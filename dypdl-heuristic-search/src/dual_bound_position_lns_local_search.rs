// Alternates between PositionLns and ModelAwareLocalSearch via a multi-armed bandit over short
// wall-clock time slices, rather than running one to a self-reported "stuck" point and handing off
// forever: PositionLns's per-iteration cost varies by orders of magnitude across domains, so any
// single stall threshold is either unreachable or trivial depending on the domain. The bandit
// sidesteps that calibration problem by having both solvers compete on measured reward-per-slice,
// using the same sliding-window UCB policy `ModelAwareNeighborhoodSelection::Adaptive` uses inside
// `ModelAwareLocalSearch`.
//
// Both solvers stay alive for the whole search (never reconstructed mid-run) so neither loses its
// own learned state across turns. Each turn is bounded by `set_turn_deadline` (an elapsed-time
// value at which `search_next` returns `terminated: false` without waiting for a real local
// optimum or the overall time limit), and whichever solver is NOT active gets synced to the other's
// incumbent via `adopt_incumbent` right before its next turn, so a solver that reached a genuine
// local optimum gets a chance to resume once the other arm improves on it.

use super::dual_bound_local_search::cabs_runner;
use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::data_structure::exceed_bound;
use super::search_algorithm::{
    CabsParameters, ModelAwareLocalSearch, ModelAwareLocalSearchParameters, PositionLns,
    PositionLnsParameters, Search, Solution, SuccessorGenerator,
};
use dypdl::variable_type::{self, Numeric};
use dypdl::Transition;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

const ARM_POSITION_LNS: usize = 0;
const ARM_LOCAL_SEARCH: usize = 1;
const NUM_ARMS: usize = 2;

// Same constants ModelAwareNeighborhoodSelection::Adaptive defaults to in ModelAwareLocalSearch --
// not re-tuned here, just reusing an already-validated starting point for the same kind of
// sliding-window UCB choice.
const DEFAULT_WINDOW_SIZE: usize = 50;
const DEFAULT_EXPLORATION_CONSTANT: f64 = std::f64::consts::SQRT_2;

// Hard cap on how much of the remaining budget a single CABS-on-stuck escape round (see
// `run_cabs`'s field doc) or the one-time initial-solution CABS round (see
// `create_dual_bound_position_lns_local_search`) may be handed as its own `time_limit`.
// `CabsParameters::time_limit` is only checked BETWEEN beam-width-doubling rounds, not within one,
// and the escape's starting beam width grows unboundedly across repeated stuck events, so an
// uncapped late escape can start from an already-huge beam and run far longer than its nominal
// budget. Capping every call bounds the worst-case overshoot and keeps the bandit itself from
// being starved of a share of the overall budget.
const MAX_CABS_CALL_SECONDS: f64 = 20.0;

// Separate, more generous cap for the one-time initial-solution CABS round: finding a first
// feasible solution is worth more runway than any single escape attempt, but it must still be
// bounded so it can't eat into the bandit's own budget.
const MAX_INITIAL_CABS_SECONDS: f64 = 60.0;

// Env-var overrides for the two caps above, read fresh each call so a sweep script can toggle them
// per-process. Defaults to the hardcoded constants above when unset.
fn max_cabs_call_seconds() -> f64 {
    std::env::var("DIDP_COMBO_MAX_CABS_CALL_SECONDS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(MAX_CABS_CALL_SECONDS)
}

fn max_initial_cabs_seconds() -> f64 {
    std::env::var("DIDP_COMBO_MAX_INITIAL_CABS_SECONDS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(MAX_INITIAL_CABS_SECONDS)
}

/// Sliding-window UCB over a small, fixed number of arms -- a stripped-down copy of the bandit
/// `ModelAwareLocalSearch`'s own `ModelAwareNeighborhoodSelection::Adaptive` already implements
/// (see its doc there for the full rationale), generalized just enough to work over `NUM_ARMS`
/// arms instead of one arm per enabled neighborhood kind.
struct SwUcb {
    window_size: usize,
    exploration_constant: f64,
    visits: [u64; NUM_ARMS],
    avg_reward: [f64; NUM_ARMS],
    reward_sum: [f64; NUM_ARMS],
    ever_selected: [bool; NUM_ARMS],
    // Arms currently excluded from selection (e.g. ModelAwareLocalSearch just exhausted every
    // neighborhood with nothing left to try against its current incumbent) -- distinct from a low
    // UCB score: a cold arm is skipped entirely until something reactivates it, not just
    // disfavored.
    cold: [bool; NUM_ARMS],
    window: VecDeque<(usize, f64)>,
}

impl SwUcb {
    fn new(window_size: usize, exploration_constant: f64) -> SwUcb {
        SwUcb {
            window_size,
            exploration_constant,
            visits: [0; NUM_ARMS],
            avg_reward: [0.0; NUM_ARMS],
            reward_sum: [0.0; NUM_ARMS],
            ever_selected: [false; NUM_ARMS],
            cold: [false; NUM_ARMS],
            window: VecDeque::new(),
        }
    }

    /// Picks the next arm: an arm that's never been tried gets an unconditional first look (same
    /// forced-exploration rule ModelAwareLocalSearch's own copy of this uses), then whichever live
    /// (non-cold) arm maximizes `average_reward + exploration_constant * sqrt(ln(window_len) /
    /// visits)`. Returns `None` only if every arm is cold (nothing left this bandit can usefully
    /// run).
    fn choose(&self) -> Option<usize> {
        let live: Vec<usize> = (0..NUM_ARMS).filter(|&i| !self.cold[i]).collect();

        if live.is_empty() {
            return None;
        }

        let never_selected: Vec<usize> =
            live.iter().copied().filter(|&i| !self.ever_selected[i]).collect();

        if let Some(&arm) = never_selected.first() {
            return Some(arm);
        }

        let window_len = (self.window.len().max(1)) as f64;

        live.into_iter().max_by(|&a, &b| {
            let score = |i: usize| {
                let visits = (self.visits[i] as f64).max(1.0);
                self.avg_reward[i] + self.exploration_constant * (window_len.ln() / visits).sqrt()
            };
            score(a).partial_cmp(&score(b)).unwrap()
        })
    }

    fn record(&mut self, arm: usize, reward: f64) {
        self.ever_selected[arm] = true;
        self.window.push_back((arm, reward));
        self.visits[arm] += 1;
        self.reward_sum[arm] += reward;

        if self.window.len() > self.window_size {
            let (evicted, evicted_reward) = self.window.pop_front().unwrap();
            self.visits[evicted] -= 1;
            self.reward_sum[evicted] -= evicted_reward;
        }

        for i in 0..NUM_ARMS {
            self.avg_reward[i] = if self.visits[i] > 0 {
                self.reward_sum[i] / self.visits[i] as f64
            } else {
                0.0
            };
        }
    }
}

/// Alternates `PositionLns` and `ModelAwareLocalSearch` via a bandit over short time slices -- see
/// this module's doc for the full design and why it replaced a stuck-threshold hand-off. Neither
/// solver's own standalone behavior is changed: this only calls their existing public
/// `Search`/`::new`/`set_turn_deadline`/`adopt_incumbent` interface.
struct BanditPositionLnsLocalSearch<T>
where
    T: Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    position_lns: PositionLns<T, Box<dyn FnMut(T, T) -> T>>,
    local_search: ModelAwareLocalSearch<T, Box<dyn FnMut(T, T) -> T>>,
    model: Rc<dypdl::Model>,
    bandit: SwUcb,
    slice_seconds: f64,
    // Each arm's own clock, tracked separately so a turn's deadline is computed against the
    // right one, not the wrapper's combined total below.
    position_lns_time: f64,
    local_search_time: f64,
    // Cumulative wall-clock time across every turn of either arm so far; each arm's own
    // `Solution::time` is relative to its own clock, not to the point CABS handed off from.
    total_elapsed: f64,
    overall_time_limit: Option<f64>,
    best: Solution<T>,
    // Gives the local_search arm its own escape mechanism: since ModelAwareLocalSearch's
    // neighborhoods are its whole search space (unlike PositionLns's combinatorially-huge
    // destroy-set choices), a genuine local optimum there is a real dead end this bandit's own
    // turn-taking can't do anything about -- PositionLns not improving on the same incumbent for a
    // while doesn't mean it never will, but LocalSearch exhausting every neighborhood is a proof,
    // not a guess.
    run_cabs: Box<dyn FnMut(CabsParameters<T>) -> (Solution<T>, usize)>,
    cabs_parameters_template: CabsParameters<T>,
    // Beam size the next stuck-triggered CABS round should start from, carried across rounds so
    // growth compounds instead of restarting from the configured initial size each time.
    cabs_beam_size: usize,
    // The first call to this wrapper's own search_next just echoes the initial CABS solution
    // before any bandit turn runs, matching the convention every other solver here follows.
    first_call: bool,
}

impl<T> BanditPositionLnsLocalSearch<T>
where
    T: Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn is_better(&self, cost: T, other: T) -> bool {
        !exceed_bound(&self.model, cost, Some(other))
    }

    fn reward(&self, before: T, after: T) -> f64 {
        if !self.is_better(after, before) {
            return 0.0;
        }

        let denom = std::cmp::max(before.abs(), after.abs()).to_continuous();

        if denom == 0.0 {
            return 1.0;
        }

        ((before - after).to_continuous().abs() / denom).min(1.0)
    }

    fn remaining_time(&self) -> Option<f64> {
        self.overall_time_limit
            .map(|limit| (limit - self.total_elapsed).max(0.0))
    }

    /// Runs one turn of `arm`, syncing it to the wrapper's current incumbent first (a no-op if it
    /// is already at least as good), bounding it to a slice via `set_turn_deadline`, and folding
    /// its outcome back into the wrapper's own bookkeeping (elapsed time, incumbent, bandit stats,
    /// cold/live status). Returns `(improved, terminated, time_out)`: `improved` is whether
    /// `self.best` changed this turn (the only thing the caller needs to decide whether to surface
    /// this turn to ITS OWN caller -- `self.best` itself is already updated in place when true).
    fn run_turn(&mut self, arm: usize) -> Result<(bool, bool, bool), Box<dyn Error>> {
        let remaining = self.remaining_time();
        let slice = remaining.map_or(self.slice_seconds, |r| r.min(self.slice_seconds));
        let cost_before = self.best.cost.unwrap();

        let (solution, terminated) = match arm {
            ARM_POSITION_LNS => {
                self.position_lns.adopt_incumbent(
                    self.best.transitions.clone(),
                    cost_before,
                    self.total_elapsed,
                );
                let deadline = self.position_lns_time + slice;
                self.position_lns.set_turn_deadline(Some(deadline));
                let (solution, terminated) = self.position_lns.search_next()?;
                self.total_elapsed += solution.time - self.position_lns_time;
                self.position_lns_time = solution.time;
                (solution, terminated)
            }
            ARM_LOCAL_SEARCH => {
                self.local_search.adopt_incumbent(
                    self.best.transitions.clone(),
                    cost_before,
                    self.total_elapsed,
                );
                let deadline = self.local_search_time + slice;
                self.local_search.set_turn_deadline(Some(deadline));
                let (solution, terminated) = self.local_search.search_next()?;
                self.total_elapsed += solution.time - self.local_search_time;
                self.local_search_time = solution.time;
                (solution, terminated)
            }
            _ => unreachable!(),
        };

        let reward = solution
            .cost
            .map_or(0.0, |cost| self.reward(cost_before, cost));
        self.bandit.record(arm, reward);

        let time_out = solution.time_out;

        // A genuine local optimum (ModelAwareLocalSearch exhausting every neighborhood) -- not the
        // whole search being over, just this arm being unproductive against its current incumbent.
        // Benched until the other arm hands it something better via adopt_incumbent, which
        // un-benches it below.
        if terminated && !time_out {
            self.bandit.cold[arm] = true;
        }

        let mut improved = solution.cost.is_some_and(|cost| self.is_better(cost, cost_before));

        if improved {
            // Un-bench the OTHER arm too, in case it was cold against the old, worse incumbent --
            // adopt_incumbent on its next turn will pick this up regardless, but clearing the flag
            // here lets the bandit consider selecting it again immediately instead of treating it
            // as permanently dead.
            self.bandit.cold[1 - arm] = false;
            self.best.cost = solution.cost;
            self.best.transitions = solution.transitions;
            self.best.time = self.total_elapsed;
        } else if arm == ARM_LOCAL_SEARCH && terminated && !time_out {
            // Give the local-search arm back its own escape mechanism -- see the `run_cabs` field's
            // doc for why PositionLns not having improved on the same incumbent yet isn't a
            // substitute for this. A no-op if no time remains.
            let remaining_for_cabs = self.remaining_time();

            if !remaining_for_cabs.is_some_and(|r| r <= 0.0) {
                let mut cabs_parameters = self.cabs_parameters_template;
                cabs_parameters.beam_search_parameters.parameters.primal_bound = Some(cost_before);
                // Capped at MAX_CABS_CALL_SECONDS (see its doc): handing this round the ENTIRE
                // remaining budget let a single already-wide beam round overshoot it catastrophically
                // under contention. A capped round that doesn't finish just means local_search stays
                // cold and the bandit moves on to PositionLns; the next stuck event retriggers the
                // escape from wherever cabs_beam_size left off, so bounded rounds still make the same
                // cumulative progress an uncapped one would, just spread over more, safer calls.
                cabs_parameters.beam_search_parameters.parameters.time_limit =
                    remaining_for_cabs.map(|r| r.min(max_cabs_call_seconds()));
                cabs_parameters.beam_search_parameters.beam_size = self.cabs_beam_size;

                let (cabs_solution, final_beam_size) = (self.run_cabs)(cabs_parameters);
                self.cabs_beam_size = final_beam_size;
                self.total_elapsed += cabs_solution.time;

                if cabs_solution
                    .cost
                    .is_some_and(|cost| self.is_better(cost, cost_before))
                {
                    self.bandit.cold[ARM_LOCAL_SEARCH] = false;
                    self.bandit.cold[ARM_POSITION_LNS] = false;
                    self.best.cost = cabs_solution.cost;
                    self.best.transitions = cabs_solution.transitions;
                    self.best.time = self.total_elapsed;
                    improved = true;
                }
                // CABS didn't find anything better either: local_search stays cold (already set
                // above) until PositionLns hands it something to work with instead.
            }
        }

        Ok((improved, terminated, time_out))
    }
}

impl<T> Search<T> for BanditPositionLnsLocalSearch<T>
where
    T: Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if self.first_call {
            self.first_call = false;

            return Ok((self.best.clone(), false));
        }

        loop {
            if self.best.cost.is_none() {
                return Ok((self.best.clone(), true));
            }

            if self.remaining_time().is_some_and(|remaining| remaining <= 0.0) {
                self.best.time_out = true;

                return Ok((self.best.clone(), true));
            }

            let Some(arm) = self.bandit.choose() else {
                // Both arms proved themselves stuck against the same incumbent with nothing to
                // hand each other -- there is nothing left this pair can do automatically.
                return Ok((self.best.clone(), true));
            };

            if std::env::var("DIDP_HYBRID_DIAG").is_ok() {
                eprintln!(
                    "[bandit diag] turn={} t={:.3} cost={:?} plns_avg={:.4}/{} ls_avg={:.4}/{}",
                    if arm == ARM_POSITION_LNS { "position_lns" } else { "local_search" },
                    self.total_elapsed,
                    self.best.cost,
                    self.bandit.avg_reward[ARM_POSITION_LNS],
                    self.bandit.visits[ARM_POSITION_LNS],
                    self.bandit.avg_reward[ARM_LOCAL_SEARCH],
                    self.bandit.visits[ARM_LOCAL_SEARCH],
                );
            }

            let (improved, terminated, time_out) = self.run_turn(arm)?;

            if improved {
                return Ok((self.best.clone(), false));
            }

            // A genuine time-out on this arm means the whole overall budget is exhausted (both
            // arms are always constructed with the full remaining budget as their own time_limit,
            // since slicing is handled entirely via turn deadlines here, not by shrinking either
            // arm's own time_limit) -- nothing left to alternate into.
            if terminated && time_out {
                self.best.time_out = true;

                return Ok((self.best.clone(), true));
            }
        }
    }
}

/// Creates a solver alternating `PositionLns` and `ModelAwareLocalSearch` via a multi-armed bandit
/// over short time slices -- see this module's doc for the full design. Both find their first
/// feasible solution the usual way (a one-time CABS round, exactly as `create_dual_bound_position_lns`
/// and `create_dual_bound_model_aware_local_search` already do on their own), and neither solver's
/// own standalone behavior is changed by this.
pub fn create_dual_bound_position_lns_local_search<T>(
    model: Rc<dypdl::Model>,
    mut position_lns_parameters: PositionLnsParameters<T>,
    mut local_search_parameters: ModelAwareLocalSearchParameters<T>,
    cabs_parameters: CabsParameters<T>,
    f_evaluator_type: FEvaluatorType,
    slice_seconds: f64,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
    let root_cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    // Find an initial feasible solution with CABS, exactly as PositionLns/ModelAwareLocalSearch/
    // LNBS do on their own when no initial solution is given. `run_cabs` and `cabs_parameters` are
    // kept afterward (not just consumed here) so the local_search arm can trigger further bounded
    // CABS rounds on its own local optima later -- see the `run_cabs` field's doc.
    //
    // Capped at MAX_INITIAL_CABS_SECONDS (see its doc) rather than handed the model's full
    // `time_limit` the way standalone CABS/PositionLns/ModelAwareLocalSearch use it: unlike those,
    // this initial round isn't the whole search, it's a preamble the bandit still needs real time
    // after -- an uncapped round that (under load or on a genuinely slow instance) ate nearly the
    // entire budget left the bandit with only a few seconds to do anything at all. `cabs_parameters`
    // itself (used unmodified below as `cabs_parameters_template` for the local-search arm's own
    // later escape rounds, each already separately capped at call time) is untouched -- only this
    // one-off initial call's own copy is capped.
    // No overall time_limit at all (None) means no starvation risk either -- the bandit would get
    // an unbounded turn afterward regardless of how long this took -- so the cap only applies when
    // there's a real budget for this round to eat into.
    let mut initial_cabs_parameters = cabs_parameters;
    initial_cabs_parameters.beam_search_parameters.parameters.time_limit = cabs_parameters
        .beam_search_parameters
        .parameters
        .time_limit
        .map(|limit| limit.min(max_initial_cabs_seconds()));
    let mut run_cabs = cabs_runner(model.clone(), generator, f_evaluator_type, root_cost);
    let (solution, initial_beam_size) = run_cabs(initial_cabs_parameters);

    let overall_time_limit = position_lns_parameters.parameters.time_limit;
    // Leave CABS's elapsed time out of the alternation's own time budget, same as every other
    // solver here does. Both arms get the FULL remaining budget as their own time_limit (never
    // shrunk per-turn): slicing is handled entirely via set_turn_deadline, so time_limit only ever
    // needs to mean "the whole thing is over," exactly as it does standalone.
    let remaining_after_cabs =
        overall_time_limit.map(|time_limit| (time_limit - solution.time).max(0.0));
    position_lns_parameters.parameters.time_limit = remaining_after_cabs;
    local_search_parameters.parameters.time_limit = remaining_after_cabs;

    // The bandit itself decides how much attention PositionLns gets; it doesn't need to
    // self-report stuck the way the old hand-off design required.
    position_lns_parameters.stall_limit = None;
    position_lns_parameters.stall_time_limit = None;

    let f_evaluator_type_a = f_evaluator_type;
    let base_cost_evaluator_a: Box<dyn FnMut(T, T) -> T> =
        Box::new(move |cost, base_cost| f_evaluator_type_a.eval(cost, base_cost));
    let f_evaluator_type_b = f_evaluator_type;
    let base_cost_evaluator_b: Box<dyn FnMut(T, T) -> T> =
        Box::new(move |cost, base_cost| f_evaluator_type_b.eval(cost, base_cost));

    let mut position_lns = PositionLns::new(
        model.clone(),
        solution.transitions.clone(),
        solution.cost,
        solution.time,
        root_cost,
        base_cost_evaluator_a,
        position_lns_parameters,
        f_evaluator_type,
    );
    let mut local_search = ModelAwareLocalSearch::new(
        model.clone(),
        solution.transitions.clone(),
        solution.cost,
        solution.time,
        root_cost,
        base_cost_evaluator_b,
        local_search_parameters,
    );

    // Consume both solvers' own one-time first_call echo now (cheap -- neither does any real work
    // for it) so it doesn't waste a real bandit turn on it later; the wrapper's own first_call
    // handles echoing the initial solution to ITS caller instead.
    let _ = position_lns.search_next();
    let _ = local_search.search_next();

    Box::new(BanditPositionLnsLocalSearch {
        position_lns,
        local_search,
        model,
        bandit: SwUcb::new(DEFAULT_WINDOW_SIZE, DEFAULT_EXPLORATION_CONSTANT),
        slice_seconds,
        position_lns_time: 0.0,
        local_search_time: 0.0,
        total_elapsed: solution.time,
        overall_time_limit,
        best: solution,
        first_call: true,
        run_cabs,
        cabs_parameters_template: cabs_parameters,
        cabs_beam_size: initial_beam_size,
    })
}
