use super::data_structure::exceed_bound;
use super::rollout::{get_trace, rollout};
use super::search::{Parameters, Search, Solution};
use super::util::{print_primal_bound, TimeKeeper};
use dypdl::{variable_type::Numeric, Model, ParentAndChildStateFunctionCache, State, Transition};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::str;

/// How a neighbor is accepted or rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalSearchMode {
    // Accept a neighbor if better than current best
    HillClimbing,
    // accept bad solutions with a probability that decreases as the temperature cools down.
    // Cooling is time-based (temperature = T0 * final_temp_ratio ^ (elapsed /
    // time_limit), see LocalSearch::sa_temperature), not a per-call
    // multiplicative decay -- a `temperature *= alpha` on every non-improving
    // candidate was tried first (mirroring deorder_lns.rs's sa_final_temp_ratio,
    // which replaced the same mistake there) and is a real, measured bug: it
    // runs on every evaluated non-improving candidate, not just accepted ones,
    // so how many decay steps happen in a fixed wall-clock budget depends
    // entirely on this problem/instance's rollout cost per neighbor -- cheap
    // instances freeze to near-hill-climbing within a tiny fraction of the
    // budget while expensive ones barely cool at all. Tying the schedule to
    // elapsed wall-clock time against time_limit instead makes it immune to
    // iteration throughput, so every instance reaches final_temp_ratio * T0
    // right as its own budget runs out regardless of how many moves it
    // managed to evaluate.
    SimulatedAnnealing {
        T0: f64,
        final_temp_ratio: f64,
        // When true, use the old per-call multiplicative decay
        // (`temperature *= alpha` after every non-improving candidate) instead
        // of the time-based schedule above. Kept only so the two schedules can
        // be A/B compared from the same binary/config; `alpha` is unused
        // unless this is set.
        legacy_cooling: bool,
        alpha: f64,
    },
}

/// Which neighbourhoods are used during this local search. At least one should be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Neighborhoods {
    pub swap: bool,
    pub relocate: bool,
    pub replace: bool,
    pub twoopt: bool,
}

// Which neighborhood kind a single call to `generate_neighbor` produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NeighborhoodKind {
    Swap,
    Relocate,
    Replace,
    TwoOpt,
}

impl NeighborhoodKind {
    fn index(self) -> usize {
        match self {
            NeighborhoodKind::Swap => 0,
            NeighborhoodKind::Relocate => 1,
            NeighborhoodKind::Replace => 2,
            NeighborhoodKind::TwoOpt => 3,
        }
    }
}

//default is use both
impl Default for Neighborhoods {
    fn default() -> Neighborhoods {
        Neighborhoods {
            swap: true,
            relocate: true,
            replace:true,
            twoopt: true
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeighborhoodSelection {
    Random,
    Sequential { iterations: usize },
    // Adaptive selection using SW-UCB (Sliding-Window UCB, Garivier & Moulines
    // 2008): picks the neighborhood maximizing
    // `average_reward + exploration_constant * sqrt(ln(window_len) / visits)`,
    // where `average_reward` and `visits` only count the last `window_size`
    // neighborhood switches. Discarding stale observations lets the bandit
    // track a non-stationary reward landscape (a neighborhood that was
    // productive early in the search but has since stalled loses its
    // influence once its samples age out of the window), unlike plain UCB1
    // whose all-time average reacts to changes ever more slowly as the run
    // goes on.
    Adaptive { iterations: usize, exploration_constant: f64, window_size: usize },
}

const NEIGHBORHOOD_GLOBAL_BEST_REWARD: f64 = 3.0;
const NEIGHBORHOOD_ACCEPTED_REWARD: f64 = 1.0;

impl Default for NeighborhoodSelection {
    fn default() -> NeighborhoodSelection {
        NeighborhoodSelection::Random
    }
}

/// Parameters for [`LocalSearch`].
#[derive(Debug, Clone, Copy)]
pub struct LocalSearchParameters<T> {
    /// Random seed.
    pub seed: u64,
    /// hill climbing or simulated annealing
    pub mode: LocalSearchMode,
    /// Which neighborhoods to try.
    pub neighborhoods: Neighborhoods,
    pub neighborhood_selection: NeighborhoodSelection,
    pub parameters: Parameters<T>, //time limit and quiet
}

pub struct LocalSearch<T: Numeric, B> {
    model: Rc<Model>,
    base_cost_evaluator: B,
    root_cost: T,
    current: Vec<Transition>,
    current_cost: T, // cost type is generic (can be int, can be float)
    // Cache of the state/cost after each prefix of current
    current_states: Vec<State>,
    current_costs: Vec<T>,
    // Best solution found so far, we return it 
    best: Solution<T>,
    // Whether the first solution found by CABS is feasible and has at least 2 transitions
    solvable: bool,
    mode: LocalSearchMode,
    neighborhood_selection: NeighborhoodSelection,
    enabled_neighborhood_kinds: Vec<NeighborhoodKind>,
    // SW-UCB statistics per neighborhood, restricted to the last `window_size`
    // neighborhood switches: number of times selected within the window, the
    // resulting average reward (recomputed after every push/evict so it always
    // reflects the current window), and the sum of rewards backing that average.
    neighborhood_visits: Vec<u64>,
    neighborhood_avg_reward: Vec<f64>,
    neighborhood_reward_sum: Vec<f64>,
    // Whether a neighborhood has EVER been selected, for the lifetime of this solver --
    // unlike `neighborhood_visits` (which only counts selections still inside the
    // sliding window and drops back to 0 once they all age out), this never resets.
    // Only a neighborhood that's never been tried at all gets the unconditional
    // forced-first-try treatment in `sample_neighborhood`; one that's merely aged out
    // of the window competes normally on score instead of being blindly retried
    // every time its window history empties out, regardless of how consistently bad
    // it's already been shown to be.
    neighborhood_ever_selected: Vec<bool>,
    // (neighborhood index, reward) of each switch still inside the window, oldest first.
    neighborhood_window: VecDeque<(usize, f64)>,
    neighborhood_cursor: usize,
    neighborhood_iterations_done: usize,
    segment_score: f64,
    stale_neighborhoods: usize,
    // Total time budget for the whole search (not the remaining time at any
    // given call) -- sa_temperature needs this as the denominator for
    // elapsed / time_limit. Stored separately from time_keeper, which only
    // exposes remaining time and an elapsed-time accumulator, not the
    // original limit itself.
    time_limit: f64,
    // Only used when `LocalSearchMode::SimulatedAnnealing`'s `legacy_cooling` is
    // set -- the time-based schedule (`sa_temperature`) doesn't need mutable
    // state, it's computed fresh from elapsed time every call.
    legacy_temperature: f64,
    rng: Pcg64Mcg, // random number generator
    time_keeper: TimeKeeper,
    quiet: bool, //do we print log/debugs output
    first_call: bool,
    candidate_transitions: Vec<Transition>, // candidate solution generated by the neighborhood exploration
    // FEASIBILITY-MEASUREMENT INSTRUMENTATION (temporary): (feasible, attempted) counts per
    // NeighborhoodKind, indexed by NeighborhoodKind::index(), recorded every time the SA path
    // (search_next's non-HillClimbing branch) evaluates a randomly generated neighbor. Used to
    // measure the empirical feasible-neighbor fraction along a real SA trajectory.
    neighbor_feasibility: [(u64, u64); 4],
}

impl<T, B> LocalSearch<T, B>
where
    T: Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    B: FnMut(T, T) -> T,
{
    /// Creates a new local search starting from `transitions`.
    pub fn new(
        model: Rc<Model>,
        transitions: Vec<Transition>, //initial solution
        cost: Option<T>, //cost of the initial solution, or None if infeasible
        initial_time: f64, //elapsed time already spent finding the initial solution (e.g. by CABS)
        root_cost: T, //base cost for rollout from the root state
        base_cost_evaluator: B, //how to combined accumulated cost
        // with cost when reaching a base state 
        parameters: LocalSearchParameters<T>,
    ) -> LocalSearch<T, B> {
        assert!(
            parameters.neighborhoods.swap || parameters.neighborhoods.relocate || parameters.neighborhoods.replace || parameters.neighborhoods.twoopt,
            "at least one neighborhood (swap, relocate, replace, or twoopt) must be enabled"
        );
        match parameters.neighborhood_selection {
            NeighborhoodSelection::Sequential { iterations } => {
                assert!(
                    iterations >= 1,
                    "iterations per neighborhood must be at least 1 in NeighborhoodSelection::Sequential"
                );
            }
            NeighborhoodSelection::Adaptive { iterations, exploration_constant, window_size } => {
                assert!(
                    iterations >= 1,
                    "iterations per neighborhood must be at least 1 in NeighborhoodSelection::Adaptive"
                );
                assert!(
                    exploration_constant >= 0.0,
                    "exploration_constant must be non-negative in NeighborhoodSelection::Adaptive"
                );
                assert!(
                    window_size >= 1,
                    "window_size must be at least 1 in NeighborhoodSelection::Adaptive"
                );
            }
            NeighborhoodSelection::Random => {}
        }
        let enabled_neighborhood_kinds: Vec<NeighborhoodKind> = [
            (parameters.neighborhoods.swap, NeighborhoodKind::Swap),
            (parameters.neighborhoods.relocate, NeighborhoodKind::Relocate),
            (parameters.neighborhoods.replace, NeighborhoodKind::Replace),
            (parameters.neighborhoods.twoopt, NeighborhoodKind::TwoOpt),
        ]
        .into_iter()
        .filter_map(|(enabled, kind)| enabled.then_some(kind))
        .collect();
        // solvable is true if we start from a valid cabs solution with at least 2 transitions
        let solvable = cost.is_some() && transitions.len() >= 2;
        let time_limit = parameters.parameters.time_limit.unwrap_or(f64::INFINITY);
        // Every transition the model can produce, used as the candidate pool for the
        // replace neighborhood. Computed once here since it never changes during search.
        let candidate_transitions: Vec<Transition> = model
            .forward_transitions
            .iter()
            .chain(model.forward_forced_transitions.iter())
            .cloned()
            .collect();
        let neighborhood_visits = vec![0; enabled_neighborhood_kinds.len()];
        let neighborhood_avg_reward = vec![0.0; enabled_neighborhood_kinds.len()];
        let neighborhood_reward_sum = vec![0.0; enabled_neighborhood_kinds.len()];
        let neighborhood_ever_selected = vec![false; enabled_neighborhood_kinds.len()];
        let legacy_temperature = match parameters.mode {
            LocalSearchMode::SimulatedAnnealing { T0, .. } => T0,
            LocalSearchMode::HillClimbing => 0.0,
        };
        // remaining time limit after dual_bound_local_search, if any, is passed to the local search
        let mut local_search = LocalSearch {
            model,
            base_cost_evaluator,
            root_cost,
            current_cost: cost.unwrap_or(root_cost),
            current: transitions.clone(),
            current_states: Vec::new(),
            current_costs: Vec::new(),
            best: Solution {
                cost,
                transitions,
                is_infeasible: cost.is_none(),
                time: initial_time,
                ..Default::default()
            },
            solvable,
            mode: parameters.mode,
            neighborhood_selection: parameters.neighborhood_selection,
            enabled_neighborhood_kinds,
            neighborhood_visits,
            neighborhood_avg_reward,
            neighborhood_reward_sum,
            neighborhood_ever_selected,
            neighborhood_window: VecDeque::new(),
            neighborhood_cursor: 0,
            neighborhood_iterations_done: 0,
            segment_score: 0.0,
            stale_neighborhoods: 0,
            time_limit,
            legacy_temperature,
            rng: Pcg64Mcg::seed_from_u64(parameters.seed),
            time_keeper: TimeKeeper::with_time_limit(time_limit),
            quiet: parameters.parameters.quiet,
            first_call: true,
            candidate_transitions,
            neighbor_feasibility: [(0, 0); 4],
        };

        if matches!(parameters.neighborhood_selection, NeighborhoodSelection::Adaptive { .. }) {
            local_search.neighborhood_cursor = local_search.sample_neighborhood(false);
        }

        if solvable {
            local_search.refresh_trace();
        }
        local_search.time_keeper.stop();

        local_search
    }

    // Recomputes the state and cost after every prefix of current
    // so we do not rollout the whole solution
    fn refresh_trace(&mut self) {
        let (states, costs) = get_trace(
            &self.model.target,
            self.root_cost,
            &self.current,
            &self.model,
        )
        .unzip();
        self.current_states = states;
        self.current_costs = costs;
    }

    fn generate_neighbor(&mut self) -> (Vec<Transition>, usize, NeighborhoodKind) {
        let n = self.current.len();
        let mut transitions = self.current.clone();

        // Two distinct indices in [0, n).
        let i = self.rng.random_range(0..n);
        let mut j = self.rng.random_range(0..n - 1);
        if j >= i {
            j += 1;
        } // j has to be different from i

        let active = self.active_neighborhood_kinds();
        let kind = active[self.rng.random_range(0..active.len())];

        let prefix_len = match kind {
            NeighborhoodKind::Swap => {
                transitions.swap(i, j); // swap neighborhood
                i.min(j)
            }
            NeighborhoodKind::Relocate => {
                let transition = transitions.remove(i); // relocation neighborhood
                transitions.insert(j, transition);
                i.min(j)
            }
            NeighborhoodKind::Replace => {
                let replacement = self.candidate_transitions
                    [self.rng.random_range(0..self.candidate_transitions.len())]
                .clone();
                transitions[i] = replacement; // replace neighborhood
                i
            }
            NeighborhoodKind::TwoOpt => {
                let (i, j) = if i < j { (i, j) } else { (j, i) };
                transitions[i + 1..=j].reverse(); // 2-opt neighborhood
                i + 1
            }
        };

        (transitions, prefix_len, kind)
    }

    fn active_neighborhood_kinds(&self) -> Vec<NeighborhoodKind> {
        match self.neighborhood_selection {
            NeighborhoodSelection::Random => self.enabled_neighborhood_kinds.clone(),
            NeighborhoodSelection::Sequential { .. } | NeighborhoodSelection::Adaptive { .. } => {
                vec![self.enabled_neighborhood_kinds[self.neighborhood_cursor]]
            }
        }
    }

    // SW-UCB selection: a neighborhood NEVER selected before (lifetime, not just within
    // the current window) is always tried first (infinite bonus), so every kind gets an
    // initial fair look. A neighborhood that HAS been selected before but whose samples
    // have since aged out of the window does NOT get this unconditional treatment --
    // that was the original design, but it meant a kind already shown to be consistently
    // useless got blindly force-retried every time its window history emptied out,
    // regardless of how many times that forced re-check had already failed: the window
    // is meant to let the *average reward* forget stale data for non-stationarity, not
    // to erase the fact that this kind has real history and should compete on its
    // merits. Once it's had its one lifetime freebie, an aged-out kind instead falls
    // through to the normal scoring branch below with its window visits floored at 1
    // (avoiding a division by zero) rather than being forced to the front of the queue.
    // Otherwise the neighborhood maximizing
    // `average_reward + exploration_constant * sqrt(ln(window_len) / visits)`
    // is picked, where `average_reward`/`visits`/`window_len` are all
    // restricted to the last `window_size` switches.
    fn sample_neighborhood(&mut self, exclude_current: bool) -> usize {
        let n = self.enabled_neighborhood_kinds.len();

        if n == 1 {
            return 0;
        }

        let exploration_constant = match self.neighborhood_selection {
            NeighborhoodSelection::Adaptive { exploration_constant, .. } => exploration_constant,
            _ => 0.0,
        };

        let candidates: Vec<usize> = (0..n)
            .filter(|&i| !exclude_current || i != self.neighborhood_cursor)
            .collect();

        let never_selected: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&i| !self.neighborhood_ever_selected[i])
            .collect();

        let chosen = if !never_selected.is_empty() {
            never_selected[self.rng.random_range(0..never_selected.len())]
        } else {
            let window_len = self.neighborhood_window.len().max(1) as f64;
            candidates
                .into_iter()
                .max_by(|&a, &b| {
                    let score = |i: usize| {
                        let visits = (self.neighborhood_visits[i] as f64).max(1.0);
                        self.neighborhood_avg_reward[i]
                            + exploration_constant * (window_len.ln() / visits).sqrt()
                    };
                    score(a).partial_cmp(&score(b)).unwrap()
                })
                .unwrap()
        };

        self.neighborhood_ever_selected[chosen] = true;

        chosen
    }

    fn switch_to_next_neighborhood(&mut self, exclude_current: bool) {
        if let NeighborhoodSelection::Adaptive { window_size, .. } = self.neighborhood_selection {
            let uses = self.neighborhood_iterations_done.max(1) as f64;
            let average_reward = self.segment_score / uses;
            let cursor = self.neighborhood_cursor;

            self.neighborhood_window.push_back((cursor, average_reward));
            self.neighborhood_visits[cursor] += 1;
            self.neighborhood_reward_sum[cursor] += average_reward;

            if self.neighborhood_window.len() > window_size {
                // Evict the oldest sample so the window (and hence every
                // average/visit count derived from it) always reflects only
                // the last `window_size` switches.
                let (evicted, evicted_reward) = self.neighborhood_window.pop_front().unwrap();
                self.neighborhood_visits[evicted] -= 1;
                self.neighborhood_reward_sum[evicted] -= evicted_reward;
            }

            for i in 0..self.neighborhood_avg_reward.len() {
                self.neighborhood_avg_reward[i] = if self.neighborhood_visits[i] > 0 {
                    self.neighborhood_reward_sum[i] / self.neighborhood_visits[i] as f64
                } else {
                    0.0
                };
            }
        }

        self.segment_score = 0.0;
        self.neighborhood_iterations_done = 0;

        self.neighborhood_cursor = match self.neighborhood_selection {
            NeighborhoodSelection::Random => self.neighborhood_cursor,
            NeighborhoodSelection::Sequential { .. } => {
                (self.neighborhood_cursor + 1) % self.enabled_neighborhood_kinds.len()
            }
            NeighborhoodSelection::Adaptive { .. } => self.sample_neighborhood(exclude_current),
        };
    }

    fn record_outcome(&mut self, reward: f64) {
        self.segment_score += reward;

        let iterations = match self.neighborhood_selection {
            NeighborhoodSelection::Random => return,
            NeighborhoodSelection::Sequential { iterations } => iterations,
            NeighborhoodSelection::Adaptive { iterations, .. } => iterations,
        };
        self.neighborhood_iterations_done += 1;

        if self.neighborhood_iterations_done >= iterations {
            self.switch_to_next_neighborhood(false);
        }
    }

    fn advance_on_exhaustion(&mut self) -> bool {
        match self.neighborhood_selection {
            NeighborhoodSelection::Random => false,
            NeighborhoodSelection::Sequential { .. } | NeighborhoodSelection::Adaptive { .. } => {
                self.neighborhood_iterations_done += 1;
                self.stale_neighborhoods += 1;
                let has_more = self.stale_neighborhoods < self.enabled_neighborhood_kinds.len();
                let exclude_current = matches!(self.neighborhood_selection, NeighborhoodSelection::Adaptive { .. });
                self.switch_to_next_neighborhood(exclude_current);

                has_more
            }
        }
    }

    // Best-improvement neighborhood exploration for hill climbing: evaluates every
    // swap and/or relocate move (O(n^2) neighbors) and returns the best feasible one.
    fn best_neighbor(&mut self) -> Option<(Vec<Transition>, usize, T)> {
        let n = self.current.len();
        let mut best: Option<(Vec<Transition>, usize, T)> = None;
        let active = self.active_neighborhood_kinds();

        if active.contains(&NeighborhoodKind::Swap) {
            let mut candidate = self.current.clone();

            for i in 0..n {
                for j in (i + 1)..n {
                    candidate.swap(i, j);

                    if let Some(cost) = self.evaluate(&candidate, i) {
                        if best
                            .as_ref()
                            .map_or(true, |&(_, _, best_cost)| self.is_better(cost, best_cost))
                        {
                            best = Some((candidate.clone(), i, cost));
                        }
                    }

                    candidate.swap(i, j); // undo the swap
                }
            }
        }

        if active.contains(&NeighborhoodKind::Relocate) {
            let mut candidate = self.current.clone();

            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let transition = candidate.remove(i);
                    candidate.insert(j, transition);
                    let prefix_len = i.min(j);

                    if let Some(cost) = self.evaluate(&candidate, prefix_len) {
                        if best
                            .as_ref()
                            .map_or(true, |&(_, _, best_cost)| self.is_better(cost, best_cost))
                        {
                            best = Some((candidate.clone(), prefix_len, cost));
                        }
                    }

                    let transition = candidate.remove(j);
                    candidate.insert(i, transition); // undo the relocation
                }
            }
        }
        if active.contains(&NeighborhoodKind::Replace) {
            let mut candidate = self.current.clone();

            let candidate_transitions = self.candidate_transitions.clone();

            for i in 0..n {
                for transition in &candidate_transitions {
                    let original_transition = candidate[i].clone();
                    candidate[i] = transition.clone();

                    if let Some(cost) = self.evaluate(&candidate, i) {
                        if best
                            .as_ref()
                            .map_or(true, |&(_, _, best_cost)| self.is_better(cost, best_cost))
                        {
                            best = Some((candidate.clone(), i, cost));
                        }
                    }

                    candidate[i] = original_transition; // undo the replacement
                }
            }
        }
        if active.contains(&NeighborhoodKind::TwoOpt) {
            let mut candidate = self.current.clone();

            for i in 0..n {
                // j = i + 1 would reverse a single-element slice (a no-op), so skip it.
                for j in (i + 2)..n {
                    candidate[i + 1..=j].reverse();

                    if let Some(cost) = self.evaluate(&candidate, i + 1) {
                        if best
                            .as_ref()
                            .map_or(true, |&(_, _, best_cost)| self.is_better(cost, best_cost))
                        {
                            best = Some((candidate.clone(), i + 1, cost));
                        }
                    }

                    candidate[i + 1..=j].reverse();
                }
            }
        }

        best
    }

    // Rollout from prefix to end & returns cost
    fn evaluate(&mut self, transitions: &[Transition], prefix_len: usize) -> Option<T> {
        let (state, cost) = if prefix_len == 0 {
            (self.model.target.clone(), self.root_cost)
        } else {
            (
                self.current_states[prefix_len - 1].clone(),
                self.current_costs[prefix_len - 1],
            )
        };
        let mut function_cache = ParentAndChildStateFunctionCache::new(&self.model.state_functions);
        let result = rollout(
            &state,
            &mut function_cache,
            cost,
            &transitions[prefix_len..],
            &mut self.base_cost_evaluator,
            &self.model,
        )?;

        result.is_base.then_some(result.cost)
    }

    // Decides whether a candidate with the given cost replaces `current`.
    fn accept(&mut self, cost: T) -> bool {
        if !exceed_bound(&self.model, cost, Some(self.current_cost)) {
            return true;
        }

        match self.mode {
            LocalSearchMode::HillClimbing => false,
            LocalSearchMode::SimulatedAnnealing { legacy_cooling, alpha, .. } => {
                let delta = if self.model.reduce_function == dypdl::ReduceFunction::Max {
                    self.current_cost - cost
                } else {
                    cost - self.current_cost
                }
                .to_continuous();

                if legacy_cooling {
                    let accept = self.rng.random::<f64>() < (-delta / self.legacy_temperature).exp();
                    self.legacy_temperature *= alpha;

                    accept
                } else {
                    self.rng.random::<f64>() < (-delta / self.sa_temperature()).exp()
                }
            }
        }
    }

    // Time-based cooling schedule: temperature = T0 * final_temp_ratio ^
    // (elapsed / time_limit) -- see LocalSearchMode::SimulatedAnnealing's doc
    // for why this replaced a per-call multiplicative `temperature *= alpha`
    // decay. elapsed_time() is TimeKeeper's accumulated active time, which is
    // exactly what time_limit is measured against everywhere else in this
    // file (check_time_limit, remaining_time_limit).
    fn sa_temperature(&self) -> f64 {
        let LocalSearchMode::SimulatedAnnealing { T0, final_temp_ratio, .. } = self.mode else {
            unreachable!("sa_temperature is only called from the SimulatedAnnealing accept branch")
        };
        let elapsed_fraction = if self.time_limit.is_finite() && self.time_limit > 0.0 {
            (self.time_keeper.elapsed_time() / self.time_limit).clamp(0.0, 1.0)
        } else {
            0.0
        };
        T0 * final_temp_ratio.powf(elapsed_fraction)
    }

    // Returns whether `cost` is a strict improvement over `other`.
    fn is_better(&self, cost: T, other: T) -> bool {
        !exceed_bound(&self.model, cost, Some(other))
    }

    // Prints a snapshot of each neighborhood's SW-UCB stats (average reward
    // and visit count restricted to the current window) tagged with the
    // current elapsed time. Called both on every new-best solution and at
    // termination, so the sequence of snapshots forms a trajectory showing
    // which neighborhood was most useful over the course of the search
    // (e.g. via `experiments/run_experiments.py`, which parses this line).
    fn log_neighborhood_stats(&self, time: f64) {
        if let NeighborhoodSelection::Adaptive { .. } = self.neighborhood_selection {
            let stats: Vec<String> = self
                .enabled_neighborhood_kinds
                .iter()
                .zip(&self.neighborhood_avg_reward)
                .zip(&self.neighborhood_visits)
                .map(|((kind, avg_reward), visits)| {
                    format!("{:?}=(avg_reward={avg_reward:.6}, visits={visits})", kind).to_lowercase()
                })
                .collect();
            println!("neighborhood stats: t={time:.6} {}", stats.join(" "));
        }
    }

    // FEASIBILITY-MEASUREMENT INSTRUMENTATION (temporary): prints the (feasible/attempted)
    // ratio recorded in `neighbor_feasibility` for each neighborhood kind, only meaningful
    // in SimulatedAnnealing mode (HillClimbing uses best_neighbor, not generate_neighbor).
    fn log_neighbor_feasibility(&self) {
        if matches!(self.mode, LocalSearchMode::HillClimbing) {
            return;
        }
        let kinds = [
            NeighborhoodKind::Swap,
            NeighborhoodKind::Relocate,
            NeighborhoodKind::Replace,
            NeighborhoodKind::TwoOpt,
        ];
        let stats: Vec<String> = kinds
            .iter()
            .map(|kind| {
                let (feasible, attempted) = self.neighbor_feasibility[kind.index()];
                format!("{:?}=({feasible}/{attempted})", kind).to_lowercase()
            })
            .collect();
        println!("neighbor feasibility: {}", stats.join(" "));
    }
}

impl<T, B> Search<T> for LocalSearch<T, B>
where
    T: Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    B: FnMut(T, T) -> T,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if !self.solvable {
            return Ok((self.best.clone(), true));
        }

        if self.first_call {
            self.first_call = false;

            return Ok((self.best.clone(), false));
        }

        self.time_keeper.start();

        loop {
            if self.time_keeper.check_time_limit(self.quiet) {
                self.best.time = self.time_keeper.elapsed_time();
                self.best.time_out = true;
                self.time_keeper.stop();
                self.log_neighborhood_stats(self.best.time);
                self.log_neighbor_feasibility();

                return Ok((self.best.clone(), true));
            }

            if matches!(self.mode, LocalSearchMode::HillClimbing) {
                if let Some((candidate, _, cost)) = self.best_neighbor() {
                    if self.accept(cost) {
                        self.current = candidate;
                        self.current_cost = cost;
                        self.refresh_trace();
                        self.stale_neighborhoods = 0;

                        let is_new_best = self.is_better(cost, self.best.cost.unwrap());
                        self.record_outcome(if is_new_best {
                            NEIGHBORHOOD_GLOBAL_BEST_REWARD
                        } else {
                            NEIGHBORHOOD_ACCEPTED_REWARD
                        });

                        if is_new_best {
                            self.best.cost = Some(cost);
                            self.best.transitions = self.current.clone();
                            self.best.time = self.time_keeper.elapsed_time();

                            if !self.quiet {
                                print_primal_bound(&self.best);
                            }

                            self.log_neighborhood_stats(self.best.time);
                            self.time_keeper.stop();

                            return Ok((self.best.clone(), false));
                        }

                        continue;
                    }
                }

                if self.advance_on_exhaustion() {
                    continue;
                }

                self.best.time = self.time_keeper.elapsed_time();
                self.time_keeper.stop();
                self.log_neighborhood_stats(self.best.time);

                return Ok((self.best.clone(), true));
            }

            let (candidate, prefix_len, kind) = self.generate_neighbor();
            let evaluated = self.evaluate(&candidate, prefix_len);

            let (feasible, attempted) = &mut self.neighbor_feasibility[kind.index()];
            *attempted += 1;
            if evaluated.is_some() {
                *feasible += 1;
            }

            let cost = match evaluated {
                Some(cost) => cost,
                None => {
                    self.record_outcome(0.0);
                    continue; // infeasible neighbor: try another one
                }
            };

            if self.accept(cost) {
                self.current = candidate;
                self.current_cost = cost;
                self.refresh_trace();

                let is_new_best = self.is_better(cost, self.best.cost.unwrap());
                self.record_outcome(if is_new_best {
                    NEIGHBORHOOD_GLOBAL_BEST_REWARD
                } else {
                    NEIGHBORHOOD_ACCEPTED_REWARD
                });

                if is_new_best {
                    self.best.cost = Some(cost);
                    self.best.transitions = self.current.clone();
                    self.best.time = self.time_keeper.elapsed_time();

                    if !self.quiet {
                        print_primal_bound(&self.best);
                    }

                    self.log_neighborhood_stats(self.best.time);
                    self.time_keeper.stop();

                    return Ok((self.best.clone(), false));
                }
            } else {
                self.record_outcome(0.0);
            }
        }
    }
}
