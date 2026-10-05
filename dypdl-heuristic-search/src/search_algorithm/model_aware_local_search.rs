use super::data_structure::{
    classify_transition_cardinality, exceed_bound, SuccessorGenerator, TransitionCardinality,
    TransitionWithId,
};
use super::rollout::{get_trace, rollout};
use super::search::{Parameters, Search, Solution};
use super::util::{print_primal_bound, TimeKeeper};
use dypdl::{
    variable_type::Numeric, Element, Model, ParentAndChildStateFunctionCache, State,
    StateFunctionCache, Transition,
};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::str;

/// How a neighbor is accepted or rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelAwareLocalSearchMode {
    // Accept a neighbor if better than current best
    HillClimbing,
    // accept bad solutions with a probability that decreases as the temperature cools down.
    // Cooling is time-based (temperature = T0 * final_temp_ratio ^ (elapsed /
    // time_limit), see ModelAwareLocalSearch::sa_temperature), not a per-call
    // multiplicative decay: a decay tied to iteration count would make the cooling rate
    // depend on this problem/instance's rollout cost per neighbor -- cheap instances would
    // freeze to near-hill-climbing within a tiny fraction of the budget while expensive ones
    // barely cool at all. Tying the schedule to elapsed wall-clock time against time_limit
    // instead makes it immune to iteration throughput.
    SimulatedAnnealing {
        T0: f64,
        final_temp_ratio: f64,
    },
}

/// Which neighbourhoods are used during this local search. At least one should be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelAwareNeighborhoods {
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

//default is use both
impl Default for ModelAwareNeighborhoods {
    fn default() -> ModelAwareNeighborhoods {
        ModelAwareNeighborhoods {
            swap: true,
            relocate: true,
            replace:true,
            twoopt: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelAwareNeighborhoodSelection {
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

impl Default for ModelAwareNeighborhoodSelection {
    fn default() -> ModelAwareNeighborhoodSelection {
        ModelAwareNeighborhoodSelection::Random
    }
}

/// Parameters for [`ModelAwareLocalSearch`].
#[derive(Debug, Clone, Copy)]
pub struct ModelAwareLocalSearchParameters<T> {
    /// Random seed.
    pub seed: u64,
    /// hill climbing or simulated annealing
    pub mode: ModelAwareLocalSearchMode,
    /// Which neighborhoods to try.
    pub neighborhoods: ModelAwareNeighborhoods,
    pub neighborhood_selection: ModelAwareNeighborhoodSelection,
    /// When true, before rolling out a candidate, check the state at the candidate's first
    /// changed position against the model's forced transitions and `transition_dominance`
    /// rules (via `SuccessorGenerator::generate_applicable_transitions`, the same machinery
    /// CABS/LNBS get automatically from forward expansion): reject without ever calling
    /// `evaluate` when a forced transition is applicable there but not the one the candidate
    /// places, or (transition_dominance only) when the placed transition is provably
    /// dominated there by another transition actually applicable at that state. Both are
    /// optimality-preserving pruning rules in DIDP's own semantics, not heuristics -- see
    /// `passes_structural_pruning`'s doc. Off by default: local search never consulted either
    /// mechanism before, and most domains declare neither.
    pub dominance_pruning: bool,
    pub parameters: Parameters<T>, //time limit and quiet
}

pub struct ModelAwareLocalSearch<T: Numeric, B> {
    model: Rc<Model>,
    base_cost_evaluator: B,
    root_cost: T,
    // Carries each position's catalog (id, forced) identity alongside the transition
    // itself, so it never needs to be re-derived by hashing (name, parameter_values)
    // -- swap/relocate/twoopt just move `TransitionWithId` elements around (identity
    // travels with the data), and replace already knows the id of whatever it's
    // substituting in, since it picked it from the catalog by id in the first place.
    current: Vec<TransitionWithId>,
    current_cost: T, // cost type is generic (can be int, can be float)
    // Cache of the state/cost after each prefix of current
    current_states: Vec<State>,
    current_costs: Vec<T>,
    // Best solution found so far, we return it 
    best: Solution<T>,
    // Whether the first solution found by CABS is feasible and has at least 2 transitions
    solvable: bool,
    mode: ModelAwareLocalSearchMode,
    neighborhood_selection: ModelAwareNeighborhoodSelection,
    enabled_neighborhood_kinds: Vec<NeighborhoodKind>,
    // Shares the same forced-transition short-circuit and transition_dominance filtering
    // CABS/LNBS use during forward expansion (see `passes_structural_pruning`). Only built
    // and consulted when `dominance_pruning` is set -- otherwise local search never touches
    // it, matching its behavior before this field existed.
    successor_generator: SuccessorGenerator<Transition>,
    dominance_pruning: bool,
    // Precomputed once: whether the model declares anything `passes_structural_pruning`
    // could ever actually rule a candidate out on. `generate_applicable_transitions`
    // itself always scans the whole catalog for applicability regardless of whether
    // dominance/forced structure exists (that's unavoidable -- it doesn't know in
    // advance which candidate it'll be compared against), so calling it once per
    // candidate is real, non-trivial overhead on every neighbor, paid whether or not
    // the model has anything to prune. On a model with neither forced transitions nor
    // transition_dominance, the check can only ever return true (the placed transition,
    // if `evaluate` would've accepted it, is necessarily in that unfiltered applicable
    // set too) -- so skip the call entirely there rather than pay full-catalog-scan cost
    // for a result that's already known.
    has_forced_or_dominance: bool,
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
    rng: Pcg64Mcg, // random number generator
    time_keeper: TimeKeeper,
    quiet: bool, //do we print log/debugs output
    first_call: bool,
    candidate_transitions: Vec<TransitionWithId>, // candidate solution generated by the neighborhood exploration
    // Same-parameter grouping (transitions sharing `parameter_values`), same key
    // `position_lns.rs`'s `compute_alternatives` groups its `by_params` map by. Used by
    // `replace_same_param_only` to restrict the Replace neighborhood's candidate pool.
    by_params: FxHashMap<Vec<Element>, Vec<TransitionWithId>>,
    // Replace-neighborhood arm (opt-in via DIDP_MALS_REPLACE_SAME_PARAM=1): restricts
    // Replace's candidate pool at a position from the full grounded catalog
    // (`candidate_transitions`) to just that position's same-parameter siblings
    // (`by_params`). Targets domains where the variant is a genuine choice (cvrp's visit
    // vs. visit-via-depot for a given customer, mdkp's pack vs. ignore -- both share
    // `parameter_values` at a position). A same-position-only pool is also far cheaper to
    // evaluate than the full catalog, which matters under `neighborhood_selection: adaptive`
    // (only one kind active at a time, so a slow Replace sweep eats into the whole run's
    // iteration budget, not just its own).
    replace_same_param_only: bool,
    // Whether `replace_pool`'s degenerate-pool fallback (a position whose own `by_params`
    // group is a singleton) may widen to the full catalog. Only safe when the model is
    // `TransitionCardinality::Flexible` (optw-style "selection" domains, where every group
    // is naturally a singleton and substituting a structurally different transition is
    // still feasible) -- NOT safe on a `FixedWithReplacement` domain like cvrp, where a
    // position's own group being a singleton (e.g. a customer with no via-depot
    // alternative) doesn't change that substituting a DIFFERENT customer's transition
    // there is just as infeasible as it would be everywhere else. Computed once at
    // construction from the same `classify_transition_cardinality` call that decides
    // whether to auto-enable `replace_same_param_only` in the first place.
    allow_full_catalog_replace_fallback: bool,
    // Set via `set_turn_deadline` by a caller time-slicing this solver against another `Search`
    // impl (see dual_bound_position_lns_local_search.rs's bandit wrapper). An elapsed-time value
    // (same clock as `time_keeper.elapsed_time()`), not a duration -- checked every loop iteration
    // alongside the existing overall time_limit check, so a caller can get control back well
    // before `time_limit` is up without losing any internal state (current solution, neighborhood
    // bandit stats, RNG, SA temperature schedule, ...): unlike stopping via `time_limit` itself,
    // which this struct treats as "genuinely done," hitting a turn deadline returns
    // `terminated: false`, the same signal already used for "found a new best," so a caller that
    // doesn't use turn deadlines (passes `None`, the default) sees no behavior change at all.
    turn_deadline: Option<f64>,
}

impl<T, B> ModelAwareLocalSearch<T, B>
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
        parameters: ModelAwareLocalSearchParameters<T>,
    ) -> ModelAwareLocalSearch<T, B> {
        assert!(
            parameters.neighborhoods.swap
                || parameters.neighborhoods.relocate
                || parameters.neighborhoods.replace
                || parameters.neighborhoods.twoopt,
            "at least one neighborhood (swap, relocate, replace, or twoopt) must be enabled"
        );
        match parameters.neighborhood_selection {
            ModelAwareNeighborhoodSelection::Sequential { iterations } => {
                assert!(
                    iterations >= 1,
                    "iterations per neighborhood must be at least 1 in ModelAwareNeighborhoodSelection::Sequential"
                );
            }
            ModelAwareNeighborhoodSelection::Adaptive { iterations, exploration_constant, window_size } => {
                assert!(
                    iterations >= 1,
                    "iterations per neighborhood must be at least 1 in ModelAwareNeighborhoodSelection::Adaptive"
                );
                assert!(
                    exploration_constant >= 0.0,
                    "exploration_constant must be non-negative in ModelAwareNeighborhoodSelection::Adaptive"
                );
                assert!(
                    window_size >= 1,
                    "window_size must be at least 1 in ModelAwareNeighborhoodSelection::Adaptive"
                );
            }
            ModelAwareNeighborhoodSelection::Random => {}
        }
        let mut enabled_neighborhood_kinds: Vec<NeighborhoodKind> = [
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
        // Grounded transition catalog (forward transitions, then forced ones, each keyed
        // by its position in that chain), used for: the candidate pool for the replace
        // neighborhood, and -- via `catalog_lookup`, needed only here -- attaching each
        // initial transition's identity once at construction. That lookup is the one
        // unavoidable cost (the caller only hands in plain `Transition`s, with no
        // identity attached), paid exactly once per solver construction rather than
        // once per accepted move: from here on, `current`'s elements carry their own
        // `(id, forced)` and every neighborhood operation just moves that data around
        // instead of re-deriving it.
        let mut catalog_lookup: FxHashMap<(String, Vec<Element>), (bool, usize)> =
            FxHashMap::default();
        let mut candidate_transitions: Vec<TransitionWithId> = Vec::with_capacity(
            model.forward_transitions.len() + model.forward_forced_transitions.len(),
        );
        for (id, t) in model.forward_transitions.iter().enumerate() {
            catalog_lookup.insert((t.name.clone(), t.parameter_values.clone()), (false, id));
            candidate_transitions.push(TransitionWithId {
                id,
                forced: false,
                transition: t.clone(),
            });
        }
        for (id, t) in model.forward_forced_transitions.iter().enumerate() {
            catalog_lookup.insert((t.name.clone(), t.parameter_values.clone()), (true, id));
            candidate_transitions.push(TransitionWithId {
                id,
                forced: true,
                transition: t.clone(),
            });
        }

        let current: Vec<TransitionWithId> = transitions
            .iter()
            .map(|t| {
                let &(forced, id) = catalog_lookup
                    .get(&(t.name.clone(), t.parameter_values.clone()))
                    .unwrap_or_else(|| {
                        panic!(
                            "transition `{}` in the initial solution is not in the model's grounded catalog",
                            t.name
                        )
                    });
                TransitionWithId {
                    id,
                    forced,
                    transition: t.clone(),
                }
            })
            .collect();

        // Some catalog transitions can never be applicable in any reachable state at all,
        // which the raw catalog count below doesn't account for. A grounded transition's
        // parameters are grounded over the WHOLE underlying object type regardless of the
        // target state (e.g. tsptw/m-pdtsp's `visit(to)` is grounded for every customer
        // including the depot), and applicability against the *current* state is instead
        // enforced separately via `elements_in_set_variable` -- "parameter value `element`
        // must currently be a member of set variable `var_id`" (see `Transition`'s doc).
        // If a set variable is never the target of an effect that could add an element to
        // it anywhere in the whole catalog, it can only ever shrink from its initial
        // (`model.target`) value for the rest of the search -- nothing ever puts an element
        // back in. So a transition requiring membership of an element that isn't in that
        // variable's initial value, for such a variable, is provably dead: e.g.
        // `visit(depot)`, since `depot` was never in `unvisited` to begin with and no effect
        // ever adds to `unvisited`.
        //
        // Whitelist, not blacklist: the only effect shape recognized as provably
        // non-adding is `(remove element var)` where the operand being removed from is a
        // direct reference back to `var` itself (the ordinary "shrink this set" pattern,
        // e.g. tsptw's `(remove to unvisited)`) -- removing from a set can only ever
        // shrink it. Every other shape -- `Add` (obviously), a `Remove` whose operand isn't
        // simply the variable itself, a `Union`/`Intersection`/table-based/complement
        // expression, or anything else this doesn't specifically recognize -- defaults to
        // "could add," so this can only under-count dead transitions, never wrongly exclude
        // a live one (a variable it wrongly treats as "could still gain elements" just
        // means Replace stays a candidate there, same as today's behavior).
        // Same-parameter grouping used by `replace_same_param_only` -- same key
        // (`parameter_values`) `position_lns.rs`'s `compute_alternatives` groups its
        // `by_params` map by. Computed up front (not just below where it's stored) so the
        // degenerate-pool check right after can use it.
        let mut by_params: FxHashMap<Vec<Element>, Vec<TransitionWithId>> = FxHashMap::default();
        for t in &candidate_transitions {
            by_params
                .entry(t.transition.parameter_values.clone())
                .or_default()
                .push(t.clone());
        }
        let mut replace_same_param_only = std::env::var("DIDP_MALS_REPLACE_SAME_PARAM")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Whether substituting a position for any OTHER catalog transition is ever
        // structurally feasible -- see `classify_transition_cardinality`'s doc for the full
        // reasoning. Computed whenever Replace is enabled at all, regardless of
        // `replace_same_param_only`'s starting value, since it also governs
        // `allow_full_catalog_replace_fallback` below (relevant even when the YAML/env var
        // already requested same-param-only mode).
        let cardinality = enabled_neighborhood_kinds
            .contains(&NeighborhoodKind::Replace)
            .then(|| {
                classify_transition_cardinality::<T>(
                    &model,
                    candidate_transitions.iter().map(|t| &t.transition),
                    &transitions,
                )
            });

        // Auto-adjust full-catalog Replace (`replace_same_param_only == false` so far --
        // if the env var already requested same-param-only, leave that choice alone here).
        // `FixedWithReplacement` (e.g. cvrp's `visit`/`visit-via-depot`, mdkp's
        // `pack`/`ignore`): full-catalog substitution is infeasible, but *same-param*
        // substitution is exactly the valid move -- so switch to `replace_same_param_only`
        // instead of dropping Replace outright, rather than silently leaving these domains
        // with no replacement-style move at all. `FixedPermutation` has no such fallback
        // (every group is a singleton, so same-param-only would be a pure no-op there);
        // Replace is simply disabled there, unless it's the only neighborhood left enabled.
        if !replace_same_param_only {
            match cardinality {
                Some(TransitionCardinality::FixedWithReplacement) => {
                    replace_same_param_only = true;

                    if !parameters.parameters.quiet {
                        println!(
                            "detected FixedWithReplacement-shaped initial solution ({} transitions) -- restricting replace neighborhood to same-parameter alternatives",
                            current.len()
                        );
                    }
                }
                Some(TransitionCardinality::FixedPermutation) => {
                    let other_enabled = enabled_neighborhood_kinds
                        .iter()
                        .any(|&kind| kind != NeighborhoodKind::Replace);

                    if other_enabled {
                        enabled_neighborhood_kinds.retain(|&kind| kind != NeighborhoodKind::Replace);

                        if !parameters.parameters.quiet {
                            println!(
                                "detected FixedPermutation-shaped initial solution ({} transitions) -- disabling replace neighborhood",
                                current.len()
                            );
                        }
                    }
                }
                Some(TransitionCardinality::Flexible) | None => {}
            }
        }

        // See `allow_full_catalog_replace_fallback`'s doc: only a `Flexible` model (or
        // Replace not enabled at all, in which case this is unused) may widen a degenerate
        // same-param pool back out to the full catalog.
        let allow_full_catalog_replace_fallback =
            matches!(cardinality, Some(TransitionCardinality::Flexible) | None);

        let successor_generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
        let has_forced_or_dominance = !model.forward_forced_transitions.is_empty()
            || !model.backward_forced_transitions.is_empty()
            || !model.transition_dominance.is_empty();

        let neighborhood_visits = vec![0; enabled_neighborhood_kinds.len()];
        let neighborhood_avg_reward = vec![0.0; enabled_neighborhood_kinds.len()];
        let neighborhood_reward_sum = vec![0.0; enabled_neighborhood_kinds.len()];
        let neighborhood_ever_selected = vec![false; enabled_neighborhood_kinds.len()];

        // remaining time limit after dual_bound_local_search, if any, is passed to the local search
        let mut local_search = ModelAwareLocalSearch {
            model,
            base_cost_evaluator,
            root_cost,
            current_cost: cost.unwrap_or(root_cost),
            current,
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
            successor_generator,
            dominance_pruning: parameters.dominance_pruning,
            has_forced_or_dominance,
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
            rng: Pcg64Mcg::seed_from_u64(parameters.seed),
            time_keeper: TimeKeeper::with_time_limit(time_limit),
            quiet: parameters.parameters.quiet,
            first_call: true,
            candidate_transitions,
            by_params,
            replace_same_param_only,
            allow_full_catalog_replace_fallback,
            turn_deadline: None,
        };

        if matches!(parameters.neighborhood_selection, ModelAwareNeighborhoodSelection::Adaptive { .. }) {
            local_search.neighborhood_cursor = local_search.sample_neighborhood(false);
        }

        if solvable {
            local_search.refresh_trace(0);
        }
        local_search.time_keeper.stop();

        local_search
    }

    // Recomputes the state and cost for every position from `prefix_len` onward,
    // reusing `current_states`/`current_costs`'s already-correct entries before it.
    // Every neighborhood kind changes only a single contiguous run starting at
    // `prefix_len` -- the same scope `evaluate` itself rolls out from -- so
    // everything before it is guaranteed unchanged by whatever move was just
    // accepted, and only the changed suffix needs a fresh rollout instead of
    // replaying the whole trajectory from the root on every accepted move.
    fn refresh_trace(&mut self, prefix_len: usize) {
        let (state, cost) = if prefix_len == 0 {
            (self.model.target.clone(), self.root_cost)
        } else {
            (
                self.current_states[prefix_len - 1].clone(),
                self.current_costs[prefix_len - 1],
            )
        };
        let (states, costs): (Vec<_>, Vec<_>) =
            get_trace(&state, cost, &self.current[prefix_len..], &self.model).unzip();
        self.current_states.truncate(prefix_len);
        self.current_states.extend(states);
        self.current_costs.truncate(prefix_len);
        self.current_costs.extend(costs);
    }

    fn generate_neighbor(&mut self) -> (Vec<TransitionWithId>, usize, NeighborhoodKind) {
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
                // Avoids cloning the whole pool for a single sample: look up its length,
                // draw an index, then clone just that one element (two `by_params` lookups
                // in the same-param case instead of one, but O(1) clone instead of O(pool
                // size) -- this fires on every Replace proposal, so for a large catalog the
                // difference is real, not cosmetic).
                //
                // Falls back to the full catalog -- not just when there's no `by_params`
                // entry at all, but also when the entry exists with only ONE sibling (the
                // transition currently at this position, with nothing else sharing its
                // parameters). A size-1 group is a guaranteed no-op under the restricted
                // pool, which on selection-style domains (e.g. optw's "which node to visit"
                // choice) would throw away a genuinely useful move -- full-catalog Replace
                // still substitutes in some structurally different transition there, it
                // just doesn't happen to share this position's exact parameter_values.
                let param_values = &transitions[i].transition.parameter_values;
                let same_param_siblings = self
                    .replace_same_param_only
                    .then(|| self.by_params.get(param_values))
                    .flatten()
                    .filter(|siblings| siblings.len() > 1);
                let pool_len = same_param_siblings.map_or(self.candidate_transitions.len(), Vec::len);
                let index = self.rng.random_range(0..pool_len);
                let replacement = same_param_siblings
                    .map(|siblings| siblings[index].clone())
                    .unwrap_or_else(|| self.candidate_transitions[index].clone());
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
            ModelAwareNeighborhoodSelection::Random => self.enabled_neighborhood_kinds.clone(),
            ModelAwareNeighborhoodSelection::Sequential { .. } | ModelAwareNeighborhoodSelection::Adaptive { .. } => {
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
            ModelAwareNeighborhoodSelection::Adaptive { exploration_constant, .. } => exploration_constant,
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
        if let ModelAwareNeighborhoodSelection::Adaptive { window_size, .. } = self.neighborhood_selection {
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
            ModelAwareNeighborhoodSelection::Random => self.neighborhood_cursor,
            ModelAwareNeighborhoodSelection::Sequential { .. } => {
                (self.neighborhood_cursor + 1) % self.enabled_neighborhood_kinds.len()
            }
            ModelAwareNeighborhoodSelection::Adaptive { .. } => self.sample_neighborhood(exclude_current),
        };
    }

    fn record_outcome(&mut self, reward: f64) {
        self.segment_score += reward;

        let iterations = match self.neighborhood_selection {
            ModelAwareNeighborhoodSelection::Random => return,
            ModelAwareNeighborhoodSelection::Sequential { iterations } => iterations,
            ModelAwareNeighborhoodSelection::Adaptive { iterations, .. } => iterations,
        };
        self.neighborhood_iterations_done += 1;

        if self.neighborhood_iterations_done >= iterations {
            self.switch_to_next_neighborhood(false);
        }
    }

    fn advance_on_exhaustion(&mut self) -> bool {
        match self.neighborhood_selection {
            ModelAwareNeighborhoodSelection::Random => false,
            ModelAwareNeighborhoodSelection::Sequential { .. } | ModelAwareNeighborhoodSelection::Adaptive { .. } => {
                self.neighborhood_iterations_done += 1;
                self.stale_neighborhoods += 1;
                let has_more = self.stale_neighborhoods < self.enabled_neighborhood_kinds.len();
                let exclude_current = matches!(self.neighborhood_selection, ModelAwareNeighborhoodSelection::Adaptive { .. });
                self.switch_to_next_neighborhood(exclude_current);

                has_more
            }
        }
    }

    // How many `evaluate` calls between time-budget checks inside `best_neighbor`'s
    // O(n^2)-candidate loops. Each `evaluate` call itself replays transitions[prefix_len..]
    // (see its doc), so a single sweep is effectively O(n^3) on a large instance -- without a
    // check inside the loop (not just once per `search_next` iteration), a single
    // best_neighbor() call on e.g. a 1000-transition instance with twoopt active can run for
    // minutes past the configured time_limit before anyone notices. 1024 is cheap to check
    // (one Instant::now() per 1024 evaluations) while keeping worst-case overrun small
    // relative to realistic per-evaluate costs.
    const TIME_CHECK_INTERVAL: u64 = 1024;

    // Best-improvement neighborhood exploration for hill climbing: evaluates every
    // active neighborhood kind's moves (O(n^2) candidates each) and returns the best
    // feasible one found before either exhausting them all or running out of time --
    // see TIME_CHECK_INTERVAL's doc for why a mid-sweep check is necessary, not just
    // defensive. A time-cut sweep returns whatever `best` it found so far, same as
    // exhausting the sweep normally would with nothing better available.
    fn best_neighbor(&mut self) -> Option<(Vec<TransitionWithId>, usize, T)> {
        let n = self.current.len();
        let mut best: Option<(Vec<TransitionWithId>, usize, T)> = None;
        let active = self.active_neighborhood_kinds();
        let mut evaluated: u64 = 0;

        if active.contains(&NeighborhoodKind::Swap) {
            let mut candidate = self.current.clone();

            'swap: for i in 0..n {
                for j in (i + 1)..n {
                    candidate.swap(i, j);

                    if self.passes_structural_pruning(&candidate, i) {
                        if let Some(cost) = self.evaluate(&candidate, i) {
                            if best
                                .as_ref()
                                .map_or(true, |&(_, _, best_cost)| self.is_better(cost, best_cost))
                            {
                                best = Some((candidate.clone(), i, cost));
                            }
                        }
                        evaluated += 1;
                    }

                    candidate.swap(i, j); // undo the swap

                    if evaluated % Self::TIME_CHECK_INTERVAL == 0 && self.time_keeper.check_time_limit(true) {
                        break 'swap;
                    }
                }
            }
        }

        if active.contains(&NeighborhoodKind::Relocate) {
            let mut candidate = self.current.clone();

            'relocate: for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let transition = candidate.remove(i);
                    candidate.insert(j, transition);
                    let prefix_len = i.min(j);

                    if self.passes_structural_pruning(&candidate, prefix_len) {
                        if let Some(cost) = self.evaluate(&candidate, prefix_len) {
                            if best
                                .as_ref()
                                .map_or(true, |&(_, _, best_cost)| self.is_better(cost, best_cost))
                            {
                                best = Some((candidate.clone(), prefix_len, cost));
                            }
                        }
                        evaluated += 1;
                    }

                    let transition = candidate.remove(j);
                    candidate.insert(i, transition); // undo the relocation

                    if evaluated % Self::TIME_CHECK_INTERVAL == 0 && self.time_keeper.check_time_limit(true) {
                        break 'relocate;
                    }
                }
            }
        }
        if active.contains(&NeighborhoodKind::Replace) {
            let mut candidate = self.current.clone();

            // Default (non-same-param) pool is the same at every position -- clone it once
            // up front, same as the pre-repair/-replace-arm code did, instead of once per
            // position (only the same-param arm genuinely needs a fresh lookup per i, since
            // its pool depends on what's currently at that position).
            let shared_pool = (!self.replace_same_param_only).then(|| self.candidate_transitions.clone());

            'replace: for i in 0..n {
                let per_position_pool;
                let pool: &Vec<TransitionWithId> = match &shared_pool {
                    Some(shared) => shared,
                    None => {
                        per_position_pool = self.replace_pool(&candidate[i]).clone();
                        &per_position_pool
                    }
                };

                for transition in pool {
                    let original_transition = candidate[i].clone();
                    candidate[i] = transition.clone();

                    if self.passes_structural_pruning(&candidate, i) {
                        if let Some(cost) = self.evaluate(&candidate, i) {
                            if best
                                .as_ref()
                                .map_or(true, |&(_, _, best_cost)| self.is_better(cost, best_cost))
                            {
                                best = Some((candidate.clone(), i, cost));
                            }
                        }
                        evaluated += 1;
                    }

                    candidate[i] = original_transition; // undo the replacement

                    if evaluated % Self::TIME_CHECK_INTERVAL == 0 && self.time_keeper.check_time_limit(true) {
                        break 'replace;
                    }
                }
            }
        }
        if active.contains(&NeighborhoodKind::TwoOpt) {
            let mut candidate = self.current.clone();

            'twoopt: for i in 0..n {
                // j = i + 1 would reverse a single-element slice (a no-op), so skip it.
                for j in (i + 2)..n {
                    candidate[i + 1..=j].reverse();

                    if self.passes_structural_pruning(&candidate, i + 1) {
                        if let Some(cost) = self.evaluate(&candidate, i + 1) {
                            if best
                                .as_ref()
                                .map_or(true, |&(_, _, best_cost)| self.is_better(cost, best_cost))
                            {
                                best = Some((candidate.clone(), i + 1, cost));
                            }
                        }
                        evaluated += 1;
                    }

                    candidate[i + 1..=j].reverse();

                    if evaluated % Self::TIME_CHECK_INTERVAL == 0 && self.time_keeper.check_time_limit(true) {
                        break 'twoopt;
                    }
                }
            }
        }

        best
    }

    // Candidate pool for the Replace neighborhood at a position currently holding `current`:
    // the full grounded catalog by default, or (DIDP_MALS_REPLACE_SAME_PARAM=1, or
    // auto-restricted for a `FixedWithReplacement` model -- see `replace_same_param_only`'s
    // doc) just `current`'s same-parameter siblings. Falls back to the full catalog if
    // `current`'s parameters happen to have no `by_params` entry, or if that entry has only
    // one sibling (current's own transition, with no genuine alternative) -- but ONLY when
    // `allow_full_catalog_replace_fallback` says that's actually safe (a genuinely
    // `Flexible`/selection-style model, where substituting a structurally different
    // transition is still feasible). On a `FixedWithReplacement` model a degenerate
    // size-1 group is just as infeasible to broaden as any other position, so the fallback
    // there instead returns that size-1 group itself -- a harmless no-op candidate, not a
    // reintroduction of full-catalog substitution's infeasibility.
    fn replace_pool(&self, current: &TransitionWithId) -> &Vec<TransitionWithId> {
        if self.replace_same_param_only {
            if let Some(siblings) = self.by_params.get(&current.transition.parameter_values) {
                if siblings.len() > 1 || !self.allow_full_catalog_replace_fallback {
                    return siblings;
                }
            }
        } else {
            return &self.candidate_transitions;
        }

        &self.candidate_transitions
    }

    // Rollout from prefix to end & returns cost
    fn evaluate(&mut self, transitions: &[TransitionWithId], prefix_len: usize) -> Option<T> {
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


    // Cheap pre-`evaluate` gate: does the state right before `prefix_len` -- the first
    // position a candidate actually changes, always known without a rollout since
    // everything before it is untouched -- already rule out `transitions[prefix_len]`
    // via the model's own forced-transition/transition_dominance semantics? Both are
    // optimality-preserving in DIDP (not heuristics): a forced transition applicable at a
    // state is provably part of some optimal continuation, so taking anything else there
    // cannot be part of an optimal solution (this is exactly why SuccessorGenerator's
    // forced-transition short-circuit is safe to use for forward-expansion pruning at
    // all -- see successor_generator.rs), and `transition_dominance` conditions are
    // declared by the model author with the same "never worse to prefer the dominating
    // one" guarantee. Every neighborhood kind (swap/relocate/replace/twoopt) changes
    // exactly one contiguous run starting at `prefix_len`, so checking only that first
    // position is the natural, cheap boundary of what's knowable without simulating the
    // rest of the candidate -- same scope `evaluate` itself rolls out from.
    //
    // A `false` here means the candidate is skipped without ever calling `evaluate` on it --
    // cheaper than the rollout it replaces, not just an additional filter on top of it.
    fn passes_structural_pruning(&mut self, transitions: &[TransitionWithId], prefix_len: usize) -> bool {
        if !self.dominance_pruning || !self.has_forced_or_dominance || prefix_len >= transitions.len() {
            return true;
        }

        let state = if prefix_len == 0 {
            self.model.target.clone()
        } else {
            self.current_states[prefix_len - 1].clone()
        };
        let mut function_cache = StateFunctionCache::new(&self.model.state_functions);
        let mut applicable = Vec::new();
        self.successor_generator
            .generate_applicable_transitions(&state, &mut function_cache, &mut applicable);

        let placed = &transitions[prefix_len];
        applicable
            .iter()
            .any(|t| t.forced == placed.forced && t.id == placed.id)
    }

    // Decides whether a candidate with the given cost replaces `current`.
    fn accept(&mut self, cost: T) -> bool {
        if !exceed_bound(&self.model, cost, Some(self.current_cost)) {
            return true;
        }

        match self.mode {
            ModelAwareLocalSearchMode::HillClimbing => false,
            ModelAwareLocalSearchMode::SimulatedAnnealing { .. } => {
                let delta = if self.model.reduce_function == dypdl::ReduceFunction::Max {
                    self.current_cost - cost
                } else {
                    cost - self.current_cost
                }
                .to_continuous();

                self.rng.random::<f64>() < (-delta / self.sa_temperature()).exp()
            }
        }
    }

    // Time-based cooling schedule: temperature = T0 * final_temp_ratio ^
    // (elapsed / time_limit) -- see ModelAwareLocalSearchMode::SimulatedAnnealing's doc
    // for why this replaced a per-call multiplicative `temperature *= alpha`
    // decay. elapsed_time() is TimeKeeper's accumulated active time, which is
    // exactly what time_limit is measured against everywhere else in this
    // file (check_time_limit, remaining_time_limit).
    fn sa_temperature(&self) -> f64 {
        let ModelAwareLocalSearchMode::SimulatedAnnealing { T0, final_temp_ratio } = self.mode else {
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
        if let ModelAwareNeighborhoodSelection::Adaptive { .. } = self.neighborhood_selection {
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

    /// Sets an elapsed-time value (same clock as `time_keeper.elapsed_time()`) at which the next
    /// call(s) to `search_next` should return early with `terminated: false`, without waiting for
    /// `time_limit` or a genuine local optimum. `None` (the default) disables this and reproduces
    /// the original behavior exactly. See the `turn_deadline` field's doc for the intended use
    /// (time-slicing this solver against another `Search` impl without losing internal state
    /// between turns).
    pub fn set_turn_deadline(&mut self, deadline: Option<f64>) {
        self.turn_deadline = deadline;
    }

    /// Adopts `transitions`/`cost` as the current working solution and incumbent if `cost` is
    /// better than this instance's own `best.cost` (a no-op otherwise). See `PositionLns`'s
    /// identically-purposed method: lets an instance stuck at a local optimum resume from a better
    /// solution found by another `Search` impl in the meantime, without reconstructing it (which
    /// would also discard `set_turn_deadline`'s doc's list of things worth preserving -- the
    /// SW-UCB neighborhood bandit's learned stats, in this struct's case). `transitions` are
    /// looked up against `candidate_transitions` to attach each one's catalog id/forced flag,
    /// exactly as the constructor does for the initial solution -- `current` needs that identity,
    /// plain `Transition`s (what every other `Search` impl's `Solution` carries) don't have it.
    /// Resets `stale_neighborhoods` since the local-optimum proof `advance_on_exhaustion` was
    /// building no longer applies to the new incumbent.
    pub fn adopt_incumbent(&mut self, transitions: Vec<Transition>, cost: T, time: f64) {
        if self.best.cost.is_some_and(|best_cost| !self.is_better(cost, best_cost)) {
            return;
        }

        // `solvable` is otherwise fixed at construction from the INITIAL solution. If that was too
        // short to work with (e.g. an empty solution on a variable-length domain like optw, where the
        // first CABS round only finds cost 0), this instance would report "terminated" on every
        // turn forever, even after an adopted incumbent gives it something real to improve. The
        // constructor's one-time first_call echo is also skipped for an unsolvable instance
        // (search_next returns before reaching it), so consume it here or the first real turn would
        // be wasted re-echoing the incumbent.
        if !self.solvable && transitions.len() >= 2 {
            self.solvable = true;
            self.first_call = false;
        }

        let catalog_lookup: FxHashMap<(String, Vec<Element>), (bool, usize)> = self
            .candidate_transitions
            .iter()
            .map(|t| {
                (
                    (t.transition.name.clone(), t.transition.parameter_values.clone()),
                    (t.forced, t.id),
                )
            })
            .collect();

        self.current = transitions
            .iter()
            .map(|t| {
                let &(forced, id) = catalog_lookup
                    .get(&(t.name.clone(), t.parameter_values.clone()))
                    .unwrap_or_else(|| {
                        panic!(
                            "transition `{}` in an adopted incumbent is not in the model's grounded catalog",
                            t.name
                        )
                    });
                TransitionWithId { id, forced, transition: t.clone() }
            })
            .collect();
        self.current_cost = cost;
        self.refresh_trace(0);
        self.stale_neighborhoods = 0;

        self.best.cost = Some(cost);
        self.best.transitions = transitions;
        self.best.is_infeasible = false;
        self.best.time = time;
    }
}

impl<T, B> Search<T> for ModelAwareLocalSearch<T, B>
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

                return Ok((self.best.clone(), true));
            }

            if self
                .turn_deadline
                .is_some_and(|deadline| self.time_keeper.elapsed_time() >= deadline)
            {
                self.best.time = self.time_keeper.elapsed_time();
                self.time_keeper.stop();

                return Ok((self.best.clone(), false));
            }

            if matches!(self.mode, ModelAwareLocalSearchMode::HillClimbing) {
                if let Some((candidate, prefix_len, cost)) = self.best_neighbor() {
                    if self.accept(cost) {
                        self.current = candidate;
                        self.current_cost = cost;
                        self.refresh_trace(prefix_len);
                        self.stale_neighborhoods = 0;

                        let is_new_best = self.is_better(cost, self.best.cost.unwrap());
                        self.record_outcome(if is_new_best {
                            NEIGHBORHOOD_GLOBAL_BEST_REWARD
                        } else {
                            NEIGHBORHOOD_ACCEPTED_REWARD
                        });

                        if is_new_best {
                            self.best.cost = Some(cost);
                            self.best.transitions = self.current.iter().map(|t| t.transition.clone()).collect();
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

            let (candidate, prefix_len, _kind) = self.generate_neighbor();

            if !self.passes_structural_pruning(&candidate, prefix_len) {
                self.record_outcome(0.0);
                continue; // ruled out by a forced-transition/dominance rule: try another one
            }

            let evaluated = self.evaluate(&candidate, prefix_len);

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
                self.refresh_trace(prefix_len);

                let is_new_best = self.is_better(cost, self.best.cost.unwrap());
                self.record_outcome(if is_new_best {
                    NEIGHBORHOOD_GLOBAL_BEST_REWARD
                } else {
                    NEIGHBORHOOD_ACCEPTED_REWARD
                });

                if is_new_best {
                    self.best.cost = Some(cost);
                    self.best.transitions = self.current.iter().map(|t| t.transition.clone()).collect();
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
