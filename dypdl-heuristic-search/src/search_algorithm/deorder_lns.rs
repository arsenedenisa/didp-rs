// Large neighborhood search whose destroy set is a partial-order relaxation
// of the incumbent, and whose repair step *optimizes* over that relaxation
// (bounded beam search over valid linear extensions) instead of sampling a
// single random one. See deorder_local_search.rs's module doc for the
// diagnosis this is a response to: the sampling-based version's failures
// (diffuse samples, a badness table with nothing reliable to learn) are all
// sampling failures, and optimizing over the neighborhood removes sampling
// from the loop entirely.
//
// The destroy step frees any pair independently -- no freeability cascade.
// deorder_local_search needs the cascade because its repair step is a
// randomized topological sort that only knows how to walk a transitively
// closed partial order; this repair step is a beam search over in_degree,
// which handles an arbitrary DAG, so the cascade was only ever throttling
// how many pairs could be freed, not protecting correctness here.
//
// Free probability is not flat either: it's redistributed across pairs in
// proportion to w[a] + w[c], where w[p] is the marginal cost of the
// transition at position p in the current incumbent (current_costs[p] -
// current_costs[p-1]). This is LNBS's delta=0 rule (Section 4.4) applied at
// pair granularity -- positions that don't move the cost needle get
// (near-)zero probability of having their precedence relaxed, since freeing
// them can't change the objective. The total expected number of frees is
// normalized to match what a flat free_probability coin flip over every
// pair would give, so this is a retargeting of the same destroy budget, not
// a change in how much gets destroyed. There's still no "avoid this pair,
// it tends to hurt" concept -- the repair step evaluates real cost for
// everything it tries and simply won't choose a bad option if a better one
// is in the beam.
//
// The repair step is a beam search over "place the next ready original
// position" (ready = every retained predecessor already placed), checking
// real DyPDL applicability/state-constraints at each step, ranked by
// accumulated cost. This is a self-contained beam search over the fixed
// known transition multiset, not the model's general SuccessorGenerator --
// deliberately simpler than integrating with the FNode/CABS dominance
// machinery, since the candidate pool at each step is already small and
// known (the incumbent's own transitions), not rediscovered from the model.
//
// One exception to "fixed known transition multiset": at each position, the
// beam may substitute any other grounded transition that shares the same
// parameter values (see `alternatives`) -- e.g. CVRP's `visit(to)` and
// `visit-via-depot(to)` both take a single `to` parameter, so they're
// alternatives for the same position, even though they differ in name,
// effect, and cost. This is what lets the repair change route-boundary
// *count*, not just which customers fall on which side of one -- pure
// reordering of a fixed multiset can never do that, since the count of
// route-reset transitions is baked into the multiset itself. The
// destroy poset, in_degree, and cascade are unaffected: they're purely
// positional and don't know or care which transition ends up placed at a
// position, only which positions precede which.

use super::data_structure::exceed_bound;
use super::rollout::get_trace;
use super::search::{Parameters, Search, Solution};
use super::util::{print_primal_bound, TimeKeeper};
use crate::f_evaluator_type::FEvaluatorType;
use dypdl::expression::{
    Condition, ElementExpression, ReferenceExpression, SetCondition, SetElementOperator,
    SetExpression, SetOperator,
};
use dypdl::{
    variable_type::{Element, Numeric},
    Model, ParentAndChildStateFunctionCache, State, StateFunctionCache, Transition,
    TransitionInterface,
};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::str;

/// Parameters for [`DeorderLns`].
#[derive(Debug, Clone, Copy)]
pub struct DeorderLnsParameters<T> {
    /// Random seed.
    pub seed: u64,
    /// Widest position gap considered when freeing a precedence pair.
    /// `None` means unbounded.
    pub max_gap: Option<usize>,
    /// Target expected fraction of all in-`max_gap` pairs freed per destroy
    /// step. No longer a flat per-pair probability: mass is redistributed by
    /// `w[a] + w[c]` (see `build_destroy_poset`), so individual pairs land
    /// above or below this rate, but the expected total count of frees
    /// matches `free_probability * total_pairs`, same as a flat coin flip
    /// would give. Unlike the sampling-based deorder solver, higher is
    /// generally better here (up to what `beam_width` can search
    /// effectively), since the repair step optimizes over the relaxation
    /// rather than sampling blindly from it.
    pub free_probability: f64,
    /// Overrides the expected-frees target used in `build_destroy_poset`
    /// with an absolute count instead of `free_probability * total_pairs`.
    /// Exists to decouple destroy-step *size* from destroy-step *targeting*
    /// when comparing against a run made before the freeability cascade was
    /// removed: the cascade capped how many pairs could ever be freed
    /// regardless of `free_probability`, so matching `free_probability`
    /// alone across a before/after comparison does not hold the destroy
    /// step's size constant -- this does. `None` uses
    /// `free_probability * total_pairs`, the uncapped default.
    pub target_frees: Option<f64>,
    /// Beam width for the repair step.
    pub beam_width: usize,
    /// Cap on how many ready positions to try expanding from a single beam
    /// entry per step, to bound work when the destroy step frees enough
    /// pairs that many positions become ready at once.
    pub max_branching: usize,
    /// Discount applied to `free_probability` for a pair that's "hard" --
    /// provably infeasible to invert (a genuine precedence constraint, not
    /// just costly), per the extraction in `compute_must_precede`. Not 0:
    /// the freeability cascade can require a hard pair to be freed anyway,
    /// as a stepping stone toward freeing a wider pair that's actually
    /// useful (see `build_destroy_poset`); this only makes hard pairs less
    /// likely to be the ones spending the destroy step's budget, not
    /// impossible to free.
    pub gamma: f64,
    pub parameters: Parameters<T>,
}

/// One partial linear extension under construction during the repair beam
/// search: which original positions have been placed (in order), the
/// remaining in-degree of every position under the destroy poset's kept
/// edges, and the resulting DyPDL state/cost of applying them in that order.
#[derive(Clone)]
struct BeamEntry<T> {
    // The actual transition placed at each step, in order -- not
    // necessarily `current[p]` for the position `p` it was placed at (see
    // `alternatives`), so this has to record the transition itself. Rc, not
    // Transition: `entry.clone()` happens for every candidate at every beam
    // step, and this list grows with search depth -- with owned Transitions
    // (full precondition/effect/cost AST trees) that's an O(depth) deep-copy
    // per candidate, which measurably wrecked search throughput across every
    // problem (single-machine/instance_16 at fp=0.3: 188 -> 582) the first
    // time this was tried. Rc::clone is a refcount bump regardless of the
    // underlying Transition's size.
    placed: Vec<Rc<Transition>>,
    // Original position filled at each step, parallel to `placed`. Lets
    // callers recover which precedence pairs the repaired order actually
    // inverted, independent of which transition (original or alternative)
    // ended up there.
    placed_positions: Vec<usize>,
    placed_mask: Vec<bool>,
    in_degree: Vec<usize>,
    state: State,
    cost: T,
    // f = cost combined with the dual-bound estimate of the remaining cost
    // to completion (via f_evaluator_type), used only to rank/prune the beam
    // -- ranking on raw cost alone is a greedy, lookahead-free beam, which
    // measurably underperforms on consequence-sensitive objectives (see the
    // module doc). Equal to `cost` when the model has no dual bounds.
    f: T,
    // True iff this entry has, at every step so far, placed position `k`
    // (in order, k = 0, 1, ...) using exactly `current[k]` -- i.e. followed
    // the incumbent's own original order *and* transitions, never an
    // alternative. The destroy poset is always a relaxation of the
    // incumbent's total order, so this lineage is always extendable (its
    // next position is always ready) and always reaches a feasible
    // completion at exactly `current_cost`. Protected from beam truncation
    // (see beam_repair) so a non-improving iteration is a real negative
    // result -- the beam searched and found nothing better than the
    // incumbent it was guaranteed to have -- rather than an artifact of
    // truncation/shuffle dropping the one candidate that couldn't lose.
    is_incumbent: bool,
}

/// A large neighborhood search whose destroy set is a partial-order
/// relaxation of the incumbent (see the module doc) and whose repair step is
/// a bounded beam search for the best valid linear extension of that
/// relaxation, rather than a single random sample.
pub struct DeorderLns<T: Numeric, B> {
    model: Rc<Model>,
    base_cost_evaluator: B,
    root_cost: T,
    current: Vec<Transition>,
    // Every grounded transition the model can produce, grouped by
    // parameter values -- static, model-wide, unaffected by reordering.
    // Kept so alternatives (below) can be cheaply re-derived whenever
    // `current` changes, instead of only once at construction.
    by_params: HashMap<Vec<Element>, Vec<Rc<Transition>>>,
    // alternatives[p]: every grounded transition sharing current[p]'s
    // parameter values, always including current[p] itself. See the module
    // doc. Rc'd so beam_repair can push a shared reference into a
    // BeamEntry.placed without cloning the underlying Transition -- see
    // BeamEntry.placed's doc.
    //
    // MUST be recomputed every time `current` is reordered (an accepted
    // move permutes which transition sits at which position), not just
    // once at construction -- this was a real, measured bug: alternatives
    // stayed frozen at the construction-time position->identity mapping,
    // so after the first accept, beam_repair silently placed whichever
    // transition originally sat at position p, not whatever current[p]
    // actually held post-reorder. Confirmed by comparing per-iteration
    // beam_repair output between this mechanism and one that reads
    // current[p] directly: byte-identical at iteration 1 (nothing
    // reordered yet), diverging from iteration 2 on (right after the
    // first accept) even under an identical fixed iteration count with
    // wall-clock timing eliminated as a variable.
    alternatives: Vec<Vec<Rc<Transition>>>,
    // transition_ids[p] / must_precede: refreshed on every accept, not
    // per-iteration -- see compute_must_precede's doc. Recomputing this
    // from scratch every iteration means calling get_full_name() (a String
    // allocation) for every position every iteration, which measurably
    // wrecked throughput the first time this was tried (single-machine/
    // instance_16 regressed from 188 to 628 at fp=0.3) -- the same
    // string-allocation anti-pattern already identified and fixed once
    // this session in deorder_local_search.rs. Like `alternatives`, this is
    // position-indexed and goes stale the moment `current` is reordered, so
    // "only recompute on accept" (not "recompute once ever") is load-
    // bearing, not just an optimization.
    transition_ids: Vec<u32>,
    must_precede: HashMap<u32, Vec<u32>>,
    current_cost: T,
    // Cumulative cost after applying current[0..=p], refreshed alongside
    // transition_ids/must_precede/alternatives on every accept (see
    // refresh_trace). Consumed by build_destroy_poset to compute each
    // position's marginal cost contribution -- see the module doc's
    // delta=0 rule paragraph.
    current_costs: Vec<T>,
    best: Solution<T>,
    solvable: bool,
    max_gap: usize,
    free_probability: f64,
    target_frees: Option<f64>,
    beam_width: usize,
    max_branching: usize,
    gamma: f64,
    rng: Pcg64Mcg,
    time_keeper: TimeKeeper,
    quiet: bool,
    first_call: bool,
    f_evaluator_type: FEvaluatorType,
    use_heuristic: bool,
}

// Recognizes precondition shapes TransitionMutex's own get_required_elements
// (transition_mutex.rs) doesn't: is_subset(constant_set, variable) and
// is_empty(intersection(variable, constant_set)) in either argument order,
// both against a table lookup already resolved to a constant set by DyPDL's
// own grounding (confirmed by inspecting the grounded AST directly -- e.g.
// single-machine's `is_subset (predecessors i) scheduled` compiles down to
// exactly IsSubset(Reference(Constant(bitset)), Reference(Variable(id)))).
// Kept as a standalone function, not added to transition_mutex.rs, per the
// instruction to extend additively rather than touch the existing path.
//
// is_subset(required, var): every element of `required` must be present in
// `var` -- a positive requirement on each of those elements.
// is_empty(intersection(var, required)): none of `required`'s elements may
// be present in `var` -- a negative requirement on each of those elements.
// (m-PDTSP's `is_empty (intersection unvisited (predecessors to))` reads as
// "no predecessor of `to` is still unvisited", i.e. every predecessor has
// already been removed from `unvisited` -- negative on `unvisited`, not
// positive; getting this backwards would produce a poset that's actively
// wrong rather than merely unhelpful.)
fn extract_required_elements(condition: &Condition) -> (Vec<(usize, Element)>, Vec<(usize, Element)>) {
    fn constant_and_var<'a>(
        a: &'a SetExpression,
        b: &'a SetExpression,
    ) -> Option<(&'a dypdl::variable_type::Set, usize)> {
        if let (
            SetExpression::Reference(ReferenceExpression::Constant(set)),
            SetExpression::Reference(ReferenceExpression::Variable(var_id)),
        ) = (a, b)
        {
            return Some((set, *var_id));
        }
        None
    }

    let mut positively = Vec::new();
    let mut negatively = Vec::new();

    match condition {
        Condition::Set(c) => match c.as_ref() {
            SetCondition::IsSubset(required, var) => {
                if let Some((set, var_id)) = constant_and_var(required, var) {
                    positively.extend(set.ones().map(|e| (var_id, e)));
                }
            }
            SetCondition::IsEmpty(inner) => {
                if let SetExpression::SetOperation(SetOperator::Intersection, a, b) = inner {
                    if let Some((set, var_id)) =
                        constant_and_var(a, b).or_else(|| constant_and_var(b, a))
                    {
                        negatively.extend(set.ones().map(|e| (var_id, e)));
                    }
                }
            }
            _ => {}
        },
        Condition::Not(inner) => {
            let (neg_positively, neg_negatively) = extract_required_elements(inner);
            positively.extend(neg_negatively);
            negatively.extend(neg_positively);
        }
        _ => {}
    }

    (positively, negatively)
}

// For every position in `transitions`, which catalog (distinct-transition-
// identity) id it holds, and a map from a catalog id to every other catalog
// id that must precede it -- derived from real DyPDL preconditions
// (extract_required_elements plus achiever/remover extraction for the
// affected side), not a heuristic.
//
// Called once at construction and again every time `current` is reassigned
// (an accepted move) -- NOT per LNS iteration, which includes many rejected/
// non-improving beam_repair results that leave `current` untouched. A
// per-iteration version was tried first and measurably wrecked throughput
// (single-machine/instance_16 regressed from 188 to 628 at fp=0.3) because
// it calls get_full_name() -- a String allocation -- for every position,
// every call; the same anti-pattern already identified and fixed once this
// session in deorder_local_search.rs. Accepts are far rarer than iterations,
// so paying this cost only there is cheap. It also has to be recomputed on
// every accept, not just once: transition_ids is a position -> identity
// mapping, and accepting a move reorders (and, via item-4, can substitute)
// what's at each position -- a one-time snapshot went stale after the very
// first accepted move, silently checking is_hard against the wrong mapping
// (this was a real, measured bug, not a hypothetical one: results looked
// like a throughput regression but persisted after fixing the throughput
// issue alone, which is what exposed it).
// alternatives[p]: every grounded transition sharing transitions[p]'s
// parameter values (always including transitions[p] itself), looked up
// from the static, model-wide `by_params` grouping. Must be called again
// every time the position->transition mapping changes (an accepted move
// reorders it) -- see `alternatives`'s doc for why a one-time snapshot is
// wrong, not just stale-and-harmless.
fn compute_alternatives(
    by_params: &HashMap<Vec<Element>, Vec<Rc<Transition>>>,
    transitions: &[Transition],
) -> Vec<Vec<Rc<Transition>>> {
    transitions
        .iter()
        .map(|t| {
            by_params
                .get(&t.parameter_values)
                .cloned()
                .filter(|alts| alts.iter().any(|alt| alt.as_ref() == t))
                .unwrap_or_else(|| vec![Rc::new(t.clone())])
        })
        .collect()
}

fn compute_must_precede(transitions: &[Transition]) -> (Vec<u32>, HashMap<u32, Vec<u32>>) {
    let mut interner: HashMap<String, u32> = HashMap::new();
    let transition_ids: Vec<u32> = transitions
        .iter()
        .map(|t| {
            let next_id = interner.len() as u32;
            *interner.entry(t.get_full_name()).or_insert(next_id)
        })
        .collect();

    let mut catalog: Vec<&Transition> = Vec::new();
    for (t, &id) in transitions.iter().zip(transition_ids.iter()) {
        if id as usize == catalog.len() {
            catalog.push(t);
        }
    }

    let mut achievers: HashMap<(usize, Element), Vec<u32>> = HashMap::new();
    let mut removers: HashMap<(usize, Element), Vec<u32>> = HashMap::new();
    for (id, t) in catalog.iter().enumerate() {
        for (var_id, expr) in &t.effect.set_effects {
            if let SetExpression::SetElementOperation(op, ElementExpression::Constant(e), _) = expr
            {
                let entry = match op {
                    SetElementOperator::Add => achievers.entry((*var_id, *e)),
                    SetElementOperator::Remove => removers.entry((*var_id, *e)),
                };
                entry.or_default().push(id as u32);
            }
        }
    }

    let mut must_precede: HashMap<u32, Vec<u32>> = HashMap::new();
    for (id, t) in catalog.iter().enumerate() {
        for condition in t.get_preconditions() {
            let (positively, negatively) = extract_required_elements(&condition);
            for (var_id, e) in positively {
                if let Some(a) = achievers.get(&(var_id, e)) {
                    if a.len() == 1 {
                        must_precede.entry(id as u32).or_default().push(a[0]);
                    }
                }
            }
            for (var_id, e) in negatively {
                if let Some(r) = removers.get(&(var_id, e)) {
                    if r.len() == 1 {
                        must_precede.entry(id as u32).or_default().push(r[0]);
                    }
                }
            }
        }
    }

    (transition_ids, must_precede)
}

impl<T, B> DeorderLns<T, B>
where
    T: Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    B: FnMut(T, T) -> T,
{
    /// Creates a new deorder LNS starting from `transitions`.
    pub fn new(
        model: Rc<Model>,
        transitions: Vec<Transition>,
        cost: Option<T>,
        initial_time: f64,
        root_cost: T,
        base_cost_evaluator: B,
        parameters: DeorderLnsParameters<T>,
        f_evaluator_type: FEvaluatorType,
    ) -> DeorderLns<T, B> {
        let solvable = cost.is_some() && transitions.len() >= 2;
        let time_limit = parameters.parameters.time_limit.unwrap_or(f64::INFINITY);
        let max_gap = parameters.max_gap.unwrap_or(usize::MAX);
        let use_heuristic = model.has_dual_bounds();

        // Group every grounded transition the model can produce by its
        // parameter values, so e.g. CVRP's visit(to) and
        // visit-via-depot(to) end up in the same bucket for a given `to`.
        // One-time O(all grounded transitions) cost at construction, not
        // per-iteration -- this part genuinely is static and model-wide,
        // unlike `alternatives` itself (see its doc).
        let mut by_params: HashMap<Vec<dypdl::variable_type::Element>, Vec<Rc<Transition>>> =
            HashMap::new();
        for t in model
            .forward_transitions
            .iter()
            .chain(model.forward_forced_transitions.iter())
        {
            by_params
                .entry(t.parameter_values.clone())
                .or_default()
                .push(Rc::new(t.clone()));
        }
        let alternatives = compute_alternatives(&by_params, &transitions);
        let (transition_ids, must_precede) = compute_must_precede(&transitions);

        let mut search = DeorderLns {
            model,
            base_cost_evaluator,
            root_cost,
            by_params,
            alternatives,
            transition_ids,
            must_precede,
            current_cost: cost.unwrap_or(root_cost),
            current_costs: Vec::new(),
            current: transitions.clone(),
            best: Solution {
                cost,
                transitions,
                is_infeasible: cost.is_none(),
                time: initial_time,
                ..Default::default()
            },
            solvable,
            max_gap,
            free_probability: parameters.free_probability,
            target_frees: parameters.target_frees,
            beam_width: parameters.beam_width.max(1),
            max_branching: parameters.max_branching.max(1),
            gamma: parameters.gamma,
            rng: Pcg64Mcg::seed_from_u64(parameters.seed),
            time_keeper: TimeKeeper::with_time_limit(time_limit),
            quiet: parameters.parameters.quiet,
            first_call: true,
            f_evaluator_type,
            use_heuristic,
        };

        if solvable {
            search.refresh_trace();
        }
        search.time_keeper.stop();

        search
    }

    // Refreshes current_costs (cumulative cost after current[0..=p]) to
    // match `current`. Called at construction and after every accepted
    // move, same cadence as transition_ids/must_precede/alternatives --
    // per-iteration would pay a full rollout of `current` for every
    // rejected beam_repair result, and accepts are far rarer than
    // iterations (see compute_must_precede's doc for the same argument).
    fn refresh_trace(&mut self) {
        self.current_costs = get_trace(&self.model.target, self.root_cost, &self.current, &self.model)
            .map(|(_, cost)| cost)
            .collect();
    }

    // Builds the destroy poset: `freed[a][k]` (c = a + 1 + k) is `true` when
    // the precedence between positions a and c has been relaxed. No
    // freeability cascade -- every pair within max_gap is an independent
    // coin flip (see the module doc for why the cascade isn't needed here).
    //
    // The coin isn't flat: pair (a, c)'s probability is proportional to
    // w[a] + w[c], where w[p] = current_costs[p] - current_costs[p-1] is
    // the marginal cost of the transition at position p (LNBS's delta=0
    // rule, Section 4.4, at pair granularity -- a position that adds
    // nothing to the cost can't be a useful thing to relax). Probabilities
    // are scaled so the expected total number of frees equals
    // `free_probability * total_pairs`, i.e. what a flat coin flip over
    // every pair would give -- this redistributes the same destroy budget,
    // it doesn't change its size.
    //
    // Hard pairs (real precedence constraints, per compute_must_precede)
    // get their probability additionally scaled down by `gamma`: measured
    // directly (single-machine 16.4%, m-PDTSP 3.0% of freed pairs were
    // hard, on this repo's benchmark instances -- see the session's
    // DIDP_HARDSOFT_FRACTION measurement), so this isn't a hypothetical
    // saving. Not scaled to 0, since a hard pair can still be worth freeing
    // in its own right now that there's no cascade requiring it as a
    // stepping stone.
    fn build_destroy_poset(&mut self, n: usize) -> Vec<Vec<bool>> {
        let max_gap = self.max_gap.min(n.saturating_sub(1));
        let mut freed: Vec<Vec<bool>> = (0..n)
            .map(|a| vec![false; max_gap.min(n - 1 - a)])
            .collect();

        let total_pairs: usize = freed.iter().map(|row| row.len()).sum();
        if total_pairs == 0 {
            return freed;
        }

        let is_hard = |a: usize, c: usize| -> bool {
            let (ida, idc) = (self.transition_ids[a], self.transition_ids[c]);
            self.must_precede
                .get(&idc)
                .is_some_and(|before| before.contains(&ida))
        };

        // w[p]: non-negative marginal cost of current[p]. Clamped at 0
        // rather than passed through raw -- a negative delta (e.g. a
        // Max-reduced objective that a later transition doesn't raise
        // further) isn't "this position undoes cost", it's "this position
        // didn't cost anything on top of what preceded it", same as an
        // exact 0 for weighting purposes.
        let mut w = vec![0.0f64; n];
        let mut prev = self.root_cost.to_continuous();
        for p in 0..n {
            let cur = self.current_costs[p].to_continuous();
            w[p] = (cur - prev).max(0.0);
            prev = cur;
        }

        let mut pair_weight: Vec<Vec<f64>> = (0..n)
            .map(|a| vec![0.0; max_gap.min(n - 1 - a)])
            .collect();
        let mut weight_sum = 0.0f64;
        for a in 0..n {
            for (k, slot) in pair_weight[a].iter_mut().enumerate() {
                let c = a + 1 + k;
                *slot = w[a] + w[c];
                weight_sum += *slot;
            }
        }

        // Target expected count, not a per-pair rate -- see the doc above.
        // target_frees (if set) pins this to an absolute count instead, so
        // destroy-step size can be held constant independent of
        // total_pairs -- see its doc.
        let target_frees = self
            .target_frees
            .unwrap_or(self.free_probability * total_pairs as f64);

        thread_local! {
            static LOGGED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        let log_this_call =
            std::env::var("DIDP_FREEABLE_DIAG").is_ok() && !LOGGED.with(|c| c.get());
        if log_this_call {
            LOGGED.with(|c| c.set(true));
        }
        let mut freed_count = 0u64;

        for a in 0..n {
            for k in 0..freed[a].len() {
                let c = a + 1 + k;
                // weight_sum == 0.0 means every position in range has zero
                // marginal cost (e.g. a fully degenerate trace) -- fall
                // back to the flat rate rather than divide by zero.
                let base_p = if weight_sum > 0.0 {
                    target_frees * pair_weight[a][k] / weight_sum
                } else {
                    self.free_probability
                };
                let p = if is_hard(a, c) {
                    base_p * self.gamma
                } else {
                    base_p
                }
                .min(1.0);

                if self.rng.random::<f64>() < p {
                    freed[a][k] = true;
                    if log_this_call {
                        freed_count += 1;
                    }
                }
            }
        }

        if log_this_call {
            eprintln!(
                "[freeable diag] total_pairs={} freed={} ({:.1}% of total)",
                total_pairs,
                freed_count,
                100.0 * freed_count as f64 / total_pairs.max(1) as f64,
            );
        }

        freed
    }

    // f = g + h ranking value for a beam entry, via the model's dual bound
    // evaluated on `state` as the heuristic estimate of remaining cost. Falls
    // back to `cost` alone when the model has no dual bounds, so this is a
    // no-op (pure-cost ranking) for models without one.
    fn eval_f(&self, cost: T, state: &State) -> T {
        if !self.use_heuristic {
            return cost;
        }

        let mut function_cache = StateFunctionCache::new(&self.model.state_functions);
        match self.model.eval_dual_bound::<State, T>(state, &mut function_cache) {
            Some(h) => self.f_evaluator_type.eval(cost, h),
            None => cost,
        }
    }

    // A tighter alternative to eval_f, available only because of the
    // fixed-multiset restriction: unlike the model's own dual bound, which
    // has to reason about an unknown suffix, this beam knows exactly which
    // original transitions remain unplaced. Simulates completing with them
    // in their original relative order -- always a valid linear extension
    // of the kept precedence edges (see is_incumbent's doc: precedence only
    // ever points from an earlier original position to a later one, so
    // appending unplaced positions in increasing order can never violate a
    // kept edge), though it can still hit an infeasible DyPDL state
    // (capacity/resource constraints the poset doesn't encode) -- falls
    // back to eval_f in that case. When it succeeds, this is a real, valid
    // total completion cost, not just an estimate: for placed_mask all
    // false (the beam root), it's exactly self.current_cost, replaying the
    // incumbent's own order.
    //
    // Capped to at most MULTISET_LOOKAHEAD simulated positions: this method
    // is called for every beam candidate at every step, so its O(remaining)
    // cost compounds into O(n^2) per repair on large-n problems (measured:
    // collapsed mosp to ~2 iterations in 15s). Near the end of a repair, few
    // positions remain and this still simulates all of them exactly as
    // before -- the cap only bites early on, where the fallback to eval_f
    // (the model's own dual bound) is a reasonable stand-in for a suffix
    // this deep.
    const MULTISET_LOOKAHEAD: usize = 24;

    fn eval_f_multiset(&self, cost: T, state: &State, placed_mask: &[bool]) -> T {
        let mut sim_state = state.clone();
        let mut sim_cost = cost;
        let mut simulated = 0usize;

        for (p, &placed) in placed_mask.iter().enumerate() {
            if placed {
                continue;
            }

            if simulated >= Self::MULTISET_LOOKAHEAD {
                return self.eval_f(sim_cost, &sim_state);
            }
            simulated += 1;

            let transition = &self.current[p];
            let mut function_cache =
                ParentAndChildStateFunctionCache::new(&self.model.state_functions);

            if !transition.is_applicable(
                &sim_state,
                &mut function_cache.child,
                &self.model.state_functions,
                &self.model.table_registry,
            ) {
                return self.eval_f(cost, state);
            }

            let new_state = transition.apply(
                &sim_state,
                &mut function_cache.child,
                &self.model.state_functions,
                &self.model.table_registry,
            );

            if !self
                .model
                .check_constraints(&new_state, &mut function_cache.child)
            {
                return self.eval_f(cost, state);
            }

            sim_cost = transition.eval_cost(
                sim_cost,
                &sim_state,
                &mut function_cache.child,
                &self.model.state_functions,
                &self.model.table_registry,
            );
            sim_state = new_state;
        }

        sim_cost
    }

    // Materializes the kept-edge graph over all n positions: successors[a]
    // lists every c with the precedence (a, c) still forced, including pairs
    // beyond max_gap (always forced, since the destroy step never considers
    // them). O(n^2), acceptable given the repair step budgets iterations in
    // the hundreds, not thousands.
    fn build_successors(&self, n: usize, freed: &[Vec<bool>]) -> (Vec<Vec<usize>>, Vec<usize>) {
        let max_gap = self.max_gap.min(n.saturating_sub(1));
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];

        for a in 0..n {
            for c in (a + 1)..n {
                let gap = c - a;
                let kept = if gap <= max_gap { !freed[a][gap - 1] } else { true };
                if kept {
                    successors[a].push(c);
                    in_degree[c] += 1;
                }
            }
        }

        (successors, in_degree)
    }

    // Bounded beam search for the best valid linear extension of the destroy
    // poset, under real DyPDL applicability/state-constraint checks at every
    // step. Returns the best complete, feasible sequence of transitions
    // found (not necessarily `current` reordered -- see `alternatives`) and
    // its cost, or `None` if every beam entry died out to infeasibility
    // before completion.
    fn beam_repair(
        &mut self,
        successors: &[Vec<usize>],
        init_in_degree: &[usize],
        n: usize,
    ) -> Option<(Vec<Transition>, Vec<usize>, T)> {
        let root_state = self.model.target.clone();
        let root_placed_mask = vec![false; n];
        let root_f = self.eval_f_multiset(self.root_cost, &root_state, &root_placed_mask);
        let mut beam = vec![BeamEntry {
            placed: Vec::with_capacity(n),
            placed_positions: Vec::with_capacity(n),
            placed_mask: root_placed_mask,
            in_degree: init_in_degree.to_vec(),
            state: root_state,
            cost: self.root_cost,
            f: root_f,
            is_incumbent: true,
        }];

        for _ in 0..n {
            let mut candidates: Vec<BeamEntry<T>> = Vec::new();

            for entry in &beam {
                let mut ready: Vec<usize> = (0..n)
                    .filter(|&i| entry.in_degree[i] == 0 && !entry.placed_mask[i])
                    .collect();

                if ready.len() > self.max_branching {
                    ready.partial_shuffle(&mut self.rng, self.max_branching);
                    ready.truncate(self.max_branching);

                    // The incumbent lineage's next position is always ready
                    // (see is_incumbent's doc) -- make sure the shuffle above
                    // didn't drop it, since losing it here would silently
                    // break the truncation guarantee below.
                    if entry.is_incumbent {
                        let next = entry.placed.len();
                        if !ready.contains(&next) {
                            ready.push(next);
                        }
                    }
                }

                for &p in &ready {
                    for transition in &self.alternatives[p] {
                        let mut function_cache =
                            ParentAndChildStateFunctionCache::new(&self.model.state_functions);

                        if !transition.is_applicable(
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        ) {
                            continue;
                        }

                        let new_state = transition.apply(
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        );

                        if !self
                            .model
                            .check_constraints(&new_state, &mut function_cache.child)
                        {
                            continue;
                        }

                        let new_cost = transition.eval_cost(
                            entry.cost,
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        );

                        // Cheap proxy for "is this the incumbent's own
                        // transition at p" -- (name, parameter_values)
                        // uniquely identifies a grounded transition within
                        // one domain (grounding is deterministic), so this
                        // is equivalent to full Transition equality without
                        // the cost of deep-comparing precondition/effect/
                        // cost AST trees for every candidate at every step.
                        let is_incumbent = entry.is_incumbent
                            && p == entry.placed.len()
                            && transition.name == self.current[p].name
                            && transition.parameter_values == self.current[p].parameter_values;

                        let mut new_entry = entry.clone();
                        new_entry.placed.push(transition.clone());
                        new_entry.placed_positions.push(p);
                        new_entry.placed_mask[p] = true;
                        new_entry.f =
                            self.eval_f_multiset(new_cost, &new_state, &new_entry.placed_mask);
                        new_entry.state = new_state;
                        new_entry.cost = new_cost;
                        new_entry.is_incumbent = is_incumbent;
                        for &successor in &successors[p] {
                            new_entry.in_degree[successor] -= 1;
                        }

                        candidates.push(new_entry);
                    }
                }
            }

            if candidates.is_empty() {
                return None;
            }

            // At most one candidate can have is_incumbent true (only the
            // incumbent lineage's own next position produces one -- see
            // is_incumbent's doc). Pull it out before truncating so the
            // sort/truncate below can never drop it, then always reinsert
            // it: this is what turns "beam found nothing better" into a
            // real result instead of an artifact of truncation.
            let incumbent_entry = candidates
                .iter()
                .position(|c| c.is_incumbent)
                .map(|i| candidates.remove(i));

            candidates.sort_by(|a, b| {
                if exceed_bound(&self.model, a.f, Some(b.f)) {
                    std::cmp::Ordering::Greater
                } else if exceed_bound(&self.model, b.f, Some(a.f)) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            let keep = self.beam_width - if incumbent_entry.is_some() { 1 } else { 0 };
            candidates.truncate(keep);
            candidates.extend(incumbent_entry);
            beam = candidates;
        }

        let mut best: Option<(&BeamEntry<T>, T)> = None;

        for entry in &beam {
            let mut function_cache = ParentAndChildStateFunctionCache::new(&self.model.state_functions);
            if let Some(base_cost) = self
                .model
                .eval_base_cost(&entry.state, &mut function_cache.child)
            {
                let final_cost = (self.base_cost_evaluator)(entry.cost, base_cost);
                let better = best
                    .as_ref()
                    .map_or(true, |(_, best_cost)| !exceed_bound(&self.model, final_cost, Some(*best_cost)));
                if better {
                    best = Some((entry, final_cost));
                }
            }
        }

        // Rc -> owned Transition happens exactly once here, for the single
        // winning entry, not per candidate -- see BeamEntry.placed's doc for
        // why per-candidate cloning was the actual performance bug.
        best.map(|(entry, final_cost)| {
            let transitions: Vec<Transition> =
                entry.placed.iter().map(|t| t.as_ref().clone()).collect();
            (transitions, entry.placed_positions.clone(), final_cost)
        })
    }

    // Returns whether `cost` is a strict improvement over `other`.
    fn is_better(&self, cost: T, other: T) -> bool {
        !exceed_bound(&self.model, cost, Some(other))
    }
}

impl<T, B> Search<T> for DeorderLns<T, B>
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

                return Ok((self.best.clone(), true));
            }

            let n = self.current.len();
            let freed = self.build_destroy_poset(n);
            let (successors, in_degree) = self.build_successors(n, &freed);
            let repaired = self.beam_repair(&successors, &in_degree, n);

            let Some((order, _positions, cost)) = repaired else {
                continue;
            };

            if self.is_better(cost, self.current_cost) {
                self.current = order;
                self.current_cost = cost;
                // self.current just got reordered (and possibly item-4
                // substituted), so everything keyed by position --
                // transition_ids/must_precede and alternatives -- must be
                // refreshed to match, or beam_repair would silently place
                // whatever transition originally sat at a position instead
                // of what's actually there now (alternatives), or is_hard
                // would check the wrong mapping (transition_ids/
                // must_precede). Both were real, measured bugs, not
                // hypothetical ones. Only done on accept, which is far
                // rarer than every iteration, so this doesn't reintroduce
                // the per-iteration cost compute_must_precede's doc warns
                // about.
                let (transition_ids, must_precede) = compute_must_precede(&self.current);
                self.transition_ids = transition_ids;
                self.must_precede = must_precede;
                self.alternatives = compute_alternatives(&self.by_params, &self.current);
                self.refresh_trace();

                if self.is_better(cost, self.best.cost.unwrap()) {
                    self.best.cost = Some(cost);
                    self.best.transitions = self.current.clone();
                    self.best.time = self.time_keeper.elapsed_time();

                    if !self.quiet {
                        print_primal_bound(&self.best);
                    }

                    self.time_keeper.stop();

                    return Ok((self.best.clone(), false));
                }
            }
        }
    }
}
