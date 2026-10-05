//! Static analysis of whether a solver's destroy/repair step may ever validly change the
//! NUMBER of transitions in a solution (insert extras, drop some), as opposed to only ever
//! reordering/replacing a fixed set of them.
//!
//! Three buckets, from most to least restrictive:
//!   - `FixedPermutation`: every reachable parameter group has exactly one grounded
//!     transition, and the incumbent uses each exactly once (classic TSP/single-machine
//!     permutation domains). Insertion, deletion, and cross-group replacement are all
//!     infeasible by construction.
//!   - `FixedWithReplacement`: some parameter groups have multiple grounded alternatives
//!     (e.g. CVRP's `visit(i)` / `visit-via-depot(i)`), but the incumbent still uses each
//!     reachable group exactly once. Replacement among a group's own alternatives is valid;
//!     insertion/deletion is not. Also covers the "unparameterized counter" shape (see
//!     `has_universal_progress_counter`'s doc) -- e.g. MDKP's `pack`/`ignore`, which have no
//!     parameters at all and so share one `by_params` group that legitimately recurs once
//!     per item, every item, in every feasible solution.
//!   - `Flexible`: neither of the above holds -- some transitions are genuinely optional
//!     (e.g. optw's early-return action, or SALBP-1's data-dependent number of
//!     `open-new-station` calls), or the root state is already a base case -- so
//!     insertion and/or deletion may be valid and worth searching.
//!
//! Shared by `position_lns.rs` and `model_aware_local_search.rs`, which each used to carry
//! their own narrower, raw-transition-level version of just the permutation half of this
//! check. That version compared `transitions.len()` against the RAW reachable catalog count
//! (every grounded transition, not grouped), which double-counts a group's alternatives and
//! so misclassifies every `FixedWithReplacement` domain as flexible.

use dypdl::{
    expression::{ReferenceExpression, SetElementOperator, SetExpression},
    variable_type::Numeric,
    Model, State, StateFunctionCache, StateInterface, Transition, TransitionInterface,
};
use std::collections::{HashMap, HashSet};

/// See the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionCardinality {
    /// Fixed-length, no replacement: every reachable parameter group is a singleton.
    FixedPermutation,
    /// Fixed-length, replacement among same-parameter-group alternatives only.
    FixedWithReplacement,
    /// Transition count is not structurally fixed -- insertion/deletion may be valid.
    Flexible,
}

/// Classifies `transitions` (an incumbent solution) against `model`'s full grounded
/// `catalog` (every forward and forced transition the model can produce). See the module
/// doc for the three buckets. `catalog` takes a plain iterator rather than a
/// `SuccessorGenerator` so each caller can pass whatever representation of the grounded
/// catalog it already has on hand (`position_lns.rs`'s `SuccessorGenerator`-backed
/// `Rc<TransitionWithId>`s, `model_aware_local_search.rs`'s plain `TransitionWithId`s)
/// without an intermediate allocation.
///
/// `T` is only used to call `Model::eval_base_cost`'s generic cost type -- any `Numeric +
/// Ord` works identically here since only `.is_some()` is read, never the cost value
/// itself.
pub fn classify_transition_cardinality<'a, T: Numeric + Ord>(
    model: &Model,
    catalog: impl Iterator<Item = &'a Transition> + Clone,
    transitions: &[Transition],
) -> TransitionCardinality {
    // Zero-transition feasibility: if the root state already satisfies the base case, the
    // domain provably admits solutions of varying length (0 included) -- the strongest,
    // cheapest possible signal that transition count is not fixed.
    let mut function_cache = StateFunctionCache::new(&model.state_functions);
    if model
        .eval_base_cost::<T, _>(&model.target, &mut function_cache)
        .is_some()
    {
        return TransitionCardinality::Flexible;
    }

    // Whitelist, not blacklist, for "provably never adds elements back to `var_id`" -- see
    // `position_lns.rs`'s identical check (`is_provably_shrink_only`) for the full
    // rationale: this can only under-count dead transitions, never wrongly treat a live one
    // as unreachable.
    let is_provably_shrink_only = |var_id: usize, expression: &SetExpression| {
        matches!(
            expression,
            SetExpression::SetElementOperation(SetElementOperator::Remove, _, set)
                if matches!(
                    set.as_ref(),
                    SetExpression::Reference(ReferenceExpression::Variable(v)) if *v == var_id
                )
        )
    };

    let mut var_ever_gains_elements = vec![false; model.target.get_number_of_set_variables()];
    for t in catalog.clone() {
        for (var_id, expression) in &t.effect.set_effects {
            if !is_provably_shrink_only(*var_id, expression) {
                var_ever_gains_elements[*var_id] = true;
            }
        }
    }
    let is_reachable = |t: &Transition| {
        t.elements_in_set_variable.iter().all(|&(var_id, element)| {
            var_ever_gains_elements[var_id] || model.target.get_set_variable(var_id).contains(element)
        })
    };

    let mut group_sizes: HashMap<Vec<_>, usize> = HashMap::new();
    for t in catalog.clone() {
        if is_reachable(t) {
            *group_sizes.entry(t.parameter_values.clone()).or_insert(0) += 1;
        }
    }

    let mut seen_groups = HashSet::with_capacity(transitions.len());
    let covers_each_reachable_group_once = transitions.len() == group_sizes.len()
        && transitions.iter().all(|t| {
            group_sizes.contains_key(&t.parameter_values) && seen_groups.insert(t.parameter_values.clone())
        });

    if covers_each_reachable_group_once {
        return if group_sizes.values().all(|&size| size == 1) {
            TransitionCardinality::FixedPermutation
        } else {
            TransitionCardinality::FixedWithReplacement
        };
    }

    if has_universal_progress_counter::<T>(model, catalog, transitions, &mut function_cache) {
        return TransitionCardinality::FixedWithReplacement;
    }

    TransitionCardinality::Flexible
}

/// Catches the one case the group-coverage check above structurally cannot: a catalog of
/// *unparameterized* transitions (so every one of them shares the single, degenerate
/// `parameter_values == []` group) that legitimately recurs many times in a row -- e.g.
/// MDKP's `pack`/`ignore`, grounded over no object at all, chosen once per item. There,
/// `transitions.len()` (one choice per item) vastly exceeds `group_sizes.len()` (1), so the
/// coverage check reports "not covered," indistinguishable on its own from a genuinely
/// variable-length domain like SALBP-1 (whose own repeating group, `open-new-station`, has a
/// data-dependent repeat count that really does vary across feasible solutions).
///
/// The distinguishing fact: replay the incumbent from `model.target` and check whether
/// EVERY reachable transition that is ever actually applicable along that replay applies
/// the exact same, state-independent delta to some single element or integer state
/// variable (MDKP: every transition does `i: (+ i 1)`, full stop). If so, that variable's
/// value after `k` transitions is `initial + k * delta` regardless of which transitions
/// were chosen -- so the total transition count is pinned by that variable's start and end
/// values alone, not by which branch was taken at each step. (SALBP-1 fails this: `do-task`
/// always applies `-1` to `|unscheduled|` while `open-new-station` applies `0`, so no single
/// variable has a universal delta across the *whole* catalog.)
///
/// A universal delta alone isn't quite enough -- it would also hold for a domain where the
/// counter keeps advancing past where some runs stop early (optw-style). So also require
/// that `eval_base_cost` is `None` at every strict prefix of the incumbent's own replay: the
/// base case is reachable at exactly one depth, not a range of them.
///
/// This is a sampling-based heuristic (checked along one incumbent's replay, not proven for
/// every reachable state), consistent with this module's existing whitelist-not-blacklist
/// style elsewhere: it can only fail to recognize a genuinely fixed-cardinality domain
/// (falling through to `Flexible`, same as today), never wrongly force a real `insertion_slack`
/// candidate off.
fn has_universal_progress_counter<'a, T: Numeric + Ord>(
    model: &Model,
    catalog: impl Iterator<Item = &'a Transition>,
    transitions: &[Transition],
    function_cache: &mut StateFunctionCache,
) -> bool {
    if transitions.is_empty() {
        return false;
    }

    let states = replay_states(model, transitions, function_cache);
    let num_element = model.target.get_number_of_element_variables();
    let num_integer = model.target.get_number_of_integer_variables();

    // Seed candidate (is_element, index, delta) triples from the incumbent's own steps,
    // dropping any variable where even the incumbent itself isn't consistent.
    let mut element_delta: Vec<Option<i64>> = vec![None; num_element];
    let mut element_consistent = vec![true; num_element];
    let mut integer_delta: Vec<Option<i64>> = vec![None; num_integer];
    let mut integer_consistent = vec![true; num_integer];

    for i in 0..transitions.len() {
        let (pre, post) = (&states[i], &states[i + 1]);
        for e in 0..num_element {
            if !element_consistent[e] {
                continue;
            }
            let d = post.get_element_variable(e) as i64 - pre.get_element_variable(e) as i64;
            match element_delta[e] {
                None => element_delta[e] = Some(d),
                Some(prev) if prev == d => {}
                _ => element_consistent[e] = false,
            }
        }
        for v in 0..num_integer {
            if !integer_consistent[v] {
                continue;
            }
            let d = post.get_integer_variable(v) as i64 - pre.get_integer_variable(v) as i64;
            match integer_delta[v] {
                None => integer_delta[v] = Some(d),
                Some(prev) if prev == d => {}
                _ => integer_consistent[v] = false,
            }
        }
    }

    let mut surviving: HashSet<(bool, usize, i64)> = element_delta
        .iter()
        .enumerate()
        .filter(|&(e, _)| element_consistent[e])
        .filter_map(|(e, d)| d.map(|d| (true, e, d)))
        .chain(
            integer_delta
                .iter()
                .enumerate()
                .filter(|&(v, _)| integer_consistent[v])
                .filter_map(|(v, d)| d.map(|d| (false, v, d))),
        )
        .collect();

    if surviving.is_empty() {
        return false;
    }

    // Narrow further: every surviving candidate must also hold for every UNTAKEN
    // alternative that was ever applicable along the replay, not just the transitions the
    // incumbent happened to choose.
    let catalog: Vec<&Transition> = catalog.collect();
    for state in &states {
        if surviving.is_empty() {
            break;
        }

        for t in &catalog {
            function_cache.clear();
            if !t.is_applicable(
                state,
                function_cache,
                &model.state_functions,
                &model.table_registry,
            ) {
                continue;
            }

            function_cache.clear();
            let next: State = t.apply(
                state,
                function_cache,
                &model.state_functions,
                &model.table_registry,
            );

            surviving.retain(|&(is_element, idx, delta)| {
                let d = if is_element {
                    next.get_element_variable(idx) as i64 - state.get_element_variable(idx) as i64
                } else {
                    next.get_integer_variable(idx) as i64 - state.get_integer_variable(idx) as i64
                };
                d == delta
            });
        }
    }

    if surviving.is_empty() {
        return false;
    }

    !states[..states.len() - 1].iter().any(|s| {
        function_cache.clear();
        model.eval_base_cost::<T, _>(s, function_cache).is_some()
    })
}

/// Replays `transitions` from `model.target`, returning the `transitions.len() + 1` states
/// visited (`states[0] == model.target`, `states[k]` after applying `transitions[..k]`).
fn replay_states(
    model: &Model,
    transitions: &[Transition],
    function_cache: &mut StateFunctionCache,
) -> Vec<State> {
    let mut states = Vec::with_capacity(transitions.len() + 1);
    states.push(model.target.clone());

    for t in transitions {
        function_cache.clear();
        let next: State = t.apply(
            states.last().unwrap(),
            function_cache,
            &model.state_functions,
            &model.table_registry,
        );
        states.push(next);
    }

    states
}
