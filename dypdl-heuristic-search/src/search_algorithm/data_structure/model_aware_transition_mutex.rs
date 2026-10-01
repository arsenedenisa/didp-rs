// Checks whether two grounded transitions commute: does swapping their order
// in an already-feasible trajectory risk changing feasibility? This is a
// narrower question than `TransitionMutex::forbidden_before/after`, which
// reasons globally about achievers during forward expansion and is silent
// once any achiever of a fact exists anywhere in the catalog -- it says
// nothing about reordering two already-scheduled transitions. Here conflict
// is determined purely from each transition's own read/write fact sets.
//
// Used only to skip candidate moves that are provably safe to reorder:
// `false` ("no known conflict") is not a feasibility guarantee -- the real
// rollout (`LocalSearch::evaluate`) still runs regardless -- while `true`
// skips the move without ever evaluating it. So `conflicts` must never
// over-claim a conflict; missing one is safe by construction.
//
// Decomposes three precondition shapes into per-(var_id, element) facts:
// plain `is_in`/`not(is_in)`, `lhs subset var` (every element of a grounded
// `lhs` is positively required in `var`), and `is_empty(var & rhs)` (every
// element of a grounded `rhs` is negatively required in `var`). Table
// lookups against grounded parameters constant-fold via
// `SetExpression::simplify`, so they decompose the same way as a literal set.

use super::transition::TransitionWithId;
use dypdl::expression::*;
use dypdl::variable_type::Element;
use dypdl::{TableRegistry, Transition, TransitionInterface};
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Default, Debug)]
struct TransitionFacts {
    achieved: FxHashSet<(usize, Element)>,
    removed: FxHashSet<(usize, Element)>,
    positively_required: FxHashSet<(usize, Element)>,
    negatively_required: FxHashSet<(usize, Element)>,
    // Variable ids affected by an effect this analysis can't decompose to a
    // specific element (e.g. a set built from a non-constant union/difference).
    // Conservatively treated as conflicting with anything touching the same
    // variable, known element or not.
    arbitrary: FxHashSet<usize>,
}

impl TransitionFacts {
    fn touches_var(&self, var_id: usize) -> bool {
        self.arbitrary.contains(&var_id)
            || self.achieved.iter().any(|(v, _)| *v == var_id)
            || self.removed.iter().any(|(v, _)| *v == var_id)
            || self.positively_required.iter().any(|(v, _)| *v == var_id)
            || self.negatively_required.iter().any(|(v, _)| *v == var_id)
    }

    fn reads_or_writes(&self, fact: (usize, Element)) -> bool {
        self.achieved.contains(&fact)
            || self.removed.contains(&fact)
            || self.positively_required.contains(&fact)
            || self.negatively_required.contains(&fact)
    }

    fn written_facts(&self) -> impl Iterator<Item = (usize, Element)> + '_ {
        self.achieved.iter().copied().chain(self.removed.iter().copied())
    }
}

fn extract_facts(transition: &Transition, registry: &TableRegistry) -> TransitionFacts {
    let mut facts = TransitionFacts::default();

    for (var_id, expression) in &transition.effect.set_effects {
        match expression {
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Constant(element),
                _,
            ) => {
                facts.achieved.insert((*var_id, *element));
            }
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(element),
                _,
            ) => {
                facts.removed.insert((*var_id, *element));
            }
            _ => {
                facts.arbitrary.insert(*var_id);
            }
        }
    }

    for condition in transition.get_preconditions() {
        collect_required_elements(&condition, registry, false, &mut facts);
    }

    facts
}

// `negated` tracks how many `Not(..)` wrappers we've walked through an odd
// number of times, so `Not(Not(is_in ...))` etc. still resolve correctly.
fn collect_required_elements(
    condition: &Condition,
    registry: &TableRegistry,
    negated: bool,
    facts: &mut TransitionFacts,
) {
    match condition {
        Condition::Not(inner) => collect_required_elements(inner, registry, !negated, facts),
        Condition::Set(set_condition) => match set_condition.as_ref() {
            SetCondition::IsIn(ElementExpression::Constant(element), set_expression) => {
                match set_expression {
                    SetExpression::Reference(ReferenceExpression::Variable(var_id)) => {
                        insert_required(facts, negated, *var_id, *element);
                    }
                    SetExpression::Complement(inner) => {
                        if let SetExpression::Reference(ReferenceExpression::Variable(var_id)) =
                            inner.as_ref()
                        {
                            insert_required(facts, !negated, *var_id, *element);
                        }
                    }
                    _ => {}
                }
            }
            // `lhs subset var`: every element the (grounded, so constant-foldable)
            // `lhs` side contains is positively required in `var`. Only decomposed
            // in the non-negated direction -- "NOT a subset" doesn't reduce to a
            // fixed set of per-element facts (it only says *some* element of lhs is
            // missing, not which one).
            SetCondition::IsSubset(
                lhs,
                SetExpression::Reference(ReferenceExpression::Variable(var_id)),
            ) if !negated => {
                if let SetExpression::Reference(ReferenceExpression::Constant(set)) =
                    lhs.simplify(registry)
                {
                    for element in set.ones() {
                        insert_required(facts, false, *var_id, element);
                    }
                }
            }
            // `is_empty(var intersect rhs)` (checked in both operand orders):
            // `var` must not contain any element of the (grounded) other side. Only
            // decomposed non-negated -- "NOT empty" only says *some* shared element
            // exists, not which one.
            SetCondition::IsEmpty(SetExpression::SetOperation(SetOperator::Intersection, x, y))
                if !negated =>
            {
                decompose_intersection_empty(x, y, registry, facts);
                decompose_intersection_empty(y, x, registry, facts);
            }
            _ => {}
        },
        _ => {}
    }
}

fn insert_required(facts: &mut TransitionFacts, negated: bool, var_id: usize, element: Element) {
    if negated {
        facts.negatively_required.insert((var_id, element));
    } else {
        facts.positively_required.insert((var_id, element));
    }
}

fn decompose_intersection_empty(
    var_side: &SetExpression,
    const_side: &SetExpression,
    registry: &TableRegistry,
    facts: &mut TransitionFacts,
) {
    if let SetExpression::Reference(ReferenceExpression::Variable(var_id)) = var_side {
        if let SetExpression::Reference(ReferenceExpression::Constant(set)) =
            const_side.simplify(registry)
        {
            for element in set.ones() {
                facts.negatively_required.insert((*var_id, element));
            }
        }
    }
}

/// Direct fact-level commutativity check between grounded transitions: does
/// swapping two transitions' relative order in an already-feasible trajectory
/// risk changing feasibility? See the module doc for why this is a different
/// (and, for this use, more appropriate) question than what
/// [`super::TransitionMutex`] answers, and for the asymmetric error tolerance
/// (`conflicts` must never over-claim a conflict; missing one is safe by
/// construction because the real rollout still runs afterward regardless).
#[derive(Default, Debug)]
pub struct ModelAwareTransitionMutex {
    facts: Vec<TransitionFacts>,
    forced_facts: Vec<TransitionFacts>,
    /// True if any grounded transition has at least one precondition this
    /// analysis (or plain `is_in`) could extract a fact from. `false` means
    /// the model has no fact-level ordering structure at all to exploit (e.g.
    /// MOSP, whose `close` transition has no preconditions) -- callers should
    /// skip filtering entirely rather than pay for a lookup that can never
    /// find a conflict.
    pub has_feasibility_structure: bool,
    // For each transition (keyed by (forced, id)), every *other* transition in the
    // catalog whose (achieved union removed) fact set is identical and non-empty --
    // e.g. CVRP's `visit(to)`/`visit-via-depot(to)` both achieve exactly
    // {(unvisited, to)}, so each is the other's sole alternative. Transitions with no
    // set effects, or a set-effect signature unique to them, have no entry (an absent
    // key means "no alternatives", same as an empty slice).
    alternatives: FxHashMap<(bool, usize), Vec<(bool, usize)>>,
    /// True if any transition has at least one alternative -- `false` means there is
    /// nothing for a ReSupport-style substitution move to ever find in this model.
    pub has_variant_structure: bool,
}

impl ModelAwareTransitionMutex {
    /// Every transition in the catalog whose achieved/removed fact set exactly matches
    /// this one's (excluding itself). Empty if this transition has no set effects, or
    /// none of them are shared by anything else in the catalog. See the module doc and
    /// `has_variant_structure` for what this is for and its scope.
    pub fn alternatives(&self, forced: bool, id: usize) -> &[(bool, usize)] {
        self.alternatives
            .get(&(forced, id))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Create a new instance from the given transitions (typically a model's
    /// full grounded catalog, i.e. `forward_transitions` chained with
    /// `forward_forced_transitions`, wrapped with their catalog ids).
    pub fn new<T>(transitions: Vec<TransitionWithId<T>>, registry: &TableRegistry) -> Self
    where
        T: TransitionInterface + Clone,
        Transition: From<T>,
    {
        let len = transitions
            .iter()
            .filter_map(|t| if !t.forced { Some(t.id) } else { None })
            .max()
            .map_or(0, |id_max| id_max + 1);
        let forced_len = transitions
            .iter()
            .filter_map(|t| if t.forced { Some(t.id) } else { None })
            .max()
            .map_or(0, |id_max| id_max + 1);

        let mut facts: Vec<TransitionFacts> = (0..len).map(|_| TransitionFacts::default()).collect();
        let mut forced_facts: Vec<TransitionFacts> =
            (0..forced_len).map(|_| TransitionFacts::default()).collect();
        let mut has_feasibility_structure = false;
        // Groups transitions by their (achieved union removed) fact signature, sorted
        // for a stable, comparable key. Only non-empty signatures are tracked, since an
        // empty signature (no set effects at all) isn't a meaningful "these two do the
        // same thing" grouping.
        let mut signature_groups: FxHashMap<Vec<(usize, Element)>, Vec<(bool, usize)>> =
            FxHashMap::default();

        for t in transitions {
            let id = t.id;
            let forced = t.forced;
            let transition = Transition::from(t.transition);
            let extracted = extract_facts(&transition, registry);
            has_feasibility_structure = has_feasibility_structure
                || !extracted.positively_required.is_empty()
                || !extracted.negatively_required.is_empty();

            let mut signature: Vec<(usize, Element)> = extracted
                .achieved
                .iter()
                .chain(extracted.removed.iter())
                .copied()
                .collect();
            if !signature.is_empty() {
                signature.sort_unstable();
                signature_groups
                    .entry(signature)
                    .or_default()
                    .push((forced, id));
            }

            if forced {
                forced_facts[id] = extracted;
            } else {
                facts[id] = extracted;
            }
        }

        let mut alternatives: FxHashMap<(bool, usize), Vec<(bool, usize)>> = FxHashMap::default();
        let mut has_variant_structure = false;
        for group in signature_groups.into_values() {
            if group.len() < 2 {
                continue;
            }
            has_variant_structure = true;
            for &member in &group {
                let others = group.iter().copied().filter(|&m| m != member).collect();
                alternatives.insert(member, others);
            }
        }

        Self {
            facts,
            forced_facts,
            has_feasibility_structure,
            alternatives,
            has_variant_structure,
        }
    }

    fn get_facts(&self, forced: bool, id: usize) -> &TransitionFacts {
        if forced {
            &self.forced_facts[id]
        } else {
            &self.facts[id]
        }
    }

    /// Returns whether swapping the relative order of these two grounded
    /// transitions could change feasibility. `false` ("no known conflict")
    /// is not a feasibility guarantee -- see the module doc.
    pub fn conflicts(&self, forced_a: bool, id_a: usize, forced_b: bool, id_b: usize) -> bool {
        let a = self.get_facts(forced_a, id_a);
        let b = self.get_facts(forced_b, id_b);

        if a.arbitrary.iter().any(|&v| b.touches_var(v))
            || b.arbitrary.iter().any(|&v| a.touches_var(v))
        {
            return true;
        }

        a.written_facts().any(|fact| b.reads_or_writes(fact))
            || b.written_facts().any(|fact| a.reads_or_writes(fact))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::prelude::*;

    // single-machine's `schedule(i)` gated on `is_subset (predecessors i)
    // scheduled`, which plain `TransitionMutex` (single-element `is_in` only)
    // cannot decompose at all.
    #[test]
    fn detects_conflict_from_is_subset_precondition() {
        let mut model = Model::default();
        let job = model.add_object_type("job", 3).unwrap();
        let scheduled = model
            .add_set_variable("scheduled", job, Set::with_capacity(3))
            .unwrap();
        let no_predecessors = Set::with_capacity(3);
        let requires_job_0 = {
            let mut s = Set::with_capacity(3);
            s.insert(0);
            s
        };
        let predecessors = model
            .add_table_1d(
                "predecessors",
                vec![no_predecessors.clone(), requires_job_0, no_predecessors.clone()],
            )
            .unwrap();

        let mut schedule_0 = Transition::new("schedule 0");
        schedule_0.add_effect(scheduled, scheduled.add(0)).unwrap();

        let mut schedule_1 = Transition::new("schedule 1");
        schedule_1.add_precondition(predecessors.element(1).is_subset(scheduled));
        schedule_1.add_effect(scheduled, scheduled.add(1)).unwrap();

        let mut schedule_2 = Transition::new("schedule 2");
        schedule_2.add_effect(scheduled, scheduled.add(2)).unwrap();

        let transitions = vec![
            TransitionWithId {
                id: 0,
                forced: false,
                transition: schedule_0,
            },
            TransitionWithId {
                id: 1,
                forced: false,
                transition: schedule_1,
            },
            TransitionWithId {
                id: 2,
                forced: false,
                transition: schedule_2,
            },
        ];
        let mutex = ModelAwareTransitionMutex::new(transitions, &model.table_registry);

        assert!(mutex.has_feasibility_structure);
        // schedule(1) requires (scheduled, 0), which schedule(0) writes: swapping
        // them is a real conflict.
        assert!(mutex.conflicts(false, 0, false, 1));
        // schedule(0) and schedule(2) touch disjoint facts (different elements of
        // the same variable): no known conflict, safe to swap.
        assert!(!mutex.conflicts(false, 0, false, 2));
        assert!(!mutex.conflicts(false, 1, false, 2));
    }

    // Regression test for m-PDTSP's `visit(to)` gated on
    // `is_empty (intersection unvisited (predecessors to))`: `predecessors`
    // is a table (grounded/constant once `to` is bound), `unvisited` is the
    // state variable -- neither operand order should matter.
    #[test]
    fn detects_conflict_from_is_empty_intersection_precondition() {
        let mut model = Model::default();
        let customer = model.add_object_type("customer", 3).unwrap();
        let full = {
            let mut s = Set::with_capacity(3);
            s.insert(0);
            s.insert(1);
            s.insert(2);
            s
        };
        let unvisited = model
            .add_set_variable("unvisited", customer, full)
            .unwrap();
        let no_predecessors = Set::with_capacity(3);
        let requires_customer_0 = {
            let mut s = Set::with_capacity(3);
            s.insert(0);
            s
        };
        let predecessors = model
            .add_table_1d(
                "predecessors",
                vec![
                    no_predecessors.clone(),
                    requires_customer_0,
                    no_predecessors,
                ],
            )
            .unwrap();

        let mut visit_0 = Transition::new("visit 0");
        visit_0
            .add_effect(unvisited, unvisited.remove(0))
            .unwrap();

        let mut visit_1 = Transition::new("visit 1");
        visit_1.add_precondition((unvisited & predecessors.element(1)).is_empty());
        visit_1
            .add_effect(unvisited, unvisited.remove(1))
            .unwrap();

        let transitions = vec![
            TransitionWithId {
                id: 0,
                forced: false,
                transition: visit_0,
            },
            TransitionWithId {
                id: 1,
                forced: false,
                transition: visit_1,
            },
        ];
        let mutex = ModelAwareTransitionMutex::new(transitions, &model.table_registry);

        assert!(mutex.has_feasibility_structure);
        // visit(1) requires customer 0 already removed from `unvisited`, which
        // visit(0) does: swapping them is a real conflict.
        assert!(mutex.conflicts(false, 0, false, 1));
    }

    // Regression test for the ReSupport substitution case: CVRP's `visit(to)` and
    // `visit-via-depot(to)` both achieve exactly {(unvisited, to)} despite differing
    // in every other respect (name, other effects, cost), so they should be each
    // other's sole alternative -- and neither should have anything in common with a
    // `visit` for a *different* customer.
    #[test]
    fn finds_alternatives_with_matching_achieved_facts() {
        let mut model = Model::default();
        let customer = model.add_object_type("customer", 2).unwrap();
        let full = {
            let mut s = Set::with_capacity(2);
            s.insert(0);
            s.insert(1);
            s
        };
        let unvisited = model
            .add_set_variable("unvisited", customer, full)
            .unwrap();
        let load = model.add_integer_variable("load", 0).unwrap();

        let mut visit_0 = Transition::new("visit");
        visit_0.add_effect(unvisited, unvisited.remove(0)).unwrap();
        visit_0.add_effect(load, load + 1).unwrap();

        let mut visit_via_depot_0 = Transition::new("visit-via-depot");
        visit_via_depot_0
            .add_effect(unvisited, unvisited.remove(0))
            .unwrap();
        visit_via_depot_0.add_effect(load, 1).unwrap();

        let mut visit_1 = Transition::new("visit");
        visit_1.add_effect(unvisited, unvisited.remove(1)).unwrap();

        let transitions = vec![
            TransitionWithId {
                id: 0,
                forced: false,
                transition: visit_0,
            },
            TransitionWithId {
                id: 1,
                forced: false,
                transition: visit_via_depot_0,
            },
            TransitionWithId {
                id: 2,
                forced: false,
                transition: visit_1,
            },
        ];
        let mutex = ModelAwareTransitionMutex::new(transitions, &model.table_registry);

        assert!(mutex.has_variant_structure);
        assert_eq!(mutex.alternatives(false, 0), &[(false, 1)]);
        assert_eq!(mutex.alternatives(false, 1), &[(false, 0)]);
        // visit(1) touches a different element (1, not 0): no alternatives.
        assert_eq!(mutex.alternatives(false, 2), &[]);
    }
}
