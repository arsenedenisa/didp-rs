// returns a solver object that implements the Search trait, which performs
// deorder LNS (partial-order relaxation as a destroy set, repaired by beam
// search over the best linear extension) on a given model.
// Like create_dual_bound_local_search, it finds a first solution with a
// one-time CABS run, then hands off to the LNS to improve it.

use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{
    beam_search, Cabs, CabsParameters, CostNode, DeorderLns, DeorderLnsParameters, FNode, Search,
    SearchInput, StateInRegistry, SuccessorGenerator,
};
use dypdl::variable_type;
use dypdl::{ParentAndChildStateFunctionCache, StateFunctionCache, Transition};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates a deorder LNS solver using the dual bound as a heuristic function for its initial solution.
pub fn create_dual_bound_deorder_lns<T>(
    model: Rc<dypdl::Model>,
    mut parameters: DeorderLnsParameters<T>,
    cabs_parameters: CabsParameters<T>,
    f_evaluator_type: FEvaluatorType,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let root_cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    // Find an initial feasible solution with CABS, exactly as LNBS/dual_bound_local_search do
    // when no initial solution is given.
    let solution = if model.has_dual_bounds() {
        let h_model = model.clone();
        let h_evaluator = move |state: &_, cache: &mut _| h_model.eval_dual_bound(state, cache);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let g_model = model.clone();
        let node_generator = move |state, cost| {
            let mut cache = StateFunctionCache::new(&g_model.state_functions);

            FNode::generate_root_node(
                state,
                &mut cache,
                cost,
                &g_model,
                &h_evaluator,
                &f_evaluator,
                cabs_parameters
                    .beam_search_parameters
                    .parameters
                    .primal_bound,
            )
        };
        let h_model = model.clone();
        let h_evaluator = move |state: &_, cache: &mut _| h_model.eval_dual_bound(state, cache);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let t_model = model.clone();
        let transition_evaluator =
            move |node: &FNode<_, _>, transition, cache: &mut _, primal_bound| {
                node.generate_successor_node(
                    transition,
                    cache,
                    &t_model,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )
            };
        let beam_search_fn = move |input: &SearchInput<_, _>, parameters| {
            beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
            )
        };

        let input = SearchInput {
            node: node_generator(StateInRegistry::from(model.target.clone()), root_cost),
            generator: generator.clone(),
            solution_suffix: &[],
        };
        let mut cabs = Cabs::<_, _, _, _>::new(input, beam_search_fn, cabs_parameters);
        let (solution, _) = cabs.search_inner();

        solution
    } else {
        let g_model = model.clone();
        let node_generator =
            move |state, cost| Some(CostNode::generate_root_node(state, cost, &g_model));
        let t_model = model.clone();
        let transition_evaluator =
            move |node: &CostNode<_, _>,
                  transition,
                  cache: &mut ParentAndChildStateFunctionCache,
                  _| { node.generate_successor_node(transition, cache, &t_model) };
        let beam_search_fn = move |input: &SearchInput<_, _>, parameters| {
            beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
            )
        };

        let input = SearchInput {
            node: node_generator(StateInRegistry::from(model.target.clone()), root_cost),
            generator: generator.clone(),
            solution_suffix: &[],
        };
        let mut cabs = Cabs::<_, _, _, _>::new(input, beam_search_fn, cabs_parameters);
        let (solution, _) = cabs.search_inner();

        solution
    };

    // Leave CABS's elapsed time out of the LNS's own time budget.
    parameters.parameters.time_limit = parameters
        .parameters
        .time_limit
        .map(|time_limit| (time_limit - solution.time).max(0.0));

    let transitions = solution
        .transitions
        .iter()
        .map(|t| t.transition.clone())
        .collect();

    Box::new(DeorderLns::new(
        model,
        transitions,
        solution.cost,
        solution.time,
        root_cost,
        base_cost_evaluator,
        parameters,
        f_evaluator_type,
    ))
}
