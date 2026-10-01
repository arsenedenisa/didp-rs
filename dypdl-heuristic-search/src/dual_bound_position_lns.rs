use super::dual_bound_local_search::cabs_runner;
use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::data_structure::exceed_bound;
use super::search_algorithm::{
    beam_search, Cabs, CabsParameters, CostNode, FNode, PositionLns, PositionLnsParameters, Search,
    SearchInput, StateInRegistry, SuccessorGenerator,
};
use dypdl::variable_type;
use dypdl::{ParentAndChildStateFunctionCache, StateFunctionCache, Transition};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates a position LNS solver using the dual bound as a heuristic function for its initial solution.
pub fn create_dual_bound_position_lns<T>(
    model: Rc<dypdl::Model>,
    mut parameters: PositionLnsParameters<T>,
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

    // Find an initial feasible solution with CABS.
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

    let overall_time_limit = parameters.parameters.time_limit;
    let mut transitions: Vec<Transition> = solution
        .transitions
        .iter()
        .map(|t| t.transition.clone())
        .collect();
    let mut cost = solution.cost;
    let mut initial_time = solution.time;

    // PositionLns needs a starting solution of at least 2 transitions (it reports itself
    // unsolvable and terminates immediately otherwise), so keep escalating CABS, each round
    // seeded with the incumbent as primal bound and the previous round's final beam size,
    // until there is a solution to work from or CABS finds nothing better.
    if cost.is_some() && transitions.len() < 2 {
        let mut run_cabs = cabs_runner(model.clone(), generator.clone(), f_evaluator_type, root_cost);
        let mut beam_size = cabs_parameters.beam_search_parameters.beam_size;

        while transitions.len() < 2 {
            let remaining = overall_time_limit.map(|limit| (limit - initial_time).max(0.0));

            if remaining.is_some_and(|r| r <= 0.0) {
                break;
            }

            let mut round_parameters = cabs_parameters;
            round_parameters.beam_search_parameters.parameters.primal_bound = cost;
            round_parameters.beam_search_parameters.parameters.time_limit = remaining;
            round_parameters.beam_search_parameters.beam_size = beam_size;
            let (round_solution, final_beam_size) = run_cabs(round_parameters);
            beam_size = final_beam_size;
            initial_time += round_solution.time;

            match round_solution.cost {
                Some(new_cost) if !exceed_bound(&model, new_cost, cost) => {
                    cost = Some(new_cost);
                    transitions = round_solution.transitions;
                }
                _ => break,
            }
        }
    }

    // Leave CABS's elapsed time out of the LNS's own time budget.
    parameters.parameters.time_limit =
        overall_time_limit.map(|time_limit| (time_limit - initial_time).max(0.0));

    Box::new(PositionLns::new(
        model,
        transitions,
        cost,
        initial_time,
        root_cost,
        base_cost_evaluator,
        parameters,
        f_evaluator_type,
    ))
}
