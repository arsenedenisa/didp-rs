// returns a solver object that implements the Search trait, which can be used to perform local search on a given model
// search_next actually returns a solution
// it is called in the yaml folder, in solve method in main.rs

use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::data_structure::exceed_bound;
use super::search_algorithm::{
    beam_search, Cabs, CabsParameters, CostNode, FNode, LocalSearch, LocalSearchMode,
    LocalSearchParameters, Search, SearchInput, Solution, StateInRegistry, SuccessorGenerator,
};
use dypdl::variable_type;
use dypdl::{ParentAndChildStateFunctionCache, StateFunctionCache, Transition};
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Builds a closure that, each time it is called, runs one bounded CABS round (fresh root node,
/// beam size starting from `cabs_parameters.beam_search_parameters.beam_size`) with the given
/// `CabsParameters<T>` and returns the resulting solution, in terms of plain `Transition`s,
/// alongside the beam size the round ended on (so a caller alternating rounds can resume growth
/// from there instead of restarting at the configured initial beam size every time).
///
/// This is the same construction `create_dual_bound_local_search` used to run exactly once
/// (to find an initial feasible solution, exactly as LNBS does), just parameterized so it can
/// be called again with a different primal bound and time limit every time hill climbing gets
/// stuck.
pub(crate) fn cabs_runner<T>(
    model: Rc<dypdl::Model>,
    generator: SuccessorGenerator<Transition>,
    f_evaluator_type: FEvaluatorType,
    root_cost: T,
) -> Box<dyn FnMut(CabsParameters<T>) -> (Solution<T>, usize)>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    if model.has_dual_bounds() {
        Box::new(move |cabs_parameters: CabsParameters<T>| {
            let primal_bound = cabs_parameters
                .beam_search_parameters
                .parameters
                .primal_bound;
            let h_model = model.clone();
            let h_evaluator = move |state: &_, cache: &mut _| h_model.eval_dual_bound(state, cache);
            let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
            let g_model = model.clone();
            let node_generator = move |state, cost| {
                let mut cache = StateFunctionCache::new(&g_model.state_functions);
                // cache some state variables that are not stored by the state itself, but are
                // needed for checking preconditions for example

                FNode::generate_root_node(
                    state,
                    &mut cache,
                    cost,
                    &g_model,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )
            };
            let h_model = model.clone();
            let h_evaluator = move |state: &_, cache: &mut _| h_model.eval_dual_bound(state, cache);
            let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
            let t_model = model.clone();
            // generates successor nodes from a given node and transition, using the model and
            // evaluators; it is passed to the beam search function to evaluate the transitions
            // and generate successor nodes
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
            let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
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
            let final_beam_size = cabs.beam_size();

            let solution = Solution {
                cost: solution.cost,
                best_bound: solution.best_bound,
                is_optimal: solution.is_optimal,
                is_infeasible: solution.is_infeasible,
                transitions: solution
                    .transitions
                    .into_iter()
                    .map(|t| t.transition)
                    .collect(),
                expanded: solution.expanded,
                generated: solution.generated,
                time: solution.time,
                time_out: solution.time_out,
            };

            (solution, final_beam_size)
        })
    } else {
        Box::new(move |cabs_parameters: CabsParameters<T>| {
            let g_model = model.clone();
            let node_generator =
                move |state, cost| Some(CostNode::generate_root_node(state, cost, &g_model));
            let t_model = model.clone();
            let transition_evaluator =
                move |node: &CostNode<_, _>,
                      transition,
                      cache: &mut ParentAndChildStateFunctionCache,
                      _| { node.generate_successor_node(transition, cache, &t_model) };
            let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
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
            let final_beam_size = cabs.beam_size();

            let solution = Solution {
                cost: solution.cost,
                best_bound: solution.best_bound,
                is_optimal: solution.is_optimal,
                is_infeasible: solution.is_infeasible,
                transitions: solution
                    .transitions
                    .into_iter()
                    .map(|t| t.transition)
                    .collect(),
                expanded: solution.expanded,
                generated: solution.generated,
                time: solution.time,
                time_out: solution.time_out,
            };

            (solution, final_beam_size)
        })
    }
}

/// Alternates between hill climbing (`LocalSearch`) and CABS.
///
/// Whenever hill climbing gets stuck in a local optimum (every enabled neighborhood kind
/// produced no improving move, but the time limit has not been reached), this runs one bounded
/// CABS round -- reusing the CABS parameters the solver was configured with -- bounded above by
/// the current incumbent's cost. If that round finds something strictly better, hill climbing
/// resumes from it; a fresh `LocalSearch` is deterministic given the same seed and bound, so if
/// it does not, there is nothing more this can do automatically and the search terminates with
/// the current best (marked optimal if CABS proved the bound infeasible).
///
/// The beam size is not reset to the configured initial value on every round: each round starts
/// from wherever the previous round's exponential widening left off (see `beam_size`), so the
/// beam only grows wider across alternations instead of re-paying the same small-beam-size work
/// every time hill climbing gets stuck again.
struct HybridLocalSearch<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    inner: LocalSearch<T, Box<dyn FnMut(T, T) -> T>>,
    run_cabs: Box<dyn FnMut(CabsParameters<T>) -> (Solution<T>, usize)>,
    model: Rc<dypdl::Model>,
    f_evaluator_type: FEvaluatorType,
    root_cost: T,
    local_search_parameters: LocalSearchParameters<T>,
    cabs_parameters_template: CabsParameters<T>,
    overall_time_limit: Option<f64>,
    // Beam size the next CABS round should start from. Seeded from the initial CABS run (the
    // one that found the first feasible solution) and updated to whatever beam size the round
    // ended on after every subsequent call, so growth carries across rounds instead of every
    // stuck-triggered round restarting the exponential widening from
    // `cabs_parameters_template.beam_search_parameters.beam_size`.
    beam_size: usize,
    // Cumulative wall-clock time spent across every hill-climbing and CABS phase so far.
    // Each phase's own `Solution::time` is relative to its own start, not cumulative, so this
    // has to be tracked here and used both to compute each phase's remaining time budget and to
    // patch reported `Solution::time` values before they reach the caller.
    total_elapsed: f64,
}

impl<T> HybridLocalSearch<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn base_cost_evaluator(&self) -> Box<dyn FnMut(T, T) -> T> {
        let f_evaluator_type = self.f_evaluator_type;
        Box::new(move |cost, base_cost| f_evaluator_type.eval(cost, base_cost))
    }

    fn remaining_time(&self) -> Option<f64> {
        self.overall_time_limit
            .map(|limit| (limit - self.total_elapsed).max(0.0))
    }
}

impl<T> Search<T> for HybridLocalSearch<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        loop {
            let (mut solution, terminated) = self.inner.search_next()?;

            if !terminated {
                solution.time += self.total_elapsed;

                return Ok((solution, false));
            }

            self.total_elapsed += solution.time;
            solution.time = self.total_elapsed;

            // A genuine time-out or an infeasible model: nothing to alternate into.
            if solution.time_out || solution.cost.is_none() {
                return Ok((solution, true));
            }

            let remaining = self.remaining_time();

            if remaining.is_some_and(|remaining| remaining <= 0.0) {
                return Ok((solution, true));
            }

            let mut cabs_parameters = self.cabs_parameters_template;
            cabs_parameters.beam_search_parameters.parameters.primal_bound = solution.cost;
            cabs_parameters.beam_search_parameters.parameters.time_limit = remaining;
            cabs_parameters.beam_search_parameters.beam_size = self.beam_size;

            let (cabs_solution, final_beam_size) = (self.run_cabs)(cabs_parameters);
            self.beam_size = final_beam_size;
            self.total_elapsed += cabs_solution.time;

            let improved = cabs_solution.cost.is_some_and(|new_cost| {
                !exceed_bound(&self.model, new_cost, solution.cost)
            });

            if !improved {
                solution.is_optimal = cabs_solution.is_infeasible;
                solution.time = self.total_elapsed;

                return Ok((solution, true));
            }

            let mut local_search_parameters = self.local_search_parameters;
            local_search_parameters.parameters.time_limit = self.remaining_time();

            self.inner = LocalSearch::new(
                self.model.clone(),
                cabs_solution.transitions,
                cabs_solution.cost,
                self.total_elapsed,
                self.root_cost,
                self.base_cost_evaluator(),
                local_search_parameters,
            );
        }
    }
}

/// Creates a local search solver using the dual bound as a heuristic function for its initial solution.
/// finds first solution with one time run CABS
/// like in lnbs
/// then uses local search to improve the solution found by CABS
///
/// If `cabs_on_stuck` is `true` and `parameters.mode` is `LocalSearchMode::HillClimbing`, the
/// solver alternates between hill climbing and CABS: whenever hill climbing gets stuck in a
/// local optimum, it runs one more bounded CABS round (see `HybridLocalSearch`) and resumes hill
/// climbing from whatever that round finds, until the time limit is reached or CABS proves the
/// incumbent optimal. Outside `HillClimbing` mode, `cabs_on_stuck` is ignored.
pub fn create_dual_bound_local_search<T>(
    model: Rc<dypdl::Model>,
    mut parameters: LocalSearchParameters<T>, //defined in local_search.rs
    cabs_parameters: CabsParameters<T>,
    f_evaluator_type: FEvaluatorType,
    cabs_on_stuck: bool,
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

    // Find an initial feasible solution with CABS, exactly as LNBS does when no
    // initial solution is given.
    let mut run_cabs = cabs_runner(model.clone(), generator, f_evaluator_type, root_cost);
    let (solution, final_beam_size) = run_cabs(cabs_parameters);

    let overall_time_limit = parameters.parameters.time_limit;
    // Leave CABS's elapsed time out of the local search's own time budget,
    // exactly as LNBS does.
    parameters.parameters.time_limit =
        overall_time_limit.map(|time_limit| (time_limit - solution.time).max(0.0));

    if cabs_on_stuck && matches!(parameters.mode, LocalSearchMode::HillClimbing) {
        let f_evaluator_type_for_evaluator = f_evaluator_type;
        let base_cost_evaluator: Box<dyn FnMut(T, T) -> T> = Box::new(move |cost, base_cost| {
            f_evaluator_type_for_evaluator.eval(cost, base_cost)
        });
        let inner = LocalSearch::new(
            model.clone(),
            solution.transitions.clone(),
            solution.cost,
            solution.time,
            root_cost,
            base_cost_evaluator,
            parameters,
        );

        Box::new(HybridLocalSearch {
            inner,
            run_cabs,
            model,
            f_evaluator_type,
            root_cost,
            local_search_parameters: parameters,
            cabs_parameters_template: cabs_parameters,
            overall_time_limit,
            total_elapsed: solution.time,
            beam_size: final_beam_size,
        })
    } else {
        let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);

        Box::new(LocalSearch::new(
            model,
            solution.transitions,
            solution.cost,
            solution.time,
            root_cost,
            base_cost_evaluator,
            parameters,
        ))
    }
}
