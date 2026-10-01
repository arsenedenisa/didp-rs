use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    create_dual_bound_model_aware_local_search, BeamSearchParameters, CabsParameters, FEvaluatorType,
    ModelAwareLocalSearchMode, ModelAwareLocalSearchParameters, ModelAwareNeighborhoods, ModelAwareNeighborhoodSelection, Search,
};
use std::error::Error;
use std::rc::Rc;
use std::{fmt, str};

pub fn load_from_yaml<T>(
    model: dypdl::Model,
    config: &yaml_rust::Yaml,
) -> Result<Box<dyn Search<T>>, Box<dyn Error>>
where
    T: Numeric + Ord + fmt::Display + Send + Sync + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            return Err(util::YamlContentErr::new(format!(
                "expected Hash for the solver config, but found `{config:?}`",
            ))
            .into())
        }
    };
    let f_evaluator_type = match map.get(&yaml_rust::Yaml::from_str("f")) {
        Some(yaml_rust::Yaml::String(string)) => match &string[..] {
            "+" => FEvaluatorType::Plus,
            "max" => FEvaluatorType::Max,
            "min" => FEvaluatorType::Min,
            "h" => FEvaluatorType::Overwrite,
            op => {
                return Err(util::YamlContentErr::new(format!(
                    "unexpected operator for `{op}` for `f`",
                ))
                .into())
            }
        },
        None => FEvaluatorType::default(),
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `f`, but found `{value:?}`",
            ))
            .into())
        }
    };
    let seed = match map.get(&yaml_rust::Yaml::from_str("seed")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as u64,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `seed`, but found `{value:?}`",
            ))
            .into())
        }
        None => 2023,
    };
    // `mode` selects between plain hill climbing and simulated annealing.
    let mode = match map.get(&yaml_rust::Yaml::from_str("mode")) {
        Some(yaml_rust::Yaml::String(value)) => match &value[..] {
            "hill_climbing" => ModelAwareLocalSearchMode::HillClimbing,
            "simulated_annealing" => {
                let initial_temperature =
                    match map.get(&yaml_rust::Yaml::from_str("initial_temperature")) {
                        Some(value) => util::get_numeric::<f64>(value)?,
                        None => 10.0,
                    };
                // Absolute temperature the cooling schedule should reach right as
                // time_limit runs out (see ModelAwareLocalSearchMode::SimulatedAnnealing's doc
                // for why cooling is time-based rather than a per-call `cooling_rate`
                // multiplicative decay -- that old key is gone because it can't be
                // calibrated without knowing how many moves a run will evaluate,
                // which varies wildly by problem/instance). Default reaches 1% of
                // initial_temperature by the end of the budget.
                let final_temperature = match map.get(&yaml_rust::Yaml::from_str("final_temperature"))
                {
                    Some(value) => util::get_numeric::<f64>(value)?,
                    None => initial_temperature * 0.01,
                };
                ModelAwareLocalSearchMode::SimulatedAnnealing {
                    T0: initial_temperature,
                    final_temp_ratio: final_temperature / initial_temperature,
                }
            }
            mode => {
                return Err(util::YamlContentErr::new(format!(
                    "unexpected value for `mode`: `{mode}`",
                ))
                .into())
            }
        },
        None => ModelAwareLocalSearchMode::HillClimbing,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `mode`, but found `{value:?}`",
            ))
            .into())
        }
    };
    // `swap`/`relocate` select which neighborhoods are tried; both default to on.
    let swap = match map.get(&yaml_rust::Yaml::from_str("swap")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => true,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `swap`, but found `{value:?}`",
            ))
            .into())
        }
    };
    let relocate = match map.get(&yaml_rust::Yaml::from_str("relocate")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => true,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `relocate`, but found `{value:?}`",
            ))
            .into())
        }
    };
    // Defaults to off, unlike `swap`/`relocate`, so existing configs that don't mention
    // `replace` keep their previous meaning instead of silently gaining a third neighborhood.
    let replace = match map.get(&yaml_rust::Yaml::from_str("replace")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `replace`, but found `{value:?}`",
            ))
            .into())
        }
    };
    // Defaults to off, same reasoning as `replace`.
    let twoopt = match map.get(&yaml_rust::Yaml::from_str("twoopt")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `twoopt`, but found `{value:?}`",
            ))
            .into())
        }
    };
    if !swap && !relocate && !replace && !twoopt {
        return Err(util::YamlContentErr::new(
            "at least one of `swap`, `relocate`, `replace`, or `twoopt` must be enabled".to_string(),
        )
        .into());
    }
    let neighborhood_selection = match map.get(&yaml_rust::Yaml::from_str("neighborhood_selection")) {
        Some(yaml_rust::Yaml::String(value)) => match &value[..] {
            "random" => ModelAwareNeighborhoodSelection::Random,
            "sequential" => {
                let iterations =
                    match map.get(&yaml_rust::Yaml::from_str("neighborhood_switch_iterations")) {
                        Some(yaml_rust::Yaml::Integer(value)) if *value >= 1 => *value as usize,
                        Some(value) => {
                            return Err(util::YamlContentErr::new(format!(
                                "expected a positive Integer for `neighborhood_switch_iterations`, but found `{value:?}`",
                            ))
                            .into())
                        }
                        None => 20,
                    };

                ModelAwareNeighborhoodSelection::Sequential { iterations }
            }
            "adaptive" => {
                let iterations =
                    match map.get(&yaml_rust::Yaml::from_str("neighborhood_switch_iterations")) {
                        Some(yaml_rust::Yaml::Integer(value)) if *value >= 1 => *value as usize,
                        Some(value) => {
                            return Err(util::YamlContentErr::new(format!(
                                "expected a positive Integer for `neighborhood_switch_iterations`, but found `{value:?}`",
                            ))
                            .into())
                        }
                        None => 20,
                    };
                let exploration_constant =
                    match map.get(&yaml_rust::Yaml::from_str("neighborhood_exploration_constant")) {
                        Some(value) => {
                            let value = util::get_numeric::<f64>(value)?;

                            if value < 0.0 {
                                return Err(util::YamlContentErr::new(format!(
                                    "expected `neighborhood_exploration_constant` to be non-negative, but found `{value}`",
                                ))
                                .into());
                            }

                            value
                        }
                        None => std::f64::consts::SQRT_2,
                    };
                let window_size =
                    match map.get(&yaml_rust::Yaml::from_str("neighborhood_window_size")) {
                        Some(yaml_rust::Yaml::Integer(value)) if *value >= 1 => *value as usize,
                        Some(value) => {
                            return Err(util::YamlContentErr::new(format!(
                                "expected a positive Integer for `neighborhood_window_size`, but found `{value:?}`",
                            ))
                            .into())
                        }
                        None => 50,
                    };

                ModelAwareNeighborhoodSelection::Adaptive { iterations, exploration_constant, window_size }
            }
            selection => {
                return Err(util::YamlContentErr::new(format!(
                    "unexpected value for `neighborhood_selection`: `{selection}`",
                ))
                .into())
            }
        },
        None => ModelAwareNeighborhoodSelection::default(),
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected String for `neighborhood_selection`, but found `{value:?}`",
            ))
            .into())
        }
    };
    let cabs_beam_size = match map.get(&yaml_rust::Yaml::from_str("cabs_initial_beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `cabs_initial_beam_size`, but found `{value:?}`",
            ))
            .into())
        }
        None => 1,
    };
    let cabs_max_beam_size = match map.get(&yaml_rust::Yaml::from_str("cabs_max_beam_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `cabs_max_beam_size`, but found `{value:?}`",
            ))
            .into())
        }
        None => None,
    };
    // When set, hill climbing alternates with CABS instead of stopping the first time it gets
    // stuck in a local optimum: each time that happens, it runs one more bounded CABS round
    // (using `cabs_initial_beam_size`/`cabs_max_beam_size` above) and resumes hill climbing from
    // whatever that round finds, until the time limit is reached or CABS proves the incumbent
    // optimal. Only meaningful with `mode: hill_climbing`.
    //
    // Defaults to true: combined with `neighborhood_selection`'s own default (Random), this
    // makes "hc_random_cabs" the default MALS configuration when a config specifies nothing
    // beyond `mode: hill_climbing`. Explicit `cabs_on_stuck: false` still opts out.
    let cabs_on_stuck = match map.get(&yaml_rust::Yaml::from_str("cabs_on_stuck")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => true,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `cabs_on_stuck`, but found `{value:?}`",
            ))
            .into())
        }
    };
    if cabs_on_stuck && !matches!(mode, ModelAwareLocalSearchMode::HillClimbing) {
        return Err(util::YamlContentErr::new(
            "`cabs_on_stuck` requires `mode: hill_climbing`".to_string(),
        )
        .into());
    }

    // When true, before rolling out a candidate move, reject it without evaluation if the
    // model's forced-transition/transition_dominance rules already rule it out at its first
    // changed position -- see ModelAwareLocalSearch::passes_structural_pruning's doc.
    // Off by default: most domains declare neither mechanism, and it's a new toggle.
    let dominance_pruning = match map.get(&yaml_rust::Yaml::from_str("dominance_pruning")) {
        Some(yaml_rust::Yaml::Boolean(value)) => *value,
        None => false,
        value => {
            return Err(util::YamlContentErr::new(format!(
                "expected Boolean for `dominance_pruning`, but found `{value:?}`",
            ))
            .into())
        }
    };

    let parameters = solver_parameters::parse_from_map(map)?;
    let local_search_parameters = ModelAwareLocalSearchParameters {
        seed,
        mode,
        neighborhoods: ModelAwareNeighborhoods { swap, relocate, replace, twoopt },
        neighborhood_selection,
        dominance_pruning,
        parameters,
    };
    let cabs_parameters = CabsParameters {
        max_beam_size: cabs_max_beam_size,
        beam_search_parameters: BeamSearchParameters {
            parameters,
            beam_size: cabs_beam_size,
            keep_all_layers: false,
        },
    };

    Ok(create_dual_bound_model_aware_local_search(
        Rc::new(model),
        local_search_parameters,
        cabs_parameters,
        f_evaluator_type,
        cabs_on_stuck,
    ))
}
