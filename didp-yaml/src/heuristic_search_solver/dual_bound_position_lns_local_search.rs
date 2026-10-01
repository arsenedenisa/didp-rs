use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    create_dual_bound_position_lns_local_search, BeamSearchParameters, CabsParameters,
    FEvaluatorType, ModelAwareLocalSearchMode, ModelAwareLocalSearchParameters,
    ModelAwareNeighborhoodSelection, ModelAwareNeighborhoods, PositionLnsParameters, Search,
};
use std::error::Error;
use std::rc::Rc;
use std::{fmt, str};

/// A multi-armed bandit alternating `dual_bound_position_lns` and
/// `dual_bound_model_aware_local_search` over short wall-clock time slices: both solvers stay
/// alive for the whole search (never reconstructed), and a sliding-window UCB policy picks which
/// one gets the next slice based on measured cost-improvement-per-slice, shifting attention toward
/// whichever is currently paying off. See dypdl-heuristic-search's
/// dual_bound_position_lns_local_search.rs module doc for the full design and why this replaced an
/// earlier "run one until stuck, then hand off" version. Accepts every parameter both standalone
/// solvers accept (see `dual_bound_position_lns` and `dual_bound_model_aware_local_search`), under
/// the same names (`dominance_pruning` and `cabs_on_stuck` from the local-search side are not
/// applicable/supported here -- `cabs_on_stuck` is meaningless once PositionLns itself already
/// plays the "escape a local optimum" role).
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

    // --- PositionLns-specific parameters (same names/defaults as `dual_bound_position_lns`) ---
    let neighborhood_size = match map.get(&yaml_rust::Yaml::from_str("neighborhood_size")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `neighborhood_size`, but found `{value:?}`",
            ))
            .into())
        }
        None => 8,
    };
    let position_weight = match map.get(&yaml_rust::Yaml::from_str("position_weight")) {
        Some(value) => util::get_numeric::<f64>(value)?,
        None => 1.0,
    };
    let cost_weight = match map.get(&yaml_rust::Yaml::from_str("cost_weight")) {
        Some(value) => util::get_numeric::<f64>(value)?,
        None => 1.0,
    };
    let relatedness_determinism =
        match map.get(&yaml_rust::Yaml::from_str("relatedness_determinism")) {
            Some(value) => util::get_numeric::<f64>(value)?,
            None => 5.0,
        };
    let full_reinsert_probability =
        match map.get(&yaml_rust::Yaml::from_str("full_reinsert_probability")) {
            Some(value) => util::get_numeric::<f64>(value)?,
            None => 1.0,
        };
    let enable_stagnation_seed =
        match map.get(&yaml_rust::Yaml::from_str("enable_stagnation_seed")) {
            Some(yaml_rust::Yaml::Boolean(value)) => *value,
            Some(value) => {
                return Err(util::YamlContentErr::new(format!(
                    "expected Boolean for `enable_stagnation_seed`, but found `{value:?}`",
                ))
                .into())
            }
            None => false,
        };
    let beam_width = match map.get(&yaml_rust::Yaml::from_str("beam_width")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `beam_width`, but found `{value:?}`",
            ))
            .into())
        }
        None => 16,
    };
    let max_branching = match map.get(&yaml_rust::Yaml::from_str("max_branching")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `max_branching`, but found `{value:?}`",
            ))
            .into())
        }
        None => 20,
    };
    let insertion_slack = match map.get(&yaml_rust::Yaml::from_str("insertion_slack")) {
        Some(yaml_rust::Yaml::Integer(value)) => *value as usize,
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `insertion_slack`, but found `{value:?}`",
            ))
            .into())
        }
        None => 0,
    };

    // How many wall-clock seconds each bandit turn gets before control returns to the bandit to
    // pick again (possibly the same arm again). See dual_bound_position_lns_local_search.rs's
    // module doc for why this replaced a per-solver "stuck" threshold.
    let slice_seconds = match map.get(&yaml_rust::Yaml::from_str("slice_seconds")) {
        Some(value) => util::get_numeric::<f64>(value)?,
        None => 2.0,
    };

    // --- ModelAwareLocalSearch-specific parameters (same names/defaults as
    // `dual_bound_model_aware_local_search`; `dominance_pruning` and `cabs_on_stuck` are not
    // supported here -- see this module's doc) ---
    let mode = match map.get(&yaml_rust::Yaml::from_str("mode")) {
        Some(yaml_rust::Yaml::String(value)) => match &value[..] {
            "hill_climbing" => ModelAwareLocalSearchMode::HillClimbing,
            "simulated_annealing" => {
                let initial_temperature =
                    match map.get(&yaml_rust::Yaml::from_str("initial_temperature")) {
                        Some(value) => util::get_numeric::<f64>(value)?,
                        None => 10.0,
                    };
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

    // --- shared CABS parameters, used for the one-time initial-solution round ---
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

    let parameters = solver_parameters::parse_from_map(map)?;
    let position_lns_parameters = PositionLnsParameters {
        seed,
        neighborhood_size,
        position_weight,
        cost_weight,
        relatedness_determinism,
        full_reinsert_probability,
        enable_stagnation_seed,
        beam_width,
        max_branching,
        insertion_slack,
        stall_limit: None,
        stall_time_limit: None,
        parameters,
    };
    let local_search_parameters = ModelAwareLocalSearchParameters {
        seed,
        mode,
        neighborhoods: ModelAwareNeighborhoods { swap, relocate, replace, twoopt },
        neighborhood_selection,
        dominance_pruning: false,
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

    Ok(create_dual_bound_position_lns_local_search(
        Rc::new(model),
        position_lns_parameters,
        local_search_parameters,
        cabs_parameters,
        f_evaluator_type,
        slice_seconds,
    ))
}
