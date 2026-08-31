use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    create_dual_bound_local_search, BeamSearchParameters, CabsParameters, FEvaluatorType,
    LocalSearchMode, LocalSearchParameters, Neighborhoods, Search,
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
            "hill_climbing" => LocalSearchMode::HillClimbing,
            "simulated_annealing" => {
                let initial_temperature =
                    match map.get(&yaml_rust::Yaml::from_str("initial_temperature")) {
                        Some(value) => util::get_numeric::<f64>(value)?,
                        None => 10.0,
                    };
                let cooling_rate = match map.get(&yaml_rust::Yaml::from_str("cooling_rate")) {
                    Some(value) => util::get_numeric::<f64>(value)?,
                    None => 0.9999,
                };
                LocalSearchMode::SimulatedAnnealing {
                    T0: initial_temperature,
                    alpha: cooling_rate,
                }
            }
            mode => {
                return Err(util::YamlContentErr::new(format!(
                    "unexpected value for `mode`: `{mode}`",
                ))
                .into())
            }
        },
        None => LocalSearchMode::HillClimbing,
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
    let local_search_parameters = LocalSearchParameters {
        seed,
        mode,
        neighborhoods: Neighborhoods { swap, relocate, replace, twoopt },
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

    Ok(create_dual_bound_local_search(
        Rc::new(model),
        local_search_parameters,
        cabs_parameters,
        f_evaluator_type,
    ))
}
