use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    create_dual_bound_deorder_lns, BeamSearchParameters, CabsParameters, DeorderLnsParameters,
    FEvaluatorType, Search,
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
    // Widest position gap considered when freeing a precedence pair; unset means unbounded.
    let max_gap = match map.get(&yaml_rust::Yaml::from_str("max_gap")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected Integer for `max_gap`, but found `{value:?}`",
            ))
            .into())
        }
        None => None,
    };
    // Flat probability of freeing each freeable pair during the destroy step.
    // Higher is generally better here than in the sampling-based deorder
    // solver, up to what `beam_width` can search effectively -- the repair
    // step optimizes over the relaxation rather than sampling blindly.
    let free_probability = match map.get(&yaml_rust::Yaml::from_str("free_probability")) {
        Some(value) => util::get_numeric::<f64>(value)?,
        None => 0.6,
    };
    // Absolute expected-frees override -- see DeorderLnsParameters::target_frees.
    let target_frees = match map.get(&yaml_rust::Yaml::from_str("target_frees")) {
        Some(value) => Some(util::get_numeric::<f64>(value)?),
        None => None,
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
    // Discount on free_probability for a pair that's a genuine precedence
    // constraint (provably infeasible to invert), not just costly. Not 0 --
    // the cascade can need a hard pair freed as a stepping stone toward a
    // wider, useful one.
    let gamma = match map.get(&yaml_rust::Yaml::from_str("gamma")) {
        Some(value) => util::get_numeric::<f64>(value)?,
        None => 0.1,
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

    let parameters = solver_parameters::parse_from_map(map)?;
    let deorder_lns_parameters = DeorderLnsParameters {
        seed,
        max_gap,
        free_probability,
        target_frees,
        beam_width,
        max_branching,
        gamma,
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

    Ok(create_dual_bound_deorder_lns(
        Rc::new(model),
        deorder_lns_parameters,
        cabs_parameters,
        f_evaluator_type,
    ))
}
