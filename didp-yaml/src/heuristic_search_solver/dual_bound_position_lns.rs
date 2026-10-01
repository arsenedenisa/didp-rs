use super::solver_parameters;
use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{
    create_dual_bound_position_lns, BeamSearchParameters, CabsParameters, FEvaluatorType,
    PositionLnsParameters, Search,
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
    // Number of positions freed per destroy step.
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
    // Weight of sequence-index distance in the relatedness score used to
    // grow the freed set.
    let position_weight = match map.get(&yaml_rust::Yaml::from_str("position_weight")) {
        Some(value) => util::get_numeric::<f64>(value)?,
        None => 1.0,
    };
    // Weight of marginal-cost similarity in the relatedness score.
    let cost_weight = match map.get(&yaml_rust::Yaml::from_str("cost_weight")) {
        Some(value) => util::get_numeric::<f64>(value)?,
        None => 1.0,
    };
    // Shaw-removal-style determinism exponent for the freed-set growth
    // step; 1.0 is uniform random among candidates, higher values pick
    // closer to the most-related candidate more often.
    let relatedness_determinism =
        match map.get(&yaml_rust::Yaml::from_str("relatedness_determinism")) {
            Some(value) => util::get_numeric::<f64>(value)?,
            None => 5.0,
        };
    // Probability of repairing with the full-reinsertion neighborhood
    // (freed positions may move anywhere), vs. restricted shuffle (freed
    // positions may only permute among themselves). Defaults to 1.0
    // (full-reinsertion only).
    let full_reinsert_probability =
        match map.get(&yaml_rust::Yaml::from_str("full_reinsert_probability")) {
            Some(value) => util::get_numeric::<f64>(value)?,
            None => 1.0,
        };
    // When true, biases the freed set's seed position toward positions that
    // recently sat in a rejected freed set, instead of picking uniformly at
    // random -- see PositionLnsParameters::enable_stagnation_seed.
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
    // Extra real-model transitions the repair beam may insert beyond the
    // incumbent's own transition count -- see
    // PositionLnsParameters::insertion_slack. 0 (default) reproduces the
    // original fixed-multiset repair behavior.
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
    // When set, search_next reports "stuck" (terminated, not timed out) after this many
    // consecutive non-improving iterations, instead of only ever stopping on the time limit --
    // see PositionLnsParameters::stall_limit's doc. Unset (default) reproduces the original
    // behavior exactly.
    let stall_limit = match map.get(&yaml_rust::Yaml::from_str("stall_limit")) {
        Some(yaml_rust::Yaml::Integer(value)) if *value >= 1 => Some(*value as u64),
        Some(value) => {
            return Err(util::YamlContentErr::new(format!(
                "expected a positive Integer for `stall_limit`, but found `{value:?}`",
            ))
            .into())
        }
        None => None,
    };
    // Same "stuck" signal as `stall_limit`, but measured in elapsed seconds since the last
    // improvement instead of a raw iteration count -- see PositionLnsParameters::stall_time_limit's
    // doc. Unset (default) reproduces the original behavior exactly.
    let stall_time_limit = match map.get(&yaml_rust::Yaml::from_str("stall_time_limit")) {
        Some(value) => Some(util::get_numeric::<f64>(value)?),
        None => None,
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
        stall_limit,
        stall_time_limit,
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

    Ok(create_dual_bound_position_lns(
        Rc::new(model),
        position_lns_parameters,
        cabs_parameters,
        f_evaluator_type,
    ))
}
