//! Reports `ModelAwareTransitionMutex::conflicts(a, b)` pairs for a grounded
//! model: whether two specific grounded transitions' read/write sets overlap
//! (i.e. can't be freely swapped in an already-feasible trajectory). Pairs
//! with identical grounded parameters (e.g. bin-packing's `pack`/
//! `open-and-pack` for the same item) are excluded -- they're alternate
//! encodings of one decision about one task, not a precedence between two
//! distinct tasks.
//!
//! Usage: model_aware_transition_dependencies <domain.yaml> <problem.yaml> [--json]

use didp_yaml::dypdl_parser;
use dypdl::Transition;
use dypdl_heuristic_search::search_algorithm::data_structure::ModelAwareTransitionMutex;
use dypdl_heuristic_search::search_algorithm::SuccessorGenerator;
use std::env;
use std::fs;
use std::process;

fn main() {
    let mut args = env::args().skip(1);
    let domain = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a domain file name.");
        process::exit(1);
    });
    let problem = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a problem file name.");
        process::exit(1);
    });
    let json_output = args.next().as_deref() == Some("--json");

    let domain_content = fs::read_to_string(&domain).unwrap_or_else(|e| {
        eprintln!("Couldn't read domain file {domain}: {e:?}");
        process::exit(1);
    });
    let domain_yaml = yaml_rust::YamlLoader::load_from_str(&domain_content).unwrap_or_else(|e| {
        eprintln!("Couldn't parse domain file {domain}: {e:?}");
        process::exit(1);
    });
    let domain_yaml = &domain_yaml[0];

    let problem_content = fs::read_to_string(&problem).unwrap_or_else(|e| {
        eprintln!("Couldn't read problem file {problem}: {e:?}");
        process::exit(1);
    });
    let problem_yaml = yaml_rust::YamlLoader::load_from_str(&problem_content).unwrap_or_else(|e| {
        eprintln!("Couldn't parse problem file {problem}: {e:?}");
        process::exit(1);
    });
    let problem_yaml = &problem_yaml[0];

    let model = dypdl_parser::load_model_from_yaml(domain_yaml, problem_yaml).unwrap_or_else(|e| {
        eprintln!("Couldn't load model from {domain} / {problem}: {e:?}");
        process::exit(1);
    });

    let model = std::rc::Rc::new(model);
    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);

    let transitions = generator
        .transitions
        .iter()
        .chain(generator.forced_transitions.iter())
        .map(|t| t.as_ref().clone())
        .collect::<Vec<_>>();
    let num_transitions = transitions.len();
    let params: Vec<_> = transitions
        .iter()
        .map(|t| (t.forced, t.id, t.transition.parameter_values.clone()))
        .collect();

    let mutex = ModelAwareTransitionMutex::new(transitions, &model.table_registry);

    // O(n^2) scan over all unordered pairs: count each transition that
    // conflicts with at least one other, differently-parameterized
    // transition, and the total number of such cross-task conflicting pairs.
    let mut num_dependent_transitions = 0usize;
    let mut num_pairs = 0usize;
    for (i, (forced_a, id_a, params_a)) in params.iter().enumerate() {
        let mut has_conflict = false;
        for (forced_b, id_b, params_b) in params.iter().skip(i + 1) {
            if params_a == params_b {
                continue;
            }
            if mutex.conflicts(*forced_a, *id_a, *forced_b, *id_b) {
                num_pairs += 1;
                has_conflict = true;
            }
        }
        if has_conflict {
            num_dependent_transitions += 1;
        }
    }
    let has_hard_dependency = num_pairs > 0;

    if json_output {
        let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        println!(
            "{{\"domain\": \"{}\", \"problem\": \"{}\", \"num_transitions\": {num_transitions}, \"has_feasibility_structure\": {}, \"num_dependent_transitions\": {num_dependent_transitions}, \"num_pairs\": {num_pairs}, \"has_hard_dependency\": {has_hard_dependency}}}",
            escape(&domain),
            escape(&problem),
            mutex.has_feasibility_structure,
        );
    } else {
        println!("domain: {domain}");
        println!("problem: {problem}");
        println!("transitions (incl. forced): {num_transitions}");
        println!("model has any extractable precondition structure: {}", mutex.has_feasibility_structure);
        println!("transitions with a cross-task conflict: {num_dependent_transitions}");
        println!("cross-task conflicting pairs: {num_pairs}");
        println!("has hard dependency: {has_hard_dependency}");
    }
}
