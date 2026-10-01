//! A module for search algorithms.

mod acps;
mod apps;
mod beam_search;
mod best_first_search;
mod breadth_first_search;
mod cabs;
mod cbfs;
pub mod data_structure;
mod dbdfs;
mod dd_lns;
mod deorder_lns;
mod dfbb;
mod forward_recursion;
mod position_lns;
mod lnbs;
mod local_search;
mod model_aware_local_search;
mod neighborhood_search;
mod randomized_restricted_dd;
mod rollout;
mod search;
pub mod util;

pub use acps::{Acps, ProgressiveSearchParameters};
pub use apps::Apps;
pub use beam_search::{beam_search, BeamSearchParameters};
pub use best_first_search::BestFirstSearch;
pub use breadth_first_search::{BreadthFirstSearch, BrfsParameters};
pub use cabs::{Cabs, CabsParameters};
pub use cbfs::Cbfs;
pub use data_structure::{
    BfsNode, CostNode, CustomFNode, FNode, FNodeEvaluators, InsertionResult, StateInRegistry,
    StateRegistry, SuccessorGenerator, TransitionMutex, TransitionWithCustomCost, TransitionWithId,
    WeightedFNode,
};
pub use dbdfs::{Dbdfs, DbdfsParameters};
pub use dd_lns::{DdLns, DdLnsParameters};
pub use deorder_lns::{DeorderLns, DeorderLnsParameters};
pub use dfbb::Dfbb;
pub use forward_recursion::ForwardRecursion;
pub use position_lns::{PositionLns, PositionLnsParameters};
pub use lnbs::{Lnbs, LnbsParameters};
pub use local_search::{
    LocalSearch, LocalSearchMode, LocalSearchParameters, Neighborhoods, NeighborhoodSelection,
};
pub use model_aware_local_search::{
    ModelAwareLocalSearch, ModelAwareLocalSearchMode, ModelAwareLocalSearchParameters,
    ModelAwareNeighborhoods, ModelAwareNeighborhoodSelection,
};
pub use neighborhood_search::NeighborhoodSearchInput;
pub use randomized_restricted_dd::{randomized_restricted_dd, RandomizedRestrictedDDParameters};
pub use rollout::{get_solution_cost_and_suffix, get_trace, rollout, RolloutResult};
pub use search::{Parameters, Search, SearchInput, Solution};
