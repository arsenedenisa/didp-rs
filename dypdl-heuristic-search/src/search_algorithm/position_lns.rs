// Large neighborhood search whose destroy step frees a *set of positions* in
// the incumbent (chosen structurally, not by a per-pair coin flip -- see
// deorder_lns.rs and graph_relaxation_lns.rs for two earlier, more complex
// designs), and whose repair step is a bounded beam search for the best
// valid linear extension of what freeing that set allows.
//
// Each iteration:
//   1. Pick a set S of positions to free by relatedness-based seed-and-grow
//      (see select_freed_positions): a seed position is chosen (uniformly
//      at random by default, or weighted toward stagnating positions when
//      enable_stagnation_seed is set), then grown one position at a time,
//      always picking (Shaw-removal style, with a determinism exponent
//      controlling how greedy the pick is) whichever remaining position is
//      closest to the freed set so far under a relatedness distance
//      combining sequence-index distance and marginal-cost similarity
//      (|w[a] - w[c]|, w = current_costs[p] - current_costs[p-1]).
//   2. Pick one of two repair neighborhoods for this iteration:
//      - Restricted shuffle: only pairs with BOTH endpoints in S are
//        relaxed, so a freed position can only permute among other freed
//        positions, never cross a frozen one.
//      - Full reinsertion: pairs with EITHER endpoint in S are relaxed, so a
//        freed position can move anywhere relative to any other position;
//        only frozen-frozen pairs stay locked in their original order.
//      Both are the same "keep everything not relaxed, beam search the
//      rest" machinery (see build_successors) -- they differ only in which
//      pairs count as relaxed.
//   3. Beam search over "place the next ready original position" (ready =
//      every retained predecessor already placed), checking real DyPDL
//      applicability/state-constraints at each step, ranked by accumulated
//      cost (or f = g + h when the model has a dual bound). Self-contained
//      over the fixed known transition multiset rather than integrating
//      with the FNode/CABS dominance machinery, since the candidate pool at
//      each step is already small and known (the incumbent's own
//      transitions).
//
// At each position the beam may substitute any other grounded transition
// sharing the same parameter values (see `compute_alternatives`), e.g.
// CVRP's `visit(to)` and `visit-via-depot(to)`.
//
// Fixed-multiset by default (insertion_slack == 0): the repair beam places
// exactly the incumbent's own n transitions, substituting only among
// same-parameter alternatives. When insertion_slack > 0, a beam entry may
// additionally spend up to that many extra steps applying any real,
// currently-applicable model transition while a freed position remains
// unplaced, which is what lets the freed region end up holding more or
// fewer transitions than it started with (an inserted transition doesn't
// consume a position or reduce any in-degree, so it doesn't affect the kept
// edges or the incumbent-lineage protection below).
//
// A sibling toggle, free_replace_enabled, widens a freed position's own
// candidates (still exactly one transition placed at that position, no
// length change) from same-parameter alternatives to the full grounded
// catalog -- a true one-for-one replace. Independent of insertion_slack,
// which grows/shrinks the sequence length instead. Auto-disabled when the
// initial solution is a permutation of the whole reachable transition
// catalog (same construction-time check as ModelAwareLocalSearch's), since
// substituting an unrelated transition into a freed slot there is provably
// infeasible for a permutation-shaped domain.
//
// Self-contained by design: duplicates the handful of small primitives
// (compute_alternatives, NO_CATALOG_ID, the forced-transition short-circuit,
// TransitionMutex-backed pruning, dominance filtering) rather than sharing a
// module with deorder_lns.rs or graph_relaxation_lns.rs.

use super::data_structure::{
    classify_transition_cardinality, exceed_bound, HashableState, SuccessorGenerator,
    TransitionCardinality, TransitionMutex, TransitionWithId,
};
use super::rollout::get_trace;
use super::search::{Parameters, Search, Solution};
use super::util::{print_primal_bound, TimeKeeper};
use crate::f_evaluator_type::FEvaluatorType;
use dypdl::{
    variable_type::{Element, Numeric},
    Model, ParentAndChildStateFunctionCache, State, StateFunctionCache, Transition,
    TransitionInterface,
};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::str;

/// Parameters for [`PositionLns`].
#[derive(Debug, Clone, Copy)]
pub struct PositionLnsParameters<T> {
    /// Random seed.
    pub seed: u64,
    /// Number of positions freed per destroy step.
    pub neighborhood_size: usize,
    /// Weight of sequence-index distance in the relatedness score used to
    /// grow the freed set (see select_freed_positions).
    pub position_weight: f64,
    /// Weight of marginal-cost similarity in the relatedness score.
    pub cost_weight: f64,
    /// Shaw-removal-style determinism exponent: at each growth step, the
    /// remaining candidates are ranked by relatedness distance and one is
    /// picked at index `floor(rng()^determinism * candidates.len())`.
    /// Higher values pick closer to the top of the ranking more often
    /// (fully deterministic "always closest" as this -> infinity); 1.0 is a
    /// uniform random pick among all candidates, ignoring the ranking.
    pub relatedness_determinism: f64,
    /// Probability of repairing with the "full reinsertion" neighborhood
    /// (freed positions may move anywhere in the sequence), vs. "restricted
    /// shuffle" (freed positions may only permute among themselves).
    pub full_reinsert_probability: f64,
    /// When true, the freed set's seed position (see select_freed_positions)
    /// is sampled with probability proportional to `1 + stagnation count`
    /// instead of uniformly at random -- biasing seeds toward positions that
    /// have recently sat in freed sets which failed to produce an accepted
    /// move. False (default) reproduces the original uniform-random seed.
    /// Helps on some domains and costs on others with no known cheap signal
    /// to auto-gate on -- set per model by hand for now.
    pub enable_stagnation_seed: bool,
    /// Beam width for the repair step.
    pub beam_width: usize,
    /// Cap on how many ready positions to try expanding from a single beam
    /// entry per step, to bound work when many positions become ready at
    /// once (mainly a full-reinsertion concern, since a freed position
    /// there can become ready far earlier than its original slot).
    pub max_branching: usize,
    /// Enables flexible-length repair -- see beam_repair's doc. At 0
    /// (default), repair is exactly the original fixed-multiset behavior:
    /// every repaired solution has exactly as many transitions as the
    /// incumbent it started from. Above 0, it's both an insertion budget
    /// (a beam entry may spend up to this many extra steps applying any
    /// real, currently-applicable model transition, not just a specific
    /// position's alternatives) AND what enables deletion (a beam entry
    /// with a freed position still unplaced is opportunistically checked
    /// against the model's base case every step, so a freed position that
    /// turns out to be unnecessary can be left unplaced instead of forced
    /// in).
    pub insertion_slack: usize,
    /// When set, `search_next` returns early (`terminated: true, time_out: false` -- a genuine
    /// "stuck" signal, distinct from running out of the time budget) once this many consecutive
    /// iterations in a row have found no improvement over `current_cost`. `None` (default)
    /// reproduces the original behavior exactly: the destroy/repair loop only ever stops by
    /// exhausting the time limit, since unlike `LocalSearch`'s hill climbing, sampling a fresh
    /// destroy set is never actually exhausted -- there's always another one to try. This gives
    /// PositionLns a way to self-report "not making progress" for use as the "stuck" half of an
    /// alternating hybrid (see `dual_bound_position_lns_local_search.rs`). Only meaningful
    /// together with `sa_enabled` off, since SA-driven acceptance of worsening moves would trip
    /// a stall counter on normal behavior, not genuine stagnation.
    pub stall_limit: Option<u64>,
    /// Same "stuck" signal as `stall_limit`, but measured in elapsed wall-clock seconds since the
    /// last improvement over `current_cost` instead of a raw iteration count. Preferred over
    /// `stall_limit` for driving an alternating hybrid: how many iterations a fixed time budget
    /// buys varies by orders of magnitude across domains and instances, so a fixed iteration
    /// count is either unreachable or trips almost immediately depending on the domain. `None`
    /// (default) reproduces the original behavior exactly. If both this and `stall_limit` are
    /// set, whichever trips first ends the search.
    pub stall_time_limit: Option<f64>,
    pub parameters: Parameters<T>,
}

/// One partial linear extension under construction during the repair beam
/// search: which original positions have been placed (in order), the
/// remaining in-degree of every position under the destroy step's kept
/// edges, and the resulting DyPDL state/cost of applying them in that order.
#[derive(Clone)]
struct BeamEntry<T> {
    // The actual transition placed at each step, in order -- not
    // necessarily `current[p]` for the position `p` it was placed at (see
    // `alternatives`). Rc, not Transition: cloned for every candidate at
    // every beam step, and this grows with search depth -- see
    // deorder_lns.rs's identical field doc for the measured cost of using
    // owned Transitions here instead.
    placed: Vec<Rc<TransitionWithId>>,
    // Original position filled at each step, parallel to `placed`.
    placed_positions: Vec<usize>,
    placed_mask: Vec<bool>,
    in_degree: Vec<usize>,
    // Grown incrementally: every time a transition is placed, its
    // TransitionMutex forbidden-after set is unioned in, so a future
    // candidate can be rejected in O(1) instead of paying for
    // is_applicable/apply/eval_cost to rediscover the same illegality.
    forbidden: HashSet<(bool, usize)>,
    state: State,
    cost: T,
    // Insertions used so far, capped at `insertion_slack` -- see
    // beam_repair's doc. Never incremented by a position-placing seed, only
    // by an inserted one (`CandidateSeed.position == None`).
    inserted: usize,
    // True iff this entry has, at every step so far, placed position `k`
    // (k = 0, 1, ...) using exactly `current[k]` -- i.e. reproduced the
    // incumbent's own order and transitions exactly. Both neighborhoods
    // here are relaxations of the incumbent's total order (restricted
    // shuffle and full reinsertion both only ever *remove* kept edges
    // relative to the incumbent, never add ones the incumbent violates), so
    // this lineage is always extendable and always reaches a feasible
    // completion at exactly `current_cost`. Protected from beam truncation
    // (see beam_repair) so a non-improving iteration is a real negative
    // result, not an artifact of truncation dropping the one candidate that
    // couldn't lose.
    is_incumbent: bool,
}

/// A large neighborhood search whose destroy set is a chosen subset of
/// positions in the incumbent (see the module doc), repaired by a bounded
/// beam search for the best valid linear extension of what freeing that
/// subset allows.
pub struct PositionLns<T: Numeric, B> {
    model: Rc<Model>,
    base_cost_evaluator: B,
    root_cost: T,
    current: Vec<Transition>,
    // Every grounded transition the model can produce, grouped by
    // parameter values -- static, model-wide, unaffected by reordering.
    by_params: HashMap<Vec<Element>, Vec<Rc<TransitionWithId>>>,
    // alternatives[p]: every grounded transition sharing current[p]'s
    // parameter values, always including current[p] itself. MUST be
    // recomputed every time `current` is reordered -- see deorder_lns.rs's
    // identical field doc for why a one-time snapshot is a real, measured
    // bug, not just stale-and-harmless.
    alternatives: Vec<Vec<Rc<TransitionWithId>>>,
    // Reused rather than rebuilt so beam_repair's forced-transition
    // short-circuit and TransitionDominance SCC pruning are the exact same
    // logic other solvers get, not a reimplementation.
    successor_generator: SuccessorGenerator,
    // O(1) "can t2 legally follow t1" lookups from set-effect/precondition
    // analysis, consumed incrementally per beam entry via
    // BeamEntry.forbidden.
    transition_mutex: TransitionMutex,
    // Cumulative cost after applying current[0..=p], and the per-position
    // marginal cost derived from it (current_costs[p] - current_costs[p-1],
    // clamped at 0 -- same clamp rationale as deorder_lns.rs's w: a
    // negative delta under a Max-reduced objective means "didn't cost
    // anything on top of what preceded it", not "undid cost"). Refreshed on
    // every accept via refresh_trace, consumed by relatedness_distance's
    // cost-similarity term. w_range caches max(w) - min(w) (floored away
    // from 0) so relatedness_distance can normalize the cost term to
    // roughly the same [0, 1] scale as the position term without
    // recomputing the range on every call.
    current_costs: Vec<T>,
    w: Vec<f64>,
    w_range: f64,
    // Incumbent's own scalar cost (distinct from current_costs, the
    // per-position trace) -- what search_next compares each repair
    // candidate against to decide accept/reject.
    current_cost: T,
    best: Solution<T>,
    solvable: bool,
    // SA-weighted beam sampling -- ported from deorder_lns.rs's own
    // (already-tuned) mechanism rather than reinvented; see beam_repair's
    // final-selection block and DIDP_POSITION_LNS_SA's doc at the env-var
    // read site for the full rationale, all identical to deorder_lns.rs's.
    // Kept env-var-gated (not a YAML parameter yet) to match this feature's
    // maturity there: it's still under evaluation, not a settled default.
    sa_enabled: bool,
    sa_final_temp_ratio: f64,
    sa_t0: f64,
    sa_calibrated: bool,
    sa_calibration_deltas: Vec<f64>,
    sa_stall_threshold: u64,
    stall_count: u64,
    // See PositionLnsParameters::stall_limit's doc.
    stall_limit: Option<u64>,
    // See PositionLnsParameters::stall_time_limit's doc.
    stall_time_limit: Option<f64>,
    // time_keeper.elapsed_time() as of the last iteration that improved on current_cost (or 0.0
    // at construction, so an instance handed a worse starting point than root doesn't count the
    // time before its first iteration as if it had already improved). Reset on every improvement,
    // exactly like stall_count, just tracking wall-clock time instead of an iteration count.
    last_improvement_time: f64,
    // Denominator for sa_temperature's elapsed-time fraction -- same role as
    // LocalSearch's own `time_limit` field, duplicated rather than derived
    // from time_keeper on every call (see its doc there).
    time_limit: f64,
    neighborhood_size: usize,
    position_weight: f64,
    cost_weight: f64,
    relatedness_determinism: f64,
    // Prefer-gaps growth bias (DIDP_POSITION_LNS_CONTIGUITY_FILL): once the
    // freed set is large enough (freed.len() >= min_freed_for_contiguity) and
    // dense enough (gaps / span <= max_gap_ratio, gaps = span - freed.len(),
    // span = max - min + 1) to call "mostly filled" meaningful,
    // select_freed_positions narrows its growth pool to just the *gap*
    // positions -- frozen slots strictly inside [min(freed), max(freed)] --
    // so a set that's already almost one solid block gets pulled the rest of
    // the way into being one, instead of relatedness_distance being free to
    // reach back outside the span and keep it scattered.
    //
    // Density is a RATIO (gaps/span), not an absolute gap count, so the same
    // density bar applies regardless of freed-set size; min_freed_for_contiguity
    // separately keeps a too-small sample from passing on a ratio that only
    // looks dense by chance.
    // This is a preference over WHICH positions fill the destroy-set size
    // the caller already picked (size_bandit's arm, if enabled), never a
    // change to that size itself -- doing it as a second size addition on
    // top would silently break update_size_bandit's reward/time
    // attribution, which assumes arm i's destroy sets are always
    // size_arms[i] positions (see select_neighborhood's doc). Falls back to
    // the unrestricted pool whenever there are no gaps to prefer (freed
    // already a perfectly solid block, or the thresholds aren't met yet) --
    // never stalls growth outright. Does not account for
    // ModelAwareTransitionMutex-confirmed-locked positions (a position with
    // exactly one valid ordering contributes nothing if forced into the
    // freed set) -- a known gap in this filter, not yet gated on. Off by
    // default.
    contiguity_fill_enabled: bool,
    min_freed_for_contiguity: usize,
    max_gap_ratio: f64,
    // When the freed set select_freed_positions returns ends up perfectly
    // contiguous (gaps == 0, span == freed.len()) and size >= 2 (size 1
    // never has a pair to shuffle, see full_reinsert's existing size <= 1
    // guard), override the usual full_reinsert_probability coin flip and
    // force restricted shuffle instead. Rationale: restricted shuffle's
    // actual constraint -- freed positions may only permute among
    // themselves -- is a coherent neighborhood precisely when freed is one
    // solid run (it's just "find the best internal order of this block");
    // on a scattered freed set it's a much stranger move. Independent
    // of contiguity_fill_enabled -- this reads whatever freed shape came
    // out of select_freed_positions, whether or not the gap-preference
    // growth bias helped produce it. Off by default.
    shuffle_on_fill_enabled: bool,
    full_reinsert_probability: f64,
    enable_stagnation_seed: bool,
    // stagnation[p]: consecutive iterations (since last reset) that position
    // p sat in a freed set whose repair either failed outright or didn't
    // improve on current_cost. Reset to 0 for every position in the freed
    // set of an accepted move, incremented for every position in the freed
    // set of a rejected one -- see search_next. Indexed by *position*, not
    // transition identity, since the transition occupying a given index
    // changes across accepts. Only read/written when enable_stagnation_seed
    // is set; resized to match `transitions.len()` on every accept since
    // insertion_slack can change that length.
    stagnation: Vec<u32>,
    beam_width: usize,
    // Per-size-arm beam width, ported from Lnbs's neighborhood_beam_size onto
    // `size_arms`/`growth_arm` instead of window identity: arm_beam_size[i]
    // starts at `beam_width`, doubles (capped at `beam_width_max`) each
    // consecutive time arm i is chosen without an improving move, and resets
    // to `beam_width` the moment arm i produces one. `growth_arm` is always a
    // concrete index (falling back to sentinel slot 0 when `size_bandit_enabled`
    // is off), so this works standalone too. On by default (env_flag_default_true)
    // -- set DIDP_POSITION_LNS_BEAM_GROWTH=0 to opt out. Not a YAML parameter yet.
    beam_width_growth_enabled: bool,
    beam_width_max: usize,
    arm_beam_size: Vec<usize>,
    max_branching: usize,
    insertion_slack: usize,
    // When set, overrides `insertion_slack` for every iteration with that
    // iteration's own destroy-set `size` (see select_neighborhood's doc),
    // staying in lockstep with size_bandit_enabled's arm choice rather than
    // needing a second, separately-tuned constant. Off by default -- a
    // manual, per-model toggle like enable_stagnation_seed, not auto-detected
    // (domains with a fixed transition count per instance, e.g. cvrp, get
    // no benefit and only added per-iteration cost from this). Set
    // DIDP_POSITION_LNS_ADAPTIVE_INSERTION_SLACK=1 by hand for a model you
    // know is flexible-length.
    adaptive_insertion_slack_enabled: bool,
    // DIDP_POSITION_LNS_FREE_REPLACE: widens a freed, ready position's
    // candidate pool in beam_repair from `alternatives[p]` (same parameter
    // values as current[p]) to the full grounded catalog
    // (successor_generator.transitions), so the repair beam can fill that
    // slot with a transition unrelated to whatever current[p] held --
    // e.g. sequence 1 2 3 4 5 6 with positions 2..4 freed repairing to
    // 1 3 2 7 5 6: positions 1/2 (0-indexed) permute among their own
    // original transitions as before, but position 3's original transition
    // is dropped in favor of an unrelated one, all still within the
    // incumbent's own n slots -- no length change, unlike insertion_slack.
    // Existing applicability/state-constraint checks and TransitionDominance
    // pruning are unchanged and apply to these candidates exactly as to any
    // other, so a replacement that leaves some other, now-orphaned
    // transition's precondition permanently unsatisfiable simply fails
    // eval_base_cost at completion like any other infeasible beam entry.
    // Off by default -- untested, and costs a per-freed-position,
    // per-beam-step scan of the full catalog, so expect it to slow
    // iteration throughput even where it helps.
    free_replace_enabled: bool,
    rng: Pcg64Mcg,
    time_keeper: TimeKeeper,
    quiet: bool,
    first_call: bool,
    f_evaluator_type: FEvaluatorType,
    use_heuristic: bool,
    // DIDP_POSITION_LNS_DIAG: gates a per-accept iteration/elapsed-time log
    // line, matching the diagnostic pattern in deorder_lns.rs and
    // graph_relaxation_lns.rs.
    accept_diag_enabled: bool,
    iteration_count: u64,
    accept_count: u64,
    // DIDP_POSITION_LNS_EFFORT_DIAG: gates a cumulative end-of-run report on
    // where beam_repair's work actually goes (beam saturation, how much
    // TransitionDominance/TransitionMutex pruning cuts, state-duplication
    // rate, lookahead fallback behavior) -- see the eprintln! sites for the
    // exact counters reported.
    effort_diag_enabled: bool,
    effort_beam_steps: u64,
    effort_beam_seeds_generated: u64,
    effort_beam_truncated_steps: u64,
    effort_dominance_pruned: u64,
    effort_dominance_checked: u64,
    effort_mutex_skipped: u64,
    effort_candidates_attempted: u64,
    // Cumulative total vs. distinct-by-DyPDL-state entry counts across
    // every beam layer built during the run -- see the counting site in
    // beam_repair for why (state-registry dedup is exactly what LNBS's
    // shared `beam_search` does per layer and this solver's self-contained
    // `beam_repair` never does). Ratio close to 1.0 means duplicate states
    // aren't costing this solver anything; well below 1.0 means a real
    // fraction of `beam_width` is wasted on redundant copies of the same
    // state.
    effort_beam_layer_total_states: u64,
    effort_beam_layer_distinct_states: u64,
    // Where eval_f_multiset's simulation actually ends, across every call
    // this run: does the fallback path (simulated transition becomes
    // inapplicable or violates constraints) fire, and at what depth?
    // `_depth_sum` counts only successful steps before the failing one
    // (see the increment site: `simulated - 1`).
    effort_fallback_cap_count: u64,
    effort_fallback_infeasible_count: u64,
    effort_fallback_infeasible_depth_sum: u64,
    effort_full_completion_count: u64,
    // DIDP_POSITION_LNS_DEDUP: state-registry-style deduplication of a
    // beam layer's candidates by DyPDL state before truncation -- see the
    // dedup site in beam_repair (right after incumbent extraction, right
    // before the truncation sort) for the full rationale and why it must
    // run pre-truncation, not post. On by default (env_flag_default_true);
    // set DIDP_POSITION_LNS_DEDUP=0 to opt out for an ablation.
    dedup_enabled: bool,
    // DIDP_POSITION_LNS_CONTIGUITY_STATS: end-of-run report on how often
    // select_freed_positions's growth happens to land on a perfectly
    // contiguous block (span == freed.len(), the same is_contiguous check
    // shuffle_on_fill_enabled reads) among iterations where it would even
    // matter (size >= 2 -- see full_reinsert's own size <= 1 guard).
    // Independent of contiguity_fill_enabled -- counts whatever freed shape
    // select_freed_positions actually produces, fill bias on or off.
    contiguity_stats_enabled: bool,
    contiguity_stat_freed_ge2: u64,
    contiguity_stat_contiguous: u64,
    // Step-level companion to contiguity_stat_freed_ge2/contiguous above,
    // gated on the same flag: the final-freed-set contiguous_rate says
    // nothing about how often the gap-preference gate actually narrows the
    // candidate pool DURING growth (select_freed_positions's per-step
    // `pool = gap_scratch` branch) -- it could fire often but still usually
    // lose to a later step reaching back outside the span.
    // contiguity_gate_steps_narrowed counts steps where the pool was
    // actually restricted to gap_scratch (non-empty); contiguity_gate_steps_total
    // counts every growth step taken with contiguity_fill_enabled on, engaged or not.
    contiguity_gate_steps_total: u64,
    contiguity_gate_steps_narrowed: u64,
    // DIDP_POSITION_LNS_MULTISET_LOOKAHEAD: overrides eval_f_multiset's fixed
    // lookahead cap (MULTISET_LOOKAHEAD = 24) with a caller-chosen value.
    // Motivated by the effort diagnostic on cvrp, where per-candidate
    // lookahead cost dominates iteration throughput. Lowering this trades
    // f-estimate quality for iteration throughput; unset keeps the original
    // fixed 24.
    multiset_lookahead_cap: usize,
    // DIDP_POSITION_LNS_SIZE_TRACE: gates a per-ITERATION (not just
    // per-accept) log line of the chosen destroy-set size and beam width.
    // Off by default -- prints every iteration, so only meant for short ad
    // hoc diagnostic runs.
    size_trace_enabled: bool,
    // Destroy-set-size bandit, ported from Lnbs's window-depth bandit onto
    // `neighborhood_size` instead of window depth. On by default
    // (env_flag_default_true) alongside beam_width_growth_enabled -- set
    // DIDP_POSITION_LNS_SIZE_BANDIT=0 to opt out for an ablation. Not a
    // YAML parameter yet.
    //
    // Unlike Lnbs, there is no `depth_exhausted` equivalent here: Lnbs marks
    // an arm exhausted when its *complete, bounded* beam search proves
    // optimal/infeasible for that window -- beam_repair is a plain heuristic
    // beam with no such completeness signal, so every size arm stays
    // eligible for the whole run.
    size_bandit_enabled: bool,
    size_arms: Vec<usize>,
    size_reward_mean: Vec<f64>,
    size_time_mean: Vec<f64>,
    size_trials: Vec<f64>,
    size_total_trials: f64,
    size_lambda: Option<f64>,
    // DIDP_POSITION_LNS_SEED_BANDIT: UCB bandit over (seed_position,
    // growth_arm) pairs, replacing select_freed_positions's uniform-random
    // (or stagnation-weighted) seed choice with one revisitable enough to
    // learn from -- see select_seed_bandit's doc for why the key is
    // (seed, growth_arm), not the literal freed set. Off by default,
    // untested. Mutually exclusive in effect with enable_stagnation_seed
    // (see select_neighborhood's priority order) -- setting both just
    // means stagnation's bookkeeping keeps running unread.
    seed_bandit_enabled: bool,
    seed_arm_stats: HashMap<(usize, usize), SeedArmStats>,
    seed_arm_total_trials: f64,
    seed_arm_lambda: Option<f64>,
    // Same time-slicing mechanism as LocalSearch's identically-named field/doc -- an elapsed-time
    // value (same clock as time_keeper.elapsed_time()) at which search_next should return early
    // with terminated: false, letting a caller alternate this against another Search impl without
    // reconstructing this instance (and losing everything above: SA calibration, the size/seed
    // bandits' learned arm stats, ...) between turns. None (default) disables this.
    turn_deadline: Option<f64>,
}

// Per-(seed, growth_arm) UCB stats for seed_bandit_enabled -- see
// select_seed_bandit's doc. A plain struct, not three parallel maps, since
// entries are created/looked-up together as a unit.
struct SeedArmStats {
    trials: f64,
    reward_mean: f64,
    time_mean: f64,
}

// Sentinel `TransitionWithId::id` for a synthetic alternatives-fallback
// entry that isn't actually in the model's grounded catalog -- never a
// valid index into TransitionMutex's or SuccessorGenerator's dominance
// tables, so every id-indexed pruning path in beam_repair must check for
// and skip it rather than index with it. Same convention as deorder_lns.rs.
const NO_CATALOG_ID: usize = usize::MAX;

// How many steps beam_repair's f-ranking simulates ahead by replaying the
// remaining unplaced positions in their original relative order (see
// eval_f_multiset). Fixed, not runtime-configurable -- deorder_lns.rs found
// this worth tuning during its own development, but this solver starts from
// its settled value (24) rather than re-exposing the knob before there's a
// reason to.
const MULTISET_LOOKAHEAD: usize = 24;

// Reads an env-var-gated boolean flag that defaults to `true`. Set the var
// to "0" or "false" (case-insensitive) to opt back out for an ablation;
// unset, or any other value, keeps the default of enabled.
fn env_flag_default_true(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => !(value == "0" || value.eq_ignore_ascii_case("false")),
        Err(_) => true,
    }
}

fn compute_alternatives(
    by_params: &HashMap<Vec<Element>, Vec<Rc<TransitionWithId>>>,
    transitions: &[Transition],
) -> Vec<Vec<Rc<TransitionWithId>>> {
    transitions
        .iter()
        .map(|t| {
            by_params
                .get(&t.parameter_values)
                .cloned()
                .filter(|alts| alts.iter().any(|alt| alt.transition == *t))
                .unwrap_or_else(|| {
                    vec![Rc::new(TransitionWithId {
                        transition: t.clone(),
                        forced: false,
                        id: NO_CATALOG_ID,
                    })]
                })
        })
        .collect()
}

impl<T, B> PositionLns<T, B>
where
    T: Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    B: FnMut(T, T) -> T,
{
    /// Creates a new position LNS starting from `transitions`.
    pub fn new(
        model: Rc<Model>,
        transitions: Vec<Transition>,
        cost: Option<T>,
        initial_time: f64,
        root_cost: T,
        base_cost_evaluator: B,
        parameters: PositionLnsParameters<T>,
        f_evaluator_type: FEvaluatorType,
    ) -> PositionLns<T, B> {
        let solvable = cost.is_some() && transitions.len() >= 2;
        let time_limit = parameters.parameters.time_limit.unwrap_or(f64::INFINITY);
        let use_heuristic = model.has_dual_bounds();

        let successor_generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);

        let transition_mutex = TransitionMutex::new(
            successor_generator
                .transitions
                .iter()
                .chain(successor_generator.forced_transitions.iter())
                .map(|t| t.as_ref().clone())
                .collect(),
        );

        let mut by_params: HashMap<Vec<Element>, Vec<Rc<TransitionWithId>>> = HashMap::new();
        for t in successor_generator
            .transitions
            .iter()
            .chain(successor_generator.forced_transitions.iter())
        {
            by_params
                .entry(t.transition.parameter_values.clone())
                .or_default()
                .push(t.clone());
        }
        let alternatives = compute_alternatives(&by_params, &transitions);

        // Static classification of whether this model's transition count is structurally
        // fixed (see `classify_transition_cardinality`'s doc for the three buckets). Both
        // free_replace_enabled and insertion_slack have the same failure mode on a
        // non-Flexible model: widening a freed position's candidates to the full catalog
        // (free_replace), or spending steps inserting/omitting transitions beyond the
        // incumbent's own multiset (insertion_slack), either duplicates a transition
        // already scheduled elsewhere or drops one still required -- provably infeasible
        // whenever every reachable parameter group must appear exactly once. This
        // replaces an earlier, narrower raw-transition permutation check that compared
        // `transitions.len()` against the UNGROUPED reachable catalog count, which
        // miscounted any `FixedWithReplacement` domain (e.g. CVRP's `visit`/
        // `visit-via-depot` pair) as flexible.
        let cardinality = classify_transition_cardinality::<T>(
            &model,
            successor_generator
                .transitions
                .iter()
                .chain(successor_generator.forced_transitions.iter())
                .map(|t| &t.transition),
            &transitions,
        );
        if std::env::var("DIDP_POSITION_LNS_CARDINALITY_DIAG").is_ok() {
            eprintln!(
                "[position_lns cardinality diag] {:?} ({} transitions)",
                cardinality,
                transitions.len()
            );
        }
        // Off by default (DIDP_POSITION_LNS_FREE_REPLACE opts in) -- NOT parity with
        // ModelAwareLocalSearch's Replace: unlike that one (one candidate substituted per
        // neighbor evaluation), this multiplies by every freed position in the destroy set
        // at once, so "full catalog" candidates scale far worse here, especially at the
        // largest destroy-size arm. The cardinality check above is a safe, validated
        // auto-*disable* regardless of the env var; auto-enabling by default is not safe
        // without also bounding the candidate pool for large catalogs.
        let free_replace_enabled = if cardinality != TransitionCardinality::Flexible {
            if !parameters.parameters.quiet {
                println!(
                    "detected {:?}-shaped initial solution ({} transitions) -- disabling position_lns free-replace",
                    cardinality,
                    transitions.len()
                );
            }
            false
        } else {
            std::env::var("DIDP_POSITION_LNS_FREE_REPLACE").is_ok()
        };
        // Same reasoning as free_replace_enabled above: an insertion or a deletion
        // (insertion_slack > 0, see its doc) is provably infeasible on a fixed-cardinality
        // model, so force it off regardless of the YAML parameter / adaptive toggle rather
        // than waste beam-repair steps attempting insertions that can never survive
        // applicability checks for the whole run.
        let (insertion_slack, adaptive_insertion_slack_enabled) =
            if cardinality != TransitionCardinality::Flexible {
                if !parameters.parameters.quiet && parameters.insertion_slack > 0 {
                    println!(
                        "detected {:?}-shaped initial solution ({} transitions) -- disabling position_lns insertion_slack",
                        cardinality,
                        transitions.len()
                    );
                }
                (0, false)
            } else {
                (
                    parameters.insertion_slack,
                    std::env::var("DIDP_POSITION_LNS_ADAPTIVE_INSERTION_SLACK").is_ok(),
                )
            };

        let transitions_len = transitions.len();
        if std::env::var("DIDP_POSITION_LNS_SIZE_DIAG").is_ok() {
            eprintln!(
                "[position_lns size diag] n={} grounded_transitions={} forced_transitions={}",
                transitions_len,
                successor_generator.transitions.len(),
                successor_generator.forced_transitions.len()
            );
        }

        // Destroy-set-size bandit arm ladder: powers of two up to
        // transitions_len, always ending in transitions_len itself -- same
        // construction as Lnbs's depth_arms (lnbs.rs), just over
        // transitions_len instead of max_depth -- plus an explicit size-1
        // arm prepended. Lnbs's own ladder starts at 2 (its smallest window
        // is a 2-position swap-equivalent under restricted shuffle), but
        // size 1 is meaningful here in a way it isn't for Lnbs: under full
        // reinsertion (see select_neighborhood, which forces it whenever
        // size == 1) it relaxes one position's constraints against
        // everything else, so beam_repair's per-position `alternatives[p]`
        // candidate generation (see its loop) can both relocate it anywhere
        // in the order AND substitute it for any parameter-matching
        // alternative -- a combined relocate-or-replace move, strictly
        // generalizing both of ModelAwareLocalSearch's separate Relocate and
        // Replace neighborhoods. Under restricted shuffle, size 1 can never
        // relax any pair (a pair needs both endpoints in the freed set) and
        // is a guaranteed no-op -- exactly why full reinsertion is forced
        // for this arm rather than left to the usual coin flip.
        //
        // transitions_len == 0 when the upstream CABS initial solve ran out of
        // time without finding any feasible solution (e.g. on m-pdtsp class1
        // instances with sparse missing-arc structure) -- leading_zeros() of 0
        // is usize::BITS, which would underflow `log` below and (in a release
        // build, where overflow checks are off) silently wrap to ~u32::MAX,
        // making `(1..=log).collect()` try to build a multi-billion-element
        // Vec. Guard it explicitly instead.
        let mut size_arms: Vec<usize> = if transitions_len == 0 {
            Vec::new()
        } else {
            let log = (usize::BITS - 1) - transitions_len.leading_zeros();
            (1..=log).map(|i| 2usize.pow(i)).collect()
        };
        if let Some(&last) = size_arms.last() {
            if last < transitions_len {
                size_arms.push(transitions_len);
            }
        } else {
            size_arms.push(transitions_len);
        }
        // DIDP_POSITION_LNS_MIN_SIZE: overrides the ladder's floor arm (default
        // 1) -- e.g. set to 2 to test dropping the size-1 arm (see its
        // construction doc above for why 1 is meaningful, not just the floor)
        // without a rebuild. Unset behavior is byte-for-byte the original: insert
        // exactly 1.
        let min_size: usize = std::env::var("DIDP_POSITION_LNS_MIN_SIZE")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&m| m >= 1)
            .unwrap_or(1);
        size_arms.retain(|&s| s >= min_size);
        if size_arms.first() != Some(&min_size) {
            size_arms.insert(0, min_size);
        }
        let size_arm_count = size_arms.len();

        let mut search = PositionLns {
            model,
            base_cost_evaluator,
            root_cost,
            by_params,
            alternatives,
            successor_generator,
            transition_mutex,
            current_costs: Vec::new(),
            w: Vec::new(),
            w_range: 1.0,
            current_cost: cost.unwrap_or(root_cost),
            current: transitions.clone(),
            best: Solution {
                cost,
                transitions,
                is_infeasible: cost.is_none(),
                time: initial_time,
                ..Default::default()
            },
            solvable,
            sa_enabled: std::env::var("DIDP_POSITION_LNS_SA").is_ok(),
            sa_final_temp_ratio: std::env::var("DIDP_SA_FINAL_RATIO")
                .ok()
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.01),
            sa_t0: 0.0,
            sa_calibrated: false,
            sa_calibration_deltas: Vec::new(),
            sa_stall_threshold: std::env::var("DIDP_SA_STALL_THRESHOLD")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(50),
            stall_count: 0,
            stall_limit: parameters.stall_limit,
            stall_time_limit: parameters.stall_time_limit,
            last_improvement_time: 0.0,
            time_limit,
            neighborhood_size: parameters.neighborhood_size.max(2),
            position_weight: parameters.position_weight,
            cost_weight: parameters.cost_weight,
            relatedness_determinism: parameters.relatedness_determinism.max(1.0),
            contiguity_fill_enabled: std::env::var("DIDP_POSITION_LNS_CONTIGUITY_FILL").is_ok(),
            min_freed_for_contiguity: std::env::var("DIDP_POSITION_LNS_MIN_FREED_FOR_CONTIGUITY")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(4),
            max_gap_ratio: std::env::var("DIDP_POSITION_LNS_MAX_GAP_RATIO")
                .ok()
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.25),
            shuffle_on_fill_enabled: std::env::var("DIDP_POSITION_LNS_SHUFFLE_ON_FILL").is_ok(),
            full_reinsert_probability: parameters.full_reinsert_probability,
            enable_stagnation_seed: parameters.enable_stagnation_seed,
            stagnation: vec![0u32; transitions_len],
            beam_width: parameters.beam_width.max(1),
            beam_width_growth_enabled: env_flag_default_true("DIDP_POSITION_LNS_BEAM_GROWTH"),
            beam_width_max: std::env::var("DIDP_POSITION_LNS_BEAM_MAX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(parameters.beam_width.max(1) * 2)
                .max(parameters.beam_width.max(1)),
            // DIDP_POSITION_LNS_BEAM_START_SMALL: seeds arm_beam_size at 1
            // instead of beam_width, matching Lnbs's neighborhood_beam_size,
            // which seeds each window small and doubles with no forced cap.
            // Off by default, untested.
            arm_beam_size: vec![
                if std::env::var("DIDP_POSITION_LNS_BEAM_START_SMALL").is_ok() {
                    1
                } else {
                    parameters.beam_width.max(1)
                };
                size_arm_count
            ],
            max_branching: parameters.max_branching.max(1),
            insertion_slack,
            adaptive_insertion_slack_enabled,
            free_replace_enabled,
            rng: Pcg64Mcg::seed_from_u64(parameters.seed),
            time_keeper: TimeKeeper::with_time_limit(time_limit),
            quiet: parameters.parameters.quiet,
            first_call: true,
            f_evaluator_type,
            use_heuristic,
            accept_diag_enabled: std::env::var("DIDP_POSITION_LNS_DIAG").is_ok(),
            iteration_count: 0,
            accept_count: 0,
            effort_diag_enabled: std::env::var("DIDP_POSITION_LNS_EFFORT_DIAG").is_ok(),
            effort_beam_steps: 0,
            effort_beam_seeds_generated: 0,
            effort_beam_truncated_steps: 0,
            effort_dominance_pruned: 0,
            effort_dominance_checked: 0,
            effort_mutex_skipped: 0,
            effort_candidates_attempted: 0,
            effort_beam_layer_total_states: 0,
            effort_beam_layer_distinct_states: 0,
            effort_fallback_cap_count: 0,
            effort_fallback_infeasible_count: 0,
            effort_fallback_infeasible_depth_sum: 0,
            effort_full_completion_count: 0,
            dedup_enabled: env_flag_default_true("DIDP_POSITION_LNS_DEDUP"),
            contiguity_stats_enabled: std::env::var("DIDP_POSITION_LNS_CONTIGUITY_STATS").is_ok(),
            contiguity_stat_freed_ge2: 0,
            contiguity_stat_contiguous: 0,
            contiguity_gate_steps_total: 0,
            contiguity_gate_steps_narrowed: 0,
            multiset_lookahead_cap: std::env::var("DIDP_POSITION_LNS_MULTISET_LOOKAHEAD")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .filter(|&v| v >= 1)
                .unwrap_or(MULTISET_LOOKAHEAD),
            size_trace_enabled: std::env::var("DIDP_POSITION_LNS_SIZE_TRACE").is_ok(),
            size_bandit_enabled: env_flag_default_true("DIDP_POSITION_LNS_SIZE_BANDIT"),
            size_arms,
            size_reward_mean: vec![0.0; size_arm_count],
            size_time_mean: vec![0.0; size_arm_count],
            size_trials: vec![0.0; size_arm_count],
            size_total_trials: 0.0,
            size_lambda: None,
            seed_bandit_enabled: std::env::var("DIDP_POSITION_LNS_SEED_BANDIT").is_ok(),
            seed_arm_stats: HashMap::new(),
            seed_arm_total_trials: 0.0,
            seed_arm_lambda: None,
            turn_deadline: None,
        };

        if search.solvable {
            search.refresh_trace();
        }
        search.time_keeper.stop();

        search
    }

    // Refreshes current_costs/w/w_range to match `current`. Called at
    // construction and after every accepted move, same cadence as
    // alternatives -- see its doc for why "only on accept" (not
    // per-iteration) is load-bearing for throughput, not just an
    // optimization.
    fn refresh_trace(&mut self) {
        self.current_costs = get_trace(&self.model.target, self.root_cost, &self.current, &self.model)
            .map(|(_, cost)| cost)
            .collect();

        let mut w = vec![0.0f64; self.current.len()];
        let mut prev = self.root_cost.to_continuous();
        for (p, slot) in w.iter_mut().enumerate() {
            let cur = self.current_costs[p].to_continuous();
            *slot = (cur - prev).max(0.0);
            prev = cur;
        }
        let (min_w, max_w) = w
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        self.w_range = (max_w - min_w).max(1e-9);
        self.w = w;
    }

    // Relatedness *distance* between positions a and c -- smaller means
    // "more related", i.e. more likely to be grown into the same freed set.
    // See the module doc for the two terms and why a third, TransitionMutex-
    // based structural term was tried and removed (it was measured to
    // always contribute exactly 0 -- see the module doc).
    fn relatedness_distance(&self, a: usize, c: usize) -> f64 {
        let n = self.current.len().max(1);
        let position_term = (a as f64 - c as f64).abs() / n as f64;
        let cost_term = (self.w[a] - self.w[c]).abs() / self.w_range;

        self.position_weight * position_term + self.cost_weight * cost_term
    }

    // Picks the set S of positions to free this iteration by relatedness
    // seed-and-grow (see the module doc), and whether this iteration
    // repairs with the full-reinsertion neighborhood (vs. restricted
    // shuffle) -- an independent coin flip, unaffected by how S itself was
    // chosen. The destroy-set size itself is either the fixed
    // `neighborhood_size` (default) or picked by `select_size`'s bandit when
    // `size_bandit_enabled`. The returned `Option<usize>` is the chosen size
    // arm index, `None` when the bandit is off, threaded back to
    // `search_next` so it can report the outcome via `update_size_bandit`.
    // The `usize` `growth_arm` is always a concrete index into
    // `arm_beam_size` (see its doc) -- the real size arm when
    // `size_bandit_enabled`, or the fixed sentinel slot 0 otherwise -- so
    // per-arm beam-width growth has a stable key to revisit even when the
    // size bandit itself is off. The final `usize`, `size`, is this
    // iteration's actual destroy-set size (post `.min(n)`) -- exposed so
    // `search_next` can feed it to `adaptive_insertion_slack` (see its doc)
    // and keep insertion_slack in lockstep with whatever size the bandit (or
    // the fixed default) chose, without duplicating that choice. The final
    // `usize`, `seed`, is the seed position select_freed_positions grew
    // from -- exposed so `search_next` can report the outcome back to
    // `update_seed_bandit` when `seed_bandit_enabled` (see its doc).
    fn select_neighborhood(&mut self, n: usize) -> (HashSet<usize>, bool, Option<usize>, usize, usize, usize) {
        let (size_arm, growth_arm, size) = if self.size_bandit_enabled {
            let (arm, size) = self.select_size();
            (Some(arm), arm, size.min(n))
        } else {
            (None, 0, self.neighborhood_size.min(n))
        };
        // Priority: seed_bandit_enabled > enable_stagnation_seed > uniform
        // random -- see seed_bandit_enabled's doc for why these two aren't
        // meant to be combined (harmless if both are set, just wasted
        // stagnation bookkeeping that nothing reads).
        let seed = if self.seed_bandit_enabled {
            self.select_seed_bandit(n, growth_arm)
        } else if self.enable_stagnation_seed {
            self.select_stagnation_seed(n)
        } else {
            self.rng.random_range(0..n)
        };
        let freed = self.select_freed_positions(n, size, seed);
        // Restricted shuffle can never relax any pair when |freed| == 1 (a
        // pair needs both endpoints in the freed set) -- a guaranteed no-op
        // -- so force full reinsertion there instead of leaving it to the
        // usual coin flip. See size_arms's construction doc for why size 1
        // is a real, meaningful arm precisely because it always gets paired
        // with full reinsertion.
        let is_contiguous = size > 1 && Self::is_contiguous(&freed);
        if self.contiguity_stats_enabled && size > 1 {
            self.contiguity_stat_freed_ge2 += 1;
            if is_contiguous {
                self.contiguity_stat_contiguous += 1;
            }
        }

        let full_reinsert = if size <= 1 {
            true
        } else if self.shuffle_on_fill_enabled && is_contiguous {
            false
        } else {
            self.rng.random::<f64>() < self.full_reinsert_probability
        };

        (freed, full_reinsert, size_arm, growth_arm, size, seed)
    }

    // Sparse UCB bandit over (seed_position, growth_arm) pairs -- same
    // budgeted-UCB score as select_size, just keyed by a pair instead of a
    // single arm index, and stored in a HashMap (not a dense Vec) since
    // most of the n * size_arms possible pairs are never visited: unlike
    // Lnbs's (start, depth) grid, which is small enough to bootstrap and
    // revisit densely, position_lns's freed sets are normally a fresh,
    // essentially-never-repeated draw (see arm_beam_size's doc) -- pairing
    // the *seed* alone with growth_arm is the compromise that keeps this
    // revisitable without tracking the literal freed set (which really
    // would be a distinct-almost-every-time key, useless for a bandit).
    // `growth_arm` (not `size_arm`) is deliberately reused as the second
    // component: it's always a concrete index (falling back to sentinel 0
    // when size_bandit_enabled is off), so this bandit works standalone,
    // exactly mirroring why arm_beam_size itself keys on growth_arm.
    //
    // Untried pairs score f64::INFINITY like select_size's bootstrap, and
    // ties resolve to the first-seen seed in 0..n order (via strict `>`,
    // not `>=`) -- same deterministic left-to-right bootstrap order
    // select_size gets from its `.rev().max_by()` trick, just written
    // directly since there's no iterator adapter doing the reversal here.
    // This means a full bootstrap costs at least n iterations before any
    // pair can be exploited over another -- expected to be a real up-front
    // cost for large n, not hidden.
    fn select_seed_bandit(&mut self, n: usize, growth_arm: usize) -> usize {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_seed = 0;

        for s in 0..n {
            let score = match self.seed_arm_stats.get(&(s, growth_arm)) {
                None => f64::INFINITY,
                Some(stats) if stats.trials < 0.5 => f64::INFINITY,
                Some(stats) => {
                    let r = stats.reward_mean;
                    let c = stats.time_mean;
                    let lambda = self.seed_arm_lambda.unwrap();
                    let epsilon = (2.0 * self.seed_arm_total_trials.ln() / stats.trials).sqrt();
                    let numerator = if r + epsilon <= 1.0 { r + epsilon } else { 1.0 };
                    let denominator = if c - epsilon >= lambda { c - epsilon } else { lambda };
                    r / c + epsilon / c + epsilon / c * numerator / denominator
                }
            };

            if score > best_score {
                best_score = score;
                best_seed = s;
            }
        }

        best_seed
    }

    // Ported from update_size_bandit's math onto the sparse (seed,
    // growth_arm) map -- see select_seed_bandit's doc.
    fn update_seed_bandit(&mut self, seed: usize, growth_arm: usize, reward: f64, time: f64) {
        if self.seed_arm_lambda.is_none() {
            self.seed_arm_lambda = Some(time / 10.0);
        }

        self.seed_arm_total_trials += 1.0;
        let stats = self
            .seed_arm_stats
            .entry((seed, growth_arm))
            .or_insert(SeedArmStats { trials: 0.0, reward_mean: 0.0, time_mean: 0.0 });
        stats.trials += 1.0;
        stats.reward_mean = (stats.reward_mean * (stats.trials - 1.0) + reward) / stats.trials;
        stats.time_mean = (stats.time_mean * (stats.trials - 1.0) + time) / stats.trials;
    }

    // Ported from Lnbs::select_ucb (lnbs.rs) onto the destroy-set-size arm
    // ladder instead of window depth -- same budgeted-UCB score (reward per
    // unit time, with a confidence term shrinking as an arm accumulates
    // trials), same "never-tried arm gets picked first" bootstrap via
    // f64::INFINITY. See size_bandit_enabled's doc for why there is no
    // exhaustion filtering here (Lnbs's `depth_exhausted` has no analogue).
    fn select_size(&mut self) -> (usize, usize) {
        self.size_arms
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                if self.size_trials[i] < 0.5 {
                    return (f64::INFINITY, (i, size));
                }

                let r = self.size_reward_mean[i];
                let c = self.size_time_mean[i];
                let lambda = self.size_lambda.unwrap();
                let epsilon = (2.0 * self.size_total_trials.ln() / self.size_trials[i]).sqrt();
                let numerator = if r + epsilon <= 1.0 { r + epsilon } else { 1.0 };
                let denominator = if c - epsilon >= lambda { c - epsilon } else { lambda };

                let score = r / c + epsilon / c + epsilon / c * numerator / denominator;

                (score, (i, size))
            })
            .rev()
            .max_by(|(a, _), (b, _)| a.total_cmp(b))
            .map(|(_, arm)| arm)
            .unwrap()
    }

    // Ported from Lnbs::update_bandit (lnbs.rs) verbatim.
    fn update_size_bandit(&mut self, arm: usize, reward: f64, time: f64) {
        if self.size_lambda.is_none() {
            self.size_lambda = Some(time / 10.0);
        }

        self.size_total_trials += 1.0;
        self.size_trials[arm] += 1.0;
        self.size_reward_mean[arm] = (self.size_reward_mean[arm] * (self.size_trials[arm] - 1.0)
            + reward)
            / self.size_trials[arm];
        self.size_time_mean[arm] = (self.size_time_mean[arm] * (self.size_trials[arm] - 1.0)
            + time)
            / self.size_trials[arm];
    }

    // Shaw-removal-style seed-and-grow: start from a random position, then
    // repeatedly add whichever remaining position is closest (by
    // relatedness_distance) to *any* position already in the freed set --
    // not just the seed, so the set can follow a chain of relatedness
    // rather than only ever growing around one anchor. Each pick is
    // randomized among the ranked candidates via relatedness_determinism
    // (see its doc) rather than always taking the single closest, so the
    // sampler explores different related clusters across iterations
    // instead of being pinned to one greedy chain per seed.
    //
    // `seed` is chosen by the caller (select_neighborhood) rather than
    // here, now that there are three mutually exclusive ways to pick it
    // (seed_bandit_enabled, enable_stagnation_seed, uniform random) -- see
    // select_neighborhood's doc for the priority order.
    fn select_freed_positions(&mut self, n: usize, size: usize, seed: usize) -> HashSet<usize> {
        let mut freed = HashSet::new();
        freed.insert(seed);
        let mut remaining: Vec<usize> = (0..n).filter(|&i| i != seed).collect();
        let mut gap_scratch: Vec<usize> = Vec::new();

        while freed.len() < size && !remaining.is_empty() {
            let base: &[usize] = &remaining;

            let pool: &[usize] = if self.contiguity_fill_enabled {
                let (span_lo, span_hi) = freed
                    .iter()
                    .fold((usize::MAX, usize::MIN), |(lo, hi), &f| (lo.min(f), hi.max(f)));
                let span = span_hi - span_lo + 1;
                let gaps = span - freed.len();
                // Gate on the PROSPECTIVE size (freed.len() + 1, what this step's own pick would
                // bring freed to), not freed's current size: growth stops as soon as freed.len()
                // == size, so a current-size gate could never fire on the step that would
                // actually reach a min_freed_for_contiguity-sized freed set.
                let gap_ratio = gaps as f64 / span as f64;
                if self.contiguity_stats_enabled {
                    self.contiguity_gate_steps_total += 1;
                }
                if freed.len() + 1 >= self.min_freed_for_contiguity && gap_ratio <= self.max_gap_ratio {
                    gap_scratch.clear();
                    gap_scratch.extend(
                        base.iter()
                            .copied()
                            .filter(|&c| c > span_lo && c < span_hi),
                    );
                    if self.contiguity_stats_enabled && !gap_scratch.is_empty() {
                        self.contiguity_gate_steps_narrowed += 1;
                    }
                    if gap_scratch.is_empty() {
                        base
                    } else {
                        &gap_scratch
                    }
                } else {
                    base
                }
            } else {
                base
            };

            let mut candidates: Vec<(usize, f64)> = pool
                .iter()
                .map(|&c| {
                    let d = freed
                        .iter()
                        .map(|&f| self.relatedness_distance(f, c))
                        .fold(f64::INFINITY, f64::min);
                    (c, d)
                })
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let r = self.rng.random::<f64>().powf(self.relatedness_determinism);
            let idx = ((r * candidates.len() as f64) as usize).min(candidates.len() - 1);
            let chosen = candidates[idx].0;

            freed.insert(chosen);
            remaining.retain(|&x| x != chosen);
        }

        freed
    }

    // True iff `freed` is a single solid run of positions -- span (max -
    // min + 1) equals freed.len(), i.e. zero gaps. Strict on purpose (unlike
    // contiguity_fill's gap_ratio threshold, which is a "close enough to
    // nudge" bar): this is used to decide the repair *method*, where "almost
    // contiguous" doesn't give restricted shuffle the same guarantee that
    // every freed position is adjacent to another freed position.
    fn is_contiguous(freed: &HashSet<usize>) -> bool {
        let (lo, hi) = freed
            .iter()
            .fold((usize::MAX, usize::MIN), |(lo, hi), &f| (lo.min(f), hi.max(f)));
        hi - lo + 1 == freed.len()
    }

    // Roulette-wheel pick over positions [0, n), weighted by `1 +
    // stagnation[p]` -- see `stagnation`'s doc. The +1 keeps every position
    // reachable (a never-stuck position still has a nonzero chance), so
    // this degrades toward "occasionally revisits everything" rather than
    // starving positions that happen to start at 0 and never get sampled.
    fn select_stagnation_seed(&mut self, n: usize) -> usize {
        let total: u64 = self.stagnation[..n].iter().map(|&s| s as u64 + 1).sum();
        let mut r = self.rng.random_range(0..total);

        for (p, &s) in self.stagnation[..n].iter().enumerate() {
            let weight = s as u64 + 1;
            if r < weight {
                return p;
            }
            r -= weight;
        }

        n - 1
    }

    // Materializes the kept-edge graph over all n positions: successors[a]
    // lists every c with the precedence (a, c) still forced. A pair (a, c)
    // is relaxed -- and so absent from this graph -- when:
    //   - full_reinsert: a or c is in `freed`
    //   - restricted shuffle: a and c are both in `freed`
    // Everything else stays forced, exactly the incumbent's own order.
    // O(n^2), matching deorder_lns.rs's build_successors -- acceptable
    // given the repair step budgets iterations in the hundreds, not
    // thousands.
    fn build_successors(
        &self,
        n: usize,
        freed: &HashSet<usize>,
        full_reinsert: bool,
    ) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];

        for a in 0..n {
            for c in (a + 1)..n {
                let relaxed = if full_reinsert {
                    freed.contains(&a) || freed.contains(&c)
                } else {
                    freed.contains(&a) && freed.contains(&c)
                };
                if !relaxed {
                    successors[a].push(c);
                    in_degree[c] += 1;
                }
            }
        }

        (successors, in_degree)
    }

    // f = g + h ranking value for a beam entry, via the model's dual bound
    // evaluated on `state` as the heuristic estimate of remaining cost.
    // Falls back to `cost` alone when the model has no dual bounds.
    fn eval_f(&self, cost: T, state: &State) -> T {
        if !self.use_heuristic {
            return cost;
        }

        let mut function_cache = StateFunctionCache::new(&self.model.state_functions);
        match self.model.eval_dual_bound::<State, T>(state, &mut function_cache) {
            Some(h) => self.f_evaluator_type.eval(cost, h),
            None => cost,
        }
    }

    // A tighter alternative to eval_f, available only because of the
    // fixed-multiset restriction: simulates completing with the unplaced
    // original positions in increasing order, which is always a valid
    // completion under the kept edges here (both neighborhoods only ever
    // relax edges relative to the incumbent's own total order, never add
    // ones it violates -- same argument as deorder_lns.rs's is_incumbent
    // doc). Falls back to eval_f on an infeasible simulated step, using
    // `sim_cost`/`sim_state` (this call's own progress up to the failing
    // step), not the function's original `cost`/`state` parameters, so the
    // real cost already incurred during simulation isn't discarded.
    //
    // `f` is not proven to be a sound lower bound over all completions:
    // even the non-fallback path only ever evaluates ONE fixed completion
    // order (replay the unplaced positions in their original relative
    // order), and nothing proves that order's cost is <= every other valid
    // completion's cost -- it's a plausible heuristic estimate, exact only
    // when the model gives it no cheaper alternative. Capped at
    // MULTISET_LOOKAHEAD simulated positions -- see its doc.
    fn eval_f_multiset(
        &mut self,
        cost: T,
        state: &State,
        placed_mask: &[bool],
        function_cache: &mut ParentAndChildStateFunctionCache,
    ) -> T {
        let mut sim_state = state.clone();
        let mut sim_cost = cost;
        let mut simulated = 0usize;
        let lookahead_cap = self.multiset_lookahead_cap;

        for (p, &placed) in placed_mask.iter().enumerate() {
            if placed {
                continue;
            }

            if simulated >= lookahead_cap {
                if self.effort_diag_enabled {
                    self.effort_fallback_cap_count += 1;
                }
                return self.eval_f(sim_cost, &sim_state);
            }
            simulated += 1;

            let transition = &self.current[p];
            function_cache.parent.clear();
            function_cache.child.clear();

            if !transition.is_applicable(
                &sim_state,
                &mut function_cache.child,
                &self.model.state_functions,
                &self.model.table_registry,
            ) {
                if self.effort_diag_enabled {
                    self.effort_fallback_infeasible_count += 1;
                    self.effort_fallback_infeasible_depth_sum += simulated as u64 - 1;
                }
                return self.eval_f(sim_cost, &sim_state);
            }

            let new_state = transition.apply(
                &sim_state,
                &mut function_cache.child,
                &self.model.state_functions,
                &self.model.table_registry,
            );

            if !self
                .model
                .check_constraints(&new_state, &mut function_cache.child)
            {
                if self.effort_diag_enabled {
                    self.effort_fallback_infeasible_count += 1;
                    self.effort_fallback_infeasible_depth_sum += simulated as u64 - 1;
                }
                return self.eval_f(sim_cost, &sim_state);
            }

            sim_cost = transition.eval_cost(
                sim_cost,
                &sim_state,
                &mut function_cache.child,
                &self.model.state_functions,
                &self.model.table_registry,
            );
            sim_state = new_state;
        }

        if self.effort_diag_enabled {
            self.effort_full_completion_count += 1;
        }

        sim_cost
    }

    // Bounded beam search for the best valid linear extension of the kept
    // edges, under real DyPDL applicability/state-constraint checks at
    // every step. Returns the best complete, feasible sequence found and
    // its cost, or `None` if every beam entry died out before completion.
    //
    // `freed` gates insertion candidates (see insertion_slack's doc): only
    // offered to an entry that still has an unplaced position in `freed`.
    // An entry that has placed all n positions is moved out of the active
    // `beam` into `done` immediately (see the loop below) rather than kept
    // around for more steps -- insertion_slack only ever extends the
    // destroy/repair region itself, never appends past it.
    //
    // `beam_width` is the truncation width for this call, computed by the
    // caller via `beam_width_for_arm` -- passed in explicitly rather than
    // read from `self.beam_width` so per-arm growth (see `arm_beam_size`'s
    // doc) can vary it per call without this function needing to know
    // which arm is active.
    fn beam_repair(
        &mut self,
        successors: &[Vec<usize>],
        init_in_degree: &[usize],
        n: usize,
        freed: &HashSet<usize>,
        beam_width: usize,
        insertion_slack: usize,
    ) -> Option<(Vec<Transition>, Vec<usize>, T)> {
        let root_state = self.model.target.clone();
        let root_placed_mask = vec![false; n];
        let mut function_cache = ParentAndChildStateFunctionCache::new(&self.model.state_functions);
        let mut beam = vec![BeamEntry {
            placed: Vec::with_capacity(n),
            placed_positions: Vec::with_capacity(n),
            placed_mask: root_placed_mask,
            in_degree: init_in_degree.to_vec(),
            forbidden: HashSet::new(),
            state: root_state,
            cost: self.root_cost,
            inserted: 0,
            is_incumbent: true,
        }];
        // Entries that have placed all n positions, plus any entry
        // opportunistically captured early below -- set aside from `beam`
        // so the loop only ever generates candidates from still-active
        // entries, and so a final base-case check doesn't have to be
        // redone for one already known to pass it. All are candidate
        // completions for the final argmin/SA selection at the bottom of
        // this function, regardless of which path put them here.
        let mut done: Vec<BeamEntry<T>> = Vec::new();
        let max_steps = n + insertion_slack;

        for _ in 0..max_steps {
            if beam.is_empty() {
                break;
            }

            // Opportunistic early-completion check -- what makes deletion
            // (not just insertion) possible: an active entry, with a freed
            // position still unplaced, may already sit at a valid
            // base-case state (e.g. the freed position turned out to be
            // unnecessary). Snapshot it into `done` as a candidate
            // completion *without* removing it from `beam`, so the search
            // still also tries placing its remaining positions -- whichever
            // turns out cheaper wins at final selection. Gated on
            // insertion_slack, like the insertion candidates below: with it
            // at 0, every entry here is still forced through exactly the
            // incumbent's own n positions, unchanged from before either
            // feature existed. An entry that only reaches base *after*
            // placing every position is deliberately not re-checked here --
            // it gets exactly one base-case check, at the very end, via
            // `done`'s own final loop.
            if insertion_slack > 0 {
                for entry in &beam {
                    function_cache.parent.clear();
                    function_cache.child.clear();

                    if self
                        .model
                        .eval_base_cost::<T, _>(&entry.state, &mut function_cache.child)
                        .is_some()
                    {
                        done.push(entry.clone());
                    }
                }
            }

            struct CandidateSeed<T> {
                parent_idx: usize,
                // None for an inserted transition: doesn't consume a
                // position or reduce any in-degree -- see insertion_slack's
                // doc.
                position: Option<usize>,
                transition: Rc<TransitionWithId>,
                state: State,
                cost: T,
                is_incumbent: bool,
            }

            struct EntryCandidate<T> {
                position: usize,
                transition: Rc<TransitionWithId>,
                state: State,
                cost: T,
            }

            let mut seeds: Vec<CandidateSeed<T>> = Vec::new();

            for (parent_idx, entry) in beam.iter().enumerate() {
                let mut ready: Vec<usize> = (0..n)
                    .filter(|&i| entry.in_degree[i] == 0 && !entry.placed_mask[i])
                    .collect();

                // Forced-transition short-circuit -- mirrors
                // SuccessorGenerator::ApplicableTransitions exactly, same as
                // deorder_lns.rs's beam_repair.
                let mut forced_applicable: Option<&Rc<TransitionWithId>> = None;
                for t in &self.successor_generator.forced_transitions {
                    function_cache.parent.clear();
                    function_cache.child.clear();

                    if t.is_applicable(
                        &entry.state,
                        &mut function_cache.child,
                        &self.model.state_functions,
                        &self.model.table_registry,
                    ) {
                        forced_applicable = Some(t);
                        break;
                    }
                }

                if let Some(forced) = forced_applicable {
                    let position = ready.iter().copied().find(|&p| {
                        self.alternatives[p]
                            .iter()
                            .any(|alt| alt.forced == forced.forced && alt.id == forced.id)
                    });

                    if let Some(position) = position {
                        let new_state = forced.apply(
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        );

                        if self
                            .model
                            .check_constraints(&new_state, &mut function_cache.child)
                        {
                            let new_cost = forced.eval_cost(
                                entry.cost,
                                &entry.state,
                                &mut function_cache.child,
                                &self.model.state_functions,
                                &self.model.table_registry,
                            );
                            let is_incumbent = entry.is_incumbent
                                && position == entry.placed.len()
                                && forced.transition.name == self.current[position].name
                                && forced.transition.parameter_values
                                    == self.current[position].parameter_values;

                            seeds.push(CandidateSeed {
                                parent_idx,
                                position: Some(position),
                                transition: forced.clone(),
                                state: new_state,
                                cost: new_cost,
                                is_incumbent,
                            });
                        }
                    }

                    continue;
                }

                if ready.len() > self.max_branching {
                    ready.partial_shuffle(&mut self.rng, self.max_branching);
                    ready.truncate(self.max_branching);

                    if entry.is_incumbent {
                        let next = entry.placed.len();
                        if !ready.contains(&next) {
                            ready.push(next);
                        }
                    }
                }

                let mut entry_candidates: Vec<EntryCandidate<T>> = Vec::new();

                for &p in &ready {
                    // free_replace_enabled (see its doc): a freed position's
                    // candidate pool is widened from same-parameter
                    // alternatives (alternatives[p]) to the full grounded
                    // catalog, so it can be filled by a transition wholly
                    // unrelated to whatever current[p] originally held --
                    // not just relocated/substituted, genuinely replaced.
                    // `extra` stays empty (and this is a plain iterator
                    // chain over alternatives[p] alone) whenever the toggle
                    // is off or p isn't freed, so behavior is byte-for-byte
                    // unchanged by default.
                    let extra: &[Rc<TransitionWithId>] =
                        if self.free_replace_enabled && freed.contains(&p) {
                            &self.successor_generator.transitions
                        } else {
                            &[]
                        };

                    for transition in self.alternatives[p].iter().chain(extra.iter()) {
                        if transition.id != NO_CATALOG_ID
                            && entry.forbidden.contains(&(transition.forced, transition.id))
                        {
                            if self.effort_diag_enabled {
                                self.effort_mutex_skipped += 1;
                            }
                            continue;
                        }
                        if self.effort_diag_enabled {
                            self.effort_candidates_attempted += 1;
                        }

                        function_cache.parent.clear();
                        function_cache.child.clear();

                        if !transition.is_applicable(
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        ) {
                            continue;
                        }

                        let new_state = transition.apply(
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        );

                        if !self
                            .model
                            .check_constraints(&new_state, &mut function_cache.child)
                        {
                            continue;
                        }

                        let new_cost = transition.eval_cost(
                            entry.cost,
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        );

                        entry_candidates.push(EntryCandidate {
                            position: p,
                            transition: transition.clone(),
                            state: new_state,
                            cost: new_cost,
                        });
                    }
                }

                // TransitionDominance pruning, scoped to this entry's own
                // applicable candidates -- see deorder_lns.rs's identical
                // block doc.
                if entry_candidates.len() > 1 {
                    let real_count = entry_candidates
                        .iter()
                        .filter(|c| c.transition.id != NO_CATALOG_ID)
                        .count();

                    if real_count > 1 {
                        let mut dominance_pool: Vec<Rc<TransitionWithId>> = entry_candidates
                            .iter()
                            .filter(|c| c.transition.id != NO_CATALOG_ID)
                            .map(|c| c.transition.clone())
                            .collect();
                        self.successor_generator.filter_dominated(
                            &entry.state,
                            &mut function_cache.child,
                            &mut dominance_pool,
                        );

                        if self.effort_diag_enabled {
                            self.effort_dominance_checked += real_count as u64;
                            self.effort_dominance_pruned +=
                                (real_count - dominance_pool.len()) as u64;
                        }

                        if dominance_pool.len() != real_count {
                            let survivors: HashSet<usize> =
                                dominance_pool.iter().map(|t| t.id).collect();
                            entry_candidates.retain(|c| {
                                c.transition.id == NO_CATALOG_ID
                                    || survivors.contains(&c.transition.id)
                            });
                        }
                    }
                }

                for c in entry_candidates {
                    let is_incumbent = entry.is_incumbent
                        && c.position == entry.placed.len()
                        && c.transition.transition.name == self.current[c.position].name
                        && c.transition.transition.parameter_values
                            == self.current[c.position].parameter_values;

                    seeds.push(CandidateSeed {
                        parent_idx,
                        position: Some(c.position),
                        transition: c.transition,
                        state: c.state,
                        cost: c.cost,
                        is_incumbent,
                    });
                }

                // Insertion candidates (see insertion_slack's doc): any
                // real, currently-applicable model transition, not just
                // this entry's ready positions' alternatives. Only offered
                // while budget remains and at least one freed position is
                // still unplaced -- an entry with nothing left to free
                // shouldn't keep spending steps inserting. Skipped
                // entirely for forced_applicable entries above (they
                // `continue` before reaching here), matching how a forced
                // transition already takes priority over every other
                // candidate kind in this file.
                if insertion_slack > 0
                    && entry.inserted < insertion_slack
                    && freed.iter().any(|&p| !entry.placed_mask[p])
                {
                    for transition in &self.successor_generator.transitions {
                        if transition.id != NO_CATALOG_ID
                            && entry.forbidden.contains(&(transition.forced, transition.id))
                        {
                            if self.effort_diag_enabled {
                                self.effort_mutex_skipped += 1;
                            }
                            continue;
                        }
                        if self.effort_diag_enabled {
                            self.effort_candidates_attempted += 1;
                        }

                        function_cache.parent.clear();
                        function_cache.child.clear();

                        if !transition.is_applicable(
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        ) {
                            continue;
                        }

                        let new_state = transition.apply(
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        );

                        if !self
                            .model
                            .check_constraints(&new_state, &mut function_cache.child)
                        {
                            continue;
                        }

                        let new_cost = transition.eval_cost(
                            entry.cost,
                            &entry.state,
                            &mut function_cache.child,
                            &self.model.state_functions,
                            &self.model.table_registry,
                        );

                        seeds.push(CandidateSeed {
                            parent_idx,
                            position: None,
                            transition: transition.clone(),
                            state: new_state,
                            cost: new_cost,
                            is_incumbent: false,
                        });
                    }
                }
            }

            if seeds.is_empty() {
                beam.clear();
                break;
            }

            if self.effort_diag_enabled {
                self.effort_beam_steps += 1;
                self.effort_beam_seeds_generated += seeds.len() as u64;
                if seeds.len() > beam_width {
                    self.effort_beam_truncated_steps += 1;
                }
            }

            let mut scored: Vec<(CandidateSeed<T>, T)> = Vec::with_capacity(seeds.len());
            for seed in seeds {
                let mut mask = beam[seed.parent_idx].placed_mask.clone();
                if let Some(p) = seed.position {
                    mask[p] = true;
                }
                let f = self.eval_f_multiset(seed.cost, &seed.state, &mask, &mut function_cache);
                scored.push((seed, f));
            }

            // At most one seed can have is_incumbent true. Pull it out
            // before truncating so the sort/truncate below can never drop
            // it, then always reinsert it -- see BeamEntry.is_incumbent's
            // doc.
            let incumbent_scored = scored
                .iter()
                .position(|(seed, _)| seed.is_incumbent)
                .map(|i| scored.remove(i));

            // DIDP_POSITION_LNS_DEDUP: collapses `scored` down to one entry
            // per distinct DyPDL state (keeping whichever has the better
            // `f`, the same criterion the truncation sort just below
            // already ranks by) before that truncation runs, not after --
            // deduping after truncation would just shrink an already-
            // truncated beam further without giving the freed-up slots to
            // some other, genuinely different candidate that truncation
            // had already discarded. Excludes the incumbent-protected seed
            // (already pulled out above) unconditionally, same as
            // truncation itself -- dedup must never break the
            // is_incumbent lineage guarantee (see its doc).
            if self.dedup_enabled {
                let mut best_for_state: HashMap<HashableState, usize> = HashMap::new();
                for (i, (seed, f)) in scored.iter().enumerate() {
                    let key = HashableState::from(seed.state.clone());
                    match best_for_state.get(&key) {
                        Some(&existing) => {
                            let existing_f = scored[existing].1;
                            if !exceed_bound(&self.model, *f, Some(existing_f)) {
                                best_for_state.insert(key, i);
                            }
                        }
                        None => {
                            best_for_state.insert(key, i);
                        }
                    }
                }
                let keep_indices: HashSet<usize> = best_for_state.into_values().collect();
                let mut i = 0usize;
                scored.retain(|_| {
                    let keep = keep_indices.contains(&i);
                    i += 1;
                    keep
                });
            }

            scored.sort_by(|(_, a), (_, b)| {
                if exceed_bound(&self.model, *a, Some(*b)) {
                    std::cmp::Ordering::Greater
                } else if exceed_bound(&self.model, *b, Some(*a)) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            let keep = beam_width - if incumbent_scored.is_some() { 1 } else { 0 };
            scored.truncate(keep);
            scored.extend(incumbent_scored);

            let mut new_beam = Vec::with_capacity(scored.len());
            for (seed, _) in scored {
                let parent = &beam[seed.parent_idx];

                let mut placed = parent.placed.clone();
                let mut forbidden = parent.forbidden.clone();
                if seed.transition.id != NO_CATALOG_ID {
                    forbidden.extend(
                        self.transition_mutex
                            .get_forbidden_after(seed.transition.forced, seed.transition.id)
                            .iter()
                            .copied(),
                    );
                }
                placed.push(seed.transition);
                let mut placed_positions = parent.placed_positions.clone();
                let mut placed_mask = parent.placed_mask.clone();
                let mut in_degree = parent.in_degree.clone();
                let mut inserted = parent.inserted;

                if let Some(position) = seed.position {
                    placed_positions.push(position);
                    placed_mask[position] = true;
                    for &successor in &successors[position] {
                        in_degree[successor] -= 1;
                    }
                } else {
                    inserted += 1;
                }

                new_beam.push(BeamEntry {
                    placed,
                    placed_positions,
                    placed_mask,
                    in_degree,
                    forbidden,
                    state: seed.state,
                    cost: seed.cost,
                    inserted,
                    is_incumbent: seed.is_incumbent,
                });
            }

            // How much of this beam layer's nominal width is actually
            // distinct search: `new_beam` is exactly the layer LNBS's own
            // `beam_search` would state-registry-deduplicate before
            // continuing, but beam_repair (self-contained, see the module
            // doc) never does -- two different orderings of the same freed
            // positions can easily land on the identical DyPDL state and
            // both survive truncation, silently shrinking how many really
            // different candidates `beam_width` is buying versus its
            // nominal size. Counted at every layer, not just the last one,
            // since duplicates lost at an early layer are duplicates for
            // every layer built on top of them.
            if self.effort_diag_enabled {
                let distinct: HashSet<HashableState> = new_beam
                    .iter()
                    .map(|entry| HashableState::from(entry.state.clone()))
                    .collect();
                self.effort_beam_layer_total_states += new_beam.len() as u64;
                self.effort_beam_layer_distinct_states += distinct.len() as u64;
            }

            // Entries that have placed every original position are set
            // aside into `done` -- see beam_repair's doc -- so the next
            // round (if any, spent purely on insertion_slack) only
            // generates candidates for entries that still need one.
            let mut next_beam = Vec::with_capacity(new_beam.len());
            for entry in new_beam {
                if entry.placed_mask.iter().all(|&placed| placed) {
                    done.push(entry);
                } else {
                    next_beam.push(entry);
                }
            }
            beam = next_beam;
        }

        beam.extend(done);

        // Deterministic argmin over the final beam's feasible completions.
        let mut best: Option<(&BeamEntry<T>, T)> = None;
        // Every entry with a valid base cost, only collected when
        // sa_enabled (see sa_sample's doc) -- unused, and its allocation
        // skipped, otherwise.
        let mut feasible: Vec<(&BeamEntry<T>, T)> = Vec::new();

        for entry in &beam {
            function_cache.parent.clear();
            function_cache.child.clear();
            if let Some(base_cost) = self
                .model
                .eval_base_cost(&entry.state, &mut function_cache.child)
            {
                let final_cost = (self.base_cost_evaluator)(entry.cost, base_cost);
                let better = best
                    .as_ref()
                    .map_or(true, |(_, best_cost)| !exceed_bound(&self.model, final_cost, Some(*best_cost)));
                if better {
                    best = Some((entry, final_cost));
                }
                if self.sa_enabled {
                    feasible.push((entry, final_cost));
                }
            }
        }

        // Metropolis-weighted sampling over the beam's own already-generated
        // candidates -- see sa_enabled's doc for why this replaced an outer
        // accept/reject on beam_repair's single result: that could never
        // fire, since the incumbent-protected entry always survives to the
        // final beam and always ties-or-beats every other completion, so
        // argmin could never return worse than current_cost for an outer
        // check to reject. The incumbent-protection mechanism above is
        // completely unchanged -- this only touches which already-produced
        // entry gets returned. Ported from deorder_lns.rs's identical block.
        let selected = match best {
            None => None,
            Some((best_entry, best_cost)) if !self.sa_enabled => Some((best_entry, best_cost)),
            Some((best_entry, best_cost)) => {
                if !self.sa_calibrated {
                    // Calibrate from the first SA_CALIBRATION_WINDOW
                    // non-tied deltas seen, NOT gated on stall_count -- see
                    // sa_stall_threshold's doc on the struct for why
                    // calibration and sampling are deliberately decoupled.
                    for &(_, cost) in &feasible {
                        let delta = self.sa_delta(cost, best_cost);
                        if delta > 0.0 {
                            self.sa_calibration_deltas.push(delta);
                        }
                    }
                    if self.sa_calibration_deltas.len() >= Self::SA_CALIBRATION_WINDOW {
                        let mean_delta = self.sa_calibration_deltas.iter().sum::<f64>()
                            / self.sa_calibration_deltas.len() as f64;
                        self.sa_t0 = mean_delta.max(1e-9);
                        self.sa_calibrated = true;
                    }
                    Some((best_entry, best_cost))
                } else if self.stall_count < self.sa_stall_threshold {
                    Some((best_entry, best_cost))
                } else {
                    Some(self.sa_sample(&feasible, best_cost))
                }
            }
        };

        selected.map(|(entry, final_cost)| {
            let transitions: Vec<Transition> =
                entry.placed.iter().map(|t| t.transition.clone()).collect();
            (transitions, entry.placed_positions.clone(), final_cost)
        })
    }

    const SA_CALIBRATION_WINDOW: usize = 30;

    // Returns whether `cost` is a strict improvement over `other`.
    fn is_better(&self, cost: T, other: T) -> bool {
        !exceed_bound(&self.model, cost, Some(other))
    }

    // Same Min/Max sign convention as LocalSearchMode::SimulatedAnnealing:
    // delta > 0 means `cost` is worse than `reference`, regardless of which
    // way the model reduces. 0 for reference itself. Ported from
    // deorder_lns.rs's identical helper.
    fn sa_delta(&self, cost: T, reference: T) -> f64 {
        if self.model.reduce_function == dypdl::ReduceFunction::Max {
            reference - cost
        } else {
            cost - reference
        }
        .to_continuous()
    }

    // Time-based cooling schedule -- identical to deorder_lns.rs's
    // sa_temperature (see its doc for why time-based, not per-call decay).
    fn sa_temperature(&self) -> f64 {
        let elapsed_fraction = if self.time_limit.is_finite() && self.time_limit > 0.0 {
            (self.time_keeper.elapsed_time() / self.time_limit).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.sa_t0 * self.sa_final_temp_ratio.powf(elapsed_fraction)
    }

    // Beam width for this iteration's repair call: fixed at `beam_width`
    // unless beam_width_growth_enabled, in which case it's whatever
    // `arm_beam_size[growth_arm]` has escalated to for that specific arm --
    // see `arm_beam_size`'s doc for why per-arm, not a global time/stall
    // dial.
    fn beam_width_for_arm(&self, growth_arm: usize) -> usize {
        if !self.beam_width_growth_enabled {
            return self.beam_width;
        }

        self.arm_beam_size[growth_arm]
    }

    // Called once per iteration after the outcome is known: doubles
    // `arm_beam_size[growth_arm]` (capped at `beam_width_max`) when this
    // arm's repair call didn't improve on current_cost, or resets it to
    // `beam_width` when it did -- see `arm_beam_size`'s doc for the Lnbs
    // parallel (neighborhood_beam_size's per-window double-or-reset rule).
    fn update_beam_growth(&mut self, growth_arm: usize, improving: bool) {
        if !self.beam_width_growth_enabled {
            return;
        }

        self.arm_beam_size[growth_arm] = if improving {
            self.beam_width
        } else {
            (self.arm_beam_size[growth_arm] * 2).min(self.beam_width_max)
        };
    }

    // Draws one entry from `feasible` with probability proportional to
    // exp(-delta_i / T), delta_i relative to best_cost (0 for the best
    // entry itself, which is therefore always in contention with weight
    // exp(0) = 1). Doesn't generate anything new: every candidate here was
    // already produced and validated by the unmodified beam search above --
    // this only changes which one gets returned. Identical to
    // deorder_lns.rs's sa_sample.
    fn sa_sample<'a>(&mut self, feasible: &[(&'a BeamEntry<T>, T)], best_cost: T) -> (&'a BeamEntry<T>, T) {
        let temperature = self.sa_temperature();
        let mut weights = Vec::with_capacity(feasible.len());
        let mut weight_sum = 0.0f64;
        for &(_, cost) in feasible {
            let w = (-self.sa_delta(cost, best_cost) / temperature).exp();
            weight_sum += w;
            weights.push(w);
        }

        let mut r = self.rng.random::<f64>() * weight_sum;
        for (i, &w) in weights.iter().enumerate() {
            if r < w {
                return feasible[i];
            }
            r -= w;
        }
        feasible[feasible.len() - 1]
    }

    /// Sets an elapsed-time value at which the next call(s) to `search_next` should return early
    /// with `terminated: false`, without waiting for `time_limit` or `stall_limit`/
    /// `stall_time_limit`. `None` (the default) disables this. See the `turn_deadline` field's
    /// doc for the intended use.
    pub fn set_turn_deadline(&mut self, deadline: Option<f64>) {
        self.turn_deadline = deadline;
    }

    /// Adopts `transitions`/`cost` as the current working solution and incumbent if `cost` is
    /// better than this instance's own `best.cost` (a no-op otherwise). For a caller alternating
    /// this against another `Search` impl (see `dual_bound_position_lns_local_search.rs`'s bandit
    /// wrapper): lets an instance whose own search has stagnated resume from a better solution
    /// found by the other solver in the meantime, without reconstructing it -- reconstructing
    /// would also discard everything `set_turn_deadline`'s doc already lists as worth preserving
    /// (SA calibration, size/seed bandit stats, ...). Mirrors exactly what the accept branch below
    /// already does when `self.current` changes (refresh `alternatives`/`current_costs`/`w`/
    /// `stagnation` to match), since skipping any of those steps here would be the same
    /// measured-bug class their own docs describe for an accepted move.
    pub fn adopt_incumbent(&mut self, transitions: Vec<Transition>, cost: T, time: f64) {
        if self.best.cost.is_some_and(|best_cost| !self.is_better(cost, best_cost)) {
            return;
        }

        // `solvable` is otherwise fixed at construction from the INITIAL solution. If that was too
        // short to work with (e.g. an empty solution on a variable-length domain like optw, where the
        // first CABS round only finds cost 0), this instance would report "terminated" on every
        // turn forever, even after an adopted incumbent gives it something real to improve. The
        // constructor's one-time first_call echo is also skipped for an unsolvable instance
        // (search_next returns before reaching it), so consume it here or the first real turn would
        // be wasted re-echoing the incumbent.
        if !self.solvable && transitions.len() >= 2 {
            self.solvable = true;
            self.first_call = false;
        }

        self.current = transitions.clone();
        self.current_cost = cost;
        self.alternatives = compute_alternatives(&self.by_params, &self.current);
        self.refresh_trace();
        self.stagnation.resize(self.current.len(), 0);
        self.stall_count = 0;
        self.last_improvement_time = self.time_keeper.elapsed_time();

        self.best.cost = Some(cost);
        self.best.transitions = transitions;
        self.best.is_infeasible = false;
        self.best.time = time;
    }
}

impl<T, B> Search<T> for PositionLns<T, B>
where
    T: Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    B: FnMut(T, T) -> T,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if !self.solvable {
            return Ok((self.best.clone(), true));
        }

        if self.first_call {
            self.first_call = false;

            return Ok((self.best.clone(), false));
        }

        self.time_keeper.start();

        loop {
            if self.time_keeper.check_time_limit(self.quiet) {
                self.best.time = self.time_keeper.elapsed_time();
                self.best.time_out = true;
                self.time_keeper.stop();

                if self.accept_diag_enabled {
                    eprintln!(
                        "[position_lns iter diag] total_iterations={} total_accepts={}",
                        self.iteration_count, self.accept_count
                    );

                    if self.size_bandit_enabled {
                        for (i, &size) in self.size_arms.iter().enumerate() {
                            eprintln!(
                                "[position_lns size bandit] size={} trials={} reward_mean={:.4} time_mean={:.4}",
                                size, self.size_trials[i], self.size_reward_mean[i], self.size_time_mean[i]
                            );
                        }
                    }

                }

                if self.effort_diag_enabled {
                    eprintln!(
                        "[position_lns effort diag] beam_steps={} beam_seeds_generated={} \
                         beam_truncated_steps={} dominance_checked={} dominance_pruned={} \
                         mutex_skipped={} candidates_attempted={} beam_layer_total_states={} \
                         beam_layer_distinct_states={}",
                        self.effort_beam_steps,
                        self.effort_beam_seeds_generated,
                        self.effort_beam_truncated_steps,
                        self.effort_dominance_checked,
                        self.effort_dominance_pruned,
                        self.effort_mutex_skipped,
                        self.effort_candidates_attempted,
                        self.effort_beam_layer_total_states,
                        self.effort_beam_layer_distinct_states,
                    );
                    let saturation = if self.effort_beam_steps > 0 {
                        self.effort_beam_truncated_steps as f64 / self.effort_beam_steps as f64
                    } else {
                        0.0
                    };
                    let dominance_ratio = if self.effort_dominance_checked > 0 {
                        self.effort_dominance_pruned as f64 / self.effort_dominance_checked as f64
                    } else {
                        0.0
                    };
                    let mutex_ratio = if self.effort_mutex_skipped + self.effort_candidates_attempted > 0 {
                        self.effort_mutex_skipped as f64
                            / (self.effort_mutex_skipped + self.effort_candidates_attempted) as f64
                    } else {
                        0.0
                    };
                    let diversity_ratio = if self.effort_beam_layer_total_states > 0 {
                        self.effort_beam_layer_distinct_states as f64
                            / self.effort_beam_layer_total_states as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[position_lns effort diag] beam_saturation_rate={saturation:.4} \
                         dominance_prune_rate={dominance_ratio:.4} mutex_skip_rate={mutex_ratio:.4} \
                         beam_layer_diversity_rate={diversity_ratio:.4}"
                    );

                    let total_calls = self.effort_fallback_cap_count
                        + self.effort_fallback_infeasible_count
                        + self.effort_full_completion_count;
                    let avg_fail_depth = if self.effort_fallback_infeasible_count > 0 {
                        self.effort_fallback_infeasible_depth_sum as f64
                            / self.effort_fallback_infeasible_count as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[position_lns effort diag] eval_f_multiset_calls={} full_completion={} \
                         cap_hit={} infeasible_fallback={} avg_infeasible_fallback_depth={:.2}",
                        total_calls,
                        self.effort_full_completion_count,
                        self.effort_fallback_cap_count,
                        self.effort_fallback_infeasible_count,
                        avg_fail_depth,
                    );
                }

                if self.contiguity_stats_enabled {
                    let rate = if self.contiguity_stat_freed_ge2 > 0 {
                        self.contiguity_stat_contiguous as f64 / self.contiguity_stat_freed_ge2 as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[position_lns contiguity diag] freed_ge2={} contiguous={} contiguous_rate={:.4}",
                        self.contiguity_stat_freed_ge2, self.contiguity_stat_contiguous, rate
                    );
                    let gate_rate = if self.contiguity_gate_steps_total > 0 {
                        self.contiguity_gate_steps_narrowed as f64 / self.contiguity_gate_steps_total as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[position_lns contiguity gate diag] gate_steps_total={} gate_steps_narrowed={} gate_engage_rate={:.4}",
                        self.contiguity_gate_steps_total, self.contiguity_gate_steps_narrowed, gate_rate
                    );
                }


                return Ok((self.best.clone(), true));
            }

            if self
                .turn_deadline
                .is_some_and(|deadline| self.time_keeper.elapsed_time() >= deadline)
            {
                self.best.time = self.time_keeper.elapsed_time();
                self.time_keeper.stop();

                return Ok((self.best.clone(), false));
            }

            self.iteration_count += 1;

            let time_start = self.time_keeper.elapsed_time();
            let n = self.current.len();
            let (freed, full_reinsert, size_arm, growth_arm, size, seed) = self.select_neighborhood(n);

            let (successors, in_degree) = self.build_successors(n, &freed, full_reinsert);
            let beam_width = self.beam_width_for_arm(growth_arm);
            let insertion_slack = if self.adaptive_insertion_slack_enabled {
                size
            } else {
                self.insertion_slack
            };
            let repaired = self.beam_repair(&successors, &in_degree, n, &freed, beam_width, insertion_slack);

            let Some((order, _positions, cost)) = repaired else {
                if self.enable_stagnation_seed {
                    for &p in &freed {
                        self.stagnation[p] = self.stagnation[p].saturating_add(1);
                    }
                }
                self.update_beam_growth(growth_arm, false);
                continue;
            };

            let improving = self.is_better(cost, self.current_cost);

            self.update_beam_growth(growth_arm, improving);
            // Feeds beam_repair's stall gate (see sa_stall_threshold's doc)
            // -- tracked unconditionally, not just when sa_enabled, so
            // turning the flag on wouldn't start from a stale count.
            if improving {
                self.stall_count = 0;
                self.last_improvement_time = self.time_keeper.elapsed_time();
            } else {
                self.stall_count += 1;
            }

            // See PositionLnsParameters::stall_limit's/stall_time_limit's docs -- a genuine "no
            // progress" signal, separate from time_out, so a caller alternating this with another
            // Search impl can tell the two apart. Checked right after the counter/clock updates
            // above so a stall detected on this iteration ends the search before its
            // (non-improving) outcome is otherwise acted on below.
            let stalled = self.stall_limit.is_some_and(|limit| self.stall_count >= limit)
                || self.stall_time_limit.is_some_and(|limit| {
                    self.time_keeper.elapsed_time() - self.last_improvement_time >= limit
                });

            if stalled {
                self.best.time = self.time_keeper.elapsed_time();
                self.best.time_out = false;
                self.time_keeper.stop();

                return Ok((self.best.clone(), true));
            }

            // Bandit feedback: reward tracks whether THIS destroy set found
            // a genuine improvement over current_cost, independent of the
            // unrelated SA-driven `accepted` decision below -- mirrors
            // Lnbs's own reward definition (lnbs.rs) exactly, normalized
            // magnitude of improvement capped at 1.0, 0.0 otherwise. Shared
            // by size_bandit and seed_bandit -- both react to the same
            // iteration's outcome, just keyed differently.
            let reward = if improving {
                let denom = cmp::max(cost.abs(), self.current_cost.abs()).to_continuous();
                let r = (cost - self.current_cost).to_continuous().abs() / denom;
                r.min(1.0)
            } else {
                0.0
            };
            let time = (self.time_keeper.elapsed_time() - time_start) / self.time_limit;

            if let Some(arm) = size_arm {
                self.update_size_bandit(arm, reward, time);
            }
            if self.seed_bandit_enabled {
                self.update_seed_bandit(seed, growth_arm, reward, time);
            }

            // Once calibrated, beam_repair's own selection already made the
            // accept/reject call (see sa_sample) -- whatever it returns
            // gets taken unconditionally, including a genuine worsening of
            // current_cost. self.best below is still keyed off
            // self.best.cost, not current_cost, so a worsening current
            // still can't lose the incumbent.
            let accepted = if self.sa_enabled && self.sa_calibrated {
                true
            } else {
                improving
            };

            if self.size_trace_enabled {
                eprintln!(
                    "[size trace] iteration={} elapsed={:.3}s size={} beam_width={} improving={} accepted={}",
                    self.iteration_count,
                    self.time_keeper.elapsed_time(),
                    size,
                    beam_width,
                    improving,
                    accepted
                );
            }

            // Credit/penalize the freed positions before self.current is
            // reordered below -- `freed` indexes into the pre-move
            // permutation, matching stagnation's own indexing (see its
            // doc). An accepted move resets every position it touched
            // (freeing it *did* make progress here); a rejected one
            // increments them (freeing them didn't, this time).
            if self.enable_stagnation_seed {
                for &p in &freed {
                    self.stagnation[p] = if accepted { 0 } else { self.stagnation[p].saturating_add(1) };
                }
            }

            if accepted {
                self.current = order;
                self.current_cost = cost;
                self.accept_count += 1;

                // self.current just got reordered (and possibly item-4
                // substituted), so alternatives -- keyed by position --
                // must be refreshed to match, or beam_repair would silently
                // place whatever transition originally sat at a position
                // instead of what's actually there now. See
                // deorder_lns.rs's identical alternatives-refresh doc for
                // why this was a real, measured bug there.
                self.alternatives = compute_alternatives(&self.by_params, &self.current);
                self.refresh_trace();
                // stagnation is indexed by position and was sized to the
                // starting transition count -- insertion_slack (see its
                // doc) can change self.current's length on accept, unlike
                // this file's original invariant that positions are only
                // ever reordered/substituted, never added or removed.
                self.stagnation.resize(self.current.len(), 0);

                if self.accept_diag_enabled {
                    eprintln!(
                        "[accept diag] iteration={} elapsed={:.3}s cost={}",
                        self.iteration_count,
                        self.time_keeper.elapsed_time(),
                        cost
                    );
                }

                if self.is_better(cost, self.best.cost.unwrap()) {
                    self.best.cost = Some(cost);
                    self.best.transitions = self.current.clone();
                    self.best.time = self.time_keeper.elapsed_time();

                    if !self.quiet {
                        print_primal_bound(&self.best);
                    }

                    self.time_keeper.stop();

                    return Ok((self.best.clone(), false));
                }
            }
        }
    }
}
