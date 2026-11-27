// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HiGHS backend of culmen: HiGHS 1.16 through `highs-sys`, with every HiGHS callback as a typed, safe
//! context, typed options, and every status.
//!
//! [`Highs`] implements [`Backend`]: load a [`Program`], solve it under [`Settings`], and HiGHS's callbacks reach a
//! [`Callback<Highs>`](culmen_core::Callback) as the generic events of culmen_core, each with a HiGHS context type
//! that offers what HiGHS reports beyond the generic traits ([Callbacks](#callbacks) maps every kind). Of the search
//! capabilities it has [`Heuristics`]: a [`HeuristicHandler`](culmen_core::HeuristicHandler) supplies solutions at
//! HiGHS's user-solution callback. HiGHS takes no lazy constraints and no user cuts and does not price, so a handler of
//! those kinds does not compile for it. The generic path, a [`Run`](culmen_core::Run) with a stop and an observer,
//! works unchanged.
//!
//! ```
//! use culmen_core::{Linear, LinExpr, Model, Outcome, Solver, Var};
//! use culmen_highs::Highs;
//!
//! // A knapsack.
//! let (value, weight) = ([10.0, 13.0, 7.0, 8.0], [5.0, 6.0, 4.0, 3.0]);
//! let mut m = Model::new();
//! let take = m.add_vars(4, |i| Var::binary().name(format!("take[{i}]")));
//! m.add_constraint(LinExpr::sum((0..4).map(|i| weight[i] * take[i])).leq(10.0));
//! m.maximize(LinExpr::sum((0..4).map(|i| value[i] * take[i])));
//!
//! let mut solver = Solver::new(Highs::new(), m.build()?)?;
//! let solved = solver.solve()?;
//! let Outcome::Optimal(best) = solved.outcome else { panic!("expected an optimum, got {:?}", solved.outcome) };
//! assert_eq!(best.objective, 21.0);
//! assert_eq!(best.values[take[1]].round(), 1.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Options and settings
//!
//! Each solve resets HiGHS's options, routes HiGHS's log to the callback only (standard output stays quiet), then
//! applies the backend's [`HighsOptions`] and then the solve's [`Settings`], so a per-solve setting always wins.
//! HiGHS measures its time limit on a clock the backend restarts before every run.
//!
//! # Callbacks
//!
//! [`Highs::EVENTS`](Backend::EVENTS) is every observational event, each from a HiGHS callback kind, with a context of
//! this crate's:
//!
//! | HiGHS 1.16 callback | Event or handler | Context |
//! |---|---|---|
//! | `kHighsCallbackLogging` | `Log` | [`HighsLog`] |
//! | `kHighsCallbackSimplexInterrupt` | `Poll` | [`HighsPoll`], [`HighsPollSource::Simplex`] |
//! | `kHighsCallbackIpmInterrupt` | `Poll` | [`HighsPoll`], [`HighsPollSource::Ipm`] |
//! | `kHighsCallbackMipInterrupt` | `Poll` | [`HighsPoll`], [`HighsPollSource::Mip`] |
//! | `kHighsCallbackMipLogging` | `Progress` | [`HighsProgress`] |
//! | `kHighsCallbackMipSolution` | `Feasible` | [`HighsSolution`] |
//! | `kHighsCallbackMipImprovingSolution` | `Incumbent` | [`HighsSolution`] |
//! | `kHighsCallbackMipUserSolution` | [`HeuristicHandler`](culmen_core::HeuristicHandler) | [`HighsHeuristic`] |
//! | `kHighsCallbackMipGetCutPool` | `Native` | [`HighsNative::CutPool`] |
//! | `kHighsCallbackMipDefineLazyConstraints` | none | HiGHS 1.16 declares it and never calls it |
//! | `kHighsCallbackQpFirstFeasiblePoint` | `Feasible` | [`HighsSolution`], no [`HighsMipData`] |
//! | `kHighsCallbackQpInterrupt` | `Poll` | [`HighsPoll`], [`HighsPollSource::Qp`] |
//!
//! HiGHS has no user cuts and never calls its lazy-constraint callback, so the backend implements neither
//! [`LazyConstraints`](culmen_core::LazyConstraints) nor [`UserCuts`](culmen_core::UserCuts), and a handler of either
//! does not compile for it; HiGHS does not price either ([`Prices`](culmen_core::Prices)). An interrupt is honoured
//! only at the four interrupt kinds, the only context with
//! [`PollContext::interrupt`](culmen_core::PollContext::interrupt); a solution is taken only at the user-solution kind,
//! the only context with [`HeuristicContext::supply`](culmen_core::HeuristicContext::supply), and started only for a
//! solve with a heuristic handler.
//!
//! The QP solver's first feasible point is a feasible solution of the loaded program, so it is `Feasible`.
//! HiGHS's active-set QP solver works on the incumbent model itself, unpresolved and unscaled (`callSolveQp` in
//! `Highs.cpp`, `solveQpAsm` in `QpAsmWrapper.cpp`), so the point holds one value per loaded column. `solveqp` in
//! `a_quass.cpp` reports it only after the starting-point computation left the QP's status unset: the point HiGHS's LP
//! solver found for the constraints (the backend passes every QP with at least one row), or, for a QP without rows,
//! the unconstrained minimizer clamped into the bounds.
//! The point is never reported again as the solve improves it, so it is no incumbent. HiGHS reports no objective with
//! it, so the trampoline values it on the loaded program.
//!
//! # Threads
//!
//! [`Highs`] is `Send`, so it can be moved to a worker thread to solve there. It is not `Sync`: HiGHS does not promise
//! that two threads may call one instance at once, not even to read it. HiGHS keeps one task scheduler per
//! thread, started by the first run on that thread at that run's `threads`; a later run on the same thread asking
//! for another number of threads fails with [`HighsError::Run`]. Dropping a [`Highs`] shuts down the scheduler of
//! the dropping thread. Leaving [`Settings::threads`](culmen_core::Settings::threads) unset never conflicts.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

mod analysis;
mod callback;
mod error;
pub mod files;
mod instance;
mod model_file;
mod options;
mod status;

use core::fmt;
use core::time::Duration;
use std::panic::resume_unwind;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

pub use callback::{
    HighsCut, HighsCutPool, HighsHeuristic, HighsLog, HighsLogType, HighsMipData, HighsNative, HighsPoll,
    HighsPollSource, HighsProgress, HighsQueryOrigin, HighsRootNodePoint, HighsSolution, ReducedCol,
};
use culmen_core::backend::{Edit, Hooks};
use culmen_core::{
    Backend, Configurable, Events, Heuristics, ModelClass, ModelClasses, Objectives, Outcome, Program, ProgramFeatures,
    QuadObjective, RelativeGap, Settings, Solution, Solved, Start, StopReason, TiSlice,
};
use culmen_core::{
    Basis, BasisAnalysis, Col, DualAnalysis, Duals, Iis, IisAnalysis, Ranging, RangingAnalysis, RayAnalysis,
    RelaxationAnalysis, Relaxed, Row, Starts, TiVec, Weights,
};
pub use error::{Dimension, HighsCode, HighsError, UnsoundQp};
pub use model_file::{FileError, ModelFormat};
pub use options::{
    EdgeWeight, HighsOption, HighsOptions, LpSolver, MipIpmSolver, MipLpSolver, OptionValue, PresolveMode,
    ScaleStrategy, SimplexStrategy, Switch,
};
pub use status::{CallStatus, HighsStatus, ModelStatus, SolutionStatus};

use callback::{Hook, NO_RUN, Payload, kinds, trampoline};
use instance::{Instance, Padding};

/// HiGHS's default relative MIP gap, below which an optimal status is a proof of optimality: the default of
/// `mip_rel_gap` in HiGHS 1.16's `lp_data/HighsOptions.h`, to be read again when the HiGHS version changes.
const DEFAULT_RELATIVE_GAP: f64 = 1e-4;
/// HiGHS's default absolute MIP gap: the default of `mip_abs_gap` in HiGHS 1.16's `lp_data/HighsOptions.h`, to be read
/// again when the HiGHS version changes.
const DEFAULT_ABSOLUTE_GAP: f64 = 1e-6;

/// The HiGHS backend: one HiGHS instance and the options every solve applies.
///
/// ```
/// use culmen_core::{Backend, Configurable, Event, ModelClass};
/// use culmen_highs::{Highs, HighsOptions, Switch};
///
/// let highs = Highs::with_options(HighsOptions { parallel: Some(Switch::Off), ..HighsOptions::default() });
/// assert_eq!(highs.options().parallel, Some(Switch::Off));
/// assert!(Highs::EVENTS.contains(Event::Native));
/// assert!(!Highs::CLASSES.contains(ModelClass::Miqp), "HiGHS 1.16 refuses to solve a MIQP");
/// ```
#[derive(Debug)]
pub struct Highs {
    instance: Instance,
    options: HighsOptions,
    loaded: Option<Loaded>,
}

/// What the backend keeps of the loaded program.
#[derive(Debug)]
pub(crate) struct Loaded {
    /// The cost of each column, one per column of the program.
    pub(crate) costs: Vec<f64>,
    pub(crate) quadratic: QuadObjective,
    /// Whether every column has two finite bounds.
    pub(crate) bounded: bool,
    pub(crate) is_mip: bool,
    pub(crate) offset: f64,
    /// How the program's objectives combine.
    combined: Combined,
}

/// How a loaded program's objectives combine: its own alone, or with further ones as HiGHS's option
/// `blend_multi_objectives` picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combined {
    /// The program's own objective alone.
    Single,
    /// Lexicographically: `blend_multi_objectives` off.
    Lexicographic,
    /// Blended: `blend_multi_objectives` on.
    Blended,
}

impl Combined {
    /// How `objectives` combine.
    fn of(objectives: &Objectives) -> Self {
        match objectives {
            _ if !objectives.are_several() => Self::Single,
            Objectives::Lexicographic { .. } => Self::Lexicographic,
            Objectives::Single | Objectives::Blended { .. } => Self::Blended,
        }
    }

    /// The value of `blend_multi_objectives` a solve sets, `None` for a single objective.
    const fn blend(self) -> Option<bool> {
        match self {
            Self::Single => None,
            Self::Lexicographic => Some(false),
            Self::Blended => Some(true),
        }
    }
}

impl Loaded {
    fn of(program: &Program) -> Self {
        Self {
            combined: Combined::of(program.objectives()),
            costs: program.columns().iter().map(|c| c.cost).collect(),
            quadratic: program.quadratic().clone(),
            bounded: program
                .columns()
                .iter()
                .all(|c| c.bounds.lower().is_finite() && c.bounds.upper().is_finite()),
            is_mip: program.class().is_discrete(),
            offset: program.offset(),
        }
    }

    pub(crate) fn num_cols(&self) -> usize {
        self.costs.len()
    }

    /// Whether the program has further objectives, whose solve leaves HiGHS's objective value and MIP bound those of
    /// its last stage or of the blend.
    pub(crate) const fn several(&self) -> bool {
        !matches!(self.combined, Combined::Single)
    }

    /// The objective value of a point with one value per column, quadratic part and offset included.
    ///
    /// # Panics
    ///
    /// When `values` has fewer values than the program has columns.
    pub(crate) fn objective(&self, values: &[f64]) -> f64 {
        let linear = self.costs.iter().zip(values).map(|(c, x)| c * x).sum::<f64>();
        self.offset + linear + self.quadratic.value(TiSlice::from_ref(values))
    }
}

/// What a run left behind.
#[derive(Debug)]
struct Ran {
    elapsed: Duration,
    run: Result<CallStatus, HighsError>,
    panic: Option<Payload>,
    unlisted: Option<HighsError>,
    last_error: Option<String>,
}

impl Highs {
    /// The largest upper bound of a semi-continuous or semi-integer column HiGHS 1.16 solves as given
    /// (`kMaxSemiVariableUpper`); [`Backend::load`] refuses a semi column above it with [`HighsError::SemiUpper`].
    ///
    /// Above it, HiGHS 1.16 lowers the bound to the limit when the lower bound is at most a tenth of the limit, and
    /// refuses the model otherwise; a solution at the lowered bound fails the run, and one below it is optimal for
    /// the original model only by chance.
    pub const SEMI_UPPER_LIMIT: f64 = 1e5;

    /// A backend with HiGHS's default options.
    ///
    /// # Panics
    ///
    /// When HiGHS cannot allocate an instance.
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(HighsOptions::default())
    }

    /// A backend with the given options.
    ///
    /// # Panics
    ///
    /// When HiGHS cannot allocate an instance.
    #[must_use]
    pub fn with_options(options: impl Into<HighsOptions>) -> Self {
        Self {
            instance: Instance::new(),
            options: options.into(),
            loaded: None,
        }
    }

    /// Changes HiGHS's model, which is the program before `edit`, into `program`.
    ///
    /// Bounds, costs, integrality and row bounds change through HiGHS's `BySet` calls, rows and columns are appended
    /// with `Highs_addRows` and `Highs_addCols`, the objective's sense and offset change through
    /// `Highs_changeObjectiveSense` and `Highs_changeObjectiveOffset`, and further objectives are passed again after a
    /// change of costs, columns, sense or offset, since HiGHS 1.16 neither reads the own objective into them nor
    /// extends them. A replaced objective changes in place when its quadratic part stays as `before` held it, and
    /// loads the program whole when that part changed. A QP without rows carries a free row HiGHS needs (see
    /// [`Padding`]); rows appended to it load the program whole.
    ///
    /// # Errors
    ///
    /// [`HighsError::SemiUpper`] for a semi column above [`Highs::SEMI_UPPER_LIMIT`], [`HighsError::TooLarge`] when
    /// the program exceeds HiGHS's 32-bit indices, [`HighsError::Edit`] when HiGHS refuses a call; HiGHS may then
    /// hold any model.
    ///
    /// # Panics
    ///
    /// When `edit` names an id `program` lacks, or rows or columns that are not its last: when `program` is not the
    /// model HiGHS holds changed as `edit` names.
    fn apply(&mut self, program: &Program, edit: &Edit, before: Option<&Loaded>) -> Result<(), HighsError> {
        match edit {
            Edit::ColBounds { cols } => {
                check_semi_upper(program)?;
                self.instance.change_col_bounds(program, cols)
            }
            Edit::ColCosts { cols } => {
                self.instance.change_col_costs(program, cols)?;
                self.pass_further_objectives(program)
            }
            Edit::ColIntegrality { cols } => {
                check_semi_upper(program)?;
                self.instance.change_col_integrality(program, cols)
            }
            Edit::RowBounds { rows } => self.instance.change_row_bounds(program, rows),
            Edit::NewRows { rows } if self.instance.num_row() + rows.len() == program.num_rows() => {
                self.instance.add_rows(program, rows)
            }
            // HiGHS holds the free row of a QP without rows (`Padding`), which the appended rows make needless.
            Edit::NewRows { .. } => self.load(program),
            Edit::NewCols { cols } => {
                check_semi_upper(program)?;
                self.instance.add_cols(program, cols)?;
                self.pass_further_objectives(program)
            }
            Edit::Sense => {
                self.instance.change_sense(program)?;
                self.pass_further_objectives(program)
            }
            Edit::Offset => {
                self.instance.change_offset(program)?;
                self.pass_further_objectives(program)
            }
            Edit::Objective if before.is_some_and(|loaded| loaded.quadratic == *program.quadratic()) => {
                self.instance.change_sense(program)?;
                self.instance.change_offset(program)?;
                let cols: Vec<Col> = program.cols().collect();
                self.instance.change_col_costs(program, &cols)?;
                self.pass_further_objectives(program)
            }
            // A changed quadratic part has no call that sets it in place.
            Edit::Objective => self.load(program),
        }
    }

    /// Passes `program`'s objectives to HiGHS again when it has further ones.
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] when the objectives exceed HiGHS's 32-bit indices, [`HighsError::Edit`] when HiGHS
    /// refuses them.
    fn pass_further_objectives(&mut self, program: &Program) -> Result<(), HighsError> {
        if !program.objectives().are_several() {
            return Ok(());
        }
        self.instance.pass_objectives(program).map_err(|error| match error {
            HighsError::Load => HighsError::Edit {
                call: "Highs_passLinearObjectives",
            },
            other => other,
        })
    }

    /// Reads HiGHS's account of the last run; a solution status HiGHS does not report reads as none.
    ///
    /// # Errors
    ///
    /// [`HighsError::Unlisted`] for a model or solution status HiGHS 1.16 does not define.
    fn status(&self, loaded: &Loaded, run: CallStatus, last_error: Option<String>) -> Result<HighsStatus, HighsError> {
        let count = |v: Option<i64>| u64::try_from(v.unwrap_or(0)).unwrap_or(0);
        let gap = self
            .instance
            .info_double(c"mip_gap")
            .filter(|g| loaded.is_mip && g.is_finite() && *g >= 0.0);
        let solution = |name: &core::ffi::CStr| {
            self.instance
                .info_int(name)
                .map_or(Ok(SolutionStatus::None), SolutionStatus::from_raw)
        };
        Ok(HighsStatus {
            run,
            model: self.instance.model_status()?,
            primal: solution(c"primal_solution_status")?,
            dual: solution(c"dual_solution_status")?,
            mip_gap: gap,
            simplex_iterations: count(self.instance.info_int(c"simplex_iteration_count").map(i64::from)),
            ipm_iterations: count(self.instance.info_int(c"ipm_iteration_count").map(i64::from)),
            qp_iterations: count(self.instance.info_int(c"qp_iteration_count").map(i64::from)),
            primal_dual_objective_error: self
                .instance
                .info_double(c"primal_dual_objective_error")
                .filter(|e| e.is_finite() && *e >= 0.0),
            last_error,
        })
    }

    /// The branch-and-bound nodes of the last run, `None` for a program without integer columns, which HiGHS solves
    /// without a search tree; and the iterations of every LP and QP solver it ran, `None` when HiGHS counted none.
    fn work(&self, loaded: &Loaded) -> (Option<u64>, Option<u64>) {
        let nodes = self
            .instance
            .info_i64(c"mip_node_count")
            .filter(|_| loaded.is_mip)
            .and_then(|n| u64::try_from(n).ok());
        // HiGHS counts a MIP's LP iterations over all nodes in `simplex_iteration_count`, and marks a count it did not
        // keep, or one beyond its integers, with -1.
        let names = [
            c"simplex_iteration_count",
            c"ipm_iteration_count",
            c"crossover_iteration_count",
            c"pdlp_iteration_count",
            c"qp_iteration_count",
        ];
        let counted: Vec<u64> = names
            .into_iter()
            .filter_map(|name| self.instance.info_int(name).and_then(|n| u64::try_from(n).ok()))
            .collect();
        let iterations = (!counted.is_empty()).then(|| counted.iter().sum());
        (nodes, iterations)
    }

    /// The generic outcome of a run HiGHS reports as `status`, after `nodes` branch-and-bound nodes.
    ///
    /// # Errors
    ///
    /// [`HighsError::Run`] when HiGHS failed, or reports an optimum without a solution; [`HighsError::UnsoundQp`]
    /// when it reports a QP optimal with a primal-dual objective error above the options'
    /// [`qp_objective_error`](HighsOptions::qp_objective_error), or unbounded although every column is bounded.
    fn outcome(&self, loaded: &Loaded, status: &HighsStatus, nodes: Option<u64>) -> Result<Outcome, HighsError> {
        if status.run == CallStatus::Error {
            return Err(run_failed(status));
        }
        let best = (status.primal == SolutionStatus::Feasible)
            .then(|| self.solution(loaded))
            .flatten();
        if let Some(why) = unsound_qp(loaded, status, best.as_ref(), self.options.qp_objective_error) {
            return Err(HighsError::UnsoundQp(why));
        }
        let bound = self.mip_bound(loaded);
        match self.ending(status, nodes) {
            Ending::Optimal => best
                .map(|solution| optimum(solution, bound))
                .ok_or_else(|| run_failed(status)),
            Ending::Empty if !self.zero_is_feasible() => Ok(Outcome::Infeasible),
            Ending::Empty => Ok(Outcome::Optimal(Solution {
                objective: loaded.offset,
                values: Vec::new().into(),
            })),
            Ending::Infeasible => Ok(Outcome::Infeasible),
            Ending::Unbounded => Ok(Outcome::Unbounded),
            Ending::Either => Ok(Outcome::InfeasibleOrUnbounded),
            Ending::Stopped(reason) => Ok(Outcome::Stopped { reason, best, bound }),
            Ending::Failed => Err(run_failed(status)),
        }
    }

    /// Whether every row HiGHS holds admits an activity of zero, the only activity a row has without columns.
    ///
    /// HiGHS reports a program without columns as empty, whatever its rows say, so a row such as `0 = 1` is read here.
    fn zero_is_feasible(&self) -> bool {
        let model = self.instance.model();
        model
            .row_lower
            .iter()
            .zip(&model.row_upper)
            .all(|(&lower, &upper)| lower <= 0.0 && 0.0 <= upper)
    }

    /// How a run HiGHS reports with `status` ended after `nodes` branch-and-bound nodes, in culmen's terms.
    fn ending(&self, status: &HighsStatus, nodes: Option<u64>) -> Ending {
        match status.model {
            ModelStatus::Optimal => Ending::Optimal,
            ModelStatus::ModelEmpty => Ending::Empty,
            ModelStatus::Infeasible => Ending::Infeasible,
            ModelStatus::Unbounded => Ending::Unbounded,
            ModelStatus::UnboundedOrInfeasible => Ending::Either,
            ModelStatus::TimeLimit => Ending::Stopped(StopReason::TimeLimit),
            ModelStatus::Interrupt => Ending::Stopped(StopReason::Interrupted),
            ModelStatus::IterationLimit => Ending::Stopped(StopReason::IterationLimit),
            ModelStatus::ObjectiveBound | ModelStatus::ObjectiveTarget => Ending::Stopped(StopReason::ObjectiveLimit),
            ModelStatus::MemoryLimit => Ending::Stopped(StopReason::MemoryLimit),
            ModelStatus::SolutionLimit => Ending::Stopped(self.search_limit(nodes.unwrap_or(0))),
            ModelStatus::Unknown | ModelStatus::HighsInterrupt => Ending::Stopped(StopReason::Other),
            ModelStatus::NotSet
            | ModelStatus::LoadError
            | ModelStatus::ModelError
            | ModelStatus::PresolveError
            | ModelStatus::SolveError
            | ModelStatus::PostsolveError => Ending::Failed,
        }
    }

    /// The best bound HiGHS proved for a MIP with one objective; `None` for any other program or without one.
    fn mip_bound(&self, loaded: &Loaded) -> Option<f64> {
        self.instance
            .info_double(c"mip_dual_bound")
            .filter(|b| loaded.is_mip && !loaded.several() && b.is_finite())
    }

    /// The last run's primal solution, valued with the program's own objective: HiGHS's objective value is the
    /// program's own unless it has further objectives, when it is the last stage's, zero after a lexicographic solve,
    /// or the blend's. `None` when HiGHS holds no primal solution.
    fn solution(&self, loaded: &Loaded) -> Option<Solution> {
        let values = self.instance.primal_values(loaded.num_cols())?;
        let objective = if loaded.several() {
            loaded.objective(&values)
        } else {
            self.instance.objective()
        };
        Some(Solution {
            objective,
            values: values.into(),
        })
    }

    /// Which of HiGHS's search limits a run that ended at its solution limit reached: the node limit HiGHS held for
    /// the run, however it was set, when the node count reached it, else a limit on solutions or leaves.
    fn search_limit(&self, nodes: u64) -> StopReason {
        let node_limit = self
            .instance
            .option_int(c"mip_max_nodes")
            .and_then(|limit| u64::try_from(limit).ok());
        if node_limit.is_some_and(|limit| nodes >= limit) {
            StopReason::NodeLimit
        } else {
            StopReason::SolutionLimit
        }
    }
}

impl Default for Highs {
    fn default() -> Self {
        Self::new()
    }
}

/// Why HiGHS's ending of a QP is not trusted: an optimum at a point `best` holding NaN, an optimum whose primal-dual
/// objective error exceeds `limit`, or unboundedness although every column is bounded; `None` for an LP or a trusted
/// ending.
fn unsound_qp(loaded: &Loaded, status: &HighsStatus, best: Option<&Solution>, limit: RelativeGap) -> Option<UnsoundQp> {
    if loaded.quadratic.is_empty() {
        return None;
    }
    let not_a_number = best.is_some_and(|best| best.values.iter().any(|v| v.is_nan()));
    match status.model {
        ModelStatus::Optimal if not_a_number => Some(UnsoundQp::NotANumber),
        ModelStatus::Optimal => status
            .primal_dual_objective_error
            .filter(|&error| error > limit.get())
            .map(|error| UnsoundQp::ObjectiveError { error }),
        ModelStatus::Unbounded => loaded.bounded.then_some(UnsoundQp::UnboundedInABox),
        _ => None,
    }
}

/// How a run ended, as [`Highs::ending`] reads HiGHS's model status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ending {
    /// Optimal, which a solution must back.
    Optimal,
    /// The program has no columns: optimal at its offset when every row admits zero, infeasible otherwise.
    Empty,
    /// No point is feasible.
    Infeasible,
    /// The objective improves without limit.
    Unbounded,
    /// Infeasible or unbounded, HiGHS did not tell which.
    Either,
    /// Stopped before proving optimality.
    Stopped(StopReason),
    /// HiGHS failed.
    Failed,
}

/// The error of a run that failed with `status`.
fn run_failed(status: &HighsStatus) -> HighsError {
    HighsError::Run {
        status: status.model,
        message: status.last_error.clone(),
    }
}

/// An optimal solution's outcome: [`Outcome::GapReached`] when the bound lies beyond HiGHS's default gaps, so a wider
/// requested gap ended the search, else [`Outcome::Optimal`].
fn optimum(solution: Solution, bound: Option<f64>) -> Outcome {
    match bound {
        Some(bound) if gap_open(solution.objective, bound) => Outcome::GapReached { solution, bound },
        _ => Outcome::Optimal(solution),
    }
}

/// Whether the gap between an objective and a bound is wider than HiGHS's default gaps, so an optimal status
/// reached it only through a wider requested gap.
fn gap_open(objective: f64, bound: f64) -> bool {
    let absolute = (objective - bound).abs();
    let relative = if objective == 0.0 {
        f64::INFINITY
    } else {
        absolute / objective.abs()
    };
    absolute > DEFAULT_ABSOLUTE_GAP && relative > DEFAULT_RELATIVE_GAP
}

/// Refuses a program with a semi column whose upper bound exceeds [`Highs::SEMI_UPPER_LIMIT`].
///
/// # Errors
///
/// [`HighsError::SemiUpper`] for the first such column.
fn check_semi_upper(program: &Program) -> Result<(), HighsError> {
    let high = program
        .columns()
        .iter_enumerated()
        .find(|(_, c)| c.integrality.is_semi() && c.bounds.upper() > Highs::SEMI_UPPER_LIMIT);
    high.map_or(Ok(()), |(col, c)| {
        Err(HighsError::SemiUpper {
            col,
            upper: c.bounds.upper(),
        })
    })
}

/// Resets `instance`'s options, routes the log, and applies `options` and then the solve's settings.
///
/// # Errors
///
/// [`HighsError::OutOfRange`] for the first exact integer beyond HiGHS's integers, and before any option is set;
/// [`HighsError::Option`], [`HighsError::OptionName`] or [`HighsError::OptionString`] for the first option refused.
fn configure(instance: &mut Instance, options: &HighsOptions, settings: &Settings) -> Result<(), HighsError> {
    let (typed, settings) = (options.assignments()?, settings_assignments(settings)?);
    instance.reset_options();
    instance.set_option("output_flag", &OptionValue::Bool(true))?;
    instance.set_option("log_to_console", &OptionValue::Bool(false))?;
    for (name, value) in typed {
        instance.set_option(name, &value)?;
    }
    for HighsOption { name, value } in &options.options {
        instance.set_option(name, value)?;
    }
    for (name, value) in settings {
        instance.set_option(name, &value)?;
    }
    Ok(())
}

/// Runs HiGHS on `loaded` with `hooks` installed for the run alone.
fn run_hooked(instance: &mut Instance, loaded: &Loaded, hooks: Hooks<'_, Highs>) -> Ran {
    let (wanted, heuristic) = (hooks.events(), hooks.wants_heuristic());
    let start = Instant::now();
    let hook = Mutex::new(Hook {
        hooks,
        wanted,
        start,
        program: loaded,
        panic: None,
        unlisted: None,
        last_error: None,
        offered: None,
    });
    let user: *mut core::ffi::c_void = (&raw const hook).cast_mut().cast();
    debug_assert!(!user.is_null(), "run_hooked: a run is installed with no data");
    // SAFETY: `hook` lives on this frame until after the second `hook` call below uninstalls it, and the
    // trampoline reaches it only through a shared reference to its `Mutex`; `trampoline` treats its arguments as
    // HiGHS passes them.
    unsafe { instance.hook(Some(trampoline), user, &kinds(wanted, heuristic)) };
    instance.zero_clocks();
    let run = instance.run();
    // SAFETY: uninstalling, with a null data pointer, before `hook` drops.
    unsafe { instance.hook(Some(trampoline), NO_RUN, &[]) };
    let hook = hook.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ran {
        elapsed: start.elapsed(),
        run,
        panic: hook.panic,
        unlisted: hook.unlisted,
        last_error: hook.last_error,
    }
}

/// The HiGHS options a solve's settings set.
///
/// # Errors
///
/// [`HighsError::OutOfRange`] when the thread count or the seed exceeds HiGHS's integers.
fn settings_assignments(settings: &Settings) -> Result<Vec<(&'static str, OptionValue)>, HighsError> {
    let mut set = Vec::new();
    if let Some(limit) = settings.time_limit {
        set.push(("time_limit", OptionValue::Double(limit.as_secs_f64())));
    }
    if let Some(gap) = settings.relative_gap {
        set.push(("mip_rel_gap", OptionValue::Double(gap.get())));
    }
    if let Some(gap) = settings.absolute_gap {
        set.push(("mip_abs_gap", OptionValue::Double(gap.get())));
    }
    if let Some(threads) = settings.threads {
        // A `usize` beyond `u64` exceeds HiGHS's integers as much as `u64::MAX` does.
        let threads = u64::try_from(threads.get()).unwrap_or(u64::MAX);
        set.push(("threads", options::exact("threads", threads)?));
    }
    if let Some(seed) = settings.seed {
        set.push(("random_seed", options::exact("random_seed", u64::from(seed))?));
    }
    if let Some(presolve) = settings.presolve {
        set.push((
            "presolve",
            OptionValue::String(if presolve { "on" } else { "off" }.to_owned()),
        ));
    }
    if let Some(nodes) = settings.node_limit {
        set.push(("mip_max_nodes", options::limit(nodes)));
    }
    if let Some(solutions) = settings.solution_limit {
        set.push(("mip_max_improving_sols", options::limit(solutions.get())));
    }
    if settings.console {
        set.push(("log_to_console", OptionValue::Bool(true)));
    }
    Ok(set)
}

impl Backend for Highs {
    type Error = HighsError;
    type Status = HighsStatus;
    type Log<'a> = HighsLog<'a>;
    type Progress<'a> = HighsProgress;
    type Poll<'a> = HighsPoll<'a>;
    type Solution<'a> = HighsSolution<'a>;
    type Native<'a> = HighsNative;

    /// Every observational event.
    const EVENTS: Events = Events::ALL;

    /// HiGHS 1.16 solves LPs, MILPs and convex QPs; `Highs::run` refuses a QP with integer columns ("Cannot solve
    /// MIQP problems with HiGHS").
    const CLASSES: ModelClasses = ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp, ModelClass::Qp]);

    /// HiGHS 1.16 takes semi-continuous and semi-integer columns (up to [`Highs::SEMI_UPPER_LIMIT`]) and multiple
    /// linear objectives.
    const FEATURES: ProgramFeatures = ProgramFeatures::ALL;

    fn load(&mut self, program: &Program) -> Result<(), HighsError> {
        self.loaded = None;
        check_semi_upper(program)?;
        self.instance.pass(program, Padding::QpFreeRow)?;
        self.instance.pass_objectives(program)?;
        self.loaded = Some(Loaded::of(program));
        Ok(())
    }

    /// Changes HiGHS's model in place, so the next run starts from the basis of the last: bounds, costs and
    /// integrality through HiGHS's `BySet` calls, new rows and columns through `Highs_addRows` and `Highs_addCols`,
    /// the objective's sense and offset through `Highs_changeObjectiveSense` and `Highs_changeObjectiveOffset`. A
    /// replaced objective whose quadratic part changed and rows added to a QP without rows load the program whole.
    ///
    /// # Panics
    ///
    /// When `program` is not the loaded program changed as `edit` names, which a [`Solver`](culmen_core::Solver)
    /// never hands it.
    fn edit(&mut self, program: &Program, edit: &Edit) -> Result<(), HighsError> {
        let before = self.loaded.take();
        self.apply(program, edit, before.as_ref())?;
        self.loaded = Some(Loaded::of(program));
        Ok(())
    }

    /// Runs HiGHS on the loaded model, with the user-solution callback started only for a heuristic handler.
    ///
    /// # Errors
    ///
    /// [`HighsError::NotLoaded`] before a load; the errors of [`HighsOptions`] and the settings;
    /// [`HighsError::Relaxation`] for a MIP when the options set `solve_relaxation`; [`HighsError::Run`] and
    /// [`HighsError::UnsoundQp`] as the outcome is read; [`HighsError::Unlisted`] when HiGHS passes or returns a value
    /// HiGHS 1.16 does not define, after the run.
    fn solve(&mut self, settings: &Settings, hooks: Hooks<'_, Self>) -> Result<Solved<HighsStatus>, HighsError> {
        let loaded = self.loaded.as_ref().ok_or(HighsError::NotLoaded)?;
        configure(&mut self.instance, &self.options, settings)?;
        if loaded.is_mip && self.instance.option_bool(c"solve_relaxation") == Some(true) {
            return Err(HighsError::Relaxation);
        }
        if let Some(blend) = loaded.combined.blend() {
            self.instance
                .set_option("blend_multi_objectives", &OptionValue::Bool(blend))?;
        }
        let ran = run_hooked(&mut self.instance, loaded, hooks);
        if let Some(payload) = ran.panic {
            resume_unwind(payload);
        }
        if let Some(error) = ran.unlisted {
            return Err(error);
        }
        let status = self.status(loaded, ran.run?, ran.last_error)?;
        let (nodes, iterations) = self.work(loaded);
        let outcome = self.outcome(loaded, &status, nodes)?;
        Ok(Solved {
            outcome,
            status,
            elapsed: ran.elapsed,
            nodes,
            iterations,
        })
    }
}

/// [`HighsOptions`], applied by every solve after resetting HiGHS's options and before the solve's settings.
impl Configurable for Highs {
    type Options = HighsOptions;

    fn options(&self) -> &HighsOptions {
        &self.options
    }

    fn options_mut(&mut self) -> &mut HighsOptions {
        &mut self.options
    }
}

/// HiGHS asks a heuristic handler for a solution at its user-solution callback, `kHighsCallbackMipUserSolution`, at
/// the points [`HighsQueryOrigin`] names.
impl Heuristics for Highs {
    type Heuristic<'a> = HighsHeuristic<'a>;
}

impl Starts for Highs {
    /// Hands HiGHS the start as its solution, which the next run starts from, through `Highs_setSparseSolution`.
    ///
    /// HiGHS chooses the columns a sparse start leaves out, and for a MIP completes them by a sub-MIP. A sparse start
    /// without entries leaves HiGHS starting from its last solution, as a solve without a start does.
    fn set_start(&mut self, start: &Start) -> Result<(), HighsError> {
        let loaded = self.loaded.as_ref().ok_or(HighsError::NotLoaded)?;
        start.check(loaded.num_cols()).map_err(|_| HighsError::Start)?;
        self.instance.set_solution(start.entries())
    }
}

/// HiGHS's dual solution; the `analysis` module documents what HiGHS gives for each analysis.
impl DualAnalysis for Highs {
    fn duals(&self, program: &Program) -> Result<Duals, HighsError> {
        analysis::duals(self, program)
    }
}

/// HiGHS's simplex basis, read through `Highs_getBasis` and set through `Highs_setBasis`.
impl BasisAnalysis for Highs {
    fn basis(&self, program: &Program) -> Result<Option<Basis>, HighsError> {
        analysis::basis(self, program)
    }

    fn set_basis(&mut self, _: &Program, basis: &Basis) -> Result<(), HighsError> {
        analysis::set_basis(self, basis)
    }
}

/// HiGHS's ranging through `Highs_getRanging`, which writes ranging data into the instance.
impl RangingAnalysis for Highs {
    fn ranging(&mut self, program: &Program) -> Result<Ranging, HighsError> {
        analysis::ranging(self, program)
    }
}

/// HiGHS's rays through `Highs_getPrimalRay` and `Highs_getDualRay`, either of which may run its simplex solver.
impl RayAnalysis for Highs {
    fn primal_ray(&mut self, program: &Program) -> Result<Option<TiVec<Col, f64>>, HighsError> {
        analysis::primal_ray(self, program)
    }

    fn dual_ray(&mut self, _: &Program) -> Result<Option<TiVec<Row, f64>>, HighsError> {
        analysis::dual_ray(self)
    }
}

/// HiGHS's IIS, computed under the backend's options and its full IIS strategy.
impl IisAnalysis for Highs {
    fn iis(&mut self, program: &Program) -> Result<Option<Iis>, HighsError> {
        analysis::iis(self, program)
    }
}

/// HiGHS's feasibility relaxation, refused for a program with semi columns ([`HighsError::AnalysisFeature`]).
impl RelaxationAnalysis for Highs {
    fn relax(&mut self, program: &Program, weights: &Weights) -> Result<Option<Relaxed>, HighsError> {
        analysis::relax(self, program, weights)
    }
}

/// A HiGHS version, ordered as versions are; displayed as `major.minor.patch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Version {
    /// The major version.
    pub major: u32,
    /// The minor version.
    pub minor: u32,
    /// The patch version.
    pub patch: u32,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The version of the HiGHS library linked in.
///
/// ```
/// let version = culmen_highs::version();
/// assert_eq!((version.major, version.minor), (1, 16));
/// assert_eq!(version.to_string(), format!("1.16.{}", version.patch));
/// ```
#[must_use]
pub fn version() -> Version {
    // SAFETY: the version getters take no arguments and return plain integers.
    let (major, minor, patch) = unsafe {
        (
            highs_sys::Highs_versionMajor(),
            highs_sys::Highs_versionMinor(),
            highs_sys::Highs_versionPatch(),
        )
    };
    // HiGHS's version numbers are never negative, so their magnitudes are the numbers.
    Version {
        major: major.unsigned_abs(),
        minor: minor.unsigned_abs(),
        patch: patch.unsigned_abs(),
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroUsize;

    use super::*;

    #[test]
    fn test_a_supplied_solution_is_valued_with_the_quadratic_part() {
        // The heuristic context values a supplied solution with `Loaded::objective`: (1 + 2)² + 3·1 - 2 + 2 = 12.
        let mut m = culmen_core::Model::new();
        let x = m.add_var(culmen_core::Var::continuous());
        let y = m.add_var(culmen_core::Var::integer());
        m.minimize(x * x + 2.0 * x * y + y * y + 3.0 * x - y + 2.0);
        let program = m.build().unwrap();
        let values = [1.0, 2.0];
        assert_eq!(Loaded::of(&program).objective(&values), 12.0);
        assert_eq!(program.objective_value(TiSlice::from_ref(&values)), 12.0);
    }

    #[test]
    fn test_a_qp_optimum_is_refused_above_the_options_objective_error_and_taken_below() {
        let mut m = culmen_core::Model::new();
        let x = m.add_var(culmen_core::Var::continuous());
        m.minimize(x * x);
        let loaded = Loaded::of(&m.build().unwrap());
        let status = HighsStatus {
            run: CallStatus::Ok,
            model: ModelStatus::Optimal,
            primal: SolutionStatus::Feasible,
            dual: SolutionStatus::Feasible,
            mip_gap: None,
            simplex_iterations: 0,
            ipm_iterations: 0,
            qp_iterations: 3,
            primal_dual_objective_error: Some(2e-5),
            last_error: None,
        };
        let default = HighsOptions::default().qp_objective_error;
        assert_eq!(default, culmen_core::relative_gap!(1e-5));
        assert_eq!(
            unsound_qp(&loaded, &status, None, default),
            Some(UnsoundQp::ObjectiveError { error: 2e-5 })
        );
        assert_eq!(
            unsound_qp(&loaded, &status, None, culmen_core::relative_gap!(1e-4)),
            None
        );
    }

    #[test]
    fn test_every_setting_maps_to_its_highs_option() {
        let settings = Settings {
            time_limit: Some(core::time::Duration::from_millis(1500)),
            relative_gap: Some(culmen_core::relative_gap!(0.01)),
            absolute_gap: Some(culmen_core::absolute_gap!(0.5)),
            threads: NonZeroUsize::new(2),
            seed: Some(7),
            presolve: Some(false),
            node_limit: Some(100),
            solution_limit: core::num::NonZeroU64::new(3),
            console: true,
        };
        assert_eq!(
            settings_assignments(&settings),
            Ok(vec![
                ("time_limit", OptionValue::Double(1.5)),
                ("mip_rel_gap", OptionValue::Double(0.01)),
                ("mip_abs_gap", OptionValue::Double(0.5)),
                ("threads", OptionValue::Int(2)),
                ("random_seed", OptionValue::Int(7)),
                ("presolve", OptionValue::String("off".to_owned())),
                ("mip_max_nodes", OptionValue::Int(100)),
                ("mip_max_improving_sols", OptionValue::Int(3)),
                ("log_to_console", OptionValue::Bool(true)),
            ])
        );
        let on = Settings {
            presolve: Some(true),
            ..Settings::new()
        };
        assert_eq!(
            settings_assignments(&on),
            Ok(vec![("presolve", OptionValue::String("on".to_owned()))])
        );
        assert_eq!(settings_assignments(&Settings::new()), Ok(vec![]));
    }

    #[test]
    fn test_seeds_and_thread_counts_beyond_highs_integers_are_refused() {
        let with = |seed, threads| {
            settings_assignments(&Settings {
                seed: Some(seed),
                threads: NonZeroUsize::new(threads),
                node_limit: Some(u64::MAX),
                ..Settings::new()
            })
        };
        let largest = 2_147_483_647;
        assert_eq!(
            with(largest, 1),
            Ok(vec![
                ("threads", OptionValue::Int(1)),
                ("random_seed", OptionValue::Int(2_147_483_647)),
                ("mip_max_nodes", OptionValue::Int(2_147_483_647)),
            ])
        );
        assert_eq!(
            with(largest + 1, 1),
            Err(HighsError::OutOfRange {
                name: "random_seed",
                value: 2_147_483_648,
                max: 2_147_483_647,
            })
        );
        assert_eq!(
            with(1, usize::MAX),
            Err(HighsError::OutOfRange {
                name: "threads",
                value: u64::MAX,
                max: 2_147_483_647,
            })
        );
    }
}
