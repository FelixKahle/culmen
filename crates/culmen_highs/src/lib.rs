// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HiGHS backend of culmen: HiGHS 1.15 through `highs-sys`, with every HiGHS callback as a typed, safe
//! context, typed options, and every status.
//!
//! [`Highs`] implements [`Backend`]: load a [`Program`], solve it under [`Settings`], and HiGHS's callbacks reach a
//! [`Callback<Highs>`](culmen_core::Callback) as the generic events of culmen_core, each with a HiGHS context type
//! that offers what HiGHS reports beyond the generic traits ([`callback`] maps every kind). The generic path, a
//! [`Request`](culmen_core::Request) with a stop and an observer, works unchanged.
//!
//! ```
//! use culmen_core::{Linear, LinExpr, Model, Outcome, Request, Solver, Var};
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
//! let solved = solver.solve(Request::new())?;
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
//! # Threads
//!
//! [`Highs`] is `Send`, so it can be moved to a worker thread to solve there. HiGHS keeps one task scheduler per
//! thread, started by the first run on that thread at that run's `threads`; a later run on the same thread asking
//! for another number of threads fails with [`HighsError::Run`]. Dropping a [`Highs`] shuts down the scheduler of
//! the dropping thread. Leaving [`Settings::threads`](culmen_core::Settings::threads) unset never conflicts.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

mod analysis;
pub mod callback;
mod error;
pub mod files;
mod instance;
mod options;
mod status;

use core::fmt;
use core::time::Duration;
use std::panic::resume_unwind;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

pub use callback::{
    Cut, CutPool, HighsHeuristic, HighsLog, HighsLogType, HighsMipProgress, HighsNative, HighsPoll, HighsSolution,
    MipData, PollSource, QueryOrigin, ReducedCol, RootNodePoint,
};
use culmen_core::{Analyses, AnalysisError, Basis, Col, Duals, Iis, Ranging, Relaxed, Row, TiVec, Weights};
use culmen_core::{
    Backend, Callback, Edit, Events, ModelClass, ModelClasses, Objectives, Outcome, Program, ProgramFeatures,
    QuadObjective, Sense, Settings, Solution, Solved, Start, StopReason, TiSlice, Unsupported,
};
pub use error::{Dimension, HighsError, UnsoundQp};
pub use options::{
    EdgeWeight, HighsOptions, LpSolver, MipIpmSolver, MipLpSolver, OnOff, OptionValue, RawOption, ScaleStrategy,
    SimplexStrategy, Switch,
};
pub use status::{CallStatus, HighsStatus, ModelStatus, SolutionStatus};

use callback::{Hook, NO_RUN, Payload, kinds, trampoline};
use instance::{Instance, Padding, UNDEFINED};

/// HiGHS's default relative MIP gap, below which an optimal status is a proof of optimality.
const DEFAULT_RELATIVE_GAP: f64 = 1e-4;
/// HiGHS's default absolute MIP gap.
const DEFAULT_ABSOLUTE_GAP: f64 = 1e-6;

/// The HiGHS backend: one HiGHS instance and the options every solve applies.
///
/// ```
/// use culmen_core::{Backend, Events, ModelClass};
/// use culmen_highs::{Highs, HighsOptions, Switch};
///
/// let highs = Highs::with_options(HighsOptions { presolve: Some(Switch::Off), ..HighsOptions::default() });
/// assert_eq!(highs.options().presolve, Some(Switch::Off));
/// assert!(Highs::EVENTS.contains(Events::HEURISTIC));
/// assert!(!Highs::EVENTS.contains(Events::LAZY), "HiGHS 1.15 never calls its lazy-constraint callback");
/// assert!(!Highs::CLASSES.contains(ModelClass::Miqp), "HiGHS 1.15 refuses to solve a MIQP");
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
    pub(crate) sense: Sense,
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
            is_mip: program.is_mip(),
            offset: program.offset(),
            sense: program.sense(),
        }
    }

    pub(crate) fn num_cols(&self) -> usize {
        self.costs.len()
    }

    /// Whether the program has further objectives, whose solve leaves HiGHS's objective value and MIP bound those of
    /// its last stage or of the blend.
    const fn several(&self) -> bool {
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
struct Ran {
    elapsed: Duration,
    run: CallStatus,
    panic: Option<Payload>,
    last_error: Option<String>,
}

impl Highs {
    /// The largest relative primal-dual objective error, `|P - D| / (1 + |P| + |D|)`, at which a QP optimum HiGHS
    /// reports is taken as optimal; above it the solve fails with [`HighsError::UnsoundQp`].
    ///
    /// HiGHS 1.15's QP solver can report optimality at a point whose primal and dual objectives disagree by tens of
    /// percent, after its iterates turned NaN, and that point need not even be feasible; on random convex QPs its
    /// correct optima stay below `1e-6`.
    pub const QP_OBJECTIVE_ERROR: f64 = 1e-5;

    /// The largest upper bound of a semi-continuous or semi-integer column HiGHS 1.15 solves as given
    /// (`kMaxSemiVariableUpper`); [`Backend::load`] refuses a semi column above it with [`HighsError::SemiUpper`].
    ///
    /// Above it, HiGHS 1.15 lowers the bound to the limit when the lower bound is at most a tenth of the limit, and
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

    /// The options every solve applies.
    #[must_use]
    pub const fn options(&self) -> &HighsOptions {
        &self.options
    }

    /// The options every solve applies, to change between solves.
    pub const fn options_mut(&mut self) -> &mut HighsOptions {
        &mut self.options
    }

    /// Changes HiGHS's model, which is the program before `edit`, into `program`.
    ///
    /// Bounds, costs, integrality and row bounds change through HiGHS's `BySet` calls, rows and columns are appended
    /// with `Highs_addRows` and `Highs_addCols`, and further objectives are passed again after a change of costs or
    /// columns, since HiGHS 1.15 neither reads the costs into them nor extends them. A QP without rows carries a free
    /// row HiGHS needs (see [`Padding`]); rows appended to it, and any edit this version does not know, load the
    /// program whole.
    ///
    /// # Errors
    ///
    /// [`HighsError::SemiUpper`] for a semi column above [`Highs::SEMI_UPPER_LIMIT`], [`HighsError::TooLarge`] when
    /// the program exceeds HiGHS's 32-bit indices, [`HighsError::Edit`] when HiGHS refuses a call; HiGHS may then
    /// hold any model.
    fn apply(&mut self, program: &Program, edit: Edit<'_>) -> Result<(), HighsError> {
        match edit {
            Edit::ColBounds(cols) => {
                check_semi_upper(program)?;
                self.instance.change_col_bounds(program, cols)
            }
            Edit::ColCosts(cols) => {
                self.instance.change_col_costs(program, cols)?;
                self.pass_further_objectives(program)
            }
            Edit::ColIntegrality(cols) => {
                check_semi_upper(program)?;
                self.instance.change_col_integrality(program, cols)
            }
            Edit::RowBounds(rows) => self.instance.change_row_bounds(program, rows),
            Edit::NewRows(rows) if self.instance.num_row() + rows.len() == program.num_rows() => {
                self.instance.add_rows(program, rows)
            }
            Edit::NewCols(cols) => {
                check_semi_upper(program)?;
                self.instance.add_cols(program, cols)?;
                self.pass_further_objectives(program)
            }
            _ => self.load(program),
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
            HighsError::Load { .. } => HighsError::Edit {
                call: "Highs_passLinearObjectives",
            },
            other => other,
        })
    }

    /// Reads HiGHS's account of the last run.
    fn status(&self, loaded: &Loaded, run: CallStatus, last_error: Option<String>) -> HighsStatus {
        let count = |v: Option<i64>| u64::try_from(v.unwrap_or(0)).unwrap_or(0);
        let gap = self
            .instance
            .info_double(c"mip_gap")
            .filter(|g| loaded.is_mip && g.is_finite() && *g >= 0.0);
        HighsStatus {
            run,
            model: self.instance.model_status(),
            primal: SolutionStatus::from_raw(self.instance.info_int(c"primal_solution_status").unwrap_or(0)),
            dual: SolutionStatus::from_raw(self.instance.info_int(c"dual_solution_status").unwrap_or(0)),
            mip_gap: gap,
            mip_nodes: count(self.instance.info_i64(c"mip_node_count")),
            simplex_iterations: count(self.instance.info_int(c"simplex_iteration_count").map(i64::from)),
            ipm_iterations: count(self.instance.info_int(c"ipm_iteration_count").map(i64::from)),
            qp_iterations: count(self.instance.info_int(c"qp_iteration_count").map(i64::from)),
            primal_dual_objective_error: self
                .instance
                .info_double(c"primal_dual_objective_error")
                .filter(|e| e.is_finite() && *e >= 0.0),
            last_error,
        }
    }

    /// The generic outcome of a run HiGHS reports as `status`.
    ///
    /// # Errors
    ///
    /// [`HighsError::Run`] when HiGHS failed, or reports an optimum without a solution; [`HighsError::UnsoundQp`]
    /// when it reports a QP optimal with a primal-dual objective error above [`Highs::QP_OBJECTIVE_ERROR`], or
    /// unbounded although every column is bounded.
    fn outcome(&self, loaded: &Loaded, settings: &Settings, status: &HighsStatus) -> Result<Outcome, HighsError> {
        if status.run == CallStatus::Error {
            return Err(run_failed(status));
        }
        let best = (status.primal == SolutionStatus::Feasible).then(|| self.solution(loaded));
        if let Some(why) = unsound_qp(loaded, status, best.as_ref()) {
            return Err(HighsError::UnsoundQp(why));
        }
        let bound = self.mip_bound(loaded);
        match self.ending(settings, status) {
            Ending::Optimal => best
                .map(|solution| optimum(solution, bound))
                .ok_or_else(|| run_failed(status)),
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

    /// How a run HiGHS reports with `status` ended, in culmen's terms.
    fn ending(&self, settings: &Settings, status: &HighsStatus) -> Ending {
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
            ModelStatus::SolutionLimit => Ending::Stopped(self.search_limit(settings, status.mip_nodes)),
            ModelStatus::Unknown | ModelStatus::HighsInterrupt => Ending::Stopped(StopReason::Other),
            ModelStatus::NotSet
            | ModelStatus::LoadError
            | ModelStatus::ModelError
            | ModelStatus::PresolveError
            | ModelStatus::SolveError
            | ModelStatus::PostsolveError
            | ModelStatus::Other(_) => Ending::Failed,
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
    /// or the blend's.
    fn solution(&self, loaded: &Loaded) -> Solution {
        let values = self.instance.primal_values(loaded.num_cols());
        let objective = if loaded.several() {
            loaded.objective(&values)
        } else {
            self.instance.objective()
        };
        Solution {
            objective,
            values: values.into(),
        }
    }

    /// Which of HiGHS's search limits a run that ended at its solution limit reached: the node limit when one is
    /// set and the node count reached it, else a limit on solutions or leaves.
    fn search_limit(&self, settings: &Settings, nodes: u64) -> StopReason {
        let node_limit = settings.node_limit.or(self.options.mip_max_nodes);
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

/// Why HiGHS's ending of a QP is not trusted: an optimum at a point `best` holding NaN, an optimum its own primal-dual
/// objective error does not support, or unboundedness although every column is bounded; `None` for an LP or a trusted
/// ending.
fn unsound_qp(loaded: &Loaded, status: &HighsStatus, best: Option<&Solution>) -> Option<UnsoundQp> {
    if loaded.quadratic.is_empty() {
        return None;
    }
    let not_a_number = best.is_some_and(|best| best.values.iter().any(|v| v.is_nan()));
    match status.model {
        ModelStatus::Optimal if not_a_number => Some(UnsoundQp::NotANumber),
        ModelStatus::Optimal => status
            .primal_dual_objective_error
            .filter(|&error| error > Highs::QP_OBJECTIVE_ERROR)
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
    /// The program has no columns: optimal at its offset.
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
    for RawOption { name, value } in &options.raw {
        instance.set_option(name, value)?;
    }
    for (name, value) in settings {
        instance.set_option(name, &value)?;
    }
    Ok(())
}

/// Runs HiGHS on `loaded` with `callback` installed for the run alone.
fn run_hooked(instance: &mut Instance, loaded: &Loaded, callback: &mut dyn Callback<Highs>) -> Ran {
    let wanted = callback.events().intersection(Highs::EVENTS);
    let start = Instant::now();
    let hook = Mutex::new(Hook {
        callback,
        wanted,
        start,
        program: loaded,
        panic: None,
        last_error: None,
        offered: None,
        incumbent: None,
    });
    let user: *mut core::ffi::c_void = (&raw const hook).cast_mut().cast();
    debug_assert!(!user.is_null(), "run_hooked: a run is installed with no data");
    // SAFETY: `hook` lives on this frame until after the second `hook` call below uninstalls it, and the
    // trampoline reaches it only through a shared reference to its `Mutex`; `trampoline` treats its arguments as
    // HiGHS passes them.
    unsafe { instance.hook(Some(trampoline), user, &kinds(wanted)) };
    instance.zero_clocks();
    let run = instance.run();
    // SAFETY: uninstalling, with a null data pointer, before `hook` drops.
    unsafe { instance.hook(Some(trampoline), NO_RUN, &[]) };
    let hook = hook.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ran {
        elapsed: start.elapsed(),
        run,
        panic: hook.panic,
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
        set.push(("mip_max_improving_sols", options::limit(solutions)));
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
    type Progress<'a> = HighsMipProgress;
    type Poll<'a> = HighsPoll<'a>;
    type Solution<'a> = HighsSolution<'a>;
    type Lazy<'a> = Unsupported;
    type Cut<'a> = Unsupported;
    type Heuristic<'a> = HighsHeuristic<'a>;
    type Native<'a> = HighsNative;

    const EVENTS: Events = Events::LOG
        .union(Events::PROGRESS)
        .union(Events::POLL)
        .union(Events::FEASIBLE)
        .union(Events::INCUMBENT)
        .union(Events::HEURISTIC)
        .union(Events::NATIVE);

    /// HiGHS 1.15 solves LPs, MILPs and convex QPs; `Highs::run` refuses a QP with integer columns ("Cannot solve
    /// MIQP problems with HiGHS").
    const CLASSES: ModelClasses = ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp, ModelClass::Qp]);

    /// Every analysis, each through its HiGHS call; the `analysis` module documents what HiGHS gives.
    const ANALYSES: Analyses = Analyses::ALL;
    /// HiGHS 1.15 takes semi-continuous and semi-integer columns (up to [`Highs::SEMI_UPPER_LIMIT`]), multiple
    /// linear objectives, and starts.
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
    /// integrality through HiGHS's `BySet` calls, new rows and columns through `Highs_addRows` and `Highs_addCols`.
    /// Rows added to a QP without rows, and any edit this version does not know, load the program whole.
    fn edit(&mut self, program: &Program, edit: Edit<'_>) -> Result<(), HighsError> {
        self.loaded = None;
        self.apply(program, edit)?;
        self.loaded = Some(Loaded::of(program));
        Ok(())
    }

    /// Hands HiGHS the start as its solution, which the next run starts from.
    ///
    /// A sparse start's other columns are HiGHS's undefined value, as `Highs_setSparseSolution` leaves them. That
    /// call is not used: it refuses a value outside its column's bounds, a semi column's zero among them.
    fn set_start(&mut self, start: &Start) -> Result<(), HighsError> {
        let loaded = self.loaded.as_ref().ok_or(HighsError::NotLoaded)?;
        start.check(loaded.num_cols()).map_err(|_| HighsError::Start)?;
        let mut values = vec![UNDEFINED; loaded.num_cols()];
        start.entries().for_each(|(col, value)| values[col.index()] = value);
        self.instance.set_solution(&values)
    }

    fn solve(
        &mut self,
        settings: &Settings,
        callback: &mut dyn Callback<Self>,
    ) -> Result<Solved<HighsStatus>, HighsError> {
        let loaded = self.loaded.as_ref().ok_or(HighsError::NotLoaded)?;
        configure(&mut self.instance, &self.options, settings)?;
        if let Some(blend) = loaded.combined.blend() {
            self.instance
                .set_option("blend_multi_objectives", &OptionValue::Bool(blend))?;
        }
        let ran = run_hooked(&mut self.instance, loaded, callback);
        if let Some(payload) = ran.panic {
            resume_unwind(payload);
        }
        let status = self.status(loaded, ran.run, ran.last_error);
        let outcome = self.outcome(loaded, settings, &status)?;
        Ok(Solved {
            outcome,
            status,
            elapsed: ran.elapsed,
        })
    }

    fn duals(&self, program: &Program) -> Result<Duals, AnalysisError<HighsError>> {
        analysis::duals(self, program)
    }

    fn basis(&self, program: &Program) -> Result<Option<Basis>, AnalysisError<HighsError>> {
        analysis::basis(self, program)
    }

    fn set_basis(&mut self, _: &Program, basis: &Basis) -> Result<(), AnalysisError<HighsError>> {
        analysis::set_basis(self, basis)
    }

    fn ranging(&mut self, program: &Program) -> Result<Ranging, AnalysisError<HighsError>> {
        analysis::ranging(self, program)
    }

    fn primal_ray(&mut self, program: &Program) -> Result<Option<TiVec<Col, f64>>, AnalysisError<HighsError>> {
        analysis::primal_ray(self, program)
    }

    fn dual_ray(&mut self, _: &Program) -> Result<Option<TiVec<Row, f64>>, AnalysisError<HighsError>> {
        analysis::dual_ray(self)
    }

    fn iis(&mut self, program: &Program) -> Result<Option<Iis>, AnalysisError<HighsError>> {
        analysis::iis(self, program)
    }

    fn relax(&mut self, program: &Program, weights: &Weights) -> Result<Option<Relaxed>, AnalysisError<HighsError>> {
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
/// assert_eq!((version.major, version.minor), (1, 15));
/// assert_eq!(version.to_string(), format!("1.15.{}", version.patch));
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
                value: 2_147_483_648
            })
        );
        assert_eq!(
            with(1, usize::MAX),
            Err(HighsError::OutOfRange {
                name: "threads",
                value: u64::MAX
            })
        );
    }
}
