// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HiGHS backend of culmen: HiGHS 1.15 through `highs-sys`, with every HiGHS callback as a typed, safe
//! context, typed options, and every status.
//!
//! [`Highs`] implements [`Backend`]: load an [`Lp`], solve it under [`Settings`], and HiGHS's callbacks reach a
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

pub mod callback;
mod error;
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
use culmen_core::{Backend, Callback, Events, Lp, Outcome, Sense, Settings, Solution, Solved, StopReason, Unsupported};
pub use error::{Dimension, HighsError};
pub use options::{
    EdgeWeight, HighsOptions, LpSolver, MipIpmSolver, MipLpSolver, OnOff, OptionValue, RawOption, ScaleStrategy,
    SimplexStrategy, Switch,
};
pub use status::{CallStatus, HighsStatus, ModelStatus, SolutionStatus};

use callback::{Hook, NO_RUN, Payload, kinds, trampoline};
use instance::Instance;

/// HiGHS's default relative MIP gap, below which an optimal status is a proof of optimality.
const DEFAULT_RELATIVE_GAP: f64 = 1e-4;
/// HiGHS's default absolute MIP gap.
const DEFAULT_ABSOLUTE_GAP: f64 = 1e-6;

/// The HiGHS backend: one HiGHS instance and the options every solve applies.
///
/// ```
/// use culmen_core::{Backend, Events};
/// use culmen_highs::{Highs, HighsOptions, Switch};
///
/// let highs = Highs::with_options(HighsOptions { presolve: Some(Switch::Off), ..HighsOptions::default() });
/// assert_eq!(highs.options().presolve, Some(Switch::Off));
/// assert!(Highs::EVENTS.contains(Events::HEURISTIC));
/// assert!(!Highs::EVENTS.contains(Events::LAZY), "HiGHS 1.15 never calls its lazy-constraint callback");
/// ```
#[derive(Debug)]
pub struct Highs {
    instance: Instance,
    options: HighsOptions,
    loaded: Option<Loaded>,
}

/// What the backend keeps of the loaded LP.
#[derive(Debug)]
pub(crate) struct Loaded {
    /// The cost of each column, one per column of the LP.
    pub(crate) costs: Vec<f64>,
    pub(crate) is_mip: bool,
    pub(crate) offset: f64,
    pub(crate) sense: Sense,
}

impl Loaded {
    fn of(lp: &Lp) -> Self {
        Self {
            costs: lp.columns().iter().map(|c| c.cost).collect(),
            is_mip: lp.is_mip(),
            offset: lp.offset(),
            sense: lp.sense(),
        }
    }

    pub(crate) fn num_cols(&self) -> usize {
        self.costs.len()
    }

    /// The objective value of a point with one value per column, offset included.
    pub(crate) fn objective(&self, values: &[f64]) -> f64 {
        self.offset + self.costs.iter().zip(values).map(|(c, x)| c * x).sum::<f64>()
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
            last_error,
        }
    }

    /// The generic outcome of a run HiGHS reports as `status`.
    ///
    /// # Errors
    ///
    /// [`HighsError::Run`] when HiGHS failed, or reports an optimum without a solution.
    fn outcome(&self, loaded: &Loaded, settings: &Settings, status: &HighsStatus) -> Result<Outcome, HighsError> {
        let failed = || HighsError::Run {
            status: status.model,
            message: status.last_error.clone(),
        };
        if status.run == CallStatus::Error {
            return Err(failed());
        }
        let best = (status.primal == SolutionStatus::Feasible).then(|| Solution {
            objective: self.instance.objective(),
            values: self.instance.primal_values(loaded.num_cols()).into(),
        });
        let bound = self
            .instance
            .info_double(c"mip_dual_bound")
            .filter(|b| loaded.is_mip && b.is_finite());
        let stopped = |reason| {
            Ok(Outcome::Stopped {
                reason,
                best: best.clone(),
                bound,
            })
        };
        match status.model {
            ModelStatus::Optimal => {
                let solution = best.clone().ok_or_else(failed)?;
                match bound {
                    Some(bound) if gap_open(solution.objective, bound) => Ok(Outcome::GapReached { solution, bound }),
                    _ => Ok(Outcome::Optimal(solution)),
                }
            }
            ModelStatus::ModelEmpty => Ok(Outcome::Optimal(Solution {
                objective: loaded.offset,
                values: Vec::new().into(),
            })),
            ModelStatus::Infeasible => Ok(Outcome::Infeasible),
            ModelStatus::Unbounded => Ok(Outcome::Unbounded),
            ModelStatus::UnboundedOrInfeasible => Ok(Outcome::InfeasibleOrUnbounded),
            ModelStatus::TimeLimit => stopped(StopReason::TimeLimit),
            ModelStatus::Interrupt => stopped(StopReason::Interrupted),
            ModelStatus::IterationLimit => stopped(StopReason::IterationLimit),
            ModelStatus::ObjectiveBound | ModelStatus::ObjectiveTarget => stopped(StopReason::ObjectiveLimit),
            ModelStatus::MemoryLimit => stopped(StopReason::MemoryLimit),
            ModelStatus::SolutionLimit => stopped(self.search_limit(settings, status.mip_nodes)),
            ModelStatus::Unknown | ModelStatus::HighsInterrupt => stopped(StopReason::Other),
            ModelStatus::NotSet
            | ModelStatus::LoadError
            | ModelStatus::ModelError
            | ModelStatus::PresolveError
            | ModelStatus::SolveError
            | ModelStatus::PostsolveError
            | ModelStatus::Other(_) => Err(failed()),
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
        lp: loaded,
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

    fn load(&mut self, lp: &Lp) -> Result<(), HighsError> {
        self.loaded = None;
        self.instance.pass(lp)?;
        self.loaded = Some(Loaded::of(lp));
        Ok(())
    }

    fn solve(
        &mut self,
        settings: &Settings,
        callback: &mut dyn Callback<Self>,
    ) -> Result<Solved<HighsStatus>, HighsError> {
        let loaded = self.loaded.as_ref().ok_or(HighsError::NotLoaded)?;
        configure(&mut self.instance, &self.options, settings)?;
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
