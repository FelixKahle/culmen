// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The SCIP backend of culmen: SCIP 10, built from source through `scip-sys`, with typed statuses and options.
//!
//! [`Scip`] implements [`Backend`]: load a [`Program`], solve it under [`Settings`], and read the generic
//! [`Outcome`] with SCIP's own [`ScipStatus`] beside it. It solves LPs, MILPs, convex QPs and MIQPs, and takes a
//! start. SCIP's callbacks, which are plugins registered on the instance, are not delivered yet, so
//! [`Scip::EVENTS`] is empty and a callback passed to a solve is never called.
//!
//! ```
//! use culmen_core::{Linear, LinExpr, Model, Outcome, Request, Solver, Var};
//! use culmen_scip::Scip;
//!
//! // A knapsack.
//! let (value, weight) = ([10.0, 13.0, 7.0, 8.0], [5.0, 6.0, 4.0, 3.0]);
//! let mut m = Model::new();
//! let take = m.add_vars(4, |i| Var::binary().name(format!("take[{i}]")));
//! m.add_constraint(LinExpr::sum((0..4).map(|i| weight[i] * take[i])).leq(10.0));
//! m.maximize(LinExpr::sum((0..4).map(|i| value[i] * take[i])));
//!
//! let mut solver = Solver::new(Scip::new(), m.build()?)?;
//! let solved = solver.solve(Request::new())?;
//! let Outcome::Optimal(best) = solved.outcome else { panic!("expected an optimum, got {:?}", solved.outcome) };
//! assert_eq!(best.objective, 21.0);
//! assert_eq!(best.values[take[1]].round(), 1.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Options and settings
//!
//! Each solve starts from the program as loaded: it frees the last solve's transformed problem, resets SCIP's
//! parameters to their defaults, keeps SCIP from catching `SIGINT` (`misc/catchctrlc`), then applies the backend's
//! [`ScipOptions`] and then the solve's [`Settings`], so a per-solve setting always wins. SCIP's log stays off
//! standard output unless [`Settings::console`] asks for it. Every field of [`Settings`] maps to SCIP:
//!
//! | `Settings` field | SCIP |
//! |---|---|
//! | `time_limit` | `limits/time`, in seconds, from the start of the solve while `misc/resetstat` keeps its default |
//! | `relative_gap` | `limits/gap` |
//! | `absolute_gap` | `limits/absgap` |
//! | `threads` | nothing to set: the solve is sequential, so it uses one thread, within any limit |
//! | `seed` | `randomization/randomseedshift`, at most `i32::MAX` |
//! | `presolve` | `SCIPsetPresolving`: off, or the presolving emphasis of the options when it is not off, else default |
//! | `node_limit` | `limits/nodes`, at most `i64::MAX` |
//! | `solution_limit` | `limits/solutions`, at most `i32::MAX` |
//! | `console` | SCIP's message handler quiet or not |
//!
//! A value beyond what SCIP's parameter takes is refused with [`ScipError::OutOfRange`] before any parameter is set.
//! SCIP prints the error messages of a call it refuses, such as a parameter it does not know, to standard error
//! through its process-wide error printer, which the backend leaves as it is.
//!
//! # Threads
//!
//! [`Scip`] is `Send`, so it can be moved to a worker thread to solve there. It is not `Sync`: SCIP is built
//! thread-safe in the sense that each instance keeps its state to itself, but one instance promises nothing to two
//! threads calling it at once, not even to read it.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

mod error;
mod instance;
mod options;
mod status;

use core::fmt;
use std::time::Instant;

use culmen_core::{
    Backend, Callback, Events, ModelClasses, Outcome, Program, ProgramFeature, ProgramFeatures, Settings, Solved,
    Start, Unsupported,
};
pub use error::{Number, RetCode, ScipError};
pub use options::{Emphasis, ParamValue, ScipOptions, ScipParam};
pub use status::{ScipStatus, Status};

use instance::{Group, Instance};

/// The SCIP backend: the options every solve applies and, once a program is loaded, the SCIP instance holding it.
///
/// ```
/// use culmen_core::{Backend, Events, ModelClass};
/// use culmen_scip::{Emphasis, Scip, ScipOptions};
///
/// let scip = Scip::with_options(ScipOptions { heuristics: Some(Emphasis::Off), ..ScipOptions::default() });
/// assert_eq!(scip.options().heuristics, Some(Emphasis::Off));
/// assert!(Scip::CLASSES.contains(ModelClass::Miqp));
/// assert_eq!(Scip::EVENTS, Events::NONE);
/// ```
#[derive(Debug, Default)]
pub struct Scip {
    options: ScipOptions,
    /// The instance holding the loaded program; `None` before a load and after a failed one.
    instance: Option<Instance>,
}

impl Scip {
    /// A backend with SCIP's default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(ScipOptions::default())
    }

    /// A backend with the given options.
    #[must_use]
    pub fn with_options(options: impl Into<ScipOptions>) -> Self {
        Self {
            options: options.into(),
            instance: None,
        }
    }

    /// The options every solve applies.
    #[must_use]
    pub const fn options(&self) -> &ScipOptions {
        &self.options
    }

    /// The options every solve applies, to change between solves.
    pub const fn options_mut(&mut self) -> &mut ScipOptions {
        &mut self.options
    }
}

/// Refuses a program of a class or with a feature this backend does not declare, which a
/// [`Solver`](culmen_core::Solver) never hands it.
///
/// # Errors
///
/// [`ScipError::Unsupported`] for the class, else [`ScipError::UnsupportedFeature`] for the first feature.
fn refuse_unsupported(program: &Program) -> Result<(), ScipError> {
    let class = program.class();
    if !Scip::CLASSES.contains(class) {
        return Err(ScipError::Unsupported { class });
    }
    match program.features().difference(Scip::FEATURES).iter().next() {
        Some(feature) => Err(ScipError::UnsupportedFeature { feature }),
        None => Ok(()),
    }
}

/// Resets SCIP's parameters, keeps SCIP from catching `SIGINT`, and applies `options` and then the solve's settings.
///
/// # Errors
///
/// [`ScipError::OutOfRange`] for the first setting beyond SCIP's parameter, before any parameter is set;
/// [`ScipError::Call`] when SCIP fails to reset or refuses an emphasis; [`ScipError::Param`] and the other parameter
/// errors for the first parameter refused.
fn configure(instance: &mut Instance, options: &ScipOptions, settings: &Settings) -> Result<(), ScipError> {
    let params = settings_params(settings)?;
    instance.reset_params()?;
    instance.set_param("misc/catchctrlc", &ParamValue::Bool(false))?;
    let emphases = [
        (Group::Heuristics, options.heuristics),
        (Group::Presolving, options.presolving),
        (Group::Separating, options.separating),
    ];
    for (group, emphasis) in emphases {
        if let Some(emphasis) = emphasis {
            instance.set_emphasis(group, emphasis)?;
        }
    }
    for ScipParam { name, value } in &options.params {
        instance.set_param(name, value)?;
    }
    if let Some(presolve) = settings.presolve {
        instance.set_emphasis(Group::Presolving, presolving(options.presolving, presolve))?;
    }
    for (name, value) in params {
        instance.set_param(name, &value)?;
    }
    instance.set_console(settings.console);
    Ok(())
}

/// The presolving emphasis of a solve whose settings ask to presolve or not: off, or the options' emphasis when it
/// presolves, else SCIP's default.
fn presolving(options: Option<Emphasis>, presolve: bool) -> Emphasis {
    match options {
        _ if !presolve => Emphasis::Off,
        Some(emphasis @ (Emphasis::Aggressive | Emphasis::Fast)) => emphasis,
        Some(Emphasis::Default | Emphasis::Off) | None => Emphasis::Default,
    }
}

/// The SCIP parameters a solve's settings set, beside its presolving and console.
///
/// # Errors
///
/// [`ScipError::OutOfRange`] when the seed, the node limit or the solution limit exceeds SCIP's parameter.
fn settings_params(settings: &Settings) -> Result<Vec<(&'static str, ParamValue)>, ScipError> {
    let mut set = Vec::new();
    if let Some(limit) = settings.time_limit {
        set.push(("limits/time", ParamValue::Real(limit.as_secs_f64())));
    }
    if let Some(gap) = settings.relative_gap {
        set.push(("limits/gap", ParamValue::Real(gap.get())));
    }
    if let Some(gap) = settings.absolute_gap {
        set.push(("limits/absgap", ParamValue::Real(gap.get())));
    }
    if let Some(seed) = settings.seed {
        let seed = int("randomization/randomseedshift", seed.into())?;
        set.push(("randomization/randomseedshift", ParamValue::Int(seed)));
    }
    if let Some(nodes) = settings.node_limit {
        set.push(("limits/nodes", ParamValue::Longint(longint("limits/nodes", nodes)?)));
    }
    if let Some(solutions) = settings.solution_limit {
        set.push((
            "limits/solutions",
            ParamValue::Int(int("limits/solutions", solutions.get())?),
        ));
    }
    Ok(set)
}

/// An exact integer setting as SCIP's `int` parameter `name`.
///
/// # Errors
///
/// [`ScipError::OutOfRange`] when `value` exceeds `i32::MAX`.
fn int(name: &'static str, value: u64) -> Result<i32, ScipError> {
    i32::try_from(value).map_err(|_| ScipError::OutOfRange {
        name,
        value,
        max: i32::MAX.unsigned_abs().into(),
    })
}

/// An exact integer setting as SCIP's `SCIP_Longint` parameter `name`.
///
/// # Errors
///
/// [`ScipError::OutOfRange`] when `value` exceeds `i64::MAX`.
fn longint(name: &'static str, value: u64) -> Result<i64, ScipError> {
    i64::try_from(value).map_err(|_| ScipError::OutOfRange {
        name,
        value,
        max: i64::MAX.unsigned_abs(),
    })
}

impl Backend for Scip {
    type Error = ScipError;
    type Status = ScipStatus;
    type Log<'a> = Unsupported;
    type Progress<'a> = Unsupported;
    type Poll<'a> = Unsupported;
    type Solution<'a> = Unsupported;
    type Lazy<'a> = Unsupported;
    type Cut<'a> = Unsupported;
    type Heuristic<'a> = Unsupported;
    type Native<'a> = Unsupported;

    /// None yet: each event needs a SCIP plugin registered on the instance (a message handler, event handlers, a
    /// constraint handler, a separator, a heuristic), which this version does not register.
    const EVENTS: Events = Events::NONE;

    /// Every class: a quadratic objective `q(x)` becomes an epigraph variable `t` with cost one, bounded by the
    /// quadratic constraint `q(x) - t <= 0` and `t >= 0` when minimizing, `q(x) - t >= 0` and `t <= 0` when
    /// maximizing, where a convex `q` is never negative and a concave one never positive. SCIP solves the constraint
    /// as a convex nonlinear constraint by cutting planes. A solution's objective takes `q(x)` itself in the place of
    /// `t`, which meets it only to SCIP's feasibility tolerance; so an optimum's objective is close to the optimal
    /// value, and its point, where the objective is flat, less close to the optimal point.
    const CLASSES: ModelClasses = ModelClasses::ALL;

    /// Starts: a full start as a SCIP solution, a sparse one as a partial solution SCIP completes.
    const FEATURES: ProgramFeatures = ProgramFeatures::of(&[ProgramFeature::StartSolution]);

    /// Builds the program into a new SCIP instance, one variable per column and one linear constraint per row (a
    /// ranged row has both sides, a free row neither), and a quadratic objective's epigraph, replacing the instance
    /// before.
    ///
    /// # Errors
    ///
    /// [`ScipError::Unsupported`] or [`ScipError::UnsupportedFeature`] for a program this backend does not declare;
    /// [`ScipError::BeyondInfinity`] for a finite number SCIP would read as infinite; [`ScipError::ColName`] or
    /// [`ScipError::RowName`] for a name holding a NUL byte; [`ScipError::Call`] when SCIP refuses a call. The
    /// backend then holds no program.
    fn load(&mut self, program: &Program) -> Result<(), ScipError> {
        self.instance = None;
        refuse_unsupported(program)?;
        self.instance = Some(Instance::load(program)?);
        Ok(())
    }

    /// Hands SCIP the start for the next solve, which SCIP checks, and completes when it is sparse, as that solve
    /// starts. An infeasible start, or a sparse one SCIP cannot complete, is dropped then, as SCIP drops any.
    ///
    /// # Errors
    ///
    /// [`ScipError::NotLoaded`] before a load, [`ScipError::Start`] for a start that does not fit the program,
    /// [`ScipError::Call`] when SCIP refuses a call.
    fn set_start(&mut self, start: &Start) -> Result<(), ScipError> {
        let instance = self.instance.as_mut().ok_or(ScipError::NotLoaded)?;
        start.check(instance.num_cols()).map_err(|_| ScipError::Start)?;
        instance.add_start(start)
    }

    /// Solves the loaded program from scratch under the backend's options and `settings`.
    ///
    /// `callback` is never called, as [`Scip::EVENTS`] is empty.
    ///
    /// # Errors
    ///
    /// [`ScipError::NotLoaded`] before a load; the errors of the options and settings the module documentation lists;
    /// [`ScipError::Call`] when SCIP fails; [`ScipError::Failed`] when SCIP ends with a status that proves nothing.
    fn solve(
        &mut self,
        settings: &Settings,
        callback: &mut dyn Callback<Self>,
    ) -> Result<Solved<ScipStatus>, ScipError> {
        // No event is delivered, so the callback is never called; the plugins of a later version reach it here.
        let _ = callback;
        let instance = self.instance.as_mut().ok_or(ScipError::NotLoaded)?;
        configure(instance, &self.options, settings)?;
        let start = Instant::now();
        instance.solve()?;
        let elapsed = start.elapsed();
        let status = instance.status();
        let outcome: Outcome = status::outcome(&status, instance.best())?;
        Ok(Solved {
            outcome,
            status,
            elapsed,
        })
    }
}

/// A SCIP version, ordered as versions are; displayed as `major.minor.patch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Version {
    /// The major version.
    pub major: u32,
    /// The minor version.
    pub minor: u32,
    /// The patch version, SCIP's technical version.
    pub patch: u32,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The version of the SCIP library linked in.
///
/// ```
/// let version = culmen_scip::version();
/// assert_eq!(version.major, 10);
/// assert_eq!(version.to_string(), format!("10.{}.{}", version.minor, version.patch));
/// ```
#[must_use]
pub fn version() -> Version {
    // SAFETY: the version getters take no arguments and return plain integers.
    let (major, minor, patch) = unsafe {
        (
            scip_sys::SCIPmajorVersion(),
            scip_sys::SCIPminorVersion(),
            scip_sys::SCIPtechVersion(),
        )
    };
    // SCIP's version numbers are never negative, so their magnitudes are the numbers.
    Version {
        major: major.unsigned_abs(),
        minor: minor.unsigned_abs(),
        patch: patch.unsigned_abs(),
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU64;
    use core::time::Duration;

    use culmen_core::{absolute_gap, relative_gap, threads};

    use super::*;

    #[test]
    fn test_settings_set_scips_parameters_by_name_and_a_thread_count_sets_none() {
        let settings = Settings {
            time_limit: Some(Duration::from_millis(1500)),
            relative_gap: Some(relative_gap!(0.01)),
            absolute_gap: Some(absolute_gap!(0.5)),
            threads: Some(threads!(4)),
            seed: Some(7),
            presolve: Some(false),
            node_limit: Some(100),
            solution_limit: NonZeroU64::new(3),
            console: true,
        };
        assert_eq!(
            settings_params(&settings),
            Ok(vec![
                ("limits/time", ParamValue::Real(1.5)),
                ("limits/gap", ParamValue::Real(0.01)),
                ("limits/absgap", ParamValue::Real(0.5)),
                ("randomization/randomseedshift", ParamValue::Int(7)),
                ("limits/nodes", ParamValue::Longint(100)),
                ("limits/solutions", ParamValue::Int(3)),
            ])
        );
        assert_eq!(settings_params(&Settings::new()), Ok(vec![]));
    }

    #[test]
    fn test_presolving_off_wins_and_on_keeps_an_emphasis_that_presolves() {
        for options in [
            None,
            Some(Emphasis::Default),
            Some(Emphasis::Aggressive),
            Some(Emphasis::Fast),
            Some(Emphasis::Off),
        ] {
            assert_eq!(presolving(options, false), Emphasis::Off, "{options:?}");
        }
        assert_eq!(presolving(None, true), Emphasis::Default);
        assert_eq!(presolving(Some(Emphasis::Off), true), Emphasis::Default);
        assert_eq!(presolving(Some(Emphasis::Default), true), Emphasis::Default);
        assert_eq!(presolving(Some(Emphasis::Aggressive), true), Emphasis::Aggressive);
        assert_eq!(presolving(Some(Emphasis::Fast), true), Emphasis::Fast);
    }
}
