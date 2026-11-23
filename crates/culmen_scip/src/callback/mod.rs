// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! culmen's callback events delivered by SCIP plugins, each event with a typed context.
//!
//! # Which plugin delivers which event
//!
//! | SCIP 10 plugin | callback | culmen event | context |
//! |---|---|---|---|
//! | message handler | info and warning messages | [`Events::LOG`] | [`ScipLog`] |
//! | event handler | `PRESOLVEROUND`, `NODEFOCUSED`, `LPSOLVED`, `NODESOLVED` | [`Events::POLL`], [`Events::PROGRESS`] | [`ScipPoll`], [`ScipProgress`] |
//! | event handler | `POORSOLFOUND`, `BESTSOLFOUND` | [`Events::FEASIBLE`] | [`ScipSolution`] |
//! | event handler | `BESTSOLFOUND` | [`Events::INCUMBENT`] | [`ScipSolution`] |
//! | constraint handler | `CONSCHECK`, `CONSENFOLP`, `CONSENFOPS`, `CONSENFORELAX` | [`Events::LAZY`] | [`ScipLazy`] |
//! | separator | `SEPAEXECLP` | [`Events::CUT`] | [`ScipCut`] |
//! | primal heuristic | `HEUREXEC` | [`Events::HEURISTIC`] | [`ScipHeuristic`] |
//!
//! # The slot and the run, and why the plugins are sound
//!
//! Every plugin is included once, when the instance is created, with a pointer to the instance's [`Slot`] as its
//! data; the slot lives on the heap at one address until the instance is freed, after SCIP. A solve installs its
//! [`Run`], which lives on the solve's stack, in the slot for the call of `SCIPsolve` alone: a guard clears it when
//! `SCIPsolve` returns, on every path, before the run is dropped. A plugin called with no run installed, as SCIP calls
//! the constraint handler's lock and the event handler's exit while it frees a transformed problem, does nothing a
//! run would need.
//!
//! SCIP solves sequentially and calls every plugin on the thread running `SCIPsolve`, one call at a time. A plugin
//! may still be called inside another's call: a solution supplied from the heuristic callback is checked and stored
//! at once, which calls the constraint handler and the event handler, and any SCIP call may write a message. The
//! callback is borrowed through a `RefCell` for each delivery, so such a nested call finds it busy: a log line or a
//! solution is then kept and delivered as soon as the outer callback returns, a poll or a progress report is dropped,
//! and no plugin ever calls into the callback twice at once. Everything a context changes in SCIP (a lazy constraint,
//! a cut) is collected while the callback runs and handed to SCIP after it returns, outside any borrow.
//!
//! A panic in a callback is caught where the callback is called, never unwinds into SCIP, and asks SCIP to stop:
//! `SCIPinterruptSolve`, at once or, in the one stage SCIP refuses it, at the next plugin call. Every later delivery of
//! the run is skipped, each plugin returns a result its callback allows (the constraint handler rejects every
//! candidate and cuts off every node, the separator and the heuristic do not run), and the panic is resumed once
//! `SCIPsolve` has returned.

mod cut;
mod events;
mod heuristic;
mod lazy;
mod log;

use core::any::Any;
use core::cell::{Cell, RefCell};
use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use culmen_core::context::check_constraint;
use culmen_core::{ActionError, AddedRow, Callback, Constraint, Events, Found, LogLevel, Phase, Progress};
pub use cut::ScipCut;
pub use events::{ScipEvent, ScipPoll, ScipProgress, ScipSolution};
pub use heuristic::ScipHeuristic;
pub use lazy::{LazySource, ScipLazy};
pub use log::ScipLog;
use scip_sys::{
    SCIP, SCIP_RETCODE, SCIP_Retcode_SCIP_ERROR, SCIP_STAGE, SCIP_Stage_SCIP_STAGE_EXITPRESOLVE,
    SCIP_Stage_SCIP_STAGE_EXITSOLVE, SCIP_Stage_SCIP_STAGE_FREETRANS, SCIP_Stage_SCIP_STAGE_INITPRESOLVE,
    SCIP_Stage_SCIP_STAGE_PRESOLVED, SCIP_Stage_SCIP_STAGE_PRESOLVING, SCIP_Stage_SCIP_STAGE_PROBLEM,
    SCIP_Stage_SCIP_STAGE_SOLVED, SCIP_Stage_SCIP_STAGE_SOLVING, SCIP_Stage_SCIP_STAGE_TRANSFORMED, SCIPgetDualbound,
    SCIPgetNLPIterations, SCIPgetNOrigBinVars, SCIPgetNOrigIntVars, SCIPgetNTotalNodes, SCIPgetPrimalbound,
    SCIPgetStage, SCIPinfinity, SCIPinterruptSolve,
};

use crate::columns::Columns;
use crate::error::check;
use crate::{Scip, ScipError};

/// A caught panic's payload, resumed once `SCIPsolve` has returned.
pub(crate) type Payload = Box<dyn Any + Send>;

/// What every plugin of an instance reaches: one per instance, at one address from the instance's creation until
/// after SCIP is freed.
#[derive(Debug)]
pub(crate) struct Slot {
    /// The instance, which a message handler's call does not name.
    scip: *mut SCIP,
    /// The run of the solve in progress; null between solves.
    run: Cell<*const ()>,
    /// Whether SCIP's messages reach standard output and standard error.
    console: Cell<bool>,
    /// Whether the constraint handler locked every variable of the transformed problem, which its unlock undoes.
    locked: Cell<bool>,
    /// The events the event handler caught on the transformed problem and their filter position; `None` when it
    /// caught none.
    caught: Cell<Option<(u64, c_int)>>,
}

impl Slot {
    /// The slot of a new instance: no run, quiet, no locks, no events caught.
    pub(crate) const fn new(scip: *mut SCIP) -> Self {
        Self {
            scip,
            run: Cell::new(ptr::null()),
            console: Cell::new(false),
            locked: Cell::new(false),
            caught: Cell::new(None),
        }
    }

    /// Whether SCIP's messages reach standard output and standard error.
    pub(crate) fn set_console(&self, console: bool) {
        self.console.set(console);
    }

    /// Installs `run` until the guard returned drops.
    pub(crate) fn install<'s>(&'s self, run: &'s Run<'_>) -> Installed<'s> {
        debug_assert!(self.run.get().is_null(), "Slot::install: a run is installed already");
        self.run.set(ptr::from_ref(run).cast());
        Installed {
            slot: self,
            run: PhantomData,
        }
    }

    /// The run installed now, `None` between solves.
    ///
    /// # Safety
    ///
    /// Called from a plugin's callback, which SCIP calls only during the call of `SCIPsolve` the run is installed
    /// for, or with no run installed; the reference is used during that callback alone.
    unsafe fn run<'s>(&'s self) -> Option<&'s Run<'s>> {
        let run = self.run.get().cast::<Run<'s>>();
        // SAFETY: a non-null pointer is a run installed by `install`, which lives until its guard clears the pointer
        // after `SCIPsolve` returns, so for the whole callback the caller is in.
        unsafe { run.as_ref() }
    }

    /// Whether the run installed now asks for lazy constraints.
    ///
    /// # Safety
    ///
    /// As [`Slot::run`].
    unsafe fn lazy(&self) -> bool {
        // SAFETY: the caller's contract is `run`'s.
        unsafe { self.run() }.is_some_and(|run| run.wanted.contains(Events::LAZY))
    }

    /// Keeps `error` for the run installed now, if any, and returns the code a plugin returns to SCIP for it.
    ///
    /// # Safety
    ///
    /// As [`Slot::run`].
    unsafe fn fail(&self, error: ScipError) -> SCIP_RETCODE {
        // SAFETY: the caller's contract is `run`'s.
        match unsafe { self.run() } {
            Some(run) => run.fail(error),
            None => SCIP_Retcode_SCIP_ERROR,
        }
    }
}

/// The slot behind a plugin's data pointer, `None` for a null one.
///
/// # Safety
///
/// `data` is null or the slot pointer the plugin was included with, which lives as long as the instance.
unsafe fn slot<'s, T>(data: *mut T) -> Option<&'s Slot> {
    // SAFETY: the caller vouches that a non-null `data` is a live slot.
    unsafe { data.cast::<Slot>().as_ref() }
}

/// A run installed in a slot, until it drops.
#[derive(Debug)]
pub(crate) struct Installed<'s> {
    slot: &'s Slot,
    run: PhantomData<&'s Run<'s>>,
}

impl Drop for Installed<'_> {
    fn drop(&mut self) {
        self.slot.run.set(ptr::null());
    }
}

/// One solve's callback and what its plugins share, on the solve's stack while `SCIPsolve` runs.
pub(crate) struct Run<'a> {
    /// The callback, borrowed by one delivery at a time.
    callback: RefCell<&'a mut dyn Callback<Scip>>,
    /// The events the callback asked for that SCIP delivers.
    wanted: Events,
    /// When the solve began.
    start: Instant,
    cols: Columns<'a>,
    /// SCIP's infinity: a number of at least this magnitude is infinite to SCIP.
    infinity: f64,
    /// Whether the program has an integer column, which makes SCIP's search a branch and bound on a MIP.
    integral: bool,
    lazy: RefCell<lazy::Store>,
    /// What a nested call could not deliver while the callback was busy, in order.
    deferred: RefCell<Vec<Deferred>>,
    /// A panic a callback raised, held until `SCIPsolve` returns.
    panic: RefCell<Option<Payload>>,
    /// The first error a plugin met, surfaced after `SCIPsolve` returns.
    error: RefCell<Option<ScipError>>,
    /// Whether a stop was asked for that SCIP has not been told of, as it refuses one in its `INITSOLVE` stage.
    stop: Cell<bool>,
}

/// An event kept while the callback was busy.
enum Deferred {
    /// A log line.
    Log {
        text: String,
        level: LogLevel,
        elapsed: Duration,
    },
    /// A solution SCIP stored, and whether it is the new incumbent.
    Solution { found: Found, best: bool },
}

/// What came of handing an event to the callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Delivered {
    /// The callback ran and returned.
    Ran,
    /// The callback is running an outer event's method.
    Busy,
    /// A callback of the run panicked, now or before.
    Stopped,
}

impl<'a> Run<'a> {
    /// The run of a solve of `scip`, which holds the columns `cols`, for the events `wanted`.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance holding `cols`, in its problem stage.
    pub(crate) unsafe fn new(
        scip: *mut SCIP,
        callback: &'a mut dyn Callback<Scip>,
        wanted: Events,
        start: Instant,
        cols: Columns<'a>,
    ) -> Self {
        // SAFETY: the instance is live and in its problem stage, where SCIP counts the original variables.
        let (infinity, integers) = unsafe {
            (
                SCIPinfinity(scip),
                SCIPgetNOrigIntVars(scip) + SCIPgetNOrigBinVars(scip),
            )
        };
        Self {
            callback: RefCell::new(callback),
            wanted,
            start,
            cols,
            infinity,
            integral: integers > 0,
            lazy: RefCell::default(),
            deferred: RefCell::default(),
            panic: RefCell::default(),
            error: RefCell::default(),
            stop: Cell::new(false),
        }
    }

    /// The panic a callback raised and the first error a plugin met, once `SCIPsolve` has returned.
    pub(crate) fn finish(self) -> (Option<Payload>, Option<ScipError>) {
        (self.panic.into_inner(), self.error.into_inner())
    }

    /// Whether a callback of the run panicked.
    fn panicked(&self) -> bool {
        self.panic.borrow().is_some()
    }

    /// Whether the callback is running an outer event's method now.
    fn busy(&self) -> bool {
        self.callback.try_borrow_mut().is_err()
    }

    /// Keeps the first error a plugin met and returns the code the plugin returns to SCIP, which makes `SCIPsolve`
    /// fail.
    fn fail(&self, error: ScipError) -> SCIP_RETCODE {
        self.error.borrow_mut().get_or_insert(error);
        SCIP_Retcode_SCIP_ERROR
    }

    /// Hands the callback to `call`, unless a callback of the run panicked or the callback is busy; keeps a new panic
    /// and asks SCIP to stop. Once `call` returns, delivers what nested calls kept meanwhile.
    fn deliver(&self, scip: *mut SCIP, call: impl FnOnce(&mut dyn Callback<Scip>)) -> Delivered {
        if self.panicked() {
            return Delivered::Stopped;
        }
        let Ok(mut callback) = self.callback.try_borrow_mut() else {
            return Delivered::Busy;
        };
        let caught = catch_unwind(AssertUnwindSafe(|| call(&mut **callback)));
        drop(callback);
        if let Err(payload) = caught {
            *self.panic.borrow_mut() = Some(payload);
            self.request_stop(scip);
            return Delivered::Stopped;
        }
        self.flush(scip);
        Delivered::Ran
    }

    /// Delivers what nested calls kept while the callback was busy.
    fn flush(&self, scip: *mut SCIP) {
        let kept = core::mem::take(&mut *self.deferred.borrow_mut());
        for item in kept {
            match item {
                Deferred::Log { text, level, elapsed } => self.log_now(scip, &text, level, elapsed),
                Deferred::Solution { found, best } => self.solution_now(scip, &found, best),
            }
        }
    }

    /// Hands a log line to the callback, or keeps it while the callback is busy.
    fn log(&self, scip: *mut SCIP, text: &str, level: LogLevel) {
        let elapsed = self.start.elapsed();
        if self.busy() {
            self.deferred.borrow_mut().push(Deferred::Log {
                text: text.to_owned(),
                level,
                elapsed,
            });
        } else {
            self.log_now(scip, text, level, elapsed);
        }
    }

    fn log_now(&self, scip: *mut SCIP, text: &str, level: LogLevel, elapsed: Duration) {
        self.deliver(scip, |callback| {
            callback.log(&mut ScipLog::new(text, level, elapsed));
        });
    }

    /// Hands a solution SCIP stored to the callback as feasible and, when it is the new incumbent, as the incumbent;
    /// keeps it while the callback is busy.
    fn solution(&self, scip: *mut SCIP, found: Found, best: bool) {
        if self.busy() {
            self.deferred.borrow_mut().push(Deferred::Solution { found, best });
        } else {
            self.solution_now(scip, &found, best);
        }
    }

    fn solution_now(&self, scip: *mut SCIP, found: &Found, best: bool) {
        if self.wanted.contains(Events::FEASIBLE) {
            self.deliver(scip, |callback| callback.feasible(&mut ScipSolution::new(found)));
        }
        if best && self.wanted.contains(Events::INCUMBENT) {
            self.deliver(scip, |callback| callback.incumbent(&mut ScipSolution::new(found)));
        }
    }

    /// Asks SCIP to stop as soon as it can.
    fn request_stop(&self, scip: *mut SCIP) {
        self.stop.set(true);
        self.pass_stop(scip);
    }

    /// Tells SCIP of a stop asked for before, unless SCIP's stage refuses it now.
    fn pass_stop(&self, scip: *mut SCIP) {
        if !self.stop.get() {
            return;
        }
        // SAFETY: the instance is live; the call reads its stage.
        let stage = unsafe { SCIPgetStage(scip) };
        // `SCIPinterruptSolve` refuses only the init, initsolve and free stages.
        let open = (SCIP_Stage_SCIP_STAGE_PROBLEM..=SCIP_Stage_SCIP_STAGE_PRESOLVED).contains(&stage)
            || (SCIP_Stage_SCIP_STAGE_SOLVING..=SCIP_Stage_SCIP_STAGE_FREETRANS).contains(&stage);
        if open {
            // SAFETY: the instance is live and in a stage that takes the call, which sets a flag SCIP polls.
            let code = unsafe { SCIPinterruptSolve(scip) };
            if check("SCIPinterruptSolve", code).is_ok() {
                self.stop.set(false);
            }
        }
    }

    /// Where the solve stands: every figure SCIP reports in its stage now, objective values in the program's sense
    /// with the offset.
    fn progress(&self, scip: *mut SCIP) -> Progress {
        // SAFETY: the instance is live; the call reads its stage.
        let stage = unsafe { SCIPgetStage(scip) };
        let bounds = (SCIP_Stage_SCIP_STAGE_TRANSFORMED..=SCIP_Stage_SCIP_STAGE_EXITSOLVE).contains(&stage);
        let finite = |value: f64| (value.abs() < self.infinity).then_some(value);
        // SAFETY: SCIP reports its bounds from the transformed stage to the exitsolve stage, which `bounds` checks.
        let objective = bounds.then(|| finite(unsafe { SCIPgetPrimalbound(scip) })).flatten();
        // SAFETY: as above.
        let bound = bounds.then(|| finite(unsafe { SCIPgetDualbound(scip) })).flatten();
        let searching = (SCIP_Stage_SCIP_STAGE_SOLVING..=SCIP_Stage_SCIP_STAGE_EXITSOLVE).contains(&stage);
        // SAFETY: SCIP counts nodes from its problem stage on, so while it searches.
        let nodes = searching.then(|| count(unsafe { SCIPgetNTotalNodes(scip) }));
        let counted = [
            SCIP_Stage_SCIP_STAGE_PRESOLVING,
            SCIP_Stage_SCIP_STAGE_PRESOLVED,
            SCIP_Stage_SCIP_STAGE_SOLVING,
            SCIP_Stage_SCIP_STAGE_SOLVED,
        ];
        // SAFETY: SCIP counts LP iterations in the stages `counted` lists.
        let iterations = counted
            .contains(&stage)
            .then(|| count(unsafe { SCIPgetNLPIterations(scip) }));
        Progress {
            elapsed: self.start.elapsed(),
            phase: self.phase(stage),
            objective,
            bound,
            nodes,
            iterations,
        }
    }

    /// The phase of a stage: presolve, or a MIP's branch and bound.
    fn phase(&self, stage: SCIP_STAGE) -> Phase {
        if (SCIP_Stage_SCIP_STAGE_INITPRESOLVE..=SCIP_Stage_SCIP_STAGE_EXITPRESOLVE).contains(&stage) {
            Phase::Presolve
        } else if self.integral && stage == SCIP_Stage_SCIP_STAGE_SOLVING {
            Phase::Mip
        } else {
            Phase::Other
        }
    }
}

impl core::fmt::Debug for Run<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Run")
            .field("wanted", &self.wanted)
            .field("start", &self.start)
            .field("panicked", &self.panicked())
            .finish_non_exhaustive()
    }
}

/// A count SCIP reports, zero for a negative one.
fn count(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

/// Checks a constraint a callback adds as a row SCIP can hold: known columns, finite coefficients and valid bounds,
/// and every number below SCIP's infinity in magnitude, with no more terms than SCIP's `int` counts.
///
/// # Errors
///
/// The errors of [`check_constraint`]; [`ActionError::Refused`] for a number SCIP would read as infinite or a row
/// too long.
fn checked(constraint: &Constraint, num_cols: usize, infinity: f64) -> Result<AddedRow, ActionError> {
    let row = check_constraint(constraint, num_cols)?;
    if let Some(&(col, value)) = row.terms.iter().find(|(_, value)| value.abs() >= infinity) {
        return Err(ActionError::Refused(format!(
            "the coefficient {value} of {col:?} is one SCIP reads as infinite, from {infinity} on"
        )));
    }
    let ends = [row.bounds.lower(), row.bounds.upper()];
    if let Some(end) = ends.into_iter().find(|end| end.is_finite() && end.abs() >= infinity) {
        return Err(ActionError::Refused(format!(
            "the bound {end} is one SCIP reads as infinite, from {infinity} on"
        )));
    }
    if c_int::try_from(row.terms.len()).is_err() {
        return Err(ActionError::Refused(format!(
            "{} terms are more than SCIP's `int` counts",
            row.terms.len()
        )));
    }
    Ok(row)
}

/// A row's bounds in SCIP's terms, an infinite end as SCIP's infinity.
fn sides(row: &AddedRow, infinity: f64) -> (f64, f64) {
    let side = |end: f64| {
        if end.is_infinite() {
            end.signum() * infinity
        } else {
            end
        }
    };
    (side(row.bounds.lower()), side(row.bounds.upper()))
}

/// Includes the message handler, before SCIP's default plugins so that it carries their messages too.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(crate) unsafe fn install_messages(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    // SAFETY: the caller's contract is `install`'s.
    unsafe { log::install(scip, slot) }
}

/// Includes the constraint handler, the separator, the heuristic and the event handler.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(crate) unsafe fn include(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    // SAFETY: the caller's contract is each `include`'s.
    unsafe {
        lazy::include(scip, slot)?;
        cut::include(scip, slot)?;
        heuristic::include(scip, slot)?;
        events::include(scip, slot)
    }
}
