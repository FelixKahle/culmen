// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! culmen's callback events and search handlers delivered by SCIP plugins, each with a typed context.
//!
//! # Which plugin delivers which event
//!
//! | SCIP 10 plugin | callback | culmen event or handler | context |
//! |---|---|---|---|
//! | message handler | info and warning messages | [`Event::Log`] | [`ScipLog`] |
//! | event handler | `PRESOLVEROUND`, `NODEFOCUSED`, `LPSOLVED`, `NODESOLVED` | [`Event::Poll`], [`Event::Progress`] | [`ScipPoll`], [`ScipProgress`] |
//! | event handler | `POORSOLFOUND`, `BESTSOLFOUND` | [`Event::Feasible`] | [`ScipSolution`] |
//! | event handler | `BESTSOLFOUND` | [`Event::Incumbent`] | [`ScipSolution`] |
//! | constraint handler | `CONSCHECK`, `CONSENFOLP`, `CONSENFOPS`, `CONSENFORELAX` | [`LazyHandler`](culmen_core::LazyHandler) | [`ScipLazy`] |
//! | separator | `SEPAEXECLP` | [`CutHandler`](culmen_core::CutHandler) | [`ScipCut`] |
//! | primal heuristic | `HEUREXEC` | [`HeuristicHandler`](culmen_core::HeuristicHandler) | [`ScipHeuristic`] |
//! | variable pricer | `PRICERREDCOST`, `PRICERFARKAS` | [`Pricer`](culmen_core::Pricer) | [`ScipPrice`] |
//!
//! A handler's plugin acts only in a solve whose [`Hooks`] hold a handler of its kind; the run reads which once, as it
//! is made, from [`Hooks::wants_lazy`] and its siblings.
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
//! may still be called inside another's call: a solution supplied from the heuristic handler is checked and stored
//! at once, which calls the constraint handler and the event handler, and any SCIP call may write a message. The
//! hooks, which hold the callback and every handler, are borrowed through a `RefCell` for each delivery, so such a
//! nested call finds them busy: a log line or a solution is then kept and delivered as soon as the outer call returns,
//! a poll or a progress report is dropped, and no plugin ever calls into the hooks twice at once. Everything a context
//! changes in SCIP (a lazy constraint, a cut, a priced column) is collected while the handler runs and handed to SCIP
//! after it returns, outside any borrow.
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
mod locks;
mod log;
mod price;

use core::any::Any;
use core::cell::{Cell, RefCell};
use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::time::Duration;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use culmen_core::backend::check_constraint;
use culmen_core::backend::{AddedRow, Found, Hooks, Priced};
use culmen_core::{ActionError, Col, Constraint, Event, Events, LogLevel, Phase, Progress, Row, Sense, TiSlice, TiVec};
pub use cut::ScipCut;
pub use events::{ScipEvent, ScipPoll, ScipProgress, ScipSolution};
pub use heuristic::ScipHeuristic;
pub use lazy::{ScipLazy, ScipLazySource};
pub use log::ScipLog;
pub(crate) use price::NAME as PRICER;
pub use price::ScipPrice;
use scip_sys::{
    SCIP, SCIP_CONS, SCIP_Objsense_SCIP_OBJSENSE_MAXIMIZE, SCIP_RETCODE, SCIP_Retcode_SCIP_ERROR,
    SCIP_Retcode_SCIP_OKAY, SCIP_SOL, SCIP_STAGE, SCIP_Stage_SCIP_STAGE_EXITPRESOLVE, SCIP_Stage_SCIP_STAGE_EXITSOLVE,
    SCIP_Stage_SCIP_STAGE_FREETRANS, SCIP_Stage_SCIP_STAGE_INITPRESOLVE, SCIP_Stage_SCIP_STAGE_PRESOLVED,
    SCIP_Stage_SCIP_STAGE_PRESOLVING, SCIP_Stage_SCIP_STAGE_PROBLEM, SCIP_Stage_SCIP_STAGE_SOLVED,
    SCIP_Stage_SCIP_STAGE_SOLVING, SCIP_Stage_SCIP_STAGE_TRANSFORMED, SCIP_VAR, SCIP_Varstatus_SCIP_VARSTATUS_MULTAGGR,
    SCIPgetDualbound, SCIPgetNLPIterations, SCIPgetNOrigBinVars, SCIPgetNOrigIntVars, SCIPgetNTotalNodes, SCIPgetNVars,
    SCIPgetObjsense, SCIPgetPrimalbound, SCIPgetProbvarSum, SCIPgetStage, SCIPinfinity, SCIPinterruptSolve,
    SCIPvarGetMultaggrConstant, SCIPvarGetMultaggrNVars, SCIPvarGetMultaggrScalars, SCIPvarGetMultaggrVars,
    SCIPvarGetStatus,
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
    /// The locks the lazy-constraint handler applied to the transformed problem, which its unlock undoes.
    lazy_locks: locks::Applied,
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
            lazy_locks: locks::Applied::new(),
            caught: Cell::new(None),
        }
    }

    /// The locks the lazy-constraint handler keeps on `var`, down and up.
    #[cfg(test)]
    pub(crate) fn lazy_locks_on(&self, var: NonNull<SCIP_VAR>) -> (c_int, c_int) {
        self.lazy_locks.on(var)
    }

    /// Whether SCIP's messages reach standard output and standard error.
    pub(crate) fn set_console(&self, console: bool) {
        self.console.set(console);
    }

    /// Installs `run` until the guard returned drops.
    pub(crate) fn install<'s>(&'s self, run: &'s Run<'_, '_>) -> Installed<'s> {
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
    unsafe fn run<'s>(&'s self) -> Option<&'s Run<'s, 's>> {
        let run = self.run.get().cast::<Run<'s, 's>>();
        // SAFETY: a non-null pointer is a run installed by `install`, which lives until its guard clears the pointer
        // after `SCIPsolve` returns, so for the whole callback the caller is in.
        unsafe { run.as_ref() }
    }

    /// Whether the run installed now has a lazy-constraint handler.
    ///
    /// # Safety
    ///
    /// As [`Slot::run`].
    unsafe fn lazy(&self) -> bool {
        // SAFETY: the caller's contract is `run`'s.
        unsafe { self.run() }.is_some_and(|run| run.wants_lazy)
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
    run: PhantomData<&'s Run<'s, 's>>,
}

impl Drop for Installed<'_> {
    fn drop(&mut self) {
        self.slot.run.set(ptr::null());
    }
}

/// One solve's hooks and what its plugins share, on the solve's stack while `SCIPsolve` runs: what the run reads of
/// the instance lives for `'a`, the hooks for `'h`.
pub(crate) struct Run<'a, 'h> {
    /// The instance's slot, which keeps the locks the lazy-constraint handler applies beyond the run.
    slot: &'a Slot,
    /// The callback and the handlers, borrowed by one delivery at a time.
    hooks: RefCell<Hooks<'h, Scip>>,
    /// The events the callback asked for that SCIP delivers.
    wanted: Events,
    /// Whether the hooks hold a lazy-constraint handler.
    wants_lazy: bool,
    /// Whether the hooks hold a cut handler.
    wants_cut: bool,
    /// Whether the hooks hold a heuristic handler.
    wants_heuristic: bool,
    /// Whether the hooks hold a pricer, and so `priced` its sink and the pricer is active.
    pricing: bool,
    /// When the solve began.
    start: Instant,
    cols: Columns<'a>,
    /// The original constraint of every program row.
    rows: &'a TiSlice<Row, NonNull<SCIP_CONS>>,
    /// Whether the program minimizes or maximizes.
    sense: Sense,
    /// The columns the solve priced: checked and numbered here, then created in SCIP as `priced_vars`; `None` for a
    /// solve without a pricer.
    priced: RefCell<Option<&'h mut Priced>>,
    /// The variable of every column priced so far, in the order of their ids; the run owns one reference to each.
    priced_vars: RefCell<Vec<NonNull<SCIP_VAR>>>,
    /// SCIP's infinity: a number of at least this magnitude is infinite to SCIP.
    infinity: f64,
    /// Whether the program, or a column priced so far, has an integer column, which makes SCIP's search a branch and
    /// bound on a MIP.
    integral: Cell<bool>,
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

/// What a run leaves once `SCIPsolve` has returned.
#[derive(Debug)]
pub(crate) struct Finished {
    /// A panic a callback raised, to resume.
    pub(crate) panic: Option<Payload>,
    /// The first error a plugin met.
    pub(crate) error: Option<ScipError>,
    /// The variable of every column the run priced, in the order of their ids, one reference to each.
    pub(crate) priced: Vec<NonNull<SCIP_VAR>>,
}

/// An event kept while the callback was busy.
#[derive(Debug)]
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

impl<'a, 'h> Run<'a, 'h> {
    /// The run of a solve of `scip`, whose slot is `slot` and which holds the columns `cols` and the rows `rows`, for
    /// `hooks`, handing every column its pricer prices to `priced`, the sink the hooks gave.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance holding `cols` and `rows`, in its problem stage.
    pub(crate) unsafe fn new(
        scip: *mut SCIP,
        slot: &'a Slot,
        hooks: Hooks<'h, Scip>,
        priced: Option<&'h mut Priced>,
        start: Instant,
        cols: Columns<'a>,
        rows: &'a TiSlice<Row, NonNull<SCIP_CONS>>,
    ) -> Self {
        // SAFETY: the instance is live and in its problem stage, where SCIP counts the original variables.
        let (infinity, integers, objsense) = unsafe {
            (
                SCIPinfinity(scip),
                SCIPgetNOrigIntVars(scip) + SCIPgetNOrigBinVars(scip),
                SCIPgetObjsense(scip),
            )
        };
        let sense = if objsense == SCIP_Objsense_SCIP_OBJSENSE_MAXIMIZE {
            Sense::Maximize
        } else {
            Sense::Minimize
        };
        Self {
            slot,
            wanted: hooks.events(),
            wants_lazy: hooks.wants_lazy(),
            wants_cut: hooks.wants_cut(),
            wants_heuristic: hooks.wants_heuristic(),
            pricing: priced.is_some(),
            hooks: RefCell::new(hooks),
            start,
            cols,
            rows,
            sense,
            priced: RefCell::new(priced),
            priced_vars: RefCell::default(),
            infinity,
            integral: Cell::new(integers > 0),
            lazy: RefCell::default(),
            deferred: RefCell::default(),
            panic: RefCell::default(),
            error: RefCell::default(),
            stop: Cell::new(false),
        }
    }

    /// The panic a callback raised, the first error a plugin met, and the variables of the columns the run priced,
    /// whose references the caller then owns, once `SCIPsolve` has returned.
    pub(crate) fn finish(self) -> Finished {
        Finished {
            panic: self.panic.into_inner(),
            error: self.error.into_inner(),
            priced: self.priced_vars.into_inner(),
        }
    }

    /// The objective of `sol` and its value of every column the problem holds now, as [`Columns::read`] reads them.
    ///
    /// # Safety
    ///
    /// As [`Columns::read`], for the instance of the run.
    unsafe fn read(&self, scip: *mut SCIP, sol: *mut SCIP_SOL) -> (f64, TiVec<Col, f64>) {
        // SAFETY: the caller's contract.
        unsafe { self.cols.read(scip, sol, &self.priced_vars.borrow()) }
    }

    /// The value of every column the problem holds now in `sol`, as [`Columns::read`] reads it.
    ///
    /// # Safety
    ///
    /// As [`Columns::read`], for the instance of the run.
    unsafe fn values(&self, scip: *mut SCIP, sol: *mut SCIP_SOL) -> TiVec<Col, f64> {
        // SAFETY: the caller's contract.
        unsafe { self.cols.values(scip, sol, &self.priced_vars.borrow()) }
    }

    /// Whether a callback of the run panicked.
    fn panicked(&self) -> bool {
        self.panic.borrow().is_some()
    }

    /// Whether a callback or handler is running an outer event's method now.
    fn busy(&self) -> bool {
        self.hooks.try_borrow_mut().is_err()
    }

    /// Keeps the first error a plugin met and returns the code the plugin returns to SCIP, which makes `SCIPsolve`
    /// fail.
    fn fail(&self, error: ScipError) -> SCIP_RETCODE {
        self.error.borrow_mut().get_or_insert(error);
        SCIP_Retcode_SCIP_ERROR
    }

    /// Keeps the first error a plugin met and asks SCIP to stop, for a callback that must not return an error code:
    /// SCIP leaves a constraint handler's enforcement on an error code without undoing its delay of the handler's
    /// constraint updates (`cons.c:3525-3531` of SCIP 10.0.2), which outlives the solve and aborts a debug SCIP at the
    /// next one (`cons.c:2524`), leaves a pricer's columns in its pricing store (`solve.c:2323`), which aborts a
    /// debug SCIP as it frees the store (`pricestore.c:142`), and leaves a separator's cuts in its separation store
    /// (`solve.c:1794`, `solve.c:1895`), which aborts a debug SCIP as it frees that store (`sepastore.c:125`). The solve fails with the error once `SCIPsolve` has
    /// returned, whatever SCIP's status.
    fn stop_with(&self, scip: *mut SCIP, error: ScipError) -> SCIP_RETCODE {
        self.error.borrow_mut().get_or_insert(error);
        self.request_stop(scip);
        SCIP_Retcode_SCIP_OKAY
    }

    /// Whether the run is ending: a callback panicked or a plugin met an error.
    fn stopping(&self) -> bool {
        self.panicked() || self.error.borrow().is_some()
    }

    /// Hands the hooks to `call`, unless a callback or handler of the run panicked or the hooks are busy; keeps a new
    /// panic and asks SCIP to stop. Once `call` returns, delivers what nested calls kept meanwhile.
    fn deliver(&self, scip: *mut SCIP, call: impl FnOnce(&mut Hooks<'h, Scip>)) -> Delivered {
        if self.panicked() {
            return Delivered::Stopped;
        }
        let Ok(mut hooks) = self.hooks.try_borrow_mut() else {
            return Delivered::Busy;
        };
        let caught = catch_unwind(AssertUnwindSafe(|| call(&mut hooks)));
        drop(hooks);
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
        self.deliver(scip, |hooks| {
            hooks.log(&mut ScipLog::new(text, level, elapsed));
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
        if self.wanted.contains(Event::Feasible) {
            self.deliver(scip, |hooks| hooks.feasible(&mut ScipSolution::new(found)));
        }
        if best && self.wanted.contains(Event::Incumbent) {
            self.deliver(scip, |hooks| hooks.incumbent(&mut ScipSolution::new(found)));
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
        } else if self.integral.get() && stage == SCIP_Stage_SCIP_STAGE_SOLVING {
            Phase::Mip
        } else {
            Phase::Other
        }
    }
}

impl core::fmt::Debug for Run<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Run")
            .field("wanted", &self.wanted)
            .field("wants_lazy", &self.wants_lazy)
            .field("wants_cut", &self.wants_cut)
            .field("wants_heuristic", &self.wants_heuristic)
            .field("pricing", &self.pricing)
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
/// and every number below SCIP's infinity in magnitude, as given and added up per column, with no more terms than
/// SCIP's `int` counts.
///
/// # Errors
///
/// The errors of [`check_constraint`]; [`ActionError::Refused`] for a number SCIP would read as infinite or a row
/// too long.
fn checked(constraint: &Constraint, num_cols: usize, infinity: f64) -> Result<AddedRow, ActionError> {
    let row = check_constraint(constraint, num_cols)?;
    // SCIP's linear handler adds up the coefficients of a column given twice, so their sum must stay finite to it.
    let mut sums: BTreeMap<Col, f64> = BTreeMap::new();
    for &(col, value) in &row.terms {
        *sums.entry(col).or_insert(0.0) += value;
    }
    let given = row.terms.iter().copied().find(|(_, value)| value.abs() >= infinity);
    if let Some((col, value)) = given.or_else(|| sums.into_iter().find(|(_, value)| value.abs() >= infinity)) {
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

/// Checks that SCIP can hold `row` as a linear constraint over the variables it keeps now: SCIP replaces each fixed,
/// aggregated, negated or multi-aggregated variable of a constraint it creates after presolve by its active
/// representation (`SCIPgetProbvarLinearSum` in `SCIPcreateConsLinear`, `cons_linear.c` of SCIP 10.0.2), moving the
/// constant that adds up into the sides, and refuses a constant of at least its infinity in magnitude
/// (`SCIP_INVALIDDATA`); a debug SCIP aborts on it before, as each variable's share of the constant is added
/// (`var.c:5263`). So every share of the constant, the constant, every coefficient of the representation and each finite
/// side less the constant stay below SCIP's infinity. The representation is walked here, through
/// `SCIPgetProbvarSum` for each variable, never through `SCIPgetProbvarLinearSum`, which carries that abort. Before
/// SCIP's transformed problem exists no variable is represented, and the row is taken as given.
///
/// # Errors
///
/// Why SCIP could not hold the row, as a sentence.
///
/// # Safety
///
/// `scip` is the live instance holding `cols` and `priced`, and `row` names only their columns.
unsafe fn representable(
    scip: *mut SCIP,
    cols: Columns<'_>,
    priced: &[NonNull<SCIP_VAR>],
    row: &AddedRow,
    infinity: f64,
) -> Result<(), String> {
    // SAFETY: the instance is live; the call reads its stage.
    let stage = unsafe { SCIPgetStage(scip) };
    if !(SCIP_Stage_SCIP_STAGE_TRANSFORMED..=SCIP_Stage_SCIP_STAGE_EXITSOLVE).contains(&stage) {
        return Ok(());
    }
    let mut given = Vec::with_capacity(row.terms.len());
    for &(col, value) in &row.terms {
        // SAFETY: the instance is transformed, and the column one of its own.
        let var = unsafe { cols.transformed(scip, col, priced) }.map_err(|error| error.to_string())?;
        given.push((var, value));
    }
    // SAFETY: the stage is checked above, and every variable is a transformed one of the instance.
    let constant = unsafe { active(scip, given, infinity) }?;
    let ends = [row.bounds.lower(), row.bounds.upper()];
    if let Some(end) = ends
        .into_iter()
        .find(|end| end.is_finite() && (end - constant).abs() >= infinity)
    {
        return Err(format!(
            "the bound {end} less the constant {constant} of the columns SCIP has fixed or aggregated is one SCIP \
             reads as infinite, from {infinity} on"
        ));
    }
    Ok(())
}

/// The constant of `given`, a linear sum of transformed variables, over the active variables SCIP represents them by
/// now, after checking that every share of it, the constant itself and every coefficient of the representation stay
/// below `infinity`.
///
/// # Errors
///
/// Why SCIP could not hold the sum, as a sentence.
///
/// # Safety
///
/// `scip` is a live instance from its transformed stage to its exitsolve stage, and every variable of `given` one of
/// its transformed variables.
unsafe fn active(scip: *mut SCIP, mut given: Vec<(*mut SCIP_VAR, f64)>, infinity: f64) -> Result<f64, String> {
    let beyond = |what: &str, value: f64| {
        (value.abs() >= infinity).then(|| {
            format!(
                "the columns SCIP has fixed or aggregated add up to the {what} {value}, which SCIP reads as infinite, \
                 from {infinity} on"
            )
        })
    };
    // Each step resolves one variable; a multi-aggregated one hands on its terms, and SCIP's aggregation graph has no
    // cycle, so the walk ends. The bound turns a walk that does not into a refusal.
    // SAFETY: the instance is live; the call reads its number of variables.
    let vars = usize::try_from(unsafe { SCIPgetNVars(scip) }).unwrap_or(0);
    let bound = (vars + 1).saturating_mul(given.len() + 1).saturating_mul(4);
    let (mut constant, mut coefficients) = (0.0, BTreeMap::<usize, f64>::new());
    for _ in 0..bound {
        let Some((mut var, mut scalar)) = given.pop() else {
            if let Some(&value) = coefficients.values().find(|value| value.abs() >= infinity) {
                return Err(beyond("coefficient", value).unwrap_or_default());
            }
            return Ok(constant);
        };
        let mut share = 0.0;
        // SAFETY: the caller vouches for the instance, its stage and `var`; the rest are locals to write to.
        let code = unsafe { SCIPgetProbvarSum(scip, &raw mut var, &raw mut scalar, &raw mut share) };
        check("SCIPgetProbvarSum", code).map_err(|error| error.to_string())?;
        constant += share;
        if let Some(why) = beyond("constant", share).or_else(|| beyond("constant", constant)) {
            return Err(why);
        }
        if var.is_null() || scalar == 0.0 {
            continue;
        }
        // SAFETY: `var` is a transformed variable of the instance, as SCIP resolved it.
        if unsafe { SCIPvarGetStatus(var) } == SCIP_Varstatus_SCIP_VARSTATUS_MULTAGGR {
            // SAFETY: as above; a multi-aggregated variable holds its terms and constant.
            let (n, terms, scalars, offset) = unsafe {
                (
                    usize::try_from(SCIPvarGetMultaggrNVars(var)).unwrap_or(0),
                    SCIPvarGetMultaggrVars(var),
                    SCIPvarGetMultaggrScalars(var),
                    SCIPvarGetMultaggrConstant(var),
                )
            };
            constant += scalar * offset;
            if let Some(why) = beyond("constant", scalar * offset).or_else(|| beyond("constant", constant)) {
                return Err(why);
            }
            for k in 0..n {
                // SAFETY: SCIP's arrays of the multi-aggregation hold `n` entries each.
                given.push(unsafe { (*terms.add(k), scalar * *scalars.add(k)) });
            }
        } else {
            *coefficients.entry(var.addr()).or_insert(0.0) += scalar;
        }
    }
    Err(format!(
        "SCIP's representation of the columns took more than {bound} steps to resolve"
    ))
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

/// Includes the constraint handler, the separator, the heuristic, the event handler and the pricer.
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
        events::include(scip, slot)?;
        price::include(scip, slot)
    }
}
