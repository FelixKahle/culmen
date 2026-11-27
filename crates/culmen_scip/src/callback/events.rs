// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Polls, progress and solutions: an event handler on SCIP's global events.
//!
//! The handler catches its events when SCIP has transformed the problem (`EVENTINIT`), so it sees presolve too, and
//! drops them before SCIP frees the transformed problem (`EVENTEXIT`). A finished presolve round, a focused node, a
//! solved node LP and a solved node each make a poll and a progress report; a pure LP has one node, so it is polled
//! after presolve, when its node is focused, when its LP is solved and when its node is solved. SCIP's LP solver runs
//! without events, so a stop asked for during one LP solve acts when it ends. Every solution SCIP stores is a
//! feasible solution, and a new best one is the incumbent too. SCIP checks and stores the starts and the last solve's
//! best solutions as it transforms the problem, before the handler can catch an event, so the handler hands those to
//! the callback as it catches its events.

use core::ffi::c_int;
use core::ptr::{self, NonNull};

use culmen_core::backend::Found;
use culmen_core::{Col, Context, Event, PollContext, Progress, ProgressContext, SolutionContext, TiSlice};
use scip_sys::{
    SCIP, SCIP_EVENT, SCIP_EVENTDATA, SCIP_EVENTHDLR, SCIP_EVENTTYPE, SCIP_RETCODE, SCIP_Retcode_SCIP_OKAY,
    SCIP_Stage_SCIP_STAGE_EXITSOLVE, SCIP_Stage_SCIP_STAGE_TRANSFORMED, SCIPcatchEvent, SCIPdropEvent, SCIPeventGetSol,
    SCIPeventGetType, SCIPeventhdlrGetData, SCIPgetBestSol, SCIPgetNSols, SCIPgetSols, SCIPgetStage,
    SCIPincludeEventhdlrBasic, SCIPsetEventhdlrExit, SCIPsetEventhdlrInit,
};

use super::{Delivered, Run, Slot, slot};
use crate::ScipError;
use crate::error::check;

/// `SCIP_EVENTTYPE_PRESOLVEROUND`.
const PRESOLVEROUND: SCIP_EVENTTYPE = 1 << 18;
/// `SCIP_EVENTTYPE_NODEFOCUSED`.
const NODEFOCUSED: SCIP_EVENTTYPE = 1 << 19;
/// `SCIP_EVENTTYPE_NODEFEASIBLE`.
const NODEFEASIBLE: SCIP_EVENTTYPE = 1 << 20;
/// `SCIP_EVENTTYPE_NODEINFEASIBLE`.
const NODEINFEASIBLE: SCIP_EVENTTYPE = 1 << 21;
/// `SCIP_EVENTTYPE_NODEBRANCHED`.
const NODEBRANCHED: SCIP_EVENTTYPE = 1 << 22;
/// `SCIP_EVENTTYPE_LPSOLVED`.
const LPSOLVED: SCIP_EVENTTYPE = 1 << 26;
/// `SCIP_EVENTTYPE_POORSOLFOUND`.
const POORSOLFOUND: SCIP_EVENTTYPE = 1 << 27;
/// `SCIP_EVENTTYPE_BESTSOLFOUND`.
const BESTSOLFOUND: SCIP_EVENTTYPE = 1 << 28;
/// `SCIP_EVENTTYPE_NODESOLVED`.
const NODESOLVED: SCIP_EVENTTYPE = NODEFEASIBLE | NODEINFEASIBLE | NODEBRANCHED;
/// The events that poll.
const POLLING: SCIP_EVENTTYPE = PRESOLVEROUND | NODEFOCUSED | LPSOLVED | NODESOLVED;
/// The events of a solution SCIP stored.
const SOLFOUND: SCIP_EVENTTYPE = POORSOLFOUND | BESTSOLFOUND;

/// The SCIP event a poll or a progress report comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScipEvent {
    /// A presolve round finished (`SCIP_EVENTTYPE_PRESOLVEROUND`).
    PresolveRound,
    /// A node became the focus node (`SCIP_EVENTTYPE_NODEFOCUSED`).
    NodeFocused,
    /// A node's LP was solved, with its cuts (`SCIP_EVENTTYPE_LPSOLVED`).
    LpSolved,
    /// A node was solved: found feasible, infeasible or cut off, or branched on (`SCIP_EVENTTYPE_NODESOLVED`).
    NodeSolved,
}

impl ScipEvent {
    /// The event of SCIP's type, `None` for one that does not poll.
    const fn from_raw(raw: SCIP_EVENTTYPE) -> Option<Self> {
        Some(if raw & PRESOLVEROUND != 0 {
            Self::PresolveRound
        } else if raw & NODEFOCUSED != 0 {
            Self::NodeFocused
        } else if raw & LPSOLVED != 0 {
            Self::LpSolved
        } else if raw & NODESOLVED != 0 {
            Self::NodeSolved
        } else {
            return None;
        })
    }
}

/// A frequent poll at which the callback may stop the solve: a presolve round, a node focused, a node LP solved or a
/// node solved.
#[derive(Debug)]
pub struct ScipPoll<'a> {
    progress: Progress,
    event: ScipEvent,
    interrupt: &'a mut bool,
}

impl ScipPoll<'_> {
    /// The SCIP event that polls.
    #[must_use]
    pub const fn event(&self) -> ScipEvent {
        self.event
    }
}

impl Context for ScipPoll<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl PollContext for ScipPoll<'_> {
    /// SCIP stops through `SCIPinterruptSolve`, at its next check of its limits.
    fn interrupt(&mut self) {
        *self.interrupt = true;
    }
}

/// A progress report at each poll.
#[derive(Debug)]
pub struct ScipProgress {
    progress: Progress,
    event: ScipEvent,
}

impl ScipProgress {
    /// The SCIP event that reports.
    #[must_use]
    pub const fn event(&self) -> ScipEvent {
        self.event
    }
}

impl Context for ScipProgress {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl ProgressContext for ScipProgress {}

/// A solution SCIP stored, read in the program's columns: any one as [`Event::Feasible`], a new best one also as
/// [`Event::Incumbent`].
#[derive(Debug)]
pub struct ScipSolution<'a> {
    found: &'a Found,
}

impl<'a> ScipSolution<'a> {
    pub(super) const fn new(found: &'a Found) -> Self {
        Self { found }
    }
}

impl Context for ScipSolution<'_> {
    fn progress(&self) -> Progress {
        self.found.progress
    }
}

impl SolutionContext for ScipSolution<'_> {
    fn objective(&self) -> f64 {
        self.found.objective
    }
    fn values(&self) -> &TiSlice<Col, f64> {
        &self.found.values
    }
}

impl Run<'_, '_> {
    /// Hands a poll and a progress report to the callback, as it asked; neither while it is busy.
    fn poll(&self, scip: *mut SCIP, event: ScipEvent) {
        let progress = self.progress(scip);
        if self.wanted.contains(Event::Poll) {
            let mut interrupt = false;
            let delivered = self.deliver(scip, |hooks| {
                hooks.poll(&mut ScipPoll {
                    progress,
                    event,
                    interrupt: &mut interrupt,
                });
            });
            if delivered == Delivered::Ran && interrupt {
                self.request_stop(scip);
            }
        }
        if self.wanted.contains(Event::Progress) {
            self.deliver(scip, |hooks| {
                hooks.progress(&mut ScipProgress { progress, event });
            });
        }
    }

    /// Hands the callback the solutions SCIP stored before the handler caught its events: the starts and the last
    /// solve's best solutions, which SCIP checks and stores as it transforms the problem. Each is feasible, the worst
    /// first, and the best is the incumbent too.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, with its problem transformed.
    unsafe fn stored(&self, scip: *mut SCIP) {
        // SAFETY: the instance is live; the call reads its stage.
        let stage = unsafe { SCIPgetStage(scip) };
        if !(SCIP_Stage_SCIP_STAGE_TRANSFORMED..=SCIP_Stage_SCIP_STAGE_EXITSOLVE).contains(&stage) {
            return;
        }
        // SAFETY: SCIP lists its stored solutions, best first, from its transformed stage to its exitsolve stage.
        let (sols, len, best) = unsafe { (SCIPgetSols(scip), SCIPgetNSols(scip), SCIPgetBestSol(scip)) };
        let len = usize::try_from(len).unwrap_or(0);
        if sols.is_null() || len == 0 {
            return;
        }
        // SAFETY: SCIP's array holds `len` solutions; it is copied before any callback runs.
        let sols = unsafe { core::slice::from_raw_parts(sols, len) }.to_vec();
        for sol in sols.into_iter().rev() {
            // SAFETY: `sol` is a stored solution of the transformed problem, which SCIP reads in this stage.
            let (objective, values) = unsafe { self.read(scip, sol) };
            let found = Found {
                progress: self.progress(scip),
                objective,
                values,
            };
            self.solution(scip, found, sol == best);
        }
    }

    /// The events to catch for the events the callback asked for.
    const fn caught(&self) -> SCIP_EVENTTYPE {
        let mut mask = 0;
        if self.wanted.contains(Event::Poll) || self.wanted.contains(Event::Progress) {
            mask |= POLLING;
        }
        if self.wanted.contains(Event::Feasible) || self.wanted.contains(Event::Incumbent) {
            mask |= SOLFOUND;
        }
        mask
    }
}

/// Includes the event handler.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(super) unsafe fn include(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    let mut eventhdlr: *mut SCIP_EVENTHDLR = ptr::null_mut();
    // SAFETY: the instance is live and in its init stage; the name and description are NUL-terminated strings SCIP
    // copies; the callbacks treat their arguments as SCIP passes them, and the data is the slot, which outlives SCIP.
    unsafe {
        check(
            "SCIPincludeEventhdlrBasic",
            SCIPincludeEventhdlrBasic(
                scip,
                &raw mut eventhdlr,
                c"culmen".as_ptr(),
                c"polls, progress and solutions for a culmen callback".as_ptr(),
                Some(exec),
                slot.as_ptr().cast(),
            ),
        )?;
        check(
            "SCIPsetEventhdlrInit",
            SCIPsetEventhdlrInit(scip, eventhdlr, Some(init)),
        )?;
        check(
            "SCIPsetEventhdlrExit",
            SCIPsetEventhdlrExit(scip, eventhdlr, Some(exit)),
        )
    }
}

/// The handler's slot.
///
/// # Safety
///
/// `eventhdlr` is the handler, called by SCIP.
unsafe fn handler_slot<'s>(eventhdlr: *mut SCIP_EVENTHDLR) -> Option<&'s Slot> {
    // SAFETY: the handler's data is its slot.
    unsafe { slot::<_>(SCIPeventhdlrGetData(eventhdlr)) }
}

/// `EVENTINIT`: catches the events the run's callback asked for, on the transformed problem.
unsafe extern "C" fn init(scip: *mut SCIP, eventhdlr: *mut SCIP_EVENTHDLR) -> SCIP_RETCODE {
    // SAFETY: SCIP calls the handler with itself.
    let Some(slot) = (unsafe { handler_slot(eventhdlr) }) else {
        return SCIP_Retcode_SCIP_OKAY;
    };
    // SAFETY: SCIP initialises the handler during the solve the run is installed for.
    let Some(run) = (unsafe { slot.run() }) else {
        return SCIP_Retcode_SCIP_OKAY;
    };
    let mask = run.caught();
    if mask == 0 {
        return SCIP_Retcode_SCIP_OKAY;
    }
    let mut position: c_int = -1;
    // SAFETY: the instance is live with its transformed problem, the mask holds global events alone, and `position`
    // is a local to write to.
    let caught = unsafe { SCIPcatchEvent(scip, mask, eventhdlr, ptr::null_mut(), &raw mut position) };
    match check("SCIPcatchEvent", caught) {
        Ok(()) => slot.caught.set(Some((mask, position))),
        Err(error) => return run.fail(error),
    }
    if mask & SOLFOUND != 0 {
        // SAFETY: SCIP initialises its plugins once the problem is transformed.
        unsafe { run.stored(scip) };
    }
    SCIP_Retcode_SCIP_OKAY
}

/// `EVENTEXIT`: drops the events caught, before SCIP frees the transformed problem.
unsafe extern "C" fn exit(scip: *mut SCIP, eventhdlr: *mut SCIP_EVENTHDLR) -> SCIP_RETCODE {
    // SAFETY: SCIP calls the handler with itself.
    let Some(slot) = (unsafe { handler_slot(eventhdlr) }) else {
        return SCIP_Retcode_SCIP_OKAY;
    };
    let Some((mask, position)) = slot.caught.take() else {
        return SCIP_Retcode_SCIP_OKAY;
    };
    // SAFETY: the instance is live with the transformed problem the events were caught on, at `position`.
    let dropped = unsafe { SCIPdropEvent(scip, mask, eventhdlr, ptr::null_mut(), position) };
    match check("SCIPdropEvent", dropped) {
        Ok(()) => SCIP_Retcode_SCIP_OKAY,
        // SAFETY: SCIP calls the handler's exit during a solve or with no run installed.
        Err(error) => unsafe { slot.fail(error) },
    }
}

/// `EVENTEXEC`: a poll and a progress report, or a solution.
unsafe extern "C" fn exec(
    scip: *mut SCIP,
    eventhdlr: *mut SCIP_EVENTHDLR,
    event: *mut SCIP_EVENT,
    _eventdata: *mut SCIP_EVENTDATA,
) -> SCIP_RETCODE {
    // SAFETY: SCIP calls the handler with itself.
    let Some(slot) = (unsafe { handler_slot(eventhdlr) }) else {
        return SCIP_Retcode_SCIP_OKAY;
    };
    // SAFETY: SCIP calls the handler during the solve the run is installed for.
    let Some(run) = (unsafe { slot.run() }) else {
        return SCIP_Retcode_SCIP_OKAY;
    };
    run.pass_stop(scip);
    // SAFETY: `event` is the event SCIP processes, valid for the call.
    let kind = unsafe { SCIPeventGetType(event) };
    if kind & SOLFOUND != 0 {
        // SAFETY: as above; a solution event carries the solution stored.
        let sol = unsafe { SCIPeventGetSol(event) };
        if !sol.is_null() {
            // SAFETY: SCIP stores solutions of the transformed problem, which it reads from its transforming stage on.
            let (objective, values) = unsafe { run.read(scip, sol) };
            let found = Found {
                progress: run.progress(scip),
                objective,
                values,
            };
            run.solution(scip, found, kind & BESTSOLFOUND != 0);
        }
    } else if let Some(event) = ScipEvent::from_raw(kind) {
        run.poll(scip, event);
    }
    SCIP_Retcode_SCIP_OKAY
}
