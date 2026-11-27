// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A [`ConstraintHandler`](culmen_core::ConstraintHandler): SCIP's constraint handler `culmen-handler`, without
//! constraints of its own, checked and enforced after every SCIP handler and before the lazy-constraint handler.
//!
//! The citations below are into SCIP 10.0.2's `src/scip/` as `scip-sys` builds it.
//!
//! # What SCIP calls, and what the handler is asked
//!
//! - `CONSCHECK`, on every solution SCIP checks (a heuristic's, a start, a sub-SCIP's, a supplied one, a solution kept
//!   from the solve before), asks `check`. The point is feasible exactly when it accepts.
//! - `CONSENFOLP`, `CONSENFOPS` and `CONSENFORELAX`, on a node's point that every handler enforcing before accepted,
//!   ask `check` first and `enforce` only for a point it rejects. SCIP stores a point every enforcement accepted
//!   without checking it (`solve.c:5388-5391`), so enforcement must reject whatever the check rejects, which asking
//!   `check` first makes so.
//! - `CONSLOCK` locks the columns the declaration involves, as [`Locks`] says, and every column a pricer adds both
//!   ways as it is created; the unlock undoes exactly those locks.
//!
//! # SCIP's result, derived from what happened
//!
//! The handler names no result code; the plugin derives it, first matching row first:
//!
//! | Enforcement | Result |
//! |---|---|
//! | a callback of the run panicked | `CUTOFF`, SCIP is being stopped |
//! | a pseudo solution worse than the incumbent | `DIDNOTRUN` |
//! | a handler enforcing before rejected the point | `FEASIBLE`, without asking: the point is no candidate |
//! | the handler is busy with an outer call | `INFEASIBLE` |
//! | `check` accepts | `FEASIBLE` |
//! | `enforce` cut the node off | `CUTOFF` |
//! | `enforce` added a constraint not added before that the point violates by SCIP's tolerance | `CONSADDED` |
//! | nothing changed, and SCIP can resolve the point: an integer variable is unfixed, the node is pure integer without an active pricer, the node has no LP yet, or the solve is stopping | `INFEASIBLE` |
//! | nothing changed, and SCIP cannot | the solve fails with [`ScipError::UnresolvedRejection`] |
//!
//! SCIP resolves an unresolved `INFEASIBLE` by branching on an unfixed integer variable, cuts the node off when every
//! integer variable is fixed and the problem has no continuous variable and no active pricer, and otherwise solves
//! the node's LP and, when the point is still rejected there, fails with `SCIP_INVALIDRESULT` (`solve.c:4705-4911`;
//! a debug build of SCIP aborts at `solve.c:4852` instead). The plugin fails the solve with the typed error first.
//! Every enforcement the handler is asked at one node is counted; past
//! [`ScipOptions::enforcement_rounds`](crate::ScipOptions::enforcement_rounds) the solve fails with
//! [`ScipError::EnforcementStalled`], so a handler that acts without end ends as an error, not as a hang.

use core::cell::{Cell, RefCell};
use core::convert::Infallible;
use core::ffi::{c_int, c_uint};
use core::marker::PhantomData;
use core::ptr::{self, NonNull};

use culmen_core::{
    ActionError, AddedRow, Bound, Bounds, CheckContext, Child, Col, Constraint, Context, EnforceContext,
    ExplainContext, Inferred, Involved, Lock, Locks, NodeAction, NodeActions, NodeContext, PresolveContext, Progress,
    PropagateContext, Reason, TiSlice, Verdict,
};
use scip_sys::{
    SCIP, SCIP_CONS, SCIP_CONSHDLR, SCIP_LOCKTYPE, SCIP_RESULT, SCIP_RETCODE, SCIP_Result_SCIP_CONSADDED,
    SCIP_Result_SCIP_CUTOFF, SCIP_Result_SCIP_DIDNOTRUN, SCIP_Result_SCIP_FEASIBLE, SCIP_Result_SCIP_INFEASIBLE,
    SCIP_Retcode_SCIP_OKAY, SCIP_SOL, SCIP_VAR, SCIPcomputeVarLbLocal, SCIPcomputeVarUbLocal, SCIPconshdlrGetData,
    SCIPgetDepth, SCIPgetFocusNode, SCIPgetNActivePricers, SCIPgetNContVars, SCIPgetNPseudoBranchCands, SCIPgetNRuns,
    SCIPhasCurrentNodeLP, SCIPincludeConshdlrBasic, SCIPisStopped, SCIPmarkDoNotMultaggrVar, SCIPnodeGetNumber,
    SCIPsetConshdlrCopy, SCIPsetConshdlrEnforelax,
};

use super::rows::{add_cons, checked, violates};
use super::{CandidateSource, Delivered, Run, Slot, slot};
use crate::columns::Columns;
use crate::error::check;
use crate::{RetCode, ScipError};

/// After every constraint handler of SCIP's, whose priorities stay above `-4_000_000`, and before the lazy-constraint
/// handler, in enforcement and in checks.
const PRIORITY: c_int = -9_999_998;

/// What a solve with a constraint handler fixes when it starts, from the handler's declaration and the options.
#[derive(Debug)]
pub(crate) struct Setup {
    /// The columns the rule involves.
    pub(crate) locks: Locks,
    /// The most enforcement rounds at one node.
    pub(crate) rounds: u64,
}

/// The constraint handler's state during one solve.
#[derive(Debug)]
pub(super) struct Active {
    locks: Locks,
    involved: Involved,
    /// The most enforcement rounds at one node.
    rounds: u64,
    /// The node the handler was last asked to enforce at, as SCIP's run and node number, and how often it was asked
    /// there.
    enforced: Cell<(c_int, i64, u64)>,
    /// Every constraint the handler added to SCIP's problem in this solve, so one added again changes nothing.
    added: RefCell<Vec<AddedRow>>,
}

impl Active {
    /// The state of a solve of a program of `program_cols` columns as `setup` fixes it.
    pub(super) fn new(setup: Setup, program_cols: usize) -> Self {
        Self {
            involved: Involved::new(&setup.locks, program_cols),
            locks: setup.locks,
            rounds: setup.rounds,
            enforced: Cell::new((0, -1, 0)),
            added: RefCell::default(),
        }
    }
}

/// A point the constraint handler judges: the plugin's `CONSCHECK`, or the start of an enforcement.
#[derive(Debug)]
pub struct ScipCheck<'a> {
    progress: Progress,
    objective: f64,
    candidate: &'a TiSlice<Col, f64>,
    source: CandidateSource,
}

impl ScipCheck<'_> {
    /// Where the point comes from.
    #[must_use]
    pub const fn source(&self) -> CandidateSource {
        self.source
    }
}

impl Context for ScipCheck<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl CheckContext for ScipCheck<'_> {
    fn objective(&self) -> f64 {
        self.objective
    }
    fn candidate(&self) -> &TiSlice<Col, f64> {
        self.candidate
    }
}

/// A node's point the constraint handler's `check` rejected, to resolve: the plugin's `CONSENFOLP`, `CONSENFOPS` or
/// `CONSENFORELAX`.
///
/// SCIP takes the actions once the handler returns. In this release SCIP takes a constraint
/// ([`EnforceContext::add_constraint`]), added to its problem for the rest of the solve, and a plain cutoff
/// ([`NodeContext::cut_off`]); [`NodeContext::tighten`], [`NodeContext::cut_off_because`] and
/// [`EnforceContext::branch`] are refused with [`ActionError::Refused`], as their SCIP facilities (bound changes
/// through culmen's propagator, conflict analysis, children of the handler's own) are not wired yet.
#[derive(Debug)]
pub struct ScipEnforce<'a> {
    scip: *mut SCIP,
    progress: Progress,
    objective: f64,
    candidate: &'a TiSlice<Col, f64>,
    source: CandidateSource,
    cols: Columns<'a>,
    /// The variables of the columns priced so far.
    priced: &'a [NonNull<SCIP_VAR>],
    infinity: f64,
    /// The columns the handler may write.
    involved: &'a Involved,
    actions: NodeActions,
}

impl ScipEnforce<'_> {
    /// Where the point comes from: [`CandidateSource::Lp`], [`CandidateSource::Pseudo`] or
    /// [`CandidateSource::Relaxation`].
    #[must_use]
    pub const fn source(&self) -> CandidateSource {
        self.source
    }

    /// The node's depth in SCIP's search tree, zero at the root.
    #[must_use]
    pub fn depth(&self) -> u32 {
        // SAFETY: the context lives during an enforcement of the live instance, solving, where SCIP reports the depth.
        u32::try_from(unsafe { SCIPgetDepth(self.scip) }).unwrap_or(0)
    }

    /// Refuses an action of kind `action` the actions recorded exclude, as [`NodeActions`] would.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`].
    fn admit(&self, action: NodeAction) -> Result<(), ActionError> {
        match (self.actions.action(), action) {
            (None, _) | (Some(NodeAction::Reduce), NodeAction::Reduce | NodeAction::CutOff) => Ok(()),
            (Some(NodeAction::Branch), NodeAction::CutOff) => Ok(()),
            (Some(after), action) => Err(ActionError::Exclusive { action, after }),
        }
    }
}

/// The refusal of an action whose SCIP facility a later release wires.
fn not_wired(what: &str) -> ActionError {
    ActionError::Refused(format!("{what} is not wired to SCIP in this release"))
}

impl Context for ScipEnforce<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl CheckContext for ScipEnforce<'_> {
    fn objective(&self) -> f64 {
        self.objective
    }
    fn candidate(&self) -> &TiSlice<Col, f64> {
        self.candidate
    }
}

impl NodeContext for ScipEnforce<'_> {
    /// The local bounds of the variable SCIP holds for the column at this node; for a column presolve
    /// multi-aggregated, the bounds its aggregation's variables give.
    fn bounds(&self, col: Col) -> Option<Bounds> {
        if col.index() >= self.candidate.len() {
            return None;
        }
        // SAFETY: the context lives during an enforcement of the live instance holding the columns, solving.
        let var = unsafe { self.cols.transformed(self.scip, col, self.priced) }.ok()?;
        // SAFETY: as above; `var` is a variable of the transformed problem, whose local bounds SCIP computes.
        let (lower, upper) = unsafe {
            (
                SCIPcomputeVarLbLocal(self.scip, var),
                SCIPcomputeVarUbLocal(self.scip, var),
            )
        };
        let end = |value: f64| {
            if value.abs() >= self.infinity {
                value.signum() * f64::INFINITY
            } else {
                value
            }
        };
        Bounds::new(end(lower), end(upper)).ok()
    }

    /// Refused in this release, after the checks every backend makes: SCIP takes a bound change of a handler through
    /// culmen's propagator, which is not wired yet.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`], [`ActionError::UnknownColumn`] or [`ActionError::NotInvolved`] as the trait
    /// states; otherwise [`ActionError::Refused`].
    fn tighten(&mut self, col: Col, bounds: Bounds, reason: Option<Reason>) -> Result<(), ActionError> {
        let _ = (bounds, reason);
        self.admit(NodeAction::Reduce)?;
        self.involved.check(col, self.candidate.len())?;
        Err(not_wired("tightening a bound during enforcement"))
    }

    fn cut_off(&mut self) {
        self.actions.cut_off();
    }

    /// Refused in this release, after the checks every backend makes: SCIP learns a conflict from the bounds named
    /// through its conflict analysis, which is not wired yet. A plain [`NodeContext::cut_off`] is taken.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] or [`ActionError::UnknownColumn`] as the trait states; otherwise
    /// [`ActionError::Refused`].
    fn cut_off_because(&mut self, reasons: &[(Col, Bound)]) -> Result<(), ActionError> {
        self.admit(NodeAction::CutOff)?;
        if let Some(&(col, _)) = reasons.iter().find(|(col, _)| col.index() >= self.candidate.len()) {
            return Err(ActionError::UnknownColumn { col });
        }
        Err(not_wired("cutting off with a conflict"))
    }
}

impl EnforceContext for ScipEnforce<'_> {
    /// SCIP adds the constraint to its problem once the handler returns, as a global linear constraint, unless the
    /// handler added the same terms and bounds before in this solve.
    ///
    /// # Errors
    ///
    /// Besides the errors the trait states, [`ActionError::Refused`] for a number SCIP would read as infinite, or
    /// more terms than SCIP's `int` counts.
    fn add_constraint(&mut self, constraint: Constraint) -> Result<(), ActionError> {
        self.admit(NodeAction::Reduce)?;
        checked(&constraint, self.candidate.len(), self.infinity)?;
        self.actions.add_constraint(&constraint)
    }

    /// Refused in this release, after the checks every backend makes: SCIP takes children of a handler's own, which
    /// are not wired yet.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after any other action; otherwise [`ActionError::Refused`].
    fn branch(&mut self, children: Vec<Child>) -> Result<(), ActionError> {
        let _ = children;
        self.admit(NodeAction::Branch)?;
        Err(not_wired("branching into children of the handler's own"))
    }
}

/// The context of a node's domain to narrow. SCIP does not call a constraint handler's
/// [`propagate`](culmen_core::ConstraintHandler::propagate) in this release, as culmen's propagator is not wired yet,
/// so no value of it exists.
#[derive(Debug)]
pub struct ScipPropagate<'a> {
    never: Infallible,
    life: PhantomData<&'a ()>,
}

impl Context for ScipPropagate<'_> {
    fn progress(&self) -> Progress {
        match self.never {}
    }
}

impl NodeContext for ScipPropagate<'_> {
    fn bounds(&self, _: Col) -> Option<Bounds> {
        match self.never {}
    }
    fn tighten(&mut self, _: Col, _: Bounds, _: Option<Reason>) -> Result<(), ActionError> {
        match self.never {}
    }
    fn cut_off(&mut self) {
        match self.never {}
    }
    fn cut_off_because(&mut self, _: &[(Col, Bound)]) -> Result<(), ActionError> {
        match self.never {}
    }
}

impl PropagateContext for ScipPropagate<'_> {}

/// The context of an inferred bound to explain. SCIP does not call a constraint handler's
/// [`explain`](culmen_core::ConstraintHandler::explain) in this release, as no bound is tightened with a reason yet,
/// so no value of it exists.
#[derive(Debug)]
pub struct ScipExplain<'a> {
    never: Infallible,
    life: PhantomData<&'a ()>,
}

impl Context for ScipExplain<'_> {
    fn progress(&self) -> Progress {
        match self.never {}
    }
}

impl ExplainContext for ScipExplain<'_> {
    fn inferred(&self) -> Inferred {
        match self.never {}
    }
    fn reason(&self) -> Reason {
        match self.never {}
    }
    fn bounds_before(&self, _: Col) -> Option<Bounds> {
        match self.never {}
    }
    fn because(&mut self, _: Col, _: Bound) -> Result<(), ActionError> {
        match self.never {}
    }
}

/// The context of the program's global domain before the search. SCIP does not call a constraint handler's
/// [`presolve`](culmen_core::ConstraintHandler::presolve) in this release, as culmen's propagator is not wired yet,
/// so no value of it exists.
#[derive(Debug)]
pub struct ScipPresolve<'a> {
    never: Infallible,
    life: PhantomData<&'a ()>,
}

impl Context for ScipPresolve<'_> {
    fn progress(&self) -> Progress {
        match self.never {}
    }
}

impl PresolveContext for ScipPresolve<'_> {
    fn bounds(&self, _: Col) -> Option<Bounds> {
        match self.never {}
    }
    fn tighten(&mut self, _: Col, _: Bounds) -> Result<(), ActionError> {
        match self.never {}
    }
    fn add_constraint(&mut self, _: Constraint) -> Result<(), ActionError> {
        match self.never {}
    }
    fn infeasible(&mut self) {
        match self.never {}
    }
}

/// How an enforcement asks the handler: the point, where it comes from, and what SCIP says of it.
#[derive(Clone, Copy, Debug)]
struct Asked {
    source: CandidateSource,
    /// A handler enforcing before rejected the point without resolving it.
    infeasible: bool,
    /// The pseudo solution is worse than the incumbent.
    objinfeasible: bool,
}

impl Run<'_, '_> {
    /// Asks the constraint handler's `check` about a point; `None` when it did not run.
    fn ask_check(
        &self,
        scip: *mut SCIP,
        objective: f64,
        candidate: &TiSlice<Col, f64>,
        source: CandidateSource,
    ) -> Option<Verdict> {
        let progress = self.progress(scip);
        let mut verdict = Verdict::Reject;
        let delivered = self.deliver(scip, |hooks| {
            verdict = hooks.check(&mut ScipCheck {
                progress,
                objective,
                candidate,
                source,
            });
        });
        (delivered == Delivered::Ran).then_some(verdict)
    }

    /// Judges a solution SCIP checks.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run and `sol` a solution SCIP checks, in a stage where SCIP reads it.
    unsafe fn judge(&self, scip: *mut SCIP, sol: *mut SCIP_SOL) -> SCIP_RESULT {
        if self.panicked() {
            return SCIP_Result_SCIP_INFEASIBLE;
        }
        // SAFETY: the caller vouches for the instance, `sol` and the stage.
        let (objective, values) = unsafe { self.read(scip, sol) };
        // A handler busy with an outer call cannot judge the point, so it is rejected; the heuristic context keeps the
        // solutions it is handed in a run with a constraint handler until its handler returns, so none meets a busy
        // handler.
        match self.ask_check(scip, objective, &values, CandidateSource::Check) {
            Some(Verdict::Accept) => SCIP_Result_SCIP_FEASIBLE,
            Some(Verdict::Reject) | None => SCIP_Result_SCIP_INFEASIBLE,
        }
    }

    /// Counts an enforcement at the focus node.
    ///
    /// # Errors
    ///
    /// [`ScipError::EnforcementStalled`] past the solve's limit at one node.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, enforcing.
    unsafe fn count_round(&self, scip: *mut SCIP, active: &Active) -> Result<(), ScipError> {
        // SAFETY: the instance is solving, with a focus node, whose number SCIP reports, as it reports its runs.
        let at = unsafe { (SCIPgetNRuns(scip), SCIPnodeGetNumber(SCIPgetFocusNode(scip))) };
        let (runs, node, count) = active.enforced.get();
        let count = if (runs, node) == at { count.saturating_add(1) } else { 1 };
        active.enforced.set((at.0, at.1, count));
        if count > active.rounds {
            return Err(ScipError::EnforcementStalled { rounds: active.rounds });
        }
        Ok(())
    }

    /// Enforces the constraint handler on a node's point, `sol`, or the current LP or pseudo solution for null.
    ///
    /// # Errors
    ///
    /// [`ScipError::EnforcementStalled`], [`ScipError::UnresolvedRejection`]; [`ScipError::Call`] when SCIP refuses
    /// a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, solving, and `sol` null or a solution of it.
    unsafe fn enforce_handler(
        &self,
        scip: *mut SCIP,
        sol: *mut SCIP_SOL,
        active: &Active,
        asked: Asked,
    ) -> Result<SCIP_RESULT, ScipError> {
        if self.panicked() {
            return Ok(SCIP_Result_SCIP_CUTOFF);
        }
        if asked.objinfeasible {
            return Ok(SCIP_Result_SCIP_DIDNOTRUN);
        }
        // A point another handler rejected is no candidate; SCIP resolves it with that handler.
        if asked.infeasible {
            return Ok(SCIP_Result_SCIP_FEASIBLE);
        }
        // SAFETY: the caller vouches for the instance, its stage and `sol`.
        unsafe { self.count_round(scip, active) }?;
        // SAFETY: as above.
        let (objective, values) = unsafe { self.read(scip, sol) };
        let verdict = self.ask_check(scip, objective, &values, asked.source);
        match verdict {
            Some(Verdict::Accept) => return Ok(SCIP_Result_SCIP_FEASIBLE),
            Some(Verdict::Reject) => {}
            None if self.panicked() => return Ok(SCIP_Result_SCIP_CUTOFF),
            // Busy, which no enforcement meets: heuristics keep their solutions until their handler returns.
            None => return Ok(SCIP_Result_SCIP_INFEASIBLE),
        }
        let Some(actions) = self.ask_enforce(scip, objective, &values, active, asked.source) else {
            return Ok(if self.panicked() {
                SCIP_Result_SCIP_CUTOFF
            } else {
                SCIP_Result_SCIP_INFEASIBLE
            });
        };
        // SAFETY: as above.
        unsafe { self.apply_actions(scip, &values, active, &actions) }
    }

    /// Asks the constraint handler's `enforce` about a rejected point; the actions it recorded, `None` when it did not
    /// run.
    fn ask_enforce(
        &self,
        scip: *mut SCIP,
        objective: f64,
        candidate: &TiSlice<Col, f64>,
        active: &Active,
        source: CandidateSource,
    ) -> Option<NodeActions> {
        let progress = self.progress(scip);
        let priced = self.priced_vars.borrow();
        let mut ctx = ScipEnforce {
            scip,
            progress,
            objective,
            candidate,
            source,
            cols: self.cols,
            priced: &priced,
            infinity: self.infinity,
            involved: &active.involved,
            actions: NodeActions::new(&active.involved, candidate.len()),
        };
        let delivered = self.deliver(scip, |hooks| hooks.enforce(&mut ctx));
        (delivered == Delivered::Ran).then_some(ctx.actions)
    }

    /// Hands SCIP the actions an enforcement recorded and derives SCIP's result from what happened.
    ///
    /// # Errors
    ///
    /// [`ScipError::UnresolvedRejection`] when nothing changed and SCIP cannot resolve the point;
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, enforcing on the point `values`.
    unsafe fn apply_actions(
        &self,
        scip: *mut SCIP,
        values: &TiSlice<Col, f64>,
        active: &Active,
        actions: &NodeActions,
    ) -> Result<SCIP_RESULT, ScipError> {
        if actions.cutoff().is_some() {
            return Ok(SCIP_Result_SCIP_CUTOFF);
        }
        let mut violated = false;
        for row in actions.constraints() {
            if active.added.borrow().contains(row) {
                continue;
            }
            // Nothing is borrowed while SCIP runs, which may write a message the callback is handed.
            // SAFETY: the caller's contract is `add_cons`'s; the row was checked against the columns of the point.
            unsafe {
                add_cons(
                    scip,
                    self.cols,
                    &self.priced_vars.borrow(),
                    row,
                    self.infinity,
                    c"culmen-handler",
                )
            }?;
            active.added.borrow_mut().push(row.clone());
            violated |= violates(scip, row, values);
        }
        if violated {
            return Ok(SCIP_Result_SCIP_CONSADDED);
        }
        // SAFETY: as above.
        if unsafe { unresolvable(scip) } {
            return Err(ScipError::UnresolvedRejection);
        }
        Ok(SCIP_Result_SCIP_INFEASIBLE)
    }

    /// Locks what the declaration involves for a transformed problem SCIP creates: every variable both ways for
    /// [`Locks::All`]; for [`Locks::Only`] each listed program column's variable in its directions, marked so presolve
    /// does not multi-aggregate it.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, transforming its problem.
    unsafe fn lock_declared(&self, scip: *mut SCIP, active: &Active, n: c_int) -> Result<(), ScipError> {
        let applied = &self.slot.handler_locks;
        let listed = match &active.locks {
            // SAFETY: the caller's contract.
            Locks::All => return unsafe { applied.lock_all(scip, n) },
            Locks::Only(listed) => listed,
        };
        // A column beyond the program's is one a pricer adds, locked both ways as it is created.
        for &(col, lock) in listed.iter().filter(|(col, _)| col.index() < self.cols.len()) {
            // SAFETY: the caller's contract; the column is one of the program's.
            let var = unsafe { self.cols.transformed(scip, col, &[]) }?;
            let var = NonNull::new(var).ok_or(ScipError::Call {
                call: "SCIPgetTransformedVar",
                code: RetCode::InvalidData,
            })?;
            let (down, up) = match lock {
                Lock::Down => (n, 0),
                Lock::Up => (0, n),
                Lock::Both => (n, n),
            };
            // SAFETY: as above; `var` is a variable of the transformed problem.
            unsafe { applied.lock(scip, var, down, up) }?;
            // The mark lives on the transformed variable, which dies with the problem; SCIP has no unmark.
            // SAFETY: as above, in the transforming stage, where SCIP takes the mark.
            check("SCIPmarkDoNotMultaggrVar", unsafe {
                SCIPmarkDoNotMultaggrVar(scip, var.as_ptr())
            })?;
        }
        Ok(())
    }
}

/// Whether SCIP cannot resolve a point the constraint handler rejected without acting: no integer variable is unfixed
/// to branch on, the node has a continuous variable or an active pricer so SCIP would not cut it off, its LP is
/// solved, and the solve is not stopping, so SCIP would fail with `SCIP_INVALIDRESULT` (`solve.c:4705-4911`).
///
/// # Safety
///
/// `scip` is a live instance, enforcing.
unsafe fn unresolvable(scip: *mut SCIP) -> bool {
    // SAFETY: the instance is solving, where SCIP reports each of these.
    unsafe {
        SCIPgetNPseudoBranchCands(scip) == 0
            && (SCIPgetNContVars(scip) > 0 || SCIPgetNActivePricers(scip) > 0)
            && SCIPhasCurrentNodeLP(scip) != 0
            && SCIPisStopped(scip) == 0
    }
}

/// Includes the plugin.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(super) unsafe fn include(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    let mut conshdlr: *mut SCIP_CONSHDLR = ptr::null_mut();
    // SAFETY: the instance is live and in its init stage; the name and description are NUL-terminated strings SCIP
    // copies; the callbacks treat their arguments as SCIP passes them, and the data is the slot, which outlives SCIP.
    // The handler needs no constraints, so SCIP calls it once per point and locks once per problem.
    check("SCIPincludeConshdlrBasic", unsafe {
        SCIPincludeConshdlrBasic(
            scip,
            &raw mut conshdlr,
            c"culmen-handler".as_ptr(),
            c"the constraint handler of a culmen solve".as_ptr(),
            PRIORITY,
            PRIORITY,
            -1,
            0,
            Some(enfolp),
            Some(enfops),
            Some(conscheck),
            Some(conslock),
            slot.as_ptr().cast(),
        )
    })?;
    // SAFETY: `conshdlr` is the plugin just included in the live instance; the callbacks treat their arguments as
    // SCIP passes them.
    unsafe {
        check(
            "SCIPsetConshdlrEnforelax",
            SCIPsetConshdlrEnforelax(scip, conshdlr, Some(enforelax)),
        )?;
        check(
            "SCIPsetConshdlrCopy",
            SCIPsetConshdlrCopy(scip, conshdlr, Some(copy), None),
        )
    }
}

/// The plugin's slot and its run, `None` when no run with a constraint handler is installed.
///
/// # Safety
///
/// `conshdlr` is the plugin, called by SCIP.
unsafe fn handler_run<'s>(scip: *mut SCIP, conshdlr: *mut SCIP_CONSHDLR) -> Option<(&'s Run<'s, 's>, &'s Active)> {
    // SAFETY: the plugin's data is its slot.
    let slot = unsafe { slot::<_>(SCIPconshdlrGetData(conshdlr)) }?;
    // SAFETY: SCIP calls the plugin during the solve the run is installed for, or with none installed.
    let run = unsafe { slot.run() }?;
    run.pass_stop(scip);
    run.constraint.as_ref().map(|active| (run, active))
}

/// `CONSCHECK`: the point is feasible exactly when the handler accepts it.
unsafe extern "C" fn conscheck(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    _conss: *mut *mut SCIP_CONS,
    _nconss: c_int,
    sol: *mut SCIP_SOL,
    _checkintegrality: c_uint,
    _checklprows: c_uint,
    _printreason: c_uint,
    _completely: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    // SAFETY: SCIP calls the plugin's check with itself and a result to write.
    let verdict = match unsafe { handler_run(scip, conshdlr) } {
        // SAFETY: SCIP checks `sol` in a stage where it reads it.
        Some((run, _)) => unsafe { run.judge(scip, sol) },
        None => SCIP_Result_SCIP_FEASIBLE,
    };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(verdict) };
    SCIP_Retcode_SCIP_OKAY
}

/// Enforces on `sol` for a callback of the plugin, and writes the result.
///
/// # Safety
///
/// SCIP calls the plugin's enforcement with itself, `sol` null or a solution of its own, and `result` to write.
unsafe fn enforced(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    sol: *mut SCIP_SOL,
    asked: Asked,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    // SAFETY: the caller's contract.
    let (verdict, code) = match unsafe { handler_run(scip, conshdlr) } {
        // SAFETY: SCIP enforces while solving, on `sol`.
        Some((run, active)) => match unsafe { run.enforce_handler(scip, sol, active, asked) } {
            Ok(verdict) => (verdict, SCIP_Retcode_SCIP_OKAY),
            // The solve fails with the error; the result closes the node should SCIP read it.
            Err(error) => (SCIP_Result_SCIP_CUTOFF, run.fail(error)),
        },
        None => (SCIP_Result_SCIP_FEASIBLE, SCIP_Retcode_SCIP_OKAY),
    };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(verdict) };
    code
}

/// `CONSENFOLP`: the node's LP solution.
unsafe extern "C" fn enfolp(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    _conss: *mut *mut SCIP_CONS,
    _nconss: c_int,
    _nusefulconss: c_int,
    solinfeasible: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    let asked = Asked {
        source: CandidateSource::Lp,
        infeasible: solinfeasible != 0,
        objinfeasible: false,
    };
    // SAFETY: SCIP calls the enforcement while solving; null is the current LP solution.
    unsafe { enforced(scip, conshdlr, ptr::null_mut(), asked, result) }
}

/// `CONSENFOPS`: the node's pseudo solution.
unsafe extern "C" fn enfops(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    _conss: *mut *mut SCIP_CONS,
    _nconss: c_int,
    _nusefulconss: c_int,
    solinfeasible: c_uint,
    objinfeasible: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    let asked = Asked {
        source: CandidateSource::Pseudo,
        infeasible: solinfeasible != 0,
        objinfeasible: objinfeasible != 0,
    };
    // SAFETY: SCIP calls the enforcement while solving; null is the current pseudo solution.
    unsafe { enforced(scip, conshdlr, ptr::null_mut(), asked, result) }
}

/// `CONSENFORELAX`: a relaxator's solution.
unsafe extern "C" fn enforelax(
    scip: *mut SCIP,
    sol: *mut SCIP_SOL,
    conshdlr: *mut SCIP_CONSHDLR,
    _conss: *mut *mut SCIP_CONS,
    _nconss: c_int,
    _nusefulconss: c_int,
    solinfeasible: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    let asked = Asked {
        source: CandidateSource::Relaxation,
        infeasible: solinfeasible != 0,
        objinfeasible: false,
    };
    // SAFETY: SCIP calls the enforcement while solving, with the relaxation's solution.
    unsafe { enforced(scip, conshdlr, sol, asked, result) }
}

/// `CONSLOCK`: what the declaration involves, for a solve with a constraint handler, undone when SCIP unlocks.
unsafe extern "C" fn conslock(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    _cons: *mut SCIP_CONS,
    _locktype: SCIP_LOCKTYPE,
    nlockspos: c_int,
    nlocksneg: c_int,
) -> SCIP_RETCODE {
    // SAFETY: the plugin's data is its slot.
    let Some(slot) = (unsafe { slot::<_>(SCIPconshdlrGetData(conshdlr)) }) else {
        return SCIP_Retcode_SCIP_OKAY;
    };
    let n = nlockspos.saturating_add(nlocksneg);
    // SCIP locks once a problem is transformed, with a run installed, and unlocks as it frees it, maybe with none.
    let applied = if n > 0 {
        // SAFETY: SCIP calls the lock during the solve the run is installed for, or with none installed.
        let Some(run) = (unsafe { slot.run() }) else {
            return SCIP_Retcode_SCIP_OKAY;
        };
        let Some(active) = run.constraint.as_ref() else {
            return SCIP_Retcode_SCIP_OKAY;
        };
        // SAFETY: SCIP locks a transformed problem it holds, in its transforming stage.
        unsafe { run.lock_declared(scip, active, n) }
    } else {
        // SAFETY: SCIP unlocks a transformed problem it holds before it frees its variables.
        unsafe { slot.handler_locks.unlock(scip) }
    };
    match applied {
        Ok(()) => SCIP_Retcode_SCIP_OKAY,
        // SAFETY: as above.
        Err(error) => unsafe { slot.fail(error) },
    }
}

/// `CONSHDLRCOPY`: a copy is valid unless the solve has a constraint handler, which a copy would not see.
unsafe extern "C" fn copy(_target: *mut SCIP, conshdlr: *mut SCIP_CONSHDLR, valid: *mut c_uint) -> SCIP_RETCODE {
    // SAFETY: the plugin's data is its slot.
    let handler = unsafe { slot::<_>(SCIPconshdlrGetData(conshdlr)) }
        // SAFETY: SCIP copies the plugin during the solve the run is installed for.
        .is_some_and(|slot| unsafe { slot.constraint() });
    // SAFETY: `valid` is SCIP's to write for the call.
    unsafe { valid.write(c_uint::from(!handler)) };
    SCIP_Retcode_SCIP_OKAY
}
