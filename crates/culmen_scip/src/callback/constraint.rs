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
//! `propagate` and `explain` are asked by culmen's propagator, not by this plugin: SCIP resolves a constraint
//! handler's inference only through a constraint object (`cons.c:7493-7550`), which a handler without constraints
//! has none of, and a propagator's inference through the propagator alone (`prop.c:739-784`).
//!
//! # SCIP's result, derived from what happened
//!
//! The handler names no result code; the plugin derives it, first matching row first:
//!
//! | Enforcement | Result |
//! |---|---|
//! | a callback of the run panicked, or a plugin met an error | `CUTOFF`, SCIP is being stopped |
//! | a pseudo solution worse than the incumbent | `DIDNOTRUN` |
//! | a handler enforcing before rejected the point | `FEASIBLE`, without asking: the point is no candidate |
//! | the handler is busy with an outer call | `INFEASIBLE` |
//! | `check` accepts | `FEASIBLE` |
//! | `enforce` cut the node off, plainly or naming the bounds SCIP's conflict analysis is handed first | `CUTOFF` |
//! | a tightening of `enforce` emptied a domain, as SCIP reports it | `CUTOFF` |
//! | `enforce` added a constraint not added before that the point violates by SCIP's tolerance (its tightenings are applied too) | `CONSADDED` |
//! | a tightening of `enforce` moved a bound, as SCIP reports it | `REDUCEDDOM` |
//! | nothing changed, and SCIP can resolve the point: an integer variable is unfixed, the node is pure integer without an active pricer, the node has no LP yet, or the solve is stopping | `INFEASIBLE` |
//! | nothing changed, and SCIP cannot | `CUTOFF`, SCIP is asked to stop, and the solve fails with [`ScipError::UnresolvedRejection`] |
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
use std::collections::HashMap;

use culmen_core::{
    ActionError, AddedRow, Bound, Bounds, CheckContext, Child, Col, Constraint, Context, EnforceContext, Involved,
    Lock, Locks, NodeAction, NodeActions, NodeContext, PresolveContext, Progress, Reason, TiSlice, Verdict,
};
use scip_sys::{
    SCIP, SCIP_CONS, SCIP_CONSHDLR, SCIP_LOCKTYPE, SCIP_RESULT, SCIP_RETCODE, SCIP_Result_SCIP_CONSADDED,
    SCIP_Result_SCIP_CUTOFF, SCIP_Result_SCIP_DIDNOTRUN, SCIP_Result_SCIP_FEASIBLE, SCIP_Result_SCIP_INFEASIBLE,
    SCIP_Result_SCIP_REDUCEDDOM, SCIP_Retcode_SCIP_OKAY, SCIP_SOL, SCIP_VAR, SCIPconshdlrGetData, SCIPgetFocusNode,
    SCIPgetNActivePricers, SCIPgetNContVars, SCIPgetNPseudoBranchCands, SCIPgetNRuns, SCIPhasCurrentNodeLP,
    SCIPincludeConshdlrBasic, SCIPisStopped, SCIPmarkDoNotMultaggrVar, SCIPnodeGetNumber, SCIPsetConshdlrCopy,
    SCIPsetConshdlrEnforelax,
};

use super::node::{Node, Settled, admit};
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
    /// The columns the handler may write.
    pub(super) involved: Involved,
    /// The most enforcement rounds at one node.
    rounds: u64,
    /// The node the handler was last asked to enforce at, as SCIP's run and node number, and how often it was asked
    /// there.
    enforced: Cell<(c_int, i64, u64)>,
    /// Every constraint the handler added to SCIP's problem in this solve, so one added again changes nothing.
    added: RefCell<Vec<AddedRow>>,
    /// The column of each transformed variable mapped so far, which an explanation maps SCIP's variable back by: the
    /// program's columns in order, then each priced one, as many as `mapped` counts.
    vars: RefCell<HashMap<NonNull<SCIP_VAR>, Col>>,
    mapped: Cell<usize>,
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
            vars: RefCell::default(),
            mapped: Cell::new(0),
        }
    }

    /// The column whose transformed variable is `var`, among the program's columns `cols` and the priced ones
    /// `priced`; `None` for another variable. SCIP hands an explanation back the variable a tightening named
    /// (`tree.c:1989`), which is the column's transformed one, and a transformed variable stays the same until SCIP
    /// frees its problem, after the run.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP holds no transformed variable for a column.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance holding `cols` and `priced`, transformed.
    pub(super) unsafe fn column_of(
        &self,
        scip: *mut SCIP,
        cols: Columns<'_>,
        priced: &[NonNull<SCIP_VAR>],
        var: *mut SCIP_VAR,
    ) -> Result<Option<Col>, ScipError> {
        let mut map = self.vars.borrow_mut();
        let total = cols.len() + priced.len();
        for k in self.mapped.get()..total {
            let col = Col::new(k);
            // SAFETY: the caller's contract; the column is one of the problem's.
            let transformed = unsafe { cols.transformed(scip, col, priced) }?;
            if let Some(transformed) = NonNull::new(transformed) {
                map.insert(transformed, col);
            }
        }
        self.mapped.set(total);
        Ok(NonNull::new(var).and_then(|var| map.get(&var).copied()))
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
/// SCIP takes the actions once the handler returns: a constraint ([`EnforceContext::add_constraint`]), added to its
/// problem for the rest of the solve; a tightening ([`NodeContext::tighten`]), at this node and below, and at the root
/// for the rest of the solve, with a reason through culmen's propagator, which answers SCIP's conflict analysis with
/// the handler's [`explain`](culmen_core::ConstraintHandler::explain); a cutoff, plain or with the bounds that make the
/// node infeasible, which SCIP's conflict analysis may learn a conflict from ([`NodeContext::cut_off_because`]).
/// [`EnforceContext::branch`] is refused with [`ActionError::Refused`], as children of a handler's own are not wired
/// yet.
#[derive(Debug)]
pub struct ScipEnforce<'a> {
    progress: Progress,
    objective: f64,
    candidate: &'a TiSlice<Col, f64>,
    source: CandidateSource,
    node: Node<'a>,
}

impl ScipEnforce<'_> {
    /// Where the point comes from: [`CandidateSource::Lp`], [`CandidateSource::Pseudo`] or
    /// [`CandidateSource::Relaxation`].
    #[must_use]
    pub const fn source(&self) -> CandidateSource {
        self.source
    }

    /// The node's depth in SCIP's search tree, zero at the root, where a tightening holds for the rest of the solve.
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.node.depth()
    }
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
        self.node.bounds(col)
    }

    /// SCIP narrows the column once the handler returns, with a reason as an inference of culmen's propagator.
    ///
    /// # Errors
    ///
    /// Besides the errors the trait states, [`ActionError::Refused`] for an end SCIP would read as infinite, or a
    /// column presolve multi-aggregated, whose bounds SCIP cannot change.
    fn tighten(&mut self, col: Col, bounds: Bounds, reason: Option<Reason>) -> Result<(), ActionError> {
        self.node.tighten(col, bounds, reason)
    }

    fn cut_off(&mut self) {
        self.node.cut_off();
    }

    /// SCIP cuts the node off once the handler returns, after handing the named bounds at the node to its conflict
    /// analysis as valid everywhere in the tree, when SCIP analyses conflicts there (never at the root).
    fn cut_off_because(&mut self, reasons: &[(Col, Bound)]) -> Result<(), ActionError> {
        self.node.cut_off_because(reasons)
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
        admit(&self.node.actions, NodeAction::Reduce)?;
        checked(&constraint, self.candidate.len(), self.node.infinity)?;
        self.node.actions.add_constraint(&constraint)
    }

    /// Refused in this release, after the checks every backend makes: SCIP takes children of a handler's own, which
    /// are not wired yet.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after any other action; otherwise [`ActionError::Refused`].
    fn branch(&mut self, children: Vec<Child>) -> Result<(), ActionError> {
        let _ = children;
        admit(&self.node.actions, NodeAction::Branch)?;
        Err(ActionError::Refused(
            "branching into children of the handler's own is not wired to SCIP in this release".to_owned(),
        ))
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
        // SCIP is being stopped: every node is closed until it does.
        if self.stopping() {
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
            progress,
            objective,
            candidate,
            source,
            node: Node::new(scip, self.cols, &priced, self.infinity, &active.involved),
        };
        let delivered = self.deliver(scip, |hooks| hooks.enforce(&mut ctx));
        (delivered == Delivered::Ran).then_some(ctx.node.actions)
    }

    /// Hands SCIP the actions an enforcement recorded and derives SCIP's result from what happened: a cutoff or an
    /// emptied domain cuts the node off; a new constraint the point violates asks SCIP to solve again; a bound SCIP
    /// moved reduces the domain; otherwise the point stays rejected.
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
        // SAFETY: the caller's contract; the actions were recorded at this node.
        let settled = unsafe { self.settle(scip, actions) }?;
        if settled == Settled::CutOff {
            return Ok(SCIP_Result_SCIP_CUTOFF);
        }
        // SAFETY: as above.
        if unsafe { self.add_new(scip, values, active, actions) }? {
            return Ok(SCIP_Result_SCIP_CONSADDED);
        }
        if settled == Settled::Reduced {
            return Ok(SCIP_Result_SCIP_REDUCEDDOM);
        }
        // SAFETY: as above.
        if unsafe { unresolvable(scip) } {
            return Err(ScipError::UnresolvedRejection);
        }
        Ok(SCIP_Result_SCIP_INFEASIBLE)
    }

    /// Adds to SCIP every constraint the actions record that the handler did not add before in this solve; whether
    /// one of them is violated by the point `values` by SCIP's tolerance.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// As [`Run::apply_actions`].
    unsafe fn add_new(
        &self,
        scip: *mut SCIP,
        values: &TiSlice<Col, f64>,
        active: &Active,
        actions: &NodeActions,
    ) -> Result<bool, ScipError> {
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
        Ok(violated)
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
            // The solve fails with the error once SCIP has stopped; the node is closed meanwhile.
            Err(error) => (SCIP_Result_SCIP_CUTOFF, run.stop_with(scip, error)),
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
