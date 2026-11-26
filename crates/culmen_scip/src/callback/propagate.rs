// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! culmen's propagator `culmen`: a [`ConstraintHandler`](culmen_core::ConstraintHandler)'s propagation and the
//! explanation of the bounds it tightened with a reason.
//!
//! The citations below are into SCIP 10.0.2's `src/scip/` as `scip-sys` builds it.
//!
//! # Why a propagator
//!
//! SCIP resolves an inferred bound during conflict analysis by asking whoever inferred it. A constraint handler's
//! inference names a constraint and is resolved through that constraint's handler (`conflict_graphanalysis.c:3248-3307`,
//! `cons.c:7493-7550`); the constraint handler of a culmen solve has no constraints, so it has nothing to name. A
//! propagator's inference names the propagator, whose `PROPRESPROP` SCIP asks (`conflict_graphanalysis.c:3310-3341`,
//! `prop.c:739-784`), and any plugin may name it. So every bound the handler tightens with a reason, in propagation or
//! in enforcement, is an inference of this propagator, and the handler's `explain` answers SCIP for it.
//!
//! # What SCIP calls, and what the handler is asked
//!
//! - `PROPEXEC`, in the propagation rounds of every node of a solve with a constraint handler (`propagating/culmen/freq`
//!   is 1 then and −1 otherwise), before, during and after the node's LP loop, and at the probing nodes SCIP's probing
//!   and strong branching create, asks `propagate`. Its priority is below zero, so SCIP's constraint handlers have
//!   propagated the node first (`solve.c:414-475`).
//! - `PROPRESPROP`, when SCIP's conflict analysis resolves a bound the handler tightened with a reason, asks `explain`.
//!   The bounds it names are kept until it answers: on [`Explained::Complete`] they go to SCIP's conflict analysis as
//!   they were just before the tightening (`SCIPaddConflictLb`/`Ub` at the bound change's index), and on
//!   [`Explained::Unknown`] they are dropped, since bounds handed to SCIP before an answer of `DIDNOTFIND` stay in its
//!   conflict queue, which a debug build of SCIP asserts against (`conflict_graphanalysis.c:3354-3361`).
//!
//! # SCIP's result, derived from what happened
//!
//! | Propagation | Result |
//! |---|---|
//! | no constraint handler, a callback of the run panicked or a plugin met an error, or the handler is busy with an outer call | `DIDNOTRUN` |
//! | `propagate` cut the node off, plainly or naming the bounds SCIP's conflict analysis is handed first | `CUTOFF` |
//! | a tightening emptied a domain, as SCIP reports it | `CUTOFF` |
//! | a tightening moved a bound, as SCIP reports it | `REDUCEDDOM` |
//! | otherwise | `DIDNOTFIND` |
//!
//! | Explanation | Result |
//! |---|---|
//! | no constraint handler, a callback of the run panicked or a plugin met an error, the handler is busy, the variable is no column's, SCIP asks for a bound stronger than it inferred, or `explain` answers `Unknown` | `DIDNOTFIND`, nothing named |
//! | `explain` answers `Complete` | `SUCCESS`, every bound it named handed to SCIP |

use core::ffi::c_int;
use core::ptr::{self, NonNull};

use culmen_core::{
    ActionError, Bound, Bounds, Col, Context, ExplainContext, Explained, Inferred, NodeActions, NodeContext, Progress,
    PropagateContext, Reason,
};
use scip_sys::{
    SCIP, SCIP_BDCHGIDX, SCIP_BOUNDTYPE, SCIP_BoundType_SCIP_BOUNDTYPE_LOWER, SCIP_PROP, SCIP_PROPTIMING,
    SCIP_PROPTIMING_ALWAYS, SCIP_RESULT, SCIP_RETCODE, SCIP_Result_SCIP_CUTOFF, SCIP_Result_SCIP_DIDNOTFIND,
    SCIP_Result_SCIP_DIDNOTRUN, SCIP_Result_SCIP_REDUCEDDOM, SCIP_Result_SCIP_SUCCESS, SCIP_Retcode_SCIP_OKAY,
    SCIP_VAR, SCIP_Varstatus_SCIP_VARSTATUS_COLUMN, SCIP_Varstatus_SCIP_VARSTATUS_LOOSE, SCIPgetVarLbAtIndex,
    SCIPgetVarUbAtIndex, SCIPincludePropBasic, SCIPisFeasGT, SCIPisFeasLT, SCIPpropGetData, SCIPsetPropResprop,
    SCIPvarGetStatus,
};

use super::constraint::Active;
use super::node::{Node, Settled, add_conflict_bound, end};
use super::{Delivered, Run, Slot, slot};
use crate::columns::Columns;
use crate::error::check;
use crate::{RetCode, ScipError};

/// The propagator's name, under which its parameters live (`propagating/culmen/…`).
const NAME: &core::ffi::CStr = c"culmen";

/// Below zero, so SCIP's constraint handlers propagate a node first, and above SCIP's expensive propagators (probing
/// at −100 000, OBBT at −1 000 000).
const PRIORITY: c_int = -10;

/// A node's domain the constraint handler narrows: the propagator's `PROPEXEC`.
///
/// SCIP takes the actions once the handler returns: a tightening ([`NodeContext::tighten`]), at this node and below,
/// and at the root for the rest of the solve, with a reason as an inference of culmen's propagator, which answers
/// SCIP's conflict analysis with the handler's [`explain`](culmen_core::ConstraintHandler::explain); a cutoff, plain or
/// with the bounds that make the node infeasible, which SCIP's conflict analysis may learn a conflict from
/// ([`NodeContext::cut_off_because`]). The node may be one of SCIP's probing nodes, whose domain is a trial.
#[derive(Debug)]
pub struct ScipPropagate<'a> {
    progress: Progress,
    node: Node<'a>,
}

impl ScipPropagate<'_> {
    /// The node's depth in SCIP's search tree, zero at the root, where a tightening holds for the rest of the solve.
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.node.depth()
    }
}

impl Context for ScipPropagate<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl NodeContext for ScipPropagate<'_> {
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

impl PropagateContext for ScipPropagate<'_> {}

/// A bound the constraint handler tightened with a reason, which SCIP's conflict analysis asks it to explain: the
/// propagator's `PROPRESPROP`.
///
/// Every bound it reads is as it was just before the tightening, at SCIP's index of the bound change; the bounds the
/// handler names reach SCIP only when it answers [`Explained::Complete`].
#[derive(Debug)]
pub struct ScipExplain<'a> {
    scip: *mut SCIP,
    progress: Progress,
    inferred: Inferred,
    reason: Reason,
    /// SCIP's index of the tightening's bound change.
    bdchgidx: *mut SCIP_BDCHGIDX,
    cols: Columns<'a>,
    /// The variables of the columns priced so far.
    priced: &'a [NonNull<SCIP_VAR>],
    infinity: f64,
    /// The bounds named so far, in order.
    named: Vec<(Col, Bound)>,
}

impl Context for ScipExplain<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl ExplainContext for ScipExplain<'_> {
    /// The column, the end and the bound SCIP asks to have explained, which may be weaker than the one tightened.
    fn inferred(&self) -> Inferred {
        self.inferred
    }

    fn reason(&self) -> Reason {
        self.reason
    }

    /// The bounds of the variable SCIP holds for the column at the index of the tightening's bound change, before it.
    fn bounds_before(&self, col: Col) -> Option<Bounds> {
        if col.index() >= self.cols.len() + self.priced.len() {
            return None;
        }
        // SAFETY: the context lives during a conflict analysis of the live instance holding the columns.
        let var = unsafe { self.cols.transformed(self.scip, col, self.priced) }.ok()?;
        // SAFETY: as above; `var` is a variable of the transformed problem and `bdchgidx` the index SCIP handed over.
        let (lower, upper) = unsafe {
            (
                SCIPgetVarLbAtIndex(self.scip, var, self.bdchgidx, 0),
                SCIPgetVarUbAtIndex(self.scip, var, self.bdchgidx, 0),
            )
        };
        Bounds::new(end(lower, self.infinity), end(upper, self.infinity)).ok()
    }

    fn because(&mut self, col: Col, end: Bound) -> Result<(), ActionError> {
        if col.index() >= self.cols.len() + self.priced.len() {
            return Err(ActionError::UnknownColumn { col });
        }
        self.named.push((col, end));
        Ok(())
    }
}

/// What SCIP's conflict analysis asks of a bound the handler tightened with a reason.
#[derive(Clone, Copy, Debug)]
struct Asked {
    /// The variable the tightening named, the column's transformed one.
    var: *mut SCIP_VAR,
    /// The handler's reason, as SCIP's `int`.
    info: c_int,
    /// The end of `var` tightened.
    boundtype: SCIP_BOUNDTYPE,
    bdchgidx: *mut SCIP_BDCHGIDX,
    /// The bound of SCIP's active variable for `var` that suffices to explain.
    relaxed: f64,
}

impl Run<'_, '_> {
    /// Asks the constraint handler's `propagate` at the current node and derives SCIP's result from what happened.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, propagating a node.
    unsafe fn propagate_handler(&self, scip: *mut SCIP, active: &Active) -> Result<SCIP_RESULT, ScipError> {
        let Some(actions) = self.ask_propagate(scip, active) else {
            return Ok(SCIP_Result_SCIP_DIDNOTRUN);
        };
        // SAFETY: the caller's contract; the actions were recorded at this node.
        Ok(match unsafe { self.settle(scip, &actions) }? {
            Settled::CutOff => SCIP_Result_SCIP_CUTOFF,
            Settled::Reduced => SCIP_Result_SCIP_REDUCEDDOM,
            Settled::Unchanged => SCIP_Result_SCIP_DIDNOTFIND,
        })
    }

    /// Asks the constraint handler's `propagate`; the actions it recorded, `None` when it did not run.
    fn ask_propagate(&self, scip: *mut SCIP, active: &Active) -> Option<NodeActions> {
        let progress = self.progress(scip);
        let priced = self.priced_vars.borrow();
        let mut ctx = ScipPropagate {
            progress,
            node: Node::new(scip, self.cols, &priced, self.infinity, &active.involved),
        };
        let delivered = self.deliver(scip, |hooks| hooks.propagate(&mut ctx));
        (delivered == Delivered::Ran).then_some(ctx.node.actions)
    }

    /// Asks the constraint handler to explain a bound it tightened with a reason, hands SCIP the bounds it named when it
    /// answers that they are complete, and derives SCIP's result.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, in a conflict analysis, and `asked` what SCIP handed over.
    unsafe fn explain_handler(&self, scip: *mut SCIP, active: &Active, asked: Asked) -> Result<SCIP_RESULT, ScipError> {
        let col = {
            let priced = self.priced_vars.borrow();
            // SAFETY: the caller's contract; the instance is transformed while it analyses a conflict.
            unsafe { active.column_of(scip, self.cols, &priced, asked.var) }?
        };
        // SAFETY: as above.
        let Some(inferred) = col.and_then(|col| unsafe { inferred(scip, col, asked) }) else {
            return Ok(SCIP_Result_SCIP_DIDNOTFIND);
        };
        let reason = Reason(asked.info.cast_unsigned());
        let Some(named) = self.ask_explain(scip, inferred, reason, asked.bdchgidx) else {
            return Ok(SCIP_Result_SCIP_DIDNOTFIND);
        };
        let vars = self.vars_of_named(scip, &named)?;
        for (var, (_, side)) in vars.into_iter().zip(named) {
            // SAFETY: as above; the analysis asked for this bound change's reasons, whose index SCIP handed over.
            unsafe { add_conflict_bound(scip, var, side, asked.bdchgidx) }?;
        }
        Ok(SCIP_Result_SCIP_SUCCESS)
    }

    /// Asks the constraint handler's `explain`; the bounds it named when it answered [`Explained::Complete`], `None`
    /// when it answered [`Explained::Unknown`] or did not run.
    fn ask_explain(
        &self,
        scip: *mut SCIP,
        inferred: Inferred,
        reason: Reason,
        bdchgidx: *mut SCIP_BDCHGIDX,
    ) -> Option<Vec<(Col, Bound)>> {
        let progress = self.progress(scip);
        let priced = self.priced_vars.borrow();
        let mut ctx = ScipExplain {
            scip,
            progress,
            inferred,
            reason,
            bdchgidx,
            cols: self.cols,
            priced: &priced,
            infinity: self.infinity,
            named: Vec::new(),
        };
        let mut answer = Explained::Unknown;
        let delivered = self.deliver(scip, |hooks| answer = hooks.explain(&mut ctx));
        (delivered == Delivered::Ran && answer == Explained::Complete).then_some(ctx.named)
    }

    /// The transformed variable of each named column, read before SCIP is called again.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP holds no transformed variable for a column.
    fn vars_of_named(&self, scip: *mut SCIP, named: &[(Col, Bound)]) -> Result<Vec<NonNull<SCIP_VAR>>, ScipError> {
        let priced = self.priced_vars.borrow();
        named
            .iter()
            .map(|&(col, _)| {
                // SAFETY: the run's instance is live and transformed while it analyses a conflict; `because` checked
                // the column against the problem's.
                let var = unsafe { self.cols.transformed(scip, col, &priced) }?;
                NonNull::new(var).ok_or(ScipError::Call {
                    call: "SCIPgetTransformedVar",
                    code: RetCode::InvalidData,
                })
            })
            .collect()
    }
}

/// The bound of `col` SCIP asks to have explained, in the column's units; `None` when SCIP asks for one stronger than
/// the bound it inferred, which no explanation of the tightening could imply.
///
/// SCIP hands `PROPRESPROP` the relaxed bound of the active variable the change landed on, without translating it to
/// the variable the tightening named, as it does for a constraint's inference (`conflict_graphanalysis.c:3283-3305`
/// against `3310-3341`). So for a column whose variable is active, the relaxed bound is the column's own; for one
/// aggregated or negated onto another, the bound the tightening set is asked for, which implies the relaxed one.
///
/// # Safety
///
/// `scip` is a live instance in a conflict analysis, and `asked` what SCIP handed over for the column's variable.
unsafe fn inferred(scip: *mut SCIP, col: Col, asked: Asked) -> Option<Inferred> {
    let lower = asked.boundtype == SCIP_BoundType_SCIP_BOUNDTYPE_LOWER;
    // SAFETY: the caller's contract; SCIP reads the variable's status and its bound just after the change.
    let (status, set) = unsafe {
        let set = if lower {
            SCIPgetVarLbAtIndex(scip, asked.var, asked.bdchgidx, 1)
        } else {
            SCIPgetVarUbAtIndex(scip, asked.var, asked.bdchgidx, 1)
        };
        (SCIPvarGetStatus(asked.var), set)
    };
    let active = status == SCIP_Varstatus_SCIP_VARSTATUS_LOOSE || status == SCIP_Varstatus_SCIP_VARSTATUS_COLUMN;
    let value = if active { asked.relaxed } else { set };
    // SAFETY: as above; the comparison reads SCIP's tolerance.
    let stronger = unsafe {
        if lower {
            SCIPisFeasGT(scip, value, set) != 0
        } else {
            SCIPisFeasLT(scip, value, set) != 0
        }
    };
    if stronger || !value.is_finite() {
        return None;
    }
    let end = if lower { Bound::Lower } else { Bound::Upper };
    Some(Inferred { col, end, value })
}

/// Includes the propagator, off until a solve with a constraint handler turns it on.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(super) unsafe fn include(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    let mut prop: *mut SCIP_PROP = ptr::null_mut();
    // SAFETY: the instance is live and in its init stage; the name and description are NUL-terminated strings SCIP
    // copies; the callbacks treat their arguments as SCIP passes them, and the data is the slot, which outlives SCIP.
    // A frequency of −1 never calls it; the solve with a constraint handler sets it to 1.
    check("SCIPincludePropBasic", unsafe {
        SCIPincludePropBasic(
            scip,
            &raw mut prop,
            NAME.as_ptr(),
            c"the propagation and explanations of a culmen solve's constraint handler".as_ptr(),
            PRIORITY,
            -1,
            0,
            SCIP_PROPTIMING_ALWAYS,
            Some(propexec),
            slot.as_ptr().cast(),
        )
    })?;
    // SAFETY: `prop` is the plugin just included in the live instance; the callback treats its arguments as SCIP
    // passes them.
    check("SCIPsetPropResprop", unsafe {
        SCIPsetPropResprop(scip, prop, Some(propresprop))
    })?;
    // SAFETY: the slot lives at one address from the instance's creation until after SCIP is freed.
    unsafe { slot.as_ref() }.prop.set(prop);
    Ok(())
}

/// The plugin's run and its constraint handler's state, `None` when no run with a constraint handler is installed.
///
/// # Safety
///
/// `prop` is the plugin, called by SCIP.
unsafe fn prop_run<'s>(scip: *mut SCIP, prop: *mut SCIP_PROP) -> Option<(&'s Run<'s, 's>, &'s Active)> {
    // SAFETY: the plugin's data is its slot.
    let slot = unsafe { slot::<_>(SCIPpropGetData(prop)) }?;
    // SAFETY: SCIP calls the plugin during the solve the run is installed for, or with none installed.
    let run = unsafe { slot.run() }?;
    run.pass_stop(scip);
    if run.stopping() {
        return None;
    }
    run.constraint.as_ref().map(|active| (run, active))
}

/// `PROPEXEC`: the constraint handler's propagation of the current node.
unsafe extern "C" fn propexec(
    scip: *mut SCIP,
    prop: *mut SCIP_PROP,
    _proptiming: SCIP_PROPTIMING,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    // SAFETY: SCIP calls the plugin's propagation with itself and a result to write.
    let (found, code) = match unsafe { prop_run(scip, prop) } {
        // SAFETY: SCIP propagates a node of the solve.
        Some((run, active)) => match unsafe { run.propagate_handler(scip, active) } {
            Ok(found) => (found, SCIP_Retcode_SCIP_OKAY),
            // The solve fails with the error; the result changes nothing should SCIP read it.
            Err(error) => (SCIP_Result_SCIP_DIDNOTRUN, run.fail(error)),
        },
        None => (SCIP_Result_SCIP_DIDNOTRUN, SCIP_Retcode_SCIP_OKAY),
    };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(found) };
    code
}

/// `PROPRESPROP`: the constraint handler's explanation of a bound it tightened with a reason.
unsafe extern "C" fn propresprop(
    scip: *mut SCIP,
    prop: *mut SCIP_PROP,
    infervar: *mut SCIP_VAR,
    inferinfo: c_int,
    boundtype: SCIP_BOUNDTYPE,
    bdchgidx: *mut SCIP_BDCHGIDX,
    relaxedbd: f64,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    let asked = Asked {
        var: infervar,
        info: inferinfo,
        boundtype,
        bdchgidx,
        relaxed: relaxedbd,
    };
    // SAFETY: SCIP calls the plugin's resolution with itself and a result to write.
    let (found, code) = match unsafe { prop_run(scip, prop) } {
        // SAFETY: SCIP analyses a conflict of the solve and hands over the tightening's variable and index.
        Some((run, active)) => match unsafe { run.explain_handler(scip, active, asked) } {
            Ok(found) => (found, SCIP_Retcode_SCIP_OKAY),
            Err(error) => (SCIP_Result_SCIP_DIDNOTFIND, run.fail(error)),
        },
        None => (SCIP_Result_SCIP_DIDNOTFIND, SCIP_Retcode_SCIP_OKAY),
    };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(found) };
    code
}
