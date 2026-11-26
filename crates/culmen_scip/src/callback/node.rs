// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node of SCIP's search as the constraint handler's enforcement and propagation see it: each column's local bounds,
//! the checked record of the handler's actions, and handing those actions to SCIP once the handler returns, as bound
//! changes and as the conflict analysis of a cutoff.
//!
//! The citations below are into SCIP 10.0.2's `src/scip/` as `scip-sys` builds it.
//!
//! # Which columns SCIP can write
//!
//! A bound change on a column goes to its transformed variable, which presolve may have fixed, aggregated, negated or
//! multi-aggregated. SCIP translates a change on an aggregated or negated variable to its active variable
//! (`tree.c:1992`), decides one on a fixed variable before it reaches the tree, as redundant or infeasible
//! (`scip_var.c:7618-7632`), and refuses one on a multi-aggregated variable with `SCIPABORT`, which ends a debug build of
//! SCIP (`tree.c:1994-1998`); the stored bounds of a multi-aggregated variable are not kept up to date either
//! (`var.c:12133-12137`: it is no parent variable). So a tightening of a column whose variable is multi-aggregated, or
//! resolves to one, is refused when the handler asks for it, a fixed one is decided here as SCIP would, and only an
//! active, aggregated or negated one reaches SCIP.
//!
//! # A tightening, with and without a reason
//!
//! A tightening with a [`Reason`] is `SCIPinferVarLbProp`/`SCIPinferVarUbProp` naming culmen's propagator, so SCIP's
//! conflict analysis asks the propagator's `PROPRESPROP`, and so the handler's `explain`, which bounds it rested on
//! (`conflict_graphanalysis.c:3310-3341`); one without is `SCIPtightenVarLb`/`SCIPtightenVarUb`, which SCIP keeps as a
//! decision it cannot resolve (`conflict_graphanalysis.c:3526-3536`), weaker for conflicts but never wrong. SCIP
//! reports whether a bound moved and whether the domain became empty; the result of the call follows from that, never
//! from what the handler asked for.
//!
//! # The conflict analysis of a cutoff
//!
//! A cutoff naming bounds hands them to SCIP's conflict analysis as the bounds that make the node infeasible
//! (`SCIPinitConflictAnalysis`, `SCIPaddConflictLb`/`Ub` at the node's bounds, `SCIPanalyzeConflict`), with valid depth
//! zero: the handler's rule and the program hold everywhere, so the named bounds exclude every accepted point anywhere
//! in the tree. As SCIP's own handlers do (`cons_linear.c:5196`), the analysis runs only while solving or probing and
//! when SCIP finds it applicable, which it never is at the root (`scip_conflict.c:320`); a plain cutoff is the rest.

use core::ffi::{c_int, c_uint};
use core::ptr::{self, NonNull};

use culmen_core::{ActionError, Bound, Bounds, Col, Involved, NodeAction, NodeActions, Reason};
use scip_sys::{
    SCIP, SCIP_BDCHGIDX, SCIP_ConflictType_SCIP_CONFTYPE_PROPAGATION, SCIP_PROP, SCIP_Stage_SCIP_STAGE_PRESOLVING,
    SCIP_Stage_SCIP_STAGE_SOLVING, SCIP_VAR, SCIP_Varstatus_SCIP_VARSTATUS_COLUMN, SCIP_Varstatus_SCIP_VARSTATUS_FIXED,
    SCIP_Varstatus_SCIP_VARSTATUS_LOOSE, SCIP_Varstatus_SCIP_VARSTATUS_MULTAGGR, SCIPaddConflictLb, SCIPaddConflictUb,
    SCIPanalyzeConflict, SCIPcomputeVarLbLocal, SCIPcomputeVarUbLocal, SCIPgetDepth, SCIPgetStage, SCIPinProbing,
    SCIPinferVarLbProp, SCIPinferVarUbProp, SCIPinitConflictAnalysis, SCIPisConflictAnalysisApplicable, SCIPisFeasGT,
    SCIPisFeasLT, SCIPtightenVarLb, SCIPtightenVarUb, SCIPvarGetProbvar, SCIPvarGetStatus,
};

use super::Run;
use crate::columns::Columns;
use crate::error::check;
use crate::{RetCode, ScipError};

/// A node's domain and the handler's actions there, as an enforcement or a propagation context holds them.
#[derive(Debug)]
pub(super) struct Node<'a> {
    pub(super) scip: *mut SCIP,
    pub(super) cols: Columns<'a>,
    /// The variables of the columns priced so far.
    pub(super) priced: &'a [NonNull<SCIP_VAR>],
    /// SCIP's infinity: a number of at least this magnitude is infinite to SCIP.
    pub(super) infinity: f64,
    /// The columns the handler may write.
    pub(super) involved: &'a Involved,
    pub(super) actions: NodeActions,
}

impl<'a> Node<'a> {
    /// A node of the live instance `scip`, solving, nothing recorded yet.
    pub(super) fn new(
        scip: *mut SCIP,
        cols: Columns<'a>,
        priced: &'a [NonNull<SCIP_VAR>],
        infinity: f64,
        involved: &'a Involved,
    ) -> Self {
        Self {
            scip,
            cols,
            priced,
            infinity,
            involved,
            actions: NodeActions::new(involved, cols.len() + priced.len()),
        }
    }

    /// The number of columns the problem holds now, the priced ones included.
    fn num_cols(&self) -> usize {
        self.cols.len() + self.priced.len()
    }

    /// The node's depth in SCIP's tree, zero at the root.
    pub(super) fn depth(&self) -> u32 {
        // SAFETY: a node context lives during a callback of the live instance, solving or probing, where SCIP reports
        // the depth.
        u32::try_from(unsafe { SCIPgetDepth(self.scip) }).unwrap_or(0)
    }

    /// The local bounds of the variable SCIP holds for the column at this node; for a column presolve
    /// multi-aggregated, the bounds its aggregation's variables give. `None` for a column the problem lacks.
    pub(super) fn bounds(&self, col: Col) -> Option<Bounds> {
        if col.index() >= self.num_cols() {
            return None;
        }
        // SAFETY: the context lives during a callback of the live instance holding the columns, solving or probing.
        let var = unsafe { self.cols.transformed(self.scip, col, self.priced) }.ok()?;
        // SAFETY: as above; `var` is a variable of the transformed problem, whose local bounds SCIP computes.
        let (lower, upper) = unsafe {
            (
                SCIPcomputeVarLbLocal(self.scip, var),
                SCIPcomputeVarUbLocal(self.scip, var),
            )
        };
        Bounds::new(end(lower, self.infinity), end(upper, self.infinity)).ok()
    }

    /// Records narrowing `col` to `bounds`, after the checks every backend makes and SCIP's own.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`], [`ActionError::UnknownColumn`] or [`ActionError::NotInvolved`] as
    /// [`NodeActions::tighten`] states; [`ActionError::Refused`] for an end SCIP would read as infinite, or a column
    /// SCIP cannot change.
    pub(super) fn tighten(&mut self, col: Col, bounds: Bounds, reason: Option<Reason>) -> Result<(), ActionError> {
        admit(&self.actions, NodeAction::Reduce)?;
        self.involved.check(col, self.num_cols())?;
        self.writable(col, bounds)?;
        self.actions.tighten(col, bounds, reason)
    }

    /// Refuses a tightening SCIP cannot take: an end SCIP reads as infinite, or a column whose variable presolve
    /// multi-aggregated.
    ///
    /// # Errors
    ///
    /// [`ActionError::Refused`].
    fn writable(&self, col: Col, bounds: Bounds) -> Result<(), ActionError> {
        let ends = [bounds.lower(), bounds.upper()];
        if let Some(end) = ends
            .into_iter()
            .find(|end| end.is_finite() && end.abs() >= self.infinity)
        {
            return Err(ActionError::Refused(format!(
                "the bound {end} is one SCIP reads as infinite, from {} on",
                self.infinity
            )));
        }
        // SAFETY: the context lives during a callback of the live instance holding the columns, solving or probing.
        let var = unsafe { self.cols.transformed(self.scip, col, self.priced) }
            .map_err(|error| ActionError::Refused(format!("SCIP holds no variable for {col:?}: {error}")))?;
        // SAFETY: as above; `var` is a variable of the transformed problem.
        if unsafe { standing(var) } == Standing::Unwritable {
            return Err(ActionError::Refused(format!(
                "presolve multi-aggregated {col:?}, so SCIP cannot change its bounds"
            )));
        }
        Ok(())
    }

    /// Records that the node is cut off.
    pub(super) fn cut_off(&mut self) {
        self.actions.cut_off();
    }

    /// Records that the node is cut off because of the bounds `reasons` name at the node.
    ///
    /// # Errors
    ///
    /// As [`NodeActions::cut_off_because`].
    pub(super) fn cut_off_because(&mut self, reasons: &[(Col, Bound)]) -> Result<(), ActionError> {
        self.actions.cut_off_because(reasons)
    }
}

/// Refuses an action of kind `action` that the actions recorded exclude, as [`NodeActions`] would, before any check of
/// its arguments.
///
/// # Errors
///
/// [`ActionError::Exclusive`].
pub(super) fn admit(actions: &NodeActions, action: NodeAction) -> Result<(), ActionError> {
    match (actions.action(), action) {
        (None, _) | (Some(NodeAction::Reduce), NodeAction::Reduce | NodeAction::CutOff) => Ok(()),
        (Some(NodeAction::Branch), NodeAction::CutOff) => Ok(()),
        (Some(after), action) => Err(ActionError::Exclusive { action, after }),
    }
}

/// A bound SCIP reports, an end of at least SCIP's infinity in magnitude as infinite.
pub(super) fn end(value: f64, infinity: f64) -> f64 {
    if value.abs() >= infinity {
        value.signum() * f64::INFINITY
    } else {
        value
    }
}

/// What a bound change on a variable reaches in SCIP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Standing {
    /// An active variable, or one aggregated or negated onto one: SCIP takes the change.
    Writable,
    /// A fixed variable, or one resolving to one: the change is redundant or empties the domain.
    Fixed,
    /// A multi-aggregated variable, or one resolving to one: SCIP aborts on the change.
    Unwritable,
}

/// Where a bound change on `var` lands: SCIP resolves it through `SCIPvarGetProbvarBound` to the variable
/// `SCIPvarGetProbvar` names (`tree.c:1990-1999`), but reads the old bounds on `var` itself, which SCIP does not keep for
/// a multi-aggregated variable even of one term.
///
/// # Safety
///
/// `var` is a variable of a live instance's transformed problem.
unsafe fn standing(var: *mut SCIP_VAR) -> Standing {
    // SAFETY: the caller vouches for `var`; SCIP resolves it within its own problem.
    let (status, active) = unsafe { (SCIPvarGetStatus(var), SCIPvarGetProbvar(var)) };
    if status == SCIP_Varstatus_SCIP_VARSTATUS_MULTAGGR || active.is_null() {
        return Standing::Unwritable;
    }
    // SAFETY: `active` is the variable SCIP resolved `var` to, of the same problem.
    let active = unsafe { SCIPvarGetStatus(active) };
    if active == SCIP_Varstatus_SCIP_VARSTATUS_LOOSE || active == SCIP_Varstatus_SCIP_VARSTATUS_COLUMN {
        Standing::Writable
    } else if active == SCIP_Varstatus_SCIP_VARSTATUS_FIXED {
        Standing::Fixed
    } else {
        Standing::Unwritable
    }
}

/// What came of handing a node's actions to SCIP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Settled {
    /// The node is cut off: the handler said so, or a tightening emptied a domain.
    CutOff,
    /// A tightening moved a bound.
    Reduced,
    /// Nothing changed.
    Unchanged,
}

impl Run<'_, '_> {
    /// Hands SCIP the cutoff or the tightenings a node's actions record and says what came of them; the constraints and
    /// children they record are the caller's.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call, or culmen's propagator is missing for a tightening with a reason.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, enforcing or propagating at the node the actions were recorded at.
    pub(super) unsafe fn settle(&self, scip: *mut SCIP, actions: &NodeActions) -> Result<Settled, ScipError> {
        if let Some(reasons) = actions.cutoff() {
            let vars = self.vars_of(scip, reasons.iter().map(|&(col, _)| col))?;
            let named: Vec<_> = vars.into_iter().zip(reasons.iter().map(|&(_, end)| end)).collect();
            // SAFETY: the caller's contract; each variable is a transformed one of the instance.
            unsafe { analyze_cutoff(scip, &named) }?;
            return Ok(Settled::CutOff);
        }
        let tightenings = actions.tightenings();
        let vars = self.vars_of(scip, tightenings.iter().map(|&(col, _, _)| col))?;
        let mut reduced = false;
        for (var, &(_, bounds, reason)) in vars.into_iter().zip(tightenings) {
            // SAFETY: as above.
            let wrote = unsafe { self.write(scip, var, bounds, reason) }?;
            if wrote.infeasible {
                return Ok(Settled::CutOff);
            }
            reduced |= wrote.tightened;
        }
        Ok(if reduced { Settled::Reduced } else { Settled::Unchanged })
    }

    /// The transformed variable of each column, the priced ones included, read before SCIP is called again, so no
    /// borrow is held while it runs.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP holds no transformed variable for a column.
    ///
    /// # Panics
    ///
    /// When a column is beyond the problem's, which a checked action never names.
    fn vars_of(&self, scip: *mut SCIP, cols: impl Iterator<Item = Col>) -> Result<Vec<NonNull<SCIP_VAR>>, ScipError> {
        let priced = self.priced_vars.borrow();
        cols.map(|col| {
            // SAFETY: the run's instance is live and transformed while a node's actions are settled.
            let var = unsafe { self.cols.transformed(scip, col, &priced) }?;
            NonNull::new(var).ok_or(ScipError::Call {
                call: "SCIPgetTransformedVar",
                code: RetCode::InvalidData,
            })
        })
        .collect()
    }

    /// Narrows `var` to `bounds` at the current node, through culmen's propagator with `reason`, and reports what SCIP
    /// did.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call, for a multi-aggregated variable, which the context refuses before,
    /// or when culmen's propagator is missing for a reason.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, solving or probing, and `var` a variable of its transformed problem.
    unsafe fn write(
        &self,
        scip: *mut SCIP,
        var: NonNull<SCIP_VAR>,
        bounds: Bounds,
        reason: Option<Reason>,
    ) -> Result<Wrote, ScipError> {
        let var = var.as_ptr();
        // SAFETY: the caller vouches for `var`.
        match unsafe { standing(var) } {
            Standing::Writable => {}
            // SAFETY: as above, at a node, where SCIP computes the variable's local bounds.
            Standing::Fixed => return Ok(unsafe { fixed(scip, var, bounds) }),
            Standing::Unwritable => {
                return Err(ScipError::Call {
                    call: "SCIPinferVarLbProp",
                    code: RetCode::InvalidData,
                });
            }
        }
        let prop = match reason {
            None => None,
            Some(_) => Some(NonNull::new(self.slot.prop.get()).ok_or(ScipError::Call {
                call: "SCIPfindProp",
                code: RetCode::PluginNotFound,
            })?),
        };
        let mut wrote = Wrote::default();
        // An infinite end narrows nothing: `Bounds` holds no lower end at +∞ and no upper end at −∞.
        let ends = [(Bound::Lower, bounds.lower()), (Bound::Upper, bounds.upper())];
        for (side, value) in ends.into_iter().filter(|(_, value)| value.is_finite()) {
            let info = reason.map_or(0, |reason| reason.0.cast_signed());
            // SAFETY: the caller vouches for the instance, its stage and `var`; `prop` is culmen's propagator.
            let once = unsafe { change(scip, var, side, value, prop.map(|prop| (prop.as_ptr(), info))) }?;
            wrote.tightened |= once.tightened;
            if once.infeasible {
                wrote.infeasible = true;
                break;
            }
        }
        Ok(wrote)
    }
}

/// What SCIP did with a bound change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Wrote {
    /// The domain became empty.
    infeasible: bool,
    /// A bound moved.
    tightened: bool,
}

/// What narrowing a fixed variable to `bounds` comes to, decided as SCIP decides it before it reaches the tree
/// (`scip_var.c:7618-7632`): nothing when the value is inside, an empty domain otherwise.
///
/// # Safety
///
/// `scip` is a live instance at a node and `var` a variable of its transformed problem.
unsafe fn fixed(scip: *mut SCIP, var: *mut SCIP_VAR, bounds: Bounds) -> Wrote {
    // SAFETY: the caller's contract; SCIP computes the local bounds of any variable and compares by its tolerance.
    let infeasible = unsafe {
        let (lower, upper) = (SCIPcomputeVarLbLocal(scip, var), SCIPcomputeVarUbLocal(scip, var));
        (bounds.lower().is_finite() && SCIPisFeasGT(scip, bounds.lower(), upper) != 0)
            || (bounds.upper().is_finite() && SCIPisFeasLT(scip, bounds.upper(), lower) != 0)
    };
    Wrote {
        infeasible,
        tightened: false,
    }
}

/// Moves one end of `var` to `value`, as an inference of `prop` with the information `info` or, without, as a plain
/// tightening.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses the call.
///
/// # Safety
///
/// `scip` is a live instance, solving or probing, `var` an active, aggregated or negated variable of its transformed
/// problem, and `prop` a propagator of it.
unsafe fn change(
    scip: *mut SCIP,
    var: *mut SCIP_VAR,
    side: Bound,
    value: f64,
    prop: Option<(*mut SCIP_PROP, c_int)>,
) -> Result<Wrote, ScipError> {
    let (mut infeasible, mut tightened): (c_uint, c_uint) = (0, 0);
    let (inf, tight) = (&raw mut infeasible, &raw mut tightened);
    // SAFETY: the caller's contract; the flags are locals SCIP writes, and `force` is off, so SCIP skips a change below
    // its bound strengthening tolerance.
    let (call, code) = unsafe {
        match (side, prop) {
            (Bound::Lower, Some((prop, info))) => (
                "SCIPinferVarLbProp",
                SCIPinferVarLbProp(scip, var, value, prop, info, 0, inf, tight),
            ),
            (Bound::Upper, Some((prop, info))) => (
                "SCIPinferVarUbProp",
                SCIPinferVarUbProp(scip, var, value, prop, info, 0, inf, tight),
            ),
            (Bound::Lower, None) => ("SCIPtightenVarLb", SCIPtightenVarLb(scip, var, value, 0, inf, tight)),
            (Bound::Upper, None) => ("SCIPtightenVarUb", SCIPtightenVarUb(scip, var, value, 0, inf, tight)),
        }
    };
    check(call, code)?;
    Ok(Wrote {
        infeasible: infeasible != 0,
        tightened: tightened != 0,
    })
}

/// Adds one end of `var` to SCIP's conflict analysis: as it was at the bound change `bdchgidx`, or at the current node
/// for null.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses the call.
///
/// # Safety
///
/// `scip` is a live instance, solving or probing, inside a conflict analysis, `var` a variable of its transformed
/// problem and `bdchgidx` null or a bound change index SCIP handed over.
pub(super) unsafe fn add_conflict_bound(
    scip: *mut SCIP,
    var: NonNull<SCIP_VAR>,
    side: Bound,
    bdchgidx: *mut SCIP_BDCHGIDX,
) -> Result<(), ScipError> {
    // SAFETY: the caller's contract.
    unsafe {
        match side {
            Bound::Lower => check("SCIPaddConflictLb", SCIPaddConflictLb(scip, var.as_ptr(), bdchgidx)),
            Bound::Upper => check("SCIPaddConflictUb", SCIPaddConflictUb(scip, var.as_ptr(), bdchgidx)),
        }
    }
}

/// Hands SCIP's conflict analysis the bounds that make the current node infeasible, valid everywhere in the tree, when
/// SCIP can analyse a conflict here; does nothing for none.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance at a node, solving or probing, and each variable one of its transformed problem.
unsafe fn analyze_cutoff(scip: *mut SCIP, named: &[(NonNull<SCIP_VAR>, Bound)]) -> Result<(), ScipError> {
    if named.is_empty() {
        return Ok(());
    }
    // SAFETY: the caller's contract; SCIP reports its stage, whether it probes and whether it analyses conflicts in
    // the stages a node exists in.
    let applicable = unsafe {
        let stage = SCIPgetStage(scip);
        (stage == SCIP_Stage_SCIP_STAGE_SOLVING
            || (stage == SCIP_Stage_SCIP_STAGE_PRESOLVING && SCIPinProbing(scip) != 0))
            && SCIPisConflictAnalysisApplicable(scip) != 0
    };
    if !applicable {
        return Ok(());
    }
    // SAFETY: as above; the analysis starts here, is handed each bound at the current node, and ends here.
    unsafe {
        check(
            "SCIPinitConflictAnalysis",
            SCIPinitConflictAnalysis(scip, SCIP_ConflictType_SCIP_CONFTYPE_PROPAGATION, 0),
        )?;
        for &(var, side) in named {
            add_conflict_bound(scip, var, side, ptr::null_mut())?;
        }
        check("SCIPanalyzeConflict", SCIPanalyzeConflict(scip, 0, ptr::null_mut()))
    }
}
