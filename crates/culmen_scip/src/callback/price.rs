// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pricing: a variable pricer with a reduced-cost and a Farkas callback (`PRICERREDCOST`, `PRICERFARKAS`).
//!
//! The pricer is included in every instance, delayed, so SCIP prices the variables it holds before it asks the
//! callback, and inactive. A solve whose callback [`prices`](culmen_core::Callback::prices) activates it and marks
//! every program row's constraint modifiable, as SCIP requires of a constraint a pricer adds variables to; a solve that
//! does not deactivates it and leaves every constraint unmodifiable, so SCIP neither calls it nor gives up a reduction
//! for it. A modifiable linear constraint is never presolved, upgraded or used to tighten bounds, so it stays a linear
//! constraint whose dual SCIP reports.
//!
//! SCIP minimizes `s·c` for the program's cost `c`, with `s = 1` when the program minimizes and `s = -1` when it
//! maximizes; SCIP's dual of a row in its transformed problem is therefore `s` times the program's, which the context
//! hands out, and a priced variable's objective in SCIP's problem is `s·c`. A Farkas multiplier does not depend on
//! the sense, as SCIP's transformation negates only the objective, and is handed out as SCIP reports it.
//!
//! Each column the callback adds is checked against what SCIP can hold (a number SCIP reads as infinite, a name
//! with a NUL byte) and then by the solve's [`Priced`], which numbers it; once the callback returns, it becomes a
//! variable of SCIP's transformed problem (`SCIPcreateVar`, `SCIPaddPricedVar`) with its coefficients in the
//! transformed constraints of its rows (`SCIPaddCoefLinear`). The run keeps one reference to each such variable,
//! which the instance holds after the solve until it frees the transformed problem, so every point read afterwards
//! has a value for every priced column.
//!
//! | Mode | the callback | SCIP's result |
//! |---|---|---|
//! | reduced cost | returned | `SUCCESS`, with the bound it reported as SCIP's lower bound |
//! | reduced cost | branched early | `SUCCESS` with `stopearly` |
//! | Farkas | returned | `SUCCESS` |
//! | Farkas | branched early | `DIDNOTRUN` |
//! | either | busy with an outer event, or panicked | `DIDNOTRUN` |
//! | reduced cost, on a node LP that is not optimal | not called | `DIDNOTRUN` |
//!
//! SCIP can stop pricing early only where it can branch: at a node without an unfixed integer variable, as in a
//! program without integer columns, an early stop fails the solve with [`ScipError::BranchEarly`] instead.

use core::ffi::{CStr, c_uint};
use core::ptr::{self, NonNull};
use std::collections::BTreeMap;
use std::ffi::CString;

use culmen_core::{
    ActionError, Bounds, Col, Integrality, NewCol, PriceContext, PriceMode, Priced, Progress, Row, Sense, TiSlice,
    TiVec,
};
use scip_sys::{
    SCIP, SCIP_CONS, SCIP_LPSolStat_SCIP_LPSOLSTAT_OPTIMAL, SCIP_PRICER, SCIP_RESULT, SCIP_RETCODE,
    SCIP_Result_SCIP_DIDNOTRUN, SCIP_Result_SCIP_SUCCESS, SCIP_Retcode_SCIP_OKAY, SCIP_VAR,
    SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS, SCIP_Vartype_SCIP_VARTYPE_INTEGER, SCIPaddCoefLinear, SCIPaddPricedVar,
    SCIPcreateVar, SCIPdualfeastol, SCIPgetDualfarkasLinear, SCIPgetDualsolLinear, SCIPgetLPObjval, SCIPgetLPSolstat,
    SCIPgetNPseudoBranchCands, SCIPgetTransformedCons, SCIPincludePricerBasic, SCIPpricerGetData, SCIPretransformObj,
    SCIPtransformObj, SCIPvarGetLbLocal, SCIPvarGetUbLocal,
};

use super::{Delivered, Run, Slot, slot};
use crate::error::check;
use crate::{RetCode, ScipError};

/// The name the pricer is included under, which the instance finds it by.
pub(crate) const NAME: &CStr = c"culmen";

/// A node LP SCIP offers for pricing: the pricer's `PRICERREDCOST` or `PRICERFARKAS`.
///
/// [`PriceContext::tolerance`] is SCIP's dual feasibility tolerance (`numerics/dualfeastol`), the magnitude up to
/// which SCIP counts a reduced cost as zero.
#[derive(Debug)]
pub struct ScipPrice<'a> {
    scip: *mut SCIP,
    progress: Progress,
    mode: PriceMode,
    sense: Sense,
    duals: &'a TiSlice<Row, f64>,
    tolerance: f64,
    infinity: f64,
    /// The variables of the columns priced before this round, after the program's columns.
    priced_vars: &'a [NonNull<SCIP_VAR>],
    /// The original variables of the program's columns.
    vars: &'a TiSlice<Col, NonNull<SCIP_VAR>>,
    priced: &'a mut Priced,
    round: &'a mut Round,
}

/// What a callback did in one round.
#[derive(Debug, Default)]
struct Round {
    /// The columns it added, in order.
    added: Vec<Col>,
    /// The bound it reported last, in the program's sense.
    bound: Option<f64>,
    /// Whether it ended pricing early.
    early: bool,
}

impl ScipPrice<'_> {
    /// The node's local bounds of a column, as branching and propagation left them on the variable SCIP holds for it;
    /// for a column added in this round, the bounds it was added with; `None` for an id the problem lacks.
    #[must_use]
    pub fn local_bounds(&self, col: Col) -> Option<Bounds> {
        let base = self.vars.len();
        let var = match col.index().checked_sub(base) {
            None => {
                // SAFETY: the context lives during a pricer callback, solving, of the instance holding the variable.
                unsafe { crate::columns::transformed(self.scip, self.vars[col].as_ptr()) }.ok()?
            }
            Some(k) => match self.priced_vars.get(k) {
                Some(var) => var.as_ptr(),
                None => return self.priced.get(col).map(|column| column.column.bounds),
            },
        };
        // SAFETY: `var` is a live variable of the transformed problem, whose local bounds SCIP reads while solving.
        let (lower, upper) = unsafe { (SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var)) };
        let end = |value: f64| {
            if value.abs() >= self.infinity {
                value.signum() * f64::INFINITY
            } else {
                value
            }
        };
        Bounds::new(end(lower), end(upper)).ok()
    }

    /// Refuses a column SCIP cannot hold: a number SCIP reads as infinite, as given or once its repeated entries are
    /// added up, or a name with a NUL byte.
    ///
    /// # Errors
    ///
    /// [`ActionError::Refused`] naming the number or the name.
    fn refuse(&self, column: &NewCol) -> Result<(), ActionError> {
        let infinity = self.infinity;
        let beyond = |value: f64| value.is_finite() && value.abs() >= infinity;
        let data = &column.column;
        let numbers = [
            ("cost", data.cost),
            ("lower bound", data.bounds.lower()),
            ("upper bound", data.bounds.upper()),
        ];
        if let Some((what, value)) = numbers.into_iter().find(|&(_, value)| beyond(value)) {
            return Err(ActionError::Refused(format!(
                "the {what} {value} is one SCIP reads as infinite, from {infinity} on"
            )));
        }
        // The coefficients add up per row in the order given, as the solve's `Priced` adds them.
        let mut sums: BTreeMap<Row, f64> = BTreeMap::new();
        for &(row, value) in &column.entries {
            *sums.entry(row).or_insert(0.0) += value;
        }
        if let Some((row, value)) = sums.into_iter().find(|&(_, value)| beyond(value)) {
            return Err(ActionError::Refused(format!(
                "the coefficient {value} in {row:?} is one SCIP reads as infinite, from {infinity} on"
            )));
        }
        if data.name.as_deref().is_some_and(|name| name.contains('\0')) {
            return Err(ActionError::Refused("the name holds a NUL byte".to_owned()));
        }
        Ok(())
    }
}

impl PriceContext for ScipPrice<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
    fn mode(&self) -> PriceMode {
        self.mode
    }
    fn sense(&self) -> Sense {
        self.sense
    }
    fn duals(&self) -> &TiSlice<Row, f64> {
        self.duals
    }
    fn tolerance(&self) -> f64 {
        self.tolerance
    }
    /// SCIP takes the column once the callback returns.
    ///
    /// # Errors
    ///
    /// Besides the core errors, [`ActionError::Refused`] for a number SCIP would read as infinite, as given or once
    /// the column's repeated entries are added up, or a name with a NUL byte; checked before the column takes an id.
    fn add_col(&mut self, column: NewCol) -> Result<Col, ActionError> {
        self.refuse(&column)?;
        let col = self.priced.push(column)?;
        self.round.added.push(col);
        Ok(col)
    }
    /// In [`PriceMode::Farkas`] SCIP takes the round as not run, and the node LP stays infeasible to it. SCIP stops
    /// pricing early only where it can branch: at a node where no integer variable is unfixed, as in a program
    /// without integer columns, the solve fails with [`ScipError::BranchEarly`] once the callback returns.
    fn branch_early(&mut self) {
        self.round.early = true;
    }
    /// # Errors
    ///
    /// [`ActionError::NotFiniteBound`] for a NaN or infinite bound, [`ActionError::Refused`] in
    /// [`PriceMode::Farkas`], which takes no bound, or for a bound SCIP reads as infinite.
    fn report_bound(&mut self, bound: f64) -> Result<(), ActionError> {
        if matches!(self.mode, PriceMode::Farkas) {
            return Err(ActionError::Refused(
                "SCIP takes no bound from a Farkas round".to_owned(),
            ));
        }
        if !bound.is_finite() {
            return Err(ActionError::NotFiniteBound { bound });
        }
        if bound.abs() >= self.infinity {
            return Err(ActionError::Refused(format!(
                "the bound {bound} is one SCIP reads as infinite, from {} on",
                self.infinity
            )));
        }
        self.round.bound = Some(bound);
        Ok(())
    }
}

/// What a pricer callback reports to SCIP.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Verdict {
    result: SCIP_RESULT,
    /// SCIP's lower bound on the node, in its transformed space; `None` to leave SCIP's default.
    lower: Option<f64>,
    stop_early: bool,
}

impl Verdict {
    const DIDNOTRUN: Self = Self {
        result: SCIP_Result_SCIP_DIDNOTRUN,
        lower: None,
        stop_early: false,
    };
}

impl Run<'_> {
    /// The multiplier of every program row in `mode`: the program's dual in reduced-cost mode, SCIP's Farkas
    /// multiplier in Farkas mode.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call or a row has no transformed constraint.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, pricing a node LP solved in `mode`.
    unsafe fn duals(&self, scip: *mut SCIP, farkas: bool) -> Result<TiVec<Row, f64>, ScipError> {
        let s = match self.sense {
            Sense::Minimize => 1.0,
            Sense::Maximize => -1.0,
        };
        let mut duals = TiVec::with_capacity(self.rows.len());
        for cons in self.rows {
            // SAFETY: the caller's contract is `transformed_cons`'.
            let cons = unsafe { transformed_cons(scip, cons.as_ptr()) }?;
            // SAFETY: the instance is pricing a solved node LP, which holds the linear constraint's row.
            duals.push(unsafe {
                if farkas {
                    SCIPgetDualfarkasLinear(scip, cons)
                } else {
                    s * SCIPgetDualsolLinear(scip, cons)
                }
            });
        }
        Ok(duals)
    }

    /// Hands a node LP to the callback for pricing and SCIP the columns it added; the pricer's verdict.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call; [`ScipError::BranchEarly`] for an early stop where SCIP cannot
    /// branch.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, in a pricer callback of `farkas`'s kind.
    unsafe fn price(&self, scip: *mut SCIP, farkas: bool) -> Result<Verdict, ScipError> {
        // SCIP also calls the reduced-cost callback for a node LP the LP solver stopped at the objective limit
        // (`OBJLIMIT`). A pricing solve keeps that limit off (`lp/disablecutoff`), so this holds unless SCIP changes;
        // such an LP is not optimal, so it is not handed to the callback, and the round counts as not run.
        // SAFETY: the instance is pricing a solved node LP.
        if !farkas && unsafe { SCIPgetLPSolstat(scip) } != SCIP_LPSolStat_SCIP_LPSOLSTAT_OPTIMAL {
            return Ok(Verdict::DIDNOTRUN);
        }
        // SAFETY: the caller's contract.
        let duals = unsafe { self.duals(scip, farkas) }?;
        let mode = if farkas {
            PriceMode::Farkas
        } else {
            // SAFETY: the node LP is solved; SCIP maps its value back into the program's space with the offset.
            let objective = unsafe { SCIPretransformObj(scip, SCIPgetLPObjval(scip)) };
            PriceMode::ReducedCost { objective }
        };
        let progress = self.progress(scip);
        // SAFETY: the instance is solving.
        let tolerance = unsafe { SCIPdualfeastol(scip) };
        let mut round = Round::default();
        let delivered = {
            let priced_vars = self.priced_vars.borrow();
            let mut priced = self.priced.borrow_mut();
            self.deliver(scip, |callback| {
                callback.price(&mut ScipPrice {
                    scip,
                    progress,
                    mode,
                    sense: self.sense,
                    duals: &duals,
                    tolerance,
                    infinity: self.infinity,
                    priced_vars: &priced_vars,
                    vars: self.cols.vars,
                    priced: &mut priced,
                    round: &mut round,
                });
            })
        };
        if delivered != Delivered::Ran {
            return Ok(Verdict::DIDNOTRUN);
        }
        for &col in &round.added {
            // SAFETY: the caller's contract; `col` is a column the solve's `Priced` handed out in this round.
            unsafe { self.apply(scip, col, &duals, farkas) }?;
        }
        // SAFETY: the instance is solving.
        if round.early && unsafe { SCIPgetNPseudoBranchCands(scip) } == 0 {
            return Err(ScipError::BranchEarly);
        }
        Ok(match (farkas, round.early) {
            (true, true) => Verdict::DIDNOTRUN,
            (true, false) => Verdict {
                result: SCIP_Result_SCIP_SUCCESS,
                lower: None,
                stop_early: false,
            },
            (false, stop_early) => Verdict {
                result: SCIP_Result_SCIP_SUCCESS,
                // SAFETY: the instance is solving; SCIP maps the bound into its transformed space.
                lower: round.bound.map(|bound| unsafe { SCIPtransformObj(scip, bound) }),
                stop_early,
            },
        })
    }

    /// Adds the priced column `col` to SCIP's transformed problem as a variable with its coefficients in the
    /// transformed constraints of its rows, and keeps the reference to it.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Panics
    ///
    /// When the solve's `Priced` did not hand `col` out.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, in a pricer callback.
    unsafe fn apply(
        &self,
        scip: *mut SCIP,
        col: Col,
        duals: &TiSlice<Row, f64>,
        farkas: bool,
    ) -> Result<(), ScipError> {
        let priced = self.priced.borrow();
        let column = priced.get(col).expect("a column the solve's `Priced` handed out");
        let data = &column.column;
        let s = match self.sense {
            Sense::Minimize => 1.0,
            Sense::Maximize => -1.0,
        };
        let side = |end: f64| {
            if end.is_infinite() {
                end.signum() * self.infinity
            } else {
                end
            }
        };
        let kind = match data.integrality {
            Integrality::Integer => SCIP_Vartype_SCIP_VARTYPE_INTEGER,
            _ => SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS,
        };
        let invalid = || ScipError::Call {
            call: "SCIPcreateVar",
            code: RetCode::InvalidData,
        };
        let name = data
            .name
            .as_deref()
            .map(CString::new)
            .transpose()
            .map_err(|_| invalid())?;
        let dot: f64 = column.entries.iter().map(|&(row, a)| duals[row] * a).sum();
        let score = if farkas { dot.abs() } else { (data.cost - dot).abs() };
        let mut var: *mut SCIP_VAR = ptr::null_mut();
        // SAFETY: the instance is solving, where a variable of the transformed problem is created; `var` is a local to
        // write to, and the name null, for a name SCIP makes up, or a NUL-terminated string SCIP copies. The
        // variable is not in the initial LP, may leave the LP, and has no data or callbacks.
        check("SCIPcreateVar", unsafe {
            SCIPcreateVar(
                scip,
                &raw mut var,
                name.as_deref().map_or(ptr::null(), CStr::as_ptr),
                side(data.bounds.lower()),
                side(data.bounds.upper()),
                s * data.cost,
                kind,
                0,
                1,
                None,
                None,
                None,
                None,
                ptr::null_mut(),
            )
        })?;
        let var = NonNull::new(var).ok_or_else(invalid)?;
        // The run owns the reference from here on, so it is released on every path.
        self.priced_vars.borrow_mut().push(var);
        // SAFETY: the instance is pricing, the one stage SCIP adds a priced variable in; `var` is its own.
        check("SCIPaddPricedVar", unsafe {
            SCIPaddPricedVar(scip, var.as_ptr(), score)
        })?;
        if kind == SCIP_Vartype_SCIP_VARTYPE_INTEGER {
            self.integral.set(true);
        }
        for &(row, a) in &column.entries {
            // SAFETY: the instance is solving; the row's constraint is one of its own.
            let cons = unsafe { transformed_cons(scip, self.rows[row].as_ptr()) }?;
            // SAFETY: as above; `cons` is a modifiable linear constraint of the transformed problem.
            check("SCIPaddCoefLinear", unsafe {
                SCIPaddCoefLinear(scip, cons, var.as_ptr(), a)
            })?;
        }
        Ok(())
    }
}

/// The transformed constraint of an original one.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses the call or the constraint has no transformed counterpart.
///
/// # Safety
///
/// `scip` is a live instance from its transformed stage on and `cons` an original constraint of it.
unsafe fn transformed_cons(scip: *mut SCIP, cons: *mut SCIP_CONS) -> Result<*mut SCIP_CONS, ScipError> {
    let mut transformed: *mut SCIP_CONS = ptr::null_mut();
    // SAFETY: the caller vouches for the instance, its stage and `cons`; `transformed` is a local to write to.
    check("SCIPgetTransformedCons", unsafe {
        SCIPgetTransformedCons(scip, cons, &raw mut transformed)
    })?;
    if transformed.is_null() {
        return Err(ScipError::Call {
            call: "SCIPgetTransformedCons",
            code: RetCode::InvalidData,
        });
    }
    Ok(transformed)
}

/// Includes the pricer, delayed and inactive.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses the call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(super) unsafe fn include(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    let mut pricer: *mut SCIP_PRICER = ptr::null_mut();
    // SAFETY: the instance is live and in its init stage; the name and description are NUL-terminated strings SCIP
    // copies; the callbacks treat their arguments as SCIP passes them, and the data is the slot, which outlives SCIP.
    // The pricer is delayed, so SCIP prices the variables it holds first, and has no copy callback, so a copy of the
    // instance is not valid while it is active.
    check("SCIPincludePricerBasic", unsafe {
        SCIPincludePricerBasic(
            scip,
            &raw mut pricer,
            NAME.as_ptr(),
            c"columns of a culmen callback".as_ptr(),
            0,
            1,
            Some(redcost),
            Some(farkas),
            slot.as_ptr().cast(),
        )
    })
}

/// Prices for the run installed in the pricer's slot; the verdict and the code to return to SCIP.
///
/// # Safety
///
/// SCIP calls the pricer with itself, pricing a node LP solved in `farkas`'s mode.
unsafe fn exec(scip: *mut SCIP, pricer: *mut SCIP_PRICER, farkas: bool) -> (Verdict, SCIP_RETCODE) {
    // SAFETY: the pricer's data is its slot.
    let run = unsafe { slot::<_>(SCIPpricerGetData(pricer)) }
        // SAFETY: SCIP calls an active pricer during the solve the run is installed for.
        .and_then(|slot| unsafe { slot.run() })
        .inspect(|run| run.pass_stop(scip))
        .filter(|run| run.pricing);
    match run {
        // SAFETY: the caller's contract.
        Some(run) => match unsafe { run.price(scip, farkas) } {
            Ok(verdict) => (verdict, SCIP_Retcode_SCIP_OKAY),
            Err(error) => (Verdict::DIDNOTRUN, run.fail(error)),
        },
        None => (Verdict::DIDNOTRUN, SCIP_Retcode_SCIP_OKAY),
    }
}

/// `PRICERREDCOST`: the columns the callback adds for an optimal node LP.
unsafe extern "C" fn redcost(
    scip: *mut SCIP,
    pricer: *mut SCIP_PRICER,
    lowerbound: *mut f64,
    stopearly: *mut c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    // SAFETY: SCIP calls the reduced-cost callback with an optimal node LP.
    let (verdict, code) = unsafe { exec(scip, pricer, false) };
    // SAFETY: `lowerbound`, `stopearly` and `result` are SCIP's to write for the call; SCIP preset the first two.
    unsafe {
        if let Some(lower) = verdict.lower {
            lowerbound.write(lower);
        }
        if verdict.stop_early {
            stopearly.write(1);
        }
        result.write(verdict.result);
    }
    code
}

/// `PRICERFARKAS`: the columns the callback adds for an infeasible node LP.
unsafe extern "C" fn farkas(scip: *mut SCIP, pricer: *mut SCIP_PRICER, result: *mut SCIP_RESULT) -> SCIP_RETCODE {
    // SAFETY: SCIP calls the Farkas callback with an infeasible node LP.
    let (verdict, code) = unsafe { exec(scip, pricer, true) };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(verdict.result) };
    code
}
