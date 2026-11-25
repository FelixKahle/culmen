// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Heuristic solutions: a primal heuristic called before and after every node (`HEUREXEC`).
//!
//! A supplied solution becomes an original solution of SCIP's (`SCIPcreateOrigSol`), the epigraph variable of a
//! quadratic objective set to the quadratic part's value, and SCIP tries it at once (`SCIPtrySolFree`): it checks it
//! against the original problem, bounds and integrality included, and stores it if it is feasible and among the
//! solutions SCIP keeps. Checking it in the original space keeps it independent of what presolve did to the variables.
//!
//! In a solve that also asks for [`Events::PRICE`], an original solution cannot hold the columns priced so far, which
//! have no original variable, so the solution is a transformed one (`SCIPcreateSol`) instead: each program column's
//! value is set through its original variable into whatever presolve made of it, each priced column's on its own
//! variable, and the solution is handed over only when SCIP reads back every program column's value as given
//! (`SCIPisEQ`). A column presolve fixed at another value or multi-aggregated over others may not read back so, and
//! the solution is then refused; SCIP checks a transformed solution against its transformed problem, which presolve
//! derived from the program, so a solution it stores is feasible in the program.
//!
//! In a solve that also asks for [`Events::LAZY`], the lazy-constraint callback must judge every solution SCIP
//! stores, and it cannot while the heuristic callback runs; so there the supplied solutions are tried once the
//! heuristic callback returns.

use core::ffi::{c_char, c_uint};
use core::ptr::{self, NonNull};

use culmen_core::context::{check_entries, check_solution};
use culmen_core::{ActionError, Col, Events, HeuristicContext, Progress, TiSlice};
use scip_sys::{
    SCIP, SCIP_HEUR, SCIP_HEURTIMING, SCIP_HEURTIMING_AFTERNODE, SCIP_HEURTIMING_BEFORENODE, SCIP_RESULT, SCIP_RETCODE,
    SCIP_Result_SCIP_DIDNOTFIND, SCIP_Result_SCIP_DIDNOTRUN, SCIP_Result_SCIP_FOUNDSOL, SCIP_Retcode_SCIP_OKAY,
    SCIP_SOL, SCIP_VAR, SCIP_Varstatus_SCIP_VARSTATUS_FIXED, SCIP_Varstatus_SCIP_VARSTATUS_MULTAGGR, SCIPcreateOrigSol,
    SCIPcreateSol, SCIPfreeSol, SCIPgetSolVal, SCIPheurGetData, SCIPincludeHeurBasic, SCIPisEQ, SCIPsetSolVal,
    SCIPtrySolFree, SCIPvarGetStatus,
};

use super::{Delivered, Run, Slot, slot};
use crate::ScipError;
use crate::columns::Columns;
use crate::error::check;

/// Before SCIP's own heuristics at each point, whose priorities stay below `1_000_000`.
const PRIORITY: i32 = 1_000_000;

/// SCIP asks for a solution of the callback's own: the heuristic's `HEUREXEC`, before and after each node.
#[derive(Debug)]
pub struct ScipHeuristic<'a> {
    scip: *mut SCIP,
    heur: *mut SCIP_HEUR,
    cols: Columns<'a>,
    /// The variables of the columns priced so far.
    priced: &'a [NonNull<SCIP_VAR>],
    /// Whether the solve prices, so a solution is a transformed one.
    pricing: bool,
    infinity: f64,
    progress: Progress,
    /// The solutions to try once the callback returns; `None` to try each at once.
    later: Option<&'a mut Vec<Vec<f64>>>,
    /// Whether SCIP stored a solution tried at once.
    found: &'a mut bool,
}

impl HeuristicContext for ScipHeuristic<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }

    /// SCIP checks the solution against the program, bounds and integrality included, and stores it when it is
    /// feasible and among the solutions it keeps (`limits/maxsol` of them), so a stored solution need not be the new
    /// incumbent. In a solve that also asks for [`Events::LAZY`], SCIP tries the solution once this callback
    /// returns, after the lazy-constraint callback judged it, and `supply` returns `Ok` when it took it to try. In a
    /// solve that also asks for [`Events::PRICE`], the solution is checked against SCIP's transformed problem, and a
    /// solution presolve's fixings or aggregations cannot hold is refused.
    ///
    /// # Errors
    ///
    /// Besides the core errors, [`ActionError::Refused`] for a value SCIP would read as infinite, a solution SCIP
    /// did not store, a solution of a pricing solve SCIP's transformed problem cannot hold, or a call SCIP refused.
    fn supply(&mut self, values: &TiSlice<Col, f64>) -> Result<(), ActionError> {
        check_solution(values, self.cols.len() + self.priced.len())?;
        if let Some((col, value)) = values.iter_enumerated().find(|(_, v)| v.abs() >= self.infinity) {
            return Err(ActionError::Refused(format!(
                "the value {value} of {col:?} is one SCIP reads as infinite, from {} on",
                self.infinity
            )));
        }
        if let Some(later) = self.later.as_deref_mut() {
            later.push(values.raw.to_vec());
            return Ok(());
        }
        let sol = Supplied {
            cols: self.cols,
            priced: self.priced,
            pricing: self.pricing,
        };
        // SAFETY: the context lives during the heuristic's call, solving, of the instance holding the columns.
        match unsafe { sol.try_sol(self.scip, self.heur, values) } {
            Ok(Tried::Stored) => {
                *self.found = true;
                Ok(())
            }
            Ok(Tried::Rejected) => Err(ActionError::Refused(
                "SCIP found the solution infeasible, or worse than every solution it keeps".into(),
            )),
            Ok(Tried::Unfit(reason)) => Err(ActionError::Refused(reason)),
            Err(error) => Err(ActionError::Refused(error.to_string())),
        }
    }

    /// Always refused once the entries pass the core checks: SCIP completes a partial solution only as a start,
    /// before a solve begins, and offers no completion during the search.
    ///
    /// # Errors
    ///
    /// The core errors, else [`ActionError::Refused`].
    fn supply_partial(&mut self, entries: &[(Col, f64)]) -> Result<(), ActionError> {
        check_entries(entries, self.cols.len() + self.priced.len())?;
        Err(ActionError::Refused(
            "SCIP completes a partial solution only as a start, before a solve begins; supply a value for every column"
                .into(),
        ))
    }
}

/// What came of trying a supplied solution.
#[derive(Debug, PartialEq)]
enum Tried {
    /// SCIP stored it.
    Stored,
    /// SCIP found it infeasible, or worse than every solution it keeps.
    Rejected,
    /// SCIP's transformed problem cannot hold it; the reason.
    Unfit(String),
}

/// Where a supplied solution's values go: the program's columns, the columns priced so far, and whether the solve
/// prices.
#[derive(Clone, Copy, Debug)]
struct Supplied<'a> {
    cols: Columns<'a>,
    priced: &'a [NonNull<SCIP_VAR>],
    pricing: bool,
}

impl Supplied<'_> {
    /// Hands SCIP a full solution to check and store: an original one, or a transformed one in a solve that prices.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance holding the columns, solving, and `heur` its heuristic; `values` holds one value
    /// per column, the priced ones included.
    unsafe fn try_sol(
        self,
        scip: *mut SCIP,
        heur: *mut SCIP_HEUR,
        values: &TiSlice<Col, f64>,
    ) -> Result<Tried, ScipError> {
        let mut sol: *mut SCIP_SOL = ptr::null_mut();
        if self.pricing {
            // SAFETY: the instance is live and solving, where a transformed solution is created; `sol` a local.
            check("SCIPcreateSol", unsafe { SCIPcreateSol(scip, &raw mut sol, heur) })?;
        } else {
            // SAFETY: the instance is live and solving, where an original solution is created; `sol` a local.
            check("SCIPcreateOrigSol", unsafe {
                SCIPcreateOrigSol(scip, &raw mut sol, heur)
            })?;
        }
        // SAFETY: `sol` is a live solution of the instance, transformed exactly when the solve prices.
        let filled = unsafe { self.fill(scip, sol, values) };
        if !matches!(filled, Ok(None)) {
            // SAFETY: `sol` is the solution created above, freed once.
            let freed = unsafe { SCIPfreeSol(scip, &raw mut sol) };
            debug_assert!(
                freed == SCIP_Retcode_SCIP_OKAY,
                "try_sol: SCIP failed to free a solution: {freed}"
            );
            return filled.map(|unfit| Tried::Unfit(unfit.unwrap_or_default()));
        }
        let mut stored: c_uint = 0;
        // SAFETY: `sol` is a complete solution SCIP takes over and frees; `stored` is a local to write to. SCIP
        // checks it with bounds, integrality and every row, as nothing about it is known.
        check("SCIPtrySolFree", unsafe {
            SCIPtrySolFree(scip, &raw mut sol, 0, 0, 1, 1, 1, &raw mut stored)
        })?;
        Ok(if stored != 0 { Tried::Stored } else { Tried::Rejected })
    }

    /// Sets every column's value in `sol`: in an original solution through the original variables, with the
    /// epigraph variable at the quadratic part's value; in a transformed one as the module documentation states.
    /// `Some` reason when the transformed problem cannot hold the values.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a value.
    ///
    /// # Safety
    ///
    /// `sol` is a live solution of the live, solving instance `scip` holding the columns, transformed exactly when the
    /// solve prices; `values` holds one value per column, the priced ones included.
    unsafe fn fill(
        self,
        scip: *mut SCIP,
        sol: *mut SCIP_SOL,
        values: &TiSlice<Col, f64>,
    ) -> Result<Option<String>, ScipError> {
        if !self.pricing {
            // SAFETY: the caller's contract.
            unsafe { fill(scip, sol, self.cols, values) }?;
            return Ok(None);
        }
        debug_assert!(
            self.cols.epigraph.is_none(),
            "Supplied::fill: a pricing solve of a quadratic objective"
        );
        for (var, &value) in self.cols.vars.iter().zip(values) {
            // SAFETY: the instance is transformed and `var` one of its original variables.
            let transformed = unsafe { crate::columns::transformed(scip, var.as_ptr()) }?;
            // SAFETY: `transformed` is a variable of the transformed problem.
            let status = unsafe { SCIPvarGetStatus(transformed) };
            // A fixed variable takes no other value and a multi-aggregated one none at all; reading back finds them.
            if status == SCIP_Varstatus_SCIP_VARSTATUS_FIXED || status == SCIP_Varstatus_SCIP_VARSTATUS_MULTAGGR {
                continue;
            }
            // SAFETY: the caller vouches for `sol`; SCIP sets the value on what presolve made of the variable.
            check("SCIPsetSolVal", unsafe {
                SCIPsetSolVal(scip, sol, var.as_ptr(), value)
            })?;
        }
        let priced = values.iter().skip(self.cols.len());
        for (var, &value) in self.priced.iter().zip(priced) {
            // SAFETY: the caller vouches for `sol`; `var` is a variable of the transformed problem.
            check("SCIPsetSolVal", unsafe {
                SCIPsetSolVal(scip, sol, var.as_ptr(), value)
            })?;
        }
        for ((col, &value), var) in values.iter_enumerated().zip(self.cols.vars) {
            // SAFETY: the caller vouches for `sol`; SCIP reads the original variable through what presolve made of it.
            let read = unsafe { SCIPgetSolVal(scip, sol, var.as_ptr()) };
            // SAFETY: the instance is live; the comparison reads its epsilon.
            if unsafe { SCIPisEQ(scip, read, value) } == 0 {
                return Ok(Some(format!(
                    "presolve fixed or aggregated {col:?} so that SCIP's problem holds it at {read}, not at {value}"
                )));
            }
        }
        Ok(None)
    }
}

/// Sets every column's value in `sol`, and the epigraph variable's to the quadratic part's value.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a value.
///
/// # Safety
///
/// `sol` is a live original solution of the live instance `scip` holding `cols`; `values` holds one value per
/// column.
unsafe fn fill(
    scip: *mut SCIP,
    sol: *mut SCIP_SOL,
    cols: Columns<'_>,
    values: &TiSlice<Col, f64>,
) -> Result<(), ScipError> {
    for (var, &value) in cols.vars.iter().zip(values) {
        // SAFETY: the caller vouches for the instance and `sol`; `var` is an original variable of the instance.
        check("SCIPsetSolVal", unsafe {
            SCIPsetSolVal(scip, sol, var.as_ptr(), value)
        })?;
    }
    if let Some((t, quadratic)) = cols.epigraph {
        // SAFETY: as above; `t` is an original variable of the instance.
        check("SCIPsetSolVal", unsafe {
            SCIPsetSolVal(scip, sol, t.as_ptr(), quadratic.value(values))
        })?;
    }
    Ok(())
}

impl Run<'_> {
    /// Asks the callback for solutions; the heuristic's result.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call trying a solution kept until the callback returned.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, solving, and `heur` its heuristic.
    unsafe fn heuristic(&self, scip: *mut SCIP, heur: *mut SCIP_HEUR) -> Result<SCIP_RESULT, ScipError> {
        let progress = self.progress(scip);
        let mut later = Vec::new();
        let mut found = false;
        let defer = self.wanted.contains(Events::LAZY);
        let priced = self.priced_vars.borrow();
        let supplied = Supplied {
            cols: self.cols,
            priced: &priced,
            pricing: self.wanted.contains(Events::PRICE),
        };
        let delivered = self.deliver(scip, |callback| {
            callback.heuristic(&mut ScipHeuristic {
                scip,
                heur,
                cols: supplied.cols,
                priced: supplied.priced,
                pricing: supplied.pricing,
                infinity: self.infinity,
                progress,
                later: defer.then_some(&mut later),
                found: &mut found,
            });
        });
        if delivered != Delivered::Ran {
            return Ok(SCIP_Result_SCIP_DIDNOTRUN);
        }
        for values in later {
            if self.panicked() {
                break;
            }
            // SAFETY: the caller's contract; `values` holds one value per column, as `supply` checked, and no column
            // was priced since, as SCIP prices only while it solves a node LP.
            let tried = unsafe { supplied.try_sol(scip, heur, TiSlice::from_ref(values.as_slice())) }?;
            found |= tried == Tried::Stored;
        }
        Ok(if found {
            SCIP_Result_SCIP_FOUNDSOL
        } else {
            SCIP_Result_SCIP_DIDNOTFIND
        })
    }
}

/// Includes the heuristic, called before and after every node.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses the call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(super) unsafe fn include(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    let mut heur: *mut SCIP_HEUR = ptr::null_mut();
    let timing: SCIP_HEURTIMING = SCIP_HEURTIMING_BEFORENODE | SCIP_HEURTIMING_AFTERNODE;
    // `'U'` is ASCII, so it is the same number as a `c_char`, signed or not.
    let dispchar = c_char::try_from(b'U').unwrap_or_default();
    // SAFETY: the instance is live and in its init stage; the name and description are NUL-terminated strings SCIP
    // copies; the callback treats its arguments as SCIP passes them, and the data is the slot, which outlives SCIP.
    check("SCIPincludeHeurBasic", unsafe {
        SCIPincludeHeurBasic(
            scip,
            &raw mut heur,
            c"culmen".as_ptr(),
            c"solutions of a culmen callback".as_ptr(),
            dispchar,
            PRIORITY,
            1,
            0,
            -1,
            timing,
            0,
            Some(exec),
            slot.as_ptr().cast(),
        )
    })
}

/// `HEUREXEC`: the callback's solutions.
unsafe extern "C" fn exec(
    scip: *mut SCIP,
    heur: *mut SCIP_HEUR,
    _heurtiming: SCIP_HEURTIMING,
    _nodeinfeasible: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    // SAFETY: the heuristic's data is its slot.
    let run = unsafe { slot::<_>(SCIPheurGetData(heur)) }
        // SAFETY: SCIP calls the heuristic during the solve the run is installed for.
        .and_then(|slot| unsafe { slot.run() })
        .inspect(|run| run.pass_stop(scip))
        .filter(|run| run.wanted.contains(Events::HEURISTIC));
    let (verdict, code) = match run {
        // SAFETY: SCIP calls the heuristic while solving.
        Some(run) => match unsafe { run.heuristic(scip, heur) } {
            Ok(verdict) => (verdict, SCIP_Retcode_SCIP_OKAY),
            Err(error) => (SCIP_Result_SCIP_DIDNOTFIND, run.fail(error)),
        },
        None => (SCIP_Result_SCIP_DIDNOTRUN, SCIP_Retcode_SCIP_OKAY),
    };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(verdict) };
    code
}
