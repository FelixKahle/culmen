// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Heuristic solutions: a primal heuristic called before and after every node (`HEUREXEC`).
//!
//! A supplied solution becomes an original solution of SCIP's (`SCIPcreateOrigSol`), the epigraph variable of a
//! quadratic objective set to the quadratic part's value, and SCIP tries it at once (`SCIPtrySolFree`): it checks it
//! against the original problem, bounds and integrality included, and stores it if it is feasible and among the
//! solutions SCIP keeps. Checking it in the original space keeps it independent of what presolve did to the variables.
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
    SCIP_SOL, SCIPcreateOrigSol, SCIPfreeSol, SCIPheurGetData, SCIPincludeHeurBasic, SCIPsetSolVal, SCIPtrySolFree,
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
    /// returns, after the lazy-constraint callback judged it, and `supply` returns `Ok` when it took it to try.
    ///
    /// # Errors
    ///
    /// Besides the core errors, [`ActionError::Refused`] for a value SCIP would read as infinite, a solution SCIP
    /// did not store, or a call SCIP refused.
    fn supply(&mut self, values: &TiSlice<Col, f64>) -> Result<(), ActionError> {
        check_solution(values, self.cols.len())?;
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
        // SAFETY: the context lives during the heuristic's call, solving, of the instance holding the columns.
        match unsafe { try_sol(self.scip, self.heur, self.cols, values) } {
            Ok(true) => {
                *self.found = true;
                Ok(())
            }
            Ok(false) => Err(ActionError::Refused(
                "SCIP found the solution infeasible, or worse than every solution it keeps".into(),
            )),
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
        check_entries(entries, self.cols.len())?;
        Err(ActionError::Refused(
            "SCIP completes a partial solution only as a start, before a solve begins; supply a value for every column"
                .into(),
        ))
    }
}

/// Hands SCIP a full solution to check and store; whether it stored it.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is the live instance holding `cols`, solving, and `heur` its heuristic; `values` holds one value per
/// column.
unsafe fn try_sol(
    scip: *mut SCIP,
    heur: *mut SCIP_HEUR,
    cols: Columns<'_>,
    values: &TiSlice<Col, f64>,
) -> Result<bool, ScipError> {
    let mut sol: *mut SCIP_SOL = ptr::null_mut();
    // SAFETY: the instance is live and solving, where an original solution is created; `sol` a local to write to.
    check("SCIPcreateOrigSol", unsafe {
        SCIPcreateOrigSol(scip, &raw mut sol, heur)
    })?;
    // SAFETY: `sol` is a live original solution of the instance.
    if let Err(error) = unsafe { fill(scip, sol, cols, values) } {
        // SAFETY: `sol` is the solution created above, freed once.
        let freed = unsafe { SCIPfreeSol(scip, &raw mut sol) };
        debug_assert!(
            freed == SCIP_Retcode_SCIP_OKAY,
            "try_sol: SCIP failed to free a solution: {freed}"
        );
        return Err(error);
    }
    let mut stored: c_uint = 0;
    // SAFETY: `sol` is a complete original solution SCIP takes over and frees; `stored` is a local to write to. SCIP
    // checks it with bounds, integrality and every row, as nothing about it is known.
    check("SCIPtrySolFree", unsafe {
        SCIPtrySolFree(scip, &raw mut sol, 0, 0, 1, 1, 1, &raw mut stored)
    })?;
    Ok(stored != 0)
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
        let delivered = self.deliver(scip, |callback| {
            callback.heuristic(&mut ScipHeuristic {
                scip,
                heur,
                cols: self.cols,
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
            // SAFETY: the caller's contract; `values` holds one value per column, as `supply` checked.
            found |= unsafe { try_sol(scip, heur, self.cols, TiSlice::from_ref(values.as_slice())) }?;
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
