// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! User cuts: a separator called on every node's LP solution (`SEPAEXECLP`).
//!
//! Each cut the callback adds becomes a global row over the columns' transformed variables. A cut that cuts the LP
//! solution off by SCIP's efficacy goes into the LP at once (`SCIPaddRow`); every cut goes into SCIP's global cut
//! pool (`SCIPaddPoolCut`), which SCIP separates again at later nodes. The separator reports `CUTOFF` when a cut makes
//! the node's LP infeasible, `SEPARATED` when a cut went into the LP, and `DIDNOTFIND` otherwise.
//!
//! SCIP takes a cut as valid: it may cut off every point the cut excludes. That every integer-feasible point of the
//! program satisfies the cut is the callback's promise, which nothing checks.

use core::ffi::{c_int, c_uint};
use core::ptr::{self, NonNull};

use culmen_core::{ActionError, AddedRow, Col, Constraint, Context, CutContext, Progress, TiSlice};
use scip_sys::{
    SCIP, SCIP_RESULT, SCIP_RETCODE, SCIP_ROW, SCIP_Result_SCIP_CUTOFF, SCIP_Result_SCIP_DIDNOTFIND,
    SCIP_Result_SCIP_DIDNOTRUN, SCIP_Result_SCIP_SEPARATED, SCIP_Retcode_SCIP_OKAY, SCIP_SEPA, SCIP_VAR,
    SCIPaddPoolCut, SCIPaddRow, SCIPaddVarsToRow, SCIPcreateEmptyRowSepa, SCIPincludeSepaBasic, SCIPisCutEfficacious,
    SCIPreleaseRow, SCIPsepaGetData,
};

use super::{Delivered, Run, Slot, checked, sides, slot};
use crate::columns::Columns;
use crate::error::check;
use crate::{RetCode, ScipError};

/// Before SCIP's own separators, whose priorities stay below `1_000_000`, so the callback sees the LP solution first.
const PRIORITY: c_int = 1_000_000;

/// A node's LP solution, to be cut off with user cuts: the separator's `SEPAEXECLP`.
#[derive(Debug)]
pub struct ScipCut<'a> {
    progress: Progress,
    relaxation: &'a TiSlice<Col, f64>,
    infinity: f64,
    added: &'a mut Vec<AddedRow>,
}

impl Context for ScipCut<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl CutContext for ScipCut<'_> {
    fn relaxation(&self) -> &TiSlice<Col, f64> {
        self.relaxation
    }
    /// SCIP takes the cut once the callback returns; a cut that does not cut the LP solution off by SCIP's efficacy
    /// goes into the cut pool alone.
    ///
    /// # Errors
    ///
    /// Besides the core errors, [`ActionError::Refused`] for a number SCIP would read as infinite, or more terms
    /// than SCIP's `int` counts.
    fn add_cut(&mut self, cut: Constraint) -> Result<(), ActionError> {
        let row = checked(&cut, self.relaxation.len(), self.infinity)?;
        self.added.push(row);
        Ok(())
    }
}

/// A row this module created, released when it drops.
#[derive(Debug)]
struct OwnedRow {
    scip: *mut SCIP,
    row: NonNull<SCIP_ROW>,
}

impl Drop for OwnedRow {
    fn drop(&mut self) {
        let mut row = self.row.as_ptr();
        // SAFETY: the instance is live and `row` the one reference this value holds, released once.
        let code = unsafe { SCIPreleaseRow(self.scip, &raw mut row) };
        debug_assert!(
            code == SCIP_Retcode_SCIP_OKAY || std::thread::panicking(),
            "OwnedRow: SCIP failed to release a row: {code}"
        );
    }
}

/// The global row of a cut over the columns' transformed variables, the priced ones `priced` included, removable from
/// the LP.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is the live instance holding `cols` and `priced`, solving, and `sepa` its separator.
unsafe fn row(
    scip: *mut SCIP,
    sepa: *mut SCIP_SEPA,
    cols: Columns<'_>,
    priced: &[NonNull<SCIP_VAR>],
    cut: &AddedRow,
    infinity: f64,
) -> Result<OwnedRow, ScipError> {
    let mut vars: Vec<*mut SCIP_VAR> = Vec::with_capacity(cut.terms.len());
    for &(col, _) in &cut.terms {
        // SAFETY: the caller vouches for the instance and its stage; the column's variable is one of its own.
        vars.push(unsafe { cols.transformed(scip, col, priced) }?);
    }
    let mut values: Vec<f64> = cut.terms.iter().map(|&(_, value)| value).collect();
    let invalid = ScipError::Call {
        call: "SCIPcreateEmptyRowSepa",
        code: RetCode::InvalidData,
    };
    let len = c_int::try_from(vars.len()).map_err(|_| invalid.clone())?;
    let (lhs, rhs) = sides(cut, infinity);
    let mut row: *mut SCIP_ROW = ptr::null_mut();
    // SAFETY: the instance is live and solving, `sepa` its separator, `row` a local to write to, and the name a
    // NUL-terminated string SCIP copies. The row is global, not modifiable, and removable.
    check("SCIPcreateEmptyRowSepa", unsafe {
        SCIPcreateEmptyRowSepa(scip, &raw mut row, sepa, c"cut".as_ptr(), lhs, rhs, 0, 0, 1)
    })?;
    let row = OwnedRow {
        scip,
        row: NonNull::new(row).ok_or(invalid)?,
    };
    // SAFETY: the row is live; `vars` and `values` hold `len` entries SCIP copies.
    check("SCIPaddVarsToRow", unsafe {
        SCIPaddVarsToRow(scip, row.row.as_ptr(), len, vars.as_mut_ptr(), values.as_mut_ptr())
    })?;
    Ok(row)
}

impl Run<'_, '_> {
    /// Hands the LP solution to the callback and the cuts it added to SCIP; the separator's result.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, solving with a node LP, and `sepa` its separator.
    unsafe fn separate(&self, scip: *mut SCIP, sepa: *mut SCIP_SEPA) -> Result<SCIP_RESULT, ScipError> {
        // SAFETY: the caller vouches for the instance and its stage; null is the current LP solution.
        let relaxation = unsafe { self.values(scip, ptr::null_mut()) };
        let progress = self.progress(scip);
        let mut added = Vec::new();
        let delivered = self.deliver(scip, |hooks| {
            hooks.cut(&mut ScipCut {
                progress,
                relaxation: &relaxation,
                infinity: self.infinity,
                added: &mut added,
            });
        });
        if delivered != Delivered::Ran {
            return Ok(SCIP_Result_SCIP_DIDNOTRUN);
        }
        let mut result = SCIP_Result_SCIP_DIDNOTFIND;
        for cut in &added {
            // SAFETY: as above.
            let row = unsafe { row(scip, sepa, self.cols, &self.priced_vars.borrow(), cut, self.infinity) }?;
            let raw = row.row.as_ptr();
            // SAFETY: the instance is live and solving with a node LP; the row is live. Null is the LP solution.
            if unsafe { SCIPisCutEfficacious(scip, ptr::null_mut(), raw) } != 0 {
                let mut infeasible: c_uint = 0;
                // SAFETY: as above; `infeasible` is a local to write to.
                check("SCIPaddRow", unsafe { SCIPaddRow(scip, raw, 0, &raw mut infeasible) })?;
                if infeasible != 0 {
                    return Ok(SCIP_Result_SCIP_CUTOFF);
                }
                result = SCIP_Result_SCIP_SEPARATED;
            }
            // SAFETY: as above; the pool takes its own reference to the global row.
            check("SCIPaddPoolCut", unsafe { SCIPaddPoolCut(scip, raw) })?;
        }
        Ok(result)
    }
}

/// Includes the separator, called at every node.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses the call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(super) unsafe fn include(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    let mut sepa: *mut SCIP_SEPA = ptr::null_mut();
    // SAFETY: the instance is live and in its init stage; the name and description are NUL-terminated strings SCIP
    // copies; the callback treats its arguments as SCIP passes them, and the data is the slot, which outlives SCIP.
    check("SCIPincludeSepaBasic", unsafe {
        SCIPincludeSepaBasic(
            scip,
            &raw mut sepa,
            c"culmen".as_ptr(),
            c"user cuts of a culmen callback".as_ptr(),
            PRIORITY,
            1,
            1.0,
            0,
            0,
            Some(execlp),
            None,
            slot.as_ptr().cast(),
        )
    })
}

/// `SEPAEXECLP`: the callback's cuts for the node's LP solution.
unsafe extern "C" fn execlp(
    scip: *mut SCIP,
    sepa: *mut SCIP_SEPA,
    result: *mut SCIP_RESULT,
    _allowlocal: c_uint,
    _depth: c_int,
) -> SCIP_RETCODE {
    // SAFETY: the separator's data is its slot.
    let run = unsafe { slot::<_>(SCIPsepaGetData(sepa)) }
        // SAFETY: SCIP calls the separator during the solve the run is installed for.
        .and_then(|slot| unsafe { slot.run() })
        .inspect(|run| run.pass_stop(scip))
        .filter(|run| run.wants_cut);
    let (verdict, code) = match run {
        // SAFETY: SCIP separates while solving, with the node's LP solved.
        Some(run) => match unsafe { run.separate(scip, sepa) } {
            Ok(verdict) => (verdict, SCIP_Retcode_SCIP_OKAY),
            Err(error) => (SCIP_Result_SCIP_DIDNOTFIND, run.fail(error)),
        },
        None => (SCIP_Result_SCIP_DIDNOTRUN, SCIP_Retcode_SCIP_OKAY),
    };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(verdict) };
    code
}
