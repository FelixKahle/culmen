// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rows a handler adds, as the lazy-constraint handler, the constraint handler and the separator share them: the check
//! of a row against what SCIP can hold, its sides in SCIP's terms, whether a point violates it by SCIP's tolerance,
//! and adding it to SCIP's transformed problem as a global linear constraint.

use core::ffi::{CStr, c_int};
use core::ptr::{self, NonNull};
use std::collections::BTreeMap;

use culmen_core::context::check_constraint;
use culmen_core::{ActionError, AddedRow, Col, Constraint, TiSlice};
use scip_sys::{
    SCIP, SCIP_CONS, SCIP_VAR, SCIPaddCons, SCIPcreateConsBasicLinear, SCIPisFeasGT, SCIPisFeasLT, SCIPreleaseCons,
};

use crate::columns::Columns;
use crate::error::check;
use crate::{RetCode, ScipError};

/// Checks a constraint a callback adds as a row SCIP can hold: known columns, finite coefficients and valid bounds,
/// and every number below SCIP's infinity in magnitude, as given and added up per column, with no more terms than
/// SCIP's `int` counts.
///
/// # Errors
///
/// The errors of [`check_constraint`]; [`ActionError::Refused`] for a number SCIP would read as infinite or a row
/// too long.
pub(super) fn checked(constraint: &Constraint, num_cols: usize, infinity: f64) -> Result<AddedRow, ActionError> {
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

/// A row's bounds in SCIP's terms, an infinite end as SCIP's infinity.
pub(super) fn sides(row: &AddedRow, infinity: f64) -> (f64, f64) {
    let side = |end: f64| {
        if end.is_infinite() {
            end.signum() * infinity
        } else {
            end
        }
    };
    (side(row.bounds.lower()), side(row.bounds.upper()))
}

/// Whether `values` violates `row` beyond SCIP's feasibility tolerance, as SCIP's linear handler judges a row.
///
/// # Panics
///
/// When `row` names a column beyond `values`.
pub(super) fn violates(scip: *mut SCIP, row: &AddedRow, values: &TiSlice<Col, f64>) -> bool {
    let activity: f64 = row.terms.iter().map(|&(col, a)| a * values[col]).sum();
    let (lower, upper) = (row.bounds.lower(), row.bounds.upper());
    // SAFETY: the instance is live; the comparisons read its tolerance.
    unsafe {
        (lower.is_finite() && SCIPisFeasLT(scip, activity, lower) != 0)
            || (upper.is_finite() && SCIPisFeasGT(scip, activity, upper) != 0)
    }
}

/// Adds `row` to SCIP's transformed problem as a global linear constraint named `name` over the columns' transformed
/// variables, the priced ones `priced` included.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Panics
///
/// When `row` names a column beyond the program's columns and `priced`.
///
/// # Safety
///
/// `scip` is the live instance holding `cols` and `priced`, solving.
pub(super) unsafe fn add_cons(
    scip: *mut SCIP,
    cols: Columns<'_>,
    priced: &[NonNull<SCIP_VAR>],
    row: &AddedRow,
    infinity: f64,
    name: &CStr,
) -> Result<(), ScipError> {
    let mut vars = Vec::with_capacity(row.terms.len());
    let mut values = Vec::with_capacity(row.terms.len());
    for &(col, value) in &row.terms {
        // SAFETY: the caller vouches for the instance and its stage; the column's variable is one of its own.
        vars.push(unsafe { cols.transformed(scip, col, priced) }?);
        values.push(value);
    }
    let invalid = ScipError::Call {
        call: "SCIPcreateConsBasicLinear",
        code: RetCode::InvalidData,
    };
    let len = c_int::try_from(vars.len()).map_err(|_| invalid.clone())?;
    let (lhs, rhs) = sides(row, infinity);
    let mut cons: *mut SCIP_CONS = ptr::null_mut();
    // SAFETY: the instance is live and solving, where a constraint over transformed variables is created; `vars` and
    // `values` hold `len` entries SCIP copies, and the name is a NUL-terminated string SCIP copies.
    check("SCIPcreateConsBasicLinear", unsafe {
        SCIPcreateConsBasicLinear(
            scip,
            &raw mut cons,
            name.as_ptr(),
            len,
            vars.as_mut_ptr(),
            values.as_mut_ptr(),
            lhs,
            rhs,
        )
    })?;
    let cons = NonNull::new(cons).ok_or(invalid)?.as_ptr();
    // SAFETY: the instance is live and solving, where SCIP adds a constraint to its transformed problem globally.
    let added = check("SCIPaddCons", unsafe { SCIPaddCons(scip, cons) });
    let mut cons = cons;
    // SAFETY: `cons` is the one reference this function holds, released once; the problem keeps its own.
    let released = check("SCIPreleaseCons", unsafe { SCIPreleaseCons(scip, &raw mut cons) });
    added.and(released)
}
