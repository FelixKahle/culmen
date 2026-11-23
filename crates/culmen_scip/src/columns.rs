// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The program's columns as SCIP holds them, and any solution of SCIP's read in them.

use core::ptr::NonNull;

use culmen_core::{Col, QuadObjective, TiSlice, TiVec};
use scip_sys::{SCIP, SCIP_SOL, SCIP_VAR, SCIPgetSolOrigObj, SCIPgetSolVal};

/// The original variable of every column, and the epigraph variable standing for a quadratic objective with the
/// quadratic part it stands for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Columns<'a> {
    pub(crate) vars: &'a TiSlice<Col, NonNull<SCIP_VAR>>,
    /// The epigraph variable `t` and the quadratic part `q`; `None` for a linear objective.
    pub(crate) epigraph: Option<(NonNull<SCIP_VAR>, &'a QuadObjective)>,
}

impl Columns<'_> {
    /// The number of columns.
    pub(crate) fn len(&self) -> usize {
        self.vars.len()
    }

    /// The objective of `sol` in the program's sense, offset included, and its value of every column.
    ///
    /// SCIP reads an original variable's value in a transformed solution, or in the current LP or pseudo solution,
    /// through the variable's transformed counterpart, resolving whatever presolve fixed, aggregated or
    /// multi-aggregated it into; so every value is the program's column's, whichever space `sol` lives in. The
    /// objective is SCIP's original objective, which counts the epigraph variable `t` where the program counts `q(x)`;
    /// `t` meets `q(x)` only to SCIP's feasibility tolerance, so `q(x)` itself takes its place.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance holding these variables, in a stage where SCIP reads `sol`: from its problem
    /// stage on for an original solution, from its transforming stage on for a transformed one, and only while
    /// solving for null, the current LP or pseudo solution.
    pub(crate) unsafe fn read(&self, scip: *mut SCIP, sol: *mut SCIP_SOL) -> (f64, TiVec<Col, f64>) {
        // SAFETY: the caller's contract is `values`'.
        let values = unsafe { self.values(scip, sol) };
        // SAFETY: the caller vouches for the instance, its stage and `sol`.
        let objective = unsafe { SCIPgetSolOrigObj(scip, sol) };
        let objective = match self.epigraph {
            None => objective,
            Some((t, quadratic)) => {
                // SAFETY: as above; `t` is a variable of the instance.
                let t = unsafe { SCIPgetSolVal(scip, sol, t.as_ptr()) };
                objective - t + quadratic.value(&values)
            }
        };
        (objective, values)
    }

    /// The value of every column in `sol`, as [`Columns::read`] reads it.
    ///
    /// # Safety
    ///
    /// As [`Columns::read`].
    pub(crate) unsafe fn values(&self, scip: *mut SCIP, sol: *mut SCIP_SOL) -> TiVec<Col, f64> {
        self.vars
            .iter()
            // SAFETY: the caller vouches for the instance, its stage and `sol`; each variable is one of its own.
            .map(|var| unsafe { SCIPgetSolVal(scip, sol, var.as_ptr()) })
            .collect()
    }
}
