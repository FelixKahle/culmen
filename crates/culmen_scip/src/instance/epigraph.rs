// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A quadratic objective as SCIP solves it: a linear objective over an epigraph variable bounded by a quadratic
//! constraint.

use core::ffi::c_int;
use core::ptr::{self, NonNull};

use culmen_core::{Program, Sense};
use scip_sys::{
    SCIP_CONS, SCIP_VAR, SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS, SCIPaddCons, SCIPaddVar,
    SCIPcreateConsBasicQuadraticNonlinear, SCIPcreateVarBasic,
};

use super::{Epigraph, Instance};
use crate::error::{Number, check};
use crate::{RetCode, ScipError};

impl Instance {
    /// Puts the epigraph of `program`'s quadratic part in the objective's place; does nothing for a linear objective.
    ///
    /// # Errors
    ///
    /// [`ScipError::BeyondInfinity`] for a coefficient SCIP would read as infinite, [`ScipError::QuadraticTooLong`]
    /// for a quadratic part beyond SCIP's counts, [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Panics
    ///
    /// When the instance holds an epigraph already, or the quadratic part names a column without a variable.
    pub(super) fn add_epigraph(&mut self, program: &Program) -> Result<(), ScipError> {
        if program.quadratic().is_empty() {
            return Ok(());
        }
        assert!(
            self.epigraph.is_none(),
            "Instance::add_epigraph: the instance holds an epigraph already"
        );
        self.add_epigraph_var(program)?;
        self.add_epigraph_cons(program)
    }

    /// Creates the epigraph variable with cost one, at least zero when minimizing and at most zero when maximizing,
    /// and adds it to the problem.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    fn add_epigraph_var(&mut self, program: &Program) -> Result<(), ScipError> {
        let infinity = self.infinity();
        // A convex quadratic form is never negative and a concave one never positive, so the bound cuts off no point
        // of the epigraph, and it hands SCIP's relaxation a bound on `t` that SCIP otherwise finds only by branching.
        let (lower, upper) = match program.sense() {
            Sense::Minimize => (0.0, infinity),
            Sense::Maximize => (-infinity, 0.0),
        };
        let continuous = SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS;
        let mut var: *mut SCIP_VAR = ptr::null_mut();
        // SAFETY: the instance is live, borrowed exclusively and in its problem stage; `var` is a local to write the
        // new variable to, and the null name asks SCIP to make one up.
        let created =
            unsafe { SCIPcreateVarBasic(self.raw(), &raw mut var, ptr::null(), lower, upper, 1.0, continuous) };
        check("SCIPcreateVarBasic", created)?;
        let var = NonNull::new(var).ok_or(ScipError::Call {
            call: "SCIPcreateVarBasic",
            code: RetCode::Error,
        })?;
        self.epigraph = Some(Epigraph {
            var,
            cons: None,
            quadratic: program.quadratic().clone(),
        });
        // SAFETY: the instance is live and `var` a variable of it, not yet in its problem.
        check("SCIPaddVar", unsafe { SCIPaddVar(self.raw(), var.as_ptr()) })
    }

    /// Creates the quadratic constraint bounding the epigraph variable and adds it to the problem.
    ///
    /// # Errors
    ///
    /// [`ScipError::BeyondInfinity`], [`ScipError::QuadraticTooLong`] or [`ScipError::Call`].
    ///
    /// # Panics
    ///
    /// When the instance holds no epigraph variable, or the quadratic part names a column without a variable.
    fn add_epigraph_cons(&mut self, program: &Program) -> Result<(), ScipError> {
        let quadratic = program.quadratic();
        let len = c_int::try_from(quadratic.len()).map_err(|_| ScipError::QuadraticTooLong { len: quadratic.len() })?;
        let (mut firsts, mut seconds, mut coefficients) = (Vec::new(), Vec::new(), Vec::new());
        for (first, second, q) in quadratic.entries() {
            firsts.push(self.vars[first].as_ptr());
            seconds.push(self.vars[second].as_ptr());
            coefficients.push(self.number(Number::QuadCoefficient { first, second }, q)?);
        }
        let infinity = self.infinity();
        // `q(x) - t` is at most zero when minimizing a convex `q`, at least zero when maximizing a concave one.
        let (lhs, rhs) = match program.sense() {
            Sense::Minimize => (-infinity, 0.0),
            Sense::Maximize => (0.0, infinity),
        };
        let epigraph = self.epigraph.as_ref().expect("the epigraph variable is created first");
        let (mut t, mut minus_one) = (epigraph.var.as_ptr(), -1.0);
        let mut cons: *mut SCIP_CONS = ptr::null_mut();
        // SAFETY: the instance is live, borrowed exclusively and in its problem stage; `cons` is a local to write the
        // new constraint to; one linear variable `t` and its coefficient are locals, and `firsts`, `seconds` and
        // `coefficients` hold `len` variables of this instance and coefficients, all of which SCIP copies; the name is
        // a NUL-terminated string SCIP copies.
        let created = unsafe {
            SCIPcreateConsBasicQuadraticNonlinear(
                self.raw(),
                &raw mut cons,
                c"objective".as_ptr(),
                1,
                &raw mut t,
                &raw mut minus_one,
                len,
                firsts.as_mut_ptr(),
                seconds.as_mut_ptr(),
                coefficients.as_mut_ptr(),
                lhs,
                rhs,
            )
        };
        check("SCIPcreateConsBasicQuadraticNonlinear", created)?;
        let cons = NonNull::new(cons).ok_or(ScipError::Call {
            call: "SCIPcreateConsBasicQuadraticNonlinear",
            code: RetCode::Error,
        })?;
        self.epigraph
            .as_mut()
            .expect("the epigraph variable is created first")
            .cons = Some(cons);
        // SAFETY: the instance is live and `cons` a constraint of it, not yet in its problem.
        check("SCIPaddCons", unsafe { SCIPaddCons(self.raw(), cons.as_ptr()) })
    }
}
