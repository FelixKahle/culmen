// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A [`Program`] built into a new SCIP instance: one variable per column, one linear constraint per row.

use core::ffi::{CStr, c_int};
use core::ptr::{self, NonNull};
use std::ffi::CString;

use culmen_core::{Bounds, Col, Integrality, Program, Row, Sense};
use scip_sys::{
    SCIP_CONS, SCIP_Objsense_SCIP_OBJSENSE_MAXIMIZE, SCIP_Objsense_SCIP_OBJSENSE_MINIMIZE, SCIP_VAR,
    SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS, SCIP_Vartype_SCIP_VARTYPE_INTEGER, SCIPaddCons, SCIPaddOrigObjoffset,
    SCIPaddVar, SCIPcreateConsBasicLinear, SCIPcreateVarBasic, SCIPsetObjsense,
};

use super::Instance;
use crate::ScipError;
use crate::error::{Number, check};

impl Instance {
    /// A new instance holding `program`, which has no semi column and one objective; a quadratic part becomes an
    /// epigraph.
    ///
    /// On an error the instance built so far is dropped, which frees every variable and constraint it created.
    ///
    /// # Errors
    ///
    /// [`ScipError::BeyondInfinity`] for a finite number SCIP would read as infinite, [`ScipError::ColName`] or
    /// [`ScipError::RowName`] for a name holding a NUL byte, [`ScipError::RowTooLong`] or
    /// [`ScipError::QuadraticTooLong`] for a row or a quadratic part beyond SCIP's counts, [`ScipError::Call`] when
    /// SCIP refuses a call.
    ///
    /// # Panics
    ///
    /// When `program` has a semi column.
    pub(crate) fn load(program: &Program) -> Result<Self, ScipError> {
        let mut instance = Self::new()?;
        instance.set_objective(program)?;
        for col in program.cols() {
            instance.add_var(program, col)?;
        }
        for row in program.rows() {
            instance.add_cons(program, row)?;
        }
        instance.add_epigraph(program)?;
        debug_assert!(
            instance.vars.len() == program.num_cols() && instance.conss.len() == program.num_rows(),
            "Instance: the problem does not hold one variable per column and one constraint per row"
        );
        Ok(instance)
    }

    /// Sets the objective's sense and offset.
    ///
    /// # Errors
    ///
    /// [`ScipError::BeyondInfinity`] for an offset SCIP would read as infinite, [`ScipError::Call`] when SCIP refuses
    /// a call.
    fn set_objective(&mut self, program: &Program) -> Result<(), ScipError> {
        let sense = match program.sense() {
            Sense::Minimize => SCIP_Objsense_SCIP_OBJSENSE_MINIMIZE,
            Sense::Maximize => SCIP_Objsense_SCIP_OBJSENSE_MAXIMIZE,
        };
        // SAFETY: the instance is live, borrowed exclusively, and in its problem stage.
        check("SCIPsetObjsense", unsafe { SCIPsetObjsense(self.raw(), sense) })?;
        let offset = self.number(Number::Offset, program.offset())?;
        // SAFETY: as above.
        check("SCIPaddOrigObjoffset", unsafe {
            SCIPaddOrigObjoffset(self.raw(), offset)
        })
    }

    /// Creates the variable of `col` and adds it to the problem.
    ///
    /// # Errors
    ///
    /// [`ScipError::BeyondInfinity`], [`ScipError::ColName`] or [`ScipError::Call`].
    ///
    /// # Panics
    ///
    /// When the column is semi-continuous or semi-integer.
    fn add_var(&mut self, program: &Program, col: Col) -> Result<(), ScipError> {
        let column = &program.columns()[col];
        let (lower, upper) = self.bounds(Number::ColBound(col), column.bounds)?;
        let cost = self.number(Number::Cost(col), column.cost)?;
        let kind = match column.integrality {
            Integrality::Continuous => SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS,
            Integrality::Integer => SCIP_Vartype_SCIP_VARTYPE_INTEGER,
            Integrality::SemiContinuous | Integrality::SemiInteger => {
                panic!("Instance::add_var: column {} is a semi column", col.index())
            }
        };
        let name = c_name(column.name.as_deref()).map_err(|()| ScipError::ColName { col })?;
        let name = name.as_deref().map_or(ptr::null(), CStr::as_ptr);
        let mut var: *mut SCIP_VAR = ptr::null_mut();
        // SAFETY: the instance is live, borrowed exclusively and in its problem stage; `var` is a local to write the
        // new variable to, and the name is null, for a name SCIP makes up, or a NUL-terminated string SCIP copies.
        let created = unsafe { SCIPcreateVarBasic(self.raw(), &raw mut var, name, lower, upper, cost, kind) };
        check("SCIPcreateVarBasic", created)?;
        let var = self.own_var(var)?;
        // SAFETY: the instance is live and `var` a variable of it, not yet in its problem.
        check("SCIPaddVar", unsafe { SCIPaddVar(self.raw(), var.as_ptr()) })
    }

    /// Takes the reference to a variable SCIP created, so dropping the instance releases it.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP reported success but wrote no variable.
    fn own_var(&mut self, var: *mut SCIP_VAR) -> Result<NonNull<SCIP_VAR>, ScipError> {
        let var = NonNull::new(var).ok_or(ScipError::Call {
            call: "SCIPcreateVarBasic",
            code: crate::RetCode::Error,
        })?;
        self.vars.push(var);
        Ok(var)
    }

    /// Creates the linear constraint of `row` and adds it to the problem.
    ///
    /// # Errors
    ///
    /// [`ScipError::BeyondInfinity`], [`ScipError::RowName`], [`ScipError::RowTooLong`] or [`ScipError::Call`].
    ///
    /// # Panics
    ///
    /// When the row names a column without a variable, which a program's rows never do once every column has one.
    fn add_cons(&mut self, program: &Program, row: Row) -> Result<(), ScipError> {
        let view = program.row(row);
        let (lhs, rhs) = self.bounds(Number::RowBound(row), view.bounds())?;
        let len = c_int::try_from(view.len()).map_err(|_| ScipError::RowTooLong { row, len: view.len() })?;
        let mut vars = Vec::with_capacity(view.len());
        let mut values = Vec::with_capacity(view.len());
        for (col, value) in view.entries() {
            vars.push(self.vars[col].as_ptr());
            values.push(self.number(Number::Coefficient { row, col }, value)?);
        }
        let name = c_name(view.name()).map_err(|()| ScipError::RowName { row })?;
        // A constraint's name may be empty but not null; an empty name keeps it out of SCIP's name table.
        let name = name.as_deref().unwrap_or(c"");
        let mut cons: *mut SCIP_CONS = ptr::null_mut();
        // SAFETY: the instance is live, borrowed exclusively and in its problem stage; `cons` is a local to write the
        // new constraint to; `vars` holds `len` variables of this instance and `values` as many coefficients, which
        // SCIP copies; the name is a NUL-terminated string SCIP copies.
        let created = unsafe {
            SCIPcreateConsBasicLinear(
                self.raw(),
                &raw mut cons,
                name.as_ptr(),
                len,
                vars.as_mut_ptr(),
                values.as_mut_ptr(),
                lhs,
                rhs,
            )
        };
        check("SCIPcreateConsBasicLinear", created)?;
        let cons = NonNull::new(cons).ok_or(ScipError::Call {
            call: "SCIPcreateConsBasicLinear",
            code: crate::RetCode::Error,
        })?;
        self.conss.push(cons);
        // SAFETY: the instance is live and `cons` a constraint of it, not yet in its problem.
        check("SCIPaddCons", unsafe { SCIPaddCons(self.raw(), cons.as_ptr()) })
    }

    /// `bounds` in SCIP's terms, an infinite end as SCIP's infinity.
    ///
    /// # Errors
    ///
    /// [`ScipError::BeyondInfinity`] naming `number` for a finite end SCIP would read as infinite.
    fn bounds(&self, number: Number, bounds: Bounds) -> Result<(f64, f64), ScipError> {
        let infinity = self.infinity();
        let end = |value: f64| {
            if value.is_infinite() {
                Ok(value.signum() * infinity)
            } else {
                self.number(number, value)
            }
        };
        Ok((end(bounds.lower())?, end(bounds.upper())?))
    }

    /// A finite number of the program, unchanged.
    ///
    /// # Errors
    ///
    /// [`ScipError::BeyondInfinity`] naming `number` when SCIP would read `value` as infinite.
    pub(super) fn number(&self, number: Number, value: f64) -> Result<f64, ScipError> {
        self.finite(value).ok_or(ScipError::BeyondInfinity {
            number,
            value,
            infinity: self.infinity(),
        })
    }
}

/// A name as a C string, `None` for no name.
///
/// # Errors
///
/// `()` when the name holds a NUL byte.
fn c_name(name: Option<&str>) -> Result<Option<CString>, ()> {
    name.map(CString::new).transpose().map_err(|_| ())
}
