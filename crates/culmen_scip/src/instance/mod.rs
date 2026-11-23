// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One SCIP instance and the problem it holds behind safe methods: the only place that calls SCIP. Building the
//! problem from a [`Program`](culmen_core::Program) is in the `load` submodule, parameters in `params`.
//!
//! Every call's `SCIP_RETCODE` goes through [`check`], which turns any code but `SCIP_OKAY` into
//! [`ScipError::Call`] naming the call; parameter calls report [`ScipError::Param`] instead.

mod load;
mod params;

pub(crate) use params::Group;

use core::ffi::c_uint;
use core::ptr::{self, NonNull};

use culmen_core::{Col, Row, Solution, Start, TiVec};
use scip_sys::{
    SCIP, SCIP_CONS, SCIP_Retcode_SCIP_OKAY, SCIP_SOL, SCIP_Stage_SCIP_STAGE_PROBLEM, SCIP_VAR, SCIPaddSolFree,
    SCIPcreate, SCIPcreatePartialSol, SCIPcreateProbBasic, SCIPcreateSol, SCIPfree, SCIPfreeSol, SCIPfreeTransform,
    SCIPgetBestSol, SCIPgetDualbound, SCIPgetNLPIterations, SCIPgetNSolsFound, SCIPgetNTotalNodes, SCIPgetSolOrigObj,
    SCIPgetSolVal, SCIPgetStage, SCIPgetStatus, SCIPincludeDefaultPlugins, SCIPinfinity, SCIPreleaseCons,
    SCIPreleaseVar, SCIPsetMessagehdlrQuiet, SCIPsetSolVal, SCIPsolve,
};

use crate::error::check;
use crate::{ScipError, ScipStatus, Status};

/// SCIP's `TRUE`.
const TRUE: c_uint = 1;

/// A SCIP instance holding one problem, from creation to drop.
///
/// It owns one reference to every variable and constraint it created, indexed by the column or row they stand for;
/// dropping it releases those references and then frees the instance, whatever stage it is in, and so frees
/// everything it created on every path, an early return from a half-built load included.
///
/// Every method that changes SCIP's state borrows the instance exclusively, so no two calls ever overlap on it.
#[derive(Debug)]
pub(crate) struct Instance {
    scip: NonNull<SCIP>,
    vars: TiVec<Col, NonNull<SCIP_VAR>>,
    conss: TiVec<Row, NonNull<SCIP_CONS>>,
}

// SAFETY: SCIP is built with `SCIP_THREADSAFE` (its `config.h` defines it), which keeps every piece of an instance's
// state in the instance and guards SCIP's few process-wide statics, so an instance may be used from any one thread
// at a time. `Instance` hands out no pointer into SCIP, and its `&mut` methods keep calls from overlapping. It is
// not `Sync`: SCIP's getters take a mutable `SCIP*` and promise nothing about concurrent calls.
unsafe impl Send for Instance {}

impl Instance {
    /// A new instance with SCIP's default plugins, quiet, holding an empty problem.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP cannot create the instance, include its plugins or create the problem.
    fn new() -> Result<Self, ScipError> {
        let mut raw: *mut SCIP = ptr::null_mut();
        // SAFETY: `SCIPcreate` writes a new instance to the pointer it is given, which points to a local.
        check("SCIPcreate", unsafe { SCIPcreate(&raw mut raw) })?;
        let scip = NonNull::new(raw).ok_or(ScipError::Call {
            call: "SCIPcreate",
            code: crate::RetCode::NoMemory,
        })?;
        // From here on, dropping the instance frees it.
        let instance = Self {
            scip,
            vars: TiVec::new(),
            conss: TiVec::new(),
        };
        // SAFETY: the instance is live. Its default message handler would print to standard output.
        unsafe { SCIPsetMessagehdlrQuiet(instance.raw(), TRUE) };
        // SAFETY: the instance is live and in its init stage, where plugins are included.
        check("SCIPincludeDefaultPlugins", unsafe {
            SCIPincludeDefaultPlugins(instance.raw())
        })?;
        // SAFETY: the instance is live with its plugins, and the name a NUL-terminated string SCIP copies.
        check("SCIPcreateProbBasic", unsafe {
            SCIPcreateProbBasic(instance.raw(), c"culmen".as_ptr())
        })?;
        Ok(instance)
    }

    fn raw(&self) -> *mut SCIP {
        self.scip.as_ptr()
    }

    /// The number of columns, one variable each.
    pub(crate) fn num_cols(&self) -> usize {
        self.vars.len()
    }

    /// SCIP's infinity: a number of at least this magnitude is infinite to SCIP.
    fn infinity(&self) -> f64 {
        // SAFETY: the instance is live; the call reads one parameter.
        unsafe { SCIPinfinity(self.raw()) }
    }

    /// A finite value of SCIP's, `None` for one SCIP reads as infinite.
    fn finite(&self, value: f64) -> Option<f64> {
        (value.abs() < self.infinity()).then_some(value)
    }

    /// Frees the last solve's transformed problem, so SCIP holds the problem as loaded and its parameters and
    /// original solutions may change. Does nothing before the first solve.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP fails to free it.
    fn free_transform(&mut self) -> Result<(), ScipError> {
        // SAFETY: the instance is live and borrowed exclusively; SCIP takes the call in every stage it can be in here.
        check("SCIPfreeTransform", unsafe { SCIPfreeTransform(self.raw()) })?;
        // SAFETY: the instance is live.
        let stage = unsafe { SCIPgetStage(self.raw()) };
        debug_assert_eq!(
            stage, SCIP_Stage_SCIP_STAGE_PROBLEM,
            "Instance: SCIP is not in its problem stage after freeing the transformed problem"
        );
        Ok(())
    }

    /// Hands SCIP a start for the next solve: a full start as a solution, a sparse one as a partial solution SCIP
    /// completes. Frees the last solve's transformed problem first.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Panics
    ///
    /// When the start names a column the instance lacks, which [`Start::check`] rules out.
    pub(crate) fn add_start(&mut self, start: &Start) -> Result<(), ScipError> {
        self.free_transform()?;
        let mut sol: *mut SCIP_SOL = ptr::null_mut();
        let created = match start {
            // SAFETY: the instance is live, in its problem stage, and `sol` a local to write the new solution to.
            Start::Full(_) => check("SCIPcreateSol", unsafe {
                SCIPcreateSol(self.raw(), &raw mut sol, ptr::null_mut())
            }),
            // SAFETY: as above.
            Start::Sparse(_) => check("SCIPcreatePartialSol", unsafe {
                SCIPcreatePartialSol(self.raw(), &raw mut sol, ptr::null_mut())
            }),
        };
        created?;
        if let Err(error) = self.fill(sol, start) {
            // SAFETY: `sol` was created above on this live instance and is freed once, here.
            let freed = unsafe { SCIPfreeSol(self.raw(), &raw mut sol) };
            debug_assert_eq!(
                freed, SCIP_Retcode_SCIP_OKAY,
                "Instance: SCIP failed to free a solution"
            );
            return Err(error);
        }
        // SCIP stores a solution added in its problem stage among its original solutions, which it checks and
        // completes when the next solve starts. It leaves `stored` false only for a solution it already holds or one
        // worse than the `limits/maxorigsol` solutions it holds, the last solve's best among them, which could not
        // improve the next solve's incumbent.
        let mut stored: c_uint = 0;
        // SAFETY: `sol` is a solution of this instance, which SCIP takes over and frees; `stored` a local to write.
        check("SCIPaddSolFree", unsafe {
            SCIPaddSolFree(self.raw(), &raw mut sol, &raw mut stored)
        })
    }

    /// Sets the start's values in `sol`.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a value.
    ///
    /// # Panics
    ///
    /// When the start names a column the instance lacks.
    fn fill(&mut self, sol: *mut SCIP_SOL, start: &Start) -> Result<(), ScipError> {
        for (col, value) in start.entries() {
            let var = self.vars[col].as_ptr();
            // SAFETY: the instance is live, `sol` one of its solutions and `var` one of its variables.
            check("SCIPsetSolVal", unsafe { SCIPsetSolVal(self.raw(), sol, var, value) })?;
        }
        Ok(())
    }

    /// Solves the problem.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP fails.
    pub(crate) fn solve(&mut self) -> Result<(), ScipError> {
        // SAFETY: the instance is live and borrowed exclusively, and holds its problem.
        check("SCIPsolve", unsafe { SCIPsolve(self.raw()) })
    }

    /// SCIP's account of the last solve.
    pub(crate) fn status(&self) -> ScipStatus {
        let count = |n: i64| u64::try_from(n).unwrap_or(0);
        // SAFETY: the instance is live; the getters read the statistics of the last solve.
        let (status, dual_bound, nodes, lp_iterations, solutions_found) = unsafe {
            (
                SCIPgetStatus(self.raw()),
                SCIPgetDualbound(self.raw()),
                SCIPgetNTotalNodes(self.raw()),
                SCIPgetNLPIterations(self.raw()),
                SCIPgetNSolsFound(self.raw()),
            )
        };
        ScipStatus {
            status: Status::from_raw(status),
            dual_bound: self.finite(dual_bound),
            nodes: count(nodes),
            lp_iterations: count(lp_iterations),
            solutions_found: count(solutions_found),
        }
    }

    /// The best solution of the last solve, its objective in the program's sense with the offset; `None` when SCIP
    /// holds none.
    pub(crate) fn best(&self) -> Option<Solution> {
        // SAFETY: the instance is live; the result is null or a solution of this instance.
        let sol = NonNull::new(unsafe { SCIPgetBestSol(self.raw()) })?.as_ptr();
        let values = self
            .vars
            .iter()
            // SAFETY: the instance is live, `sol` one of its solutions and each variable one of its variables.
            .map(|var| unsafe { SCIPgetSolVal(self.raw(), sol, var.as_ptr()) })
            .collect();
        // SAFETY: as above.
        let objective = unsafe { SCIPgetSolOrigObj(self.raw(), sol) };
        Some(Solution { objective, values })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        let raw = self.raw();
        let released = |code| code == SCIP_Retcode_SCIP_OKAY || std::thread::panicking();
        for mut cons in self.conss.drain(..).map(NonNull::as_ptr) {
            // SAFETY: the instance is live and `cons` the one reference to the constraint this instance owns.
            let code = unsafe { SCIPreleaseCons(raw, &raw mut cons) };
            debug_assert!(released(code), "Instance: SCIP failed to release a constraint: {code}");
        }
        for mut var in self.vars.drain(..).map(NonNull::as_ptr) {
            // SAFETY: the instance is live and `var` the one reference to the variable this instance owns.
            let code = unsafe { SCIPreleaseVar(raw, &raw mut var) };
            debug_assert!(released(code), "Instance: SCIP failed to release a variable: {code}");
        }
        let mut raw = raw;
        // SAFETY: the instance is live, this instance holds no further reference into it, and it is freed once.
        let code = unsafe { SCIPfree(&raw mut raw) };
        debug_assert!(released(code), "Instance: SCIP failed to free the instance: {code}");
    }
}
