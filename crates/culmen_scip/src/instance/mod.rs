// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One SCIP instance and the problem it holds behind safe methods: the only place that calls SCIP. Building the
//! problem from a [`Program`](culmen_core::Program) is in the `load` submodule, a quadratic objective's epigraph in
//! `epigraph`, parameters in `params`.
//!
//! Every call's `SCIP_RETCODE` goes through [`check`], which turns any code but `SCIP_OKAY` into
//! [`ScipError::Call`] naming the call; parameter calls report [`ScipError::Param`] instead.

mod epigraph;
mod load;
mod params;

pub(crate) use params::Group;

use core::ffi::c_uint;
use core::ptr::{self, NonNull};

use std::time::Instant;

use culmen_core::{Callback, Col, Events, QuadObjective, Row, Solution, Start, TiVec};
use scip_sys::{
    SCIP, SCIP_CONS, SCIP_Retcode_SCIP_OKAY, SCIP_SOL, SCIP_Stage_SCIP_STAGE_PROBLEM, SCIP_VAR, SCIPaddSolFree,
    SCIPcheckSolOrig, SCIPcreate, SCIPcreatePartialSol, SCIPcreateProbBasic, SCIPcreateSol, SCIPfree, SCIPfreeSol,
    SCIPfreeTransform, SCIPgetBestSol, SCIPgetDualbound, SCIPgetNLPIterations, SCIPgetNSolsFound, SCIPgetNTotalNodes,
    SCIPgetStage, SCIPgetStatus, SCIPincludeDefaultPlugins, SCIPinfinity, SCIPreleaseCons, SCIPreleaseVar,
    SCIPsetSolVal, SCIPsolve,
};

use crate::callback::{self, Payload, Run, Slot};
use crate::columns::Columns;
use crate::error::check;
use crate::{Scip, ScipError, ScipStatus, Status};

/// SCIP's `TRUE`.
const TRUE: c_uint = 1;

/// A SCIP instance holding one problem, from creation to drop.
///
/// It owns one reference to every variable and constraint it created, indexed by the column or row they stand for,
/// and to those of a quadratic objective's epigraph; dropping it releases those references and then frees the
/// instance, whatever stage it is in, and so frees everything it created on every path, an early return from a
/// half-built load included.
///
/// Every method that changes SCIP's state borrows the instance exclusively, so no two calls ever overlap on it.
///
/// It owns the slot its callback plugins share, which it frees after SCIP.
#[derive(Debug)]
pub(crate) struct Instance {
    scip: NonNull<SCIP>,
    /// The slot of the callback plugins, from `Box::into_raw`, freed in `Drop` after SCIP.
    slot: NonNull<Slot>,
    vars: TiVec<Col, NonNull<SCIP_VAR>>,
    conss: TiVec<Row, NonNull<SCIP_CONS>>,
    /// The epigraph of a quadratic objective; `None` for a linear one.
    epigraph: Option<Epigraph>,
}

/// What stands for a quadratic part `q(x)` of the objective: a variable `t` in its place, with cost one, bounded by a
/// quadratic constraint, `q(x) - t <= 0` when minimizing and `q(x) - t >= 0` when maximizing.
#[derive(Debug)]
struct Epigraph {
    /// The variable `t`.
    var: NonNull<SCIP_VAR>,
    /// The constraint bounding `t`; `None` until it is created.
    cons: Option<NonNull<SCIP_CONS>>,
    /// The quadratic part, which values a solution's `q(x)` in place of `t`.
    quadratic: QuadObjective,
}

// SAFETY: SCIP is built with `SCIP_THREADSAFE` (its `config.h` defines it), which keeps every piece of an instance's
// state in the instance and guards SCIP's few process-wide statics, so an instance may be used from any one thread
// at a time. `Instance` hands out no pointer into SCIP, and its `&mut` methods keep calls from overlapping. It is
// not `Sync`: SCIP's getters take a mutable `SCIP*` and promise nothing about concurrent calls.
unsafe impl Send for Instance {}

impl Instance {
    /// A new instance with SCIP's default plugins and culmen's callback plugins, quiet, holding an empty problem.
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
        // From here on, dropping the instance frees it and then the slot.
        let instance = Self {
            scip,
            slot: NonNull::from(Box::leak(Box::new(Slot::new(raw)))),
            vars: TiVec::new(),
            conss: TiVec::new(),
            epigraph: None,
        };
        // SAFETY: the instance is live and in its init stage, and the slot lives until after SCIP is freed. The
        // handler replaces SCIP's default one, which would print to standard output.
        unsafe { callback::install_messages(instance.raw(), instance.slot) }?;
        // SAFETY: the instance is live and in its init stage, where plugins are included.
        check("SCIPincludeDefaultPlugins", unsafe {
            SCIPincludeDefaultPlugins(instance.raw())
        })?;
        // SAFETY: as above.
        unsafe { callback::include(instance.raw(), instance.slot) }?;
        // SAFETY: the instance is live with its plugins, and the name a NUL-terminated string SCIP copies.
        check("SCIPcreateProbBasic", unsafe {
            SCIPcreateProbBasic(instance.raw(), c"culmen".as_ptr())
        })?;
        Ok(instance)
    }

    fn raw(&self) -> *mut SCIP {
        self.scip.as_ptr()
    }

    /// The slot of the callback plugins.
    fn slot(&self) -> &Slot {
        // SAFETY: the slot lives from the instance's creation until after SCIP is freed in `Drop`, and is only ever
        // reached through shared references.
        unsafe { self.slot.as_ref() }
    }

    /// Whether SCIP's messages reach standard output and standard error.
    pub(crate) fn set_console(&mut self, console: bool) {
        self.slot().set_console(console);
    }

    /// The columns' variables and the epigraph.
    fn columns(&self) -> Columns<'_> {
        Columns {
            vars: &self.vars,
            epigraph: self.epigraph.as_ref().map(|e| (e.var, &e.quadratic)),
        }
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

    /// Solves the problem, delivering the events `wanted` to `callback`; a panic of the callback is returned, not
    /// resumed.
    ///
    /// The callback's run is installed in the slot for the call of `SCIPsolve` alone, and SCIP reads the columns
    /// through shared references during it.
    pub(crate) fn solve(&mut self, callback: &mut dyn Callback<Scip>, wanted: Events, start: Instant) -> Ran {
        let this: &Self = self;
        // SAFETY: the instance is live, holds the columns, and is in its problem stage, as configuring left it.
        let run = unsafe { Run::new(this.raw(), callback, wanted, start, this.columns()) };
        let solved = {
            let _installed = this.slot().install(&run);
            // SAFETY: the instance is live and borrowed exclusively by the caller, and holds its problem; the run its
            // plugins reach lives until after the guard uninstalls it.
            check("SCIPsolve", unsafe { SCIPsolve(this.raw()) })
        };
        let (panic, error) = run.finish();
        Ran {
            result: error.map_or(solved, Err),
            panic,
        }
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
        // SAFETY: the instance is live after a solve, in a stage where it reads its own solution `sol`.
        let (objective, values) = unsafe { self.columns().read(self.raw(), sol) };
        Some(Solution { objective, values })
    }

    /// Whether SCIP's own check finds the best solution of the last solve feasible in the original problem, its
    /// bounds, integrality and constraints, to SCIP's feasibility tolerance (`SCIPcheckSolOrig`); `true` when SCIP
    /// holds no solution.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses the check.
    pub(crate) fn best_feasible(&self) -> Result<bool, ScipError> {
        // SAFETY: the instance is live; the result is null or a solution of this instance.
        let Some(sol) = NonNull::new(unsafe { SCIPgetBestSol(self.raw()) }) else {
            return Ok(true);
        };
        let mut feasible: c_uint = 0;
        // SAFETY: the instance is live after a solve, in a stage from its transformed stage to its solved stage, where
        // SCIP checks a complete solution of its own; `feasible` is a local to write to. No reason is printed.
        check("SCIPcheckSolOrig", unsafe {
            SCIPcheckSolOrig(self.raw(), sol.as_ptr(), &raw mut feasible, 0, 0)
        })?;
        Ok(feasible != 0)
    }
}

/// What came of a solve: SCIP's result, or the first error a plugin met, and a panic of the callback to resume.
#[derive(Debug)]
pub(crate) struct Ran {
    pub(crate) result: Result<(), ScipError>,
    pub(crate) panic: Option<Payload>,
}

impl Drop for Instance {
    fn drop(&mut self) {
        let raw = self.raw();
        let released = |code| code == SCIP_Retcode_SCIP_OKAY || std::thread::panicking();
        let (epigraph_var, epigraph_cons) = match self.epigraph.take() {
            Some(Epigraph { var, cons, .. }) => (Some(var), cons),
            None => (None, None),
        };
        if let Some(mut cons) = epigraph_cons.map(NonNull::as_ptr) {
            // SAFETY: the instance is live and `cons` the one reference to the constraint this instance owns.
            let code = unsafe { SCIPreleaseCons(raw, &raw mut cons) };
            debug_assert!(
                released(code),
                "Instance: SCIP failed to release the epigraph's constraint: {code}"
            );
        }
        if let Some(mut var) = epigraph_var.map(NonNull::as_ptr) {
            // SAFETY: the instance is live and `var` the one reference to the variable this instance owns.
            let code = unsafe { SCIPreleaseVar(raw, &raw mut var) };
            debug_assert!(
                released(code),
                "Instance: SCIP failed to release the epigraph's variable: {code}"
            );
        }
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
        // SAFETY: the slot came from `Box::into_raw` and SCIP, whose plugins were the only other holders of it, is
        // freed; it is freed once, here.
        drop(unsafe { Box::from_raw(self.slot.as_ptr()) });
    }
}
