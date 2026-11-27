// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lazy constraints: a constraint handler without constraints of its own, checked and enforced after every other.
//!
//! The handler sees every candidate SCIP would accept: each solution SCIP checks (`CONSCHECK`, from a heuristic, a
//! start, a sub-SCIP or a supplied solution) and each LP, pseudo or relaxation solution that every other handler
//! found feasible (`CONSENFOLP`, `CONSENFOPS`, `CONSENFORELAX`). It keeps every constraint the callback adds for the
//! rest of the solve, so a later candidate is held to all of them:
//!
//! - A candidate that violates a constraint kept before is rejected without asking the callback.
//! - Otherwise the callback is asked. A candidate violating a constraint it adds is rejected.
//! - SCIP allows no constraint to be added while it checks a solution, so a constraint added at a check waits in the
//!   handler, which rejects every later candidate violating it, until the next enforcement adds it to SCIP's problem.
//! - An enforcement adds each waiting constraint to SCIP's transformed problem as a global linear constraint
//!   (`SCIPaddCons`), which SCIP's linear handler then enforces, separates and checks with everything else, and
//!   reports `CONSADDED` when the candidate violates one, so SCIP solves the node again with it.
//!
//! A constraint added again, the same terms and bounds as one kept, is not added twice and rejects nothing new, so a
//! callback that keeps adding one constraint cannot keep SCIP enforcing the same candidate.
//!
//! The handler may constrain any variable, so its lock (`CONSLOCK`) locks every variable of the transformed problem
//! both ways, and each variable a pricer adds both ways as it is created, which keeps presolve and propagation from
//! fixing a variable by a dual argument that a constraint added later would refute; its unlock undoes exactly the
//! locks it applied, before SCIP frees the variables. It does so only for a solve with a lazy-constraint handler;
//! that solve also turns off SCIP's symmetry handling and its dual reductions, which reason about the constraints
//! SCIP holds, and its copies are invalid, so a sub-SCIP's solution is checked by the main SCIP and its components
//! presolver does not run.
//!
//! An enforcement that meets an error, SCIP refusing a constraint, keeps it, asks SCIP to stop and cuts its node off
//! instead of returning an error code, which would leave SCIP's delay of the handler's constraint updates in place
//! (`cons.c:3525-3531` of SCIP 10.0.2) and abort a debug SCIP at the next solve of the instance (`cons.c:2524`). Every
//! later enforcement cuts its node off until SCIP stops, and the solve fails with the error.

use core::ffi::{c_int, c_uint};
use core::ptr::{self, NonNull};

use culmen_core::{ActionError, AddedRow, Col, Constraint, Context, LazyContext, Progress, TiSlice};
use scip_sys::{
    SCIP, SCIP_CONS, SCIP_CONSHDLR, SCIP_LOCKTYPE, SCIP_RESULT, SCIP_RETCODE, SCIP_Result_SCIP_CONSADDED,
    SCIP_Result_SCIP_CUTOFF, SCIP_Result_SCIP_FEASIBLE, SCIP_Result_SCIP_INFEASIBLE, SCIP_Retcode_SCIP_OKAY, SCIP_SOL,
    SCIP_VAR, SCIPaddCons, SCIPconshdlrGetData, SCIPcreateConsBasicLinear, SCIPgetNOrigVars, SCIPgetOrigVars,
    SCIPincludeConshdlrBasic, SCIPisFeasGT, SCIPisFeasLT, SCIPreleaseCons, SCIPsetConshdlrCopy,
    SCIPsetConshdlrEnforelax, SCIPsolIsOriginal,
};

use super::locks::Applied;
use super::{Delivered, Run, Slot, checked, sides, slot};
use crate::columns::Columns;
use crate::error::check;
use crate::{RetCode, ScipError};

/// After every constraint handler of SCIP's in enforcement and in checks; SCIP's own priorities stay above
/// `-4_000_000`.
const PRIORITY: c_int = -9_999_999;

/// The constraints a callback added during one solve.
#[derive(Debug, Default)]
pub(super) struct Store {
    rows: Vec<Kept>,
    /// The last candidate the callback accepted, so SCIP checking the solution it just enforced does not ask again.
    accepted: Option<Vec<f64>>,
}

/// A constraint the callback added.
#[derive(Debug)]
struct Kept {
    row: AddedRow,
    /// Whether SCIP's problem holds it; until then the handler alone enforces it.
    in_scip: bool,
}

/// Which of the kept constraints a candidate is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// Every one: an original solution, which SCIP checks against the original problem, without the constraints
    /// added to its transformed problem.
    All,
    /// Those SCIP's problem does not hold yet: SCIP's linear handler checks the others.
    Waiting,
}

impl Store {
    /// Whether `values` violates a kept constraint of those `held`.
    fn violated(&self, scip: *mut SCIP, values: &TiSlice<Col, f64>, held: Held) -> bool {
        let held = |kept: &&Kept| held == Held::All || !kept.in_scip;
        self.rows
            .iter()
            .filter(held)
            .any(|kept| violates(scip, &kept.row, values))
    }

    /// Whether the callback accepted `values` last.
    fn accepted(&self, values: &TiSlice<Col, f64>) -> bool {
        self.accepted.as_deref() == Some(&values.raw)
    }

    /// Keeps the constraints not kept yet.
    fn keep(&mut self, rows: Vec<AddedRow>) {
        for row in rows {
            if !self.rows.iter().any(|kept| kept.row == row) {
                self.rows.push(Kept { row, in_scip: false });
            }
        }
    }
}

/// Whether `values` violates `row` beyond SCIP's feasibility tolerance, as SCIP's linear handler judges a row.
fn violates(scip: *mut SCIP, row: &AddedRow, values: &TiSlice<Col, f64>) -> bool {
    let activity: f64 = row.terms.iter().map(|&(col, a)| a * values[col]).sum();
    let (lower, upper) = (row.bounds.lower(), row.bounds.upper());
    // SAFETY: the instance is live; the comparisons read its tolerance.
    unsafe {
        (lower.is_finite() && SCIPisFeasLT(scip, activity, lower) != 0)
            || (upper.is_finite() && SCIPisFeasGT(scip, activity, upper) != 0)
    }
}

/// Where a lazy-constraint candidate comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScipLazySource {
    /// A solution SCIP checks before storing it: a heuristic's, a start, a sub-SCIP's, a supplied one (`CONSCHECK`).
    Check,
    /// The LP solution of a node, which every other constraint handler found feasible (`CONSENFOLP`).
    Lp,
    /// The pseudo solution of a node whose LP is not solved, every variable at its better bound (`CONSENFOPS`).
    Pseudo,
    /// A relaxator's solution (`CONSENFORELAX`).
    Relaxation,
}

/// A candidate solution SCIP would accept, to be cut off with lazy constraints: the constraint handler's `CONSCHECK`
/// or enforcement.
///
/// The candidate satisfies every lazy constraint added before in this solve, to SCIP's feasibility tolerance.
/// A constraint added here holds for the rest of the solve, also for candidates SCIP checks while it cannot add a
/// constraint: SCIP's problem takes it at the next enforcement, and until then the handler holds every candidate to
/// it.
#[derive(Debug)]
pub struct ScipLazy<'a> {
    progress: Progress,
    objective: f64,
    candidate: &'a TiSlice<Col, f64>,
    source: ScipLazySource,
    infinity: f64,
    added: &'a mut Vec<AddedRow>,
}

impl ScipLazy<'_> {
    /// Where the candidate comes from.
    #[must_use]
    pub const fn source(&self) -> ScipLazySource {
        self.source
    }
}

impl Context for ScipLazy<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl LazyContext for ScipLazy<'_> {
    fn objective(&self) -> f64 {
        self.objective
    }
    fn candidate(&self) -> &TiSlice<Col, f64> {
        self.candidate
    }
    /// SCIP takes the constraint once the callback returns.
    ///
    /// # Errors
    ///
    /// Besides the core errors, [`ActionError::Refused`] for a number SCIP would read as infinite, or more terms
    /// than SCIP's `int` counts.
    fn add_lazy(&mut self, constraint: Constraint) -> Result<(), ActionError> {
        let row = checked(&constraint, self.candidate.len(), self.infinity)?;
        self.added.push(row);
        Ok(())
    }
}

impl Run<'_, '_> {
    /// Asks the callback about a candidate; the constraints it added, `None` when it did not run.
    fn ask_lazy(
        &self,
        scip: *mut SCIP,
        objective: f64,
        candidate: &TiSlice<Col, f64>,
        source: ScipLazySource,
    ) -> Option<Vec<AddedRow>> {
        let progress = self.progress(scip);
        let mut added = Vec::new();
        let delivered = self.deliver(scip, |hooks| {
            hooks.lazy(&mut ScipLazy {
                progress,
                objective,
                candidate,
                source,
                infinity: self.infinity,
                added: &mut added,
            });
        });
        (delivered == Delivered::Ran).then_some(added)
    }

    /// Adds every waiting constraint to SCIP's problem; whether `values` violates one of them.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, solving.
    unsafe fn add_waiting(&self, scip: *mut SCIP, values: &TiSlice<Col, f64>) -> Result<bool, ScipError> {
        let waiting: Vec<(usize, AddedRow)> = {
            let store = self.lazy.borrow();
            let rows = store.rows.iter().enumerate();
            rows.filter(|(_, kept)| !kept.in_scip)
                .map(|(i, kept)| (i, kept.row.clone()))
                .collect()
        };
        let mut violated = false;
        for (i, row) in waiting {
            #[cfg(test)]
            if self.slot.refuse_lazy.get() {
                return Err(ScipError::Call {
                    call: "SCIPaddCons",
                    code: RetCode::Error,
                });
            }
            // The store is not borrowed while SCIP runs, which may write a message the callback is handed.
            // SAFETY: the caller's contract is `add_cons`'s.
            unsafe { add_cons(scip, self.cols, &self.priced_vars.borrow(), &row, self.infinity) }?;
            self.lazy.borrow_mut().rows[i].in_scip = true;
            violated |= violates(scip, &row, values);
        }
        Ok(violated)
    }

    /// Judges a solution SCIP checks.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run and `sol` a solution SCIP checks, in a stage where SCIP reads it.
    unsafe fn check(&self, scip: *mut SCIP, sol: *mut SCIP_SOL) -> SCIP_RESULT {
        if self.panicked() {
            return SCIP_Result_SCIP_INFEASIBLE;
        }
        // SAFETY: the caller vouches for the instance, `sol` and the stage.
        let (objective, values) = unsafe { self.read(scip, sol) };
        // SAFETY: as above.
        let original = !sol.is_null() && unsafe { SCIPsolIsOriginal(sol) } != 0;
        let held = if original { Held::All } else { Held::Waiting };
        if self.lazy.borrow().violated(scip, &values, held) {
            return SCIP_Result_SCIP_INFEASIBLE;
        }
        if self.lazy.borrow().accepted(&values) {
            return SCIP_Result_SCIP_FEASIBLE;
        }
        // A callback busy with an outer event cannot judge the candidate, so it is rejected; the heuristic context
        // keeps the solutions it is handed in a run asking for lazy constraints until its callback returns, so no
        // candidate meets a busy callback.
        let Some(rows) = self.ask_lazy(scip, objective, &values, ScipLazySource::Check) else {
            return SCIP_Result_SCIP_INFEASIBLE;
        };
        let mut store = self.lazy.borrow_mut();
        let rejected = rows
            .iter()
            .filter(|row| !store.rows.iter().any(|kept| kept.row == **row))
            .any(|row| violates(scip, row, &values));
        store.keep(rows);
        if rejected {
            SCIP_Result_SCIP_INFEASIBLE
        } else {
            store.accepted = Some(values.raw);
            SCIP_Result_SCIP_FEASIBLE
        }
    }

    /// Enforces the lazy constraints on a node's solution, `sol`, or the current LP or pseudo solution for null.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses a call.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance of the run, solving, and `sol` null or a solution of it.
    unsafe fn enforce(
        &self,
        scip: *mut SCIP,
        sol: *mut SCIP_SOL,
        source: ScipLazySource,
        infeasible: bool,
    ) -> Result<SCIP_RESULT, ScipError> {
        // SCIP is being stopped: every node is closed until it does.
        if self.stopping() {
            return Ok(SCIP_Result_SCIP_CUTOFF);
        }
        // SAFETY: the caller vouches for the instance, its stage and `sol`.
        let (objective, values) = unsafe { self.read(scip, sol) };
        // SAFETY: as above.
        if unsafe { self.add_waiting(scip, &values) }? {
            return Ok(SCIP_Result_SCIP_CONSADDED);
        }
        // A candidate another handler rejected is no candidate; the constraints waiting are in SCIP's problem now.
        if infeasible || self.lazy.borrow().accepted(&values) {
            return Ok(SCIP_Result_SCIP_FEASIBLE);
        }
        let Some(rows) = self.ask_lazy(scip, objective, &values, source) else {
            // The callback panicked, and SCIP is asked to stop; or it is busy, which no enforcement meets.
            return Ok(if self.panicked() {
                SCIP_Result_SCIP_CUTOFF
            } else {
                SCIP_Result_SCIP_INFEASIBLE
            });
        };
        self.lazy.borrow_mut().keep(rows);
        // SAFETY: as above.
        if unsafe { self.add_waiting(scip, &values) }? {
            return Ok(SCIP_Result_SCIP_CONSADDED);
        }
        self.lazy.borrow_mut().accepted = Some(values.raw);
        Ok(SCIP_Result_SCIP_FEASIBLE)
    }
}

/// Adds `row` to SCIP's transformed problem as a global linear constraint over the columns' transformed variables,
/// the priced ones `priced` included.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is the live instance holding `cols` and `priced`, solving.
unsafe fn add_cons(
    scip: *mut SCIP,
    cols: Columns<'_>,
    priced: &[NonNull<SCIP_VAR>],
    row: &AddedRow,
    infinity: f64,
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
            c"lazy".as_ptr(),
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

/// Locks every variable of the transformed problem `n` times both ways, through the original variables, which SCIP
/// keeps from the transforming stage until the transformed problem is freed, and keeps the locks in `applied`.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance from its transforming stage until it frees its transformed problem, not in its solved
/// stage.
unsafe fn lock_all(scip: *mut SCIP, applied: &Applied, n: c_int) -> Result<(), ScipError> {
    // SAFETY: the instance is live in a stage where it holds its original variables.
    let (vars, len) = unsafe { (SCIPgetOrigVars(scip), SCIPgetNOrigVars(scip)) };
    let len = usize::try_from(len).unwrap_or(0);
    if vars.is_null() || len == 0 {
        return Ok(());
    }
    // SAFETY: SCIP's array of original variables holds `len` of them, unchanged during this call.
    let vars = unsafe { core::slice::from_raw_parts(vars, len) };
    for &var in vars {
        // SAFETY: the instance is live and transformed, and `var` one of its original variables.
        let var = unsafe { crate::columns::transformed(scip, var) }?;
        let var = NonNull::new(var).ok_or(ScipError::Call {
            call: "SCIPgetTransformedVar",
            code: RetCode::InvalidData,
        })?;
        // SAFETY: as above; `var` is a transformed variable of the instance.
        unsafe { applied.lock(scip, var, n, n) }?;
    }
    Ok(())
}

/// Includes the handler.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(super) unsafe fn include(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    let mut conshdlr: *mut SCIP_CONSHDLR = ptr::null_mut();
    // SAFETY: the instance is live and in its init stage; the name and description are NUL-terminated strings SCIP
    // copies; the callbacks treat their arguments as SCIP passes them, and the data is the slot, which outlives SCIP.
    check("SCIPincludeConshdlrBasic", unsafe {
        SCIPincludeConshdlrBasic(
            scip,
            &raw mut conshdlr,
            c"culmen".as_ptr(),
            c"lazy constraints of a culmen callback".as_ptr(),
            PRIORITY,
            PRIORITY,
            -1,
            0,
            Some(enfolp),
            Some(enfops),
            Some(conscheck),
            Some(conslock),
            slot.as_ptr().cast(),
        )
    })?;
    // SAFETY: `conshdlr` is the handler just included in the live instance; the callbacks treat their arguments as
    // SCIP passes them.
    unsafe {
        check(
            "SCIPsetConshdlrEnforelax",
            SCIPsetConshdlrEnforelax(scip, conshdlr, Some(enforelax)),
        )?;
        check(
            "SCIPsetConshdlrCopy",
            SCIPsetConshdlrCopy(scip, conshdlr, Some(copy), None),
        )
    }
}

/// The handler's slot and its run, `None` when no run with a lazy-constraint handler is installed.
///
/// # Safety
///
/// `conshdlr` is the handler, called by SCIP.
unsafe fn lazy_run<'s>(scip: *mut SCIP, conshdlr: *mut SCIP_CONSHDLR) -> Option<&'s Run<'s, 's>> {
    // SAFETY: the handler's data is its slot.
    let slot = unsafe { slot::<_>(SCIPconshdlrGetData(conshdlr)) }?;
    // SAFETY: SCIP calls the handler during the solve the run is installed for, or with none installed.
    let run = unsafe { slot.run() }?;
    run.pass_stop(scip);
    run.wants_lazy.then_some(run)
}

/// `CONSCHECK`: the candidate is feasible unless it violates a lazy constraint.
unsafe extern "C" fn conscheck(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    _conss: *mut *mut SCIP_CONS,
    _nconss: c_int,
    sol: *mut SCIP_SOL,
    _checkintegrality: c_uint,
    _checklprows: c_uint,
    _printreason: c_uint,
    _completely: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    // SAFETY: SCIP calls the handler's check with itself and a result to write.
    let verdict = match unsafe { lazy_run(scip, conshdlr) } {
        // SAFETY: SCIP checks `sol` in a stage where it reads it.
        Some(run) => unsafe { run.check(scip, sol) },
        None => SCIP_Result_SCIP_FEASIBLE,
    };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(verdict) };
    SCIP_Retcode_SCIP_OKAY
}

/// Enforces on `sol` for a callback of the handler, and writes the result.
///
/// # Safety
///
/// SCIP calls the handler's enforcement with itself, `sol` null or a solution of its own, and `result` to write.
unsafe fn enforced(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    sol: *mut SCIP_SOL,
    source: ScipLazySource,
    infeasible: bool,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    // SAFETY: the caller's contract.
    let (verdict, code) = match unsafe { lazy_run(scip, conshdlr) } {
        // SAFETY: SCIP enforces while solving, on `sol`.
        Some(run) => match unsafe { run.enforce(scip, sol, source, infeasible) } {
            Ok(verdict) => (verdict, SCIP_Retcode_SCIP_OKAY),
            // The solve fails with the error once SCIP has stopped; the node is closed meanwhile.
            Err(error) => (SCIP_Result_SCIP_CUTOFF, run.stop_with(scip, error)),
        },
        None => (SCIP_Result_SCIP_FEASIBLE, SCIP_Retcode_SCIP_OKAY),
    };
    // SAFETY: `result` is SCIP's to write for the call.
    unsafe { result.write(verdict) };
    code
}

/// `CONSENFOLP`: the node's LP solution.
unsafe extern "C" fn enfolp(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    _conss: *mut *mut SCIP_CONS,
    _nconss: c_int,
    _nusefulconss: c_int,
    solinfeasible: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    let infeasible = solinfeasible != 0;
    // SAFETY: SCIP calls the enforcement while solving; null is the current LP solution.
    unsafe { enforced(scip, conshdlr, ptr::null_mut(), ScipLazySource::Lp, infeasible, result) }
}

/// `CONSENFOPS`: the node's pseudo solution.
unsafe extern "C" fn enfops(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    _conss: *mut *mut SCIP_CONS,
    _nconss: c_int,
    _nusefulconss: c_int,
    solinfeasible: c_uint,
    objinfeasible: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    // A pseudo solution worse than the incumbent ends its node whatever the handler says.
    let infeasible = solinfeasible != 0 || objinfeasible != 0;
    // SAFETY: SCIP calls the enforcement while solving; null is the current pseudo solution.
    unsafe {
        enforced(
            scip,
            conshdlr,
            ptr::null_mut(),
            ScipLazySource::Pseudo,
            infeasible,
            result,
        )
    }
}

/// `CONSENFORELAX`: a relaxator's solution.
unsafe extern "C" fn enforelax(
    scip: *mut SCIP,
    sol: *mut SCIP_SOL,
    conshdlr: *mut SCIP_CONSHDLR,
    _conss: *mut *mut SCIP_CONS,
    _nconss: c_int,
    _nusefulconss: c_int,
    solinfeasible: c_uint,
    result: *mut SCIP_RESULT,
) -> SCIP_RETCODE {
    let infeasible = solinfeasible != 0;
    // SAFETY: SCIP calls the enforcement while solving, with the relaxation's solution.
    unsafe { enforced(scip, conshdlr, sol, ScipLazySource::Relaxation, infeasible, result) }
}

/// `CONSLOCK`: every variable both ways for a solve with a lazy-constraint handler, undone when SCIP unlocks.
unsafe extern "C" fn conslock(
    scip: *mut SCIP,
    conshdlr: *mut SCIP_CONSHDLR,
    _cons: *mut SCIP_CONS,
    _locktype: SCIP_LOCKTYPE,
    nlockspos: c_int,
    nlocksneg: c_int,
) -> SCIP_RETCODE {
    // SAFETY: the handler's data is its slot.
    let Some(slot) = (unsafe { slot::<_>(SCIPconshdlrGetData(conshdlr)) }) else {
        return SCIP_Retcode_SCIP_OKAY;
    };
    let n = nlockspos.saturating_add(nlocksneg);
    // SCIP locks once a problem is transformed, with a run installed, and unlocks as it frees it, maybe with none.
    let applied = if n > 0 {
        // SAFETY: SCIP calls the lock during the solve the run is installed for, or with none installed.
        if !unsafe { slot.lazy() } {
            return SCIP_Retcode_SCIP_OKAY;
        }
        // SAFETY: SCIP locks a transformed problem it holds, in its transforming stage.
        unsafe { lock_all(scip, &slot.lazy_locks, n) }
    } else {
        // SAFETY: SCIP unlocks a transformed problem it holds before it frees its variables.
        unsafe { slot.lazy_locks.unlock(scip) }
    };
    match applied {
        Ok(()) => SCIP_Retcode_SCIP_OKAY,
        // SAFETY: as above.
        Err(error) => unsafe { slot.fail(error) },
    }
}

/// `CONSHDLRCOPY`: a copy is valid unless the solve has a lazy-constraint handler, which a copy would not see.
unsafe extern "C" fn copy(_target: *mut SCIP, conshdlr: *mut SCIP_CONSHDLR, valid: *mut c_uint) -> SCIP_RETCODE {
    // SAFETY: the handler's data is its slot.
    let lazy = unsafe { slot::<_>(SCIPconshdlrGetData(conshdlr)) }
        // SAFETY: SCIP copies the handler during the solve the run is installed for.
        .is_some_and(|slot| unsafe { slot.lazy() });
    // SAFETY: `valid` is SCIP's to write for the call.
    unsafe { valid.write(c_uint::from(!lazy)) };
    SCIP_Retcode_SCIP_OKAY
}
