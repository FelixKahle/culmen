// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The variable locks the lazy-constraint handler, a handler without constraints, applies to SCIP's transformed
//! problem, kept so they are undone exactly.
//!
//! SCIP calls such a handler's lock twice per transformed problem: to lock when the problem is created, with a run
//! installed, and to unlock when it is freed, maybe with none (`S/cons.c:4276-4313`, `S/prob.c:461-470`, `625-633`
//! of SCIP 10.0.2). A variable a pricer creates while solving exists at neither call, so it is locked when it is
//! created; and a variable SCIP frees must hold no lock (`S/var.c:3669-3670`). So every lock the handler applies goes
//! into its [`Applied`] in the slot, which outlives the run, and the unlock undoes exactly what it holds, the priced
//! variables' locks included.

use core::cell::RefCell;
use core::ffi::c_int;
use core::ptr::NonNull;

use scip_sys::{SCIP, SCIP_LockType_SCIP_LOCKTYPE_MODEL, SCIP_VAR, SCIPaddVarLocksType};

use crate::ScipError;
use crate::error::check;

/// The model locks the handler applied to the variables of SCIP's transformed problem, in order: each variable with the
/// number of locks down and up.
#[derive(Debug)]
pub(super) struct Applied(RefCell<Vec<(NonNull<SCIP_VAR>, c_int, c_int)>>);

impl Applied {
    /// No lock applied.
    pub(super) const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    /// Adds `down` locks down and `up` locks up to `var` and keeps them to undo.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses the call; nothing is kept then.
    ///
    /// # Safety
    ///
    /// `scip` is a live instance from its transforming stage until it frees its transformed problem, not in its
    /// solved stage, and `var` a variable of its transformed problem.
    pub(super) unsafe fn lock(
        &self,
        scip: *mut SCIP,
        var: NonNull<SCIP_VAR>,
        down: c_int,
        up: c_int,
    ) -> Result<(), ScipError> {
        // SAFETY: the caller vouches for the instance, its stage and `var`.
        check("SCIPaddVarLocksType", unsafe {
            SCIPaddVarLocksType(scip, var.as_ptr(), SCIP_LockType_SCIP_LOCKTYPE_MODEL, down, up)
        })?;
        self.0.borrow_mut().push((var, down, up));
        Ok(())
    }

    /// Removes every lock kept, newest first, and forgets them.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] for the first call SCIP refuses; the other locks are removed still.
    ///
    /// # Safety
    ///
    /// `scip` is the live instance whose transformed problem holds every variable kept, from its transforming stage
    /// until it frees that problem, before it frees the variables.
    pub(super) unsafe fn unlock(&self, scip: *mut SCIP) -> Result<(), ScipError> {
        let applied = core::mem::take(&mut *self.0.borrow_mut());
        let mut unlocked = Ok(());
        for (var, down, up) in applied.into_iter().rev() {
            // SAFETY: the caller vouches for the instance, its stage and that `var` is still a variable of it.
            let code =
                unsafe { SCIPaddVarLocksType(scip, var.as_ptr(), SCIP_LockType_SCIP_LOCKTYPE_MODEL, -down, -up) };
            unlocked = unlocked.and(check("SCIPaddVarLocksType", code));
        }
        unlocked
    }

    /// The locks kept on `var`, down and up, summed.
    #[cfg(test)]
    pub(super) fn on(&self, var: NonNull<SCIP_VAR>) -> (c_int, c_int) {
        self.0
            .borrow()
            .iter()
            .filter(|&&(v, _, _)| v == var)
            .fold((0, 0), |(d, u), &(_, down, up)| (d + down, u + up))
    }
}
