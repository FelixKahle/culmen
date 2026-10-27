// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! When an instance may be destroyed. `Highs_destroy` also shuts down the HiGHS task scheduler of the destroying
//! thread, so an instance dropped while that thread is inside a run, or inside a callback on a HiGHS worker thread,
//! would free the scheduler the run is still using. Such an instance is kept until the last run that is in progress on
//! the dropping thread when it drops has returned, and destroyed then; an instance dropped anywhere else is destroyed
//! at once.

use core::cell::Cell;
use core::ffi::c_void;
use core::ptr::NonNull;
use std::sync::{Mutex, PoisonError};

use super::Instance;

thread_local! {
    /// How many runs and callbacks this thread is inside.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// The instances dropped inside a run or a callback, waiting for a run to end with its thread outside every other.
static ORPHANS: Mutex<Vec<Instance>> = Mutex::new(Vec::new());

/// A run or a callback in progress on this thread, from creation to drop.
#[derive(Debug)]
pub(crate) struct Busy {
    /// Whether it is a run, whose end may destroy the orphans: a callback may be on a HiGHS worker thread, whose
    /// scheduler must not be shut down.
    run: bool,
}

impl Busy {
    /// A run in progress on this thread.
    pub(super) fn run() -> Self {
        Self::enter(true)
    }

    /// A callback in progress on this thread, which may be a HiGHS worker thread.
    pub(crate) fn callback() -> Self {
        Self::enter(false)
    }

    fn enter(run: bool) -> Self {
        DEPTH.set(DEPTH.get() + 1);
        Self { run }
    }

    /// Whether this thread is inside a run or a callback.
    fn active() -> bool {
        DEPTH.get() > 0
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        let depth = DEPTH.get() - 1;
        DEPTH.set(depth);
        if self.run && depth == 0 {
            // The lock is released before the orphans drop, so their own drops find it free.
            let orphans = core::mem::take(&mut *ORPHANS.lock().unwrap_or_else(PoisonError::into_inner));
            drop(orphans);
        }
    }
}

/// Whether the instance at `raw`, which its owner is dropping, is kept for the end of the last run in progress on this
/// thread; when it is not, the owner destroys it now.
pub(super) fn defer(raw: NonNull<c_void>) -> bool {
    let busy = Busy::active();
    if busy {
        // The dropping owner never uses `raw` again, so the orphan is its one owner from here on.
        ORPHANS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Instance(raw));
    }
    busy
}
