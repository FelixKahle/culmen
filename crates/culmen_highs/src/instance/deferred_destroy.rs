// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! When a HiGHS instance is destroyed. `Highs_destroy` also shuts down the HiGHS task scheduler of the destroying
//! thread, so an instance released while that thread is inside a run, or inside a callback on a HiGHS worker thread,
//! would free the scheduler the run is still using. Such an instance waits until a run ends with its thread outside
//! every other run and callback, and is destroyed then, on that thread; an instance released anywhere else is
//! destroyed at once.

use core::cell::Cell;
use core::ffi::c_void;
use core::ptr::NonNull;
use std::sync::{Mutex, PoisonError};

use highs_sys::{Highs_create, Highs_destroy};

thread_local! {
    /// How many runs and callbacks this thread is inside.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// The instances released inside a run or a callback, waiting for a run to end.
static WAITING: Mutex<Vec<Handle>> = Mutex::new(Vec::new());

/// A live HiGHS instance, destroyed by `Highs_destroy` when it drops. Only [`release`] and the end of a run drop
/// one, so it never drops while its thread is inside a run or a callback.
pub(super) struct Handle(NonNull<c_void>);

// SAFETY: a HiGHS instance has no affinity to the thread that created it: it holds no thread-local state, and the
// task scheduler HiGHS keeps per thread is looked up afresh on the calling thread by every `Highs_run`. Moving the
// owner to another thread and calling it there is what HiGHS supports; `Handle` is not `Sync`, so it is never
// called from two threads at once.
unsafe impl Send for Handle {}

impl Handle {
    /// A new instance with HiGHS's default options.
    ///
    /// # Panics
    ///
    /// When HiGHS cannot allocate an instance.
    pub(super) fn create() -> Self {
        // SAFETY: `Highs_create` takes no arguments and returns a new instance or null.
        let raw = unsafe { Highs_create() };
        Self(NonNull::new(raw).expect("HiGHS could not allocate an instance"))
    }

    pub(super) fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        debug_assert!(DEPTH.get() == 0, "Handle: destroyed inside a run or a callback");
        // SAFETY: the instance is live and is never used again. `Highs_destroy` also shuts down the HiGHS task
        // scheduler of this thread, which the next run on it starts afresh; a handle drops only outside every run
        // and callback on this thread, so no run is using that scheduler.
        unsafe { Highs_destroy(self.as_ptr()) };
    }
}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Handle").field(&self.0).finish()
    }
}

/// Destroys the instance now when this thread is outside every run and callback; otherwise it waits for the end of a
/// run.
pub(super) fn release(handle: Handle) {
    if DEPTH.get() == 0 {
        drop(handle);
    } else {
        WAITING.lock().unwrap_or_else(PoisonError::into_inner).push(handle);
    }
}

/// A run in progress on this thread, from creation to drop. Its end, when this thread is then outside every other run
/// and callback, destroys the waiting instances.
// The private field keeps a guard from being built without entering.
#[derive(Debug)]
pub(crate) struct RunGuard(());

impl RunGuard {
    pub(super) fn enter() -> Self {
        enter();
        Self(())
    }
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        if leave() == 0 {
            // The lock is released before the instances drop.
            let waiting = core::mem::take(&mut *WAITING.lock().unwrap_or_else(PoisonError::into_inner));
            drop(waiting);
        }
    }
}

/// A callback in progress on this thread, from creation to drop. Its end never destroys an instance: this may be a
/// HiGHS worker thread, whose scheduler the run is using.
#[derive(Debug)]
pub(crate) struct CallbackGuard(());

impl CallbackGuard {
    pub(crate) fn enter() -> Self {
        enter();
        Self(())
    }
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        leave();
    }
}

fn enter() {
    DEPTH.set(DEPTH.get() + 1);
}

/// Leaves a run or a callback, returning how many this thread is still inside.
///
/// # Panics
///
/// When this thread is inside none.
fn leave() -> usize {
    let depth = DEPTH
        .get()
        .checked_sub(1)
        .expect("left a run or a callback never entered");
    DEPTH.set(depth);
    depth
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    /// Serialises the tests: a run ending in one would destroy the instances another is watching wait.
    static SERIAL: Mutex<()> = Mutex::new(());

    /// Releases a new instance and returns its address, to look for among the waiting instances.
    fn release_new() -> usize {
        let handle = Handle::create();
        let addr = handle.as_ptr().addr();
        release(handle);
        addr
    }

    fn waiting(addr: usize) -> bool {
        WAITING
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .any(|handle| handle.as_ptr().addr() == addr)
    }

    #[test]
    fn test_an_instance_released_outside_every_run_is_destroyed_at_once() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let addr = release_new();
        assert!(!waiting(addr));
    }

    #[test]
    fn test_an_instance_released_in_a_run_waits_for_the_outermost_run_to_end() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let outer = RunGuard::enter();
        let callback = CallbackGuard::enter();
        let inner = RunGuard::enter();
        let addr = release_new();
        assert!(waiting(addr));
        drop(inner);
        assert!(waiting(addr), "a nested run ending inside another destroys nothing");
        drop(callback);
        assert!(waiting(addr), "a callback ending destroys nothing");
        drop(outer);
        assert!(!waiting(addr), "the outermost run ending destroys the instance");
    }

    #[test]
    fn test_an_instance_released_on_a_worker_thread_waits_for_the_run_to_end() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let run = RunGuard::enter();
        let addr = thread::spawn(|| {
            let _callback = CallbackGuard::enter();
            release_new()
        })
        .join()
        .unwrap();
        assert!(waiting(addr), "the worker's callback ending destroys nothing");
        drop(run);
        assert!(!waiting(addr));
    }
}
