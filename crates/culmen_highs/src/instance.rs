// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One HiGHS instance behind safe methods: the only place, with the callback trampoline, that calls HiGHS.

use core::ffi::{CStr, c_char, c_int, c_void};
use core::ptr::{self, NonNull};
use std::ffi::CString;

use culmen_core::{Integrality, Lp, Sense};
use highs_sys::{
    Highs_create, Highs_destroy, Highs_getDoubleInfoValue, Highs_getInt64InfoValue, Highs_getIntInfoValue,
    Highs_getModelStatus, Highs_getObjectiveValue, Highs_getSolution, Highs_passModel, Highs_resetOptions, Highs_run,
    Highs_setBoolOptionValue, Highs_setCallback, Highs_setDoubleOptionValue, Highs_setIntOptionValue,
    Highs_setStringOptionValue, Highs_startCallback, Highs_stopCallback, Highs_zeroAllClocks, HighsCCallbackType,
    HighsInt, kHighsMatrixFormatRowwise, kHighsObjSenseMaximize, kHighsObjSenseMinimize, kHighsStatusError,
    kHighsVarTypeContinuous, kHighsVarTypeInteger,
};

use crate::{CallStatus, Dimension, HighsError, ModelStatus, OptionValue};

/// A HiGHS instance, created by `Highs_create` and destroyed by `Highs_destroy` on drop.
///
/// The pointer is valid from creation to drop, and every method borrows the instance as HiGHS needs it: `&mut`
/// for anything that changes it, so no two calls ever overlap on one instance.
pub(crate) struct Instance(NonNull<c_void>);

// SAFETY: a HiGHS instance has no affinity to the thread that created it: it holds no thread-local state, and the
// task scheduler HiGHS keeps per thread is looked up afresh on the calling thread by every `Highs_run`. Moving the
// owner to another thread and calling it there is what HiGHS supports; `Instance` is not `Sync`, so it is never
// called from two threads at once.
unsafe impl Send for Instance {}

impl Instance {
    /// A new instance with HiGHS's default options.
    ///
    /// # Panics
    ///
    /// When HiGHS cannot allocate an instance.
    pub(crate) fn new() -> Self {
        // SAFETY: `Highs_create` takes no arguments and returns a new instance or null.
        let raw = unsafe { Highs_create() };
        let mut instance = Self(NonNull::new(raw).expect("HiGHS could not allocate an instance"));
        // HiGHS prints its banner to standard output at the first model passed while its log is on, which it is by
        // default; the log stays off until a run routes it to the callback.
        let _ = instance.set_option("output_flag", &OptionValue::Bool(false));
        instance
    }

    fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    /// Passes `lp` to HiGHS, replacing any model before.
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] when a count exceeds HiGHS's 32-bit indices, [`HighsError::Load`] when HiGHS
    /// refuses the LP.
    pub(crate) fn pass(&mut self, lp: &Lp) -> Result<(), HighsError> {
        let (num_col, num_row) = (
            index(lp.num_cols(), Dimension::Columns)?,
            index(lp.num_rows(), Dimension::Rows)?,
        );
        let num_nz = index(lp.num_entries(), Dimension::Entries)?;
        let a = Arrays::of(lp);
        let sense = match lp.sense() {
            Sense::Minimize => kHighsObjSenseMinimize,
            Sense::Maximize => kHighsObjSenseMaximize,
        };
        let integrality = if lp.is_mip() {
            a.integrality.as_ptr()
        } else {
            ptr::null()
        };
        let (no_int, no_double) = (ptr::null(), ptr::null());
        // SAFETY: the instance is live and borrowed exclusively. The column arrays hold `num_col` values, the row
        // bounds `num_row`, `starts` holds `num_row + 1` row starts (HiGHS reads the first `num_row`), `cols` and
        // the values hold `num_nz` entries, and `integrality` is null or holds `num_col` values. There is no
        // Hessian: its count is zero and its arrays null. HiGHS copies every array before returning.
        let status = unsafe {
            Highs_passModel(
                self.raw(),
                num_col,
                num_row,
                num_nz,
                0,
                kHighsMatrixFormatRowwise,
                0,
                sense,
                lp.offset(),
                a.costs.as_ptr(),
                a.lower.as_ptr(),
                a.upper.as_ptr(),
                a.row_lower.as_ptr(),
                a.row_upper.as_ptr(),
                a.starts.as_ptr(),
                a.cols.as_ptr(),
                lp.matrix().values.as_ptr(),
                no_int,
                no_int,
                no_double,
                integrality,
            )
        };
        match CallStatus::from_raw(status) {
            CallStatus::Error => Err(HighsError::Load { message: None }),
            CallStatus::Ok | CallStatus::Warning => Ok(()),
        }
    }

    /// Resets every option to HiGHS's default.
    pub(crate) fn reset_options(&mut self) {
        // SAFETY: the instance is live and borrowed exclusively.
        let _ = unsafe { Highs_resetOptions(self.raw()) };
    }

    /// Sets an option by name.
    ///
    /// # Errors
    ///
    /// [`HighsError::OptionName`] when the name holds a NUL byte, [`HighsError::OptionString`] when a string value
    /// does, [`HighsError::Option`] when HiGHS refuses the option: an unknown name, a value of the wrong kind or out
    /// of range.
    pub(crate) fn set_option(&mut self, name: &str, value: &OptionValue) -> Result<(), HighsError> {
        let c_name = CString::new(name).map_err(|_| HighsError::OptionName { name: name.to_owned() })?;
        let raw = self.raw();
        let name_ptr = c_name.as_ptr();
        let status = match value {
            // SAFETY: the instance is live and borrowed exclusively, the name a NUL-terminated string HiGHS reads
            // during the call.
            OptionValue::Bool(v) => unsafe { Highs_setBoolOptionValue(raw, name_ptr, HighsInt::from(*v)) },
            // SAFETY: as above.
            OptionValue::Int(v) => unsafe { Highs_setIntOptionValue(raw, name_ptr, *v) },
            // SAFETY: as above.
            OptionValue::Double(v) => unsafe { Highs_setDoubleOptionValue(raw, name_ptr, *v) },
            OptionValue::String(v) => {
                let c_value =
                    CString::new(v.as_str()).map_err(|_| HighsError::OptionString { name: name.to_owned() })?;
                // SAFETY: as above, and the value is a NUL-terminated string HiGHS copies during the call.
                unsafe { Highs_setStringOptionValue(raw, name_ptr, c_value.as_ptr()) }
            }
        };
        if status == kHighsStatusError {
            return Err(HighsError::Option {
                name: name.to_owned(),
                value: value.clone(),
            });
        }
        Ok(())
    }

    /// Restarts HiGHS's run clock, which `time_limit` is measured on and which otherwise counts across runs.
    pub(crate) fn zero_clocks(&mut self) {
        // SAFETY: the instance is live and borrowed exclusively.
        let _ = unsafe { Highs_zeroAllClocks(self.raw()) };
    }

    /// Installs `callback` with `user` as its data and starts the callback kinds in `kinds`; or, with a null
    /// `user`, stops every kind and leaves the callback with nothing to reach.
    ///
    /// # Safety
    ///
    /// `user` is null or points to data `callback` may use until this is called again with a null `user`, and
    /// `callback` treats its arguments as HiGHS passes them.
    pub(crate) unsafe fn hook(&mut self, callback: HighsCCallbackType, user: *mut c_void, kinds: &[c_int]) {
        // SAFETY: the instance is live and borrowed exclusively; the caller vouches for `user` and `callback`.
        let _ = unsafe { Highs_setCallback(self.raw(), callback, user) };
        for kind in 0..=highs_sys::kHighsCallbackCallbackMipUserSolution {
            // SAFETY: the instance is live and borrowed exclusively, and `kind` is a callback kind of HiGHS 1.15.
            let _ = unsafe {
                if !user.is_null() && kinds.contains(&kind) {
                    Highs_startCallback(self.raw(), kind)
                } else {
                    Highs_stopCallback(self.raw(), kind)
                }
            };
        }
    }

    /// Runs HiGHS on the loaded model.
    pub(crate) fn run(&mut self) -> CallStatus {
        // SAFETY: the instance is live and borrowed exclusively. Any callback it calls was installed by `hook`,
        // whose caller keeps the callback's data alive for the run.
        CallStatus::from_raw(unsafe { Highs_run(self.raw()) })
    }

    /// The model status of the last run.
    pub(crate) fn model_status(&self) -> ModelStatus {
        // SAFETY: the instance is live.
        ModelStatus::from_raw(unsafe { Highs_getModelStatus(self.raw()) })
    }

    /// The objective value of the last run's solution, offset included.
    pub(crate) fn objective(&self) -> f64 {
        // SAFETY: the instance is live.
        unsafe { Highs_getObjectiveValue(self.raw()) }
    }

    /// An integer info value, `None` when HiGHS does not know the name or has no valid info.
    pub(crate) fn info_int(&self, name: &CStr) -> Option<i32> {
        let mut value: HighsInt = 0;
        // SAFETY: the instance is live, the name NUL-terminated, and `value` one integer to write.
        let status = unsafe { Highs_getIntInfoValue(self.raw(), name.as_ptr(), &raw mut value) };
        (status != kHighsStatusError).then_some(value)
    }

    /// A 64-bit integer info value, `None` when HiGHS does not know the name or has no valid info.
    pub(crate) fn info_i64(&self, name: &CStr) -> Option<i64> {
        let mut value: i64 = 0;
        // SAFETY: the instance is live, the name NUL-terminated, and `value` one 64-bit integer to write.
        let status = unsafe { Highs_getInt64InfoValue(self.raw(), name.as_ptr(), &raw mut value) };
        (status != kHighsStatusError).then_some(value)
    }

    /// A floating-point info value, `None` when HiGHS does not know the name or has no valid info.
    pub(crate) fn info_double(&self, name: &CStr) -> Option<f64> {
        let mut value = 0.0;
        // SAFETY: the instance is live, the name NUL-terminated, and `value` one double to write.
        let status = unsafe { Highs_getDoubleInfoValue(self.raw(), name.as_ptr(), &raw mut value) };
        (status != kHighsStatusError).then_some(value)
    }

    /// The column values of the last run's primal solution.
    ///
    /// Call it only when the last run reports a feasible primal solution of the loaded model with `num_cols`
    /// columns.
    pub(crate) fn primal_values(&self, num_cols: usize) -> Vec<f64> {
        let mut values = vec![0.0; num_cols];
        // SAFETY: the instance is live. A run that reports a primal solution holds one value per column of its
        // model, `num_cols` values, which is what `values` has room for; the dual and row arrays are null, which
        // HiGHS skips.
        let _ = unsafe {
            Highs_getSolution(
                self.raw(),
                values.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        values
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: the instance is live and is never used again. `Highs_destroy` also shuts down the HiGHS task
        // scheduler of the dropping thread, which the next run on that thread starts afresh; no run is in progress
        // on this thread while it drops.
        unsafe { Highs_destroy(self.raw()) };
    }
}

impl core::fmt::Debug for Instance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Instance").field(&self.0).finish()
    }
}

/// An LP's arrays in the layout `Highs_passModel` reads.
struct Arrays {
    costs: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    row_lower: Vec<f64>,
    row_upper: Vec<f64>,
    starts: Vec<HighsInt>,
    cols: Vec<HighsInt>,
    integrality: Vec<HighsInt>,
}

impl Arrays {
    /// The arrays of `lp`, whose counts the caller checked against HiGHS's index type.
    fn of(lp: &Lp) -> Self {
        let columns = lp.columns();
        let matrix = lp.matrix();
        // Every start is at most the entry count and every column below the column count, both checked by the
        // caller, so the conversions are exact.
        Self {
            costs: columns.iter().map(|c| c.cost).collect(),
            lower: columns.iter().map(|c| c.bounds.lower()).collect(),
            upper: columns.iter().map(|c| c.bounds.upper()).collect(),
            row_lower: lp.row_bounds().iter().map(|b| b.lower()).collect(),
            row_upper: lp.row_bounds().iter().map(|b| b.upper()).collect(),
            starts: matrix.starts.iter().map(|&s| s as HighsInt).collect(),
            cols: matrix.cols.iter().map(|c| c.index() as HighsInt).collect(),
            integrality: columns
                .iter()
                .map(|c| match c.integrality {
                    Integrality::Continuous => kHighsVarTypeContinuous,
                    Integrality::Integer => kHighsVarTypeInteger,
                })
                .collect(),
        }
    }
}

/// A count as HiGHS's index type.
///
/// # Errors
///
/// [`HighsError::TooLarge`] when the count exceeds it.
fn index(count: usize, what: Dimension) -> Result<HighsInt, HighsError> {
    HighsInt::try_from(count).map_err(|_| HighsError::TooLarge { what, count })
}

/// HiGHS's C string as text, lossily for bytes that are not UTF-8.
///
/// # Safety
///
/// `text` is null or a NUL-terminated string that stays valid for `'a`.
pub(crate) unsafe fn text<'a>(text: *const c_char) -> std::borrow::Cow<'a, str> {
    if text.is_null() {
        return std::borrow::Cow::Borrowed("");
    }
    // SAFETY: the caller vouches that `text` is a NUL-terminated string valid for `'a`.
    unsafe { CStr::from_ptr(text) }.to_string_lossy()
}
