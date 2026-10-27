// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One HiGHS instance behind safe methods: the only place, with the callback trampoline, that calls HiGHS. The calls
//! behind the analyses are in its `analysis` submodule, those behind an edit of the model in `edit`; when a dropped
//! instance is destroyed, in `deferred_destroy`.

mod analysis;
mod deferred_destroy;
mod edit;

pub(crate) use deferred_destroy::CallbackGuard;

use core::ffi::{CStr, c_char, c_int, c_void};
use core::mem::ManuallyDrop;
use core::ptr;
use std::ffi::CString;

use culmen_core::{
    AbsoluteGap, Col, Integrality, LinearObjective, Objectives, Program, RelativeGap, Sense, Tolerance, Weight,
};
use highs_sys::{
    Highs_clearLinearObjectives, Highs_getColName, Highs_getDoubleInfoValue, Highs_getHessianNumNz,
    Highs_getInt64InfoValue, Highs_getIntInfoValue, Highs_getModel, Highs_getModelStatus, Highs_getNumCol,
    Highs_getNumNz, Highs_getObjectiveValue, Highs_getRowName, Highs_getSolution, Highs_passColName,
    Highs_passLinearObjectives, Highs_passModel, Highs_passRowName, Highs_readModel, Highs_resetOptions, Highs_run,
    Highs_setBoolOptionValue, Highs_setCallback, Highs_setDoubleOptionValue, Highs_setIntOptionValue,
    Highs_setSolution, Highs_setStringOptionValue, Highs_startCallback, Highs_stopCallback, Highs_writeModel,
    Highs_zeroAllClocks, HighsCCallbackType, HighsInt, kHighsHessianFormatTriangular, kHighsMatrixFormatRowwise,
    kHighsObjSenseMaximize, kHighsObjSenseMinimize, kHighsStatusError, kHighsVarTypeContinuous, kHighsVarTypeInteger,
    kHighsVarTypeSemiContinuous, kHighsVarTypeSemiInteger,
};

pub(crate) use analysis::{Penalties, RangingRecord, RawIis};

use crate::{CallStatus, Dimension, HighsError, ModelStatus, OptionValue};
use deferred_destroy::{Handle, RunGuard};

/// A HiGHS instance, live from creation to drop and destroyed once no run on the dropping thread needs HiGHS's task
/// scheduler there.
///
/// Every method borrows the instance as HiGHS needs it: `&mut` for anything that changes it, so no two calls ever
/// overlap on one instance.
#[derive(Debug)]
pub(crate) struct Instance(ManuallyDrop<Handle>);

impl Instance {
    /// A new instance with HiGHS's default options.
    ///
    /// # Panics
    ///
    /// When HiGHS cannot allocate an instance.
    pub(crate) fn new() -> Self {
        let mut instance = Self(ManuallyDrop::new(Handle::create()));
        // HiGHS prints its banner to standard output at the first model passed while its log is on, which it is by
        // default; the log stays off until a run routes it to the callback.
        let _ = instance.set_option("output_flag", &OptionValue::Bool(false));
        instance
    }

    fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    /// The number of columns of the model HiGHS holds.
    fn num_col(&self) -> usize {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live.
        let count = unsafe { Highs_getNumCol(self.raw()) };
        usize::try_from(count).unwrap_or(0)
    }

    /// Passes `program` to HiGHS, replacing any model before; `padding` says whether a QP without rows gets the free
    /// row HiGHS needs to solve it (see [`Arrays`]).
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] when a count exceeds HiGHS's 32-bit indices, [`HighsError::Load`] when HiGHS
    /// refuses the program.
    pub(crate) fn pass(&mut self, program: &Program, padding: Padding) -> Result<(), HighsError> {
        let arrays = Arrays::of(program, padding);
        let counts = Counts::of(program, &arrays)?;
        let hessian = Hessian::of(program);
        debug_assert!(
            arrays.agrees_with(program, padding),
            "Instance: the arrays passed to HiGHS do not match the program's counts"
        );
        debug_assert!(
            hessian.agrees_with(program) && hessian.values.len() == program.quadratic().len(),
            "Instance: the Hessian passed to HiGHS does not match the program's quadratic part"
        );
        match CallStatus::from_raw(self.pass_model(program, &arrays, &hessian, &counts)) {
            CallStatus::Error => Err(HighsError::Load { message: None }),
            CallStatus::Ok | CallStatus::Warning => Ok(()),
        }
    }

    /// Hands HiGHS the model the arrays describe and returns `Highs_passModel`'s status; `arrays`, `hessian` and
    /// `counts` are those of `program`.
    fn pass_model(&mut self, program: &Program, a: &Arrays, q: &Hessian, counts: &Counts) -> HighsInt {
        let integrality = if program.is_mip() {
            a.integrality.as_ptr()
        } else {
            ptr::null()
        };
        let (q_format, q_start, q_index, q_value) = q.raw();
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively. The column arrays hold `counts.cols` values, the row
        // bounds `counts.rows`, `starts` holds `counts.rows + 1` row starts (HiGHS reads the first `counts.rows`),
        // `cols` and the values hold `counts.entries` entries, and `integrality` is null or holds `counts.cols`
        // values. Without a quadratic part the Hessian's count is zero and its arrays null; with one, `q_start` holds
        // `counts.cols + 1` column starts (HiGHS reads the first `counts.cols`) and `q_index` and `q_value` hold
        // `counts.hessian` entries of the lower triangle, column-wise. HiGHS copies every array before returning.
        unsafe {
            Highs_passModel(
                self.raw(),
                counts.cols,
                counts.rows,
                counts.entries,
                counts.hessian,
                kHighsMatrixFormatRowwise,
                q_format,
                objective_sense(program.sense()),
                program.offset(),
                a.costs.as_ptr(),
                a.lower.as_ptr(),
                a.upper.as_ptr(),
                a.row_lower.as_ptr(),
                a.row_upper.as_ptr(),
                a.starts.as_ptr(),
                a.cols.as_ptr(),
                program.matrix().values.as_ptr(),
                q_start,
                q_index,
                q_value,
                integrality,
            )
        }
    }

    /// Resets every option to HiGHS's default.
    pub(crate) fn reset_options(&mut self) {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
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
        debug_assert!(!raw.is_null(), "Instance: the HiGHS instance is null");
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
                debug_assert!(
                    !c_value.as_bytes().contains(&0),
                    "Instance: the option value holds a NUL byte"
                );
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
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
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
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            user.is_null() || callback.is_some(),
            "Instance: callback data is installed without a callback"
        );
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
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        let _run = RunGuard::enter();
        // SAFETY: the instance is live and borrowed exclusively. Any callback it calls was installed by `hook`,
        // whose caller keeps the callback's data alive for the run.
        CallStatus::from_raw(unsafe { Highs_run(self.raw()) })
    }

    /// The model status of the last run.
    pub(crate) fn model_status(&self) -> ModelStatus {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live.
        ModelStatus::from_raw(unsafe { Highs_getModelStatus(self.raw()) })
    }

    /// The objective value of the last run's solution, offset included.
    pub(crate) fn objective(&self) -> f64 {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live.
        unsafe { Highs_getObjectiveValue(self.raw()) }
    }

    /// An integer info value, `None` when HiGHS does not know the name or has no valid info.
    pub(crate) fn info_int(&self, name: &CStr) -> Option<i32> {
        let mut value: HighsInt = 0;
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live, the name NUL-terminated, and `value` one integer to write.
        let status = unsafe { Highs_getIntInfoValue(self.raw(), name.as_ptr(), &raw mut value) };
        (status != kHighsStatusError).then_some(value)
    }

    /// A 64-bit integer info value, `None` when HiGHS does not know the name or has no valid info.
    pub(crate) fn info_i64(&self, name: &CStr) -> Option<i64> {
        let mut value: i64 = 0;
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live, the name NUL-terminated, and `value` one 64-bit integer to write.
        let status = unsafe { Highs_getInt64InfoValue(self.raw(), name.as_ptr(), &raw mut value) };
        (status != kHighsStatusError).then_some(value)
    }

    /// A floating-point info value, `None` when HiGHS does not know the name or has no valid info.
    pub(crate) fn info_double(&self, name: &CStr) -> Option<f64> {
        let mut value = 0.0;
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
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
        debug_assert_eq!(
            self.num_col(),
            num_cols,
            "Instance: the buffer does not fit HiGHS's model"
        );
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

/// What a program hands HiGHS beyond its model: its further objectives and a start.
impl Instance {
    /// Replaces HiGHS's linear objectives with `program`'s: none for a program with one objective, else every
    /// objective of the program, its own first, as [`LinearObjectives`] lays them out.
    ///
    /// Call it after [`Instance::pass`] passed `program`, so HiGHS holds its columns.
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] when the objectives' coefficients exceed HiGHS's 32-bit indices,
    /// [`HighsError::Load`] when HiGHS refuses them.
    pub(crate) fn pass_objectives(&mut self, program: &Program) -> Result<(), HighsError> {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively.
        let _ = unsafe { Highs_clearLinearObjectives(self.raw()) };
        if !program.objectives().are_several() {
            return Ok(());
        }
        let o = LinearObjectives::of(program);
        let count = index(o.weights.len(), Dimension::Objectives)?;
        let _ = index(o.coefficients.len(), Dimension::Objectives)?;
        debug_assert!(
            o.agrees_with(self.num_col()),
            "Instance: the objectives passed to HiGHS do not match its model's columns"
        );
        // SAFETY: the instance is live and borrowed exclusively. The weights, offsets, tolerances and priorities hold
        // `count` values each, and the coefficients `count` times the column count of the model HiGHS holds, objective
        // by objective, which is what HiGHS reads; it copies them before returning.
        let status = unsafe {
            Highs_passLinearObjectives(
                self.raw(),
                count,
                o.weights.as_ptr(),
                o.offsets.as_ptr(),
                o.coefficients.as_ptr(),
                o.absolute.as_ptr(),
                o.relative.as_ptr(),
                o.priorities.as_ptr(),
            )
        };
        match CallStatus::from_raw(status) {
            CallStatus::Error => Err(HighsError::Load { message: None }),
            CallStatus::Ok | CallStatus::Warning => Ok(()),
        }
    }

    /// Hands HiGHS `values`, one per column of its model, as the solution the next run starts from; a value of
    /// [`UNDEFINED`] leaves the column to HiGHS.
    ///
    /// # Errors
    ///
    /// [`HighsError::Start`] when HiGHS refuses the solution.
    pub(crate) fn set_solution(&mut self, values: &[f64]) -> Result<(), HighsError> {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert_eq!(
            values.len(),
            self.num_col(),
            "Instance: the start does not fit HiGHS's model"
        );
        // SAFETY: the instance is live and borrowed exclusively. `values` holds one value per column of the model HiGHS
        // holds, which is what it reads; the row values and duals are null, which it skips. It copies the values.
        let status = unsafe { Highs_setSolution(self.raw(), values.as_ptr(), ptr::null(), ptr::null(), ptr::null()) };
        match CallStatus::from_raw(status) {
            CallStatus::Error => Err(HighsError::Start),
            CallStatus::Ok | CallStatus::Warning => Ok(()),
        }
    }
}

/// HiGHS's `kHighsUndefined`, `kHighsInf`: a start's value HiGHS chooses itself.
pub(crate) const UNDEFINED: f64 = f64::INFINITY;

/// A program's objectives in the layout `Highs_passLinearObjectives` reads, the program's own first.
///
/// HiGHS's lexicographic solve optimizes the objectives by descending priority, each minimized for a positive
/// weight and maximized otherwise, and keeps each optimum `f*` within `min(abs, rel · |f*|)` for the later ones,
/// leaving out a negative tolerance; its blend minimizes `Σ weight · objective`. So a lexicographic objective gets
/// the weight `±1` by its sense and its tolerances, `-1` for an absent one; a blended one its weight signed by its
/// sense and no tolerances. Every objective gets its own priority, the first the highest, which a blend ignores.
struct LinearObjectives {
    weights: Vec<f64>,
    offsets: Vec<f64>,
    coefficients: Vec<f64>,
    absolute: Vec<f64>,
    relative: Vec<f64>,
    priorities: Vec<HighsInt>,
}

/// One objective as HiGHS ranks it: its weight and its absolute and relative tolerances, `-1` for none.
type Rank = (f64, f64, f64);

impl LinearObjectives {
    /// The objectives of `program`, which has several, whose counts the caller checks against HiGHS's index type.
    fn of(program: &Program) -> Self {
        let own = LinearObjective {
            sense: program.sense(),
            terms: program
                .columns()
                .iter_enumerated()
                .map(|(col, c)| (col, c.cost))
                .collect(),
            offset: program.offset(),
        };
        let ranked: Vec<(&LinearObjective, Rank)> = match program.objectives() {
            Objectives::Single => Vec::new(),
            Objectives::Lexicographic { first, then } => core::iter::once((&own, lexicographic(own.sense, *first)))
                .chain(
                    then.iter()
                        .map(|s| (&s.objective, lexicographic(s.objective.sense, s.tolerance))),
                )
                .collect(),
            Objectives::Blended { first, with } => core::iter::once((&own, blended(own.sense, *first)))
                .chain(
                    with.iter()
                        .map(|w| (&w.objective, blended(w.objective.sense, w.weight))),
                )
                .collect(),
        };
        let count = ranked.len();
        Self {
            weights: ranked.iter().map(|(_, rank)| rank.0).collect(),
            offsets: ranked.iter().map(|(o, _)| o.offset).collect(),
            coefficients: ranked.iter().flat_map(|(o, _)| dense(o, program.num_cols())).collect(),
            absolute: ranked.iter().map(|(_, rank)| rank.1).collect(),
            relative: ranked.iter().map(|(_, rank)| rank.2).collect(),
            // The caller checked the count against HiGHS's index type, so every priority converts exactly.
            priorities: (0..count).rev().map(|p| p as HighsInt).collect(),
        }
    }

    /// Whether every array holds what `Highs_passLinearObjectives` reads for a model of `num_cols` columns.
    fn agrees_with(&self, num_cols: usize) -> bool {
        let count = self.weights.len();
        [&self.offsets, &self.absolute, &self.relative]
            .iter()
            .all(|a| a.len() == count)
            && self.priorities.len() == count
            && self.coefficients.len() == count * num_cols
    }
}

/// The rank of a lexicographic objective: the weight `±1` by its sense, and its tolerances.
fn lexicographic(sense: Sense, tolerance: Tolerance) -> Rank {
    (
        signed(sense, 1.0),
        tolerance.absolute.map_or(-1.0, AbsoluteGap::get),
        tolerance.relative.map_or(-1.0, RelativeGap::get),
    )
}

/// The rank of a blended objective: its weight signed by its sense, and no tolerances.
fn blended(sense: Sense, weight: Weight) -> Rank {
    (signed(sense, weight.get()), -1.0, -1.0)
}

/// `value` for minimizing, `-value` for maximizing.
const fn signed(sense: Sense, value: f64) -> f64 {
    match sense {
        Sense::Minimize => value,
        Sense::Maximize => -value,
    }
}

/// An objective's coefficients, one per column of a program of `num_cols` columns.
///
/// # Panics
///
/// When a term names a column at or beyond `num_cols`.
fn dense(objective: &LinearObjective, num_cols: usize) -> Vec<f64> {
    let mut values = vec![0.0; num_cols];
    objective.terms.iter().for_each(|&(col, c)| values[col.index()] += c);
    values
}

/// Model files: HiGHS's readers and writers, and the model and names HiGHS holds.
impl Instance {
    /// Reads the model file at `path` into HiGHS, replacing any model before; HiGHS picks the reader by the
    /// extension, `.mps` or `.lp`.
    pub(crate) fn read_model(&mut self, path: &CStr) -> CallStatus {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively, and the path a NUL-terminated string HiGHS reads
        // during the call.
        CallStatus::from_raw(unsafe { Highs_readModel(self.raw(), path.as_ptr()) })
    }

    /// Writes the model HiGHS holds to the file at `path`; HiGHS picks the writer by the extension, `.mps` or `.lp`.
    pub(crate) fn write_model(&mut self, path: &CStr) -> CallStatus {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively, and the path a NUL-terminated string HiGHS reads
        // during the call.
        CallStatus::from_raw(unsafe { Highs_writeModel(self.raw(), path.as_ptr()) })
    }

    /// Names a column or a row of the model HiGHS holds.
    ///
    /// # Errors
    ///
    /// [`HighsError::Load`] when HiGHS refuses the name.
    pub(crate) fn pass_name(&mut self, line: Line, index: HighsInt, name: &CStr) -> Result<(), HighsError> {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            usize::try_from(index).is_ok_and(|i| i < self.count(line)),
            "Instance: a name is passed for a line HiGHS's model does not have"
        );
        // SAFETY: the instance is live and borrowed exclusively, `index` a line of its model, and the name a
        // NUL-terminated string HiGHS copies during the call.
        let status = unsafe {
            match line {
                Line::Col => Highs_passColName(self.raw(), index, name.as_ptr()),
                Line::Row => Highs_passRowName(self.raw(), index, name.as_ptr()),
            }
        };
        match CallStatus::from_raw(status) {
            CallStatus::Error => Err(HighsError::Load { message: None }),
            CallStatus::Ok | CallStatus::Warning => Ok(()),
        }
    }

    /// The name of a column or a row of the model HiGHS holds, read through `buffer`; `None` when HiGHS has none.
    ///
    /// # Safety
    ///
    /// Every name of the model HiGHS holds is shorter than `buffer`: HiGHS copies the name with its terminating NUL
    /// into the buffer without a bound.
    pub(crate) unsafe fn name(&self, line: Line, index: HighsInt, buffer: &mut [u8]) -> Option<String> {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            usize::try_from(index).is_ok_and(|i| i < self.count(line)),
            "Instance: a name is read for a line HiGHS's model does not have"
        );
        debug_assert!(!buffer.is_empty(), "Instance: a name is read into no buffer");
        let out: *mut c_char = buffer.as_mut_ptr().cast();
        // SAFETY: the instance is live, `index` a line of its model, and the caller vouches that the name and its NUL
        // fit `buffer`, which HiGHS writes during the call.
        let status = unsafe {
            match line {
                Line::Col => Highs_getColName(self.raw(), index, out),
                Line::Row => Highs_getRowName(self.raw(), index, out),
            }
        };
        if status == kHighsStatusError {
            return None;
        }
        let name = CStr::from_bytes_until_nul(buffer).ok()?.to_string_lossy().into_owned();
        (!name.is_empty()).then_some(name)
    }

    /// The number of columns or rows of the model HiGHS holds.
    fn count(&self, line: Line) -> usize {
        match line {
            Line::Col => self.num_col(),
            Line::Row => self.num_row(),
        }
    }

    /// The model HiGHS holds: its matrix row-wise, its Hessian's lower triangle column-wise.
    pub(crate) fn model(&self) -> RawModel {
        let mut model = self.model_buffers();
        self.fill(&mut model);
        model
    }

    /// Buffers sized for the counts of the model HiGHS holds.
    fn model_buffers(&self) -> RawModel {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live.
        let (num_nz, q_num_nz) = unsafe { (Highs_getNumNz(self.raw()), Highs_getHessianNumNz(self.raw())) };
        RawModel::sized(
            self.num_col(),
            self.num_row(),
            usize::try_from(num_nz).unwrap_or(0),
            usize::try_from(q_num_nz).unwrap_or(0),
        )
    }

    /// Copies the model HiGHS holds into `m`, made by [`Instance::model_buffers`] for it, and trims every buffer to
    /// the counts HiGHS reports.
    fn fill(&self, m: &mut RawModel) {
        let (mut num_col, mut num_row, mut nz, mut q_nz): (HighsInt, HighsInt, HighsInt, HighsInt) = (0, 0, 0, 0);
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            m.fits(self.num_col(), self.num_row()),
            "Instance: the model buffers are too small"
        );
        // SAFETY: the instance is live. Every buffer holds at least what HiGHS writes for the counts it reported when
        // the buffers were made: one value per column, per row, per matrix entry and per Hessian entry, one start per
        // row and per column, and never less than one element, so no pointer is dangling. The counts are written to
        // locals.
        let _ = unsafe {
            Highs_getModel(
                self.raw(),
                kHighsMatrixFormatRowwise,
                kHighsHessianFormatTriangular,
                &raw mut num_col,
                &raw mut num_row,
                &raw mut nz,
                &raw mut q_nz,
                &raw mut m.sense,
                &raw mut m.offset,
                m.costs.as_mut_ptr(),
                m.lower.as_mut_ptr(),
                m.upper.as_mut_ptr(),
                m.row_lower.as_mut_ptr(),
                m.row_upper.as_mut_ptr(),
                m.starts.as_mut_ptr(),
                m.indices.as_mut_ptr(),
                m.values.as_mut_ptr(),
                m.q_starts.as_mut_ptr(),
                m.q_indices.as_mut_ptr(),
                m.q_values.as_mut_ptr(),
                m.integrality.as_mut_ptr(),
            )
        };
        m.trim(num_col, num_row, nz, q_nz);
    }
}

/// A column or a row, for the calls that take either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Line {
    /// A column.
    Col,
    /// A row.
    Row,
}

/// The model HiGHS holds as `Highs_getModel` writes it: the matrix row-wise, row `r`'s entries at
/// `starts[r]..starts[r + 1]` of `indices` and `values`; the Hessian's lower triangle column-wise alike in `q_starts`,
/// `q_indices` and `q_values`; HiGHS's variable types in `integrality`.
#[derive(Debug)]
pub(crate) struct RawModel {
    pub(crate) sense: HighsInt,
    pub(crate) offset: f64,
    pub(crate) costs: Vec<f64>,
    pub(crate) lower: Vec<f64>,
    pub(crate) upper: Vec<f64>,
    pub(crate) row_lower: Vec<f64>,
    pub(crate) row_upper: Vec<f64>,
    /// One start per row and the entry count after them.
    pub(crate) starts: Vec<HighsInt>,
    pub(crate) indices: Vec<HighsInt>,
    pub(crate) values: Vec<f64>,
    /// One start per column and the Hessian's entry count after them; empty without a Hessian.
    pub(crate) q_starts: Vec<HighsInt>,
    pub(crate) q_indices: Vec<HighsInt>,
    pub(crate) q_values: Vec<f64>,
    pub(crate) integrality: Vec<HighsInt>,
}

impl RawModel {
    /// Buffers for a model of these counts, each at least one element long, every variable type continuous.
    fn sized(cols: usize, rows: usize, nz: usize, q_nz: usize) -> Self {
        let at_least_one = |n: usize| n.max(1);
        Self {
            sense: kHighsObjSenseMinimize,
            offset: 0.0,
            costs: vec![0.0; at_least_one(cols)],
            lower: vec![0.0; at_least_one(cols)],
            upper: vec![0.0; at_least_one(cols)],
            row_lower: vec![0.0; at_least_one(rows)],
            row_upper: vec![0.0; at_least_one(rows)],
            starts: vec![0; at_least_one(rows + 1)],
            indices: vec![0; at_least_one(nz)],
            values: vec![0.0; at_least_one(nz)],
            q_starts: vec![0; at_least_one(cols + 1)],
            q_indices: vec![0; at_least_one(q_nz)],
            q_values: vec![0.0; at_least_one(q_nz)],
            integrality: vec![kHighsVarTypeContinuous; at_least_one(cols)],
        }
    }

    /// Whether the buffers hold a model of `cols` columns and `rows` rows.
    fn fits(&self, cols: usize, rows: usize) -> bool {
        [&self.costs, &self.lower, &self.upper].iter().all(|a| a.len() >= cols)
            && self.integrality.len() >= cols
            && self.q_starts.len() > cols
            && [&self.row_lower, &self.row_upper].iter().all(|a| a.len() >= rows)
            && self.starts.len() > rows
    }

    /// Cuts the buffers to the counts HiGHS wrote and closes the starts with the entry counts.
    fn trim(&mut self, num_col: HighsInt, num_row: HighsInt, num_nz: HighsInt, q_num_nz: HighsInt) {
        let [cols, rows, nz, q_nz] = [num_col, num_row, num_nz, q_num_nz].map(|n| usize::try_from(n).unwrap_or(0));
        [&mut self.costs, &mut self.lower, &mut self.upper]
            .into_iter()
            .for_each(|a| a.truncate(cols));
        self.integrality.truncate(cols);
        [&mut self.row_lower, &mut self.row_upper]
            .into_iter()
            .for_each(|a| a.truncate(rows));
        self.starts.truncate(rows);
        self.starts.push(num_nz);
        self.indices.truncate(nz);
        self.values.truncate(nz);
        self.q_indices.truncate(q_nz);
        self.q_values.truncate(q_nz);
        self.q_starts.truncate(if q_nz == 0 { 0 } else { cols });
        if q_nz > 0 {
            self.q_starts.push(q_num_nz);
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: the handle is taken once, here, and `self` is never used again.
        let handle = unsafe { ManuallyDrop::take(&mut self.0) };
        deferred_destroy::release(handle);
    }
}

/// Whether a QP without rows gets one free row without entries.
///
/// HiGHS 1.15 solves a QP without rows from the unconstrained minimizer clamped into the bounds, and can report that
/// point optimal after no iteration when it is not; with a row it finds its starting point with its LP solver and
/// solves correctly. A free row constrains nothing, and the backend reads no row values; a model written to a file
/// goes without it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Padding {
    /// A QP without rows gets the free row, to be solved.
    QpFreeRow,
    /// The program as it is, to be written.
    Exact,
}

/// A program's arrays in the layout `Highs_passModel` reads, with a free row for a QP without rows when padded.
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
    /// The arrays of `program`, with a free row for a QP without rows when `padding` asks for it.
    fn of(program: &Program, padding: Padding) -> Self {
        let columns = program.columns();
        let matrix = program.matrix();
        let free_row = needs_free_row(program, padding);
        // Every start is at most the entry count and every column below the column count, both checked by the
        // caller before it passes the arrays, so the conversions it passes are exact.
        Self {
            costs: columns.iter().map(|c| c.cost).collect(),
            lower: columns.iter().map(|c| c.bounds.lower()).collect(),
            upper: columns.iter().map(|c| c.bounds.upper()).collect(),
            row_lower: program
                .row_bounds()
                .iter()
                .map(|b| b.lower())
                .chain(free_row.then_some(f64::NEG_INFINITY))
                .collect(),
            row_upper: program
                .row_bounds()
                .iter()
                .map(|b| b.upper())
                .chain(free_row.then_some(f64::INFINITY))
                .collect(),
            starts: matrix
                .starts
                .iter()
                .map(|&s| s as HighsInt)
                .chain(free_row.then_some(0))
                .collect(),
            cols: matrix.cols.iter().map(|c| c.index() as HighsInt).collect(),
            integrality: columns.iter().map(|c| var_type(c.integrality)).collect(),
        }
    }
}

impl Arrays {
    /// Whether every array holds the length `Highs_passModel` reads for `program`: one value per column, one per row
    /// (the program's rows, and the free row a QP without rows gets), one start per row and one more, one column per
    /// entry.
    fn agrees_with(&self, program: &Program, padding: Padding) -> bool {
        let (cols, entries) = (program.num_cols(), program.num_entries());
        let rows = program.num_rows() + usize::from(needs_free_row(program, padding));
        [&self.costs, &self.lower, &self.upper].iter().all(|a| a.len() == cols)
            && self.integrality.len() == cols
            && [&self.row_lower, &self.row_upper].iter().all(|a| a.len() == rows)
            && self.starts.len() == rows + 1
            && self.cols.len() == entries
            && program.matrix().values.len() == entries
    }
}

/// The Hessian of a program's quadratic part in the layout `Highs_passModel` reads: its lower triangle in compressed
/// column form, `rows[starts[c]..starts[c + 1]]` with `values` alike for column `c`, the diagonal first, as
/// [`QuadObjective::hessian`](culmen_core::QuadObjective::hessian) gives it for HiGHS's `½ xᵀQx`.
struct Hessian {
    starts: Vec<HighsInt>,
    rows: Vec<HighsInt>,
    values: Vec<f64>,
}

impl Hessian {
    /// The format, starts, rows and values `Highs_passModel` takes: a format of zero and null arrays without
    /// entries.
    fn raw(&self) -> (HighsInt, *const HighsInt, *const HighsInt, *const f64) {
        if self.values.is_empty() {
            (0, ptr::null(), ptr::null(), ptr::null())
        } else {
            (
                kHighsHessianFormatTriangular,
                self.starts.as_ptr(),
                self.rows.as_ptr(),
                self.values.as_ptr(),
            )
        }
    }

    /// The Hessian of `program`, whose counts the caller checked against HiGHS's index type.
    fn of(program: &Program) -> Self {
        let entries: Vec<(Col, Col, f64)> = program.quadratic().hessian().collect();
        // Every start is at most the entry count and every row below the column count, both checked by the
        // caller before it passes the arrays, so the conversions it passes are exact.
        Self {
            starts: (0..=program.num_cols())
                .map(|col| entries.partition_point(|e| e.0.index() < col) as HighsInt)
                .collect(),
            rows: entries.iter().map(|e| e.1.index() as HighsInt).collect(),
            values: entries.iter().map(|e| e.2).collect(),
        }
    }

    /// Whether the arrays hold the length `Highs_passModel` reads for `program`'s Hessian: one start per column and
    /// one more, ending at the entry count, and one row and one value per entry.
    fn agrees_with(&self, program: &Program) -> bool {
        let entries = self.values.len();
        self.starts.len() == program.num_cols() + 1
            && self.rows.len() == entries
            && self
                .starts
                .last()
                .is_some_and(|&end| usize::try_from(end).is_ok_and(|end| end == entries))
    }
}

/// Whether `program` is a QP without rows, which HiGHS 1.15 solves only with one free, empty row added, and `padding`
/// asks for that row.
fn needs_free_row(program: &Program, padding: Padding) -> bool {
    padding == Padding::QpFreeRow && program.num_rows() == 0 && !program.quadratic().is_empty()
}

/// HiGHS's variable type of an integrality.
pub(crate) const fn var_type(integrality: Integrality) -> HighsInt {
    match integrality {
        Integrality::Continuous => kHighsVarTypeContinuous,
        Integrality::Integer => kHighsVarTypeInteger,
        Integrality::SemiContinuous => kHighsVarTypeSemiContinuous,
        Integrality::SemiInteger => kHighsVarTypeSemiInteger,
    }
}

/// The counts `Highs_passModel` reads, each within HiGHS's index type.
struct Counts {
    cols: HighsInt,
    rows: HighsInt,
    entries: HighsInt,
    hessian: HighsInt,
}

impl Counts {
    /// The counts of `program` passed as `arrays`, whose rows may include the free row of a QP without rows.
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] naming the first count beyond HiGHS's 32-bit indices.
    fn of(program: &Program, arrays: &Arrays) -> Result<Self, HighsError> {
        Ok(Self {
            cols: index(program.num_cols(), Dimension::Columns)?,
            rows: index(arrays.row_lower.len(), Dimension::Rows)?,
            entries: index(program.num_entries(), Dimension::Entries)?,
            hessian: index(program.quadratic().len(), Dimension::HessianEntries)?,
        })
    }
}

/// HiGHS's number for an objective sense.
const fn objective_sense(sense: Sense) -> HighsInt {
    match sense {
        Sense::Minimize => kHighsObjSenseMinimize,
        Sense::Maximize => kHighsObjSenseMaximize,
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
