// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HiGHS calls behind the analyses, in HiGHS's own numbering; `crate::analysis` gives them culmen's types.

use core::ptr;

use highs_sys::{
    Highs_feasibilityRelaxation, Highs_getBasis, Highs_getDualRay, Highs_getIis, Highs_getNumRow, Highs_getPrimalRay,
    Highs_getRanging, Highs_getSolution, Highs_setBasis, HighsInt,
};

use super::Instance;
use crate::CallStatus;

/// One direction of HiGHS's ranging of every column or row: HiGHS's `HighsRangingRecord`.
#[derive(Debug)]
pub(crate) struct RangingRecord {
    pub(crate) value: Vec<f64>,
    pub(crate) objective: Vec<f64>,
    /// HiGHS's index of the entering variable: a column below the column count, a row after it, `-1` for none.
    pub(crate) entering: Vec<HighsInt>,
    /// The leaving variable, numbered as `entering`.
    pub(crate) leaving: Vec<HighsInt>,
}

impl RangingRecord {
    /// A record of `len` zeros.
    fn zeros(len: usize) -> Self {
        Self {
            value: vec![0.0; len],
            objective: vec![0.0; len],
            entering: vec![0; len],
            leaving: vec![0; len],
        }
    }

    /// The four arrays as `Highs_getRanging` takes them, in its order.
    fn raw(&mut self) -> (*mut f64, *mut f64, *mut HighsInt, *mut HighsInt) {
        (
            self.value.as_mut_ptr(),
            self.objective.as_mut_ptr(),
            self.entering.as_mut_ptr(),
            self.leaving.as_mut_ptr(),
        )
    }

    /// Whether every array holds `len` entries.
    fn has_len(&self, len: usize) -> bool {
        [
            self.value.len(),
            self.objective.len(),
            self.entering.len(),
            self.leaving.len(),
        ]
        .iter()
        .all(|&l| l == len)
    }
}

/// HiGHS's ranging in its six directions, each over every column or every row of its model.
#[derive(Debug)]
pub(crate) struct RawRanging {
    pub(crate) cost_up: RangingRecord,
    pub(crate) cost_down: RangingRecord,
    pub(crate) col_bound_up: RangingRecord,
    pub(crate) col_bound_down: RangingRecord,
    pub(crate) row_bound_up: RangingRecord,
    pub(crate) row_bound_down: RangingRecord,
}

/// HiGHS's IIS: the columns and rows, each with HiGHS's `kHighsIisBound*` of it.
#[derive(Debug, Default)]
pub(crate) struct RawIis {
    pub(crate) cols: Vec<(HighsInt, HighsInt)>,
    pub(crate) rows: Vec<(HighsInt, HighsInt)>,
}

/// The penalties `Highs_feasibilityRelaxation` takes, one per column bound and per row of HiGHS's model; a negative
/// penalty keeps its bound or row hard.
#[derive(Debug)]
pub(crate) struct Penalties {
    pub(crate) lower: Vec<f64>,
    pub(crate) upper: Vec<f64>,
    pub(crate) rows: Vec<f64>,
}

impl Instance {
    /// The number of rows of the model HiGHS holds, the free row of a QP without rows included.
    pub(crate) fn num_row(&self) -> usize {
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live.
        let count = unsafe { Highs_getNumRow(self.raw()) };
        usize::try_from(count).unwrap_or(0)
    }

    /// The column duals and row duals of the last run, one per column and per row of HiGHS's model.
    ///
    /// Call it only when the last run reports a dual solution.
    pub(crate) fn duals(&self) -> (Vec<f64>, Vec<f64>) {
        let (mut cols, mut rows) = (vec![0.0; self.num_col()], vec![0.0; self.num_row()]);
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert_eq!(
            self.info_int(c"dual_solution_status"),
            Some(2),
            "Instance: duals are read without a dual solution"
        );
        // SAFETY: the instance is live. A run that reports a dual solution holds one column dual per column and one
        // row dual per row of its model, which is what the buffers have room for; the value arrays are null, which
        // HiGHS skips.
        let _ = unsafe {
            Highs_getSolution(
                self.raw(),
                ptr::null_mut(),
                cols.as_mut_ptr(),
                ptr::null_mut(),
                rows.as_mut_ptr(),
            )
        };
        (cols, rows)
    }

    /// The basis statuses of every column and row, as `kHighsBasisStatus*`; `None` when HiGHS holds no valid basis.
    pub(crate) fn basis(&self) -> Option<(Vec<HighsInt>, Vec<HighsInt>)> {
        if self.info_int(c"basis_validity") != Some(1) {
            return None;
        }
        let (mut cols, mut rows) = (vec![0; self.num_col()], vec![0; self.num_row()]);
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live, and its basis is valid, so it holds one status per column and one per row of
        // the model, which is what the buffers have room for.
        let _ = unsafe { Highs_getBasis(self.raw(), cols.as_mut_ptr(), rows.as_mut_ptr()) };
        Some((cols, rows))
    }

    /// Sets the basis the next run starts from, as `kHighsBasisStatus*` per column and per row.
    ///
    /// # Panics
    ///
    /// When the slices do not hold one status per column and per row of HiGHS's model.
    pub(crate) fn set_basis(&mut self, cols: &[HighsInt], rows: &[HighsInt]) -> CallStatus {
        assert!(
            cols.len() == self.num_col() && rows.len() == self.num_row(),
            "Instance: the basis does not fit HiGHS's model"
        );
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively, and HiGHS reads one status per column and per row
        // of its model, which the slices hold.
        CallStatus::from_raw(unsafe { Highs_setBasis(self.raw(), cols.as_ptr(), rows.as_ptr()) })
    }

    /// HiGHS's ranging of the last run; `None` when HiGHS refuses, as without an optimal simplex basis.
    pub(crate) fn ranging(&mut self) -> Option<RawRanging> {
        let (n, m) = (self.num_col(), self.num_row());
        let mut r = RawRanging {
            cost_up: RangingRecord::zeros(n),
            cost_down: RangingRecord::zeros(n),
            col_bound_up: RangingRecord::zeros(n),
            col_bound_down: RangingRecord::zeros(n),
            row_bound_up: RangingRecord::zeros(m),
            row_bound_down: RangingRecord::zeros(m),
        };
        let (cu, cd, bu, bd) = (
            r.cost_up.raw(),
            r.cost_down.raw(),
            r.col_bound_up.raw(),
            r.col_bound_down.raw(),
        );
        let (ru, rd) = (r.row_bound_up.raw(), r.row_bound_down.raw());
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            [&r.cost_up, &r.cost_down, &r.col_bound_up, &r.col_bound_down]
                .iter()
                .all(|x| x.has_len(n))
                && [&r.row_bound_up, &r.row_bound_down].iter().all(|x| x.has_len(m)),
            "Instance: the ranging buffers do not fit HiGHS's model"
        );
        // SAFETY: the instance is live and borrowed exclusively. On success HiGHS copies one entry per column into
        // each of the sixteen column arrays and one per row into each of the eight row arrays, which is what the
        // buffers hold; on failure it writes nothing.
        let status = unsafe {
            Highs_getRanging(
                self.raw(),
                cu.0,
                cu.1,
                cu.2,
                cu.3,
                cd.0,
                cd.1,
                cd.2,
                cd.3,
                bu.0,
                bu.1,
                bu.2,
                bu.3,
                bd.0,
                bd.1,
                bd.2,
                bd.3,
                ru.0,
                ru.1,
                ru.2,
                ru.3,
                rd.0,
                rd.1,
                rd.2,
                rd.3,
            )
        };
        (CallStatus::from_raw(status) != CallStatus::Error).then_some(r)
    }

    /// HiGHS's primal ray, one value per column; `Ok(None)` when it has none, `Err` when the call fails.
    ///
    /// HiGHS may run its simplex solver again to find one, which replaces the last run's result.
    pub(crate) fn primal_ray(&mut self) -> Result<Option<Vec<f64>>, CallStatus> {
        let mut ray = vec![0.0; self.num_col()];
        let mut has: HighsInt = 0;
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively; HiGHS writes one flag and at most one value per
        // column of its model, which `ray` has room for.
        let status = unsafe { Highs_getPrimalRay(self.raw(), &raw mut has, ray.as_mut_ptr()) };
        ray_result(status, has, ray)
    }

    /// HiGHS's dual ray, one value per row; `Ok(None)` when it has none, `Err` when the call fails.
    ///
    /// HiGHS may run its simplex solver again to find one, which replaces the last run's result.
    pub(crate) fn dual_ray(&mut self) -> Result<Option<Vec<f64>>, CallStatus> {
        let mut ray = vec![0.0; self.num_row()];
        let mut has: HighsInt = 0;
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively; HiGHS writes one flag and at most one value per
        // row of its model, which `ray` has room for.
        let status = unsafe { Highs_getDualRay(self.raw(), &raw mut has, ray.as_mut_ptr()) };
        ray_result(status, has, ray)
    }

    /// HiGHS's IIS of the loaded model, computed under the instance's options; `None` when the call fails.
    ///
    /// HiGHS may run its simplex solver to find one, which replaces the last run's result.
    pub(crate) fn iis(&mut self) -> Option<RawIis> {
        let (mut num_cols, mut num_rows): (HighsInt, HighsInt) = (0, 0);
        let null = ptr::null_mut();
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively; with every array null HiGHS computes the IIS, keeps
        // it, and writes only the two counts.
        let status = unsafe {
            Highs_getIis(
                self.raw(),
                &raw mut num_cols,
                &raw mut num_rows,
                null,
                null,
                null,
                null,
                null,
                null,
            )
        };
        if CallStatus::from_raw(status) == CallStatus::Error {
            return None;
        }
        let (n, m) = (usize::try_from(num_cols).ok()?, usize::try_from(num_rows).ok()?);
        let (mut col_index, mut col_bound, mut row_index, mut row_bound) =
            (vec![0; n], vec![0; n], vec![0; m], vec![0; m]);
        // SAFETY: as above; HiGHS answers this second call from the IIS it kept, of the same `n` columns and `m`
        // rows, so it writes `n` entries into each column array and `m` into each row array, which they hold. The
        // per-model status arrays are null, which HiGHS skips.
        let status = unsafe {
            Highs_getIis(
                self.raw(),
                &raw mut num_cols,
                &raw mut num_rows,
                col_index.as_mut_ptr(),
                row_index.as_mut_ptr(),
                col_bound.as_mut_ptr(),
                row_bound.as_mut_ptr(),
                null,
                null,
            )
        };
        debug_assert!(
            usize::try_from(num_cols).ok() == Some(n) && usize::try_from(num_rows).ok() == Some(m),
            "Instance: HiGHS changed its IIS between two reads"
        );
        (CallStatus::from_raw(status) != CallStatus::Error).then(|| RawIis {
            cols: col_index.into_iter().zip(col_bound).collect(),
            rows: row_index.into_iter().zip(row_bound).collect(),
        })
    }

    /// Solves HiGHS's feasibility relaxation of the loaded model under `penalties` and returns the relaxed point,
    /// one value per column; `None` when the call fails.
    ///
    /// # Panics
    ///
    /// When the penalties do not hold one entry per column bound and per row of HiGHS's model.
    pub(crate) fn feasibility_relaxation(&mut self, penalties: &Penalties) -> Option<Vec<f64>> {
        let (n, m) = (self.num_col(), self.num_row());
        assert!(
            penalties.lower.len() == n && penalties.upper.len() == n && penalties.rows.len() == m,
            "Instance: the penalties do not fit HiGHS's model"
        );
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        // SAFETY: the instance is live and borrowed exclusively, and HiGHS reads one penalty per column bound and
        // per row of its model, which the slices hold; the negative global penalties are not read beside them.
        let status = unsafe {
            Highs_feasibilityRelaxation(
                self.raw(),
                -1.0,
                -1.0,
                -1.0,
                penalties.lower.as_ptr(),
                penalties.upper.as_ptr(),
                penalties.rows.as_ptr(),
            )
        };
        if CallStatus::from_raw(status) == CallStatus::Error {
            return None;
        }
        // HiGHS restores the model after solving its elastic LP and, as after every call that changes the model,
        // resizes its solution to the model's columns, so the relaxed point is read as any primal solution is.
        let values = self.primal_values(n);
        Some(values)
    }
}

/// A ray call's result: the ray when the call succeeded and HiGHS has one.
///
/// # Errors
///
/// The call's status when it failed.
fn ray_result(status: HighsInt, has: HighsInt, ray: Vec<f64>) -> Result<Option<Vec<f64>>, CallStatus> {
    match CallStatus::from_raw(status) {
        CallStatus::Error => Err(CallStatus::Error),
        CallStatus::Ok | CallStatus::Warning => Ok((has != 0).then_some(ray)),
    }
}
