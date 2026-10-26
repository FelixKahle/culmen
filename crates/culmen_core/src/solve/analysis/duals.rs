// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The dual solution of an optimal continuous program.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::TiVec;

use crate::{Col, Row};

/// The dual values of an optimal LP or QP, read with the program's handles: [`Solver::duals`](crate::Solver::duals).
///
/// # Sign convention
///
/// Every dual value is the rate at which the optimal objective changes as the bound it belongs to rises, in either
/// sense. So at the optimum `x`, with costs `c`, the quadratic part's gradient `∇q(x)` and the matrix `A`,
///
/// ```text
/// c + ∇q(x) = Aᵀ·rows + cols
/// ```
///
/// a row or column strictly inside its bounds has dual zero, and in a minimization a line at its lower bound has a
/// dual of zero or above and one at its upper bound a dual of zero or below; a maximization reverses both signs.
/// The optimal objective is then `offset + Σ rows[r]·(bound row r is at) + Σ cols[c]·(bound column c is at)` for an
/// LP. This is HiGHS's convention.
///
/// A row's activity is not a dual value; it is read from the solution with
/// [`Program::activities`](crate::Program::activities).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Duals {
    /// The dual value of every row.
    pub rows: TiVec<Row, f64>,
    /// The dual value of every column's bounds: its reduced cost.
    pub cols: TiVec<Col, f64>,
}
