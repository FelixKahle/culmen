// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cost and bound ranging of an optimal basis.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::TiVec;

use super::Line;
use crate::{Col, Row};

/// How far each cost and bound of an optimal LP may move before its optimal basis changes:
/// [`Solver::ranging`](crate::Solver::ranging).
///
/// Each entry holds the [`Limit`] of a move down and of a move up, one parameter at a time, all others fixed.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ranging {
    /// The ranging of every column's cost.
    pub costs: TiVec<Col, Ranged>,
    /// The ranging of every column's value, moved by its bounds.
    pub col_bounds: TiVec<Col, Ranged>,
    /// The ranging of every row's activity, moved by its bounds.
    pub row_bounds: TiVec<Row, Ranged>,
}

/// The limits of moving one cost or bound down and up.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ranged {
    /// The limit of a decrease.
    pub down: Limit,
    /// The limit of an increase.
    pub up: Limit,
}

/// Where a move of one cost or bound changes the optimal basis, and what the objective is there.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Limit {
    /// The cost or value at which the basis changes; infinite when it never does in this direction.
    pub value: f64,
    /// The optimal objective value at [`value`](Self::value), offset included; infinite when the program becomes
    /// unbounded or infeasible first.
    pub objective: f64,
    /// The line that enters the basis there; `None` when none does.
    pub entering: Option<Line>,
    /// The line that leaves the basis there; `None` when none does.
    pub leaving: Option<Line>,
}
