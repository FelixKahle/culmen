// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed handles of the columns and rows of an [`Lp`](super::Lp).

use core::iter::FusedIterator;
use core::ops::Range;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A column of an LP: a variable of the model it was built from.
///
/// A `Col` is a plain `Copy` index. It is handed out by [`LpPlan::col`](super::LpPlan::col) and
/// [`Model::add_var`](crate::Model::add_var), and the column of a model variable is the same `Col` in the LP the
/// model builds, so a solution is read with the handle the model returned. A `Col` from one LP used on another
/// reads that LP's column of the same index; every read that cannot answer for the index panics or returns `None`
/// as its docs say, never anything undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct Col(usize);

impl Col {
    /// The column at a position, counting from zero.
    ///
    /// ```
    /// use culmen_core::Col;
    /// assert_eq!(Col::new(3).index(), 3);
    /// ```
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// The column's position, counting from zero.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for Col {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl From<Col> for usize {
    fn from(col: Col) -> Self {
        col.0
    }
}

/// A row of an LP: a constraint of the model it was built from.
///
/// Like [`Col`], a plain `Copy` index, handed out by [`LpPlan::row`](super::LpPlan::row) and
/// [`Model::add_constraint`](crate::Model::add_constraint).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct Row(usize);

impl Row {
    /// The row at a position, counting from zero.
    ///
    /// ```
    /// use culmen_core::Row;
    /// assert_eq!(Row::new(2).index(), 2);
    /// ```
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// The row's position, counting from zero.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for Row {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl From<Row> for usize {
    fn from(row: Row) -> Self {
        row.0
    }
}

/// Every column of an LP, in order; made by [`Lp::cols`](super::Lp::cols).
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Cols(Range<usize>);

impl Cols {
    /// The first `count` columns.
    pub(crate) const fn new(count: usize) -> Self {
        Self(0..count)
    }
}

impl Iterator for Cols {
    type Item = Col;

    fn next(&mut self) -> Option<Col> {
        self.0.next().map(Col)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for Cols {
    fn next_back(&mut self) -> Option<Col> {
        self.0.next_back().map(Col)
    }
}

impl ExactSizeIterator for Cols {}

impl FusedIterator for Cols {}

/// Every row of an LP, in order; made by [`Lp::rows`](super::Lp::rows).
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Rows(Range<usize>);

impl Rows {
    /// The first `count` rows.
    pub(crate) const fn new(count: usize) -> Self {
        Self(0..count)
    }
}

impl Iterator for Rows {
    type Item = Row;

    fn next(&mut self) -> Option<Row> {
        self.0.next().map(Row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for Rows {
    fn next_back(&mut self) -> Option<Row> {
        self.0.next_back().map(Row)
    }
}

impl ExactSizeIterator for Rows {}

impl FusedIterator for Rows {}
