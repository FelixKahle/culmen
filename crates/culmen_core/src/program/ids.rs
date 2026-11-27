// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed handles of the columns and rows of a [`Program`](super::Program).

use core::iter::FusedIterator;
use core::ops::Range;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A column of a program: a variable of the model it was built from.
///
/// A `Col` is a plain `Copy` index. It is handed out by [`ProgramPlan::col`](super::ProgramPlan::col) and
/// [`Model::add_var`](crate::Model::add_var), and the column of a model variable is the same `Col` in the program the
/// model builds, so a solution is read with the handle the model returned. A `Col` from one program used on another
/// reads that program's column of the same index; every read that cannot answer for the index panics or returns `None`
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

/// A row of a program: a constraint of the model it was built from.
///
/// Like [`Col`], a plain `Copy` index, handed out by [`ProgramPlan::row`](super::ProgramPlan::row) and
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

/// A run of consecutive columns of a program, in order: every column, made by [`Program::cols`](super::Program::cols),
/// or the columns one edit appended, returned by [`Solver::add_cols`](crate::Solver::add_cols).
///
/// It is an iterator, and a borrowed run iterates without being used up, so it can be read more than once:
///
/// ```
/// use culmen_core::{Col, ProgramColumn, ProgramPlan};
/// let mut plan = ProgramPlan::new();
/// (0..3).for_each(|_| { plan.col(ProgramColumn::binary()); });
/// let cols = plan.build()?.cols();
/// assert_eq!((cols.len(), cols.start(), cols.end()), (3, Col::new(0), Col::new(3)));
/// assert!(cols.contains(Col::new(2)) && !cols.contains(Col::new(3)) && !cols.is_empty());
/// assert_eq!((&cols).into_iter().count() + (&cols).into_iter().count(), 6);
/// # Ok::<(), culmen_core::ProgramError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Cols(Range<usize>);

impl Cols {
    /// The first `count` columns.
    pub(crate) const fn new(count: usize) -> Self {
        Self::range(0, count)
    }

    /// The columns from index `start` up to, not including, index `end`; none when `end <= start`.
    pub(crate) const fn range(start: usize, end: usize) -> Self {
        Self(start..end)
    }

    /// The first col not yet iterated; [`Cols::end`] when there is none.
    #[must_use]
    pub const fn start(&self) -> Col {
        Col(self.0.start)
    }

    /// The col after the last, which a program that ends with these columns does not have; for the
    /// columns an edit appended, the program's count after it.
    #[must_use]
    pub const fn end(&self) -> Col {
        Col(if self.0.end < self.0.start {
            self.0.start
        } else {
            self.0.end
        })
    }

    /// Whether no col is left.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.start >= self.0.end
    }

    /// Whether `col` is one of the columns not yet iterated.
    #[must_use]
    pub const fn contains(&self, col: Col) -> bool {
        self.0.start <= col.0 && col.0 < self.0.end
    }
}

/// The columns again, leaving the run as it is: `for col in &added` reads it more than once.
impl IntoIterator for &Cols {
    type Item = Col;
    type IntoIter = Cols;

    fn into_iter(self) -> Cols {
        self.clone()
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

/// A run of consecutive rows of a program, in order: every row, made by [`Program::rows`](super::Program::rows), or
/// the rows one edit appended, returned by [`Solver::add_rows`](crate::Solver::add_rows).
///
/// Read like [`Cols`]: an iterator, and a borrowed run iterates without being used up.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Rows(Range<usize>);

impl Rows {
    /// The first `count` rows.
    pub(crate) const fn new(count: usize) -> Self {
        Self::range(0, count)
    }

    /// The rows from index `start` up to, not including, index `end`; none when `end <= start`.
    pub(crate) const fn range(start: usize, end: usize) -> Self {
        Self(start..end)
    }

    /// The first row not yet iterated; [`Rows::end`] when there is none.
    #[must_use]
    pub const fn start(&self) -> Row {
        Row(self.0.start)
    }

    /// The row after the last, which a program that ends with these rows does not have; for the
    /// rows an edit appended, the program's count after it.
    #[must_use]
    pub const fn end(&self) -> Row {
        Row(if self.0.end < self.0.start {
            self.0.start
        } else {
            self.0.end
        })
    }

    /// Whether no row is left.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.start >= self.0.end
    }

    /// Whether `row` is one of the rows not yet iterated.
    #[must_use]
    pub const fn contains(&self, row: Row) -> bool {
        self.0.start <= row.0 && row.0 < self.0.end
    }
}

/// The rows again, leaving the run as it is: `for row in &added` reads it more than once.
impl IntoIterator for &Rows {
    type Item = Row;
    type IntoIter = Rows;

    fn into_iter(self) -> Rows {
        self.clone()
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

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use proptest::prelude::*;

    use super::{Col, Cols, Row, Rows};

    proptest! {
        /// However far a run is iterated from either end, its reads agree with the indices left, kept in a queue;
        /// iterating a borrowed run yields them all and leaves the run as it was.
        #[test]
        fn test_runs_read_as_the_indices_left(
            start in 0usize..6,
            count in 0usize..6,
            steps in prop::collection::vec(any::<bool>(), 0..8),
        ) {
            let (mut cols, mut rows) = (Cols::range(start, start + count), Rows::range(start, start + count));
            let mut left: VecDeque<usize> = (start..start + count).collect();
            for front in steps {
                let (col, row, index) = if front {
                    (cols.next(), rows.next(), left.pop_front())
                } else {
                    (cols.next_back(), rows.next_back(), left.pop_back())
                };
                prop_assert_eq!((col, row), (index.map(Col::new), index.map(Row::new)));
                prop_assert_eq!((cols.len(), rows.len(), cols.is_empty()), (left.len(), left.len(), left.is_empty()));
                if let (Some(&first), Some(&last)) = (left.front(), left.back()) {
                    prop_assert_eq!((cols.start(), cols.end()), (Col::new(first), Col::new(last + 1)));
                    prop_assert_eq!((rows.start(), rows.end()), (Row::new(first), Row::new(last + 1)));
                } else {
                    prop_assert_eq!(cols.start(), cols.end());
                    prop_assert_eq!(rows.start(), rows.end());
                }
                for i in 0..12 {
                    prop_assert_eq!(cols.contains(Col::new(i)), left.contains(&i));
                    prop_assert_eq!(rows.contains(Row::new(i)), left.contains(&i));
                }
                let twice: Vec<Col> = (&cols).into_iter().chain(&cols).collect();
                let expected: Vec<Col> = left.iter().chain(&left).map(|&i| Col::new(i)).collect();
                prop_assert_eq!(twice, expected);
                prop_assert_eq!((&rows).into_iter().count(), left.len());
            }
        }
    }
}
