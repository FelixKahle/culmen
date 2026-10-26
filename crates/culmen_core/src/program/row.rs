// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Borrowed views of one row of a [`Program`](super::Program).

use core::iter::{Copied, FusedIterator, Zip};
use core::slice;

use typed_index_collections::TiSlice;

use super::{Bounds, Col};

/// One row of a [`Program`](super::Program): its bounds, its name and its entries, sorted by column, each column at
/// most once and with a non-zero coefficient.
///
/// ```
/// use culmen_core::{ProgramColumn, ProgramPlan, bounds};
///
/// let mut plan = ProgramPlan::new();
/// let x = plan.col(ProgramColumn::binary(1.0));
/// let y = plan.col(ProgramColumn::binary(1.0));
/// let r = plan.row(bounds!(0.0, 1.0), [(y, 2.0), (x, 1.0)]);
/// let program = plan.build()?;
/// let row = program.row(r);
/// assert_eq!((row.bounds(), row.name(), row.len()), (bounds!(0.0, 1.0), None, 2));
/// assert_eq!(row.into_iter().collect::<Vec<_>>(), [(x, 1.0), (y, 2.0)]);
/// # Ok::<(), culmen_core::ProgramError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowView<'a> {
    bounds: Bounds,
    name: Option<&'a str>,
    cols: &'a [Col],
    values: &'a [f64],
}

impl<'a> RowView<'a> {
    /// A view of a row whose columns and coefficients are `cols` and `values`, entry by entry.
    pub(crate) fn new(bounds: Bounds, name: Option<&'a str>, cols: &'a [Col], values: &'a [f64]) -> Self {
        Self {
            bounds,
            name,
            cols,
            values,
        }
    }

    /// The interval the row's activity must lie in.
    #[must_use]
    pub const fn bounds(&self) -> Bounds {
        self.bounds
    }

    /// The row's name; `None` names it by its index.
    #[must_use]
    pub const fn name(&self) -> Option<&'a str> {
        self.name
    }

    /// The number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.cols.len()
    }

    /// Whether the row has no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }

    /// The entries as `(column, coefficient)`, sorted by column.
    pub fn entries(&self) -> RowEntries<'a> {
        RowEntries {
            inner: self.cols.iter().copied().zip(self.values.iter().copied()),
        }
    }

    /// The row's activity `Σ coefficient · value` at a point.
    ///
    /// # Panics
    ///
    /// When `values` has no value for a column of the row.
    #[must_use]
    pub fn activity(&self, values: &TiSlice<Col, f64>) -> f64 {
        self.entries().map(|(col, a)| a * values[col]).sum()
    }
}

impl<'a> IntoIterator for RowView<'a> {
    type Item = (Col, f64);
    type IntoIter = RowEntries<'a>;

    fn into_iter(self) -> RowEntries<'a> {
        self.entries()
    }
}

/// The entries of a row of a [`Program`](super::Program) as `(column, coefficient)`, sorted by column; made by
/// [`RowView::entries`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct RowEntries<'a> {
    inner: Zip<Copied<slice::Iter<'a, Col>>, Copied<slice::Iter<'a, f64>>>,
}

impl Iterator for RowEntries<'_> {
    type Item = (Col, f64);

    fn next(&mut self) -> Option<(Col, f64)> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for RowEntries<'_> {
    fn next_back(&mut self) -> Option<(Col, f64)> {
        self.inner.next_back()
    }
}

impl ExactSizeIterator for RowEntries<'_> {}

impl FusedIterator for RowEntries<'_> {}
