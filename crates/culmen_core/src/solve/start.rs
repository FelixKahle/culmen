// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A starting solution handed to the solver before a solve, and why a solve refused its request.

use core::fmt;
use core::iter::{Copied, Enumerate, FusedIterator};
use core::slice;
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::TiVec;

use crate::{Col, ProgramFeature, ProgramFeatures};

/// A starting solution for one solve: a value for every column, or for some.
///
/// The solver may start its search from it: a MIP solver takes a feasible start as its first incumbent and may
/// complete a partial or infeasible one, an LP solver may start from it. A start need not be feasible, but every value
/// must be finite and name a column of the program; [`Start::check`] states it, and
/// [`Solver::solve`](crate::Solver::solve) checks it before the backend sees the start.
///
/// ```
/// use culmen_core::{Col, Start, StartError, TiVec};
///
/// let full = Start::from(TiVec::<Col, f64>::from(vec![1.0, 0.0, 2.5]));
/// assert_eq!(full.check(3), Ok(()));
/// assert_eq!(full.check(2), Err(StartError::Length { expected: 2, found: 3 }));
///
/// let sparse = Start::from(vec![(Col::new(2), 2.5), (Col::new(0), 1.0)]);
/// assert_eq!(sparse.entries().collect::<Vec<_>>(), [(Col::new(2), 2.5), (Col::new(0), 1.0)]);
/// assert_eq!(sparse.check(2), Err(StartError::UnknownColumn { col: Col::new(2) }));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Start {
    /// A value for every column, indexed by [`Col`].
    Full(TiVec<Col, f64>),
    /// Values for some columns as `(column, value)`, each column at most once; the solver chooses the rest.
    Sparse(Vec<(Col, f64)>),
}

impl Start {
    /// Whether the start fits a program of `num_cols` columns: a full start has one value per column, a sparse one
    /// names each column at most once and only columns below `num_cols`, and every value is finite.
    ///
    /// # Errors
    ///
    /// [`StartError::Length`] for a full start of another length, else [`StartError::UnknownColumn`],
    /// [`StartError::RepeatedColumn`] or [`StartError::NotFinite`] for the first entry at fault.
    pub fn check(&self, num_cols: usize) -> Result<(), StartError> {
        if let Self::Full(values) = self
            && values.len() != num_cols
        {
            return Err(StartError::Length {
                expected: num_cols,
                found: values.len(),
            });
        }
        let mut seen = HashSet::new();
        self.entries().try_for_each(|(col, value)| {
            if col.index() >= num_cols {
                Err(StartError::UnknownColumn { col })
            } else if !seen.insert(col) {
                Err(StartError::RepeatedColumn { col })
            } else if !value.is_finite() {
                Err(StartError::NotFinite { col, value })
            } else {
                Ok(())
            }
        })
    }

    /// The start's entries as `(column, value)`: every column in order for a full start, the entries as given for a
    /// sparse one.
    pub fn entries(&self) -> StartEntries<'_> {
        StartEntries {
            inner: match self {
                Self::Full(values) => Entries::Full(values.iter().copied().enumerate()),
                Self::Sparse(entries) => Entries::Sparse(entries.iter().copied()),
            },
        }
    }
}

/// A full start.
impl From<TiVec<Col, f64>> for Start {
    fn from(values: TiVec<Col, f64>) -> Self {
        Self::Full(values)
    }
}

/// A sparse start.
impl From<Vec<(Col, f64)>> for Start {
    fn from(entries: Vec<(Col, f64)>) -> Self {
        Self::Sparse(entries)
    }
}

/// The entries of a [`Start`] as `(column, value)`; made by [`Start::entries`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct StartEntries<'a> {
    inner: Entries<'a>,
}

/// The variant a [`StartEntries`] walks.
#[derive(Clone, Debug)]
enum Entries<'a> {
    Full(Enumerate<Copied<slice::Iter<'a, f64>>>),
    Sparse(Copied<slice::Iter<'a, (Col, f64)>>),
}

impl Iterator for StartEntries<'_> {
    type Item = (Col, f64);

    fn next(&mut self) -> Option<(Col, f64)> {
        match &mut self.inner {
            Entries::Full(values) => values.next().map(|(i, v)| (Col::new(i), v)),
            Entries::Sparse(entries) => entries.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            Entries::Full(values) => values.size_hint(),
            Entries::Sparse(entries) => entries.size_hint(),
        }
    }
}

impl ExactSizeIterator for StartEntries<'_> {}

impl FusedIterator for StartEntries<'_> {}

/// Why [`Start::check`] refused a start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StartError {
    /// A full start holds another number of values than the program has columns.
    Length {
        /// The number of columns.
        expected: usize,
        /// The number of values given.
        found: usize,
    },
    /// A sparse start names a column the program does not have.
    UnknownColumn {
        /// The column named.
        col: Col,
    },
    /// A sparse start names a column twice.
    RepeatedColumn {
        /// The column named again.
        col: Col,
    },
    /// A value is NaN or infinite.
    NotFinite {
        /// The column of the value.
        col: Col,
        /// The value given.
        value: f64,
    },
}

impl fmt::Display for StartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { expected, found } => {
                write!(f, "Start: {found} values given for {expected} columns")
            }
            Self::UnknownColumn { col } => write!(f, "Start: {col:?} is not a column of the program"),
            Self::RepeatedColumn { col } => write!(f, "Start: {col:?} is given twice"),
            Self::NotFinite { col, value } => write!(f, "Start: the value {value} of {col:?} is not finite"),
        }
    }
}

impl core::error::Error for StartError {}

/// Why [`Solver::solve`](crate::Solver::solve) failed: the request uses a feature the backend lacks, its start does
/// not fit the program, or the backend failed with its own error `E`.
///
/// The first two are the same for every backend: they are decided before the backend sees the request.
///
/// ```
/// use culmen_core::{ProgramFeature, ProgramFeatures, SolveError};
///
/// let error = SolveError::<std::io::Error>::UnsupportedFeature {
///     feature: ProgramFeature::StartSolution,
///     supported: ProgramFeatures::NONE,
/// };
/// assert_eq!(error.to_string(), "Solver: the backend takes {}, not a start solution");
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum SolveError<E> {
    /// The request uses a feature not in [`Backend::FEATURES`](crate::Backend::FEATURES).
    UnsupportedFeature {
        /// The feature the request uses.
        feature: ProgramFeature,
        /// The features the backend takes.
        supported: ProgramFeatures,
    },
    /// The request's start does not fit the program.
    Start(StartError),
    /// The backend failed.
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for SolveError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFeature { feature, supported } => {
                write!(f, "Solver: the backend takes {supported}, not {feature}")
            }
            Self::Start(error) => write!(f, "Solver: the request's start is refused: {error}"),
            Self::Backend(error) => write!(f, "Solver: the backend failed: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for SolveError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::UnsupportedFeature { .. } => None,
            Self::Start(error) => Some(error),
            Self::Backend(error) => Some(error),
        }
    }
}
