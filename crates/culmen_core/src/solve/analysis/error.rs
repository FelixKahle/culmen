// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why an analysis was refused.

use core::fmt;

use super::{Analyses, Analysis, BasisStatus, Line};
use crate::{ModelClass, Program};

/// Why a [`Solver`](crate::Solver) refused an analysis, or the backend failed it with its own error `E`.
///
/// Every check but the backend's own is the same for every backend: the solver makes it before the backend is asked.
///
/// ```
/// use culmen_core::{Analyses, Analysis, AnalysisError, ModelClass};
///
/// let error = AnalysisError::<std::io::Error>::Unsupported { analysis: Analysis::Iis, supported: Analyses::NONE };
/// assert_eq!(error.to_string(), "Solver: the backend provides {}, not IIS");
/// let error = AnalysisError::<std::io::Error>::Class { analysis: Analysis::Duals, class: ModelClass::Milp };
/// assert_eq!(error.to_string(), "Solver: duals are defined for {LP, QP}, not a MILP");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnalysisError<E> {
    /// The backend does not provide the analysis.
    Unsupported {
        /// The analysis asked for.
        analysis: Analysis,
        /// The analyses the backend provides.
        supported: Analyses,
    },
    /// The analysis is not defined for the program's class.
    Class {
        /// The analysis asked for.
        analysis: Analysis,
        /// The program's class.
        class: ModelClass,
    },
    /// The analysis reads a solve, and there is none to read: no solve since the program was loaded, the last solve
    /// failed, or a later basis, IIS or relaxation replaced its result.
    NotSolved {
        /// The analysis asked for.
        analysis: Analysis,
    },
    /// The analysis reads an optimal solve, and the last solve did not end [`Optimal`](crate::Outcome::Optimal).
    NotOptimal {
        /// The analysis asked for.
        analysis: Analysis,
    },
    /// A basis or the weights given do not fit the program.
    Misfit(Misfit),
    /// The backend failed.
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for AnalysisError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { analysis, supported } => {
                write!(f, "Solver: the backend provides {supported}, not {analysis}")
            }
            Self::Class { analysis, class } => {
                let are = analysis.agree("is", "are");
                write!(
                    f,
                    "Solver: {analysis} {are} defined for {}, not a {class}",
                    analysis.classes()
                )
            }
            Self::NotSolved { analysis } => {
                let need = analysis.agree("needs", "need");
                write!(f, "Solver: {analysis} {need} a solve, and there is none to read")
            }
            Self::NotOptimal { analysis } => {
                write!(
                    f,
                    "Solver: {analysis} {} an optimal solve",
                    analysis.agree("needs", "need")
                )
            }
            Self::Misfit(misfit) => write!(f, "Solver: {misfit}"),
            Self::Backend(error) => write!(f, "Solver: the backend failed the analysis: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for AnalysisError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            Self::Misfit(misfit) => Some(misfit),
            Self::Unsupported { .. } | Self::Class { .. } | Self::NotSolved { .. } | Self::NotOptimal { .. } => None,
        }
    }
}

impl<E> From<Misfit> for AnalysisError<E> {
    fn from(misfit: Misfit) -> Self {
        Self::Misfit(misfit)
    }
}

/// How a [`Basis`](crate::Basis) or [`Weights`](crate::Weights) fail to fit a program.
///
/// ```
/// use culmen_core::{Misfit, Table};
/// let misfit = Misfit::Length { table: Table::BasisRows, expected: 2, found: 3 };
/// assert_eq!(misfit.to_string(), "the basis has 3 row statuses for 2 rows");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Misfit {
    /// A table holds the wrong number of entries.
    Length {
        /// The table.
        table: Table,
        /// The number of columns or rows of the program.
        expected: usize,
        /// The number of entries given.
        found: usize,
    },
    /// A basis has not one basic line per row.
    BasicCount {
        /// The number of rows.
        expected: usize,
        /// The number of basic columns and rows.
        found: usize,
    },
    /// A basis puts a line at a bound the line does not have.
    Status {
        /// The line.
        line: Line,
        /// Its status, [`BasisStatus::Lower`] or [`BasisStatus::Upper`].
        status: BasisStatus,
    },
}

impl fmt::Display for Misfit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { table, expected, found } => {
                let (what, lines) = table.names();
                write!(f, "{what} {found} {} for {expected} {lines}", table.entries())
            }
            Self::BasicCount { expected, found } => {
                write!(f, "the basis has {found} basic lines for {expected} rows")
            }
            Self::Status { line, status } => {
                write!(f, "the basis puts {line} at its {status} bound, which is infinite")
            }
        }
    }
}

impl core::error::Error for Misfit {}

/// A per-column or per-row table an analysis takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Table {
    /// [`Basis::cols`](crate::Basis::cols).
    BasisCols,
    /// [`Basis::rows`](crate::Basis::rows).
    BasisRows,
    /// [`Weights::lower`](crate::Weights::lower).
    LowerWeights,
    /// [`Weights::upper`](crate::Weights::upper).
    UpperWeights,
    /// [`Weights::rows`](crate::Weights::rows).
    RowWeights,
}

impl Table {
    /// Whether the table is indexed by column rather than by row.
    const fn is_columns(self) -> bool {
        matches!(self, Self::BasisCols | Self::LowerWeights | Self::UpperWeights)
    }

    /// The owner with its verb, and the lines it counts, as text.
    const fn names(self) -> (&'static str, &'static str) {
        let owner = match self {
            Self::BasisCols | Self::BasisRows => "the basis has",
            Self::LowerWeights | Self::UpperWeights | Self::RowWeights => "the weights have",
        };
        (owner, if self.is_columns() { "columns" } else { "rows" })
    }

    /// The kind of entry, as text.
    const fn entries(self) -> &'static str {
        match self {
            Self::BasisCols => "column statuses",
            Self::BasisRows => "row statuses",
            Self::LowerWeights => "lower-bound weights",
            Self::UpperWeights => "upper-bound weights",
            Self::RowWeights => "row weights",
        }
    }

    /// Checks that the table has one entry per line of `program` it is indexed by.
    ///
    /// # Errors
    ///
    /// [`Misfit::Length`] when it has not.
    pub(crate) fn check(self, len: usize, program: &Program) -> Result<(), Misfit> {
        let expected = if self.is_columns() {
            program.num_cols()
        } else {
            program.num_rows()
        };
        if len == expected {
            Ok(())
        } else {
            Err(Misfit::Length {
                table: self,
                expected,
                found: len,
            })
        }
    }
}
