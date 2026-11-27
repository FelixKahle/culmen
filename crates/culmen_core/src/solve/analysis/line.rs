// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A column or a row, and an end of its bounds: what the analyses name.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Bounds, Col, Program, Row};

/// A line of a program's constraint matrix: a column, or a row.
///
/// In the simplex method a column is a structural variable and a row the logical variable of its activity, so a basis
/// change, a bound and a violation each name a `Line`.
///
/// ```
/// use culmen_core::{Col, Line, Row};
/// assert_eq!(Line::from(Col::new(2)), Line::Col(Col::new(2)));
/// assert_eq!(Line::Row(Row::new(0)).to_string(), "row 0");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Line {
    /// A column.
    Col(Col),
    /// A row.
    Row(Row),
}

impl Line {
    /// The line's bounds in `program`: a column's bounds, or a row's activity bounds.
    ///
    /// # Panics
    ///
    /// When the line is not a column or row of `program`.
    #[must_use]
    pub fn bounds(self, program: &Program) -> Bounds {
        match self {
            Self::Col(col) => program.columns()[col].bounds,
            Self::Row(row) => program.row_bounds()[row],
        }
    }
}

impl From<Col> for Line {
    fn from(col: Col) -> Self {
        Self::Col(col)
    }
}

impl From<Row> for Line {
    fn from(row: Row) -> Self {
        Self::Row(row)
    }
}

/// `column 3` or `row 0`, by position.
impl fmt::Display for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Col(col) => write!(f, "column {}", col.index()),
            Self::Row(row) => write!(f, "row {}", row.index()),
        }
    }
}

/// One end of a line's bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Bound {
    /// The lower end.
    Lower,
    /// The upper end.
    Upper,
}

impl Bound {
    /// This end of `bounds`, infinite when the interval is open there.
    #[must_use]
    pub const fn of(self, bounds: Bounds) -> f64 {
        match self {
            Self::Lower => bounds.lower(),
            Self::Upper => bounds.upper(),
        }
    }
}

/// `lower` or `upper`.
impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Lower => "lower",
            Self::Upper => "upper",
        })
    }
}
