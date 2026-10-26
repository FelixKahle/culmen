// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The feasibility relaxation: the least weighted violation of a program's bounds and rows.

use core::fmt;
use core::iter::FusedIterator;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::{TiSlice, TiVec};

use super::{Bound, Line, Table};
use crate::{Col, Misfit, Program, Row};

/// The price of one unit of violation of a bound or row: a finite number, zero or above.
///
/// Literals are written with [`penalty!`](crate::penalty) and checked when the code is compiled; numbers read at
/// run time are checked by [`Penalty::new`] or `TryFrom<f64>`.
///
/// ```
/// use culmen_core::{InvalidPenalty, Penalty, penalty};
/// assert_eq!(penalty!(2.0).to_string(), "Penalty(2)");
/// assert_eq!(Penalty::try_from(-1.0), Err(InvalidPenalty::Negative));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "f64", into = "f64")
)]
pub struct Penalty(f64);

impl Penalty {
    /// The price `value` per unit of violation.
    ///
    /// # Errors
    ///
    /// [`InvalidPenalty`] when `value` is not finite or below zero.
    pub const fn new(value: f64) -> Result<Self, InvalidPenalty> {
        if !value.is_finite() {
            Err(InvalidPenalty::NotFinite)
        } else if value < 0.0 {
            Err(InvalidPenalty::Negative)
        } else {
            Ok(Self(value))
        }
    }

    /// The price as a number.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// The penalty of a literal, or a panic naming the defect: in a `const` context, a compile error.
    ///
    /// # Panics
    ///
    /// When [`Penalty::new`] refuses `value`.
    #[doc(hidden)]
    #[track_caller]
    #[must_use]
    pub const fn literal(value: f64) -> Self {
        match Self::new(value) {
            Ok(penalty) => penalty,
            Err(InvalidPenalty::NotFinite) => panic!("Penalty: not a finite number"),
            Err(InvalidPenalty::Negative) => panic!("Penalty: below zero"),
        }
    }
}

impl TryFrom<f64> for Penalty {
    type Error = InvalidPenalty;

    fn try_from(value: f64) -> Result<Self, InvalidPenalty> {
        Self::new(value)
    }
}

impl From<Penalty> for f64 {
    fn from(penalty: Penalty) -> Self {
        penalty.0
    }
}

impl fmt::Display for Penalty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Penalty({})", self.0)
    }
}

/// Why [`Penalty::new`] refused a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvalidPenalty {
    /// The number is NaN or infinite.
    NotFinite,
    /// The number is below zero.
    Negative,
}

impl fmt::Display for InvalidPenalty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite => write!(f, "Penalty: not a finite number"),
            Self::Negative => write!(f, "Penalty: below zero"),
        }
    }
}

impl core::error::Error for InvalidPenalty {}

/// A literal [`Penalty`], checked when the code is compiled.
///
/// ```
/// let penalty = culmen_core::penalty!(0.5);
/// assert_eq!(penalty.get(), 0.5);
/// ```
#[macro_export]
macro_rules! penalty {
    ($value:expr $(,)?) => {
        const { $crate::Penalty::literal($value) }
    };
}

/// Which bounds and rows a feasibility relaxation may violate, and at what price per unit:
/// [`Solver::feasibility_relaxation`](crate::Solver::feasibility_relaxation).
///
/// A `None` keeps that bound or row hard. A row's weight prices a violation of either of its bounds.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Weights {
    /// The price of every column's value below its lower bound.
    pub lower: TiVec<Col, Option<Penalty>>,
    /// The price of every column's value above its upper bound.
    pub upper: TiVec<Col, Option<Penalty>>,
    /// The price of every row's activity outside its bounds.
    pub rows: TiVec<Row, Option<Penalty>>,
}

impl Weights {
    /// The same price for every column bound but a semi column's, which stays hard, and the same for every row, for
    /// `program`.
    ///
    /// ```
    /// use culmen_core::{Linear, Model, Var, Weights, penalty};
    ///
    /// let mut m = Model::new();
    /// let x = m.add_var(Var::continuous());
    /// let cap = m.add_constraint(x.leq(4.0));
    /// let weights = Weights::uniform(&m.build()?, None, Some(penalty!(1.0)));
    /// assert_eq!((weights.lower[x], weights.rows[cap]), (None, Some(penalty!(1.0))));
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    #[must_use]
    pub fn uniform(program: &Program, cols: Option<Penalty>, rows: Option<Penalty>) -> Self {
        let col = |c: &crate::ProgramColumn| cols.filter(|_| !c.integrality.is_semi());
        Self {
            lower: program.columns().iter().map(col).collect(),
            upper: program.columns().iter().map(col).collect(),
            rows: program.rows().map(|_| rows).collect(),
        }
    }

    /// Checks that the weights hold one entry per column bound and per row of `program`, and keep every semi
    /// column's bounds hard.
    ///
    /// # Errors
    ///
    /// [`Misfit::Length`] for the first table of the wrong length, else [`Misfit::SoftSemiBound`] for the first semi
    /// column with a priced bound.
    pub fn check(&self, program: &Program) -> Result<(), Misfit> {
        Table::LowerWeights.check(self.lower.len(), program)?;
        Table::UpperWeights.check(self.upper.len(), program)?;
        Table::RowWeights.check(self.rows.len(), program)?;
        let soft = program.cols().find(|&col| {
            program.columns()[col].integrality.is_semi() && (self.lower[col].is_some() || self.upper[col].is_some())
        });
        soft.map_or(Ok(()), |col| Err(Misfit::SoftSemiBound { col }))
    }

    /// The price of one unit of violation of `bound` of `line`, `None` when it is hard.
    ///
    /// # Panics
    ///
    /// When the line has no entry in the weights.
    #[must_use]
    pub fn price(&self, line: Line, bound: Bound) -> Option<Penalty> {
        match (line, bound) {
            (Line::Col(col), Bound::Lower) => self.lower[col],
            (Line::Col(col), Bound::Upper) => self.upper[col],
            (Line::Row(row), _) => self.rows[row],
        }
    }

    /// The weighted violation of `violations`: the sum of each soft violation's amount times its price; a hard
    /// violation counts nothing.
    ///
    /// # Panics
    ///
    /// When a violation names a line that has no entry in the weights.
    #[must_use]
    pub fn penalty(&self, violations: impl IntoIterator<Item = Violation>) -> f64 {
        violations
            .into_iter()
            .filter_map(|v| self.price(v.line, v.bound).map(|price| price.get() * v.amount))
            .sum()
    }
}

/// A point that violates a program's bounds and rows least, by the [`Weights`] it was relaxed with.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Relaxed {
    /// The value of every column. It satisfies every hard bound and row and every integrality, within the solver's
    /// tolerances.
    pub values: TiVec<Col, f64>,
}

impl Relaxed {
    /// Every bound and row of `program` the point violates, columns first; a semi-continuous or semi-integer column
    /// violates its bounds only by as much as zero, which it may also take, is farther off.
    ///
    /// # Panics
    ///
    /// The iterator panics when the point holds fewer values than `program` has columns.
    pub fn violations<'a>(&'a self, program: &'a Program) -> Violations<'a> {
        Violations::new(program, &self.values)
    }
}

/// By how much a point lies outside one end of a line's bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Violation {
    /// The column or row.
    pub line: Line,
    /// The end of its bounds the value lies beyond.
    pub bound: Bound,
    /// How far beyond it, above zero.
    pub amount: f64,
}

/// The violations of a point, columns first then rows, each line at most once; made by [`Relaxed::violations`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Violations<'a> {
    program: &'a Program,
    values: &'a TiSlice<Col, f64>,
    /// The position of the next line to look at: columns, then rows after them.
    next: usize,
}

impl<'a> Violations<'a> {
    /// The violations of `values` in `program`.
    pub(crate) const fn new(program: &'a Program, values: &'a TiSlice<Col, f64>) -> Self {
        Self {
            program,
            values,
            next: 0,
        }
    }

    /// The violation of the line at `position`, `None` when there is none.
    ///
    /// # Panics
    ///
    /// When `position` is past the last row, or `values` holds fewer values than the program has columns.
    fn at(&self, position: usize) -> Option<Violation> {
        let num_cols = self.program.num_cols();
        let (line, value) = if position < num_cols {
            let col = Col::new(position);
            (Line::Col(col), self.values[col])
        } else {
            let row = Row::new(position - num_cols);
            (Line::Row(row), self.program.row(row).activity(self.values))
        };
        let bounds = line.bounds(self.program);
        let amount = match line {
            // A semi column may also be zero, so it violates its bounds only by as much as zero is farther off.
            Line::Col(col) if self.program.columns()[col].integrality.is_semi() => {
                bounds.violation(value).min(value.abs())
            }
            Line::Col(_) | Line::Row(_) => bounds.violation(value),
        };
        let bound = if value < bounds.lower() {
            Bound::Lower
        } else {
            Bound::Upper
        };
        (amount > 0.0).then_some(Violation { line, bound, amount })
    }

    /// The number of lines.
    fn end(&self) -> usize {
        self.program.num_cols() + self.program.num_rows()
    }
}

impl Iterator for Violations<'_> {
    type Item = Violation;

    fn next(&mut self) -> Option<Violation> {
        let end = self.end();
        let found = (self.next..end).find_map(|position| self.at(position).map(|v| (position, v)));
        self.next = found.map_or(end, |(position, _)| position + 1);
        found.map(|(_, violation)| violation)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.end().saturating_sub(self.next)))
    }
}

impl FusedIterator for Violations<'_> {}
