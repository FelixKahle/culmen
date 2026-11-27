// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A simplex basis: where each column and row stands, read after a solve and set before the next.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::TiVec;

use super::{Bound, Line, Misfit, Table};
use crate::{Col, Program, Row};

/// Where a column or row stands in a simplex basis; HiGHS's `kHighsBasisStatus*`.
///
/// A row's status refers to its activity: [`Lower`](Self::Lower) is an activity at the row's lower bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BasisStatus {
    /// Nonbasic at its lower bound.
    Lower,
    /// Basic: its value is determined by the others.
    Basic,
    /// Nonbasic at its upper bound.
    Upper,
    /// Nonbasic at zero, as a free line is.
    Zero,
    /// Nonbasic, at a bound the solver chooses.
    NonBasic,
}

impl BasisStatus {
    /// The bound the status puts its line at, `None` when it names none.
    #[must_use]
    pub const fn bound(self) -> Option<Bound> {
        match self {
            Self::Lower => Some(Bound::Lower),
            Self::Upper => Some(Bound::Upper),
            Self::Basic | Self::Zero | Self::NonBasic => None,
        }
    }
}

/// `lower`, `basic`, `upper`, `zero` or `nonbasic`.
impl fmt::Display for BasisStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Lower => "lower",
            Self::Basic => "basic",
            Self::Upper => "upper",
            Self::Zero => "zero",
            Self::NonBasic => "nonbasic",
        })
    }
}

/// A simplex basis of a program: one status per column and one per row, read with the program's handles.
///
/// [`Solver::basis`](crate::Solver::basis) reads it after a solve and
/// [`Solver::set_basis`](crate::Solver::set_basis) sets it as the start of the next, after [`Basis::check`] found
/// that it fits the program.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Basis {
    /// The status of every column.
    pub cols: TiVec<Col, BasisStatus>,
    /// The status of every row's activity.
    pub rows: TiVec<Row, BasisStatus>,
}

impl Basis {
    /// The basis every solve may start from: every row basic, every column nonbasic at a bound it has, or at zero
    /// when free.
    ///
    /// ```
    /// use culmen_core::{Basis, Linear, Model, Var};
    ///
    /// let mut m = Model::new();
    /// let x = m.add_var(Var::continuous());
    /// m.add_constraint(x.leq(4.0));
    /// let program = m.build()?;
    /// assert_eq!(Basis::slack(&program).check(&program), Ok(()));
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    #[must_use]
    pub fn slack(program: &Program) -> Self {
        let status = |line: Line| {
            let bounds = line.bounds(program);
            match (bounds.lower().is_finite(), bounds.upper().is_finite()) {
                (true, _) => BasisStatus::Lower,
                (false, true) => BasisStatus::Upper,
                (false, false) => BasisStatus::Zero,
            }
        };
        Self {
            cols: program.cols().map(|col| status(col.into())).collect(),
            rows: program.rows().map(|_| BasisStatus::Basic).collect(),
        }
    }

    /// Checks that the basis fits `program`: one status per column and per row, as many basic lines as rows, and
    /// no line at a bound it does not have.
    ///
    /// # Errors
    ///
    /// The first [`Misfit`] found, in that order.
    pub fn check(&self, program: &Program) -> Result<(), Misfit> {
        Table::BasisCols.check(self.cols.len(), program)?;
        Table::BasisRows.check(self.rows.len(), program)?;
        let basic = self.lines().filter(|(_, s)| *s == BasisStatus::Basic).count();
        if basic != program.num_rows() {
            return Err(Misfit::BasicCount {
                expected: program.num_rows(),
                found: basic,
            });
        }
        match self
            .lines()
            .find(|&(line, status)| !at_own_bound(program, line, status))
        {
            Some((line, status)) => Err(Misfit::Status { line, status }),
            None => Ok(()),
        }
    }

    /// Every line with its status, columns first.
    fn lines(&self) -> impl Iterator<Item = (Line, BasisStatus)> + '_ {
        let cols = self.cols.iter_enumerated().map(|(col, &s)| (Line::Col(col), s));
        cols.chain(self.rows.iter_enumerated().map(|(row, &s)| (Line::Row(row), s)))
    }
}

/// Whether `status` puts `line` at no bound, or at one of its finite bounds.
///
/// # Panics
///
/// When the line is not a column or row of `program`.
fn at_own_bound(program: &Program, line: Line, status: BasisStatus) -> bool {
    status
        .bound()
        .is_none_or(|bound| bound.of(line.bounds(program)).is_finite())
}
