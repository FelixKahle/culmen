// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An irreducible infeasible subsystem.

use core::fmt;
use std::collections::BTreeMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::Bound;
use crate::{Col, Row};

/// Which of a line's bounds belong to a subsystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Side {
    /// The lower bound alone.
    Lower,
    /// The upper bound alone.
    Upper,
    /// Both bounds, as of an equality or a range.
    Both,
}

impl Side {
    /// Whether the side includes `bound`.
    #[must_use]
    pub const fn includes(self, bound: Bound) -> bool {
        matches!(
            (self, bound),
            (Self::Both, _) | (Self::Lower, Bound::Lower) | (Self::Upper, Bound::Upper)
        )
    }
}

/// `lower`, `upper` or `both`.
impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Lower => "lower",
            Self::Upper => "upper",
            Self::Both => "both",
        })
    }
}

/// An irreducible infeasible subsystem of a program's constraints: [`Solver::iis`](crate::Solver::iis).
///
/// The subsystem is the listed rows with the listed sides of their bounds, and the listed sides of the listed
/// columns' bounds; every other bound is dropped, every column stays continuous. It is infeasible, and dropping any
/// one entry makes it feasible. A column of the subsystem that has no bound in it is not listed.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Iis {
    /// The columns whose bounds belong to the subsystem, with which of them.
    pub cols: BTreeMap<Col, Side>,
    /// The rows of the subsystem, with which of their bounds.
    pub rows: BTreeMap<Row, Side>,
}
