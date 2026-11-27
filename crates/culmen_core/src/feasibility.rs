// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A checked tolerance of a solver's numerical tests: how far a solution may violate a constraint, or a reduced cost
//! have the wrong sign, and still count.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A tolerance of a solver's feasibility or optimality test: a number from [`FeasibilityTolerance::MIN`] to
/// [`FeasibilityTolerance::MAX`], the range every backend takes.
///
/// It is a solve's [`Settings::feasibility_tolerance`](crate::Settings::feasibility_tolerance) and
/// [`Settings::optimality_tolerance`](crate::Settings::optimality_tolerance). Literals are written with
/// [`feasibility_tolerance!`](crate::feasibility_tolerance) and checked when the code is compiled; numbers read at run
/// time are checked by [`FeasibilityTolerance::new`] or `TryFrom<f64>`.
///
/// ```
/// use culmen_core::{FeasibilityTolerance, InvalidFeasibilityTolerance, feasibility_tolerance};
/// assert_eq!(feasibility_tolerance!(1e-6).to_string(), "FeasibilityTolerance(0.000001)");
/// assert_eq!(FeasibilityTolerance::try_from(0.1), Err(InvalidFeasibilityTolerance::AboveMaximum));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "f64", into = "f64")
)]
pub struct FeasibilityTolerance(f64);

impl FeasibilityTolerance {
    /// The least tolerance: HiGHS's least, `1e-10`.
    pub const MIN: f64 = 1e-10;

    /// The greatest tolerance: SCIP's greatest, `1e-3`.
    pub const MAX: f64 = 1e-3;

    /// The tolerance `value`.
    ///
    /// # Errors
    ///
    /// [`InvalidFeasibilityTolerance`] when `value` is NaN or lies outside [`FeasibilityTolerance::MIN`] to
    /// [`FeasibilityTolerance::MAX`].
    pub const fn new(value: f64) -> Result<Self, InvalidFeasibilityTolerance> {
        if value.is_nan() {
            Err(InvalidFeasibilityTolerance::NotANumber)
        } else if value < Self::MIN {
            Err(InvalidFeasibilityTolerance::BelowMinimum)
        } else if value > Self::MAX {
            Err(InvalidFeasibilityTolerance::AboveMaximum)
        } else {
            Ok(Self(value))
        }
    }

    /// The tolerance as a number.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// The tolerance of a literal, or a panic naming the defect: in a `const` context, a compile error.
    ///
    /// # Panics
    ///
    /// When [`FeasibilityTolerance::new`] refuses `value`.
    #[doc(hidden)]
    #[track_caller]
    #[must_use]
    pub const fn literal(value: f64) -> Self {
        match Self::new(value) {
            Ok(tolerance) => tolerance,
            Err(InvalidFeasibilityTolerance::NotANumber) => panic!("FeasibilityTolerance: NaN"),
            Err(InvalidFeasibilityTolerance::BelowMinimum) => panic!("FeasibilityTolerance: below 1e-10"),
            Err(InvalidFeasibilityTolerance::AboveMaximum) => panic!("FeasibilityTolerance: above 1e-3"),
        }
    }
}

impl TryFrom<f64> for FeasibilityTolerance {
    type Error = InvalidFeasibilityTolerance;

    fn try_from(value: f64) -> Result<Self, InvalidFeasibilityTolerance> {
        Self::new(value)
    }
}

impl From<FeasibilityTolerance> for f64 {
    fn from(tolerance: FeasibilityTolerance) -> Self {
        tolerance.0
    }
}

impl fmt::Display for FeasibilityTolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FeasibilityTolerance({})", self.0)
    }
}

/// Why [`FeasibilityTolerance::new`] refused a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvalidFeasibilityTolerance {
    /// The number is NaN.
    NotANumber,
    /// The number is below [`FeasibilityTolerance::MIN`].
    BelowMinimum,
    /// The number is above [`FeasibilityTolerance::MAX`].
    AboveMaximum,
}

impl fmt::Display for InvalidFeasibilityTolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotANumber => write!(f, "FeasibilityTolerance: NaN"),
            Self::BelowMinimum => write!(f, "FeasibilityTolerance: below 1e-10"),
            Self::AboveMaximum => write!(f, "FeasibilityTolerance: above 1e-3"),
        }
    }
}

impl core::error::Error for InvalidFeasibilityTolerance {}

/// A literal [`FeasibilityTolerance`], checked when the code is compiled.
///
/// ```
/// use culmen_core::feasibility_tolerance;
/// assert_eq!(feasibility_tolerance!(1e-7).get(), 1e-7);
/// ```
///
/// ```compile_fail,E0080
/// let loose = culmen_core::feasibility_tolerance!(0.1); // FeasibilityTolerance: above 1e-3
/// ```
#[macro_export]
macro_rules! feasibility_tolerance {
    ($value:expr $(,)?) => {
        const { $crate::FeasibilityTolerance::literal($value) }
    };
}
