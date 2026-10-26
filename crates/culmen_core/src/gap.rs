// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked gaps between objective values: a solve's MIP gaps and a lexicographic objective's tolerance.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A relative gap between two objective values: a finite number, zero or above.
///
/// It is a solve's relative MIP gap in [`Settings`](crate::Settings) and a lexicographic objective's relative
/// allowance in a [`Tolerance`](crate::Tolerance).
///
/// Literals are written with [`relative_gap!`](crate::relative_gap) and checked when the code is compiled; numbers
/// read at run time are checked by [`RelativeGap::new`] or `TryFrom<f64>`.
///
/// ```
/// use culmen_core::{InvalidRelativeGap, RelativeGap, relative_gap};
/// assert_eq!(relative_gap!(0.01).to_string(), "RelativeGap(0.01)");
/// assert_eq!(RelativeGap::try_from(-0.5), Err(InvalidRelativeGap::Negative));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "f64", into = "f64")
)]
pub struct RelativeGap(f64);

impl RelativeGap {
    /// The gap `value`.
    ///
    /// # Errors
    ///
    /// [`InvalidRelativeGap`] when `value` is not finite or below zero.
    pub const fn new(value: f64) -> Result<Self, InvalidRelativeGap> {
        if !value.is_finite() {
            Err(InvalidRelativeGap::NotFinite)
        } else if value < 0.0 {
            Err(InvalidRelativeGap::Negative)
        } else {
            Ok(Self(value))
        }
    }

    /// The gap as a number.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// The gap of a literal, or a panic naming the defect: in a `const` context, a compile error.
    ///
    /// # Panics
    ///
    /// When [`RelativeGap::new`] refuses `value`.
    #[doc(hidden)]
    #[track_caller]
    #[must_use]
    pub const fn literal(value: f64) -> Self {
        match Self::new(value) {
            Ok(gap) => gap,
            Err(InvalidRelativeGap::NotFinite) => panic!("RelativeGap: not a finite number"),
            Err(InvalidRelativeGap::Negative) => panic!("RelativeGap: below zero"),
        }
    }
}

impl TryFrom<f64> for RelativeGap {
    type Error = InvalidRelativeGap;

    fn try_from(value: f64) -> Result<Self, InvalidRelativeGap> {
        Self::new(value)
    }
}

impl From<RelativeGap> for f64 {
    fn from(gap: RelativeGap) -> Self {
        gap.0
    }
}

impl fmt::Display for RelativeGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RelativeGap({})", self.0)
    }
}

/// Why [`RelativeGap::new`] refused a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvalidRelativeGap {
    /// The number is NaN or infinite.
    NotFinite,
    /// The number is below zero.
    Negative,
}

impl fmt::Display for InvalidRelativeGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite => write!(f, "RelativeGap: not a finite number"),
            Self::Negative => write!(f, "RelativeGap: below zero"),
        }
    }
}

impl core::error::Error for InvalidRelativeGap {}

/// An absolute gap between two objective values, in objective units: a finite number, zero or above.
///
/// It is a solve's absolute MIP gap in [`Settings`](crate::Settings) and a lexicographic objective's absolute
/// allowance in a [`Tolerance`](crate::Tolerance).
///
/// Literals are written with [`absolute_gap!`](crate::absolute_gap) and checked when the code is compiled; numbers
/// read at run time are checked by [`AbsoluteGap::new`] or `TryFrom<f64>`.
///
/// ```
/// use culmen_core::{AbsoluteGap, InvalidAbsoluteGap, absolute_gap};
/// assert_eq!(absolute_gap!(2.5).to_string(), "AbsoluteGap(2.5)");
/// assert_eq!(AbsoluteGap::try_from(f64::NAN), Err(InvalidAbsoluteGap::NotFinite));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "f64", into = "f64")
)]
pub struct AbsoluteGap(f64);

impl AbsoluteGap {
    /// The gap `value`, in objective units.
    ///
    /// # Errors
    ///
    /// [`InvalidAbsoluteGap`] when `value` is not finite or below zero.
    pub const fn new(value: f64) -> Result<Self, InvalidAbsoluteGap> {
        if !value.is_finite() {
            Err(InvalidAbsoluteGap::NotFinite)
        } else if value < 0.0 {
            Err(InvalidAbsoluteGap::Negative)
        } else {
            Ok(Self(value))
        }
    }

    /// The gap as a number, in objective units.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// The gap of a literal, or a panic naming the defect: in a `const` context, a compile error.
    ///
    /// # Panics
    ///
    /// When [`AbsoluteGap::new`] refuses `value`.
    #[doc(hidden)]
    #[track_caller]
    #[must_use]
    pub const fn literal(value: f64) -> Self {
        match Self::new(value) {
            Ok(gap) => gap,
            Err(InvalidAbsoluteGap::NotFinite) => panic!("AbsoluteGap: not a finite number"),
            Err(InvalidAbsoluteGap::Negative) => panic!("AbsoluteGap: below zero"),
        }
    }
}

impl TryFrom<f64> for AbsoluteGap {
    type Error = InvalidAbsoluteGap;

    fn try_from(value: f64) -> Result<Self, InvalidAbsoluteGap> {
        Self::new(value)
    }
}

impl From<AbsoluteGap> for f64 {
    fn from(gap: AbsoluteGap) -> Self {
        gap.0
    }
}

impl fmt::Display for AbsoluteGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AbsoluteGap({})", self.0)
    }
}

/// Why [`AbsoluteGap::new`] refused a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvalidAbsoluteGap {
    /// The number is NaN or infinite.
    NotFinite,
    /// The number is below zero.
    Negative,
}

impl fmt::Display for InvalidAbsoluteGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite => write!(f, "AbsoluteGap: not a finite number"),
            Self::Negative => write!(f, "AbsoluteGap: below zero"),
        }
    }
}

impl core::error::Error for InvalidAbsoluteGap {}

/// A literal [`RelativeGap`], checked when the code is compiled.
///
/// ```
/// let gap = culmen_core::relative_gap!(0.005);
/// assert_eq!(gap.get(), 0.005);
/// ```
#[macro_export]
macro_rules! relative_gap {
    ($value:expr $(,)?) => {
        const { $crate::RelativeGap::literal($value) }
    };
}

/// A literal [`AbsoluteGap`], checked when the code is compiled.
///
/// ```
/// let gap = culmen_core::absolute_gap!(1.0);
/// assert_eq!(gap.get(), 1.0);
/// ```
#[macro_export]
macro_rules! absolute_gap {
    ($value:expr $(,)?) => {
        const { $crate::AbsoluteGap::literal($value) }
    };
}
