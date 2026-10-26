// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The settings every backend understands, applied per solve.

use core::fmt;
use core::num::NonZeroUsize;
use core::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// What a solve may spend and when it may stop early, in terms every backend maps to its own parameters.
///
/// Plain data; `None` leaves a setting to the backend's own options, which in turn default to the solver's.
/// Settings given here take precedence over a backend's options for the one solve they are passed to.
///
/// ```
/// use core::time::Duration;
/// use culmen_core::{Settings, relative_gap, threads};
///
/// let settings = Settings {
///     time_limit: Some(Duration::from_secs(30)),
///     relative_gap: Some(relative_gap!(0.01)),
///     threads: Some(threads!(4)),
///     ..Settings::default()
/// };
/// assert_eq!(settings.relative_gap.unwrap().get(), 0.01);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Settings {
    /// The wall-clock time the solve may take.
    pub time_limit: Option<Duration>,
    /// The relative gap between the best solution and the best bound at which a MIP counts as solved.
    pub relative_gap: Option<RelativeGap>,
    /// The absolute gap, in objective units, at which a MIP counts as solved.
    pub absolute_gap: Option<AbsoluteGap>,
    /// The number of threads the solver may use.
    pub threads: Option<NonZeroUsize>,
    /// The seed of the solver's random choices, for reproducible runs.
    pub seed: Option<u32>,
    /// Whether the solver presolves the problem first.
    pub presolve: Option<bool>,
    /// The number of branch-and-bound nodes after which a MIP stops.
    pub node_limit: Option<u64>,
    /// The number of feasible solutions after which a MIP stops.
    pub solution_limit: Option<u64>,
    /// Whether the solver also prints its log to standard output; the log reaches callbacks either way.
    pub console: bool,
}

impl Settings {
    /// Every setting left to the backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            time_limit: None,
            relative_gap: None,
            absolute_gap: None,
            threads: None,
            seed: None,
            presolve: None,
            node_limit: None,
            solution_limit: None,
            console: false,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::new()
    }
}

/// A relative MIP gap: a finite number, zero or above.
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

/// An absolute MIP gap in objective units: a finite number, zero or above.
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

/// A literal thread count, a [`NonZeroUsize`] checked when the code is compiled.
///
/// ```
/// let four = culmen_core::threads!(4);
/// assert_eq!(four.get(), 4);
/// ```
#[macro_export]
macro_rules! threads {
    ($value:expr $(,)?) => {
        const {
            match ::core::num::NonZeroUsize::new($value) {
                ::core::option::Option::Some(n) => n,
                ::core::option::Option::None => panic!("threads: zero"),
            }
        }
    };
}
