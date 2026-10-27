// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked intervals, integrality and the objective sense.

use core::fmt;
use core::ops::{RangeFrom, RangeFull, RangeInclusive, RangeToInclusive};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A closed interval `lower ..= upper` a column or a row activity must lie in.
///
/// Either end may be infinite, and the interval is never empty: neither end is NaN, `lower <= upper`, the lower end
/// is never `+∞` and the upper end never `-∞`. A literal interval is written with [`bounds!`](crate::bounds), which
/// checks it when the code is compiled; [`Bounds::new`] and the `TryFrom` conversions from `lower..=upper`,
/// `lower..` and `..=upper` check numbers read at run time, and `..` converts into [`Bounds::FREE`].
///
/// ```
/// use culmen_core::{Bounds, bounds};
///
/// const STOCK: Bounds = bounds!(0.0, 40.0);
/// assert_eq!((STOCK.lower(), STOCK.upper()), (0.0, 40.0));
/// assert!(Bounds::new(2.0, 1.0).is_err());
/// assert!(Bounds::at_least(0.0).unwrap().contains(1e300));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "BoundsForm", into = "BoundsForm")
)]
pub struct Bounds {
    lower: f64,
    upper: f64,
}

impl Bounds {
    /// Every real number.
    pub const FREE: Self = Self {
        lower: f64::NEG_INFINITY,
        upper: f64::INFINITY,
    };
    /// Zero and above.
    pub const NON_NEGATIVE: Self = Self {
        lower: 0.0,
        upper: f64::INFINITY,
    };
    /// Zero to one: with integrality, a binary column.
    pub const BINARY: Self = Self { lower: 0.0, upper: 1.0 };

    /// The interval `lower ..= upper`.
    ///
    /// # Errors
    ///
    /// [`InvalidBounds`] when an end is NaN, the lower end is `+∞`, the upper end is `-∞`, or `lower > upper`.
    pub const fn new(lower: f64, upper: f64) -> Result<Self, InvalidBounds> {
        if lower.is_nan() || upper.is_nan() {
            Err(InvalidBounds::NotANumber)
        } else if lower == f64::INFINITY {
            Err(InvalidBounds::LowerIsPlusInfinity)
        } else if upper == f64::NEG_INFINITY {
            Err(InvalidBounds::UpperIsMinusInfinity)
        } else if lower > upper {
            Err(InvalidBounds::Empty { lower, upper })
        } else {
            Ok(Self { lower, upper })
        }
    }

    /// The single value `value ..= value`.
    ///
    /// # Errors
    ///
    /// [`InvalidBounds`] when `value` is NaN or infinite.
    pub const fn fixed(value: f64) -> Result<Self, InvalidBounds> {
        Self::new(value, value)
    }

    /// The interval `lower ..`.
    ///
    /// # Errors
    ///
    /// [`InvalidBounds`] when `lower` is NaN or `+∞`.
    pub const fn at_least(lower: f64) -> Result<Self, InvalidBounds> {
        Self::new(lower, f64::INFINITY)
    }

    /// The interval `..= upper`.
    ///
    /// # Errors
    ///
    /// [`InvalidBounds`] when `upper` is NaN or `-∞`.
    pub const fn at_most(upper: f64) -> Result<Self, InvalidBounds> {
        Self::new(f64::NEG_INFINITY, upper)
    }

    /// The lower end, `-∞` when there is none.
    #[must_use]
    pub const fn lower(self) -> f64 {
        self.lower
    }

    /// The upper end, `+∞` when there is none.
    #[must_use]
    pub const fn upper(self) -> f64 {
        self.upper
    }

    /// Whether `value` lies in the interval.
    #[must_use]
    pub const fn contains(self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }

    /// How far `value` lies outside the interval, zero inside it and infinite for NaN, which lies in no interval.
    #[must_use]
    pub const fn violation(self, value: f64) -> f64 {
        if value.is_nan() {
            f64::INFINITY
        } else if value < self.lower {
            self.lower - value
        } else if value > self.upper {
            value - self.upper
        } else {
            0.0
        }
    }

    /// The interval moved by `shift`.
    ///
    /// # Errors
    ///
    /// [`InvalidBounds`] when `shift` is not finite, or moving a finite end overflows it to an infinity.
    pub fn shifted(self, shift: f64) -> Result<Self, InvalidBounds> {
        if !shift.is_finite() {
            return Err(InvalidBounds::NonFiniteShift { shift });
        }
        let lower = self.lower + shift;
        let upper = self.upper + shift;
        if (self.lower.is_finite() && !lower.is_finite()) || (self.upper.is_finite() && !upper.is_finite()) {
            return Err(InvalidBounds::Overflow);
        }
        Self::new(lower, upper)
    }

    /// The bounds of a literal, or a panic naming the defect: in a `const` context, a compile error.
    ///
    /// # Panics
    ///
    /// When [`Bounds::new`] refuses the interval.
    #[doc(hidden)]
    #[track_caller]
    #[must_use]
    pub const fn literal(lower: f64, upper: f64) -> Self {
        match Self::new(lower, upper) {
            Ok(bounds) => bounds,
            Err(InvalidBounds::NotANumber) => panic!("Bounds: an end is NaN"),
            Err(InvalidBounds::LowerIsPlusInfinity) => panic!("Bounds: the lower end is +inf"),
            Err(InvalidBounds::UpperIsMinusInfinity) => panic!("Bounds: the upper end is -inf"),
            Err(InvalidBounds::Empty { .. }) => panic!("Bounds: the lower end exceeds the upper end"),
            Err(InvalidBounds::Overflow) => panic!("Bounds: an end overflows"),
            Err(InvalidBounds::NonFiniteShift { .. }) => panic!("Bounds: the shift is not finite"),
        }
    }
}

/// A literal [`Bounds`], checked when the code is compiled.
///
/// `bounds!(lower, upper)` is the interval `lower ..= upper`; either end may be `f64::INFINITY` or
/// `f64::NEG_INFINITY`. An invalid literal is a compile error.
///
/// ```
/// use culmen_core::bounds;
/// let b = bounds!(-1.0, f64::INFINITY);
/// assert!(b.contains(7.0));
/// ```
#[macro_export]
macro_rules! bounds {
    ($lower:expr, $upper:expr $(,)?) => {
        const { $crate::Bounds::literal($lower, $upper) }
    };
}

/// The interval `start ..= end`, as [`Bounds::new`] checks it.
///
/// ```
/// use culmen_core::{Bounds, InvalidBounds, bounds};
/// assert_eq!(Bounds::try_from(0.0..=40.0), Ok(bounds!(0.0, 40.0)));
/// assert_eq!(Bounds::try_from(2.0..=1.0), Err(InvalidBounds::Empty { lower: 2.0, upper: 1.0 }));
/// ```
impl TryFrom<RangeInclusive<f64>> for Bounds {
    type Error = InvalidBounds;

    fn try_from(range: RangeInclusive<f64>) -> Result<Self, InvalidBounds> {
        Self::new(*range.start(), *range.end())
    }
}

/// The interval `start ..`, as [`Bounds::at_least`] checks it.
///
/// ```
/// use culmen_core::Bounds;
/// assert_eq!(Bounds::try_from(0.0..), Ok(Bounds::NON_NEGATIVE));
/// assert!(Bounds::try_from(f64::NAN..).is_err());
/// ```
impl TryFrom<RangeFrom<f64>> for Bounds {
    type Error = InvalidBounds;

    fn try_from(range: RangeFrom<f64>) -> Result<Self, InvalidBounds> {
        Self::at_least(range.start)
    }
}

/// The interval `..= end`, as [`Bounds::at_most`] checks it.
///
/// ```
/// use culmen_core::{Bounds, bounds};
/// assert_eq!(Bounds::try_from(..=1.0), Ok(bounds!(f64::NEG_INFINITY, 1.0)));
/// assert!(Bounds::try_from(..=f64::NEG_INFINITY).is_err());
/// ```
impl TryFrom<RangeToInclusive<f64>> for Bounds {
    type Error = InvalidBounds;

    fn try_from(range: RangeToInclusive<f64>) -> Result<Self, InvalidBounds> {
        Self::at_most(range.end)
    }
}

/// [`Bounds::FREE`].
///
/// ```
/// use culmen_core::Bounds;
/// assert_eq!(Bounds::from(..), Bounds::FREE);
/// ```
impl From<RangeFull> for Bounds {
    fn from(_: RangeFull) -> Self {
        Self::FREE
    }
}

/// The interval in mathematical notation, an infinite end open; the formatter's options apply to each finite end.
///
/// ```
/// use culmen_core::{Bounds, bounds};
/// assert_eq!(bounds!(0.0, 40.0).to_string(), "[0, 40]");
/// assert_eq!(Bounds::NON_NEGATIVE.to_string(), "[0, +inf)");
/// assert_eq!(bounds!(f64::NEG_INFINITY, 1.0).to_string(), "(-inf, 1]");
/// assert_eq!(Bounds::FREE.to_string(), "(-inf, +inf)");
/// assert_eq!(format!("{:.1}", bounds!(0.5, 2.0)), "[0.5, 2.0]");
/// ```
impl fmt::Display for Bounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.lower.is_finite() {
            f.write_str("[")?;
            fmt::Display::fmt(&self.lower, f)?;
        } else {
            f.write_str("(-inf")?;
        }
        f.write_str(", ")?;
        if self.upper.is_finite() {
            fmt::Display::fmt(&self.upper, f)?;
            f.write_str("]")
        } else {
            f.write_str("+inf)")
        }
    }
}

/// Why [`Bounds::new`] refused an interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InvalidBounds {
    /// An end is NaN.
    NotANumber,
    /// The lower end is `+∞`.
    LowerIsPlusInfinity,
    /// The upper end is `-∞`.
    UpperIsMinusInfinity,
    /// The lower end exceeds the upper end.
    Empty {
        /// The lower end given.
        lower: f64,
        /// The upper end given.
        upper: f64,
    },
    /// Shifting a finite end overflowed it to an infinity.
    Overflow,
    /// The shift is NaN or infinite.
    NonFiniteShift {
        /// The shift given.
        shift: f64,
    },
}

impl fmt::Display for InvalidBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotANumber => write!(f, "Bounds: an end is NaN"),
            Self::LowerIsPlusInfinity => write!(f, "Bounds: the lower end is +inf"),
            Self::UpperIsMinusInfinity => write!(f, "Bounds: the upper end is -inf"),
            Self::Empty { lower, upper } => write!(f, "Bounds: the lower end {lower} exceeds the upper end {upper}"),
            Self::Overflow => write!(f, "Bounds: shifting a finite end overflowed it"),
            Self::NonFiniteShift { shift } => write!(f, "Bounds: the shift {shift} is not finite"),
        }
    }
}

impl core::error::Error for InvalidBounds {}

/// The serialized form of [`Bounds`]: an absent end is infinite, since JSON has no infinity.
#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct BoundsForm {
    lower: Option<f64>,
    upper: Option<f64>,
}

#[cfg(feature = "serde")]
impl From<Bounds> for BoundsForm {
    fn from(bounds: Bounds) -> Self {
        Self {
            lower: bounds.lower.is_finite().then_some(bounds.lower),
            upper: bounds.upper.is_finite().then_some(bounds.upper),
        }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<BoundsForm> for Bounds {
    type Error = InvalidBounds;

    fn try_from(form: BoundsForm) -> Result<Self, InvalidBounds> {
        Self::new(
            form.lower.unwrap_or(f64::NEG_INFINITY),
            form.upper.unwrap_or(f64::INFINITY),
        )
    }
}

/// Which values in its bounds a column takes: any, only whole ones, or, for a semi-continuous or semi-integer
/// column, zero as well.
///
/// A binary column is an integer column with [`Bounds::BINARY`]; the program stores no third kind, since the bounds
/// already say it. A semi column is zero or in its bounds, which a [`Program`](super::Program) holds only when they
/// are `lower ..= upper` with `0 < lower` and `upper` finite, so zero lies outside them.
///
/// ```
/// use culmen_core::Integrality;
/// assert!(Integrality::SemiInteger.is_integral() && Integrality::SemiInteger.is_semi());
/// assert!(!Integrality::Continuous.is_integral() && !Integrality::Integer.is_semi());
/// assert!(Integrality::SemiContinuous.is_discrete() && !Integrality::Continuous.is_discrete());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Integrality {
    /// Any value in the bounds.
    #[default]
    Continuous,
    /// Only whole values in the bounds.
    Integer,
    /// Zero, or any value in the bounds.
    SemiContinuous,
    /// Zero, or a whole value in the bounds.
    SemiInteger,
}

impl Integrality {
    /// Whether the column's nonzero values are whole: [`Integrality::Integer`] or [`Integrality::SemiInteger`].
    #[must_use]
    pub const fn is_integral(self) -> bool {
        matches!(self, Self::Integer | Self::SemiInteger)
    }

    /// Whether the column may also be zero outside its bounds: [`Integrality::SemiContinuous`] or
    /// [`Integrality::SemiInteger`].
    #[must_use]
    pub const fn is_semi(self) -> bool {
        matches!(self, Self::SemiContinuous | Self::SemiInteger)
    }

    /// Whether the column's values are not one interval, so a program with it is mixed-integer: anything but
    /// [`Integrality::Continuous`].
    #[must_use]
    pub const fn is_discrete(self) -> bool {
        !matches!(self, Self::Continuous)
    }

    /// How far `value` lies from the values a column of this integrality in `bounds` takes, zero for one of them and
    /// infinite for NaN.
    ///
    /// ```
    /// use culmen_core::{Integrality, bounds};
    /// let semi = Integrality::SemiInteger;
    /// assert_eq!(semi.violation(bounds!(2.0, 5.0), 0.0), 0.0);
    /// assert_eq!(semi.violation(bounds!(2.0, 5.0), 0.5), 0.5);
    /// assert_eq!(semi.violation(bounds!(2.0, 5.0), 3.25), 0.25);
    /// assert_eq!(Integrality::Integer.violation(bounds!(2.0, 5.0), 0.0), 2.0);
    /// ```
    #[must_use]
    pub fn violation(self, bounds: Bounds, value: f64) -> f64 {
        let fraction = (value - value.round()).abs();
        let inside = match self {
            Self::Continuous | Self::SemiContinuous => bounds.violation(value),
            Self::Integer | Self::SemiInteger => bounds.violation(value).max(fraction),
        };
        if self.is_semi() {
            inside.min(value.abs())
        } else {
            inside
        }
    }
}

/// Whether the objective is minimized or maximized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Sense {
    /// The least objective value is best.
    #[default]
    Minimize,
    /// The greatest objective value is best.
    Maximize,
}

impl Sense {
    /// Whether objective value `a` is strictly better than `b` under this sense.
    ///
    /// ```
    /// use culmen_core::Sense;
    /// assert!(Sense::Minimize.better(1.0, 2.0));
    /// assert!(Sense::Maximize.better(2.0, 1.0));
    /// ```
    #[must_use]
    pub const fn better(self, a: f64, b: f64) -> bool {
        match self {
            Self::Minimize => a < b,
            Self::Maximize => a > b,
        }
    }
}
