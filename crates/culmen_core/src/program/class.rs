// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The class of a model, and the set of classes a backend solves.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// What kind of program a [`Program`](super::Program) is: whether it has integer columns, and whether its objective has
/// a quadratic part. Its constraints are linear in every class.
///
/// [`Program::class`](super::Program::class) computes it, and a backend declares the classes it solves in
/// [`Backend::CLASSES`](crate::Backend::CLASSES). The enum is non-exhaustive: a class of quadratic constraints
/// would be a new variant, and a backend that does not declare it refuses it.
///
/// ```
/// use culmen_core::ModelClass;
/// assert_eq!(ModelClass::Miqp.to_string(), "MIQP");
/// assert!(ModelClass::Miqp.is_discrete() && ModelClass::Miqp.is_quadratic());
/// assert!(!ModelClass::Lp.is_discrete() && !ModelClass::Lp.is_quadratic());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ModelClass {
    /// A linear program: continuous columns, a linear objective.
    Lp,
    /// A mixed-integer linear program: some integer columns, a linear objective.
    Milp,
    /// A convex quadratic program: continuous columns, a convex quadratic objective.
    Qp,
    /// A mixed-integer convex quadratic program: some integer columns, a convex quadratic objective.
    Miqp,
}

impl ModelClass {
    /// Every class, in the order of their bits in [`ModelClasses`].
    const ALL: [Self; 4] = [Self::Lp, Self::Milp, Self::Qp, Self::Miqp];

    /// Whether the class has discrete columns: integer, semi-continuous or semi-integer ones, whose values are not
    /// one interval.
    #[must_use]
    pub const fn is_discrete(self) -> bool {
        matches!(self, Self::Milp | Self::Miqp)
    }

    /// Whether the class has a quadratic objective part.
    #[must_use]
    pub const fn is_quadratic(self) -> bool {
        matches!(self, Self::Qp | Self::Miqp)
    }

    /// The class with integer columns and the same objective.
    pub(crate) const fn with_integer(self) -> Self {
        match self {
            Self::Lp | Self::Milp => Self::Milp,
            Self::Qp | Self::Miqp => Self::Miqp,
        }
    }

    /// The class's bit in a [`ModelClasses`].
    const fn bit(self) -> u8 {
        1 << self as u8
    }

    /// The class's usual abbreviation.
    const fn name(self) -> &'static str {
        match self {
            Self::Lp => "LP",
            Self::Milp => "MILP",
            Self::Qp => "QP",
            Self::Miqp => "MIQP",
        }
    }
}

/// The class's usual abbreviation: `LP`, `MILP`, `QP` or `MIQP`.
impl fmt::Display for ModelClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of [`ModelClass`]es: what a [`Backend`](crate::Backend) solves ([`Backend::CLASSES`]).
///
/// [`Solver::new`](crate::Solver::new) loads a [`Program`](super::Program) only into a backend whose set contains its
/// class, and refuses any other with [`LoadError::UnsupportedClass`](crate::LoadError::UnsupportedClass).
///
/// [`Backend::CLASSES`]: crate::Backend::CLASSES
///
/// ```
/// use culmen_core::{ModelClass, ModelClasses};
///
/// const LINEAR: ModelClasses = ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]);
/// let classes = LINEAR | ModelClass::Qp;
/// assert!(classes.contains(ModelClass::Qp));
/// assert!(!classes.contains(ModelClass::Miqp));
/// assert_eq!(classes & LINEAR, LINEAR);
/// assert_eq!(classes.iter().collect::<ModelClasses>(), classes);
/// assert_eq!(format!("{classes:?}"), "ModelClasses(LP | MILP | QP)");
/// assert_eq!(classes.to_string(), "{LP, MILP, QP}");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(from = "Vec<ModelClass>", into = "Vec<ModelClass>")
)]
pub struct ModelClasses(u8);

/// The classes of a [`ModelClasses`] set, in the order of [`ModelClass`]'s variants; made by
/// [`ModelClasses::iter`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ModelClassesIter(ModelClasses);

crate::set::element_set!(ModelClasses, ModelClassesIter, ModelClass, u8, "classes", "{}");
