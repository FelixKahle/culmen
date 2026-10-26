// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The class of a model, and the set of classes a backend solves.

use core::fmt;
use core::iter::FusedIterator;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

use itertools::Itertools;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// What kind of program an [`Lp`](super::Lp) is: whether it has integer columns, and whether its objective has a
/// quadratic part. Its constraints are linear in every class.
///
/// [`Lp::class`](super::Lp::class) computes it, and a backend declares the classes it solves in
/// [`Backend::CLASSES`](crate::Backend::CLASSES). The enum is non-exhaustive: a class of quadratic constraints
/// would be a new variant, and a backend that does not declare it refuses it.
///
/// ```
/// use culmen_core::ModelClass;
/// assert_eq!(ModelClass::Miqp.to_string(), "MIQP");
/// assert!(ModelClass::Miqp.is_integer() && ModelClass::Miqp.is_quadratic());
/// assert!(!ModelClass::Lp.is_integer() && !ModelClass::Lp.is_quadratic());
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

    /// Whether the class has integer columns.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::Milp | Self::Miqp)
    }

    /// Whether the class has a quadratic objective part.
    #[must_use]
    pub const fn is_quadratic(self) -> bool {
        matches!(self, Self::Qp | Self::Miqp)
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
/// [`Solver::new`](crate::Solver::new) loads an [`Lp`](super::Lp) only into a backend whose set contains its
/// class, and refuses any other with [`LoadError::Unsupported`](crate::LoadError::Unsupported).
///
/// [`Backend::CLASSES`]: crate::Backend::CLASSES
///
/// ```
/// use culmen_core::{ModelClass, ModelClasses};
///
/// const LINEAR: ModelClasses = ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]);
/// let classes = LINEAR | ModelClass::Qp.into();
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

impl ModelClasses {
    /// No class.
    pub const NONE: Self = Self(0);
    /// Every class.
    pub const ALL: Self = Self::of(&ModelClass::ALL);

    /// The set of the given classes.
    #[must_use]
    pub const fn of(classes: &[ModelClass]) -> Self {
        let mut bits = 0;
        let mut i = 0;
        while i < classes.len() {
            bits |= classes[i].bit();
            i += 1;
        }
        Self(bits)
    }

    /// The classes in either set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The classes in both sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Whether `class` is in the set.
    #[must_use]
    pub const fn contains(self, class: ModelClass) -> bool {
        self.0 & class.bit() != 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The number of classes in the set.
    #[must_use]
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Each class of the set, in the order of [`ModelClass`]'s variants.
    pub const fn iter(self) -> ModelClassesIter {
        ModelClassesIter(self)
    }
}

impl From<ModelClass> for ModelClasses {
    fn from(class: ModelClass) -> Self {
        Self(class.bit())
    }
}

impl BitOr for ModelClasses {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for ModelClasses {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

impl BitAnd for ModelClasses {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        self.intersection(other)
    }
}

impl BitAndAssign for ModelClasses {
    fn bitand_assign(&mut self, other: Self) {
        *self = self.intersection(other);
    }
}

impl FromIterator<ModelClass> for ModelClasses {
    fn from_iter<I: IntoIterator<Item = ModelClass>>(iter: I) -> Self {
        iter.into_iter().map(Self::from).fold(Self::NONE, Self::union)
    }
}

impl Extend<ModelClass> for ModelClasses {
    fn extend<I: IntoIterator<Item = ModelClass>>(&mut self, iter: I) {
        *self = iter.into_iter().map(Self::from).fold(*self, Self::union);
    }
}

impl IntoIterator for ModelClasses {
    type Item = ModelClass;
    type IntoIter = ModelClassesIter;

    fn into_iter(self) -> ModelClassesIter {
        self.iter()
    }
}

impl From<Vec<ModelClass>> for ModelClasses {
    fn from(classes: Vec<ModelClass>) -> Self {
        classes.into_iter().collect()
    }
}

impl From<ModelClasses> for Vec<ModelClass> {
    fn from(classes: ModelClasses) -> Self {
        classes.iter().collect()
    }
}

/// `ModelClasses(LP | QP)`, each class by its abbreviation.
impl fmt::Debug for ModelClasses {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelClasses({})", self.iter().format(" | "))
    }
}

/// `{LP, QP}`, each class by its abbreviation; `{}` for none.
impl fmt::Display for ModelClasses {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}}}", self.iter().format(", "))
    }
}

/// The classes of a [`ModelClasses`] set, in the order of [`ModelClass`]'s variants; made by
/// [`ModelClasses::iter`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ModelClassesIter(ModelClasses);

impl ModelClassesIter {
    /// Removes `class` from the remaining set and returns it.
    fn remove(&mut self, class: ModelClass) -> ModelClass {
        (self.0).0 &= !class.bit();
        class
    }
}

impl Iterator for ModelClassesIter {
    type Item = ModelClass;

    fn next(&mut self) -> Option<ModelClass> {
        let rest = self.0;
        ModelClass::ALL
            .into_iter()
            .find(|&c| rest.contains(c))
            .map(|c| self.remove(c))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }
}

impl DoubleEndedIterator for ModelClassesIter {
    fn next_back(&mut self) -> Option<ModelClass> {
        let rest = self.0;
        ModelClass::ALL
            .into_iter()
            .rfind(|&c| rest.contains(c))
            .map(|c| self.remove(c))
    }
}

impl ExactSizeIterator for ModelClassesIter {}

impl FusedIterator for ModelClassesIter {}
