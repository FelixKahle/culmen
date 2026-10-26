// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The features a program or a solve uses beyond its class, and the set of features a backend takes.

use core::fmt;
use core::iter::FusedIterator;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

use itertools::Itertools;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Something a program or a solve uses that not every backend takes, beyond its [`ModelClass`](super::ModelClass).
///
/// [`Program::features`](super::Program::features) computes the features a program uses; a [`Start`](crate::Start)
/// in a [`Request`](crate::Request) uses [`ProgramFeature::StartSolution`]. A backend declares the features it takes
/// in [`Backend::FEATURES`](crate::Backend::FEATURES): [`Solver::new`](crate::Solver::new) refuses a program using
/// any other with [`LoadError::UnsupportedFeature`](crate::LoadError::UnsupportedFeature), and
/// [`Solver::solve`](crate::Solver::solve) refuses a start alike with
/// [`SolveError::UnsupportedFeature`](crate::SolveError::UnsupportedFeature).
///
/// ```
/// use culmen_core::ProgramFeature;
/// assert_eq!(ProgramFeature::SemiContinuous.to_string(), "semi-continuous columns");
/// assert!(ProgramFeature::StartSolution.is_per_solve() && !ProgramFeature::SemiInteger.is_per_solve());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ProgramFeature {
    /// A column of [`Integrality::SemiContinuous`](super::Integrality::SemiContinuous).
    SemiContinuous,
    /// A column of [`Integrality::SemiInteger`](super::Integrality::SemiInteger).
    SemiInteger,
    /// More than one objective: [`Objectives::Lexicographic`](super::Objectives::Lexicographic) or
    /// [`Objectives::Blended`](super::Objectives::Blended) with a further objective.
    MultipleObjectives,
    /// A starting solution handed to the solver before a solve; a feature of a [`Request`](crate::Request), never of
    /// a program.
    StartSolution,
}

impl ProgramFeature {
    /// Every feature, in the order of their bits in [`ProgramFeatures`].
    const ALL: [Self; 4] = [
        Self::SemiContinuous,
        Self::SemiInteger,
        Self::MultipleObjectives,
        Self::StartSolution,
    ];

    /// Whether the feature belongs to one solve's [`Request`](crate::Request) rather than to the program.
    #[must_use]
    pub const fn is_per_solve(self) -> bool {
        matches!(self, Self::StartSolution)
    }

    /// The feature's bit in a [`ProgramFeatures`].
    const fn bit(self) -> u8 {
        1 << self as u8
    }

    /// The feature in words.
    const fn name(self) -> &'static str {
        match self {
            Self::SemiContinuous => "semi-continuous columns",
            Self::SemiInteger => "semi-integer columns",
            Self::MultipleObjectives => "multiple objectives",
            Self::StartSolution => "a start solution",
        }
    }
}

/// The feature in words, such as `semi-continuous columns`.
impl fmt::Display for ProgramFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of [`ProgramFeature`]s: what a program uses ([`Program::features`](super::Program::features)) or what a
/// backend takes ([`Backend::FEATURES`](crate::Backend::FEATURES)).
///
/// ```
/// use culmen_core::{ProgramFeature, ProgramFeatures};
///
/// const SEMI: ProgramFeatures = ProgramFeatures::of(&[ProgramFeature::SemiContinuous, ProgramFeature::SemiInteger]);
/// let features = SEMI | ProgramFeature::StartSolution.into();
/// assert!(features.contains(ProgramFeature::StartSolution));
/// assert!(!features.contains(ProgramFeature::MultipleObjectives));
/// assert_eq!(features & SEMI, SEMI);
/// assert_eq!(features.iter().collect::<ProgramFeatures>(), features);
/// assert_eq!(features.difference(SEMI).iter().collect::<Vec<_>>(), [ProgramFeature::StartSolution]);
/// assert_eq!(format!("{SEMI:?}"), "ProgramFeatures(SemiContinuous | SemiInteger)");
/// assert_eq!(SEMI.to_string(), "{semi-continuous columns, semi-integer columns}");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(from = "Vec<ProgramFeature>", into = "Vec<ProgramFeature>")
)]
pub struct ProgramFeatures(u8);

impl ProgramFeatures {
    /// No feature.
    pub const NONE: Self = Self(0);
    /// Every feature.
    pub const ALL: Self = Self::of(&ProgramFeature::ALL);

    /// The set of the given features.
    #[must_use]
    pub const fn of(features: &[ProgramFeature]) -> Self {
        let mut bits = 0;
        let mut i = 0;
        while i < features.len() {
            bits |= features[i].bit();
            i += 1;
        }
        Self(bits)
    }

    /// The features in either set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The features in both sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// The features in this set and not in `other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Whether `feature` is in the set.
    #[must_use]
    pub const fn contains(self, feature: ProgramFeature) -> bool {
        self.0 & feature.bit() != 0
    }

    /// Whether every feature of `other` is in the set.
    #[must_use]
    pub const fn contains_all(self, other: Self) -> bool {
        other.0 & !self.0 == 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The number of features in the set.
    #[must_use]
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Each feature of the set, in the order of [`ProgramFeature`]'s variants.
    pub const fn iter(self) -> ProgramFeaturesIter {
        ProgramFeaturesIter(self)
    }
}

impl From<ProgramFeature> for ProgramFeatures {
    fn from(feature: ProgramFeature) -> Self {
        Self(feature.bit())
    }
}

impl BitOr for ProgramFeatures {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for ProgramFeatures {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

impl BitAnd for ProgramFeatures {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        self.intersection(other)
    }
}

impl BitAndAssign for ProgramFeatures {
    fn bitand_assign(&mut self, other: Self) {
        *self = self.intersection(other);
    }
}

impl FromIterator<ProgramFeature> for ProgramFeatures {
    fn from_iter<I: IntoIterator<Item = ProgramFeature>>(iter: I) -> Self {
        iter.into_iter().map(Self::from).fold(Self::NONE, Self::union)
    }
}

impl Extend<ProgramFeature> for ProgramFeatures {
    fn extend<I: IntoIterator<Item = ProgramFeature>>(&mut self, iter: I) {
        *self = iter.into_iter().map(Self::from).fold(*self, Self::union);
    }
}

impl IntoIterator for ProgramFeatures {
    type Item = ProgramFeature;
    type IntoIter = ProgramFeaturesIter;

    fn into_iter(self) -> ProgramFeaturesIter {
        self.iter()
    }
}

impl From<Vec<ProgramFeature>> for ProgramFeatures {
    fn from(features: Vec<ProgramFeature>) -> Self {
        features.into_iter().collect()
    }
}

impl From<ProgramFeatures> for Vec<ProgramFeature> {
    fn from(features: ProgramFeatures) -> Self {
        features.iter().collect()
    }
}

/// `ProgramFeatures(SemiContinuous | StartSolution)`, each feature by its variant's name.
impl fmt::Debug for ProgramFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProgramFeatures({})",
            self.iter().map(|feature| format!("{feature:?}")).format(" | ")
        )
    }
}

/// `{semi-continuous columns, a start solution}`, each feature in words; `{}` for none.
impl fmt::Display for ProgramFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}}}", self.iter().format(", "))
    }
}

/// The features of a [`ProgramFeatures`] set, in the order of [`ProgramFeature`]'s variants; made by
/// [`ProgramFeatures::iter`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ProgramFeaturesIter(ProgramFeatures);

impl ProgramFeaturesIter {
    /// Removes `feature` from the remaining set and returns it.
    fn remove(&mut self, feature: ProgramFeature) -> ProgramFeature {
        (self.0).0 &= !feature.bit();
        feature
    }
}

impl Iterator for ProgramFeaturesIter {
    type Item = ProgramFeature;

    fn next(&mut self) -> Option<ProgramFeature> {
        let rest = self.0;
        ProgramFeature::ALL
            .into_iter()
            .find(|&f| rest.contains(f))
            .map(|f| self.remove(f))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }
}

impl DoubleEndedIterator for ProgramFeaturesIter {
    fn next_back(&mut self) -> Option<ProgramFeature> {
        let rest = self.0;
        ProgramFeature::ALL
            .into_iter()
            .rfind(|&f| rest.contains(f))
            .map(|f| self.remove(f))
    }
}

impl ExactSizeIterator for ProgramFeaturesIter {}

impl FusedIterator for ProgramFeaturesIter {}
