// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The features a program uses beyond its class, and the set of features a backend takes.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Something a program uses that not every backend takes, beyond its [`ModelClass`](super::ModelClass).
///
/// [`Program::features`](super::Program::features) computes the features a program uses. A backend declares the
/// features it takes in [`Backend::FEATURES`](crate::Backend::FEATURES): [`Solver::new`](crate::Solver::new) refuses
/// a program using any other with [`LoadError::UnsupportedFeature`](crate::LoadError::UnsupportedFeature), and an
/// edit alike with [`EditError::UnsupportedFeature`](crate::EditError::UnsupportedFeature). A start is no feature: a
/// backend takes one by implementing [`Starts`](crate::Starts).
///
/// ```
/// use culmen_core::ProgramFeature;
/// assert_eq!(ProgramFeature::SemiContinuous.to_string(), "semi-continuous columns");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProgramFeature {
    /// A column of [`Integrality::SemiContinuous`](super::Integrality::SemiContinuous).
    SemiContinuous,
    /// A column of [`Integrality::SemiInteger`](super::Integrality::SemiInteger).
    SemiInteger,
    /// More than one objective: [`Objectives::Lexicographic`](super::Objectives::Lexicographic) or
    /// [`Objectives::Blended`](super::Objectives::Blended) with a further objective.
    MultipleObjectives,
}

impl ProgramFeature {
    /// Every feature, in the order of their bits in [`ProgramFeatures`].
    const ALL: [Self; 3] = [Self::SemiContinuous, Self::SemiInteger, Self::MultipleObjectives];

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
/// let features = SEMI | ProgramFeature::MultipleObjectives;
/// assert!(features.contains(ProgramFeature::MultipleObjectives));
/// assert!(!SEMI.contains(ProgramFeature::MultipleObjectives));
/// assert_eq!(features & SEMI, SEMI);
/// assert_eq!(features.iter().collect::<ProgramFeatures>(), features);
/// assert_eq!(features.difference(SEMI).iter().collect::<Vec<_>>(), [ProgramFeature::MultipleObjectives]);
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

/// The features of a [`ProgramFeatures`] set, in the order of [`ProgramFeature`]'s variants; made by
/// [`ProgramFeatures::iter`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ProgramFeaturesIter(ProgramFeatures);

crate::set::element_set!(
    ProgramFeatures,
    ProgramFeaturesIter,
    ProgramFeature,
    u8,
    "features",
    "{:?}"
);
