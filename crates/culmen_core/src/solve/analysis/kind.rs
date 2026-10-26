// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kinds of analysis a solve can be followed by, and the set of them a backend provides.

use core::fmt;
use core::iter::FusedIterator;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

use itertools::Itertools;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{ModelClass, ModelClasses, ProgramFeature, ProgramFeatures};

/// What a [`Solver`](crate::Solver) can report beyond a solve's [`Outcome`](crate::Outcome).
///
/// A backend declares the analyses it provides in [`Backend::ANALYSES`](crate::Backend::ANALYSES); each is defined
/// only for the [`classes`](Analysis::classes) of program where its mathematics holds. The enum is non-exhaustive: a
/// new analysis is a new variant, and a backend that does not declare it refuses it.
///
/// ```
/// use culmen_core::{Analysis, ModelClass};
/// assert_eq!(Analysis::Iis.to_string(), "IIS");
/// assert!(Analysis::Duals.classes().contains(ModelClass::Qp));
/// assert!(!Analysis::Duals.classes().contains(ModelClass::Milp), "a MIP has no duals");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Analysis {
    /// Row duals and reduced costs of an optimal continuous program: [`Solver::duals`](crate::Solver::duals).
    Duals,
    /// The simplex basis, read after a solve and set before the next: [`Solver::basis`](crate::Solver::basis),
    /// [`Solver::set_basis`](crate::Solver::set_basis).
    Basis,
    /// Cost and bound ranging of an optimal basis: [`Solver::ranging`](crate::Solver::ranging).
    Ranging,
    /// A primal ray of an unbounded program and a dual ray of an infeasible one:
    /// [`Solver::primal_ray`](crate::Solver::primal_ray), [`Solver::dual_ray`](crate::Solver::dual_ray).
    Rays,
    /// An irreducible infeasible subsystem: [`Solver::iis`](crate::Solver::iis).
    Iis,
    /// The least weighted violation of bounds and rows:
    /// [`Solver::feasibility_relaxation`](crate::Solver::feasibility_relaxation).
    FeasibilityRelaxation,
}

impl Analysis {
    /// Every analysis, in the order of their bits in [`Analyses`].
    const ALL: [Self; 6] = [
        Self::Duals,
        Self::Basis,
        Self::Ranging,
        Self::Rays,
        Self::Iis,
        Self::FeasibilityRelaxation,
    ];

    /// The classes of program the analysis is defined for.
    ///
    /// Duals exist for continuous programs; a basis, ranging and rays are simplex notions of a linear program; an
    /// IIS is a subsystem of linear constraints over continuous columns, so a QP has one exactly when its
    /// constraints do; a feasibility relaxation minimizes a linear violation under the program's integrality.
    #[must_use]
    pub const fn classes(self) -> ModelClasses {
        match self {
            Self::Duals | Self::Iis => ModelClasses::of(&[ModelClass::Lp, ModelClass::Qp]),
            Self::Basis | Self::Ranging | Self::Rays => ModelClasses::of(&[ModelClass::Lp]),
            Self::FeasibilityRelaxation => ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]),
        }
    }

    /// The features of a program the analysis is not defined for.
    ///
    /// Duals, a basis, ranging and rays belong to one objective, so a program with several has none; HiGHS, for one,
    /// leaves them invalid after a multi-objective solve. Semi columns need no entry: they make a program a MILP or a
    /// MIQP, which [`Analysis::classes`] already decides.
    #[must_use]
    pub const fn refused(self) -> ProgramFeatures {
        match self {
            Self::Duals | Self::Basis | Self::Ranging | Self::Rays => {
                ProgramFeatures::of(&[ProgramFeature::MultipleObjectives])
            }
            Self::Iis | Self::FeasibilityRelaxation => ProgramFeatures::NONE,
        }
    }

    /// The analysis's bit in an [`Analyses`].
    const fn bit(self) -> u8 {
        1 << self as u8
    }

    /// `singular` or `plural`, whichever agrees with the analysis's name as a subject.
    pub(super) const fn agree(self, singular: &'static str, plural: &'static str) -> &'static str {
        match self {
            Self::Duals | Self::Rays => plural,
            Self::Basis | Self::Ranging | Self::Iis | Self::FeasibilityRelaxation => singular,
        }
    }

    /// The analysis's name as text.
    const fn name(self) -> &'static str {
        match self {
            Self::Duals => "duals",
            Self::Basis => "basis",
            Self::Ranging => "ranging",
            Self::Rays => "rays",
            Self::Iis => "IIS",
            Self::FeasibilityRelaxation => "feasibility relaxation",
        }
    }
}

/// The analysis's name: `duals`, `basis`, `ranging`, `rays`, `IIS` or `feasibility relaxation`.
impl fmt::Display for Analysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of [`Analysis`] kinds: what a [`Backend`](crate::Backend) provides ([`Backend::ANALYSES`]).
///
/// A [`Solver`](crate::Solver) asks its backend only for an analysis in the set, and refuses any other with
/// [`AnalysisError::Unsupported`](crate::AnalysisError::Unsupported).
///
/// [`Backend::ANALYSES`]: crate::Backend::ANALYSES
///
/// ```
/// use culmen_core::{Analyses, Analysis};
///
/// const LP: Analyses = Analyses::of(&[Analysis::Duals, Analysis::Basis]);
/// let analyses = LP | Analysis::Iis.into();
/// assert!(analyses.contains(Analysis::Iis));
/// assert!(!analyses.contains(Analysis::Rays));
/// assert_eq!(analyses & LP, LP);
/// assert_eq!(analyses.iter().collect::<Analyses>(), analyses);
/// assert_eq!(format!("{analyses:?}"), "Analyses(duals | basis | IIS)");
/// assert_eq!(analyses.to_string(), "{duals, basis, IIS}");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(from = "Vec<Analysis>", into = "Vec<Analysis>")
)]
pub struct Analyses(u8);

impl Analyses {
    /// No analysis.
    pub const NONE: Self = Self(0);
    /// Every analysis.
    pub const ALL: Self = Self::of(&Analysis::ALL);

    /// The set of the given analyses.
    #[must_use]
    pub const fn of(analyses: &[Analysis]) -> Self {
        let mut bits = 0;
        let mut i = 0;
        while i < analyses.len() {
            bits |= analyses[i].bit();
            i += 1;
        }
        Self(bits)
    }

    /// The analyses in either set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The analyses in both sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Whether `analysis` is in the set.
    #[must_use]
    pub const fn contains(self, analysis: Analysis) -> bool {
        self.0 & analysis.bit() != 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The number of analyses in the set.
    #[must_use]
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Each analysis of the set, in the order of [`Analysis`]'s variants.
    pub const fn iter(self) -> AnalysesIter {
        AnalysesIter(self)
    }
}

impl From<Analysis> for Analyses {
    fn from(analysis: Analysis) -> Self {
        Self(analysis.bit())
    }
}

impl BitOr for Analyses {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for Analyses {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

impl BitAnd for Analyses {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        self.intersection(other)
    }
}

impl BitAndAssign for Analyses {
    fn bitand_assign(&mut self, other: Self) {
        *self = self.intersection(other);
    }
}

impl FromIterator<Analysis> for Analyses {
    fn from_iter<I: IntoIterator<Item = Analysis>>(iter: I) -> Self {
        iter.into_iter().map(Self::from).fold(Self::NONE, Self::union)
    }
}

impl Extend<Analysis> for Analyses {
    fn extend<I: IntoIterator<Item = Analysis>>(&mut self, iter: I) {
        *self = iter.into_iter().map(Self::from).fold(*self, Self::union);
    }
}

impl IntoIterator for Analyses {
    type Item = Analysis;
    type IntoIter = AnalysesIter;

    fn into_iter(self) -> AnalysesIter {
        self.iter()
    }
}

impl From<Vec<Analysis>> for Analyses {
    fn from(analyses: Vec<Analysis>) -> Self {
        analyses.into_iter().collect()
    }
}

impl From<Analyses> for Vec<Analysis> {
    fn from(analyses: Analyses) -> Self {
        analyses.iter().collect()
    }
}

/// `Analyses(duals | IIS)`, each analysis by its name.
impl fmt::Debug for Analyses {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Analyses({})", self.iter().format(" | "))
    }
}

/// `{duals, IIS}`, each analysis by its name; `{}` for none.
impl fmt::Display for Analyses {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}}}", self.iter().format(", "))
    }
}

/// The analyses of an [`Analyses`] set, in the order of [`Analysis`]'s variants; made by [`Analyses::iter`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct AnalysesIter(Analyses);

impl AnalysesIter {
    /// Removes `analysis` from the remaining set and returns it.
    fn remove(&mut self, analysis: Analysis) -> Analysis {
        (self.0).0 &= !analysis.bit();
        analysis
    }
}

impl Iterator for AnalysesIter {
    type Item = Analysis;

    fn next(&mut self) -> Option<Analysis> {
        let rest = self.0;
        Analysis::ALL
            .into_iter()
            .find(|&a| rest.contains(a))
            .map(|a| self.remove(a))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }
}

impl DoubleEndedIterator for AnalysesIter {
    fn next_back(&mut self) -> Option<Analysis> {
        let rest = self.0;
        Analysis::ALL
            .into_iter()
            .rfind(|&a| rest.contains(a))
            .map(|a| self.remove(a))
    }
}

impl ExactSizeIterator for AnalysesIter {}

impl FusedIterator for AnalysesIter {}
