// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kinds of analysis a solve can be followed by.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{ModelClass, ModelClasses, ProgramFeature, ProgramFeatures};

/// What a [`Solver`](crate::Solver) can report beyond a solve's [`Outcome`](crate::Outcome).
///
/// A backend provides an analysis by implementing its trait, such as [`DualAnalysis`](crate::DualAnalysis); each
/// analysis is defined only for the [`classes`](Analysis::classes) of program where its mathematics holds. It names the
/// analysis in an [`AnalysisError`](crate::AnalysisError). The enum is non-exhaustive: a new analysis is a new variant.
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
