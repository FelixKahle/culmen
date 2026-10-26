// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A further objective of a model and how a model's objectives combine.

use super::expr::{LinExpr, Linear};
use crate::{Sense, Tolerance, Weight};

/// A further linear objective to add to a [`Model`](super::Model): its sense, expression, and its rank in either
/// combination.
///
/// Built by chaining, and checked only when the model is built. The model's [`Combination`] picks which of the two
/// ranks counts: the tolerance in a lexicographic solve, the weight in a blend.
///
/// ```
/// use culmen_core::{Model, Objective, Tolerance, Var, absolute_gap, weight};
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous().upper(4));
/// let co2 = Objective::minimize(3.0 * x)
///     .tolerance(Tolerance { absolute: Some(absolute_gap!(1.0)), relative: None })
///     .weight(weight!(0.5));
/// let id = m.add_objective(co2);
/// assert_eq!(id.index(), 1);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct Objective {
    pub(super) sense: Sense,
    pub(super) expr: LinExpr,
    pub(super) tolerance: Tolerance,
    pub(super) weight: Weight,
}

impl Objective {
    /// `expr` optimized in `sense`, with [`Tolerance::EXACT`] and [`Weight::ONE`].
    pub fn new(sense: Sense, expr: impl Linear) -> Self {
        Self {
            sense,
            expr: expr.into_expr(),
            tolerance: Tolerance::EXACT,
            weight: Weight::ONE,
        }
    }

    /// `expr` minimized.
    pub fn minimize(expr: impl Linear) -> Self {
        Self::new(Sense::Minimize, expr)
    }

    /// `expr` maximized.
    pub fn maximize(expr: impl Linear) -> Self {
        Self::new(Sense::Maximize, expr)
    }

    /// How far later stages of a lexicographic solve may move this objective from its optimum.
    pub const fn tolerance(mut self, tolerance: Tolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// The objective's weight in a blend.
    pub const fn weight(mut self, weight: Weight) -> Self {
        self.weight = weight;
        self
    }
}

/// How a [`Model`](super::Model)'s objectives combine once it has a further one: the
/// [`Objectives`](crate::Objectives) variant its program gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Combination {
    /// One after another in the order added, the model's own objective first, each kept within its tolerance by
    /// the later ones: [`Objectives::Lexicographic`](crate::Objectives::Lexicographic).
    #[default]
    Lexicographic,
    /// Their weighted sum, signed by sense, minimized: [`Objectives::Blended`](crate::Objectives::Blended).
    Blended,
}
