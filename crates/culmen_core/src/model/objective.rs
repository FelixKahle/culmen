// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A further objective of a model and how a model's objectives combine.

use super::expr::{LinExpr, Linear};
use crate::{Sense, Tolerance, Weight};

/// A further linear objective to add to a [`Model`](super::Model): its sense and expression.
///
/// Its rank comes with it when it is added: [`Model::add_stage`](super::Model::add_stage) adds it as a stage of a
/// lexicographic solve with a tolerance, [`Model::add_weighted`](super::Model::add_weighted) as a part of a blend
/// with a weight. It is checked only when the model is built.
///
/// ```
/// use culmen_core::{Model, Objective, Tolerance, Var, absolute_gap};
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous().upper(4));
/// let co2 = Objective::minimize(3.0 * x);
/// let id = m.add_stage(co2, Tolerance { absolute: Some(absolute_gap!(1.0)), relative: None });
/// assert_eq!(id.index(), 1);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct Objective {
    pub(super) sense: Sense,
    pub(super) expr: LinExpr,
}

impl Objective {
    /// `expr` optimized in `sense`.
    pub fn new(sense: Sense, expr: impl Linear) -> Self {
        Self {
            sense,
            expr: expr.into_expr(),
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
}

/// How a [`Model`](super::Model)'s objectives combine once it has a further one: the
/// [`Objectives`](crate::Objectives) variant its program gets, which the rank of its objectives decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Combination {
    /// One after another in the order added, the model's own objective first, each kept within its tolerance by
    /// the later ones: [`Objectives::Lexicographic`](crate::Objectives::Lexicographic), for stages.
    Lexicographic,
    /// Their weighted sum, signed by sense, minimized: [`Objectives::Blended`](crate::Objectives::Blended), for
    /// weighted objectives.
    Blended,
}

/// The rank of one objective of a model: a stage's tolerance or a blended objective's weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Rank {
    /// A stage of a lexicographic solve.
    Tolerance(Tolerance),
    /// A part of a blend.
    Weight(Weight),
}

impl Rank {
    /// The combination this rank belongs to.
    pub(super) const fn combination(self) -> Combination {
        match self {
            Self::Tolerance(_) => Combination::Lexicographic,
            Self::Weight(_) => Combination::Blended,
        }
    }
}
