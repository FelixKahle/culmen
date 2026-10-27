// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Further linear objectives of a program and how they combine with its own: lexicographically or blended.

use core::fmt;
use core::iter::FusedIterator;
use core::slice;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::TiSlice;

use super::{Col, Sense, merge_terms};
use crate::{AbsoluteGap, RelativeGap};

/// An objective of a [`Program`](super::Program): [`ObjectiveId::FIRST`] is the program's own, and the further
/// objectives of its [`Objectives`] follow in order from `ObjectiveId::new(1)`.
///
/// A plain `Copy` index like [`Col`]; [`Model::add_objective`](crate::Model::add_objective) hands it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct ObjectiveId(usize);

impl ObjectiveId {
    /// The program's own objective: its sense, costs, quadratic part and offset.
    pub const FIRST: Self = Self(0);

    /// The objective at a position, counting from zero for the program's own.
    ///
    /// ```
    /// use culmen_core::ObjectiveId;
    /// assert_eq!(ObjectiveId::new(0), ObjectiveId::FIRST);
    /// assert_eq!(ObjectiveId::new(2).index(), 2);
    /// ```
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// The objective's position, counting from zero for the program's own.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for ObjectiveId {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl From<ObjectiveId> for usize {
    fn from(id: ObjectiveId) -> Self {
        id.0
    }
}

/// The weight of an objective in a blend: a finite number above zero.
///
/// Literals are written with [`weight!`](crate::weight) and checked when the code is compiled; numbers read at run
/// time are checked by [`Weight::new`] or `TryFrom<f64>`.
///
/// ```
/// use culmen_core::{InvalidWeight, Weight, weight};
/// assert_eq!(weight!(2.5).to_string(), "Weight(2.5)");
/// assert_eq!(Weight::try_from(0.0), Err(InvalidWeight::NotPositive));
/// assert_eq!(Weight::default(), weight!(1.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "f64", into = "f64")
)]
pub struct Weight(f64);

impl Weight {
    /// The weight one.
    pub const ONE: Self = Self(1.0);

    /// The weight `value`.
    ///
    /// # Errors
    ///
    /// [`InvalidWeight`] when `value` is not finite or not above zero.
    pub const fn new(value: f64) -> Result<Self, InvalidWeight> {
        if !value.is_finite() {
            Err(InvalidWeight::NotFinite)
        } else if value <= 0.0 {
            Err(InvalidWeight::NotPositive)
        } else {
            Ok(Self(value))
        }
    }

    /// The weight as a number.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// The weight of a literal, or a panic naming the defect: in a `const` context, a compile error.
    ///
    /// # Panics
    ///
    /// When [`Weight::new`] refuses `value`.
    #[doc(hidden)]
    #[track_caller]
    #[must_use]
    pub const fn literal(value: f64) -> Self {
        match Self::new(value) {
            Ok(weight) => weight,
            Err(InvalidWeight::NotFinite) => panic!("Weight: not a finite number"),
            Err(InvalidWeight::NotPositive) => panic!("Weight: not above zero"),
        }
    }
}

/// [`Weight::ONE`].
impl Default for Weight {
    fn default() -> Self {
        Self::ONE
    }
}

impl TryFrom<f64> for Weight {
    type Error = InvalidWeight;

    fn try_from(value: f64) -> Result<Self, InvalidWeight> {
        Self::new(value)
    }
}

impl From<Weight> for f64 {
    fn from(weight: Weight) -> Self {
        weight.0
    }
}

impl fmt::Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Weight({})", self.0)
    }
}

/// Why [`Weight::new`] refused a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvalidWeight {
    /// The number is NaN or infinite.
    NotFinite,
    /// The number is zero or below.
    NotPositive,
}

impl fmt::Display for InvalidWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite => write!(f, "Weight: not a finite number"),
            Self::NotPositive => write!(f, "Weight: not above zero"),
        }
    }
}

impl core::error::Error for InvalidWeight {}

/// A literal [`Weight`], checked when the code is compiled.
///
/// ```
/// let w = culmen_core::weight!(0.5);
/// assert_eq!(w.get(), 0.5);
/// ```
#[macro_export]
macro_rules! weight {
    ($value:expr $(,)?) => {
        const { $crate::Weight::literal($value) }
    };
}

/// How far a later stage of a lexicographic solve may move an objective from the optimum its own stage found.
///
/// Once an objective's stage found its optimum `f*`, offset included, every later stage keeps the objective within
/// `min(absolute, relative · |f*|)` of `f*`, on the side worse for its sense; an absent tolerance is left out of the
/// minimum, and with both absent later stages do not keep the objective at all. Plain data: the two tolerances are
/// independent.
///
/// ```
/// use culmen_core::{Tolerance, absolute_gap, relative_gap};
/// let loose = Tolerance { absolute: Some(absolute_gap!(5.0)), relative: Some(relative_gap!(0.1)) };
/// assert_eq!(loose.allowance(20.0), 2.0);
/// assert_eq!(loose.allowance(-100.0), 5.0);
/// assert_eq!(Tolerance::EXACT.allowance(7.0), 0.0);
/// assert_eq!(Tolerance::default(), Tolerance::EXACT);
/// assert_eq!(Tolerance { absolute: None, relative: None }.allowance(1.0), f64::INFINITY);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tolerance {
    /// The allowance in objective units; `None` leaves it out.
    pub absolute: Option<AbsoluteGap>,
    /// The allowance relative to `|f*|`; `None` leaves it out.
    pub relative: Option<RelativeGap>,
}

impl Tolerance {
    /// No allowance: later stages keep the objective at its optimum.
    pub const EXACT: Self = Self {
        absolute: Some(AbsoluteGap::literal(0.0)),
        relative: None,
    };

    /// How far from an optimum `optimum` later stages may move the objective: `+∞` when both tolerances are absent.
    #[must_use]
    pub fn allowance(self, optimum: f64) -> f64 {
        let absolute = self.absolute.map_or(f64::INFINITY, AbsoluteGap::get);
        let relative = self.relative.map_or(f64::INFINITY, |r| r.get() * optimum.abs());
        absolute.min(relative)
    }
}

/// [`Tolerance::EXACT`].
impl Default for Tolerance {
    fn default() -> Self {
        Self::EXACT
    }
}

/// A further linear objective: `Σ coefficient · column + offset`, minimized or maximized.
///
/// In a [`ProgramPlan`](super::ProgramPlan) its terms may repeat a column and add up when the plan is built; in a
/// [`Program`](super::Program) they are sorted by column, each column at most once with a finite, non-zero
/// coefficient, and the offset is finite.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LinearObjective {
    /// Whether the objective is minimized or maximized.
    pub sense: Sense,
    /// The terms as `(column, coefficient)`.
    pub terms: Vec<(Col, f64)>,
    /// The constant added to the objective.
    pub offset: f64,
}

impl LinearObjective {
    /// The objective's value at a point, offset included.
    ///
    /// # Panics
    ///
    /// When `values` has no value for a column of a term.
    #[must_use]
    pub fn value(&self, values: &TiSlice<Col, f64>) -> f64 {
        self.offset + self.terms.iter().map(|&(col, c)| c * values[col]).sum::<f64>()
    }

    /// The objective with its terms sorted, merged in the order given and stripped of zeros.
    fn canonical(self) -> Self {
        Self {
            terms: merge_terms(self.terms).into_iter().filter(|&(_, c)| c != 0.0).collect(),
            ..self
        }
    }
}

/// A stage of a lexicographic solve: a further objective and how far later stages may move it from its optimum.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Stage {
    /// The objective optimized in this stage.
    pub objective: LinearObjective,
    /// How far later stages may move it from the optimum this stage finds.
    pub tolerance: Tolerance,
}

/// A further objective of a blend and its weight.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Weighted {
    /// The objective.
    pub objective: LinearObjective,
    /// Its weight in the blend.
    pub weight: Weight,
}

/// How a program's own objective combines with further linear objectives.
///
/// - [`Objectives::Single`]: the program optimizes its own objective alone.
/// - [`Objectives::Lexicographic`]: the program's own objective is optimized first; then each stage's objective in
///   order, over the points that keep every earlier objective within its [`Tolerance`] of the optimum its stage
///   found. The order is the priority: earlier objectives matter more.
/// - [`Objectives::Blended`]: one objective is minimized, `Σ s · w · f` over every objective `f` with its weight `w`
///   and `s` one for minimizing, minus one for maximizing.
///
/// A program with a further objective has a linear objective of its own:
/// [`ProgramPlan::build`](super::ProgramPlan::build) refuses a quadratic part beside further objectives. Its
/// solutions report the program's own objective in [`Solution::objective`](crate::Solution::objective);
/// [`Program::objective_values`](super::Program::objective_values) gives every objective's value.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Objectives {
    /// The program's own objective alone.
    #[default]
    Single,
    /// The program's own objective first, then each stage's in order.
    Lexicographic {
        /// How far later stages may move the program's own objective from its optimum.
        first: Tolerance,
        /// The further stages, in order of priority.
        then: Vec<Stage>,
    },
    /// The weighted sum of the program's own objective and the further ones, signed by sense, minimized.
    Blended {
        /// The weight of the program's own objective.
        first: Weight,
        /// The further objectives with their weights.
        with: Vec<Weighted>,
    },
}

impl Objectives {
    /// The further objectives, in order: the first is [`ObjectiveId::new(1)`](ObjectiveId::new).
    pub fn further(&self) -> FurtherObjectives<'_> {
        FurtherObjectives {
            inner: match self {
                Self::Single => Further::Stages([].iter()),
                Self::Lexicographic { then, .. } => Further::Stages(then.iter()),
                Self::Blended { with, .. } => Further::Weighted(with.iter()),
            },
        }
    }

    /// Whether there is a further objective, so the program has several.
    #[must_use]
    pub fn are_several(&self) -> bool {
        self.further().len() > 0
    }

    /// The objectives with every further objective's terms sorted, merged and stripped of zeros, and without further
    /// objectives [`Objectives::Single`].
    pub(crate) fn canonical(self) -> Self {
        match self {
            Self::Lexicographic { then, .. } if then.is_empty() => Self::Single,
            Self::Blended { with, .. } if with.is_empty() => Self::Single,
            Self::Single => Self::Single,
            Self::Lexicographic { first, then } => Self::Lexicographic {
                first,
                then: then
                    .into_iter()
                    .map(|s| Stage {
                        objective: s.objective.canonical(),
                        ..s
                    })
                    .collect(),
            },
            Self::Blended { first, with } => Self::Blended {
                first,
                with: with
                    .into_iter()
                    .map(|w| Weighted {
                        objective: w.objective.canonical(),
                        ..w
                    })
                    .collect(),
            },
        }
    }
}

/// The further objectives of an [`Objectives`], in order; made by [`Objectives::further`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct FurtherObjectives<'a> {
    inner: Further<'a>,
}

/// The variant a [`FurtherObjectives`] walks.
#[derive(Clone, Debug)]
enum Further<'a> {
    Stages(slice::Iter<'a, Stage>),
    Weighted(slice::Iter<'a, Weighted>),
}

impl<'a> Iterator for FurtherObjectives<'a> {
    type Item = &'a LinearObjective;

    fn next(&mut self) -> Option<&'a LinearObjective> {
        match &mut self.inner {
            Further::Stages(stages) => stages.next().map(|s| &s.objective),
            Further::Weighted(weighted) => weighted.next().map(|w| &w.objective),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            Further::Stages(stages) => stages.size_hint(),
            Further::Weighted(weighted) => weighted.size_hint(),
        }
    }
}

impl DoubleEndedIterator for FurtherObjectives<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            Further::Stages(stages) => stages.next_back().map(|s| &s.objective),
            Further::Weighted(weighted) => weighted.next_back().map(|w| &w.objective),
        }
    }
}

impl ExactSizeIterator for FurtherObjectives<'_> {}

impl FusedIterator for FurtherObjectives<'_> {}
