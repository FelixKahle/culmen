// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The program as data: a checked, immutable value with typed handles.

mod bounds;
mod class;
mod features;
mod ids;
mod lp_file;
mod objectives;
mod plan;
mod quad;
mod row;

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::{TiSlice, TiVec};

pub use bounds::{Bounds, Integrality, InvalidBounds, Sense};
pub use class::{ModelClass, ModelClasses, ModelClassesIter};
pub use features::{ProgramFeature, ProgramFeatures, ProgramFeaturesIter};
pub use ids::{Col, Cols, Row, Rows};
pub use lp_file::LpFileError;
pub use objectives::{
    FurtherObjectives, InvalidWeight, LinearObjective, ObjectiveId, Objectives, Stage, Tolerance, Weight, Weighted,
};
pub use plan::{ProgramColumn, ProgramPlan, ProgramRow};
pub use quad::{HessianEntries, QuadEntries, QuadObjective};
pub(crate) use quad::{check_convex, merge_pairs};
pub use row::{RowEntries, RowView};

/// A linear, mixed-integer or convex quadratic program: minimize or maximize
/// `cost · x + Σ q · xᵢ · xⱼ + offset` over `x` with every column in its bounds, the integer columns whole, every
/// semi column zero or in its bounds, and every row's activity in its bounds; and, with further linear objectives
/// ([`Objectives`]), optimize those lexicographically or blended.
///
/// A `Program` is built once, by value, from a [`ProgramPlan`] or a [`Model`](crate::Model), and every number in it was
/// checked then: costs, coefficients and offsets are finite, each row and each further objective holds each column at
/// most once, with a non-zero coefficient, a semi column's bounds are `lower ..= upper` with `0 < lower` and `upper`
/// finite, the quadratic part ([`QuadObjective`]) is canonical and passes the convexity check of
/// [`Program::CONVEXITY_TOLERANCE`], and it is empty beside further objectives. Its constraints are linear;
/// [`Program::class`] names the kind of program it is and [`Program::features`] what it uses beyond its class. It is
/// plain data a [`Backend`](crate::Backend) loads; it holds no solver state.
///
/// ```
/// use culmen_core::{Bounds, ProgramColumn, ProgramPlan, Sense, bounds};
///
/// let mut plan = ProgramPlan::new();
/// plan.sense = Sense::Maximize;
/// let x = plan.col(ProgramColumn::binary(5.0));
/// let y = plan.col(ProgramColumn::binary(4.0));
/// let r = plan.row(bounds!(f64::NEG_INFINITY, 1.0), [(x, 1.0), (y, 1.0), (x, 0.0)]);
/// let program = plan.build()?;
/// assert!(program.is_mip());
/// assert_eq!(program.row(r).entries().collect::<Vec<_>>(), [(x, 1.0), (y, 1.0)]);
/// # Ok::<(), culmen_core::ProgramError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "ProgramPlan", into = "ProgramPlan")
)]
pub struct Program {
    sense: Sense,
    offset: f64,
    columns: TiVec<Col, ProgramColumn>,
    row_bounds: TiVec<Row, Bounds>,
    row_names: TiVec<Row, Option<String>>,
    /// Where each row's entries start in `entry_cols` and `entry_values`, with one more element at the end.
    row_starts: Vec<usize>,
    entry_cols: Vec<Col>,
    entry_values: Vec<f64>,
    quadratic: QuadObjective,
    objectives: Objectives,
}

/// The constraint matrix of a [`Program`] in compressed row form: row `r`'s entries are
/// `cols[starts[r]..starts[r + 1]]` with `values` alike, sorted by column.
#[derive(Clone, Copy, Debug)]
pub struct RowMatrix<'a> {
    /// Where each row's entries start, with the total entry count appended.
    pub starts: &'a [usize],
    /// The column of each entry.
    pub cols: &'a [Col],
    /// The coefficient of each entry.
    pub values: &'a [f64],
}

impl Program {
    /// The tolerance of the convexity check, relative to the largest Hessian value of each block.
    ///
    /// The Hessian `H` of the quadratic part (see [`QuadObjective`]) splits into blocks of columns joined by its
    /// entries. A `Program` is built only when, for every block, `s · H + δ · I` has a Cholesky factor in floating
    /// point, where `s` is `1` for minimizing and `-1` for maximizing and `δ` is this tolerance times the block's
    /// largest `|Hᵢⱼ|`. So every eigenvalue of `s · H` on a block is at least about `-δ`: the objective is convex
    /// for minimizing and concave for maximizing up to that tolerance and the factorization's rounding. A block
    /// costs a dense factorization, cubic in its number of columns.
    pub const CONVEXITY_TOLERANCE: f64 = quad::CONVEXITY_TOLERANCE;

    /// Whether the objective is minimized or maximized.
    #[must_use]
    pub const fn sense(&self) -> Sense {
        self.sense
    }

    /// The constant added to the objective.
    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }

    /// The number of columns.
    #[must_use]
    pub fn num_cols(&self) -> usize {
        self.columns.len()
    }

    /// The number of rows.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.row_bounds.len()
    }

    /// The number of non-zero coefficients.
    #[must_use]
    pub const fn num_entries(&self) -> usize {
        self.entry_values.len()
    }

    /// Whether any column is integer, semi-continuous or semi-integer: whether its values are not one interval.
    #[must_use]
    pub fn is_mip(&self) -> bool {
        self.columns.iter().any(|c| c.integrality.is_discrete())
    }

    /// The quadratic part of the objective, empty for a linear objective.
    #[must_use]
    pub const fn quadratic(&self) -> &QuadObjective {
        &self.quadratic
    }

    /// The kind of program: whether it has integer columns and whether its objective has a quadratic part.
    ///
    /// A semi-continuous or semi-integer column counts as integer: its values are not one interval, so a program with
    /// one is a MILP or a MIQP. Further objectives do not change the class; [`Program::features`] lists them.
    ///
    /// ```
    /// use culmen_core::{Model, ModelClass, Var};
    ///
    /// let mut m = Model::new();
    /// let x = m.add_var(Var::continuous());
    /// let n = m.add_var(Var::integer());
    /// m.minimize(x + n);
    /// assert_eq!(m.build()?.class(), ModelClass::Milp);
    /// m.minimize(x * x + n);
    /// assert_eq!(m.build()?.class(), ModelClass::Miqp);
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    #[must_use]
    pub fn class(&self) -> ModelClass {
        match (self.is_mip(), self.quadratic.is_empty()) {
            (false, true) => ModelClass::Lp,
            (true, true) => ModelClass::Milp,
            (false, false) => ModelClass::Qp,
            (true, false) => ModelClass::Miqp,
        }
    }

    /// How the program's own objective combines with further linear objectives.
    #[must_use]
    pub const fn objectives(&self) -> &Objectives {
        &self.objectives
    }

    /// What the program uses beyond its [`class`](Program::class): semi columns and multiple objectives; never
    /// [`ProgramFeature::StartSolution`], which belongs to a solve.
    ///
    /// ```
    /// use culmen_core::{Model, Objective, ProgramFeature, ProgramFeatures, Var};
    ///
    /// let mut m = Model::new();
    /// let x = m.add_var(Var::semi_continuous().bounds(2, 10));
    /// m.minimize(x);
    /// assert_eq!(m.build()?.features(), ProgramFeature::SemiContinuous.into());
    /// m.add_objective(Objective::maximize(x));
    /// let both = ProgramFeatures::of(&[ProgramFeature::SemiContinuous, ProgramFeature::MultipleObjectives]);
    /// assert_eq!(m.build()?.features(), both);
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    #[must_use]
    pub fn features(&self) -> ProgramFeatures {
        let columns = self.columns.iter().filter_map(|c| match c.integrality {
            Integrality::SemiContinuous => Some(ProgramFeature::SemiContinuous),
            Integrality::SemiInteger => Some(ProgramFeature::SemiInteger),
            Integrality::Continuous | Integrality::Integer => None,
        });
        let objectives = self
            .objectives
            .are_several()
            .then_some(ProgramFeature::MultipleObjectives);
        columns.chain(objectives).collect()
    }

    /// The value of every objective at a point, offset included: the program's own at [`ObjectiveId::FIRST`], quadratic
    /// part included, then each further objective in order.
    ///
    /// # Panics
    ///
    /// When `values` holds fewer values than the program has columns.
    #[must_use]
    pub fn objective_values(&self, values: &TiSlice<Col, f64>) -> TiVec<ObjectiveId, f64> {
        let further = self.objectives.further().map(|objective| objective.value(values));
        core::iter::once(self.objective_value(values)).chain(further).collect()
    }

    /// Every column, in order.
    pub fn cols(&self) -> Cols {
        Cols::new(self.num_cols())
    }

    /// Every row, in order.
    pub fn rows(&self) -> Rows {
        Rows::new(self.num_rows())
    }

    /// The columns' data, indexed by [`Col`].
    #[must_use]
    pub fn columns(&self) -> &TiSlice<Col, ProgramColumn> {
        &self.columns
    }

    /// The rows' bounds, indexed by [`Row`].
    #[must_use]
    pub fn row_bounds(&self) -> &TiSlice<Row, Bounds> {
        &self.row_bounds
    }

    /// A row's bounds, name and entries.
    ///
    /// # Panics
    ///
    /// When the program has no row `row`.
    #[must_use]
    pub fn row(&self, row: Row) -> RowView<'_> {
        let range = self.row_starts[row.index()]..self.row_starts[row.index() + 1];
        RowView::new(
            self.row_bounds[row],
            self.row_names[row].as_deref(),
            &self.entry_cols[range.clone()],
            &self.entry_values[range],
        )
    }

    /// The whole constraint matrix in compressed row form.
    #[must_use]
    pub fn matrix(&self) -> RowMatrix<'_> {
        RowMatrix {
            starts: &self.row_starts,
            cols: &self.entry_cols,
            values: &self.entry_values,
        }
    }

    /// The value of the program's own objective at a point, quadratic part and offset included.
    ///
    /// # Panics
    ///
    /// When `values` holds fewer values than the program has columns.
    #[must_use]
    pub fn objective_value(&self, values: &TiSlice<Col, f64>) -> f64 {
        let linear = self
            .columns
            .iter_enumerated()
            .map(|(col, c)| c.cost * values[col])
            .sum::<f64>();
        self.offset + linear + self.quadratic.value(values)
    }

    /// The activity of every row at a point: the sum of its coefficients times the point's values.
    ///
    /// # Panics
    ///
    /// When `values` holds fewer values than the program has columns.
    #[must_use]
    pub fn activities(&self, values: &TiSlice<Col, f64>) -> TiVec<Row, f64> {
        self.rows().map(|row| self.row(row).activity(values)).collect()
    }

    /// The largest amount by which a point violates a bound, a row, integrality or a semi column's choice between zero
    /// and its bounds ([`Integrality::violation`]), zero for a feasible point and infinite for one holding NaN.
    ///
    /// # Panics
    ///
    /// When `values` holds fewer values than the program has columns.
    #[must_use]
    pub fn max_violation(&self, values: &TiSlice<Col, f64>) -> f64 {
        let columns = self
            .columns
            .iter_enumerated()
            .map(|(col, c)| c.integrality.violation(c.bounds, values[col]));
        let rows = self.rows().map(|row| {
            let view = self.row(row);
            view.bounds().violation(view.activity(values))
        });
        columns.chain(rows).fold(0.0, f64::max)
    }

    /// Checks a plan and builds the program from it.
    ///
    /// The plan is accepted when its offset and costs are finite, every term names a column of the plan with a
    /// finite coefficient, in every row the coefficients of each column, added up in the order given, are finite,
    /// every quadratic term names columns of the plan with a finite coefficient, the coefficients of each unordered
    /// pair of columns, added up in the order given, are finite, and the quadratic part passes the convexity check
    /// of [`Program::CONVEXITY_TOLERANCE`] under the plan's sense.
    ///
    /// # Errors
    ///
    /// As [`ProgramPlan::build`].
    fn from_plan(plan: ProgramPlan) -> Result<Self, ProgramError> {
        check_plan(&plan)?;
        Ok(Self::from_plan_unchecked(plan))
    }

    /// Builds the program from a plan [`Program::from_plan`] accepts; a plan it refuses gives an unspecified program,
    /// never undefined behaviour.
    pub(crate) fn from_plan_unchecked(plan: ProgramPlan) -> Self {
        debug_assert_eq!(check_plan(&plan), Ok(()), "Program: an unchecked plan is refused");
        let mut program = Self {
            sense: plan.sense,
            offset: plan.offset,
            row_bounds: TiVec::with_capacity(plan.rows.len()),
            row_names: TiVec::with_capacity(plan.rows.len()),
            row_starts: Vec::with_capacity(plan.rows.len() + 1),
            entry_cols: Vec::new(),
            entry_values: Vec::new(),
            columns: plan.columns,
            quadratic: QuadObjective::from_terms(plan.quadratic),
            objectives: plan.objectives.canonical(),
        };
        program.row_starts.push(0);
        for data in plan.rows {
            program.push_row(data);
        }
        program
    }

    /// Appends a row with its terms sorted, merged and stripped of zeros.
    fn push_row(&mut self, data: ProgramRow) {
        for (col, value) in merge_terms(data.terms) {
            if value != 0.0 {
                self.entry_cols.push(col);
                self.entry_values.push(value);
            }
        }
        self.row_starts.push(self.entry_values.len());
        self.row_bounds.push(data.bounds);
        self.row_names.push(data.name);
    }
}

/// Checks a plan as [`Program::from_plan`] states.
///
/// # Errors
///
/// [`ProgramError`] naming the first non-finite offset, cost or coefficient, the first semi column with bounds it may
/// not have, the first term whose column the plan does not have, a quadratic part that is not convex, or one beside
/// further objectives.
fn check_plan(plan: &ProgramPlan) -> Result<(), ProgramError> {
    if !plan.offset.is_finite() {
        return Err(ProgramError::NonFiniteOffset { value: plan.offset });
    }
    check_columns(&plan.columns)?;
    let num_cols = plan.columns.len();
    plan.rows
        .iter_enumerated()
        .try_for_each(|(row, data)| bad_term(&data.terms, num_cols).map_or(Ok(()), |bad| Err(bad.in_row(row))))?;
    check_quadratic(plan)?;
    check_objectives(plan)
}

/// Checks each column's cost, and a semi column's bounds.
///
/// # Errors
///
/// [`ProgramError::NonFiniteCost`] or [`ProgramError::SemiBounds`] for the first column at fault.
fn check_columns(columns: &TiSlice<Col, ProgramColumn>) -> Result<(), ProgramError> {
    columns.iter_enumerated().try_for_each(|(col, c)| {
        if !c.cost.is_finite() {
            Err(ProgramError::NonFiniteCost { col, value: c.cost })
        } else if c.integrality.is_semi() && !semi_bounds(c.bounds) {
            Err(ProgramError::SemiBounds { col, bounds: c.bounds })
        } else {
            Ok(())
        }
    })
}

/// Whether `bounds` may be a semi column's: `0 < lower` and `upper` finite, so zero lies outside them.
pub(crate) const fn semi_bounds(bounds: Bounds) -> bool {
    bounds.lower() > 0.0 && bounds.upper().is_finite()
}

/// Checks a plan's further objectives, and that its own objective is linear beside them.
///
/// # Errors
///
/// [`ProgramError::QuadraticWithObjectives`], or for the first further objective at fault
/// [`ProgramError::NonFiniteObjectiveOffset`], [`ProgramError::UnknownObjectiveColumn`] or
/// [`ProgramError::NonFiniteObjectiveCoefficient`].
fn check_objectives(plan: &ProgramPlan) -> Result<(), ProgramError> {
    let quadratic = QuadObjective::from_terms(plan.quadratic.iter().copied());
    if plan.objectives.are_several() && !quadratic.is_empty() {
        return Err(ProgramError::QuadraticWithObjectives);
    }
    let num_cols = plan.columns.len();
    plan.objectives.further().zip(1..).try_for_each(|(further, index)| {
        let objective = ObjectiveId::new(index);
        if !further.offset.is_finite() {
            return Err(ProgramError::NonFiniteObjectiveOffset {
                objective,
                value: further.offset,
            });
        }
        bad_term(&further.terms, num_cols).map_or(Ok(()), |bad| Err(bad.in_objective(objective)))
    })
}

/// A term a plan may not hold.
#[derive(Clone, Copy, Debug)]
enum BadTerm {
    /// The term names a column the plan does not have.
    Unknown(Col),
    /// The coefficient of the column is NaN or infinite, as given or added up.
    NonFinite(Col, f64),
}

impl BadTerm {
    /// The error of a row holding this term.
    const fn in_row(self, row: Row) -> ProgramError {
        match self {
            Self::Unknown(col) => ProgramError::UnknownColumn { row, col },
            Self::NonFinite(col, value) => ProgramError::NonFiniteCoefficient { row, col, value },
        }
    }

    /// The error of a further objective holding this term.
    const fn in_objective(self, objective: ObjectiveId) -> ProgramError {
        match self {
            Self::Unknown(col) => ProgramError::UnknownObjectiveColumn { objective, col },
            Self::NonFinite(col, value) => ProgramError::NonFiniteObjectiveCoefficient { objective, col, value },
        }
    }
}

/// The first term naming a column at or beyond `num_cols` or holding a non-finite coefficient, as given; else the
/// first column whose coefficients, added up in the order given, are not finite.
fn bad_term(terms: &[(Col, f64)], num_cols: usize) -> Option<BadTerm> {
    let given = terms.iter().find_map(|&(col, value)| {
        if col.index() >= num_cols {
            Some(BadTerm::Unknown(col))
        } else {
            (!value.is_finite()).then_some(BadTerm::NonFinite(col, value))
        }
    });
    given.or_else(|| {
        merge_terms(terms.to_vec())
            .into_iter()
            .find(|(_, value)| !value.is_finite())
            .map(|(col, value)| BadTerm::NonFinite(col, value))
    })
}

/// Checks a plan's quadratic terms as [`Program::from_plan`] states.
///
/// # Errors
///
/// [`ProgramError::UnknownQuadColumn`] or [`ProgramError::NonFiniteQuadCoefficient`] for the first defective term, as
/// given or merged, or [`ProgramError::NonConvexObjective`].
fn check_quadratic(plan: &ProgramPlan) -> Result<(), ProgramError> {
    let num_cols = plan.columns.len();
    plan.quadratic.iter().try_for_each(|&(first, second, value)| {
        match [first, second].into_iter().find(|col| col.index() >= num_cols) {
            Some(col) => Err(ProgramError::UnknownQuadColumn { col }),
            None => finite_quad(first, second, value),
        }
    })?;
    merge_pairs(plan.quadratic.iter().copied())
        .into_iter()
        .try_for_each(|(first, second, value)| finite_quad(first, second, value))?;
    let quadratic = QuadObjective::from_terms(plan.quadratic.iter().copied());
    check_convex(plan.sense, &quadratic, num_cols)
        .map_err(|col| ProgramError::NonConvexObjective { sense: plan.sense, col })
}

/// # Errors
///
/// [`ProgramError::NonFiniteQuadCoefficient`] when `value` is NaN or infinite.
fn finite_quad(first: Col, second: Col, value: f64) -> Result<(), ProgramError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ProgramError::NonFiniteQuadCoefficient { first, second, value })
    }
}

/// The terms sorted by column, each column once with its coefficients added up in the order given; zeros are kept.
pub(crate) fn merge_terms(mut terms: Vec<(Col, f64)>) -> Vec<(Col, f64)> {
    // A stable sort keeps each column's terms in the order given, so every caller adds them up alike.
    terms.sort_by_key(|&(col, _)| col);
    let mut merged: Vec<(Col, f64)> = Vec::with_capacity(terms.len());
    for (col, value) in terms {
        match merged.last_mut() {
            Some(last) if last.0 == col => last.1 += value,
            _ => merged.push((col, value)),
        }
    }
    merged
}

impl From<Program> for ProgramPlan {
    fn from(program: Program) -> Self {
        let rows = program
            .rows()
            .map(|row| ProgramRow {
                bounds: program.row_bounds[row],
                terms: program.row(row).entries().collect(),
                name: program.row_names[row].clone(),
            })
            .collect();
        Self {
            sense: program.sense,
            offset: program.offset,
            columns: program.columns,
            rows,
            quadratic: program.quadratic.entries().collect(),
            objectives: program.objectives,
        }
    }
}

impl TryFrom<ProgramPlan> for Program {
    type Error = ProgramError;

    fn try_from(plan: ProgramPlan) -> Result<Self, ProgramError> {
        plan.build()
    }
}

/// Why a [`ProgramPlan`] could not be built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProgramError {
    /// The objective offset is NaN or infinite.
    NonFiniteOffset {
        /// The offset given.
        value: f64,
    },
    /// A column's cost is NaN or infinite.
    NonFiniteCost {
        /// The column.
        col: Col,
        /// The cost given.
        value: f64,
    },
    /// A row's coefficient is NaN or infinite, as given or after repeated terms were added up.
    NonFiniteCoefficient {
        /// The row.
        row: Row,
        /// The column of the coefficient.
        col: Col,
        /// The coefficient.
        value: f64,
    },
    /// A row's term names a column the plan does not have.
    UnknownColumn {
        /// The row.
        row: Row,
        /// The column named.
        col: Col,
    },
    /// A quadratic term names a column the plan does not have.
    UnknownQuadColumn {
        /// The column named.
        col: Col,
    },
    /// A quadratic coefficient is NaN or infinite, as given or after the terms of its pair were added up.
    NonFiniteQuadCoefficient {
        /// The first column of the pair.
        first: Col,
        /// The second column of the pair.
        second: Col,
        /// The coefficient.
        value: f64,
    },
    /// The quadratic part is not convex for minimizing, or not concave for maximizing, within
    /// [`Program::CONVEXITY_TOLERANCE`].
    NonConvexObjective {
        /// The objective's sense.
        sense: Sense,
        /// The smallest column of the first block of the Hessian that fails the check.
        col: Col,
    },
    /// A semi-continuous or semi-integer column's bounds are not `lower ..= upper` with `0 < lower` and `upper`
    /// finite.
    SemiBounds {
        /// The column.
        col: Col,
        /// The bounds given.
        bounds: Bounds,
    },
    /// A further objective's offset is NaN or infinite.
    NonFiniteObjectiveOffset {
        /// The objective.
        objective: ObjectiveId,
        /// The offset given.
        value: f64,
    },
    /// A further objective's coefficient is NaN or infinite, as given or after repeated terms were added up.
    NonFiniteObjectiveCoefficient {
        /// The objective.
        objective: ObjectiveId,
        /// The column of the coefficient.
        col: Col,
        /// The coefficient.
        value: f64,
    },
    /// A further objective's term names a column the plan does not have.
    UnknownObjectiveColumn {
        /// The objective.
        objective: ObjectiveId,
        /// The column named.
        col: Col,
    },
    /// The objective has a quadratic part beside further objectives, which are linear and need it linear too.
    QuadraticWithObjectives,
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteOffset { value } => write!(f, "Program: the objective offset {value} is not finite"),
            Self::NonFiniteCost { col, value } => write!(f, "Program: the cost {value} of {col:?} is not finite"),
            Self::NonFiniteCoefficient { row, col, value } => {
                write!(
                    f,
                    "Program: the coefficient {value} of {col:?} in {row:?} is not finite"
                )
            }
            Self::UnknownColumn { row, col } => {
                write!(f, "Program: {row:?} names {col:?}, which the plan does not have")
            }
            Self::UnknownQuadColumn { col } => {
                write!(
                    f,
                    "Program: a quadratic term names {col:?}, which the plan does not have"
                )
            }
            Self::NonFiniteQuadCoefficient { first, second, value } => {
                write!(
                    f,
                    "Program: the quadratic coefficient {value} of {first:?} · {second:?} is not finite"
                )
            }
            Self::NonConvexObjective {
                sense: Sense::Minimize,
                col,
            } => write!(
                f,
                "Program: the quadratic objective is not convex in the block of {col:?}"
            ),
            Self::NonConvexObjective {
                sense: Sense::Maximize,
                col,
            } => write!(
                f,
                "Program: the quadratic objective is not concave in the block of {col:?}"
            ),
            Self::SemiBounds { col, bounds } => write!(
                f,
                "Program: the semi column {col:?} has the bounds {bounds}, not 0 < lower <= upper < +inf"
            ),
            Self::NonFiniteObjectiveOffset { objective, value } => {
                write!(f, "Program: the offset {value} of {objective:?} is not finite")
            }
            Self::NonFiniteObjectiveCoefficient { objective, col, value } => write!(
                f,
                "Program: the coefficient {value} of {col:?} in {objective:?} is not finite"
            ),
            Self::UnknownObjectiveColumn { objective, col } => {
                write!(f, "Program: {objective:?} names {col:?}, which the plan does not have")
            }
            Self::QuadraticWithObjectives => {
                write!(
                    f,
                    "Program: the objective has a quadratic part beside further objectives"
                )
            }
        }
    }
}

impl core::error::Error for ProgramError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;

    use super::{Bounds, Col, Program, ProgramColumn, ProgramPlan, ProgramRow};

    /// A number from a small finite range, a huge one whose sums overflow, an infinity or NaN.
    fn number() -> impl Strategy<Value = f64> {
        prop_oneof![
            6 => (-3i32..=3).prop_map(f64::from),
            2 => prop_oneof![Just(1e308), Just(-1e308)],
            1 => prop_oneof![Just(f64::INFINITY), Just(f64::NAN)],
        ]
    }

    /// A plan of up to three columns and three rows whose terms name up to four columns, minimizing an objective
    /// with up to three squares of up to four columns.
    fn plan() -> impl Strategy<Value = ProgramPlan> {
        let row = prop::collection::vec((0..4usize, number()), 0..5).prop_map(|terms| ProgramRow {
            bounds: Bounds::FREE,
            terms: terms.into_iter().map(|(i, value)| (Col::new(i), value)).collect(),
            name: None,
        });
        let columns = prop::collection::vec(number(), 0..4);
        let squares = prop::collection::vec((0..4usize, number()), 0..4);
        let parts = (number(), columns, prop::collection::vec(row, 0..4), squares);
        parts.prop_map(|(offset, costs, rows, squares)| ProgramPlan {
            offset,
            columns: costs
                .into_iter()
                .map(|cost| ProgramColumn::continuous(cost, Bounds::FREE))
                .collect(),
            rows: rows.into(),
            quadratic: squares
                .into_iter()
                .map(|(i, q)| (Col::new(i), Col::new(i), q))
                .collect(),
            ..ProgramPlan::new()
        })
    }

    /// Whether the plan meets `Program::from_plan`'s precondition, read from its prose: each row's coefficients are
    /// summed per column in a map, in the order given, and so are the squares' coefficients, whose sums a convex
    /// objective of squares alone has non-negative.
    fn accepted(plan: &ProgramPlan) -> bool {
        let num_cols = plan.columns.len();
        let finite = plan.offset.is_finite() && plan.columns.iter().all(|c| c.cost.is_finite());
        let rows_ok = plan
            .rows
            .iter()
            .all(|row| sums(row.terms.iter().copied(), num_cols).is_some());
        let squares = sums(plan.quadratic.iter().map(|&(i, _, q)| (i, q)), num_cols);
        finite && rows_ok && squares.is_some_and(|sums| sums.values().all(|&sum| sum >= 0.0))
    }

    /// The coefficients summed per column, when every term names one of `num_cols` columns with a finite
    /// coefficient and every sum is finite.
    fn sums(terms: impl Iterator<Item = (Col, f64)>, num_cols: usize) -> Option<BTreeMap<Col, f64>> {
        let mut sums = BTreeMap::new();
        let mut terms = terms;
        let terms_ok = terms.all(|(col, value)| {
            *sums.entry(col).or_insert(0.0) += value;
            col.index() < num_cols && value.is_finite()
        });
        (terms_ok && sums.values().all(|sum: &f64| sum.is_finite())).then_some(sums)
    }

    proptest! {
        /// `from_plan` builds what `from_plan_unchecked` builds from every plan it accepts, and refuses every other.
        #[test]
        fn test_from_plan_is_from_plan_unchecked_on_accepted_plans(plan in plan()) {
            if accepted(&plan) {
                prop_assert_eq!(Program::from_plan(plan.clone()), Ok(Program::from_plan_unchecked(plan)));
            } else {
                prop_assert!(Program::from_plan(plan).is_err());
            }
        }
    }
}
