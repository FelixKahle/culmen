// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The LP as data: a checked, immutable value with typed handles.

mod bounds;
mod class;
mod ids;
mod lp_file;
mod plan;
mod quad;
mod row;

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::{TiSlice, TiVec};

pub use bounds::{Bounds, Integrality, InvalidBounds, Sense};
pub use class::{ModelClass, ModelClasses, ModelClassesIter};
pub use ids::{Col, Cols, Row, Rows};
pub use plan::{LpColumn, LpPlan, LpRow};
pub use quad::{HessianEntries, QuadEntries, QuadObjective};
pub(crate) use quad::{check_convex, merge_pairs};
pub use row::{RowEntries, RowView};

/// A linear, mixed-integer or convex quadratic program: minimize or maximize
/// `cost · x + Σ q · xᵢ · xⱼ + offset` over `x` with every column in its bounds, the integer columns whole, and
/// every row's activity in its bounds.
///
/// An `Lp` is built once, by value, from an [`LpPlan`] or a [`Model`](crate::Model), and every number in it was
/// checked then: costs, coefficients and the offset are finite, each row holds each column at most once, with a
/// non-zero coefficient, and the quadratic part ([`QuadObjective`]) is canonical and passes the convexity check of
/// [`Lp::CONVEXITY_TOLERANCE`]. Its constraints are linear; [`Lp::class`] names the kind of program it is. It is
/// plain data a [`Backend`](crate::Backend) loads; it holds no solver state.
///
/// ```
/// use culmen_core::{Bounds, LpColumn, LpPlan, Sense, bounds};
///
/// let mut plan = LpPlan::new();
/// plan.sense = Sense::Maximize;
/// let x = plan.col(LpColumn::binary(5.0));
/// let y = plan.col(LpColumn::binary(4.0));
/// let r = plan.row(bounds!(f64::NEG_INFINITY, 1.0), [(x, 1.0), (y, 1.0), (x, 0.0)]);
/// let lp = plan.build()?;
/// assert!(lp.is_mip());
/// assert_eq!(lp.row(r).entries().collect::<Vec<_>>(), [(x, 1.0), (y, 1.0)]);
/// # Ok::<(), culmen_core::LpError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "LpPlan", into = "LpPlan")
)]
pub struct Lp {
    sense: Sense,
    offset: f64,
    columns: TiVec<Col, LpColumn>,
    row_bounds: TiVec<Row, Bounds>,
    row_names: TiVec<Row, Option<String>>,
    /// Where each row's entries start in `entry_cols` and `entry_values`, with one more element at the end.
    row_starts: Vec<usize>,
    entry_cols: Vec<Col>,
    entry_values: Vec<f64>,
    quadratic: QuadObjective,
}

/// The constraint matrix of an [`Lp`] in compressed row form: row `r`'s entries are
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

impl Lp {
    /// The tolerance of the convexity check, relative to the largest Hessian value of each block.
    ///
    /// The Hessian `H` of the quadratic part (see [`QuadObjective`]) splits into blocks of columns joined by its
    /// entries. An `Lp` is built only when, for every block, `s · H + δ · I` has a Cholesky factor in floating
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

    /// Whether any column is integer.
    #[must_use]
    pub fn is_mip(&self) -> bool {
        self.columns.iter().any(|c| c.integrality == Integrality::Integer)
    }

    /// The quadratic part of the objective, empty for a linear objective.
    #[must_use]
    pub const fn quadratic(&self) -> &QuadObjective {
        &self.quadratic
    }

    /// The kind of program: whether it has integer columns and whether its objective has a quadratic part.
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
    pub fn columns(&self) -> &TiSlice<Col, LpColumn> {
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
    /// When the LP has no row `row`.
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

    /// The objective value of a point, quadratic part and offset included.
    ///
    /// # Panics
    ///
    /// When `values` holds fewer values than the LP has columns.
    #[must_use]
    pub fn objective_value(&self, values: &TiSlice<Col, f64>) -> f64 {
        let linear = self
            .columns
            .iter_enumerated()
            .map(|(col, c)| c.cost * values[col])
            .sum::<f64>();
        self.offset + linear + self.quadratic.value(values)
    }

    /// The largest amount by which a point violates a bound, a row or integrality, zero for a feasible point.
    ///
    /// # Panics
    ///
    /// When `values` holds fewer values than the LP has columns.
    #[must_use]
    pub fn max_violation(&self, values: &TiSlice<Col, f64>) -> f64 {
        let columns = self.columns.iter_enumerated().map(|(col, c)| {
            let value = values[col];
            let whole = match c.integrality {
                Integrality::Continuous => 0.0,
                Integrality::Integer => (value - value.round()).abs(),
            };
            c.bounds.violation(value).max(whole)
        });
        let rows = self.rows().map(|row| {
            let view = self.row(row);
            view.bounds().violation(view.activity(values))
        });
        columns.chain(rows).fold(0.0, f64::max)
    }

    /// Checks a plan and builds the LP from it.
    ///
    /// The plan is accepted when its offset and costs are finite, every term names a column of the plan with a
    /// finite coefficient, in every row the coefficients of each column, added up in the order given, are finite,
    /// every quadratic term names columns of the plan with a finite coefficient, the coefficients of each unordered
    /// pair of columns, added up in the order given, are finite, and the quadratic part passes the convexity check
    /// of [`Lp::CONVEXITY_TOLERANCE`] under the plan's sense.
    ///
    /// # Errors
    ///
    /// As [`LpPlan::build`].
    fn from_plan(plan: LpPlan) -> Result<Self, LpError> {
        check_plan(&plan)?;
        Ok(Self::from_plan_unchecked(plan))
    }

    /// Builds the LP from a plan [`Lp::from_plan`] accepts; a plan it refuses gives an unspecified LP, never
    /// undefined behaviour.
    pub(crate) fn from_plan_unchecked(plan: LpPlan) -> Self {
        debug_assert_eq!(check_plan(&plan), Ok(()), "Lp: an unchecked plan is refused");
        let mut lp = Self {
            sense: plan.sense,
            offset: plan.offset,
            row_bounds: TiVec::with_capacity(plan.rows.len()),
            row_names: TiVec::with_capacity(plan.rows.len()),
            row_starts: Vec::with_capacity(plan.rows.len() + 1),
            entry_cols: Vec::new(),
            entry_values: Vec::new(),
            columns: plan.columns,
            quadratic: QuadObjective::from_terms(plan.quadratic),
        };
        lp.row_starts.push(0);
        for data in plan.rows {
            lp.push_row(data);
        }
        lp
    }

    /// Appends a row with its terms sorted, merged and stripped of zeros.
    fn push_row(&mut self, data: LpRow) {
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

/// Checks a plan as [`Lp::from_plan`] states.
///
/// # Errors
///
/// [`LpError`] naming the first non-finite offset, cost or coefficient, or the first term whose column the plan
/// does not have.
fn check_plan(plan: &LpPlan) -> Result<(), LpError> {
    if !plan.offset.is_finite() {
        return Err(LpError::NonFiniteOffset { value: plan.offset });
    }
    if let Some((col, c)) = plan.columns.iter_enumerated().find(|(_, c)| !c.cost.is_finite()) {
        return Err(LpError::NonFiniteCost { col, value: c.cost });
    }
    for (row, data) in plan.rows.iter_enumerated() {
        check_terms(row, &data.terms, plan.columns.len())?;
        let merged = merge_terms(data.terms.clone());
        if let Some(&(col, value)) = merged.iter().find(|(_, value)| !value.is_finite()) {
            return Err(LpError::NonFiniteCoefficient { row, col, value });
        }
    }
    check_quadratic(plan)
}

/// Checks a plan's quadratic terms as [`Lp::from_plan`] states.
///
/// # Errors
///
/// [`LpError::UnknownQuadColumn`] or [`LpError::NonFiniteQuadCoefficient`] for the first defective term, as given
/// or merged, or [`LpError::NonConvexObjective`].
fn check_quadratic(plan: &LpPlan) -> Result<(), LpError> {
    let num_cols = plan.columns.len();
    plan.quadratic.iter().try_for_each(|&(first, second, value)| {
        match [first, second].into_iter().find(|col| col.index() >= num_cols) {
            Some(col) => Err(LpError::UnknownQuadColumn { col }),
            None => finite_quad(first, second, value),
        }
    })?;
    merge_pairs(plan.quadratic.iter().copied())
        .into_iter()
        .try_for_each(|(first, second, value)| finite_quad(first, second, value))?;
    let quadratic = QuadObjective::from_terms(plan.quadratic.iter().copied());
    check_convex(plan.sense, &quadratic, num_cols).map_err(|col| LpError::NonConvexObjective { sense: plan.sense, col })
}

/// # Errors
///
/// [`LpError::NonFiniteQuadCoefficient`] when `value` is NaN or infinite.
fn finite_quad(first: Col, second: Col, value: f64) -> Result<(), LpError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(LpError::NonFiniteQuadCoefficient { first, second, value })
    }
}

/// Checks a row's terms as given: known columns, finite coefficients.
///
/// # Errors
///
/// [`LpError::UnknownColumn`] or [`LpError::NonFiniteCoefficient`] for the first defective term.
fn check_terms(row: Row, terms: &[(Col, f64)], num_cols: usize) -> Result<(), LpError> {
    for &(col, value) in terms {
        if col.index() >= num_cols {
            return Err(LpError::UnknownColumn { row, col });
        }
        if !value.is_finite() {
            return Err(LpError::NonFiniteCoefficient { row, col, value });
        }
    }
    Ok(())
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

impl From<Lp> for LpPlan {
    fn from(lp: Lp) -> Self {
        let rows = lp
            .rows()
            .map(|row| LpRow {
                bounds: lp.row_bounds[row],
                terms: lp.row(row).entries().collect(),
                name: lp.row_names[row].clone(),
            })
            .collect();
        Self {
            sense: lp.sense,
            offset: lp.offset,
            columns: lp.columns,
            rows,
            quadratic: lp.quadratic.entries().collect(),
        }
    }
}

impl TryFrom<LpPlan> for Lp {
    type Error = LpError;

    fn try_from(plan: LpPlan) -> Result<Self, LpError> {
        plan.build()
    }
}

/// Why an [`LpPlan`] could not be built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LpError {
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
    /// [`Lp::CONVEXITY_TOLERANCE`].
    NonConvexObjective {
        /// The objective's sense.
        sense: Sense,
        /// The smallest column of the first block of the Hessian that fails the check.
        col: Col,
    },
}

impl fmt::Display for LpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteOffset { value } => write!(f, "Lp: the objective offset {value} is not finite"),
            Self::NonFiniteCost { col, value } => write!(f, "Lp: the cost {value} of {col:?} is not finite"),
            Self::NonFiniteCoefficient { row, col, value } => {
                write!(f, "Lp: the coefficient {value} of {col:?} in {row:?} is not finite")
            }
            Self::UnknownColumn { row, col } => write!(f, "Lp: {row:?} names {col:?}, which the plan does not have"),
            Self::UnknownQuadColumn { col } => {
                write!(f, "Lp: a quadratic term names {col:?}, which the plan does not have")
            }
            Self::NonFiniteQuadCoefficient { first, second, value } => {
                write!(
                    f,
                    "Lp: the quadratic coefficient {value} of {first:?} · {second:?} is not finite"
                )
            }
            Self::NonConvexObjective {
                sense: Sense::Minimize,
                col,
            } => write!(f, "Lp: the quadratic objective is not convex in the block of {col:?}"),
            Self::NonConvexObjective {
                sense: Sense::Maximize,
                col,
            } => write!(f, "Lp: the quadratic objective is not concave in the block of {col:?}"),
        }
    }
}

impl core::error::Error for LpError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;

    use super::{Bounds, Col, Lp, LpColumn, LpPlan, LpRow};

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
    fn plan() -> impl Strategy<Value = LpPlan> {
        let row = prop::collection::vec((0..4usize, number()), 0..5).prop_map(|terms| LpRow {
            bounds: Bounds::FREE,
            terms: terms.into_iter().map(|(i, value)| (Col::new(i), value)).collect(),
            name: None,
        });
        let columns = prop::collection::vec(number(), 0..4);
        let squares = prop::collection::vec((0..4usize, number()), 0..4);
        let parts = (number(), columns, prop::collection::vec(row, 0..4), squares);
        parts.prop_map(|(offset, costs, rows, squares)| LpPlan {
            offset,
            columns: costs
                .into_iter()
                .map(|cost| LpColumn::continuous(cost, Bounds::FREE))
                .collect(),
            rows: rows.into(),
            quadratic: squares
                .into_iter()
                .map(|(i, q)| (Col::new(i), Col::new(i), q))
                .collect(),
            ..LpPlan::new()
        })
    }

    /// Whether the plan meets `Lp::from_plan`'s precondition, read from its prose: each row's coefficients are
    /// summed per column in a map, in the order given, and so are the squares' coefficients, whose sums a convex
    /// objective of squares alone has non-negative.
    fn accepted(plan: &LpPlan) -> bool {
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
                prop_assert_eq!(Lp::from_plan(plan.clone()), Ok(Lp::from_plan_unchecked(plan)));
            } else {
                prop_assert!(Lp::from_plan(plan).is_err());
            }
        }
    }
}
