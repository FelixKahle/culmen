// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The LP as data: a checked, immutable value with typed handles.

mod bounds;
mod ids;
mod lp_file;
mod plan;
mod row;

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::{TiSlice, TiVec};

pub use bounds::{Bounds, Integrality, InvalidBounds, Sense};
pub use ids::{Col, Cols, Row, Rows};
pub use plan::{LpColumn, LpPlan, LpRow};
pub use row::{RowEntries, RowView};

/// A linear or mixed-integer program: minimize or maximize `cost · x + offset` over `x` with every column in its
/// bounds, the integer columns whole, and every row's activity in its bounds.
///
/// An `Lp` is built once, by value, from an [`LpPlan`] or a [`Model`](crate::Model), and every number in it was
/// checked then: costs, coefficients and the offset are finite, and each row holds each column at most once, with
/// a non-zero coefficient. It is plain data a [`Backend`](crate::Backend) loads; it holds no solver state.
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

    /// The objective value of a point, offset included.
    ///
    /// # Panics
    ///
    /// When `values` holds fewer values than the LP has columns.
    #[must_use]
    pub fn objective_value(&self, values: &TiSlice<Col, f64>) -> f64 {
        self.offset
            + self
                .columns
                .iter_enumerated()
                .map(|(col, c)| c.cost * values[col])
                .sum::<f64>()
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
    /// finite coefficient, and in every row the coefficients of each column, added up in the order given, are
    /// finite.
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
    Ok(())
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

    /// A plan of up to three columns and three rows whose terms name up to four columns.
    fn plan() -> impl Strategy<Value = LpPlan> {
        let row = prop::collection::vec((0..4usize, number()), 0..5).prop_map(|terms| LpRow {
            bounds: Bounds::FREE,
            terms: terms.into_iter().map(|(i, value)| (Col::new(i), value)).collect(),
            name: None,
        });
        let columns = prop::collection::vec(number(), 0..4);
        (number(), columns, prop::collection::vec(row, 0..4)).prop_map(|(offset, costs, rows)| LpPlan {
            offset,
            columns: costs
                .into_iter()
                .map(|cost| LpColumn::continuous(cost, Bounds::FREE))
                .collect(),
            rows: rows.into(),
            ..LpPlan::new()
        })
    }

    /// Whether the plan meets `Lp::from_plan`'s precondition, read from its prose: each row's coefficients are
    /// summed per column in a map, in the order given.
    fn accepted(plan: &LpPlan) -> bool {
        let finite = plan.offset.is_finite() && plan.columns.iter().all(|c| c.cost.is_finite());
        finite
            && plan.rows.iter().all(|row| {
                let mut sums = BTreeMap::new();
                let terms_ok = row.terms.iter().all(|&(col, value)| {
                    *sums.entry(col).or_insert(0.0) += value;
                    col.index() < plan.columns.len() && value.is_finite()
                });
                terms_ok && sums.values().all(|sum: &f64| sum.is_finite())
            })
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
