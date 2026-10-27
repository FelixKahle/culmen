// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The solver-neutral callback contexts: what each event lets a callback read and do.
//!
//! A [`Backend`](crate::Backend) names one concrete type per event kind, implementing the matching trait here. Code
//! generic over backends uses the traits; code written for one backend gets that backend's type, with whatever the
//! solver offers beyond the traits as its inherent methods. A backend that cannot deliver an event names
//! [`Unsupported`], which has no values, so a callback for that event can never run.

use core::fmt;

use typed_index_collections::TiSlice;

use crate::{Bounds, Col, Constraint, InvalidBounds, LogLevel, Progress};

/// A chunk of the solver's log. A line may arrive in several chunks; [`Watch`](crate::Watch) joins them.
pub trait LogContext {
    /// The text as the solver wrote it, line breaks included.
    fn text(&self) -> &str;
    /// How much the chunk matters.
    fn level(&self) -> LogLevel;
    /// Where the solve stands; often only the elapsed time.
    fn progress(&self) -> Progress;
}

/// A progress report the solver issues at its own pace.
pub trait ProgressContext {
    /// Where the solve stands.
    fn progress(&self) -> Progress;
}

/// A frequent poll at which the callback may stop the solve.
pub trait PollContext {
    /// Where the solve stands.
    fn progress(&self) -> Progress;
    /// Asks the solver to stop as soon as it can. The solve then ends as
    /// [`StopReason::Interrupted`](crate::StopReason::Interrupted), with the best solution found.
    fn interrupt(&mut self);
}

/// A feasible solution the solver found: any one, or a new incumbent.
pub trait SolutionContext {
    /// Where the solve stands.
    fn progress(&self) -> Progress;
    /// The solution's objective value, offset included.
    fn objective(&self) -> f64;
    /// The solution's value of every column.
    fn values(&self) -> &TiSlice<Col, f64>;
}

/// A candidate solution the callback may cut off with lazy constraints before the solver accepts it.
pub trait LazyContext {
    /// Where the solve stands.
    fn progress(&self) -> Progress;
    /// The candidate's objective value, offset included.
    fn objective(&self) -> f64;
    /// The candidate's value of every column.
    fn candidate(&self) -> &TiSlice<Col, f64>;
    /// Adds a constraint the candidate may violate; it holds for the rest of the search.
    ///
    /// # Errors
    ///
    /// [`ActionError`] when the constraint names an unknown column, has a non-finite number, or the solver
    /// refuses it.
    fn add_lazy(&mut self, constraint: Constraint) -> Result<(), ActionError>;
}

/// A node relaxation the callback may cut off with user cuts.
pub trait CutContext {
    /// Where the solve stands.
    fn progress(&self) -> Progress;
    /// The node relaxation's value of every column.
    fn relaxation(&self) -> &TiSlice<Col, f64>;
    /// Adds a cut: a constraint every integer-feasible point satisfies.
    ///
    /// # Errors
    ///
    /// [`ActionError`] when the cut names an unknown column, has a non-finite number, or the solver refuses it.
    fn add_cut(&mut self, cut: Constraint) -> Result<(), ActionError>;
}

/// A point at which the callback may hand the solver a heuristic solution.
pub trait HeuristicContext {
    /// Where the solve stands.
    fn progress(&self) -> Progress;
    /// Offers a value for every column. The solver checks it and keeps it if it is feasible and better.
    ///
    /// # Errors
    ///
    /// [`ActionError`] when `values` does not hold one finite value per column, or the solver refuses it.
    fn supply(&mut self, values: &TiSlice<Col, f64>) -> Result<(), ActionError>;
    /// Offers values for some columns, for the solver to complete into a feasible solution if it can.
    ///
    /// # Errors
    ///
    /// [`ActionError`] when an entry names an unknown column or a non-finite value, or the solver offers no
    /// completion.
    fn supply_partial(&mut self, entries: &[(Col, f64)]) -> Result<(), ActionError>;
}

/// The context of an event a backend cannot deliver. It has no values, so a callback for it never runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unsupported {}

impl LogContext for Unsupported {
    fn text(&self) -> &str {
        match *self {}
    }
    fn level(&self) -> LogLevel {
        match *self {}
    }
    fn progress(&self) -> Progress {
        match *self {}
    }
}

impl ProgressContext for Unsupported {
    fn progress(&self) -> Progress {
        match *self {}
    }
}

impl PollContext for Unsupported {
    fn progress(&self) -> Progress {
        match *self {}
    }
    fn interrupt(&mut self) {
        match *self {}
    }
}

impl SolutionContext for Unsupported {
    fn progress(&self) -> Progress {
        match *self {}
    }
    fn objective(&self) -> f64 {
        match *self {}
    }
    fn values(&self) -> &TiSlice<Col, f64> {
        match *self {}
    }
}

impl LazyContext for Unsupported {
    fn progress(&self) -> Progress {
        match *self {}
    }
    fn objective(&self) -> f64 {
        match *self {}
    }
    fn candidate(&self) -> &TiSlice<Col, f64> {
        match *self {}
    }
    fn add_lazy(&mut self, _: Constraint) -> Result<(), ActionError> {
        match *self {}
    }
}

impl CutContext for Unsupported {
    fn progress(&self) -> Progress {
        match *self {}
    }
    fn relaxation(&self) -> &TiSlice<Col, f64> {
        match *self {}
    }
    fn add_cut(&mut self, _: Constraint) -> Result<(), ActionError> {
        match *self {}
    }
}

impl HeuristicContext for Unsupported {
    fn progress(&self) -> Progress {
        match *self {}
    }
    fn supply(&mut self, _: &TiSlice<Col, f64>) -> Result<(), ActionError> {
        match *self {}
    }
    fn supply_partial(&mut self, _: &[(Col, f64)]) -> Result<(), ActionError> {
        match *self {}
    }
}

/// Why a callback's action was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ActionError {
    /// A full solution holds the wrong number of values.
    Length {
        /// The number of columns.
        expected: usize,
        /// The number of values given.
        found: usize,
    },
    /// An entry names a column the problem does not have.
    UnknownColumn {
        /// The column named.
        col: Col,
    },
    /// A value or coefficient is NaN or infinite.
    NotFinite {
        /// The column it belongs to.
        col: Col,
    },
    /// A constraint's bounds, after its constant moved into them, are invalid.
    Bounds(InvalidBounds),
    /// The solver refused the action; the text is its reason.
    Refused(String),
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { expected, found } => {
                write!(f, "callback action: {found} values given for {expected} columns")
            }
            Self::UnknownColumn { col } => write!(f, "callback action: {col:?} is not a column of the problem"),
            Self::NotFinite { col } => write!(f, "callback action: the number for {col:?} is not finite"),
            Self::Bounds(error) => write!(f, "callback action: {error}"),
            Self::Refused(reason) => write!(f, "callback action: the solver refused it: {reason}"),
        }
    }
}

impl core::error::Error for ActionError {}

/// Checks a full solution against the number of columns: one finite value each.
///
/// For backend authors, so every backend refuses the same solutions the same way.
///
/// # Errors
///
/// [`ActionError::Length`] or [`ActionError::NotFinite`].
pub fn check_solution(values: &TiSlice<Col, f64>, num_cols: usize) -> Result<(), ActionError> {
    if values.len() != num_cols {
        return Err(ActionError::Length {
            expected: num_cols,
            found: values.len(),
        });
    }
    match values.iter_enumerated().find(|(_, v)| !v.is_finite()) {
        Some((col, _)) => Err(ActionError::NotFinite { col }),
        None => Ok(()),
    }
}

/// Checks partial solution entries against the number of columns: known columns, finite values.
///
/// # Errors
///
/// [`ActionError::UnknownColumn`] or [`ActionError::NotFinite`].
pub fn check_entries(entries: &[(Col, f64)], num_cols: usize) -> Result<(), ActionError> {
    for &(col, value) in entries {
        if col.index() >= num_cols {
            return Err(ActionError::UnknownColumn { col });
        }
        if !value.is_finite() {
            return Err(ActionError::NotFinite { col });
        }
    }
    Ok(())
}

/// A constraint a callback added, as checked by [`check_constraint`]: `Σ coefficient · column` in the bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct AddedRow {
    /// The terms as `(column, coefficient)`, repeated columns kept as given.
    pub terms: Vec<(Col, f64)>,
    /// The interval the terms' activity must lie in, the constraint's constant moved into it.
    pub bounds: Bounds,
}

/// Checks a constraint against the number of columns and returns it as a row: known columns, finite coefficients,
/// and valid bounds once its constant moved into them.
///
/// # Errors
///
/// [`ActionError::UnknownColumn`], [`ActionError::NotFinite`] or [`ActionError::Bounds`].
pub fn check_constraint(constraint: &Constraint, num_cols: usize) -> Result<AddedRow, ActionError> {
    let terms = constraint.row_terms();
    check_entries(&terms, num_cols)?;
    let bounds = constraint.row_bounds().map_err(ActionError::Bounds)?;
    Ok(AddedRow { terms, bounds })
}
