// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Editing a loaded program: the rows and columns to append, the change a backend applies, and why an edit is
//! refused.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{Backend, Solver};
use crate::{
    Bounds, Col, Cols, Constraint, Integrality, InvalidBounds, ModelClass, ModelClasses, ProgramColumn, ProgramFeature,
    ProgramFeatures, ProgramRow, Row, Rows,
};

/// What one edit changed in a [`Solver`]'s program; the new values are read from the program.
///
/// Every slice is sorted ascending and holds each id once; new rows and columns follow every one before them.
/// [`Backend::edit`] receives it beside the edited program.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Edit<'a> {
    /// The bounds of these columns.
    ColBounds(&'a [Col]),
    /// The costs of these columns in the program's own objective.
    ColCosts(&'a [Col]),
    /// The integrality of these columns.
    ColIntegrality(&'a [Col]),
    /// The bounds of these rows.
    RowBounds(&'a [Row]),
    /// These rows, appended.
    NewRows(Rows),
    /// These columns, appended with their entries in the rows before them.
    NewCols(Cols),
}

/// A row to append to a [`Solver`]'s program: a model constraint, named or not, or a program row.
///
/// ```
/// use culmen_core::{Linear, Model, NewRow, Var};
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous());
/// let unnamed = NewRow::from(x.leq(4));
/// let named = NewRow::from(("cut", x.geq(1)));
/// assert!(matches!(unnamed, NewRow::Constraint { name: None, .. }));
/// assert!(matches!(named, NewRow::Constraint { name: Some(ref n), .. } if n == "cut"));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum NewRow {
    /// A constraint as a model writes it; its constant moves into its bounds as
    /// [`Model::build`](crate::Model::build) moves it.
    Constraint {
        /// The constraint.
        constraint: Constraint,
        /// The row's name in diagnostics and LP files; `None` names it by its index.
        name: Option<String>,
    },
    /// A row as program data; its terms may repeat a column and add up, and terms that cancel to zero are dropped.
    Row(ProgramRow),
}

/// An unnamed row.
impl From<Constraint> for NewRow {
    fn from(constraint: Constraint) -> Self {
        Self::Constraint { constraint, name: None }
    }
}

/// A named row, given as the closures of [`Model::add_constraint_map`](crate::Model::add_constraint_map) give it.
impl<N: Into<String>> From<(N, Constraint)> for NewRow {
    fn from((name, constraint): (N, Constraint)) -> Self {
        Self::Constraint {
            constraint,
            name: Some(name.into()),
        }
    }
}

impl From<ProgramRow> for NewRow {
    fn from(row: ProgramRow) -> Self {
        Self::Row(row)
    }
}

/// A column to append to a [`Solver`]'s program: its data and its coefficients in rows the program already has.
///
/// Its coefficient in every further objective is zero, and it has no quadratic term.
///
/// ```
/// use culmen_core::{Bounds, NewCol, ProgramColumn, Row};
///
/// let pattern = NewCol {
///     column: ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE),
///     entries: vec![(Row::new(0), 2.0), (Row::new(2), 1.0)],
/// };
/// assert_eq!(NewCol::from(pattern.column.clone()).entries, []);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NewCol {
    /// The column's cost in the program's own objective, its bounds, integrality and name.
    pub column: ProgramColumn,
    /// Its coefficients as `(row, coefficient)`; a row may repeat and its coefficients add up; empty for a column in
    /// no row.
    pub entries: Vec<(Row, f64)>,
}

/// A column in no row.
impl From<ProgramColumn> for NewCol {
    fn from(column: ProgramColumn) -> Self {
        Self {
            column,
            entries: Vec::new(),
        }
    }
}

/// Why a [`Solver`] refused an edit; the program and the backend are as they were before it.
///
/// The id of a new row or column in an error is the id it would have had.
///
/// ```
/// use culmen_core::{Col, EditError};
/// let error = EditError::<std::io::Error>::UnknownCol { col: Col::new(7) };
/// assert_eq!(error.to_string(), "Solver: the program has no Col(7)");
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum EditError<E> {
    /// A column the program does not have: an edit's target or a new row's term.
    UnknownCol {
        /// The column named.
        col: Col,
    },
    /// A row the program does not have: an edit's target or a new column's entry.
    UnknownRow {
        /// The row named.
        row: Row,
    },
    /// A cost is NaN or infinite.
    NonFiniteCost {
        /// The column.
        col: Col,
        /// The cost given.
        value: f64,
    },
    /// A coefficient of a new row or column is NaN or infinite, as given or after repeated terms were added up.
    NonFiniteCoefficient {
        /// The row of the coefficient.
        row: Row,
        /// The column of the coefficient.
        col: Col,
        /// The coefficient.
        value: f64,
    },
    /// A new constraint's bounds, after its constant moved into them, are invalid.
    RowBounds {
        /// The new row.
        row: Row,
        /// What is wrong with its bounds.
        source: InvalidBounds,
    },
    /// A semi-continuous or semi-integer column would have bounds that are not `lower ..= upper` with `0 < lower` and
    /// `upper` finite.
    SemiBounds {
        /// The column.
        col: Col,
        /// The bounds it would have.
        bounds: Bounds,
    },
    /// The backend does not solve the class the edited program would have.
    Unsupported {
        /// The class the edited program would have.
        class: ModelClass,
        /// The classes the backend solves.
        supported: ModelClasses,
    },
    /// The backend does not take a feature the edited program would use.
    UnsupportedFeature {
        /// The first feature the edited program would use that the backend does not take, in the order of
        /// [`ProgramFeature`].
        feature: ProgramFeature,
        /// The features the backend takes.
        supported: ProgramFeatures,
    },
    /// The backend refused the edit.
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for EditError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCol { col } => write!(f, "Solver: the program has no {col:?}"),
            Self::UnknownRow { row } => write!(f, "Solver: the program has no {row:?}"),
            Self::NonFiniteCost { col, value } => write!(f, "Solver: the cost {value} of {col:?} is not finite"),
            Self::NonFiniteCoefficient { row, col, value } => {
                write!(f, "Solver: the coefficient {value} of {col:?} in {row:?} is not finite")
            }
            Self::RowBounds { row, source } => write!(
                f,
                "Solver: the bounds of the new {row:?}, its constant moved into them, are invalid: {source}"
            ),
            Self::SemiBounds { col, bounds } => write!(
                f,
                "Solver: the semi column {col:?} would have the bounds {bounds}, not 0 < lower <= upper < +inf"
            ),
            Self::Unsupported { class, supported } => {
                write!(f, "Solver: the backend solves {supported}, not a {class}")
            }
            Self::UnsupportedFeature { feature, supported } => {
                write!(f, "Solver: the backend takes {supported}, not {feature}")
            }
            Self::Backend(error) => write!(f, "Solver: the backend refused the edit: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for EditError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::RowBounds { source, .. } => Some(source),
            Self::Backend(error) => Some(error),
            Self::UnknownCol { .. }
            | Self::UnknownRow { .. }
            | Self::NonFiniteCost { .. }
            | Self::NonFiniteCoefficient { .. }
            | Self::SemiBounds { .. }
            | Self::Unsupported { .. }
            | Self::UnsupportedFeature { .. } => None,
        }
    }
}

/// Edits of the loaded program. Each call is one edit, all or none: when it returns `Ok`, [`Solver::program`] is
/// the edited program and the backend holds it; when it returns `Err`, both are as they were before it. Ids never
/// change meaning: an edit appends rows and columns after every one before, and nothing is deleted. To switch a row
/// off, free its bounds; to take a column out, fix it at zero.
impl<B: Backend> Solver<B> {
    /// Sets the bounds of columns, all or none, the last bounds given for a column holding.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] for a column the program does not have, [`EditError::SemiBounds`] for a semi column
    /// given bounds it may not have, both before the backend sees the edit; [`EditError::Backend`] when the backend
    /// refuses it.
    pub fn set_col_bounds<T: Into<Bounds>>(
        &mut self,
        bounds: impl IntoIterator<Item = (Col, T)>,
    ) -> Result<(), EditError<B::Error>> {
        let _ = bounds;
        todo!("Solver::set_col_bounds")
    }

    /// Sets the costs of columns in the program's own objective, all or none, the last cost given for a column
    /// holding.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] for a column the program does not have, [`EditError::NonFiniteCost`] for a cost that
    /// is NaN or infinite, both before the backend sees the edit; [`EditError::Backend`] when the backend refuses it.
    pub fn set_col_costs<T: Into<f64>>(
        &mut self,
        costs: impl IntoIterator<Item = (Col, T)>,
    ) -> Result<(), EditError<B::Error>> {
        let _ = costs;
        todo!("Solver::set_col_costs")
    }

    /// Sets the integrality of columns, all or none, the last integrality given for a column holding.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] for a column the program does not have, [`EditError::SemiBounds`] for a column made
    /// semi whose bounds a semi column may not have, [`EditError::Unsupported`] or [`EditError::UnsupportedFeature`]
    /// when the backend does not solve the class or take a feature the edited program would have, all before the
    /// backend sees the edit; [`EditError::Backend`] when the backend refuses it.
    pub fn set_col_integrality(
        &mut self,
        integrality: impl IntoIterator<Item = (Col, Integrality)>,
    ) -> Result<(), EditError<B::Error>> {
        let _ = integrality;
        todo!("Solver::set_col_integrality")
    }

    /// Sets the bounds of rows, all or none, the last bounds given for a row holding; `..`, [`Bounds::FREE`],
    /// switches a row off.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownRow`] for a row the program does not have, before the backend sees the edit;
    /// [`EditError::Backend`] when the backend refuses it.
    pub fn set_row_bounds<T: Into<Bounds>>(
        &mut self,
        bounds: impl IntoIterator<Item = (Row, T)>,
    ) -> Result<(), EditError<B::Error>> {
        let _ = bounds;
        todo!("Solver::set_row_bounds")
    }

    /// Appends a row and returns it.
    ///
    /// # Errors
    ///
    /// As [`Solver::add_rows`].
    pub fn add_row(&mut self, row: impl Into<NewRow>) -> Result<Row, EditError<B::Error>> {
        let _ = row;
        todo!("Solver::add_row")
    }

    /// Appends rows, all or none, and returns them in order after every row before.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] for a term naming a column the program does not have,
    /// [`EditError::NonFiniteCoefficient`] for a coefficient that is NaN or infinite as given or added up,
    /// [`EditError::RowBounds`] for a constraint whose bounds are invalid once its constant moved into them, all
    /// before the backend sees the edit; [`EditError::Backend`] when the backend refuses it.
    pub fn add_rows<R: Into<NewRow>>(
        &mut self,
        rows: impl IntoIterator<Item = R>,
    ) -> Result<Rows, EditError<B::Error>> {
        let _ = rows;
        todo!("Solver::add_rows")
    }

    /// Appends a column with its entries in existing rows and returns it.
    ///
    /// # Errors
    ///
    /// As [`Solver::add_cols`].
    pub fn add_col(&mut self, column: impl Into<NewCol>) -> Result<Col, EditError<B::Error>> {
        let _ = column;
        todo!("Solver::add_col")
    }

    /// Appends columns, all or none, and returns them in order after every column before.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownRow`] for an entry naming a row the program does not have, [`EditError::NonFiniteCost`]
    /// for a cost that is NaN or infinite, [`EditError::NonFiniteCoefficient`] for a coefficient that is NaN or
    /// infinite as given or added up, [`EditError::SemiBounds`] for a semi column whose bounds a semi column may not
    /// have, [`EditError::Unsupported`] or [`EditError::UnsupportedFeature`] when the backend does not solve the
    /// class or take a feature the edited program would have, all before the backend sees the edit;
    /// [`EditError::Backend`] when the backend refuses it.
    pub fn add_cols<C: Into<NewCol>>(
        &mut self,
        columns: impl IntoIterator<Item = C>,
    ) -> Result<Cols, EditError<B::Error>> {
        let _ = columns;
        todo!("Solver::add_cols")
    }
}
