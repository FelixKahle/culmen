// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Edits of a loaded program as plain data, so a property test can generate them, replay them on a
//! [`Solver`] and judge them with the [reference](crate::reference::edit_plan).

use culmen_core::{Backend, Bounds, Col, EditError, Integrality, NewCol, NewRow, Row, Solver};

/// One edit of a [`Solver`]'s program, one variant per edit method.
#[derive(Clone, Debug, PartialEq)]
pub enum EditOp {
    /// [`Solver::set_col_bounds`].
    ColBounds(Vec<(Col, Bounds)>),
    /// [`Solver::set_col_costs`].
    ColCosts(Vec<(Col, f64)>),
    /// [`Solver::set_col_integrality`].
    ColIntegrality(Vec<(Col, Integrality)>),
    /// [`Solver::set_row_bounds`].
    RowBounds(Vec<(Row, Bounds)>),
    /// [`Solver::add_rows`].
    AddRows(Vec<NewRow>),
    /// [`Solver::add_cols`].
    AddCols(Vec<NewCol>),
}

/// What an accepted edit returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Applied {
    /// A setter returned nothing.
    Changed,
    /// [`Solver::add_rows`] returned these rows.
    Rows(Vec<Row>),
    /// [`Solver::add_cols`] returned these columns.
    Cols(Vec<Col>),
}

impl EditOp {
    /// Applies the edit to `solver` through its edit method.
    ///
    /// # Errors
    ///
    /// The solver's refusal.
    pub fn apply<B: Backend>(&self, solver: &mut Solver<B>) -> Result<Applied, EditError<B::Error>> {
        match self.clone() {
            Self::ColBounds(bounds) => solver.set_col_bounds(bounds).map(|()| Applied::Changed),
            Self::ColCosts(costs) => solver.set_col_costs(costs).map(|()| Applied::Changed),
            Self::ColIntegrality(integrality) => solver.set_col_integrality(integrality).map(|()| Applied::Changed),
            Self::RowBounds(bounds) => solver.set_row_bounds(bounds).map(|()| Applied::Changed),
            Self::AddRows(rows) => solver.add_rows(rows).map(|rows| Applied::Rows(rows.collect())),
            Self::AddCols(cols) => solver.add_cols(cols).map(|cols| Applied::Cols(cols.collect())),
        }
    }
}
