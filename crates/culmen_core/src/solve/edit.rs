// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Editing a loaded program: the rows and columns to append, the change a backend applies, and why an edit is
//! refused.

use core::fmt;
use std::collections::BTreeMap;

use super::{Backend, Solver, lack};
use crate::program::{BadColumn, BadTerm, bad_column, bad_term, merge_terms};
use crate::{
    Bounds, Col, Cols, Constraint, Integrality, InvalidBounds, ModelClass, ModelClasses, NewCol, Program,
    ProgramColumn, ProgramFeature, ProgramFeatures, ProgramRow, Row, Rows,
};

/// What one edit changed in a [`Solver`]'s program; the new values are read from the program.
///
/// Only a [`Solver`] makes one, so its ids are the edited program's, sorted ascending, each once, and new rows and
/// columns follow every one before them. [`Backend::edit`] receives it beside the edited program and matches it as
/// `Edit::ColBounds { cols, .. }`; every variant is non-exhaustive, so a later version may add to it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Edit {
    /// The bounds of these columns changed.
    #[non_exhaustive]
    ColBounds {
        /// The columns, ascending, each once.
        cols: Vec<Col>,
    },
    /// The costs of these columns in the program's own objective changed.
    #[non_exhaustive]
    ColCosts {
        /// The columns, ascending, each once.
        cols: Vec<Col>,
    },
    /// The integrality of these columns changed.
    #[non_exhaustive]
    ColIntegrality {
        /// The columns, ascending, each once.
        cols: Vec<Col>,
    },
    /// The bounds of these rows changed.
    #[non_exhaustive]
    RowBounds {
        /// The rows, ascending, each once.
        rows: Vec<Row>,
    },
    /// These rows were appended.
    #[non_exhaustive]
    NewRows {
        /// The rows, the program's last.
        rows: Rows,
    },
    /// These columns were appended, with their entries in the rows before them.
    #[non_exhaustive]
    NewCols {
        /// The columns, the program's last.
        cols: Cols,
    },
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

/// Why a [`Solver`] refused an edit.
///
/// After every refusal [`Solver::program`] is the program before the edit. The backend holds it too, except after
/// [`EditError::Backend`] when the backend refused the edit and then the program loaded into it again, or refused a
/// load still pending from such a failure: the solver then loads the program again before the next call that runs
/// the backend. The id of a new row or column in an error is the id it would have had.
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
    /// The backend refused the edit, or the program the solver loaded into it first, after an earlier edit the
    /// backend refused and whose reload failed.
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
/// the edited program and the backend holds it; when it returns `Err`, the program is as it was before it (see
/// [`EditError`] for the backend). Ids never change meaning: an edit appends rows and columns after every one before,
/// and nothing is deleted. To switch a row off, free its bounds; to take a column out, fix it at zero.
impl<B: Backend> Solver<B> {
    /// Sets the bounds of columns, all or none, the last bounds given for a column holding.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] for a column the program does not have, [`EditError::SemiBounds`] for a semi column
    /// given bounds it may not have, both before the backend sees the edit; [`EditError::Backend`] when the backend
    /// refuses it, or refuses the program the solver loads first after an earlier failed edit.
    pub fn set_col_bounds<T: Into<Bounds>>(
        &mut self,
        bounds: impl IntoIterator<Item = (Col, T)>,
    ) -> Result<(), EditError<B::Error>> {
        self.apply_col_bounds(latest(bounds.into_iter().map(|(col, bounds)| (col, bounds.into()))))
    }

    /// Sets the costs of columns in the program's own objective, all or none, the last cost given for a column
    /// holding.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] for a column the program does not have, [`EditError::NonFiniteCost`] for a cost that
    /// is NaN or infinite, both before the backend sees the edit; [`EditError::Backend`] when the backend refuses it,
    /// or refuses the program the solver loads first after an earlier failed edit.
    pub fn set_col_costs<T: Into<f64>>(
        &mut self,
        costs: impl IntoIterator<Item = (Col, T)>,
    ) -> Result<(), EditError<B::Error>> {
        self.apply_col_costs(latest(costs.into_iter().map(|(col, cost)| (col, cost.into()))))
    }

    /// Sets the integrality of columns, all or none, the last integrality given for a column holding.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] for a column the program does not have, [`EditError::SemiBounds`] for a column made
    /// semi whose bounds a semi column may not have, [`EditError::Unsupported`] or [`EditError::UnsupportedFeature`]
    /// when the backend does not solve the class or take a feature the edited program would have, all before the
    /// backend sees the edit; [`EditError::Backend`] when the backend refuses it, or refuses the program the solver
    /// loads first after an earlier failed edit.
    pub fn set_col_integrality(
        &mut self,
        integrality: impl IntoIterator<Item = (Col, Integrality)>,
    ) -> Result<(), EditError<B::Error>> {
        self.apply_col_integrality(latest(integrality))
    }

    /// Sets the bounds of rows, all or none, the last bounds given for a row holding; `..`, [`Bounds::FREE`],
    /// switches a row off.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownRow`] for a row the program does not have, before the backend sees the edit;
    /// [`EditError::Backend`] when the backend refuses it, or refuses the program the solver loads first after an
    /// earlier failed edit.
    pub fn set_row_bounds<T: Into<Bounds>>(
        &mut self,
        bounds: impl IntoIterator<Item = (Row, T)>,
    ) -> Result<(), EditError<B::Error>> {
        self.apply_row_bounds(latest(bounds.into_iter().map(|(row, bounds)| (row, bounds.into()))))
    }

    /// Appends a row and returns it.
    ///
    /// # Errors
    ///
    /// As [`Solver::add_rows`].
    pub fn add_row(&mut self, row: impl Into<NewRow>) -> Result<Row, EditError<B::Error>> {
        let new = Row::new(self.program.num_rows());
        self.append_rows(vec![row.into()]).map(|_| new)
    }

    /// Appends rows, all or none, and returns them in order after every row before.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] for a term naming a column the program does not have,
    /// [`EditError::NonFiniteCoefficient`] for a coefficient that is NaN or infinite as given or added up,
    /// [`EditError::RowBounds`] for a constraint whose bounds are invalid once its constant moved into them, all
    /// before the backend sees the edit; [`EditError::Backend`] when the backend refuses it, or refuses the program
    /// the solver loads first after an earlier failed edit.
    pub fn add_rows<R: Into<NewRow>>(
        &mut self,
        rows: impl IntoIterator<Item = R>,
    ) -> Result<Rows, EditError<B::Error>> {
        self.append_rows(rows.into_iter().map(Into::into).collect())
    }

    /// Appends a column with its entries in existing rows and returns it.
    ///
    /// # Errors
    ///
    /// As [`Solver::add_cols`].
    pub fn add_col(&mut self, column: impl Into<NewCol>) -> Result<Col, EditError<B::Error>> {
        let new = Col::new(self.program.num_cols());
        self.append_cols(vec![column.into()]).map(|_| new)
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
    /// [`EditError::Backend`] when the backend refuses it, or refuses the program the solver loads first after an
    /// earlier failed edit.
    pub fn add_cols<C: Into<NewCol>>(
        &mut self,
        columns: impl IntoIterator<Item = C>,
    ) -> Result<Cols, EditError<B::Error>> {
        self.append_cols(columns.into_iter().map(Into::into).collect())
    }
}

/// The private machinery of the edits.
impl<B: Backend> Solver<B> {
    /// [`Solver::set_col_bounds`] with each column once.
    ///
    /// # Errors
    ///
    /// As [`Solver::set_col_bounds`].
    fn apply_col_bounds(&mut self, bounds: BTreeMap<Col, Bounds>) -> Result<(), EditError<B::Error>> {
        self.sync().map_err(EditError::Backend)?;
        known_cols(&bounds, self.program.num_cols())?;
        check_edited(&self.program, &bounds, |c, bounds| {
            bad_column(c.cost, bounds, c.integrality)
        })?;
        let edit = |cols| Edit::ColBounds { cols };
        self.set(bounds, Program::replace_col_bounds, edit, Undo::ColBounds)
    }

    /// [`Solver::set_col_costs`] with each column once.
    ///
    /// # Errors
    ///
    /// As [`Solver::set_col_costs`].
    fn apply_col_costs(&mut self, costs: BTreeMap<Col, f64>) -> Result<(), EditError<B::Error>> {
        self.sync().map_err(EditError::Backend)?;
        known_cols(&costs, self.program.num_cols())?;
        check_edited(&self.program, &costs, |c, cost| {
            bad_column(cost, c.bounds, c.integrality)
        })?;
        let edit = |cols| Edit::ColCosts { cols };
        self.set(costs, Program::replace_col_cost, edit, Undo::ColCosts)
    }

    /// [`Solver::set_col_integrality`] with each column once.
    ///
    /// # Errors
    ///
    /// As [`Solver::set_col_integrality`].
    fn apply_col_integrality(&mut self, integrality: BTreeMap<Col, Integrality>) -> Result<(), EditError<B::Error>> {
        self.sync().map_err(EditError::Backend)?;
        known_cols(&integrality, self.program.num_cols())?;
        check_edited(&self.program, &integrality, |c, integrality| {
            bad_column(c.cost, c.bounds, integrality)
        })?;
        let edited = self
            .program
            .columns()
            .iter_enumerated()
            .map(|(col, c)| integrality.get(&col).copied().unwrap_or(c.integrality));
        self.supports(edited)?;
        let edit = |cols| Edit::ColIntegrality { cols };
        self.set(
            integrality,
            Program::replace_col_integrality,
            edit,
            Undo::ColIntegrality,
        )
    }

    /// [`Solver::set_row_bounds`] with each row once.
    ///
    /// # Errors
    ///
    /// As [`Solver::set_row_bounds`].
    fn apply_row_bounds(&mut self, bounds: BTreeMap<Row, Bounds>) -> Result<(), EditError<B::Error>> {
        self.sync().map_err(EditError::Backend)?;
        if let Some((&row, _)) = bounds.range(Row::new(self.program.num_rows())..).next() {
            return Err(EditError::UnknownRow { row });
        }
        let edit = |rows| Edit::RowBounds { rows };
        self.set(bounds, Program::replace_row_bounds, edit, Undo::RowBounds)
    }

    /// [`Solver::add_rows`] of rows as given.
    ///
    /// # Errors
    ///
    /// As [`Solver::add_rows`].
    fn append_rows(&mut self, rows: Vec<NewRow>) -> Result<Rows, EditError<B::Error>> {
        self.sync().map_err(EditError::Backend)?;
        let (num_cols, num_rows) = (self.program.num_cols(), self.program.num_rows());
        let rows = rows
            .into_iter()
            .zip(num_rows..)
            .map(|(new, row)| new.checked(Row::new(row), num_cols))
            .collect::<Result<Vec<_>, _>>()?;
        let added = Rows::range(num_rows, num_rows + rows.len());
        let undo = Undo::Appended {
            cols: num_cols,
            rows: num_rows,
        };
        self.commit(Edit::NewRows { rows: added.clone() }, |program| {
            program.push_rows(rows);
            undo
        })?;
        Ok(added)
    }

    /// [`Solver::add_cols`] of columns as given.
    ///
    /// # Errors
    ///
    /// As [`Solver::add_cols`].
    fn append_cols(&mut self, columns: Vec<NewCol>) -> Result<Cols, EditError<B::Error>> {
        self.sync().map_err(EditError::Backend)?;
        let (num_cols, num_rows) = (self.program.num_cols(), self.program.num_rows());
        let columns = columns
            .into_iter()
            .zip(num_cols..)
            .map(|(new, col)| new.checked(Col::new(col), num_rows))
            .collect::<Result<Vec<_>, _>>()?;
        let old = self.program.columns().iter().map(|c| c.integrality);
        self.supports(old.chain(columns.iter().map(|new| new.column.integrality)))?;
        let added = Cols::range(num_cols, num_cols + columns.len());
        let undo = Undo::Appended {
            cols: num_cols,
            rows: num_rows,
        };
        self.commit(Edit::NewCols { cols: added.clone() }, |program| {
            program.push_cols(columns);
            undo
        })?;
        Ok(added)
    }

    /// Writes each value of `given` with `replace` as one edit, which `edit` names by the ids written, through
    /// [`Solver::commit`].
    ///
    /// # Errors
    ///
    /// As [`Solver::commit`].
    ///
    /// # Panics
    ///
    /// When `replace` panics for an id of `given`, as it does for an id the program lacks; the callers checked every
    /// id.
    fn set<I: Copy + Ord, V>(
        &mut self,
        given: BTreeMap<I, V>,
        replace: fn(&mut Program, I, V) -> V,
        edit: fn(Vec<I>) -> Edit,
        undo: fn(Vec<(I, V)>) -> Undo,
    ) -> Result<(), EditError<B::Error>> {
        let ids = given.keys().copied().collect();
        self.commit(edit(ids), |program| {
            undo(
                given
                    .into_iter()
                    .map(|(id, value)| (id, replace(program, id, value)))
                    .collect(),
            )
        })
    }

    /// Unless `edit` names nothing, applies `change` to the program, which then shows `edit`, and has the backend
    /// apply `edit`; when the backend refuses it, undoes `change` with the [`Undo`] it returned and loads the program
    /// before it into the backend again.
    ///
    /// The backend counts as behind the program before `change` runs, so a panic in `change`, the backend or the
    /// undo leaves the solver to load its program before the next call that runs the backend.
    ///
    /// # Errors
    ///
    /// [`EditError::Backend`] when the backend refuses the edit, the program then as before it.
    ///
    /// # Panics
    ///
    /// When `change`, the backend or the undo panics.
    fn commit(&mut self, edit: Edit, change: impl FnOnce(&mut Program) -> Undo) -> Result<(), EditError<B::Error>> {
        if edit.is_empty() {
            return Ok(());
        }
        self.synced = false;
        self.ended = None;
        let undo = change(&mut self.program);
        match self.backend.edit(&self.program, &edit) {
            Ok(()) => {
                self.synced = true;
                Ok(())
            }
            Err(error) => {
                undo.apply(&mut self.program);
                self.synced = self.backend.load(&self.program).is_ok();
                Err(EditError::Backend(error))
            }
        }
    }

    /// Checks that the backend solves the class and takes the features the program would have with columns of
    /// these integralities, as many as it would have.
    ///
    /// # Errors
    ///
    /// [`EditError::Unsupported`] or [`EditError::UnsupportedFeature`].
    fn supports(&self, integralities: impl Iterator<Item = Integrality> + Clone) -> Result<(), EditError<B::Error>> {
        let class = self.program.class_with(integralities.clone());
        Ok(lack::<B>(class, self.program.features_with(integralities))?)
    }
}

impl Edit {
    /// Whether the edit names no id.
    fn is_empty(&self) -> bool {
        match self {
            Self::ColBounds { cols } | Self::ColCosts { cols } | Self::ColIntegrality { cols } => cols.is_empty(),
            Self::RowBounds { rows } => rows.is_empty(),
            Self::NewRows { rows } => rows.len() == 0,
            Self::NewCols { cols } => cols.len() == 0,
        }
    }
}

/// What restores a program an edit changed.
#[derive(Debug)]
enum Undo {
    /// These columns' bounds before.
    ColBounds(Vec<(Col, Bounds)>),
    /// These columns' costs before.
    ColCosts(Vec<(Col, f64)>),
    /// These columns' integralities before.
    ColIntegrality(Vec<(Col, Integrality)>),
    /// These rows' bounds before.
    RowBounds(Vec<(Row, Bounds)>),
    /// The counts before rows or columns were appended.
    Appended {
        /// The number of columns before.
        cols: usize,
        /// The number of rows before.
        rows: usize,
    },
}

impl Undo {
    /// Restores `program` to what it was before the edit.
    ///
    /// # Panics
    ///
    /// When `program` is not the program the edit made: it lacks an id the undo names, or has fewer rows than it
    /// had before the edit.
    fn apply(self, program: &mut Program) {
        /// Writes each old value back with `replace`.
        ///
        /// # Panics
        ///
        /// When `replace` panics for an id of `old`.
        fn restore<I, V>(program: &mut Program, old: Vec<(I, V)>, replace: fn(&mut Program, I, V) -> V) {
            old.into_iter().for_each(|(id, value)| {
                replace(program, id, value);
            });
        }
        match self {
            Self::ColBounds(old) => restore(program, old, Program::replace_col_bounds),
            Self::ColCosts(old) => restore(program, old, Program::replace_col_cost),
            Self::ColIntegrality(old) => restore(program, old, Program::replace_col_integrality),
            Self::RowBounds(old) => restore(program, old, Program::replace_row_bounds),
            Self::Appended { cols, rows } => program.truncate(cols, rows),
        }
    }
}

/// The values given, each id once with the last value given for it, in the order of the ids.
fn latest<I: Ord, V>(given: impl IntoIterator<Item = (I, V)>) -> BTreeMap<I, V> {
    given.into_iter().fold(BTreeMap::new(), |mut latest, (id, value)| {
        latest.insert(id, value);
        latest
    })
}

/// Checks that every column `given` names is one of the program's `num_cols`.
///
/// # Errors
///
/// [`EditError::UnknownCol`] for the first column beyond them.
fn known_cols<V, E>(given: &BTreeMap<Col, V>, num_cols: usize) -> Result<(), EditError<E>> {
    given
        .range(Col::new(num_cols)..)
        .next()
        .map_or(Ok(()), |(&col, _)| Err(EditError::UnknownCol { col }))
}

/// Checks each column `given` names as it would be with its new value, which `bad` judges from the column and the
/// value.
///
/// # Errors
///
/// [`EditError::NonFiniteCost`] or [`EditError::SemiBounds`] for the first column `bad` finds at fault.
///
/// # Panics
///
/// When `given` names a column the program lacks.
fn check_edited<V: Copy, E>(
    program: &Program,
    given: &BTreeMap<Col, V>,
    bad: impl Fn(&ProgramColumn, V) -> Option<BadColumn>,
) -> Result<(), EditError<E>> {
    let columns = program.columns();
    given
        .iter()
        .find_map(|(&col, &value)| bad(&columns[col], value).map(|fault| column_error(col, fault)))
        .map_or(Ok(()), Err)
}

/// The refusal of column `col` for `fault`.
const fn column_error<E>(col: Col, fault: BadColumn) -> EditError<E> {
    match fault {
        BadColumn::NonFiniteCost(value) => EditError::NonFiniteCost { col, value },
        BadColumn::SemiBounds(bounds) => EditError::SemiBounds { col, bounds },
    }
}

impl NewRow {
    /// The program row of this new row, which would be `row` of a program with `num_cols` columns.
    ///
    /// # Errors
    ///
    /// [`EditError::UnknownCol`] or [`EditError::NonFiniteCoefficient`] for the first term at fault, else
    /// [`EditError::RowBounds`] for a constraint whose bounds are invalid once its constant moved into them.
    fn checked<E>(self, row: Row, num_cols: usize) -> Result<ProgramRow, EditError<E>> {
        let (terms, bounds, name) = match self {
            Self::Row(ProgramRow { bounds, terms, name }) => (terms, Ok(bounds), name),
            Self::Constraint { constraint, name } => (constraint.row_terms(), constraint.row_bounds(), name),
        };
        match bad_term(&terms, num_cols) {
            Some(BadTerm::Unknown(col)) => return Err(EditError::UnknownCol { col }),
            Some(BadTerm::NonFinite(col, value)) => return Err(EditError::NonFiniteCoefficient { row, col, value }),
            None => {}
        }
        let bounds = bounds.map_err(|source| EditError::RowBounds { row, source })?;
        Ok(ProgramRow { bounds, terms, name })
    }
}

impl NewCol {
    /// This new column, which would be `col` of a program with `num_rows` rows, with its entries sorted by row,
    /// each row once with its coefficients added up, and none zero.
    ///
    /// # Errors
    ///
    /// [`EditError::NonFiniteCost`] or [`EditError::SemiBounds`] for the column, else [`EditError::UnknownRow`] or
    /// [`EditError::NonFiniteCoefficient`] for the first entry at fault.
    fn checked<E>(self, col: Col, num_rows: usize) -> Result<Self, EditError<E>> {
        let Self { column, entries } = self;
        if let Some(fault) = bad_column(column.cost, column.bounds, column.integrality) {
            return Err(column_error(col, fault));
        }
        match bad_term(&entries, num_rows) {
            Some(BadTerm::Unknown(row)) => return Err(EditError::UnknownRow { row }),
            Some(BadTerm::NonFinite(row, value)) => return Err(EditError::NonFiniteCoefficient { row, col, value }),
            None => {}
        }
        let entries = merge_terms(entries)
            .into_iter()
            .filter(|&(_, value)| value != 0.0)
            .collect();
        Ok(Self { column, entries })
    }
}
