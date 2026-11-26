// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The columns one solve prices in, checked and numbered after the program's.

use crate::program::{BadTerm, bad_term};
use crate::{ActionError, Col, Cols, ModelClass, ModelClasses, NewCol, Program};

/// The columns one solve prices in, numbered after the program's columns: what a backend's price context hands every
/// added column to, so that the [`Solver`](crate::Solver) can append them to its program.
///
/// The [`Solver`](crate::Solver) makes one for each solve and hands it to [`Backend::solve`](crate::Backend::solve) in
/// the [`Hooks`](crate::Hooks), where a backend that prices takes it ([`Hooks::take_priced`](crate::Hooks::take_priced));
/// code that drives a backend directly makes its own, registers it with the pricer in
/// [`Hooks::register_pricer`](crate::Hooks::register_pricer), and owns the columns it holds afterwards
/// ([`Priced::into_cols`]).
///
/// ```
/// use culmen_core::{Backend, Bounds, Col, Linear, Model, NewCol, Priced, ProgramColumn, Script, Var};
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous());
/// let row = m.add_constraint(x.geq(1));
/// let program = m.build()?;
///
/// let mut priced = Priced::new(&program, Script::CLASSES);
/// let new = NewCol { column: ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE), entries: vec![(row, 1.0)] };
/// assert_eq!(priced.push(new), Ok(Col::new(1)));
/// assert_eq!(priced.num_cols(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Priced {
    /// The number of rows an entry may name.
    num_rows: usize,
    /// The number of columns the program had when the solve began; the first priced column's index.
    base: usize,
    /// The classes the backend solves.
    classes: ModelClasses,
    /// The class of the program with every column priced so far.
    class: ModelClass,
    /// The columns priced so far, each as checked.
    cols: Vec<NewCol>,
}

impl Priced {
    /// No column yet, for a solve of `program` by a backend that solves `classes`.
    #[must_use]
    pub fn new(program: &Program, classes: ModelClasses) -> Self {
        Self {
            num_rows: program.num_rows(),
            base: program.num_cols(),
            classes,
            class: program.class(),
            cols: Vec::new(),
        }
    }

    /// Checks a column as [`PriceContext::add_col`](crate::PriceContext::add_col) states, and appends it with its
    /// entries sorted by row, each row once with its coefficients added up, and none zero; its id, after the
    /// program's columns and every one priced before. A refused column takes no id.
    ///
    /// # Errors
    ///
    /// As [`PriceContext::add_col`](crate::PriceContext::add_col), except a solver's own refusal.
    pub fn push(&mut self, column: NewCol) -> Result<Col, ActionError> {
        let col = Col::new(self.num_cols());
        let class = self.check(col, &column)?;
        self.class = class;
        self.cols.push(column.merged());
        Ok(col)
    }

    /// Checks `column`, which would be `col`, and returns the class the program would have with it.
    ///
    /// # Errors
    ///
    /// As [`Priced::push`].
    fn check(&self, col: Col, column: &NewCol) -> Result<ModelClass, ActionError> {
        let NewCol { column: data, entries } = column;
        if !data.cost.is_finite() {
            return Err(ActionError::NotFinite { col });
        }
        let class = if data.integrality.is_discrete() {
            self.class.with_integer()
        } else {
            self.class
        };
        if data.integrality.is_semi() || !self.classes.contains(class) {
            return Err(ActionError::Integrality {
                col,
                integrality: data.integrality,
            });
        }
        if !data.bounds.contains(0.0) {
            return Err(ActionError::ExcludesZero {
                col,
                bounds: data.bounds,
            });
        }
        match bad_term(entries, self.num_rows) {
            Some(BadTerm::Unknown(row)) => Err(ActionError::UnknownRow { row }),
            Some(BadTerm::NonFinite(row, _)) => Err(ActionError::NotFiniteEntry { row, col }),
            None => Ok(class),
        }
    }

    /// The number of columns: the program's and those priced so far.
    #[must_use]
    pub fn num_cols(&self) -> usize {
        self.base + self.cols.len()
    }

    /// The ids priced so far, in order.
    pub fn cols(&self) -> Cols {
        Cols::range(self.base, self.num_cols())
    }

    /// A priced column as [`Priced::push`] kept it; `None` for an id it did not hand out.
    #[must_use]
    pub fn get(&self, col: Col) -> Option<&NewCol> {
        col.index().checked_sub(self.base).and_then(|i| self.cols.get(i))
    }

    /// The columns priced, in the order of their ids.
    #[must_use]
    pub fn into_cols(self) -> Vec<NewCol> {
        self.cols
    }

    /// Takes the columns priced, leaving none: the ids handed out keep their meaning only in the program the columns
    /// are appended to.
    pub(crate) fn take_cols(&mut self) -> Vec<NewCol> {
        let cols = core::mem::take(&mut self.cols);
        self.base += cols.len();
        cols
    }
}
