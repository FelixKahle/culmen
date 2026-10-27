// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A program changed in place, for a [`Solver`](crate::Solver) editing the program it holds: each change gives back
//! what undoes it.
//!
//! None of these checks its input: the solver checks an edit against the program before it changes anything, so the
//! program stays as checked as [`Program::from_plan`] leaves it.

use core::mem;

use itertools::Itertools;

use super::{Bounds, Col, Cols, Integrality, Program, ProgramColumn, ProgramRow, Row, Rows};

/// A column to append and its entries as `(row, coefficient)`, sorted by row, each row once and none zero.
pub(crate) type ColumnEntries = (ProgramColumn, Vec<(Row, f64)>);

impl Program {
    /// Sets a column's bounds and returns the bounds it had.
    ///
    /// # Panics
    ///
    /// When the program has no column `col`.
    pub(crate) fn replace_col_bounds(&mut self, col: Col, bounds: Bounds) -> Bounds {
        mem::replace(&mut self.columns[col].bounds, bounds)
    }

    /// Sets a column's cost in the program's own objective and returns the cost it had.
    ///
    /// # Panics
    ///
    /// When the program has no column `col`.
    pub(crate) fn replace_col_cost(&mut self, col: Col, cost: f64) -> f64 {
        mem::replace(&mut self.columns[col].cost, cost)
    }

    /// Sets a column's integrality and returns the integrality it had.
    ///
    /// # Panics
    ///
    /// When the program has no column `col`.
    pub(crate) fn replace_col_integrality(&mut self, col: Col, integrality: Integrality) -> Integrality {
        mem::replace(&mut self.columns[col].integrality, integrality)
    }

    /// Sets a row's bounds and returns the bounds it had.
    ///
    /// # Panics
    ///
    /// When the program has no row `row`.
    pub(crate) fn replace_row_bounds(&mut self, row: Row, bounds: Bounds) -> Bounds {
        mem::replace(&mut self.row_bounds[row], bounds)
    }

    /// Appends rows, as [`Program::from_plan`] appends a plan's, and returns them.
    pub(crate) fn push_rows(&mut self, rows: impl IntoIterator<Item = ProgramRow>) -> Rows {
        let start = self.num_rows();
        rows.into_iter().for_each(|row| self.push_row(row));
        Rows::range(start, self.num_rows())
    }

    /// Appends columns, each with its entries, and returns them.
    ///
    /// # Panics
    ///
    /// When an entry names a row the program does not have.
    pub(crate) fn push_cols(&mut self, cols: impl IntoIterator<Item = ColumnEntries>) -> Cols {
        let start = self.num_cols();
        let mut added: Vec<(Row, Col, f64)> = Vec::new();
        for (column, entries) in cols {
            let col = self.columns.push_and_get_key(column);
            added.extend(entries.into_iter().map(|(row, value)| (row, col, value)));
        }
        debug_assert!(
            added
                .iter()
                .all(|&(row, _, value)| row.index() < self.num_rows() && value != 0.0),
            "Program: a new column's entry names no row or is zero"
        );
        // A stable sort keeps each row's new entries in column order, which every new column follows.
        added.sort_by_key(|&(row, ..)| row);
        self.append_to_rows(&added);
        Cols::range(start, self.num_cols())
    }

    /// Appends `added`, sorted by row and within a row by column, each beyond every column the row holds, to the
    /// end of its row.
    fn append_to_rows(&mut self, added: &[(Row, Col, f64)]) {
        if added.is_empty() {
            return;
        }
        let total = self.num_entries() + added.len();
        let (mut cols, mut values) = (Vec::with_capacity(total), Vec::with_capacity(total));
        let mut starts = Vec::with_capacity(self.row_starts.len());
        starts.push(0);
        let mut added = added.iter().peekable();
        for (row, (&from, &to)) in self.row_starts.iter().tuple_windows().enumerate() {
            cols.extend_from_slice(&self.entry_cols[from..to]);
            values.extend_from_slice(&self.entry_values[from..to]);
            for &(_, col, value) in added.peeking_take_while(|entry| entry.0.index() == row) {
                cols.push(col);
                values.push(value);
            }
            starts.push(values.len());
        }
        self.row_starts = starts;
        self.entry_cols = cols;
        self.entry_values = values;
    }

    /// Removes every column from `num_cols` on, with its entries, and every row from `num_rows` on: the program as it
    /// was before columns and rows were appended to its first `num_cols` columns and `num_rows` rows.
    ///
    /// # Panics
    ///
    /// When the program has fewer than `num_rows` rows.
    pub(crate) fn truncate(&mut self, num_cols: usize, num_rows: usize) {
        self.row_bounds.truncate(num_rows);
        self.row_names.truncate(num_rows);
        self.row_starts.truncate(num_rows + 1);
        let entries = self.row_starts[num_rows];
        self.entry_cols.truncate(entries);
        self.entry_values.truncate(entries);
        if num_cols < self.num_cols() {
            self.columns.truncate(num_cols);
            self.drop_entries_from(Col::new(num_cols));
        }
    }

    /// Removes every entry of a column from `first` on; such entries end their rows, which are sorted by column.
    fn drop_entries_from(&mut self, first: Col) {
        let (mut cols, mut values) = (Vec::new(), Vec::new());
        let mut starts = Vec::with_capacity(self.row_starts.len());
        starts.push(0);
        for (&from, &to) in self.row_starts.iter().tuple_windows() {
            let end = from + self.entry_cols[from..to].partition_point(|&col| col < first);
            cols.extend_from_slice(&self.entry_cols[from..end]);
            values.extend_from_slice(&self.entry_values[from..end]);
            starts.push(values.len());
        }
        self.row_starts = starts;
        self.entry_cols = cols;
        self.entry_values = values;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProgramPlan, bounds};

    /// Two columns, two rows: `x + y` in `[0, 4]`, `-x` free.
    fn program() -> Program {
        let mut plan = ProgramPlan::new();
        let x = plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
        let y = plan.col(ProgramColumn::continuous(2.0, Bounds::NON_NEGATIVE));
        plan.row(bounds!(0.0, 4.0), [(x, 1.0), (y, 1.0)]);
        plan.row(Bounds::FREE, [(x, -1.0)]);
        plan.build().unwrap()
    }

    #[test]
    fn test_appending_columns_and_rows_and_truncating_restores_the_program() {
        let before = program();
        let mut program = before.clone();
        let cols = program.push_cols([
            (ProgramColumn::binary(0.0), vec![(Row::new(1), 3.0)]),
            (
                ProgramColumn::binary(1.0),
                vec![(Row::new(0), 2.0), (Row::new(1), -1.0)],
            ),
        ]);
        assert_eq!(cols.collect::<Vec<_>>(), [Col::new(2), Col::new(3)]);
        let rows = program.push_rows([ProgramRow {
            bounds: Bounds::FREE,
            terms: vec![(Col::new(3), 1.0), (Col::new(0), 1.0)],
            name: None,
        }]);
        assert_eq!(rows.collect::<Vec<_>>(), [Row::new(2)]);
        assert_eq!(
            program.row(Row::new(1)).entries().collect::<Vec<_>>(),
            [(Col::new(0), -1.0), (Col::new(2), 3.0), (Col::new(3), -1.0)]
        );
        assert_eq!(
            program.row(Row::new(2)).entries().collect::<Vec<_>>(),
            [(Col::new(0), 1.0), (Col::new(3), 1.0)]
        );
        program.truncate(2, 2);
        assert_eq!(program, before);
    }
}
