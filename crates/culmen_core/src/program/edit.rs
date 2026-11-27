// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A program changed in place, for a [`Solver`](crate::Solver) editing the program it holds: each change gives back
//! what undoes it.
//!
//! None of these checks its input: the solver checks an edit against the program before it changes anything, so the
//! program stays as checked as [`Program::from_plan`] leaves it.

use core::mem;

use itertools::Itertools;
use typed_index_collections::TiVec;

use super::{Bounds, Col, Integrality, NewCol, Program, ProgramRow, QuadObjective, Row, Sense};

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

    /// Sets the sense of the program's own objective and returns the sense it had.
    pub(crate) const fn replace_sense(&mut self, sense: Sense) -> Sense {
        mem::replace(&mut self.sense, sense)
    }

    /// Sets the offset of the program's own objective and returns the offset it had.
    pub(crate) const fn replace_offset(&mut self, offset: f64) -> f64 {
        mem::replace(&mut self.offset, offset)
    }

    /// Replaces the program's own objective and returns the one it had.
    ///
    /// # Panics
    ///
    /// When `objective` does not hold one cost per column of the program.
    pub(crate) fn replace_objective(&mut self, objective: OwnObjective) -> OwnObjective {
        assert_eq!(
            objective.costs.len(),
            self.num_cols(),
            "Program::replace_objective: the objective holds one cost per column"
        );
        let costs = self
            .columns
            .iter_mut()
            .zip(objective.costs)
            .map(|(column, cost)| mem::replace(&mut column.cost, cost))
            .collect();
        OwnObjective {
            sense: self.replace_sense(objective.sense),
            costs,
            offset: self.replace_offset(objective.offset),
            quadratic: mem::replace(&mut self.quadratic, objective.quadratic),
        }
    }

    /// Appends rows, as [`Program::from_plan`] appends a plan's.
    pub(crate) fn push_rows(&mut self, rows: impl IntoIterator<Item = ProgramRow>) {
        rows.into_iter().for_each(|row| self.push_row(row));
    }

    /// Appends columns, each with its entries sorted by row, each row once and none zero.
    ///
    /// # Panics
    ///
    /// When an entry names a row the program does not have.
    pub(crate) fn push_cols(&mut self, cols: impl IntoIterator<Item = NewCol>) {
        let mut added: Vec<(Row, Col, f64)> = Vec::new();
        for NewCol { column, entries } in cols {
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

/// A program's own objective: its sense, one cost per column, its offset and its quadratic part.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OwnObjective {
    pub(crate) sense: Sense,
    pub(crate) costs: TiVec<Col, f64>,
    pub(crate) offset: f64,
    pub(crate) quadratic: QuadObjective,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProgramColumn, ProgramPlan, bounds};

    /// Two columns, two rows: `x + y` in `[0, 4]`, `-x` free.
    fn program() -> Program {
        let mut plan = ProgramPlan::new();
        let x = plan.col(ProgramColumn::continuous().cost(1.0));
        let y = plan.col(ProgramColumn::continuous().cost(2.0));
        plan.row((bounds!(0.0, 4.0), [(x, 1.0), (y, 1.0)]));
        plan.row((Bounds::FREE, [(x, -1.0)]));
        plan.build().unwrap()
    }

    #[test]
    fn test_appending_columns_and_rows_and_truncating_restores_the_program() {
        let before = program();
        let mut program = before.clone();
        program.push_cols([
            NewCol {
                column: ProgramColumn::binary(),
                entries: vec![(Row::new(1), 3.0)],
            },
            NewCol {
                column: ProgramColumn::binary().cost(1.0),
                entries: vec![(Row::new(0), 2.0), (Row::new(1), -1.0)],
            },
        ]);
        assert_eq!(program.num_cols(), 4);
        program.push_rows([ProgramRow {
            bounds: Bounds::FREE,
            terms: vec![(Col::new(3), 1.0), (Col::new(0), 1.0)],
            name: None,
        }]);
        assert_eq!(program.num_rows(), 3);
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

    #[test]
    #[should_panic(expected = "one cost per column")]
    fn test_replacing_the_objective_with_too_few_costs_panics() {
        let mut program = program();
        program.replace_objective(OwnObjective {
            sense: Sense::Minimize,
            costs: TiVec::from(vec![1.0]),
            offset: 0.0,
            quadratic: QuadObjective::default(),
        });
    }
}
