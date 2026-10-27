// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The HiGHS calls behind an edit of the model an [`Instance`] holds: each changes the model in place to the
//! program's new values, so HiGHS keeps its basis for the next run.

use culmen_core::{Bounds, Col, Cols, Integrality, Program, Row, Rows};
use highs_sys::{
    Highs_addCols, Highs_addRows, Highs_changeColsBoundsBySet, Highs_changeColsCostBySet,
    Highs_changeColsIntegralityByRange, Highs_changeColsIntegralityBySet, Highs_changeRowsBoundsBySet, HighsInt,
};

use super::{Instance, index, var_type};
use crate::{CallStatus, Dimension, HighsError};

impl Instance {
    /// Sets the bounds of columns `cols` of the model HiGHS holds to their bounds in `program`.
    ///
    /// # Errors
    ///
    /// [`HighsError::Edit`] when HiGHS refuses the change.
    pub(crate) fn change_col_bounds(&mut self, program: &Program, cols: &[Col]) -> Result<(), HighsError> {
        let set = indices(cols, Dimension::Columns)?;
        let (lower, upper) = ends(cols.iter().map(|&col| program.columns()[col].bounds));
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(self.holds(cols), "Instance: a column HiGHS's model lacks");
        // SAFETY: the instance is live and borrowed exclusively. `set`, `lower` and `upper` hold `set.count` values
        // each, every index a column of the model; HiGHS reads them during the call.
        let status = unsafe {
            Highs_changeColsBoundsBySet(
                self.raw(),
                set.count,
                set.indices.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
            )
        };
        edited(status, "Highs_changeColsBoundsBySet")
    }

    /// Sets the costs of columns `cols` of the model HiGHS holds to their costs in `program`.
    ///
    /// # Errors
    ///
    /// [`HighsError::Edit`] when HiGHS refuses the change.
    pub(crate) fn change_col_costs(&mut self, program: &Program, cols: &[Col]) -> Result<(), HighsError> {
        let set = indices(cols, Dimension::Columns)?;
        let costs: Vec<f64> = cols.iter().map(|&col| program.columns()[col].cost).collect();
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(self.holds(cols), "Instance: a column HiGHS's model lacks");
        // SAFETY: the instance is live and borrowed exclusively. `set` and `costs` hold `set.count` values each, every
        // index a column of the model; HiGHS reads them during the call.
        let status = unsafe { Highs_changeColsCostBySet(self.raw(), set.count, set.indices.as_ptr(), costs.as_ptr()) };
        edited(status, "Highs_changeColsCostBySet")
    }

    /// Sets the integrality of columns `cols` of the model HiGHS holds to their integrality in `program`.
    ///
    /// # Errors
    ///
    /// [`HighsError::Edit`] when HiGHS refuses the change.
    pub(crate) fn change_col_integrality(&mut self, program: &Program, cols: &[Col]) -> Result<(), HighsError> {
        let set = indices(cols, Dimension::Columns)?;
        let types: Vec<HighsInt> = cols
            .iter()
            .map(|&col| var_type(program.columns()[col].integrality))
            .collect();
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(self.holds(cols), "Instance: a column HiGHS's model lacks");
        // SAFETY: the instance is live and borrowed exclusively. `set` and `types` hold `set.count` values each, every
        // index a column of the model and every type a HiGHS variable type; HiGHS reads them during the call.
        let status =
            unsafe { Highs_changeColsIntegralityBySet(self.raw(), set.count, set.indices.as_ptr(), types.as_ptr()) };
        edited(status, "Highs_changeColsIntegralityBySet")
    }

    /// Sets the bounds of rows `rows` of the model HiGHS holds to their bounds in `program`.
    ///
    /// # Errors
    ///
    /// [`HighsError::Edit`] when HiGHS refuses the change.
    pub(crate) fn change_row_bounds(&mut self, program: &Program, rows: &[Row]) -> Result<(), HighsError> {
        let set = indices(rows, Dimension::Rows)?;
        let (lower, upper) = ends(rows.iter().map(|&row| program.row_bounds()[row]));
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            rows.iter().all(|row| row.index() < self.num_row()),
            "Instance: a row HiGHS's model lacks"
        );
        // SAFETY: the instance is live and borrowed exclusively. `set`, `lower` and `upper` hold `set.count` values
        // each, every index a row of the model; HiGHS reads them during the call.
        let status = unsafe {
            Highs_changeRowsBoundsBySet(
                self.raw(),
                set.count,
                set.indices.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
            )
        };
        edited(status, "Highs_changeRowsBoundsBySet")
    }

    /// Appends `rows`, the last rows of `program`, to the model HiGHS holds, which holds `program`'s rows before
    /// them and no other.
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] when `program` exceeds HiGHS's 32-bit indices, [`HighsError::Edit`] when HiGHS
    /// refuses the rows.
    pub(crate) fn add_rows(&mut self, program: &Program, rows: Rows) -> Result<(), HighsError> {
        let new = NewRows::of(program, rows)?;
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            self.num_row() + new.lower.len() == program.num_rows() && self.num_col() == program.num_cols(),
            "Instance: HiGHS's model is not the program before its new rows"
        );
        debug_assert!(new.agrees(), "Instance: the new rows' arrays disagree");
        // SAFETY: the instance is live and borrowed exclusively. The bounds and starts hold `new.count` values each,
        // the columns and values `new.entries`, every start at most `new.entries` and every column one of the model;
        // HiGHS copies them before returning.
        let status = unsafe {
            Highs_addRows(
                self.raw(),
                new.count,
                new.lower.as_ptr(),
                new.upper.as_ptr(),
                new.entries,
                new.starts.as_ptr(),
                new.lines.as_ptr(),
                new.values.as_ptr(),
            )
        };
        edited(status, "Highs_addRows")
    }

    /// Appends `cols`, the last columns of `program`, with their entries and integrality, to the model HiGHS holds,
    /// which holds `program`'s columns before them and all its rows.
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] when `program` exceeds HiGHS's 32-bit indices, [`HighsError::Edit`] when HiGHS
    /// refuses the columns or their integrality.
    pub(crate) fn add_cols(&mut self, program: &Program, cols: Cols) -> Result<(), HighsError> {
        let new = NewCols::of(program, cols)?;
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            self.num_col() + new.costs.len() == program.num_cols() && self.num_row() >= program.num_rows(),
            "Instance: HiGHS's model is not the program before its new columns"
        );
        debug_assert!(new.agrees(), "Instance: the new columns' arrays disagree");
        // SAFETY: the instance is live and borrowed exclusively. The costs, bounds and starts hold `new.count` values
        // each, the rows and values `new.entries`, every start at most `new.entries` and every row one of the model;
        // HiGHS copies them before returning.
        let status = unsafe {
            Highs_addCols(
                self.raw(),
                new.count,
                new.costs.as_ptr(),
                new.lower.as_ptr(),
                new.upper.as_ptr(),
                new.entries,
                new.starts.as_ptr(),
                new.lines.as_ptr(),
                new.values.as_ptr(),
            )
        };
        edited(status, "Highs_addCols")?;
        self.set_new_integrality(&new)
    }

    /// Gives the columns `new` appended their integrality, unless every one is continuous, as HiGHS adds them.
    ///
    /// # Errors
    ///
    /// [`HighsError::Edit`] when HiGHS refuses the integrality.
    fn set_new_integrality(&mut self, new: &NewCols) -> Result<(), HighsError> {
        if new.types.iter().all(|&t| t == var_type(Integrality::Continuous)) {
            return Ok(());
        }
        debug_assert!(!self.raw().is_null(), "Instance: the HiGHS instance is null");
        debug_assert!(
            usize::try_from(new.last).is_ok_and(|last| last < self.num_col()) && new.first <= new.last,
            "Instance: the new columns are not the model's last"
        );
        // SAFETY: the instance is live and borrowed exclusively. `new.first ..= new.last` are columns of the model, and
        // `types` holds one HiGHS variable type for each; HiGHS reads them during the call.
        let status = unsafe { Highs_changeColsIntegralityByRange(self.raw(), new.first, new.last, new.types.as_ptr()) };
        edited(status, "Highs_changeColsIntegralityByRange")
    }

    /// Whether every column of `cols` is one of the model's.
    fn holds(&self, cols: &[Col]) -> bool {
        let num_col = self.num_col();
        cols.iter().all(|col| col.index() < num_col)
    }
}

/// The ids a `BySet` call changes, as HiGHS's indices.
struct Set {
    count: HighsInt,
    indices: Vec<HighsInt>,
}

/// The ids `lines` as HiGHS's indices, each a column or row of the model HiGHS holds.
///
/// # Errors
///
/// [`HighsError::TooLarge`] when a count or an index exceeds HiGHS's 32-bit indices, which a model HiGHS holds never
/// does.
fn indices<I: Copy + Into<usize>>(lines: &[I], what: Dimension) -> Result<Set, HighsError> {
    Ok(Set {
        count: index(lines.len(), what)?,
        indices: lines
            .iter()
            .map(|&line| index(line.into(), what))
            .collect::<Result<_, _>>()?,
    })
}

/// The lower and the upper ends of `bounds`.
fn ends(bounds: impl Iterator<Item = Bounds>) -> (Vec<f64>, Vec<f64>) {
    bounds.map(|b| (b.lower(), b.upper())).unzip()
}

/// The result of an edit call that returned `status`.
///
/// # Errors
///
/// [`HighsError::Edit`] naming `call` when HiGHS reports an error.
fn edited(status: HighsInt, call: &'static str) -> Result<(), HighsError> {
    match CallStatus::from_raw(status) {
        CallStatus::Error => Err(HighsError::Edit { call }),
        CallStatus::Ok | CallStatus::Warning => Ok(()),
    }
}

/// A program's last rows in the layout `Highs_addRows` reads: row-wise, starts relative to the first new row.
struct NewRows {
    count: HighsInt,
    entries: HighsInt,
    lower: Vec<f64>,
    upper: Vec<f64>,
    starts: Vec<HighsInt>,
    lines: Vec<HighsInt>,
    values: Vec<f64>,
}

impl NewRows {
    /// The rows `rows`, the last of `program`.
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] when the program's rows or entries exceed HiGHS's 32-bit indices.
    fn of(program: &Program, rows: Rows) -> Result<Self, HighsError> {
        index(program.num_rows(), Dimension::Rows)?;
        index(program.num_entries(), Dimension::Entries)?;
        let first = program.num_rows() - rows.len();
        let matrix = program.matrix();
        let base = matrix.starts[first];
        let (lower, upper) = ends(rows.map(|row| program.row_bounds()[row]));
        Ok(Self {
            count: index(lower.len(), Dimension::Rows)?,
            entries: index(matrix.values.len() - base, Dimension::Entries)?,
            lower,
            upper,
            starts: positions(&matrix.starts[first..program.num_rows()], base)?,
            lines: matrix.cols[base..]
                .iter()
                .map(|&col| index(col.index(), Dimension::Columns))
                .collect::<Result<_, _>>()?,
            values: matrix.values[base..].to_vec(),
        })
    }

    /// Whether every array holds what `Highs_addRows` reads.
    fn agrees(&self) -> bool {
        let (rows, entries) = (self.lower.len(), self.values.len());
        index(rows, Dimension::Rows).is_ok_and(|count| count == self.count)
            && index(entries, Dimension::Entries).is_ok_and(|count| count == self.entries)
            && self.upper.len() == rows
            && self.starts.len() == rows
            && self.lines.len() == entries
    }
}

/// A program's last columns in the layout `Highs_addCols` reads: column-wise, starts relative to the first new
/// column; and their HiGHS variable types for `Highs_changeColsIntegralityByRange` from `first` to `last`.
struct NewCols {
    count: HighsInt,
    entries: HighsInt,
    costs: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    starts: Vec<HighsInt>,
    lines: Vec<HighsInt>,
    values: Vec<f64>,
    first: HighsInt,
    last: HighsInt,
    types: Vec<HighsInt>,
}

impl NewCols {
    /// The columns `cols`, the last of `program`, with their entries, which end their rows.
    ///
    /// # Errors
    ///
    /// [`HighsError::TooLarge`] when the program's columns or entries exceed HiGHS's 32-bit indices.
    fn of(program: &Program, cols: Cols) -> Result<Self, HighsError> {
        index(program.num_cols(), Dimension::Columns)?;
        index(program.num_entries(), Dimension::Entries)?;
        let first = program.num_cols() - cols.len();
        let entries = new_entries(program, Col::new(first));
        let columns = &program.columns()[Col::new(first)..];
        let (lower, upper) = ends(columns.iter().map(|c| c.bounds));
        let starts: Vec<usize> = (first..program.num_cols())
            .map(|col| entries.partition_point(|e| e.0.index() < col))
            .collect();
        Ok(Self {
            count: index(columns.len(), Dimension::Columns)?,
            entries: index(entries.len(), Dimension::Entries)?,
            costs: columns.iter().map(|c| c.cost).collect(),
            lower,
            upper,
            starts: positions(&starts, 0)?,
            lines: entries
                .iter()
                .map(|e| index(e.1.index(), Dimension::Rows))
                .collect::<Result<_, _>>()?,
            values: entries.iter().map(|e| e.2).collect(),
            first: index(first, Dimension::Columns)?,
            last: index(program.num_cols().saturating_sub(1), Dimension::Columns)?,
            types: columns.iter().map(|c| var_type(c.integrality)).collect(),
        })
    }

    /// Whether every array holds what `Highs_addCols` and `Highs_changeColsIntegralityByRange` read.
    fn agrees(&self) -> bool {
        let (cols, entries) = (self.costs.len(), self.values.len());
        index(cols, Dimension::Columns).is_ok_and(|count| count == self.count)
            && index(entries, Dimension::Entries).is_ok_and(|count| count == self.entries)
            && [&self.lower, &self.upper].iter().all(|a| a.len() == cols)
            && self.starts.len() == cols
            && self.types.len() == cols
            && self.lines.len() == entries
    }
}

/// The entries of every column from `first` on, as `(column, row, coefficient)` sorted by column and then by row.
fn new_entries(program: &Program, first: Col) -> Vec<(Col, Row, f64)> {
    let matrix = program.matrix();
    let mut entries: Vec<(Col, Row, f64)> = program
        .rows()
        .flat_map(|row| {
            let (from, to) = (matrix.starts[row.index()], matrix.starts[row.index() + 1]);
            // A row's entries are sorted by column, so its new columns' entries end it.
            let tail = from + matrix.cols[from..to].partition_point(|&col| col < first);
            (tail..to).map(move |entry| (matrix.cols[entry], row, matrix.values[entry]))
        })
        .collect();
    // A stable sort keeps each column's entries in row order.
    entries.sort_by_key(|e| e.0);
    entries
}

/// Each position minus `base`, as HiGHS's index type.
///
/// # Errors
///
/// [`HighsError::TooLarge`] when a position exceeds HiGHS's 32-bit indices.
fn positions(starts: &[usize], base: usize) -> Result<Vec<HighsInt>, HighsError> {
    starts
        .iter()
        .map(|&start| index(start - base, Dimension::Entries))
        .collect()
}

#[cfg(test)]
mod tests {
    use culmen_core::{
        Bounds, Integrality, Linear, Model, NewCol, NewRow, Program, ProgramColumn, ProgramPlan, Solver, Var, bounds,
    };

    use crate::Highs;
    use crate::files::program_of;

    /// The model HiGHS holds for `solver`, read back as a program without names and without the free row of a QP
    /// without rows.
    fn held(solver: &Solver<Highs>) -> Program {
        let mut model = solver.backend().instance.model();
        let rows = solver.program().num_rows();
        model.row_lower.truncate(rows);
        model.row_upper.truncate(rows);
        model.starts.truncate(rows + 1);
        let cols = model.costs.len();
        program_of(&model, vec![None; cols], vec![None; rows]).unwrap()
    }

    /// The solver's program without names, as HiGHS holds it.
    fn unnamed(solver: &Solver<Highs>) -> Program {
        let mut plan = ProgramPlan::from(solver.program().clone());
        plan.columns.iter_mut().for_each(|c| c.name = None);
        plan.rows.iter_mut().for_each(|r| r.name = None);
        plan.build().unwrap()
    }

    #[test]
    fn test_highs_holds_the_program_after_each_kind_of_edit_of_an_lp() {
        let mut m = Model::new();
        let x = m.add_var(Var::continuous().upper(4).name("x"));
        let y = m.add_var(Var::continuous().upper(4));
        let cap = m.add_named_constraint("cap", (x + y).leq(5));
        let floor = m.add_constraint((x - y).geq(-1));
        m.maximize(x + 2.0 * y);
        let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
        assert_eq!(held(&solver), unnamed(&solver));

        solver
            .set_col_bounds([(x, bounds!(1.0, 3.0)), (y, bounds!(0.5, 2.0))])
            .unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "column bounds");
        solver.set_col_costs([(y, -1.0)]).unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "costs");
        solver
            .set_col_integrality([(x, Integrality::SemiContinuous), (y, Integrality::Integer)])
            .unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "integrality");
        solver
            .set_row_bounds([(floor, Bounds::FREE), (cap, bounds!(1.0, 6.0))])
            .unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "row bounds");
        let rows = solver
            .add_rows([NewRow::from((2.0 * x - y + 1).leq(3)), NewRow::from(("lone", y.geq(0)))])
            .unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "rows");
        solver
            .set_row_bounds(rows.map(|row| (row, bounds!(f64::NEG_INFINITY, 2.0))))
            .unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "the new rows' bounds");
        let both = NewCol {
            column: ProgramColumn::integer(1.0, bounds!(0.0, 5.0)),
            entries: vec![(cap, 1.0), (floor, -2.0)],
        };
        let cols = solver
            .add_cols([both, NewCol::from(ProgramColumn::continuous(0.0, Bounds::FREE))])
            .unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "columns");
        solver.set_col_costs(cols.map(|col| (col, 3.0))).unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "the new columns' costs");
    }

    #[test]
    fn test_highs_holds_the_program_after_columns_and_a_row_are_added_to_a_qp_without_rows() {
        let mut m = Model::new();
        let x = m.add_var(Var::free());
        m.minimize(x * x - 2.0 * x);
        let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
        let y = solver
            .add_col(ProgramColumn::continuous(1.0, bounds!(0.0, 2.0)))
            .unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "a column beside the free row");
        solver.add_row((x + y).geq(1)).unwrap();
        assert_eq!(held(&solver), unnamed(&solver), "a row in place of the free row");
        assert_eq!(solver.backend().instance.num_row(), 1);
    }
}
