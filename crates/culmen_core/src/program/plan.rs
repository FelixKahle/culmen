// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The plain data a [`Program`] is built from.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::TiVec;

use super::{Bounds, Col, Integrality, Objectives, Program, ProgramError, Row, Sense};

/// A column of a [`ProgramPlan`]: its objective coefficient, bounds, integrality and name.
///
/// The cost is checked when the plan is built; everything else carries its own invariant.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProgramColumn {
    /// The objective coefficient.
    pub cost: f64,
    /// The values the column may take.
    pub bounds: Bounds,
    /// Which values in its bounds the column takes, and whether zero too.
    pub integrality: Integrality,
    /// The column's name in diagnostics and LP files; `None` names it by its index.
    pub name: Option<String>,
}

impl ProgramColumn {
    /// A continuous column.
    ///
    /// ```
    /// use culmen_core::{Bounds, ProgramColumn};
    /// let c = ProgramColumn::continuous(2.5, Bounds::NON_NEGATIVE);
    /// assert_eq!(c.cost, 2.5);
    /// ```
    #[must_use]
    pub const fn continuous(cost: f64, bounds: Bounds) -> Self {
        Self {
            cost,
            bounds,
            integrality: Integrality::Continuous,
            name: None,
        }
    }

    /// An integer column.
    #[must_use]
    pub const fn integer(cost: f64, bounds: Bounds) -> Self {
        Self {
            cost,
            bounds,
            integrality: Integrality::Integer,
            name: None,
        }
    }

    /// A binary column: integer, between zero and one.
    #[must_use]
    pub const fn binary(cost: f64) -> Self {
        Self::integer(cost, Bounds::BINARY)
    }

    /// A semi-continuous column: zero, or any value in `bounds`, which the plan's build checks are
    /// `lower ..= upper` with `0 < lower` and `upper` finite.
    ///
    /// ```
    /// use culmen_core::{Integrality, ProgramColumn, ProgramPlan, bounds};
    /// let mut plan = ProgramPlan::new();
    /// let x = plan.col(ProgramColumn::semi_continuous(1.0, bounds!(2.0, 10.0)));
    /// assert_eq!(plan.build()?.columns()[x].integrality, Integrality::SemiContinuous);
    /// # Ok::<(), culmen_core::ProgramError>(())
    /// ```
    #[must_use]
    pub const fn semi_continuous(cost: f64, bounds: Bounds) -> Self {
        Self {
            cost,
            bounds,
            integrality: Integrality::SemiContinuous,
            name: None,
        }
    }

    /// A semi-integer column: zero, or a whole value in `bounds`, checked as for
    /// [`ProgramColumn::semi_continuous`].
    #[must_use]
    pub const fn semi_integer(cost: f64, bounds: Bounds) -> Self {
        Self {
            cost,
            bounds,
            integrality: Integrality::SemiInteger,
            name: None,
        }
    }
}

/// A column to append to a [`Solver`](crate::Solver)'s program: its data and its coefficients in rows the program already has.
///
/// Its coefficient in every further objective is zero, and it has no quadratic term.
///
/// ```
/// use culmen_core::{Bounds, Linear, Model, NewCol, ProgramColumn, Script, Solver, Var};
///
/// // Cover the demands a and b, one pattern each so far.
/// let mut m = Model::new();
/// let only_a = m.add_var(Var::continuous());
/// let only_b = m.add_var(Var::continuous());
/// let a = m.add_constraint(only_a.geq(1));
/// let b = m.add_constraint(only_b.geq(1));
/// m.minimize(only_a + only_b);
/// let mut solver = Solver::new(Script::optimal(2.0, vec![1.0, 1.0]), m.build()?)?;
///
/// // A pattern covering both.
/// let both = solver.add_col(NewCol {
///     column: ProgramColumn::continuous(1.5, Bounds::NON_NEGATIVE),
///     entries: vec![(a, 1.0), (b, 1.0)],
/// })?;
/// assert_eq!(solver.program().row(a).entries().last(), Some((both, 1.0)));
///
/// // A column in no row.
/// let spare = solver.add_col(ProgramColumn::continuous(0.0, Bounds::NON_NEGATIVE))?;
/// assert_eq!(solver.program().columns()[spare].cost, 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
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

/// A row of a [`ProgramPlan`]: the bounds of a linear activity, its terms and its name.
///
/// Terms may repeat a column, and their coefficients add up when the plan is built; terms that cancel to zero are
/// dropped.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProgramRow {
    /// The interval the activity `Σ coefficient · column` must lie in.
    pub bounds: Bounds,
    /// The activity's terms as `(column, coefficient)`.
    pub terms: Vec<(Col, f64)>,
    /// The row's name in diagnostics and LP files; `None` names it by its index.
    pub name: Option<String>,
}

/// A program as plain data, checked into a [`Program`] by [`ProgramPlan::build`].
///
/// Its objective is `Σ cost · x + Σ q · xᵢ · xⱼ + offset`, the quadratic terms `(i, j, q)` in
/// [`ProgramPlan::quadratic`].
///
/// ```
/// use culmen_core::{Bounds, ProgramColumn, ProgramPlan, Sense, bounds};
///
/// // Maximize 3x + 2y subject to x + y <= 4, x + 3y <= 6, x, y >= 0.
/// let mut plan = ProgramPlan::new();
/// plan.sense = Sense::Maximize;
/// let x = plan.col(ProgramColumn::continuous(3.0, Bounds::NON_NEGATIVE));
/// let y = plan.col(ProgramColumn::continuous(2.0, Bounds::NON_NEGATIVE));
/// plan.row(bounds!(f64::NEG_INFINITY, 4.0), [(x, 1.0), (y, 1.0)]);
/// plan.row(bounds!(f64::NEG_INFINITY, 6.0), [(x, 1.0), (y, 3.0)]);
/// let program = plan.build()?;
/// assert_eq!((program.num_cols(), program.num_rows()), (2, 2));
/// # Ok::<(), culmen_core::ProgramError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProgramPlan {
    /// Whether the objective is minimized or maximized.
    pub sense: Sense,
    /// The constant added to the objective.
    pub offset: f64,
    /// The columns, in handle order.
    pub columns: TiVec<Col, ProgramColumn>,
    /// The rows, in handle order.
    pub rows: TiVec<Row, ProgramRow>,
    /// The objective's quadratic terms as `(i, j, q)`, each adding `q · xᵢ · xⱼ`; empty for a linear objective.
    ///
    /// A pair may come in either order and repeat, and its coefficients add up when the plan is built; pairs that
    /// cancel to zero are dropped. Absent from serialized data written before quadratic objectives, it reads as
    /// empty.
    #[cfg_attr(feature = "serde", serde(default))]
    pub quadratic: Vec<(Col, Col, f64)>,
    /// How the objective combines with further linear objectives; [`Objectives::Single`] for none.
    ///
    /// Absent from serialized data written before further objectives, it reads as [`Objectives::Single`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub objectives: Objectives,
}

impl ProgramPlan {
    /// An empty plan: minimize zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sense: Sense::Minimize,
            offset: 0.0,
            columns: TiVec::new(),
            rows: TiVec::new(),
            quadratic: Vec::new(),
            objectives: Objectives::Single,
        }
    }

    /// Adds a column and returns its handle.
    pub fn col(&mut self, column: ProgramColumn) -> Col {
        self.columns.push_and_get_key(column)
    }

    /// Adds an unnamed row and returns its handle.
    pub fn row(&mut self, bounds: Bounds, terms: impl IntoIterator<Item = (Col, f64)>) -> Row {
        self.rows.push_and_get_key(ProgramRow {
            bounds,
            terms: terms.into_iter().collect(),
            name: None,
        })
    }

    /// Checks every number and every handle, every semi column's bounds, the convexity of the quadratic part and its
    /// absence beside further objectives, and builds the program.
    ///
    /// # Errors
    ///
    /// [`ProgramError`] naming the first non-finite offset, cost or coefficient, a semi column whose bounds are not
    /// `lower ..= upper` with `0 < lower` and `upper` finite, a term whose column the plan does not have, a quadratic
    /// part that is not convex for minimizing or concave for maximizing, or a quadratic part beside further
    /// objectives.
    pub fn build(self) -> Result<Program, ProgramError> {
        Program::from_plan(self)
    }
}
