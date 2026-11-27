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

/// Built by chaining from its integrality, as a [`Var`](crate::Var) is: a column costs nothing and lies in the bounds
/// its constructor names until given others, and the plan's build checks it.
///
/// ```
/// use culmen_core::{Integrality, ProgramColumn, ProgramPlan, bounds};
/// let mut plan = ProgramPlan::new();
/// let x = plan.col(ProgramColumn::continuous().cost(2.5));
/// let y = plan.col(ProgramColumn::semi_continuous().cost(1).bounds(bounds!(2.0, 10.0)).name("y"));
/// let program = plan.build()?;
/// assert_eq!(program.columns()[x].cost, 2.5);
/// assert_eq!(program.columns()[y].integrality, Integrality::SemiContinuous);
/// # Ok::<(), culmen_core::ProgramError>(())
/// ```
impl ProgramColumn {
    /// A column of `integrality` in `bounds`, costing nothing and unnamed.
    const fn of(integrality: Integrality, bounds: Bounds) -> Self {
        Self {
            cost: 0.0,
            bounds,
            integrality,
            name: None,
        }
    }

    /// A continuous column, zero and above.
    #[must_use]
    pub const fn continuous() -> Self {
        Self::of(Integrality::Continuous, Bounds::NON_NEGATIVE)
    }

    /// A continuous column without bounds.
    #[must_use]
    pub const fn free() -> Self {
        Self::of(Integrality::Continuous, Bounds::FREE)
    }

    /// An integer column, zero and above.
    #[must_use]
    pub const fn integer() -> Self {
        Self::of(Integrality::Integer, Bounds::NON_NEGATIVE)
    }

    /// A binary column: integer, between zero and one.
    #[must_use]
    pub const fn binary() -> Self {
        Self::of(Integrality::Integer, Bounds::BINARY)
    }

    /// A semi-continuous column: zero, or any value in its bounds, which it needs: the plan's build and the edits
    /// check that they are `lower ..= upper` with `0 < lower` and `upper` finite.
    #[must_use]
    pub const fn semi_continuous() -> Self {
        Self::of(Integrality::SemiContinuous, Bounds::NON_NEGATIVE)
    }

    /// A semi-integer column: zero, or a whole value in its bounds, which it needs as
    /// [`ProgramColumn::semi_continuous`] does.
    #[must_use]
    pub const fn semi_integer() -> Self {
        Self::of(Integrality::SemiInteger, Bounds::NON_NEGATIVE)
    }

    /// The objective coefficient.
    #[must_use]
    pub fn cost(mut self, cost: impl Into<f64>) -> Self {
        self.cost = cost.into();
        self
    }

    /// The values the column may take.
    #[must_use]
    pub fn bounds(mut self, bounds: impl Into<Bounds>) -> Self {
        self.bounds = bounds.into();
        self
    }

    /// The column's name in diagnostics and LP files.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// A column to append to a [`Solver`](crate::Solver)'s program or to price in: its data and its coefficients in rows
/// the program already has.
///
/// Built by chaining as a [`ProgramColumn`] is, with its entries after it, and checked when it is appended. Its
/// coefficient in every further objective is zero, and it has no quadratic term.
///
/// ```
/// use culmen_core::{Linear, Model, NewCol, Script, Solver, Var};
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
/// let both = solver.add_col(NewCol::continuous().cost(1.5).entry(a, 1).entry(b, 1))?;
/// assert_eq!(solver.program().row(a).entries().last(), Some((both, 1.0)));
///
/// // A column in no row.
/// let spare = solver.add_col(NewCol::binary())?;
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

/// The constructors and setters of [`ProgramColumn`], with entries after them.
impl NewCol {
    /// The column `column` in no row yet.
    const fn of(column: ProgramColumn) -> Self {
        Self {
            column,
            entries: Vec::new(),
        }
    }

    /// A continuous column, zero and above, as [`ProgramColumn::continuous`].
    #[must_use]
    pub const fn continuous() -> Self {
        Self::of(ProgramColumn::continuous())
    }

    /// A continuous column without bounds, as [`ProgramColumn::free`].
    #[must_use]
    pub const fn free() -> Self {
        Self::of(ProgramColumn::free())
    }

    /// An integer column, zero and above, as [`ProgramColumn::integer`].
    #[must_use]
    pub const fn integer() -> Self {
        Self::of(ProgramColumn::integer())
    }

    /// A binary column, as [`ProgramColumn::binary`].
    #[must_use]
    pub const fn binary() -> Self {
        Self::of(ProgramColumn::binary())
    }

    /// A semi-continuous column, which needs bounds, as [`ProgramColumn::semi_continuous`].
    #[must_use]
    pub const fn semi_continuous() -> Self {
        Self::of(ProgramColumn::semi_continuous())
    }

    /// A semi-integer column, which needs bounds, as [`ProgramColumn::semi_integer`].
    #[must_use]
    pub const fn semi_integer() -> Self {
        Self::of(ProgramColumn::semi_integer())
    }

    /// The cost in the program's own objective.
    #[must_use]
    pub fn cost(mut self, cost: impl Into<f64>) -> Self {
        self.column.cost = cost.into();
        self
    }

    /// The values the column may take.
    #[must_use]
    pub fn bounds(mut self, bounds: impl Into<Bounds>) -> Self {
        self.column.bounds = bounds.into();
        self
    }

    /// The column's name in diagnostics and LP files.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.column.name = Some(name.into());
        self
    }

    /// Adds the coefficient in `row`; coefficients of a repeated row add up.
    #[must_use]
    pub fn entry(mut self, row: Row, coefficient: impl Into<f64>) -> Self {
        self.entries.push((row, coefficient.into()));
        self
    }

    /// Adds a coefficient per row, as [`NewCol::entry`] adds one.
    #[must_use]
    pub fn entries<T: Into<f64>>(mut self, entries: impl IntoIterator<Item = (Row, T)>) -> Self {
        self.entries
            .extend(entries.into_iter().map(|(row, coefficient)| (row, coefficient.into())));
        self
    }
}

/// A column in no row.
impl From<ProgramColumn> for NewCol {
    fn from(column: ProgramColumn) -> Self {
        Self::of(column)
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

/// An unnamed row of its bounds and terms.
impl<B: Into<Bounds>, T: IntoIterator<Item = (Col, f64)>> From<(B, T)> for ProgramRow {
    fn from((bounds, terms): (B, T)) -> Self {
        Self {
            bounds: bounds.into(),
            terms: terms.into_iter().collect(),
            name: None,
        }
    }
}

/// A named row of its bounds and terms.
impl<N: Into<String>, B: Into<Bounds>, T: IntoIterator<Item = (Col, f64)>> From<(N, B, T)> for ProgramRow {
    fn from((name, bounds, terms): (N, B, T)) -> Self {
        Self {
            name: Some(name.into()),
            ..(bounds, terms).into()
        }
    }
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
/// let x = plan.col(ProgramColumn::continuous().cost(3.0));
/// let y = plan.col(ProgramColumn::continuous().cost(2.0));
/// plan.row((bounds!(f64::NEG_INFINITY, 4.0), [(x, 1.0), (y, 1.0)]));
/// plan.row((bounds!(f64::NEG_INFINITY, 6.0), [(x, 1.0), (y, 3.0)]));
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

    /// Adds a row and returns its handle: a [`ProgramRow`], or its parts as `(bounds, terms)` for an unnamed row and
    /// `(name, bounds, terms)` for a named one, the bounds anything that converts into [`Bounds`].
    ///
    /// ```
    /// use culmen_core::{Bounds, ProgramColumn, ProgramPlan, bounds};
    ///
    /// let mut plan = ProgramPlan::new();
    /// let x = plan.col(ProgramColumn::binary().cost(1));
    /// let free = plan.row((.., [(x, 1.0)]));
    /// let cap = plan.row(("cap", bounds!(0.0, 1.0), [(x, 2.0)]));
    /// assert_eq!(plan.rows[free].bounds, Bounds::FREE);
    /// assert_eq!(plan.rows[cap].name.as_deref(), Some("cap"));
    /// ```
    pub fn row(&mut self, row: impl Into<ProgramRow>) -> Row {
        self.rows.push_and_get_key(row.into())
    }

    /// Checks every number and every handle, every semi column's bounds, the convexity of the quadratic part and its
    /// absence beside further objectives, and builds the program.
    ///
    /// # Errors
    ///
    /// [`ProgramError`] naming the first non-finite offset, cost or coefficient (a square's coefficient whose doubled
    /// Hessian value overflows included), a semi column whose bounds are not
    /// `lower ..= upper` with `0 < lower` and `upper` finite, a term whose column the plan does not have, a quadratic
    /// part that is not convex for minimizing or concave for maximizing, or a quadratic part beside further
    /// objectives.
    pub fn build(self) -> Result<Program, ProgramError> {
        Program::from_plan(self)
    }
}
