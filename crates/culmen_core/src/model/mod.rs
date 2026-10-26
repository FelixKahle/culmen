// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The modelling layer: variables, expressions and constraints written the way the mathematics reads, built into
//! an [`Lp`].

mod error;
mod expr;
mod keyed;

use core::hash::Hash;

use typed_index_collections::TiVec;

pub use error::{ExprPlace, ModelConstraint, ModelError, ModelVar};
pub use expr::{Constraint, LinExpr, Linear, Term};
pub use keyed::{
    ConstraintMap, Handles, IndexSet, Keyed, KeyedIter, Keys, Members, Product, Values, VarMap, product, set,
};

use crate::lp::merge_terms;
use crate::{Bounds, Col, Integrality, Lp, LpColumn, LpPlan, LpRow, Row, Sense};

/// A variable to add to a [`Model`]: its integrality, bounds and name.
///
/// Built by chaining, and checked only when the model is built, where an error names the variable:
///
/// ```
/// use culmen_core::Var;
/// let stock = Var::integer().bounds(0, 40).name("stock");
/// let flow = Var::continuous().upper(12.5);
/// let open = Var::binary();
/// let slack = Var::free();
/// # let _ = (stock, flow, open, slack);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct Var {
    integrality: Integrality,
    lower: f64,
    upper: f64,
    name: Option<String>,
}

impl Var {
    /// A continuous variable, zero and above unless bounded otherwise.
    pub const fn continuous() -> Self {
        Self {
            integrality: Integrality::Continuous,
            lower: 0.0,
            upper: f64::INFINITY,
            name: None,
        }
    }

    /// A continuous variable without bounds.
    pub const fn free() -> Self {
        Self {
            integrality: Integrality::Continuous,
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            name: None,
        }
    }

    /// An integer variable, zero and above unless bounded otherwise.
    pub const fn integer() -> Self {
        Self {
            integrality: Integrality::Integer,
            lower: 0.0,
            upper: f64::INFINITY,
            name: None,
        }
    }

    /// A binary variable: integer, between zero and one.
    pub const fn binary() -> Self {
        Self {
            integrality: Integrality::Integer,
            lower: 0.0,
            upper: 1.0,
            name: None,
        }
    }

    /// The least value; `f64::NEG_INFINITY` for none.
    pub fn lower(mut self, lower: impl Into<f64>) -> Self {
        self.lower = lower.into();
        self
    }

    /// The greatest value; `f64::INFINITY` for none.
    pub fn upper(mut self, upper: impl Into<f64>) -> Self {
        self.upper = upper.into();
        self
    }

    /// Both bounds.
    pub fn bounds(self, lower: impl Into<f64>, upper: impl Into<f64>) -> Self {
        self.lower(lower).upper(upper)
    }

    /// The name in errors, logs and LP files.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl From<Bounds> for Var {
    /// A continuous variable in the bounds.
    fn from(bounds: Bounds) -> Self {
        Self::continuous().bounds(bounds.lower(), bounds.upper())
    }
}

/// A row of a [`Model`]: the constraint and its name.
#[derive(Clone, Debug, PartialEq)]
struct Named {
    constraint: Constraint,
    name: Option<String>,
}

/// A linear or mixed-integer model, written with handles, expressions and constraints and built into an [`Lp`].
///
/// Every number is stored as given and checked once, by [`Model::build`], whose error names the variable or
/// constraint at fault. The handles the model returns are the columns and rows of the LP it builds, so a solution
/// is read with them.
///
/// ```
/// use culmen_core::{Linear, LinExpr, Model, Sense, Var};
///
/// // A knapsack: take items of the most value within the capacity.
/// let (value, weight) = ([10.0, 13.0, 7.0], [5.0, 6.0, 4.0]);
/// let mut m = Model::new();
/// let take = m.add_vars(3, |i| Var::binary().name(format!("take[{i}]")));
/// let load = LinExpr::sum((0..3).map(|i| weight[i] * take[i]));
/// let cap = m.add_named_constraint("capacity", load.leq(10.0));
/// m.set_objective(Sense::Maximize, (0..3).map(|i| value[i] * take[i]).sum::<LinExpr>());
///
/// let lp = m.build()?;
/// assert_eq!((lp.num_cols(), lp.num_rows()), (3, 1));
/// assert_eq!(lp.row(cap).name(), Some("capacity"));
/// assert_eq!(lp.columns()[take[1]].cost, 13.0);
/// # Ok::<(), culmen_core::ModelError>(())
/// ```
#[derive(Clone, Debug, Default)]
pub struct Model {
    vars: TiVec<Col, Var>,
    rows: TiVec<Row, Named>,
    sense: Sense,
    objective: LinExpr,
}

impl Model {
    /// An empty model: minimize zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vars: TiVec::new(),
            rows: TiVec::new(),
            sense: Sense::Minimize,
            objective: LinExpr::new(),
        }
    }

    /// Adds a variable and returns its column; [`Bounds`] add a continuous variable in them.
    ///
    /// ```
    /// use culmen_core::{Model, Var, bounds};
    /// let mut m = Model::new();
    /// let open = m.add_var(Var::binary());
    /// let flow = m.add_var(bounds!(0.0, 12.5));
    /// assert_eq!(m.build()?.columns()[flow].bounds, bounds!(0.0, 12.5));
    /// # let _ = open;
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    pub fn add_var(&mut self, var: impl Into<Var>) -> Col {
        self.vars.push_and_get_key(var.into())
    }

    /// Adds `count` variables, the `i`-th made by `var(i)`, and returns their columns in order.
    pub fn add_vars(&mut self, count: usize, mut var: impl FnMut(usize) -> Var) -> Vec<Col> {
        (0..count).map(|i| self.add_var(var(i))).collect()
    }

    /// Adds a variable per key, made by `var(&key)`, and returns them keyed.
    ///
    /// Filter the keys with the iterator, and key a two-index family with [`product`]:
    ///
    /// ```
    /// use culmen_core::{Model, Var, product};
    /// let fits = |p: usize, l: usize| p != l;
    /// let mut m = Model::new();
    /// let x = m.add_var_map(product(&(0..3), &(0..3)).filter(|&(p, l)| fits(p, l)), |(p, l)| {
    ///     Var::binary().name(format!("x[{p},{l}]"))
    /// });
    /// assert_eq!(x.len(), 6);
    /// assert_eq!(m.var_name(x[(0, 2)]), Some("x[0,2]"));
    /// ```
    ///
    /// # Panics
    ///
    /// When `keys` repeats a key.
    #[track_caller]
    pub fn add_var_map<K: Hash + Eq + Clone>(
        &mut self,
        keys: impl IntoIterator<Item = K>,
        mut var: impl FnMut(&K) -> Var,
    ) -> VarMap<K> {
        let mut map = VarMap::new();
        for key in keys {
            let col = self.add_var(var(&key));
            map.insert(key, col);
        }
        map
    }

    /// Adds an unnamed constraint and returns its row.
    pub fn add_constraint(&mut self, constraint: Constraint) -> Row {
        self.rows.push_and_get_key(Named { constraint, name: None })
    }

    /// Adds a named constraint and returns its row.
    pub fn add_named_constraint(&mut self, name: impl Into<String>, constraint: Constraint) -> Row {
        self.rows.push_and_get_key(Named {
            constraint,
            name: Some(name.into()),
        })
    }

    /// Adds a constraint per key, named and made by `constraint(&key)`, and returns them keyed.
    ///
    /// ```
    /// use culmen_core::{Linear, LinExpr, Model, Var, product};
    /// let mut m = Model::new();
    /// let x = m.add_var_map(product(&(0..2), &(0..3)), |_| Var::binary());
    /// let once = m.add_constraint_map(0..2, |&p| {
    ///     (format!("once[{p}]"), LinExpr::sum((0..3).map(|l| x[(p, l)])).equals(1))
    /// });
    /// assert_eq!(m.constraint_name(once[1]), Some("once[1]"));
    /// ```
    ///
    /// # Panics
    ///
    /// When `keys` repeats a key.
    #[track_caller]
    pub fn add_constraint_map<K: Hash + Eq + Clone, N: Into<String>>(
        &mut self,
        keys: impl IntoIterator<Item = K>,
        mut constraint: impl FnMut(&K) -> (N, Constraint),
    ) -> ConstraintMap<K> {
        let mut map = ConstraintMap::new();
        for key in keys {
            let (name, c) = constraint(&key);
            let row = self.add_named_constraint(name, c);
            map.insert(key, row);
        }
        map
    }

    /// Sets the objective, replacing any before.
    pub fn set_objective(&mut self, sense: Sense, objective: impl Linear) {
        self.sense = sense;
        self.objective = objective.into_expr();
    }

    /// Sets the objective to minimize `objective`.
    pub fn minimize(&mut self, objective: impl Linear) {
        self.set_objective(Sense::Minimize, objective);
    }

    /// Sets the objective to maximize `objective`.
    pub fn maximize(&mut self, objective: impl Linear) {
        self.set_objective(Sense::Maximize, objective);
    }

    /// The number of variables.
    #[must_use]
    pub fn num_vars(&self) -> usize {
        self.vars.len()
    }

    /// The number of constraints.
    #[must_use]
    pub fn num_constraints(&self) -> usize {
        self.rows.len()
    }

    /// A variable's name, `None` when it has none or the model has no such variable.
    #[must_use]
    pub fn var_name(&self, col: Col) -> Option<&str> {
        self.vars.get(col)?.name.as_deref()
    }

    /// A constraint's name, `None` when it has none or the model has no such constraint.
    #[must_use]
    pub fn constraint_name(&self, row: Row) -> Option<&str> {
        self.rows.get(row)?.name.as_deref()
    }

    /// Checks every number and handle and builds the LP; the model stays as it is, to be changed and built again.
    ///
    /// These checks are the LP's own, made in the model's terms, so every model they accept builds and every error
    /// names the variable or constraint at fault.
    ///
    /// # Errors
    ///
    /// [`ModelError`] naming the first variable or constraint at fault: invalid bounds, a non-finite objective or
    /// constraint coefficient, or a handle the model did not hand out.
    pub fn build(&self) -> Result<Lp, ModelError> {
        let mut plan = LpPlan {
            sense: self.sense,
            ..LpPlan::new()
        };
        for (col, var) in self.vars.iter_enumerated() {
            let bounds = Bounds::new(var.lower, var.upper).map_err(|source| ModelError::VarBounds {
                var: self.model_var(col),
                source,
            })?;
            plan.columns.push(LpColumn {
                cost: 0.0,
                bounds,
                integrality: var.integrality,
                name: var.name.clone(),
            });
        }
        self.build_objective(&mut plan)?;
        for (row, named) in self.rows.iter_enumerated() {
            plan.rows.push(self.build_row(row, named)?);
        }
        // The checks above are the LP's own, made in the model's terms: every bound is a `Bounds`, every cost and
        // the offset finite, and every row's terms known and finite as given and as merged the way the LP merges.
        Ok(Lp::from_plan_unchecked(plan))
    }

    /// Writes the objective's coefficients and constant into the plan.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`], [`ModelError::ObjectiveCoefficient`] or [`ModelError::ObjectiveConstant`].
    fn build_objective(&self, plan: &mut LpPlan) -> Result<(), ModelError> {
        for term in &self.objective.terms {
            let column = plan.columns.get_mut(term.col).ok_or(ModelError::UnknownVar {
                place: ExprPlace::Objective,
                col: term.col,
            })?;
            column.cost += term.coefficient;
            if !column.cost.is_finite() {
                let var = self.model_var(term.col);
                return Err(ModelError::ObjectiveCoefficient {
                    var,
                    value: column.cost,
                });
            }
        }
        if !self.objective.constant.is_finite() {
            return Err(ModelError::ObjectiveConstant {
                value: self.objective.constant,
            });
        }
        plan.offset = self.objective.constant;
        Ok(())
    }

    /// One constraint as a checked plan row, its constant moved into its bounds and its repeated columns merged.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`], [`ModelError::Coefficient`] or [`ModelError::ConstraintBounds`].
    fn build_row(&self, row: Row, named: &Named) -> Result<LpRow, ModelError> {
        let c = &named.constraint;
        let terms = self.row_terms(row, &c.expr)?;
        let bounds = Bounds::new(c.lower, c.upper)
            .and_then(|b| b.shifted(-c.expr.constant))
            .map_err(|source| ModelError::ConstraintBounds {
                constraint: self.model_constraint(row),
                source,
            })?;
        Ok(LpRow {
            bounds,
            terms,
            name: named.name.clone(),
        })
    }

    /// A constraint's terms, each column once, every coefficient finite as given and as added up.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`] or [`ModelError::Coefficient`] for the first defective term.
    fn row_terms(&self, row: Row, expr: &LinExpr) -> Result<Vec<(Col, f64)>, ModelError> {
        for term in &expr.terms {
            if term.col.index() >= self.vars.len() {
                return Err(ModelError::UnknownVar {
                    place: ExprPlace::Constraint(self.model_constraint(row)),
                    col: term.col,
                });
            }
            self.check_coefficient(row, term.col, term.coefficient)?;
        }
        // The LP adds repeated columns up the same way, so a sum that passes here passes there.
        let merged = merge_terms(expr.terms.iter().map(|t| (t.col, t.coefficient)).collect());
        for &(col, value) in &merged {
            self.check_coefficient(row, col, value)?;
        }
        Ok(merged)
    }

    /// # Errors
    ///
    /// [`ModelError::Coefficient`] when `value` is NaN or infinite.
    fn check_coefficient(&self, row: Row, col: Col, value: f64) -> Result<(), ModelError> {
        if value.is_finite() {
            return Ok(());
        }
        Err(ModelError::Coefficient {
            constraint: self.model_constraint(row),
            var: self.model_var(col),
            value,
        })
    }

    /// The variable at `col` as the model names it.
    fn model_var(&self, col: Col) -> ModelVar {
        let name = self.var_name(col).map(str::to_owned);
        ModelVar { col, name }
    }

    /// The constraint at `row` as the model names it.
    fn model_constraint(&self, row: Row) -> ModelConstraint {
        let name = self.constraint_name(row).map(str::to_owned);
        ModelConstraint { row, name }
    }
}
