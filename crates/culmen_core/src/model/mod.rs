// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The modelling layer: variables, expressions and constraints written the way the mathematics reads, built into
//! a [`Program`].

mod error;
mod expr;
mod keyed;
mod objective;
mod quad;

use core::hash::Hash;

use typed_index_collections::TiVec;

pub use error::{ExprPlace, ModelConstraint, ModelError, ModelVar};
pub use expr::{Constraint, LinExpr, Linear, Term};
pub use keyed::{
    ConstraintMap, Handles, IndexSet, Keyed, KeyedIter, Keys, Members, Product, Values, VarMap, product, set,
};
pub use objective::{Combination, Objective};
pub use quad::{QuadExpr, QuadTerm, Quadratic, Summand};

use crate::program::{check_convex, merge_pairs, merge_terms, semi_bounds};
use crate::{
    Bounds, Col, Integrality, LinearObjective, ObjectiveId, Objectives, Program, ProgramColumn, ProgramPlan,
    ProgramRow, QuadObjective, Row, Sense, Stage, Tolerance, Weight, Weighted,
};

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

    /// A semi-continuous variable: zero, or any value in its bounds, which it needs: `lower ..= upper` with
    /// `0 < lower` and `upper` finite, as [`Model::build`] checks.
    ///
    /// ```
    /// use culmen_core::{Model, ModelError, Var};
    /// let mut m = Model::new();
    /// m.add_var(Var::semi_continuous().bounds(2, 10).name("flow"));
    /// assert!(m.build().is_ok());
    /// m.add_var(Var::semi_continuous().name("unbounded"));
    /// assert!(matches!(m.build(), Err(ModelError::SemiBounds { .. })));
    /// ```
    pub const fn semi_continuous() -> Self {
        Self {
            integrality: Integrality::SemiContinuous,
            lower: 0.0,
            upper: f64::INFINITY,
            name: None,
        }
    }

    /// A semi-integer variable: zero, or a whole value in its bounds, which it needs as
    /// [`Var::semi_continuous`] does.
    pub const fn semi_integer() -> Self {
        Self {
            integrality: Integrality::SemiInteger,
            lower: 0.0,
            upper: f64::INFINITY,
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

/// A linear, mixed-integer or convex quadratic model, written with handles, expressions and constraints and built
/// into a [`Program`].
///
/// Constraints are linear; the objective is linear or quadratic, and a quadratic objective must be convex to
/// minimize or concave to maximize.
///
/// Every number is stored as given and checked once, by [`Model::build`], whose error names the variable or
/// constraint at fault. The handles the model returns are the columns and rows of the program it builds, so a solution
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
/// let program = m.build()?;
/// assert_eq!((program.num_cols(), program.num_rows()), (3, 1));
/// assert_eq!(program.row(cap).name(), Some("capacity"));
/// assert_eq!(program.columns()[take[1]].cost, 13.0);
/// # Ok::<(), culmen_core::ModelError>(())
/// ```
#[derive(Clone, Debug, Default)]
pub struct Model {
    vars: TiVec<Col, Var>,
    rows: TiVec<Row, Named>,
    sense: Sense,
    objective: QuadExpr,
    further: Vec<Objective>,
    combination: Combination,
    first_tolerance: Tolerance,
    first_weight: Weight,
}

impl Model {
    /// An empty model: minimize zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vars: TiVec::new(),
            rows: TiVec::new(),
            sense: Sense::Minimize,
            objective: QuadExpr::new(),
            further: Vec::new(),
            combination: Combination::Lexicographic,
            first_tolerance: Tolerance::EXACT,
            first_weight: Weight::ONE,
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

    /// Sets the objective, linear or quadratic, replacing any before.
    ///
    /// ```
    /// use culmen_core::{Model, Var};
    /// let mut m = Model::new();
    /// let x = m.add_var(Var::free());
    /// let y = m.add_var(Var::free());
    /// m.minimize(x * x + 2.0 * x * y + 2.0 * y * y - 4.0 * x + 3.0);
    /// let program = m.build()?;
    /// assert_eq!(program.quadratic().entries().collect::<Vec<_>>(), [(x, x, 1.0), (x, y, 2.0), (y, y, 2.0)]);
    /// assert_eq!((program.columns()[x].cost, program.offset()), (-4.0, 3.0));
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    pub fn set_objective(&mut self, sense: Sense, objective: impl Quadratic) {
        self.sense = sense;
        self.objective = objective.into_quad_expr();
    }

    /// Sets the objective to minimize `objective`.
    pub fn minimize(&mut self, objective: impl Quadratic) {
        self.set_objective(Sense::Minimize, objective);
    }

    /// Sets the objective to maximize `objective`.
    pub fn maximize(&mut self, objective: impl Quadratic) {
        self.set_objective(Sense::Maximize, objective);
    }

    /// Adds a further linear objective and returns its id; the model's own objective is [`ObjectiveId::FIRST`] and
    /// the further ones follow in the order added.
    ///
    /// With a further objective the model's objectives combine as [`Model::set_combination`] says, lexicographically
    /// unless set otherwise, and its own objective must be linear.
    ///
    /// ```
    /// use culmen_core::{Linear, Model, Objective, Objectives, TiSlice, Var};
    ///
    /// // Least cost first; among the cheapest plans, the least emissions.
    /// let mut m = Model::new();
    /// let coal = m.add_var(Var::continuous());
    /// let gas = m.add_var(Var::continuous());
    /// m.add_constraint((coal + gas).geq(10));
    /// m.minimize(2.0 * coal + 2.0 * gas);
    /// let co2 = m.add_objective(Objective::minimize(3.0 * coal + gas));
    /// let program = m.build()?;
    /// assert!(matches!(program.objectives(), Objectives::Lexicographic { then, .. } if then.len() == 1));
    /// assert_eq!(program.objective_values(TiSlice::from_ref(&[0.0, 10.0]))[co2], 10.0);
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    pub fn add_objective(&mut self, objective: Objective) -> ObjectiveId {
        self.further.push(objective);
        ObjectiveId::new(self.further.len())
    }

    /// How the objectives combine once there is a further one; [`Combination::Lexicographic`] until set.
    pub const fn set_combination(&mut self, combination: Combination) {
        self.combination = combination;
    }

    /// How far later stages of a lexicographic solve may move the model's own objective from its optimum;
    /// [`Tolerance::EXACT`] until set.
    pub const fn set_objective_tolerance(&mut self, tolerance: Tolerance) {
        self.first_tolerance = tolerance;
    }

    /// The weight of the model's own objective in a blend; [`Weight::ONE`] until set.
    pub const fn set_objective_weight(&mut self, weight: Weight) {
        self.first_weight = weight;
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

    /// Checks every number and handle and builds the program; the model stays as it is, to be changed and built again.
    ///
    /// These checks are the program's own, made in the model's terms, so every model they accept builds and every error
    /// names the variable, constraint or objective at fault.
    ///
    /// # Errors
    ///
    /// [`ModelError`] naming the first variable, constraint or objective at fault: invalid bounds, a semi variable
    /// whose bounds are not `lower ..= upper` with `0 < lower` and `upper` finite, a non-finite objective or
    /// constraint coefficient or constant, a handle the model did not hand out, a quadratic objective that is not
    /// convex to minimize or concave to maximize, as [`Program::CONVEXITY_TOLERANCE`] states, or a quadratic
    /// objective beside further objectives.
    pub fn build(&self) -> Result<Program, ModelError> {
        let mut plan = ProgramPlan {
            sense: self.sense,
            ..ProgramPlan::new()
        };
        plan.columns = self
            .vars
            .iter_enumerated()
            .map(|(col, var)| self.build_column(col, var))
            .collect::<Result<_, _>>()?;
        self.build_objective(&mut plan)?;
        plan.objectives = self.build_objectives()?;
        if plan.objectives.are_several() && !self.objective.terms.is_empty() {
            return Err(ModelError::QuadraticWithObjectives);
        }
        plan.rows = self
            .rows
            .iter_enumerated()
            .map(|(row, named)| self.build_row(row, named))
            .collect::<Result<_, _>>()?;
        // The checks above are the program's own, made in the model's terms: every bound is a `Bounds`, a semi
        // column's as the program needs, every cost and offset finite, every row's and objective's terms known and
        // finite as given and as merged the way the program merges, the quadratic part convex as the program checks
        // it, and absent beside further objectives: the model refuses any quadratic term there, even one that cancels.
        Ok(Program::from_plan_unchecked(plan))
    }

    /// One variable as a plan column without cost.
    ///
    /// # Errors
    ///
    /// [`ModelError::VarBounds`] or [`ModelError::SemiBounds`].
    fn build_column(&self, col: Col, var: &Var) -> Result<ProgramColumn, ModelError> {
        let bounds = Bounds::new(var.lower, var.upper).map_err(|source| ModelError::VarBounds {
            var: self.model_var(col),
            source,
        })?;
        if var.integrality.is_semi() && !semi_bounds(bounds) {
            return Err(ModelError::SemiBounds {
                var: self.model_var(col),
                bounds,
            });
        }
        Ok(ProgramColumn {
            cost: 0.0,
            bounds,
            integrality: var.integrality,
            name: var.name.clone(),
        })
    }

    /// The further objectives, checked, combined as the model's [`Combination`] says.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`], [`ModelError::FurtherCoefficient`] or [`ModelError::FurtherConstant`] for the first
    /// further objective at fault.
    fn build_objectives(&self) -> Result<Objectives, ModelError> {
        let built: Vec<(LinearObjective, &Objective)> = self
            .further
            .iter()
            .zip(1..)
            .map(|(objective, index)| Ok((self.build_further(ObjectiveId::new(index), objective)?, objective)))
            .collect::<Result<_, ModelError>>()?;
        Ok(match self.combination {
            Combination::Lexicographic => Objectives::Lexicographic {
                first: self.first_tolerance,
                then: built
                    .into_iter()
                    .map(|(objective, given)| Stage {
                        objective,
                        tolerance: given.tolerance,
                    })
                    .collect(),
            },
            Combination::Blended => Objectives::Blended {
                first: self.first_weight,
                with: built
                    .into_iter()
                    .map(|(objective, given)| Weighted {
                        objective,
                        weight: given.weight,
                    })
                    .collect(),
            },
        })
    }

    /// One further objective as plan data, each column once, every coefficient and the constant finite.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`], [`ModelError::FurtherCoefficient`] or [`ModelError::FurtherConstant`].
    fn build_further(&self, id: ObjectiveId, objective: &Objective) -> Result<LinearObjective, ModelError> {
        let expr = &objective.expr;
        let unknown = expr.terms.iter().find(|t| t.col.index() >= self.vars.len());
        if let Some(term) = unknown {
            return Err(ModelError::UnknownVar {
                place: ExprPlace::Further(id),
                col: term.col,
            });
        }
        let given = expr.terms.iter().map(|t| (t.col, t.coefficient));
        // The program adds repeated columns up the same way, so a sum that passes here passes there.
        let merged = merge_terms(given.clone().collect());
        let bad = given
            .chain(merged.iter().copied())
            .find(|(_, value)| !value.is_finite());
        if let Some((col, value)) = bad {
            let var = self.model_var(col);
            return Err(ModelError::FurtherCoefficient {
                objective: id,
                var,
                value,
            });
        }
        if !expr.constant.is_finite() {
            let value = expr.constant;
            return Err(ModelError::FurtherConstant { objective: id, value });
        }
        Ok(LinearObjective {
            sense: objective.sense,
            terms: merged,
            offset: expr.constant,
        })
    }

    /// Writes the objective's coefficients, constant and quadratic terms into the plan.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`], [`ModelError::ObjectiveCoefficient`], [`ModelError::ObjectiveConstant`],
    /// [`ModelError::ObjectiveQuadCoefficient`] or [`ModelError::NonConvexObjective`].
    fn build_objective(&self, plan: &mut ProgramPlan) -> Result<(), ModelError> {
        self.build_linear_objective(plan)?;
        plan.quadratic = self.quadratic_terms()?;
        let quadratic = QuadObjective::from_terms(plan.quadratic.iter().copied());
        check_convex(self.sense, &quadratic, self.vars.len()).map_err(|col| ModelError::NonConvexObjective {
            sense: self.sense,
            var: self.model_var(col),
        })
    }

    /// Writes the objective's linear coefficients and constant into the plan.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`], [`ModelError::ObjectiveCoefficient`] or [`ModelError::ObjectiveConstant`].
    fn build_linear_objective(&self, plan: &mut ProgramPlan) -> Result<(), ModelError> {
        let linear = &self.objective.linear;
        for term in &linear.terms {
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
        if !linear.constant.is_finite() {
            return Err(ModelError::ObjectiveConstant { value: linear.constant });
        }
        plan.offset = linear.constant;
        Ok(())
    }

    /// The objective's quadratic terms as plan terms, each known and finite as given and as merged.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`] or [`ModelError::ObjectiveQuadCoefficient`] for the first defective term.
    fn quadratic_terms(&self) -> Result<Vec<(Col, Col, f64)>, ModelError> {
        let terms: Vec<(Col, Col, f64)> = self
            .objective
            .terms
            .iter()
            .map(|t| (t.first, t.second, t.coefficient))
            .collect();
        terms.iter().try_for_each(|&(first, second, value)| {
            match [first, second].into_iter().find(|col| col.index() >= self.vars.len()) {
                Some(col) => Err(ModelError::UnknownVar {
                    place: ExprPlace::Objective,
                    col,
                }),
                None => self.check_quad_coefficient(first, second, value),
            }
        })?;
        // The program adds repeated pairs up the same way, so a sum that passes here passes there.
        merge_pairs(terms.iter().copied())
            .into_iter()
            .try_for_each(|(first, second, value)| self.check_quad_coefficient(first, second, value))?;
        Ok(terms)
    }

    /// # Errors
    ///
    /// [`ModelError::ObjectiveQuadCoefficient`] when `value` is NaN or infinite.
    fn check_quad_coefficient(&self, first: Col, second: Col, value: f64) -> Result<(), ModelError> {
        if value.is_finite() {
            return Ok(());
        }
        Err(ModelError::ObjectiveQuadCoefficient {
            first: self.model_var(first),
            second: self.model_var(second),
            value,
        })
    }

    /// One constraint as a checked plan row, its constant moved into its bounds and its repeated columns merged.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`], [`ModelError::Coefficient`] or [`ModelError::ConstraintBounds`].
    fn build_row(&self, row: Row, named: &Named) -> Result<ProgramRow, ModelError> {
        let c = &named.constraint;
        let terms = self.row_terms(row, &c.expr)?;
        let bounds = c.row_bounds().map_err(|source| ModelError::ConstraintBounds {
            constraint: self.model_constraint(row),
            source,
        })?;
        Ok(ProgramRow {
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
        // The program adds repeated columns up the same way, so a sum that passes here passes there.
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
