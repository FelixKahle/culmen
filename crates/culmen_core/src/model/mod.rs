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
use objective::Rank;
pub use objective::{Combination, Objective};
pub use quad::{QuadExpr, QuadTerm, Quadratic, Summand};

use crate::program::{check_convex, hessian_value, merge_pairs, merge_terms, semi_bounds};
use crate::{
    Bounds, Col, Integrality, LinearObjective, ObjectiveId, Objectives, Program, ProgramColumn, ProgramPlan,
    ProgramRow, QuadObjective, Row, Sense, Stage, Tolerance, Weight, Weighted,
};

/// A variable to add to a [`Model`]: its integrality, bounds and name.
///
/// Built by chaining; bounds set with [`Var::lower`] and [`Var::upper`] are checked only when the model is built,
/// where an error names the variable:
///
/// ```
/// use culmen_core::{Var, bounds};
/// let stock = Var::integer().bounds(bounds!(0.0, 40.0)).name("stock");
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
    /// use culmen_core::{Model, ModelError, Var, bounds};
    /// let mut m = Model::new();
    /// m.add_var(Var::semi_continuous().bounds(bounds!(2.0, 10.0)).name("flow"));
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

    /// Both bounds, an interval checked where it is made, as a [`ProgramColumn`]'s and a [`NewCol`]'s are.
    ///
    /// [`NewCol`]: crate::NewCol
    pub fn bounds(self, bounds: impl Into<Bounds>) -> Self {
        let bounds = bounds.into();
        self.lower(bounds.lower()).upper(bounds.upper())
    }

    /// The least value; `f64::NEG_INFINITY` for none. A number known only at run time, which [`Model::build`] checks
    /// with the upper bound, naming the variable.
    pub fn lower(mut self, lower: impl Into<f64>) -> Self {
        self.lower = lower.into();
        self
    }

    /// The greatest value; `f64::INFINITY` for none, checked as [`Var::lower`] is.
    pub fn upper(mut self, upper: impl Into<f64>) -> Self {
        self.upper = upper.into();
        self
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
        Self::continuous().bounds(bounds)
    }
}

/// A constraint to add to a [`Model`] or append to a [`Solver`](crate::Solver)'s program: a model constraint, named or
/// not, or a program row. [`Model::add_constraint`] and [`Solver::add_row`](crate::Solver::add_row) take anything that
/// converts into one, so both read alike.
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
    /// A constraint as a model writes it; its constant moves into its bounds as [`Model::build`] moves it.
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

/// A named row: `("capacity", load.leq(10))`.
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

impl NewRow {
    /// The row's name; `None` names it by its index.
    fn name(&self) -> Option<&str> {
        match self {
            Self::Constraint { name, .. } | Self::Row(ProgramRow { name, .. }) => name.as_deref(),
        }
    }
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
/// let cap = m.add_constraint(("capacity", load.leq(10.0)));
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
    rows: TiVec<Row, NewRow>,
    sense: Sense,
    objective: QuadExpr,
    /// The further objectives, each with its rank.
    further: Vec<(Objective, Rank)>,
    /// The rank of the model's own objective; `None` until set, when the further objectives' rank decides.
    first: Option<Rank>,
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
            first: None,
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

    /// Adds `count` variables, the `i`-th made by `var(i)`, and returns their columns in order; like
    /// [`Model::add_var`], the closure may return [`Bounds`] for a continuous variable in them.
    ///
    /// ```
    /// use culmen_core::{Model, Var, bounds};
    /// let mut m = Model::new();
    /// let take = m.add_vars(3, |i| Var::binary().name(format!("take[{i}]")));
    /// let flow = m.add_vars(2, |_| bounds!(0.0, 4.0));
    /// assert_eq!((take.len(), m.build()?.columns()[flow[1]].bounds), (3, bounds!(0.0, 4.0)));
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    pub fn add_vars<V: Into<Var>>(&mut self, count: usize, mut var: impl FnMut(usize) -> V) -> Vec<Col> {
        (0..count).map(|i| self.add_var(var(i))).collect()
    }

    /// Adds a variable per key, made by `var(&key)` as a [`Var`] or [`Bounds`], and returns them keyed.
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
    pub fn add_var_map<K: Hash + Eq + Clone, V: Into<Var>>(
        &mut self,
        keys: impl IntoIterator<Item = K>,
        mut var: impl FnMut(&K) -> V,
    ) -> VarMap<K> {
        let mut map = VarMap::new();
        for key in keys {
            let col = self.add_var(var(&key));
            map.insert(key, col);
        }
        map
    }

    /// Adds a constraint and returns its row: a [`Constraint`], a named one as `(name, constraint)`, or a
    /// [`ProgramRow`], as [`NewRow`] lists.
    ///
    /// ```
    /// use culmen_core::{Linear, Model, Var};
    /// let mut m = Model::new();
    /// let x = m.add_var(Var::continuous());
    /// let cap = m.add_constraint(("cap", x.leq(4)));
    /// let floor = m.add_constraint(x.geq(1));
    /// assert_eq!((m.constraint_name(cap), m.constraint_name(floor)), (Some("cap"), None));
    /// ```
    pub fn add_constraint(&mut self, constraint: impl Into<NewRow>) -> Row {
        self.rows.push_and_get_key(constraint.into())
    }

    /// Adds a constraint per key, made by `constraint(&key)` as [`Model::add_constraint`] takes it, usually named as
    /// `(name, constraint)`, and returns them keyed.
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
    pub fn add_constraint_map<K: Hash + Eq + Clone, R: Into<NewRow>>(
        &mut self,
        keys: impl IntoIterator<Item = K>,
        mut constraint: impl FnMut(&K) -> R,
    ) -> ConstraintMap<K> {
        let mut map = ConstraintMap::new();
        for key in keys {
            let row = self.add_constraint(constraint(&key));
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

    /// Adds a further linear objective as the next stage of a lexicographic solve and returns its id; the model's own
    /// objective is [`ObjectiveId::FIRST`] and the further ones follow in the order added.
    ///
    /// The objectives are optimized one after another in that order, and `tolerance` says how far the later stages
    /// may move this one from the optimum its stage finds: [`Tolerance::EXACT`] for not at all. The model's own
    /// objective must then be linear, and every further objective a stage: [`Model::build`] refuses a model that
    /// also has a weighted objective.
    ///
    /// ```
    /// use culmen_core::{Linear, Model, Objective, Objectives, TiSlice, Tolerance, Var};
    ///
    /// // Least cost first; among the cheapest plans, the least emissions.
    /// let mut m = Model::new();
    /// let coal = m.add_var(Var::continuous());
    /// let gas = m.add_var(Var::continuous());
    /// m.add_constraint((coal + gas).geq(10));
    /// m.minimize(2.0 * coal + 2.0 * gas);
    /// let co2 = m.add_stage(Objective::minimize(3.0 * coal + gas), Tolerance::EXACT);
    /// let program = m.build()?;
    /// assert!(matches!(program.objectives(), Objectives::Lexicographic { then, .. } if then.len() == 1));
    /// assert_eq!(program.objective_values(TiSlice::from_ref(&[0.0, 10.0]))[co2], 10.0);
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    pub fn add_stage(&mut self, objective: Objective, tolerance: Tolerance) -> ObjectiveId {
        self.push_further(objective, Rank::Tolerance(tolerance))
    }

    /// Adds a further linear objective to a blend with `weight` and returns its id, numbered as
    /// [`Model::add_stage`] numbers it.
    ///
    /// The blend minimizes the sum of every objective times its weight, signed by its sense: plus to minimize, minus
    /// to maximize. The model's own objective must then be linear, and every further objective weighted:
    /// [`Model::build`] refuses a model that also has a stage.
    ///
    /// ```
    /// use culmen_core::{Linear, Model, Objective, Objectives, Var, weight};
    /// let mut m = Model::new();
    /// let x = m.add_var(Var::continuous().upper(4));
    /// m.maximize(x);
    /// m.set_objective_weight(weight!(2.0));
    /// m.add_weighted(Objective::minimize(3.0 * x), weight!(0.5));
    /// assert!(matches!(m.build()?.objectives(), Objectives::Blended { with, .. } if with.len() == 1));
    /// # Ok::<(), culmen_core::ModelError>(())
    /// ```
    pub fn add_weighted(&mut self, objective: Objective, weight: Weight) -> ObjectiveId {
        self.push_further(objective, Rank::Weight(weight))
    }

    /// Appends a further objective and returns its id.
    fn push_further(&mut self, objective: Objective, rank: Rank) -> ObjectiveId {
        self.further.push((objective, rank));
        ObjectiveId::new(self.further.len())
    }

    /// How far later stages of a lexicographic solve may move the model's own objective from its optimum;
    /// [`Tolerance::EXACT`] until set. [`Model::build`] refuses it beside weighted objectives and without a further
    /// objective.
    pub const fn set_objective_tolerance(&mut self, tolerance: Tolerance) {
        self.first = Some(Rank::Tolerance(tolerance));
    }

    /// The weight of the model's own objective in a blend; [`Weight::ONE`] until set. [`Model::build`] refuses it
    /// beside stages and without a further objective.
    pub const fn set_objective_weight(&mut self, weight: Weight) {
        self.first = Some(Rank::Weight(weight));
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
        self.rows.get(row)?.name()
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
    /// convex to minimize or concave to maximize, as [`Program::CONVEXITY_TOLERANCE`] states, a quadratic objective
    /// beside further objectives, stages beside weighted objectives, or a rank of the model's own objective without a
    /// further one.
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
            .map(|(row, new)| self.build_row(row, new))
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

    /// The further objectives, checked, combined as their ranks say.
    ///
    /// # Errors
    ///
    /// [`ModelError::MixedObjectives`] for the first objective whose rank belongs to the other combination than the
    /// first ranked objective's, [`ModelError::RankAlone`] for a rank of the model's own objective alone, else
    /// [`ModelError::UnknownVar`], [`ModelError::FurtherCoefficient`] or [`ModelError::FurtherConstant`] for the first
    /// further objective at fault.
    fn build_objectives(&self) -> Result<Objectives, ModelError> {
        let Some(combination) = self.combination()? else {
            return Ok(Objectives::Single);
        };
        let built: Vec<(LinearObjective, Rank)> = self
            .further
            .iter()
            .zip(1..)
            .map(|((objective, rank), index)| Ok((self.build_further(ObjectiveId::new(index), objective)?, *rank)))
            .collect::<Result<_, ModelError>>()?;
        // Every rank belongs to `combination`, as `combination` checked.
        Ok(match combination {
            Combination::Lexicographic => Objectives::Lexicographic {
                first: match self.first {
                    Some(Rank::Tolerance(tolerance)) => tolerance,
                    Some(Rank::Weight(_)) | None => Tolerance::EXACT,
                },
                then: built
                    .into_iter()
                    .map(|(objective, rank)| Stage {
                        objective,
                        tolerance: match rank {
                            Rank::Tolerance(tolerance) => tolerance,
                            Rank::Weight(_) => Tolerance::EXACT,
                        },
                    })
                    .collect(),
            },
            Combination::Blended => Objectives::Blended {
                first: match self.first {
                    Some(Rank::Weight(weight)) => weight,
                    Some(Rank::Tolerance(_)) | None => Weight::ONE,
                },
                with: built
                    .into_iter()
                    .map(|(objective, rank)| Weighted {
                        objective,
                        weight: match rank {
                            Rank::Weight(weight) => weight,
                            Rank::Tolerance(_) => Weight::ONE,
                        },
                    })
                    .collect(),
            },
        })
    }

    /// How the objectives combine: as the first ranked objective's rank says, `None` without a further objective.
    ///
    /// # Errors
    ///
    /// [`ModelError::MixedObjectives`] for the first objective whose rank belongs to the other combination,
    /// [`ModelError::RankAlone`] for a rank of the model's own objective without a further one.
    fn combination(&self) -> Result<Option<Combination>, ModelError> {
        if self.further.is_empty() {
            return match self.first {
                Some(rank) => Err(ModelError::RankAlone {
                    combination: rank.combination(),
                }),
                None => Ok(None),
            };
        }
        let ranks = self.first.into_iter().map(|rank| (ObjectiveId::FIRST, rank));
        let mut ranks = ranks.chain(
            self.further
                .iter()
                .zip(1..)
                .map(|((_, rank), i)| (ObjectiveId::new(i), *rank)),
        );
        // A further objective exists, so there is a first rank.
        let combination = ranks
            .next()
            .map_or(Combination::Lexicographic, |(_, rank)| rank.combination());
        match ranks.find(|(_, rank)| rank.combination() != combination) {
            Some((objective, _)) => Err(ModelError::MixedObjectives { objective, combination }),
            None => Ok(Some(combination)),
        }
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

    /// The objective's quadratic terms as plan terms, each known and finite as given and as merged, and a square's
    /// Hessian value finite.
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
        // The program adds repeated pairs up the same way, so a sum that passes here passes there; a square's Hessian
        // value, twice its sum, must be finite there too.
        merge_pairs(terms.iter().copied())
            .into_iter()
            .try_for_each(|entry| self.check_quad_coefficient(entry.0, entry.1, hessian_value(entry)))?;
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

    /// One constraint as a checked plan row, a model constraint's constant moved into its bounds and its repeated
    /// columns merged.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`], [`ModelError::Coefficient`] or [`ModelError::ConstraintBounds`].
    fn build_row(&self, row: Row, new: &NewRow) -> Result<ProgramRow, ModelError> {
        let (bounds, given) = match new {
            NewRow::Constraint { constraint, .. } => (constraint.row_bounds(), constraint.row_terms()),
            NewRow::Row(data) => (Ok(data.bounds), data.terms.clone()),
        };
        let terms = self.row_terms(row, given)?;
        let bounds = bounds.map_err(|source| ModelError::ConstraintBounds {
            constraint: self.model_constraint(row),
            source,
        })?;
        Ok(ProgramRow {
            bounds,
            terms,
            name: new.name().map(str::to_owned),
        })
    }

    /// A constraint's terms, each column once, every coefficient finite as given and as added up.
    ///
    /// # Errors
    ///
    /// [`ModelError::UnknownVar`] or [`ModelError::Coefficient`] for the first defective term.
    fn row_terms(&self, row: Row, given: Vec<(Col, f64)>) -> Result<Vec<(Col, f64)>, ModelError> {
        for &(col, value) in &given {
            if col.index() >= self.vars.len() {
                return Err(ModelError::UnknownVar {
                    place: ExprPlace::Constraint(self.model_constraint(row)),
                    col,
                });
            }
            self.check_coefficient(row, col, value)?;
        }
        // The program adds repeated columns up the same way, so a sum that passes here passes there.
        let merged = merge_terms(given);
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
