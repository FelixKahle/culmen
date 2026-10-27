// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An exact reference solver for tiny problems, sharing none of a real solver's reasoning, and a reference model of
//! editing a loaded program, read from the edit rules alone.
//!
//! It enumerates every integer point of the integer columns' box and, for each, every vertex of the polytope the
//! continuous columns span, solving each vertex's linear system in exact rational arithmetic. Every column must have
//! finite bounds, so the polytope is bounded and an optimum, if any point is feasible, lies at a vertex. A semi
//! column is read from its definition, zero or in its bounds: every choice of zero or bounds for every semi column is
//! solved as a program without semi columns, and the best of them is the optimum.

use culmen_core::{
    Bounds, Col, Constraint, Integrality, ModelClasses, NewCol, NewRow, Program, ProgramColumn, ProgramFeatures,
    ProgramPlan, ProgramRow, Row, Sense,
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

use crate::op::EditOp;

/// What the reference solver proves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reference {
    /// No point satisfies the problem.
    Infeasible,
    /// The exact optimal objective value, offset included.
    Optimal(BigRational),
}

impl Reference {
    /// The optimal objective value as the nearest `f64`, `None` when infeasible.
    #[must_use]
    pub fn objective(&self) -> Option<f64> {
        match self {
            Self::Infeasible => None,
            Self::Optimal(value) => value.to_f64(),
        }
    }
}

/// One inequality `Σ a · x <= b` over the continuous columns.
struct Inequality {
    a: Vec<BigRational>,
    b: BigRational,
}

/// Solves a tiny LP or MILP exactly, semi columns included; further objectives are ignored.
///
/// # Panics
///
/// When a column has an infinite bound, the integer box holds more than a million points, or there are more than
/// twelve semi columns.
#[must_use]
pub fn solve(program: &Program) -> Reference {
    let semi: Vec<Col> = program
        .cols()
        .filter(|&col| program.columns()[col].integrality.is_semi())
        .collect();
    assert!(semi.len() <= 12, "too many semi columns to enumerate their choices");
    (0..1u32 << semi.len())
        .map(|on| solve_box(&chosen(program, &semi, on)))
        .reduce(|a, b| better(program.sense(), a, b))
        .unwrap_or(Reference::Infeasible)
}

/// The better of two references under `sense`; a feasible one beats an infeasible one.
fn better(sense: Sense, a: Reference, b: Reference) -> Reference {
    match (&a, &b) {
        (Reference::Infeasible, _) => b,
        (_, Reference::Infeasible) => a,
        (Reference::Optimal(x), Reference::Optimal(y)) => {
            let b_better = match sense {
                Sense::Minimize => y < x,
                Sense::Maximize => y > x,
            };
            if b_better { b } else { a }
        }
    }
}

/// The program without semi columns in which the `k`-th semi column is zero when bit `k` of `on` is clear, and
/// continuous or integer in its bounds when it is set.
fn chosen(program: &Program, semi: &[Col], on: u32) -> Program {
    let mut plan = ProgramPlan::from(program.clone());
    for (k, &col) in semi.iter().enumerate() {
        let column = &mut plan.columns[col];
        if on & (1 << k) == 0 {
            column.integrality = Integrality::Continuous;
            column.bounds = Bounds::fixed(0.0).expect("zero is finite");
        } else if column.integrality == Integrality::SemiInteger {
            column.integrality = Integrality::Integer;
        } else {
            column.integrality = Integrality::Continuous;
        }
    }
    plan.build().expect("the choice of a valid program is valid")
}

/// Solves a tiny LP or MILP without semi columns exactly.
fn solve_box(program: &Program) -> Reference {
    let ints: Vec<usize> = (0..program.num_cols()).filter(|&j| is_integer(program, j)).collect();
    let conts: Vec<usize> = (0..program.num_cols()).filter(|&j| !is_integer(program, j)).collect();
    let ranges = integer_ranges(program, &ints);
    let points: i64 = ranges.iter().map(|&(l, u)| (u - l + 1).max(0)).product();
    assert!(points <= 1_000_000, "the integer box is too large to enumerate");
    if ranges.iter().any(|&(l, u)| l > u) {
        return Reference::Infeasible;
    }
    let mut best: Option<BigRational> = None;
    let mut point: Vec<i64> = ranges.iter().map(|&(l, _)| l).collect();
    loop {
        if let Some(value) = best_continuous(program, &ints, &point, &conts) {
            let better = best.as_ref().is_none_or(|b| match program.sense() {
                Sense::Minimize => value < *b,
                Sense::Maximize => value > *b,
            });
            if better {
                best = Some(value);
            }
        }
        if !advance(&mut point, &ranges) {
            break;
        }
    }
    best.map_or(Reference::Infeasible, Reference::Optimal)
}

/// The whole values each integer column `ints` may take, as `(least, greatest)`.
///
/// # Panics
///
/// When a bound is infinite or beyond `i64`.
fn integer_ranges(program: &Program, ints: &[usize]) -> Vec<(i64, i64)> {
    let whole =
        |v: f64, round: fn(&BigRational) -> BigRational| round(&exact(v)).to_integer().to_i64().expect("a small bound");
    ints.iter()
        .map(|&j| {
            let bounds = program.columns()[Col::new(j)].bounds;
            (
                whole(bounds.lower(), BigRational::ceil),
                whole(bounds.upper(), BigRational::floor),
            )
        })
        .collect()
}

/// A finite number as an exact rational.
///
/// # Panics
///
/// When `v` is not finite.
fn exact(v: f64) -> BigRational {
    BigRational::from_float(v).expect("a finite number")
}

fn is_integer(program: &Program, j: usize) -> bool {
    program.columns()[Col::new(j)].integrality.is_integral()
}

/// Steps `point` to the next integer point of the box, `false` after the last.
fn advance(point: &mut [i64], ranges: &[(i64, i64)]) -> bool {
    for (value, &(lower, upper)) in point.iter_mut().zip(ranges) {
        if *value < upper {
            *value += 1;
            return true;
        }
        *value = lower;
    }
    false
}

/// The best objective value over the continuous columns with the integer columns fixed at `point`, `None` when no
/// continuous point is feasible.
///
/// # Panics
///
/// When a continuous column has an infinite bound.
fn best_continuous(program: &Program, ints: &[usize], point: &[i64], conts: &[usize]) -> Option<BigRational> {
    let fixed = fixed_values(program, ints, point);
    let mut inequalities = row_inequalities(program, &fixed, conts);
    inequalities.extend(column_inequalities(program, conts));
    vertices(&inequalities, conts.len())
        .into_iter()
        .map(|x| objective_at(program, &fixed, conts, &x))
        .reduce(|a, b| match program.sense() {
            Sense::Minimize => a.min(b),
            Sense::Maximize => a.max(b),
        })
}

/// Each column's value when it is an integer column `ints`, fixed at `point`; `None` for a continuous column.
fn fixed_values(program: &Program, ints: &[usize], point: &[i64]) -> Vec<Option<BigRational>> {
    (0..program.num_cols())
        .map(|j| {
            ints.iter()
                .position(|&i| i == j)
                .map(|k| BigRational::from_integer(BigInt::from(point[k])))
        })
        .collect()
}

/// Every row's bounds as inequalities over the continuous columns `conts`, the fixed columns moved into constants.
///
/// # Panics
///
/// When a column is neither fixed nor in `conts`.
fn row_inequalities(program: &Program, fixed: &[Option<BigRational>], conts: &[usize]) -> Vec<Inequality> {
    let position = |j: usize| conts.iter().position(|&c| c == j).expect("a continuous column");
    let mut inequalities = Vec::new();
    for row in program.rows() {
        let mut a = vec![BigRational::zero(); conts.len()];
        let mut constant = BigRational::zero();
        for (col, coefficient) in program.row(row).entries() {
            let coefficient = exact(coefficient);
            match &fixed[col.index()] {
                Some(value) => constant += &coefficient * value,
                None => a[position(col.index())] += coefficient,
            }
        }
        push_bounds(&mut inequalities, &a, &constant, program.row_bounds()[row]);
    }
    inequalities
}

/// Every continuous column's bounds as inequalities over the continuous columns `conts`.
///
/// # Panics
///
/// When a continuous column has an infinite bound.
fn column_inequalities(program: &Program, conts: &[usize]) -> Vec<Inequality> {
    let mut inequalities = Vec::new();
    for (k, &j) in conts.iter().enumerate() {
        let mut a = vec![BigRational::zero(); conts.len()];
        a[k] = BigRational::one();
        let bounds = program.columns()[Col::new(j)].bounds;
        assert!(
            bounds.lower().is_finite() && bounds.upper().is_finite(),
            "every continuous column is bounded"
        );
        push_bounds(&mut inequalities, &a, &BigRational::zero(), bounds);
    }
    inequalities
}

/// The program's linear objective, offset included, at the point whose fixed columns are `fixed` and whose
/// continuous columns `conts` take `x`.
///
/// # Panics
///
/// When a column is neither fixed nor in `conts`.
fn objective_at(program: &Program, fixed: &[Option<BigRational>], conts: &[usize], x: &[BigRational]) -> BigRational {
    let position = |j: usize| conts.iter().position(|&c| c == j).expect("a continuous column");
    (0..program.num_cols()).fold(exact(program.offset()), |value, j| {
        let x_j = fixed[j].clone().unwrap_or_else(|| x[position(j)].clone());
        value + exact(program.columns()[Col::new(j)].cost) * x_j
    })
}

/// Adds `lower <= a · x + constant <= upper` as up to two inequalities `a · x <= b`.
fn push_bounds(out: &mut Vec<Inequality>, a: &[BigRational], constant: &BigRational, bounds: culmen_core::Bounds) {
    if bounds.upper().is_finite() {
        let b = BigRational::from_float(bounds.upper()).expect("finite") - constant;
        out.push(Inequality { a: a.to_vec(), b });
    }
    if bounds.lower().is_finite() {
        let b = constant - BigRational::from_float(bounds.lower()).expect("finite");
        out.push(Inequality {
            a: a.iter().map(|v| -v).collect(),
            b,
        });
    }
}

/// Every feasible vertex of `{x : a · x <= b for every inequality}` in `n` dimensions; the empty point when `n` is
/// zero and every inequality holds.
fn vertices(inequalities: &[Inequality], n: usize) -> Vec<Vec<BigRational>> {
    let holds = |x: &[BigRational]| {
        inequalities
            .iter()
            .all(|q| q.a.iter().zip(x).map(|(a, x)| a * x).sum::<BigRational>() <= q.b)
    };
    let mut found = Vec::new();
    for subset in subsets(inequalities.len(), n) {
        let rows: Vec<&Inequality> = subset.iter().map(|&i| &inequalities[i]).collect();
        if let Some(x) = solve_square(&rows, n)
            && holds(&x)
        {
            found.push(x);
        }
    }
    found
}

/// Every subset of `k` indices below `n`, in increasing order.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if n < k {
        return Vec::new();
    }
    let mut with_last: Vec<Vec<usize>> = subsets(n - 1, k - 1);
    for s in &mut with_last {
        s.push(n - 1);
    }
    let mut out = subsets(n - 1, k);
    out.extend(with_last);
    out
}

/// The unique solution of the square system with the inequalities as equalities, `None` when it is singular.
fn solve_square(rows: &[&Inequality], n: usize) -> Option<Vec<BigRational>> {
    let mut m: Vec<Vec<BigRational>> = rows
        .iter()
        .map(|q| {
            let mut r = q.a.clone();
            r.push(q.b.clone());
            r
        })
        .collect();
    for c in 0..n {
        let pivot = (c..n).find(|&r| !m[r][c].is_zero())?;
        m.swap(c, pivot);
        let p = m[c][c].clone();
        for v in &mut m[c] {
            *v = &*v / &p;
        }
        for r in 0..n {
            if r != c && !m[r][c].is_zero() {
                let factor = m[r][c].clone();
                let pivot_row = m[c].clone();
                for (v, pv) in m[r].iter_mut().zip(&pivot_row) {
                    *v -= &factor * pv;
                }
            }
        }
    }
    Some(m.into_iter().map(|r| r[n].clone()).collect())
}

/// Why the edit rules refuse an edit, in words for a failing test's message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    /// The rule the edit breaks.
    pub reason: String,
}

/// The program a `Solver` holding `program` holds after `op`, read from the edit rules alone, or why it refuses `op`.
///
/// The rules, applied to the plain fields of `program`'s plan in the order the edit gives its items:
///
/// - Every id an edit names is one of `program`'s: a setter's target, a new row's term, a new column's entry.
/// - A setter writes each value given into its column or row, so the last value given for an id holds.
/// - A new constraint is the row of its terms in the bounds `lower - constant ..= upper - constant`: its bounds as
///   given are a valid interval, its constant is finite, and no finite end becomes infinite by the move.
/// - A new column is appended after every column, and each entry `(row, a)` adds the term `(column, a)` to `row`.
/// - The edited plan builds, the backend solves its class (`classes`) and takes its features (`features`).
///
/// # Errors
///
/// [`Refused`] naming the first rule `op` breaks.
pub fn edit_plan(
    program: &Program,
    op: &EditOp,
    classes: ModelClasses,
    features: ProgramFeatures,
) -> Result<Program, Refused> {
    let mut plan = ProgramPlan::from(program.clone());
    let (num_cols, num_rows) = (program.num_cols(), program.num_rows());
    match op {
        EditOp::ColBounds(given) => given
            .iter()
            .try_for_each(|&(c, b)| column(&mut plan, c).map(|c| c.bounds = b))?,
        EditOp::ColCosts(given) => given
            .iter()
            .try_for_each(|&(c, v)| column(&mut plan, c).map(|c| c.cost = v))?,
        EditOp::ColIntegrality(given) => {
            given
                .iter()
                .try_for_each(|&(c, i)| column(&mut plan, c).map(|c| c.integrality = i))?;
        }
        EditOp::RowBounds(given) => given
            .iter()
            .try_for_each(|&(r, b)| row(&mut plan, r).map(|r| r.bounds = b))?,
        EditOp::AddRows(new) => {
            for new in new {
                let row = plan_row(new, num_cols)?;
                plan.rows.push(row);
            }
        }
        EditOp::AddCols(new) => new.iter().try_for_each(|new| append_column(&mut plan, new, num_rows))?,
    }
    let built = plan
        .build()
        .map_err(|e| refused(format!("the edited plan does not build: {e}")))?;
    if !classes.contains(built.class()) {
        return Err(refused(format!(
            "the backend solves {classes}, not a {}",
            built.class()
        )));
    }
    if !features.contains_all(built.features()) {
        return Err(refused(format!(
            "the backend takes {features}, not {}",
            built.features()
        )));
    }
    Ok(built)
}

/// A refusal for `reason`.
fn refused(reason: impl Into<String>) -> Refused {
    Refused { reason: reason.into() }
}

/// The plan's column `col`.
///
/// # Errors
///
/// [`Refused`] when the plan has no such column.
fn column(plan: &mut ProgramPlan, col: Col) -> Result<&mut ProgramColumn, Refused> {
    plan.columns
        .get_mut(col)
        .ok_or_else(|| refused(format!("the program has no {col:?}")))
}

/// The plan's row `row`.
///
/// # Errors
///
/// [`Refused`] when the plan has no such row.
fn row(plan: &mut ProgramPlan, row: Row) -> Result<&mut ProgramRow, Refused> {
    plan.rows
        .get_mut(row)
        .ok_or_else(|| refused(format!("the program has no {row:?}")))
}

/// The plan row of a new row, whose terms name columns below `num_cols`.
///
/// # Errors
///
/// [`Refused`] for a term naming another column, or a constraint whose bounds the move of its constant refuses.
fn plan_row(new: &NewRow, num_cols: usize) -> Result<ProgramRow, Refused> {
    let row = match new {
        NewRow::Row(row) => row.clone(),
        NewRow::Constraint { constraint, name } => ProgramRow {
            bounds: moved(constraint)?,
            terms: constraint.expr.terms.iter().map(|t| (t.col, t.coefficient)).collect(),
            name: name.clone(),
        },
    };
    match row.terms.iter().find(|(col, _)| col.index() >= num_cols) {
        Some((col, _)) => Err(refused(format!(
            "a new row names {col:?}, which the program does not have"
        ))),
        None => Ok(row),
    }
}

/// A constraint's bounds with its constant moved into them.
///
/// # Errors
///
/// [`Refused`] when the bounds as given are invalid, the constant is not finite, a finite end becomes infinite, or
/// the moved bounds are invalid.
fn moved(constraint: &Constraint) -> Result<Bounds, Refused> {
    let (lower, upper, constant) = (constraint.lower, constraint.upper, constraint.expr.constant);
    Bounds::new(lower, upper).map_err(|e| refused(format!("a new constraint's bounds are invalid: {e}")))?;
    if !constant.is_finite() {
        return Err(refused(format!("a new constraint's constant {constant} is not finite")));
    }
    let (moved_lower, moved_upper) = (lower - constant, upper - constant);
    if (lower.is_finite() && !moved_lower.is_finite()) || (upper.is_finite() && !moved_upper.is_finite()) {
        return Err(refused(
            "a new constraint's finite bound overflows as its constant moves into it",
        ));
    }
    Bounds::new(moved_lower, moved_upper)
        .map_err(|e| refused(format!("a new constraint's moved bounds are invalid: {e}")))
}

/// Appends a new column and its entries, which name rows below `num_rows`.
///
/// # Errors
///
/// [`Refused`] for an entry naming another row.
fn append_column(plan: &mut ProgramPlan, new: &NewCol, num_rows: usize) -> Result<(), Refused> {
    if let Some((row, _)) = new.entries.iter().find(|(row, _)| row.index() >= num_rows) {
        return Err(refused(format!(
            "a new column names {row:?}, which the program does not have"
        )));
    }
    let col = plan.columns.push_and_get_key(new.column.clone());
    for &(row, value) in &new.entries {
        plan.rows[row].terms.push((col, value));
    }
    Ok(())
}

/// Whether two objective values agree within `tolerance`, relative to their size and at least absolutely.
#[must_use]
pub fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use culmen_core::{Linear, Model, ModelClass, ProgramFeature, Var, bounds};

    use super::*;

    #[test]
    fn test_reference_solves_hand_worked_problems() {
        // Maximize 3x + 2y subject to x + y <= 4, x + 3y <= 6, x, y in [0, 10]: the optimum is x = 4, y = 0.
        let mut m = Model::new();
        let x = m.add_var(Var::continuous().upper(10.0));
        let y = m.add_var(Var::continuous().upper(10.0));
        m.add_constraint((x + y).leq(4.0));
        m.add_constraint((x + 3.0 * y).leq(6.0));
        m.maximize(3.0 * x + 2.0 * y);
        assert_eq!(solve(&m.build().unwrap()).objective(), Some(12.0));

        // The same with y integer and x <= 2.5: x = 2.5, y = 1 gives 9.5.
        let mut m = Model::new();
        let x = m.add_var(Var::continuous().upper(2.5));
        let y = m.add_var(Var::integer().upper(10.0));
        m.add_constraint((x + y).leq(4.0));
        m.add_constraint((x + 3.0 * y).leq(6.0));
        m.maximize(3.0 * x + 2.0 * y + 1);
        assert_eq!(
            solve(&m.build().unwrap()),
            Reference::Optimal(BigRational::new(21.into(), 2.into()))
        );

        // x + y = 1.5 with both integer: infeasible.
        let mut m = Model::new();
        let x = m.add_var(Var::integer().upper(3.0));
        let y = m.add_var(Var::integer().upper(3.0));
        m.add_constraint((x + y).equals(1.5));
        assert_eq!(solve(&m.build().unwrap()), Reference::Infeasible);
        assert!(close(1.0, 1.0 + 1e-9, 1e-6));
    }

    #[test]
    fn test_reference_solves_hand_worked_semi_problems() {
        // x is zero or in [2, 10]: maximizing it below 1.5 leaves it at zero; minimizing it above 0.5 lifts it to 2.
        let mut m = Model::new();
        let x = m.add_var(Var::semi_continuous().bounds(2, 10));
        m.add_constraint(x.leq(1.5));
        m.maximize(x);
        assert_eq!(solve(&m.build().unwrap()).objective(), Some(0.0));
        let mut m = Model::new();
        let x = m.add_var(Var::semi_continuous().bounds(2, 10));
        m.add_constraint(x.geq(0.5));
        m.minimize(x);
        assert_eq!(solve(&m.build().unwrap()).objective(), Some(2.0));

        // y is zero or a whole number in [2.5, 10]: at least 0.5 makes it 3; between 0.5 and 2.9 nothing fits.
        let mut m = Model::new();
        let y = m.add_var(Var::semi_integer().bounds(2.5, 10));
        m.add_constraint(y.geq(0.5));
        m.minimize(y);
        assert_eq!(solve(&m.build().unwrap()).objective(), Some(3.0));
        m.add_constraint(y.leq(2.9));
        assert_eq!(solve(&m.build().unwrap()), Reference::Infeasible);
    }

    /// Minimize `x + y` over `x`, `y` in `[0, 4]` and the row `x + y >= 1`.
    fn small() -> (Program, Col, Col, Row) {
        let mut m = Model::new();
        let x = m.add_var(Var::continuous().upper(4));
        let y = m.add_var(Var::continuous().upper(4));
        let row = m.add_constraint((x + y).geq(1));
        m.minimize(x + y);
        (m.build().unwrap(), x, y, row)
    }

    /// Every class and every feature.
    fn any(program: &Program, op: &EditOp) -> Result<Program, Refused> {
        edit_plan(program, op, ModelClasses::ALL, ProgramFeatures::ALL)
    }

    #[test]
    fn test_edit_plan_moves_a_constraint_constant_into_its_bounds() {
        let (program, x, y, _) = small();
        let edited = any(
            &program,
            &EditOp::AddRows(vec![NewRow::from(("r", (x + 2.0 * y + 1).leq(7)))]),
        )
        .unwrap();
        let new = edited.row(Row::new(1));
        assert_eq!(new.bounds(), bounds!(f64::NEG_INFINITY, 6.0));
        assert_eq!(new.entries().collect::<Vec<_>>(), [(x, 1.0), (y, 2.0)]);
        assert_eq!(new.name(), Some("r"));
    }

    #[test]
    fn test_edit_plan_appends_a_column_with_its_entries_added_up() {
        let (program, x, y, row) = small();
        let column = ProgramColumn::continuous(-1.0, bounds!(0.0, 1.0));
        let new = NewCol {
            column: column.clone(),
            entries: vec![(row, 1.0), (row, 2.0)],
        };
        let edited = any(&program, &EditOp::AddCols(vec![new])).unwrap();
        let z = Col::new(2);
        assert_eq!(edited.columns()[z], column);
        assert_eq!(
            edited.row(row).entries().collect::<Vec<_>>(),
            [(x, 1.0), (y, 1.0), (z, 3.0)]
        );
    }

    #[test]
    fn test_edit_plan_lets_the_last_value_given_hold() {
        let (program, x, ..) = small();
        let edited = any(&program, &EditOp::ColCosts(vec![(x, f64::NAN), (x, 5.0)])).unwrap();
        assert_eq!(edited.columns()[x].cost, 5.0);
        assert!(any(&program, &EditOp::ColCosts(vec![(x, 5.0), (x, f64::NAN)])).is_err());
    }

    #[test]
    fn test_edit_plan_refuses_what_the_rules_refuse() {
        let (program, x, _, row) = small();
        let refused = |op: EditOp| any(&program, &op).is_err();
        assert!(refused(EditOp::ColBounds(vec![(Col::new(2), Bounds::FREE)])));
        assert!(refused(EditOp::RowBounds(vec![(Row::new(1), Bounds::FREE)])));
        assert!(refused(EditOp::AddRows(vec![NewRow::from(Col::new(5).leq(1))])));
        assert!(refused(EditOp::AddRows(vec![NewRow::from(x.between(3, 1))])));
        assert!(refused(EditOp::AddRows(vec![NewRow::from(
            (1e308 * x + 1e308 * x).leq(1)
        )])));
        let unknown_row = NewCol {
            column: ProgramColumn::binary(0.0),
            entries: vec![(Row::new(1), 1.0)],
        };
        assert!(refused(EditOp::AddCols(vec![unknown_row])));
        assert!(refused(EditOp::ColIntegrality(vec![(x, Integrality::SemiContinuous)])));
        assert!(!refused(EditOp::RowBounds(vec![(row, Bounds::FREE)])));

        let linear = ModelClasses::of(&[ModelClass::Lp]);
        let integer = EditOp::ColIntegrality(vec![(x, Integrality::Integer)]);
        assert!(edit_plan(&program, &integer, linear, ProgramFeatures::ALL).is_err());
        let semi = EditOp::AddCols(vec![NewCol::from(ProgramColumn::semi_continuous(
            1.0,
            bounds!(1.0, 2.0),
        ))]);
        assert!(edit_plan(&program, &semi, ModelClasses::ALL, ProgramFeature::SemiInteger.into()).is_err());
        assert!(
            edit_plan(
                &program,
                &semi,
                ModelClasses::ALL,
                ProgramFeature::SemiContinuous.into()
            )
            .is_ok()
        );
    }
}
