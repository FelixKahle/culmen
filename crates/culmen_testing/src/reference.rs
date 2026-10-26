// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An exact reference solver for tiny problems, sharing none of a real solver's reasoning.
//!
//! It enumerates every integer point of the integer columns' box and, for each, every vertex of the polytope the
//! continuous columns span, solving each vertex's linear system in exact rational arithmetic. Every column must have
//! finite bounds, so the polytope is bounded and an optimum, if any point is feasible, lies at a vertex.

use culmen_core::{Integrality, Program, Sense};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

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

/// Solves a tiny LP or MILP exactly.
///
/// # Panics
///
/// When a column has an infinite bound, or the integer box holds more than a million points.
#[must_use]
pub fn solve(program: &Program) -> Reference {
    let exact = |v: f64| BigRational::from_float(v).expect("a finite number");
    let ints: Vec<usize> = (0..program.num_cols()).filter(|&j| is_integer(program, j)).collect();
    let conts: Vec<usize> = (0..program.num_cols()).filter(|&j| !is_integer(program, j)).collect();
    let ranges: Vec<(i64, i64)> = ints
        .iter()
        .map(|&j| {
            let bounds = program.columns()[culmen_core::Col::new(j)].bounds;
            let lower = exact(bounds.lower())
                .ceil()
                .to_integer()
                .to_i64()
                .expect("a small lower bound");
            let upper = exact(bounds.upper())
                .floor()
                .to_integer()
                .to_i64()
                .expect("a small upper bound");
            (lower, upper)
        })
        .collect();
    let points: i64 = ranges.iter().map(|&(l, u)| (u - l + 1).max(0)).product();
    assert!(points <= 1_000_000, "the integer box is too large to enumerate");
    let mut best: Option<BigRational> = None;
    let mut point: Vec<i64> = ranges.iter().map(|&(l, _)| l).collect();
    if ranges.iter().any(|&(l, u)| l > u) {
        return Reference::Infeasible;
    }
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

fn is_integer(program: &Program, j: usize) -> bool {
    program.columns()[culmen_core::Col::new(j)].integrality == Integrality::Integer
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
fn best_continuous(program: &Program, ints: &[usize], point: &[i64], conts: &[usize]) -> Option<BigRational> {
    let exact = |v: f64| BigRational::from_float(v).expect("a finite number");
    let fixed: Vec<Option<BigRational>> = (0..program.num_cols())
        .map(|j| {
            ints.iter()
                .position(|&i| i == j)
                .map(|k| BigRational::from_integer(BigInt::from(point[k])))
        })
        .collect();
    let position = |j: usize| conts.iter().position(|&c| c == j);
    let mut inequalities = Vec::new();
    for row in program.rows() {
        let mut a = vec![BigRational::zero(); conts.len()];
        let mut constant = BigRational::zero();
        for (col, coefficient) in program.row(row).entries() {
            let coefficient = exact(coefficient);
            match &fixed[col.index()] {
                Some(value) => constant += &coefficient * value,
                None => a[position(col.index()).expect("a continuous column")] += coefficient,
            }
        }
        push_bounds(&mut inequalities, &a, &constant, program.row_bounds()[row]);
    }
    for (k, &j) in conts.iter().enumerate() {
        let mut a = vec![BigRational::zero(); conts.len()];
        a[k] = BigRational::one();
        let bounds = program.columns()[culmen_core::Col::new(j)].bounds;
        assert!(
            bounds.lower().is_finite() && bounds.upper().is_finite(),
            "every continuous column is bounded"
        );
        push_bounds(&mut inequalities, &a, &BigRational::zero(), bounds);
    }
    let objective = |x: &[BigRational]| -> BigRational {
        let mut value = exact(program.offset());
        for j in 0..program.num_cols() {
            let cost = exact(program.columns()[culmen_core::Col::new(j)].cost);
            let x_j = fixed[j]
                .clone()
                .unwrap_or_else(|| x[position(j).expect("a continuous column")].clone());
            value += cost * x_j;
        }
        value
    };
    vertices(&inequalities, conts.len())
        .into_iter()
        .map(|x| objective(&x))
        .reduce(|a, b| match program.sense() {
            Sense::Minimize => a.min(b),
            Sense::Maximize => a.max(b),
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

/// Whether two objective values agree within `tolerance`, relative to their size and at least absolutely.
#[must_use]
pub fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use culmen_core::{Linear, Model, Var};

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
}
