// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property-test strategies: tiny bounded LPs and MILPs the reference solver enumerates, and small convex QPs.

use culmen_core::{Bounds, Col, Integrality, Program, ProgramColumn, ProgramPlan, ProgramRow, Sense, TiVec};
use itertools::Itertools;
use proptest::prelude::*;

/// A column: integrality, bounds and cost, all small integers.
fn column() -> impl Strategy<Value = ProgramColumn> {
    (any::<bool>(), -3i32..=1, 0i32..=4, -5i32..=5).prop_map(|(integer, lower, width, cost)| ProgramColumn {
        cost: f64::from(cost),
        bounds: Bounds::new(f64::from(lower), f64::from(lower + width)).expect("lower <= upper"),
        integrality: if integer {
            Integrality::Integer
        } else {
            Integrality::Continuous
        },
        name: None,
    })
}

/// Row bounds: both ends, or one end open.
fn row_bounds() -> impl Strategy<Value = Bounds> {
    (-8i32..=4, 0i32..=8, 0u8..3).prop_map(|(lower, width, shape)| {
        let (lower, upper) = (f64::from(lower), f64::from(lower + width));
        match shape {
            0 => Bounds::new(lower, upper),
            1 => Bounds::at_most(upper),
            _ => Bounds::at_least(lower),
        }
        .expect("valid bounds")
    })
}

/// A tiny bounded LP or MILP: one to four columns of which at most two continuous, up to three rows with small
/// integer coefficients, either sense and a small offset.
///
/// # Panics
///
/// When a generated plan fails to build, which a strategy defect would cause.
pub fn small_lp() -> impl Strategy<Value = Program> {
    prop::collection::vec(column(), 1..=4)
        .prop_filter("at most two continuous columns", |columns| {
            columns
                .iter()
                .filter(|c| c.integrality == Integrality::Continuous)
                .count()
                <= 2
        })
        .prop_flat_map(|columns| {
            let n = columns.len();
            let row = (row_bounds(), prop::collection::vec((0..n, -4i32..=4), 1..=4));
            (
                Just(columns),
                prop::collection::vec(row, 0..=3),
                any::<bool>(),
                -3i32..=3,
            )
        })
        .prop_map(|(columns, rows, maximize, offset)| {
            let rows = rows
                .into_iter()
                .map(|(bounds, terms)| ProgramRow {
                    bounds,
                    terms: terms.into_iter().map(|(j, a)| (Col::new(j), f64::from(a))).collect(),
                    name: None,
                })
                .collect();
            let sense = if maximize { Sense::Maximize } else { Sense::Minimize };
            ProgramPlan {
                sense,
                offset: f64::from(offset),
                columns: columns.into(),
                rows,
                quadratic: Vec::new(),
            }
            .build()
            .expect("a valid plan")
        })
}

/// The weight on the identity in a QP's Hessian `AᵀA + εI`, which makes it positive definite.
pub const QP_RIDGE: f64 = 0.1;

/// A small convex QP: two to four continuous columns in boxes of small integers, the quadratic part `½ xᵀHx`
/// with `H = AᵀA + εI` ([`QP_RIDGE`]) for a random integer `A`, small integer costs, and up to two rows that hold
/// at the boxes' midpoint, so it is feasible and bounded. Maximizing negates the objective, so it stays concave.
///
/// # Panics
///
/// When a generated plan fails to build, which a strategy defect would cause.
pub fn small_qp() -> impl Strategy<Value = Program> {
    (2..=4usize)
        .prop_flat_map(|n| {
            let boxes = prop::collection::vec((-3i32..=1, 1i32..=4), n);
            let a = prop::collection::vec(prop::collection::vec(-3i32..=3, n), 1..=n);
            let costs = prop::collection::vec(-5i32..=5, n);
            let rows = prop::collection::vec((prop::collection::vec(-3i32..=3, n), 0i32..=3, 0i32..=3), 0..=2);
            (boxes, a, costs, rows, any::<bool>())
        })
        .prop_map(|(boxes, a, costs, rows, maximize)| {
            let sign = if maximize { -1.0 } else { 1.0 };
            let columns: TiVec<Col, ProgramColumn> = boxes
                .iter()
                .zip(&costs)
                .map(|(&(lower, width), &cost)| {
                    let bounds = Bounds::new(f64::from(lower), f64::from(lower + width)).expect("lower <= upper");
                    ProgramColumn::continuous(sign * f64::from(cost), bounds)
                })
                .collect();
            let mid: Vec<f64> = boxes.iter().map(|&(l, w)| f64::from(l) + f64::from(w) / 2.0).collect();
            ProgramPlan {
                sense: if maximize { Sense::Maximize } else { Sense::Minimize },
                offset: 0.0,
                rows: rows.into_iter().map(|row| row_through(&mid, row)).collect(),
                quadratic: half_gram(&a, boxes.len())
                    .into_iter()
                    .map(|(i, j, q)| (i, j, sign * q))
                    .collect(),
                columns,
            }
            .build()
            .expect("a convex plan")
        })
}

/// The entries `(i, j, q)` of `½ xᵀ(AᵀA + εI)x`, `i <= j`: `q = ½ Hᵢᵢ` on the diagonal and `Hᵢⱼ` off it.
fn half_gram(a: &[Vec<i32>], n: usize) -> Vec<(Col, Col, f64)> {
    let h = |i: usize, j: usize| a.iter().map(|row| f64::from(row[i]) * f64::from(row[j])).sum::<f64>();
    (0..n)
        .array_combinations()
        .map(|[i, j]| (Col::new(i), Col::new(j), h(i, j)))
        .chain((0..n).map(|i| (Col::new(i), Col::new(i), (h(i, i) + QP_RIDGE) / 2.0)))
        .collect()
}

/// A row with the given coefficients whose bounds hold its activity at `point`, widened by `below` and `above`.
fn row_through(point: &[f64], (coefficients, below, above): (Vec<i32>, i32, i32)) -> ProgramRow {
    let activity: f64 = coefficients.iter().zip(point).map(|(&c, x)| f64::from(c) * x).sum();
    ProgramRow {
        bounds: Bounds::new(activity - f64::from(below), activity + f64::from(above)).expect("lower <= upper"),
        terms: coefficients
            .into_iter()
            .enumerate()
            .map(|(j, c)| (Col::new(j), f64::from(c)))
            .collect(),
        name: None,
    }
}
