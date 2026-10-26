// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property-test strategies: tiny bounded LPs and MILPs the reference solver enumerates.

use culmen_core::{Bounds, Col, Integrality, Lp, LpColumn, LpPlan, LpRow, Sense};
use proptest::prelude::*;

/// A column: integrality, bounds and cost, all small integers.
fn column() -> impl Strategy<Value = LpColumn> {
    (any::<bool>(), -3i32..=1, 0i32..=4, -5i32..=5).prop_map(|(integer, lower, width, cost)| LpColumn {
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
pub fn small_lp() -> impl Strategy<Value = Lp> {
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
                .map(|(bounds, terms)| LpRow {
                    bounds,
                    terms: terms.into_iter().map(|(j, a)| (Col::new(j), f64::from(a))).collect(),
                    name: None,
                })
                .collect();
            let sense = if maximize { Sense::Maximize } else { Sense::Minimize };
            LpPlan {
                sense,
                offset: f64::from(offset),
                columns: columns.into(),
                rows,
            }
            .build()
            .expect("a valid plan")
        })
}
