// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instances hard enough that a solver searches for seconds: generalized assignment problems.

use culmen_core::{LinExpr, Linear, Model, Program, Var, VarMap, product};

/// A generalized assignment problem: each product goes to exactly one location, each location holds at most its
/// capacity in volume, and the walking cost of picks is minimized.
#[derive(Debug)]
pub struct Assignment {
    /// The built program.
    pub program: Program,
    /// The assignment variables, keyed by `(product, location)`.
    pub x: VarMap<(usize, usize)>,
}

/// A deterministic generator of small numbers.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

/// A generalized assignment problem of `products × locations` binaries, tight enough to be hard from about 40
/// products and 20 locations; `offset` is added to the objective.
///
/// # Panics
///
/// When the model fails to build, which a generator defect would cause.
#[must_use]
pub fn assignment(products: usize, locations: usize, seed: u64, offset: f64) -> Assignment {
    let mut rng = Lcg(seed);
    let picks: Vec<f64> = (0..products).map(|_| (1 + rng.below(50)) as f64).collect();
    let volume: Vec<f64> = (0..products).map(|_| (5 + rng.below(30)) as f64).collect();
    let distance: Vec<f64> = (0..locations).map(|l| 5.0 + 3.0 * l as f64).collect();
    let total: f64 = volume.iter().sum();
    let capacity = (total / locations as f64 * 1.15).round();

    let mut m = Model::new();
    let x = m.add_var_map(product(&(0..products), &(0..locations)), |(p, l)| {
        Var::binary().name(format!("x[{p},{l}]"))
    });
    for p in 0..products {
        m.add_constraint(LinExpr::sum((0..locations).map(|l| x[(p, l)])).equals(1));
    }
    for l in 0..locations {
        m.add_constraint(LinExpr::sum((0..products).map(|p| volume[p] * x[(p, l)])).leq(capacity));
    }
    let walk = LinExpr::sum(x.iter().map(|(&(p, l), col)| picks[p] * 2.0 * distance[l] * col));
    m.minimize(walk + offset);
    Assignment {
        program: m.build().expect("a valid model"),
        x,
    }
}
