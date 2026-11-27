// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instances hard enough that a solver searches for seconds: generalized assignment problems; and instances whose
//! outcome turns on a solver's tolerances.

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
#[derive(Debug)]
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

/// Two columns in `[0, 1]` held to `x + y >= 1` and `x + y <= 1 - 1e-5`, integer or not, minimizing `x`: infeasible by
/// `1e-5`, so infeasible to the solvers' default feasibility tolerances (`1e-6` and below) and feasible to `1e-4`.
/// Integer, its one point a tolerance admits has `x + y = 1`.
///
/// # Panics
///
/// Never: the model is valid.
#[must_use]
pub fn infeasible_by_a_hair(integer: bool) -> Program {
    let mut m = Model::new();
    let kind = if integer { Var::integer() } else { Var::continuous() };
    let x = m.add_var(kind.clone().upper(1));
    let y = m.add_var(kind.upper(1));
    m.add_constraint((x + y).geq(1));
    m.add_constraint((x + y).leq(1.0 - 1e-5));
    m.minimize(x);
    m.build().expect("a valid model")
}

/// Minimize `x - 1e-4 y` over `y - x <= 1`, `x, y >= 0`: the optimum is `-1e-4` at `(0, 1)`, which an LP solver
/// started at the origin finds only when it counts the reduced cost `-1e-4` of `y` as wrong, so not to an optimality
/// tolerance of `1e-3`, where the origin, objective zero, counts as optimal. The cost of `x` keeps the largest cost
/// at one, so a solver that scales the objective to it leaves that reduced cost as it is, and no column has an upper
/// bound, so the objective is bounded only through the row and only an LP proves the optimum.
///
/// # Panics
///
/// Never: the model is valid.
#[must_use]
pub fn flat_costs() -> Program {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    m.add_constraint((y - x).leq(1));
    m.minimize(x - 1e-4 * y);
    m.build().expect("a valid model")
}
