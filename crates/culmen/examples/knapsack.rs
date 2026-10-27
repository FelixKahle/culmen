// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A 0/1 knapsack written with the plain API: pick the items of most value that fit the capacity.
//!
//! `cargo run -p culmen --example knapsack`; `knapsack_macros` writes the same model with the macros.

use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let items = [
        ("tent", 11.0, 7.0),
        ("stove", 6.0, 4.0),
        ("rope", 3.0, 2.0),
        ("lamp", 4.0, 3.0),
        ("book", 2.0, 1.0),
    ];
    let capacity = 12.0;

    let mut m = Model::new();
    let take = m.add_var_map(items.iter().map(|&(name, ..)| name), |name| {
        Var::binary().name(format!("take[{name}]"))
    });
    let weight = LinExpr::sum(items.iter().map(|&(name, _, weight)| weight * take[name]));
    m.add_named_constraint("capacity", weight.leq(capacity));
    m.maximize(LinExpr::sum(items.iter().map(|&(name, value, _)| value * take[name])));

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let solved = solver.solve(Request::new())?;
    let Outcome::Optimal(best) = solved.outcome else {
        return Err(format!("no optimum: {:?}", solved.outcome).into());
    };
    println!("value {}", best.objective);
    for (name, value) in take.values(&best.values) {
        if value > 0.5 {
            println!("  take {name}");
        }
    }
    Ok(())
}
