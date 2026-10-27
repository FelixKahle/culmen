// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The 0/1 knapsack of `knapsack`, written with the modelling macros.
//!
//! `cargo run -p culmen --example knapsack_macros --features macros`.

use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let items = [
        ("tent", 11.0, 7.0),
        ("stove", 6.0, 4.0),
        ("rope", 3.0, 2.0),
        ("lamp", 4.0, 3.0),
        ("book", 2.0, 1.0),
    ];
    let names: Vec<&str> = items.iter().map(|&(name, ..)| name).collect();
    let value = |i: &str| items.iter().find(|item| item.0 == i).map_or(0.0, |item| item.1);
    let weight = |i: &str| items.iter().find(|item| item.0 == i).map_or(0.0, |item| item.2);
    let capacity = 12.0;

    let mut m = Model::new();
    variable!(m, take[i in names], Bin);
    constraint!(m, capacity_limit, sum!(weight(i) * take[i] for i in names) <= capacity);
    objective!(m, Max, sum!(value(i) * take[i] for i in names));

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
