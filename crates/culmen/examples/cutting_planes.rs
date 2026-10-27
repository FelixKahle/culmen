// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cutting planes for a 0/1 knapsack: solve its LP relaxation, add a cover inequality the relaxed optimum violates,
//! and solve again from the last basis, until no cover is violated; then make the items whole and solve the MIP.
//!
//! `cargo run -p culmen --example cutting_planes`

use culmen::prelude::*;
use culmen::{Integrality, TiSlice};

/// Each item's name, value and weight.
const ITEMS: [(&str, f64, f64); 8] = [
    ("tent", 16.0, 9.0),
    ("stove", 13.0, 7.0),
    ("rope", 9.0, 5.0),
    ("lamp", 8.0, 4.0),
    ("pan", 7.0, 4.0),
    ("map", 5.0, 3.0),
    ("knife", 4.0, 2.0),
    ("book", 2.0, 1.0),
];

/// The weight the knapsack holds.
const CAPACITY: f64 = 22.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let take = m.add_vars(ITEMS.len(), |i| {
        Var::continuous().upper(1.0).name(format!("take[{}]", ITEMS[i].0))
    });
    let weight = LinExpr::sum(ITEMS.iter().zip(&take).map(|(&(_, _, weight), &x)| weight * x));
    m.add_named_constraint("capacity", weight.leq(CAPACITY));
    m.maximize(LinExpr::sum(
        ITEMS.iter().zip(&take).map(|(&(_, value, _), &x)| value * x),
    ));

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    loop {
        let (relaxed, iterations) = solve(&mut solver)?;
        println!(
            "relaxation {:.3} after {iterations} simplex iterations",
            relaxed.objective
        );
        let Some(cover) = violated_cover(&take, &relaxed.values) else {
            break;
        };
        let names: Vec<&str> = cover.iter().map(|&x| ITEMS[x.index()].0).collect();
        // Of the items of a cover, which together outweigh the capacity, at most all but one fit.
        let fit = cover.len() as f64 - 1.0;
        let cut = solver.add_row((
            format!("at most {fit} of {}", names.join(", ")),
            LinExpr::sum(cover).leq(fit),
        ))?;
        println!("  cut: {}", solver.program().row(cut).name().unwrap_or_default());
    }

    solver.set_col_integrality(take.iter().map(|&x| (x, Integrality::Integer)))?;
    let (best, _) = solve(&mut solver)?;
    let taken: Vec<&str> = take
        .iter()
        .filter(|&&x| best.values[x] > 0.5)
        .map(|&x| ITEMS[x.index()].0)
        .collect();
    println!("whole items: value {} with {taken:?}", best.objective);
    Ok(())
}

/// Solves and returns the optimal solution and the simplex iterations the solve took.
fn solve(solver: &mut Solver<Highs>) -> Result<(Solution, u64), Box<dyn std::error::Error>> {
    let solved = solver.solve(Request::new())?;
    match solved.outcome {
        Outcome::Optimal(solution) => Ok((solution, solved.status.simplex_iterations)),
        other => Err(format!("no optimum: {other:?}").into()),
    }
}

/// A cover the point `values` violates: the items taken by how little of each the point leaves out per unit of
/// weight, until their weight exceeds the capacity. `None` when every item fits, or the point meets the cover.
fn violated_cover(take: &[Col], values: &TiSlice<Col, f64>) -> Option<Vec<Col>> {
    let left_out = |&x: &Col| (1.0 - values[x]) / ITEMS[x.index()].2;
    let mut order = take.to_vec();
    order.sort_by(|a, b| left_out(a).total_cmp(&left_out(b)));
    let cover: Vec<Col> = order
        .into_iter()
        .scan(0.0, |weight, x| {
            let fits = *weight <= CAPACITY;
            *weight += ITEMS[x.index()].2;
            fits.then_some(x)
        })
        .collect();
    let weight: f64 = cover.iter().map(|x| ITEMS[x.index()].2).sum();
    let taken: f64 = cover.iter().map(|&x| values[x]).sum();
    (weight > CAPACITY && taken > cover.len() as f64 - 1.0 + 1e-6).then_some(cover)
}
