// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cutting stock by column generation: cut rolls of width 100 into pieces to meet each width's demand with as few
//! rolls as possible. The master LP holds one column per cutting pattern. Each round, a pricing knapsack, its costs
//! set to the master's duals, finds the pattern that lowers the master's objective most; it is appended to the
//! master, which solves again from its last basis. Once no pattern helps, the patterns are made whole.
//!
//! `cargo run -p culmen --example column_generation`

use culmen::prelude::*;
use culmen::{Integrality, NewCol, ProgramColumn};

/// The width of a roll.
const ROLL: f64 = 100.0;

/// Each piece's width and demand.
const PIECES: [(f64, f64); 4] = [(45.0, 97.0), (36.0, 610.0), (31.0, 395.0), (14.0, 211.0)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The master: at first one pattern per width, as many pieces of it as a roll holds.
    let mut m = Model::new();
    let only = m.add_vars(PIECES.len(), |i| Var::continuous().name(format!("only[{i}]")));
    let demand: Vec<Row> = PIECES
        .iter()
        .zip(&only)
        .enumerate()
        .map(|(i, (&(width, demand), &x))| {
            m.add_named_constraint(format!("demand[{i}]"), ((ROLL / width).floor() * x).geq(demand))
        })
        .collect();
    m.minimize(LinExpr::sum(only.iter().copied()));
    let mut master = Solver::new(Highs::new(), m.build()?)?;

    // The pricing knapsack: how many pieces of each width one roll yields. Its costs are the master's duals, set
    // before each solve.
    let mut m = Model::new();
    let yields = m.add_vars(PIECES.len(), |i| Var::integer().name(format!("yield[{i}]")));
    m.add_constraint(LinExpr::sum(PIECES.iter().zip(&yields).map(|(&(width, _), &a)| width * a)).leq(ROLL));
    m.maximize(LinExpr::sum(yields.iter().copied()));
    let mut pricing = Solver::new(Highs::new(), m.build()?)?;

    loop {
        let (relaxed, iterations) = solve(&mut master)?;
        println!(
            "master {:.3} rolls after {iterations} simplex iterations",
            relaxed.objective
        );
        let duals = master.duals()?;
        pricing.set_col_costs(yields.iter().zip(&demand).map(|(&a, &row)| (a, duals.rows[row])))?;
        let (pattern, _) = solve(&mut pricing)?;
        // A pattern costs one roll, so it lowers the master's objective only when its pieces are worth more.
        if pattern.objective <= 1.0 + 1e-9 {
            break;
        }
        let entries: Vec<(Row, f64)> = demand
            .iter()
            .zip(&yields)
            .map(|(&row, &a)| (row, pattern.values[a].round()))
            .filter(|&(_, pieces)| pieces > 0.0)
            .collect();
        println!("  new pattern: {entries:?}");
        master.add_col(NewCol {
            column: ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE),
            entries,
        })?;
    }

    let patterns: Vec<Col> = master.program().cols().collect();
    master.set_col_integrality(patterns.iter().map(|&col| (col, Integrality::Integer)))?;
    let (rolls, _) = solve(&mut master)?;
    println!("{} whole rolls", rolls.objective);
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
