// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cutting stock by branch-and-price on SCIP: cut rolls of width 100 into pieces to meet each width's demand with as
//! few whole rolls as possible, the instance of `column_generation`. The master holds one integer column per cutting
//! pattern, at first one pattern per width. SCIP solves it by branch and bound and, at every node, hands the node LP's
//! duals to the pricer (a `Pricer`, which exists only for a backend that prices), which prices patterns in with a
//! knapsack solved by a second solver, until no pattern improves the node LP. `column_generation` is the same model
//! priced in an outer loop on HiGHS and made whole only at the end; here the patterns are priced inside the search, so
//! the search is exact.
//!
//! Pricing on SCIP is column generation at every node, with SCIP's default branching on the master's columns. It is
//! exact when the pricing is exact and never adds a column the master already holds, as here: the knapsack excludes
//! every pattern the master holds, so it finds the best pattern the master lacks. culmen has no branching event, so
//! there is no Ryan–Foster or other branching on the pricing problem; branching on master columns is correct but weak,
//! and a master needing a large tree is better served by a branching rule of its own.
//!
//! `cargo run -p culmen --example branch_and_price --features scip`

use std::time::Duration;

use culmen::prelude::*;

/// The width of a roll.
const ROLL: f64 = 100.0;

/// Each piece's width and demand.
const PIECES: [(f64, f64); 4] = [(45.0, 97.0), (36.0, 610.0), (31.0, 395.0), (14.0, 211.0)];

/// More pricing rounds than this search can need: each round that adds a pattern adds one the master lacked, and
/// the node limit bounds the rounds that add none. A defect ends here instead of looping.
const MAX_ROUNDS: usize = 10_000;

/// The pricing problem: which pattern a roll is cut into, solved as a knapsack by a solver of its own.
///
/// `pick[i][k]` is one when the roll yields `k` pieces of width `i`, so a pattern is one pick per width, and a
/// pattern the master holds is excluded by the row "not all of its picks". Its objective is the value of a roll's
/// pieces at the master's duals, set before each solve.
struct Patterns {
    knapsack: Solver<Scip>,
    pick: Vec<Vec<Col>>,
    /// The master's demand row of each width.
    demand: Vec<Row>,
    rounds: usize,
    priced: usize,
}

impl Patterns {
    fn new(demand: Vec<Row>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut m = Model::new();
        let pick: Vec<Vec<Col>> = PIECES
            .iter()
            .enumerate()
            .map(|(i, &(width, _))| {
                let most = (ROLL / width).floor() as usize;
                m.add_vars(most + 1, |k| Var::binary().name(format!("pick[{i},{k}]")))
            })
            .collect();
        for choices in &pick {
            m.add_constraint(LinExpr::sum(choices.iter().copied()).equals(1));
        }
        let used = PIECES
            .iter()
            .zip(&pick)
            .flat_map(|(&(width, _), choices)| choices.iter().enumerate().map(move |(k, &z)| k as f64 * width * z));
        m.add_constraint(LinExpr::sum(used).leq(ROLL));
        m.maximize(LinExpr::sum(pick.iter().flatten().copied()));
        let knapsack = Solver::new(Scip::new(), m.build()?)?;
        let mut patterns = Self {
            knapsack,
            pick,
            demand,
            rounds: 0,
            priced: 0,
        };
        // The master's first patterns: as many pieces of one width as a roll holds.
        for (i, &(width, _)) in PIECES.iter().enumerate() {
            let mut only = vec![0; PIECES.len()];
            only[i] = (ROLL / width).floor() as usize;
            patterns.exclude(&only)?;
        }
        Ok(patterns)
    }

    /// Keeps the knapsack from finding `pattern` again, since the master holds it.
    fn exclude(&mut self, pattern: &[usize]) -> Result<(), Box<dyn std::error::Error>> {
        let picks = pattern.iter().zip(&self.pick).map(|(&k, choices)| choices[k]);
        self.knapsack
            .add_row(LinExpr::sum(picks).leq(PIECES.len() as f64 - 1.0))?;
        Ok(())
    }

    /// The pattern of most value at `duals`, among those the master lacks, as pieces per width.
    fn best(&mut self, duals: &[f64]) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let costs = self
            .pick
            .iter()
            .zip(duals)
            .flat_map(|(choices, &dual)| choices.iter().enumerate().map(move |(k, &z)| (z, k as f64 * dual)));
        self.knapsack.set_col_costs(costs.collect::<Vec<_>>())?;
        let solved = self.knapsack.run().time_limit(Duration::from_secs(10)).solve()?;
        let Outcome::Optimal(best) = solved.outcome else {
            return Err(format!("the knapsack has no optimum: {:?}", solved.outcome).into());
        };
        let pieces = |choices: &Vec<Col>| choices.iter().position(|&z| best.values[z] > 0.5).unwrap_or(0);
        Ok(self.pick.iter().map(pieces).collect())
    }
}

impl<B: Prices> Pricer<B> for Patterns {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        self.rounds += 1;
        assert!(self.rounds <= MAX_ROUNDS, "pricing ran {} rounds", self.rounds);
        // In a Farkas round the duals are the Farkas multipliers and the cost counts as zero, so the same knapsack
        // finds the pattern that repairs an infeasible node LP most.
        let duals: Vec<f64> = self.demand.iter().map(|&row| ctx.duals()[row]).collect();
        let pattern = self.best(&duals).expect("the pricing knapsack solves");
        let pieces = self.demand.iter().zip(&pattern).filter(|&(_, &k)| k > 0);
        let column = NewCol::integer()
            .cost(1)
            .entries(pieces.map(|(&row, &k)| (row, k as f64)));
        // The knapsack is exact over the patterns the master lacks, so when its best does not improve, none does.
        if ctx.improves(&column) {
            if self.rounds <= 5 || matches!(ctx.mode(), PriceMode::Farkas) {
                println!(
                    "  {:?}: pattern {pattern:?}, reduced cost {:.4}",
                    ctx.mode(),
                    ctx.reduced_cost(&column)
                );
            }
            self.exclude(&pattern).expect("the knapsack takes the row");
            ctx.add_col(column).expect("a valid pattern");
            self.priced += 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The master: at first one pattern per width, as many pieces of it as a roll holds.
    let mut m = Model::new();
    let only = m.add_vars(PIECES.len(), |i| Var::integer().name(format!("only[{i}]")));
    let demand: Vec<Row> = PIECES
        .iter()
        .zip(&only)
        .enumerate()
        .map(|(i, (&(width, demand), &x))| {
            m.add_constraint((format!("demand[{i}]"), ((ROLL / width).floor() * x).geq(demand)))
        })
        .collect();
    m.minimize(LinExpr::sum(only.iter().copied()));
    let mut master = Solver::new(Scip::new(), m.build()?)?;

    // Every pattern costs one roll, so the objective is whole: a solution within less than one roll of the bound is
    // optimal, and the absolute gap stops the search there.
    let mut patterns = Patterns::new(demand.clone())?;
    let solved = master
        .run()
        .settings(Settings {
            absolute_gap: Some(absolute_gap!(0.999)),
            node_limit: Some(10_000),
            ..Settings::new()
        })
        .time_limit(Duration::from_secs(60))
        .pricer(&mut patterns)
        .solve()?;

    let (rolls, bound) = match &solved.outcome {
        Outcome::Optimal(best) => (best, best.objective),
        Outcome::GapReached { solution, bound } => (solution, *bound),
        other => return Err(format!("no solution: {other:?}").into()),
    };
    println!(
        "{} patterns priced in {} rounds over {} nodes",
        patterns.priced,
        patterns.rounds,
        solved.nodes.unwrap_or(0)
    );
    let program = master.program();
    for col in program.cols() {
        let used = rolls.values[col].round();
        if used > 0.0 {
            let pattern: Vec<f64> = demand.iter().map(|&row| program.coefficient(row, col)).collect();
            println!(
                "  {used} rolls cut into {pattern:?} pieces of widths {:?}",
                PIECES.map(|(width, _)| width)
            );
        }
    }
    println!(
        "{} whole rolls; no cutting plan uses fewer than {bound:.2}",
        rolls.objective
    );
    Ok(())
}
