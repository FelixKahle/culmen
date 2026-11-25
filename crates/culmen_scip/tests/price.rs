// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pricing on SCIP: the two hand-worked fixtures with their duals, the integer cutting stock,
//! a panic and an interrupt during pricing, pricing beside lazy constraints, cuts, heuristic solutions and a start,
//! the refusals, and random programs split into kept and withheld columns against exact enumeration of the full
//! program.
//!
//! Every solve here sets a node limit and a time limit, and every pricing callback counts its rounds and fails past
//! a bound, so a defect ends as a failed assertion.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};

use culmen_core::{
    ActionError, Bounds, Callback, Col, CutContext, Events, HeuristicContext, LazyContext, Linear, NewCol, Outcome,
    PollContext, PriceContext, PriceMode, Program, ProgramColumn, ProgramPlan, Request, Row, Sense, Settings, Solution,
    SolutionContext, SolveError, Solver, StopReason, TiVec, bounds,
};
use culmen_scip::{Scip, ScipError, ScipOptions, ScipPrice};
use culmen_testing::reference::{Reference, close, solve};
use culmen_testing::strategy::{PricedSplit, priced_split};
use proptest::prelude::*;

/// The most pricing rounds any test here expects; a callback past it fails.
const MAX_ROUNDS: usize = 2_000;

/// The most lazy-constraint or cut calls any test here expects; a callback past it fails.
const MAX_CALLS: usize = 20_000;

/// A request with a node limit and a time limit, presolving or not.
fn limited(presolve: bool) -> Request<'static> {
    Request::new().settings_from(Settings {
        node_limit: Some(10_000),
        time_limit: Some(Duration::from_secs(30)),
        presolve: Some(presolve),
        ..Settings::new()
    })
}

fn optimal(outcome: Outcome) -> Solution {
    match outcome {
        Outcome::Optimal(solution) => solution,
        other => panic!("expected an optimum, got {other:?}"),
    }
}

/// A column of cost `cost` with the given bounds and entries.
fn column(cost: f64, bounds: Bounds, integer: bool, entries: &[(Row, f64)]) -> NewCol {
    let column = if integer {
        ProgramColumn::integer(cost, bounds)
    } else {
        ProgramColumn::continuous(cost, bounds)
    };
    NewCol {
        column,
        entries: entries.to_vec(),
    }
}

/// One pricing round as the callback saw it.
#[derive(Clone, Debug)]
struct Seen {
    mode: PriceMode,
    duals: Vec<f64>,
    /// Each watched column's reduced cost and whether it improves.
    columns: Vec<(f64, bool)>,
}

/// Watches every column of `watched` at each round and adds, from those withheld, every one not added yet that
/// improves, or only the first or only the most improving.
struct Pricer {
    watched: Vec<NewCol>,
    withheld: Vec<usize>,
    pick: Pick,
    /// The watched column each added `Col` stands for, in the order added.
    added: Vec<(usize, Col)>,
    rounds: Vec<Seen>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pick {
    Every,
    First,
    Most,
}

impl Pricer {
    fn new(watched: Vec<NewCol>, withheld: impl IntoIterator<Item = usize>, pick: Pick) -> Self {
        Self {
            watched,
            withheld: withheld.into_iter().collect(),
            pick,
            added: Vec::new(),
            rounds: Vec::new(),
        }
    }

    fn col_of(&self, watched: usize) -> Option<Col> {
        self.added.iter().find(|&&(i, _)| i == watched).map(|&(_, col)| col)
    }

    /// The improvement of a column in minimization form: how far below `-tolerance` its reduced cost lies.
    fn gain(ctx: &ScipPrice<'_>, column: &NewCol) -> f64 {
        let d = ctx.reduced_cost(column);
        match (ctx.mode(), ctx.sense()) {
            (PriceMode::ReducedCost { .. }, Sense::Maximize) => d,
            _ => -d,
        }
    }

    fn price_round(&mut self, ctx: &mut ScipPrice<'_>) {
        assert!(
            self.rounds.len() < MAX_ROUNDS,
            "more pricing rounds than any test expects"
        );
        let columns = self
            .watched
            .iter()
            .map(|c| (ctx.reduced_cost(c), ctx.improves(c)))
            .collect();
        self.rounds.push(Seen {
            mode: ctx.mode(),
            duals: ctx.duals().iter().copied().collect(),
            columns,
        });
        let open: Vec<usize> = self
            .withheld
            .iter()
            .copied()
            .filter(|&i| self.col_of(i).is_none() && ctx.improves(&self.watched[i]))
            .collect();
        let chosen: Vec<usize> = match self.pick {
            Pick::Every => open,
            Pick::First => open.into_iter().take(1).collect(),
            Pick::Most => open
                .into_iter()
                .max_by(|&a, &b| Self::gain(ctx, &self.watched[a]).total_cmp(&Self::gain(ctx, &self.watched[b])))
                .into_iter()
                .collect(),
        };
        for i in chosen {
            let col = ctx.add_col(self.watched[i].clone()).expect("a column SCIP takes");
            self.added.push((i, col));
        }
    }
}

impl Callback<Scip> for Pricer {
    fn events(&self) -> Events {
        Events::NONE
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        self.price_round(ctx);
    }
}

/// The first hand-worked fixture: cutting stock of rolls of width 10 into pieces of width 3, 4 and 5 with demands 7,
/// 2 and 3, every pattern costing one roll. The program starts with the patterns A = (3,0,0), D = (0,2,0) and
/// F = (0,0,2); it returns every one of the 11 patterns, those three first, then the eight it withholds:
/// (1,0,0), (2,0,0), (0,1,0), (1,1,0), B = (2,1,0), (0,0,1), C = (1,0,1), E = (0,1,1).
///
/// The master LP over the three starting patterns is `min a + d + f` with `3a >= 7`, `2d >= 2`, `2f >= 3`: optimal at
/// `a = 7/3, d = 1, f = 3/2`, worth 29/6, with duals (1/3, 1/2, 1/2). B then prices at `1 - (2/3 + 1/2) = -1/6` and
/// improves, E at `1 - (1/2 + 1/2) = 0` and does not. With B, the LP is optimal at `a = 1, d = 0, f = 3/2, b = 2`,
/// worth 9/2, with duals (1/3, 1/3, 1/2) from `3y1 = 1`, `2y1 + y2 = 1` and `2y3 = 1`, and nothing improves.
fn cutting_stock(integer: bool) -> (Program, Vec<NewCol>) {
    let patterns = [
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let rows = [Row::new(0), Row::new(1), Row::new(2)];
    let columns: Vec<NewCol> = patterns
        .iter()
        .map(|p| {
            let entries: Vec<(Row, f64)> = rows
                .iter()
                .zip(p)
                .map(|(&r, &a)| (r, a))
                .filter(|&(_, a)| a != 0.0)
                .collect();
            column(1.0, Bounds::NON_NEGATIVE, integer, &entries)
        })
        .collect();
    let mut plan = ProgramPlan::new();
    let start: Vec<Col> = columns[..3].iter().map(|c| plan.col(c.column.clone())).collect();
    for (piece, demand) in [7.0, 2.0, 3.0].into_iter().enumerate() {
        let terms = start
            .iter()
            .zip(&patterns)
            .map(|(&col, p)| (col, p[piece]))
            .filter(|&(_, a)| a != 0.0);
        plan.row(Bounds::at_least(demand).unwrap(), terms);
    }
    (plan.build().unwrap(), columns)
}

/// The full program of a fixture: every column of `columns` up front.
fn full(program: &Program, columns: &[NewCol]) -> Program {
    let mut plan = ProgramPlan::from(program.clone());
    plan.columns = TiVec::new();
    for row in &mut plan.rows {
        row.terms.clear();
    }
    for (j, column) in columns.iter().enumerate() {
        plan.columns.push(column.column.clone());
        for &(row, a) in &column.entries {
            plan.rows[row].terms.push((Col::new(j), a));
        }
    }
    plan.build().unwrap()
}

/// A point of the solver's program mapped to the full program of a fixture: a column the pricer added takes its
/// value, a column the start held its own, every other zero.
fn in_full(pricer: &Pricer, start: usize, values: &TiVec<Col, f64>) -> TiVec<Col, f64> {
    (0..pricer.watched.len())
        .map(|i| match pricer.col_of(i) {
            Some(col) => values[col],
            None if i < start => values[Col::new(i)],
            None => 0.0,
        })
        .collect()
}

fn assert_duals(seen: &Seen, want: &[f64]) {
    assert_eq!(seen.duals.len(), want.len(), "{seen:?}");
    for (got, want) in seen.duals.iter().zip(want) {
        assert!((got - want).abs() <= 1e-9, "duals {:?}, want {want:?}", seen.duals);
    }
}

fn assert_objective(seen: &Seen, want: f64) {
    match seen.mode {
        PriceMode::ReducedCost { objective } => assert!((objective - want).abs() <= 1e-9, "{seen:?}, want {want}"),
        PriceMode::Farkas => panic!("expected reduced-cost mode, got {seen:?}"),
    }
}

#[test]
fn test_fixture_one_prices_pattern_b_alone_and_reaches_nine_halves() {
    for presolve in [true, false] {
        let (program, patterns) = cutting_stock(false);
        let mut solver = Solver::new(Scip::new(), program).unwrap();
        let mut pricer = Pricer::new(patterns, 3..11, Pick::Every);
        let solved = solver.solve_with(limited(presolve), &mut pricer).unwrap();

        // Round 1: duals (1/3, 1/2, 1/2) at 29/6; only B improves, E sits on the boundary.
        assert_objective(&pricer.rounds[0], 29.0 / 6.0);
        assert_duals(&pricer.rounds[0], &[1.0 / 3.0, 0.5, 0.5]);
        let (b, e) = (7, 10);
        assert!((pricer.rounds[0].columns[b].0 + 1.0 / 6.0).abs() <= 1e-9);
        assert!(pricer.rounds[0].columns[b].1);
        assert!(
            !pricer.rounds[0].columns[e].1,
            "E has reduced cost {}",
            pricer.rounds[0].columns[e].0
        );
        assert_eq!(pricer.added, vec![(b, Col::new(3))]);
        // Round 2: duals (1/3, 1/3, 1/2) at 9/2, and nothing improves; SCIP may offer the same LP again.
        assert_objective(&pricer.rounds[1], 4.5);
        assert_duals(&pricer.rounds[1], &[1.0 / 3.0, 1.0 / 3.0, 0.5]);
        assert!(
            pricer.rounds[1..]
                .iter()
                .all(|r| r.columns.iter().skip(3).all(|&(_, improves)| !improves))
        );

        let best = optimal(solved.outcome);
        assert!(close(best.objective, 4.5, 1e-9), "{best:?}");
        let want = [1.0, 0.0, 1.5, 2.0];
        assert_eq!(best.values.len(), 4);
        for (got, want) in best.values.iter().zip(want) {
            assert!((got - want).abs() <= 1e-9, "{best:?}");
        }
        assert_eq!(solver.program().num_cols(), 4);
        assert_eq!(solver.program().columns()[Col::new(3)], pricer.watched[b].column);
    }
}

#[test]
fn test_integer_cutting_stock_needs_five_rolls_and_its_solution_cuts_the_full_program() {
    for presolve in [true, false] {
        let (program, patterns) = cutting_stock(true);
        let full = full(&program, &patterns);
        let mut solver = Solver::new(Scip::new(), program).unwrap();
        let mut pricer = Pricer::new(patterns, 3..11, Pick::Every);
        let solved = solver.solve_with(limited(presolve), &mut pricer).unwrap();
        let best = optimal(solved.outcome);
        assert!(close(best.objective, 5.0, 1e-9), "{best:?}");
        assert_eq!(best.values.len(), solver.program().num_cols());
        let point = in_full(&pricer, 3, &best.values);
        assert!(full.max_violation(&point) <= 1e-6, "{point:?}");
        assert!(close(full.objective_value(&point), 5.0, 1e-9));
    }
}

/// The second hand-worked fixture: maximize `3x1 + 2x2 + 7x3` subject to `x1 + x2 + 2x3 <= 4` and `x1 + x3 >= 1`,
/// every column in `[0, 10]`; the program starts with x2, and withholds x1 and then x3.
///
/// With `x2` alone the row `x1 + x3 >= 1` cannot hold, so the first round is Farkas, with multipliers `f0 <= 0 < f1`.
/// With `x1` the LP is optimal at `x1 = 4`, worth 12, with duals (3, 0), and `x3` prices at `7 - 2·3 = 1 > 0`, which
/// improves a maximization; with `x3` it is optimal at `x3 = 2`, worth 14, with duals (7/2, 0).
fn max_fixture() -> (Program, Vec<NewCol>) {
    let ten = bounds!(0.0, 10.0);
    let (r0, r1) = (Row::new(0), Row::new(1));
    let columns = vec![
        column(2.0, ten, false, &[(r0, 1.0)]),
        column(3.0, ten, false, &[(r0, 1.0), (r1, 1.0)]),
        column(7.0, ten, false, &[(r0, 2.0), (r1, 1.0)]),
    ];
    let mut plan = ProgramPlan::new();
    plan.sense = Sense::Maximize;
    let x2 = plan.col(columns[0].column.clone());
    plan.row(bounds!(f64::NEG_INFINITY, 4.0), [(x2, 1.0)]);
    plan.row(bounds!(1.0, f64::INFINITY), []);
    (plan.build().unwrap(), columns)
}

#[test]
fn test_fixture_two_prices_through_farkas_then_with_the_maximization_sign() {
    for presolve in [true, false] {
        let (program, columns) = max_fixture();
        let mut solver = Solver::new(Scip::new(), program).unwrap();
        let mut pricer = Pricer::new(columns, [1, 2], Pick::First);
        let solved = solver.solve_with(limited(presolve), &mut pricer).unwrap();

        // Round 1: the start is infeasible; any valid ray has f0 <= 0 < f1, and x1 improves.
        let farkas = &pricer.rounds[0];
        assert_eq!(farkas.mode, PriceMode::Farkas);
        assert!(farkas.duals[0] <= 0.0 && farkas.duals[1] > 0.0, "{farkas:?}");
        assert!(farkas.columns[1].1, "{farkas:?}");
        // Round 2: x1 = 4 at 12, duals (3, 0); x3 prices at 7 - 2·3 = 1 > 0, which improves a maximization.
        assert_objective(&pricer.rounds[1], 12.0);
        assert_duals(&pricer.rounds[1], &[3.0, 0.0]);
        assert!((pricer.rounds[1].columns[2].0 - 1.0).abs() <= 1e-9);
        assert!(pricer.rounds[1].columns[2].1);
        // Round 3: x3 = 2 at 14, duals (7/2, 0).
        assert_objective(&pricer.rounds[2], 14.0);
        assert_duals(&pricer.rounds[2], &[3.5, 0.0]);
        assert_eq!(pricer.added, vec![(1, Col::new(1)), (2, Col::new(2))]);

        let best = optimal(solved.outcome);
        assert!(close(best.objective, 14.0, 1e-9), "{best:?}");
        for (got, want) in best.values.iter().zip([0.0, 0.0, 2.0]) {
            assert!((got - want).abs() <= 1e-9, "{best:?}");
        }
    }
}

/// Prices as a `Pricer` and then panics in the round given.
struct Panics {
    pricer: Pricer,
    round: usize,
}

impl Callback<Scip> for Panics {
    fn events(&self) -> Events {
        Events::NONE
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        self.pricer.price_round(ctx);
        assert!(self.pricer.rounds.len() < self.round, "a panic in round {}", self.round);
    }
}

#[test]
fn test_a_panic_in_price_unwinds_after_scip_stopped_and_the_program_keeps_the_column() {
    let (program, patterns) = cutting_stock(false);
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut panics = Panics {
        pricer: Pricer::new(patterns.clone(), 3..11, Pick::Every),
        round: 1,
    };
    let caught = catch_unwind(AssertUnwindSafe(|| solver.solve_with(limited(true), &mut panics)));
    assert!(caught.is_err());
    assert_eq!(panics.pricer.added, vec![(7, Col::new(3))]);
    assert_eq!(solver.program().num_cols(), 4);
    // The backend reloads the grown program and solves it, B now an ordinary column.
    let mut pricer = Pricer::new(patterns, 3..11, Pick::Every);
    pricer.added.push((7, Col::new(3)));
    let solved = solver.solve_with(limited(true), &mut pricer).unwrap();
    assert!(close(optimal(solved.outcome).objective, 4.5, 1e-9));
}

/// Prices as a `Pricer` and interrupts at the first poll after the first pricing round.
struct Interrupts {
    pricer: Pricer,
    interrupted: bool,
}

impl Callback<Scip> for Interrupts {
    fn events(&self) -> Events {
        Events::POLL
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        self.pricer.price_round(ctx);
    }
    fn poll(&mut self, ctx: &mut culmen_scip::ScipPoll<'_>) {
        if !self.pricer.rounds.is_empty() && !self.interrupted {
            self.interrupted = true;
            ctx.interrupt();
        }
    }
}

#[test]
fn test_an_interrupt_during_pricing_stops_the_solve_and_keeps_the_columns() {
    let (program, patterns) = cutting_stock(true);
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut interrupts = Interrupts {
        pricer: Pricer::new(patterns, 3..11, Pick::Every),
        interrupted: false,
    };
    let solved = solver.solve_with(limited(true), &mut interrupts).unwrap();
    assert!(interrupts.interrupted);
    let Outcome::Stopped { reason, best, .. } = solved.outcome else {
        panic!("expected a stop, got {:?}", solved.outcome)
    };
    assert_eq!(reason, StopReason::Interrupted);
    let added = interrupts.pricer.added.len();
    assert!(added >= 1);
    assert_eq!(solver.program().num_cols(), 3 + added);
    if let Some(best) = best {
        assert_eq!(best.values.len(), 3 + added);
    }
}

/// Prices as a `Pricer`, and holds the pattern B to at most one roll as a lazy constraint once it is priced.
struct AtMostOneB {
    pricer: Pricer,
    /// Every candidate the lazy callback accepted.
    accepted: Vec<TiVec<Col, f64>>,
    calls: usize,
}

impl Callback<Scip> for AtMostOneB {
    fn events(&self) -> Events {
        Events::LAZY
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        self.pricer.price_round(ctx);
    }
    fn lazy(&mut self, ctx: &mut culmen_scip::ScipLazy<'_>) {
        self.calls += 1;
        assert!(self.calls < MAX_CALLS, "more lazy calls than a tiny program needs");
        let candidate = ctx.candidate();
        if let Some(b) = self.pricer.col_of(7)
            && candidate[b] > 1.0 + 1e-6
        {
            ctx.add_lazy(b.leq(1)).expect("a constraint on a priced column");
            return;
        }
        self.accepted.push(candidate.to_owned());
    }
}

#[test]
fn test_a_lazy_constraint_on_a_priced_column_holds_and_the_optimum_stays_five() {
    let (program, patterns) = cutting_stock(true);
    let full = full(&program, &patterns);
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut callback = AtMostOneB {
        pricer: Pricer::new(patterns, 3..11, Pick::Every),
        accepted: Vec::new(),
        calls: 0,
    };
    let solved = solver.solve_with(limited(true), &mut callback).unwrap();
    let best = optimal(solved.outcome);
    assert!(close(best.objective, 5.0, 1e-9), "{best:?}");
    let b = callback.pricer.col_of(7).expect("B is priced");
    assert!(best.values[b] <= 1.0 + 1e-6, "{best:?}");
    assert!(full.max_violation(&in_full(&callback.pricer, 3, &best.values)) <= 1e-6);
    for candidate in &callback.accepted {
        assert!(candidate.get(b).is_none_or(|&v| v <= 1.0 + 1e-6), "{candidate:?}");
    }
}

/// Prices as a `Pricer`, and adds a cut on the pattern B at every node once it is priced.
struct CutsB {
    pricer: Pricer,
    cuts: usize,
    calls: usize,
}

impl Callback<Scip> for CutsB {
    fn events(&self) -> Events {
        Events::CUT
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        self.pricer.price_round(ctx);
    }
    fn cut(&mut self, ctx: &mut culmen_scip::ScipCut<'_>) {
        self.calls += 1;
        assert!(self.calls < MAX_CALLS, "more cut calls than a tiny program needs");
        if let Some(b) = self.pricer.col_of(7) {
            assert_eq!(ctx.relaxation().len(), 3 + self.pricer.added.len());
            ctx.add_cut(b.geq(0)).expect("a cut on a priced column");
            self.cuts += 1;
        }
    }
}

#[test]
fn test_a_cut_may_name_a_priced_column() {
    let (program, patterns) = cutting_stock(true);
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut callback = CutsB {
        pricer: Pricer::new(patterns, 3..11, Pick::Every),
        cuts: 0,
        calls: 0,
    };
    let solved = solver.solve_with(limited(true), &mut callback).unwrap();
    assert!(close(optimal(solved.outcome).objective, 5.0, 1e-9));
    assert!(callback.cuts > 0);
}

/// Prices as a `Pricer`, supplies one solution at the first heuristic call once every column it names is held, and
/// records every feasible solution SCIP reports.
struct Supplies {
    pricer: Pricer,
    /// The solution, by watched column; columns not watched yet read zero.
    solution: Vec<(usize, f64)>,
    results: Vec<Result<(), ActionError>>,
    feasible: Vec<TiVec<Col, f64>>,
}

impl Callback<Scip> for Supplies {
    fn events(&self) -> Events {
        Events::HEURISTIC | Events::FEASIBLE
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        self.pricer.price_round(ctx);
    }
    fn heuristic(&mut self, ctx: &mut culmen_scip::ScipHeuristic<'_>) {
        if !self.results.is_empty() {
            return;
        }
        let n = 3 + self.pricer.added.len();
        let mut values: TiVec<Col, f64> = vec![0.0; n].into();
        for &(i, value) in &self.solution {
            let col = if i < 3 {
                Some(Col::new(i))
            } else {
                self.pricer.col_of(i)
            };
            match col {
                Some(col) if col.index() < n => values[col] = value,
                _ => return,
            }
        }
        self.results.push(ctx.supply(&values));
    }
    fn feasible(&mut self, ctx: &mut culmen_scip::ScipSolution<'_>) {
        assert!(self.feasible.len() < MAX_CALLS, "more solutions than any test expects");
        self.feasible.push(ctx.values().to_owned());
    }
}

#[test]
fn test_a_heuristic_solution_through_priced_columns_is_stored_with_their_values() {
    // A + 2B + 2F, five rolls: 3 + 4 >= 7 pieces of width 3, 2 >= 2 of width 4, 4 >= 3 of width 5. B is priced at the
    // root, where the LP's 9/2 is fractional, so the heuristic after the root holds it. Presolve and SCIP's own
    // heuristics are off, so the supplied solution is the one SCIP stores, with B's value on B's priced variable.
    let (program, patterns) = cutting_stock(true);
    let full = full(&program, &patterns);
    let options = ScipOptions {
        heuristics: Some(culmen_scip::Emphasis::Off),
        ..ScipOptions::default()
    };
    let mut solver = Solver::new(Scip::with_options(options), program).unwrap();
    let mut callback = Supplies {
        pricer: Pricer::new(patterns, [7], Pick::Every),
        solution: vec![(0, 1.0), (7, 2.0), (2, 2.0)],
        results: Vec::new(),
        feasible: Vec::new(),
    };
    let solved = solver.solve_with(limited(false), &mut callback).unwrap();
    let best = optimal(solved.outcome);
    assert!(close(best.objective, 5.0, 1e-9), "{best:?}");
    assert!(full.max_violation(&in_full(&callback.pricer, 3, &best.values)) <= 1e-6);
    assert_eq!(callback.results, [Ok(())]);
    let b = callback.pricer.col_of(7).expect("B is priced");
    let supplied = |values: &TiVec<Col, f64>| {
        let at = |col: Col, want: f64| values.get(col).is_some_and(|&v| (v - want).abs() <= 1e-9);
        at(Col::new(0), 1.0) && at(Col::new(1), 0.0) && at(Col::new(2), 2.0) && at(b, 2.0)
    };
    assert!(
        callback.feasible.iter().any(supplied),
        "no reported solution holds the supplied values: {:?}",
        callback.feasible
    );
}

#[test]
fn test_a_heuristic_solution_presolve_cannot_hold_is_refused_with_the_reason() {
    // Fixture 1 with a column of cost 1 in no row, which presolve fixes at zero; a solution setting it to one does
    // not fit SCIP's problem and is refused, while one leaving it at zero is taken.
    let (program, patterns) = cutting_stock(false);
    let mut plan = ProgramPlan::from(program);
    let z = plan.col(ProgramColumn::continuous(1.0, bounds!(0.0, 5.0)));
    let program = plan.build().unwrap();
    let mut patterns = patterns;
    patterns.insert(3, column(1.0, bounds!(0.0, 5.0), false, &[]));
    for (z_value, refused) in [(1.0, true), (0.0, false)] {
        let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
        let mut callback = Supplies {
            pricer: Pricer::new(patterns.clone(), [], Pick::Every),
            solution: vec![(0, 3.0), (1, 1.0), (2, 2.0)],
            results: Vec::new(),
            feasible: Vec::new(),
        };
        callback.pricer.added.push((3, z));
        callback.solution.push((3, z_value));
        let solved = solver.solve_with(limited(true), &mut callback).unwrap();
        assert!(close(optimal(solved.outcome).objective, 29.0 / 6.0, 1e-9));
        let result = callback.results.first().expect("the heuristic ran");
        if refused {
            assert!(
                matches!(result, Err(ActionError::Refused(reason)) if reason.contains("presolve")),
                "{result:?}"
            );
        } else {
            assert!(result.is_ok(), "{result:?}");
        }
    }
}

#[test]
fn test_a_pricing_solve_starts_from_a_start_over_the_programs_columns() {
    let (program, patterns) = cutting_stock(true);
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut pricer = Pricer::new(patterns, 3..11, Pick::Every);
    // A = 3, D = 1, F = 2: six rolls, feasible.
    let request = limited(true).start(vec![3.0, 1.0, 2.0]);
    let solved = solver.solve_with(request, &mut pricer).unwrap();
    assert!(close(optimal(solved.outcome).objective, 5.0, 1e-9));
}

#[test]
fn test_pricing_a_program_with_a_quadratic_objective_is_refused_before_scip_runs() {
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(0.0, bounds!(0.0, 4.0)));
    plan.row(bounds!(1.0, f64::INFINITY), [(x, 1.0)]);
    plan.quadratic = vec![(x, x, 1.0)];
    let program = plan.build().unwrap();
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut pricer = Pricer::new(vec![], [], Pick::Every);
    let refused = solver.solve_with(limited(true), &mut pricer);
    assert!(
        matches!(refused, Err(SolveError::Backend(ScipError::PricingQuadratic))),
        "{refused:?}"
    );
    assert!(pricer.rounds.is_empty());
    // Without pricing the same solver solves it: x = 1, x² = 1.
    let solved = solver.solve(limited(true)).unwrap();
    assert!(close(optimal(solved.outcome).objective, 1.0, 1e-6));
}

/// Tries the refusals SCIP adds to the core's at the first round, then prices as a `Pricer`.
struct Refusals {
    pricer: Pricer,
    results: Vec<Result<Col, ActionError>>,
    bound: Option<Result<(), ActionError>>,
}

impl Callback<Scip> for Refusals {
    fn events(&self) -> Events {
        Events::NONE
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        if self.results.is_empty() {
            let row = Row::new(0);
            let huge = column(1.0, Bounds::NON_NEGATIVE, false, &[(row, 1e21)]);
            let summed = column(1.0, Bounds::NON_NEGATIVE, false, &[(row, 6e19), (row, 6e19)]);
            let upper = column(1.0, bounds!(0.0, 1e25), false, &[(row, 1.0)]);
            let mut named = column(1.0, Bounds::NON_NEGATIVE, false, &[(row, 1.0)]);
            named.column.name = Some("a\0b".to_owned());
            let excludes = column(1.0, bounds!(1.0, 2.0), false, &[(row, 1.0)]);
            for c in [huge, summed, upper, named, excludes] {
                self.results.push(ctx.add_col(c));
            }
            assert_eq!(ctx.local_bounds(Col::new(0)), Some(Bounds::NON_NEGATIVE));
            assert_eq!(ctx.local_bounds(Col::new(99)), None);
            self.bound = Some(ctx.report_bound(f64::NAN));
        }
        self.pricer.price_round(ctx);
    }
}

#[test]
fn test_scip_refuses_a_column_it_cannot_hold_before_it_takes_an_id() {
    let (program, patterns) = cutting_stock(false);
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut callback = Refusals {
        pricer: Pricer::new(patterns, 3..11, Pick::Every),
        results: Vec::new(),
        bound: None,
    };
    let solved = solver.solve_with(limited(true), &mut callback).unwrap();
    for result in &callback.results[..4] {
        assert!(matches!(result, Err(ActionError::Refused(_))), "{:?}", callback.results);
    }
    assert!(
        matches!(callback.results[4], Err(ActionError::ExcludesZero { col, .. }) if col == Col::new(3)),
        "{:?}",
        callback.results
    );
    assert!(matches!(callback.bound, Some(Err(ActionError::NotFiniteBound { .. }))));
    // No refused column took an id: B is the first column priced.
    assert_eq!(callback.pricer.added, vec![(7, Col::new(3))]);
    assert!(close(optimal(solved.outcome).objective, 4.5, 1e-9));
}

/// Branches early at the first round, with a bound, and prices exactly afterwards.
struct Early {
    pricer: Pricer,
    early: Option<Result<(), ActionError>>,
}

impl Callback<Scip> for Early {
    fn events(&self) -> Events {
        Events::NONE
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        if self.early.is_none() && matches!(ctx.mode(), PriceMode::ReducedCost { .. }) {
            self.early = Some(ctx.report_bound(4.0));
            ctx.branch_early();
            return;
        }
        self.pricer.price_round(ctx);
    }
}

#[test]
fn test_branching_early_with_a_bound_still_reaches_the_integer_optimum() {
    let (program, patterns) = cutting_stock(true);
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut early = Early {
        pricer: Pricer::new(patterns, 3..11, Pick::Every),
        early: None,
    };
    let solved = solver.solve_with(limited(true), &mut early).unwrap();
    assert_eq!(early.early, Some(Ok(())));
    assert!(close(optimal(solved.outcome).objective, 5.0, 1e-9));
}

#[test]
fn test_branching_early_where_scip_cannot_branch_fails_the_solve_with_a_typed_error() {
    let (program, patterns) = cutting_stock(false);
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut early = Early {
        pricer: Pricer::new(patterns, 3..11, Pick::Every),
        early: None,
    };
    let failed = solver.solve_with(limited(true), &mut early);
    assert!(
        matches!(failed, Err(SolveError::Backend(ScipError::BranchEarly))),
        "{failed:?}"
    );
}

#[test]
fn test_a_degenerate_node_lp_hands_out_duals_that_price_its_bounds() {
    // Maximize 2x0 subject to x1 - x0 - x2 = 0, x0 in [0, 2] and x1, x2 in [0, 1], starting without x1: the start
    // forces x0 = 0, and every dual of the row that prices x0 out at its lower bound is at most -2, so x1, of
    // reduced cost 0 - y·1 >= 2, improves. The full program's optimum is x0 = x1 = 1, worth 2. A proptest seed found
    // SCIP's LP presolving handing out the dual 0, which prices nothing and ends at 0.
    let mut plan = ProgramPlan::new();
    plan.sense = Sense::Maximize;
    let x0 = plan.col(ProgramColumn::continuous(2.0, bounds!(0.0, 2.0)));
    let x2 = plan.col(ProgramColumn::continuous(0.0, bounds!(0.0, 1.0)));
    plan.row(bounds!(0.0, 0.0), [(x0, -1.0), (x2, -1.0)]);
    let program = plan.build().unwrap();
    for presolve in [true, false] {
        let x1 = column(0.0, bounds!(0.0, 1.0), false, &[(Row::new(0), 1.0)]);
        let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
        let mut pricer = Pricer::new(vec![x1], [0], Pick::Every);
        let solved = solver.solve_with(limited(presolve), &mut pricer).unwrap();
        assert!(pricer.rounds[0].duals[0] <= -2.0 + 1e-9, "{:?}", pricer.rounds);
        assert!(
            close(optimal(solved.outcome).objective, 2.0, 1e-9),
            "{:?}",
            pricer.rounds
        );
    }
}

#[test]
fn test_no_cut_from_the_objective_over_the_columns_held_cuts_off_a_column_pricing_needs() {
    // Maximize 3x1 - 2x2 - 3x4 subject to -2x1 + x2 + 2x4 >= 1, every column in [0, 1] and x4 integer, starting with
    // x2 alone. Pricing adds x4, the root LP is x4 = 1/2 at -3/2, and at x4 = 1 the row is slack, its dual 0, and x1
    // of reduced cost 3 improves: the optimum is x4 = 1, x1 = 1/2, worth -3/2. A proptest seed found SCIP's
    // aggregation separator deriving x4 <= 0 at the root from the objective cutoff over the columns held, which no
    // column priced later obeys, and the solve ending at -2.
    let mut plan = ProgramPlan::new();
    plan.sense = Sense::Maximize;
    let x2 = plan.col(ProgramColumn::continuous(-2.0, bounds!(0.0, 1.0)));
    plan.row(bounds!(1.0, f64::INFINITY), [(x2, 1.0)]);
    let program = plan.build().unwrap();
    let r = Row::new(0);
    let x1 = column(3.0, bounds!(0.0, 1.0), false, &[(r, -2.0)]);
    let x4 = column(-3.0, bounds!(0.0, 1.0), true, &[(r, 2.0)]);
    for presolve in [true, false] {
        let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
        let mut pricer = Pricer::new(vec![x1.clone(), x4.clone()], [0, 1], Pick::Every);
        let solved = solver.solve_with(limited(presolve), &mut pricer).unwrap();
        let best = optimal(solved.outcome);
        assert!(close(best.objective, -1.5, 1e-9), "{best:?}, {:?}", pricer.rounds);
    }
}

/// Prices the withheld columns of a split, records every bound a poll reports, and counts its rounds.
struct SplitPricer {
    withheld: Vec<NewCol>,
    pick: Pick,
    added: Vec<(usize, Col)>,
    rounds: usize,
    bounds: Vec<f64>,
}

impl Callback<Scip> for SplitPricer {
    fn events(&self) -> Events {
        Events::POLL
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        self.rounds += 1;
        assert!(
            self.rounds < MAX_ROUNDS,
            "more pricing rounds than a tiny program needs"
        );
        let open = (0..self.withheld.len())
            .filter(|&i| !self.added.iter().any(|&(j, _)| j == i) && ctx.improves(&self.withheld[i]));
        let chosen: Vec<usize> = match self.pick {
            Pick::Every => open.collect(),
            Pick::First => open.take(1).collect(),
            Pick::Most => open
                .max_by(|&a, &b| Pricer::gain(ctx, &self.withheld[a]).total_cmp(&Pricer::gain(ctx, &self.withheld[b])))
                .into_iter()
                .collect(),
        };
        for i in chosen {
            let col = ctx.add_col(self.withheld[i].clone()).expect("a column SCIP takes");
            self.added.push((i, col));
        }
    }
    fn poll(&mut self, ctx: &mut culmen_scip::ScipPoll<'_>) {
        if let Some(bound) = ctx.progress().bound {
            self.bounds.push(bound);
        }
    }
}

/// The relative tolerance of the property test's comparisons with the exact reference.
const TOLERANCE: f64 = 1e-5;

/// Checks one pricing solve of a split, on a backend with `options`, against the reference optimum of its full program.
fn check_split(
    split: &PricedSplit,
    reference: &Reference,
    pick: Pick,
    presolve: bool,
    options: &ScipOptions,
) -> Result<(), TestCaseError> {
    let PricedSplit {
        full,
        start,
        kept,
        withheld,
    } = split;
    let mut solver = Solver::new(Scip::with_options(options.clone()), start.clone()).unwrap();
    let mut pricer = SplitPricer {
        withheld: withheld.iter().map(|(_, c)| c.clone()).collect(),
        pick,
        added: Vec::new(),
        rounds: 0,
        bounds: Vec::new(),
    };
    let solved = solver
        .solve_with(limited(presolve), &mut pricer)
        .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
    let context = format!("{pick:?} presolve {presolve}: {solved:?}, reference {reference:?}");
    let program = solver.program();
    // 3: the program holds the kept and the added columns, and the solution a value for each.
    prop_assert_eq!(program.num_cols(), kept.len() + pricer.added.len(), "{}", context);
    // 4: every added column is the withheld column it stands for.
    for &(i, col) in &pricer.added {
        let want = &withheld[i].1;
        prop_assert_eq!(&program.columns()[col], &want.column, "{}", context);
        let entries: Vec<(Row, f64)> = program
            .rows()
            .filter_map(|row| {
                program
                    .row(row)
                    .entries()
                    .find(|&(c, _)| c == col)
                    .map(|(_, a)| (row, a))
            })
            .collect();
        prop_assert_eq!(&entries, &want.entries, "{}", context);
    }
    let sense = full.sense();
    match (reference, &solved.outcome) {
        // 1: infeasible exactly when the reference is.
        (Reference::Infeasible, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded) => {}
        (Reference::Optimal(_), Outcome::Optimal(best)) => {
            let opt = reference.objective().unwrap();
            // 2: the optimum is the reference's. SCIP meets a row to its relative feasibility tolerance, 1e-6 of the
            // row's side, which on these sides allows a few millionths of violation and of objective; the checks
            // allow 1e-5, as the reference test of the plain solves does.
            prop_assert!(close(best.objective, opt, TOLERANCE), "{}", context);
            prop_assert_eq!(best.values.len(), program.num_cols(), "{}", context);
            // 5: mapped back to the full program, the point is feasible and optimal there, to the tolerance of 2, and
            // its objective there is the one the solve reported.
            let mut point: TiVec<Col, f64> = vec![0.0; full.num_cols()].into();
            for (col, &original) in kept.iter_enumerated() {
                point[original] = best.values[col];
            }
            for &(i, col) in &pricer.added {
                point[withheld[i].0] = best.values[col];
            }
            prop_assert!(full.max_violation(&point) <= TOLERANCE, "{:?}: {}", point, context);
            let objective = full.objective_value(&point);
            prop_assert!(close(objective, best.objective, 1e-9), "{:?}: {}", point, context);
            prop_assert!(close(objective, opt, TOLERANCE), "{:?}: {}", point, context);
        }
        _ => prop_assert!(false, "{}", context),
    }
    // 6: no bound a poll reported beats the optimum.
    if let Some(opt) = reference.objective() {
        let tolerance = TOLERANCE * opt.abs().max(1.0);
        for &bound in &pricer.bounds {
            let valid = match sense {
                Sense::Minimize => bound <= opt + tolerance,
                Sense::Maximize => bound >= opt - tolerance,
            };
            prop_assert!(valid, "bound {} beats the optimum {}: {}", bound, opt, context);
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4000))]

    /// Pricing the withheld columns of a random program, with every improving column or the most improving one, with
    /// presolve on and off, reaches the optimum exact enumeration finds for the full program.
    #[test]
    fn test_pricing_agrees_with_enumeration_of_the_full_program(split in priced_split()) {
        let reference = solve(&split.full);
        for pick in [Pick::Every, Pick::Most] {
            for presolve in [true, false] {
                check_split(&split, &reference, pick, presolve, &ScipOptions::default())?;
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// With the LP solver's objective limit turned back on by option (`lp/disablecutoff = 0`), SCIP could stop a node
    /// LP at the cutoff (`OBJLIMIT`) and still price it; a pricing solve keeps the limit off, so it hands the callback
    /// only optimal or infeasible node LPs and reaches the reference optimum.
    #[test]
    fn test_pricing_with_the_lp_cutoff_turned_on_by_option_agrees_with_enumeration(split in priced_split()) {
        let reference = solve(&split.full);
        let options = ScipOptions::default().param("lp/disablecutoff", 0);
        for presolve in [true, false] {
            check_split(&split, &reference, Pick::Most, presolve, &options)?;
        }
    }
}
