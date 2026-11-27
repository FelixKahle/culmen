// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pricing event, driven by the scripted backend: the numbering of priced columns, the lengths later events
//! check against, the refusals, the reduced costs of two hand-worked fixtures, and a solver that keeps every priced
//! column whether the solve returned, failed, was interrupted or panicked.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};

use culmen_core::backend::{Hooks, Priced, Unsupported};
use culmen_core::script::Found;
use culmen_core::script::{Script, ScriptError, ScriptPrice, Step};
use culmen_core::{
    ActionError, Analysis, AnalysisError, Backend, Bounds, Callback, Col, Context, Duals, Event, Events, Integrality,
    LazyContext, LazyHandler, Linear, ModelClass, ModelClasses, NewCol, Outcome, Phase, PollContext, PriceContext,
    PriceMode, Pricer, Prices, Program, ProgramColumn, ProgramPlan, Progress, Row, Sense, Settings, Solution, Solved,
    Solver, StopReason, TiSlice, TiVec, bounds,
};
use culmen_testing::mirror::{Call, Mirror};

/// The comparison of a hand-worked number with one computed in floating point.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

fn progress() -> Progress {
    Progress::at(Duration::from_secs(1), Phase::Simplex)
}

fn price(mode: PriceMode, duals: impl Into<Vec<f64>>) -> Step {
    Step::Price {
        progress: progress(),
        mode,
        duals: TiVec::from(duals.into()),
    }
}

fn optimum(objective: f64, values: impl Into<Vec<f64>>) -> Outcome {
    Outcome::Optimal(Solution {
        objective,
        values: TiVec::from(values.into()),
    })
}

/// A column of cost `cost` in `[0, ∞)` with the given entries.
fn column(cost: f64, entries: &[(Row, f64)]) -> culmen_core::NewCol {
    culmen_core::NewCol {
        column: ProgramColumn::continuous().cost(cost),
        entries: entries.to_vec(),
    }
}

/// Minimize `x + y` over `x, y >= 0` and the rows `x + y >= 1` and `x <= 3`.
fn small() -> Program {
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous().cost(1.0));
    let y = plan.col(ProgramColumn::continuous().cost(1.0));
    plan.row((bounds!(1.0, f64::INFINITY), [(x, 1.0), (y, 1.0)]));
    plan.row((bounds!(f64::NEG_INFINITY, 3.0), [(x, 1.0)]));
    plan.build().unwrap()
}

/// Watches the reduced cost of every column in `watched` at each round and adds, from those `withheld`, every one
/// not added yet that improves, or only the first when `first_only`.
struct Watching {
    watched: Vec<culmen_core::NewCol>,
    withheld: Vec<usize>,
    first_only: bool,
    added: Vec<(usize, Col)>,
    /// Per round: the mode, and each watched column's reduced cost and whether it improves.
    rounds: Vec<(PriceMode, Vec<(f64, bool)>)>,
}

impl Watching {
    fn new(watched: Vec<culmen_core::NewCol>, withheld: impl IntoIterator<Item = usize>, first_only: bool) -> Self {
        Self {
            watched,
            withheld: withheld.into_iter().collect(),
            first_only,
            added: Vec::new(),
            rounds: Vec::new(),
        }
    }

    fn reduced_costs(&self, round: usize) -> Vec<f64> {
        self.rounds[round].1.iter().map(|&(d, _)| d).collect()
    }

    fn improving(&self, round: usize) -> Vec<usize> {
        let seen = &self.rounds[round].1;
        (0..seen.len()).filter(|&i| seen[i].1).collect()
    }
}

impl<B: Prices> Pricer<B> for Watching {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        let seen = self
            .watched
            .iter()
            .map(|c| (ctx.reduced_cost(c), ctx.improves(c)))
            .collect();
        self.rounds.push((ctx.mode(), seen));
        for &i in &self.withheld {
            let done = self.added.iter().any(|&(j, _)| j == i);
            if !done && ctx.improves(&self.watched[i]) {
                let col = ctx.add_col(self.watched[i].clone()).unwrap();
                self.added.push((i, col));
                if self.first_only {
                    break;
                }
            }
        }
    }
}

/// The first hand-worked fixture: cutting stock of rolls of width 10 into pieces of width 3, 4 and 5 with demands 7,
/// 2 and 3, every pattern costing one roll. The program starts with the patterns (3,0,0), (0,2,0) and (0,0,2).
///
/// The master LP over the three starting patterns is `min a + d + f` with `3a >= 7`, `2d >= 2`, `2f >= 3`: optimal at
/// `a = 7/3, d = 1, f = 3/2`, worth 29/6, with duals (1/3, 1/2, 1/2). B then prices at `1 - (2/3 + 1/2) = -1/6` and
/// improves, E at `1 - (1/2 + 1/2) = 0` and does not. With B, the LP is optimal at `a = 1, d = 0, f = 3/2, b = 2`,
/// worth 9/2, with duals (1/3, 1/3, 1/2) from `3y1 = 1`, `2y1 + y2 = 1` and `2y3 = 1`, and nothing improves.
///
/// Returns the program and every pattern: the three it holds, then the eight it withholds in the order
/// (1,0,0), (2,0,0), (0,1,0), (1,1,0), B = (2,1,0), (0,0,1), C = (1,0,1), E = (0,1,1).
fn cutting_stock() -> (Program, Vec<culmen_core::NewCol>) {
    let mut plan = ProgramPlan::new();
    let start = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
    let cols: Vec<Col> = start
        .iter()
        .map(|_| plan.col(ProgramColumn::continuous().cost(1.0)))
        .collect();
    let demands = [
        bounds!(7.0, f64::INFINITY),
        bounds!(2.0, f64::INFINITY),
        bounds!(3.0, f64::INFINITY),
    ];
    let rows: Vec<Row> = demands
        .into_iter()
        .enumerate()
        .map(|(piece, demand)| {
            let terms = cols.iter().zip(&start).map(|(&col, p)| (col, p[piece]));
            plan.row((demand, terms.filter(|&(_, a)| a != 0.0)))
        })
        .collect();
    let withheld = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let patterns = start
        .iter()
        .chain(&withheld)
        .map(|p| {
            let entries: Vec<(Row, f64)> = rows
                .iter()
                .zip(p)
                .map(|(&r, &a)| (r, a))
                .filter(|&(_, a)| a != 0.0)
                .collect();
            column(1.0, &entries)
        })
        .collect();
    (plan.build().unwrap(), patterns)
}

#[test]
fn test_fixture_one_prices_pattern_b_and_nothing_after_it() {
    let (program, patterns) = cutting_stock();
    let (b, e) = (3 + 4, 3 + 7);
    let steps = vec![
        price(PriceMode::ReducedCost { objective: 29.0 / 6.0 }, [1.0 / 3.0, 0.5, 0.5]),
        price(PriceMode::ReducedCost { objective: 4.5 }, [1.0 / 3.0, 1.0 / 3.0, 0.5]),
    ];
    let end = optimum(4.5, [1.0, 0.0, 1.5, 2.0]);
    let mut solver = Solver::new(Script::new(steps, end), program).unwrap();
    let mut pricer = Watching::new(patterns, 3..11, false);
    let solved = solver.run().pricer(&mut pricer).solve().unwrap();

    // Round 1: the hand-worked reduced costs; only B improves, E sits on the boundary and does not.
    let first = [
        0.0,
        0.0,
        0.0,
        2.0 / 3.0,
        1.0 / 3.0,
        0.5,
        1.0 / 6.0,
        -1.0 / 6.0,
        0.5,
        1.0 / 6.0,
        0.0,
    ];
    let found = pricer.reduced_costs(0);
    assert!(found.iter().zip(&first).all(|(&a, &b)| close(a, b)), "{found:?}");
    assert!(close(found[b], -1.0 / 6.0));
    assert_eq!(pricer.improving(0), [b]);
    assert!(!pricer.rounds[0].1[e].1);
    // Round 2: nothing improves; B is basic, D prices at 1/3.
    let second = [
        0.0,
        1.0 / 3.0,
        0.0,
        2.0 / 3.0,
        1.0 / 3.0,
        2.0 / 3.0,
        1.0 / 3.0,
        0.0,
        0.5,
        1.0 / 6.0,
        1.0 / 6.0,
    ];
    let found = pricer.reduced_costs(1);
    assert!(found.iter().zip(&second).all(|(&a, &b)| close(a, b)), "{found:?}");
    assert!(pricer.improving(1).is_empty());

    assert_eq!(pricer.added, [(b, Col::new(3))]);
    let record = &solver.backend().record;
    assert_eq!(record.priced.len(), 2);
    assert_eq!(record.priced[0].added.clone().collect::<Vec<_>>(), [Col::new(3)]);
    assert_eq!(record.priced[1].added.len(), 0);
    let program = solver.program();
    assert_eq!(program.num_cols(), 4);
    let Outcome::Optimal(best) = solved.outcome else {
        panic!("expected the scripted optimum")
    };
    assert_eq!(best.values.len(), program.num_cols());
    assert!(program.max_violation(&best.values) <= 1e-12);
    assert!(close(program.objective_value(&best.values), 4.5));
}

/// The second hand-worked fixture: maximize `3x1 + 2x2 + 7x3` over `[0, 10]` each, subject to `x1 + x2 + 2x3 <= 4`
/// and `x1 + x3 >= 1`, starting with `x2` alone. Returns the program and the columns x2, x1, x3 in that order.
///
/// With `x2` alone the row `x1 + x3 >= 1` cannot hold, so the first round is Farkas, with multipliers `f0 <= 0 < f1`.
/// With `x1` the LP is optimal at `x1 = 4`, worth 12, with duals (3, 0), and `x3` prices at `7 - 2·3 = 1 > 0`, which
/// improves a maximization; with `x3` it is optimal at `x3 = 2`, worth 14, with duals (7/2, 0).
fn farkas_then_max() -> (Program, Vec<culmen_core::NewCol>) {
    let mut plan = ProgramPlan::new();
    plan.sense = Sense::Maximize;
    let x2 = plan.col(ProgramColumn::continuous().cost(2.0).bounds(bounds!(0.0, 10.0)));
    let r0 = plan.row((bounds!(f64::NEG_INFINITY, 4.0), [(x2, 1.0)]));
    let r1 = plan.row((bounds!(1.0, f64::INFINITY), []));
    let col = |cost, entries: Vec<(Row, f64)>| culmen_core::NewCol {
        column: ProgramColumn::continuous().cost(cost).bounds(bounds!(0.0, 10.0)),
        entries,
    };
    let columns = vec![
        col(2.0, vec![(r0, 1.0)]),
        col(3.0, vec![(r0, 1.0), (r1, 1.0)]),
        col(7.0, vec![(r0, 2.0), (r1, 1.0)]),
    ];
    (plan.build().unwrap(), columns)
}

#[test]
fn test_fixture_two_prices_by_farkas_then_by_the_maximizing_sign() {
    let (program, columns) = farkas_then_max();
    let steps = vec![
        price(PriceMode::Farkas, [-0.2, 1.0]),
        price(PriceMode::ReducedCost { objective: 12.0 }, [3.0, 0.0]),
        price(PriceMode::ReducedCost { objective: 14.0 }, [3.5, 0.0]),
    ];
    let end = optimum(14.0, [0.0, 0.0, 2.0]);
    let mut solver = Solver::new(Script::new(steps, end), program).unwrap();
    let mut pricer = Watching::new(columns, [1, 2], true);
    solver.run().pricer(&mut pricer).solve().unwrap();

    // Farkas: the cost counts as zero, so x2 prices at -f0 = 0.2 and does not help; x1 at -0.8 and x3 at -0.6 do.
    assert_eq!(pricer.rounds[0].0, PriceMode::Farkas);
    let found = pricer.reduced_costs(0);
    assert!(
        close(found[0], 0.2) && close(found[1], -0.8) && close(found[2], -0.6),
        "{found:?}"
    );
    assert_eq!(pricer.improving(0), [1, 2]);
    // Maximizing at 12: x2 -1 and x1 0 do not improve, x3 at +1 does.
    let found = pricer.reduced_costs(1);
    assert!(
        close(found[0], -1.0) && close(found[1], 0.0) && close(found[2], 1.0),
        "{found:?}"
    );
    assert_eq!(pricer.improving(1), [2]);
    // Maximizing at 14: nothing improves.
    let found = pricer.reduced_costs(2);
    assert!(
        close(found[0], -1.5) && close(found[1], -0.5) && close(found[2], 0.0),
        "{found:?}"
    );
    assert!(pricer.improving(2).is_empty());

    assert_eq!(pricer.added, [(1, Col::new(1)), (2, Col::new(2))]);
    let program = solver.program();
    assert_eq!(program.num_cols(), 3);
    let values: TiVec<Col, f64> = vec![0.0, 0.0, 2.0].into();
    assert!(program.max_violation(&values) <= 1e-12);
    assert!(close(program.objective_value(&values), 14.0));
}

/// Adds the given columns at every price step, recording what each add returned.
#[derive(Default)]
struct Adds {
    columns: Vec<culmen_core::NewCol>,
    results: Vec<Result<Col, ActionError>>,
}

impl Adds {
    fn of(columns: impl IntoIterator<Item = culmen_core::NewCol>) -> Self {
        Self {
            columns: columns.into_iter().collect(),
            results: Vec::new(),
        }
    }
}

impl<B: Prices> Pricer<B> for Adds {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        for column in &self.columns {
            self.results.push(ctx.add_col(column.clone()));
        }
    }
}

#[test]
fn test_a_priced_column_continues_the_numbering_and_is_kept_as_checked() {
    let program = small();
    let n = program.num_cols();
    let (r0, r1) = (Row::new(0), Row::new(1));
    let given = culmen_core::NewCol {
        column: ProgramColumn::continuous().cost(0.5).bounds(bounds!(0.0, 2.0)),
        entries: vec![(r1, 2.0), (r0, 1.0), (r1, -2.0), (r0, 0.5)],
    };
    let steps = vec![price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0])];
    let mut solver = Solver::new(Script::new(steps, optimum(1.0, [1.0, 0.0, 0.0])), program).unwrap();
    let mut adds = Adds::of([given.clone()]);
    solver.run().pricer(&mut adds).solve().unwrap();

    assert_eq!(adds.results, [Ok(Col::new(n))]);
    let program = solver.program();
    assert_eq!(program.num_cols(), n + 1);
    assert_eq!(program.columns()[Col::new(n)], given.column);
    // The entries of one row add up and a sum of zero is dropped.
    assert_eq!(program.row(r0).entries().next_back(), Some((Col::new(n), 1.5)));
    assert!(program.row(r1).entries().all(|(col, _)| col != Col::new(n)));
    let rounds = &solver.backend().record.priced;
    assert_eq!(rounds.len(), 1);
    assert_eq!(rounds[0].added.clone().collect::<Vec<_>>(), [Col::new(n)]);
    assert_eq!((rounds[0].bound, rounds[0].early), (None, false));
}

#[test]
fn test_a_refused_column_names_its_fault_and_takes_no_id() {
    let program = small();
    let (n, m) = (program.num_cols(), program.num_rows());
    let col = Col::new(n);
    let r0 = Row::new(0);
    let with = |column: ProgramColumn, entries: Vec<(Row, f64)>| culmen_core::NewCol { column, entries };
    let any = ProgramColumn::continuous().cost(1.0);
    let columns = [
        with(any.clone(), vec![(Row::new(m), 1.0)]),
        with(ProgramColumn::continuous().cost(f64::NAN), vec![]),
        with(any.clone(), vec![(r0, f64::NAN)]),
        with(any.clone(), vec![(r0, f64::MAX), (r0, f64::MAX)]),
        with(ProgramColumn::continuous().cost(1.0).bounds(bounds!(1.0, 2.0)), vec![]),
        with(
            ProgramColumn::semi_continuous().cost(1.0).bounds(bounds!(1.0, 2.0)),
            vec![],
        ),
        with(
            ProgramColumn::continuous().cost(1.0).bounds(bounds!(-1.0, 0.0)),
            vec![(r0, 1.0)],
        ),
    ];
    let steps = vec![price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0])];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), program).unwrap();
    let mut adds = Adds::of(columns);
    solver.run().pricer(&mut adds).solve().unwrap();
    assert_eq!(
        adds.results,
        [
            Err(ActionError::UnknownRow { row: Row::new(m) }),
            Err(ActionError::NonFinite { col }),
            Err(ActionError::NonFiniteEntry { row: r0, col }),
            Err(ActionError::NonFiniteEntry { row: r0, col }),
            Err(ActionError::ExcludesZero {
                col,
                bounds: bounds!(1.0, 2.0)
            }),
            Err(ActionError::Integrality {
                col,
                integrality: Integrality::SemiContinuous
            }),
            Ok(col),
        ]
    );
    assert_eq!(solver.program().num_cols(), n + 1);
}

#[test]
fn test_priced_refuses_a_class_the_backend_does_not_solve_and_reads_back_what_it_kept() {
    let program = small();
    let n = program.num_cols();
    let mut priced = Priced::new(&program, ModelClasses::of(&[ModelClass::Lp]));
    assert_eq!((priced.num_cols(), priced.cols().len()), (n, 0));
    let integer = culmen_core::NewCol::from(ProgramColumn::integer().cost(1.0).bounds(bounds!(0.0, 3.0)));
    assert_eq!(
        priced.push(integer),
        Err(ActionError::Integrality {
            col: Col::new(n),
            integrality: Integrality::Integer
        })
    );
    let given = column(2.0, &[(Row::new(1), 1.0), (Row::new(0), 0.0)]);
    assert_eq!(priced.push(given.clone()), Ok(Col::new(n)));
    let kept = column(2.0, &[(Row::new(1), 1.0)]);
    assert_eq!(priced.get(Col::new(n)), Some(&kept));
    assert_eq!(priced.get(Col::new(0)), None);
    assert_eq!(priced.get(Col::new(n + 1)), None);
    assert_eq!(priced.cols().collect::<Vec<_>>(), [Col::new(n)]);
    assert_eq!(priced.num_cols(), n + 1);
    assert_eq!(priced.into_cols(), [kept]);

    // A backend that solves MILPs takes an integer column.
    let mut priced = Priced::new(&program, ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]));
    let integer = culmen_core::NewCol::from(ProgramColumn::integer().cost(1.0).bounds(bounds!(0.0, 3.0)));
    assert_eq!(priced.push(integer), Ok(Col::new(n)));
}

/// Adds one column in each price step.
#[derive(Default)]
struct AddOne(Vec<Col>);

impl<B: Prices> Pricer<B> for AddOne {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        self.0.push(ctx.add_col(column(1.0, &[(Row::new(0), 1.0)])).unwrap());
    }
}

/// Adds a lazy constraint on `col` and on the column after it.
struct LazyOn {
    col: Col,
    results: Vec<Result<(), ActionError>>,
}

impl LazyHandler<Script> for LazyOn {
    fn lazy(&mut self, ctx: &mut <Script as culmen_core::LazyConstraints>::Lazy<'_>) {
        self.results.push(ctx.add_lazy(self.col.leq(1)));
        self.results.push(ctx.add_lazy(Col::new(self.col.index() + 1).leq(1)));
    }
}

#[test]
fn test_an_event_after_pricing_counts_the_priced_columns() {
    let program = small();
    let n = program.num_cols();
    let candidate = Found {
        progress: progress(),
        objective: 1.0,
        values: vec![1.0, 0.0, 0.0].into(),
    };
    let steps = vec![
        price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0]),
        Step::Lazy(candidate),
    ];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), program).unwrap();
    let mut pricer = AddOne::default();
    let mut lazy = LazyOn {
        col: Col::new(n),
        results: Vec::new(),
    };
    solver.run().pricer(&mut pricer).lazy(&mut lazy).solve().unwrap();
    assert_eq!(pricer.0, [Col::new(n)]);
    assert_eq!(
        lazy.results,
        [Ok(()), Err(ActionError::UnknownCol { col: Col::new(n + 1) })]
    );
}

/// Adds a column, then panics.
struct AddThenPanic;

impl<B: Prices> Pricer<B> for AddThenPanic {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        ctx.add_col(column(1.0, &[(Row::new(0), 1.0)])).unwrap();
        panic!("pricer failed");
    }
}

#[test]
fn test_a_panic_after_pricing_keeps_the_column_and_the_next_solve_loads_it() {
    let program = small();
    let n = program.num_cols();
    let steps = vec![price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0])];
    let script = Script::new(steps, optimum(1.0, [1.0, 0.0, 0.0]));
    let mut solver = Solver::new(Mirror::new(script), program).unwrap();
    let unwound = catch_unwind(AssertUnwindSafe(|| solver.run().pricer(&mut AddThenPanic).solve()));
    assert!(unwound.is_err());
    assert_eq!(solver.program().num_cols(), n + 1);
    assert_eq!(
        solver.program().row(Row::new(0)).entries().next_back(),
        Some((Col::new(n), 1.0))
    );

    let solved = solver.solve().unwrap();
    assert_eq!(solved.outcome, optimum(1.0, [1.0, 0.0, 0.0]));
    let Some(Call::Load(loaded)) = solver.backend().calls.last() else {
        panic!(
            "expected a load before the second solve, got {:?}",
            solver.backend().calls
        )
    };
    assert_eq!(loaded.num_cols(), n + 1);
    assert_eq!(solver.backend().held(), Some(solver.program()));
}

/// Interrupts at each poll.
struct Interrupt;

impl<B: Backend> Callback<B> for Interrupt {
    fn events(&self) -> Events {
        Event::Poll.into()
    }
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        ctx.interrupt();
    }
}

#[test]
fn test_an_interrupt_after_pricing_stops_the_solve_and_keeps_the_columns() {
    let program = small();
    let n = program.num_cols();
    let steps = vec![
        price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0]),
        Step::Poll(progress()),
        price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0]),
    ];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), program).unwrap();
    let (mut pricer, mut interrupt) = (AddOne::default(), Interrupt);
    let solved = solver
        .run()
        .pricer(&mut pricer)
        .callback(&mut interrupt)
        .solve()
        .unwrap();
    assert!(matches!(
        solved.outcome,
        Outcome::Stopped {
            reason: StopReason::Interrupted,
            ..
        }
    ));
    assert_eq!(pricer.0, [Col::new(n)]);
    assert_eq!(solver.program().num_cols(), n + 1);
}

#[test]
fn test_a_pair_prices_in_one_context_the_first_callback_first() {
    let program = small();
    let n = program.num_cols();
    let steps = vec![price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0])];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), program).unwrap();
    let mut a = Adds::of([column(1.0, &[(Row::new(0), 1.0)])]);
    let mut b = Adds::of([column(2.0, &[(Row::new(0), 1.0)])]);
    solver.run().pricer(&mut (&mut a, &mut b)).solve().unwrap();
    assert_eq!(a.results, [Ok(Col::new(n))]);
    assert_eq!(b.results, [Ok(Col::new(n + 1))]);
    let program = solver.program();
    assert_eq!(program.columns()[Col::new(n + 1)].cost, 2.0);
    let rounds = &solver.backend().record.priced;
    assert_eq!(rounds.len(), 1);
    assert_eq!(
        rounds[0].added.clone().collect::<Vec<_>>(),
        [Col::new(n), Col::new(n + 1)]
    );
}

/// Reports bounds and ends pricing early, recording what each report returned.
#[derive(Default)]
struct Bounding(Vec<Result<(), ActionError>>);

impl<B: Prices> Pricer<B> for Bounding {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        match ctx.mode() {
            PriceMode::Farkas => {
                self.0.push(ctx.report_bound(0.0));
                self.0.push(ctx.branch_early());
            }
            PriceMode::ReducedCost { objective } => {
                self.0.push(ctx.report_bound(f64::NAN));
                self.0.push(ctx.report_bound(objective - 1.0));
                self.0.push(ctx.branch_early());
            }
        }
    }
}

#[test]
fn test_a_bound_and_an_early_stop_are_recorded_and_farkas_takes_neither() {
    let program = small();
    let steps = vec![
        price(PriceMode::Farkas, [1.0, 0.0]),
        price(PriceMode::ReducedCost { objective: 3.0 }, [1.0, 0.0]),
    ];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), program).unwrap();
    let mut bounding = Bounding::default();
    solver.run().pricer(&mut bounding).solve().unwrap();
    assert!(
        matches!(
            &bounding.0[..2],
            [Err(ActionError::Refused(_)), Err(ActionError::Refused(_))]
        ),
        "{:?}",
        bounding.0
    );
    assert!(matches!(bounding.0[2], Err(ActionError::NonFiniteBound { bound }) if bound.is_nan()));
    assert_eq!(bounding.0[3..], [Ok(()), Ok(())]);
    let rounds = &solver.backend().record.priced;
    assert_eq!(rounds.len(), 2);
    assert_eq!(
        (rounds[0].bound, rounds[0].early, rounds[0].added.len()),
        (None, false, 0)
    );
    assert_eq!(
        (rounds[1].bound, rounds[1].early, rounds[1].added.len()),
        (Some(2.0), true, 0)
    );
    assert_eq!(solver.program().num_cols(), 2, "nothing priced, nothing appended");
}

/// A callback that leaves `events` at its default, so it asks for every event, and counts them.
struct EveryEvent(usize);

impl<B: Backend> Callback<B> for EveryEvent {
    fn poll(&mut self, _: &mut B::Poll<'_>) {
        self.0 += 1;
    }
}

#[test]
fn test_a_price_step_reaches_only_a_registered_pricer_and_never_a_callback_asking_for_every_event() {
    let steps = vec![
        price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0]),
        Step::Poll(progress()),
    ];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), small()).unwrap();
    let mut callback = EveryEvent(0);
    assert_eq!(Callback::<Script>::events(&callback), Events::ALL);
    solver.run().callback(&mut callback).solve().unwrap();
    assert_eq!(callback.0, 1);
    assert!(solver.backend().record.priced.is_empty());
    assert_eq!(solver.program().num_cols(), 2);
    assert_eq!(Script::EVENTS, Events::ALL);
}

/// A price context no solve makes, for a backend whose pricing is its own.
#[derive(Debug)]
enum NoRound {}

impl Context for NoRound {
    fn progress(&self) -> Progress {
        match *self {}
    }
}

impl PriceContext for NoRound {
    fn mode(&self) -> PriceMode {
        match *self {}
    }
    fn sense(&self) -> Sense {
        match *self {}
    }
    fn duals(&self) -> &TiSlice<Row, f64> {
        match *self {}
    }
    fn tolerance(&self) -> f64 {
        match *self {}
    }
    fn add_col(&mut self, _: NewCol) -> Result<Col, ActionError> {
        match *self {}
    }
    fn branch_early(&mut self) -> Result<(), ActionError> {
        match *self {}
    }
    fn report_bound(&mut self, _: f64) -> Result<(), ActionError> {
        match *self {}
    }
}

/// A pricer that is never asked.
struct Idle;

impl<B: Prices> Pricer<B> for Idle {
    fn price(&mut self, _: &mut B::Price<'_>) {}
}

/// A backend that, in a solve with a pricer, prices one column into the sink itself when `prices`; it ends optimal,
/// and answers duals with zeros.
#[derive(Debug)]
struct SelfPricing {
    /// Its one option: whether a solve with a pricer prices a column.
    prices: bool,
    num_cols: usize,
}

impl Backend for SelfPricing {
    type Error = ScriptError;
    type Status = ();
    type Log<'a> = Unsupported;
    type Progress<'a> = Unsupported;
    type Poll<'a> = Unsupported;
    type Solution<'a> = Unsupported;
    type Native<'a> = Unsupported;

    const EVENTS: Events = Events::NONE;
    const CLASSES: ModelClasses = ModelClasses::ALL;

    fn load(&mut self, program: &Program) -> Result<(), ScriptError> {
        self.num_cols = program.num_cols();
        Ok(())
    }

    fn solve(&mut self, _: &Settings, mut hooks: Hooks<'_, Self>) -> Result<Solved<()>, ScriptError> {
        let num_cols = match hooks.take_priced() {
            Some(priced) => {
                if self.prices {
                    priced.push(column(1.0, &[(Row::new(0), 1.0)])).unwrap();
                }
                priced.num_cols()
            }
            None => self.num_cols,
        };
        let values: TiVec<Col, f64> = (0..num_cols).map(|_| 0.0).collect();
        Ok(Solved {
            outcome: Outcome::Optimal(Solution { objective: 0.0, values }),
            status: (),
            elapsed: Duration::ZERO,
            nodes: None,
            iterations: None,
        })
    }
}

impl culmen_core::Configurable for SelfPricing {
    type Options = bool;

    fn options(&self) -> &bool {
        &self.prices
    }

    fn options_mut(&mut self) -> &mut bool {
        &mut self.prices
    }
}

impl Prices for SelfPricing {
    type Price<'a> = NoRound;
}

impl culmen_core::DualAnalysis for SelfPricing {
    fn duals(&self, program: &Program) -> Result<Duals, ScriptError> {
        Ok(Duals {
            rows: program.rows().map(|_| 0.0).collect(),
            cols: program.cols().map(|_| 0.0).collect(),
        })
    }
}

#[test]
fn test_analyses_after_a_solve_that_priced_find_no_solve_to_read() {
    let backend = SelfPricing {
        prices: false,
        num_cols: 0,
    };
    let mut solver = Solver::new(backend, small()).unwrap();
    solver.run().pricer(&mut Idle).solve().unwrap();
    assert!(solver.duals().is_ok(), "a solve that priced nothing is read as before");
    *solver.options_mut() = true;
    solver.run().pricer(&mut Idle).solve().unwrap();
    assert_eq!(solver.program().num_cols(), 3);
    assert_eq!(
        solver.duals(),
        Err(AnalysisError::NotSolved {
            analysis: Analysis::Duals
        })
    );
    *solver.options_mut() = false;
    solver.solve().unwrap();
    assert!(
        solver.duals().is_ok(),
        "the next solve that prices nothing is read again"
    );
}

#[test]
fn test_the_new_action_errors_name_their_fault() {
    use core::error::Error;
    let (col, row) = (Col::new(4), Row::new(2));
    let cases = [
        (
            ActionError::UnknownRow { row },
            "callback action: Row(2) is not a row of the program",
        ),
        (
            ActionError::NonFiniteEntry { row, col },
            "callback action: the coefficient of Col(4) in Row(2) is not finite",
        ),
        (
            ActionError::ExcludesZero {
                col,
                bounds: bounds!(1.0, 2.0),
            },
            "callback action: the priced column Col(4) has the bounds [1, 2], which do not hold zero",
        ),
        (
            ActionError::Integrality {
                col,
                integrality: Integrality::SemiInteger,
            },
            "callback action: the priced column Col(4) is SemiInteger, which pricing or the backend does not allow",
        ),
        (
            ActionError::NonFiniteBound { bound: f64::INFINITY },
            "callback action: the bound inf is not finite",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
        assert!(error.source().is_none());
    }
}

#[test]
fn test_the_script_price_context_reads_back_its_step() {
    /// Records what the context reads.
    #[derive(Default)]
    struct Reads(Vec<(Progress, PriceMode, Sense, Vec<f64>, f64)>);

    impl Pricer<Script> for Reads {
        fn price(&mut self, ctx: &mut ScriptPrice<'_>) {
            let duals = ctx.duals().iter().copied().collect();
            self.0
                .push((ctx.progress(), ctx.mode(), ctx.sense(), duals, ctx.tolerance()));
        }
    }

    let (program, _) = farkas_then_max();
    let steps = vec![price(PriceMode::ReducedCost { objective: 12.0 }, [3.0, 0.0])];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), program).unwrap();
    let mut reads = Reads::default();
    solver.run().pricer(&mut reads).solve().unwrap();
    assert_eq!(
        reads.0,
        [(
            progress(),
            PriceMode::ReducedCost { objective: 12.0 },
            Sense::Maximize,
            vec![3.0, 0.0],
            ScriptPrice::TOLERANCE
        )]
    );
}

#[test]
fn test_improves_reads_the_reduced_cost_by_the_bounds_the_sense_and_the_mode() {
    /// For one row with dual 2, a column of cost `cost` and entry 1 in it has the reduced cost `cost - 2`, or `-2` in
    /// Farkas mode; records whether each column improves.
    struct Judge(Vec<(f64, (f64, f64), bool)>, Vec<bool>);

    impl Pricer<Script> for Judge {
        fn price(&mut self, ctx: &mut ScriptPrice<'_>) {
            for &(cost, (lower, upper), _) in &self.0 {
                let column = culmen_core::NewCol {
                    column: ProgramColumn::continuous()
                        .cost(cost)
                        .bounds(Bounds::new(lower, upper).unwrap()),
                    entries: vec![(Row::new(0), 1.0)],
                };
                self.1.push(ctx.improves(&column));
            }
        }
    }

    // (cost, bounds, improves): a column rising from zero improves a minimization when `d < -tol`; one falling below
    // zero when `d > tol`; one free around zero either way; none at `d = 0`.
    let rising = (0.0, 4.0);
    let falling = (-4.0, 0.0);
    let both = (-4.0, 4.0);
    let minimize = [
        (1.0, rising, true),      // d = -1
        (3.0, rising, false),     // d = 1
        (1.0, falling, false),    // d = -1
        (3.0, falling, true),     // d = 1: lowering it below zero lowers the objective
        (3.0, both, true),        // d = 1, falling
        (1.0, both, true),        // d = -1, rising
        (2.0, both, false),       // d = 0
        (3.0, (0.0, 0.0), false), // fixed at zero: it cannot move
    ];
    // Maximizing, t = -d: the same columns improve where `-d` says so.
    let maximize = [
        (1.0, rising, false),
        (3.0, rising, true),
        (1.0, falling, true),
        (3.0, falling, false),
        (2.0, both, false),
    ];
    // Farkas, t = d = -2 whatever the cost: a rising column improves, a falling one does not.
    let farkas = [(5.0, rising, true), (5.0, falling, false), (5.0, both, true)];
    for (sense, mode, cases) in [
        (
            Sense::Minimize,
            PriceMode::ReducedCost { objective: 2.0 },
            &minimize[..],
        ),
        (
            Sense::Maximize,
            PriceMode::ReducedCost { objective: 2.0 },
            &maximize[..],
        ),
        (Sense::Minimize, PriceMode::Farkas, &farkas[..]),
    ] {
        let mut plan = ProgramPlan::new();
        plan.sense = sense;
        let x = plan.col(ProgramColumn::continuous().cost(1.0).bounds(bounds!(0.0, 1.0)));
        plan.row((bounds!(1.0, 1.0), [(x, 1.0)]));
        let steps = vec![price(mode, [2.0])];
        let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), plan.build().unwrap()).unwrap();
        let mut judge = Judge(cases.to_vec(), Vec::new());
        solver.run().pricer(&mut judge).solve().unwrap();
        let want: Vec<bool> = cases.iter().map(|&(.., improves)| improves).collect();
        assert_eq!(judge.1, want, "{sense:?} {mode:?}");
    }
}

#[test]
fn test_a_backend_driven_directly_prices_into_the_sink_it_is_given() {
    let program = small();
    let n = program.num_cols();
    let steps = vec![price(PriceMode::ReducedCost { objective: 1.0 }, [1.0, 0.0])];
    let mut script = Script::new(steps, Outcome::Infeasible);
    script.load(&program).unwrap();
    let mut priced = Priced::new(&program, Script::CLASSES);
    let mut pricer = AddOne::default();
    let hooks = Hooks::new().register_pricer(&mut pricer, &mut priced);
    script.solve(&Settings::new(), hooks).unwrap();
    assert_eq!(pricer.0, [Col::new(n)]);
    assert_eq!(priced.num_cols(), n + 1);
}
