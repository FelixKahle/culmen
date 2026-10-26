// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every HiGHS callback kind HiGHS can be made to fire, through the generic layer and the HiGHS contexts.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use culmen_core::{
    Backend, Callback, Col, Events, HeuristicContext, Improvement, LinExpr, Linear, LogLine, Model, Observer, Outcome,
    Phase, PollContext, Program, Progress, Request, Settings, Solution, SolutionContext, Solver, StopReason, TiVec,
    Var, relative_gap,
};
use culmen_highs::{Highs, HighsNative, HighsOptions, LpSolver, PollSource, QueryOrigin, Switch};
use culmen_testing::instances::assignment;
use culmen_testing::reference::close;

/// Everything an observer is told.
#[derive(Default)]
struct Seen {
    progress: Vec<Progress>,
    improved: Vec<(f64, TiVec<Col, f64>)>,
    lines: Vec<String>,
}

impl Observer for Seen {
    fn progress(&mut self, progress: &Progress) {
        self.progress.push(*progress);
    }
    fn improved(&mut self, improvement: &Improvement<'_>) {
        self.improved
            .push((improvement.objective, improvement.values.to_owned()));
    }
    fn log(&mut self, line: &LogLine<'_>) {
        self.lines.push(line.text.to_owned());
    }
}

/// The hard instance: a 60 × 30 generalized assignment HiGHS does not finish in seconds.
fn hard() -> Program {
    assignment(60, 30, 12_345, 0.0).program
}

fn knapsack() -> (Program, Vec<Col>) {
    let (value, weight) = ([10.0, 13.0, 7.0, 8.0, 9.0, 4.0], [5.0, 6.0, 4.0, 3.0, 5.0, 2.0]);
    let mut m = Model::new();
    let take = m.add_vars(6, |_| Var::binary());
    m.add_constraint(LinExpr::sum((0..6).map(|i| weight[i] * take[i])).leq(12.0));
    m.maximize(LinExpr::sum((0..6).map(|i| value[i] * take[i])));
    (m.build().unwrap(), take)
}

#[test]
fn test_log_lines_reach_the_observer_and_stdout_stays_quiet() {
    if std::env::var_os("CULMEN_LOG_CHILD").is_some() {
        let mut seen = Seen::default();
        let mut solver = Solver::new(Highs::new(), knapsack().0).unwrap();
        solver.solve(Request::new().observer(&mut seen)).unwrap();
        assert!(
            seen.lines.iter().any(|l| l.starts_with("Solving report")),
            "{:?}",
            seen.lines
        );
        assert!(seen.lines.iter().all(|l| !l.contains('\n')));
        println!("CHILD-DONE {}", seen.lines.len());
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let output = Command::new(exe)
        .args([
            "--exact",
            "test_log_lines_reach_the_observer_and_stdout_stays_quiet",
            "--nocapture",
        ])
        .env("CULMEN_LOG_CHILD", "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("CHILD-DONE"), "{stdout}");
    assert!(!stdout.contains("HiGHS"), "HiGHS wrote to standard output:\n{stdout}");
}

#[test]
fn test_improving_solutions_arrive_feasible_and_improving() {
    let program = assignment(30, 12, 7, 0.0).program;
    let mut seen = Seen::default();
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let solved = solver
        .solve(Request::new().time_limit(Duration::from_secs(20)).observer(&mut seen))
        .unwrap();
    assert!(!seen.improved.is_empty());
    for (objective, values) in &seen.improved {
        assert!(program.max_violation(values) <= 1e-6);
        assert!(close(program.objective_value(values), *objective, 1e-9));
    }
    assert!(
        seen.improved.windows(2).all(|w| w[1].0 < w[0].0),
        "each improvement is better"
    );
    let best = solved.outcome.solution().unwrap().objective;
    assert!(close(seen.improved.last().unwrap().0, best, 1e-9));
}

#[test]
fn test_progress_arrives_about_once_per_interval() {
    let mut seen = Seen::default();
    let mut solver = Solver::new(Highs::new(), hard()).unwrap();
    let request = Request::new()
        .time_limit(Duration::from_secs(3))
        .progress_interval(Duration::from_millis(500))
        .observer(&mut seen);
    let solved = solver.solve(request).unwrap();
    assert!(matches!(
        solved.outcome,
        Outcome::Stopped {
            reason: StopReason::TimeLimit,
            best: Some(_),
            ..
        }
    ));
    let count = seen.progress.len();
    assert!((4..=8).contains(&count), "{count} progress reports in 3 s at 500 ms");
    for pair in seen.progress.windows(2) {
        assert!(pair[1].elapsed >= pair[0].elapsed + Duration::from_millis(500));
    }
    let last = seen.progress.last().unwrap();
    assert_eq!(last.phase, Phase::Mip);
    assert!(
        last.bound.unwrap() <= last.objective.unwrap(),
        "a minimization's bound lies below its incumbent"
    );
    assert!(last.gap().unwrap() > 0.0);
}

#[test]
fn test_stop_request_ends_a_hard_mip_early_with_its_incumbent() {
    let program = hard();
    let stop = Arc::new(AtomicBool::new(false));
    let trigger = Arc::clone(&stop);
    let timer = thread::spawn(move || {
        thread::sleep(Duration::from_secs(1));
        trigger.store(true, Ordering::Release);
    });
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let solved = solver
        .solve(Request::new().time_limit(Duration::from_secs(30)).stop(&stop))
        .unwrap();
    timer.join().unwrap();
    assert!(
        solved.elapsed < Duration::from_secs(3),
        "stopped after {:?}",
        solved.elapsed
    );
    let Outcome::Stopped {
        reason: StopReason::Interrupted,
        best: Some(best),
        bound: Some(bound),
    } = solved.outcome
    else {
        panic!(
            "expected an interrupted solve with an incumbent, got {:?}",
            solved.outcome
        );
    };
    assert!(program.max_violation(&best.values) <= 1e-6);
    assert!(bound <= best.objective);
}

#[test]
fn test_time_limit_stops_with_the_incumbent() {
    let mut solver = Solver::new(Highs::new(), hard()).unwrap();
    let solved = solver
        .solve(Request::new().time_limit(Duration::from_millis(800)))
        .unwrap();
    assert!(solved.elapsed < Duration::from_secs(3));
    assert!(matches!(
        solved.outcome,
        Outcome::Stopped {
            reason: StopReason::TimeLimit,
            best: Some(_),
            bound: Some(_)
        }
    ));
    // The clock restarts with every solve: a second solve gets its own limit.
    let solved = solver
        .solve(Request::new().time_limit(Duration::from_millis(800)))
        .unwrap();
    assert!(
        solved.elapsed >= Duration::from_millis(700),
        "the second solve ran {:?}",
        solved.elapsed
    );
}

#[test]
fn test_wide_gap_ends_as_gap_reached() {
    let mut solver = Solver::new(Highs::new(), hard()).unwrap();
    let request = Request::new()
        .settings_from(Settings {
            relative_gap: Some(relative_gap!(0.5)),
            ..Settings::new()
        })
        .time_limit(Duration::from_secs(30));
    let solved = solver.solve(request).unwrap();
    let Outcome::GapReached { solution, bound } = solved.outcome else {
        panic!("expected the gap to be reached, got {:?}", solved.outcome);
    };
    assert!(bound <= solution.objective);
    assert!((solution.objective - bound) / solution.objective <= 0.5);
}

/// A panicking observer.
struct Panicking;

impl Observer for Panicking {
    fn improved(&mut self, _: &Improvement<'_>) {
        panic!("observer failed");
    }
}

#[test]
fn test_a_panicking_observer_panics_out_of_solve_and_leaves_highs_usable() {
    let (program, take) = knapsack();
    let mut solver = Solver::new(Highs::new(), program).unwrap();
    let caught = catch_unwind(AssertUnwindSafe(|| {
        solver.solve(Request::new().observer(&mut Panicking))
    }));
    let payload = caught.expect_err("the panic propagates");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"observer failed"));
    let solved = solver.solve(Request::new()).unwrap();
    let Outcome::Optimal(best) = solved.outcome else {
        panic!("{:?}", solved.outcome)
    };
    // Items 1, 3 and 5 weigh 11 and are worth 25; nothing heavier fits.
    assert_eq!(best.objective, 25.0);
    assert_eq!(best.values[take[1]].round(), 1.0);
}

/// Records every HiGHS callback kind it receives, through the HiGHS contexts.
#[derive(Default)]
struct Kinds {
    polls: Vec<PollSource>,
    interrupt_at_first_poll: bool,
    mip_progress: usize,
    feasible: Vec<f64>,
    incumbents: Vec<f64>,
    queries: Vec<QueryOrigin>,
    supply: Option<Vec<f64>>,
    supply_partial: Option<Vec<(Col, f64)>>,
    supplied: Vec<Result<(), culmen_core::ActionError>>,
    /// Each cut pool's column count and entry count.
    cut_pools: Vec<(usize, usize)>,
}

impl Callback<Highs> for Kinds {
    fn poll(&mut self, ctx: &mut <Highs as Backend>::Poll<'_>) {
        self.polls.push(*ctx.source());
        if self.interrupt_at_first_poll {
            ctx.interrupt();
        }
    }
    fn progress(&mut self, ctx: &mut <Highs as Backend>::Progress<'_>) {
        assert!(ctx.mip().nodes < u64::MAX);
        self.mip_progress += 1;
    }
    fn feasible(&mut self, ctx: &mut <Highs as Backend>::Solution<'_>) {
        self.feasible.push(ctx.objective());
    }
    fn incumbent(&mut self, ctx: &mut <Highs as Backend>::Solution<'_>) {
        self.incumbents.push(ctx.objective());
    }
    fn heuristic(&mut self, ctx: &mut <Highs as Backend>::Heuristic<'_>) {
        self.queries.push(ctx.origin());
        if ctx.origin() != QueryOrigin::AfterSetup {
            return;
        }
        if let Some(values) = self.supply.take() {
            self.supplied.push(ctx.supply(&TiVec::from(values)));
        }
        if let Some(entries) = self.supply_partial.take() {
            self.supplied.push(ctx.supply_partial(&entries));
        }
    }
    fn native(&mut self, ctx: &mut <Highs as Backend>::Native<'_>) {
        match ctx {
            HighsNative::CutPool(pool) => {
                let entries = pool.cuts.iter().flat_map(|cut| &cut.entries);
                assert!(entries.clone().all(|(col, _)| col.index() < pool.num_cols));
                assert!(pool.cuts.iter().all(|cut| cut.lower <= cut.upper));
                self.cut_pools.push((pool.num_cols, entries.count()));
            }
            _ => unreachable!("HiGHS 1.15 reports no other native event"),
        }
    }
}

fn lp_problem() -> Program {
    let mut m = Model::new();
    let xs = m.add_vars(30, |_| Var::continuous().upper(10.0));
    for i in 0..30 {
        m.add_constraint(LinExpr::sum((0..30).map(|j| ((i * 7 + j * 3) % 11) as f64 * xs[j])).leq(50.0 + i as f64));
    }
    m.maximize(LinExpr::sum((0..30).map(|j| (1 + j % 5) as f64 * xs[j])));
    m.build().unwrap()
}

#[test]
fn test_simplex_poll_interrupts_an_lp() {
    let options = HighsOptions {
        presolve: Some(Switch::Off),
        solver: Some(LpSolver::Simplex),
        ..HighsOptions::default()
    };
    let mut solver = Solver::new(Highs::with_options(options), lp_problem()).unwrap();
    let mut kinds = Kinds {
        interrupt_at_first_poll: true,
        ..Kinds::default()
    };
    let solved = solver.solve_with(Request::new(), &mut kinds).unwrap();
    assert!(
        matches!(kinds.polls[0], PollSource::Simplex { .. }),
        "{:?}",
        kinds.polls
    );
    assert!(matches!(
        solved.outcome,
        Outcome::Stopped {
            reason: StopReason::Interrupted,
            bound: None,
            ..
        }
    ));
}

#[test]
fn test_ipm_poll_interrupts_an_lp() {
    let options = HighsOptions {
        presolve: Some(Switch::Off),
        solver: Some(LpSolver::Ipx),
        run_crossover: Some(Switch::Off),
        ..HighsOptions::default()
    };
    let mut solver = Solver::new(Highs::with_options(options), lp_problem()).unwrap();
    let mut kinds = Kinds {
        interrupt_at_first_poll: true,
        ..Kinds::default()
    };
    let solved = solver.solve_with(Request::new(), &mut kinds).unwrap();
    assert!(
        kinds.polls.iter().any(|p| matches!(p, PollSource::Ipm { .. })),
        "{:?}",
        kinds.polls
    );
    assert!(
        matches!(
            solved.outcome,
            Outcome::Stopped {
                reason: StopReason::Interrupted,
                ..
            }
        ),
        "{:?}",
        solved
    );
}

#[test]
fn test_mip_kinds_fire_progress_solutions_queries_and_cut_pool() {
    let program = assignment(30, 12, 7, 0.0).program;
    let mut solver = Solver::new(Highs::new(), program).unwrap();
    let mut kinds = Kinds::default();
    let solved = solver
        .solve_with(Request::new().time_limit(Duration::from_secs(20)), &mut kinds)
        .unwrap();
    assert!(kinds.polls.iter().any(|p| matches!(p, PollSource::Mip(_))));
    assert!(kinds.mip_progress > 0, "MIP log lines fire progress events");
    assert!(!kinds.incumbents.is_empty());
    assert!(
        kinds.feasible.len() >= kinds.incumbents.len(),
        "every incumbent is also a feasible solution"
    );
    assert!(kinds.queries.contains(&QueryOrigin::AfterSetup), "{:?}", kinds.queries);
    assert!(
        !kinds.cut_pools.is_empty(),
        "the cut pool is read after the root, again after each restart"
    );
    assert!(kinds.cut_pools.iter().all(|&(num_cols, _)| num_cols > 0));
    assert!(
        kinds.cut_pools.iter().any(|&(_, entries)| entries > 0),
        "{:?}",
        kinds.cut_pools
    );
    let best = solved.outcome.solution().unwrap().objective;
    assert!(close(*kinds.incumbents.last().unwrap(), best, 1e-9));
}

#[test]
fn test_a_supplied_solution_becomes_the_first_incumbent() {
    let (program, take) = knapsack();
    // HiGHS finds the optimum once; a second solve is handed it at the first query.
    let optimum = Solver::new(Highs::new(), program.clone())
        .unwrap()
        .solve(Request::new())
        .unwrap()
        .outcome;
    let Outcome::Optimal(Solution { objective, values }) = optimum else {
        panic!("{optimum:?}")
    };
    let options = HighsOptions {
        presolve: Some(Switch::Off),
        ..HighsOptions::default()
    };
    let mut solver = Solver::new(Highs::with_options(options), program).unwrap();
    let mut kinds = Kinds {
        supply: Some(values.raw.clone()),
        ..Kinds::default()
    };
    let solved = solver.solve_with(Request::new(), &mut kinds).unwrap();
    assert_eq!(kinds.supplied, [Ok(())]);
    assert_eq!(
        kinds.incumbents.first().copied(),
        Some(objective),
        "{:?}",
        kinds.incumbents
    );
    assert_eq!(solved.outcome.solution().unwrap().objective, objective);
    assert_eq!(values.len(), take.len());
}

#[test]
fn test_a_partial_solution_is_repaired_into_an_incumbent() {
    let (program, take) = knapsack();
    let options = HighsOptions {
        presolve: Some(Switch::Off),
        ..HighsOptions::default()
    };
    let mut solver = Solver::new(Highs::with_options(options), program).unwrap();
    let partial = vec![(take[0], 1.0), (take[1], 1.0)];
    let mut kinds = Kinds {
        supply_partial: Some(partial),
        ..Kinds::default()
    };
    solver.solve_with(Request::new(), &mut kinds).unwrap();
    assert_eq!(kinds.supplied, [Ok(())]);
    let first = *kinds.incumbents.first().unwrap();
    assert!(
        first >= 23.0,
        "the completion takes items 0 and 1 (23) and maybe more: {first}"
    );
}

#[test]
fn test_a_wrong_solution_is_refused_before_highs_sees_it() {
    let (program, _) = knapsack();
    let options = HighsOptions {
        presolve: Some(Switch::Off),
        ..HighsOptions::default()
    };
    let mut solver = Solver::new(Highs::with_options(options), program).unwrap();
    let mut kinds = Kinds {
        supply: Some(vec![1.0; 3]),
        ..Kinds::default()
    };
    solver.solve_with(Request::new(), &mut kinds).unwrap();
    assert_eq!(
        kinds.supplied,
        [Err(culmen_core::ActionError::Length { expected: 6, found: 3 })]
    );
}

/// Asks for no event, so HiGHS starts only what culmen always needs.
struct Nothing;

impl Callback<Highs> for Nothing {
    fn events(&self) -> Events {
        Events::NONE
    }
    fn poll(&mut self, _: &mut <Highs as Backend>::Poll<'_>) {
        unreachable!("not asked for");
    }
}

#[test]
fn test_events_not_asked_for_are_not_delivered() {
    let mut solver = Solver::new(Highs::new(), knapsack().0).unwrap();
    let solved = solver.solve_with(Request::new(), &mut Nothing).unwrap();
    assert!(matches!(solved.outcome, Outcome::Optimal(_)));
}

#[test]
fn test_poll_context_reports_mip_progress_in_generic_terms() {
    struct FirstMipPoll(Option<Progress>);
    impl Callback<Highs> for FirstMipPoll {
        fn events(&self) -> Events {
            Events::POLL
        }
        fn poll(&mut self, ctx: &mut <Highs as Backend>::Poll<'_>) {
            if let PollSource::Mip(mip) = ctx.source()
                && mip.primal_bound.is_some()
                && self.0.is_none()
            {
                self.0 = Some(ctx.progress());
                assert_eq!(ctx.progress().objective, mip.primal_bound);
                ctx.interrupt();
            }
        }
    }
    let mut solver = Solver::new(Highs::new(), hard()).unwrap();
    let mut first = FirstMipPoll(None);
    let solved = solver
        .solve_with(Request::new().time_limit(Duration::from_secs(30)), &mut first)
        .unwrap();
    let progress = first.0.unwrap();
    assert_eq!(progress.phase, Phase::Mip);
    assert!(progress.nodes.is_some());
    assert!(matches!(
        solved.outcome,
        Outcome::Stopped {
            reason: StopReason::Interrupted,
            best: Some(_),
            ..
        }
    ));
}

#[test]
fn test_offset_reaches_every_reported_objective_and_bound() {
    let shifted = assignment(30, 12, 7, 1000.0).program;
    let plain = assignment(30, 12, 7, 0.0).program;
    let solve = |program: Program, seen: &mut Seen| {
        Solver::new(Highs::new(), program)
            .unwrap()
            .solve(
                Request::new()
                    .settings_from(Settings {
                        relative_gap: Some(relative_gap!(0.0)),
                        ..Settings::new()
                    })
                    .observer(seen)
                    .progress_interval(Duration::ZERO),
            )
            .unwrap()
    };
    let (mut a, mut b) = (Seen::default(), Seen::default());
    let (shifted, plain) = (solve(shifted, &mut a), solve(plain, &mut b));
    let objective = |o: &Outcome| o.solution().unwrap().objective;
    assert!(close(
        objective(&shifted.outcome),
        objective(&plain.outcome) + 1000.0,
        1e-9
    ));
    assert!(a.improved.iter().all(|(o, _)| *o > 1000.0));
    let bounds: Vec<f64> = a.progress.iter().filter_map(|p| p.bound).collect();
    assert!(!bounds.is_empty() && bounds.iter().all(|&b| b > 1000.0), "{bounds:?}");
}

#[test]
fn test_maximization_bounds_lie_above_the_incumbent() {
    let mut m = Model::new();
    let x = m.add_vars(40, |_| Var::binary());
    for i in 0..8 {
        m.add_constraint(LinExpr::sum((0..40).map(|j| (1 + (i * 13 + j * 7) % 17) as f64 * x[j])).leq(60.0));
    }
    m.maximize(LinExpr::sum((0..40).map(|j| (5 + (j * 11) % 23) as f64 * x[j])) + 7);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    let mut seen = Seen::default();
    solver
        .solve(Request::new().observer(&mut seen).progress_interval(Duration::ZERO))
        .unwrap();
    for p in &seen.progress {
        if let (Some(objective), Some(bound)) = (p.objective, p.bound) {
            assert!(bound >= objective - 1e-9, "{p:?}");
        }
    }
}

/// Solves a knapsack inside its first MIP poll, on another instance, and keeps what that inner solve observed.
#[derive(Default)]
struct Nested {
    inner: Option<Seen>,
}

impl Callback<Highs> for Nested {
    fn events(&self) -> Events {
        Events::POLL
    }
    fn poll(&mut self, ctx: &mut <Highs as Backend>::Poll<'_>) {
        if self.inner.is_none() && matches!(ctx.source(), PollSource::Mip(_)) {
            let mut seen = Seen::default();
            let mut solver = Solver::new(Highs::new(), knapsack().0).unwrap();
            solver.solve(Request::new().observer(&mut seen)).unwrap();
            self.inner = Some(seen);
            ctx.interrupt();
        }
    }
}

#[test]
fn test_a_solve_nested_in_a_callback_gets_its_own_callbacks() {
    let mut solver = Solver::new(Highs::new(), assignment(30, 12, 7, 0.0).program).unwrap();
    let mut nested = Nested::default();
    let solved = solver.solve_with(Request::new(), &mut nested).unwrap();
    let inner = nested.inner.unwrap();
    assert!(
        !inner.improved.is_empty() && !inner.lines.is_empty(),
        "the inner solve was observed"
    );
    assert!(matches!(
        solved.outcome,
        Outcome::Stopped {
            reason: StopReason::Interrupted,
            ..
        }
    ));
}
