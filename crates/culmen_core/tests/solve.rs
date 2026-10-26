// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generic callback layer, the observer convenience and the solver, driven by the scripted backend.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;
use std::sync::atomic::AtomicBool;

use proptest::prelude::*;

use culmen_core::{
    ActionError, AddedRow, Backend, Callback, Col, CutContext, Events, Found, HeuristicContext, Improvement,
    LazyContext, Linear, LogLevel, LogLine, Model, Observer, Outcome, Phase, PollContext, Progress, Request, Script,
    ScriptStatus, Settings, Solution, SolutionContext, Solver, Step, StopReason, TiVec, Var, bounds, stop_fn,
};

fn secs(s: u64) -> Duration {
    Duration::from_secs(s)
}

fn mip(elapsed: u64, objective: Option<f64>, bound: Option<f64>) -> Progress {
    Progress {
        objective,
        bound,
        nodes: Some(elapsed * 10),
        ..Progress::at(secs(elapsed), Phase::Mip)
    }
}

fn found(elapsed: u64, objective: f64, values: [f64; 2]) -> Found {
    Found {
        progress: mip(elapsed, Some(objective), None),
        objective,
        values: values.to_vec().into(),
    }
}

fn two_binaries() -> (Model, Col, Col) {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    m.maximize(x + y);
    (m, x, y)
}

/// Everything an observer is told, in order.
#[derive(Default)]
struct Seen(Vec<String>);

impl Observer for Seen {
    fn progress(&mut self, p: &Progress) {
        self.0.push(format!(
            "progress {} {:?} {:?}",
            p.elapsed.as_secs(),
            p.objective,
            p.gap()
        ));
    }
    fn improved(&mut self, i: &Improvement<'_>) {
        self.0.push(format!(
            "improved {} {:?}",
            i.objective,
            i.values.iter().collect::<Vec<_>>()
        ));
    }
    fn log(&mut self, line: &LogLine<'_>) {
        self.0.push(format!("log {:?} {}", line.level, line.text));
    }
}

#[test]
fn test_watch_joins_log_chunks_throttles_progress_and_reports_incumbents() {
    let (m, ..) = two_binaries();
    let log = |elapsed, text: &str| Step::Log {
        elapsed: secs(elapsed),
        level: LogLevel::Info,
        text: text.to_owned(),
    };
    let steps = vec![
        log(0, "Running "),
        log(0, "HiGHS-like\nsecond line\r\nthird"),
        Step::Poll(mip(0, None, Some(4.0))),
        Step::Poll(mip(0, None, Some(4.0))),
        Step::Incumbent(found(1, 1.0, [1.0, 0.0])),
        Step::Progress(mip(1, Some(1.0), Some(2.0))),
        Step::Poll(mip(2, Some(1.0), Some(2.0))),
        Step::Feasible(found(2, 0.0, [0.0, 0.0])),
    ];
    let end = Outcome::Optimal(Solution {
        objective: 2.0,
        values: vec![1.0, 1.0].into(),
    });
    let mut solver = Solver::new(Script::new(steps, end.clone()), m.build().unwrap()).unwrap();
    let mut seen = Seen::default();
    let solved = solver.solve(Request::new().observer(&mut seen)).unwrap();
    assert_eq!(solved.outcome, end);
    assert_eq!(solved.status, ScriptStatus::Finished);
    assert_eq!(solved.elapsed, secs(2));
    assert_eq!(
        seen.0,
        [
            "log Info Running HiGHS-like",
            "log Info second line",
            "progress 0 None None",
            "improved 1 [1.0, 0.0]",
            "progress 1 Some(1.0) Some(1.0)",
            "progress 2 Some(1.0) Some(1.0)",
            "log Info third",
        ]
    );
}

#[test]
fn test_stop_interrupts_at_the_next_poll_with_the_incumbent_and_bound() {
    let (m, x, _) = two_binaries();
    let steps = vec![
        Step::Incumbent(found(1, 1.0, [1.0, 0.0])),
        Step::Poll(mip(2, Some(1.0), Some(2.0))),
        Step::Incumbent(found(3, 2.0, [1.0, 1.0])),
    ];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), m.build().unwrap()).unwrap();
    let stop = AtomicBool::new(true);
    let solved = solver.solve(Request::new().stop(&stop)).unwrap();
    assert_eq!(solved.status, ScriptStatus::Interrupted { step: 1 });
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
    assert_eq!((best.values[x], best.objective, bound), (1.0, 1.0, 2.0));

    let solved = solver.solve(Request::new().stop(&stop_fn(|| false))).unwrap();
    assert_eq!(
        solved.outcome,
        Outcome::Infeasible,
        "a stop that never fires lets the script finish"
    );
}

/// A callback for one backend, using every context the scripted backend offers.
#[derive(Default)]
struct Everything {
    seen: Vec<String>,
}

impl Callback<Script> for Everything {
    fn feasible(&mut self, ctx: &mut <Script as Backend>::Solution<'_>) {
        self.seen.push(format!("feasible {}", ctx.objective()));
    }
    fn lazy(&mut self, ctx: &mut <Script as Backend>::Lazy<'_>) {
        let x = Col::new(0);
        assert_eq!(ctx.candidate()[x], 1.0);
        ctx.add_lazy(x.leq(0.0)).unwrap();
        assert_eq!(
            ctx.add_lazy(Col::new(5).leq(0.0)),
            Err(ActionError::UnknownColumn { col: Col::new(5) })
        );
        self.seen.push(format!("lazy {}", ctx.objective()));
    }
    fn cut(&mut self, ctx: &mut <Script as Backend>::Cut<'_>) {
        let (x, y) = (Col::new(0), Col::new(1));
        assert_eq!(ctx.relaxation()[y], 0.5);
        ctx.add_cut((x + y + 1).leq(2.0)).unwrap();
        self.seen.push(format!("cut at {:?}", ctx.progress().elapsed));
    }
    fn heuristic(&mut self, ctx: &mut <Script as Backend>::Heuristic<'_>) {
        let full: TiVec<Col, f64> = vec![1.0, 1.0].into();
        ctx.supply(&full).unwrap();
        ctx.supply_partial(&[(Col::new(1), 1.0)]).unwrap();
        let short: TiVec<Col, f64> = vec![1.0].into();
        assert_eq!(ctx.supply(&short), Err(ActionError::Length { expected: 2, found: 1 }));
        assert_eq!(
            ctx.supply_partial(&[(Col::new(0), f64::NAN)]),
            Err(ActionError::NotFinite { col: Col::new(0) })
        );
        self.seen.push("heuristic".to_owned());
    }
    fn native(&mut self, ctx: &mut <Script as Backend>::Native<'_>) {
        self.seen.push(format!("note {}", ctx.text()));
    }
}

#[test]
fn test_every_generic_event_reaches_a_backend_specific_callback() {
    let (m, x, y) = two_binaries();
    let steps = vec![
        Step::Feasible(found(0, 1.0, [1.0, 0.0])),
        Step::Lazy(found(1, 1.0, [1.0, 0.0])),
        Step::Cut {
            progress: mip(2, None, Some(2.0)),
            relaxation: vec![1.0, 0.5].into(),
        },
        Step::Heuristic(mip(3, None, Some(2.0))),
        Step::Note("presolve removed 0 rows".to_owned()),
    ];
    let mut script = Script::new(steps, Outcome::Unbounded);
    let mut everything = Everything::default();
    script.load(&m.build().unwrap()).unwrap();
    let solved = script.solve(&culmen_core::Settings::new(), &mut everything).unwrap();
    assert_eq!(solved.outcome, Outcome::Unbounded);
    assert_eq!(
        everything.seen,
        [
            "feasible 1",
            "lazy 1",
            "cut at 2s",
            "heuristic",
            "note presolve removed 0 rows"
        ]
    );
    assert_eq!(
        script.record.lazy,
        [AddedRow {
            terms: vec![(x, 1.0)],
            bounds: bounds!(f64::NEG_INFINITY, 0.0)
        }]
    );
    assert_eq!(
        script.record.cuts,
        [AddedRow {
            terms: vec![(x, 1.0), (y, 1.0)],
            bounds: bounds!(f64::NEG_INFINITY, 1.0)
        }]
    );
    assert_eq!(script.record.supplied, [vec![(x, 1.0), (y, 1.0)], vec![(y, 1.0)]]);
}

/// A generic callback that asks only for polls.
struct PollCounter(usize);

impl<B: Backend> Callback<B> for PollCounter {
    fn events(&self) -> Events {
        Events::POLL
    }
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        self.0 += 1;
        if ctx.progress().elapsed >= secs(2) {
            ctx.interrupt();
        }
    }
    fn incumbent(&mut self, _: &mut B::Solution<'_>) {
        unreachable!("an event the callback did not ask for is never delivered");
    }
}

#[test]
fn test_a_paired_callback_runs_beside_the_watch_and_gets_only_its_events() {
    let (m, ..) = two_binaries();
    let steps = vec![
        Step::Poll(mip(1, None, None)),
        Step::Incumbent(found(1, 1.0, [1.0, 0.0])),
        Step::Poll(mip(2, None, None)),
        Step::Poll(mip(3, None, None)),
    ];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), m.build().unwrap()).unwrap();
    let (mut seen, mut counter) = (Seen::default(), PollCounter(0));
    let solved = solver
        .solve_with(Request::new().observer(&mut seen), &mut counter)
        .unwrap();
    assert_eq!(counter.0, 2);
    assert_eq!(solved.status, ScriptStatus::Interrupted { step: 2 });
    assert_eq!(
        seen.0,
        ["progress 1 None None", "improved 1 [1.0, 0.0]", "progress 2 None None"]
    );
}

/// An observer that panics on the first improvement.
struct Panicking;

impl Observer for Panicking {
    fn improved(&mut self, _: &Improvement<'_>) {
        panic!("observer failed");
    }
}

#[test]
#[should_panic(expected = "observer failed")]
fn test_a_panicking_observer_panics_out_of_solve() {
    let (m, ..) = two_binaries();
    let steps = vec![Step::Incumbent(found(1, 1.0, [1.0, 0.0]))];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), m.build().unwrap()).unwrap();
    let _ = solver.solve(Request::new().observer(&mut Panicking));
}

#[test]
fn test_settings_reach_the_backend_and_events_print_by_name() {
    let (m, ..) = two_binaries();
    let mut solver = Solver::new(Script::optimal(2.0, vec![1.0, 1.0]), m.build().unwrap()).unwrap();
    let request = Request::new()
        .settings_from(Settings {
            console: true,
            threads: Some(culmen_core::threads!(3)),
            ..Settings::new()
        })
        .time_limit(secs(5));
    solver.solve(request).unwrap();
    let settings = &solver.backend().record.settings[0];
    assert_eq!(
        (settings.time_limit, settings.console, settings.threads.map(|t| t.get())),
        (Some(secs(5)), true, Some(3))
    );
    assert_eq!(format!("{:?}", Events::LOG | Events::CUT), "Events(LOG | CUT)");
    assert_eq!(Script::EVENTS, Events::ALL);
}

#[test]
fn test_solving_unloaded_script_is_refused() {
    let mut script = Script::optimal(0.0, Vec::new());
    assert!(script.solve(&culmen_core::Settings::new(), &mut ()).is_err());
}

/// An observer that counts progress reports.
#[derive(Default)]
struct CountProgress(usize);

impl Observer for CountProgress {
    fn progress(&mut self, _: &Progress) {
        self.0 += 1;
    }
}

#[test]
fn test_an_unbounded_progress_interval_reports_once() {
    let (m, ..) = two_binaries();
    let steps = vec![Step::Progress(mip(1, None, None)), Step::Poll(mip(2, None, None))];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), m.build().unwrap()).unwrap();
    let mut count = CountProgress::default();
    let request = Request::new().observer(&mut count).progress_interval(Duration::MAX);
    let solved = solver.solve(request).unwrap();
    assert_eq!(solved.outcome, Outcome::Infeasible);
    assert_eq!(count.0, 1);
}

#[test]
fn test_a_script_outcome_does_not_depend_on_the_events_asked_for() {
    let (m, ..) = two_binaries();
    let steps = vec![Step::Progress(mip(1, None, Some(5.0))), Step::Poll(mip(2, None, None))];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), m.build().unwrap()).unwrap();
    let stop = AtomicBool::new(true);
    let solved = solver.solve(Request::new().stop(&stop)).unwrap();
    assert_eq!(
        solved.outcome,
        Outcome::Stopped {
            reason: StopReason::Interrupted,
            best: None,
            bound: Some(5.0)
        }
    );
    assert_eq!(solved.elapsed, secs(2));
}

#[test]
fn test_events_combine_and_iterate_one_by_one() {
    let mut set = Events::LOG | Events::CUT;
    set |= Events::POLL;
    assert_eq!(set.len(), 3);
    assert_eq!(set & Events::POLL, Events::POLL);
    set &= Events::LOG | Events::POLL | Events::NATIVE;
    assert_eq!(set, Events::LOG | Events::POLL);
    assert_eq!(set.iter().collect::<Vec<_>>(), [Events::LOG, Events::POLL]);
    assert_eq!(set.iter().rev().collect::<Vec<_>>(), [Events::POLL, Events::LOG]);
    set.extend([Events::NATIVE, Events::FEASIBLE]);
    assert_eq!(format!("{set:?}"), "Events(LOG | POLL | FEASIBLE | NATIVE)");
    assert_eq!(Events::ALL.iter().len(), 9);
    assert_eq!(Events::ALL.into_iter().collect::<Events>(), Events::ALL);
    assert_eq!(format!("{:?}", Events::NONE), "Events()");
    let mut iter = Events::LOG.iter();
    assert_eq!(
        (iter.next(), iter.next(), iter.next_back()),
        (Some(Events::LOG), None, None)
    );
}

const SINGLE: [Events; 9] = [
    Events::LOG,
    Events::PROGRESS,
    Events::POLL,
    Events::FEASIBLE,
    Events::INCUMBENT,
    Events::LAZY,
    Events::CUT,
    Events::HEURISTIC,
    Events::NATIVE,
];

proptest! {
    /// A set built from chosen single events iterates exactly those, in the order of the constants, both ways.
    #[test]
    fn test_events_iterate_the_chosen_singles(chosen in prop::collection::vec(any::<bool>(), 9)) {
        let singles: Vec<Events> = SINGLE.iter().zip(&chosen).filter(|&(_, &c)| c).map(|(&e, _)| e).collect();
        let set: Events = singles.iter().copied().collect();
        prop_assert_eq!(set.iter().collect::<Vec<_>>(), singles.clone());
        prop_assert_eq!(set.iter().len(), singles.len());
        prop_assert_eq!(set.len(), singles.len());
        let mut reversed = singles.clone();
        reversed.reverse();
        prop_assert_eq!(set.iter().rev().collect::<Vec<_>>(), reversed);
        for (event, &c) in SINGLE.iter().zip(&chosen) {
            prop_assert_eq!(set.contains(*event), c);
        }
    }
}

#[test]
fn test_script_steps_name_their_event_kind() {
    let p = mip(0, None, None);
    let kinds: Events = [
        Step::Log {
            elapsed: secs(0),
            level: LogLevel::Info,
            text: String::new(),
        },
        Step::Progress(p),
        Step::Poll(p),
        Step::Feasible(found(0, 0.0, [0.0, 0.0])),
        Step::Incumbent(found(0, 0.0, [0.0, 0.0])),
        Step::Lazy(found(0, 0.0, [0.0, 0.0])),
        Step::Cut {
            progress: p,
            relaxation: TiVec::new(),
        },
        Step::Heuristic(p),
        Step::Note(String::new()),
    ]
    .iter()
    .map(Step::event)
    .collect();
    assert_eq!(kinds, Events::ALL);
}
