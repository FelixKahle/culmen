// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generic callback layer, the observer convenience and the solver, driven by the scripted backend.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use proptest::prelude::*;

use culmen_core::{
    ActionError, AddedRow, Backend, Callback, Col, Context, CutContext, CutHandler, Events, Found, HeuristicContext,
    HeuristicHandler, Heuristics, Hooks, Improvement, InvalidBounds, LazyConstraints, LazyContext, LazyHandler, Linear,
    LogLevel, LogLine, Model, Observer, Outcome, Phase, PollContext, PriceMode, ProgramPlan, Progress, Script,
    ScriptStatus, Settings, Solution, SolutionContext, Solver, Step, StopReason, TiVec, UserCuts, Var, bounds, stop_fn,
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
    let solved = solver.run().observer(&mut seen).solve().unwrap();
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
fn test_watch_drops_the_carriage_return_of_a_line_break_split_across_chunks() {
    let (m, ..) = two_binaries();
    let log = |text: &str| Step::Log {
        elapsed: secs(0),
        level: LogLevel::Info,
        text: text.to_owned(),
    };
    let steps = vec![log("first\r"), log("\nsecond\r"), log("\r\nthird\r")];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), m.build().unwrap()).unwrap();
    let mut seen = Seen::default();
    solver.run().observer(&mut seen).solve().unwrap();
    assert_eq!(seen.0, ["log Info first", "log Info second", "log Info third"]);
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
    let solved = solver.run().stop(&stop).solve().unwrap();
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

    let solved = solver.run().stop(&stop_fn(|| false)).solve().unwrap();
    assert_eq!(
        solved.outcome,
        Outcome::Infeasible,
        "a stop that never fires lets the script finish"
    );
}

/// A callback and handlers for one backend, using every context the scripted backend offers; each role is a copy that
/// writes to one shared list.
#[derive(Clone, Copy)]
struct Everything<'a>(&'a Mutex<Vec<String>>);

impl Everything<'_> {
    fn push(&self, seen: String) {
        self.0.lock().unwrap().push(seen);
    }
}

impl Callback<Script> for Everything<'_> {
    fn feasible(&mut self, ctx: &mut <Script as Backend>::Solution<'_>) {
        self.push(format!("feasible {}", ctx.objective()));
    }
    fn native(&mut self, ctx: &mut <Script as Backend>::Native<'_>) {
        self.push(format!("note {}", ctx.text()));
    }
}

impl LazyHandler<Script> for Everything<'_> {
    fn lazy(&mut self, ctx: &mut <Script as LazyConstraints>::Lazy<'_>) {
        let x = Col::new(0);
        assert_eq!(ctx.candidate()[x], 1.0);
        ctx.add_lazy(x.leq(0.0)).unwrap();
        assert_eq!(
            ctx.add_lazy(Col::new(5).leq(0.0)),
            Err(ActionError::UnknownColumn { col: Col::new(5) })
        );
        self.push(format!("lazy {}", ctx.objective()));
    }
}

impl CutHandler<Script> for Everything<'_> {
    fn cut(&mut self, ctx: &mut <Script as UserCuts>::Cut<'_>) {
        let (x, y) = (Col::new(0), Col::new(1));
        assert_eq!(ctx.relaxation()[y], 0.5);
        ctx.add_cut((x + y + 1).leq(2.0)).unwrap();
        self.push(format!("cut at {:?}", ctx.progress().elapsed));
    }
}

impl HeuristicHandler<Script> for Everything<'_> {
    fn heuristic(&mut self, ctx: &mut <Script as Heuristics>::Heuristic<'_>) {
        let full: TiVec<Col, f64> = vec![1.0, 1.0].into();
        ctx.supply(&full).unwrap();
        ctx.supply_partial(&[(Col::new(1), 1.0)]).unwrap();
        let short: TiVec<Col, f64> = vec![1.0].into();
        assert_eq!(ctx.supply(&short), Err(ActionError::Length { expected: 2, found: 1 }));
        assert_eq!(
            ctx.supply_partial(&[(Col::new(0), f64::NAN)]),
            Err(ActionError::NotFinite { col: Col::new(0) })
        );
        self.push("heuristic".to_owned());
    }
}

/// The steps every role of [`Everything`] answers, one each.
fn every_step() -> Vec<Step> {
    vec![
        Step::Feasible(found(0, 1.0, [1.0, 0.0])),
        Step::Lazy(found(1, 1.0, [1.0, 0.0])),
        Step::Cut {
            progress: mip(2, None, Some(2.0)),
            relaxation: vec![1.0, 0.5].into(),
        },
        Step::Heuristic(mip(3, None, Some(2.0))),
        Step::Note("presolve removed 0 rows".to_owned()),
    ]
}

#[test]
fn test_every_generic_event_reaches_a_backend_specific_callback_and_handlers() {
    let (m, x, y) = two_binaries();
    let mut script = Script::new(every_step(), Outcome::Unbounded);
    let seen = Mutex::new(Vec::new());
    let (mut callback, mut lazy, mut cut, mut heuristic) = (
        Everything(&seen),
        Everything(&seen),
        Everything(&seen),
        Everything(&seen),
    );
    script.load(&m.build().unwrap()).unwrap();
    let hooks = Hooks::new()
        .register_callback(&mut callback)
        .register_lazy(&mut lazy)
        .register_cut(&mut cut)
        .register_heuristic(&mut heuristic);
    let solved = script.solve(&Settings::new(), hooks).unwrap();
    assert_eq!(solved.outcome, Outcome::Unbounded);
    assert_eq!(
        seen.into_inner().unwrap(),
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

#[test]
fn test_a_search_step_reaches_only_a_registered_handler_whatever_the_callback_asks_for() {
    let (m, ..) = two_binaries();
    let mut solver = Solver::new(Script::new(every_step(), Outcome::Unbounded), m.build().unwrap()).unwrap();
    let seen = Mutex::new(Vec::new());
    let mut callback = Everything(&seen);
    solver.run().callback(&mut callback).solve().unwrap();
    assert_eq!(
        seen.into_inner().unwrap(),
        ["feasible 1", "note presolve removed 0 rows"]
    );
    let record = &solver.backend().record;
    assert!(record.lazy.is_empty() && record.cuts.is_empty() && record.supplied.is_empty());
}

/// A separator that cuts off integer and fractional points alike, counting its calls in each role.
#[derive(Default)]
struct Separator {
    lazy: usize,
    cut: usize,
}

impl<B: LazyConstraints + UserCuts> LazyHandler<B> for Separator {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        self.lazy += 1;
        ctx.add_lazy(Col::new(0).leq(0)).unwrap();
    }
}

impl<B: LazyConstraints + UserCuts> CutHandler<B> for Separator {
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        self.cut += 1;
        ctx.add_cut(Col::new(1).leq(0)).unwrap();
    }
}

#[test]
fn test_one_handler_registered_in_both_roles_gets_candidates_and_relaxations() {
    let (m, x, y) = two_binaries();
    let mut solver = Solver::new(Script::new(every_step(), Outcome::Unbounded), m.build().unwrap()).unwrap();
    let mut separator = Separator::default();
    solver.run().lazy_and_cut(&mut separator).solve().unwrap();
    assert_eq!((separator.lazy, separator.cut), (1, 1));
    let record = &solver.backend().record;
    assert_eq!(record.lazy[0].terms, [(x, 1.0)]);
    assert_eq!(record.cuts[0].terms, [(y, 1.0)]);

    // A cut handler registered after the pair takes the cuts over; the pair keeps the candidates.
    let mut other = Separator::default();
    solver
        .run()
        .lazy_and_cut(&mut separator)
        .cut(&mut other)
        .solve()
        .unwrap();
    assert_eq!((separator.lazy, separator.cut), (2, 1));
    assert_eq!((other.lazy, other.cut), (0, 1));
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
    let solved = solver.run().observer(&mut seen).callback(&mut counter).solve().unwrap();
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
    let _ = solver.run().observer(&mut Panicking).solve();
}

#[test]
fn test_settings_reach_the_backend_and_events_print_by_name() {
    let (m, ..) = two_binaries();
    let mut solver = Solver::new(Script::optimal(2.0, vec![1.0, 1.0]), m.build().unwrap()).unwrap();
    let settings = Settings {
        console: true,
        threads: Some(culmen_core::threads!(3)),
        ..Settings::new()
    };
    solver
        .run()
        .settings(settings.clone())
        .time_limit(secs(5))
        .solve()
        .unwrap();
    let got = &solver.backend().record.settings[0];
    assert_eq!(
        (got.time_limit, got.console, got.threads.map(|t| t.get())),
        (Some(secs(5)), true, Some(3))
    );
    // Settings given after a time limit replace it with their own, here none.
    solver.run().time_limit(secs(5)).settings(settings).solve().unwrap();
    let got = &solver.backend().record.settings[1];
    assert_eq!((got.time_limit, got.console), (None, true));
    assert_eq!(format!("{:?}", Events::LOG | Events::NATIVE), "Events(LOG | NATIVE)");
    assert_eq!(Script::EVENTS, Events::ALL);
}

#[test]
fn test_an_action_error_on_invalid_bounds_names_them_as_its_source() {
    use core::error::Error;
    let error = ActionError::Bounds(InvalidBounds::NotANumber);
    assert_eq!(
        error.source().map(ToString::to_string),
        Some(InvalidBounds::NotANumber.to_string())
    );
    assert!(ActionError::Refused("no".to_owned()).source().is_none());
    assert!(ActionError::NotFinite { col: Col::new(0) }.source().is_none());
}

#[test]
fn test_solving_unloaded_script_is_refused() {
    let mut script = Script::optimal(0.0, Vec::new());
    assert!(script.solve(&Settings::new(), Hooks::new()).is_err());
    assert!(script.load(&ProgramPlan::new().build().unwrap()).is_ok());
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
    let solved = solver
        .run()
        .observer(&mut count)
        .progress_interval(Duration::MAX)
        .solve()
        .unwrap();
    assert_eq!(solved.outcome, Outcome::Infeasible);
    assert_eq!(count.0, 1);
}

#[test]
fn test_a_script_outcome_does_not_depend_on_the_events_asked_for() {
    let (m, ..) = two_binaries();
    let steps = vec![Step::Progress(mip(1, None, Some(5.0))), Step::Poll(mip(2, None, None))];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), m.build().unwrap()).unwrap();
    let stop = AtomicBool::new(true);
    let solved = solver.run().stop(&stop).solve().unwrap();
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
    let mut set = Events::LOG | Events::PROGRESS;
    set |= Events::POLL;
    assert_eq!(set.len(), 3);
    assert_eq!(set & Events::POLL, Events::POLL);
    set &= Events::LOG | Events::POLL | Events::NATIVE;
    assert_eq!(set, Events::LOG | Events::POLL);
    assert_eq!(set.iter().collect::<Vec<_>>(), [Events::LOG, Events::POLL]);
    assert_eq!(set.iter().rev().collect::<Vec<_>>(), [Events::POLL, Events::LOG]);
    set.extend([Events::NATIVE, Events::FEASIBLE]);
    assert_eq!(format!("{set:?}"), "Events(LOG | POLL | FEASIBLE | NATIVE)");
    assert_eq!(Events::ALL.iter().len(), 6);
    assert_eq!(Events::ALL.into_iter().collect::<Events>(), Events::ALL);
    assert_eq!(format!("{:?}", Events::NONE), "Events()");
    let mut iter = Events::LOG.iter();
    assert_eq!(
        (iter.next(), iter.next(), iter.next_back()),
        (Some(Events::LOG), None, None)
    );
}

const SINGLE: [Events; 6] = [
    Events::LOG,
    Events::PROGRESS,
    Events::POLL,
    Events::FEASIBLE,
    Events::INCUMBENT,
    Events::NATIVE,
];

proptest! {
    /// A set built from chosen single events iterates exactly those, in the order of the constants, both ways.
    #[test]
    fn test_events_iterate_the_chosen_singles(chosen in prop::collection::vec(any::<bool>(), 6)) {
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

    /// The set operations hold each single event exactly as the boolean operation on its membership does.
    #[test]
    fn test_events_set_operations_agree_with_membership(
        a in prop::collection::vec(any::<bool>(), 6),
        b in prop::collection::vec(any::<bool>(), 6),
    ) {
        let set = |chosen: &[bool]| -> Events {
            SINGLE.iter().zip(chosen).filter(|&(_, &c)| c).map(|(&e, _)| e).collect()
        };
        let (x, y) = (set(&a), set(&b));
        let mut assigned = x;
        assigned -= y;
        prop_assert_eq!(assigned, x - y);
        for (i, event) in SINGLE.iter().enumerate() {
            prop_assert_eq!((x | y).contains(*event), a[i] || b[i]);
            prop_assert_eq!((x & y).contains(*event), a[i] && b[i]);
            prop_assert_eq!((x - y).contains(*event), a[i] && !b[i]);
            prop_assert_eq!((!x).contains(*event), !a[i]);
        }
        prop_assert_eq!(!!x, x);
    }
}

#[test]
fn test_events_display_as_a_set_of_names() {
    assert_eq!(Events::NONE.to_string(), "{}");
    assert_eq!((Events::NATIVE | Events::LOG).to_string(), "{LOG, NATIVE}");
    assert_eq!(Events::ALL.complement(), Events::NONE);
    assert_eq!(
        (Events::ALL - Events::NATIVE).to_string(),
        "{LOG, PROGRESS, POLL, FEASIBLE, INCUMBENT}"
    );
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
        Step::Price {
            progress: p,
            mode: PriceMode::Farkas,
            duals: TiVec::new(),
        },
        Step::Note(String::new()),
    ]
    .iter()
    .filter_map(Step::event)
    .collect();
    assert_eq!(kinds, Events::ALL);
    let search = [
        Step::Lazy(found(0, 0.0, [0.0, 0.0])),
        Step::Cut {
            progress: p,
            relaxation: TiVec::new(),
        },
        Step::Heuristic(p),
        Step::Price {
            progress: p,
            mode: PriceMode::Farkas,
            duals: TiVec::new(),
        },
    ];
    assert!(search.iter().all(|step| step.event().is_none()));
}

#[test]
fn test_a_callbacks_row_is_checked_with_its_coefficients_added_up() {
    let x = Col::new(0);
    let twice = (1e308 * x + 1e308 * x).leq(1.0);
    assert_eq!(
        culmen_core::context::check_constraint(&twice, 1),
        Err(ActionError::NotFinite { col: x })
    );
    // Terms that cancel are fine: the row is `0 x <= 1`.
    let cancel = (1e308 * x - 1e308 * x).leq(1.0);
    assert!(culmen_core::context::check_constraint(&cancel, 1).is_ok());
}
