// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constraint handlers on SCIP: the hand-worked fixtures of the design (all-different, parity, an unresolved
//! rejection, precise locks, pricing with a rule), the two typed failures, a panic in `check` and in `enforce`, the
//! actions this release refuses, and random tiny integer programs with a rule against the reference model
//! `solve_where`, which enumerates the rule's prose.
//!
//! Every solve here sets a node limit and a time limit, and every handler counts each method's calls and fails past
//! a bound, so a defect ends as a failed assertion.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

use culmen_core::{
    ActionError, Bound, Bounds, Callback, CheckContext, Child, Col, ConstraintHandler, Declaration, EnforceContext,
    Events, HeuristicContext, HeuristicHandler, LinExpr, Linear, Lock, Locks, Model, NewCol, NodeAction, NodeContext,
    Outcome, PriceContext, Pricer, Program, ProgramColumn, ProgramPlan, Row, Sense, Settings, Solution,
    SolutionContext, SolveError, Solver, TiSlice, TiVec, Var, Verdict, bounds, relative_gap,
};
use culmen_scip::{
    CandidateSource, Scip, ScipCheck, ScipEnforce, ScipError, ScipHeuristic, ScipOptions, ScipPrice, ScipSolution,
};
use culmen_testing::reference::{Reference, solve, solve_where};
use culmen_testing::rules::Rule;
use culmen_testing::strategy::small_integer_with_rule;
use culmen_testing::tolerance::Tolerances;
use proptest::prelude::*;

/// The most calls of one method of a handler any test here expects; a handler past it fails.
const MAX_CALLS: usize = 100_000;

/// A solve to a zero gap with a node limit and a time limit.
fn limited() -> Settings {
    Settings {
        relative_gap: Some(relative_gap!(0.0)),
        node_limit: Some(10_000),
        time_limit: Some(Duration::from_secs(20)),
        ..Settings::new()
    }
}

fn optimal(outcome: Outcome) -> Solution {
    match outcome {
        Outcome::Optimal(solution) => solution,
        other => panic!("expected an optimum, got {other:?}"),
    }
}

/// Each method's calls, each failing past [`MAX_CALLS`].
#[derive(Debug, Default)]
struct Calls {
    check: usize,
    enforce: usize,
}

impl Calls {
    fn check(&mut self) {
        self.check += 1;
        assert!(self.check <= MAX_CALLS, "check was called {} times", self.check);
    }

    fn enforce(&mut self) {
        self.enforce += 1;
        assert!(self.enforce <= MAX_CALLS, "enforce was called {} times", self.enforce);
    }
}

/// A point of integer columns, rounded.
fn rounded(values: &TiSlice<Col, f64>) -> Vec<i64> {
    values.iter().map(|v| v.round() as i64).collect()
}

/// How a [`Ruled`] handler resolves a point its rule rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    /// It does nothing, which leaves the point to SCIP's branching.
    Reject,
    /// It adds every inequality the rule implies over the columns' bounds that the point violates, and otherwise does
    /// nothing.
    Constraint,
    /// It cuts the node off once every column of the rule is fixed there, and otherwise does nothing.
    CutOffWhenFixed,
}

/// Enforces a [`Rule`] over integer columns in `[0, upper]`, each position of the rule the column of that index.
struct Ruled {
    rule: Rule,
    style: Style,
    /// Whether the declaration lists the rule's columns, both ways, rather than every column.
    only: bool,
    /// The upper bound every column of the rule shares.
    upper: i64,
    calls: Calls,
    /// Where each point `check` was asked about came from.
    sources: Vec<CandidateSource>,
}

impl Ruled {
    fn new(rule: Rule, style: Style, only: bool, upper: i64) -> Self {
        Self {
            rule,
            style,
            only,
            upper,
            calls: Calls::default(),
            sources: Vec::new(),
        }
    }

    /// Inequalities the rule implies for every column of it in `[0, upper]`: for all-different, each set of `s` of its
    /// columns adds up to at least `0 + 1 + … + (s − 1)` and at most `upper + (upper − 1) + … + (upper − s + 1)`; for
    /// no-two-adjacent, two neighbours add up to at most `upper`; parity implies none.
    fn implied(&self) -> Vec<(Vec<usize>, i64, i64)> {
        let at = self.rule.positions();
        match &self.rule {
            Rule::AllDifferent(_) => (1u32..1 << at.len())
                .map(|set| {
                    at.iter()
                        .enumerate()
                        .filter(|(k, _)| set & (1 << k) != 0)
                        .map(|(_, &j)| j)
                        .collect()
                })
                .filter(|set: &Vec<usize>| set.len() >= 2)
                .map(|set| {
                    let s = i64::try_from(set.len()).unwrap();
                    let least = s * (s - 1) / 2;
                    let most = (0..s).map(|i| self.upper - i).sum();
                    (set, least, most)
                })
                .collect(),
            Rule::NoTwoAdjacent(path) => path.windows(2).map(|pair| (pair.to_vec(), 0, self.upper)).collect(),
            Rule::EvenParity(_) => Vec::new(),
        }
    }
}

impl ConstraintHandler<Scip> for Ruled {
    fn declaration(&self) -> Declaration {
        if !self.only {
            return Declaration::default();
        }
        let listed = self
            .rule
            .positions()
            .iter()
            .map(|&j| (Col::new(j), Lock::Both))
            .collect();
        Declaration {
            locks: Locks::Only(listed),
            ..Declaration::default()
        }
    }

    fn check(&mut self, ctx: &mut ScipCheck<'_>) -> Verdict {
        self.calls.check();
        self.sources.push(ctx.source());
        if self.rule.holds(&rounded(ctx.candidate())) {
            Verdict::Accept
        } else {
            Verdict::Reject
        }
    }

    fn enforce(&mut self, ctx: &mut ScipEnforce<'_>) {
        self.calls.enforce();
        let point = rounded(ctx.candidate());
        assert!(
            !self.rule.holds(&point),
            "enforce was asked about a point check accepts: {point:?}"
        );
        match self.style {
            Style::Reject => {}
            Style::Constraint => {
                for (set, least, most) in self.implied() {
                    let sum: i64 = set.iter().map(|&j| point[j]).sum();
                    if sum < least || sum > most {
                        let expr = LinExpr::sum(set.iter().map(|&j| Col::new(j)));
                        ctx.add_constraint(expr.between(least as f64, most as f64))
                            .expect("a valid constraint");
                    }
                }
            }
            Style::CutOffWhenFixed => {
                let fixed = self.rule.positions().iter().all(|&j| {
                    let bounds = ctx.bounds(Col::new(j)).expect("a column of the problem");
                    bounds.lower() == bounds.upper()
                });
                if fixed {
                    ctx.cut_off();
                }
            }
        }
    }
}

/// Keeps the value of every solution SCIP stores.
#[derive(Default)]
struct Stored(Vec<TiVec<Col, f64>>);

impl Callback<Scip> for Stored {
    fn events(&self) -> Events {
        Events::FEASIBLE
    }
    fn feasible(&mut self, ctx: &mut ScipSolution<'_>) {
        assert!(self.0.len() < MAX_CALLS, "more solutions than any test expects");
        self.0.push(ctx.values().to_owned());
    }
}

/// Solves `program` with `handler`, keeping every stored solution.
fn solve_ruled(program: &Program, handler: &mut Ruled) -> (Result<Outcome, SolveError<ScipError>>, Stored) {
    let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
    let mut stored = Stored::default();
    let solved = solver
        .run()
        .settings(limited())
        .callback(&mut stored)
        .constraint(handler)
        .solve();
    (solved.map(|solved| solved.outcome), stored)
}

/// Three columns in {0, 1, 2}, pairwise different, `x0 + x1 >= 2`, maximize `x0 + 2x1 + 3x2`: the best is
/// (0, 2, 1), worth 7; without the rule (2, 2, 2), worth 12.
fn all_different_fixture() -> Program {
    let mut m = Model::new();
    let x = m.add_vars(3, |i| Var::integer().bounds(0, 2).name(format!("x[{i}]")));
    m.add_constraint((x[0] + x[1]).geq(2));
    m.maximize(x[0] + 2.0 * x[1] + 3.0 * x[2]);
    m.build().unwrap()
}

#[test]
fn test_all_different_by_cutoff_constraints_or_rejection_reaches_seven_at_0_2_1() {
    // The design's fixture branches on a clash; branching is refused in this release, so the clash is resolved by
    // cutting off a node that fixes all three columns, by the inequalities the rule implies, or by SCIP's branching.
    let program = all_different_fixture();
    assert_eq!(solve(&program).objective(), Some(12.0));
    for style in [Style::CutOffWhenFixed, Style::Constraint, Style::Reject] {
        for only in [false, true] {
            let mut handler = Ruled::new(Rule::AllDifferent(vec![0, 1, 2]), style, only, 2);
            let (outcome, stored) = solve_ruled(&program, &mut handler);
            let best = optimal(outcome.unwrap());
            assert_eq!(best.objective, 7.0, "{style:?} {only}");
            assert_eq!(rounded(&best.values), [0, 2, 1], "{style:?} {only}");
            assert!(handler.calls.check > 0 && handler.calls.enforce > 0, "{style:?} {only}");
            for values in &stored.0 {
                assert!(
                    handler.rule.holds(&rounded(values)),
                    "SCIP stored {values:?} ({style:?} {only})"
                );
            }
        }
    }
}

#[test]
fn test_parity_by_no_good_constraints_reaches_five() {
    // Binaries x1..x4 with an even sum of at most 3, maximize 3x1 + 2x2 + 2x3 + x4: x1 and one of x2, x3, worth 5.
    let mut m = Model::new();
    let x = m.add_vars(4, |_| Var::binary());
    m.add_constraint(LinExpr::sum(x.iter().copied()).leq(3));
    m.maximize(3.0 * x[0] + 2.0 * x[1] + 2.0 * x[2] + x[3]);
    let program = m.build().unwrap();

    /// Cuts off an odd point of binaries with the inequality only that point violates.
    struct NoGood(Calls);

    impl ConstraintHandler<Scip> for NoGood {
        fn check(&mut self, ctx: &mut ScipCheck<'_>) -> Verdict {
            self.0.check();
            let sum: i64 = rounded(ctx.candidate()).iter().sum();
            if sum % 2 == 0 { Verdict::Accept } else { Verdict::Reject }
        }

        fn enforce(&mut self, ctx: &mut ScipEnforce<'_>) {
            self.0.enforce();
            // Σ over the ones of (1 − x) + Σ over the zeros of x >= 1 holds at every binary point but this one.
            let point = rounded(ctx.candidate());
            let ones = i32::try_from(point.iter().filter(|&&v| v == 1).count()).unwrap();
            let expr = LinExpr::sum(
                point
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| if v == 1 { -1.0 * Col::new(j) } else { 1.0 * Col::new(j) }),
            );
            ctx.add_constraint(expr.geq(1 - ones)).expect("a valid constraint");
        }
    }

    let mut handler = NoGood(Calls::default());
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let best = optimal(
        solver
            .run()
            .settings(limited())
            .constraint(&mut handler)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, 5.0);
    assert_eq!(rounded(&best.values).iter().sum::<i64>(), 2);
}

/// A binary `x` and a continuous `y` in `[0, 1]`, maximize `x + y`, with the rule "x = 0".
fn x_is_zero() -> Program {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::continuous().upper(1));
    m.maximize(x + y);
    m.build().unwrap()
}

/// Rejects every point with `x = 1`; with `cut_off`, cuts off a node that fixes `x` at 1.
struct XIsZero {
    cut_off: bool,
    calls: Calls,
}

impl ConstraintHandler<Scip> for XIsZero {
    fn check(&mut self, ctx: &mut ScipCheck<'_>) -> Verdict {
        self.calls.check();
        if ctx.candidate()[Col::new(0)] > 0.5 {
            Verdict::Reject
        } else {
            Verdict::Accept
        }
    }

    fn enforce(&mut self, ctx: &mut ScipEnforce<'_>) {
        self.calls.enforce();
        if self.cut_off && ctx.bounds(Col::new(0)) == Some(bounds!(1.0, 1.0)) {
            ctx.cut_off();
        }
    }
}

#[test]
fn test_a_rejection_scip_cannot_resolve_fails_typed_and_a_cutoff_resolves_it() {
    // The LP point has x = 1; SCIP branches on x, and the child x = 1 has nothing left to branch on and the
    // continuous y, so a rejection without action there is one SCIP cannot resolve.
    let program = x_is_zero();
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut rejects = XIsZero {
        cut_off: false,
        calls: Calls::default(),
    };
    let failed = solver.run().settings(limited()).constraint(&mut rejects).solve();
    assert!(
        matches!(failed, Err(SolveError::Backend(ScipError::UnresolvedRejection))),
        "{failed:?}"
    );

    // The same rule, with the node fixing x at 1 cut off: x = 0, y = 1.
    let mut cuts = XIsZero {
        cut_off: true,
        calls: Calls::default(),
    };
    let best = optimal(
        solver
            .run()
            .settings(limited())
            .constraint(&mut cuts)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, 1.0);
    assert_eq!(best.values[Col::new(0)], 0.0);
}

#[test]
fn test_a_handler_that_never_resolves_its_point_ends_as_a_stalled_enforcement() {
    // x in [0, 1], maximize x, with a rule rejecting every x > 0 that the handler only ever shaves: it adds
    // x <= 1/k for k = 2, 3, ..., each new and violated by the last LP point, so SCIP asks again at the root without
    // end.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    m.maximize(x);
    let program = m.build().unwrap();

    struct Shaves(Calls, f64);

    impl ConstraintHandler<Scip> for Shaves {
        fn check(&mut self, ctx: &mut ScipCheck<'_>) -> Verdict {
            self.0.check();
            if ctx.candidate()[Col::new(0)] > 1e-9 {
                Verdict::Reject
            } else {
                Verdict::Accept
            }
        }

        fn enforce(&mut self, ctx: &mut ScipEnforce<'_>) {
            self.0.enforce();
            self.1 += 1.0;
            ctx.add_constraint(Col::new(0).leq(1.0 / self.1))
                .expect("a valid constraint");
        }
    }

    let options = ScipOptions {
        enforcement_rounds: 50,
        ..ScipOptions::default()
    };
    let mut solver = Solver::new(Scip::with_options(options), program).unwrap();
    let mut shaves = Shaves(Calls::default(), 1.0);
    let failed = solver.run().settings(limited()).constraint(&mut shaves).solve();
    assert!(
        matches!(
            failed,
            Err(SolveError::Backend(ScipError::EnforcementStalled { rounds: 50 }))
        ),
        "{failed:?}"
    );
    assert_eq!(shaves.0.enforce, 50);
    // The backend solves again.
    assert_eq!(solver.solve().unwrap().outcome.objective(), Some(1.0));
}

#[test]
fn test_precise_locks_and_every_lock_reach_the_optimum_of_y_at_most_x() {
    // Binaries x and y with the rule y <= x in the handler alone, maximize 2y - x: x = y = 1, worth 1. Moving y up or
    // x down can break the rule, so the precise locks are y up and x down.
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    m.maximize(2.0 * y - x);
    let program = m.build().unwrap();

    struct YAtMostX {
        x: Col,
        y: Col,
        only: bool,
        calls: Calls,
    }

    impl ConstraintHandler<Scip> for YAtMostX {
        fn declaration(&self) -> Declaration {
            let locks = if self.only {
                Locks::Only(vec![(self.y, Lock::Up), (self.x, Lock::Down)])
            } else {
                Locks::All
            };
            Declaration {
                locks,
                ..Declaration::default()
            }
        }

        fn check(&mut self, ctx: &mut ScipCheck<'_>) -> Verdict {
            self.calls.check();
            let point = ctx.candidate();
            if point[self.y] > point[self.x] + 0.5 {
                Verdict::Reject
            } else {
                Verdict::Accept
            }
        }

        fn enforce(&mut self, ctx: &mut ScipEnforce<'_>) {
            self.calls.enforce();
            ctx.add_constraint(self.y.leq(self.x)).expect("a valid constraint");
        }
    }

    for only in [true, false] {
        let mut handler = YAtMostX {
            x,
            y,
            only,
            calls: Calls::default(),
        };
        let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
        let best = optimal(
            solver
                .run()
                .settings(limited())
                .constraint(&mut handler)
                .solve()
                .unwrap()
                .outcome,
        );
        assert_eq!(best.objective, 1.0, "{only}");
        assert_eq!((best.values[x], best.values[y]), (1.0, 1.0), "{only}");
    }
}

#[test]
fn test_actions_of_later_releases_are_refused_and_the_rules_of_a_node_hold() {
    // Two binaries that must differ, resolved by SCIP's branching; the handler tries every action this release
    // refuses at its first enforcement and records what came back.
    let mut m = Model::new();
    let x = m.add_vars(2, |_| Var::binary());
    m.maximize(x[0] + x[1]);
    let program = m.build().unwrap();

    struct Tries {
        calls: Calls,
        errors: Vec<ActionError>,
        after_constraint: Vec<ActionError>,
        depth: Option<u32>,
        source: Option<CandidateSource>,
    }

    impl ConstraintHandler<Scip> for Tries {
        fn declaration(&self) -> Declaration {
            Declaration {
                locks: Locks::Only(vec![(Col::new(0), Lock::Both)]),
                ..Declaration::default()
            }
        }

        fn check(&mut self, ctx: &mut ScipCheck<'_>) -> Verdict {
            self.calls.check();
            let point = rounded(ctx.candidate());
            if point[0] == point[1] {
                Verdict::Reject
            } else {
                Verdict::Accept
            }
        }

        fn enforce(&mut self, ctx: &mut ScipEnforce<'_>) {
            self.calls.enforce();
            if !self.errors.is_empty() {
                return;
            }
            self.depth = Some(ctx.depth());
            self.source = Some(ctx.source());
            let (a, b) = (Col::new(0), Col::new(1));
            let one = bounds!(1.0, 1.0);
            self.errors.extend(ctx.tighten(a, one, None).err());
            self.errors.extend(ctx.tighten(b, one, None).err());
            self.errors.extend(ctx.tighten(Col::new(7), one, None).err());
            let child = Child {
                bounds: vec![(a, one)],
                ..Child::default()
            };
            self.errors.extend(ctx.branch(vec![child]).err());
            self.errors.extend(ctx.cut_off_because(&[(a, Bound::Lower)]).err());
            self.errors
                .extend(ctx.cut_off_because(&[(Col::new(7), Bound::Lower)]).err());
            assert_eq!(ctx.bounds(Col::new(7)), None);
            // A constraint is taken, and then a branching is refused for following it.
            ctx.add_constraint((a + b).leq(1)).expect("a valid constraint");
            self.after_constraint.extend(ctx.branch(Vec::new()).err());
        }
    }

    let mut tries = Tries {
        calls: Calls::default(),
        errors: Vec::new(),
        after_constraint: Vec::new(),
        depth: None,
        source: None,
    };
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let best = optimal(
        solver
            .run()
            .settings(limited())
            .constraint(&mut tries)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, 1.0);
    let refused = |e: &ActionError| matches!(e, ActionError::Refused(_));
    assert!(refused(&tries.errors[0]), "{:?}", tries.errors);
    assert_eq!(tries.errors[1], ActionError::NotInvolved { col: Col::new(1) });
    assert_eq!(tries.errors[2], ActionError::UnknownColumn { col: Col::new(7) });
    assert!(refused(&tries.errors[3]), "{:?}", tries.errors);
    assert!(refused(&tries.errors[4]), "{:?}", tries.errors);
    assert_eq!(tries.errors[5], ActionError::UnknownColumn { col: Col::new(7) });
    assert_eq!(tries.errors.len(), 6);
    assert_eq!(
        tries.after_constraint,
        [ActionError::Exclusive {
            action: NodeAction::Branch,
            after: NodeAction::Reduce
        }]
    );
    assert_eq!(tries.depth, Some(0));
    assert!(
        matches!(tries.source, Some(CandidateSource::Lp | CandidateSource::Pseudo)),
        "{:?}",
        tries.source
    );
}

/// Panics in `check` or in `enforce`, as it is told.
struct PanicsIn(&'static str);

impl ConstraintHandler<Scip> for PanicsIn {
    fn check(&mut self, _: &mut ScipCheck<'_>) -> Verdict {
        assert!(self.0 != "check", "panic at check");
        Verdict::Reject
    }

    fn enforce(&mut self, _: &mut ScipEnforce<'_>) {
        panic!("panic at {}", self.0);
    }
}

#[test]
fn test_a_panic_in_check_or_enforce_resumes_after_scip_returns_and_scip_solves_again() {
    let program = all_different_fixture();
    for at in ["check", "enforce"] {
        let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let _ = solver.run().settings(limited()).constraint(&mut PanicsIn(at)).solve();
        }));
        let payload = caught.expect_err("the panic propagates");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        assert_eq!(message, Some(format!("panic at {at}").as_str()));
        let best = optimal(solver.run().settings(limited()).solve().unwrap().outcome);
        assert_eq!(best.objective, 12.0, "{at}");
    }
}

/// Supplies the point with every column at its upper bound, which the all-different rule rejects, once, and then the
/// rule's optimum.
struct SuppliesBoth {
    calls: usize,
    results: Vec<Result<(), ActionError>>,
}

impl HeuristicHandler<Scip> for SuppliesBoth {
    fn heuristic(&mut self, ctx: &mut ScipHeuristic<'_>) {
        self.calls += 1;
        assert!(self.calls <= MAX_CALLS, "the heuristic ran without end");
        if self.calls == 1 {
            let clash: TiVec<Col, f64> = vec![2.0, 2.0, 2.0].into();
            let best: TiVec<Col, f64> = vec![0.0, 2.0, 1.0].into();
            self.results.push(ctx.supply(&clash));
            self.results.push(ctx.supply(&best));
        }
    }
}

#[test]
fn test_a_supplied_solution_meets_the_constraint_handler_before_scip_keeps_it() {
    // The heuristic's solutions are tried once it returns, so the handler judges each: the clash is dropped, the
    // optimum kept, and every stored solution satisfies the rule.
    let program = all_different_fixture();
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut handler = Ruled::new(Rule::AllDifferent(vec![0, 1, 2]), Style::CutOffWhenFixed, false, 2);
    let mut heuristic = SuppliesBoth {
        calls: 0,
        results: Vec::new(),
    };
    let mut stored = Stored::default();
    let best = optimal(
        solver
            .run()
            .settings(limited())
            .callback(&mut stored)
            .heuristic(&mut heuristic)
            .constraint(&mut handler)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, 7.0);
    assert_eq!(heuristic.results, [Ok(()), Ok(())]);
    assert!(
        handler.sources.contains(&CandidateSource::Check),
        "{:?}",
        handler.sources
    );
    assert!(!stored.0.is_empty());
    for values in &stored.0 {
        assert!(handler.rule.holds(&rounded(values)), "SCIP stored {values:?}");
    }
}

/// Prices every withheld column that improves, each once, and keeps which column of the full program each added
/// `Col` stands for, shared with the constraint handler.
struct Prices<'a> {
    columns: Vec<NewCol>,
    /// The full program's index of every column the solver holds, in the order of their ids.
    full: &'a Mutex<Vec<usize>>,
    rounds: usize,
}

impl Pricer<Scip> for Prices<'_> {
    fn price(&mut self, ctx: &mut ScipPrice<'_>) {
        self.rounds += 1;
        assert!(self.rounds <= 2_000, "more pricing rounds than the test expects");
        let mut full = self.full.lock().unwrap();
        for (j, column) in self.columns.iter().enumerate() {
            if !full.contains(&j) && ctx.improves(column) {
                ctx.add_col(column.clone()).expect("a column SCIP takes");
                full.push(j);
            }
        }
    }
}

/// No two neighbours along the full program's columns are both nonzero; cuts off a node that holds both columns of a
/// clashing pair at least one, which no column priced later changes.
struct NoTwoAdjacentPriced<'a> {
    full: &'a Mutex<Vec<usize>>,
    n: usize,
    calls: Calls,
}

impl NoTwoAdjacentPriced<'_> {
    /// The point in the full program's columns, a column not priced yet at zero.
    fn in_full(&self, values: &TiSlice<Col, f64>) -> Vec<i64> {
        let mut point = vec![0; self.n];
        for (k, &j) in self.full.lock().unwrap().iter().enumerate() {
            point[j] = values[Col::new(k)].round() as i64;
        }
        point
    }

    /// The solver's column of the full program's column `j`, `None` before it is priced.
    fn col(&self, j: usize) -> Option<Col> {
        self.full.lock().unwrap().iter().position(|&f| f == j).map(Col::new)
    }
}

impl ConstraintHandler<Scip> for NoTwoAdjacentPriced<'_> {
    fn check(&mut self, ctx: &mut ScipCheck<'_>) -> Verdict {
        self.calls.check();
        let path: Vec<usize> = (0..self.n).collect();
        if culmen_testing::rules::no_two_adjacent(&path, &self.in_full(ctx.candidate())) {
            Verdict::Accept
        } else {
            Verdict::Reject
        }
    }

    fn enforce(&mut self, ctx: &mut ScipEnforce<'_>) {
        self.calls.enforce();
        let point = self.in_full(ctx.candidate());
        for j in 0..self.n - 1 {
            if point[j] == 0 || point[j + 1] == 0 {
                continue;
            }
            let held = |j: usize| {
                let col = self.col(j).expect("a nonzero column is priced");
                ctx.bounds(col).expect("a column of the problem").lower() >= 1.0
            };
            if held(j) && held(j + 1) {
                ctx.cut_off();
                return;
            }
        }
    }
}

#[test]
fn test_pricing_with_a_rule_over_the_priced_columns_reaches_the_reference_over_every_column() {
    // Maximize 3x0 + 4x1 + 5x2 + 3x3 with 2x0 + 3x1 + 4x2 + 2x3 <= 9, every column integer in [0, 2], no two
    // neighbours both nonzero; the program starts with x0 and prices the others.
    let (costs, weights) = ([3.0, 4.0, 5.0, 3.0], [2.0, 3.0, 4.0, 2.0]);
    let columns: Vec<NewCol> = costs
        .iter()
        .zip(weights)
        .map(|(&cost, weight)| NewCol {
            column: ProgramColumn::integer(cost, bounds!(0.0, 2.0)),
            entries: vec![(Row::new(0), weight)],
        })
        .collect();
    let mut plan = ProgramPlan::new();
    plan.sense = Sense::Maximize;
    let x0 = plan.col(columns[0].column.clone());
    plan.row(Bounds::at_most(9.0).unwrap(), [(x0, weights[0])]);
    let program = plan.build().unwrap();
    let mut full = ProgramPlan::from(program.clone());
    for column in &columns[1..] {
        let col = full.col(column.column.clone());
        full.rows[Row::new(0)].terms.push((col, column.entries[0].1));
    }
    let full = full.build().unwrap();
    let path = [0, 1, 2, 3];
    let reference = solve_where(&full, &|p: &[i64]| culmen_testing::rules::no_two_adjacent(&path, p));
    assert_ne!(reference, solve(&full), "the rule moves the optimum");

    let shared = Mutex::new(vec![0]);
    let mut pricer = Prices {
        columns,
        full: &shared,
        rounds: 0,
    };
    let mut handler = NoTwoAdjacentPriced {
        full: &shared,
        n: 4,
        calls: Calls::default(),
    };
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let solved = solver
        .run()
        .settings(limited())
        .pricer(&mut pricer)
        .constraint(&mut handler)
        .solve()
        .unwrap();
    let best = optimal(solved.outcome);
    assert_eq!(Some(best.objective), reference.objective(), "{best:?}");
    let point = handler.in_full(&best.values);
    assert!(culmen_testing::rules::no_two_adjacent(&path, &point), "{point:?}");
    // The solver solves the program with the priced columns again, a new problem whose locks start balanced.
    let again = optimal(solver.run().settings(limited()).solve().unwrap().outcome);
    assert!(again.objective >= best.objective);
}

/// Whether SCIP's outcome is the reference's over the points the rule accepts, its point satisfying the rule.
fn agrees(program: &Program, rule: &Rule, outcome: &Outcome) -> Result<(), TestCaseError> {
    let expected: Reference = solve_where(program, &|p: &[i64]| rule.holds(p));
    match (Tolerances::SCIP.window(program, &expected), outcome) {
        (None, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded) => Ok(()),
        (Some(window), Outcome::Optimal(best)) => {
            prop_assert!(
                window.contains(best.objective),
                "SCIP {} vs exact {expected:?}, window {window:?}",
                best.objective
            );
            let point = rounded(&best.values);
            prop_assert!(rule.holds(&point), "{point:?} breaks {rule:?}");
            let stray = Tolerances::SCIP.stray(program, &best.values);
            prop_assert!(stray.is_none(), "{stray:?} {best:?}");
            Ok(())
        }
        (_, got) => Err(TestCaseError::fail(format!("exact {expected:?}, SCIP {got:?}"))),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// For every enforcement style this release has and both lock modes, SCIP's optimum is the reference's over the
    /// points the rule accepts, and every solution SCIP stores satisfies the rule.
    #[test]
    fn test_the_scip_optimum_equals_solve_where_the_rule_holds(
        (program, rule) in small_integer_with_rule(),
        style in prop_oneof![Just(Style::Reject), Just(Style::Constraint), Just(Style::CutOffWhenFixed)],
        only in any::<bool>(),
    ) {
        let mut handler = Ruled::new(rule.clone(), style, only, 3);
        let (outcome, stored) = solve_ruled(&program, &mut handler);
        let outcome = outcome.map_err(|e| TestCaseError::fail(format!("{e}")))?;
        agrees(&program, &rule, &outcome)?;
        for values in &stored.0 {
            prop_assert!(rule.holds(&rounded(values)), "SCIP stored {values:?}, which breaks {rule:?}");
        }
    }
}
