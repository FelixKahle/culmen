// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constraint handlers on SCIP: the hand-worked fixtures of the design (all-different, parity, an unresolved
//! rejection, precise locks, pricing with a rule), each propagating with and without explanations, a conflict SCIP
//! learns from, tightening during enforcement, the two typed failures, a panic in every method SCIP calls, the actions
//! this release refuses, and random tiny integer programs with a rule against the reference model `solve_where`,
//! which enumerates the rule's prose.
//!
//! The propagating handler checks SCIP's plumbing inside each call, against every point the program and the rule
//! accept (`accepted_points`, an enumeration of the rule's prose, not of the handler's reasoning): every tightening
//! keeps each accepted point of the node's box, every conflict's box holds none, every explanation is asked for a
//! tightening made with that reason, and every complete explanation implies its bound on each accepted point inside
//! the named bounds as they were before, the original bounds elsewhere.
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
    Events, ExplainContext, Explained, HeuristicContext, HeuristicHandler, Inferred, LinExpr, Linear, Lock, Locks,
    Model, NewCol, NodeAction, NodeContext, Outcome, PriceContext, Pricer, Program, ProgramColumn, ProgramPlan, Reason,
    Row, Sense, Settings, Solution, SolutionContext, SolveError, Solver, TiSlice, TiVec, Var, Verdict, bounds,
    relative_gap,
};
use culmen_scip::{
    CandidateSource, ParamValue, Scip, ScipCheck, ScipEnforce, ScipError, ScipExplain, ScipHeuristic, ScipOptions,
    ScipParam, ScipPrice, ScipPropagate, ScipSolution,
};
use culmen_testing::reference::{Reference, accepted_points, solve, solve_where};
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
    propagate: usize,
    explain: usize,
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

    fn propagate(&mut self) {
        self.propagate += 1;
        assert!(
            self.propagate <= MAX_CALLS,
            "propagate was called {} times",
            self.propagate
        );
    }

    fn explain(&mut self) {
        self.explain += 1;
        assert!(self.explain <= MAX_CALLS, "explain was called {} times", self.explain);
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

/// Whether a propagating [`Ruled`] handler gives its tightenings reasons, and how it explains them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Explain {
    /// No reasons, so `explain` is never asked; an infeasible node is cut off plainly.
    Never,
    /// Reasons, explained as unknown; an infeasible node is cut off plainly.
    Unknown,
    /// Reasons, explained completely; an infeasible node is cut off naming the bounds that make it so.
    Complete,
}

/// What a [`Ruled`] handler's propagation knows of the program: every point the program and the rule accept, listed
/// from the rule's prose, and each column's original bounds. The model-free checks inside the handler test SCIP's
/// plumbing against them: the bounds a context reports, the reasons it hands back, and what it asks to explain.
#[derive(Debug)]
struct Propagation {
    explain: Explain,
    accepted: Vec<Vec<i64>>,
    original: Vec<(i64, i64)>,
    /// Every tightening the propagation asked for with a reason: the column, the end it moved and the reason.
    tagged: Vec<(Col, Bound, u32)>,
    /// How many cutoffs named the bounds that make the node infeasible.
    because: usize,
    /// How many explanations answered complete.
    complete: usize,
}

/// What a propagation of a rule finds at a node's box.
#[derive(Debug, PartialEq, Eq)]
enum Inference {
    /// Each position's column narrowed to `[lower, upper]`, with the reason the explanation reads.
    Tighten(Vec<(usize, i64, i64, u32)>),
    /// The node holds no accepted point because of these bounds of the node.
    Conflict(Vec<(usize, Bound)>),
}

/// What a rule implies at a box of integer columns, each `(lower, upper)`, written from the rule's prose:
///
/// - all-different: a column fixed at `v` takes `v` from either end of every other column's domain; two columns
///   fixed at the same value are a conflict;
/// - even parity: with every column but one fixed, the last takes the parity that makes the sum even at both ends;
///   no such value, or an odd sum of fixed columns, is a conflict;
/// - no-two-adjacent: a column at least one forces each neighbour on the path to at most zero, when that neighbour is
///   at least zero; two neighbours at least one are a conflict.
fn infer(rule: &Rule, node: &[(i64, i64)]) -> Inference {
    let fixed = |j: usize| node[j].0 == node[j].1;
    let both = |j: usize| [(j, Bound::Lower), (j, Bound::Upper)];
    let mut tighten = Vec::new();
    match rule {
        Rule::AllDifferent(at) => {
            for (i, &a) in at.iter().enumerate().filter(|&(_, &a)| fixed(a)) {
                let v = node[a].0;
                for &c in at.iter().filter(|&&c| c != a) {
                    let (lower, upper) = node[c];
                    if lower == v && upper == v {
                        return Inference::Conflict(both(a).into_iter().chain(both(c)).collect());
                    } else if lower == v {
                        tighten.push((c, v + 1, upper, u32::try_from(i).unwrap()));
                    } else if upper == v {
                        tighten.push((c, lower, v - 1, u32::try_from(i).unwrap()));
                    }
                }
            }
        }
        Rule::EvenParity(at) => {
            let free: Vec<usize> = at.iter().copied().filter(|&j| !fixed(j)).collect();
            let sum: i64 = at.iter().filter(|&&j| fixed(j)).map(|&j| node[j].0).sum();
            let others = |c: Option<usize>| at.iter().filter(move |&&j| Some(j) != c).flat_map(|&j| both(j));
            match free[..] {
                [] if sum.rem_euclid(2) == 1 => return Inference::Conflict(others(None).collect()),
                [c] => {
                    let parity = sum.rem_euclid(2);
                    let (lower, upper) = node[c];
                    let lower = if lower.rem_euclid(2) == parity {
                        lower
                    } else {
                        lower + 1
                    };
                    let upper = if upper.rem_euclid(2) == parity {
                        upper
                    } else {
                        upper - 1
                    };
                    if lower > upper {
                        return Inference::Conflict(others(None).collect());
                    }
                    if (lower, upper) != node[c] {
                        let k = at.iter().position(|&j| j == c).unwrap();
                        tighten.push((c, lower, upper, u32::try_from(k).unwrap()));
                    }
                }
                _ => {}
            }
        }
        Rule::NoTwoAdjacent(path) => {
            for (k, &a) in path.iter().enumerate().filter(|&(_, &a)| node[a].0 >= 1) {
                let neighbours = [k.checked_sub(1), Some(k + 1)];
                for nb in neighbours.into_iter().flatten().filter_map(|m| path.get(m).copied()) {
                    let (lower, upper) = node[nb];
                    if lower >= 1 {
                        return Inference::Conflict(vec![(a, Bound::Lower), (nb, Bound::Lower)]);
                    } else if lower >= 0 && upper > 0 {
                        tighten.push((nb, lower, 0, u32::try_from(k).unwrap()));
                    }
                }
            }
        }
    }
    Inference::Tighten(tighten)
}

/// The bounds an inference of `rule` with reason `reason` on `inferred` rested on, by the rule's prose: for
/// all-different, both ends of the fixed column at position `reason` and the inferred column's own end; for even
/// parity, both ends of every other column and the inferred column's own end; for no-two-adjacent, the lower end of the
/// column at position `reason` on the path.
fn rested_on(rule: &Rule, reason: u32, inferred: &Inferred) -> Vec<(usize, Bound)> {
    let at = rule.positions();
    let r = usize::try_from(reason).unwrap();
    let c = inferred.col.index();
    match rule {
        Rule::AllDifferent(_) => vec![(at[r], Bound::Lower), (at[r], Bound::Upper), (c, inferred.end)],
        Rule::EvenParity(_) => at
            .iter()
            .filter(|&&j| j != c)
            .flat_map(|&j| [(j, Bound::Lower), (j, Bound::Upper)])
            .chain([(c, inferred.end)])
            .collect(),
        Rule::NoTwoAdjacent(path) => vec![(path[r], Bound::Lower)],
    }
}

/// A node's bounds of integer columns, rounded inward.
fn integral(bounds: Bounds) -> (i64, i64) {
    (
        (bounds.lower() - 1e-6).ceil() as i64,
        (bounds.upper() + 1e-6).floor() as i64,
    )
}

/// Whether `point` lies in `area`, a box of integer columns.
fn inside(point: &[i64], area: &[(i64, i64)]) -> bool {
    point
        .iter()
        .zip(area)
        .all(|(&v, &(lower, upper))| lower <= v && v <= upper)
}

/// `original` narrowed by the named ends of `bounds`.
fn named_box(
    original: &[(i64, i64)],
    named: &[(usize, Bound)],
    bounds: impl Fn(usize) -> (i64, i64),
) -> Vec<(i64, i64)> {
    let mut area = original.to_vec();
    for &(j, end) in named {
        let (lower, upper) = bounds(j);
        match end {
            Bound::Lower => area[j].0 = area[j].0.max(lower),
            Bound::Upper => area[j].1 = area[j].1.min(upper),
        }
    }
    area
}

/// Enforces a [`Rule`] over integer columns in `[0, upper]`, each position of the rule the column of that index; with
/// a [`Propagation`], also propagates and explains it.
struct Ruled {
    rule: Rule,
    style: Style,
    /// Whether the declaration lists the rule's columns, both ways, rather than every column.
    only: bool,
    /// The upper bound every column of the rule shares.
    upper: i64,
    propagation: Option<Propagation>,
    /// The method that panics, if any.
    panics_in: Option<&'static str>,
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
            propagation: None,
            panics_in: None,
            calls: Calls::default(),
            sources: Vec::new(),
        }
    }

    /// The same handler, propagating `program`'s rule and explaining as `explain` says.
    fn propagating(mut self, program: &Program, explain: Explain) -> Self {
        let rule = self.rule.clone();
        let original = program
            .cols()
            .map(|col| integral(program.columns()[col].bounds))
            .collect();
        self.propagation = Some(Propagation {
            explain,
            accepted: accepted_points(program, &|p: &[i64]| rule.holds(p)),
            original,
            tagged: Vec::new(),
            because: 0,
            complete: 0,
        });
        self
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

    fn propagate(&mut self, ctx: &mut ScipPropagate<'_>) {
        self.calls.propagate();
        assert!(self.panics_in != Some("propagate"), "panic at propagate");
        let Some(propagation) = self.propagation.as_mut() else {
            return;
        };
        let node: Vec<(i64, i64)> = (0..propagation.original.len())
            .map(|j| integral(ctx.bounds(Col::new(j)).expect("a column of the problem")))
            .collect();
        let held: Vec<&Vec<i64>> = propagation.accepted.iter().filter(|p| inside(p, &node)).collect();
        match infer(&self.rule, &node) {
            Inference::Conflict(named) => {
                // The named bounds of the node, with the original bounds elsewhere, hold no accepted point.
                let area = named_box(&propagation.original, &named, |j| node[j]);
                let stray = propagation.accepted.iter().find(|p| inside(p, &area));
                assert!(
                    stray.is_none(),
                    "{stray:?} is accepted in the conflict {named:?} at {node:?}"
                );
                if propagation.explain == Explain::Complete {
                    let named: Vec<(Col, Bound)> = named.iter().map(|&(j, end)| (Col::new(j), end)).collect();
                    ctx.cut_off_because(&named).expect("columns of the problem");
                    propagation.because += 1;
                } else {
                    ctx.cut_off();
                }
            }
            Inference::Tighten(found) => {
                for (j, lower, upper, reason) in found {
                    // Every accepted point of the node keeps the tightening.
                    let lost = held.iter().find(|p| p[j] < lower || p[j] > upper);
                    assert!(
                        lost.is_none(),
                        "{lost:?} is lost by x{j} in [{lower}, {upper}] at {node:?}"
                    );
                    let col = Col::new(j);
                    let reason = (propagation.explain != Explain::Never).then_some(Reason(reason));
                    if let Some(Reason(r)) = reason {
                        for (end, moved) in [(Bound::Lower, lower > node[j].0), (Bound::Upper, upper < node[j].1)] {
                            if moved {
                                propagation.tagged.push((col, end, r));
                            }
                        }
                    }
                    let bounds = Bounds::new(lower as f64, upper as f64).expect("a non-empty interval");
                    ctx.tighten(col, bounds, reason)
                        .expect("a column the declaration involves");
                }
            }
        }
    }

    fn explain(&mut self, ctx: &mut ScipExplain<'_>) -> Explained {
        self.calls.explain();
        assert!(self.panics_in != Some("explain"), "panic at explain");
        let propagation = self.propagation.as_mut().expect("explain follows a propagation");
        let (inferred, Reason(reason)) = (ctx.inferred(), ctx.reason());
        // The reason reaches the inference it was given with.
        assert!(
            propagation.tagged.contains(&(inferred.col, inferred.end, reason)),
            "{inferred:?} with reason {reason} was never inferred so"
        );
        if propagation.explain != Explain::Complete {
            return Explained::Unknown;
        }
        let named = rested_on(&self.rule, reason, &inferred);
        for &(j, end) in &named {
            ctx.because(Col::new(j), end).expect("a column of the problem");
        }
        // Every accepted point inside the named bounds as they were before, with the original bounds elsewhere,
        // satisfies the inferred bound.
        let before = |j: usize| integral(ctx.bounds_before(Col::new(j)).expect("a column of the problem"));
        let area = named_box(&propagation.original, &named, before);
        let c = inferred.col.index();
        let breaks = propagation.accepted.iter().filter(|p| inside(p, &area)).find(|p| {
            let v = p[c] as f64;
            match inferred.end {
                Bound::Lower => v < inferred.value - 1e-9,
                Bound::Upper => v > inferred.value + 1e-9,
            }
        });
        assert!(breaks.is_none(), "{breaks:?} in {area:?} breaks {inferred:?}");
        propagation.complete += 1;
        Explained::Complete
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

/// SCIP's options with conflict analysis for programs of any size: SCIP gives up a conflict of more than 15 % of the
/// integer variables (`conflict/maxvarsfac`), which in a program of a few columns is every conflict, unless
/// `conflict/minmaxvars` allows more.
fn learning() -> ScipOptions {
    ScipOptions {
        params: vec![ScipParam {
            name: "conflict/minmaxvars".to_owned(),
            value: ParamValue::Int(100),
        }],
        ..ScipOptions::default()
    }
}

/// Solves `program` with `handler`, keeping every stored solution, with conflict analysis for any program size.
fn solve_ruled(program: &Program, handler: &mut Ruled) -> (Result<Outcome, SolveError<ScipError>>, Stored) {
    let mut solver = Solver::new(Scip::with_options(learning()), program.clone()).unwrap();
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

/// Binaries x1..x4 with an even sum of at most 3, maximize 3x1 + 2x2 + 2x3 + x4: x1 and one of x2, x3, worth 5;
/// without the rule x1, x2 and x3, worth 7.
fn parity_fixture() -> Program {
    let mut m = Model::new();
    let x = m.add_vars(4, |_| Var::binary());
    m.add_constraint(LinExpr::sum(x.iter().copied()).leq(3));
    m.maximize(3.0 * x[0] + 2.0 * x[1] + 2.0 * x[2] + x[3]);
    m.build().unwrap()
}

#[test]
fn test_parity_by_no_good_constraints_reaches_five() {
    let program = parity_fixture();

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

/// Asserts what every solve of a propagating [`Ruled`] handler promises: the handler propagated, `explain` ran only for
/// tightenings with a reason, and every stored solution satisfies the rule.
fn propagated(handler: &Ruled, stored: &Stored, what: &str) {
    let explain = handler.propagation.as_ref().map(|p| p.explain);
    assert!(handler.calls.propagate > 0, "{what}");
    if explain == Some(Explain::Never) {
        assert_eq!(handler.calls.explain, 0, "{what}");
    }
    for values in &stored.0 {
        assert!(handler.rule.holds(&rounded(values)), "SCIP stored {values:?} ({what})");
    }
}

#[test]
fn test_a_propagating_all_different_reaches_seven_with_and_without_explanations() {
    let program = all_different_fixture();
    for explain in [Explain::Never, Explain::Unknown, Explain::Complete] {
        for style in [Style::Reject, Style::CutOffWhenFixed] {
            for only in [false, true] {
                let rule = Rule::AllDifferent(vec![0, 1, 2]);
                let mut handler = Ruled::new(rule, style, only, 2).propagating(&program, explain);
                let (outcome, stored) = solve_ruled(&program, &mut handler);
                let what = format!("{explain:?} {style:?} {only}");
                let best = optimal(outcome.unwrap());
                assert_eq!(best.objective, 7.0, "{what}");
                assert_eq!(rounded(&best.values), [0, 2, 1], "{what}");
                propagated(&handler, &stored, &what);
            }
        }
    }
}

#[test]
fn test_a_propagating_parity_reaches_five_with_and_without_explanations() {
    let program = parity_fixture();
    for explain in [Explain::Never, Explain::Unknown, Explain::Complete] {
        for only in [false, true] {
            let rule = Rule::EvenParity(vec![0, 1, 2, 3]);
            let mut handler = Ruled::new(rule, Style::Reject, only, 1).propagating(&program, explain);
            let (outcome, stored) = solve_ruled(&program, &mut handler);
            let what = format!("{explain:?} {only}");
            let best = optimal(outcome.unwrap());
            assert_eq!(best.objective, 5.0, "{what}");
            assert_eq!(rounded(&best.values).iter().sum::<i64>(), 2, "{what}");
            propagated(&handler, &stored, &what);
        }
    }
}

/// Four columns in {0, 1, 2, 3}, pairwise different, `x0 + x1 >= 5`, maximize `x0 + 2x1 + 3x2 + 4x3`: `x0` and `x1`
/// take 2 and 3, `x2` and `x3` take 0 and 1, and the best is (2, 3, 0, 1), worth 12; without the rule (3, 3, 3, 3),
/// worth 30.
fn permutation_fixture() -> Program {
    let mut m = Model::new();
    let x = m.add_vars(4, |_| Var::integer().bounds(0, 3));
    m.add_constraint((x[0] + x[1]).geq(5));
    m.maximize(x[0] + 2.0 * x[1] + 3.0 * x[2] + 4.0 * x[3]);
    m.build().unwrap()
}

/// Four columns in {0, 1, 2}, pairwise different, which no point satisfies: four columns, three values.
fn pigeonhole_fixture() -> Program {
    let mut m = Model::new();
    let x = m.add_vars(4, |_| Var::integer().bounds(0, 2));
    m.maximize(LinExpr::sum(x.iter().copied()));
    m.build().unwrap()
}

#[test]
fn test_cutoffs_naming_their_bounds_teach_scip_conflicts_and_keep_the_optimum() {
    let program = permutation_fixture();
    assert_eq!(solve(&program).objective(), Some(30.0));
    for explain in [Explain::Never, Explain::Unknown, Explain::Complete] {
        let rule = Rule::AllDifferent(vec![0, 1, 2, 3]);
        let mut handler = Ruled::new(rule, Style::Reject, true, 3).propagating(&program, explain);
        let (outcome, stored) = solve_ruled(&program, &mut handler);
        let what = format!("{explain:?}");
        let best = optimal(outcome.unwrap());
        assert_eq!(best.objective, 12.0, "{what}");
        assert_eq!(rounded(&best.values), [2, 3, 0, 1], "{what}");
        propagated(&handler, &stored, &what);
    }

    // No point satisfies the pigeonhole: every branch ends in a conflict the propagation names, and SCIP's conflict
    // analysis asks for the explanations of the tightenings it rests on.
    let program = pigeonhole_fixture();
    let mut handler = Ruled::new(Rule::AllDifferent(vec![0, 1, 2, 3]), Style::Reject, false, 2)
        .propagating(&program, Explain::Complete);
    let (outcome, stored) = solve_ruled(&program, &mut handler);
    assert!(matches!(outcome, Ok(Outcome::Infeasible)), "{outcome:?}");
    propagated(&handler, &stored, "pigeonhole");
    let propagation = handler.propagation.as_ref().unwrap();
    assert!(propagation.because > 0, "{:?}", handler.calls);
    assert!(propagation.complete > 0, "{:?}", handler.calls);
}

/// A binary `x` and a continuous `y` in `[0, 1]`, maximize `x + y`, with the rule "x = 0".
fn x_is_zero() -> Program {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::continuous().upper(1));
    m.maximize(x + y);
    m.build().unwrap()
}

/// How [`XIsZero`] resolves a point with `x = 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resolve {
    /// It does nothing.
    Not,
    /// It cuts off a node that fixes `x` at 1.
    CutOff,
    /// It cuts off a node that fixes `x` at 1, naming the lower bound of `x` as the reason.
    CutOffBecause,
    /// It tightens `x` to 0, with the reason or without.
    Tighten(Option<Reason>),
}

/// Rejects every point with `x = 1`, and resolves it as `resolve` says.
struct XIsZero {
    resolve: Resolve,
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
        let x = Col::new(0);
        let fixed = ctx.bounds(x) == Some(bounds!(1.0, 1.0));
        match self.resolve {
            Resolve::Not => {}
            Resolve::CutOff if fixed => ctx.cut_off(),
            Resolve::CutOffBecause if fixed => {
                ctx.cut_off_because(&[(x, Bound::Lower)])
                    .expect("a column of the problem");
            }
            Resolve::CutOff | Resolve::CutOffBecause => {}
            Resolve::Tighten(reason) => ctx.tighten(x, bounds!(0.0, 0.0), reason).expect("an involved column"),
        }
    }

    fn explain(&mut self, ctx: &mut ScipExplain<'_>) -> Explained {
        self.calls.explain();
        // The rule alone forces x to 0: no bound is needed.
        assert_eq!(self.resolve, Resolve::Tighten(Some(ctx.reason())));
        assert_eq!(ctx.inferred().col, Col::new(0));
        assert_eq!(ctx.inferred().end, Bound::Upper);
        Explained::Complete
    }
}

#[test]
fn test_a_rejection_scip_cannot_resolve_fails_typed_and_a_cutoff_resolves_it() {
    // The LP point has x = 1; SCIP branches on x, and the child x = 1 has nothing left to branch on and the
    // continuous y, so a rejection without action there is one SCIP cannot resolve.
    let program = x_is_zero();
    let mut solver = Solver::new(Scip::with_options(learning()), program).unwrap();
    let mut rejects = XIsZero {
        resolve: Resolve::Not,
        calls: Calls::default(),
    };
    let failed = solver.run().settings(limited()).constraint(&mut rejects).solve();
    assert!(
        matches!(failed, Err(SolveError::Backend(ScipError::UnresolvedRejection))),
        "{failed:?}"
    );

    // The same rule, with the node fixing x at 1 cut off, plainly or naming x's lower bound, which SCIP may learn
    // x <= 0 from, or with x tightened to 0 during enforcement, with a reason or without: x = 0, y = 1.
    let resolves = [
        Resolve::CutOff,
        Resolve::CutOffBecause,
        Resolve::Tighten(None),
        Resolve::Tighten(Some(Reason(7))),
    ];
    for resolve in resolves {
        let mut resolves = XIsZero {
            resolve,
            calls: Calls::default(),
        };
        let best = optimal(
            solver
                .run()
                .settings(limited())
                .constraint(&mut resolves)
                .solve()
                .unwrap()
                .outcome,
        );
        assert_eq!(best.objective, 1.0, "{resolve:?}");
        assert_eq!(best.values[Col::new(0)], 0.0, "{resolve:?}");
        assert!(resolves.calls.enforce > 0, "{resolve:?}");
    }
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
fn test_the_rules_of_a_node_hold_and_branching_is_refused() {
    // Two binaries that must differ, resolved by a constraint; the handler tries every action SCIP refuses at its first
    // enforcement and records what came back.
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
                // The constraint added at the first enforcement resolves (1, 1); (0, 0) is worse than either optimum.
                ctx.add_constraint((Col::new(0) + Col::new(1)).geq(1))
                    .expect("a valid constraint");
                return;
            }
            self.depth = Some(ctx.depth());
            self.source = Some(ctx.source());
            let (a, b) = (Col::new(0), Col::new(1));
            let one = bounds!(1.0, 1.0);
            self.errors.extend(ctx.tighten(b, one, None).err());
            self.errors.extend(ctx.tighten(Col::new(7), one, None).err());
            self.errors
                .extend(ctx.cut_off_because(&[(Col::new(7), Bound::Lower)]).err());
            let child = Child {
                bounds: vec![(a, one)],
                ..Child::default()
            };
            self.errors.extend(ctx.branch(vec![child]).err());
            self.errors.extend(ctx.tighten(a, bounds!(-1e30, 1.0), None).err());
            assert_eq!(ctx.bounds(Col::new(7)), None);
            // A tightening that narrows nothing and a constraint are taken, and then a branching is refused for
            // following them.
            ctx.tighten(a, bounds!(0.0, 1.0), None)
                .expect("a column the declaration involves");
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
    assert_eq!(tries.errors[0], ActionError::NotInvolved { col: Col::new(1) });
    assert_eq!(tries.errors[1], ActionError::UnknownColumn { col: Col::new(7) });
    assert_eq!(tries.errors[2], ActionError::UnknownColumn { col: Col::new(7) });
    assert!(refused(&tries.errors[3]), "{:?}", tries.errors);
    assert!(refused(&tries.errors[4]), "{:?}", tries.errors);
    assert_eq!(tries.errors.len(), 5);
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

#[test]
fn test_a_panic_in_propagate_or_explain_resumes_after_scip_returns_and_scip_solves_again() {
    // The pigeonhole's conflicts make SCIP ask for explanations, as the conflict fixture shows.
    let program = pigeonhole_fixture();
    for at in ["propagate", "explain"] {
        let mut solver = Solver::new(Scip::with_options(learning()), program.clone()).unwrap();
        let mut handler = Ruled::new(Rule::AllDifferent(vec![0, 1, 2, 3]), Style::Reject, false, 2)
            .propagating(&program, Explain::Complete);
        handler.panics_in = Some(at);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let _ = solver.run().settings(limited()).constraint(&mut handler).solve();
        }));
        let payload = caught.expect_err("the panic propagates");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        assert_eq!(message, Some(format!("panic at {at}").as_str()));
        let best = optimal(solver.run().settings(limited()).solve().unwrap().outcome);
        assert_eq!(best.objective, 8.0, "{at}");
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

    /// For every enforcement style, both lock modes, propagation on and off and every way of explaining, SCIP's optimum
    /// is the reference's over the points the rule accepts, every solution SCIP stores satisfies the rule, and the
    /// handler's model-free checks hold in every call.
    #[test]
    fn test_the_scip_optimum_equals_solve_where_the_rule_holds(
        (program, rule) in small_integer_with_rule(),
        style in prop_oneof![Just(Style::Reject), Just(Style::Constraint), Just(Style::CutOffWhenFixed)],
        only in any::<bool>(),
        propagate in any::<bool>(),
        explain in prop_oneof![Just(Explain::Never), Just(Explain::Unknown), Just(Explain::Complete)],
    ) {
        let mut handler = Ruled::new(rule.clone(), style, only, 3);
        if propagate {
            handler = handler.propagating(&program, explain);
        }
        let (outcome, stored) = solve_ruled(&program, &mut handler);
        let outcome = outcome.map_err(|e| TestCaseError::fail(format!("{e}")))?;
        agrees(&program, &rule, &outcome)?;
        for values in &stored.0 {
            prop_assert!(rule.holds(&rounded(values)), "SCIP stored {values:?}, which breaks {rule:?}");
        }
        if !propagate || explain == Explain::Never {
            prop_assert_eq!(handler.calls.explain, 0);
        }
    }
}
