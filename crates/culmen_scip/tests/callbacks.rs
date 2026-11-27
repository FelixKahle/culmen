// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every event SCIP delivers, through the generic layer and the SCIP contexts: lazy constraints against hand-worked
//! and brute-forced optima, cuts, heuristic solutions, polls, progress, the log, and panics in every plugin.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;
use std::sync::atomic::AtomicBool;

use culmen_core::{
    ActionError, Backend, Callback, Col, Constraint, Context, CutContext, CutHandler, Events, Found, HeuristicContext,
    HeuristicHandler, Heuristics, LazyConstraints, LazyContext, LazyHandler, LinExpr, Linear, LogContext, LogLevel,
    Model, Observer, Outcome, Phase, PollContext, Program, ProgramPlan, ProgramRow, Progress, Script, Settings,
    Solution, SolutionContext, Solver, Step, StopReason, Term, TiSlice, TiVec, UserCuts, Var, relative_gap,
};
use culmen_scip::{Emphasis, LazySource, Scip, ScipEvent, ScipOptions};
use culmen_testing::instances::assignment;
use culmen_testing::reference::{close, solve};
use culmen_testing::strategy::small_lp;
use culmen_testing::tolerance::Tolerances;
use proptest::prelude::*;

/// A solve to a zero gap, so an optimum is the optimum.
fn exact() -> Settings {
    Settings {
        relative_gap: Some(relative_gap!(0.0)),
        time_limit: Some(Duration::from_secs(60)),
        ..Settings::new()
    }
}

/// The optimal solution of an outcome the test expects to be optimal.
fn optimal(outcome: Outcome) -> Solution {
    match outcome {
        Outcome::Optimal(solution) => solution,
        other => panic!("expected an optimum, got {other:?}"),
    }
}

/// Maximize `3x + 2y + z` over binaries. Without the conflict `x + y <= 1`, which only the lazy callback knows, the
/// optimum takes all three, worth 6; with it, `x` and `z`, worth 4.
fn conflict() -> (Program, [Col; 3]) {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    let z = m.add_var(Var::binary());
    m.maximize(3.0 * x + 2.0 * y + z);
    (m.build().unwrap(), [x, y, z])
}

/// Forbids taking both `x` and `y`, on any backend, and keeps every candidate shown after it did.
struct NotBoth {
    x: Col,
    y: Col,
    added: bool,
    after: Vec<TiVec<Col, f64>>,
    calls: usize,
}

impl NotBoth {
    fn new(x: Col, y: Col) -> Self {
        Self {
            x,
            y,
            added: false,
            after: Vec::new(),
            calls: 0,
        }
    }
}

impl<B: LazyConstraints> LazyHandler<B> for NotBoth {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        self.calls += 1;
        assert!(self.calls <= 1000, "the lazy callback was called {} times", self.calls);
        let candidate = ctx.candidate();
        if self.added {
            self.after.push(candidate.to_owned());
        }
        if candidate[self.x] + candidate[self.y] > 1.5 {
            ctx.add_lazy((self.x + self.y).leq(1)).unwrap();
            self.added = true;
        }
    }
}

#[test]
fn test_a_lazy_conflict_moves_the_optimum_and_every_later_candidate_meets_it() {
    let (program, [x, y, z]) = conflict();
    let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
    let mut not_both = NotBoth::new(x, y);
    let best = optimal(
        solver
            .run()
            .settings(exact())
            .lazy(&mut not_both)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, 4.0);
    assert_eq!(
        [best.values[x], best.values[y], best.values[z]].map(f64::round),
        [1.0, 0.0, 1.0]
    );
    assert!(not_both.added, "the unconstrained optimum takes x and y");
    for candidate in &not_both.after {
        assert_eq!(candidate.len(), 3, "a candidate holds the program's columns");
        assert!(candidate[x] + candidate[y] <= 1.0 + 1e-6, "{candidate:?}");
    }
    // Without the callback the conflict is unknown.
    let plain = optimal(solver.run().settings(exact()).solve().unwrap().outcome);
    assert_eq!(plain.objective, 6.0);
}

#[test]
fn test_the_same_lazy_callback_on_the_script_adds_the_row_that_gives_scips_optimum() {
    let (program, [x, y, _]) = conflict();
    let at = Progress::at(Duration::ZERO, Phase::Mip);
    let both = Found {
        progress: at,
        objective: 6.0,
        values: vec![1.0, 1.0, 1.0].into(),
    };
    let one = Found {
        progress: at,
        objective: 4.0,
        values: vec![1.0, 0.0, 1.0].into(),
    };
    let script = Script::new([Step::Lazy(both), Step::Lazy(one)], Outcome::Infeasible);
    let mut scripted = Solver::new(script, program.clone()).unwrap();
    scripted.run().lazy(&mut NotBoth::new(x, y)).solve().unwrap();
    let rows = scripted.backend().record.lazy.clone();
    assert_eq!(rows.len(), 1);
    // The program with the scripted record's rows has SCIP's lazy optimum as its plain optimum.
    let mut plan = ProgramPlan::from(program.clone());
    for row in rows {
        plan.rows.push(ProgramRow {
            bounds: row.bounds,
            terms: row.terms,
            name: None,
        });
    }
    let with_rows = Solver::new(Scip::new(), plan.build().unwrap()).unwrap();
    let mut with_rows = with_rows;
    let plain = optimal(with_rows.run().settings(exact()).solve().unwrap().outcome);
    let lazy = optimal(
        Solver::new(Scip::new(), program)
            .unwrap()
            .run()
            .settings(exact())
            .lazy(&mut NotBoth::new(x, y))
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(plain.objective, lazy.objective);
}

/// Six cities on a plane, in two clusters of three far apart, so the degree constraints alone close two triangles.
const CITIES: [(f64, f64); 6] = [
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.5),
    (10.0, 0.0),
    (11.5, 0.5),
    (10.0, 2.0),
];

fn distance(i: usize, j: usize) -> f64 {
    let ((a, b), (c, d)) = (CITIES[i], CITIES[j]);
    (a - c).hypot(b - d)
}

/// The length of the shortest tour, by trying every order of the cities after the first.
fn shortest_tour() -> f64 {
    fn extend(path: &mut Vec<usize>, left: &mut Vec<usize>, best: &mut f64) {
        if left.is_empty() {
            let last = *path.last().unwrap();
            let length: f64 = path.windows(2).map(|w| distance(w[0], w[1])).sum::<f64>() + distance(last, 0);
            *best = best.min(length);
            return;
        }
        for k in 0..left.len() {
            let city = left.remove(k);
            path.push(city);
            extend(path, left, best);
            path.pop();
            left.insert(k, city);
        }
    }
    let mut best = f64::INFINITY;
    extend(&mut vec![0], &mut (1..CITIES.len()).collect(), &mut best);
    best
}

/// The symmetric TSP over the cities: one binary per edge, every city of degree two, the length minimized; the
/// subtour constraints are left to the lazy callback.
fn tsp() -> (Program, Vec<(usize, usize, Col)>) {
    let n = CITIES.len();
    let mut m = Model::new();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            edges.push((i, j, m.add_var(Var::binary())));
        }
    }
    for city in 0..n {
        let touching = edges
            .iter()
            .filter(|&&(i, j, _)| i == city || j == city)
            .map(|&(_, _, x)| x);
        m.add_constraint(LinExpr::sum(touching).equals(2));
    }
    m.minimize(LinExpr::sum(edges.iter().map(|&(i, j, x)| distance(i, j) * x)));
    (m.build().unwrap(), edges)
}

/// The cities of each connected component of the edges a candidate takes.
fn components(edges: &[(usize, usize, Col)], values: &TiSlice<Col, f64>) -> Vec<Vec<usize>> {
    let n = CITIES.len();
    let mut root: Vec<usize> = (0..n).collect();
    fn find(root: &mut [usize], mut city: usize) -> usize {
        // A root chain is at most as long as the number of cities.
        for _ in 0..root.len() {
            if root[city] == city {
                return city;
            }
            city = root[city];
        }
        panic!("the union-find holds a cycle");
    }
    for &(i, j, x) in edges {
        if values[x] > 0.5 {
            let (a, b) = (find(&mut root, i), find(&mut root, j));
            root[a] = b;
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n];
    for city in 0..n {
        let r = find(&mut root, city);
        groups[r].push(city);
    }
    groups.into_iter().filter(|g| !g.is_empty()).collect()
}

/// Adds, for each subtour a candidate closes, that its cities hold fewer edges than cities; and counts candidates
/// shown after a constraint that violate it.
struct Subtours {
    edges: Vec<(usize, usize, Col)>,
    added: Vec<Vec<usize>>,
    late_violations: usize,
    calls: usize,
}

impl Subtours {
    fn inside(&self, cities: &[usize]) -> impl Iterator<Item = Col> + '_ {
        let set = cities.to_vec();
        self.edges
            .iter()
            .filter(move |&&(i, j, _)| set.contains(&i) && set.contains(&j))
            .map(|&(_, _, x)| x)
    }
}

impl<B: LazyConstraints> LazyHandler<B> for Subtours {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        self.calls += 1;
        assert!(
            self.calls <= 10_000,
            "the lazy callback was called {} times",
            self.calls
        );
        let candidate = ctx.candidate().to_owned();
        for cities in &self.added {
            let held: f64 = self.inside(cities).map(|x| candidate[x]).sum();
            if held > cities.len() as f64 - 1.0 + 1e-6 {
                self.late_violations += 1;
            }
        }
        for cities in components(&self.edges, &candidate) {
            let held: f64 = self.inside(&cities).map(|x| candidate[x]).sum();
            if cities.len() < CITIES.len() && held > cities.len() as f64 - 1.0 + 0.5 {
                let cut = LinExpr::sum(self.inside(&cities)).leq(cities.len() as f64 - 1.0);
                ctx.add_lazy(cut).unwrap();
                self.added.push(cities);
            }
        }
    }
}

#[test]
fn test_lazy_subtour_constraints_find_the_shortest_tour_found_by_brute_force() {
    let (program, edges) = tsp();
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut subtours = Subtours {
        edges: edges.clone(),
        added: Vec::new(),
        late_violations: 0,
        calls: 0,
    };
    let best = optimal(
        solver
            .run()
            .settings(exact())
            .lazy(&mut subtours)
            .solve()
            .unwrap()
            .outcome,
    );
    assert!(
        close(best.objective, shortest_tour(), 1e-6),
        "{} vs {}",
        best.objective,
        shortest_tour()
    );
    assert_eq!(components(&edges, &best.values).len(), 1, "the optimum is one tour");
    assert!(
        !subtours.added.is_empty(),
        "the degree constraints alone admit subtours"
    );
    assert_eq!(subtours.late_violations, 0);
}

#[test]
fn test_a_lazy_constraint_on_a_qp_moves_its_optimum() {
    // The point of x + y <= 2 nearest (2, 2) is (1, 1), at squared distance 2.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(0, 5));
    let y = m.add_var(Var::continuous().bounds(0, 5));
    m.minimize((x - 2) * (x - 2) + (y - 2) * (y - 2));
    let program = m.build().unwrap();

    struct Diagonal(Col, Col, Vec<(f64, f64)>);
    impl LazyHandler<Scip> for Diagonal {
        fn lazy(&mut self, ctx: &mut <Scip as LazyConstraints>::Lazy<'_>) {
            assert!(self.2.len() < 10_000);
            let c = ctx.candidate();
            self.2
                .push((ctx.objective(), (c[self.0] - 2.0).powi(2) + (c[self.1] - 2.0).powi(2)));
            if c[self.0] + c[self.1] > 2.0 + 1e-9 {
                ctx.add_lazy((self.0 + self.1).leq(2)).unwrap();
            }
        }
    }
    let mut diagonal = Diagonal(x, y, Vec::new());
    let best = optimal(
        Solver::new(Scip::new(), program)
            .unwrap()
            .run()
            .settings(exact())
            .lazy(&mut diagonal)
            .solve()
            .unwrap()
            .outcome,
    );
    assert!(close(best.objective, 2.0, 1e-5), "{best:?}");
    assert!(
        (best.values[x] - 1.0).abs() <= 1e-2 && (best.values[y] - 1.0).abs() <= 1e-2,
        "{best:?}"
    );
    // A candidate's objective is the program's at the candidate, the quadratic part included.
    for &(objective, value) in &diagonal.2 {
        assert!(close(objective, value, 1e-6), "{objective} vs {value}");
    }
}

/// Every lazy candidate's source, for one backend's own contexts.
#[derive(Default)]
struct Sources(Vec<LazySource>);

impl LazyHandler<Scip> for Sources {
    fn lazy(&mut self, ctx: &mut <Scip as LazyConstraints>::Lazy<'_>) {
        assert!(self.0.len() < 10_000);
        self.0.push(ctx.source());
    }
}

#[test]
fn test_lazy_candidates_come_from_checked_solutions_and_node_lps() {
    let (program, _) = conflict();
    let mut sources = Sources::default();
    Solver::new(Scip::new(), program)
        .unwrap()
        .run()
        .settings(exact())
        .lazy(&mut sources)
        .solve()
        .unwrap();
    assert!(sources.0.contains(&LazySource::Check), "{:?}", sources.0);
}

#[test]
fn test_an_invalid_lazy_constraint_is_refused_typed() {
    let (program, [x, _, _]) = conflict();
    struct Bad(Col, Vec<ActionError>);
    impl LazyHandler<Scip> for Bad {
        fn lazy(&mut self, ctx: &mut <Scip as LazyConstraints>::Lazy<'_>) {
            if self.1.is_empty() {
                let x = self.0;
                let rows = [
                    Col::new(7).leq(1),
                    (1e25 * x).leq(1),
                    x.leq(1e25),
                    // Each below SCIP's infinity of 1e20, but 1.2e20 added up, as SCIP adds them.
                    (6e19 * x + 6e19 * x).leq(1),
                ];
                for row in rows {
                    self.1.push(
                        ctx.add_lazy(row)
                            .err()
                            .unwrap_or(ActionError::Refused("taken".to_owned())),
                    );
                }
            }
        }
    }
    let mut bad = Bad(x, Vec::new());
    let best = optimal(
        Solver::new(Scip::new(), program)
            .unwrap()
            .run()
            .settings(exact())
            .lazy(&mut bad)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, 6.0, "nothing refused is added");
    assert_eq!(bad.1[0], ActionError::UnknownCol { col: Col::new(7) });
    assert!(
        matches!(&bad.1[1], ActionError::Refused(reason) if reason.contains("infinite")),
        "{:?}",
        bad.1
    );
    assert!(
        matches!(&bad.1[2], ActionError::Refused(reason) if reason.contains("infinite")),
        "{:?}",
        bad.1
    );
    assert!(
        matches!(&bad.1[3], ActionError::Refused(reason) if reason.contains("infinite")),
        "{:?}",
        bad.1
    );
}

/// The knapsack of the cutting-planes example, its items whole: value and weight of each.
const ITEMS: [(f64, f64); 8] = [
    (16.0, 9.0),
    (13.0, 7.0),
    (9.0, 5.0),
    (8.0, 4.0),
    (7.0, 4.0),
    (5.0, 3.0),
    (4.0, 2.0),
    (2.0, 1.0),
];
const CAPACITY: f64 = 22.0;

fn knapsack() -> (Program, Vec<Col>) {
    let mut m = Model::new();
    let take = m.add_vars(ITEMS.len(), |_| Var::binary());
    m.add_constraint(LinExpr::sum(ITEMS.iter().zip(&take).map(|(&(_, w), &x)| w * x)).leq(CAPACITY));
    m.maximize(LinExpr::sum(ITEMS.iter().zip(&take).map(|(&(v, _), &x)| v * x)));
    (m.build().unwrap(), take)
}

/// The knapsack's optimal set of items, as bits, and its value, by trying every subset of the items.
fn best_set() -> (u32, f64) {
    let mut best = (0, 0.0);
    for set in 0u32..1 << ITEMS.len() {
        let (value, weight) = ITEMS
            .iter()
            .enumerate()
            .filter(|&(i, _)| set & (1 << i) != 0)
            .fold((0.0, 0.0), |(v, w), (_, &(vi, wi))| (v + vi, w + wi));
        if weight <= CAPACITY && value > best.1 {
            best = (set, value);
        }
    }
    best
}

/// The knapsack's optimum.
fn best_subset() -> f64 {
    best_set().1
}

/// A cover the point violates: items by how little of each the point leaves out per unit of weight, until their
/// weight exceeds the capacity; as in the cutting-planes example.
fn violated_cover(take: &[Col], values: &TiSlice<Col, f64>) -> Option<Vec<Col>> {
    let left_out = |x: &Col| (1.0 - values[*x]) / ITEMS[x.index()].1;
    let mut order = take.to_vec();
    order.sort_by(|a, b| left_out(a).total_cmp(&left_out(b)));
    let mut weight = 0.0;
    let mut cover = Vec::new();
    for x in order {
        if weight > CAPACITY {
            break;
        }
        weight += ITEMS[x.index()].1;
        cover.push(x);
    }
    let taken: f64 = cover.iter().map(|&x| values[x]).sum();
    (weight > CAPACITY && taken > cover.len() as f64 - 1.0 + 1e-6).then_some(cover)
}

/// Adds a violated cover inequality at each node relaxation.
struct Covers {
    take: Vec<Col>,
    accepted: usize,
    calls: usize,
}

impl<B: UserCuts> CutHandler<B> for Covers {
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        self.calls += 1;
        assert!(self.calls <= 10_000, "the cut callback was called {} times", self.calls);
        if let Some(cover) = violated_cover(&self.take, ctx.relaxation()) {
            let fit = cover.len() as f64 - 1.0;
            if ctx.add_cut(LinExpr::sum(cover).leq(fit)).is_ok() {
                self.accepted += 1;
            }
        }
    }
}

/// SCIP with its own heuristics off and without presolve, so its root LP is the knapsack's relaxation.
fn bare() -> Scip {
    Scip::with_options(ScipOptions {
        heuristics: Some(Emphasis::Off),
        ..ScipOptions::default()
    })
}

fn without_presolve() -> Settings {
    Settings {
        presolve: Some(false),
        ..exact()
    }
}

#[test]
fn test_cover_cuts_are_taken_and_leave_the_knapsack_optimum() {
    let (program, take) = knapsack();
    let mut solver = Solver::new(bare(), program).unwrap();
    let mut covers = Covers {
        take,
        accepted: 0,
        calls: 0,
    };
    let best = optimal(
        solver
            .run()
            .settings(without_presolve())
            .cut(&mut covers)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, best_subset());
    assert!(covers.accepted >= 1, "the root relaxation takes a fraction of an item");
    let plain = optimal(solver.run().settings(without_presolve()).solve().unwrap().outcome);
    assert_eq!(plain.objective, best.objective);
}

/// What a heuristic handler supplies at its first call, and what came of it.
#[derive(Default)]
struct Supplier {
    full: Option<Vec<f64>>,
    results: Vec<Result<(), ActionError>>,
    calls: usize,
}

impl HeuristicHandler<Scip> for Supplier {
    fn heuristic(&mut self, ctx: &mut <Scip as Heuristics>::Heuristic<'_>) {
        self.calls += 1;
        assert!(
            self.calls <= 10_000,
            "the heuristic handler was called {} times",
            self.calls
        );
        if let Some(values) = self.full.take() {
            self.results.push(ctx.supply(TiSlice::from_ref(values.as_slice())));
        }
    }
}

/// Every incumbent's objective.
#[derive(Default)]
struct Incumbents(Vec<f64>);

impl Callback<Scip> for Incumbents {
    fn events(&self) -> Events {
        Events::INCUMBENT
    }
    fn incumbent(&mut self, ctx: &mut <Scip as Backend>::Solution<'_>) {
        self.0.push(ctx.objective());
    }
}

#[test]
fn test_a_supplied_optimum_becomes_the_first_incumbent() {
    let (program, take) = knapsack();
    let mut solver = Solver::new(bare(), program).unwrap();
    let (set, value) = best_set();
    let chosen: Vec<f64> = (0..take.len()).map(|i| f64::from(set >> i & 1)).collect();
    let mut supplier = Supplier {
        full: Some(chosen),
        ..Supplier::default()
    };
    let mut incumbents = Incumbents::default();
    let best = optimal(
        solver
            .run()
            .settings(without_presolve())
            .heuristic(&mut supplier)
            .callback(&mut incumbents)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(supplier.results, [Ok(())]);
    assert_eq!(incumbents.0.first(), Some(&value), "{:?}", incumbents.0);
    assert_eq!(best.objective, value);
}

#[test]
fn test_an_infeasible_or_wrong_solution_is_refused() {
    // SCIP's heuristic context has no partial solution to supply: SCIP completes one only as a start.
    let (program, take) = knapsack();
    let mut solver = Solver::new(bare(), program).unwrap();
    let mut infeasible = Supplier {
        full: Some(vec![1.0; take.len()]),
        ..Supplier::default()
    };
    let mut incumbents = Incumbents::default();
    let best = optimal(
        solver
            .run()
            .settings(without_presolve())
            .heuristic(&mut infeasible)
            .callback(&mut incumbents)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, best_subset());
    assert!(
        matches!(&infeasible.results[0], Err(ActionError::Refused(reason)) if reason.contains("infeasible")),
        "{:?}",
        infeasible.results
    );
    assert!(incumbents.0.iter().all(|&o| o <= best_subset()), "{:?}", incumbents.0);
    let mut short = Supplier {
        full: Some(vec![1.0; 3]),
        ..Supplier::default()
    };
    solver
        .run()
        .settings(without_presolve())
        .heuristic(&mut short)
        .solve()
        .unwrap();
    assert_eq!(short.results, [Err(ActionError::Length { expected: 8, found: 3 })]);
}

/// Supplies, once, a solution taking all three items of the conflict.
#[derive(Default)]
struct SupplyAll(Vec<Result<(), ActionError>>);

impl HeuristicHandler<Scip> for SupplyAll {
    fn heuristic(&mut self, ctx: &mut <Scip as Heuristics>::Heuristic<'_>) {
        if self.0.is_empty() {
            self.0.push(ctx.supply(TiSlice::from_ref(&[1.0, 1.0, 1.0][..])));
        }
    }
}

#[test]
fn test_a_supplied_solution_meets_the_lazy_handler_before_scip_keeps_it() {
    let (program, [x, y, _]) = conflict();
    let (mut not_both, mut supply, mut incumbents) = (NotBoth::new(x, y), SupplyAll::default(), Incumbents::default());
    let best = optimal(
        Solver::new(bare(), program)
            .unwrap()
            .run()
            .settings(exact())
            .lazy(&mut not_both)
            .heuristic(&mut supply)
            .callback(&mut incumbents)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(
        supply.0,
        [Ok(())],
        "a run with a lazy-constraint handler tries the solution after the heuristic handler"
    );
    assert!(not_both.added, "the lazy-constraint handler saw the supplied solution");
    assert!(incumbents.0.iter().all(|&o| o <= 4.0), "{:?}", incumbents.0);
    assert_eq!(best.objective, 4.0);
}

/// Interrupts at the first poll that satisfies a test, and keeps every poll's event and progress.
struct InterruptWhen<F> {
    when: F,
    polls: Vec<(ScipEvent, Progress)>,
}

impl<F: FnMut(&Progress) -> bool + Send> Callback<Scip> for InterruptWhen<F> {
    fn events(&self) -> Events {
        Events::POLL
    }
    fn poll(&mut self, ctx: &mut <Scip as Backend>::Poll<'_>) {
        self.polls.push((ctx.event(), ctx.progress()));
        if (self.when)(&ctx.progress()) {
            ctx.interrupt();
        }
    }
}

#[test]
fn test_an_interrupt_at_the_first_poll_stops_the_solve() {
    let program = assignment(30, 12, 7, 0.0).program;
    let mut solver = Solver::new(Scip::new(), program).unwrap();
    let mut first = InterruptWhen {
        when: |_: &Progress| true,
        polls: Vec::new(),
    };
    let solved = solver.run().settings(exact()).callback(&mut first).solve().unwrap();
    assert!(
        matches!(
            solved.outcome,
            Outcome::Stopped {
                reason: StopReason::Interrupted,
                ..
            }
        ),
        "{:?}",
        solved.outcome
    );
    assert_eq!(first.polls[0].0, ScipEvent::PresolveRound, "{:?}", first.polls);
    assert_eq!(first.polls[0].1.phase, Phase::Presolve);
}

#[test]
fn test_an_interrupt_once_there_is_an_incumbent_keeps_it() {
    let program = assignment(60, 30, 12_345, 0.0).program;
    let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
    let mut found = InterruptWhen {
        when: |p: &Progress| p.objective.is_some(),
        polls: Vec::new(),
    };
    let solved = solver.run().settings(exact()).callback(&mut found).solve().unwrap();
    let Outcome::Stopped {
        reason: StopReason::Interrupted,
        best: Some(best),
        ..
    } = solved.outcome
    else {
        panic!(
            "expected an interrupted solve with an incumbent, got {:?}",
            solved.outcome
        );
    };
    assert!(program.max_violation(&best.values) <= 1e-6);
    assert!(close(program.objective_value(&best.values), best.objective, 1e-9));
}

#[test]
fn test_a_stop_request_set_before_the_solve_stops_it_at_the_first_poll() {
    let (program, _) = knapsack();
    let stop = AtomicBool::new(true);
    let solved = Solver::new(Scip::new(), program)
        .unwrap()
        .run()
        .stop(&stop)
        .solve()
        .unwrap();
    assert!(
        matches!(
            solved.outcome,
            Outcome::Stopped {
                reason: StopReason::Interrupted,
                ..
            }
        ),
        "{:?}",
        solved.outcome
    );
}

#[test]
fn test_a_pure_lp_is_polled_at_its_one_node() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(0, 4));
    let y = m.add_var(Var::continuous().bounds(0, 4));
    m.add_constraint((x + 2 * y).leq(6));
    m.maximize(x + y);
    let mut polls = InterruptWhen {
        when: |_: &Progress| false,
        polls: Vec::new(),
    };
    // Presolve alone would solve so small an LP, before any node.
    let best = optimal(
        Solver::new(Scip::new(), m.build().unwrap())
            .unwrap()
            .run()
            .settings(without_presolve())
            .callback(&mut polls)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, 5.0);
    let events: Vec<ScipEvent> = polls.polls.iter().map(|&(e, _)| e).collect();
    assert!(events.contains(&ScipEvent::NodeFocused), "{events:?}");
    assert!(events.contains(&ScipEvent::NodeSolved), "{events:?}");
    assert!(
        polls.polls.iter().all(|(_, p)| p.phase != Phase::Mip),
        "an LP has no branch and bound"
    );
}

/// Every progress report.
#[derive(Default)]
struct Reports(Vec<Progress>);

impl Callback<Scip> for Reports {
    fn events(&self) -> Events {
        Events::PROGRESS
    }
    fn progress(&mut self, ctx: &mut <Scip as Backend>::Progress<'_>) {
        self.0.push(ctx.progress());
    }
}

#[test]
fn test_progress_moves_one_way_and_the_bound_stays_below_the_incumbent() {
    let program = assignment(30, 12, 7, 0.0).program;
    let mut reports = Reports::default();
    Solver::new(Scip::new(), program)
        .unwrap()
        .run()
        .time_limit(Duration::from_secs(5))
        .callback(&mut reports)
        .solve()
        .unwrap();
    let reports = reports.0;
    assert!(reports.iter().any(|p| p.phase == Phase::Presolve));
    assert!(reports.iter().any(|p| p.phase == Phase::Mip));
    for pair in reports.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(b.elapsed >= a.elapsed);
        let rises = |x: Option<u64>, y: Option<u64>| matches!((x, y), (Some(x), Some(y)) if y < x);
        assert!(!rises(a.nodes, b.nodes), "{a:?} then {b:?}");
        assert!(!rises(a.iterations, b.iterations), "{a:?} then {b:?}");
        // A minimization's incumbent only falls and its bound only rises.
        if let (Some(x), Some(y)) = (a.objective, b.objective) {
            assert!(y <= x + 1e-9, "{a:?} then {b:?}");
        }
        if let (Some(x), Some(y)) = (a.bound, b.bound) {
            assert!(y >= x - 1e-6 * x.abs().max(1.0), "{a:?} then {b:?}");
        }
    }
    for p in &reports {
        if let (Some(objective), Some(bound)) = (p.objective, p.bound) {
            assert!(bound <= objective + 1e-6, "{p:?}");
        }
    }
}

/// Every log chunk, and the lines an observer is told.
#[derive(Default)]
struct Lines {
    chunks: Vec<(LogLevel, String)>,
}

impl Callback<Scip> for Lines {
    fn events(&self) -> Events {
        Events::LOG
    }
    fn log(&mut self, ctx: &mut <Scip as Backend>::Log<'_>) {
        self.chunks.push((ctx.level(), ctx.text().to_owned()));
    }
}

#[derive(Default)]
struct Told(Vec<String>);

impl Observer for Told {
    fn log(&mut self, line: &culmen_core::LogLine<'_>) {
        self.0.push(line.text.to_owned());
    }
}

/// Solves the knapsack in a child process with the console on or off, and returns what it wrote to standard output.
fn child_stdout(test: &str, console: bool) -> String {
    let exe = std::env::current_exe().unwrap();
    let output = Command::new(exe)
        .args(["--exact", test, "--nocapture"])
        .env("CULMEN_SCIP_LOG_CHILD", if console { "console" } else { "quiet" })
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("CHILD-DONE"), "{stdout}");
    stdout
}

#[test]
fn test_log_lines_arrive_whole_and_reach_stdout_only_with_the_console() {
    const NAME: &str = "test_log_lines_arrive_whole_and_reach_stdout_only_with_the_console";
    if let Some(mode) = std::env::var_os("CULMEN_SCIP_LOG_CHILD") {
        let (program, _) = knapsack();
        let mut solver = Solver::new(Scip::new(), program).unwrap();
        let mut lines = Lines::default();
        let mut told = Told::default();
        solver
            .run()
            .settings(Settings {
                console: mode == "console",
                ..Settings::new()
            })
            .observer(&mut told)
            .callback(&mut lines)
            .solve()
            .unwrap();
        assert!(!lines.chunks.is_empty());
        assert!(
            lines.chunks.iter().all(|(_, text)| text.ends_with('\n')),
            "{:?}",
            lines.chunks
        );
        assert!(told.0.iter().any(|l| l.starts_with("SCIP Status")), "{:?}", told.0);
        assert!(told.0.iter().all(|l| !l.contains('\n')));
        println!("CHILD-DONE {}", told.0.len());
        return;
    }
    let quiet = child_stdout(NAME, false);
    assert!(
        !quiet.contains("SCIP Status"),
        "SCIP wrote to standard output:\n{quiet}"
    );
    let console = child_stdout(NAME, true);
    assert!(
        console.contains("SCIP Status"),
        "the console shows SCIP's log:\n{console}"
    );
}

/// Panics at the first event of one kind.
struct PanicAt(Events);

impl PanicAt {
    fn fire(&self) {
        panic!("panic at {}", self.0);
    }
}

impl Callback<Scip> for PanicAt {
    fn events(&self) -> Events {
        self.0
    }
    fn log(&mut self, _: &mut <Scip as Backend>::Log<'_>) {
        self.fire();
    }
    fn progress(&mut self, _: &mut <Scip as Backend>::Progress<'_>) {
        self.fire();
    }
    fn poll(&mut self, _: &mut <Scip as Backend>::Poll<'_>) {
        self.fire();
    }
    fn feasible(&mut self, _: &mut <Scip as Backend>::Solution<'_>) {
        self.fire();
    }
    fn incumbent(&mut self, _: &mut <Scip as Backend>::Solution<'_>) {
        self.fire();
    }
}

/// Panics at its first call as a search handler, naming its kind.
struct PanicIn(&'static str);

impl LazyHandler<Scip> for PanicIn {
    fn lazy(&mut self, _: &mut <Scip as LazyConstraints>::Lazy<'_>) {
        panic!("panic at {}", self.0);
    }
}

impl CutHandler<Scip> for PanicIn {
    fn cut(&mut self, _: &mut <Scip as UserCuts>::Cut<'_>) {
        panic!("panic at {}", self.0);
    }
}

impl HeuristicHandler<Scip> for PanicIn {
    fn heuristic(&mut self, _: &mut <Scip as Heuristics>::Heuristic<'_>) {
        panic!("panic at {}", self.0);
    }
}

/// Solves `program` with the hooks `solve` registers, expects the panic `expected` out of the solve, and then a plain
/// solve of the same solver to the knapsack's optimum.
fn panics_out(program: &Program, expected: &str, solve: impl FnOnce(&mut Solver<Scip>)) {
    let mut solver = Solver::new(bare(), program.clone()).unwrap();
    let caught = catch_unwind(AssertUnwindSafe(|| solve(&mut solver)));
    let payload = caught.expect_err("the panic propagates");
    assert_eq!(payload.downcast_ref::<String>().map(String::as_str), Some(expected));
    let best = optimal(solver.run().settings(without_presolve()).solve().unwrap().outcome);
    assert_eq!(best.objective, best_subset(), "{expected}");
}

#[test]
fn test_a_panic_in_every_plugin_panics_out_of_solve_and_leaves_scip_usable() {
    let (program, _) = knapsack();
    for event in Scip::EVENTS {
        panics_out(&program, &format!("panic at {event}"), |solver| {
            let _ = solver
                .run()
                .settings(without_presolve())
                .callback(&mut PanicAt(event))
                .solve();
        });
    }
    panics_out(&program, "panic at lazy", |solver| {
        let _ = solver
            .run()
            .settings(without_presolve())
            .lazy(&mut PanicIn("lazy"))
            .solve();
    });
    panics_out(&program, "panic at cut", |solver| {
        let _ = solver
            .run()
            .settings(without_presolve())
            .cut(&mut PanicIn("cut"))
            .solve();
    });
    panics_out(&program, "panic at heuristic", |solver| {
        let _ = solver
            .run()
            .settings(without_presolve())
            .heuristic(&mut PanicIn("heuristic"))
            .solve();
    });
}

/// Records which events arrive.
#[derive(Default)]
struct Seen(Events);

impl Callback<Scip> for Seen {
    fn events(&self) -> Events {
        Events::ALL
    }
    fn log(&mut self, _: &mut <Scip as Backend>::Log<'_>) {
        self.0 |= Events::LOG;
    }
    fn progress(&mut self, _: &mut <Scip as Backend>::Progress<'_>) {
        self.0 |= Events::PROGRESS;
    }
    fn poll(&mut self, _: &mut <Scip as Backend>::Poll<'_>) {
        self.0 |= Events::POLL;
    }
    fn feasible(&mut self, _: &mut <Scip as Backend>::Solution<'_>) {
        self.0 |= Events::FEASIBLE;
    }
    fn incumbent(&mut self, _: &mut <Scip as Backend>::Solution<'_>) {
        self.0 |= Events::INCUMBENT;
    }
}

/// Counts its calls as a search handler, adding nothing.
#[derive(Default)]
struct Hit(usize);

impl Hit {
    fn count(&mut self) {
        self.0 += 1;
        assert!(self.0 <= 10_000, "the handler was called {} times", self.0);
    }
}

impl LazyHandler<Scip> for Hit {
    fn lazy(&mut self, _: &mut <Scip as LazyConstraints>::Lazy<'_>) {
        self.count();
    }
}

impl CutHandler<Scip> for Hit {
    fn cut(&mut self, _: &mut <Scip as UserCuts>::Cut<'_>) {
        self.count();
    }
}

impl HeuristicHandler<Scip> for Hit {
    fn heuristic(&mut self, _: &mut <Scip as Heuristics>::Heuristic<'_>) {
        self.count();
    }
}

#[test]
fn test_scips_events_are_exactly_the_events_it_delivers_and_every_handler_runs() {
    assert_eq!(Scip::EVENTS, Events::ALL - Events::NATIVE);
    let (program, _) = knapsack();
    let mut seen = Seen::default();
    let (mut lazy, mut cut, mut heuristic) = (Hit::default(), Hit::default(), Hit::default());
    Solver::new(Scip::new(), program)
        .unwrap()
        .run()
        .settings(without_presolve())
        .callback(&mut seen)
        .lazy(&mut lazy)
        .cut(&mut cut)
        .heuristic(&mut heuristic)
        .solve()
        .unwrap();
    assert_eq!(seen.0, Scip::EVENTS);
    assert!(
        lazy.0 > 0 && cut.0 > 0 && heuristic.0 > 0,
        "{} {} {}",
        lazy.0,
        cut.0,
        heuristic.0
    );
}

/// Asks for nothing and fails on any event.
struct Nothing;

impl Callback<Scip> for Nothing {
    fn events(&self) -> Events {
        Events::NONE
    }
    fn log(&mut self, _: &mut <Scip as Backend>::Log<'_>) {
        unreachable!("not asked for");
    }
    fn poll(&mut self, _: &mut <Scip as Backend>::Poll<'_>) {
        unreachable!("not asked for");
    }
}

#[test]
fn test_events_not_asked_for_are_not_delivered() {
    let (program, _) = knapsack();
    let best = optimal(
        Solver::new(Scip::new(), program)
            .unwrap()
            .run()
            .settings(without_presolve())
            .callback(&mut Nothing)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, best_subset());
}

/// Counts the polls of a solve, asking for every event.
#[derive(Default)]
struct Everything(usize);

impl Callback<Scip> for Everything {
    fn events(&self) -> Events {
        Events::ALL
    }
    fn poll(&mut self, _: &mut <Scip as Backend>::Poll<'_>) {
        self.0 += 1;
    }
}

#[test]
fn test_a_callback_asking_for_every_event_solves_a_qp_without_pricing() {
    // The point of x + y >= 1 nearest the origin is (1/2, 1/2), at squared distance 1/2.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(0, 5));
    let y = m.add_var(Var::continuous().bounds(0, 5));
    m.add_constraint((x + y).geq(1));
    m.minimize(x * x + y * y);
    let mut everything = Everything::default();
    let solved = Solver::new(Scip::new(), m.build().unwrap())
        .unwrap()
        .run()
        .settings(exact())
        .callback(&mut everything)
        .solve();
    let best = optimal(solved.unwrap().outcome);
    assert!(close(best.objective, 0.5, 1e-5), "{best:?}");
    assert!(everything.0 > 0);
}

/// Every solution reported, with its objective.
#[derive(Default)]
struct Solutions {
    feasible: Vec<(f64, TiVec<Col, f64>)>,
    incumbents: Vec<f64>,
}

impl Callback<Scip> for Solutions {
    fn events(&self) -> Events {
        Events::FEASIBLE | Events::INCUMBENT
    }
    fn feasible(&mut self, ctx: &mut <Scip as Backend>::Solution<'_>) {
        self.feasible.push((ctx.objective(), ctx.values().to_owned()));
    }
    fn incumbent(&mut self, ctx: &mut <Scip as Backend>::Solution<'_>) {
        self.incumbents.push(ctx.objective());
    }
}

#[test]
fn test_solutions_arrive_in_the_programs_columns_with_the_offset_and_improve() {
    let mut m = Model::new();
    let take = m.add_vars(ITEMS.len(), |_| Var::binary());
    m.add_constraint(LinExpr::sum(ITEMS.iter().zip(&take).map(|(&(_, w), &x)| w * x)).leq(CAPACITY));
    m.maximize(LinExpr::sum(ITEMS.iter().zip(&take).map(|(&(v, _), &x)| v * x)) + 7);
    let program = m.build().unwrap();
    let mut solutions = Solutions::default();
    let best = optimal(
        Solver::new(Scip::new(), program.clone())
            .unwrap()
            .run()
            .settings(exact())
            .callback(&mut solutions)
            .solve()
            .unwrap()
            .outcome,
    );
    assert_eq!(best.objective, best_subset() + 7.0);
    assert!(!solutions.feasible.is_empty());
    for (objective, values) in &solutions.feasible {
        assert_eq!(values.len(), ITEMS.len());
        assert!(program.max_violation(values) <= 1e-6, "{values:?}");
        assert!(close(program.objective_value(values), *objective, 1e-9));
    }
    assert!(
        solutions.incumbents.windows(2).all(|w| w[1] > w[0]),
        "{:?}",
        solutions.incumbents
    );
    assert_eq!(solutions.incumbents.last(), Some(&best.objective));
}

/// A program row as the constraint a callback adds.
fn constraint(row: &ProgramRow) -> Constraint {
    let terms = row.terms.iter().map(|&(col, coefficient)| Term { col, coefficient });
    Constraint {
        expr: LinExpr {
            terms: terms.collect(),
            constant: 0.0,
        },
        lower: row.bounds.lower(),
        upper: row.bounds.upper(),
    }
}

/// Adds each withheld row when a candidate violates it.
struct Withheld {
    rows: Vec<ProgramRow>,
    calls: usize,
}

impl<B: LazyConstraints> LazyHandler<B> for Withheld {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        self.calls += 1;
        assert!(
            self.calls <= 10_000,
            "the lazy callback was called {} times",
            self.calls
        );
        let candidate = ctx.candidate().to_owned();
        for row in &self.rows {
            let activity: f64 = row.terms.iter().map(|&(col, a)| a * candidate[col]).sum();
            if row.bounds.violation(activity) > 1e-9 {
                ctx.add_lazy(constraint(row)).unwrap();
            }
        }
    }
}

/// Adds the same cuts at every node relaxation.
struct Implied {
    cuts: Vec<Constraint>,
    calls: usize,
}

impl<B: UserCuts> CutHandler<B> for Implied {
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        self.calls += 1;
        assert!(self.calls <= 10_000, "the cut callback was called {} times", self.calls);
        for cut in &self.cuts {
            ctx.add_cut(cut.clone()).unwrap();
        }
    }
}

/// Cuts every feasible point of the program satisfies: the sum of its rows with an upper bound is at most the sum of
/// those bounds, and the sum of its columns lies between the sums of their bounds.
fn implied_cuts(program: &Program) -> Vec<Constraint> {
    let mut cuts = Vec::new();
    let capped: Vec<ProgramRow> = ProgramPlan::from(program.clone())
        .rows
        .into_iter()
        .filter(|row| row.bounds.upper().is_finite())
        .collect();
    if !capped.is_empty() {
        let sum = ProgramRow {
            bounds: culmen_core::Bounds::at_most(capped.iter().map(|row| row.bounds.upper()).sum()).unwrap(),
            terms: capped.iter().flat_map(|row| row.terms.iter().copied()).collect(),
            name: None,
        };
        cuts.push(constraint(&sum));
    }
    let columns = program.columns();
    let lower: f64 = columns.iter().map(|c| c.bounds.lower()).sum();
    let upper: f64 = columns.iter().map(|c| c.bounds.upper()).sum();
    cuts.push(LinExpr::sum(program.cols()).between(lower, upper));
    cuts
}

/// Whether SCIP's outcome is the exact reference's, its optimum in the window SCIP's tolerances leave around the exact
/// objective and its point meeting the full program to those tolerances.
fn agrees(program: &Program, outcome: &Outcome) -> Result<(), TestCaseError> {
    let expected = solve(program);
    match (Tolerances::SCIP.window(program, &expected), outcome) {
        (None, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded) => Ok(()),
        (Some(window), Outcome::Optimal(best)) => {
            prop_assert!(
                window.contains(best.objective),
                "SCIP {} vs exact {expected:?}, window {window:?}",
                best.objective
            );
            let stray = Tolerances::SCIP.stray(program, &best.values);
            prop_assert!(stray.is_none(), "{stray:?} {best:?}");
            Ok(())
        }
        (_, got) => Err(TestCaseError::fail(format!("exact {expected:?}, SCIP {got:?}"))),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// A program whose rows SCIP learns only through the lazy callback, each when a candidate violates it, has the
    /// optimum of the full program, found by exact enumeration, or is infeasible with it.
    #[test]
    fn test_rows_withheld_and_added_lazily_give_the_optimum_of_the_full_program(
        (program, withheld) in small_lp().prop_flat_map(|program| {
            let rows = program.num_rows();
            (Just(program), prop::collection::vec(any::<bool>(), rows))
        })
    ) {
        let mut plan = ProgramPlan::from(program.clone());
        let rows: Vec<ProgramRow> = core::mem::take(&mut plan.rows).into_iter().collect();
        let (held, kept): (Vec<_>, Vec<_>) = rows.into_iter().zip(withheld).partition(|&(_, held)| held);
        plan.rows = kept.into_iter().map(|(row, _)| row).collect();
        let mut withheld = Withheld { rows: held.into_iter().map(|(row, _)| row).collect(), calls: 0 };
        let solved = Solver::new(Scip::new(), plan.build().unwrap())
            .unwrap()
            .run().settings(exact()).lazy(&mut withheld).solve()
            .unwrap();
        agrees(&program, &solved.outcome)?;
    }

    /// Cuts every feasible point satisfies leave the exact optimum.
    #[test]
    fn test_cuts_implied_by_the_program_leave_its_optimum(program in small_lp()) {
        let mut implied = Implied { cuts: implied_cuts(&program), calls: 0 };
        let solved = Solver::new(Scip::new(), program.clone())
            .unwrap()
            .run().settings(exact()).cut(&mut implied).solve()
            .unwrap();
        agrees(&program, &solved.outcome)?;
    }
}
