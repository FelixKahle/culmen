// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-worked outcomes of the SCIP backend: every ending a solve can have, and programs at the edges of what a
//! program holds.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;

use culmen_core::{
    Bounds, Col, LinExpr, Linear, Model, ModelClass, Outcome, Program, ProgramColumn, ProgramPlan, Sense, Settings,
    Solution, Solved, Solver, StopReason, Var, bounds, feasibility_tolerance, relative_gap, solutions,
};
use culmen_scip::{Emphasis, Scip, ScipOptions, ScipStatus, Status};
use culmen_testing::instances::{flat_costs, infeasible_by_a_hair};

fn solve(program: Program) -> Solved<ScipStatus> {
    solve_with(Scip::new(), program, Settings::new())
}

fn solve_with(scip: Scip, program: Program, settings: Settings) -> Solved<ScipStatus> {
    Solver::new(scip, program)
        .unwrap()
        .run()
        .settings(settings)
        .solve()
        .unwrap()
}

/// Solves from `start` when one is given.
fn solve_from(scip: Scip, program: Program, settings: Settings, start: Option<Vec<f64>>) -> Solved<ScipStatus> {
    let mut solver = Solver::new(scip, program).unwrap();
    let run = solver.run().settings(settings);
    match start {
        Some(start) => run.start(start).solve(),
        None => run.solve(),
    }
    .unwrap()
}

fn optimum(solved: &Solved<ScipStatus>) -> &Solution {
    match &solved.outcome {
        Outcome::Optimal(solution) => solution,
        other => panic!("expected an optimum, got {other:?} ({:?})", solved.status),
    }
}

fn assert_close(got: f64, want: f64) {
    assert!((got - want).abs() <= 1e-6, "{got} is not {want}");
}

#[test]
fn test_an_lp_maximum_includes_the_offset_and_reports_its_work() {
    // Maximize 3x + 2y + 10 subject to x + y <= 4, x + 3y <= 6, x, y >= 0: x = 4, y = 0, objective 22.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + y).leq(4.0));
    m.add_constraint((x + 3.0 * y).leq(6.0));
    m.maximize(3.0 * x + 2.0 * y + 10);
    let solved = solve(m.build().unwrap());
    let best = optimum(&solved);
    assert_close(best.objective, 22.0);
    assert_close(best.values[x], 4.0);
    assert_close(best.values[y], 0.0);
    assert_eq!(solved.status.status, Status::Optimal);
    assert!(solved.status.solutions_found >= 1);
    // SCIP solves an LP at the root of its tree.
    assert_eq!(solved.nodes, Some(1));
    assert!(solved.iterations.is_some());
}

#[test]
fn test_a_milp_minimum_is_whole() {
    // Minimize x + y subject to 2x + 3y >= 7.5, x, y integer and at least zero: (0, 3) costs 3, (2, 2) costs 4, and
    // (1, 2) misses by half; the optimum is 3.
    let mut m = Model::new();
    let x = m.add_var(Var::integer().lower(0));
    let y = m.add_var(Var::integer().lower(0));
    m.add_constraint((2.0 * x + 3.0 * y).geq(7.5));
    m.minimize(x + y);
    let solved = solve(m.build().unwrap());
    let best = optimum(&solved);
    assert_close(best.objective, 3.0);
    assert_close(best.values[x], 0.0);
    assert_close(best.values[y], 3.0);
}

fn infeasible(integer: bool) -> Model {
    let mut m = Model::new();
    let kind = if integer { Var::integer() } else { Var::continuous() };
    let x = m.add_var(kind.clone().upper(5.0));
    let y = m.add_var(kind.upper(5.0));
    m.add_constraint((x + y).geq(4.0));
    m.add_constraint((x + y).leq(3.0));
    m.minimize(x);
    m
}

#[test]
fn test_an_infeasible_lp_and_milp_are_infeasible() {
    for integer in [false, true] {
        let solved = solve(infeasible(integer).build().unwrap());
        assert!(
            matches!(solved.outcome, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded),
            "integer {integer}: {solved:?}"
        );
        assert!(matches!(
            solved.status.status,
            Status::Infeasible | Status::InfeasibleOrUnbounded
        ));
        assert_eq!(solved.outcome.solution(), None);
    }
}

#[test]
fn test_an_unbounded_lp_and_milp_are_unbounded() {
    for integer in [false, true] {
        let mut m = Model::new();
        let x = m.add_var(if integer { Var::integer() } else { Var::continuous() });
        let y = m.add_var(Var::continuous().upper(1.0));
        m.add_constraint((x - y).geq(0.0));
        m.maximize(x + y);
        let solved = solve(m.build().unwrap());
        assert!(
            matches!(solved.outcome, Outcome::Unbounded | Outcome::InfeasibleOrUnbounded),
            "integer {integer}: {solved:?}"
        );
    }
}

#[test]
fn test_an_empty_program_is_optimal_at_its_offset() {
    let mut m = Model::new();
    m.minimize(LinExpr::from(3.5));
    let solved = solve(m.build().unwrap());
    let best = optimum(&solved);
    assert_eq!((best.objective, best.values.len()), (3.5, 0));
}

#[test]
fn test_a_program_without_columns_and_a_false_row_is_infeasible() {
    // Without columns every row's activity is zero, so a row whose bounds exclude zero has no satisfying point.
    for (lower, upper) in [(1.0, 1.0), (f64::NEG_INFINITY, -1.0), (0.5, f64::INFINITY)] {
        let mut m = Model::new();
        m.add_constraint(LinExpr::from(0.0).between(lower, upper));
        let solved = solve(m.build().unwrap());
        assert!(
            matches!(solved.outcome, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded),
            "{lower} ..= {upper}: {solved:?}"
        );
    }
}

#[test]
fn test_a_program_without_columns_and_true_rows_is_optimal_at_its_offset() {
    let mut m = Model::new();
    m.add_constraint(LinExpr::from(0.0).equals(0));
    m.add_constraint(LinExpr::from(2.0).between(-1.0, 3.0));
    m.maximize(LinExpr::from(-1.5));
    let solved = solve(m.build().unwrap());
    let best = optimum(&solved);
    assert_eq!((best.objective, best.values.len()), (-1.5, 0));
}

#[test]
fn test_a_program_with_only_bounds_takes_the_best_bound_of_each_column() {
    // Minimize x - y over x in [1, 3] and integer y in [-2, 5]: x = 1, y = 5, objective -4.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(bounds!(1.0, 3.0)));
    let y = m.add_var(Var::integer().bounds(bounds!(-2.0, 5.0)));
    m.minimize(x - y);
    let solved = solve(m.build().unwrap());
    let best = optimum(&solved);
    assert_close(best.objective, -4.0);
    assert_close(best.values[x], 1.0);
    assert_close(best.values[y], 5.0);
}

#[test]
fn test_a_free_row_binds_nothing_and_a_ranged_row_binds_both_sides() {
    // Maximize 2x + y over x <= 2, y >= 0, a free row x - y and 1 <= x + y <= 3: x = 2, y = 1, objective 5; minimizing
    // 2x + y over x, y >= 0 meets the range's lower side at x = 0, y = 1, objective 1.
    let program = |sense| {
        let mut plan = ProgramPlan::new();
        plan.sense = sense;
        let x = plan.col(ProgramColumn::continuous().cost(2.0).bounds(bounds!(0.0, 2.0)));
        let y = plan.col(ProgramColumn::continuous().cost(1.0));
        plan.row((Bounds::FREE, [(x, 1.0), (y, -1.0)]));
        plan.row((bounds!(1.0, 3.0), [(x, 1.0), (y, 1.0)]));
        (plan.build().unwrap(), x, y)
    };
    let (max, x, y) = program(Sense::Maximize);
    let solved = solve(max);
    let best = optimum(&solved);
    assert_close(best.objective, 5.0);
    assert_close(best.values[x], 2.0);
    assert_close(best.values[y], 1.0);
    let (min, x, y) = program(Sense::Minimize);
    let solved = solve(min);
    let best = optimum(&solved);
    assert_close(best.objective, 1.0);
    assert_close(best.values[x], 0.0);
    assert_close(best.values[y], 1.0);
}

#[test]
fn test_a_fixed_column_keeps_its_value() {
    // Maximize x + y with x fixed at 2.5, y integer in [0, 10] and x + y <= 6: y = 3, objective 5.5.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(bounds!(2.5, 2.5)));
    let y = m.add_var(Var::integer().bounds(bounds!(0.0, 10.0)));
    m.add_constraint((x + y).leq(6.0));
    m.maximize(x + y);
    let solved = solve(m.build().unwrap());
    let best = optimum(&solved);
    assert_close(best.objective, 5.5);
    assert_close(best.values[x], 2.5);
    assert_close(best.values[y], 3.0);
}

#[test]
fn test_infinite_bounds_leave_a_column_free() {
    // Minimize x over a free x and y >= 0 subject to x + y >= -3 and y <= 1 in a row: x = -4, y = 1, objective -4.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(bounds!(f64::NEG_INFINITY, f64::INFINITY)));
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + y).geq(-3.0));
    m.add_constraint(LinExpr::from(y).leq(1.0));
    m.minimize(x);
    let solved = solve(m.build().unwrap());
    let best = optimum(&solved);
    assert_close(best.objective, -4.0);
    assert_close(best.values[x], -4.0);
    assert_close(best.values[y], 1.0);
}

#[test]
fn test_an_integer_column_whose_bounds_hold_no_whole_number_is_infeasible() {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(bounds!(0.25, 0.75)));
    m.minimize(x);
    let solved = solve(m.build().unwrap());
    assert!(
        matches!(solved.outcome, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded),
        "{solved:?}"
    );
}

/// Maximize `Σ cᵢ · xᵢ` over six binaries with `Σ 2 · xᵢ <= 5`, the values `c` falling from 1.5 to 1.0 by tenths: the
/// optimum takes the first two, 2.9, and the LP relaxation's 3.55 takes half the third too. With heuristics and
/// separation off and without presolving, the root LP is fractional, so are both children of any branching on it,
/// and the search finds no solution at the root.
fn branching() -> (Program, Vec<Col>) {
    let mut m = Model::new();
    let take = m.add_vars(6, |_| Var::binary());
    m.add_constraint(LinExpr::sum(take.iter().map(|&x| 2.0 * x)).leq(5.0));
    m.maximize(LinExpr::sum(
        take.iter().zip([1.5, 1.4, 1.3, 1.2, 1.1, 1.0]).map(|(&x, c)| c * x),
    ));
    (m.build().unwrap(), take)
}

/// A start taking the last two items, worth 2.1.
fn last_two() -> Option<Vec<f64>> {
    Some(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
}

/// A backend whose heuristics and separators are off, so a search finds a solution only by branching or a start.
fn plain() -> Scip {
    Scip::with_options(ScipOptions {
        heuristics: Some(Emphasis::Off),
        separating: Some(Emphasis::Off),
        ..ScipOptions::default()
    })
}

/// `settings` with presolving off.
fn plain_settings(settings: Settings) -> Settings {
    Settings {
        presolve: Some(false),
        ..settings
    }
}

#[test]
fn test_a_node_limit_stops_with_the_root_bound_and_without_a_solution() {
    let (program, ..) = branching();
    let settings = Settings {
        node_limit: Some(1),
        ..Settings::new()
    };
    let solved = solve_with(plain(), program, plain_settings(settings));
    assert_eq!(solved.status.status, Status::NodeLimit, "{solved:?}");
    let Outcome::Stopped { reason, best, bound } = solved.outcome else {
        panic!("{solved:?}")
    };
    assert_eq!((reason, best), (StopReason::NodeLimit, None));
    assert_close(bound.unwrap(), 3.55);
}

#[test]
fn test_a_node_limit_stops_with_the_start_as_its_best_solution() {
    let (program, take) = branching();
    let settings = Settings {
        node_limit: Some(1),
        ..Settings::new()
    };
    let solved = solve_from(plain(), program, plain_settings(settings), last_two());
    let Outcome::Stopped { reason, best, bound } = solved.outcome else {
        panic!("{solved:?}")
    };
    assert_eq!(reason, StopReason::NodeLimit);
    let best = best.unwrap();
    assert_close(best.objective, 2.1);
    assert_close(best.values[take[4]] + best.values[take[5]], 2.0);
    assert_close(bound.unwrap(), 3.55);
}

#[test]
fn test_a_time_limit_of_zero_stops_with_and_without_a_solution() {
    let (program, ..) = branching();
    let settings = Settings {
        time_limit: Some(Duration::ZERO),
        ..Settings::new()
    };
    let without = solve_with(plain(), program.clone(), plain_settings(settings.clone()));
    assert_eq!(without.status.status, Status::TimeLimit, "{without:?}");
    assert!(
        matches!(
            without.outcome,
            Outcome::Stopped {
                reason: StopReason::TimeLimit,
                best: None,
                ..
            }
        ),
        "{without:?}"
    );
    let with = solve_from(plain(), program, plain_settings(settings), last_two());
    let Outcome::Stopped { reason, best, .. } = with.outcome else {
        panic!("{with:?}")
    };
    assert_eq!(reason, StopReason::TimeLimit);
    assert_close(best.unwrap().objective, 2.1);
}

#[test]
fn test_a_solution_limit_ends_with_a_solution() {
    // A knapsack: a limit of one solution ends with one, optimal or stopped at the limit.
    let mut m = Model::new();
    let take = m.add_vars(6, |_| Var::binary());
    let weights = [5.0, 4.0, 6.0, 3.0, 7.0, 2.0];
    m.add_constraint(LinExpr::sum(take.iter().zip(weights).map(|(&x, w)| w * x)).leq(12.0));
    m.maximize(LinExpr::sum(
        take.iter().zip([7.0, 5.0, 8.0, 4.0, 9.0, 2.0]).map(|(&x, v)| v * x),
    ));
    let settings = Settings {
        solution_limit: Some(solutions!(1)),
        ..Settings::new()
    };
    let solved = solve_with(Scip::new(), m.build().unwrap(), settings);
    assert!(
        matches!(
            solved.outcome,
            Outcome::Optimal(_)
                | Outcome::Stopped {
                    reason: StopReason::SolutionLimit,
                    best: Some(_),
                    ..
                }
        ),
        "{solved:?}"
    );
}

#[test]
fn test_a_solution_limit_reached_by_the_start_stops_at_the_start() {
    let (program, ..) = branching();
    let settings = Settings {
        solution_limit: Some(solutions!(1)),
        ..Settings::new()
    };
    let solved = solve_from(plain(), program, plain_settings(settings), last_two());
    assert_eq!(solved.status.status, Status::BestSolutionLimit, "{solved:?}");
    let Outcome::Stopped { reason, best, .. } = solved.outcome else {
        panic!("{solved:?}")
    };
    assert_eq!(reason, StopReason::SolutionLimit);
    assert_close(best.unwrap().objective, 2.1);
}

#[test]
fn test_a_wide_relative_gap_is_reached_at_the_root() {
    // The optimum 2.9 as the start against the root bound 3.55 is a gap of 0.65 / 2.9, about 22%, which a relative
    // gap of 30% accepts at the root, before any branching could close it.
    let (program, ..) = branching();
    let settings = Settings {
        relative_gap: Some(relative_gap!(0.3)),
        ..Settings::new()
    };
    let start = Some(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let solved = solve_from(plain(), program, plain_settings(settings), start);
    assert_eq!(solved.status.status, Status::GapLimit, "{solved:?}");
    let Outcome::GapReached { solution, bound } = solved.outcome else {
        panic!("{solved:?}")
    };
    assert_close(solution.objective, 2.9);
    assert_close(bound, 3.55);
}

#[test]
fn test_a_sparse_start_is_taken_and_the_solve_still_proves_the_optimum() {
    let (program, take) = branching();
    let solved = Solver::new(Scip::new(), program)
        .unwrap()
        .run()
        .start(vec![(take[5], 1.0)])
        .solve()
        .unwrap();
    let best = optimum(&solved);
    assert_close(best.objective, 2.9);
    assert_close(best.values[take[0]] + best.values[take[1]], 2.0);
}

#[test]
fn test_a_second_solve_takes_its_own_settings() {
    let (program, ..) = branching();
    let mut solver = Solver::new(plain(), program).unwrap();
    let limited = Settings {
        node_limit: Some(1),
        ..Settings::new()
    };
    let first = solver.run().settings(plain_settings(limited)).solve().unwrap();
    assert_eq!(first.status.status, Status::NodeLimit);
    let second = solver.run().settings(plain_settings(Settings::new())).solve().unwrap();
    assert_close(optimum(&second).objective, 2.9);
    let third = solver.solve().unwrap();
    assert_close(optimum(&third).objective, 2.9);
}

#[test]
fn test_a_second_solve_starts_from_the_last_solves_best_solution_until_an_edit() {
    // Without heuristics, separation or presolve, one node finds no solution of `branching` on its own (see
    // `test_a_node_limit_stops_with_the_root_bound_and_without_a_solution`); after a solve to the optimum, the same
    // one-node solve holds that optimum as its best solution, which only the last solve can have handed it. An edit
    // loads the program into a new instance, which holds no solution of before.
    let (program, take) = branching();
    let mut solver = Solver::new(plain(), program).unwrap();
    let one_node = || {
        plain_settings(Settings {
            node_limit: Some(1),
            ..Settings::new()
        })
    };
    let first = solver.run().settings(plain_settings(Settings::new())).solve().unwrap();
    assert_close(optimum(&first).objective, 2.9);
    let again = solver.run().settings(one_node()).solve().unwrap();
    let Outcome::Stopped { reason, best, .. } = again.outcome else {
        panic!("{again:?}")
    };
    assert_eq!(reason, StopReason::NodeLimit);
    let best = best.unwrap();
    assert_close(best.objective, 2.9);
    assert_close(best.values[take[0]] + best.values[take[1]], 2.0);
    // The last item is worth 1.0 still less than any other, so the optimum stays; only the instance is new.
    solver.set_col_costs([(take[5], 0.9)]).unwrap();
    let edited = solver.run().settings(one_node()).solve().unwrap();
    assert!(
        matches!(
            edited.outcome,
            Outcome::Stopped {
                reason: StopReason::NodeLimit,
                best: None,
                ..
            }
        ),
        "{edited:?}"
    );
}

#[test]
fn test_an_edit_reloads_the_program() {
    // Raising y's upper bound to 5 lets maximize x + y over x + y <= 4 reach 4 instead of 3.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(bounds!(0.0, 1.0)));
    let y = m.add_var(Var::continuous().bounds(bounds!(0.0, 2.0)));
    m.add_constraint((x + y).leq(4.0));
    m.maximize(x + y);
    let mut solver = Solver::new(Scip::new(), m.build().unwrap()).unwrap();
    assert_close(optimum(&solver.solve().unwrap()).objective, 3.0);
    solver.set_col_bounds([(y, bounds!(0.0, 5.0))]).unwrap();
    assert_close(optimum(&solver.solve().unwrap()).objective, 4.0);
}

#[test]
fn test_a_coefficient_edit_reloads_the_program() {
    // Maximize x + y over x + y <= 4, x, y in [0, 3]: 4. With the row 2x + y <= 4 instead: 3.5 at (0.5, 3); with y
    // taken out of the row: 3 + 2 = 5 at (2, 3), x bound by 2x <= 4.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(3));
    let y = m.add_var(Var::continuous().upper(3));
    let cap = m.add_constraint((x + y).leq(4.0));
    m.maximize(x + y);
    let mut solver = Solver::new(Scip::new(), m.build().unwrap()).unwrap();
    assert_close(optimum(&solver.solve().unwrap()).objective, 4.0);
    solver.set_coefficients([(cap, x, 2.0)]).unwrap();
    assert_close(optimum(&solver.solve().unwrap()).objective, 3.5);
    solver.set_coefficients([(cap, y, 0.0)]).unwrap();
    assert_close(optimum(&solver.solve().unwrap()).objective, 5.0);
}

#[test]
fn test_columns_and_rows_may_share_a_name_and_go_without_one() {
    let mut plan = ProgramPlan::new();
    plan.sense = Sense::Maximize;
    let named = |name: &str| ProgramColumn {
        name: Some(name.to_owned()),
        ..ProgramColumn::integer().cost(1.0).bounds(bounds!(0.0, 3.0))
    };
    let x = plan.col(named("twin"));
    let y = plan.col(named("twin"));
    let z = plan.col(named(""));
    let w = plan.col(ProgramColumn::continuous().cost(1.0).bounds(bounds!(0.0, 0.5)));
    let row = plan.row((bounds!(f64::NEG_INFINITY, 4.0), [(x, 1.0), (y, 1.0), (z, 1.0)]));
    plan.rows[row].name = Some("twin".to_owned());
    let other = plan.row((bounds!(f64::NEG_INFINITY, 10.0), [(w, 1.0)]));
    plan.rows[other].name = Some("twin".to_owned());
    let solved = solve(plan.build().unwrap());
    assert_close(optimum(&solved).objective, 4.5);
}

#[test]
fn test_a_moved_backend_solves_on_another_thread() {
    let mut solver = Solver::new(Scip::new(), infeasible(false).build().unwrap()).unwrap();
    let solved = std::thread::spawn(move || solver.solve().unwrap()).join().unwrap();
    assert!(matches!(
        solved.outcome,
        Outcome::Infeasible | Outcome::InfeasibleOrUnbounded
    ));
}

/// The parameters these settings set are read in `settings_params`'s unit test; this solve shows SCIP accepts them.
#[test]
fn test_threads_a_seed_and_the_console_setting_are_accepted() {
    let (program, ..) = branching();
    let settings = Settings {
        threads: Some(culmen_core::threads!(4)),
        seed: Some(7),
        absolute_gap: Some(culmen_core::absolute_gap!(0.0)),
        console: false,
        ..Settings::new()
    };
    let solved = solve_with(Scip::new(), program, settings);
    assert_close(optimum(&solved).objective, 2.9);
}

#[test]
fn test_the_objective_edits_reach_each_hand_worked_optimum() {
    // Minimize x + 2y over x + y >= 1, x, y in [0, 4] with x whole: 1 at (1, 0). Maximized: 12 at (4, 4).
    let mut m = Model::new();
    let x = m.add_var(Var::integer().upper(4));
    let y = m.add_var(Var::continuous().upper(4));
    m.add_constraint((x + y).geq(1));
    m.minimize(x + 2 * y);
    let mut solver = Solver::new(Scip::new(), m.build().unwrap()).unwrap();
    let best = |solver: &mut Solver<Scip>| optimum(&solver.solve().unwrap()).objective;
    assert!((best(&mut solver) - 1.0).abs() < 1e-9);
    solver.set_sense(Sense::Maximize).unwrap();
    assert!((best(&mut solver) - 12.0).abs() < 1e-9);
    solver.set_offset(-2).unwrap();
    assert!((best(&mut solver) - 10.0).abs() < 1e-9);
    // Minimize (x - 2)² + y = x² - 4x + 4 + y: 0 at (2, 0), a MIQP.
    solver.set_objective(Sense::Minimize, x * x - 4 * x + 4 + y).unwrap();
    assert!(best(&mut solver).abs() < 1e-6);
    assert_eq!(solver.program().class(), ModelClass::Miqp);
}

#[test]
fn test_the_feasibility_tolerance_admits_a_hair_of_infeasibility() {
    for integer in [false, true] {
        let program = infeasible_by_a_hair(integer);
        let strict = Solver::new(Scip::new(), program.clone()).unwrap().solve().unwrap();
        assert!(
            matches!(strict.outcome, Outcome::Infeasible),
            "integer {integer}: {:?}",
            strict.outcome
        );
        let loose = Settings {
            feasibility_tolerance: Some(feasibility_tolerance!(1e-4)),
            ..Settings::new()
        };
        let solved = Solver::new(Scip::new(), program.clone())
            .unwrap()
            .run()
            .settings(loose)
            .solve()
            .unwrap();
        let Outcome::Optimal(solution) = &solved.outcome else {
            panic!("integer {integer}: {:?}", solved.outcome)
        };
        let violation = program.max_violation(&solution.values);
        assert!(violation > 1e-6 && violation <= 1e-4, "integer {integer}: {violation}");
    }
}

#[test]
fn test_the_optimality_tolerance_takes_a_flat_reduced_cost_as_optimal() {
    let program = flat_costs();
    let solve = |optimality_tolerance| {
        let settings = Settings {
            presolve: Some(false),
            optimality_tolerance,
            ..Settings::new()
        };
        let solved = Solver::new(Scip::new(), program.clone())
            .unwrap()
            .run()
            .settings(settings)
            .solve()
            .unwrap();
        match solved.outcome {
            Outcome::Optimal(solution) => solution.objective,
            other => panic!("{other:?}"),
        }
    };
    assert!((solve(None) - -1e-4).abs() <= 1e-12, "{}", solve(None));
    let loose = solve(Some(feasibility_tolerance!(1e-3)));
    assert!(loose.abs() <= 1e-12, "{loose}");
}
