// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Convex quadratic objectives on SCIP, through the epigraph: hand-worked QP and MIQP optima, random convex QPs
//! against feasible points, and random pure-integer MIQPs against exhaustive enumeration.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{
    Col, Integrality, Linear, Model, ModelClass, Outcome, Program, ProgramPlan, Request, Sense, Solution, SolveError,
    Solver, TiVec, Var,
};
use culmen_scip::{Scip, ScipError, ScipOptions, Status};
use culmen_testing::reference::close;
use culmen_testing::strategy::small_qp;
use proptest::prelude::*;

/// The optimum SCIP finds, which the test expects to exist: feasible, and valued as the program values its point.
fn optimum(program: &Program) -> Solution {
    let solved = Solver::new(Scip::new(), program.clone())
        .unwrap()
        .solve(Request::new())
        .unwrap();
    let Outcome::Optimal(solution) = solved.outcome else {
        panic!("expected an optimum, got {solved:?}")
    };
    assert_eq!(solved.status.status, Status::Optimal);
    assert!(program.max_violation(&solution.values) <= 1e-6, "{solution:?}");
    assert!(
        close(program.objective_value(&solution.values), solution.objective, 1e-9),
        "{solution:?}"
    );
    solution
}

/// Asserts the optimum's objective to 1e-6 and its point to 1e-3. SCIP meets the epigraph's quadratic constraint to
/// its feasibility tolerance by cutting planes, so its objective is near the optimal value, and since the objective
/// is flat at an optimum of these curvatures, its point is near the optimal point only to about the tolerance's
/// square root.
fn assert_point(solution: &Solution, objective: f64, point: &[f64]) {
    assert!(
        close(solution.objective, objective, 1e-6),
        "{solution:?}, want {objective}"
    );
    for (&got, &want) in solution.values.iter().zip(point) {
        assert!((got - want).abs() <= 1e-3, "{solution:?}, want {point:?}");
    }
}

#[test]
fn test_squared_distance_is_least_at_its_centre() {
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::free());
    m.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2));
    let program = m.build().unwrap();
    assert_eq!(program.class(), ModelClass::Qp);
    assert_point(&optimum(&program), 0.0, &[1.0, 2.0]);
}

#[test]
fn test_a_constraint_projects_the_centre_onto_its_half_plane() {
    // The point of x + y <= 1 nearest (1, 2) is (0, 1), at squared distance 2.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::free());
    m.add_constraint((x + y).leq(1));
    m.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2));
    assert_point(&optimum(&m.build().unwrap()), 2.0, &[0.0, 1.0]);
}

#[test]
fn test_a_concave_objective_is_maximized_with_its_offset() {
    // Maximize 5 - (x - 3)² - 2y² + y over x in [0, 2], y free: x = 2, y = 1/4, objective 5 - 1 + 1/8.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(0, 2));
    let y = m.add_var(Var::free());
    m.maximize(5.0 - (x - 3) * (x - 3) - 2.0 * y * y + y);
    assert_point(&optimum(&m.build().unwrap()), 4.125, &[2.0, 0.25]);
}

#[test]
fn test_a_cross_term_counts_at_its_written_weight() {
    // Minimize x² + xy + y² - 3x over free x, y: the gradient 2x + y - 3 = 0, x + 2y = 0 gives x = 2, y = -1, and
    // the objective 4 - 2 + 1 - 6 = -3.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::free());
    m.minimize(x * x + x * y + y * y - 3.0 * x);
    assert_point(&optimum(&m.build().unwrap()), -3.0, &[2.0, -1.0]);
}

#[test]
fn test_a_miqp_rounds_the_centre_to_the_nearest_whole_point() {
    // Minimize (x - 1.4)² + (y - 2.6)² over integers: (1, 3), at 0.16 + 0.16.
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(-10, 10));
    let y = m.add_var(Var::integer().bounds(-10, 10));
    m.minimize((x - 1.4) * (x - 1.4) + (y - 2.6) * (y - 2.6));
    let program = m.build().unwrap();
    assert_eq!(program.class(), ModelClass::Miqp);
    assert_point(&optimum(&program), 0.32, &[1.0, 3.0]);
}

#[test]
fn test_a_miqp_with_a_continuous_column_and_a_row() {
    // Maximize 10 - (n - 2.5)² - c² + c over an integer n and a continuous c with n + c <= 2.75: n = 2 leaves
    // c <= 0.75 and its best c = 0.5, objective 10 - 0.25 - 0.25 + 0.5 = 10; n = 3 forces c <= -0.25, objective
    // 10 - 0.25 - 0.0625 - 0.25 = 9.4375.
    let mut m = Model::new();
    let n = m.add_var(Var::integer().bounds(0, 5));
    let c = m.add_var(Var::free());
    m.add_constraint((n + c).leq(2.75));
    m.maximize(10.0 - (n - 2.5) * (n - 2.5) - c * c + c);
    assert_point(&optimum(&m.build().unwrap()), 10.0, &[2.0, 0.5]);
}

/// The MIQP `-2a + 2b + q(a, b, c)` over integers `a, b` in `[-1, 1]` and `c` in `[-1, 0]`, with the convex
/// `q = 2.05a² - 6ab + 2ac + 9.05b² - 12bc + 5.05c²`, in `sense`; maximizing negates the objective. Of its 18 whole
/// points the origin is the only one at 0; every other is at least 0.05, at (1, 0, 0), then 0.1 at (0, -1, -1) and
/// 0.15 at (-1, -1, -1). Presolve makes `b` and `c` binary, and probing under the cutoff of the incumbent 0 finds
/// that `c = 0` forces `b = 0`, a clique only better solutions obey.
fn binary_products_miqp(sense: Sense) -> Program {
    let mut m = Model::new();
    let a = m.add_var(Var::integer().bounds(-1, 1));
    let b = m.add_var(Var::integer().bounds(-1, 1));
    let c = m.add_var(Var::integer().bounds(-1, 0));
    let objective =
        -2.0 * a + 2.0 * b + 2.05 * a * a - 6.0 * a * b + 2.0 * a * c + 9.05 * b * b - 12.0 * b * c + 5.05 * c * c;
    match sense {
        Sense::Minimize => m.minimize(objective),
        Sense::Maximize => m.maximize(-objective),
    }
    m.build().unwrap()
}

#[test]
fn test_a_miqp_whose_columns_presolve_makes_binary_keeps_its_optimum() {
    let program = binary_products_miqp(Sense::Minimize);
    assert_eq!(program.class(), ModelClass::Miqp);
    assert_point(&optimum(&program), 0.0, &[0.0, 0.0, 0.0]);
}

#[test]
fn test_the_concave_mirror_of_that_miqp_keeps_its_optimum() {
    assert_point(&optimum(&binary_products_miqp(Sense::Maximize)), 0.0, &[0.0, 0.0, 0.0]);
}

#[test]
fn test_a_best_solution_scip_finds_infeasible_in_the_original_problem_is_an_error_not_an_optimum() {
    // With the rewrite of binary products back on, SCIP's presolve rewrites the epigraph through a clique only better
    // solutions obey, and its best solution lies below the epigraph; SCIP's own check in the original problem says
    // so, and the backend reports that instead of the solution.
    for sense in [Sense::Minimize, Sense::Maximize] {
        let options = ScipOptions::default().param("constraints/nonlinear/reformbinprods", true);
        let mut solver = Solver::new(Scip::with_options(options), binary_products_miqp(sense)).unwrap();
        let solved = solver.solve(Request::new());
        assert!(
            matches!(
                solved,
                Err(SolveError::Backend(ScipError::InfeasibleBest {
                    status: Status::Optimal,
                    ..
                }))
            ),
            "{sense:?}: {solved:?}"
        );
    }
}

#[test]
fn test_a_qp_with_conflicting_rows_is_infeasible() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(0, 5));
    m.add_constraint(x.geq(4));
    m.add_constraint(x.leq(3));
    m.minimize(x * x);
    let solved = Solver::new(Scip::new(), m.build().unwrap())
        .unwrap()
        .solve(Request::new())
        .unwrap();
    assert!(
        matches!(solved.outcome, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded),
        "{solved:?}"
    );
}

/// A point in the program's boxes, each column at its fraction of the way from its lower to its upper bound.
fn point_in_boxes(program: &Program, fractions: &[f64]) -> TiVec<Col, f64> {
    program
        .columns()
        .iter()
        .zip(fractions)
        .map(|(c, t)| c.bounds.lower() + t * (c.bounds.upper() - c.bounds.lower()))
        .collect()
}

/// Whether objective value `a` is no worse than `b` under `sense`, up to `tolerance`.
fn no_worse(sense: Sense, a: f64, b: f64, tolerance: f64) -> bool {
    match sense {
        Sense::Minimize => a <= b + tolerance,
        Sense::Maximize => a >= b - tolerance,
    }
}

/// Every whole point of the program's boxes, which hold at most five whole values per column.
fn whole_points(program: &Program) -> Vec<TiVec<Col, f64>> {
    let mut points: Vec<TiVec<Col, f64>> = vec![TiVec::new()];
    for column in program.columns() {
        let (lower, upper) = (column.bounds.lower(), column.bounds.upper());
        assert!(upper - lower <= 4.0, "a box of more than five whole values");
        let values: Vec<f64> = (0..=4).map(|k| lower + f64::from(k)).filter(|&v| v <= upper).collect();
        points = points
            .into_iter()
            .flat_map(|point| {
                values.iter().map(move |&v| {
                    let mut next = point.clone();
                    next.push(v);
                    next
                })
            })
            .collect();
    }
    points
}

/// The program with every column integer.
fn all_integer(program: Program) -> Program {
    let mut plan = ProgramPlan::from(program);
    for column in &mut plan.columns {
        column.integrality = Integrality::Integer;
    }
    plan.build().unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// On random convex QPs SCIP's optimum is feasible, valued as `Program::objective_value` values it, and no worse
    /// than any feasible point in the boxes.
    #[test]
    fn test_scip_qp_optimum_is_no_worse_than_feasible_points(
        program in small_qp(),
        fractions in prop::collection::vec(prop::collection::vec(0.0..=1.0f64, 4), 8),
    ) {
        let solved = Solver::new(Scip::new(), program.clone()).unwrap().solve(Request::new()).unwrap();
        let Outcome::Optimal(best) = solved.outcome else {
            return Err(TestCaseError::fail(format!("{solved:?} on {program:?}")));
        };
        prop_assert!(program.max_violation(&best.values) <= 1e-6, "{best:?}");
        prop_assert!(close(program.objective_value(&best.values), best.objective, 1e-9));
        let tolerance = 1e-5 * best.objective.abs().max(1.0);
        let mut points: Vec<TiVec<Col, f64>> = fractions.iter().map(|f| point_in_boxes(&program, f)).collect();
        points.push(point_in_boxes(&program, &[0.5; 4]));
        for point in points.into_iter().filter(|p| program.max_violation(p) == 0.0) {
            let at_point = program.objective_value(&point);
            prop_assert!(
                no_worse(program.sense(), best.objective, at_point, tolerance),
                "{best:?} loses to {point:?} at {at_point}"
            );
        }
    }

    /// On random pure-integer convex MIQPs SCIP proves the best objective value over every whole point of the boxes,
    /// or infeasibility where none is feasible.
    #[test]
    fn test_scip_agrees_with_enumeration_on_integer_miqps(program in small_qp().prop_map(all_integer)) {
        let sense = program.sense();
        let expected = whole_points(&program)
            .into_iter()
            .filter(|p| program.max_violation(p) == 0.0)
            .map(|p| program.objective_value(&p))
            .reduce(|a, b| if no_worse(sense, a, b, 0.0) { a } else { b });
        let solved = Solver::new(Scip::new(), program.clone()).unwrap().solve(Request::new()).unwrap();
        match (expected, &solved.outcome) {
            (None, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded) => {}
            (Some(want), Outcome::Optimal(best)) => {
                prop_assert!(close(best.objective, want, 1e-6), "SCIP {} vs enumeration {want}", best.objective);
                prop_assert!(program.max_violation(&best.values) <= 1e-6, "{best:?}");
            }
            (want, got) => prop_assert!(false, "enumeration {want:?}, SCIP {got:?}, status {:?}", solved.status),
        }
    }
}
