// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Convex quadratic objectives on HiGHS: hand-worked optima, the ½ factor of HiGHS's Hessian, the refusal of a
//! MIQP, the LP file read back by HiGHS, and random convex QPs.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use std::ffi::CString;
use std::time::Duration;

use culmen_core::{
    Backend, Bounds, Col, Events, Linear, LoadError, Model, ModelClass, Outcome, Program, ProgramColumn, ProgramPlan,
    Request, Sense, Solution, Solved, Solver, StopReason, TiVec, Var,
};
use culmen_highs::{Highs, HighsError, UnsoundQp};
use culmen_testing::reference::close;
use culmen_testing::strategy::small_qp;
use proptest::prelude::*;

/// The optimum HiGHS finds, which the test expects to exist.
fn optimum(program: &Program) -> Solution {
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let solved = solver.solve(Request::new()).unwrap();
    let Outcome::Optimal(solution) = solved.outcome else {
        panic!("expected an optimum, got {solved:?}")
    };
    assert!(program.max_violation(&solution.values) <= 1e-7, "{solution:?}");
    assert!(close(
        program.objective_value(&solution.values),
        solution.objective,
        1e-7
    ));
    solution
}

fn assert_point(solution: &Solution, objective: f64, point: &[f64]) {
    assert!(close(solution.objective, objective, 1e-6), "{solution:?}");
    for (&got, &want) in solution.values.iter().zip(point) {
        assert!((got - want).abs() <= 1e-5, "{solution:?}, want {point:?}");
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
    // The point of `x + y <= 1` nearest (1, 2) is (0, 1), at squared distance 2.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::free());
    m.add_constraint((x + y).leq(1));
    m.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2));
    assert_point(&optimum(&m.build().unwrap()), 2.0, &[0.0, 1.0]);
}

#[test]
fn test_a_concave_objective_is_maximized() {
    // 5 - (x - 3)² - (y + 1)² peaks at (3, -1); the bound x <= 2 moves it to (2, -1), where it is 4.
    let mut m = Model::new();
    let x = m.add_var(Var::free().upper(2));
    let y = m.add_var(Var::free());
    m.maximize(5.0 - (x - 3) * (x - 3) - (y + 1) * (y + 1));
    assert_point(&optimum(&m.build().unwrap()), 4.0, &[2.0, -1.0]);
}

#[test]
fn test_a_square_and_a_cross_term_reach_highs_at_their_written_weight() {
    // x² + xy + y² - 3x: the gradient (2x + y - 3, x + 2y) vanishes at (2, -1), where the objective is -3. The
    // coefficients passed as HiGHS's Hessian, without its ½, would make the objective ½(x + y)² - 3x, which has no
    // minimum.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::free());
    m.minimize(x * x + x * y + y * y - 3.0 * x);
    assert_point(&optimum(&m.build().unwrap()), -3.0, &[2.0, -1.0]);

    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(-3.0, Bounds::FREE));
    let y = plan.col(ProgramColumn::continuous(0.0, Bounds::FREE));
    plan.quadratic = vec![(y, y, 1.0), (y, x, 0.5), (x, x, 1.0), (x, y, 0.5)];
    assert_point(&optimum(&plan.build().unwrap()), -3.0, &[2.0, -1.0]);
}

#[test]
fn test_a_linear_program_solves_as_before() {
    // Maximize 3x + 2y subject to x + y <= 4, x + 3y <= 6, x, y >= 0: the vertex (4, 0) at 12.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + y).leq(4));
    m.add_constraint((x + 3 * y).leq(6));
    m.maximize(3 * x + 2 * y);
    let program = m.build().unwrap();
    assert_eq!(program.class(), ModelClass::Lp);
    assert!(program.quadratic().is_empty());
    assert_point(&optimum(&program), 12.0, &[4.0, 0.0]);
}

#[test]
fn test_a_qp_with_bounds_alone_leaves_its_starting_point() {
    // Found by the property test below. On a QP without rows HiGHS 1.15 starts from the unconstrained minimizer
    // clamped into the bounds, here the origin, and reports it optimal after no iteration although the gradient
    // there, -1 on x3, points into the box. Its optimum is x2 = 20/51, x3 = 1, at -859/1020: the minimum of
    // 2.55·x2² - 2·x2 - 0.45, where the gradient is (40/51, 1, 0, 0.1 - 40/51), pointing out of the box.
    let mut plan = ProgramPlan::new();
    let x: Vec<Col> = [0.0, 0.0, 0.0, -1.0]
        .into_iter()
        .map(|cost| plan.col(ProgramColumn::continuous(cost, Bounds::new(0.0, 1.0).unwrap())))
        .collect();
    plan.quadratic = vec![
        (x[0], x[0], 2.05),
        (x[0], x[1], 4.0),
        (x[0], x[2], 2.0),
        (x[1], x[1], 2.55),
        (x[1], x[3], 1.0),
        (x[2], x[2], 2.55),
        (x[2], x[3], -2.0),
        (x[3], x[3], 0.55),
    ];
    assert_point(
        &optimum(&plan.build().unwrap()),
        -859.0 / 1020.0,
        &[0.0, 0.0, 20.0 / 51.0, 1.0],
    );
}

#[test]
fn test_a_qp_optimum_highs_cannot_support_is_not_reported_optimal() {
    // Found by the property test below. HiGHS 1.15 reports this QP optimal at about 34.27 with a relative
    // primal-dual objective error near 1, while the boxes' midpoint (1.5, -1, 0, -2) is feasible at 14.99. The
    // optimum, from its KKT conditions in exact arithmetic, is about 6.3397 at (2.2599, 0, -0.3997, -2.3602).
    let mut plan = ProgramPlan::new();
    let boxes = [(0.0, 3.0, 0.0), (-2.0, 0.0, -5.0), (-1.0, 1.0, 0.0), (-3.0, -1.0, -2.0)];
    let x: Vec<Col> = boxes
        .into_iter()
        .map(|(lower, upper, cost)| plan.col(ProgramColumn::continuous(cost, Bounds::new(lower, upper).unwrap())))
        .collect();
    let exactly_12 = Bounds::new(12.0, 12.0).unwrap();
    plan.row(exactly_12, [(x[0], 2.0), (x[1], -3.0), (x[2], -1.0), (x[3], -3.0)]);
    let range = Bounds::new(-0.5, 3.5).unwrap();
    plan.row(range, [(x[0], 1.0), (x[1], -2.0), (x[2], 1.0), (x[3], 1.0)]);
    plan.quadratic = vec![
        (x[0], x[0], 4.55),
        (x[0], x[1], -9.0),
        (x[0], x[2], -9.0),
        (x[0], x[3], 9.0),
        (x[1], x[1], 5.05),
        (x[1], x[2], 6.0),
        (x[1], x[3], -8.0),
        (x[2], x[2], 9.05),
        (x[2], x[3], -12.0),
        (x[3], x[3], 5.05),
    ];
    let solved = Solver::new(Highs::new(), plan.build().unwrap())
        .unwrap()
        .solve(Request::new());
    match solved {
        Ok(Solved {
            outcome: Outcome::Optimal(solution),
            ..
        }) => assert_point(
            &solution,
            6.339_658_331_975_892,
            &[2.259_895_7, 0.0, -0.399_739_4, -2.360_156_4],
        ),
        Err(HighsError::UnsoundQp(UnsoundQp::ObjectiveError { error })) => assert!(error > 1e-5),
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_a_bounded_qp_reported_unbounded_is_refused() {
    // Found by the property test below. HiGHS 1.15 reports this QP unbounded although every column lies in a
    // finite box, where a continuous objective is bounded.
    let mut plan = ProgramPlan::new();
    let boxes = [(0.0, 2.0, 0.0), (-2.0, 2.0, -3.0), (1.0, 5.0, -2.0), (-1.0, 1.0, 0.0)];
    let x: Vec<Col> = boxes
        .into_iter()
        .map(|(lower, upper, cost)| plan.col(ProgramColumn::continuous(cost, Bounds::new(lower, upper).unwrap())))
        .collect();
    plan.row(Bounds::new(-1.0, 0.0).unwrap(), [(x[1], -2.0), (x[3], -1.0)]);
    plan.quadratic = vec![
        (x[0], x[0], 5.05),
        (x[0], x[1], 3.0),
        (x[0], x[2], -1.0),
        (x[1], x[1], 4.55),
        (x[1], x[2], -3.0),
        (x[1], x[3], -9.0),
        (x[2], x[2], 0.55),
        (x[2], x[3], 3.0),
        (x[3], x[3], 5.05),
    ];
    let program = plan.build().unwrap();
    match Solver::new(Highs::new(), program.clone())
        .unwrap()
        .solve(Request::new())
    {
        Ok(Solved {
            outcome: Outcome::Optimal(solution),
            ..
        }) => assert!(program.max_violation(&solution.values) <= 1e-6),
        Err(HighsError::UnsoundQp(UnsoundQp::UnboundedInABox)) => {}
        other => panic!("{other:?}"),
    }
}

/// min (x - 0.4)² + n with n integer in 0..=3.
fn miqp() -> Program {
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let n = m.add_var(Var::integer().upper(3));
    m.minimize((x - 0.4) * (x - 0.4) + n);
    m.build().unwrap()
}

#[test]
fn test_a_miqp_is_refused_before_highs_sees_it() {
    let program = miqp();
    assert_eq!(program.class(), ModelClass::Miqp);
    let error = Solver::new(Highs::new(), program).unwrap_err();
    assert_eq!(
        error,
        LoadError::Unsupported {
            class: ModelClass::Miqp,
            supported: Highs::CLASSES,
        }
    );
    assert_eq!(
        error.to_string(),
        "Solver: the backend solves {LP, MILP, QP}, not a MIQP"
    );
}

#[test]
fn test_highs_itself_cannot_solve_a_miqp() {
    // The declaration matches HiGHS 1.15: loaded around the class check, a MIQP fails at the run.
    let mut highs = Highs::new();
    highs.load(&miqp()).unwrap();
    let error = highs.solve(&culmen_core::Settings::new(), &mut ()).unwrap_err();
    assert!(matches!(error, HighsError::Run { .. }), "{error:?}");
}

/// The event kinds a callback is called with.
struct Kinds(Events);

impl culmen_core::Callback<Highs> for Kinds {
    fn log(&mut self, _: &mut culmen_highs::HighsLog<'_>) {
        self.0 |= Events::LOG;
    }
    fn progress(&mut self, _: &mut culmen_highs::HighsMipProgress) {
        self.0 |= Events::PROGRESS;
    }
    fn poll(&mut self, _: &mut culmen_highs::HighsPoll<'_>) {
        self.0 |= Events::POLL;
    }
    fn feasible(&mut self, _: &mut culmen_highs::HighsSolution<'_>) {
        self.0 |= Events::FEASIBLE;
    }
    fn incumbent(&mut self, _: &mut culmen_highs::HighsSolution<'_>) {
        self.0 |= Events::INCUMBENT;
    }
    fn heuristic(&mut self, _: &mut culmen_highs::HighsHeuristic<'_>) {
        self.0 |= Events::HEURISTIC;
    }
    fn native(&mut self, _: &mut culmen_highs::HighsNative) {
        self.0 |= Events::NATIVE;
    }
}

#[test]
fn test_a_qp_solve_reports_its_log_alone() {
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::free());
    m.add_constraint((x + y).leq(1));
    m.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2));
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    let mut kinds = Kinds(Events::NONE);
    solver.solve_with(Request::new(), &mut kinds).unwrap();
    assert_eq!(kinds.0, Events::LOG);
}

/// The program written to a file, read by HiGHS's own LP-file reader and solved there: its objective value.
fn solve_lp_file(program: &Program) -> f64 {
    let path = std::env::temp_dir().join(format!("culmen-quadratic-{}.lp", std::process::id()));
    std::fs::write(&path, program.to_lp_string()).unwrap();
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    // SAFETY: the instance is created here, used only on this thread, and destroyed before the block ends; the path
    // is a NUL-terminated string HiGHS reads during the call; the option name is a NUL-terminated literal.
    let (read, run, objective) = unsafe {
        let highs = highs_sys::Highs_create();
        highs_sys::Highs_setBoolOptionValue(highs, c"output_flag".as_ptr(), 0);
        let read = highs_sys::Highs_readModel(highs, c_path.as_ptr());
        let run = highs_sys::Highs_run(highs);
        let objective = highs_sys::Highs_getObjectiveValue(highs);
        highs_sys::Highs_destroy(highs);
        (read, run, objective)
    };
    std::fs::remove_file(&path).unwrap();
    assert_eq!((read, run), (highs_sys::kHighsStatusOk, highs_sys::kHighsStatusOk));
    objective
}

#[test]
fn test_highs_reads_the_lp_file_of_a_qp_as_the_same_problem() {
    let mut m = Model::new();
    let x = m.add_var(Var::free().name("x"));
    let y = m.add_var(Var::continuous().upper(3).name("y"));
    m.add_constraint((x + 2 * y).geq(1));
    m.minimize(x * x + x * y + 2.0 * y * y - 3.0 * x + 1.5);
    let program = m.build().unwrap();
    assert_eq!(
        program.to_lp_string(),
        "Minimize\n obj: -3 x + [ 2 x ^ 2 + 2 x * y + 4 y ^ 2 ] / 2 + 1.5\nSubject To\n r0: 1 x + 2 y >= 1\n\
         Bounds\n x free\n 0 <= y <= 3\nEnd\n"
    );
    let file = solve_lp_file(&program);
    assert!(
        close(file, optimum(&program).objective, 1e-7),
        "the file solves to {file}"
    );

    m.set_objective(Sense::Maximize, -(x * x) - 4.0 * y);
    let program = m.build().unwrap();
    let file = solve_lp_file(&program);
    assert!(
        close(file, optimum(&program).objective, 1e-7),
        "the file solves to {file}"
    );
}

/// The value of `program`'s objective at `values`, in the direction it is optimized: lower is better.
fn loss(program: &Program, values: &TiVec<Col, f64>) -> f64 {
    match program.sense() {
        Sense::Minimize => program.objective_value(values),
        Sense::Maximize => -program.objective_value(values),
    }
}

/// A point at fractions of the way through each column's bounds.
fn point_in_boxes(program: &Program, fractions: &[f64]) -> TiVec<Col, f64> {
    program
        .columns()
        .iter()
        .zip(fractions)
        .map(|(c, t)| c.bounds.lower() + t * (c.bounds.upper() - c.bounds.lower()))
        .collect()
}

/// A short projected gradient step from `values`, back into the bounds.
fn gradient_step(program: &Program, values: &TiVec<Col, f64>) -> TiVec<Col, f64> {
    let mut gradient: TiVec<Col, f64> = program.columns().iter().map(|c| c.cost).collect();
    for (i, j, q) in program.quadratic().entries() {
        gradient[i] += q * values[j];
        gradient[j] += q * values[i];
    }
    let direction = match program.sense() {
        Sense::Minimize => -1e-3,
        Sense::Maximize => 1e-3,
    };
    program
        .columns()
        .iter_enumerated()
        .map(|(col, c)| (values[col] + direction * gradient[col]).clamp(c.bounds.lower(), c.bounds.upper()))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// On random convex QPs HiGHS's optimum is feasible, valued as `Program::objective_value` values it, and no worse
    /// than any feasible point in the boxes nor than a projected gradient step from it. A run HiGHS fails, or whose
    /// optimum its own primal-dual objective error does not support, ends in a typed error, and one that does not
    /// terminate stops at its time limit.
    #[test]
    fn test_highs_qp_optimum_is_no_worse_than_feasible_points(
        program in small_qp(),
        fractions in prop::collection::vec(prop::collection::vec(0.0..=1.0f64, 4), 8),
    ) {
        let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
        // HiGHS 1.15's QP solver can fail to terminate on such a QP; the time limit ends it.
        let best = match solver.solve(Request::new().time_limit(Duration::from_secs(2))) {
            Ok(Solved { outcome: Outcome::Optimal(best), .. }) => best,
            Ok(Solved { outcome: Outcome::Stopped { reason: StopReason::TimeLimit, .. }, .. })
            | Err(HighsError::Run { .. } | HighsError::UnsoundQp(_)) => return Ok(()),
            other => return Err(TestCaseError::fail(format!("{other:?} on {program:?}"))),
        };
        prop_assert!(program.max_violation(&best.values) <= 1e-6, "{best:?}");
        prop_assert!(close(program.objective_value(&best.values), best.objective, 1e-7));
        let at_best = loss(&program, &best.values);
        let tolerance = 1e-5 * at_best.abs().max(1.0);
        let mut points: Vec<TiVec<Col, f64>> = fractions.iter().map(|f| point_in_boxes(&program, f)).collect();
        points.push(point_in_boxes(&program, &[0.5; 4]));
        for point in points.into_iter().filter(|p| program.max_violation(p) == 0.0) {
            prop_assert!(at_best <= loss(&program, &point) + tolerance, "{best:?} loses to {point:?}");
        }
        if program.num_rows() == 0 {
            let step = gradient_step(&program, &best.values);
            prop_assert!(at_best <= loss(&program, &step) + tolerance, "{best:?} loses to its step {step:?}");
        }
    }
}
