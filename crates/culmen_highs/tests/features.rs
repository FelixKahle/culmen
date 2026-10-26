// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Semi columns, further objectives and starts solved by HiGHS: hand-worked optima, HiGHS's quirks around them, and
//! semi programs against the exact reference solver.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{
    Analysis, AnalysisError, Col, Combination, Improvement, LinExpr, Linear, LoadError, Misfit, Model, ModelClass,
    Objective, ObjectiveId, Observer, Outcome, Program, ProgramFeature, Request, Solution, SolveError, Solver, Start,
    StartError, TiSlice, TiVec, Tolerance, Var, Weights, absolute_gap, penalty, relative_gap, weight,
};
use culmen_highs::{Highs, HighsError};
use culmen_testing::reference::{self, close};
use culmen_testing::strategy::small_semi;
use proptest::prelude::*;

/// The optimal solution of a program HiGHS solves.
fn optimum(program: &Program) -> Solution {
    let solved = Solver::new(Highs::new(), program.clone())
        .unwrap()
        .solve(Request::new())
        .unwrap();
    match solved.outcome {
        Outcome::Optimal(solution) => solution,
        other => panic!("expected an optimum, got {other:?}"),
    }
}

#[test]
fn test_a_semi_continuous_column_is_zero_or_in_its_bounds() {
    // x is zero or in [2, 10]. Below 1.5 the most it can be is zero, where a continuous x would reach 1.5.
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(2, 10));
    m.add_constraint(x.leq(1.5));
    m.maximize(x);
    let best = optimum(&m.build().unwrap());
    assert_eq!((best.objective, best.values[x]), (0.0, 0.0));

    // Above 0.5 the least it can be is its lower bound 2, where a continuous x would stay at 0.5.
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(2, 10));
    m.add_constraint(x.geq(0.5));
    m.minimize(x);
    let best = optimum(&m.build().unwrap());
    assert!(
        close(best.objective, 2.0, 1e-9) && close(best.values[x], 2.0, 1e-9),
        "{best:?}"
    );
}

#[test]
fn test_a_semi_integer_column_is_zero_or_whole_in_its_bounds() {
    // y is zero or a whole number in [2.5, 10]: above 0.5 the least is 3; below 2.9 the most is zero.
    let mut m = Model::new();
    let y = m.add_var(Var::semi_integer().bounds(2.5, 10));
    m.add_constraint(y.geq(0.5));
    m.minimize(y);
    let best = optimum(&m.build().unwrap());
    assert_eq!((best.objective, best.values[y]), (3.0, 3.0));

    let mut m = Model::new();
    let y = m.add_var(Var::semi_integer().bounds(2.5, 10));
    m.add_constraint(y.leq(2.9));
    m.maximize(y);
    assert_eq!(optimum(&m.build().unwrap()).values[y], 0.0);
}

#[test]
fn test_a_semi_upper_bound_above_highs_limit_is_refused() {
    // HiGHS 1.15 lowers a semi column's upper bound above 1e5 to 1e5 (HighsLpUtils.cpp, assessSemiVariables), so
    // the optimum x = 2e5 of this program would be out of its reach.
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(1, 2e5));
    m.maximize(x);
    let error = Solver::new(Highs::new(), m.build().unwrap()).unwrap_err();
    assert_eq!(error, LoadError::Backend(HighsError::SemiUpper { col: x, upper: 2e5 }));
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(1, Highs::SEMI_UPPER_LIMIT));
    m.maximize(x);
    assert_eq!(optimum(&m.build().unwrap()).objective, Highs::SEMI_UPPER_LIMIT);
}

/// Maximize `x + y` over `x + y <= 10`, `x <= 6`, `y <= 10`; then minimize `y`.
///
/// The first stage's optimum is 10. Kept at 10, the second stage takes `y = 4`; allowed to drop by 2 to 8, `y = 2`;
/// by `min(5, 10% of 10) = 1` to 9, `y = 3`.
fn two_stage() -> (Model, Col, Col, ObjectiveId) {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(6));
    let y = m.add_var(Var::continuous().upper(10));
    m.add_constraint((x + y).leq(10));
    m.maximize(x + y);
    let second = m.add_objective(Objective::minimize(y));
    (m, x, y, second)
}

#[test]
fn test_lexicographic_objectives_keep_each_optimum_within_its_tolerance() {
    let (mut m, x, y, second) = two_stage();
    let cases = [
        (Tolerance::EXACT, 4.0),
        (
            Tolerance {
                absolute: Some(absolute_gap!(2.0)),
                relative: None,
            },
            2.0,
        ),
        (
            Tolerance {
                absolute: Some(absolute_gap!(5.0)),
                relative: Some(relative_gap!(0.1)),
            },
            3.0,
        ),
    ];
    for (tolerance, expected_y) in cases {
        m.set_objective_tolerance(tolerance);
        let program = m.build().unwrap();
        let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
        // A second solve gives the same: HiGHS's own costs, which its lexicographic solve overwrites, are not used.
        for _ in 0..2 {
            let best = solver.solve(Request::new()).unwrap().outcome.into_solution().unwrap();
            assert!(close(best.values[x], 6.0, 1e-9), "{tolerance:?}: {best:?}");
            assert!(close(best.values[y], expected_y, 1e-9), "{tolerance:?}: {best:?}");
            // The solution reports the program's own objective, not HiGHS's objective value, which a lexicographic
            // solve leaves at zero (HighsInterface.cpp, returnFromLexicographicOptimization).
            assert!(close(best.objective, 6.0 + expected_y, 1e-9), "{best:?}");
            let values = program.objective_values(&best.values);
            assert!(close(values[second], expected_y, 1e-9) && close(values[ObjectiveId::FIRST], best.objective, 0.0));
        }
    }
}

#[test]
fn test_lexicographic_objectives_of_a_mip() {
    // Integer x, y in [0, 5] with x + y <= 7: maximize x + y (7, at (2, 5) … (5, 2)), then maximize x - y: (5, 2).
    let mut m = Model::new();
    let x = m.add_var(Var::integer().upper(5));
    let y = m.add_var(Var::integer().upper(5));
    m.add_constraint((x + y).leq(7));
    m.maximize(x + y);
    m.add_objective(Objective::maximize(x - y));
    let best = optimum(&m.build().unwrap());
    assert_eq!(
        (best.values[x].round(), best.values[y].round(), best.objective),
        (5.0, 2.0, 7.0)
    );

    // Minimizing x - y second instead: (2, 5).
    let mut m2 = Model::new();
    let x = m2.add_var(Var::integer().upper(5));
    let y = m2.add_var(Var::integer().upper(5));
    m2.add_constraint((x + y).leq(7));
    m2.maximize(x + y);
    m2.add_objective(Objective::minimize(x - y));
    let best = optimum(&m2.build().unwrap());
    assert_eq!((best.values[x].round(), best.values[y].round()), (2.0, 5.0));
}

#[test]
fn test_blended_objectives_minimize_their_signed_weighted_sum() {
    // Over the two-stage program, minimize -w₀·(x + y) + w₁·y. For w₀ = 1, w₁ = 2 that is -x + y, at (6, 0); for
    // w₁ = 0.5, -x - 0.5 y, at (6, 4); for w₀ = 0.25, w₁ = 1, -0.25 x + 0.75 y, at (6, 0).
    let cases = [
        (weight!(1.0), weight!(2.0), 0.0),
        (weight!(1.0), weight!(0.5), 4.0),
        (weight!(0.25), weight!(1.0), 0.0),
    ];
    for (first, second, expected_y) in cases {
        let mut m = Model::new();
        let x = m.add_var(Var::continuous().upper(6));
        let y = m.add_var(Var::continuous().upper(10));
        m.add_constraint((x + y).leq(10));
        m.maximize(x + y);
        m.add_objective(Objective::minimize(y).weight(second));
        m.set_objective_weight(first);
        m.set_combination(Combination::Blended);
        let best = optimum(&m.build().unwrap());
        assert!(
            close(best.values[x], 6.0, 1e-9) && close(best.values[y], expected_y, 1e-9),
            "{first}, {second}: {best:?}"
        );
        // The solution reports the program's own objective, not HiGHS's blended one.
        assert!(close(best.objective, 6.0 + expected_y, 1e-9), "{best:?}");
    }
}

/// The objective of every improvement an observer is told of, in order.
#[derive(Default)]
struct Improvements(Vec<f64>);

impl Observer for Improvements {
    fn improved(&mut self, improvement: &Improvement<'_>) {
        self.0.push(improvement.objective);
    }
}

/// A knapsack of forty items HiGHS does not solve at its root without a start.
fn knapsack() -> Program {
    let weights: Vec<f64> = (0..40).map(|i| f64::from((i * 37 % 23) + 5)).collect();
    let values: Vec<f64> = (0..40).map(|i| f64::from((i * 53 % 29) + 7)).collect();
    let mut m = Model::new();
    let take = m.add_vars(40, |_| Var::binary());
    m.add_constraint(LinExpr::sum((0..40).map(|i| weights[i] * take[i])).leq(200));
    m.maximize(LinExpr::sum((0..40).map(|i| values[i] * take[i])));
    m.build().unwrap()
}

#[test]
fn test_a_full_start_is_the_first_incumbent() {
    let program = knapsack();
    let best = optimum(&program);
    let mut fresh = Solver::new(Highs::new(), program).unwrap();
    let mut seen = Improvements::default();
    let solved = fresh
        .solve(Request::new().observer(&mut seen).start(best.values.clone()))
        .unwrap();
    assert_eq!(seen.0.first(), Some(&best.objective), "{:?}", seen.0);
    assert_eq!(solved.outcome.solution().map(|s| s.objective), Some(best.objective));
}

#[test]
fn test_a_sparse_start_is_completed_by_highs() {
    let program = knapsack();
    let best = optimum(&program);
    let chosen: Vec<(Col, f64)> = best
        .values
        .iter_enumerated()
        .filter(|&(_, &v)| v > 0.5)
        .map(|(col, _)| (col, 1.0))
        .collect();
    let mut fresh = Solver::new(Highs::new(), program).unwrap();
    let mut seen = Improvements::default();
    fresh.solve(Request::new().observer(&mut seen).start(chosen)).unwrap();
    assert_eq!(seen.0.first(), Some(&best.objective), "{:?}", seen.0);
}

#[test]
fn test_a_start_of_an_lp_leaves_its_optimum() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(6));
    let y = m.add_var(Var::continuous().upper(10));
    m.add_constraint((x + y).leq(10));
    m.maximize(x + 2 * y);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    for start in [Start::Full(vec![1.0, 1.0].into()), Start::Sparse(vec![(y, 3.0)])] {
        let best = solver
            .solve(Request::new().start(start))
            .unwrap()
            .outcome
            .into_solution()
            .unwrap();
        assert_eq!((best.objective, best.values[y]), (20.0, 10.0));
    }
    let _ = x;
}

#[test]
fn test_a_sparse_start_may_hold_a_semi_columns_zero() {
    // Zero lies outside a semi column's bounds, which HiGHS's own sparse start refuses (see the fixture below); the
    // backend passes a sparse start whole, so the zero is taken.
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(2, 10));
    let y = m.add_var(Var::integer().upper(3));
    m.add_constraint((x + y).geq(1));
    m.minimize(x + 2 * y);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    let best = solver
        .solve(Request::new().start(vec![(x, 0.0), (y, 1.0)]))
        .unwrap()
        .outcome
        .into_solution()
        .unwrap();
    assert_eq!((best.objective, best.values[x], best.values[y]), (2.0, 0.0, 1.0));
}

#[test]
fn test_highs_sparse_start_refuses_a_semi_columns_zero() {
    // HiGHS 1.15's Highs_setSparseSolution refuses any value outside its column's bounds (Highs.cpp, setSolution),
    // a semi column's zero among them: the reason the backend does not call it.
    let lower = [2.0];
    let upper = [10.0];
    let cost = [1.0];
    let integrality = [highs_sys::kHighsVarTypeSemiContinuous];
    let (index, value) = ([0], [0.0]);
    // SAFETY: the instance is created here, used only on this thread, and destroyed before the block ends. The model
    // has one column and no rows: each column array holds one value, the row and matrix arrays are null with zero
    // counts, which HiGHS skips; the sparse solution names one column with one value.
    let (passed, set) = unsafe {
        let highs = highs_sys::Highs_create();
        debug_assert!(!highs.is_null(), "HiGHS allocates an instance");
        highs_sys::Highs_setBoolOptionValue(highs, c"output_flag".as_ptr(), 0);
        let passed = highs_sys::Highs_passMip(
            highs,
            1,
            0,
            0,
            highs_sys::kHighsMatrixFormatRowwise,
            highs_sys::kHighsObjSenseMinimize,
            0.0,
            cost.as_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            integrality.as_ptr(),
        );
        let set = highs_sys::Highs_setSparseSolution(highs, 1, index.as_ptr(), value.as_ptr());
        highs_sys::Highs_destroy(highs);
        (passed, set)
    };
    assert_eq!((passed, set), (highs_sys::kHighsStatusOk, highs_sys::kHighsStatusError));
}

#[test]
fn test_a_start_that_does_not_fit_is_refused_typed() {
    let program = knapsack();
    let mut solver = Solver::new(Highs::new(), program).unwrap();
    let refused = |solver: &mut Solver<Highs>, start: Start| match solver.solve(Request::new().start(start)) {
        Err(SolveError::Start(error)) => error,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        refused(&mut solver, Start::Full(TiVec::from(vec![0.0; 39]))),
        StartError::Length {
            expected: 40,
            found: 39
        }
    );
    assert_eq!(
        refused(&mut solver, Start::Sparse(vec![(Col::new(40), 1.0)])),
        StartError::UnknownColumn { col: Col::new(40) }
    );
    assert_eq!(
        refused(&mut solver, Start::Sparse(vec![(Col::new(1), f64::INFINITY)])),
        StartError::NotFinite {
            col: Col::new(1),
            value: f64::INFINITY
        }
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// HiGHS reaches the exact reference's optimum of a tiny program with semi columns, and every semi column of its
    /// solution is zero or in its bounds.
    #[test]
    fn test_semi_programs_match_the_reference(program in small_semi()) {
        let solved = Solver::new(Highs::new(), program.clone()).unwrap().solve(Request::new()).unwrap();
        match (reference::solve(&program).objective(), solved.outcome) {
            (None, Outcome::Infeasible) => {}
            (Some(expected), Outcome::Optimal(best)) => {
                prop_assert!(close(best.objective, expected, 1e-6), "{best:?} against {expected}");
                prop_assert!(program.max_violation(&best.values) <= 1e-6, "{best:?}");
                let semi_ok = program.cols().filter(|&c| program.columns()[c].integrality.is_semi()).all(|c| {
                    let v = best.values[c];
                    v.abs() <= 1e-9 || program.columns()[c].bounds.violation(v) <= 1e-6
                });
                prop_assert!(semi_ok, "{best:?}");
            }
            (expected, outcome) => prop_assert!(false, "{outcome:?} against {expected:?} on {program:?}"),
        }
    }
}

#[test]
fn test_objective_values_follow_the_program() {
    let (m, x, y, second) = two_stage();
    let program = m.build().unwrap();
    let values = program.objective_values(TiSlice::from_ref(&[1.0, 2.0]));
    assert_eq!((values[ObjectiveId::FIRST], values[second]), (3.0, 2.0));
    let _ = (x, y);
}

/// Minimize the violation of `x + y >= 1.8` (price 1) and `x <= 0.5` (price 10) with `x` semi-continuous in
/// `[2, 10]` and `y` in `[0, 1]`, their bounds hard. Off, `x = 0` and `y = 1` miss the first row by 0.8, a penalty of
/// 0.8; on, `x >= 2` misses the second by at least 1.5, a penalty of 15. So the relaxed point has the semi column at
/// zero, which violates none of its bounds.
fn semi_relaxation() -> (Program, Col, Col) {
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(2, 10));
    let y = m.add_var(Var::continuous().upper(1));
    m.add_constraint((x + y).geq(1.8));
    m.add_constraint(x.leq(0.5));
    m.minimize(x);
    (m.build().unwrap(), x, y)
}

#[test]
fn test_highs_relaxation_of_a_semi_program_is_refused() {
    // HiGHS 1.15's feasibility relaxation keeps x within [2, 10]: it returns x = 2, a penalty of 15, however high the
    // second row's price, where x = 0 costs 0.8 (a raw fixture below). The backend refuses it.
    let (program, ..) = semi_relaxation();
    let weights = Weights::uniform(&program, None, Some(penalty!(1.0)));
    let mut solver = Solver::new(Highs::new(), program).unwrap();
    assert_eq!(
        solver.feasibility_relaxation(&weights).unwrap_err(),
        AnalysisError::Backend(HighsError::AnalysisFeature {
            analysis: Analysis::FeasibilityRelaxation,
            feature: ProgramFeature::SemiContinuous
        })
    );
}

#[test]
fn test_highs_relaxation_keeps_a_semi_column_on() {
    let (lower, upper, cost) = ([2.0, 0.0], [10.0, 1.0], [1.0, 0.0]);
    let integrality = [
        highs_sys::kHighsVarTypeSemiContinuous,
        highs_sys::kHighsVarTypeContinuous,
    ];
    let (row_lower, row_upper) = ([1.8, f64::NEG_INFINITY], [f64::INFINITY, 0.5]);
    let (starts, index, value) = ([0, 2], [0, 1, 0], [1.0, 1.0, 1.0]);
    let (col_penalty, row_penalty) = ([-1.0, -1.0], [1.0, 100.0]);
    let mut x = [0.0; 2];
    // SAFETY: the instance is created here, used only on this thread, and destroyed before the block ends. The model
    // has two columns and two rows: the column arrays hold two values, the row arrays two, the row-wise matrix two
    // starts and three entries; the penalties one per column bound and per row; the solution buffer two values and
    // the other solution arrays are null, which HiGHS skips.
    let status = unsafe {
        let highs = highs_sys::Highs_create();
        debug_assert!(!highs.is_null(), "HiGHS allocates an instance");
        highs_sys::Highs_setBoolOptionValue(highs, c"output_flag".as_ptr(), 0);
        highs_sys::Highs_passMip(
            highs,
            2,
            2,
            3,
            highs_sys::kHighsMatrixFormatRowwise,
            highs_sys::kHighsObjSenseMinimize,
            0.0,
            cost.as_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            row_lower.as_ptr(),
            row_upper.as_ptr(),
            starts.as_ptr(),
            index.as_ptr(),
            value.as_ptr(),
            integrality.as_ptr(),
        );
        let status = highs_sys::Highs_feasibilityRelaxation(
            highs,
            -1.0,
            -1.0,
            -1.0,
            col_penalty.as_ptr(),
            col_penalty.as_ptr(),
            row_penalty.as_ptr(),
        );
        highs_sys::Highs_getSolution(
            highs,
            x.as_mut_ptr(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        highs_sys::Highs_destroy(highs);
        status
    };
    assert_ne!(status, highs_sys::kHighsStatusError);
    assert_eq!(
        x[0], 2.0,
        "the semi column stays on, at a penalty of 150 against 0.8 off"
    );
}

#[test]
fn test_a_soft_bound_of_a_semi_column_is_refused() {
    let (program, x, _) = semi_relaxation();
    let mut weights = Weights::uniform(&program, None, Some(penalty!(1.0)));
    weights.lower[x] = Some(penalty!(1.0));
    let mut solver = Solver::new(Highs::new(), program).unwrap();
    assert_eq!(
        solver.feasibility_relaxation(&weights).unwrap_err(),
        AnalysisError::Misfit(Misfit::SoftSemiBound { col: x })
    );
}

#[test]
fn test_analyses_of_one_objective_are_refused_for_several() {
    let (m, ..) = two_stage();
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    solver.solve(Request::new()).unwrap();
    let refused = AnalysisError::Feature {
        analysis: Analysis::Duals,
        feature: ProgramFeature::MultipleObjectives,
    };
    assert_eq!(solver.duals().unwrap_err(), refused);
    assert_eq!(
        refused.to_string(),
        "Solver: duals are not defined for a program with multiple objectives"
    );
    assert!(matches!(
        solver.basis(),
        Err(AnalysisError::Feature {
            analysis: Analysis::Basis,
            ..
        })
    ));
    assert!(matches!(
        solver.ranging(),
        Err(AnalysisError::Feature {
            analysis: Analysis::Ranging,
            ..
        })
    ));
    assert!(matches!(
        solver.primal_ray(),
        Err(AnalysisError::Feature {
            analysis: Analysis::Rays,
            ..
        })
    ));
    // An IIS concerns the constraints alone: the program's are feasible.
    assert_eq!(solver.iis().unwrap(), None);

    // A semi program is a MILP, which has no duals.
    let (program, ..) = semi_relaxation();
    let mut solver = Solver::new(Highs::new(), program).unwrap();
    solver.solve(Request::new()).unwrap();
    assert!(matches!(
        solver.duals(),
        Err(AnalysisError::Class {
            class: ModelClass::Milp,
            ..
        })
    ));
}
