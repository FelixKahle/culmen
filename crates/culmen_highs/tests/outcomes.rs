// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Outcomes, statuses, options and errors of the HiGHS backend.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{
    Backend, Integrality, LinExpr, Linear, Model, Outcome, Settings, SolveError, Solver, StopReason, Var, solutions,
    threads,
};
use culmen_highs::{
    CallStatus, Dimension, Highs, HighsError, HighsOptions, ModelStatus, OptionValue, RawOption, SimplexStrategy,
    SolutionStatus, Version,
};

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
fn test_infeasible_lp_and_mip() {
    for integer in [false, true] {
        let mut solver = Solver::new(Highs::new(), infeasible(integer).build().unwrap()).unwrap();
        let solved = solver.solve().unwrap();
        assert!(
            matches!(solved.outcome, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded),
            "integer {integer}: {solved:?}"
        );
        assert!(matches!(
            solved.status.model,
            ModelStatus::Infeasible | ModelStatus::UnboundedOrInfeasible
        ));
    }
}

#[test]
fn test_unbounded_lp_and_mip() {
    for integer in [false, true] {
        let mut m = Model::new();
        let x = m.add_var(if integer { Var::integer() } else { Var::continuous() });
        let y = m.add_var(Var::continuous().upper(1.0));
        m.add_constraint((x - y).geq(0.0));
        m.maximize(x + y);
        let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
        let solved = solver.solve().unwrap();
        assert!(
            matches!(solved.outcome, Outcome::Unbounded | Outcome::InfeasibleOrUnbounded),
            "integer {integer}: {solved:?}"
        );
    }
}

#[test]
fn test_lp_optimum_and_status_with_offset() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + y).leq(4.0));
    m.add_constraint((x + 3.0 * y).leq(6.0));
    m.maximize(3.0 * x + 2.0 * y + 10);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    let solved = solver.solve().unwrap();
    let Outcome::Optimal(solution) = &solved.outcome else {
        panic!("{solved:?}")
    };
    assert!((solution.objective - 22.0).abs() < 1e-9);
    assert!((solution.values[x] - 4.0).abs() < 1e-9);
    assert_eq!(solved.status.run, CallStatus::Ok);
    assert_eq!(solved.status.model, ModelStatus::Optimal);
    assert_eq!(
        (solved.status.primal, solved.status.dual),
        (SolutionStatus::Feasible, SolutionStatus::Feasible)
    );
    assert_eq!(solved.status.mip_gap, None);
    assert!(solved.status.simplex_iterations > 0);
}

#[test]
fn test_empty_model_is_optimal_at_its_offset() {
    let mut m = Model::new();
    m.minimize(culmen_core::LinExpr::from(3.5));
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    let solved = solver.solve().unwrap();
    assert_eq!(solved.status.model, ModelStatus::ModelEmpty);
    let Outcome::Optimal(solution) = solved.outcome else {
        panic!()
    };
    assert_eq!((solution.objective, solution.values.len()), (3.5, 0));
}

#[test]
fn test_empty_model_with_a_false_row_is_infeasible() {
    // Without columns every row's activity is zero, so a row whose bounds exclude zero has no satisfying point.
    for (lower, upper) in [(1.0, 1.0), (f64::NEG_INFINITY, -1.0), (0.5, f64::INFINITY)] {
        let mut m = Model::new();
        m.add_constraint(culmen_core::LinExpr::from(0.0).between(lower, upper));
        let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
        let solved = solver.solve().unwrap();
        assert_eq!(solved.outcome, Outcome::Infeasible, "{lower} ..= {upper}");
    }
}

#[test]
fn test_empty_model_with_true_rows_is_optimal_at_its_offset() {
    let mut m = Model::new();
    m.add_constraint(culmen_core::LinExpr::from(0.0).equals(0));
    m.add_constraint(culmen_core::LinExpr::from(2.0).between(-1.0, 3.0));
    m.minimize(culmen_core::LinExpr::from(-1.5));
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    let Outcome::Optimal(solution) = solver.solve().unwrap().outcome else {
        panic!()
    };
    assert_eq!((solution.objective, solution.values.len()), (-1.5, 0));
}

#[test]
fn test_options_are_applied_and_refused_by_name() {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);
    let program = m.build().unwrap();
    let options = HighsOptions {
        simplex_strategy: Some(SimplexStrategy::Primal),
        ..HighsOptions::default()
    };
    let mut solver = Solver::new(Highs::with_options(options), program.clone()).unwrap();
    assert!(matches!(solver.solve().unwrap().outcome, Outcome::Optimal(_)));

    solver.options_mut().raw.push(RawOption {
        name: "no_such_option".to_owned(),
        value: OptionValue::Int(1),
    });
    let error = solver.solve().unwrap_err();
    assert_eq!(
        error,
        SolveError::Backend(HighsError::Option {
            name: "no_such_option".to_owned(),
            value: OptionValue::Int(1)
        })
    );

    let wrong_kind = HighsOptions::default().option("mip_rel_gap", "large");
    let mut solver = Solver::new(Highs::with_options(wrong_kind), program).unwrap();
    assert!(matches!(
        solver.solve(),
        Err(SolveError::Backend(HighsError::Option { .. }))
    ));
}

#[test]
fn test_a_nul_byte_is_blamed_on_the_name_or_the_value_that_holds_it() {
    let program = infeasible(false).build().unwrap();
    let solve = |options: HighsOptions| match Solver::new(Highs::with_options(options), program.clone())
        .unwrap()
        .solve()
    {
        Err(SolveError::Backend(error)) => error,
        other => panic!("{other:?}"),
    };
    let typed = HighsOptions {
        log_file: Some("a\0b".to_owned()),
        ..HighsOptions::default()
    };
    for options in [typed, HighsOptions::default().option("log_file", "a\0b")] {
        let error = solve(options);
        assert_eq!(
            error,
            HighsError::OptionString {
                name: "log_file".to_owned()
            }
        );
        assert_eq!(
            error.to_string(),
            "HiGHS: the value of option `log_file` holds a NUL byte"
        );
    }
    let error = solve(HighsOptions::default().option("log\0file", 1));
    assert_eq!(
        error,
        HighsError::OptionName {
            name: "log\0file".to_owned()
        }
    );
}

#[test]
fn test_errors_name_the_dimension_and_the_model_status() {
    let too_large = HighsError::TooLarge {
        what: Dimension::Rows,
        count: 1 << 40,
    };
    assert_eq!(
        too_large.to_string(),
        "HiGHS: 1099511627776 rows exceed HiGHS's 32-bit indices"
    );
    let run = HighsError::Run {
        status: ModelStatus::SolveError,
        message: Some("no".to_owned()),
    };
    assert_eq!(run.to_string(), "HiGHS: the run failed (solve error): no");
    assert_eq!(ModelStatus::TimeLimit.to_string(), "time limit reached");
}

#[test]
fn test_solving_before_loading_is_refused() {
    let mut highs = Highs::new();
    assert_eq!(
        highs.solve(&Settings::new(), culmen_core::Hooks::new()).unwrap_err(),
        HighsError::NotLoaded
    );
}

#[test]
fn test_a_thread_count_conflict_on_one_thread_is_an_error() {
    // HiGHS's scheduler belongs to the thread; this test's thread starts it at two threads, then asks for three.
    let mut solver = Solver::new(Highs::new(), infeasible(true).build().unwrap()).unwrap();
    let on = |threads| Settings {
        threads: Some(threads),
        ..Settings::new()
    };
    solver.run().settings(on(threads!(2))).solve().unwrap();
    let error = solver.run().settings(on(threads!(3))).solve().unwrap_err();
    let SolveError::Backend(HighsError::Run { message, .. }) = error else {
        panic!("{error:?}")
    };
    assert!(message.unwrap().contains("threads"));
}

#[test]
fn test_dropping_the_backend_frees_the_threads_scheduler_for_another_count() {
    // A thread of its own, so no other test's scheduler is in play: once the solver that started the scheduler at two
    // threads is dropped, a new one on the same thread may ask for three.
    let on = |threads| Settings {
        threads: Some(threads),
        ..Settings::new()
    };
    std::thread::spawn(move || {
        let mut first = Solver::new(Highs::new(), infeasible(true).build().unwrap()).unwrap();
        first.run().settings(on(threads!(2))).solve().unwrap();
        drop(first);
        let mut second = Solver::new(Highs::new(), infeasible(true).build().unwrap()).unwrap();
        second.run().settings(on(threads!(3))).solve().unwrap();
    })
    .join()
    .unwrap();
}

#[test]
fn test_a_moved_backend_solves_on_another_thread() {
    let mut solver = Solver::new(Highs::new(), infeasible(false).build().unwrap()).unwrap();
    let solved = std::thread::spawn(move || solver.solve().unwrap()).join().unwrap();
    assert!(matches!(
        solved.outcome,
        Outcome::Infeasible | Outcome::InfeasibleOrUnbounded
    ));
}

#[test]
fn test_a_seed_beyond_highs_integers_is_refused() {
    let program = infeasible(true).build().unwrap();
    let settings = Settings {
        seed: Some(u32::MAX),
        ..Settings::new()
    };
    let refused = HighsError::OutOfRange {
        name: "random_seed",
        value: 4_294_967_295,
    };
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let solved = solver.run().settings(settings).solve();
    assert_eq!(solved.unwrap_err(), SolveError::Backend(refused.clone()));

    assert_eq!(
        refused.to_string(),
        "HiGHS: option `random_seed` takes at most 2147483647, not 4294967295"
    );

    // An exact integer field of the options is refused alike.
    let options = HighsOptions {
        simplex_max_concurrency: Some(u32::MAX),
        ..HighsOptions::default()
    };
    let mut solver = Solver::new(Highs::with_options(options), program).unwrap();
    let solved = solver.solve();
    let refused = HighsError::OutOfRange {
        name: "simplex_max_concurrency",
        value: 4_294_967_295,
    };
    assert_eq!(solved.unwrap_err(), SolveError::Backend(refused));
}

#[cfg(feature = "serde")]
#[test]
fn test_options_and_statuses_round_trip() {
    fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + core::fmt::Debug>(value: &T) {
        let json = serde_json::to_string(value).unwrap();
        assert_eq!(&serde_json::from_str::<T>(&json).unwrap(), value, "{json}");
    }
    let options = HighsOptions {
        parallel: Some(culmen_highs::Switch::Off),
        presolve_mode: Some(culmen_highs::PresolveMode::Light),
        mip_heuristic_effort: Some(0.2),
        ..HighsOptions::default()
    }
    .option("mip_pool_soft_limit", 100)
    .option("log_file", "highs.log");
    round_trip(&options);
    round_trip(&options.raw[1]);
    let mut solver = Solver::new(Highs::new(), infeasible(true).build().unwrap()).unwrap();
    let solved = solver.solve().unwrap();
    round_trip(&solved.status);
    round_trip(&solved);
    round_trip(&ModelStatus::Other(42));
    round_trip(&culmen_highs::version());
    // HiGHS's own default objective bound is +inf, and its infinite bound a double option set by name.
    let infinite = HighsOptions {
        objective_bound: Some(f64::INFINITY),
        objective_target: Some(f64::NEG_INFINITY),
        ..HighsOptions::default()
    }
    .option("infinite_bound", f64::INFINITY);
    round_trip(&infinite);
    round_trip(&culmen_highs::OptionValue::Double(f64::NEG_INFINITY));
    use culmen_highs::{EdgeWeight, LpSolver, MipIpmSolver, MipLpSolver, OnOff, ScaleStrategy, files::ModelFormat};
    for format in [ModelFormat::Mps, ModelFormat::Lp] {
        round_trip(&format);
    }
    round_trip(&OnOff::On);
    round_trip(&LpSolver::QpAsm);
    round_trip(&MipLpSolver::Hipo);
    round_trip(&MipIpmSolver::Ipx);
    round_trip(&SimplexStrategy::DualPami);
    round_trip(&ScaleStrategy::ForcedEquilibration);
    round_trip(&EdgeWeight::SteepestEdge);
    round_trip(&culmen_core::LogLevel::Warning);
}

#[test]
fn test_versions_order_and_display_as_versions_do() {
    let version = |major, minor, patch| Version { major, minor, patch };
    assert!(version(1, 9, 3) < version(1, 15, 0));
    assert!(version(1, 15, 0) < version(2, 0, 0));
    assert_eq!(version(1, 15, 2).to_string(), "1.15.2");
}

#[test]
fn test_a_solution_limit_ends_a_mip_with_a_solution() {
    // A knapsack: every limit of at least one solution ends with one, optimal or stopped at the limit.
    let mut m = Model::new();
    let take = m.add_vars(6, |_| Var::binary());
    let weights = [5.0, 4.0, 6.0, 3.0, 7.0, 2.0];
    m.add_constraint(LinExpr::sum(take.iter().zip(weights).map(|(&x, w)| w * x)).leq(12.0));
    m.maximize(LinExpr::sum(
        take.iter().zip([7.0, 5.0, 8.0, 4.0, 9.0, 2.0]).map(|(&x, v)| v * x),
    ));
    let program = m.build().unwrap();
    let limit = Settings {
        solution_limit: Some(solutions!(1)),
        ..Settings::new()
    };
    let in_settings = Solver::new(Highs::new(), program.clone())
        .unwrap()
        .run()
        .settings(limit)
        .solve()
        .unwrap();
    let options = HighsOptions::default().option("mip_max_improving_sols", 1);
    let in_options = Solver::new(Highs::with_options(options), program)
        .unwrap()
        .solve()
        .unwrap();
    for solved in [in_settings, in_options] {
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
}

/// A generalized assignment HiGHS does not finish at its root.
fn hard_mip() -> culmen_core::Program {
    culmen_testing::instances::assignment(60, 30, 12_345, 0.0).program
}

/// The reason a solve that is expected to stop gives.
fn stop_reason(solved: &culmen_core::Solved<culmen_highs::HighsStatus>) -> StopReason {
    match &solved.outcome {
        Outcome::Stopped { reason, .. } => *reason,
        other => panic!("expected a stop, got {other:?}"),
    }
}

#[test]
fn test_a_node_limit_stops_a_mip_at_the_node_limit_however_it_is_set() {
    let limited = Settings {
        node_limit: Some(1),
        time_limit: Some(core::time::Duration::from_secs(60)),
        ..Settings::new()
    };
    let in_settings = Solver::new(Highs::new(), hard_mip())
        .unwrap()
        .run()
        .settings(limited)
        .solve()
        .unwrap();
    assert_eq!(stop_reason(&in_settings), StopReason::NodeLimit, "{in_settings:?}");
    let by_name = HighsOptions::default().option("mip_max_nodes", 1);
    let solved = Solver::new(Highs::with_options(by_name), hard_mip())
        .unwrap()
        .run()
        .time_limit(core::time::Duration::from_secs(60))
        .solve()
        .unwrap();
    assert_eq!(stop_reason(&solved), StopReason::NodeLimit, "{solved:?}");
}

#[test]
fn test_an_objective_target_stops_a_mip_at_its_first_good_enough_solution() {
    // Every assignment walks less than a billion, so the first solution meets the target.
    let target = HighsOptions {
        objective_target: Some(1e9),
        ..HighsOptions::default()
    };
    let solved = Solver::new(Highs::with_options(target), hard_mip())
        .unwrap()
        .run()
        .time_limit(core::time::Duration::from_secs(60))
        .solve()
        .unwrap();
    assert_eq!(stop_reason(&solved), StopReason::ObjectiveLimit, "{solved:?}");
    assert!(solved.outcome.objective().is_some_and(|objective| objective <= 1e9));
}

#[test]
fn test_an_objective_bound_stops_the_dual_simplex_once_the_lp_cannot_beat_it() {
    // The assignment's LP relaxation walks far more than 1, which the dual simplex proves on its way to the optimum.
    let mut plan = culmen_core::ProgramPlan::from(hard_mip());
    for column in &mut plan.columns {
        column.integrality = culmen_core::Integrality::Continuous;
    }
    let bounded = HighsOptions {
        objective_bound: Some(1.0),
        solver: Some(culmen_highs::LpSolver::Simplex),
        simplex_strategy: Some(SimplexStrategy::Dual),
        ..HighsOptions::default()
    };
    let solved = Solver::new(Highs::with_options(bounded), plan.build().unwrap())
        .unwrap()
        .run()
        .settings(Settings {
            presolve: Some(false),
            ..Settings::new()
        })
        .solve()
        .unwrap();
    assert_eq!(stop_reason(&solved), StopReason::ObjectiveLimit, "{solved:?}");
}

#[test]
fn test_a_coefficient_highs_finds_excessive_is_refused_at_load_and_at_an_edit() {
    // HiGHS refuses a matrix value at or above its `large_matrix_value`, 1e15.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    m.add_constraint((1e16 * x).leq(1));
    m.maximize(x);
    let refused = Solver::new(Highs::new(), m.build().unwrap()).unwrap_err();
    assert_eq!(refused, culmen_core::LoadError::Backend(HighsError::Load));
    assert_eq!(HighsError::Load.to_string(), "HiGHS: the program was refused");

    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    m.maximize(x);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    let edit = solver.add_row((1e16 * x).leq(1)).unwrap_err();
    assert_eq!(
        edit,
        culmen_core::EditError::Backend(HighsError::Edit { call: "Highs_addRows" })
    );
    assert_eq!(
        HighsError::Edit { call: "Highs_addRows" }.to_string(),
        "HiGHS: Highs_addRows refused the edit"
    );
    // The solver holds the program before the edit again.
    assert_eq!(solver.solve().unwrap().outcome.objective(), Some(1.0));
}

#[test]
fn test_a_mip_never_reports_its_relaxations_optimum_as_its_own() {
    // The relaxation's optimum is x = y = 3/4, worth 3/2; the MIP's takes one of the two, worth 1.
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    m.add_constraint((2.0 * x + 2.0 * y).leq(3.0));
    m.maximize(x + y);
    let relaxing = HighsOptions::default().option("solve_relaxation", true);
    let mut solver = Solver::new(Highs::with_options(relaxing), m.build().unwrap()).unwrap();
    assert_eq!(solver.solve(), Err(SolveError::Backend(HighsError::Relaxation)));
    assert_eq!(
        HighsError::Relaxation.to_string(),
        "HiGHS: the option `solve_relaxation` would report a MIP's continuous relaxation as its optimum"
    );

    // Relaxing the program itself solves the relaxation, reported as the optimum of the program it is.
    solver
        .set_col_integrality([(x, Integrality::Continuous), (y, Integrality::Continuous)])
        .unwrap();
    assert_eq!(solver.solve().unwrap().outcome.objective(), Some(1.5));
}
