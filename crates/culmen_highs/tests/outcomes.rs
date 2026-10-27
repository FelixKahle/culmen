// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Outcomes, statuses, options and errors of the HiGHS backend.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{Backend, Linear, Model, Outcome, Request, Settings, SolveError, Solver, Var, threads};
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
        let solved = solver.solve(Request::new()).unwrap();
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
        let solved = solver.solve(Request::new()).unwrap();
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
    let solved = solver.solve(Request::new()).unwrap();
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
    let solved = solver.solve(Request::new()).unwrap();
    assert_eq!(solved.status.model, ModelStatus::ModelEmpty);
    let Outcome::Optimal(solution) = solved.outcome else {
        panic!()
    };
    assert_eq!((solution.objective, solution.values.len()), (3.5, 0));
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
    assert!(matches!(
        solver.solve(Request::new()).unwrap().outcome,
        Outcome::Optimal(_)
    ));

    solver.backend_mut().options_mut().raw.push(RawOption {
        name: "no_such_option".to_owned(),
        value: OptionValue::Int(1),
    });
    let error = solver.solve(Request::new()).unwrap_err();
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
        solver.solve(Request::new()),
        Err(SolveError::Backend(HighsError::Option { .. }))
    ));
}

#[test]
fn test_a_nul_byte_is_blamed_on_the_name_or_the_value_that_holds_it() {
    let program = infeasible(false).build().unwrap();
    let solve = |options: HighsOptions| match Solver::new(Highs::with_options(options), program.clone())
        .unwrap()
        .solve(Request::new())
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
        highs.solve(&Settings::new(), &mut ()).unwrap_err(),
        HighsError::NotLoaded
    );
}

#[test]
fn test_a_thread_count_conflict_on_one_thread_is_an_error() {
    // HiGHS's scheduler belongs to the thread; this test's thread starts it at two threads, then asks for three.
    let mut solver = Solver::new(Highs::new(), infeasible(true).build().unwrap()).unwrap();
    let on = |threads| {
        Request::new().settings_from(Settings {
            threads: Some(threads),
            ..Settings::new()
        })
    };
    solver.solve(on(threads!(2))).unwrap();
    let error = solver.solve(on(threads!(3))).unwrap_err();
    let SolveError::Backend(HighsError::Run { message, .. }) = error else {
        panic!("{error:?}")
    };
    assert!(message.unwrap().contains("threads"));
}

#[test]
fn test_a_moved_backend_solves_on_another_thread() {
    let mut solver = Solver::new(Highs::new(), infeasible(false).build().unwrap()).unwrap();
    let solved = std::thread::spawn(move || solver.solve(Request::new()).unwrap())
        .join()
        .unwrap();
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
    let solved = solver.solve(Request::new().settings_from(settings));
    assert_eq!(solved.unwrap_err(), SolveError::Backend(refused.clone()));

    let options = HighsOptions {
        random_seed: Some(u32::MAX),
        ..HighsOptions::default()
    };
    let mut solver = Solver::new(Highs::with_options(options), program).unwrap();
    let solved = solver.solve(Request::new());
    assert_eq!(solved.unwrap_err(), SolveError::Backend(refused.clone()));
    assert_eq!(
        refused.to_string(),
        "HiGHS: option `random_seed` takes at most 2147483647, not 4294967295"
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_options_and_statuses_round_trip() {
    fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + core::fmt::Debug>(value: &T) {
        let json = serde_json::to_string(value).unwrap();
        assert_eq!(&serde_json::from_str::<T>(&json).unwrap(), value, "{json}");
    }
    let options = HighsOptions {
        presolve: Some(culmen_highs::Switch::Off),
        time_limit: Some(core::time::Duration::from_secs(3)),
        mip_rel_gap: Some(0.01),
        ..HighsOptions::default()
    }
    .option("mip_pool_soft_limit", 100)
    .option("log_file", "highs.log");
    round_trip(&options);
    round_trip(&options.raw[1]);
    let mut solver = Solver::new(Highs::new(), infeasible(true).build().unwrap()).unwrap();
    let solved = solver.solve(Request::new()).unwrap();
    round_trip(&solved.status);
    round_trip(&solved);
    round_trip(&ModelStatus::Other(42));
    round_trip(&culmen_highs::version());
}

#[test]
fn test_versions_order_and_display_as_versions_do() {
    let version = |major, minor, patch| Version { major, minor, patch };
    assert!(version(1, 9, 3) < version(1, 15, 0));
    assert!(version(1, 15, 0) < version(2, 0, 0));
    assert_eq!(version(1, 15, 2).to_string(), "1.15.2");
}
