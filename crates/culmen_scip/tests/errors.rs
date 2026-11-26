// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the SCIP backend refuses, and why: settings beyond SCIP's parameters, parameters SCIP refuses, numbers SCIP
//! would read as infinite, names C cannot carry, programs it does not solve, and calls out of order.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::num::NonZeroU64;

use culmen_core::{
    Backend, Bounds, Col, Hooks, Linear, LoadError, Model, Program, ProgramColumn, ProgramFeature, ProgramPlan,
    Request, Row, Settings, SolveError, Solver, Start, Starts, Var, bounds,
};
use culmen_scip::{Number, ParamValue, RetCode, Scip, ScipError, ScipOptions};

fn knapsack() -> Program {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    m.add_constraint((2.0 * x + 3.0 * y).leq(4.0));
    m.maximize(x + 2.0 * y);
    m.build().unwrap()
}

fn refusal(scip: Scip, settings: Settings) -> ScipError {
    match Solver::new(scip, knapsack())
        .unwrap()
        .solve(Request::new().settings_from(settings))
    {
        Err(SolveError::Backend(error)) => error,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn test_settings_beyond_scips_parameters_are_refused_by_name() {
    let seed = refusal(
        Scip::new(),
        Settings {
            seed: Some(u32::MAX),
            ..Settings::new()
        },
    );
    assert_eq!(
        seed,
        ScipError::OutOfRange {
            name: "randomization/randomseedshift",
            value: 4_294_967_295,
            max: 2_147_483_647
        }
    );
    assert_eq!(
        seed.to_string(),
        "SCIP: parameter `randomization/randomseedshift` takes at most 2147483647, not 4294967295"
    );
    let nodes = refusal(
        Scip::new(),
        Settings {
            node_limit: Some(u64::MAX),
            ..Settings::new()
        },
    );
    assert_eq!(
        nodes,
        ScipError::OutOfRange {
            name: "limits/nodes",
            value: u64::MAX,
            max: 9_223_372_036_854_775_807
        }
    );
    let solutions = NonZeroU64::new(1 << 31);
    let solutions = refusal(
        Scip::new(),
        Settings {
            solution_limit: solutions,
            ..Settings::new()
        },
    );
    assert_eq!(
        solutions,
        ScipError::OutOfRange {
            name: "limits/bestsol",
            value: 1 << 31,
            max: 2_147_483_647
        }
    );
}

#[test]
fn test_the_largest_values_scips_parameters_take_are_accepted() {
    let settings = Settings {
        seed: Some(2_147_483_647),
        node_limit: Some(9_223_372_036_854_775_807),
        solution_limit: NonZeroU64::new(2_147_483_647),
        ..Settings::new()
    };
    let solved = Solver::new(Scip::new(), knapsack())
        .unwrap()
        .solve(Request::new().settings_from(settings))
        .unwrap();
    assert_eq!(solved.outcome.objective(), Some(2.0));
}

#[test]
fn test_parameters_scip_refuses_name_the_parameter_and_scips_code() {
    let with = |name: &str, value: ParamValue| {
        refusal(
            Scip::with_options(ScipOptions::default().param(name, value)),
            Settings::new(),
        )
    };
    assert_eq!(
        with("no/such/parameter", ParamValue::Int(1)),
        ScipError::Param {
            name: "no/such/parameter".to_owned(),
            value: ParamValue::Int(1),
            code: RetCode::ParameterUnknown
        }
    );
    assert_eq!(
        with("limits/nodes", ParamValue::Int(1)),
        ScipError::Param {
            name: "limits/nodes".to_owned(),
            value: ParamValue::Int(1),
            code: RetCode::ParameterWrongType
        }
    );
    let negative = with("limits/time", ParamValue::Real(-1.0));
    assert_eq!(
        negative,
        ScipError::Param {
            name: "limits/time".to_owned(),
            value: ParamValue::Real(-1.0),
            code: RetCode::ParameterWrongValue
        }
    );
    assert_eq!(
        negative.to_string(),
        "SCIP: parameter `limits/time` refused the value -1: the value is invalid for the given parameter"
    );
    assert_eq!(
        core::error::Error::source(&negative).map(ToString::to_string),
        Some(RetCode::ParameterWrongValue.to_string())
    );
}

#[test]
fn test_parameters_of_every_type_are_set() {
    let options = ScipOptions::default()
        .param("misc/catchctrlc", false)
        .param("limits/maxsol", 50)
        .param("limits/totalnodes", 1_000_000_i64)
        .param("limits/memory", 4096.0)
        .param("nodeselection/childsel", 'h')
        .param("visual/vbcfilename", "-");
    let solved = Solver::new(Scip::with_options(options), knapsack())
        .unwrap()
        .solve(Request::new())
        .unwrap();
    assert_eq!(solved.outcome.objective(), Some(2.0));
}

#[test]
fn test_names_and_values_c_cannot_carry_are_refused_before_scip_sees_them() {
    let with = |options: ScipOptions| refusal(Scip::with_options(options), Settings::new());
    assert_eq!(
        with(ScipOptions::default().param("limits\0time", 1.0)),
        ScipError::ParamName {
            name: "limits\0time".to_owned()
        }
    );
    assert_eq!(
        with(ScipOptions::default().param("visual/vbcfilename", "a\0b")),
        ScipError::ParamString {
            name: "visual/vbcfilename".to_owned()
        }
    );
    let char = with(ScipOptions::default().param("nodeselection/childsel", 'é'));
    assert_eq!(
        char,
        ScipError::ParamChar {
            name: "nodeselection/childsel".to_owned(),
            value: 'é'
        }
    );
    assert_eq!(
        char.to_string(),
        "SCIP: the value 'é' of parameter `nodeselection/childsel` is not an ASCII character"
    );
}

/// The program of `plan` loaded into SCIP, or SCIP's refusal.
fn load(plan: ProgramPlan) -> Result<(), ScipError> {
    Scip::new().load(&plan.build().unwrap())
}

#[test]
fn test_numbers_scip_would_read_as_infinite_are_refused_where_they_stand() {
    let infinity = 1e20;
    let refused = |number, value| {
        Err(ScipError::BeyondInfinity {
            number,
            value,
            infinity,
        })
    };
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(1.0, bounds!(0.0, 1e20)));
    assert_eq!(load(plan), refused(Number::ColBound(x), 1e20));

    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(-1e25, Bounds::NON_NEGATIVE));
    assert_eq!(load(plan), refused(Number::Cost(x), -1e25));

    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(1.0, Bounds::FREE));
    let row = plan.row(bounds!(-1e21, 0.0), [(x, 1.0)]);
    assert_eq!(load(plan), refused(Number::RowBound(row), -1e21));

    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
    let row = plan.row(bounds!(0.0, 1.0), [(x, 3e20)]);
    assert_eq!(load(plan), refused(Number::Coefficient { row, col: x }, 3e20));

    let mut plan = ProgramPlan::new();
    plan.offset = 1e20;
    assert_eq!(load(plan), refused(Number::Offset, 1e20));

    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(0.0, Bounds::FREE));
    let y = plan.col(ProgramColumn::continuous(0.0, Bounds::FREE));
    plan.quadratic = vec![(x, x, 1e21), (y, y, 1e21)];
    assert_eq!(
        load(plan),
        refused(Number::QuadCoefficient { first: x, second: x }, 1e21)
    );
    assert_eq!(
        ScipError::BeyondInfinity {
            number: Number::Coefficient {
                row: Row::new(2),
                col: Col::new(1)
            },
            value: 3e20,
            infinity
        }
        .to_string(),
        "SCIP: the coefficient of column 1 in row 2 is 300000000000000000000, which SCIP reads as infinite from \
         100000000000000000000 on"
    );

    // Just below SCIP's infinity, a number is finite to SCIP and loads.
    let mut plan = ProgramPlan::new();
    plan.col(ProgramColumn::continuous(1.0, bounds!(-9.9e19, 9.9e19)));
    assert_eq!(load(plan), Ok(()));
}

#[test]
fn test_a_name_holding_a_nul_byte_is_refused_for_its_column_or_row() {
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn {
        name: Some("x\0".to_owned()),
        ..ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE)
    });
    assert_eq!(load(plan), Err(ScipError::ColName { col: x }));

    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
    let row = plan.row(bounds!(0.0, 1.0), [(x, 1.0)]);
    plan.rows[row].name = Some("r\0w".to_owned());
    let program = plan.build().unwrap();
    assert_eq!(
        Solver::new(Scip::new(), program).unwrap_err(),
        LoadError::Backend(ScipError::RowName { row })
    );
}

#[test]
fn test_a_failed_load_leaves_no_program_and_a_later_load_works() {
    let mut scip = Scip::new();
    scip.load(&knapsack()).unwrap();
    let mut plan = ProgramPlan::new();
    plan.col(ProgramColumn::continuous(1e30, Bounds::NON_NEGATIVE));
    assert!(scip.load(&plan.build().unwrap()).is_err());
    assert_eq!(
        scip.solve(&Settings::new(), Hooks::new()).unwrap_err(),
        ScipError::NotLoaded
    );
    scip.load(&knapsack()).unwrap();
    let solved = scip.solve(&Settings::new(), Hooks::new()).unwrap();
    assert_eq!(solved.outcome.objective(), Some(2.0));
}

#[test]
fn test_programs_with_features_it_does_not_declare_are_refused() {
    let mut m = Model::new();
    m.add_var(Var::semi_continuous().bounds(1, 2));
    let semi = m.build().unwrap();
    assert_eq!(
        Scip::new().load(&semi),
        Err(ScipError::UnsupportedFeature {
            feature: ProgramFeature::SemiContinuous
        })
    );
    assert!(matches!(
        Solver::new(Scip::new(), semi),
        Err(LoadError::UnsupportedFeature {
            feature: ProgramFeature::SemiContinuous,
            ..
        })
    ));
}

#[test]
fn test_solving_or_starting_before_loading_is_refused() {
    let mut scip = Scip::new();
    assert_eq!(
        scip.solve(&Settings::new(), Hooks::new()).unwrap_err(),
        ScipError::NotLoaded
    );
    assert_eq!(scip.set_start(&Start::from(vec![1.0])), Err(ScipError::NotLoaded));
    assert_eq!(ScipError::NotLoaded.to_string(), "SCIP: no program is loaded");
}

#[test]
fn test_a_start_that_does_not_fit_is_refused() {
    let mut scip = Scip::new();
    scip.load(&knapsack()).unwrap();
    assert_eq!(scip.set_start(&Start::from(vec![1.0])), Err(ScipError::Start));
    assert_eq!(
        scip.set_start(&Start::from(vec![(Col::new(2), 1.0)])),
        Err(ScipError::Start)
    );
    assert_eq!(scip.set_start(&Start::from(vec![1.0, 0.0])), Ok(()));
}
