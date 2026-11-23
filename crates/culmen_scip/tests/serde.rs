// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every serializable type of the SCIP backend survives a round trip through JSON.
#![cfg(feature = "serde")]
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::fmt::Debug;

use culmen_core::{Linear, Model, Request, Solver, Var};
use culmen_scip::{Emphasis, ParamValue, RetCode, Scip, ScipOptions, ScipStatus, Status, Version};
use serde::Serialize;
use serde::de::DeserializeOwned;

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).unwrap();
    assert_eq!(&serde_json::from_str::<T>(&json).unwrap(), value, "{json}");
}

#[test]
fn test_options_and_parameters_round_trip() {
    let options = ScipOptions {
        heuristics: Some(Emphasis::Off),
        presolving: Some(Emphasis::Aggressive),
        separating: Some(Emphasis::Fast),
        ..ScipOptions::default()
    }
    .param("misc/catchctrlc", true)
    .param("limits/maxsol", 50)
    .param("limits/totalnodes", 7_i64)
    .param("limits/memory", 0.5)
    .param("nodeselection/childsel", 'h')
    .param("visual/vbcfilename", "tree.vbc");
    round_trip(&options);
    round_trip(&ScipOptions::default());
    for param in &options.params {
        round_trip(param);
        round_trip(&param.value);
    }
    round_trip(&Emphasis::Default);
    round_trip(&ParamValue::String(String::new()));
}

#[test]
fn test_statuses_codes_and_versions_round_trip() {
    for status in [Status::Optimal, Status::GapLimit, Status::Other(99)] {
        round_trip(&status);
    }
    for code in [RetCode::LpError, RetCode::Other(-42)] {
        round_trip(&code);
    }
    round_trip(&ScipStatus {
        status: Status::NodeLimit,
        dual_bound: None,
        nodes: 3,
        lp_iterations: 17,
        solutions_found: 1,
    });
    round_trip(&culmen_scip::version());
    round_trip(&Version {
        major: 10,
        minor: 0,
        patch: 2,
    });
}

#[test]
fn test_a_solved_knapsack_round_trips() {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    m.add_constraint((2.0 * x + 3.0 * y).leq(4.0));
    m.maximize(x + 2.0 * y);
    let solved = Solver::new(Scip::new(), m.build().unwrap())
        .unwrap()
        .solve(Request::new())
        .unwrap();
    round_trip(&solved.status);
    round_trip(&solved);
}
