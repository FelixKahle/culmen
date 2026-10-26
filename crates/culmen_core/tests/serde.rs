// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every serializable type survives a round trip through JSON, and checked types are checked again as they are
//! read.
#![cfg(feature = "serde")]
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::fmt::Debug;
use core::time::Duration;

use culmen_core::{
    AbsoluteGap, Bounds, Col, Integrality, Linear, Lp, LpPlan, Model, Outcome, Phase, Progress, RelativeGap, Row,
    Sense, Settings, Solution, Solved, StopReason, Var, absolute_gap, bounds, relative_gap, threads,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).unwrap();
    let back: T = serde_json::from_str(&json).unwrap();
    assert_eq!(&back, value, "{json}");
}

fn sample_lp() -> Lp {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(0.0, 4.0).name("x"));
    let y = m.add_var(Var::free());
    m.add_named_constraint("c", (x + 2.0 * y).leq(3.0));
    m.add_constraint((x - y).geq(-1.0));
    m.maximize(x + y + 1.5);
    m.build().unwrap()
}

#[test]
fn test_plain_data_round_trips() {
    round_trip(&bounds!(-1.0, 2.0));
    round_trip(&Bounds::FREE);
    round_trip(&Integrality::Integer);
    round_trip(&Sense::Maximize);
    round_trip(&Col::new(3));
    round_trip(&Row::new(1));
    round_trip(&relative_gap!(0.01));
    round_trip(&absolute_gap!(2.0));
    round_trip(&sample_lp());
    round_trip(&LpPlan::from(sample_lp()));
    round_trip(&Settings {
        time_limit: Some(Duration::from_millis(1500)),
        threads: Some(threads!(2)),
        ..Settings::new()
    });
    let progress = Progress {
        objective: Some(3.0),
        bound: None,
        ..Progress::at(Duration::from_secs(2), Phase::Mip)
    };
    round_trip(&progress);
    let solution = Solution {
        objective: 2.5,
        values: vec![1.0, 1.5].into(),
    };
    round_trip(&Outcome::Optimal(solution.clone()));
    round_trip(&Outcome::GapReached {
        solution: solution.clone(),
        bound: 3.0,
    });
    round_trip(&Outcome::Stopped {
        reason: StopReason::TimeLimit,
        best: Some(solution),
        bound: None,
    });
    round_trip(&Outcome::InfeasibleOrUnbounded);
    round_trip(&Solved {
        outcome: Outcome::Infeasible,
        status: 8_i32,
        elapsed: Duration::from_secs(1),
    });
}

#[test]
fn test_checked_types_are_checked_as_they_are_read() {
    assert!(serde_json::from_str::<Bounds>(r#"{"lower":2.0,"upper":1.0}"#).is_err());
    let error = serde_json::from_str::<RelativeGap>("-0.5").unwrap_err();
    assert!(error.to_string().starts_with("RelativeGap: below zero"), "{error}");
    let error = serde_json::from_str::<AbsoluteGap>("-1").unwrap_err();
    assert!(error.to_string().starts_with("AbsoluteGap: below zero"), "{error}");
    let mut plan = LpPlan::from(sample_lp());
    plan.rows[Row::new(0)].terms.push((Col::new(9), 1.0));
    let json = serde_json::to_string(&plan).unwrap();
    assert!(
        serde_json::from_str::<Lp>(&json).is_err(),
        "a term on an unknown column is refused"
    );
}
