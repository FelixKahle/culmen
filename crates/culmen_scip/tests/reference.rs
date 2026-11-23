// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SCIP against the exact reference solver on tiny LPs and MILPs.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{Outcome, Request, Settings, Solver, relative_gap};
use culmen_scip::Scip;
use culmen_testing::reference::{Reference, close, solve};
use culmen_testing::strategy::small_lp;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// SCIP proves the same optimum as exhaustive exact enumeration, or infeasibility where enumeration finds no
    /// point; its solution is feasible and its objective is the solution's.
    #[test]
    fn test_scip_agrees_with_exact_enumeration(program in small_lp()) {
        let expected = solve(&program);
        let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
        let exact = Settings { relative_gap: Some(relative_gap!(0.0)), ..Settings::new() };
        let solved = solver.solve(Request::new().settings_from(exact)).unwrap();
        match (&expected, &solved.outcome) {
            (Reference::Infeasible, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded) => {}
            (Reference::Optimal(_), Outcome::Optimal(solution)) => {
                let want = expected.objective().unwrap();
                // SCIP meets rows and bounds to its feasibility tolerance, `numerics/feastol` = 1e-6, which rounding
                // can exceed; both checks allow ten times that tolerance.
                prop_assert!(close(solution.objective, want, 1e-5), "SCIP {} vs exact {}", solution.objective, want);
                prop_assert!(program.max_violation(&solution.values) <= 1e-5, "an infeasible solution: {solution:?}");
                prop_assert!(close(program.objective_value(&solution.values), solution.objective, 1e-6));
            }
            (want, got) => prop_assert!(false, "exact {want:?}, SCIP {got:?}, status {:?}", solved.status),
        }
    }
}
