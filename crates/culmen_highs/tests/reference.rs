// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HiGHS against the exact reference solver on tiny LPs and MILPs.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{Outcome, Request, Settings, Solver, relative_gap};
use culmen_highs::Highs;
use culmen_testing::reference::{Reference, close, solve};
use culmen_testing::strategy::small_lp;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// HiGHS proves the same optimum as exhaustive exact enumeration, or infeasibility where enumeration finds no
    /// point; its solution is feasible and its objective is the solution's.
    #[test]
    fn test_highs_agrees_with_exact_enumeration(program in small_lp()) {
        let expected = solve(&program);
        let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
        let solved = solver.solve(Request::new().settings_from(Settings { relative_gap: Some(relative_gap!(0.0)), ..Settings::new() })).unwrap();
        match (&expected, &solved.outcome) {
            (Reference::Infeasible, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded) => {}
            (Reference::Optimal(_), Outcome::Optimal(solution)) => {
                let want = expected.objective().unwrap();
                prop_assert!(close(solution.objective, want, 1e-6), "HiGHS {} vs exact {}", solution.objective, want);
                prop_assert!(program.max_violation(&solution.values) <= 1e-6, "an infeasible solution: {solution:?}");
                prop_assert!(close(program.objective_value(&solution.values), solution.objective, 1e-6));
            }
            (want, got) => prop_assert!(false, "exact {want:?}, HiGHS {got:?}, status {:?}", solved.status),
        }
    }
}
