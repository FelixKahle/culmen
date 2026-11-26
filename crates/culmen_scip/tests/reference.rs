// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SCIP against the exact reference solver on tiny LPs and MILPs: plain, from starts, after edits, and refusing the
//! semi columns it does not take.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{
    Backend, Col, Integrality, LoadError, Outcome, Program, ProgramFeature, Settings, Solver, relative_gap,
};
use culmen_scip::Scip;
use culmen_testing::reference::{Reference, close, edit_plan, solve};
use culmen_testing::strategy::{small_lp, small_semi, with_edits};
use culmen_testing::tolerance::Tolerances;
use proptest::prelude::*;

/// A solve to a zero gap, so an optimum is the optimum.
fn exact() -> Settings {
    Settings {
        relative_gap: Some(relative_gap!(0.0)),
        ..Settings::new()
    }
}

/// Whether SCIP's outcome is the exact reference's: infeasible alike, or an optimum in the window SCIP's tolerances
/// leave around the exact objective, whose point meets the program to those tolerances and is valued as the program
/// values it.
fn agrees(program: &Program, expected: &Reference, outcome: &Outcome) -> Result<(), TestCaseError> {
    match (Tolerances::SCIP.window(program, expected), outcome) {
        (None, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded) => Ok(()),
        (Some(window), Outcome::Optimal(solution)) => {
            prop_assert!(
                window.contains(solution.objective),
                "SCIP {} vs exact {expected:?}, window {window:?}",
                solution.objective
            );
            let stray = Tolerances::SCIP.stray(program, &solution.values);
            prop_assert!(stray.is_none(), "an infeasible solution: {stray:?} {solution:?}");
            prop_assert!(close(
                program.objective_value(&solution.values),
                solution.objective,
                1e-6
            ));
            Ok(())
        }
        (_, got) => Err(TestCaseError::fail(format!("exact {expected:?}, SCIP {got:?}"))),
    }
}

/// The exact reference of a program small enough to enumerate: every bound finite, at most three continuous columns
/// and at most two thousand points of whole columns; `None` for any other.
fn enumerable(program: &Program) -> Option<Reference> {
    let columns = program.columns();
    let finite = columns
        .iter()
        .all(|c| c.bounds.lower().is_finite() && c.bounds.upper().is_finite());
    if !program.quadratic().is_empty() || !finite || columns.len() > 7 {
        return None;
    }
    let continuous = columns
        .iter()
        .filter(|c| c.integrality == Integrality::Continuous)
        .count();
    let points: f64 = columns
        .iter()
        .filter(|c| c.integrality != Integrality::Continuous)
        .map(|c| (c.bounds.upper().floor() - c.bounds.lower().ceil() + 2.0).max(1.0))
        .product();
    (continuous <= 3 && points <= 2000.0).then(|| solve(program))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// SCIP proves the same optimum as exhaustive exact enumeration, or infeasibility where enumeration finds no
    /// point; its solution is feasible and its objective is the solution's.
    #[test]
    fn test_scip_agrees_with_exact_enumeration(program in small_lp()) {
        let expected = solve(&program);
        let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
        let solved = solver.run().settings(exact()).solve().unwrap();
        agrees(&program, &expected, &solved.outcome)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// A full start at SCIP's own optimum, and a sparse start of any values, feasible or not, leave the exact
    /// optimum.
    #[test]
    fn test_starts_leave_the_exact_optimum(
        program in small_lp(),
        picks in prop::collection::vec((0usize..4, -4i32..=5), 0..=4),
    ) {
        let expected = solve(&program);
        let mut solver = Solver::new(Scip::new(), program.clone()).unwrap();
        let first = solver.run().settings(exact()).solve().unwrap().outcome;
        agrees(&program, &expected, &first)?;
        if let Some(best) = first.solution() {
            let again = solver.run().settings(exact()).start(best.values.clone()).solve().unwrap();
            agrees(&program, &expected, &again.outcome)?;
        }
        let mut sparse: Vec<(Col, f64)> = Vec::new();
        for (j, value) in picks {
            if j < program.num_cols() && sparse.iter().all(|&(col, _)| col.index() != j) {
                sparse.push((Col::new(j), f64::from(value)));
            }
        }
        let solved = Solver::new(Scip::new(), program.clone()).unwrap().run().settings(exact()).start(sparse).solve().unwrap();
        agrees(&program, &expected, &solved.outcome)?;
    }

    /// A program with a semi column is refused, naming the feature, before SCIP sees it.
    #[test]
    fn test_semi_programs_are_refused_typed(program in small_semi()) {
        let error = Solver::new(Scip::new(), program).unwrap_err();
        prop_assert!(
            matches!(
                error,
                LoadError::UnsupportedFeature {
                    feature: ProgramFeature::SemiContinuous | ProgramFeature::SemiInteger,
                    supported,
                } if supported == Scip::FEATURES
            ),
            "{error:?}"
        );
    }

    /// After every edit, accepted or refused, the solver holds the program the edit rules give, and SCIP reaches the
    /// exact optimum of it.
    #[test]
    fn test_an_edited_program_reaches_the_exact_optimum((program, ops) in with_edits(small_lp())) {
        let mut solver = Solver::new(Scip::new(), program).unwrap();
        for op in &ops {
            let before = solver.program().clone();
            let want = edit_plan(&before, op, Scip::CLASSES, Scip::FEATURES);
            let got = op.apply(&mut solver);
            match (&got, want) {
                (Ok(_), Ok(want)) => prop_assert_eq!(solver.program(), &want),
                (Err(_), Err(_)) => prop_assert_eq!(solver.program(), &before),
                (got, want) => prop_assert!(false, "{op:?}: the solver gave {got:?}, the rules {want:?}"),
            }
            if let Some(expected) = enumerable(solver.program()) {
                let program = solver.program().clone();
                let solved = solver.run().settings(exact()).solve().unwrap();
                agrees(&program, &expected, &solved.outcome)?;
            }
        }
    }
}
