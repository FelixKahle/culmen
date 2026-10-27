// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Editing a program HiGHS holds: hand-worked optima after each kind of edit, the cases HiGHS 1.15 makes hard, the
//! warm start an edit keeps, the analyses an edit clears, and a property test whose reference is a fresh solver on
//! the edited program.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{
    Analysis, AnalysisError, Backend, Bounds, Col, EditError, Integrality, LinExpr, Linear, Model, ModelClass, NewCol,
    Objective, Outcome, Program, ProgramColumn, ProgramPlan, Request, Settings, Solution, SolveError, Solver, Var,
    bounds, relative_gap,
};
use culmen_highs::{Highs, HighsError, HighsOptions, Switch};
use culmen_testing::instances::assignment;
use culmen_testing::mirror::{Faults, Mirror, MirrorError};
use culmen_testing::reference::{Reference, close, edit_plan, solve};
use culmen_testing::strategy::{small_lp, small_qp, small_semi, with_edits};
use proptest::prelude::*;

/// Solves with a relative gap of zero and returns the outcome, or why the solve failed.
fn try_solve_exactly<B: Backend>(solver: &mut Solver<B>) -> Result<Outcome, SolveError<B::Error>> {
    let exact = Settings {
        relative_gap: Some(relative_gap!(0.0)),
        ..Settings::new()
    };
    solver
        .solve(Request::new().settings_from(exact))
        .map(|solved| solved.outcome)
}

/// Solves with a relative gap of zero and returns the outcome.
fn solve_exactly<B: Backend>(solver: &mut Solver<B>) -> Outcome {
    try_solve_exactly(solver).unwrap()
}

/// Whether the edited solver's failure is the fresh solver's: HiGHS's own error of the same kind in both.
fn same_failure(edited: &SolveError<MirrorError<HighsError>>, fresh: &SolveError<HighsError>) -> bool {
    match (edited, fresh) {
        (SolveError::Backend(MirrorError::Inner(edited)), SolveError::Backend(fresh)) => {
            core::mem::discriminant(edited) == core::mem::discriminant(fresh)
        }
        _ => false,
    }
}

/// The optimal objective value of the next solve.
fn optimum<B: Backend>(solver: &mut Solver<B>) -> f64 {
    match solve_exactly(solver) {
        Outcome::Optimal(solution) => solution.objective,
        other => panic!("expected an optimum, got {other:?}"),
    }
}

#[test]
fn test_a_chain_of_edits_reaches_each_hand_worked_optimum() {
    // Maximize x + y over x, y in [0, 1].
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    let y = m.add_var(Var::continuous().upper(1));
    m.maximize(x + y);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    assert!(close(optimum(&mut solver), 2.0, 1e-9));

    // The cut x + y <= 1.5.
    let cut = solver.add_row((x + y).leq(1.5)).unwrap();
    assert!(close(optimum(&mut solver), 1.5, 1e-9));
    // The cut switched off.
    solver.set_row_bounds([(cut, ..)]).unwrap();
    assert!(close(optimum(&mut solver), 2.0, 1e-9));
    // x fixed at zero.
    solver.set_col_bounds([(x, bounds!(0.0, 0.0))]).unwrap();
    assert!(close(optimum(&mut solver), 1.0, 1e-9));
    // y worth 3.
    solver.set_col_costs([(y, 3)]).unwrap();
    assert!(close(optimum(&mut solver), 3.0, 1e-9));
    // y at most 0.5, then whole: y = 0.
    solver.set_col_bounds([(y, bounds!(0.0, 0.5))]).unwrap();
    assert!(close(optimum(&mut solver), 1.5, 1e-9));
    solver.set_col_integrality([(y, Integrality::Integer)]).unwrap();
    assert!(close(optimum(&mut solver), 0.0, 1e-9));
    // A new column z in [0, 1] worth 2, in the free cut row; then the cut back at x + y + z <= 0.5.
    let z = solver
        .add_col(NewCol {
            column: ProgramColumn::continuous(2.0, bounds!(0.0, 1.0)),
            entries: vec![(cut, 1.0)],
        })
        .unwrap();
    assert!(close(optimum(&mut solver), 2.0, 1e-9));
    solver.set_row_bounds([(cut, bounds!(f64::NEG_INFINITY, 0.5))]).unwrap();
    assert!(close(optimum(&mut solver), 1.0, 1e-9));
    assert_eq!(
        solver.program().row(cut).entries().collect::<Vec<_>>(),
        [(x, 1.0), (y, 1.0), (z, 1.0)]
    );
}

#[test]
fn test_column_generation_finds_the_cheaper_pattern() {
    // Cover the demands a and b at least once each: one pattern per item, each costing 1, is 2 in all; the
    // pattern covering both, costing 1.5, brings it down to 1.5.
    let mut m = Model::new();
    let only_a = m.add_var(Var::continuous());
    let only_b = m.add_var(Var::continuous());
    let a = m.add_constraint(only_a.geq(1));
    let b = m.add_constraint(only_b.geq(1));
    m.minimize(only_a + only_b);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    assert!(close(optimum(&mut solver), 2.0, 1e-9));
    assert!(solver.duals().is_ok());

    let both = solver
        .add_col(NewCol {
            column: ProgramColumn::continuous(1.5, Bounds::NON_NEGATIVE),
            entries: vec![(a, 1.0), (b, 1.0)],
        })
        .unwrap();
    let Outcome::Optimal(solution) = solve_exactly(&mut solver) else {
        panic!("expected an optimum")
    };
    assert!(close(solution.objective, 1.5, 1e-9));
    assert!(close(solution.values[both], 1.0, 1e-9));
}

#[test]
fn test_adding_a_column_beside_further_objectives_solves() {
    // HiGHS 1.15 does not extend its further objectives' coefficients when columns are added, and refuses to solve
    // with coefficient vectors shorter than the model: the edit must pass the objectives again.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    let y = m.add_var(Var::continuous().upper(1));
    m.minimize(x + y);
    m.add_objective(Objective::maximize(x));
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    assert!(close(optimum(&mut solver), 0.0, 1e-9));

    let z = solver
        .add_col(ProgramColumn::continuous(-1.0, bounds!(0.0, 1.0)))
        .unwrap();
    let Outcome::Optimal(solution) = solve_exactly(&mut solver) else {
        panic!("expected an optimum")
    };
    assert!(close(solution.objective, -1.0, 1e-9));
    assert!(close(solution.values[z], 1.0, 1e-9));
    let mut fresh = Solver::new(Highs::new(), solver.program().clone()).unwrap();
    assert!(close(optimum(&mut fresh), solution.objective, 1e-9));
}

#[test]
fn test_adding_rows_to_a_qp_without_rows_solves_it_right() {
    // HiGHS 1.15 is given a QP without rows with one free row added; rows added later must not be read past it.
    // Minimize (x - 1)² + (y - 2)²: zero; with x + y <= 1, the point nearest (1, 2) is (0, 1), at 2.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::free());
    m.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2));
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    assert!(close(optimum(&mut solver), 0.0, 1e-6));

    let row = solver.add_row((x + y).leq(1)).unwrap();
    let Outcome::Optimal(solution) = solve_exactly(&mut solver) else {
        panic!("expected an optimum")
    };
    assert!(close(solution.objective, 2.0, 1e-5));
    assert!(close(solution.values[x], 0.0, 1e-4) && close(solution.values[y], 1.0, 1e-4));
    assert!(solver.program().row(row).bounds() == bounds!(f64::NEG_INFINITY, 1.0));
}

#[test]
fn test_a_semi_column_beyond_the_highs_limit_is_refused_and_the_solver_still_solves() {
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(1, 10));
    m.maximize(x);
    let program = m.build().unwrap();
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    assert!(close(optimum(&mut solver), 10.0, 1e-9));

    let beyond = 2.0 * Highs::SEMI_UPPER_LIMIT;
    let refused = solver.set_col_bounds([(x, Bounds::new(1.0, beyond).unwrap())]);
    assert_eq!(
        refused,
        Err(EditError::Backend(HighsError::SemiUpper { col: x, upper: beyond }))
    );
    assert_eq!(solver.program(), &program);
    assert!(close(optimum(&mut solver), 10.0, 1e-9));
}

#[test]
fn test_an_integer_column_in_a_qp_is_refused_as_a_miqp() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(2));
    m.minimize((x - 1) * (x - 1));
    let program = m.build().unwrap();
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let miqp = EditError::Unsupported {
        class: ModelClass::Miqp,
        supported: Highs::CLASSES,
    };
    assert_eq!(
        solver.set_col_integrality([(x, Integrality::Integer)]),
        Err(miqp.clone())
    );
    assert_eq!(solver.add_col(ProgramColumn::binary(0.0)), Err(miqp));
    assert_eq!(solver.program(), &program);
    assert!(close(optimum(&mut solver), 0.0, 1e-6));
}

#[test]
fn test_an_edit_clears_the_last_solve_for_analyses() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    m.maximize(x);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    solve_exactly(&mut solver);
    assert!(solver.duals().is_ok());
    solver.set_col_costs([(x, 2.0)]).unwrap();
    assert_eq!(
        solver.duals(),
        Err(AnalysisError::NotSolved {
            analysis: Analysis::Duals
        })
    );
}

#[test]
fn test_a_refused_edit_keeps_the_last_solve_for_analyses() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    m.maximize(x);
    let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    solve_exactly(&mut solver);
    let duals = solver.duals().unwrap();
    assert_eq!(
        solver.set_col_costs([(Col::new(9), 1.0)]),
        Err(EditError::UnknownCol { col: Col::new(9) })
    );
    assert_eq!(solver.duals(), Ok(duals));
}

#[test]
fn test_a_highs_edit_failing_after_it_applied_is_undone_by_a_reload() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    let y = m.add_var(Var::continuous().upper(1));
    m.maximize(x + y);
    let program = m.build().unwrap();
    let mut solver = Solver::new(Mirror::new(Highs::new()), program.clone()).unwrap();
    assert!(close(optimum(&mut solver), 2.0, 1e-9));

    solver.backend_mut().faults = Faults {
        refuse_edits: 1,
        ..Faults::default()
    };
    assert_eq!(
        solver.set_col_costs([(x, 10.0)]),
        Err(EditError::Backend(MirrorError::Injected))
    );
    assert_eq!(solver.program(), &program);
    assert!(close(optimum(&mut solver), 2.0, 1e-9));
    assert_eq!(solver.backend().held(), Some(solver.program()));
}

/// The LP relaxation of a generalized assignment problem, with presolve off so a fresh solve and a warm one count
/// simplex iterations alike.
fn relaxed_assignment() -> (Program, HighsOptions) {
    let mut plan = ProgramPlan::from(assignment(30, 10, 7, 0.0).program);
    for column in &mut plan.columns {
        column.integrality = Integrality::Continuous;
    }
    let options = HighsOptions {
        presolve: Some(Switch::Off),
        ..HighsOptions::default()
    };
    (plan.build().unwrap(), options)
}

/// The simplex iterations and solution of an optimal solve.
fn iterations_and_solution(solver: &mut Solver<Highs>) -> (u64, Solution) {
    let solved = solver.solve(Request::new()).unwrap();
    let Outcome::Optimal(solution) = solved.outcome else {
        panic!("expected an optimum, got {:?}", solved.outcome)
    };
    (solved.status.simplex_iterations, solution)
}

/// The simplex iterations and objective of an optimal solve.
fn iterations_and_optimum(solver: &mut Solver<Highs>) -> (u64, f64) {
    let (iterations, solution) = iterations_and_solution(solver);
    (iterations, solution.objective)
}

#[test]
fn test_a_warm_resolve_after_a_cut_takes_fewer_iterations_than_a_fresh_solve() {
    let (program, options) = relaxed_assignment();
    // The first ten columns are the first product's, at its ten locations.
    let first_product: Vec<Col> = program.cols().take(10).collect();
    let mut solver = Solver::new(Highs::with_options(options.clone()), program).unwrap();
    let (_, relaxed) = iterations_and_solution(&mut solver);

    // The first product at most half at the locations the relaxation puts it: a cut its optimum violates, since the
    // product's assignment row holds those locations at one in all.
    let used = first_product.into_iter().filter(|&c| relaxed.values[c] > 1e-6);
    solver.add_row(LinExpr::sum(used).leq(0.5)).unwrap();
    let (warm, after) = iterations_and_optimum(&mut solver);
    let mut fresh = Solver::new(Highs::with_options(options), solver.program().clone()).unwrap();
    let (cold, fresh_after) = iterations_and_optimum(&mut fresh);
    assert!(close(after, fresh_after, 1e-7), "warm {after} vs fresh {fresh_after}");
    assert!(after >= relaxed.objective - 1e-7, "a cut cannot improve a minimum");
    assert!(warm < cold, "warm {warm} iterations, fresh {cold}");
}

#[test]
fn test_a_warm_resolve_after_fixing_columns_takes_fewer_iterations_than_a_fresh_solve() {
    let (program, options) = relaxed_assignment();
    let fixed: Vec<Col> = program.cols().step_by(10).take(5).collect();
    let mut solver = Solver::new(Highs::with_options(options.clone()), program).unwrap();
    iterations_and_optimum(&mut solver);

    solver
        .set_col_bounds(fixed.iter().map(|&c| (c, bounds!(0.0, 0.0))))
        .unwrap();
    let (warm, after) = iterations_and_optimum(&mut solver);
    let mut fresh = Solver::new(Highs::with_options(options), solver.program().clone()).unwrap();
    let (cold, fresh_after) = iterations_and_optimum(&mut fresh);
    assert!(close(after, fresh_after, 1e-7), "warm {after} vs fresh {fresh_after}");
    assert!(warm < cold, "warm {warm} iterations, fresh {cold}");
}

/// The objective of an optimal outcome, or of one that reached the requested gap of zero.
const fn optimal(outcome: &Outcome) -> Option<f64> {
    match outcome {
        Outcome::Optimal(solution) | Outcome::GapReached { solution, .. } => Some(solution.objective),
        _ => None,
    }
}

/// Whether two outcomes of one program agree: both optimal at objectives within `1e-5`, or both without an optimum
/// for lack of a feasible point or a bound.
fn agree(edited: &Outcome, fresh: &Outcome) -> bool {
    let without = |o: &Outcome| {
        matches!(
            o,
            Outcome::Infeasible | Outcome::Unbounded | Outcome::InfeasibleOrUnbounded
        )
    };
    match (optimal(edited), optimal(fresh)) {
        (Some(a), Some(b)) => close(a, b, 1e-5),
        _ => without(edited) && without(fresh),
    }
}

/// The exact reference of a program small enough to enumerate: linear, every bound finite, at most three
/// continuous columns and at most two thousand points of whole and semi columns; `None` for any other.
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

/// Whether HiGHS's outcome is the exact reference's.
fn matches_reference(outcome: &Outcome, reference: &Reference) -> bool {
    match (reference, outcome) {
        (Reference::Infeasible, Outcome::Infeasible | Outcome::InfeasibleOrUnbounded) => true,
        (Reference::Optimal(_), outcome) => optimal(outcome)
            .zip(reference.objective())
            .is_some_and(|(got, want)| close(got, want, 1e-5)),
        (Reference::Infeasible, _) => false,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// After every edit, accepted or refused, and whether or not HiGHS fails it after applying it, the edited solver
    /// reaches the optimum a fresh solver on `solver.program()` reaches, the exact reference's where it can
    /// enumerate, and its program is the edit rules'.
    #[test]
    fn test_an_edited_solver_reaches_the_optimum_of_a_fresh_solver(
        (program, ops) in with_edits(prop_oneof![small_lp(), small_semi(), small_qp()]),
        faults in prop::collection::vec(any::<bool>(), 5),
    ) {
        let mut solver = Solver::new(Mirror::new(Highs::new()), program).unwrap();
        for (op, &refuse) in ops.iter().zip(&faults) {
            let before = solver.program().clone();
            let want = edit_plan(&before, op, Highs::CLASSES, Highs::FEATURES);
            solver.backend_mut().faults.refuse_edits = usize::from(refuse);
            let got = op.apply(&mut solver);
            solver.backend_mut().faults = Faults::default();
            match (&got, want) {
                (Ok(_), Ok(want)) => prop_assert_eq!(solver.program(), &want),
                (Err(EditError::Backend(MirrorError::Injected)), Ok(_)) if refuse => {
                    prop_assert_eq!(solver.program(), &before);
                }
                (Err(error), Err(_)) if !matches!(error, EditError::Backend(_)) => {
                    prop_assert_eq!(solver.program(), &before);
                }
                (got, want) => prop_assert!(false, "{op:?}: the solver gave {got:?}, the rules {want:?}"),
            }
            let edited = try_solve_exactly(&mut solver);
            prop_assert_eq!(solver.backend().held(), Some(solver.program()));
            let fresh = try_solve_exactly(&mut Solver::new(Highs::new(), solver.program().clone()).unwrap());
            // Some edits make a QP HiGHS 1.15 cannot solve; a fresh solver failing the same way is agreement.
            let (edited, fresh) = match (edited, fresh) {
                (Ok(edited), Ok(fresh)) => (edited, fresh),
                (Err(edited), Err(fresh)) => {
                    prop_assert!(same_failure(&edited, &fresh), "edited {edited:?}, fresh {fresh:?}");
                    continue;
                }
                (edited, fresh) => {
                    return Err(TestCaseError::fail(format!("edited {edited:?}, fresh {fresh:?}")));
                }
            };
            prop_assert!(agree(&edited, &fresh), "edited {edited:?}, fresh {fresh:?}");
            if let Some(solution) = edited.solution() {
                prop_assert!(solver.program().max_violation(&solution.values) <= 1e-5, "{solution:?}");
            }
            if let Some(reference) = enumerable(solver.program()) {
                prop_assert!(matches_reference(&edited, &reference), "exact {reference:?}, edited {edited:?}");
            }
        }
    }
}
