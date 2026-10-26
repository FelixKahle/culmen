// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The analyses on HiGHS: duals, basis, ranging, rays, IIS and feasibility relaxation on hand-worked programs whose
//! answers were written from the mathematics first, the typed refusals, and strong duality on random LPs.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use std::collections::BTreeMap;

use culmen_core::{
    Analysis, AnalysisError, Basis, BasisStatus, Bound, Bounds, Col, Iis, Line, Linear, Misfit, Model, ModelClass,
    Outcome, Program, ProgramPlan, Request, Row, Sense, Settings, Side, Solver, Table, TiVec, Var, Violation, Weights,
    penalty,
};
use culmen_highs::{Highs, HighsStatus};
use culmen_testing::reference::{Reference, close, solve};
use culmen_testing::strategy::small_lp;
use proptest::prelude::*;

/// Whether `got` is within `1e-6` of `want`, relative to their size above one.
fn near(got: f64, want: f64) -> bool {
    close(got, want, 1e-6)
}

fn assert_all_near(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    assert!(got.iter().zip(want).all(|(&g, &w)| near(g, w)), "{got:?} vs {want:?}");
}

/// A solver on `program` after one solve, and that solve's status.
fn solved(program: &Program) -> (Solver<Highs>, Outcome, HighsStatus) {
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let solved = solver.solve(Request::new()).unwrap();
    (solver, solved.outcome, solved.status)
}

/// `sense 3x + 2y` subject to `r1: x + y <= 4`, `r2: x + 3y <= 7`, `0 <= x <= 3`, `y >= 0`: the optimum is `x = 3`
/// at its upper bound, `y = 1` basic, `r1` tight at its upper bound, `r2` slack. Minimizing `-3x - 2y` gives the
/// same point.
fn production(sense: Sense) -> (Program, [Col; 2], [Row; 2]) {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(3.0));
    let y = m.add_var(Var::continuous());
    let r1 = m.add_constraint((x + y).leq(4.0));
    let r2 = m.add_constraint((x + 3.0 * y).leq(7.0));
    match sense {
        Sense::Minimize => m.minimize(-3.0 * x - 2.0 * y),
        Sense::Maximize => m.maximize(3.0 * x + 2.0 * y),
    }
    (m.build().unwrap(), [x, y], [r1, r2])
}

#[test]
fn test_duals_of_a_minimization_are_the_objective_rates_of_its_bounds() {
    // c = Aᵀy + d: y basic gives -2 = y1, so y1 = -2; x at its upper bound gives -3 = y1 + dx, so dx = -1; r2 is
    // slack, y2 = 0. The objective -11 = 4·y1 + 3·dx.
    let (program, [x, y], [r1, r2]) = production(Sense::Minimize);
    let (solver, outcome, _) = solved(&program);
    let solution = outcome.into_solution().unwrap();
    assert!(near(solution.objective, -11.0));
    let duals = solver.duals().unwrap();
    assert_all_near(&duals.rows.raw, &[-2.0, 0.0]);
    assert_all_near(&duals.cols.raw, &[-1.0, 0.0]);
    assert_eq!((duals.rows[r2], duals.cols[y]), (0.0, 0.0));
    assert!(near(4.0 * duals.rows[r1] + 3.0 * duals.cols[x], solution.objective));
    assert_all_near(&program.activities(&solution.values).raw, &[4.0, 6.0]);
}

#[test]
fn test_duals_of_a_maximization_have_the_opposite_signs() {
    // The same point maximizing 3x + 2y: y1 = 2, dx = 1, and 11 = 4·y1 + 3·dx.
    let (program, _, _) = production(Sense::Maximize);
    let (solver, _, _) = solved(&program);
    let duals = solver.duals().unwrap();
    assert_all_near(&duals.rows.raw, &[2.0, 0.0]);
    assert_all_near(&duals.cols.raw, &[1.0, 0.0]);
}

#[test]
fn test_a_row_at_its_lower_bound_has_a_non_negative_dual_in_a_minimization() {
    // Minimize x + y subject to x + 2y >= 4, x, y >= 0: y = 2, the row's dual 1/2 and x's reduced cost 1/2.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + 2.0 * y).geq(4.0));
    m.minimize(x + y);
    let (solver, _, _) = solved(&m.build().unwrap());
    let duals = solver.duals().unwrap();
    assert_all_near(&duals.rows.raw, &[0.5]);
    assert_all_near(&duals.cols.raw, &[0.5, 0.0]);
}

#[test]
fn test_duals_of_a_qp_include_the_gradient_of_its_quadratic_part() {
    // Minimize x² - 2x subject to x <= 1/2: at x = 1/2 the gradient is 2x - 2 = -1, which is the row's dual.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    m.add_constraint(x.leq(0.5));
    m.minimize(x * x - 2.0 * x);
    let (solver, _, _) = solved(&m.build().unwrap());
    let duals = solver.duals().unwrap();
    assert_all_near(&duals.rows.raw, &[-1.0]);
    assert_all_near(&duals.cols.raw, &[0.0]);
}

#[test]
fn test_a_qp_without_rows_has_no_row_duals() {
    // HiGHS is given a free row for a QP without rows; it is not one of the program's rows.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(3.0));
    m.minimize((x - 5) * (x - 5));
    let (solver, _, _) = solved(&m.build().unwrap());
    let duals = solver.duals().unwrap();
    assert!(duals.rows.is_empty());
    // x stops at its upper bound 3, where the gradient 2(x - 5) = -4 is its reduced cost.
    assert_all_near(&duals.cols.raw, &[-4.0]);
}

#[test]
fn test_the_basis_of_an_optimum_names_where_each_line_stands() {
    let (program, _, _) = production(Sense::Minimize);
    let (solver, _, _) = solved(&program);
    let basis = solver.basis().unwrap().unwrap();
    assert_eq!(basis.cols.raw, vec![BasisStatus::Upper, BasisStatus::Basic]);
    assert_eq!(basis.rows.raw, vec![BasisStatus::Upper, BasisStatus::Basic]);
    assert_eq!(basis.check(&program), Ok(()));
}

/// A transportation LP of four sources and five sinks: supply rows `<=`, demand rows `>=`, costs from a fixed
/// pattern; enough work that a cold start takes simplex iterations.
fn transportation() -> Program {
    let (supply, demand) = ([30.0, 25.0, 35.0, 20.0], [15.0, 20.0, 25.0, 18.0, 22.0]);
    let mut m = Model::new();
    let x: Vec<Vec<Col>> = (0..4)
        .map(|_| (0..5).map(|_| m.add_var(Var::continuous())).collect())
        .collect();
    for (s, row) in x.iter().enumerate() {
        m.add_constraint(culmen_core::LinExpr::sum(row.iter().copied()).leq(supply[s]));
    }
    for (d, &want) in demand.iter().enumerate() {
        m.add_constraint(culmen_core::LinExpr::sum(x.iter().map(|row| row[d])).geq(want));
    }
    let cost = |s: usize, d: usize| f64::from(u32::try_from((s * 7 + d * 3) % 11 + 1).unwrap());
    m.minimize(culmen_core::LinExpr::sum(
        (0..4)
            .flat_map(|s| (0..5).map(move |d| (s, d)))
            .map(|(s, d)| cost(s, d) * x[s][d]),
    ));
    m.build().unwrap()
}

#[test]
fn test_an_optimal_basis_warm_starts_a_fresh_solver_without_iterations() {
    let program = transportation();
    let no_presolve = Settings {
        presolve: Some(false),
        ..Settings::new()
    };
    let mut cold = Solver::new(Highs::new(), program.clone()).unwrap();
    let first = cold.solve(Request::new().settings_from(no_presolve.clone())).unwrap();
    assert!(first.status.simplex_iterations > 0, "{:?}", first.status);
    let basis = cold.basis().unwrap().unwrap();

    let mut warm = Solver::new(Highs::new(), program).unwrap();
    warm.set_basis(&basis).unwrap();
    let second = warm.solve(Request::new().settings_from(no_presolve)).unwrap();
    assert_eq!(second.status.simplex_iterations, 0, "{:?}", second.status);
    let objective = |o: &Outcome| o.solution().unwrap().objective;
    assert!(near(objective(&second.outcome), objective(&first.outcome)));
}

#[test]
fn test_a_basis_that_does_not_fit_is_refused_before_highs_sees_it() {
    let (program, [x, _], _) = production(Sense::Minimize);
    let (mut solver, _, _) = solved(&program);
    let mut basis = Basis::slack(&program);
    basis.rows.pop();
    let misfit = Misfit::Length {
        table: Table::BasisRows,
        expected: 2,
        found: 1,
    };
    assert_eq!(solver.set_basis(&basis), Err(AnalysisError::Misfit(misfit)));
    let mut basis = Basis::slack(&program);
    basis.rows[Row::new(0)] = BasisStatus::Lower;
    assert_eq!(
        solver.set_basis(&basis),
        Err(AnalysisError::Misfit(Misfit::BasicCount { expected: 2, found: 1 }))
    );
    // A basis of another program of the same shape: x's lower bound here is minus infinity.
    let mut other = ProgramPlan::from(program.clone());
    other.columns[x].bounds = Bounds::at_most(3.0).unwrap();
    let other = other.build().unwrap();
    let mut basis = Basis::slack(&program);
    basis.cols[x] = BasisStatus::Lower;
    let mut on_other = Solver::new(Highs::new(), other).unwrap();
    assert_eq!(
        on_other.set_basis(&basis),
        Err(AnalysisError::Misfit(Misfit::Status {
            line: Line::Col(x),
            status: BasisStatus::Lower
        }))
    );
}

#[test]
fn test_ranging_of_a_basic_cost_ends_where_a_reduced_cost_or_dual_reaches_zero() {
    // With c_y moving and y basic, dx = c_x - c_y <= 0 holds for c_y >= -3, and y1 = c_y <= 0 for c_y <= 0. At
    // c_y = -3, x enters by leaving its bound and r2 becomes tight: z = -9 - 3 = -12. At c_y = 0, r1 enters and y
    // falls to zero: z = -9.
    let (program, [x, y], [r1, r2]) = production(Sense::Minimize);
    let (mut solver, _, _) = solved(&program);
    let ranging = solver.ranging().unwrap();
    let cost = ranging.costs[y];
    assert!(
        near(cost.down.value, -3.0) && near(cost.down.objective, -12.0),
        "{cost:?}"
    );
    assert!(near(cost.up.value, 0.0) && near(cost.up.objective, -9.0), "{cost:?}");
    assert_eq!(
        (cost.down.entering, cost.down.leaving),
        (Some(Line::Col(x)), Some(Line::Row(r2)))
    );
    assert_eq!(
        (cost.up.entering, cost.up.leaving),
        (Some(Line::Row(r1)), Some(Line::Col(y)))
    );
    // x, nonbasic at its upper bound, stays so while dx = c_x + 2 <= 0: up to c_x = -2, z = -6 - 2 = -8.
    let cost = ranging.costs[x];
    assert!(near(cost.up.value, -2.0) && near(cost.up.objective, -8.0), "{cost:?}");
    assert_eq!(cost.down.value, f64::NEG_INFINITY, "{cost:?}");
    // r1's bound b keeps the basis while y = b - 3 >= 0 and 3 + 3(b - 3) <= 7: b in [3, 13/3], z = -9 - 2(b - 3).
    let row = ranging.row_bounds[r1];
    assert!(near(row.down.value, 3.0) && near(row.down.objective, -9.0), "{row:?}");
    assert!(
        near(row.up.value, 13.0 / 3.0) && near(row.up.objective, -35.0 / 3.0),
        "{row:?}"
    );
    assert_eq!(
        (ranging.costs.len(), ranging.col_bounds.len(), ranging.row_bounds.len()),
        (2, 2, 2)
    );
}

#[test]
fn test_ranging_objectives_include_the_offset() {
    // The production LP with 10 added to its objective: every objective of its ranging moves by 10.
    let (program, [_, y], _) = production(Sense::Minimize);
    let mut plan = ProgramPlan::from(program);
    plan.offset = 10.0;
    let (mut solver, _, _) = solved(&plan.build().unwrap());
    let cost = solver.ranging().unwrap().costs[y];
    assert!(
        near(cost.down.objective, -2.0) && near(cost.up.objective, 1.0),
        "{cost:?}"
    );
}

/// The value of `Σ coefficient · value` that `max` (else min) reaches over `bounds` for each coefficient.
fn support(terms: impl IntoIterator<Item = (f64, Bounds)>, max: bool) -> f64 {
    terms
        .into_iter()
        .filter(|&(a, _)| a != 0.0)
        .map(|(a, b)| a * if (a > 0.0) == max { b.upper() } else { b.lower() })
        .sum()
}

/// Whether `d` is a ray of `program`: every line's recession allows it, and the objective improves along it.
fn is_primal_ray(program: &Program, d: &TiVec<Col, f64>) -> bool {
    let recedes = |value: f64, bounds: Bounds| {
        (value >= -1e-9 || bounds.lower().is_infinite()) && (value <= 1e-9 || bounds.upper().is_infinite())
    };
    let cols = program.cols().all(|c| recedes(d[c], program.columns()[c].bounds));
    let rows = program
        .rows()
        .all(|r| recedes(program.row(r).activity(d), program.row_bounds()[r]));
    let slope = program.cols().map(|c| program.columns()[c].cost * d[c]).sum::<f64>();
    let improves = match program.sense() {
        Sense::Minimize => slope < -1e-9,
        Sense::Maximize => slope > 1e-9,
    };
    cols && rows && improves
}

/// Whether `y` is a Farkas certificate of `program`'s infeasibility as culmen orients it: the largest `yᵀAx` over
/// the columns' bounds is below the least `yᵀr` over the rows' bounds.
fn is_farkas(program: &Program, y: &TiVec<Row, f64>) -> bool {
    let aty = |c: Col| -> f64 {
        let column = |r: Row| {
            program
                .row(r)
                .entries()
                .filter(move |e| e.0 == c)
                .map(move |e| y[r] * e.1)
        };
        program.rows().flat_map(column).sum()
    };
    let over_cols = support(program.cols().map(|c| (aty(c), program.columns()[c].bounds)), true);
    let over_rows = support(program.rows().map(|r| (y[r], program.row_bounds()[r])), false);
    over_cols + 1e-9 < over_rows
}

fn unbounded(sense: Sense) -> Program {
    // x - y <= 1, x, y >= 0: (1, 1) recedes, and -x - y (or x + y maximized) improves along it.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    m.add_constraint((x - y).leq(1.0));
    match sense {
        Sense::Minimize => m.minimize(-1.0 * x - y),
        Sense::Maximize => m.maximize(x + y),
    }
    m.build().unwrap()
}

#[test]
fn test_a_primal_ray_proves_unboundedness() {
    for sense in [Sense::Minimize, Sense::Maximize] {
        let program = unbounded(sense);
        let (mut solver, outcome, status) = solved(&program);
        assert!(
            matches!(outcome, Outcome::Unbounded | Outcome::InfeasibleOrUnbounded),
            "{outcome:?} {status:?}"
        );
        let ray = solver.primal_ray().unwrap().expect("a ray");
        assert!(is_primal_ray(&program, &ray), "{sense:?}: {ray:?}");
        assert_eq!(
            solver.dual_ray().unwrap(),
            None,
            "an unbounded LP has no Farkas certificate"
        );
    }
}

#[test]
fn test_an_lp_without_rows_has_a_primal_ray_along_one_column() {
    // HiGHS 1.15 reports no ray for an LP without rows; the backend gives the improving column's direction.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(2.0));
    let y = m.add_var(Var::free());
    m.maximize(x - 3.0 * y);
    let program = m.build().unwrap();
    let (mut solver, outcome, _) = solved(&program);
    assert!(
        matches!(outcome, Outcome::Unbounded | Outcome::InfeasibleOrUnbounded),
        "{outcome:?}"
    );
    let ray = solver.primal_ray().unwrap().unwrap();
    assert_eq!(ray.raw, vec![0.0, -1.0]);
    assert!(is_primal_ray(&program, &ray));
}

/// `x + y <= 1` and `x + y >= 3` over `x, y` in `[0, 10]`: infeasible.
fn contradictory(sense: Sense) -> Program {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(10.0));
    let y = m.add_var(Var::continuous().upper(10.0));
    m.add_constraint((x + y).leq(1.0));
    m.add_constraint((x + y).geq(3.0));
    m.set_objective(sense, x + 2.0 * y);
    m.build().unwrap()
}

#[test]
fn test_a_dual_ray_is_a_farkas_certificate() {
    for (sense, presolve) in [
        (Sense::Minimize, true),
        (Sense::Maximize, true),
        (Sense::Minimize, false),
    ] {
        let program = contradictory(sense);
        let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
        let settings = Settings {
            presolve: Some(presolve),
            ..Settings::new()
        };
        let solved = solver.solve(Request::new().settings_from(settings)).unwrap();
        assert!(matches!(solved.outcome, Outcome::Infeasible), "{solved:?}");
        let ray = solver.dual_ray().unwrap().expect("a certificate");
        assert!(is_farkas(&program, &ray), "{sense:?}, presolve {presolve}: {ray:?}");
        assert_eq!(solver.primal_ray().unwrap(), None);
    }
}

#[test]
fn test_rays_are_none_after_an_optimum() {
    let (program, _, _) = production(Sense::Minimize);
    let (mut solver, _, _) = solved(&program);
    assert_eq!(solver.primal_ray(), Ok(None));
    assert_eq!(solver.dual_ray(), Ok(None));
}

/// Whether the subsystem of `iis` in `program` is feasible, with `drop` left out: its rows with their listed sides,
/// its columns' listed bounds, every column continuous and otherwise free, and no objective.
fn subsystem_feasible(program: &Program, iis: &Iis, drop: Option<Line>) -> bool {
    let keep = |side: Side, bounds: Bounds| {
        let lower = if side.includes(Bound::Lower) {
            bounds.lower()
        } else {
            f64::NEG_INFINITY
        };
        let upper = if side.includes(Bound::Upper) {
            bounds.upper()
        } else {
            f64::INFINITY
        };
        Bounds::new(lower, upper).unwrap()
    };
    let mut plan = ProgramPlan::from(program.clone());
    plan.quadratic.clear();
    for (col, column) in plan.columns.iter_mut_enumerated() {
        column.cost = 0.0;
        column.integrality = culmen_core::Integrality::Continuous;
        column.bounds = match iis.cols.get(&col) {
            Some(&side) if drop != Some(Line::Col(col)) => keep(side, column.bounds),
            _ => Bounds::FREE,
        };
    }
    let rows = std::mem::take(&mut plan.rows);
    plan.rows = rows
        .into_iter()
        .enumerate()
        .filter_map(|(i, mut row)| {
            let r = Row::new(i);
            let &side = iis.rows.get(&r).filter(|_| drop != Some(Line::Row(r)))?;
            row.bounds = keep(side, row.bounds);
            Some(row)
        })
        .collect();
    let (_, outcome, _) = solved(&plan.build().unwrap());
    matches!(outcome, Outcome::Optimal(_))
}

/// Asserts that `iis` is an IIS of `program`: infeasible, and feasible with any one entry dropped.
fn assert_irreducible(program: &Program, iis: &Iis) {
    assert!(
        !subsystem_feasible(program, iis, None),
        "the subsystem is feasible: {iis:?}"
    );
    let lines = iis
        .cols
        .keys()
        .map(|&c| Line::Col(c))
        .chain(iis.rows.keys().map(|&r| Line::Row(r)));
    for line in lines {
        assert!(
            subsystem_feasible(program, iis, Some(line)),
            "dropping {line} leaves it infeasible: {iis:?}"
        );
    }
}

#[test]
fn test_the_iis_of_three_rows_leaves_out_the_irrelevant_one() {
    // x + y >= 5 with x <= 1 and y <= 2 as rows conflicts; x - y <= 10 has no part in it.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(10.0));
    let y = m.add_var(Var::continuous().upper(10.0));
    let r1 = m.add_constraint((x + y).geq(5.0));
    let r2 = m.add_constraint(x.leq(1.0));
    let r3 = m.add_constraint(y.leq(2.0));
    m.add_constraint((x - y).leq(10.0));
    let program = m.build().unwrap();
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let iis = solver.iis().unwrap().expect("an IIS");
    assert_eq!(
        iis.rows,
        BTreeMap::from([(r1, Side::Lower), (r2, Side::Upper), (r3, Side::Upper)])
    );
    assert_eq!(iis.cols, BTreeMap::new());
    assert_irreducible(&program, &iis);
}

#[test]
fn test_the_iis_names_the_column_bounds_in_conflict() {
    // x + y >= 5 with x <= 1 and y <= 2 as column bounds, both columns free below.
    let mut m = Model::new();
    let x = m.add_var(Var::free().upper(1.0));
    let y = m.add_var(Var::free().upper(2.0));
    let r = m.add_constraint((x + y).geq(5.0));
    let program = m.build().unwrap();
    let (mut solver, outcome, _) = solved(&program);
    assert_eq!(outcome, Outcome::Infeasible);
    let iis = solver.iis().unwrap().unwrap();
    assert_eq!(iis.rows, BTreeMap::from([(r, Side::Lower)]));
    assert_eq!(iis.cols, BTreeMap::from([(x, Side::Upper), (y, Side::Upper)]));
    assert_irreducible(&program, &iis);
}

#[test]
fn test_a_feasible_program_has_no_iis() {
    let (program, _, _) = production(Sense::Minimize);
    let mut fresh = Solver::new(Highs::new(), program.clone()).unwrap();
    assert_eq!(fresh.iis(), Ok(None), "found without a solve");
    let (mut solver, _, _) = solved(&program);
    assert_eq!(solver.iis(), Ok(None));
}

#[test]
fn test_the_iis_of_an_infeasible_qp_is_that_of_its_constraints() {
    let mut m = Model::new();
    let x = m.add_var(Var::free().upper(1.0));
    let y = m.add_var(Var::free().upper(2.0));
    let r = m.add_constraint((x + y).geq(5.0));
    m.minimize(x * x + y * y);
    let program = m.build().unwrap();
    assert_eq!(program.class(), ModelClass::Qp);
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let iis = solver.iis().unwrap().unwrap();
    assert_eq!(iis.rows, BTreeMap::from([(r, Side::Lower)]));
    assert_eq!(iis.cols, BTreeMap::from([(x, Side::Upper), (y, Side::Upper)]));
}

/// `r1: x + y >= 5` and `r2: x + y <= 3` over `x, y` in `[0, 10]`.
fn gap_of_two() -> (Program, [Row; 2]) {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(10.0));
    let y = m.add_var(Var::continuous().upper(10.0));
    let r1 = m.add_constraint((x + y).geq(5.0));
    let r2 = m.add_constraint((x + y).leq(3.0));
    m.minimize(x - y + 7);
    (m.build().unwrap(), [r1, r2])
}

#[test]
fn test_a_relaxation_violates_the_cheaper_row_by_the_gap() {
    // The rows are 2 apart; violating r1 by 2 costs 2, violating r2 by 2 costs 6.
    let (program, [r1, r2]) = gap_of_two();
    let mut weights = Weights::uniform(&program, None, Some(penalty!(1.0)));
    weights.rows[r2] = Some(penalty!(3.0));
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let relaxed = solver.feasibility_relaxation(&weights).unwrap().unwrap();
    let violations: Vec<Violation> = relaxed.violations(&program).collect();
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert_eq!((violations[0].line, violations[0].bound), (Line::Row(r1), Bound::Lower));
    assert!(near(violations[0].amount, 2.0));
    assert!(near(weights.penalty(violations), 2.0));
}

#[test]
fn test_a_relaxation_may_move_a_column_past_a_soft_bound() {
    // x in [0, 1] with the row x >= 3 hard: x's upper bound gives way by 2.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1.0));
    m.add_constraint(x.geq(3.0));
    m.minimize(x);
    let program = m.build().unwrap();
    let mut weights = Weights::uniform(&program, None, None);
    weights.upper[x] = Some(penalty!(1.0));
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let relaxed = solver.feasibility_relaxation(&weights).unwrap().unwrap();
    assert!(near(relaxed.values[x], 3.0), "{relaxed:?}");
    let violations: Vec<Violation> = relaxed.violations(&program).collect();
    assert_eq!(violations.len(), 1);
    assert_eq!((violations[0].line, violations[0].bound), (Line::Col(x), Bound::Upper));
}

#[test]
fn test_a_relaxation_with_only_hard_bounds_and_rows_finds_nothing() {
    // HiGHS 1.15 does not report that its elastic program is infeasible; the backend checks the point it returns.
    let (program, _) = gap_of_two();
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    assert_eq!(
        solver.feasibility_relaxation(&Weights::uniform(&program, None, None)),
        Ok(None)
    );
    let soft_columns = Weights::uniform(&program, Some(penalty!(1.0)), None);
    assert_eq!(
        solver.feasibility_relaxation(&soft_columns),
        Ok(None),
        "the rows conflict however far x, y move"
    );
}

#[test]
fn test_highs_restores_the_relaxed_solution_to_the_columns_of_the_model() {
    // What the backend's buffer relies on: after `Highs_feasibilityRelaxation` solved its elastic LP (2 columns and 2
    // elastic ones here), HiGHS 1.15's solution holds one value per column of the model, and `Highs_getSolution`
    // writes 2 values, not 4.
    let (program, _) = gap_of_two();
    let path = std::env::temp_dir().join(format!("culmen-relax-{}.lp", std::process::id()));
    std::fs::write(&path, program.to_lp_string().unwrap()).unwrap();
    let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    let (hard, soft) = ([-1.0; 2], [1.0; 2]);
    let mut values = [f64::NAN; 10];
    // SAFETY: the instance is created here, used only on this thread, and destroyed before the block ends; the path
    // and option name are NUL-terminated strings HiGHS reads during the calls; the penalties hold one entry per
    // column and per row, and `values` has room for every column of the elastic LP, which HiGHS does not write.
    let (read, relaxed) = unsafe {
        let highs = highs_sys::Highs_create();
        highs_sys::Highs_setBoolOptionValue(highs, c"output_flag".as_ptr(), 0);
        let read = highs_sys::Highs_readModel(highs, c_path.as_ptr());
        let relaxed = highs_sys::Highs_feasibilityRelaxation(
            highs,
            -1.0,
            -1.0,
            -1.0,
            hard.as_ptr(),
            hard.as_ptr(),
            soft.as_ptr(),
        );
        highs_sys::Highs_getSolution(
            highs,
            values.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        highs_sys::Highs_destroy(highs);
        (read, relaxed)
    };
    std::fs::remove_file(&path).unwrap();
    assert_eq!((read, relaxed), (highs_sys::kHighsStatusOk, highs_sys::kHighsStatusOk));
    assert_eq!(values.iter().filter(|v| !v.is_nan()).count(), 2, "{values:?}");
}

#[test]
fn test_a_relaxation_of_a_milp_keeps_integrality() {
    // 2x = 3 with x integer: the nearest even activity is 2 or 4, one unit off.
    let mut m = Model::new();
    let x = m.add_var(Var::integer().upper(10.0));
    m.add_constraint((2.0 * x).equals(3.0));
    m.minimize(x);
    let program = m.build().unwrap();
    let weights = Weights::uniform(&program, None, Some(penalty!(1.0)));
    let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
    let relaxed = solver.feasibility_relaxation(&weights).unwrap().unwrap();
    assert!(near(relaxed.values[x], relaxed.values[x].round()), "{relaxed:?}");
    assert!(near(weights.penalty(relaxed.violations(&program)), 1.0), "{relaxed:?}");
}

#[test]
fn test_a_solve_after_a_relaxation_solves_the_program_again() {
    let (program, _, _) = production(Sense::Minimize);
    let (mut solver, _, _) = solved(&program);
    let weights = Weights::uniform(&program, Some(penalty!(1.0)), Some(penalty!(1.0)));
    let relaxed = solver.feasibility_relaxation(&weights).unwrap().unwrap();
    assert_eq!(
        relaxed.violations(&program).count(),
        0,
        "a feasible program needs no violation"
    );
    assert!(matches!(
        solver.duals(),
        Err(AnalysisError::NotSolved {
            analysis: Analysis::Duals
        })
    ));
    let again = solver.solve(Request::new()).unwrap();
    assert!(near(again.outcome.solution().unwrap().objective, -11.0));
    assert!(near(solver.duals().unwrap().rows[Row::new(0)], -2.0));
}

#[test]
fn test_analyses_are_refused_by_class_and_state() {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().upper(3.0));
    m.add_constraint(x.leq(2.5));
    m.maximize(x);
    let (mut solver, _, _) = solved(&m.build().unwrap());
    let class = ModelClass::Milp;
    assert_eq!(
        solver.duals(),
        Err(AnalysisError::Class {
            analysis: Analysis::Duals,
            class
        })
    );
    assert_eq!(
        solver.iis(),
        Err(AnalysisError::Class {
            analysis: Analysis::Iis,
            class
        })
    );
    assert_eq!(
        solver.ranging().unwrap_err(),
        AnalysisError::Class {
            analysis: Analysis::Ranging,
            class
        }
    );

    let (program, _, _) = production(Sense::Minimize);
    let fresh = Solver::new(Highs::new(), program).unwrap();
    assert_eq!(
        fresh.duals(),
        Err(AnalysisError::NotSolved {
            analysis: Analysis::Duals
        })
    );
    assert_eq!(
        fresh.basis(),
        Err(AnalysisError::NotSolved {
            analysis: Analysis::Basis
        })
    );

    let (mut infeasible, _, _) = solved(&contradictory(Sense::Minimize));
    assert_eq!(
        infeasible.duals(),
        Err(AnalysisError::NotOptimal {
            analysis: Analysis::Duals
        })
    );
    assert_eq!(
        infeasible.ranging().unwrap_err(),
        AnalysisError::NotOptimal {
            analysis: Analysis::Ranging
        }
    );

    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    m.minimize(x * x);
    let mut qp = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
    let weights = Weights::uniform(qp.program(), None, None);
    assert_eq!(
        qp.feasibility_relaxation(&weights),
        Err(AnalysisError::Class {
            analysis: Analysis::FeasibilityRelaxation,
            class: ModelClass::Qp
        })
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// On a random feasible LP, HiGHS's duals meet the exact optimum: each dual times the bound its sign selects
    /// (lower for a positive dual when minimizing, upper when maximizing, and the reverse) sums with the offset to
    /// the optimum exact enumeration finds; the line sits at that bound; and `c = Aᵀy + d` holds.
    #[test]
    fn test_strong_duality_and_complementary_slackness(program in small_lp().prop_filter("an LP", |p| p.class() == ModelClass::Lp)) {
        let Reference::Optimal(_) = solve(&program) else { return Ok(()) };
        let exact = solve(&program).objective().unwrap();
        let (solver, outcome, status) = solved(&program);
        let Outcome::Optimal(solution) = outcome else { panic!("{outcome:?} {status:?}") };
        let duals = solver.duals().unwrap();
        let activities = program.activities(&solution.values);
        let at = |dual: f64, bounds: Bounds, value: f64| -> Option<f64> {
            if dual.abs() <= 1e-9 { return Some(0.0) }
            let lower = (dual > 0.0) == (program.sense() == Sense::Minimize);
            let bound = if lower { bounds.lower() } else { bounds.upper() };
            (bound.is_finite() && (value - bound).abs() <= 1e-6).then_some(dual * bound)
        };
        let rows: Option<f64> = program.rows().map(|r| at(duals.rows[r], program.row_bounds()[r], activities[r])).sum();
        let cols: Option<f64> = program.cols().map(|c| at(duals.cols[c], program.columns()[c].bounds, solution.values[c])).sum();
        let (Some(rows), Some(cols)) = (rows, cols) else { panic!("a dual at the wrong bound: {duals:?} {solution:?}") };
        prop_assert!(close(program.offset() + rows + cols, exact, 1e-6), "dual {} vs exact {exact}", program.offset() + rows + cols);
        for c in program.cols() {
            let aty: f64 = program.rows().map(|r| duals.rows[r] * program.row(r).entries().filter(|e| e.0 == c).map(|e| e.1).sum::<f64>()).sum();
            prop_assert!((program.columns()[c].cost - aty - duals.cols[c]).abs() <= 1e-6, "c = Aᵀy + d fails at {c:?}");
        }
    }

    /// On a random infeasible LP, the IIS HiGHS finds is infeasible and becomes feasible with any one entry dropped.
    #[test]
    fn test_an_iis_is_infeasible_and_irreducible(program in small_lp().prop_map(continuous)) {
        let (_, outcome, _) = solved(&program);
        prop_assume!(outcome == Outcome::Infeasible);
        let mut solver = Solver::new(Highs::new(), program.clone()).unwrap();
        let iis = solver.iis().unwrap().expect("an infeasible LP has an IIS");
        assert_irreducible(&program, &iis);
    }
}

/// The program with every column continuous: an LP.
fn continuous(program: Program) -> Program {
    let mut plan = ProgramPlan::from(program);
    plan.columns
        .iter_mut()
        .for_each(|c| c.integrality = culmen_core::Integrality::Continuous);
    plan.build().unwrap()
}
