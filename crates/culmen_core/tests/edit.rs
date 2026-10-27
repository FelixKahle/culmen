// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Editing a loaded program: what each edit makes of the program, the ids it hands out, every refusal and its
//! sentence, the class and feature checks, and the failure paths that keep the solver and its backend in sync.
//!
//! The expected programs are written out by hand; the property tests judge every edit by the reference model of the
//! edit rules in `culmen_testing::reference::edit_plan`.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::fmt;
use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};

use culmen_core::{
    Backend, Bounds, Callback, Col, Constraint, EditError, Events, Integrality, InvalidBounds, LinExpr, Linear, Model,
    ModelClass, ModelClasses, NewCol, NewRow, Outcome, Program, ProgramColumn, ProgramFeature, ProgramFeatures,
    ProgramPlan, ProgramRow, Request, Row, Script, ScriptError, Sense, Settings, SolveError, Solved, Solver, Term,
    Unsupported, Var, bounds,
};
use culmen_testing::mirror::{Call, Edited, Faults, Mirror, MirrorError};
use culmen_testing::op::{Applied, EditOp};
use culmen_testing::reference::edit_plan;
use culmen_testing::strategy::{small_lp, small_qp, small_semi, with_edits};
use proptest::prelude::*;

/// A solver over a recording script.
type Mirrored = Solver<Mirror<Script>>;

/// The refusal of a [`Mirrored`] solver.
type Refusal = EditError<MirrorError<ScriptError>>;

/// Minimize `2x + 3y + 1` over `x` in `[0, 4]`, `y` in `[0, 3]`, the row `cap`, `x + y <= 5`, and an unnamed row
/// `floor`, `x - y >= -1`.
fn base() -> (Program, Col, Col, Row, Row) {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(4).name("x"));
    let y = m.add_var(Var::continuous().upper(3).name("y"));
    let cap = m.add_named_constraint("cap", (x + y).leq(5));
    let floor = m.add_constraint((x - y).geq(-1));
    m.minimize(2.0 * x + 3.0 * y + 1);
    (m.build().unwrap(), x, y, cap, floor)
}

/// The base program as plain data, written out by hand.
fn base_plan() -> ProgramPlan {
    let (x, y) = (Col::new(0), Col::new(1));
    ProgramPlan {
        sense: Sense::Minimize,
        offset: 1.0,
        columns: vec![
            ProgramColumn {
                cost: 2.0,
                bounds: bounds!(0.0, 4.0),
                integrality: Integrality::Continuous,
                name: Some("x".to_owned()),
            },
            ProgramColumn {
                cost: 3.0,
                bounds: bounds!(0.0, 3.0),
                integrality: Integrality::Continuous,
                name: Some("y".to_owned()),
            },
        ]
        .into(),
        rows: vec![
            ProgramRow {
                bounds: bounds!(f64::NEG_INFINITY, 5.0),
                terms: vec![(x, 1.0), (y, 1.0)],
                name: Some("cap".to_owned()),
            },
            ProgramRow {
                bounds: bounds!(-1.0, f64::INFINITY),
                terms: vec![(x, 1.0), (y, -1.0)],
                name: None,
            },
        ]
        .into(),
        ..ProgramPlan::new()
    }
}

/// The base program loaded into a recording script.
fn mirrored() -> Mirrored {
    Solver::new(Mirror::new(Script::optimal(0.0, vec![0.0, 0.0])), base().0).unwrap()
}

/// The solver's backend holds exactly the solver's program.
fn assert_in_sync(solver: &Mirrored) {
    assert_eq!(solver.backend().held(), Some(solver.program()));
}

/// The last call the backend received.
fn last_call(solver: &Mirrored) -> &Call {
    solver.backend().calls.last().unwrap()
}

#[test]
fn test_the_base_plan_written_by_hand_is_the_base_program() {
    assert_eq!(base().0, base_plan().build().unwrap());
}

#[test]
fn test_set_col_bounds_changes_only_those_bounds_and_the_last_given_holds() {
    let (_, x, y, ..) = base();
    let mut solver = mirrored();
    let given = [(y, bounds!(1.0, 2.0)), (x, bounds!(0.0, 1.0)), (x, bounds!(0.5, 2.0))];
    solver.set_col_bounds(given).unwrap();
    let mut want = base_plan();
    want.columns[x].bounds = bounds!(0.5, 2.0);
    want.columns[y].bounds = bounds!(1.0, 2.0);
    let want = want.build().unwrap();
    assert_eq!(solver.program(), &want);
    assert_eq!(last_call(&solver), &Call::Edit(Edited::ColBounds(vec![x, y]), want));
    assert_in_sync(&solver);
}

#[test]
fn test_set_col_costs_changes_the_own_objective_and_takes_integers() {
    let (_, _, y, ..) = base();
    let mut solver = mirrored();
    solver.set_col_costs([(y, 5)]).unwrap();
    let mut want = base_plan();
    want.columns[y].cost = 5.0;
    let want = want.build().unwrap();
    assert_eq!(solver.program(), &want);
    assert_eq!(last_call(&solver), &Call::Edit(Edited::ColCosts(vec![y]), want));
    assert_in_sync(&solver);
}

#[test]
fn test_set_col_integrality_makes_an_lp_a_milp() {
    let (_, _, y, ..) = base();
    let mut solver = mirrored();
    solver.set_col_integrality([(y, Integrality::Integer)]).unwrap();
    let mut want = base_plan();
    want.columns[y].integrality = Integrality::Integer;
    let want = want.build().unwrap();
    assert_eq!(solver.program(), &want);
    assert_eq!(solver.program().class(), ModelClass::Milp);
    assert_eq!(last_call(&solver), &Call::Edit(Edited::ColIntegrality(vec![y]), want));
    assert_in_sync(&solver);
}

#[test]
fn test_set_row_bounds_frees_a_row_given_a_full_range() {
    let (_, _, _, cap, _) = base();
    let mut solver = mirrored();
    solver.set_row_bounds([(cap, ..)]).unwrap();
    let mut want = base_plan();
    want.rows[cap].bounds = Bounds::FREE;
    let want = want.build().unwrap();
    assert_eq!(solver.program(), &want);
    assert_eq!(last_call(&solver), &Call::Edit(Edited::RowBounds(vec![cap]), want));
    assert_in_sync(&solver);
}

#[test]
fn test_add_row_moves_a_constraint_constant_into_its_bounds_and_keeps_its_name() {
    let (_, x, y, ..) = base();
    let mut solver = mirrored();
    let first = solver.add_row((x + 2.0 * y + 1).leq(7)).unwrap();
    let cut = solver.add_row(("cut", x.geq(1))).unwrap();
    assert_eq!((first, cut), (Row::new(2), Row::new(3)));
    let mut want = base_plan();
    want.rows.push(ProgramRow {
        bounds: bounds!(f64::NEG_INFINITY, 6.0),
        terms: vec![(x, 1.0), (y, 2.0)],
        name: None,
    });
    want.rows.push(ProgramRow {
        bounds: bounds!(1.0, f64::INFINITY),
        terms: vec![(x, 1.0)],
        name: Some("cut".to_owned()),
    });
    let want = want.build().unwrap();
    assert_eq!(solver.program(), &want);
    assert_eq!(solver.program().row(cut).name(), Some("cut"));
    assert_eq!(last_call(&solver), &Call::Edit(Edited::NewRows(vec![cut]), want));
    assert_in_sync(&solver);
}

#[test]
fn test_add_row_adds_up_repeated_terms_and_drops_those_that_cancel() {
    let (_, x, y, ..) = base();
    let mut solver = mirrored();
    let sum = solver
        .add_row(ProgramRow {
            bounds: bounds!(0.0, 1.0),
            terms: vec![(x, 1.0), (y, 2.0), (x, -1.0), (y, 1.0)],
            name: Some("sum".to_owned()),
        })
        .unwrap();
    assert_eq!(solver.program().row(sum).entries().collect::<Vec<_>>(), [(y, 3.0)]);
    assert_in_sync(&solver);
}

#[test]
fn test_add_rows_returns_the_new_rows_in_order_after_every_row_before() {
    let (_, x, y, ..) = base();
    let mut solver = mirrored();
    let rows = solver
        .add_rows([
            NewRow::from(x.leq(1)),
            NewRow::from(("b", y.leq(2))),
            NewRow::Row(ProgramRow {
                bounds: bounds!(-1.0, 1.0),
                terms: vec![(y, 1.0)],
                name: None,
            }),
        ])
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(rows, [Row::new(2), Row::new(3), Row::new(4)]);
    let mut want = base_plan();
    want.rows.push(ProgramRow {
        bounds: bounds!(f64::NEG_INFINITY, 1.0),
        terms: vec![(x, 1.0)],
        name: None,
    });
    want.rows.push(ProgramRow {
        bounds: bounds!(f64::NEG_INFINITY, 2.0),
        terms: vec![(y, 1.0)],
        name: Some("b".to_owned()),
    });
    want.rows.push(ProgramRow {
        bounds: bounds!(-1.0, 1.0),
        terms: vec![(y, 1.0)],
        name: None,
    });
    let want = want.build().unwrap();
    assert_eq!(solver.program(), &want);
    assert_eq!(last_call(&solver), &Call::Edit(Edited::NewRows(rows), want));
    assert_in_sync(&solver);
}

#[test]
fn test_add_col_appends_a_column_with_its_entries_added_up_in_existing_rows() {
    let (_, x, y, cap, floor) = base();
    let mut solver = mirrored();
    let z = solver
        .add_col(NewCol {
            column: ProgramColumn {
                cost: -1.0,
                bounds: bounds!(0.0, 2.0),
                integrality: Integrality::Continuous,
                name: Some("z".to_owned()),
            },
            entries: vec![(cap, 1.0), (floor, 2.0), (cap, 0.5)],
        })
        .unwrap();
    assert_eq!(z, Col::new(2));
    let mut want = base_plan();
    want.columns.push(ProgramColumn {
        cost: -1.0,
        bounds: bounds!(0.0, 2.0),
        integrality: Integrality::Continuous,
        name: Some("z".to_owned()),
    });
    want.rows[cap].terms = vec![(x, 1.0), (y, 1.0), (z, 1.5)];
    want.rows[floor].terms = vec![(x, 1.0), (y, -1.0), (z, 2.0)];
    let want = want.build().unwrap();
    assert_eq!(solver.program(), &want);
    assert_eq!(last_call(&solver), &Call::Edit(Edited::NewCols(vec![z]), want));
    assert_in_sync(&solver);
}

#[test]
fn test_add_col_drops_entries_that_cancel_and_takes_a_bare_column() {
    let (_, x, y, cap, _) = base();
    let mut solver = mirrored();
    let cancelled = solver
        .add_col(NewCol {
            column: ProgramColumn::continuous(0.0, Bounds::NON_NEGATIVE),
            entries: vec![(cap, 1.0), (cap, -1.0)],
        })
        .unwrap();
    let open = solver.add_col(ProgramColumn::binary(1.0)).unwrap();
    assert_eq!((cancelled, open), (Col::new(2), Col::new(3)));
    assert_eq!(
        solver.program().row(cap).entries().collect::<Vec<_>>(),
        [(x, 1.0), (y, 1.0)]
    );
    assert_eq!(solver.program().columns()[open], ProgramColumn::binary(1.0));
    assert_eq!(solver.program().class(), ModelClass::Milp);
    assert_in_sync(&solver);
}

#[test]
fn test_add_cols_returns_the_new_columns_in_order_after_every_column_before() {
    let mut solver = mirrored();
    let cols = solver
        .add_cols([
            ProgramColumn::continuous(1.0, Bounds::FREE),
            ProgramColumn::continuous(2.0, Bounds::NON_NEGATIVE),
        ])
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(cols, [Col::new(2), Col::new(3)]);
    let mut want = base_plan();
    want.columns.push(ProgramColumn::continuous(1.0, Bounds::FREE));
    want.columns.push(ProgramColumn::continuous(2.0, Bounds::NON_NEGATIVE));
    let want = want.build().unwrap();
    assert_eq!(solver.program(), &want);
    assert_eq!(last_call(&solver), &Call::Edit(Edited::NewCols(cols), want));
    assert_in_sync(&solver);
}

#[test]
fn test_ids_handed_out_keep_their_meaning_through_edits() {
    let (program, x, y, cap, floor) = base();
    let mut solver = mirrored();
    solver.add_row(y.leq(1)).unwrap();
    solver
        .add_col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE))
        .unwrap();
    solver.set_row_bounds([(floor, ..)]).unwrap();
    let edited = solver.program();
    assert_eq!(
        edited.row(cap).entries().collect::<Vec<_>>(),
        program.row(cap).entries().collect::<Vec<_>>()
    );
    assert_eq!(edited.row(cap).name(), Some("cap"));
    assert_eq!(edited.columns()[x], program.columns()[x]);
    assert_eq!(edited.columns()[y], program.columns()[y]);
}

#[test]
fn test_an_empty_edit_changes_nothing_and_never_reaches_the_backend() {
    let mut solver = mirrored();
    solver.set_col_bounds(Vec::<(Col, Bounds)>::new()).unwrap();
    solver.set_col_costs(Vec::<(Col, f64)>::new()).unwrap();
    solver.set_col_integrality([]).unwrap();
    solver.set_row_bounds(Vec::<(Row, Bounds)>::new()).unwrap();
    assert_eq!(solver.add_rows(Vec::<NewRow>::new()).unwrap().len(), 0);
    assert_eq!(solver.add_cols(Vec::<NewCol>::new()).unwrap().len(), 0);
    assert_eq!(solver.program(), &base().0);
    assert_eq!(solver.backend().calls, [Call::Load(base().0)]);
}

/// An edit of the base program, given its ids `x`, `y` and `cap`, and the refusal it must meet.
type Case = (fn(&mut Mirrored, Col, Col, Row) -> Result<(), Refusal>, Refusal);

/// Every refusal the checks make, each with what it names.
fn refusals() -> Vec<Case> {
    let (new_row, new_col) = (Row::new(2), Col::new(2));
    let inf = f64::INFINITY;
    vec![
        (
            |s, _, _, _| s.set_col_bounds([(Col::new(9), Bounds::FREE)]),
            EditError::UnknownCol { col: Col::new(9) },
        ),
        (
            |s, _, _, _| s.set_col_costs([(Col::new(9), 1.0)]),
            EditError::UnknownCol { col: Col::new(9) },
        ),
        (
            |s, _, _, _| s.set_col_integrality([(Col::new(9), Integrality::Integer)]),
            EditError::UnknownCol { col: Col::new(9) },
        ),
        (
            |s, _, _, _| s.set_row_bounds([(Row::new(5), ..)]),
            EditError::UnknownRow { row: Row::new(5) },
        ),
        (
            |s, x, _, _| s.set_col_costs([(x, f64::INFINITY)]),
            EditError::NonFiniteCost {
                col: Col::new(0),
                value: inf,
            },
        ),
        (
            |s, x, _, _| s.set_col_integrality([(x, Integrality::SemiContinuous)]),
            EditError::SemiBounds {
                col: Col::new(0),
                bounds: bounds!(0.0, 4.0),
            },
        ),
        (
            |s, _, _, _| s.add_row(Col::new(9).leq(1)).map(drop),
            EditError::UnknownCol { col: Col::new(9) },
        ),
        (
            |s, x, y, _| s.add_row((x + f64::INFINITY * y).leq(1)).map(drop),
            EditError::NonFiniteCoefficient {
                row: new_row,
                col: Col::new(1),
                value: inf,
            },
        ),
        (
            |s, x, _, _| s.add_row((1e308 * x + 1e308 * x).leq(1)).map(drop),
            EditError::NonFiniteCoefficient {
                row: new_row,
                col: Col::new(0),
                value: inf,
            },
        ),
        (
            |s, x, _, _| s.add_row(x.between(3, 1)).map(drop),
            EditError::RowBounds {
                row: new_row,
                source: InvalidBounds::Empty { lower: 3.0, upper: 1.0 },
            },
        ),
        (
            |s, x, _, _| {
                let expr = LinExpr {
                    terms: vec![Term {
                        col: x,
                        coefficient: 1.0,
                    }],
                    constant: 1e308,
                };
                s.add_row(Constraint {
                    expr,
                    lower: f64::NEG_INFINITY,
                    upper: -1e308,
                })
                .map(drop)
            },
            EditError::RowBounds {
                row: new_row,
                source: InvalidBounds::Overflow,
            },
        ),
        (
            |s, x, y, _| s.add_rows([x.leq(1), y.between(2, 1)]).map(drop),
            EditError::RowBounds {
                row: Row::new(3),
                source: InvalidBounds::Empty { lower: 2.0, upper: 1.0 },
            },
        ),
        (
            |s, _, _, _| {
                let column = ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE);
                s.add_col(NewCol {
                    column,
                    entries: vec![(Row::new(9), 1.0)],
                })
                .map(drop)
            },
            EditError::UnknownRow { row: Row::new(9) },
        ),
        (
            |s, _, _, _| {
                s.add_col(ProgramColumn::continuous(f64::INFINITY, Bounds::NON_NEGATIVE))
                    .map(drop)
            },
            EditError::NonFiniteCost {
                col: new_col,
                value: inf,
            },
        ),
        (
            |s, _, _, cap| {
                let column = ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE);
                s.add_col(NewCol {
                    column,
                    entries: vec![(cap, f64::INFINITY)],
                })
                .map(drop)
            },
            EditError::NonFiniteCoefficient {
                row: Row::new(0),
                col: new_col,
                value: inf,
            },
        ),
        (
            |s, _, _, cap| {
                let column = ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE);
                s.add_col(NewCol {
                    column,
                    entries: vec![(cap, 1e308), (cap, 1e308)],
                })
                .map(drop)
            },
            EditError::NonFiniteCoefficient {
                row: Row::new(0),
                col: new_col,
                value: inf,
            },
        ),
        (
            |s, _, _, _| {
                s.add_col(ProgramColumn::semi_continuous(1.0, bounds!(0.0, 1.0)))
                    .map(drop)
            },
            EditError::SemiBounds {
                col: new_col,
                bounds: bounds!(0.0, 1.0),
            },
        ),
    ]
}

#[test]
fn test_each_refusal_names_what_is_at_fault_and_changes_nothing() {
    let (program, x, y, cap, _) = base();
    for (index, (edit, refusal)) in refusals().into_iter().enumerate() {
        let mut solver = mirrored();
        assert_eq!(edit(&mut solver, x, y, cap), Err(refusal), "case {index}");
        assert_eq!(solver.program(), &program, "case {index}");
        assert_eq!(solver.backend().calls, [Call::Load(program.clone())], "case {index}");
        assert_in_sync(&solver);
    }
}

#[test]
fn test_every_edit_error_displays_its_sentence_and_its_source() {
    let cases: [(EditError<ScriptError>, &str, bool); 9] = [
        (
            EditError::UnknownCol { col: Col::new(3) },
            "Solver: the program has no Col(3)",
            false,
        ),
        (
            EditError::UnknownRow { row: Row::new(2) },
            "Solver: the program has no Row(2)",
            false,
        ),
        (
            EditError::NonFiniteCost {
                col: Col::new(1),
                value: f64::INFINITY,
            },
            "Solver: the cost inf of Col(1) is not finite",
            false,
        ),
        (
            EditError::NonFiniteCoefficient {
                row: Row::new(0),
                col: Col::new(1),
                value: f64::NAN,
            },
            "Solver: the coefficient NaN of Col(1) in Row(0) is not finite",
            false,
        ),
        (
            EditError::RowBounds {
                row: Row::new(4),
                source: InvalidBounds::Empty { lower: 3.0, upper: 1.0 },
            },
            "Solver: the bounds of the new Row(4), its constant moved into them, are invalid: Bounds: the lower end 3 \
             exceeds the upper end 1",
            true,
        ),
        (
            EditError::SemiBounds {
                col: Col::new(2),
                bounds: bounds!(0.0, 1.0),
            },
            "Solver: the semi column Col(2) would have the bounds [0, 1], not 0 < lower <= upper < +inf",
            false,
        ),
        (
            EditError::Unsupported {
                class: ModelClass::Milp,
                supported: ModelClasses::of(&[ModelClass::Lp]),
            },
            "Solver: the backend solves {LP}, not a MILP",
            false,
        ),
        (
            EditError::UnsupportedFeature {
                feature: ProgramFeature::SemiContinuous,
                supported: ProgramFeatures::NONE,
            },
            "Solver: the backend takes {}, not semi-continuous columns",
            false,
        ),
        (
            EditError::Backend(ScriptError::NotLoaded),
            "Solver: the backend refused the edit: Script: no program is loaded",
            true,
        ),
    ];
    for (error, sentence, has_source) in cases {
        assert_eq!(error.to_string(), sentence);
        assert_eq!(core::error::Error::source(&error).is_some(), has_source, "{sentence}");
    }
}

/// A backend that solves LPs, and MILPs when `MILP`, takes no feature, and counts its loads; its edits are the
/// default, a load.
#[derive(Debug, Default)]
struct Plain<const MILP: bool> {
    loads: usize,
}

/// Why [`Plain`] refused; it never does.
#[derive(Clone, Debug, PartialEq)]
struct Never;

impl fmt::Display for Never {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Plain: never")
    }
}

impl core::error::Error for Never {}

impl<const MILP: bool> Backend for Plain<MILP> {
    type Error = Never;
    type Status = ();
    type Log<'a> = Unsupported;
    type Progress<'a> = Unsupported;
    type Poll<'a> = Unsupported;
    type Solution<'a> = Unsupported;
    type Lazy<'a> = Unsupported;
    type Cut<'a> = Unsupported;
    type Heuristic<'a> = Unsupported;
    type Native<'a> = Unsupported;

    const EVENTS: Events = Events::NONE;
    const CLASSES: ModelClasses = if MILP {
        ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp])
    } else {
        ModelClasses::of(&[ModelClass::Lp])
    };

    fn load(&mut self, _: &Program) -> Result<(), Never> {
        self.loads += 1;
        Ok(())
    }

    fn solve(&mut self, _: &Settings, _: &mut dyn Callback<Self>) -> Result<Solved<()>, Never> {
        Ok(Solved {
            outcome: Outcome::Infeasible,
            status: (),
            elapsed: Duration::ZERO,
        })
    }
}

#[test]
fn test_the_default_edit_reloads_the_whole_program() {
    let (program, x, ..) = base();
    let mut solver = Solver::new(Plain::<false>::default(), program).unwrap();
    solver.set_col_costs([(x, 7.0)]).unwrap();
    assert_eq!(solver.backend().loads, 2);
}

#[test]
fn test_an_lp_backend_refuses_an_integer_column_before_it_sees_the_edit() {
    let (program, _, y, ..) = base();
    let mut solver = Solver::new(Plain::<false>::default(), program.clone()).unwrap();
    let milp = EditError::Unsupported {
        class: ModelClass::Milp,
        supported: ModelClasses::of(&[ModelClass::Lp]),
    };
    assert_eq!(
        solver.set_col_integrality([(y, Integrality::Integer)]),
        Err(milp.clone())
    );
    assert_eq!(solver.add_col(ProgramColumn::binary(1.0)), Err(milp));
    assert_eq!(solver.program(), &program);
    assert_eq!(solver.backend().loads, 1);
}

#[test]
fn test_a_backend_without_semi_columns_refuses_one_before_it_sees_the_edit() {
    let (program, x, ..) = base();
    let mut solver = Solver::new(Plain::<true>::default(), program).unwrap();
    solver.set_col_bounds([(x, bounds!(1.0, 4.0))]).unwrap();
    let before = solver.program().clone();
    let refusal = EditError::UnsupportedFeature {
        feature: ProgramFeature::SemiContinuous,
        supported: ProgramFeatures::NONE,
    };
    assert_eq!(
        solver.set_col_integrality([(x, Integrality::SemiContinuous)]),
        Err(refusal.clone())
    );
    assert_eq!(
        solver.add_col(ProgramColumn::semi_continuous(1.0, bounds!(1.0, 2.0))),
        Err(refusal)
    );
    assert_eq!(solver.program(), &before);
    assert_eq!(solver.backend().loads, 2);
}

#[test]
fn test_a_refused_edit_restores_the_program_and_reloads_it() {
    let (program, x, ..) = base();
    let mut solver = mirrored();
    solver.backend_mut().faults.refuse_edits = 1;
    assert_eq!(
        solver.set_col_costs([(x, 9.0)]),
        Err(EditError::Backend(MirrorError::Injected))
    );
    assert_eq!(solver.program(), &program);
    let mut tried = base_plan();
    tried.columns[x].cost = 9.0;
    let calls = &solver.backend().calls;
    assert_eq!(
        calls[calls.len() - 2..],
        [
            Call::Edit(Edited::ColCosts(vec![x]), tried.build().unwrap()),
            Call::Load(program)
        ]
    );
    assert_in_sync(&solver);
}

#[test]
fn test_a_refused_add_row_hands_out_no_row() {
    let (_, x, y, ..) = base();
    let mut solver = mirrored();
    solver.backend_mut().faults.refuse_edits = 1;
    assert!(solver.add_row(x.leq(1)).is_err());
    assert_eq!(solver.add_row(y.leq(2)), Ok(Row::new(2)));
    assert_eq!(solver.program().num_rows(), 3);
    assert_eq!(
        solver.program().row(Row::new(2)).entries().collect::<Vec<_>>(),
        [(y, 1.0)]
    );
    assert_in_sync(&solver);
}

#[test]
fn test_a_failed_restore_is_made_good_before_the_next_solve() {
    let (program, x, ..) = base();
    let mut solver = mirrored();
    solver.backend_mut().faults = Faults {
        refuse_edits: 1,
        refuse_loads: 1,
        ..Faults::default()
    };
    assert_eq!(
        solver.set_col_costs([(x, 9.0)]),
        Err(EditError::Backend(MirrorError::Injected))
    );
    assert_eq!(solver.program(), &program);
    assert_eq!(solver.backend().held(), None);
    solver.solve(Request::new()).unwrap();
    assert_eq!(last_call(&solver), &Call::Load(program));
    assert_in_sync(&solver);
}

#[test]
fn test_a_reload_that_fails_again_fails_the_next_solve_and_a_later_one_reloads() {
    let (program, x, ..) = base();
    let mut solver = mirrored();
    solver.backend_mut().faults = Faults {
        refuse_edits: 1,
        refuse_loads: 2,
        ..Faults::default()
    };
    assert!(solver.set_col_costs([(x, 9.0)]).is_err());
    assert!(matches!(
        solver.solve(Request::new()),
        Err(SolveError::Backend(MirrorError::Injected))
    ));
    solver.solve(Request::new()).unwrap();
    assert_eq!(solver.program(), &program);
    assert_in_sync(&solver);
}

#[test]
fn test_an_edit_on_a_solver_out_of_sync_reloads_first() {
    let (program, x, y, ..) = base();
    let mut solver = mirrored();
    solver.backend_mut().faults = Faults {
        refuse_edits: 1,
        refuse_loads: 1,
        ..Faults::default()
    };
    assert!(solver.set_col_costs([(x, 9.0)]).is_err());
    solver.set_col_costs([(y, 4.0)]).unwrap();
    let mut want = base_plan();
    want.columns[y].cost = 4.0;
    let calls = &solver.backend().calls;
    assert_eq!(
        calls[calls.len() - 2..],
        [
            Call::Load(program),
            Call::Edit(Edited::ColCosts(vec![y]), want.build().unwrap())
        ]
    );
    assert_in_sync(&solver);
}

#[test]
fn test_an_edit_on_a_solver_whose_reload_fails_is_refused_unchanged() {
    let (program, x, y, ..) = base();
    let mut solver = mirrored();
    solver.backend_mut().faults = Faults {
        refuse_edits: 1,
        refuse_loads: 2,
        ..Faults::default()
    };
    assert!(solver.set_col_costs([(x, 9.0)]).is_err());
    assert_eq!(
        solver.set_col_costs([(y, 4.0)]),
        Err(EditError::Backend(MirrorError::Injected))
    );
    assert_eq!(solver.program(), &program);
    assert_eq!(last_call(&solver), &Call::Load(program));
}

#[test]
fn test_a_panicking_edit_leaves_the_solver_to_reload_before_its_next_solve() {
    let (_, x, ..) = base();
    let mut solver = mirrored();
    solver.backend_mut().faults.panic_edits = 1;
    let edited = catch_unwind(AssertUnwindSafe(|| solver.set_col_costs([(x, 9.0)])));
    assert!(edited.is_err());
    let mut tried = base_plan();
    tried.columns[x].cost = 9.0;
    assert_eq!(
        last_call(&solver),
        &Call::Edit(Edited::ColCosts(vec![x]), tried.build().unwrap()),
        "the panic came from the backend"
    );
    solver.solve(Request::new()).unwrap();
    assert_eq!(last_call(&solver), &Call::Load(solver.program().clone()));
    assert_in_sync(&solver);
}

#[cfg(feature = "serde")]
#[test]
fn test_a_new_column_survives_a_json_round_trip() {
    let column = NewCol {
        column: ProgramColumn {
            cost: -1.5,
            bounds: bounds!(0.0, f64::INFINITY),
            integrality: Integrality::SemiInteger,
            name: Some("pattern".to_owned()),
        },
        entries: vec![(Row::new(0), 2.0), (Row::new(3), -1.0)],
    };
    let json = serde_json::to_string(&column).unwrap();
    assert_eq!(serde_json::from_str::<NewCol>(&json).unwrap(), column, "{json}");
}

/// The ids an accepted edit must return: the new rows or columns of `after` beyond `before`, or nothing.
fn returned(op: &EditOp, before: &Program, after: &Program) -> Applied {
    match op {
        EditOp::AddRows(_) => Applied::Rows((before.num_rows()..after.num_rows()).map(Row::new).collect()),
        EditOp::AddCols(_) => Applied::Cols((before.num_cols()..after.num_cols()).map(Col::new).collect()),
        _ => Applied::Changed,
    }
}

/// Whether an edit gives no item at all, so it changes nothing and never reaches the backend.
fn is_empty(op: &EditOp) -> bool {
    match op {
        EditOp::ColBounds(items) => items.is_empty(),
        EditOp::ColCosts(items) => items.is_empty(),
        EditOp::ColIntegrality(items) => items.is_empty(),
        EditOp::RowBounds(items) => items.is_empty(),
        EditOp::AddRows(items) => items.is_empty(),
        EditOp::AddCols(items) => items.is_empty(),
    }
}

/// Applies every edit to a solver over `backend` and checks each against the reference rules for its declarations:
/// an accepted edit leaves the reference's program and returns its new ids, a refused one leaves the program as it
/// was, and the backend holds the solver's program throughout.
fn agrees_with_the_rules<B: Backend>(backend: B, program: Program, ops: &[EditOp]) -> Result<(), TestCaseError> {
    let mut solver = Solver::new(Mirror::new(backend), program).unwrap();
    for op in ops {
        let before = solver.program().clone();
        let want = edit_plan(&before, op, B::CLASSES, B::FEATURES);
        match (op.apply(&mut solver), want) {
            (Ok(applied), Ok(want)) => {
                prop_assert_eq!(solver.program(), &want);
                prop_assert_eq!(applied, returned(op, &before, &want));
            }
            (Err(EditError::Backend(error)), _) => prop_assert!(false, "a backend refusal: {error:?}"),
            (Err(_), Err(_)) => prop_assert_eq!(solver.program(), &before),
            (got, want) => prop_assert!(false, "{op:?}: the solver gave {got:?}, the rules {want:?}"),
        }
        prop_assert_eq!(solver.backend().held(), Some(solver.program()));
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Every edit of every kind is accepted exactly when the edit rules accept it, makes the program they make,
    /// returns the ids they append, and leaves a refused edit's program as it was.
    #[test]
    fn test_every_edit_agrees_with_the_reference_rules(
        (program, ops) in with_edits(prop_oneof![small_lp(), small_semi(), small_qp()])
    ) {
        agrees_with_the_rules(Script::optimal(0.0, vec![]), program, &ops)?;
    }

    /// A backend that lacks classes and features is never handed a program using them, and every refusal of a class
    /// or feature is one the rules make.
    #[test]
    fn test_a_backend_is_never_handed_a_class_or_feature_it_lacks(
        (program, ops) in with_edits(small_lp())
    ) {
        agrees_with_the_rules(Plain::<true>::default(), program, &ops)?;
    }

    /// Whatever the backend does, failing an edit after applying it, failing the reload, or both, the solver's program
    /// is the rules' after an accepted edit and the one before after any refusal, and the next solve finds the
    /// backend holding it.
    #[test]
    fn test_a_failing_backend_never_leaves_the_solver_out_of_sync(
        (program, ops) in with_edits(prop_oneof![small_lp(), small_semi(), small_qp()]),
        faults in prop::collection::vec((any::<bool>(), any::<bool>()), 5),
    ) {
        let mut solver = Solver::new(Mirror::new(Script::optimal(0.0, vec![])), program).unwrap();
        for (op, &(refuse_edit, refuse_load)) in ops.iter().zip(&faults) {
            let before = solver.program().clone();
            let want = edit_plan(&before, op, ModelClasses::ALL, ProgramFeatures::ALL);
            solver.backend_mut().faults = Faults {
                refuse_edits: usize::from(refuse_edit),
                refuse_loads: usize::from(refuse_load),
                ..Faults::default()
            };
            let got = op.apply(&mut solver);
            match want {
                Ok(want) if is_empty(op) || !refuse_edit => {
                    prop_assert!(got.is_ok(), "{op:?}: {got:?}");
                    prop_assert_eq!(solver.program(), &want);
                }
                Ok(_) => {
                    prop_assert_eq!(got.as_ref().err(), Some(&EditError::Backend(MirrorError::Injected)));
                    prop_assert_eq!(solver.program(), &before);
                }
                Err(_) => {
                    prop_assert!(got.is_err());
                    prop_assert_eq!(solver.program(), &before);
                }
            }
            solver.solve(Request::new()).unwrap();
            prop_assert_eq!(solver.backend().held(), Some(solver.program()));
            solver.backend_mut().faults = Faults::default();
        }
    }
}
