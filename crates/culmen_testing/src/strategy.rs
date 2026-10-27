// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property-test strategies: tiny bounded LPs and MILPs, with semi columns or without, the reference solver
//! enumerates, small convex QPs, and sequences of edits of a loaded program.

use culmen_core::{
    Bounds, Col, Constraint, Integrality, LinExpr, NewCol, NewRow, Program, ProgramColumn, ProgramPlan, ProgramRow,
    Row, Sense, Term, TiVec,
};
use itertools::Itertools;
use proptest::prelude::*;

use crate::op::EditOp;

/// A column: integrality, bounds and cost, all small integers.
fn column() -> impl Strategy<Value = ProgramColumn> {
    (any::<bool>(), -3i32..=1, 0i32..=4, -5i32..=5).prop_map(|(integer, lower, width, cost)| ProgramColumn {
        cost: f64::from(cost),
        bounds: Bounds::new(f64::from(lower), f64::from(lower + width)).expect("lower <= upper"),
        integrality: if integer {
            Integrality::Integer
        } else {
            Integrality::Continuous
        },
        name: None,
    })
}

/// Row bounds: both ends, or one end open.
fn row_bounds() -> impl Strategy<Value = Bounds> {
    (-8i32..=4, 0i32..=8, 0u8..3).prop_map(|(lower, width, shape)| {
        let (lower, upper) = (f64::from(lower), f64::from(lower + width));
        match shape {
            0 => Bounds::new(lower, upper),
            1 => Bounds::at_most(upper),
            _ => Bounds::at_least(lower),
        }
        .expect("valid bounds")
    })
}

/// A tiny bounded LP or MILP: one to four columns of which at most two continuous, up to three rows with small
/// integer coefficients, either sense and a small offset.
///
/// # Panics
///
/// When a generated plan fails to build, which a strategy defect would cause.
pub fn small_lp() -> impl Strategy<Value = Program> {
    tiny(
        prop::collection::vec(column(), 1..=4).prop_filter("at most two continuous columns", |columns| {
            columns.iter().filter(|c| !c.integrality.is_integral()).count() <= 2
        }),
    )
}

/// A semi column: semi-continuous or semi-integer, its bounds `1 ..= 4` at most, its cost a small integer.
fn semi_column() -> impl Strategy<Value = ProgramColumn> {
    (any::<bool>(), 1i32..=2, 0i32..=2, -5i32..=5).prop_map(|(integer, lower, width, cost)| {
        let bounds = Bounds::new(f64::from(lower), f64::from(lower + width)).expect("lower <= upper");
        if integer {
            ProgramColumn::semi_integer(f64::from(cost), bounds)
        } else {
            ProgramColumn::semi_continuous(f64::from(cost), bounds)
        }
    })
}

/// A tiny bounded MILP with semi columns: one to four columns of which one to three semi and at most two not
/// integral, up to three rows with small integer coefficients, either sense and a small offset.
///
/// # Panics
///
/// When a generated plan fails to build, which a strategy defect would cause.
pub fn small_semi() -> impl Strategy<Value = Program> {
    let any_column = prop_oneof![column(), semi_column()];
    let columns = (semi_column(), prop::collection::vec(any_column, 0..=3)).prop_map(|(semi, mut rest)| {
        rest.insert(0, semi);
        rest
    });
    tiny(columns.prop_filter("at most two columns not integral", |columns| {
        columns.iter().filter(|c| !c.integrality.is_integral()).count() <= 2
    }))
}

/// A tiny program over the given columns: up to three rows with small integer coefficients, either sense and a
/// small offset.
///
/// # Panics
///
/// When a generated plan fails to build, which a strategy defect would cause.
fn tiny(columns: impl Strategy<Value = Vec<ProgramColumn>>) -> impl Strategy<Value = Program> {
    columns
        .prop_flat_map(|columns| {
            let n = columns.len();
            let row = (row_bounds(), prop::collection::vec((0..n, -4i32..=4), 1..=4));
            (
                Just(columns),
                prop::collection::vec(row, 0..=3),
                any::<bool>(),
                -3i32..=3,
            )
        })
        .prop_map(|(columns, rows, maximize, offset)| {
            let rows = rows
                .into_iter()
                .map(|(bounds, terms)| ProgramRow {
                    bounds,
                    terms: terms.into_iter().map(|(j, a)| (Col::new(j), f64::from(a))).collect(),
                    name: None,
                })
                .collect();
            let sense = if maximize { Sense::Maximize } else { Sense::Minimize };
            ProgramPlan {
                sense,
                offset: f64::from(offset),
                columns: columns.into(),
                rows,
                ..ProgramPlan::new()
            }
            .build()
            .expect("a valid plan")
        })
}

/// The weight on the identity in a QP's Hessian `AᵀA + εI`, which makes it positive definite.
pub const QP_RIDGE: f64 = 0.1;

/// A small convex QP: two to four continuous columns in boxes of small integers, the quadratic part `½ xᵀHx`
/// with `H = AᵀA + εI` ([`QP_RIDGE`]) for a random integer `A`, small integer costs, and up to two rows that hold
/// at the boxes' midpoint, so it is feasible and bounded. Maximizing negates the objective, so it stays concave.
///
/// # Panics
///
/// When a generated plan fails to build, which a strategy defect would cause.
pub fn small_qp() -> impl Strategy<Value = Program> {
    (2..=4usize)
        .prop_flat_map(|n| {
            let boxes = prop::collection::vec((-3i32..=1, 1i32..=4), n);
            let a = prop::collection::vec(prop::collection::vec(-3i32..=3, n), 1..=n);
            let costs = prop::collection::vec(-5i32..=5, n);
            let rows = prop::collection::vec((prop::collection::vec(-3i32..=3, n), 0i32..=3, 0i32..=3), 0..=2);
            (boxes, a, costs, rows, any::<bool>())
        })
        .prop_map(|(boxes, a, costs, rows, maximize)| {
            let sign = if maximize { -1.0 } else { 1.0 };
            let columns: TiVec<Col, ProgramColumn> = boxes
                .iter()
                .zip(&costs)
                .map(|(&(lower, width), &cost)| {
                    let bounds = Bounds::new(f64::from(lower), f64::from(lower + width)).expect("lower <= upper");
                    ProgramColumn::continuous(sign * f64::from(cost), bounds)
                })
                .collect();
            let mid: Vec<f64> = boxes.iter().map(|&(l, w)| f64::from(l) + f64::from(w) / 2.0).collect();
            ProgramPlan {
                sense: if maximize { Sense::Maximize } else { Sense::Minimize },
                offset: 0.0,
                rows: rows.into_iter().map(|row| row_through(&mid, row)).collect(),
                quadratic: half_gram(&a, boxes.len())
                    .into_iter()
                    .map(|(i, j, q)| (i, j, sign * q))
                    .collect(),
                columns,
                ..ProgramPlan::new()
            }
            .build()
            .expect("a convex plan")
        })
}

/// The entries `(i, j, q)` of `½ xᵀ(AᵀA + εI)x`, `i <= j`: `q = ½ Hᵢᵢ` on the diagonal and `Hᵢⱼ` off it.
fn half_gram(a: &[Vec<i32>], n: usize) -> Vec<(Col, Col, f64)> {
    let h = |i: usize, j: usize| a.iter().map(|row| f64::from(row[i]) * f64::from(row[j])).sum::<f64>();
    (0..n)
        .array_combinations()
        .map(|[i, j]| (Col::new(i), Col::new(j), h(i, j)))
        .chain((0..n).map(|i| (Col::new(i), Col::new(i), (h(i, i) + QP_RIDGE) / 2.0)))
        .collect()
}

/// A row with the given coefficients whose bounds hold its activity at `point`, widened by `below` and `above`.
fn row_through(point: &[f64], (coefficients, below, above): (Vec<i32>, i32, i32)) -> ProgramRow {
    let activity: f64 = coefficients.iter().zip(point).map(|(&c, x)| f64::from(c) * x).sum();
    ProgramRow {
        bounds: Bounds::new(activity - f64::from(below), activity + f64::from(above)).expect("lower <= upper"),
        terms: coefficients
            .into_iter()
            .enumerate()
            .map(|(j, c)| (Col::new(j), f64::from(c)))
            .collect(),
        name: None,
    }
}

/// A number in an edit: mostly a small integer, rarely NaN or an infinity.
///
/// No huge finite number: a solver may refuse one the program accepts, as HiGHS refuses coefficients from `1e15`.
fn edit_number() -> impl Strategy<Value = f64> {
    prop_oneof![
        12 => (-3i32..=3).prop_map(f64::from),
        1 => prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
    ]
}

/// Bounds in an edit: a box of small integers, a box above zero as a semi column needs, one end open, or free.
fn edit_bounds() -> impl Strategy<Value = Bounds> {
    let between =
        |(lower, width): (i32, i32)| Bounds::new(f64::from(lower), f64::from(lower + width)).expect("lower <= upper");
    prop_oneof![
        4 => (-3i32..=2, 0i32..=4).prop_map(between),
        2 => (1i32..=2, 0i32..=2).prop_map(between),
        1 => (-3i32..=3).prop_map(|upper| Bounds::at_most(f64::from(upper)).expect("a finite end")),
        1 => (-3i32..=3).prop_map(|lower| Bounds::at_least(f64::from(lower)).expect("a finite end")),
        1 => Just(Bounds::FREE),
    ]
}

/// An integrality, the semi ones rarer.
fn edit_integrality() -> impl Strategy<Value = Integrality> {
    prop_oneof![
        3 => Just(Integrality::Continuous),
        3 => Just(Integrality::Integer),
        1 => Just(Integrality::SemiContinuous),
        1 => Just(Integrality::SemiInteger),
    ]
}

/// A name or none.
fn edit_name() -> impl Strategy<Value = Option<String>> {
    prop::option::of(0u8..3).prop_map(|n| n.map(|n| format!("edited{n}")))
}

/// A constraint's two ends: mostly a valid interval of small integers or a half-line, rarely an empty interval or
/// NaN.
fn constraint_ends() -> impl Strategy<Value = (f64, f64)> {
    prop_oneof![
        6 => (-4i32..=4, 0i32..=6, 0u8..3).prop_map(|(lower, width, shape)| {
            let (lower, upper) = (f64::from(lower), f64::from(lower + width));
            match shape {
                0 => (lower, upper),
                1 => (f64::NEG_INFINITY, upper),
                _ => (lower, f64::INFINITY),
            }
        }),
        1 => (1i32..=3).prop_map(|gap| (f64::from(gap), 0.0)),
        1 => Just((f64::NAN, 1.0)),
    ]
}

/// Terms naming columns below `cols`.
fn edit_terms(cols: usize) -> impl Strategy<Value = Vec<(Col, f64)>> {
    prop::collection::vec(((0..cols).prop_map(Col::new), edit_number()), 0..=3)
}

/// A new row whose terms name columns below `cols`: a constraint with a constant, named or not, or a program row.
fn new_row(cols: usize) -> impl Strategy<Value = NewRow> {
    let constraint = (edit_terms(cols), edit_number(), constraint_ends(), edit_name()).prop_map(
        |(terms, constant, (lower, upper), name)| NewRow::Constraint {
            constraint: Constraint {
                expr: LinExpr {
                    terms: terms
                        .into_iter()
                        .map(|(col, coefficient)| Term { col, coefficient })
                        .collect(),
                    constant,
                },
                lower,
                upper,
            },
            name,
        },
    );
    let row = (edit_terms(cols), edit_bounds(), edit_name())
        .prop_map(|(terms, bounds, name)| NewRow::Row(ProgramRow { bounds, terms, name }));
    prop_oneof![2 => constraint, 1 => row]
}

/// A new column whose entries name rows below `rows`.
fn new_col(rows: usize) -> impl Strategy<Value = NewCol> {
    let entries = prop::collection::vec(((0..rows).prop_map(Row::new), edit_number()), 0..=3);
    (edit_number(), edit_bounds(), edit_integrality(), edit_name(), entries).prop_map(
        |(cost, bounds, integrality, name, entries)| NewCol {
            column: ProgramColumn {
                cost,
                bounds,
                integrality,
                name,
            },
            entries,
        },
    )
}

/// One edit whose ids lie below `cols + 2` columns and `rows + 2` rows, so some name nothing.
fn edit_op(cols: usize, rows: usize) -> impl Strategy<Value = EditOp> {
    let (cols, rows) = (cols + 2, rows + 2);
    let col = move || (0..cols).prop_map(Col::new);
    let row = move || (0..rows).prop_map(Row::new);
    prop_oneof![
        prop::collection::vec((col(), edit_bounds()), 0..=3).prop_map(EditOp::ColBounds),
        prop::collection::vec((col(), edit_number()), 0..=3).prop_map(EditOp::ColCosts),
        prop::collection::vec((col(), edit_integrality()), 0..=3).prop_map(EditOp::ColIntegrality),
        prop::collection::vec((row(), edit_bounds()), 0..=3).prop_map(EditOp::RowBounds),
        prop::collection::vec(new_row(cols), 0..=2).prop_map(EditOp::AddRows),
        prop::collection::vec(new_col(rows), 0..=2).prop_map(EditOp::AddCols),
    ]
}

/// One to five edits of `program`: ids below its counts plus two, numbers mostly small integers and rarely NaN or
/// infinite, bounds valid but not always for a semi column, integrality of every kind, rows and columns of every
/// shape. Many are refused; every refusal is one the edit rules state.
pub fn edits(program: &Program) -> impl Strategy<Value = Vec<EditOp>> + use<> {
    prop::collection::vec(edit_op(program.num_cols(), program.num_rows()), 1..=5)
}

/// Programs from `programs`, each with edits of it from [`edits`].
pub fn with_edits(programs: impl Strategy<Value = Program>) -> impl Strategy<Value = (Program, Vec<EditOp>)> {
    programs.prop_flat_map(|program| {
        let ops = edits(&program);
        (Just(program), ops)
    })
}

#[cfg(test)]
mod tests {
    use culmen_core::{ModelClasses, ProgramFeatures};
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    use super::*;
    use crate::reference::edit_plan;

    #[test]
    fn test_edits_of_every_kind_are_both_accepted_and_refused() {
        let mut runner = TestRunner::deterministic();
        let strategy = with_edits(prop_oneof![small_lp(), small_semi(), small_qp()]);
        let (mut accepted, mut refused) = ([0usize; 6], [0usize; 6]);
        for _ in 0..2000 {
            let (program, ops) = strategy.new_tree(&mut runner).unwrap().current();
            for op in &ops {
                let kind = match op {
                    EditOp::ColBounds(_) => 0,
                    EditOp::ColCosts(_) => 1,
                    EditOp::ColIntegrality(_) => 2,
                    EditOp::RowBounds(_) => 3,
                    EditOp::AddRows(_) => 4,
                    EditOp::AddCols(_) => 5,
                };
                match edit_plan(&program, op, ModelClasses::ALL, ProgramFeatures::ALL) {
                    Ok(_) => accepted[kind] += 1,
                    Err(_) => refused[kind] += 1,
                }
            }
        }
        assert!(accepted.iter().all(|&n| n >= 100), "accepted {accepted:?}");
        assert!(refused.iter().all(|&n| n >= 50), "refused {refused:?}");
    }
}
