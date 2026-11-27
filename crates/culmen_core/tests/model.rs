// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The program plan, the model, expressions and the LP-file writer.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{
    Bounds, Col, Constraint, ExprPlace, Integrality, InvalidBounds, LinExpr, Linear, Model, ModelConstraint,
    ModelError, ModelVar, NewCol, ProgramColumn, ProgramError, ProgramPlan, ProgramRow, Row, Sense, Term, TiSlice,
    TiVec, Var, VarMap, bounds, product, set,
};
use proptest::prelude::*;

#[test]
fn test_bounds_refuse_every_empty_or_undefined_interval() {
    assert_eq!(Bounds::new(f64::NAN, 1.0), Err(InvalidBounds::NotANumber));
    assert_eq!(
        Bounds::new(f64::INFINITY, f64::INFINITY),
        Err(InvalidBounds::LowerIsPlusInfinity)
    );
    assert_eq!(
        Bounds::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        Err(InvalidBounds::UpperIsMinusInfinity)
    );
    assert_eq!(
        Bounds::new(2.0, 1.0),
        Err(InvalidBounds::Empty { lower: 2.0, upper: 1.0 })
    );
    assert_eq!(Bounds::FREE.shifted(1.0), Ok(Bounds::FREE));
    assert!(matches!(
        bounds!(0.0, 1.0).shifted(f64::NAN),
        Err(InvalidBounds::NonFiniteShift { shift }) if shift.is_nan()
    ));
    assert_eq!(bounds!(0.0, f64::MAX).shifted(f64::MAX), Err(InvalidBounds::Overflow));
    assert_eq!(bounds!(1.0, 2.0).violation(3.5), 1.5);
}

#[test]
fn test_plan_merges_repeated_terms_and_drops_zeros() {
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous().cost(1.0));
    let y = plan.col(ProgramColumn::binary().cost(2.0));
    let r = plan.row(bounds!(0.0, 4.0), [(y, 1.0), (x, 2.0), (x, -2.0), (y, 0.5)]);
    let program = plan.build().unwrap();
    assert_eq!(program.row(r).entries().collect::<Vec<_>>(), [(y, 1.5)]);
    assert_eq!(program.num_entries(), 1);
    assert!(program.class().is_discrete());
}

#[test]
fn test_columns_are_built_by_chaining_with_the_defaults_their_integrality_names() {
    let column = |cost: f64, bounds, integrality| ProgramColumn {
        cost,
        bounds,
        integrality,
        name: None,
    };
    let cases = [
        (
            ProgramColumn::continuous(),
            column(0.0, Bounds::NON_NEGATIVE, Integrality::Continuous),
        ),
        (
            ProgramColumn::free(),
            column(0.0, Bounds::FREE, Integrality::Continuous),
        ),
        (
            ProgramColumn::integer(),
            column(0.0, Bounds::NON_NEGATIVE, Integrality::Integer),
        ),
        (
            ProgramColumn::binary(),
            column(0.0, Bounds::BINARY, Integrality::Integer),
        ),
        (
            ProgramColumn::semi_continuous(),
            column(0.0, Bounds::NON_NEGATIVE, Integrality::SemiContinuous),
        ),
        (
            ProgramColumn::semi_integer(),
            column(0.0, Bounds::NON_NEGATIVE, Integrality::SemiInteger),
        ),
    ];
    for (built, expected) in cases {
        assert_eq!(built.clone(), expected);
        assert_eq!(
            NewCol::from(built),
            NewCol {
                column: expected,
                entries: Vec::new()
            }
        );
    }
    let named = ProgramColumn::integer().cost(3u8).bounds(bounds!(1.0, 4.0)).name("n");
    assert_eq!(
        named,
        ProgramColumn {
            cost: 3.0,
            bounds: bounds!(1.0, 4.0),
            integrality: Integrality::Integer,
            name: Some("n".to_owned()),
        }
    );
    let (a, b) = (Row::new(0), Row::new(2));
    let new = NewCol::integer()
        .cost(3u8)
        .bounds(bounds!(1.0, 4.0))
        .name("n")
        .entry(a, 2)
        .entries([(b, 0.5), (a, 1.0)]);
    assert_eq!(
        new,
        NewCol {
            column: named,
            entries: vec![(a, 2.0), (b, 0.5), (a, 1.0)],
        }
    );
    assert_eq!(NewCol::free(), NewCol::from(ProgramColumn::free()));
    assert_eq!(NewCol::binary().column, ProgramColumn::binary());
    assert_eq!(NewCol::continuous().column, ProgramColumn::continuous());
    assert_eq!(NewCol::semi_continuous().column, ProgramColumn::semi_continuous());
    assert_eq!(NewCol::semi_integer().column, ProgramColumn::semi_integer());
}

#[test]
fn test_plan_names_the_first_defect() {
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::free().cost(f64::NAN));
    assert!(matches!(plan.clone().build(), Err(ProgramError::NonFiniteCost { col, .. }) if col == x));
    plan.columns[x].cost = 1.0;
    let r = plan.row(Bounds::FREE, [(x, f64::INFINITY)]);
    assert_eq!(
        plan.clone().build(),
        Err(ProgramError::NonFiniteCoefficient {
            row: r,
            col: x,
            value: f64::INFINITY
        })
    );
    plan.rows[r].terms = vec![(Col::new(7), 1.0)];
    assert_eq!(
        plan.clone().build(),
        Err(ProgramError::UnknownCol {
            row: r,
            col: Col::new(7)
        })
    );
    plan.rows[r].terms = vec![(x, f64::MAX), (x, f64::MAX)];
    assert!(
        matches!(plan.clone().build(), Err(ProgramError::NonFiniteCoefficient { .. })),
        "the sum overflows"
    );
    plan.rows[r].terms.clear();
    plan.offset = f64::NEG_INFINITY;
    assert_eq!(
        plan.build(),
        Err(ProgramError::NonFiniteOffset {
            value: f64::NEG_INFINITY
        })
    );
}

#[test]
fn test_model_errors_name_the_variable_or_constraint() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().bounds(3.0, 1.0).name("stock"));
    assert_eq!(
        m.build().unwrap_err(),
        ModelError::VarBounds {
            var: ModelVar {
                col: x,
                name: Some("stock".to_owned())
            },
            source: InvalidBounds::Empty { lower: 3.0, upper: 1.0 }
        }
    );

    let mut m = Model::new();
    let x2 = m.add_var(Var::continuous().name("flow"));
    m.add_constraint(("cap", (f64::NAN * x2).leq(1.0)));
    let error = m.build().unwrap_err();
    assert_eq!(
        error.to_string(),
        "Model: constraint `cap` has the non-finite coefficient NaN on variable `flow`"
    );

    let mut m = Model::new();
    let y = m.add_var(Var::continuous());
    let row = m.add_constraint(y.between(2.0, 1.0));
    let error = m.build().unwrap_err();
    assert!(matches!(&error, ModelError::ConstraintBounds { constraint, .. }
        if *constraint == ModelConstraint { row, name: None }));
    assert_eq!(
        error.to_string(),
        "Model: constraint Row(0) has invalid bounds: Bounds: the lower end 2 exceeds the upper end 1"
    );

    let mut m = Model::new();
    m.minimize(x + 1.0);
    assert!(matches!(m.build(), Err(ModelError::UnknownVar { place: ExprPlace::Objective, col }) if col == x));
    let mut m = Model::new();
    let row = m.add_constraint(("cap", (2.0 * x).leq(1.0)));
    let error = m.build().unwrap_err();
    assert_eq!(
        error,
        ModelError::UnknownVar {
            place: ExprPlace::Constraint(ModelConstraint {
                row,
                name: Some("cap".to_owned())
            }),
            col: x
        }
    );
    assert_eq!(
        error.to_string(),
        "Model: `cap` names Col(0), which this model did not hand out"
    );
}

#[test]
fn test_model_names_a_constraint_whose_repeated_terms_sum_past_finite() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().name("flow"));
    let row = m.add_constraint(("cap", (1e308 * x + 1e308 * x).leq(1.0)));
    assert_eq!(
        m.build().unwrap_err(),
        ModelError::Coefficient {
            constraint: ModelConstraint {
                row,
                name: Some("cap".to_owned())
            },
            var: ModelVar {
                col: x,
                name: Some("flow".to_owned())
            },
            value: f64::INFINITY
        }
    );
}

#[test]
fn test_bounds_shifted_by_an_infinity_name_the_shift() {
    let error = bounds!(0.0, 1.0).shifted(f64::INFINITY).unwrap_err();
    assert_eq!(error, InvalidBounds::NonFiniteShift { shift: f64::INFINITY });
    assert_eq!(error.to_string(), "Bounds: the shift inf is not finite");
}

#[test]
fn test_model_moves_constants_into_bounds_and_sums_objective_terms() {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().upper(9.0).name("x"));
    let y = m.add_var(Var::free());
    let r = m.add_constraint((x + 1).leq(3.0 - y));
    m.maximize(2.0 * x + x - y / 2.0 + 5);
    let program = m.build().unwrap();
    assert_eq!(program.sense(), Sense::Maximize);
    assert_eq!(program.offset(), 5.0);
    assert_eq!(program.columns()[x].cost, 3.0);
    assert_eq!(program.columns()[y].cost, -0.5);
    assert_eq!(program.columns()[x].integrality, Integrality::Integer);
    assert_eq!(program.columns()[x].bounds, bounds!(0.0, 9.0));
    assert_eq!(program.columns()[y].bounds, Bounds::FREE);
    assert_eq!(program.row_bounds()[r], bounds!(f64::NEG_INFINITY, 2.0));
    assert_eq!(program.row(r).entries().collect::<Vec<_>>(), [(x, 1.0), (y, 1.0)]);
}

#[test]
fn test_operators_build_the_expected_terms() {
    let (x, y) = (Col::new(0), Col::new(1));
    let e = 3 * x - (y * 2.0) + 1.5 - x;
    assert_eq!(
        e,
        LinExpr {
            terms: vec![
                Term {
                    col: x,
                    coefficient: 3.0
                },
                Term {
                    col: y,
                    coefficient: -2.0
                },
                Term {
                    col: x,
                    coefficient: -1.0
                },
            ],
            constant: 1.5,
        }
    );
    assert_eq!(
        -(x + 2) * 2,
        LinExpr {
            terms: vec![Term {
                col: x,
                coefficient: -2.0
            }],
            constant: -4.0
        }
    );
    assert_eq!(
        1.0 - x,
        LinExpr {
            terms: vec![Term {
                col: x,
                coefficient: -1.0
            }],
            constant: 1.0
        }
    );
    let mut sum = LinExpr::new();
    sum += x;
    sum -= 2.0 * y;
    sum *= 3.0;
    assert_eq!(sum, 3.0 * x.into_expr() - 6.0 * y);
    assert_eq!(
        x.equals(y),
        Constraint {
            expr: x - y,
            lower: 0.0,
            upper: 0.0
        }
    );
    assert_eq!([x, y].into_iter().map(|c| 2 * c).sum::<LinExpr>(), 2 * x + 2 * y);
    assert_eq!([x, y].into_iter().map(|c| 2 * c).collect::<LinExpr>(), 2 * x + 2 * y);
    let mut extended = LinExpr::from(3);
    extended.extend([x, y]);
    assert_eq!(extended, x + y + 3);
    assert_eq!(x / 4, x * 0.25);
    assert_eq!((2 * x) / 4, x * 0.5);
    let mut halved = 2 * x + 4;
    halved /= 2.0;
    assert_eq!(halved, x + 2);
    assert_eq!((2 * x + 4) / 2, x + 2);
}

#[test]
fn test_every_number_that_converts_into_f64_without_loss_scales_and_compares() {
    let (x, y) = (Col::new(0), Col::new(1));
    let (a, b, c, d, e, f) = (2u8, 2u16, 2u32, 2i8, 2i16, 2.0f32);
    let doubled = 2.0 * x;
    for scaled in [x * a, x * b, x * c, x * d, x * e, x * f, Term::from(x) * a] {
        assert_eq!(scaled, doubled);
    }
    assert_eq!(x / c, x * 0.5);
    let limit: u32 = 7;
    assert_eq!(x.leq(limit), x.leq(7.0));
    assert_eq!(x.geq(a), x.geq(2.0));
    assert_eq!(x + c - d, x + 0.0);
    assert_eq!((x + y) * e, 2.0 * x + 2.0 * y);
    assert_eq!(LinExpr::from(f), LinExpr::from(2.0));
    let mut sum = x + 1;
    sum *= a;
    sum /= b;
    assert_eq!(sum, x + 1);
    assert_eq!((x * y) * c, 2.0 * x * y);
    assert_eq!((x * x + x) * f, 2.0 * (x * x) + 2.0 * x);
    // An unsuffixed literal still infers through a method call on the product.
    assert_eq!((2 * x).leq(4), (x * 2.0).leq(4.0));
    assert_eq!((x + 2 * y).geq(1), (x + 2.0 * y).geq(1.0));
}

#[test]
fn test_keyed_families_over_filtered_products() {
    let mut m = Model::new();
    let fits = |p: usize, l: char| !(p == 1 && l == 'b');
    let x = m.add_var_map(product(&(0..2), &['a', 'b']).filter(|&(p, l)| fits(p, l)), |&(p, l)| {
        Var::binary().name(format!("x[{p},{l}]"))
    });
    assert_eq!(x.len(), 3);
    assert_eq!(x.get(&(1, 'b')), None);
    assert_eq!(m.var_name(x[(1, 'a')]), Some("x[1,a]"));
    let values: TiVec<Col, f64> = vec![1.0, 0.0, 1.0].into();
    assert_eq!(
        x.values(&values).collect::<Vec<_>>(),
        [(&(0, 'a'), 1.0), (&(0, 'b'), 0.0), (&(1, 'a'), 1.0)]
    );
}

#[test]
fn test_keyed_families_are_read_by_a_borrowed_key() {
    let products = vec!["cork".to_owned(), "glass".to_owned()];
    let mut m = Model::new();
    let take = m.add_var_map(products.clone(), |p| Var::binary().name(format!("take[{p}]")));
    for p in &products {
        assert_eq!(Some(take[p]), take.get(p.as_str()));
        assert_eq!(take[p], take[p.clone()]);
    }
    assert!(take.contains_key("glass") && !take.contains_key("tin"));
    assert_eq!(m.var_name(take[&products[1]]), Some("take[glass]"));
}

#[test]
fn test_a_model_takes_a_constraint_named_or_not_or_a_program_row_as_a_solver_does() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    let unnamed = m.add_constraint((x + y).leq(4));
    let named = m.add_constraint(("cap", (x + 1).leq(3)));
    let data = m.add_constraint(ProgramRow {
        bounds: bounds!(1.0, 2.0),
        terms: vec![(y, 1.0), (x, 2.0), (y, 1.0)],
        name: Some("data".to_owned()),
    });
    let family = m.add_constraint_map(0..2, |&i| x.geq(f64::from(i as u8)));
    assert_eq!(m.constraint_name(family[&1]), None);
    let program = m.build().unwrap();
    assert_eq!(program.row(unnamed).name(), None);
    assert_eq!(
        (program.row(named).name(), program.row(named).bounds()),
        (Some("cap"), bounds!(f64::NEG_INFINITY, 2.0))
    );
    assert_eq!(program.row(data).entries().collect::<Vec<_>>(), [(x, 2.0), (y, 2.0)]);
    assert_eq!(program.row(data).name(), Some("data"));

    // A program row is checked in the model's terms.
    m.add_constraint(ProgramRow {
        bounds: Bounds::FREE,
        terms: vec![(Col::new(9), 1.0)],
        name: Some("stray".to_owned()),
    });
    assert!(matches!(
        m.build(),
        Err(ModelError::UnknownVar { place: ExprPlace::Constraint(ModelConstraint { name: Some(n), .. }), col })
            if n == "stray" && col == Col::new(9)
    ));
}

#[test]
fn test_a_family_of_variables_is_made_from_bounds() {
    let mut m = Model::new();
    let flow = m.add_var_map(
        ['a', 'b'],
        |&k| if k == 'a' { bounds!(0.0, 1.0) } else { bounds!(0.0, 2.0) },
    );
    let program = m.build().unwrap();
    assert_eq!(program.columns()[flow['b']].bounds, bounds!(0.0, 2.0));
    assert_eq!(program.columns()[flow['a']].integrality, Integrality::Continuous);
}

#[test]
#[should_panic(expected = "a key is inserted twice")]
fn test_keyed_family_refuses_repeated_keys() {
    let mut m = Model::new();
    let _ = m.add_var_map([1, 1], |_| Var::binary());
}

#[test]
fn test_lp_file_writes_every_section() {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(-2.0, 5.0).name("x[1]"));
    let y = m.add_var(Var::free().name("1y"));
    m.add_constraint(("range", (x + y).between(-1.0, 4.0)));
    m.add_constraint(("fix", (x - y).equals(0.0)));
    m.minimize(x - 2.0 * y + 3.0);
    let text = m.build().unwrap().to_lp_string().unwrap();
    let expected = "Minimize\n obj: 1 x_1_ - 2 _1y + 3\nSubject To\n range_lo: 1 x_1_ + 1 _1y >= -1\n \
                    range_hi: 1 x_1_ + 1 _1y <= 4\n fix: 1 x_1_ - 1 _1y = 0\nBounds\n -2 <= x_1_ <= 5\n _1y free\n\
                    General\n x_1_\nEnd\n";
    assert_eq!(text, expected);
}

#[test]
fn test_lp_file_escapes_names_the_format_reserves() {
    // HiGHS 1.16 refuses to read a file whose column is named `sos`, `min`, `max`, `minimize`, `maximum` or
    // `minimise`: they are keywords of the format.
    let mut m = Model::new();
    let x = m.add_var(Var::integer().upper(3).name("max"));
    let y = m.add_var(Var::continuous().upper(3).name("SOS"));
    let z = m.add_var(Var::continuous().upper(3).name("Minimize"));
    m.add_constraint(("min", (x + y + z).geq(1)));
    m.minimize(2.0 * x + y + z);
    let text = m.build().unwrap().to_lp_string().unwrap();
    let expected = "Minimize\n obj: 2 _max + 1 _SOS + 1 _Minimize\nSubject To\n _min: 1 _max + 1 _SOS + 1 _Minimize >= 1\n\
                    Bounds\n 0 <= _max <= 3\n 0 <= _SOS <= 3\n 0 <= _Minimize <= 3\nGeneral\n _max\nEnd\n";
    assert_eq!(text, expected);
}

#[test]
fn test_lp_file_labels_every_column_and_row_apart() {
    // Named lines keep their names in order, the ranged `band` first; every other line takes the next free suffix.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().name("x1"));
    let y = m.add_var(Var::continuous());
    let a = m.add_var(Var::continuous().name("a"));
    let b = m.add_var(Var::continuous().name("a"));
    m.add_constraint(("r1", (x - y).geq(1)));
    m.add_constraint((a - b).geq(1));
    m.add_constraint(("band", (x + a).between(1, 3)));
    m.add_constraint(("band_lo", (y + b).leq(5)));
    m.add_constraint(("obj", (x + b).leq(7)));
    m.minimize(x + 2 * y + a + b);
    let text = m.build().unwrap().to_lp_string().unwrap();
    let expected = "Minimize\n obj: 1 x1 + 2 x1~1 + 1 a + 1 a~1\nSubject To\n r1: 1 x1 - 1 x1~1 >= 1\n \
                    r1~1: 1 a - 1 a~1 >= 1\n band_lo: 1 x1 + 1 a >= 1\n band_hi: 1 x1 + 1 a <= 3\n \
                    band_lo~1: 1 x1~1 + 1 a~1 <= 5\n obj~1: 1 x1 + 1 a~1 <= 7\nBounds\nEnd\n";
    assert_eq!(text, expected);
}

#[test]
fn test_product_ends_for_good_over_an_inner_set_that_is_not_fused() {
    /// Every other call yields the next number: `Some(1)`, `None`, `Some(3)`, `None`, …
    #[derive(Clone)]
    struct Flicker(u8);

    impl Iterator for Flicker {
        type Item = u8;

        fn next(&mut self) -> Option<u8> {
            self.0 = self.0.wrapping_add(1);
            (self.0 % 2 == 1).then_some(self.0)
        }
    }

    fn fused<I: core::iter::FusedIterator>(iter: I) -> I {
        iter
    }

    let inner = set(Flicker(0));
    let mut pairs = fused(product(&(0..1), &inner));
    assert_eq!(pairs.next(), Some((0, 1)));
    assert_eq!(pairs.next(), None);
    assert_eq!(pairs.next(), None, "a fused iterator stays at its end");
}

/// A small expression over `n` columns: terms with repeats, and a constant.
fn expression(n: usize) -> impl Strategy<Value = LinExpr> {
    (prop::collection::vec((0..n, -5i32..=5), 0..8), -10i32..=10).prop_map(|(terms, constant)| LinExpr {
        terms: terms
            .into_iter()
            .map(|(i, c)| Term {
                col: Col::new(i),
                coefficient: f64::from(c),
            })
            .collect(),
        constant: f64::from(constant),
    })
}

proptest! {
    /// A built row holds exactly the constraint: at every point, the row's activity lies in its bounds exactly when
    /// the expression, evaluated directly, lies in the constraint's. Small integers keep the arithmetic exact.
    #[test]
    fn test_built_rows_keep_the_constraint(
        expr in expression(3),
        lower in -20i32..=0,
        width in 0i32..=20,
        point in prop::collection::vec(-4i32..=4, 3),
    ) {
        let mut m = Model::new();
        m.add_vars(3, |_| Var::free());
        let (lower, upper) = (f64::from(lower), f64::from(lower + width));
        let row = m.add_constraint(expr.clone().between(lower, upper));
        let program = m.build().unwrap();
        let values: TiVec<Col, f64> = point.iter().map(|&v| f64::from(v)).collect();
        let direct = expr.value(&values);
        let activity: f64 = program.row(row).entries().map(|(c, a)| a * values[c]).sum();
        let held = (lower..=upper).contains(&direct);
        prop_assert_eq!(program.row_bounds()[row].contains(activity), held);
        prop_assert_eq!(program.max_violation(&values) == 0.0, held);
        prop_assert!(program.row(row).entries().all(|(_, a)| a != 0.0));
        prop_assert_eq!(program.rows().collect::<Vec<_>>(), [Row::new(0)]);
    }
}

#[test]
fn test_row_views_read_bounds_name_and_entries_both_ways() {
    let mut m = Model::new();
    let x = m.add_vars(3, |_| Var::continuous());
    let r = m.add_constraint(("mix", (2 * x[2] + x[0] - x[1]).leq(4.0)));
    let program = m.build().unwrap();
    let row = program.row(r);
    assert_eq!(row.bounds(), bounds!(f64::NEG_INFINITY, 4.0));
    assert_eq!(row.name(), Some("mix"));
    assert_eq!((row.len(), row.is_empty()), (3, false));
    let values: TiVec<Col, f64> = vec![1.0, 2.0, 3.0].into();
    assert_eq!(row.activity(&values), 5.0);

    let mut entries = program.row(r).entries();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries.next_back(), Some((x[2], 2.0)));
    assert_eq!(entries.len(), 2);
    assert_eq!(entries.clone().collect::<Vec<_>>(), [(x[0], 1.0), (x[1], -1.0)]);
    assert_eq!(entries.next(), Some((x[0], 1.0)));
    assert_eq!(entries.next(), Some((x[1], -1.0)));
    assert_eq!(
        (entries.next(), entries.next_back(), entries.next()),
        (None, None, None)
    );
    assert_eq!(
        row.into_iter().rev().collect::<Vec<_>>(),
        [(x[2], 2.0), (x[1], -1.0), (x[0], 1.0)]
    );
}

#[test]
fn test_keyed_iterators_run_both_ways_with_exact_lengths() {
    let mut m = Model::new();
    let x: VarMap<char> = m.add_var_map(['a', 'b', 'c'], |_| Var::binary());
    let mut iter = x.iter();
    assert_eq!(iter.len(), 3);
    assert_eq!(iter.next_back(), Some((&'c', x['c'])));
    assert_eq!(iter.len(), 2);
    assert_eq!(iter.clone().count(), 2);
    assert_eq!(x.keys().rev().collect::<String>(), "cba");
    assert_eq!(x.keys().len(), 3);
    assert_eq!(x.handles().rev().collect::<Vec<_>>(), [x['c'], x['b'], x['a']]);
    assert_eq!(x.handles().len(), 3);
    let mut through_ref = Vec::new();
    for (key, col) in &x {
        through_ref.push((*key, col));
    }
    assert_eq!(through_ref, x.iter().map(|(k, c)| (*k, c)).collect::<Vec<_>>());
    let mut keys = x.keys();
    keys.by_ref().for_each(drop);
    assert_eq!((keys.next(), keys.next_back()), (None, None));
    assert!(format!("{:?}", x.handles()).starts_with("Handles("));

    let point: TiVec<Col, f64> = vec![1.0, 0.0, 1.0].into();
    let mut values = x.values(&point);
    assert_eq!(values.len(), 3);
    assert_eq!(values.next_back(), Some((&'c', 1.0)));
    assert_eq!(values.clone().collect::<Vec<_>>(), [(&'a', 1.0), (&'b', 0.0)]);
    assert!(format!("{values:?}").starts_with("Values"));
}

#[test]
fn test_program_cols_and_rows_run_both_ways_with_exact_lengths() {
    let mut m = Model::new();
    let x = m.add_vars(3, |_| Var::binary());
    let once = m.add_constraint(LinExpr::sum(x.iter().copied()).leq(1.0));
    let program = m.build().unwrap();
    let mut cols = program.cols();
    assert_eq!(cols.len(), 3);
    assert_eq!(cols.next_back(), Some(x[2]));
    assert_eq!(cols.clone().collect::<Vec<_>>(), x[..2]);
    cols.by_ref().for_each(drop);
    assert_eq!((cols.next(), cols.next_back()), (None, None));
    assert_eq!(program.rows().rev().collect::<Vec<_>>(), [once]);
    assert_eq!(program.rows().len(), 1);
}

proptest! {
    /// `product` pairs every member of the first set with every member of the second, the first outermost, as two
    /// nested loops over the members do; whichever set is empty, it pairs nothing.
    #[test]
    fn test_product_is_the_nested_loop(
        a in prop::collection::vec(0u8..4, 0..4),
        b in prop::collection::btree_set(0u8..6, 0..4),
        keep in 0u8..3,
    ) {
        let mut expected = Vec::new();
        for x in &a {
            for y in &b {
                expected.push((*x, *y));
            }
        }
        prop_assert_eq!(product(&a, &b).collect::<Vec<_>>(), expected.clone());
        let filtered = set(b.iter().copied().filter(|y| y % 3 != keep));
        let kept: Vec<_> = expected.into_iter().filter(|(_, y)| y % 3 != keep).collect();
        let pairs = product(&a, &filtered);
        prop_assert_eq!(pairs.clone().collect::<Vec<_>>(), kept.clone());
        let mut fused = pairs.fuse();
        fused.by_ref().for_each(drop);
        prop_assert_eq!(fused.next(), None);
    }
}

/// An expression whose coefficients are huge or small, with repeated columns, over `n` columns.
fn huge_expression(n: usize) -> impl Strategy<Value = LinExpr> {
    let coefficient = prop::sample::select(vec![f64::MAX, -f64::MAX, 1e308, -1e308, 1.0, 0.0]);
    prop::collection::vec((0..n, coefficient), 0..6).prop_map(|terms| LinExpr {
        terms: terms
            .into_iter()
            .map(|(i, coefficient)| Term {
                col: Col::new(i),
                coefficient,
            })
            .collect(),
        constant: 0.0,
    })
}

proptest! {
    /// The model's checks are the program's: a build fails with a model error naming the constraint exactly when a
    /// column's coefficients add up past finite, and a model that builds is a plan the program accepts as it is.
    #[test]
    fn test_models_never_build_a_program_the_program_refuses(expr in huge_expression(2)) {
        let mut m = Model::new();
        m.add_vars(2, |_| Var::free());
        m.add_constraint(expr.clone().leq(1.0));
        let overflows = (0..2).any(|i| {
            let sum: f64 = expr.terms.iter().filter(|t| t.col.index() == i).map(|t| t.coefficient).sum();
            !sum.is_finite()
        });
        match m.build() {
            Ok(program) => {
                prop_assert!(!overflows);
                prop_assert_eq!(ProgramPlan::from(program.clone()).build(), Ok(program));
            }
            Err(error) => {
                prop_assert!(matches!(error, ModelError::Coefficient { .. }), "{error:?}");
                prop_assert!(overflows);
            }
        }
    }

    /// A row's entries read backwards are its entries forwards, reversed, and every length is exact.
    #[test]
    fn test_row_entries_reverse_and_count_exactly(expr in expression(4)) {
        let mut m = Model::new();
        m.add_vars(4, |_| Var::free());
        let row = m.add_constraint(expr.leq(0.0));
        let program = m.build().unwrap();
        let forwards: Vec<_> = program.row(row).entries().collect();
        let mut backwards: Vec<_> = program.row(row).entries().rev().collect();
        backwards.reverse();
        prop_assert_eq!(&forwards, &backwards);
        prop_assert_eq!(program.row(row).entries().len(), forwards.len());
        prop_assert_eq!(program.row(row).len(), forwards.len());
        prop_assert!(forwards.windows(2).all(|w| w[0].0 < w[1].0));
    }
}

#[test]
fn test_a_point_with_a_nan_is_infinitely_far_from_feasible() {
    // Found through HiGHS 1.15, which reported an optimal QP point holding a NaN: every comparison with NaN is
    // false, so the point read as inside every bound.
    assert_eq!(bounds!(0.0, 1.0).violation(f64::NAN), f64::INFINITY);
    assert_eq!(
        Integrality::Integer.violation(bounds!(0.0, 1.0), f64::NAN),
        f64::INFINITY
    );
    assert_eq!(
        Integrality::SemiContinuous.violation(bounds!(1.0, 2.0), f64::NAN),
        f64::INFINITY
    );
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous().cost(1.0).bounds(bounds!(0.0, 1.0)));
    let y = plan.col(ProgramColumn::continuous().cost(1.0).bounds(bounds!(0.0, 1.0)));
    plan.row(bounds!(f64::NEG_INFINITY, 1.0), [(x, 1.0), (y, 1.0)]);
    let program = plan.build().unwrap();
    assert_eq!(
        program.max_violation(TiSlice::from_ref(&[0.5, f64::NAN])),
        f64::INFINITY
    );
    assert_eq!(program.max_violation(TiSlice::from_ref(&[0.5, 0.25])), 0.0);
}

#[test]
fn test_an_infinite_value_lies_inside_a_bound_open_on_its_side() {
    // The closed intervals of the extended reals, as `Bounds::contains` reads them.
    let open = bounds!(0.0, f64::INFINITY);
    assert!(open.contains(f64::INFINITY));
    assert_eq!(open.violation(f64::INFINITY), 0.0);
    assert_eq!(open.violation(f64::NEG_INFINITY), f64::INFINITY);
    assert_eq!(Integrality::Integer.violation(open, f64::INFINITY), 0.0);
    assert_eq!(
        Integrality::Integer.violation(bounds!(0.0, 1.0), f64::INFINITY),
        f64::INFINITY
    );
}
