// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The LP plan, the model, expressions and the LP-file writer.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen_core::{
    Bounds, Col, Constraint, ExprPlace, Integrality, InvalidBounds, LinExpr, Linear, LpColumn, LpError, LpPlan, Model,
    ModelConstraint, ModelError, ModelVar, Row, Sense, Term, TiVec, Var, VarMap, bounds, product, set,
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
    let mut plan = LpPlan::new();
    let x = plan.col(LpColumn::continuous(1.0, Bounds::NON_NEGATIVE));
    let y = plan.col(LpColumn::binary(2.0));
    let r = plan.row(bounds!(0.0, 4.0), [(y, 1.0), (x, 2.0), (x, -2.0), (y, 0.5)]);
    let lp = plan.build().unwrap();
    assert_eq!(lp.row(r).entries().collect::<Vec<_>>(), [(y, 1.5)]);
    assert_eq!(lp.num_entries(), 1);
    assert!(lp.is_mip());
}

#[test]
fn test_plan_names_the_first_defect() {
    let mut plan = LpPlan::new();
    let x = plan.col(LpColumn::continuous(f64::NAN, Bounds::FREE));
    assert!(matches!(plan.clone().build(), Err(LpError::NonFiniteCost { col, .. }) if col == x));
    plan.columns[x].cost = 1.0;
    let r = plan.row(Bounds::FREE, [(x, f64::INFINITY)]);
    assert_eq!(
        plan.clone().build(),
        Err(LpError::NonFiniteCoefficient {
            row: r,
            col: x,
            value: f64::INFINITY
        })
    );
    plan.rows[r].terms = vec![(Col::new(7), 1.0)];
    assert_eq!(
        plan.clone().build(),
        Err(LpError::UnknownColumn {
            row: r,
            col: Col::new(7)
        })
    );
    plan.rows[r].terms = vec![(x, f64::MAX), (x, f64::MAX)];
    assert!(
        matches!(plan.clone().build(), Err(LpError::NonFiniteCoefficient { .. })),
        "the sum overflows"
    );
    plan.rows[r].terms.clear();
    plan.offset = f64::NEG_INFINITY;
    assert_eq!(
        plan.build(),
        Err(LpError::NonFiniteOffset {
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
    m.add_named_constraint("cap", (f64::NAN * x2).leq(1.0));
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
    let row = m.add_named_constraint("cap", (2.0 * x).leq(1.0));
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
    let row = m.add_named_constraint("cap", (1e308 * x + 1e308 * x).leq(1.0));
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
    let lp = m.build().unwrap();
    assert_eq!(lp.sense(), Sense::Maximize);
    assert_eq!(lp.offset(), 5.0);
    assert_eq!(lp.columns()[x].cost, 3.0);
    assert_eq!(lp.columns()[y].cost, -0.5);
    assert_eq!(lp.columns()[x].integrality, Integrality::Integer);
    assert_eq!(lp.columns()[x].bounds, bounds!(0.0, 9.0));
    assert_eq!(lp.columns()[y].bounds, Bounds::FREE);
    assert_eq!(lp.row_bounds()[r], bounds!(f64::NEG_INFINITY, 2.0));
    assert_eq!(lp.row(r).entries().collect::<Vec<_>>(), [(x, 1.0), (y, 1.0)]);
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
    m.add_named_constraint("range", (x + y).between(-1.0, 4.0));
    m.add_named_constraint("fix", (x - y).equals(0.0));
    m.minimize(x - 2.0 * y + 3.0);
    let text = m.build().unwrap().to_lp_string();
    let expected = "Minimize\n obj: 1 x_1_ - 2 _1y + 3\nSubject To\n range_lo: 1 x_1_ + 1 _1y >= -1\n \
                    range_hi: 1 x_1_ + 1 _1y <= 4\n fix: 1 x_1_ - 1 _1y = 0\nBounds\n -2 <= x_1_ <= 5\n _1y free\n\
                    General\n x_1_\nEnd\n";
    assert_eq!(text, expected);
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
        let lp = m.build().unwrap();
        let values: TiVec<Col, f64> = point.iter().map(|&v| f64::from(v)).collect();
        let direct = expr.value(&values);
        let activity: f64 = lp.row(row).entries().map(|(c, a)| a * values[c]).sum();
        let held = (lower..=upper).contains(&direct);
        prop_assert_eq!(lp.row_bounds()[row].contains(activity), held);
        prop_assert_eq!(lp.max_violation(&values) == 0.0, held);
        prop_assert!(lp.row(row).entries().all(|(_, a)| a != 0.0));
        prop_assert_eq!(lp.rows().collect::<Vec<_>>(), [Row::new(0)]);
    }
}

#[test]
fn test_row_views_read_bounds_name_and_entries_both_ways() {
    let mut m = Model::new();
    let x = m.add_vars(3, |_| Var::continuous());
    let r = m.add_named_constraint("mix", (2 * x[2] + x[0] - x[1]).leq(4.0));
    let lp = m.build().unwrap();
    let row = lp.row(r);
    assert_eq!(row.bounds(), bounds!(f64::NEG_INFINITY, 4.0));
    assert_eq!(row.name(), Some("mix"));
    assert_eq!((row.len(), row.is_empty()), (3, false));
    let values: TiVec<Col, f64> = vec![1.0, 2.0, 3.0].into();
    assert_eq!(row.activity(&values), 5.0);

    let mut entries = lp.row(r).entries();
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
fn test_lp_cols_and_rows_run_both_ways_with_exact_lengths() {
    let mut m = Model::new();
    let x = m.add_vars(3, |_| Var::binary());
    let once = m.add_constraint(LinExpr::sum(x.iter().copied()).leq(1.0));
    let lp = m.build().unwrap();
    let mut cols = lp.cols();
    assert_eq!(cols.len(), 3);
    assert_eq!(cols.next_back(), Some(x[2]));
    assert_eq!(cols.clone().collect::<Vec<_>>(), x[..2]);
    cols.by_ref().for_each(drop);
    assert_eq!((cols.next(), cols.next_back()), (None, None));
    assert_eq!(lp.rows().rev().collect::<Vec<_>>(), [once]);
    assert_eq!(lp.rows().len(), 1);
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
    /// The model's checks are the LP's: a build fails with a model error naming the constraint exactly when a
    /// column's coefficients add up past finite, and a model that builds is a plan the LP accepts as it is.
    #[test]
    fn test_models_never_build_an_lp_the_lp_refuses(expr in huge_expression(2)) {
        let mut m = Model::new();
        m.add_vars(2, |_| Var::free());
        m.add_constraint(expr.clone().leq(1.0));
        let overflows = (0..2).any(|i| {
            let sum: f64 = expr.terms.iter().filter(|t| t.col.index() == i).map(|t| t.coefficient).sum();
            !sum.is_finite()
        });
        match m.build() {
            Ok(lp) => {
                prop_assert!(!overflows);
                prop_assert_eq!(LpPlan::from(lp.clone()).build(), Ok(lp));
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
        let lp = m.build().unwrap();
        let forwards: Vec<_> = lp.row(row).entries().collect();
        let mut backwards: Vec<_> = lp.row(row).entries().rev().collect();
        backwards.reverse();
        prop_assert_eq!(&forwards, &backwards);
        prop_assert_eq!(lp.row(row).entries().len(), forwards.len());
        prop_assert_eq!(lp.row(row).len(), forwards.len());
        prop_assert!(forwards.windows(2).all(|w| w[0].0 < w[1].0));
    }
}
