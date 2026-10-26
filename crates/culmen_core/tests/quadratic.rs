// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quadratic objectives: the operators, the model and plan checks, the convexity check, model classes, and the
//! typed refusal of a class a backend does not solve.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::fmt;
use core::time::Duration;
use std::sync::atomic::{AtomicUsize, Ordering};

use culmen_core::{
    Backend, Bounds, Callback, Col, Events, LinExpr, Linear, LoadError, Lp, LpColumn, LpError, LpPlan, Model,
    ModelClass, ModelClasses, ModelError, ModelVar, Outcome, QuadExpr, QuadTerm, Quadratic, Script, Sense, Settings,
    Solved, Solver, Term, TiSlice, TiVec, Unsupported, Var,
};
use proptest::prelude::*;

/// Two free variables `x` and `y`, the first two columns.
fn xy() -> (Model, Col, Col) {
    let mut m = Model::new();
    let x = m.add_var(Var::free().name("x"));
    let y = m.add_var(Var::free().name("y"));
    (m, x, y)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn test_products_of_columns_and_terms_are_quadratic_terms() {
    let (_, x, y) = xy();
    let term = |first, second, coefficient| QuadTerm {
        first,
        second,
        coefficient,
    };
    assert_eq!(x * y, term(x, y, 1.0));
    assert_eq!(x * x, term(x, x, 1.0));
    assert_eq!(x * (3.0 * y), term(x, y, 3.0));
    assert_eq!((3.0 * x) * y, term(x, y, 3.0));
    assert_eq!(2.0 * x * y, term(x, y, 2.0));
    assert_eq!((2 * x) * (y * 0.5), term(x, y, 1.0));
    assert_eq!(x * y * 4.0, term(x, y, 4.0));
    assert_eq!(4.0 * (x * y), term(x, y, 4.0));
    assert_eq!(x * y * 3, term(x, y, 3.0));
    assert_eq!(3 * (x * y), term(x, y, 3.0));
    assert_eq!(x * y / 4.0, term(x, y, 0.25));
    assert_eq!(-(x * y), term(x, y, -1.0));
}

#[test]
fn test_a_product_of_linear_expressions_expands() {
    let (_, x, y) = xy();
    // (x + 1)(2y - 3) = 2xy - 3x + 2y - 3.
    let e = (x + 1) * (2 * y - 3);
    assert_eq!(
        e.terms,
        [QuadTerm {
            first: x,
            second: y,
            coefficient: 2.0
        }]
    );
    let linear: Vec<(Col, f64)> = e.linear.terms.iter().map(|t| (t.col, t.coefficient)).collect();
    assert_eq!(linear, [(x, -3.0), (y, 2.0)]);
    assert_eq!(e.linear.constant, -3.0);
    // A factor without a constant adds no zero terms.
    let e = x * (y + 2);
    assert_eq!((e.terms.len(), e.linear.terms.len(), e.linear.constant), (1, 1, 0.0));
}

#[test]
fn test_quadratic_sums_build_one_expression_in_order() {
    let (_, x, y) = xy();
    let e = x * x + 2.0 * x * y + y * y - 4.0 * x + 3.0;
    assert_eq!(e.terms.len(), 3);
    assert_eq!(
        e.linear.terms,
        [Term {
            col: x,
            coefficient: -4.0
        }]
    );
    assert_eq!(e.linear.constant, 3.0);
    let mut acc = QuadExpr::new();
    acc += x * y;
    acc += &e;
    acc -= x;
    acc *= 2.0;
    acc.add_term(y, y, 1.0);
    assert_eq!(acc.terms.len(), 5);
    assert_eq!(acc.linear.terms.len(), 2);
    assert_eq!(acc.linear.constant, 6.0);
    let squares: QuadExpr = [x, y].into_iter().map(|v| v * v).sum();
    assert_eq!(squares, QuadExpr::sum([x * x, y * y]));
    assert_eq!([x * x, y * y].into_iter().collect::<QuadExpr>(), squares);
    let mut extended = QuadExpr::from(x * x);
    extended.extend([y * y]);
    assert_eq!(extended, squares);
    let mixed: QuadExpr = [QuadExpr::from(x * y), QuadExpr::from(x)].into_iter().sum();
    assert_eq!((mixed.terms.len(), mixed.linear.terms.len()), (1, 1));
    assert_eq!(QuadExpr::from(2.5).linear.constant, 2.5);
    assert_eq!(QuadExpr::from(3.0 * x).linear.terms.len(), 1);
    assert_eq!(QuadExpr::from(LinExpr::from(x)), x.into_quad_expr());
    assert_eq!(QuadExpr::default(), QuadExpr::new());
}

/// Every operator combination, as `(name, expression, its value at (u, v) by plain arithmetic)`.
#[expect(clippy::too_many_arguments, reason = "the free values of one table")]
fn cases(x: Col, y: Col, a: f64, b: f64, k: i32, u: f64, v: f64, d: f64) -> Vec<(&'static str, QuadExpr, f64)> {
    let kf = f64::from(k);
    let q = x * y;
    let e = x * y + x;
    let ev = u * v + u;
    let mut assigned = e.clone();
    assigned *= a;
    let mut grown = e.clone();
    grown += y * y;
    grown -= a * x;
    vec![
        ("col * col", (x * y).into(), u * v),
        ("col * term", (x * (a * y)).into(), a * u * v),
        ("term * col", ((a * x) * y).into(), a * u * v),
        ("term * term", ((a * x) * (b * y)).into(), a * b * u * v),
        ("lin * lin", (x + a) * (y - b), (u + a) * (v - b)),
        ("col * lin", x * (y + a), u * (v + a)),
        ("lin * col", (x + a) * y, (u + a) * v),
        ("term * lin", (a * x) * (y + b), a * u * (v + b)),
        ("lin * term", (x + a) * (b * y), (u + a) * b * v),
        ("quad term * f64", (q * a).into(), a * u * v),
        ("f64 * quad term", (a * q).into(), a * u * v),
        ("quad term * i32", (q * k).into(), kf * u * v),
        ("i32 * quad term", (k * q).into(), kf * u * v),
        ("quad term / f64", (q / d).into(), u * v / d),
        ("-quad term", (-q).into(), -u * v),
        ("quad expr * f64", e.clone() * a, a * ev),
        ("f64 * quad expr", a * e.clone(), a * ev),
        ("quad expr * i32", e.clone() * k, kf * ev),
        ("i32 * quad expr", k * e.clone(), kf * ev),
        ("quad expr / f64", e.clone() / d, ev / d),
        ("-quad expr", -e.clone(), -ev),
        ("quad expr *= f64", assigned, a * ev),
        ("quad expr += and -=", grown, ev + v * v - a * u),
        ("quad term + col", q + x, u * v + u),
        ("quad term - term", q - a * y, u * v - a * v),
        ("quad term + quad term", q + x * x, u * v + u * u),
        ("quad term - lin", q - (x + b), u * v - u - b),
        ("quad term + f64", q + a, u * v + a),
        ("quad term - i32", q - k, u * v - kf),
        ("quad term + quad expr", q + e.clone(), u * v + ev),
        ("quad term - &quad expr", q - &e, u * v - ev),
        ("quad expr + col", e.clone() + y, ev + v),
        ("quad expr - quad term", e.clone() - q, ev - u * v),
        ("quad expr + quad expr", e.clone() + e.clone(), 2.0 * ev),
        ("quad expr - &quad expr", e.clone() - &e, 0.0),
        ("quad expr + f64", e.clone() + a, ev + a),
        ("col + quad term", x + q, u + u * v),
        ("col - quad expr", x - e.clone(), u - ev),
        ("term + quad term", a * x + q, a * u + u * v),
        ("term - quad expr", a * x - e.clone(), a * u - ev),
        ("lin + quad term", (x + y) + q, u + v + u * v),
        ("lin - quad expr", (x + y) - e.clone(), u + v - ev),
        ("f64 + quad term", a + q, a + u * v),
        ("f64 - quad expr", a - e.clone(), a - ev),
        ("i32 - quad term", k - q, kf - u * v),
        ("i32 + quad expr", k + e.clone(), kf + ev),
        ("col + quad expr", x + e.clone(), u + ev),
        ("term + quad expr", a * x + e.clone(), a * u + ev),
        ("lin - quad term", (x - y) - q, u - v - u * v),
        ("f64 - quad term", a - q, a - u * v),
        ("i32 + quad term", k + q, kf + u * v),
        ("i32 - quad expr", k - e.clone(), kf - ev),
        ("f64 + quad expr", a + e.clone(), a + ev),
        ("lin + quad expr", (x + y) + e.clone(), u + v + ev),
        ("term - quad term", a * x - q, a * u - u * v),
        ("col - quad term", x - q, u - u * v),
    ]
}

proptest! {
    /// Every operator builds the expression whose value is what plain arithmetic computes from the same numbers.
    #[test]
    fn test_every_operator_agrees_with_arithmetic(
        a in -4.0..4.0f64,
        b in -4.0..4.0f64,
        k in -3i32..=3,
        u in -5.0..5.0f64,
        v in -5.0..5.0f64,
        d in prop_oneof![-4.0..-0.5f64, 0.5..4.0f64],
    ) {
        let (_, x, y) = xy();
        let point: TiVec<Col, f64> = vec![u, v].into();
        for (name, expr, want) in cases(x, y, a, b, k, u, v, d) {
            let got = expr.value(&point);
            prop_assert!(close(got, want), "{name}: {got} vs {want}");
        }
    }

    /// A sum of squared affine forms plus a linear part builds an LP whose objective value is the expression's at
    /// every point: the merge into canonical entries keeps the value.
    #[test]
    fn test_the_lp_values_the_objective_as_the_expression_does(
        forms in prop::collection::vec((prop::collection::vec(-3i32..=3, 3), -2i32..=2), 1..4),
        costs in prop::collection::vec(-3i32..=3, 3),
        point in prop::collection::vec(-5.0..5.0f64, 3),
        maximize in any::<bool>(),
    ) {
        let mut m = Model::new();
        let xs = m.add_vars(3, |_| Var::free());
        let affine = |(coefficients, constant): &(Vec<i32>, i32)| {
            LinExpr::sum(xs.iter().zip(coefficients).map(|(&x, &c)| f64::from(c) * x)) + *constant
        };
        let squares = QuadExpr::sum(forms.iter().map(|form| affine(form) * affine(form)));
        let linear = LinExpr::sum(xs.iter().zip(&costs).map(|(&x, &c)| f64::from(c) * x));
        let objective = if maximize { -(squares + linear) } else { squares + linear };
        m.set_objective(if maximize { Sense::Maximize } else { Sense::Minimize }, objective.clone());
        let lp = m.build().unwrap();
        let point: TiVec<Col, f64> = point.into();
        prop_assert!(close(lp.objective_value(&point), objective.value(&point)));
        let entries: Vec<(Col, Col, f64)> = lp.quadratic().entries().collect();
        prop_assert!(entries.iter().all(|&(i, j, q)| i <= j && q != 0.0));
        prop_assert!(entries.windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)), "sorted, each pair once");
    }
}

#[test]
fn test_the_model_merges_pairs_in_either_order_and_drops_cancelled_ones() {
    let (mut m, x, y) = xy();
    m.minimize(x * y + y * x + 3.0 * x * x - x * x + 2.0 * y * y + y * x - x * y - 2.0 * x * y);
    let lp = m.build().unwrap();
    assert_eq!(lp.quadratic().entries().collect::<Vec<_>>(), [(x, x, 2.0), (y, y, 2.0)]);
    assert_eq!(lp.quadratic().len(), 2);
    assert!(!lp.quadratic().is_empty());
    // The Hessian of x² + y² as HiGHS reads it, ½ xᵀHx: twice each coefficient on the diagonal.
    assert_eq!(lp.quadratic().hessian().collect::<Vec<_>>(), [(x, x, 4.0), (y, y, 4.0)]);
    assert_eq!(lp.quadratic().hessian().rev().count(), 2);
}

#[test]
fn test_the_hessian_doubles_the_diagonal_alone() {
    let (mut m, x, y) = xy();
    // x² + 3xy + 4y²: H = [[2, 3], [3, 8]], whose lower triangle holds 2, 3 and 8.
    m.minimize(x * x + 3.0 * x * y + 4.0 * y * y);
    let lp = m.build().unwrap();
    assert_eq!(
        lp.quadratic().hessian().collect::<Vec<_>>(),
        [(x, x, 2.0), (x, y, 3.0), (y, y, 8.0)]
    );
    // ½ xᵀHx at (1, 2): ½(2 + 2·3·2 + 8·4) = 23 = 1 + 6 + 16.
    let point: &TiSlice<Col, f64> = TiSlice::from_ref(&[1.0, 2.0]);
    assert_eq!(lp.quadratic().value(point), 23.0);
    assert_eq!(lp.objective_value(point), 23.0);
}

#[test]
fn test_the_model_names_a_defective_quadratic_term() {
    let (mut m, x, _) = xy();
    m.minimize(x * Col::new(9));
    assert_eq!(
        m.build(),
        Err(ModelError::UnknownVar {
            place: culmen_core::ExprPlace::Objective,
            col: Col::new(9)
        })
    );
    m.minimize(f64::NAN * x * x);
    let error = m.build().unwrap_err();
    assert!(
        matches!(&error, ModelError::ObjectiveQuadCoefficient { first, second, value }
            if first.col == x && second.col == x && value.is_nan()),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "Model: the objective coefficient NaN of `x` · `x` is not finite"
    );
    m.minimize(f64::MAX * x * x + f64::MAX * x * x);
    assert!(
        matches!(m.build(), Err(ModelError::ObjectiveQuadCoefficient { value, .. }) if value == f64::INFINITY),
        "the merged coefficient overflows"
    );
}

/// The variable `name` at `col`, as a model error names it.
fn var(col: Col, name: &str) -> ModelVar {
    ModelVar {
        col,
        name: Some(name.to_owned()),
    }
}

#[test]
fn test_a_non_convex_objective_is_a_typed_model_error() {
    let (mut m, x, y) = xy();
    let z = m.add_var(Var::free().name("z"));
    let w = m.add_var(Var::free().name("w"));
    let refused = |m: &Model| match m.build() {
        Err(ModelError::NonConvexObjective { sense, var }) => (sense, var),
        other => panic!("{other:?}"),
    };
    // x·y alone has the eigenvalues ±1.
    m.minimize(x * y);
    assert_eq!(refused(&m), (Sense::Minimize, var(x, "x")));
    // x² + 3xy + y² has the eigenvalue -1/2 in the direction x = -y.
    m.minimize(x * x + 3.0 * x * y + y * y);
    assert_eq!(refused(&m), (Sense::Minimize, var(x, "x")));
    // Maximizing a convex objective.
    m.maximize(x * x);
    assert_eq!(refused(&m), (Sense::Maximize, var(x, "x")));
    // The first block, {x, y}, is convex; the second, {z, w}, is not, and is named by its first variable.
    m.minimize((x + y) * (x + y) + z * w + w * w);
    assert_eq!(refused(&m), (Sense::Minimize, var(z, "z")));
    assert_eq!(
        m.build().unwrap_err().to_string(),
        "Model: the objective is not convex, as minimizing needs, in the block of variable `z`"
    );
    m.maximize(z * w);
    assert_eq!(
        m.build().unwrap_err().to_string(),
        "Model: the objective is not concave, as maximizing needs, in the block of variable `z`"
    );
}

#[test]
fn test_semidefinite_and_concave_objectives_build() {
    let (mut m, x, y) = xy();
    // (x + y)² = x² + 2xy + y² has the eigenvalues 0 and 4: convex, not strictly.
    m.minimize(x * x + 2.0 * x * y + y * y);
    assert_eq!(m.build().unwrap().class(), ModelClass::Qp);
    m.maximize(5.0 - (x - 3) * (x - 3) - (y + 1) * (y + 1));
    assert_eq!(m.build().unwrap().sense(), Sense::Maximize);
    // Many separate squares are many blocks of one column each.
    let mut m = Model::new();
    let xs = m.add_vars(3000, |_| Var::continuous());
    m.minimize(QuadExpr::sum(xs.iter().map(|&x| x * x)));
    assert_eq!(m.build().unwrap().quadratic().len(), 3000);
}

#[test]
fn test_the_plan_checks_its_quadratic_terms() {
    let mut plan = LpPlan::new();
    let x = plan.col(LpColumn::continuous(0.0, Bounds::FREE));
    let y = plan.col(LpColumn::continuous(0.0, Bounds::FREE));
    plan.quadratic = vec![(x, Col::new(5), 1.0)];
    assert_eq!(
        plan.clone().build(),
        Err(LpError::UnknownQuadColumn { col: Col::new(5) })
    );
    plan.quadratic = vec![(x, y, f64::INFINITY)];
    assert_eq!(
        plan.clone().build(),
        Err(LpError::NonFiniteQuadCoefficient {
            first: x,
            second: y,
            value: f64::INFINITY
        })
    );
    plan.quadratic = vec![(y, x, f64::MAX), (x, y, f64::MAX)];
    assert!(
        matches!(plan.clone().build(), Err(LpError::NonFiniteQuadCoefficient { value, .. }) if value == f64::INFINITY),
        "the pair's coefficients add up across both orders"
    );
    plan.quadratic = vec![(x, y, 1.0)];
    let error = plan.clone().build().unwrap_err();
    assert_eq!(
        error,
        LpError::NonConvexObjective {
            sense: Sense::Minimize,
            col: x
        }
    );
    assert_eq!(
        error.to_string(),
        "Lp: the quadratic objective is not convex in the block of Col(0)"
    );
    plan.sense = Sense::Maximize;
    plan.quadratic = vec![(y, y, 1.0)];
    assert_eq!(
        plan.clone().build().unwrap_err().to_string(),
        "Lp: the quadratic objective is not concave in the block of Col(1)"
    );
    plan.quadratic = vec![(y, y, -1.0), (x, x, 0.0)];
    let lp = plan.build().unwrap();
    assert_eq!(lp.quadratic().entries().collect::<Vec<_>>(), [(y, y, -1.0)]);
    assert_eq!(LpPlan::from(lp).quadratic, [(y, y, -1.0)]);
}

/// The plan of `½ xᵀ(sign · (AᵀA + ridge · I))x` over free columns.
fn gram_plan(a: &[Vec<i32>], ridge: f64, sign: f64, sense: Sense) -> LpPlan {
    let n = a[0].len();
    let h = |i: usize, j: usize| a.iter().map(|row| f64::from(row[i] * row[j])).sum::<f64>();
    let mut plan = LpPlan::new();
    plan.sense = sense;
    let cols: Vec<Col> = (0..n)
        .map(|_| plan.col(LpColumn::continuous(0.0, Bounds::FREE)))
        .collect();
    for i in 0..n {
        plan.quadratic.push((cols[i], cols[i], sign * (h(i, i) + ridge) / 2.0));
        for j in i + 1..n {
            plan.quadratic.push((cols[j], cols[i], sign * h(i, j)));
        }
    }
    plan
}

proptest! {
    /// A Gram matrix `AᵀA` is positive semidefinite, so its objective is accepted to minimize and its negation to
    /// maximize; with a ridge `εI` it is positive definite, so the opposite senses are refused.
    #[test]
    fn test_gram_objectives_pass_the_convexity_check_in_their_sense_alone(
        a in (1..=5usize).prop_flat_map(|n| prop::collection::vec(prop::collection::vec(-4i32..=4, n), 1..=n)),
    ) {
        prop_assume!(a.iter().flatten().any(|&v| v != 0));
        prop_assert!(gram_plan(&a, 0.0, 1.0, Sense::Minimize).build().is_ok());
        prop_assert!(gram_plan(&a, 0.0, -1.0, Sense::Maximize).build().is_ok());
        prop_assert!(gram_plan(&a, 0.0, 1.0, Sense::Maximize).build().is_err());
        prop_assert!(gram_plan(&a, 0.5, -1.0, Sense::Minimize).build().is_err());
        prop_assert!(gram_plan(&a, 0.5, 1.0, Sense::Maximize).build().is_err());
    }
}

#[test]
fn test_the_class_reads_integrality_and_the_objective_degree() {
    let (mut m, x, _) = xy();
    m.minimize(x);
    assert_eq!(m.build().unwrap().class(), ModelClass::Lp);
    m.minimize(x * x + x);
    assert_eq!(m.build().unwrap().class(), ModelClass::Qp);
    // An integer column makes the class mixed-integer, whether the objective names it or not.
    let n = m.add_var(Var::integer().upper(3));
    assert_eq!(m.build().unwrap().class(), ModelClass::Miqp);
    m.minimize(x + n);
    assert_eq!(m.build().unwrap().class(), ModelClass::Milp);
    for class in ModelClasses::ALL {
        let degree = matches!(class, ModelClass::Qp | ModelClass::Miqp);
        let integer = matches!(class, ModelClass::Milp | ModelClass::Miqp);
        assert_eq!((class.is_quadratic(), class.is_integer()), (degree, integer), "{class}");
    }
}

#[test]
fn test_class_sets_combine_iterate_and_print_by_name() {
    let linear = ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]);
    let qp = ModelClasses::from(ModelClass::Qp);
    let three = linear | qp;
    assert!(three.contains(ModelClass::Qp) && !three.contains(ModelClass::Miqp));
    assert_eq!(three & linear, linear);
    assert_eq!(three.intersection(qp), qp);
    assert_eq!(linear.union(qp), three);
    let mut grown = linear;
    grown |= qp;
    assert_eq!(grown, three);
    grown &= qp;
    assert_eq!(grown, qp);
    let mut extended = ModelClasses::NONE;
    extended.extend([ModelClass::Miqp, ModelClass::Lp]);
    assert_eq!(extended, ModelClasses::of(&[ModelClass::Lp, ModelClass::Miqp]));
    assert_eq!((three.len(), three.iter().len(), three.is_empty()), (3, 3, false));
    assert_eq!(three.iter().take(2).count(), 2, "the iterator's adaptors are std's");
    assert!(ModelClasses::NONE.is_empty() && ModelClasses::default().is_empty());
    assert_eq!(
        three.iter().collect::<Vec<_>>(),
        [ModelClass::Lp, ModelClass::Milp, ModelClass::Qp]
    );
    assert_eq!(
        three.iter().rev().collect::<Vec<_>>(),
        [ModelClass::Qp, ModelClass::Milp, ModelClass::Lp]
    );
    assert_eq!(three.into_iter().collect::<ModelClasses>(), three);
    assert_eq!(ModelClasses::ALL.len(), 4);
    assert_eq!(
        format!("{:?}", ModelClasses::ALL),
        "ModelClasses(LP | MILP | QP | MIQP)"
    );
    assert_eq!(format!("{:?}", ModelClasses::NONE), "ModelClasses()");
    assert_eq!(ModelClasses::NONE.to_string(), "{}");
    assert_eq!(ModelClasses::ALL.to_string(), "{LP, MILP, QP, MIQP}");
    assert!(ModelClass::Lp < ModelClass::Miqp);
    assert_eq!(Script::CLASSES, ModelClasses::ALL);
}

static LOADS: AtomicUsize = AtomicUsize::new(0);

/// A backend that solves linear classes alone and counts its loads; it refuses an LP without columns.
#[derive(Debug)]
struct LinearOnly;

/// Why [`LinearOnly`] refused.
#[derive(Debug, PartialEq)]
struct NoColumns;

impl fmt::Display for NoColumns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("LinearOnly: the LP has no columns")
    }
}

impl core::error::Error for NoColumns {}

impl Backend for LinearOnly {
    type Error = NoColumns;
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
    const CLASSES: ModelClasses = ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]);

    fn load(&mut self, lp: &Lp) -> Result<(), NoColumns> {
        LOADS.fetch_add(1, Ordering::SeqCst);
        if lp.num_cols() == 0 { Err(NoColumns) } else { Ok(()) }
    }

    fn solve(&mut self, _: &Settings, _: &mut dyn Callback<Self>) -> Result<Solved<()>, NoColumns> {
        Ok(Solved {
            outcome: Outcome::Infeasible,
            status: (),
            elapsed: Duration::ZERO,
        })
    }
}

#[test]
fn test_a_class_outside_the_backend_set_is_refused_before_load() {
    let (mut m, x, y) = xy();
    m.minimize(x + y);
    let before = LOADS.load(Ordering::SeqCst);
    assert!(Solver::new(LinearOnly, m.build().unwrap()).is_ok());
    assert_eq!(LOADS.load(Ordering::SeqCst), before + 1);

    m.minimize(x * x + y * y);
    let error = Solver::new(LinearOnly, m.build().unwrap()).unwrap_err();
    assert_eq!(
        error,
        LoadError::Unsupported {
            class: ModelClass::Qp,
            supported: LinearOnly::CLASSES
        }
    );
    assert_eq!(LOADS.load(Ordering::SeqCst), before + 1, "the backend never saw the QP");
    assert_eq!(error.to_string(), "Solver: the backend solves {LP, MILP}, not a QP");
    assert!(core::error::Error::source(&error).is_none());

    let error = Solver::new(LinearOnly, Model::new().build().unwrap()).unwrap_err();
    assert_eq!(error, LoadError::Backend(NoColumns));
    assert_eq!(
        error.to_string(),
        "Solver: the backend refused the LP: LinearOnly: the LP has no columns"
    );
    let source = core::error::Error::source(&error).unwrap();
    assert_eq!(source.to_string(), "LinearOnly: the LP has no columns");

    // The script replays any class.
    let mut solver = Solver::new(Script::optimal(2.0, vec![1.0, 1.0]), m.build().unwrap()).unwrap();
    assert_eq!(solver.lp().class(), ModelClass::Qp);
    assert!(solver.solve(culmen_core::Request::new()).is_ok());
}

#[test]
fn test_the_lp_file_writes_the_quadratic_part_halved_in_brackets() {
    let (mut m, x, y) = xy();
    m.minimize(x * x + 0.5 * x * y + y * y);
    assert!(
        m.build()
            .unwrap()
            .to_lp_string()
            .starts_with("Minimize\n obj: [ 2 x ^ 2 + 1 x * y + 2 y ^ 2 ] / 2\nSubject To\n"),
        "no linear term: the bracket opens the objective"
    );
    m.maximize(2.0 * x - x * x - 3.0 * y * y - 1.5);
    assert!(
        m.build()
            .unwrap()
            .to_lp_string()
            .starts_with("Maximize\n obj: 2 x + [ -2 x ^ 2 - 6 y ^ 2 ] / 2 - 1.5\nSubject To\n"),
    );
    m.minimize(QuadExpr::new() + 4.0);
    assert!(m.build().unwrap().to_lp_string().starts_with("Minimize\n obj: 0 + 4\n"));
}

#[test]
fn test_linear_items_stay_linear_in_sums_and_constraints() {
    let (_, x, y) = xy();
    // A linear sum stays a `LinExpr`, which constraints take.
    let load: LinExpr = [x, y].into_iter().sum();
    let c = load.leq(4);
    assert_eq!((c.upper, c.expr.constant), (0.0, -4.0));
    let summed: LinExpr = [2.0 * x, 3.0 * y].into_iter().sum();
    assert_eq!(summed.terms.len(), 2);
}
