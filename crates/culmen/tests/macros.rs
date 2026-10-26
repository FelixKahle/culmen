// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The modelling macros build exactly the LP the plain methods build.
#![cfg(feature = "macros")]
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use culmen::prelude::*;
use culmen::{Integrality, LpError, QuadExpr};

#[test]
fn test_assignment_written_both_ways_builds_the_same_lp() {
    let (products, locations) = (vec![0_usize, 1, 2, 3], 0..3_usize);
    let volume = [4.0, 3.0, 5.0, 2.0];
    let cap = [8.0, 7.0, 6.0];
    let walk = [[1.0, 3.0, 2.0], [2.0, 1.0, 4.0], [4.0, 2.0, 1.0], [1.0, 1.0, 3.0]];
    let fits = |p: usize, l: usize| p != 2 || l != 0;

    let mut plain = Model::new();
    let x_plain = plain.add_var_map(
        product(&products, &locations).filter(|&(p, l)| fits(p, l)),
        |&(p, l)| Var::binary().name(format!("x[{p},{l}]")),
    );
    plain.add_constraint_map(products.clone(), |&p| {
        let placed = LinExpr::sum(locations.clone().filter(|&l| fits(p, l)).map(|l| x_plain[(p, l)]));
        (format!("once[{p}]"), placed.equals(1))
    });
    plain.add_constraint_map(locations.clone(), |&l| {
        let load = LinExpr::sum(
            products
                .iter()
                .filter(|&&p| fits(p, l))
                .map(|&p| volume[p] * x_plain[(p, l)]),
        );
        (format!("cap[{l}]"), load.leq(cap[l]))
    });
    let walked = x_plain.iter().map(|(&(p, l), x)| walk[p][l] * x);
    plain.set_objective(Sense::Minimize, LinExpr::sum(walked));

    let mut m = Model::new();
    variable!(m, x[p in products, l in locations if fits(p, l)], Bin);
    constraint!(m, once[p in products], sum!(x[p, l] for l in locations if fits(p, l)) == 1);
    constraint!(m, cap[l in locations], sum!(volume[p] * x[p, l] for p in products if fits(p, l)) <= cap[l]);
    objective!(
        m,
        Min,
        sum!(walk[p][l] * x[p, l] for p in products, l in locations if fits(p, l))
    );

    assert_eq!(m.build().unwrap(), plain.build().unwrap());
    assert_eq!(x.len(), 11);
    assert_eq!(m.constraint_name(once[3]), Some("once[3]"));
    assert_eq!(m.constraint_name(cap[2]), Some("cap[2]"));
    assert_eq!(products.len(), 4, "the domain is still usable after the macros read it");
}

#[test]
fn test_every_variable_form_writes_its_var() {
    let ub = [3.0, 4.0];
    let mut m = Model::new();
    variable!(m, a);
    variable!(m, b >= -2.0);
    variable!(m, c <= 5, Int);
    variable!(m, d == 2.5, Free);
    variable!(m, -1 <= e <= 1, Free);
    variable!(m, 9 >= f >= 1, Integer);
    variable!(m, g, Binary);
    variable!(m, 0 <= h[i in 0..2] <= ub[i], Cont);
    variable!(m, 7.0 <= k);

    let mut p = Model::new();
    p.add_var(Var::continuous().name("a"));
    p.add_var(Var::continuous().lower(-2.0).name("b"));
    p.add_var(Var::integer().upper(5).name("c"));
    p.add_var(Var::free().bounds(2.5, 2.5).name("d"));
    p.add_var(Var::free().lower(-1).upper(1).name("e"));
    p.add_var(Var::integer().lower(1).upper(9).name("f"));
    p.add_var(Var::binary().name("g"));
    p.add_var_map(0..2, |&i| {
        Var::continuous().lower(0).upper(ub[i]).name(format!("h[{i}]"))
    });
    p.add_var(Var::continuous().lower(7.0).name("k"));

    let (lp, expected) = (m.build().unwrap(), p.build().unwrap());
    assert_eq!(lp, expected);
    assert_eq!(lp.columns()[c].integrality, Integrality::Integer);
    assert_eq!(lp.columns()[h[1]].bounds, bounds!(0.0, 4.0));
    assert_eq!(
        (lp.columns()[d].bounds.lower(), lp.columns()[a].bounds.lower()),
        (2.5, 0.0)
    );
    let _ = (b, e, f, g, k);
}

#[test]
fn test_every_constraint_form_writes_its_relation() {
    let mut m = Model::new();
    variable!(m, x);
    variable!(m, y);
    let value = constraint!(x + y <= 4);
    assert_eq!(value, (x + y).leq(4));
    assert_eq!(constraint!(x >= y), x.geq(y));
    assert_eq!(constraint!(2 * x == y - 1), (2 * x).equals(y - 1));
    assert_eq!(constraint!(-1 <= x - y <= 1), (x - y).between(-1, 1));
    assert_eq!(constraint!(1 >= x - y >= -1), (x - y).between(-1, 1));
    let unnamed = constraint!(m, x + 2.0 * y >= 1.0);
    constraint!(m, budget, x + y <= 10.0);
    constraint!(m, pair[i in 0..3, j in 0..3 if i < j], (i as f64) * x + (j as f64) * y <= 6.0);
    assert_eq!(m.constraint_name(unnamed), None);
    assert_eq!(m.constraint_name(budget), Some("budget"));
    assert_eq!(pair.len(), 3);
    assert_eq!(m.constraint_name(pair[(0, 2)]), Some("pair[0,2]"));
}

#[test]
fn test_sum_runs_its_loop_nest_with_patterns_and_filters() {
    let arcs = vec![(0_usize, 1_usize), (1, 2), (0, 2)];
    let cost = [[0.0, 1.0, 4.0], [0.0, 0.0, 2.0], [0.0; 3]];
    let mut m = Model::new();
    variable!(m, flow[(a, b) in arcs], Cont);
    let total = sum!(cost[a][b] * flow[a, b] for (a, b) in arcs);
    let expected = LinExpr::sum(arcs.iter().map(|&(a, b)| cost[a][b] * flow[(a, b)]));
    assert_eq!(total, expected);
    let from_zero = sum!(flow[a, b] for (a, b) in arcs if a == 0);
    assert_eq!(from_zero, flow[(0, 1)] + flow[(0, 2)]);
    let triangle = sum!(1 for i in 0..4, _j in i + 1..4);
    assert_eq!(triangle.constant, 6.0);
    assert_eq!(m.var_name(flow[(1, 2)]), Some("flow[1,2]"));
}

#[test]
fn test_a_quadratic_objective_written_both_ways_builds_the_same_lp() {
    // Least squares on a budget: track targets, spending at most 4 in all.
    let (items, target) = (0..3_usize, [1.0, 2.0, 3.0]);

    let mut plain = Model::new();
    let x_plain = plain.add_var_map(items.clone(), |&i| Var::continuous().name(format!("x[{i}]")));
    plain.add_named_constraint("budget", LinExpr::sum(items.clone().map(|i| x_plain[i])).leq(4));
    let squares = QuadExpr::sum(items.clone().map(|i| x_plain[i] * x_plain[i]));
    let linear = LinExpr::sum(items.clone().map(|i| -2.0 * target[i] * x_plain[i]));
    plain.minimize(squares + linear);

    let mut m = Model::new();
    variable!(m, x[i in items]);
    constraint!(m, budget, sum!(x[i] for i in items) <= 4);
    objective!(
        m,
        Min,
        sum!(x[i] * x[i] for i in items) + sum!(-2.0 * target[i] * x[i] for i in items)
    );

    assert_eq!(m.build().unwrap(), plain.build().unwrap());
    let energy: QuadExpr = sum!(x[i] * x[i] for i in items);
    let load: LinExpr = sum!(x[i] for i in items);
    assert_eq!((energy.terms.len(), load.terms.len()), (3, 3));
    let _ = budget;
}

#[test]
fn test_string_keys_are_cloned_into_lookups() {
    let products = vec!["cork".to_owned(), "glass".to_owned()];
    let sites = ["north", "south"];
    let mut m = Model::new();
    variable!(m, x[p in products, s in sites], Bin);
    constraint!(m, once[p in products], sum!(x[p, s] for s in sites) == 1);
    objective!(m, Max, sum!(x[p, s] for p in products, s in sites));
    assert_eq!(m.var_name(x[("glass".to_owned(), "south")]), Some("x[glass,south]"));
    assert_eq!(m.constraint_name(once["cork".to_owned()]), Some("once[cork]"));
}

#[test]
fn test_model_errors_name_macro_declared_entries() {
    let mut m = Model::new();
    variable!(m, 3 <= z[i in 0..2] <= 1);
    let error = m.build().unwrap_err();
    assert!(error.to_string().contains("`z[0]`"), "{error}");
    let _ = (z, LpError::NonFiniteOffset { value: 0.0 });
}
