// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Warehouse slotting as a generalized assignment, written with the plain API: 60 products, 30 locations, frozen
//! products only in cooled locations, solved live.
//!
//! `cargo run -p culmen --release --example warehouse -- 20 --log`: stop after 20 s (default 10) or on Ctrl-C,
//! printing HiGHS's log too. `warehouse_macros` writes the same model with the macros.

#[path = "support/warehouse.rs"]
mod support;

use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let w = support::Warehouse::generate();
    let mut m = Model::new();
    let pairs = product(&w.products, &w.locations).filter(|&(p, l)| w.fits(p, l));
    let x = m.add_var_map(pairs, |&(p, l)| Var::binary().name(format!("x[{p},{l}]")));
    m.add_constraint_map(w.products.clone(), |&p| {
        let placed = LinExpr::sum(w.locations.iter().filter_map(|&l| x.get(&(p, l))));
        (format!("once[{p}]"), placed.equals(1))
    });
    m.add_constraint_map(w.locations.clone(), |&l| {
        let load = LinExpr::sum(w.products.iter().filter_map(|&p| Some(w.volume[p] * x.get(&(p, l))?)));
        (format!("cap[{l}]"), load.leq(w.capacity[l]))
    });
    m.minimize(LinExpr::sum(x.iter().map(|(&(p, l), x)| w.walk(p, l) * x)));
    support::solve(&w, m.build()?, &x)
}
