// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The warehouse slotting of `warehouse`, written with the modelling macros.
//!
//! `cargo run -p culmen --release --example warehouse_macros --features macros -- 20 --log`.

#[path = "support/warehouse.rs"]
mod support;

use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let w = support::Warehouse::generate();
    let (products, locations, volume, capacity) = (&w.products, &w.locations, &w.volume, &w.capacity);
    let mut m = Model::new();
    variable!(m, x[p in products, l in locations if w.fits(p, l)], Bin);
    constraint!(m, once[p in products], sum!(x[p, l] for l in locations if w.fits(p, l)) == 1);
    constraint!(m, cap[l in locations], sum!(volume[p] * x[p, l] for p in products if w.fits(p, l)) <= capacity[l]);
    objective!(
        m,
        Min,
        sum!(w.walk(p, l) * x[p, l] for p in products, l in locations if w.fits(p, l))
    );
    support::solve(&w, m.build()?, &x)
}
