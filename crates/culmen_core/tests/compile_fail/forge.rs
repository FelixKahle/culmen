// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A checked value cannot be built around its constructor: neither an empty interval nor a program whose numbers were
//! never checked. Every field is given, so both refusals are the privacy check's.

use culmen_core::{Bounds, Program};

fn empty() -> Bounds {
    Bounds { lower: 2.0, upper: 1.0 }
}

fn unchecked() -> Program {
    Program {
        sense: Default::default(),
        offset: f64::NAN,
        columns: Default::default(),
        row_bounds: Default::default(),
        row_names: Default::default(),
        row_starts: Default::default(),
        entry_cols: Default::default(),
        entry_values: Default::default(),
        quadratic: Default::default(),
        objectives: Default::default(),
    }
}

fn main() {
    let _ = (empty(), unchecked());
}
