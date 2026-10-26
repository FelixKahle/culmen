// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Linear, mixed-integer and convex quadratic programming in Rust: a model written the way the mathematics reads,
//! solved on a switchable backend, with a live view of the search (best solution, bound, gap, solver log) and a
//! cooperative stop.
//!
//! This is the crate to depend on. It re-exports [`culmen_core`] at its root, the HiGHS backend as [`Highs`] (and
//! the whole backend as [`highs`]) with the default `highs` feature, and the modelling macros with the `macros`
//! feature. [`prelude`] holds what a model and a solve need.
//!
//! # A model, two ways
//!
//! Every model is written with plain methods and operators; the macros write the same calls more briefly and build
//! the same [`Program`]. A generalized assignment: put each product in one location, within each location's capacity,
//! walking least.
//!
//! ```
//! # #[cfg(feature = "highs")] {
//! use culmen::prelude::*;
//!
//! let (products, locations) = (0..4, 0..2);
//! let volume = [4.0, 3.0, 5.0, 2.0];
//! let capacity = [8.0, 7.0];
//! let walk = [[1.0, 3.0], [2.0, 1.0], [4.0, 2.0], [1.0, 1.0]];
//!
//! let mut m = Model::new();
//! let x = m.add_var_map(product(&products, &locations), |(p, l)| Var::binary().name(format!("x[{p},{l}]")));
//! m.add_constraint_map(products.clone(), |&p| {
//!     (format!("once[{p}]"), LinExpr::sum(locations.clone().map(|l| x[(p, l)])).equals(1))
//! });
//! m.add_constraint_map(locations.clone(), |&l| {
//!     (format!("cap[{l}]"), LinExpr::sum(products.clone().map(|p| volume[p] * x[(p, l)])).leq(capacity[l]))
//! });
//! m.minimize(x.iter().map(|(&(p, l), x)| walk[p][l] * x).sum::<LinExpr>());
//!
//! let mut solver = Solver::new(Highs::new(), m.build()?)?;
//! let solved = solver.solve(Request::new())?;
//! let Outcome::Optimal(best) = solved.outcome else { panic!("{:?}", solved.outcome) };
//! assert_eq!(best.objective, 6.0);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! With the `macros` feature:
//!
//! ```
//! # #[cfg(feature = "macros")] {
//! use culmen::prelude::*;
//!
//! let (products, locations) = (0..4, 0..2);
//! let volume = [4.0, 3.0, 5.0, 2.0];
//! let cap = [8.0, 7.0];
//! let walk = [[1.0, 3.0], [2.0, 1.0], [4.0, 2.0], [1.0, 1.0]];
//!
//! let mut m = Model::new();
//! variable!(m, x[p in products, l in locations], Bin);
//! constraint!(m, once[p in products], sum!(x[p, l] for l in locations) == 1);
//! constraint!(m, cap[l in locations], sum!(volume[p] * x[p, l] for p in products) <= cap[l]);
//! objective!(m, Min, sum!(walk[p][l] * x[p, l] for p in products, l in locations));
//! assert_eq!(m.constraint_name(cap[(1)]), Some("cap[1]"));
//! # }
//! ```
//!
//! # The macros and the methods they write
//!
//! | Macro form | Plain API |
//! |---|---|
//! | `variable!(m, x)` | `let x = m.add_var(Var::continuous().name("x"));` |
//! | `variable!(m, 0 <= y <= 9, Int)` | `let y = m.add_var(Var::integer().lower(0).upper(9).name("y"));` |
//! | `variable!(m, z == 2.5, Free)` | `let z = m.add_var(Var::free().bounds(2.5, 2.5).name("z"));` |
//! | `variable!(m, x[i in I, j in J if f(i, j)], Bin)` | `let x = m.add_var_map(keys, \|&(i, j)\| Var::binary().name(format!("x[{i},{j}]")));` with `keys` the filtered product |
//! | `constraint!(e <= f)` | `Linear::leq(e, f)` (also `geq`, `equals`) |
//! | `constraint!(lo <= e <= hi)` | `Linear::between(e, lo, hi)` |
//! | `constraint!(m, e == f)` | `m.add_constraint(Linear::equals(e, f))` |
//! | `constraint!(m, c, e >= f)` | `let c = m.add_named_constraint("c", Linear::geq(e, f));` |
//! | `constraint!(m, c[i in I], e <= f)` | `let c = m.add_constraint_map(I, \|&i\| (format!("c[{i}]"), Linear::leq(e, f)));` |
//! | `objective!(m, Min, e)` | `m.set_objective(Sense::Minimize, e)` |
//! | `sum!(e for i in I, j in J if g)` | `LinExpr` or, for a quadratic `e`, `QuadExpr`, summed in nested `for` loops |
//! | `x[i, j]` inside a macro | `x[(i.clone(), j.clone())]` |
//!
//! Each domain is evaluated where its loop starts and iterated as an [`IndexSet`], which yields members by value:
//! `p in products` binds `p: usize` over a `Vec<usize>`, a `&Vec<usize>` or a range alike, and leaves `products`
//! usable. Wrap any other cloneable iterator with [`set`]. An entry of a family is named
//! with each index's `Display`. A variable's unwritten bound keeps its kind's default: zero below for `Cont` and
//! `Int`, none for `Free`; a `Bin` takes no bounds.
//!
//! # Watching and stopping a solve
//!
//! ```
//! # #[cfg(feature = "highs")] {
//! use std::sync::Arc;
//! use std::sync::atomic::AtomicBool;
//! use std::time::Duration;
//!
//! use culmen::prelude::*;
//!
//! struct Print;
//!
//! impl Observer for Print {
//!     fn progress(&mut self, p: &Progress) {
//!         println!("{:?}: best {:?}, bound {:?}, gap {:?}", p.elapsed, p.objective, p.bound, p.gap());
//!     }
//!     fn improved(&mut self, i: &Improvement<'_>) {
//!         println!("new best {}", i.objective);
//!     }
//!     fn log(&mut self, line: &LogLine<'_>) {
//!         println!("highs: {}", line.text);
//!     }
//! }
//!
//! let mut m = Model::new();
//! let x = m.add_vars(3, |_| Var::binary());
//! m.add_constraint(LinExpr::sum([2.0 * x[0], 3.0 * x[1], 4.0 * x[2]]).leq(5.0));
//! m.maximize(LinExpr::sum([3.0 * x[0], 4.0 * x[1], 5.0 * x[2]]));
//!
//! let cancel = Arc::new(AtomicBool::new(false)); // set from another thread to stop
//! let mut print = Print;
//! let mut solver = Solver::new(Highs::new(), m.build()?)?;
//! let request = Request::new()
//!     .time_limit(Duration::from_secs(10))
//!     .stop(&cancel)
//!     .observer(&mut print)
//!     .progress_interval(Duration::from_secs(1));
//! let solved = solver.solve(request)?;
//! assert_eq!(solved.outcome.solution().map(|s| s.objective), Some(7.0));
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Features
//!
//! - **`highs`** (default): the HiGHS backend, built from HiGHS's bundled sources (needs cmake and libclang).
//! - **`macros`**: `variable!`, `constraint!`, `objective!` and `sum!`.
//! - **`serde`**: the plain data serializes and reads back.

pub use culmen_core::*;
#[cfg(feature = "highs")]
pub use culmen_highs as highs;
#[cfg(feature = "highs")]
pub use culmen_highs::Highs;
#[cfg(feature = "macros")]
pub use culmen_macros::{constraint, objective, sum, variable};

/// The README and the user guide under `docs/`, compiled as doctests so every snippet in it stays true. A page whose snippets
/// need a feature is included only with that feature.
#[cfg(doctest)]
mod docs {
    #[cfg(all(feature = "highs", feature = "macros"))]
    #[doc = include_str!("../../../README.md")]
    struct Readme;

    #[doc = include_str!("../../../docs/README.md")]
    struct Index;

    #[cfg(all(feature = "highs", feature = "macros"))]
    #[doc = include_str!("../../../docs/getting-started.md")]
    struct GettingStarted;

    #[doc = include_str!("../../../docs/modelling.md")]
    struct Modelling;

    #[cfg(feature = "macros")]
    #[doc = include_str!("../../../docs/macros.md")]
    struct Macros;

    #[doc = include_str!("../../../docs/program.md")]
    struct ProgramAsData;

    #[cfg(feature = "highs")]
    #[doc = include_str!("../../../docs/solving.md")]
    struct Solving;

    #[cfg(feature = "highs")]
    #[doc = include_str!("../../../docs/callbacks.md")]
    struct Callbacks;

    #[cfg(feature = "highs")]
    #[doc = include_str!("../../../docs/analysis.md")]
    struct Analysis;

    #[cfg(feature = "highs")]
    #[doc = include_str!("../../../docs/highs.md")]
    struct HighsBackend;

    #[cfg(feature = "highs")]
    #[doc = include_str!("../../../docs/testing.md")]
    struct Testing;

    #[doc = include_str!("../../../docs/design.md")]
    struct Design;
}

/// What a model and a solve need, in one import.
pub mod prelude {
    #[cfg(feature = "highs")]
    pub use culmen_highs::Highs;
    #[cfg(feature = "macros")]
    pub use culmen_macros::{constraint, objective, sum, variable};

    pub use culmen_core::{
        Backend, Bounds, Callback, Col, Constraint, ConstraintMap, Events, Improvement, IndexSet, LinExpr, Linear,
        LogLine, Model, Objective, Observer, Outcome, Program, Progress, QuadExpr, Quadratic, Request, Row, Sense,
        Settings, Solution, SolveError, Solved, Solver, Start, Stop, StopReason, Tolerance, Var, VarMap, absolute_gap,
        bounds, product, relative_gap, set, stop_fn, threads, weight,
    };
}
