// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Linear, mixed-integer and convex quadratic programming in Rust: a model written the way the mathematics reads,
//! solved on a switchable backend, with a live view of the search (best solution, bound, gap, solver log) and a
//! cooperative stop.
//!
//! This is the crate to depend on. It re-exports [`culmen_core`] at its root, the HiGHS backend as the module [`highs`]
//! with the default `highs` feature, the SCIP backend as the module `scip` with the `scip` feature, and the modelling
//! macros with the `macros` feature. [`prelude`] holds what a model and a solve need, `Highs` and `Scip` among them.
//!
//! # Where an item lives
//!
//! Each item has one address, chosen by who names it. The root holds what code that models and solves names: the
//! model and the program, [`Solver`] and [`Run`], settings, outcomes, the callback and handler traits, the context
//! traits a handler calls, the capability traits a generic bound names ([`LazyConstraints`], [`DualAnalysis`], …),
//! the analyses' data and the errors. [`script`] holds the scripted backend for tests, and [`backend`] what only code
//! implementing or driving a backend names. Each solver's own types are in its module, [`highs`] or `scip`.
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
//! let solved = solver.solve()?;
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
//! assert_eq!(m.constraint_name(cap[1]), Some("cap[1]"));
//! # }
//! ```
//!
//! # The macros and the methods they write
//!
//! | Macro form | Plain API |
//! |---|---|
//! | `variable!(m, x)` | `let x = m.add_var(Var::continuous().name("x"));` |
//! | `variable!(m, 0 <= y <= 9, Int)` | `let y = m.add_var(Var::integer().lower(0).upper(9).name("y"));` |
//! | `variable!(m, z == 2.5, Free)` | `let z = m.add_var(Var::free().lower(2.5).upper(2.5).name("z"));` |
//! | `variable!(m, x[i in I, j in J if f(i, j)], Bin)` | `let x = m.add_var_map(keys, \|&(i, j)\| Var::binary().name(format!("x[{i},{j}]")));` with `keys` the filtered product |
//! | `constraint!(e <= f)` | `Linear::leq(e, f)` (also `geq`, `equals`) |
//! | `constraint!(lo <= e <= hi)` | `Linear::between(e, lo, hi)` |
//! | `constraint!(m, e == f)` | `m.add_constraint(Linear::equals(e, f))` |
//! | `constraint!(m, c, e >= f)` | `let c = m.add_constraint(("c", Linear::geq(e, f)));` |
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
//! let solved = solver
//!     .run()
//!     .time_limit(Duration::from_secs(10))
//!     .stop(&cancel)
//!     .observer(&mut print)
//!     .progress_interval(Duration::from_secs(1))
//!     .solve()?;
//! assert_eq!(solved.outcome.objective(), Some(7.0));
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Features
//!
//! - **`highs`** (default): the HiGHS backend, built from HiGHS's bundled sources (needs cmake and libclang).
//! - **`scip`**: the SCIP backend, for MIQPs, lazy constraints, user cuts and branch-and-price. Not a default: its
//!   build downloads the SCIP Optimization Suite sources from GitHub and compiles SCIP and SoPlex with cmake (needs
//!   cmake, a C and C++ compiler, libclang and network access), which takes several times as long as HiGHS's build,
//!   and most models are served by HiGHS alone.
//! - **`macros`**: `variable!`, `constraint!`, `objective!` and `sum!`.
//! - **`serde`**: the plain data serializes and reads back, and so do the options and statuses of each backend
//!   enabled.

pub use culmen_core::*;
#[cfg(feature = "highs")]
pub use culmen_highs as highs;
#[cfg(feature = "macros")]
pub use culmen_macros::{constraint, objective, sum, variable};
#[cfg(feature = "scip")]
pub use culmen_scip as scip;

/// The README and the user guide under `docs/`, compiled as doctests so every snippet in it stays true. A page whose
/// snippets need a feature is included only with that feature.
#[cfg(doctest)]
mod docs {
    #[cfg(all(feature = "highs", feature = "scip", feature = "macros"))]
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

    #[cfg(feature = "scip")]
    #[doc = include_str!("../../../docs/scip.md")]
    struct ScipBackend;

    #[cfg(feature = "highs")]
    #[doc = include_str!("../../../docs/testing.md")]
    struct Testing;

    #[doc = include_str!("../../../docs/design.md")]
    struct Design;

    #[doc = include_str!("../../../docs/backends.md")]
    struct Backends;
}

/// What nearly every program that models and solves names, in one import: `use culmen::prelude::*;`.
///
/// The prelude holds the modelling layer ([`Model`], [`Var`], [`LinExpr`], the literal macros, the families), the
/// program's handles and the plain data a model or a solve passes around ([`Col`], [`Row`], [`Bounds`],
/// [`Integrality`], [`NewCol`], the [`TiVec`] and [`TiSlice`] a solution's values come in), the solve ([`Solver`],
/// [`Settings`], [`Outcome`], [`Solved`], [`Start`], the [`Stop`] and the [`Observer`]), and the backends enabled.
/// Writing a callback or a search handler needs no further import: the prelude holds [`Callback`] and [`Events`], each
/// handler trait with its capability trait, which a generic handler names in its bound, and every context trait, which
/// must be in scope for a handler's calls on its context to resolve, with the [`PriceMode`] a pricer reads.
///
/// What only some programs name is not in it: the errors beyond [`SolveError`], the analyses and their capability
/// traits (a `Solver`'s analysis methods resolve without them), the program's own building blocks such as
/// [`ProgramPlan`], and the modules [`script`] and [`backend`].
pub mod prelude {
    #[cfg(feature = "highs")]
    pub use culmen_highs::Highs;
    #[cfg(feature = "macros")]
    pub use culmen_macros::{constraint, objective, sum, variable};
    #[cfg(feature = "scip")]
    pub use culmen_scip::Scip;

    pub use culmen_core::{
        Backend, Bounds, Col, Constraint, ConstraintMap, Improvement, IndexSet, Integrality, LinExpr, Linear, LogLine,
        Model, NewCol, Objective, Observer, Outcome, Program, Progress, QuadExpr, Quadratic, Row, Sense, Settings,
        Solution, SolveError, Solved, Solver, Start, Stop, StopReason, TiSlice, TiVec, Tolerance, Var, VarMap,
        absolute_gap, bounds, feasibility_tolerance, product, relative_gap, set, solutions, stop_fn, threads, weight,
    };
    // A callback names the events it asks for.
    pub use culmen_core::{Callback, Event, Events};
    // A generic callback or handler calls its context through these traits, which must be in scope for the calls to
    // resolve; a pricer reads the mode of its round.
    pub use culmen_core::{
        Context, CutContext, HeuristicContext, LazyContext, LogContext, PollContext, PriceContext, PriceMode,
        ProgressContext, SolutionContext,
    };
    // The search capabilities a backend may have, and the handler each one takes.
    pub use culmen_core::{
        CutHandler, HeuristicHandler, Heuristics, LazyConstraints, LazyHandler, Pricer, Prices, UserCuts,
    };
}
