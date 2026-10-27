// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]
//! The core of culmen: linear, mixed-integer and convex quadratic programs as data, a modelling layer that reads
//! like the mathematics, and a solver-neutral seam with rich callbacks, a cooperative stop and a throttled observer.
//!
//! This crate holds no solver and no `unsafe`. A solver plugs in by implementing [`Backend`]; the `culmen` facade
//! re-exports this crate with the HiGHS backend and the modelling macros.
//!
//! # The line through the crate
//!
//! - **The program is data.** A [`Program`] is built once, by value, from a [`ProgramPlan`] or a [`Model`]; every
//!   number in it was checked then. [`Col`] and [`Row`] are typed `Copy` handles, and the handles a model returns
//!   are the columns and rows of the program it builds.
//! - **Models read like the mathematics.** [`Var`] builds a variable, the operators on [`Col`], [`Term`] and
//!   [`LinExpr`] build expressions, [`Linear::leq`], [`Linear::geq`], [`Linear::equals`] and [`Linear::between`]
//!   compare them into a [`Constraint`], and [`Model::add_var_map`] keys a family by domain keys. A product of
//!   columns is a [`QuadTerm`], and a convex quadratic objective a [`QuadExpr`]; constraints stay linear.
//! - **The solver is a seam.** A [`Backend`] loads a [`Program`] and solves it under [`Settings`], calling a
//!   [`Callback`] with one typed context per event kind ([`Events`]); what it cannot deliver it declares, and its
//!   context for that event is [`Unsupported`], so such a callback never runs. It declares the [`ModelClasses`] it
//!   solves, and [`Solver::new`] refuses a program of any other [`ModelClass`] with a typed [`LoadError`]. Whatever a
//!   solver offers beyond the neutral contexts is on its own context types.
//! - **A solve is a value.** A [`Solver`] holds a loaded backend; each [`Solver::solve`] takes a [`Request`] of
//!   settings, a [`Stop`] and an [`Observer`], and returns [`Solved`]: a generic [`Outcome`] and the backend's own
//!   status.
//!
//! ```text
//! Model ──build──▶ Program ──▶ Solver::new(backend, program) ──solve(Request: settings, stop, observer)──▶ Solved
//!   │                                         └──solve_with(request, &mut callback)       ──▶ Solved
//!   └─ add_var, add_var_map, add_constraint, set_objective        Outcome: Optimal | GapReached | Stopped | …
//! ```
//!
//! # A model, two ways
//!
//! Every model is written with these methods; with the `macros` feature of `culmen`, `variable!`, `constraint!`,
//! `objective!` and `sum!` write the same calls more briefly.
//!
//! ```
//! use culmen_core::{Linear, LinExpr, Model, Var, product};
//!
//! // Assign each product to one location within capacity, at the least walking cost.
//! let (products, locations) = (0..3, 0..2);
//! let vol = [4.0, 3.0, 5.0];
//! let cap = [8.0, 6.0];
//! let walk = [[1.0, 3.0], [2.0, 1.0], [4.0, 2.0]];
//!
//! let mut m = Model::new();
//! let x = m.add_var_map(product(&products, &locations), |(p, l)| Var::binary().name(format!("x[{p},{l}]")));
//! m.add_constraint_map(products.clone(), |&p| {
//!     (format!("once[{p}]"), LinExpr::sum(locations.clone().map(|l| x[(p, l)])).equals(1))
//! });
//! m.add_constraint_map(locations.clone(), |&l| {
//!     (format!("cap[{l}]"), LinExpr::sum(products.clone().map(|p| vol[p] * x[(p, l)])).leq(cap[l]))
//! });
//! m.minimize(LinExpr::sum(x.iter().map(|(&(p, l), x)| walk[p][l] * x)));
//!
//! let program = m.build()?;
//! assert_eq!((program.num_cols(), program.num_rows()), (6, 5));
//! # Ok::<(), culmen_core::ModelError>(())
//! ```
//!
//! # Features
//!
//! - **`serde`**: the plain data ([`Program`], [`ProgramPlan`], [`Bounds`], [`Settings`], [`Outcome`], [`Solution`],
//!   [`Progress`], …) serializes and reads back; a [`Program`] and every checked value are checked again as they are
//!   read.

mod gap;
mod model;
mod program;
mod script;
mod solve;

pub use gap::{AbsoluteGap, InvalidAbsoluteGap, InvalidRelativeGap, RelativeGap};
pub use model::{
    Combination, Constraint, ConstraintMap, ExprPlace, Handles, IndexSet, Keyed, KeyedIter, Keys, LinExpr, Linear,
    Members, Model, ModelConstraint, ModelError, ModelVar, Objective, Product, QuadExpr, QuadTerm, Quadratic, Summand,
    Term, Values, Var, VarMap, product, set,
};
pub use program::{
    Bounds, Col, Cols, FurtherObjectives, HessianEntries, Integrality, InvalidBounds, InvalidWeight, LinearObjective,
    LpFileError, ModelClass, ModelClasses, ModelClassesIter, ObjectiveId, Objectives, Program, ProgramColumn,
    ProgramError, ProgramFeature, ProgramFeatures, ProgramFeaturesIter, ProgramPlan, ProgramRow, QuadEntries,
    QuadObjective, Row, RowEntries, RowMatrix, RowView, Rows, Sense, Stage, Tolerance, Weight, Weighted,
};
pub use script::{
    Found, Record, Script, ScriptCut, ScriptError, ScriptHeuristic, ScriptLazy, ScriptLog, ScriptNote, ScriptPoll,
    ScriptProgress, ScriptSolution, ScriptStatus, Step,
};
pub use solve::context;
pub use solve::{
    ActionError, AddedRow, Backend, Callback, CutContext, Edit, EditError, Events, EventsIter, HeuristicContext,
    Improvement, LazyContext, LoadError, LogContext, LogLevel, LogLine, NewCol, NewRow, Observer, Outcome, Phase,
    PollContext, Progress, ProgressContext, Request, Settings, Solution, SolutionContext, SolveError, Solved, Solver,
    Start, StartEntries, StartError, Stop, StopFn, StopReason, Unsupported, Watch, stop_fn,
};
pub use solve::{
    Analyses, AnalysesIter, Analysis, AnalysisError, Basis, BasisStatus, Bound, Duals, Iis, InvalidPenalty, Limit,
    Line, Misfit, Penalty, Ranged, Ranging, Relaxed, Side, Table, Violation, Violations, Weights,
};
pub use typed_index_collections::{TiSlice, TiVec};
