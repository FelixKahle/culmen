// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a solve reports beyond its outcome: duals, the basis, ranging, rays, an IIS and a feasibility relaxation.
//!
//! The types here are solver-neutral plain data read with the program's handles. A [`Solver`](crate::Solver) checks
//! every request the same way for every backend (the backend provides the analysis, the analysis is defined for
//! the program's class, a basis or weights fit the program, the solve it reads exists) and only then asks the
//! backend.

mod basis;
mod duals;
mod error;
mod iis;
mod kind;
mod line;
mod ranging;
mod relax;
mod solver;

pub use basis::{Basis, BasisStatus};
pub use duals::Duals;
pub use error::{AnalysisError, Misfit, Table};
pub use iis::{Iis, Side};
pub use kind::{Analyses, AnalysesIter, Analysis};
pub use line::{Bound, Line};
pub use ranging::{Limit, Ranged, Ranging};
pub use relax::{InvalidPenalty, Penalty, Relaxed, Violation, Violations, Weights};
pub(crate) use solver::Ending;
