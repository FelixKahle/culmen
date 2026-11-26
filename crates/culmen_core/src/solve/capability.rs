// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a backend can do beyond [`Backend`]: take a start, and the analyses after a solve.
//!
//! Each capability is a trait a backend implements or does not, and each [`Solver`](crate::Solver) method that needs
//! one exists only for a backend that has it, so asking a backend for what it cannot do does not compile. What
//! depends on the program or on the solve (an analysis defined for the program's class, a solve to read, a basis
//! that fits) stays a run-time refusal, [`AnalysisError`](crate::AnalysisError), checked the same way for every
//! backend before the backend is asked.
//!
//! # Contract
//!
//! A [`Solver`](crate::Solver) calls an analysis method only for an analysis defined for the program's class
//! ([`Analysis::classes`](crate::Analysis::classes)) and features ([`Analysis::refused`](crate::Analysis::refused)),
//! with the loaded program, and only in the state its [`Solver`](crate::Solver) method documents:
//! [`DualAnalysis::duals`] and [`RangingAnalysis::ranging`] after a solve that ended optimal,
//! [`BasisAnalysis::basis`] after a solve, the rays after a solve that ended as they prove, a basis or weights that fit
//! the program. Called otherwise, it may fail or give an unspecified result, never undefined behaviour.

use crate::{Backend, Basis, Col, Duals, Iis, Program, Ranging, Relaxed, Row, Start, TiVec, Weights};

/// The backend takes a start: a solution, full or partial, the next solve may begin its search from.
///
/// [`Solver::solve_from`](crate::Solver::solve_from) exists only for a backend with this capability.
#[diagnostic::on_unimplemented(message = "`{Self}` does not take a start solution")]
pub trait Starts: Backend {
    /// Hands the solver a start for the next solve of the loaded program.
    ///
    /// Called by [`Solver::solve_from`](crate::Solver::solve_from) right before [`Backend::solve`], with a start
    /// [`Start::check`] accepts for the loaded program.
    ///
    /// # Errors
    ///
    /// When the solver refuses the start.
    fn set_start(&mut self, start: &Start) -> Result<(), Self::Error>;
}

/// The backend reports the row duals and reduced costs of an optimal continuous solve.
///
/// [`Solver::duals`](crate::Solver::duals) exists only for a backend with this capability.
#[diagnostic::on_unimplemented(message = "`{Self}` does not report duals")]
pub trait DualAnalysis: Backend {
    /// The row duals and reduced costs of the last solve, in the sign convention of [`Duals`].
    ///
    /// # Errors
    ///
    /// When the solver fails to report them.
    fn duals(&self, program: &Program) -> Result<Duals, Self::Error>;
}

/// The backend reads the simplex basis after a solve and sets the one the next solve starts from.
///
/// [`Solver::basis`](crate::Solver::basis) and [`Solver::set_basis`](crate::Solver::set_basis) exist only for a
/// backend with this capability.
#[diagnostic::on_unimplemented(message = "`{Self}` does not read or set a basis")]
pub trait BasisAnalysis: Backend {
    /// The basis of the last solve, `None` when the backend holds none.
    ///
    /// # Errors
    ///
    /// When the solver fails to report it.
    fn basis(&self, program: &Program) -> Result<Option<Basis>, Self::Error>;

    /// Sets the basis the next solve starts from.
    ///
    /// # Errors
    ///
    /// When the solver refuses the basis.
    fn set_basis(&mut self, program: &Program, basis: &Basis) -> Result<(), Self::Error>;
}

/// The backend ranges the costs and bounds of an optimal basis.
///
/// [`Solver::ranging`](crate::Solver::ranging) exists only for a backend with this capability.
#[diagnostic::on_unimplemented(message = "`{Self}` does not range costs and bounds")]
pub trait RangingAnalysis: Backend {
    /// The cost and bound ranging of the last solve's optimal basis.
    ///
    /// It takes `&mut self` because a solver may compute ranging by work on its own state, as HiGHS's ranging call
    /// rebuilds simplex data inside its instance.
    ///
    /// # Errors
    ///
    /// When the solver fails to compute it.
    fn ranging(&mut self, program: &Program) -> Result<Ranging, Self::Error>;
}

/// The backend reports a primal ray of an unbounded LP and a dual ray of an infeasible one.
///
/// [`Solver::primal_ray`](crate::Solver::primal_ray) and [`Solver::dual_ray`](crate::Solver::dual_ray) exist only for
/// a backend with this capability.
#[diagnostic::on_unimplemented(message = "`{Self}` does not report rays")]
pub trait RayAnalysis: Backend {
    /// A direction of unboundedness, as [`Solver::primal_ray`](crate::Solver::primal_ray) states it.
    ///
    /// It takes `&mut self` because a solver may run again to find a ray, as HiGHS may run its simplex solver.
    ///
    /// # Errors
    ///
    /// When the solver fails to compute one.
    fn primal_ray(&mut self, program: &Program) -> Result<Option<TiVec<Col, f64>>, Self::Error>;

    /// A Farkas certificate of infeasibility, as [`Solver::dual_ray`](crate::Solver::dual_ray) states it.
    ///
    /// It takes `&mut self` because a solver may run again to find a ray, as HiGHS may run its simplex solver.
    ///
    /// # Errors
    ///
    /// When the solver fails to compute one.
    fn dual_ray(&mut self, program: &Program) -> Result<Option<TiVec<Row, f64>>, Self::Error>;
}

/// The backend finds an irreducible infeasible subsystem of the program's constraints.
///
/// [`Solver::iis`](crate::Solver::iis) exists only for a backend with this capability.
#[diagnostic::on_unimplemented(message = "`{Self}` does not find an IIS")]
pub trait IisAnalysis: Backend {
    /// An irreducible infeasible subsystem of the loaded program, `None` when its constraints are feasible.
    ///
    /// It takes `&mut self` because finding one runs the solver, which replaces the last solve's result.
    ///
    /// # Errors
    ///
    /// When the solver fails.
    fn iis(&mut self, program: &Program) -> Result<Option<Iis>, Self::Error>;
}

/// The backend solves a feasibility relaxation: the least weighted violation of the program's bounds and rows.
///
/// [`Solver::feasibility_relaxation`](crate::Solver::feasibility_relaxation) exists only for a backend with this
/// capability.
#[diagnostic::on_unimplemented(message = "`{Self}` does not solve a feasibility relaxation")]
pub trait RelaxationAnalysis: Backend {
    /// A point of least weighted violation of the loaded program under `weights`, `None` when no point meets the hard
    /// bounds and rows.
    ///
    /// It takes `&mut self` because the relaxation is a solve, which replaces the last solve's result.
    ///
    /// # Errors
    ///
    /// When the solver fails or refuses the program.
    fn relax(&mut self, program: &Program, weights: &Weights) -> Result<Option<Relaxed>, Self::Error>;
}
