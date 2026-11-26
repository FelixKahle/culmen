// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a backend can do beyond [`Backend`]: the handlers that change its search, take a start, and the analyses
//! after a solve.
//!
//! Each capability is a trait a backend implements or does not, and each method that needs one, a
//! [`Solver`](crate::Solver) method or a registration on [`Hooks`](crate::Hooks), exists only for a backend that has
//! it, so asking a backend for what it cannot do does not compile: a lazy-constraint handler registered for a backend
//! without [`LazyConstraints`] is a compile error, not a solve that ignores it and reports the relaxation's optimum.
//! What depends on the program or on the solve (an analysis defined for the program's class, a solve to read, a basis
//! that fits, a column a pricer adds) stays a run-time refusal, checked the same way for every backend.
//!
//! # Contract
//!
//! A backend with a search capability calls the registered handler through its [`Hooks`](crate::Hooks), only while
//! [`Backend::solve`] runs, never two handlers or a handler and a callback at once, and only when the hooks have a
//! handler of that kind ([`Hooks::wants_lazy`](crate::Hooks::wants_lazy) and its siblings). A solve without a handler
//! of a kind behaves as if the backend lacked the capability, and gives up nothing for it.
//!
//! A [`Solver`](crate::Solver) calls an analysis method only for an analysis defined for the program's class
//! ([`Analysis::classes`](crate::Analysis::classes)) and features ([`Analysis::refused`](crate::Analysis::refused)),
//! with the loaded program, and only in the state its [`Solver`](crate::Solver) method documents:
//! [`DualAnalysis::duals`] and [`RangingAnalysis::ranging`] after a solve that ended optimal,
//! [`BasisAnalysis::basis`] after a solve, the rays after a solve that ended as they prove, a basis or weights that fit
//! the program. Called otherwise, it may fail or give an unspecified result, never undefined behaviour.

use crate::{
    Backend, Basis, Col, CutContext, Duals, HeuristicContext, Iis, LazyContext, PriceContext, Program, Ranging,
    Relaxed, Row, Start, TiVec, Weights,
};

/// The backend takes lazy constraints: it hands every candidate solution it would accept to a registered
/// [`LazyHandler`](crate::LazyHandler), which may cut it off.
///
/// Every constraint the handler adds holds for the rest of the search, so a candidate it cut off is never accepted.
#[diagnostic::on_unimplemented(message = "`{Self}` does not take lazy constraints")]
pub trait LazyConstraints: Backend {
    /// The context of a candidate solution.
    type Lazy<'a>: LazyContext;
}

/// The backend takes user cuts: it hands node relaxations to a registered [`CutHandler`](crate::CutHandler), which
/// may cut them off with constraints every integer-feasible point satisfies.
#[diagnostic::on_unimplemented(message = "`{Self}` does not take user cuts")]
pub trait UserCuts: Backend {
    /// The context of a node relaxation.
    type Cut<'a>: CutContext;
}

/// The backend takes heuristic solutions: it asks a registered [`HeuristicHandler`](crate::HeuristicHandler) for
/// solutions of its own at points of its choosing.
#[diagnostic::on_unimplemented(message = "`{Self}` does not take heuristic solutions")]
pub trait Heuristics: Backend {
    /// The context of a point at which the handler may supply a solution.
    type Heuristic<'a>: HeuristicContext;
}

/// The backend prices: it hands every node LP it solves to a registered [`Pricer`](crate::Pricer), which answers with
/// columns that improve it, and takes those columns into the search.
///
/// # Contract
///
/// The backend calls the pricer only after a node LP is solved, optimal in
/// [`PriceMode::ReducedCost`](crate::PriceMode::ReducedCost) and infeasible in
/// [`PriceMode::Farkas`](crate::PriceMode::Farkas), with every program row's multiplier in the convention of
/// [`PriceContext::duals`]. It ends pricing at a node only after a round that added no column and did not branch
/// early, and counts the node LP as a bound only then. Every column the pricer adds goes through [`Priced::push`] of
/// the sink [`Hooks::take_priced`](crate::Hooks::take_priced) gives, enters the search with its cost, bounds,
/// integrality and entries, stays for the rest of the solve, and is part of every point the backend reports
/// afterwards.
///
/// [`Priced::push`]: crate::Priced::push
#[diagnostic::on_unimplemented(message = "`{Self}` does not price columns")]
pub trait Prices: Backend {
    /// The context of a pricing round: a node LP whose duals the pricer answers with columns.
    type Price<'a>: PriceContext;
}

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
