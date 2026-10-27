// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The seam a solver plugs into, and the callbacks it drives.

use core::fmt;

use super::context::{
    CutContext, HeuristicContext, LazyContext, LogContext, PollContext, ProgressContext, SolutionContext,
};
use crate::{
    Analyses, Analysis, AnalysisError, Basis, Col, Duals, Edit, Events, Iis, ModelClasses, Program, ProgramFeatures,
    Ranging, Relaxed, Row, Settings, Solved, Start, TiVec, Weights,
};

/// A solver culmen can drive: it loads a [`Program`], solves it under [`Settings`], and calls a [`Callback`] as the
/// search goes.
///
/// The trait is open: any crate may implement it for its solver. Each associated context type implements the
/// solver-neutral trait of its event and may offer more as inherent methods; an event the solver cannot deliver
/// takes [`Unsupported`](crate::Unsupported) and is left out of [`Backend::EVENTS`]. The classes of program it
/// solves are [`Backend::CLASSES`].
///
/// # Contract
///
/// - [`Solver::new`](crate::Solver::new) calls [`Backend::load`] only with a program whose
///   [`class`](Program::class) is in [`Backend::CLASSES`] and whose [`features`](Program::features) are in
///   [`Backend::FEATURES`], so `load` need not check them.
/// - A [`Solver`](crate::Solver) calls [`Backend::edit`] only right after the backend held the program before the
///   edit, and only with an edited program whose class is in [`Backend::CLASSES`] and whose features are in
///   [`Backend::FEATURES`]. When `edit` fails, the `Solver` calls [`Backend::load`] with the program before the edit.
/// - [`Backend::solve`] solves the last program [`Backend::load`] or [`Backend::edit`] accepted, and its [`Solved`]
///   reads with that program's handles: a [`Solution`](crate::Solution) holds one value per column, every objective value and
///   bound includes the program's quadratic part and offset.
/// - It calls a callback method only for an event in both [`Backend::EVENTS`] and the callback's
///   [`Callback::events`], and only while `solve` runs; it never calls two callback methods at once.
/// - A panic in a callback stops the solve as soon as the solver allows and then continues unwinding out of
///   `solve`; it never aborts the process and never leaves the backend unusable.
/// - [`Settings`] apply to this one solve and take precedence over the backend's own options.
/// - A [`Solver`](crate::Solver) calls an analysis method only for an analysis in [`Backend::ANALYSES`] and
///   defined for the program's class ([`Analysis::classes`]), with the loaded program, and only in the state its
///   [`Solver`](crate::Solver) method documents: [`Backend::duals`] and [`Backend::ranging`] after a solve that ended
///   optimal, [`Backend::basis`] after a solve, the rays after a solve that ended as they prove, a basis or weights
///   that fit the program. Called otherwise, it may fail or give an unspecified result, never undefined behaviour.
pub trait Backend: Sized {
    /// Why loading or solving failed.
    type Error: core::error::Error + Send + Sync + 'static;
    /// The solver's own status at the end of a solve, kept for diagnostics.
    type Status: fmt::Debug + Clone + PartialEq + Send + 'static;
    /// The context of a [`Events::LOG`] event.
    type Log<'a>: LogContext;
    /// The context of a [`Events::PROGRESS`] event.
    type Progress<'a>: ProgressContext;
    /// The context of a [`Events::POLL`] event.
    type Poll<'a>: PollContext;
    /// The context of a [`Events::FEASIBLE`] or [`Events::INCUMBENT`] event.
    type Solution<'a>: SolutionContext;
    /// The context of a [`Events::LAZY`] event.
    type Lazy<'a>: LazyContext;
    /// The context of a [`Events::CUT`] event.
    type Cut<'a>: CutContext;
    /// The context of a [`Events::HEURISTIC`] event.
    type Heuristic<'a>: HeuristicContext;
    /// The context of a [`Events::NATIVE`] event: whatever only this solver reports.
    type Native<'a>: fmt::Debug;

    /// The events this backend delivers.
    const EVENTS: Events;

    /// The classes of program this backend solves; [`Solver::new`](crate::Solver::new) refuses any other.
    const CLASSES: ModelClasses;

    /// The analyses this backend provides after a solve; none unless declared. A
    /// [`Solver`](crate::Solver) refuses any other with [`AnalysisError::Unsupported`], so the default bodies of the
    /// analysis methods, which return that error, run only when one is called directly.
    const ANALYSES: Analyses = Analyses::NONE;
    /// The features this backend takes; [`Solver::new`](crate::Solver::new) refuses a program using any other, and
    /// [`Solver::solve`](crate::Solver::solve) a request using any other. None unless declared.
    const FEATURES: ProgramFeatures = ProgramFeatures::NONE;

    /// Loads a program, replacing the one before.
    ///
    /// Called by [`Solver::new`](crate::Solver::new) only with a program of a class in [`Backend::CLASSES`]; called
    /// directly with another, it may refuse the program or fail to solve it.
    ///
    /// # Errors
    ///
    /// When the solver refuses the program.
    fn load(&mut self, program: &Program) -> Result<(), Self::Error>;

    /// Hands the solver a start for the next solve of the loaded program.
    ///
    /// Called by [`Solver::solve`](crate::Solver::solve) right before [`Backend::solve`], only when
    /// [`Backend::FEATURES`] holds [`ProgramFeature::StartSolution`](crate::ProgramFeature::StartSolution) and with a
    /// start [`Start::check`] accepts for the loaded program. The default does nothing, as befits a backend that does
    /// not declare the feature.
    ///
    /// # Errors
    ///
    /// When the solver refuses the start.
    fn set_start(&mut self, start: &Start) -> Result<(), Self::Error> {
        let _ = start;
        Ok(())
    }

    /// Changes the loaded program into `program`, which is the loaded program changed as `edit` names.
    ///
    /// The default loads `program` whole: always correct, and without any warm start. A backend overrides it to change
    /// its model in place and keep its warm start, and falls back to [`Backend::load`] for any [`Edit`] it does not
    /// handle, since the enum is non-exhaustive.
    ///
    /// # Errors
    ///
    /// When the solver refuses the change. The backend may then hold any program: the [`Solver`](crate::Solver) loads
    /// the program before the edit again.
    fn edit(&mut self, program: &Program, edit: Edit<'_>) -> Result<(), Self::Error> {
        let _ = edit;
        self.load(program)
    }

    /// Solves the loaded program, calling `callback` as the search goes.
    ///
    /// # Errors
    ///
    /// When nothing is loaded, a setting cannot be applied, or the solver fails.
    ///
    /// # Panics
    ///
    /// When `callback` panics, after the solver has stopped.
    fn solve(
        &mut self,
        settings: &Settings,
        callback: &mut dyn Callback<Self>,
    ) -> Result<Solved<Self::Status>, Self::Error>;

    /// The row duals and reduced costs of the last solve, in the sign convention of [`Duals`].
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] unless [`Backend::ANALYSES`] has [`Analysis::Duals`]; the backend's own error.
    fn duals(&self, program: &Program) -> Result<Duals, AnalysisError<Self::Error>> {
        let _ = program;
        Err(unsupported::<Self>(Analysis::Duals))
    }

    /// The basis of the last solve, `None` when the backend holds none.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] unless [`Backend::ANALYSES`] has [`Analysis::Basis`]; the backend's own error.
    fn basis(&self, program: &Program) -> Result<Option<Basis>, AnalysisError<Self::Error>> {
        let _ = program;
        Err(unsupported::<Self>(Analysis::Basis))
    }

    /// Sets the basis the next solve starts from.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] unless [`Backend::ANALYSES`] has [`Analysis::Basis`]; the backend's own error.
    fn set_basis(&mut self, program: &Program, basis: &Basis) -> Result<(), AnalysisError<Self::Error>> {
        let _ = (program, basis);
        Err(unsupported::<Self>(Analysis::Basis))
    }

    /// The cost and bound ranging of the last solve's optimal basis.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] unless [`Backend::ANALYSES`] has [`Analysis::Ranging`]; the backend's own
    /// error.
    fn ranging(&mut self, program: &Program) -> Result<Ranging, AnalysisError<Self::Error>> {
        let _ = program;
        Err(unsupported::<Self>(Analysis::Ranging))
    }

    /// A direction of unboundedness, as [`Solver::primal_ray`](crate::Solver::primal_ray) states it.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] unless [`Backend::ANALYSES`] has [`Analysis::Rays`]; the backend's own error.
    fn primal_ray(&mut self, program: &Program) -> Result<Option<TiVec<Col, f64>>, AnalysisError<Self::Error>> {
        let _ = program;
        Err(unsupported::<Self>(Analysis::Rays))
    }

    /// A Farkas certificate of infeasibility, as [`Solver::dual_ray`](crate::Solver::dual_ray) states it.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] unless [`Backend::ANALYSES`] has [`Analysis::Rays`]; the backend's own error.
    fn dual_ray(&mut self, program: &Program) -> Result<Option<TiVec<Row, f64>>, AnalysisError<Self::Error>> {
        let _ = program;
        Err(unsupported::<Self>(Analysis::Rays))
    }

    /// An irreducible infeasible subsystem, `None` when the constraints are feasible.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] unless [`Backend::ANALYSES`] has [`Analysis::Iis`]; the backend's own error.
    fn iis(&mut self, program: &Program) -> Result<Option<Iis>, AnalysisError<Self::Error>> {
        let _ = program;
        Err(unsupported::<Self>(Analysis::Iis))
    }

    /// A point of least weighted violation, `None` when no point meets the hard bounds and rows.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] unless [`Backend::ANALYSES`] has [`Analysis::FeasibilityRelaxation`]; the
    /// backend's own error.
    fn relax(&mut self, program: &Program, weights: &Weights) -> Result<Option<Relaxed>, AnalysisError<Self::Error>> {
        let _ = (program, weights);
        Err(unsupported::<Self>(Analysis::FeasibilityRelaxation))
    }
}

/// The refusal of an analysis `B` does not provide.
const fn unsupported<B: Backend>(analysis: Analysis) -> AnalysisError<B::Error> {
    AnalysisError::Unsupported {
        analysis,
        supported: B::ANALYSES,
    }
}

/// What runs during a solve: one method per event kind, each with the backend's context for it.
///
/// Every method does nothing by default. Implement it generically (`impl<B: Backend> Callback<B> for …`) to run
/// on every backend through the neutral context traits, or for one backend to reach its own context types. Two
/// callbacks run together as a pair `(a, b)`, `a` first.
///
/// ```
/// use culmen_core::{Backend, Callback, Events, PollContext};
/// use std::time::Duration;
///
/// /// Stops any backend's solve after a fixed time.
/// struct Deadline(Duration);
///
/// impl<B: Backend> Callback<B> for Deadline {
///     fn events(&self) -> Events {
///         Events::POLL
///     }
///     fn poll(&mut self, ctx: &mut B::Poll<'_>) {
///         if ctx.progress().elapsed >= self.0 {
///             ctx.interrupt();
///         }
///     }
/// }
/// ```
pub trait Callback<B: Backend>: Send {
    /// The events this callback wants; every event by default.
    fn events(&self) -> Events {
        Events::ALL
    }
    /// A chunk of the solver's log.
    fn log(&mut self, ctx: &mut B::Log<'_>) {
        let _ = ctx;
    }
    /// A progress report.
    fn progress(&mut self, ctx: &mut B::Progress<'_>) {
        let _ = ctx;
    }
    /// A poll at which the solve may be interrupted.
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        let _ = ctx;
    }
    /// Any feasible solution.
    fn feasible(&mut self, ctx: &mut B::Solution<'_>) {
        let _ = ctx;
    }
    /// A new incumbent.
    fn incumbent(&mut self, ctx: &mut B::Solution<'_>) {
        let _ = ctx;
    }
    /// A candidate solution, to be cut off by lazy constraints.
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        let _ = ctx;
    }
    /// A node relaxation, to be cut off by user cuts.
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        let _ = ctx;
    }
    /// A chance to hand the solver a heuristic solution.
    fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>) {
        let _ = ctx;
    }
    /// An event only this backend knows.
    fn native(&mut self, ctx: &mut B::Native<'_>) {
        let _ = ctx;
    }
}

impl<B: Backend> Callback<B> for () {
    fn events(&self) -> Events {
        Events::NONE
    }
}

impl<B: Backend, C: Callback<B> + ?Sized> Callback<B> for &mut C {
    fn events(&self) -> Events {
        (**self).events()
    }
    fn log(&mut self, ctx: &mut B::Log<'_>) {
        (**self).log(ctx);
    }
    fn progress(&mut self, ctx: &mut B::Progress<'_>) {
        (**self).progress(ctx);
    }
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        (**self).poll(ctx);
    }
    fn feasible(&mut self, ctx: &mut B::Solution<'_>) {
        (**self).feasible(ctx);
    }
    fn incumbent(&mut self, ctx: &mut B::Solution<'_>) {
        (**self).incumbent(ctx);
    }
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        (**self).lazy(ctx);
    }
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        (**self).cut(ctx);
    }
    fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>) {
        (**self).heuristic(ctx);
    }
    fn native(&mut self, ctx: &mut B::Native<'_>) {
        (**self).native(ctx);
    }
}

/// Each callback of the pair runs only for the events it asked for, the first before the second.
impl<B: Backend, X: Callback<B>, Y: Callback<B>> Callback<B> for (X, Y) {
    fn events(&self) -> Events {
        self.0.events() | self.1.events()
    }
    fn log(&mut self, ctx: &mut B::Log<'_>) {
        if self.0.events().contains(Events::LOG) {
            self.0.log(ctx);
        }
        if self.1.events().contains(Events::LOG) {
            self.1.log(ctx);
        }
    }
    fn progress(&mut self, ctx: &mut B::Progress<'_>) {
        if self.0.events().contains(Events::PROGRESS) {
            self.0.progress(ctx);
        }
        if self.1.events().contains(Events::PROGRESS) {
            self.1.progress(ctx);
        }
    }
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        if self.0.events().contains(Events::POLL) {
            self.0.poll(ctx);
        }
        if self.1.events().contains(Events::POLL) {
            self.1.poll(ctx);
        }
    }
    fn feasible(&mut self, ctx: &mut B::Solution<'_>) {
        if self.0.events().contains(Events::FEASIBLE) {
            self.0.feasible(ctx);
        }
        if self.1.events().contains(Events::FEASIBLE) {
            self.1.feasible(ctx);
        }
    }
    fn incumbent(&mut self, ctx: &mut B::Solution<'_>) {
        if self.0.events().contains(Events::INCUMBENT) {
            self.0.incumbent(ctx);
        }
        if self.1.events().contains(Events::INCUMBENT) {
            self.1.incumbent(ctx);
        }
    }
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        if self.0.events().contains(Events::LAZY) {
            self.0.lazy(ctx);
        }
        if self.1.events().contains(Events::LAZY) {
            self.1.lazy(ctx);
        }
    }
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        if self.0.events().contains(Events::CUT) {
            self.0.cut(ctx);
        }
        if self.1.events().contains(Events::CUT) {
            self.1.cut(ctx);
        }
    }
    fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>) {
        if self.0.events().contains(Events::HEURISTIC) {
            self.0.heuristic(ctx);
        }
        if self.1.events().contains(Events::HEURISTIC) {
            self.1.heuristic(ctx);
        }
    }
    fn native(&mut self, ctx: &mut B::Native<'_>) {
        if self.0.events().contains(Events::NATIVE) {
            self.0.native(ctx);
        }
        if self.1.events().contains(Events::NATIVE) {
            self.1.native(ctx);
        }
    }
}
