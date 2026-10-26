// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The seam a solver plugs into, and the callbacks it drives.

use core::fmt;

use super::context::{
    CutContext, HeuristicContext, LazyContext, LogContext, PollContext, ProgressContext, SolutionContext,
};
use crate::{Events, Lp, ModelClasses, Settings, Solved};

/// A solver culmen can drive: it loads an [`Lp`], solves it under [`Settings`], and calls a [`Callback`] as the
/// search goes.
///
/// The trait is open: any crate may implement it for its solver. Each associated context type implements the
/// solver-neutral trait of its event and may offer more as inherent methods; an event the solver cannot deliver
/// takes [`Unsupported`](crate::Unsupported) and is left out of [`Backend::EVENTS`]. The classes of program it
/// solves are [`Backend::CLASSES`].
///
/// # Contract
///
/// - [`Solver::new`](crate::Solver::new) calls [`Backend::load`] only with an LP whose
///   [`class`](Lp::class) is in [`Backend::CLASSES`], so `load` need not check it.
/// - [`Backend::solve`] solves the last LP [`Backend::load`] accepted, and its [`Solved`] reads with that LP's
///   handles: a [`Solution`](crate::Solution) holds one value per column, every objective value and bound
///   includes the LP's quadratic part and offset.
/// - It calls a callback method only for an event in both [`Backend::EVENTS`] and the callback's
///   [`Callback::events`], and only while `solve` runs; it never calls two callback methods at once.
/// - A panic in a callback stops the solve as soon as the solver allows and then continues unwinding out of
///   `solve`; it never aborts the process and never leaves the backend unusable.
/// - [`Settings`] apply to this one solve and take precedence over the backend's own options.
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

    /// Loads an LP, replacing the one before.
    ///
    /// Called by [`Solver::new`](crate::Solver::new) only with an LP of a class in [`Backend::CLASSES`]; called
    /// directly with another, it may refuse the LP or fail to solve it.
    ///
    /// # Errors
    ///
    /// When the solver refuses the LP.
    fn load(&mut self, lp: &Lp) -> Result<(), Self::Error>;

    /// Solves the loaded LP, calling `callback` as the search goes.
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
