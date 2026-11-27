// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The seam a solver plugs into, and the observational callbacks it drives.

use core::fmt;

use super::context::{LogContext, PollContext, ProgressContext, SolutionContext};
use super::{Edit, Hooks};
use crate::{Event, Events, ModelClasses, Program, ProgramFeatures, Settings, Solved};

/// A solver culmen can drive: it loads a [`Program`], solves it under [`Settings`], and calls the [`Hooks`] of the
/// solve as the search goes.
///
/// The trait is open: any crate may implement it for its solver. Each associated context type implements the
/// solver-neutral trait of its event and may offer more as inherent methods; an observational event the solver cannot
/// deliver takes [`Unsupported`](crate::backend::Unsupported) and is left out of [`Backend::EVENTS`]. The classes of
/// program it solves are [`Backend::CLASSES`]. What changes the search (lazy constraints, user cuts, heuristic
/// solutions, pricing), taking a start and the analyses are capability traits of their own
/// ([`LazyConstraints`](crate::LazyConstraints), [`Starts`](crate::Starts), [`DualAnalysis`](crate::DualAnalysis) and
/// their siblings), each with its own contract; the [`backend`](crate::backend) module states what they share.
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
///   reads with that program's handles: a [`Solution`](crate::Solution) holds one value per column, the columns
///   priced in the solve included, and every objective value and bound includes the program's quadratic part and
///   offset.
/// - It delivers an observational event only when it is in [`Hooks::events`], which is within [`Backend::EVENTS`],
///   and only while `solve` runs; it never calls two callback or handler methods at once.
/// - A panic in a callback or handler stops the solve as soon as the solver allows and then continues unwinding out
///   of `solve`; it never aborts the process and never leaves the backend unusable.
/// - [`Settings`] apply to this one solve and take precedence over the backend's own options, when it is
///   [`Configurable`](crate::Configurable).
pub trait Backend: Sized {
    /// Why loading or solving failed.
    type Error: core::error::Error + Send + Sync + 'static;
    /// The solver's own status at the end of a solve, kept for diagnostics.
    type Status: fmt::Debug + Clone + PartialEq + Send + 'static;
    /// The context of a [`Event::Log`] event.
    type Log<'a>: LogContext;
    /// The context of a [`Event::Progress`] event.
    type Progress<'a>: ProgressContext;
    /// The context of a [`Event::Poll`] event.
    type Poll<'a>: PollContext;
    /// The context of a [`Event::Feasible`] or [`Event::Incumbent`] event.
    type Solution<'a>: SolutionContext;
    /// The context of a [`Event::Native`] event: whatever only this solver reports.
    type Native<'a>: fmt::Debug;

    /// The observational events this backend delivers.
    const EVENTS: Events;

    /// The classes of program this backend solves; [`Solver::new`](crate::Solver::new) refuses any other.
    const CLASSES: ModelClasses;

    /// The features of a program this backend takes; [`Solver::new`](crate::Solver::new) and the edits refuse a
    /// program using any other. None unless declared.
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

    /// Changes the loaded program into `program`, which is the loaded program changed as `edit` names.
    ///
    /// The default loads `program` whole: always correct, and without any warm start. A backend overrides it to change
    /// its model in place and keep its warm start. The override matches every variant of [`Edit`] by name, with no
    /// wildcard arm, and calls [`Backend::load`] explicitly for each one it does not change in place, so a variant a
    /// later version of culmen adds fails to compile in the backend instead of loading silently.
    ///
    /// # Errors
    ///
    /// When the solver refuses the change. The backend may then hold any program: the [`Solver`](crate::Solver) loads
    /// the program before the edit again.
    fn edit(&mut self, program: &Program, edit: &Edit) -> Result<(), Self::Error> {
        let _ = edit;
        self.load(program)
    }

    /// Solves the loaded program, delivering events to `hooks` as the search goes and calling the handlers it holds.
    ///
    /// A backend that [`Prices`](crate::Prices) hands every column the pricer adds to the sink
    /// [`Hooks::take_priced`] gives, which was made for the loaded program. The columns in it when `solve` returns or
    /// unwinds belong to the caller: a [`Solver`](crate::Solver) appends them to its program and loads it again
    /// before the next call that runs the backend.
    ///
    /// # Errors
    ///
    /// When nothing is loaded, a setting cannot be applied, or the solver fails.
    ///
    /// # Panics
    ///
    /// When a callback or handler panics, after the solver has stopped.
    fn solve(&mut self, settings: &Settings, hooks: Hooks<'_, Self>) -> Result<Solved<Self::Status>, Self::Error>;
}

/// What watches a solve: one method per observational event kind, each with the backend's context for it.
///
/// Every method does nothing by default, and [`Callback::events`] asks for every event. A callback only observes:
/// what changes the search is a handler of its own kind ([`LazyHandler`](crate::LazyHandler),
/// [`CutHandler`](crate::CutHandler), [`HeuristicHandler`](crate::HeuristicHandler), [`Pricer`](crate::Pricer)),
/// so asking for every event costs a solver no reduction it would otherwise make. Implement it generically
/// (`impl<B: Backend> Callback<B> for …`) to run on every backend through the neutral context traits, or for one
/// backend to reach its own context types. Two callbacks run together as a pair `(a, b)`, `a` first.
///
/// ```
/// use culmen_core::{Backend, Callback, Context, Event, Events, PollContext};
/// use std::time::Duration;
///
/// /// Stops any backend's solve after a fixed time.
/// struct Deadline(Duration);
///
/// impl<B: Backend> Callback<B> for Deadline {
///     fn events(&self) -> Events {
///         Event::Poll.into()
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
        if self.0.events().contains(Event::Log) {
            self.0.log(ctx);
        }
        if self.1.events().contains(Event::Log) {
            self.1.log(ctx);
        }
    }
    fn progress(&mut self, ctx: &mut B::Progress<'_>) {
        if self.0.events().contains(Event::Progress) {
            self.0.progress(ctx);
        }
        if self.1.events().contains(Event::Progress) {
            self.1.progress(ctx);
        }
    }
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        if self.0.events().contains(Event::Poll) {
            self.0.poll(ctx);
        }
        if self.1.events().contains(Event::Poll) {
            self.1.poll(ctx);
        }
    }
    fn feasible(&mut self, ctx: &mut B::Solution<'_>) {
        if self.0.events().contains(Event::Feasible) {
            self.0.feasible(ctx);
        }
        if self.1.events().contains(Event::Feasible) {
            self.1.feasible(ctx);
        }
    }
    fn incumbent(&mut self, ctx: &mut B::Solution<'_>) {
        if self.0.events().contains(Event::Incumbent) {
            self.0.incumbent(ctx);
        }
        if self.1.events().contains(Event::Incumbent) {
            self.1.incumbent(ctx);
        }
    }
    fn native(&mut self, ctx: &mut B::Native<'_>) {
        if self.0.events().contains(Event::Native) {
            self.0.native(ctx);
        }
        if self.1.events().contains(Event::Native) {
            self.1.native(ctx);
        }
    }
}
