// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One solve to configure: its settings, stop, observer, callback, start and handlers.

use core::fmt;
use core::time::Duration;

use super::{Hooks, Watch};
use crate::{
    Backend, Callback, CutHandler, HeuristicHandler, Heuristics, LazyConstraints, LazyHandler, Observer, Pricer,
    Prices, Settings, SolveError, Solved, Solver, Start, Starts, Stop, UserCuts,
};

/// Checks a start against the solver's program and hands it to the backend; set only where `B` [`Starts`].
pub(super) type HandStart<B> = fn(&mut Solver<B>, &Start) -> Result<(), SolveError<<B as Backend>::Error>>;

/// One solve of a [`Solver`] to configure, made by [`Solver::run`] and run by [`Run::solve`].
///
/// Each method replaces what it sets and leaves the rest as it was, so the last call of a method is the one that
/// counts; [`Run::settings`] sets every setting, the time limit included, and [`Run::time_limit`] sets only that.
/// The methods that register a start or a handler exist only for a backend with the capability, so a handler the
/// backend would ignore does not compile.
///
/// ```
/// use core::time::Duration;
/// use std::sync::atomic::AtomicBool;
/// use culmen_core::{Model, Outcome, Settings, Solver, Var, relative_gap};
/// use culmen_core::script::Script;
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::binary());
/// m.maximize(x);
/// let mut solver = Solver::new(Script::optimal(1.0, vec![1.0]), m.build()?)?;
///
/// let stop = AtomicBool::new(false);
/// let mut seen = ();
/// let solved = solver
///     .run()
///     .settings(Settings { relative_gap: Some(relative_gap!(0.01)), ..Settings::new() })
///     .time_limit(Duration::from_secs(60))
///     .stop(&stop)
///     .observer(&mut seen)
///     .progress_interval(Duration::from_millis(500))
///     .start(vec![(x, 1.0)])
///     .solve()?;
/// assert!(matches!(solved.outcome, Outcome::Optimal(_)));
/// assert_eq!(solver.backend().record.settings[0].time_limit, Some(Duration::from_secs(60)));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use = "a run solves nothing until `.solve()`"]
pub struct Run<'s, 'h, B: Backend> {
    solver: &'s mut Solver<B>,
    settings: Settings,
    stop: Option<&'h dyn Stop>,
    observer: Option<&'h mut dyn Observer>,
    progress_interval: Duration,
    start: Option<(Start, HandStart<B>)>,
    hooks: Hooks<'h, B>,
}

impl<'s, 'h, B: Backend> Run<'s, 'h, B> {
    /// The backend's defaults, no stop, no observer, no callback, no start and no handler.
    pub(super) const fn new(solver: &'s mut Solver<B>) -> Self {
        Self {
            solver,
            settings: Settings::new(),
            stop: None,
            observer: None,
            progress_interval: Watch::DEFAULT_PROGRESS_INTERVAL,
            start: None,
            hooks: Hooks::new(),
        }
    }

    /// Solves under every setting of `settings`, its time limit included: a time limit set before with
    /// [`Run::time_limit`] is replaced, and one set after replaces `settings.time_limit`.
    pub const fn settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    /// The wall-clock time the solve may take, replacing only the time limit: the other settings given before with
    /// [`Run::settings`] stay, and a later [`Run::settings`] replaces this time limit with its own.
    pub const fn time_limit(mut self, limit: Duration) -> Self {
        self.settings.time_limit = Some(limit);
        self
    }

    /// Polls `stop` while the solve runs, and interrupts the solve once it asks to.
    pub fn stop(mut self, stop: &'h impl Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Reports progress, improving solutions and complete log lines to `observer`.
    pub fn observer(mut self, observer: &'h mut impl Observer) -> Self {
        self.observer = Some(observer);
        self
    }

    /// The least time between two progress reports to the observer; [`Watch::DEFAULT_PROGRESS_INTERVAL`] when not
    /// set.
    pub const fn progress_interval(mut self, interval: Duration) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Runs `callback` for the observational events it asks for, with the backend's full context types, after the
    /// stop and the observer.
    pub fn callback(mut self, callback: &'h mut impl Callback<B>) -> Self {
        self.hooks = self.hooks.register_callback(callback);
        self
    }

    /// Solves, polling the stop and reporting to the observer, and returns how the solve ended.
    ///
    /// A start is checked against the program with [`Start::check`] before the backend sees it; a start the check
    /// refuses leaves the last solve readable. A solve with a [`Pricer`] appends the columns it priced to the
    /// program, in the order of their ids, however it ends: returning, failing or unwinding from a panic. The
    /// backend loads the grown program at the next call that runs it, and the analyses find no solve to read until
    /// the next solve that prices nothing.
    ///
    /// # Errors
    ///
    /// [`SolveError::Start`] for a start [`Start::check`] refuses, before the backend sees it;
    /// [`SolveError::Backend`] when the backend refuses the program it reloads after an edit whose reload failed or
    /// the start, a setting cannot be applied or the solver fails.
    ///
    /// # Panics
    ///
    /// When the observer, the callback or a handler panics: the panic continues once the solver has stopped.
    pub fn solve(self) -> Result<Solved<B::Status>, SolveError<B::Error>> {
        let mut watch = Watch::new(self.stop, self.observer, self.progress_interval);
        self.solver
            .execute(&self.settings, &mut watch, self.start.as_ref(), self.hooks)
    }
}

impl<B: Starts> Run<'_, '_, B> {
    /// Begins the search from `start`: a full [`TiVec<Col, f64>`](crate::TiVec) or `Vec<f64>`, or a sparse
    /// `Vec<(Col, f64)>`.
    pub fn start(mut self, start: impl Into<Start>) -> Self {
        self.start = Some((start.into(), Solver::hand_start));
        self
    }
}

impl<'h, B: LazyConstraints> Run<'_, 'h, B> {
    /// Runs `handler` for every candidate solution, to cut it off with lazy constraints.
    pub fn lazy(mut self, handler: &'h mut impl LazyHandler<B>) -> Self {
        self.hooks = self.hooks.register_lazy(handler);
        self
    }
}

impl<'h, B: UserCuts> Run<'_, 'h, B> {
    /// Runs `handler` for node relaxations, to cut them off with user cuts.
    pub fn cut(mut self, handler: &'h mut impl CutHandler<B>) -> Self {
        self.hooks = self.hooks.register_cut(handler);
        self
    }
}

impl<'h, B: LazyConstraints + UserCuts> Run<'_, 'h, B> {
    /// Runs one `handler` both for every candidate solution and for node relaxations, as a separator that cuts off
    /// integer and fractional points alike, replacing the lazy-constraint and cut handlers registered before; a later
    /// [`Run::lazy`] or [`Run::cut`] takes its role over.
    pub fn lazy_and_cut(mut self, handler: &'h mut (impl LazyHandler<B> + CutHandler<B>)) -> Self {
        self.hooks = self.hooks.register_lazy_and_cut(handler);
        self
    }
}

impl<'h, B: Heuristics> Run<'_, 'h, B> {
    /// Runs `handler` at the points the backend takes heuristic solutions.
    pub fn heuristic(mut self, handler: &'h mut impl HeuristicHandler<B>) -> Self {
        self.hooks = self.hooks.register_heuristic(handler);
        self
    }
}

impl<'h, B: Prices> Run<'_, 'h, B> {
    /// Runs `pricer` at every node LP, to add columns that improve it; the solver appends them to its program.
    pub fn pricer(mut self, pricer: &'h mut impl Pricer<B>) -> Self {
        self.hooks = self.hooks.solver_pricer(pricer);
        self
    }
}

impl<B: Backend> fmt::Debug for Run<'_, '_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Run")
            .field("settings", &self.settings)
            .field("stop", &self.stop.is_some())
            .field("observer", &self.observer.is_some())
            .field("progress_interval", &self.progress_interval)
            .field("start", &self.start.as_ref().map(|(start, _)| start))
            .field("hooks", &self.hooks)
            .finish_non_exhaustive()
    }
}
