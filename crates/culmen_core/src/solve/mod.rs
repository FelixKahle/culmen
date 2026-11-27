// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Solving: the backend seam, the generic callbacks, the observer convenience, and the solver value.

mod analysis;
mod backend;
mod capability;
mod context;
mod edit;
mod events;
mod hooks;
mod load;
mod outcome;
mod priced;
mod run;
mod settings;
mod start;
mod watch;

pub use analysis::{
    Analysis, AnalysisError, Basis, BasisStatus, Bound, Duals, Iis, InvalidPenalty, Limit, Line, Misfit, Penalty,
    Ranged, Ranging, Relaxed, Side, Table, Violation, Violations, Weights,
};
pub use backend::{Backend, Callback};
pub use capability::{
    BasisAnalysis, Configurable, DualAnalysis, Heuristics, IisAnalysis, LazyConstraints, Prices, RangingAnalysis,
    RayAnalysis, RelaxationAnalysis, Starts, UserCuts,
};
pub use context::{
    ActionError, AddedRow, Context, CutContext, HeuristicContext, LazyContext, LogContext, PollContext, PriceContext,
    PriceMode, ProgressContext, SolutionContext, Unsupported, check_constraint, check_entries, check_solution,
};
pub use edit::{Edit, EditError};
pub use events::{Event, Events, EventsIter, LogLevel, Phase, Progress};
pub use hooks::{CutHandler, HeuristicHandler, Hooks, LazyHandler, Pricer, Relay};
pub use load::LoadError;
pub use outcome::{Outcome, Solution, Solved, StopReason};
pub use priced::Priced;
pub use run::Run;
pub use settings::Settings;
pub use start::{SolveError, Start, StartEntries, StartError};
pub use watch::{Improvement, LogLine, Observer, Stop, StopFn, Watch, stop_fn};

use crate::{ModelClass, ModelClasses, Program, ProgramFeature, ProgramFeatures};
use analysis::Ended;
use run::HandStart;

/// A program loaded into a backend, solved as often as needed, each solve configured on its own [`Run`].
///
/// ```
/// use culmen_core::{Model, Outcome, Solver, Var};
/// use culmen_core::script::Script;
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::binary());
/// m.maximize(x);
/// let program = m.build()?;
///
/// // The scripted backend stands in for a solver here; with the `highs` feature of `culmen`, `Highs::new()`.
/// let mut solver = Solver::new(Script::optimal(1.0, vec![1.0]), program)?;
/// let solved = solver.solve()?;
/// let Outcome::Optimal(solution) = solved.outcome else { panic!("expected an optimum") };
/// assert_eq!(solution.values[x], 1.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Solver<B> {
    backend: B,
    program: Program,
    /// How the last solve ended; `None` before one, once a later solve reached the backend, until it returns `Ok`, or
    /// once another later call replaced its result. A solve refused before the backend sees it keeps it. A ray asked
    /// of the backend keeps the ending but marks the result as no longer held, since the backend may have run again.
    ended: Option<Ended>,
    /// Whether the backend holds `program`. An edit clears it while the backend changes; it stays clear when that
    /// change and the reload after it both fail, or a panic interrupts them, until the next call that runs the
    /// backend reloads `program`.
    synced: bool,
}

/// Appends the columns a solve priced to the solver's program when it is dropped, so on every path out of the solve,
/// unwinding included, and marks the backend as behind the program.
#[derive(Debug)]
struct Grow<'s> {
    priced: Priced,
    program: &'s mut Program,
    synced: &'s mut bool,
}

impl Drop for Grow<'_> {
    fn drop(&mut self) {
        let cols = self.priced.take_cols();
        if !cols.is_empty() {
            *self.synced = false;
            self.program.push_cols(cols);
        }
    }
}

/// What a backend lacks to hold a program: the program's class, or a feature it uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lack {
    /// The backend does not solve the class.
    Class {
        /// The class.
        class: ModelClass,
        /// The classes the backend solves.
        supported: ModelClasses,
    },
    /// The backend does not take the feature.
    Feature {
        /// The first feature the backend lacks, in the order of [`ProgramFeature`].
        feature: ProgramFeature,
        /// The features the backend takes.
        supported: ProgramFeatures,
    },
}

/// Checks that `B` solves `class` and takes `features`.
///
/// # Errors
///
/// [`Lack`] naming the class, else the first feature, `B` lacks.
fn lack<B: Backend>(class: ModelClass, features: ProgramFeatures) -> Result<(), Lack> {
    if !B::CLASSES.contains(class) {
        return Err(Lack::Class {
            class,
            supported: B::CLASSES,
        });
    }
    features
        .difference(B::FEATURES)
        .iter()
        .next()
        .map_or(Ok(()), |feature| {
            Err(Lack::Feature {
                feature,
                supported: B::FEATURES,
            })
        })
}

impl<E> From<Lack> for LoadError<E> {
    fn from(lack: Lack) -> Self {
        match lack {
            Lack::Class { class, supported } => Self::UnsupportedClass { class, supported },
            Lack::Feature { feature, supported } => Self::UnsupportedFeature { feature, supported },
        }
    }
}

impl<E> From<Lack> for EditError<E> {
    fn from(lack: Lack) -> Self {
        match lack {
            Lack::Class { class, supported } => Self::UnsupportedClass { class, supported },
            Lack::Feature { feature, supported } => Self::UnsupportedFeature { feature, supported },
        }
    }
}

impl<B: Backend> Solver<B> {
    /// Loads `program` into `backend`, when the backend solves its class and takes its features.
    ///
    /// # Errors
    ///
    /// [`LoadError::UnsupportedClass`] when [`Backend::CLASSES`] lacks the program's [`class`](Program::class), else
    /// [`LoadError::UnsupportedFeature`] when [`Backend::FEATURES`] lacks one of its [`features`](Program::features),
    /// both before the backend sees it; [`LoadError::Backend`] when the backend refuses the program.
    pub fn new(mut backend: B, program: Program) -> Result<Self, LoadError<B::Error>> {
        lack::<B>(program.class(), program.features())?;
        backend.load(&program).map_err(LoadError::Backend)?;
        Ok(Self {
            backend,
            program,
            ended: None,
            synced: true,
        })
    }

    /// The loaded program.
    #[must_use]
    pub const fn program(&self) -> &Program {
        &self.program
    }

    /// The backend, to read its own state.
    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// The backend and the program, taken apart. The backend holds the program unless an edit failed and so did the
    /// reload after it, a panic interrupted them, or the last solve priced columns.
    #[must_use]
    pub fn into_parts(self) -> (B, Program) {
        (self.backend, self.program)
    }

    /// Solves under the backend's defaults, with no stop, observer, callback, start or handler: [`Run::solve`] of
    /// [`Solver::run`] with nothing set.
    ///
    /// # Errors
    ///
    /// As [`Run::solve`].
    pub fn solve(&mut self) -> Result<Solved<B::Status>, SolveError<B::Error>> {
        self.run().solve()
    }

    /// One solve to configure: its settings, stop, observer, callback, and the start and handlers the backend
    /// takes, ended by [`Run::solve`].
    ///
    /// ```
    /// use core::time::Duration;
    /// use culmen_core::{Backend, Callback, Model, Outcome, Phase, Progress, Solver, Var};
    /// use culmen_core::script::{Script, Step};
    ///
    /// /// Counts the polls a solve delivers.
    /// #[derive(Default)]
    /// struct Count(usize);
    ///
    /// impl<B: Backend> Callback<B> for Count {
    ///     fn poll(&mut self, _: &mut B::Poll<'_>) {
    ///         self.0 += 1;
    ///     }
    /// }
    ///
    /// let mut m = Model::new();
    /// m.add_var(Var::binary());
    /// let poll = Step::Poll(Progress::at(Duration::from_secs(1), Phase::Mip));
    /// let mut solver = Solver::new(Script::new([poll.clone(), poll], Outcome::Infeasible), m.build()?)?;
    /// let mut count = Count::default();
    /// solver.run().time_limit(Duration::from_secs(10)).callback(&mut count).solve()?;
    /// assert_eq!(count.0, 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn run(&mut self) -> Run<'_, '_, B> {
        Run::new(self)
    }

    /// Syncs the backend, hands it `start` when given, and solves under `settings` with `watch` run before `hooks`.
    ///
    /// # Errors
    ///
    /// As [`Run::solve`].
    ///
    /// # Panics
    ///
    /// When the observer, the callback or a handler panics: the panic continues once the solver has stopped.
    fn execute(
        &mut self,
        settings: &Settings,
        watch: &mut Watch<'_>,
        start: Option<&(Start, HandStart<B>)>,
        hooks: Hooks<'_, B>,
    ) -> Result<Solved<B::Status>, SolveError<B::Error>> {
        self.sync().map_err(SolveError::Backend)?;
        if let Some((start, hand)) = start {
            hand(self, start)?;
        }
        // The backend's result is replaced from here on, so the last solve is read no longer even when the backend
        // fails or a panic unwinds out of it.
        self.ended = None;
        let mut grow = Grow {
            priced: Priced::new(&self.program, B::CLASSES),
            program: &mut self.program,
            synced: &mut self.synced,
        };
        let solved = self.backend.solve(settings, hooks.install(watch, &mut grow.priced));
        watch.finish();
        // The analyses would read the backend's answer for the program before the priced columns.
        if grow.priced.cols().is_empty() {
            self.ended = solved.as_ref().ok().map(|s| Ended::of(&s.outcome));
        }
        drop(grow);
        solved.map_err(SolveError::Backend)
    }

    /// Loads the program into the backend again unless the backend holds it.
    ///
    /// # Errors
    ///
    /// The backend's error when it refuses the program.
    fn sync(&mut self) -> Result<(), B::Error> {
        if !self.synced {
            self.backend.load(&self.program)?;
            self.synced = true;
        }
        Ok(())
    }
}

impl<B: Configurable> Solver<B> {
    /// The backend's own options, which every solve applies before its settings.
    #[must_use]
    pub fn options(&self) -> &B::Options {
        self.backend.options()
    }

    /// The backend's own options, to change between solves. The backend is reachable mutably only through them, so
    /// the program it holds stays the solver's.
    pub fn options_mut(&mut self) -> &mut B::Options {
        self.backend.options_mut()
    }
}

impl<B: Starts> Solver<B> {
    /// Checks a start against the program, and hands it to the backend.
    ///
    /// # Errors
    ///
    /// As [`Run::solve`] states for a start.
    fn hand_start(&mut self, start: &Start) -> Result<(), SolveError<B::Error>> {
        start.check(self.program.num_cols()).map_err(SolveError::Start)?;
        self.ended = None;
        self.backend.set_start(start).map_err(SolveError::Backend)
    }
}
