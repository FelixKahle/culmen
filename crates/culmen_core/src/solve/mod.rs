// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Solving: the backend seam, the generic callbacks, the observer convenience, and the solver value.

mod analysis;
mod backend;
pub mod context;
mod edit;
mod events;
mod load;
mod outcome;
mod settings;
mod start;
mod watch;

pub use analysis::{
    Analyses, AnalysesIter, Analysis, AnalysisError, Basis, BasisStatus, Bound, Duals, Iis, InvalidPenalty, Limit,
    Line, Misfit, Penalty, Ranged, Ranging, Relaxed, Side, Table, Violation, Violations, Weights,
};
pub use backend::{Backend, Callback};
pub use context::{
    ActionError, AddedRow, CutContext, HeuristicContext, LazyContext, LogContext, PollContext, ProgressContext,
    SolutionContext, Unsupported,
};
pub use edit::{Edit, EditError, NewCol, NewRow};
pub use events::{Events, EventsIter, LogLevel, Phase, Progress};
pub use load::LoadError;
pub use outcome::{Outcome, Solution, Solved, StopReason};
pub use settings::Settings;
pub use start::{SolveError, Start, StartEntries, StartError};
pub use watch::{Improvement, LogLine, Observer, Request, Stop, StopFn, Watch, stop_fn};

use crate::{ModelClass, Program, ProgramFeature, ProgramFeatures};
use analysis::Ending;

/// A program loaded into a backend, solved as often as needed, each solve with its own [`Request`].
///
/// ```
/// use culmen_core::{Model, Outcome, Request, Script, Solver, Var};
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::binary());
/// m.maximize(x);
/// let program = m.build()?;
///
/// // The scripted backend stands in for a solver here; with the `highs` feature of `culmen`, `Highs::new()`.
/// let mut solver = Solver::new(Script::optimal(1.0, vec![1.0]), program)?;
/// let solved = solver.solve(Request::new())?;
/// let Outcome::Optimal(solution) = solved.outcome else { panic!("expected an optimum") };
/// assert_eq!(solution.values[x], 1.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Solver<B> {
    backend: B,
    program: Program,
    /// How the last solve ended; `None` before one, after one failed, or once a later call replaced its result.
    ended: Option<Ending>,
    /// Whether the backend holds `program`. An edit clears it while the backend changes; it stays clear when that
    /// change and the reload after it both fail, or a panic interrupts them, until the next call that runs the
    /// backend reloads `program`.
    synced: bool,
}

/// What a backend lacks to hold a program: the program's class, or a feature it uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lack {
    /// The class, which [`Backend::CLASSES`] lacks.
    Class(ModelClass),
    /// The first feature [`Backend::FEATURES`] lacks, in the order of [`ProgramFeature`].
    Feature(ProgramFeature),
}

/// What `B` lacks to hold a program of `class` that uses `features`; `None` when it lacks nothing.
fn lack<B: Backend>(class: ModelClass, features: ProgramFeatures) -> Option<Lack> {
    if B::CLASSES.contains(class) {
        features.difference(B::FEATURES).iter().next().map(Lack::Feature)
    } else {
        Some(Lack::Class(class))
    }
}

impl<B: Backend> Solver<B> {
    /// Loads `program` into `backend`, when the backend solves its class and takes its features.
    ///
    /// # Errors
    ///
    /// [`LoadError::Unsupported`] when [`Backend::CLASSES`] lacks the program's [`class`](Program::class), else
    /// [`LoadError::UnsupportedFeature`] when [`Backend::FEATURES`] lacks one of its [`features`](Program::features),
    /// both before the backend sees it; [`LoadError::Backend`] when the backend refuses the program.
    pub fn new(mut backend: B, program: Program) -> Result<Self, LoadError<B::Error>> {
        match lack::<B>(program.class(), program.features()) {
            Some(Lack::Class(class)) => Err(LoadError::Unsupported {
                class,
                supported: B::CLASSES,
            }),
            Some(Lack::Feature(feature)) => Err(LoadError::UnsupportedFeature {
                feature,
                supported: B::FEATURES,
            }),
            None => {
                backend.load(&program).map_err(LoadError::Backend)?;
                Ok(Self {
                    backend,
                    program,
                    ended: None,
                    synced: true,
                })
            }
        }
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

    /// The backend, to change its own options between solves.
    pub const fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// The backend and the program, taken apart. The backend holds the program unless an edit failed and so did the
    /// reload after it, or a panic interrupted them.
    #[must_use]
    pub fn into_parts(self) -> (B, Program) {
        (self.backend, self.program)
    }

    /// Solves under the request's settings, from its start, polling its stop and reporting to its observer.
    ///
    /// # Errors
    ///
    /// [`SolveError::Backend`] when the backend refuses the program it reloads after an edit whose reload failed;
    /// [`SolveError::UnsupportedFeature`] for a start when [`Backend::FEATURES`] lacks
    /// [`ProgramFeature::StartSolution`], else [`SolveError::Start`] for a start [`Start::check`] refuses, both before
    /// the backend sees the request; [`SolveError::Backend`] when the backend refuses the start, a setting cannot be
    /// applied or the solver fails.
    ///
    /// # Panics
    ///
    /// When the observer panics: the panic continues once the solver has stopped.
    pub fn solve(&mut self, request: Request<'_>) -> Result<Solved<B::Status>, SolveError<B::Error>> {
        self.solve_with(request, &mut ())
    }

    /// Solves as [`Solver::solve`] does, and also runs `callback` with the backend's full context types.
    ///
    /// # Errors
    ///
    /// As [`Solver::solve`].
    ///
    /// # Panics
    ///
    /// When the observer or `callback` panics: the panic continues once the solver has stopped.
    pub fn solve_with(
        &mut self,
        request: Request<'_>,
        callback: &mut impl Callback<B>,
    ) -> Result<Solved<B::Status>, SolveError<B::Error>> {
        let (watch, settings, start) = request.into_watch();
        self.sync().map_err(SolveError::Backend)?;
        if let Some(start) = &start {
            self.hand_start(start)?;
        }
        let mut both = (watch, callback);
        let solved = self.backend.solve(&settings, &mut both);
        both.0.finish();
        self.ended = solved.as_ref().ok().map(|s| Ending::of(&s.outcome));
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

    /// Checks a start against the backend's features and the program, and hands it to the backend.
    ///
    /// # Errors
    ///
    /// As [`Solver::solve`] states for a start.
    fn hand_start(&mut self, start: &Start) -> Result<(), SolveError<B::Error>> {
        if !B::FEATURES.contains(ProgramFeature::StartSolution) {
            return Err(SolveError::UnsupportedFeature {
                feature: ProgramFeature::StartSolution,
                supported: B::FEATURES,
            });
        }
        start.check(self.program.num_cols()).map_err(SolveError::Start)?;
        self.backend.set_start(start).map_err(SolveError::Backend)
    }
}
