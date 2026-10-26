// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Solving: the backend seam, the generic callbacks, the observer convenience, and the solver value.

mod backend;
pub mod context;
mod events;
mod load;
mod outcome;
mod settings;
mod watch;

pub use backend::{Backend, Callback};
pub use context::{
    ActionError, AddedRow, CutContext, HeuristicContext, LazyContext, LogContext, PollContext, ProgressContext,
    SolutionContext, Unsupported,
};
pub use events::{Events, EventsIter, LogLevel, Phase, Progress};
pub use load::LoadError;
pub use outcome::{Outcome, Solution, Solved, StopReason};
pub use settings::{AbsoluteGap, InvalidAbsoluteGap, InvalidRelativeGap, RelativeGap, Settings};
pub use watch::{Improvement, LogLine, Observer, Request, Stop, StopFn, Watch, stop_fn};

use crate::Lp;

/// An LP loaded into a backend, solved as often as needed, each solve with its own [`Request`].
///
/// ```
/// use culmen_core::{Model, Outcome, Request, Script, Solver, Var};
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::binary());
/// m.maximize(x);
/// let lp = m.build()?;
///
/// // The scripted backend stands in for a solver here; with the `highs` feature of `culmen`, `Highs::new()`.
/// let mut solver = Solver::new(Script::optimal(1.0, vec![1.0]), lp)?;
/// let solved = solver.solve(Request::new())?;
/// let Outcome::Optimal(solution) = solved.outcome else { panic!("expected an optimum") };
/// assert_eq!(solution.values[x], 1.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Solver<B> {
    backend: B,
    lp: Lp,
}

impl<B: Backend> Solver<B> {
    /// Loads `lp` into `backend`, when the backend solves its class.
    ///
    /// # Errors
    ///
    /// [`LoadError::Unsupported`] when [`Backend::CLASSES`] lacks the LP's [`class`](Lp::class), before the
    /// backend sees it; [`LoadError::Backend`] when the backend refuses the LP.
    pub fn new(mut backend: B, lp: Lp) -> Result<Self, LoadError<B::Error>> {
        let class = lp.class();
        if !B::CLASSES.contains(class) {
            return Err(LoadError::Unsupported {
                class,
                supported: B::CLASSES,
            });
        }
        backend.load(&lp).map_err(LoadError::Backend)?;
        Ok(Self { backend, lp })
    }

    /// The loaded LP.
    #[must_use]
    pub const fn lp(&self) -> &Lp {
        &self.lp
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

    /// The backend and the LP, taken apart.
    #[must_use]
    pub fn into_parts(self) -> (B, Lp) {
        (self.backend, self.lp)
    }

    /// Solves under the request's settings, polling its stop and reporting to its observer.
    ///
    /// # Errors
    ///
    /// The backend's error when a setting cannot be applied or the solver fails.
    ///
    /// # Panics
    ///
    /// When the observer panics: the panic continues once the solver has stopped.
    pub fn solve(&mut self, request: Request<'_>) -> Result<Solved<B::Status>, B::Error> {
        self.solve_with(request, &mut ())
    }

    /// Solves as [`Solver::solve`] does, and also runs `callback` with the backend's full context types.
    ///
    /// # Errors
    ///
    /// The backend's error when a setting cannot be applied or the solver fails.
    ///
    /// # Panics
    ///
    /// When the observer or `callback` panics: the panic continues once the solver has stopped.
    pub fn solve_with(
        &mut self,
        request: Request<'_>,
        callback: &mut impl Callback<B>,
    ) -> Result<Solved<B::Status>, B::Error> {
        let (watch, settings) = request.into_watch();
        let mut both = (watch, callback);
        let solved = self.backend.solve(&settings, &mut both);
        both.0.finish();
        solved
    }
}
