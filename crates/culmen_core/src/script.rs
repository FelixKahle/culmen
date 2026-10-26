// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An in-memory backend that replays a script of events: for testing code that drives a solve, without a solver.

use core::fmt;
use core::time::Duration;

use typed_index_collections::{TiSlice, TiVec};

use crate::solve::context::{check_constraint, check_entries, check_solution};
use crate::{
    ActionError, AddedRow, Backend, Callback, Col, Constraint, CutContext, Events, HeuristicContext, LazyContext,
    LogContext, LogLevel, Lp, ModelClasses, Outcome, Phase, PollContext, Progress, ProgressContext, Settings, Solution,
    SolutionContext, Solved, StopReason,
};

/// A solution a script step reports.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    /// Where the solve stands.
    pub progress: Progress,
    /// The objective value, offset included.
    pub objective: f64,
    /// One value per column.
    pub values: TiVec<Col, f64>,
}

/// One event a [`Script`] delivers, in the generic terms of [`Callback`].
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// A log chunk.
    Log {
        /// When it is written.
        elapsed: Duration,
        /// How much it matters.
        level: LogLevel,
        /// Its text, line breaks included.
        text: String,
    },
    /// A progress report.
    Progress(Progress),
    /// A poll at which the solve may be interrupted.
    Poll(Progress),
    /// A feasible solution.
    Feasible(Found),
    /// A new incumbent.
    Incumbent(Found),
    /// A candidate solution offered for lazy constraints.
    Lazy(Found),
    /// A node relaxation offered for cuts.
    Cut {
        /// Where the solve stands.
        progress: Progress,
        /// One value per column.
        relaxation: TiVec<Col, f64>,
    },
    /// A chance to supply a heuristic solution.
    Heuristic(Progress),
    /// A note only this backend knows: its native event.
    Note(String),
}

impl Step {
    /// The kind of event the step is delivered as.
    ///
    /// ```
    /// use culmen_core::{Events, Step};
    /// assert_eq!(Step::Note("presolve done".to_owned()).event(), Events::NATIVE);
    /// ```
    #[must_use]
    pub const fn event(&self) -> Events {
        match self {
            Self::Log { .. } => Events::LOG,
            Self::Progress(_) => Events::PROGRESS,
            Self::Poll(_) => Events::POLL,
            Self::Feasible(_) => Events::FEASIBLE,
            Self::Incumbent(_) => Events::INCUMBENT,
            Self::Lazy(_) => Events::LAZY,
            Self::Cut { .. } => Events::CUT,
            Self::Heuristic(_) => Events::HEURISTIC,
            Self::Note(_) => Events::NATIVE,
        }
    }
}

/// What callbacks did during a [`Script`]'s solves.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Record {
    /// The settings of each solve, in order.
    pub settings: Vec<Settings>,
    /// The lazy constraints added, as checked.
    pub lazy: Vec<AddedRow>,
    /// The cuts added, as checked.
    pub cuts: Vec<AddedRow>,
    /// The heuristic solutions supplied, as entries.
    pub supplied: Vec<Vec<(Col, f64)>>,
}

/// A backend that delivers a fixed list of [`Step`]s to the callback and then ends with a fixed [`Outcome`].
///
/// It supports every generic event and loads every [`ModelClass`](crate::ModelClass), since it replays rather
/// than solves, so it exercises the whole callback layer without a solver, and it runs under Miri. A poll that interrupts ends the solve at once as [`StopReason::Interrupted`], with the last incumbent
/// and the last bound reported before. The outcome and the elapsed time count every step, whichever events the
/// callback asks for.
///
/// ```
/// use core::time::Duration;
/// use culmen_core::{Model, Outcome, Phase, Progress, Request, Script, Solver, Step, StopReason, Var};
/// use std::sync::atomic::AtomicBool;
///
/// let mut m = Model::new();
/// m.add_var(Var::binary());
/// let poll = Step::Poll(Progress::at(Duration::from_secs(1), Phase::Mip));
/// let script = Script::new([poll], Outcome::Infeasible);
/// let mut solver = Solver::new(script, m.build()?)?;
///
/// let stop = AtomicBool::new(true);
/// let solved = solver.solve(Request::new().stop(&stop))?;
/// assert_eq!(solved.outcome, Outcome::Stopped { reason: StopReason::Interrupted, best: None, bound: None });
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Script {
    steps: Vec<Step>,
    end: Outcome,
    num_cols: Option<usize>,
    /// What callbacks did, across every solve.
    pub record: Record,
}

impl Script {
    /// A script of steps ending with `end`.
    #[must_use]
    pub fn new(steps: impl Into<Vec<Step>>, end: Outcome) -> Self {
        Self {
            steps: steps.into(),
            end,
            num_cols: None,
            record: Record {
                settings: Vec::new(),
                lazy: Vec::new(),
                cuts: Vec::new(),
                supplied: Vec::new(),
            },
        }
    }

    /// A script without steps that ends optimal with the given objective and values.
    #[must_use]
    pub fn optimal(objective: f64, values: impl Into<TiVec<Col, f64>>) -> Self {
        Self::new(
            [],
            Outcome::Optimal(Solution {
                objective,
                values: values.into(),
            }),
        )
    }

    /// Delivers one step: the solve's state sees it whatever the callback asks for, and the callback gets it only
    /// when it asks for its kind.
    ///
    /// # Panics
    ///
    /// When the callback panics.
    fn deliver(&mut self, index: usize, callback: &mut dyn Callback<Self>, run: &mut Run) {
        let step = &self.steps[index];
        run.observe(step);
        if callback.events().contains(step.event()) {
            let num_cols = self.num_cols.unwrap_or(0);
            dispatch(step, callback, run, &mut self.record, num_cols);
        }
    }
}

/// Calls the callback's method for a step, with the step's context.
///
/// # Panics
///
/// When the callback panics.
fn dispatch(step: &Step, callback: &mut dyn Callback<Script>, run: &mut Run, record: &mut Record, num_cols: usize) {
    match step {
        Step::Log { elapsed, level, text } => callback.log(&mut ScriptLog {
            text,
            level: *level,
            elapsed: *elapsed,
        }),
        Step::Progress(progress) => callback.progress(&mut ScriptProgress { progress: *progress }),
        Step::Poll(progress) => callback.poll(&mut ScriptPoll {
            progress: *progress,
            interrupted: &mut run.interrupted,
        }),
        Step::Feasible(found) => callback.feasible(&mut ScriptSolution { found }),
        Step::Incumbent(found) => callback.incumbent(&mut ScriptSolution { found }),
        Step::Lazy(found) => callback.lazy(&mut ScriptLazy {
            found,
            num_cols,
            added: &mut record.lazy,
        }),
        Step::Cut { progress, relaxation } => callback.cut(&mut ScriptCut {
            progress: *progress,
            relaxation,
            num_cols,
            added: &mut record.cuts,
        }),
        Step::Heuristic(progress) => callback.heuristic(&mut ScriptHeuristic {
            progress: *progress,
            num_cols,
            supplied: &mut record.supplied,
        }),
        Step::Note(text) => callback.native(&mut ScriptNote { text }),
    }
}

/// The state of one solve.
#[derive(Default)]
struct Run {
    interrupted: bool,
    best: Option<Solution>,
    bound: Option<f64>,
    elapsed: Duration,
}

impl Run {
    /// Takes in a step's time, bound and incumbent.
    fn observe(&mut self, step: &Step) {
        match step {
            Step::Log { elapsed, .. } => self.elapsed = self.elapsed.max(*elapsed),
            Step::Progress(progress)
            | Step::Poll(progress)
            | Step::Cut { progress, .. }
            | Step::Heuristic(progress) => self.advance(progress),
            Step::Feasible(found) | Step::Lazy(found) => self.advance(&found.progress),
            Step::Incumbent(found) => {
                self.advance(&found.progress);
                self.best = Some(Solution {
                    objective: found.objective,
                    values: found.values.clone(),
                });
            }
            Step::Note(_) => {}
        }
    }

    fn advance(&mut self, progress: &Progress) {
        self.bound = progress.bound.or(self.bound);
        self.elapsed = self.elapsed.max(progress.elapsed);
    }
}

/// How a [`Script`] solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScriptStatus {
    /// Every step ran and the scripted outcome was returned.
    Finished,
    /// A poll interrupted the solve.
    Interrupted {
        /// The position of the interrupting step.
        step: usize,
    },
}

/// Why a [`Script`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScriptError {
    /// A solve was asked for before any LP was loaded.
    NotLoaded,
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLoaded => write!(f, "Script: no LP is loaded"),
        }
    }
}

impl core::error::Error for ScriptError {}

impl Backend for Script {
    type Error = ScriptError;
    type Status = ScriptStatus;
    type Log<'a> = ScriptLog<'a>;
    type Progress<'a> = ScriptProgress;
    type Poll<'a> = ScriptPoll<'a>;
    type Solution<'a> = ScriptSolution<'a>;
    type Lazy<'a> = ScriptLazy<'a>;
    type Cut<'a> = ScriptCut<'a>;
    type Heuristic<'a> = ScriptHeuristic<'a>;
    type Native<'a> = ScriptNote<'a>;

    const EVENTS: Events = Events::ALL;
    const CLASSES: ModelClasses = ModelClasses::ALL;

    fn load(&mut self, lp: &Lp) -> Result<(), ScriptError> {
        self.num_cols = Some(lp.num_cols());
        Ok(())
    }

    fn solve(
        &mut self,
        settings: &Settings,
        callback: &mut dyn Callback<Self>,
    ) -> Result<Solved<ScriptStatus>, ScriptError> {
        if self.num_cols.is_none() {
            return Err(ScriptError::NotLoaded);
        }
        self.record.settings.push(settings.clone());
        let mut run = Run::default();
        for index in 0..self.steps.len() {
            self.deliver(index, callback, &mut run);
            if run.interrupted {
                let outcome = Outcome::Stopped {
                    reason: StopReason::Interrupted,
                    best: run.best,
                    bound: run.bound,
                };
                return Ok(Solved {
                    outcome,
                    status: ScriptStatus::Interrupted { step: index },
                    elapsed: run.elapsed,
                });
            }
        }
        Ok(Solved {
            outcome: self.end.clone(),
            status: ScriptStatus::Finished,
            elapsed: run.elapsed,
        })
    }
}

/// The log context of a [`Script`].
#[derive(Debug)]
pub struct ScriptLog<'a> {
    text: &'a str,
    level: LogLevel,
    elapsed: Duration,
}

impl LogContext for ScriptLog<'_> {
    fn text(&self) -> &str {
        self.text
    }
    fn level(&self) -> LogLevel {
        self.level
    }
    fn progress(&self) -> Progress {
        Progress::at(self.elapsed, Phase::Other)
    }
}

/// The progress context of a [`Script`].
#[derive(Debug)]
pub struct ScriptProgress {
    progress: Progress,
}

impl ProgressContext for ScriptProgress {
    fn progress(&self) -> Progress {
        self.progress
    }
}

/// The poll context of a [`Script`].
#[derive(Debug)]
pub struct ScriptPoll<'a> {
    progress: Progress,
    interrupted: &'a mut bool,
}

impl PollContext for ScriptPoll<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
    fn interrupt(&mut self) {
        *self.interrupted = true;
    }
}

/// The solution context of a [`Script`].
#[derive(Debug)]
pub struct ScriptSolution<'a> {
    found: &'a Found,
}

impl SolutionContext for ScriptSolution<'_> {
    fn progress(&self) -> Progress {
        self.found.progress
    }
    fn objective(&self) -> f64 {
        self.found.objective
    }
    fn values(&self) -> &TiSlice<Col, f64> {
        &self.found.values
    }
}

/// The lazy-constraint context of a [`Script`]: added constraints are checked and recorded.
#[derive(Debug)]
pub struct ScriptLazy<'a> {
    found: &'a Found,
    num_cols: usize,
    added: &'a mut Vec<AddedRow>,
}

impl LazyContext for ScriptLazy<'_> {
    fn progress(&self) -> Progress {
        self.found.progress
    }
    fn objective(&self) -> f64 {
        self.found.objective
    }
    fn candidate(&self) -> &TiSlice<Col, f64> {
        &self.found.values
    }
    fn add_lazy(&mut self, constraint: Constraint) -> Result<(), ActionError> {
        self.added.push(check_constraint(&constraint, self.num_cols)?);
        Ok(())
    }
}

/// The cut context of a [`Script`]: added cuts are checked and recorded.
#[derive(Debug)]
pub struct ScriptCut<'a> {
    progress: Progress,
    relaxation: &'a TiSlice<Col, f64>,
    num_cols: usize,
    added: &'a mut Vec<AddedRow>,
}

impl CutContext for ScriptCut<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
    fn relaxation(&self) -> &TiSlice<Col, f64> {
        self.relaxation
    }
    fn add_cut(&mut self, cut: Constraint) -> Result<(), ActionError> {
        self.added.push(check_constraint(&cut, self.num_cols)?);
        Ok(())
    }
}

/// The heuristic context of a [`Script`]: supplied solutions are checked and recorded.
#[derive(Debug)]
pub struct ScriptHeuristic<'a> {
    progress: Progress,
    num_cols: usize,
    supplied: &'a mut Vec<Vec<(Col, f64)>>,
}

impl HeuristicContext for ScriptHeuristic<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
    fn supply(&mut self, values: &TiSlice<Col, f64>) -> Result<(), ActionError> {
        check_solution(values, self.num_cols)?;
        self.supplied
            .push(values.iter_enumerated().map(|(col, &v)| (col, v)).collect());
        Ok(())
    }
    fn supply_partial(&mut self, entries: &[(Col, f64)]) -> Result<(), ActionError> {
        check_entries(entries, self.num_cols)?;
        self.supplied.push(entries.to_vec());
        Ok(())
    }
}

/// The native event of a [`Script`]: a note, standing in for what a real solver reports beyond the generic events.
#[derive(Debug)]
pub struct ScriptNote<'a> {
    text: &'a str,
}

impl ScriptNote<'_> {
    /// The note's text.
    #[must_use]
    pub const fn text(&self) -> &str {
        self.text
    }
}
