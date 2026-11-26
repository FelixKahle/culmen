// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An in-memory backend that replays a script of events: for testing code that drives a solve, without a solver.

use core::fmt;
use core::time::Duration;

use typed_index_collections::{TiSlice, TiVec};

use crate::solve::context::{check_constraint, check_entries, check_solution};
use crate::{
    ActionError, AddedRow, Backend, Bound, Bounds, CheckContext, Child, Col, Cols, Constraint, Context,
    CustomConstraints, CutContext, Declaration, EnforceContext, Events, ExplainContext, Explained, HeuristicContext,
    Heuristics, Hooks, Inferred, Involved, LazyConstraints, LazyContext, Locks, LogContext, LogLevel, ModelClasses,
    NewCol, NodeActions, NodeContext, Outcome, Phase, PollContext, PresolveContext, PriceContext, PriceMode, Priced,
    Prices, Program, ProgramFeatures, Progress, ProgressContext, PropagateContext, Reason, Row, Sense, Settings,
    Solution, SolutionContext, Solved, Start, Starts, StopReason, UserCuts, Verdict,
};

/// A solution found during a solve, with where the solve stood when it was found: what a [`Script`] step reports, and
/// what a backend may keep of a solution it delivers to a callback later.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    /// Where the solve stands.
    pub progress: Progress,
    /// The objective value, offset included.
    pub objective: f64,
    /// One value per column.
    pub values: TiVec<Col, f64>,
}

/// One event a [`Script`] delivers, in the generic terms of [`Callback`](crate::Callback) and the search handlers.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
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
    /// A candidate solution offered to the lazy-constraint handler.
    Lazy(Found),
    /// A node relaxation offered to the cut handler.
    Cut {
        /// Where the solve stands.
        progress: Progress,
        /// One value per column.
        relaxation: TiVec<Col, f64>,
    },
    /// A chance for the heuristic handler to supply a solution.
    Heuristic(Progress),
    /// A node LP offered to the pricer.
    Price {
        /// Where the solve stands.
        progress: Progress,
        /// Whether the node LP is optimal, with its objective, or infeasible.
        mode: PriceMode,
        /// The multiplier of every program row, as [`PriceContext::duals`] states it.
        duals: TiVec<Row, f64>,
    },
    /// A point offered to the constraint handler's check.
    Check(Found),
    /// A point at a node offered to the constraint handler: checked first, and enforced when the check rejects it.
    Enforce {
        /// The point.
        found: Found,
        /// Every column's bounds at the node.
        bounds: TiVec<Col, Bounds>,
    },
    /// A node's domain offered to the constraint handler's propagation.
    Propagate {
        /// Where the solve stands.
        progress: Progress,
        /// Every column's bounds at the node.
        bounds: TiVec<Col, Bounds>,
    },
    /// A tightening the constraint handler gave a reason, offered to its explanation.
    Explain {
        /// Where the solve stands.
        progress: Progress,
        /// The bound to explain.
        inferred: Inferred,
        /// The handler's tag on the tightening.
        reason: Reason,
        /// Every column's bounds just before the tightening.
        before: TiVec<Col, Bounds>,
    },
    /// The program's global domain offered to the constraint handler's presolve.
    Presolve {
        /// Where the solve stands.
        progress: Progress,
        /// Every column's global bounds.
        bounds: TiVec<Col, Bounds>,
    },
    /// A note only this backend knows: its native event.
    Note(String),
}

impl Step {
    /// The kind of observational event the step is delivered as; `None` for a search step ([`Step::Lazy`],
    /// [`Step::Cut`], [`Step::Heuristic`], [`Step::Price`], and the constraint handler's [`Step::Check`],
    /// [`Step::Enforce`], [`Step::Propagate`], [`Step::Explain`], [`Step::Presolve`]), which is delivered to the
    /// handler of its kind whenever one is registered.
    ///
    /// ```
    /// use culmen_core::{Events, Step};
    /// assert_eq!(Step::Note("presolve done".to_owned()).event(), Some(Events::NATIVE));
    /// ```
    #[must_use]
    pub const fn event(&self) -> Option<Events> {
        Some(match self {
            Self::Log { .. } => Events::LOG,
            Self::Progress(_) => Events::PROGRESS,
            Self::Poll(_) => Events::POLL,
            Self::Feasible(_) => Events::FEASIBLE,
            Self::Incumbent(_) => Events::INCUMBENT,
            Self::Lazy(_)
            | Self::Cut { .. }
            | Self::Heuristic(_)
            | Self::Price { .. }
            | Self::Check(_)
            | Self::Enforce { .. }
            | Self::Propagate { .. }
            | Self::Explain { .. }
            | Self::Presolve { .. } => return None,
            Self::Note(_) => Events::NATIVE,
        })
    }
}

/// What callbacks and handlers did during a [`Script`]'s solves.
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
    /// The starts handed over, as checked, one per solve that had one.
    pub starts: Vec<Start>,
    /// One round per price step whose pricer ran and returned, in order; the columns themselves are the program's.
    pub priced: Vec<PriceRound>,
    /// The constraint handler's declaration, read once in each solve that had a handler.
    pub declarations: Vec<Declaration>,
    /// The constraint handler's verdict on the point of each check and enforce step, in order.
    pub verdicts: Vec<Verdict>,
    /// The actions of each enforcement, one per enforce step whose point the check rejected, in order.
    pub enforced: Vec<NodeActions>,
    /// The actions of each propagation, in order.
    pub propagated: Vec<NodeActions>,
    /// Each explanation's answer and the bounds it named, as named, in order.
    pub explained: Vec<(Explained, Vec<(Col, Bound)>)>,
    /// What the constraint handler did at each presolve step, in order.
    pub presolved: Vec<PresolveRound>,
}

/// What a constraint handler did at one [`Step::Presolve`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PresolveRound {
    /// The global bounds tightened, as given, in order.
    pub tightenings: Vec<(Col, Bounds)>,
    /// The constraints added, as checked, in order.
    pub constraints: Vec<AddedRow>,
    /// Whether it declared the program infeasible.
    pub infeasible: bool,
}

/// What a pricer did at one [`Step::Price`].
#[derive(Clone, Debug, PartialEq)]
pub struct PriceRound {
    /// The columns it added, in order; the program holds them after the solve.
    pub added: Cols,
    /// The last bound it reported; `None` when it reported none.
    pub bound: Option<f64>,
    /// Whether it ended pricing early.
    pub early: bool,
}

/// A backend that delivers a fixed list of [`Step`]s to the callback and the handlers and then ends with a fixed
/// [`Outcome`].
///
/// It supports every generic event and has every search capability ([`LazyConstraints`], [`UserCuts`],
/// [`Heuristics`], [`Prices`], [`CustomConstraints`]), loads every [`ModelClass`](crate::ModelClass) and takes every
/// [`ProgramFeature`](crate::ProgramFeature) and start, recording each start in [`Record::starts`], since it replays
/// rather than solves, so it exercises the whole callback layer without a solver, and it runs under Miri. A poll that
/// interrupts ends the solve at once as [`StopReason::Interrupted`], with the last incumbent and the last bound
/// reported before. The outcome and the elapsed time count every step, whichever events the callback asks for and
/// whichever handlers are registered. A price step's columns are checked and numbered by the solve's [`Priced`], and
/// every later step counts them; the scripted points and outcome are taken as given, so they hold one value per
/// column the solve would have.
///
/// ```
/// use core::time::Duration;
/// use culmen_core::{Model, Outcome, Phase, Progress, Script, Solver, Step, StopReason, Var};
/// use std::sync::atomic::AtomicBool;
///
/// let mut m = Model::new();
/// m.add_var(Var::binary());
/// let poll = Step::Poll(Progress::at(Duration::from_secs(1), Phase::Mip));
/// let script = Script::new([poll], Outcome::Infeasible);
/// let mut solver = Solver::new(script, m.build()?)?;
///
/// let stop = AtomicBool::new(true);
/// let solved = solver.run().stop(&stop).solve()?;
/// assert_eq!(solved.outcome, Outcome::Stopped { reason: StopReason::Interrupted, best: None, bound: None });
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct Script {
    steps: Vec<Step>,
    end: Outcome,
    /// The loaded program's sense; `None` before a load.
    sense: Option<Sense>,
    /// The number of columns of the loaded program.
    num_cols: usize,
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
            sense: None,
            num_cols: 0,
            record: Record::default(),
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

    /// Delivers one step: the solve's state sees it whatever the hooks ask for, and the hooks get it only when they
    /// ask for its kind, or, for a search step, when a handler of its kind is registered.
    ///
    /// # Panics
    ///
    /// When a callback or handler panics.
    fn deliver(&mut self, index: usize, hooks: &mut Hooks<'_, Self>, run: &mut Run, priced: Option<&mut Priced>) {
        let step = &self.steps[index];
        run.observe(step);
        let wanted = match step {
            Step::Lazy(_) => hooks.wants_lazy(),
            Step::Cut { .. } => hooks.wants_cut(),
            Step::Heuristic(_) => hooks.wants_heuristic(),
            Step::Price { .. } => priced.is_some(),
            Step::Check(_)
            | Step::Enforce { .. }
            | Step::Propagate { .. }
            | Step::Explain { .. }
            | Step::Presolve { .. } => hooks.wants_constraint(),
            _ => step.event().is_some_and(|event| hooks.events().contains(event)),
        };
        if wanted {
            let sense = self.sense.unwrap_or_default();
            let num_cols = priced.as_deref().map_or(self.num_cols, Priced::num_cols);
            let state = State {
                run,
                record: &mut self.record,
                num_cols,
                sense,
            };
            dispatch(step, hooks, state, priced);
        }
    }
}

/// What a step's context reaches beyond the step.
struct State<'a> {
    run: &'a mut Run,
    record: &'a mut Record,
    /// The number of columns the problem holds, the columns priced so far included.
    num_cols: usize,
    sense: Sense,
}

/// Hands a step to the hooks, with the step's context.
///
/// # Panics
///
/// When a callback or handler panics, and for a price step without `priced`.
fn dispatch(step: &Step, hooks: &mut Hooks<'_, Script>, state: State<'_>, priced: Option<&mut Priced>) {
    let State {
        run,
        record,
        num_cols,
        sense,
    } = state;
    match step {
        Step::Log { elapsed, level, text } => hooks.log(&mut ScriptLog {
            text,
            level: *level,
            elapsed: *elapsed,
        }),
        Step::Progress(progress) => hooks.progress(&mut ScriptProgress { progress: *progress }),
        Step::Poll(progress) => hooks.poll(&mut ScriptPoll {
            progress: *progress,
            interrupted: &mut run.interrupted,
        }),
        Step::Feasible(found) => hooks.feasible(&mut ScriptSolution { found }),
        Step::Incumbent(found) => hooks.incumbent(&mut ScriptSolution { found }),
        Step::Lazy(found) => hooks.lazy(&mut ScriptLazy {
            found,
            num_cols,
            added: &mut record.lazy,
        }),
        Step::Cut { progress, relaxation } => hooks.cut(&mut ScriptCut {
            progress: *progress,
            relaxation,
            num_cols,
            added: &mut record.cuts,
        }),
        Step::Heuristic(progress) => hooks.heuristic(&mut ScriptHeuristic {
            progress: *progress,
            num_cols,
            supplied: &mut record.supplied,
        }),
        Step::Price { progress, mode, duals } => {
            let priced = priced.expect("Script: a price step is delivered only with the solve's sink");
            let mut ctx = ScriptPrice::new(*progress, *mode, duals, sense, priced);
            hooks.price(&mut ctx);
            record.priced.push(ctx.round);
        }
        Step::Note(text) => hooks.native(&mut ScriptNote { text }),
        Step::Check(_)
        | Step::Enforce { .. }
        | Step::Propagate { .. }
        | Step::Explain { .. }
        | Step::Presolve { .. } => {
            dispatch_constraint(step, hooks, &run.involved, record, num_cols);
        }
    }
}

/// Hands a constraint handler's step to the hooks, with the step's context, and records what the handler did.
///
/// # Panics
///
/// When the handler panics.
fn dispatch_constraint(
    step: &Step,
    hooks: &mut Hooks<'_, Script>,
    involved: &Involved,
    record: &mut Record,
    num_cols: usize,
) {
    match step {
        Step::Check(found) => record.verdicts.push(hooks.check(&mut ScriptCheck { found })),
        Step::Enforce { found, bounds } => {
            let verdict = hooks.check(&mut ScriptCheck { found });
            record.verdicts.push(verdict);
            if verdict == Verdict::Reject {
                let node = ScriptNode::new(found.progress, bounds, involved, num_cols);
                let mut ctx = ScriptEnforce { found, node };
                hooks.enforce(&mut ctx);
                record.enforced.push(ctx.node.actions);
            }
        }
        Step::Propagate { progress, bounds } => {
            let mut ctx = ScriptPropagate(ScriptNode::new(*progress, bounds, involved, num_cols));
            hooks.propagate(&mut ctx);
            record.propagated.push(ctx.0.actions);
        }
        Step::Explain {
            progress,
            inferred,
            reason,
            before,
        } => {
            let mut ctx = ScriptExplain {
                progress: *progress,
                inferred: *inferred,
                reason: *reason,
                before,
                num_cols,
                named: Vec::new(),
            };
            let answer = hooks.explain(&mut ctx);
            record.explained.push((answer, ctx.named));
        }
        Step::Presolve { progress, bounds } => {
            let mut ctx = ScriptPresolve {
                progress: *progress,
                bounds,
                involved,
                num_cols,
                round: PresolveRound::default(),
            };
            hooks.presolve(&mut ctx);
            record.presolved.push(ctx.round);
        }
        _ => {}
    }
}

/// The state of one solve.
struct Run {
    interrupted: bool,
    best: Option<Solution>,
    bound: Option<f64>,
    elapsed: Duration,
    /// The columns the constraint handler's declaration involves; every column without a handler.
    involved: Involved,
}

impl Run {
    /// A solve that has seen no step, with the constraint handler's columns `involved`.
    const fn new(involved: Involved) -> Self {
        Self {
            interrupted: false,
            best: None,
            bound: None,
            elapsed: Duration::ZERO,
            involved,
        }
    }

    /// Takes in a step's time, bound and incumbent.
    fn observe(&mut self, step: &Step) {
        match step {
            Step::Log { elapsed, .. } => self.elapsed = self.elapsed.max(*elapsed),
            Step::Progress(progress)
            | Step::Poll(progress)
            | Step::Cut { progress, .. }
            | Step::Heuristic(progress)
            | Step::Price { progress, .. } => self.advance(progress),
            Step::Feasible(found) | Step::Lazy(found) | Step::Check(found) | Step::Enforce { found, .. } => {
                self.advance(&found.progress)
            }
            Step::Propagate { progress, .. } | Step::Explain { progress, .. } | Step::Presolve { progress, .. } => {
                self.advance(progress)
            }
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
#[non_exhaustive]
pub enum ScriptError {
    /// A solve was asked for before any program was loaded.
    NotLoaded,
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLoaded => write!(f, "Script: no program is loaded"),
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
    type Native<'a> = ScriptNote<'a>;
    type Options = ();

    const EVENTS: Events = Events::ALL;
    const CLASSES: ModelClasses = ModelClasses::ALL;
    const FEATURES: ProgramFeatures = ProgramFeatures::ALL;

    /// A script has no options.
    fn options(&self) -> &() {
        &()
    }

    /// A script has no options.
    fn options_mut(&mut self) -> &mut () {
        // A box of a zero-sized value allocates nothing, so leaking it leaks nothing.
        Box::leak(Box::new(()))
    }

    fn load(&mut self, program: &Program) -> Result<(), ScriptError> {
        self.sense = Some(program.sense());
        self.num_cols = program.num_cols();
        Ok(())
    }

    /// Replays the steps; in a solve with a pricer, the lengths a lazy constraint, a cut or a heuristic solution is
    /// checked against are the sink's, so they count the columns priced before.
    fn solve(&mut self, settings: &Settings, mut hooks: Hooks<'_, Self>) -> Result<Solved<ScriptStatus>, ScriptError> {
        if self.sense.is_none() {
            return Err(ScriptError::NotLoaded);
        }
        let mut priced = hooks.take_priced();
        self.record.settings.push(settings.clone());
        let declaration = hooks.declaration();
        let locks = declaration
            .as_ref()
            .map_or(&Locks::All, |declaration| &declaration.locks);
        let mut run = Run::new(Involved::new(locks, self.num_cols));
        self.record.declarations.extend(declaration);
        for index in 0..self.steps.len() {
            self.deliver(index, &mut hooks, &mut run, priced.as_deref_mut());
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

/// A script hands every candidate of a [`Step::Lazy`] to the handler, and records the constraints it adds in
/// [`Record::lazy`].
impl LazyConstraints for Script {
    type Lazy<'a> = ScriptLazy<'a>;
}

/// A script hands every relaxation of a [`Step::Cut`] to the handler, and records the cuts it adds in
/// [`Record::cuts`].
impl UserCuts for Script {
    type Cut<'a> = ScriptCut<'a>;
}

/// A script asks the handler at every [`Step::Heuristic`], and records the solutions it supplies in
/// [`Record::supplied`].
impl Heuristics for Script {
    type Heuristic<'a> = ScriptHeuristic<'a>;
}

/// A script hands every node LP of a [`Step::Price`] to the pricer, numbers the columns it adds in the solve's sink,
/// and records each round in [`Record::priced`].
impl Prices for Script {
    type Price<'a> = ScriptPrice<'a>;
}

/// A script reads the handler's declaration once per solve into [`Record::declarations`], hands it the point of every
/// [`Step::Check`] and every [`Step::Enforce`], recording each verdict in [`Record::verdicts`], enforces an enforce
/// step's point only when the check rejects it, and hands it every [`Step::Propagate`], [`Step::Explain`] and
/// [`Step::Presolve`], recording what it did in [`Record::enforced`], [`Record::propagated`], [`Record::explained`]
/// and [`Record::presolved`]. Each node's actions are checked by a [`NodeActions`]; it refuses nothing else.
impl CustomConstraints for Script {
    type Check<'a> = ScriptCheck<'a>;
    type Enforce<'a> = ScriptEnforce<'a>;
    type Propagate<'a> = ScriptPropagate<'a>;
    type Explain<'a> = ScriptExplain<'a>;
    type Presolve<'a> = ScriptPresolve<'a>;
}

/// A script records every start in [`Record::starts`]. It implements no analysis trait: it replays events and an
/// outcome, and has no duals, basis or other analysis to report honestly.
impl Starts for Script {
    fn set_start(&mut self, start: &Start) -> Result<(), ScriptError> {
        if self.sense.is_none() {
            return Err(ScriptError::NotLoaded);
        }
        self.record.starts.push(start.clone());
        Ok(())
    }
}

/// The log context of a [`Script`].
#[derive(Debug)]
pub struct ScriptLog<'a> {
    text: &'a str,
    level: LogLevel,
    elapsed: Duration,
}

impl Context for ScriptLog<'_> {
    fn progress(&self) -> Progress {
        Progress::at(self.elapsed, Phase::Other)
    }
}

impl LogContext for ScriptLog<'_> {
    fn text(&self) -> &str {
        self.text
    }
    fn level(&self) -> LogLevel {
        self.level
    }
}

/// The progress context of a [`Script`].
#[derive(Debug)]
pub struct ScriptProgress {
    progress: Progress,
}

impl Context for ScriptProgress {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl ProgressContext for ScriptProgress {}

/// The poll context of a [`Script`].
#[derive(Debug)]
pub struct ScriptPoll<'a> {
    progress: Progress,
    interrupted: &'a mut bool,
}

impl Context for ScriptPoll<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl PollContext for ScriptPoll<'_> {
    fn interrupt(&mut self) {
        *self.interrupted = true;
    }
}

/// The solution context of a [`Script`].
#[derive(Debug)]
pub struct ScriptSolution<'a> {
    found: &'a Found,
}

impl Context for ScriptSolution<'_> {
    fn progress(&self) -> Progress {
        self.found.progress
    }
}

impl SolutionContext for ScriptSolution<'_> {
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

impl Context for ScriptLazy<'_> {
    fn progress(&self) -> Progress {
        self.found.progress
    }
}

impl LazyContext for ScriptLazy<'_> {
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

impl Context for ScriptCut<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl CutContext for ScriptCut<'_> {
    fn relaxation(&self) -> &TiSlice<Col, f64> {
        self.relaxation
    }
    fn add_cut(&mut self, cut: Constraint) -> Result<(), ActionError> {
        self.added.push(check_constraint(&cut, self.num_cols)?);
        Ok(())
    }
}

/// The heuristic context of a [`Script`]: supplied solutions, full or partial, are checked and recorded.
#[derive(Debug)]
pub struct ScriptHeuristic<'a> {
    progress: Progress,
    num_cols: usize,
    supplied: &'a mut Vec<Vec<(Col, f64)>>,
}

impl ScriptHeuristic<'_> {
    /// Offers values for some columns, which the script checks and records as given, standing in for a solver that
    /// completes a partial solution.
    ///
    /// # Errors
    ///
    /// [`ActionError::UnknownColumn`] or [`ActionError::NotFinite`] for an entry at fault.
    pub fn supply_partial(&mut self, entries: &[(Col, f64)]) -> Result<(), ActionError> {
        check_entries(entries, self.num_cols)?;
        self.supplied.push(entries.to_vec());
        Ok(())
    }
}

impl Context for ScriptHeuristic<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl HeuristicContext for ScriptHeuristic<'_> {
    fn supply(&mut self, values: &TiSlice<Col, f64>) -> Result<(), ActionError> {
        check_solution(values, self.num_cols)?;
        self.supplied
            .push(values.iter_enumerated().map(|(col, &v)| (col, v)).collect());
        Ok(())
    }
}

/// The price context of a [`Script`]: columns are checked and numbered by the solve's [`Priced`], and the round is
/// recorded in [`Record::priced`]. A bound reported in [`PriceMode::Farkas`] is refused with [`ActionError::Refused`].
#[derive(Debug)]
pub struct ScriptPrice<'a> {
    progress: Progress,
    mode: PriceMode,
    duals: &'a TiSlice<Row, f64>,
    sense: Sense,
    priced: &'a mut Priced,
    /// The index of the first column this round may add.
    start: usize,
    round: PriceRound,
}

impl<'a> ScriptPrice<'a> {
    /// The tolerance the script's context reports.
    pub const TOLERANCE: f64 = 1e-9;

    /// The context of a price step, whose round adds nothing yet.
    fn new(
        progress: Progress,
        mode: PriceMode,
        duals: &'a TiSlice<Row, f64>,
        sense: Sense,
        priced: &'a mut Priced,
    ) -> Self {
        let start = priced.num_cols();
        Self {
            progress,
            mode,
            duals,
            sense,
            priced,
            start,
            round: PriceRound {
                added: Cols::range(start, start),
                bound: None,
                early: false,
            },
        }
    }
}

impl Context for ScriptPrice<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl PriceContext for ScriptPrice<'_> {
    fn mode(&self) -> PriceMode {
        self.mode
    }
    fn sense(&self) -> Sense {
        self.sense
    }
    fn duals(&self) -> &TiSlice<Row, f64> {
        self.duals
    }
    fn tolerance(&self) -> f64 {
        Self::TOLERANCE
    }
    fn add_col(&mut self, column: NewCol) -> Result<Col, ActionError> {
        let col = self.priced.push(column)?;
        self.round.added = Cols::range(self.start, self.priced.num_cols());
        Ok(col)
    }
    fn branch_early(&mut self) {
        self.round.early = true;
    }
    fn report_bound(&mut self, bound: f64) -> Result<(), ActionError> {
        if matches!(self.mode, PriceMode::Farkas) {
            return Err(ActionError::Refused("Script: a Farkas round takes no bound".to_owned()));
        }
        if !bound.is_finite() {
            return Err(ActionError::NotFiniteBound { bound });
        }
        self.round.bound = Some(bound);
        Ok(())
    }
}

/// The check context of a [`Script`]: the scripted point.
#[derive(Debug)]
pub struct ScriptCheck<'a> {
    found: &'a Found,
}

impl Context for ScriptCheck<'_> {
    fn progress(&self) -> Progress {
        self.found.progress
    }
}

impl CheckContext for ScriptCheck<'_> {
    fn objective(&self) -> f64 {
        self.found.objective
    }
    fn candidate(&self) -> &TiSlice<Col, f64> {
        &self.found.values
    }
}

/// A node of a [`Script`]: the scripted bounds, and the actions taken there.
#[derive(Debug)]
struct ScriptNode<'a> {
    progress: Progress,
    bounds: &'a TiSlice<Col, Bounds>,
    actions: NodeActions,
}

impl<'a> ScriptNode<'a> {
    fn new(progress: Progress, bounds: &'a TiSlice<Col, Bounds>, involved: &Involved, num_cols: usize) -> Self {
        Self {
            progress,
            bounds,
            actions: NodeActions::new(involved, num_cols),
        }
    }

    fn bounds(&self, col: Col) -> Option<Bounds> {
        self.bounds.get(col).copied()
    }
}

/// The enforcement context of a [`Script`]: the scripted point and bounds; the actions are checked by a
/// [`NodeActions`] and recorded in [`Record::enforced`].
#[derive(Debug)]
pub struct ScriptEnforce<'a> {
    found: &'a Found,
    node: ScriptNode<'a>,
}

impl Context for ScriptEnforce<'_> {
    fn progress(&self) -> Progress {
        self.node.progress
    }
}

impl CheckContext for ScriptEnforce<'_> {
    fn objective(&self) -> f64 {
        self.found.objective
    }
    fn candidate(&self) -> &TiSlice<Col, f64> {
        &self.found.values
    }
}

impl NodeContext for ScriptEnforce<'_> {
    fn bounds(&self, col: Col) -> Option<Bounds> {
        self.node.bounds(col)
    }
    fn tighten(&mut self, col: Col, bounds: Bounds, reason: Option<Reason>) -> Result<(), ActionError> {
        self.node.actions.tighten(col, bounds, reason)
    }
    fn cut_off(&mut self) {
        self.node.actions.cut_off();
    }
    fn cut_off_because(&mut self, reasons: &[(Col, Bound)]) -> Result<(), ActionError> {
        self.node.actions.cut_off_because(reasons)
    }
}

impl EnforceContext for ScriptEnforce<'_> {
    fn add_constraint(&mut self, constraint: Constraint) -> Result<(), ActionError> {
        self.node.actions.add_constraint(&constraint)
    }
    fn branch(&mut self, children: Vec<Child>) -> Result<(), ActionError> {
        let bounds = self.node.bounds;
        self.node.actions.branch(children, &|col| bounds.get(col).copied())
    }
}

/// The propagation context of a [`Script`]: the scripted bounds; the actions are checked by a [`NodeActions`] and
/// recorded in [`Record::propagated`].
#[derive(Debug)]
pub struct ScriptPropagate<'a>(ScriptNode<'a>);

impl Context for ScriptPropagate<'_> {
    fn progress(&self) -> Progress {
        self.0.progress
    }
}

impl NodeContext for ScriptPropagate<'_> {
    fn bounds(&self, col: Col) -> Option<Bounds> {
        self.0.bounds(col)
    }
    fn tighten(&mut self, col: Col, bounds: Bounds, reason: Option<Reason>) -> Result<(), ActionError> {
        self.0.actions.tighten(col, bounds, reason)
    }
    fn cut_off(&mut self) {
        self.0.actions.cut_off();
    }
    fn cut_off_because(&mut self, reasons: &[(Col, Bound)]) -> Result<(), ActionError> {
        self.0.actions.cut_off_because(reasons)
    }
}

impl PropagateContext for ScriptPropagate<'_> {}

/// The explanation context of a [`Script`]: the scripted bound, reason and bounds before; the bounds named are
/// recorded in [`Record::explained`] with the answer.
#[derive(Debug)]
pub struct ScriptExplain<'a> {
    progress: Progress,
    inferred: Inferred,
    reason: Reason,
    before: &'a TiSlice<Col, Bounds>,
    num_cols: usize,
    named: Vec<(Col, Bound)>,
}

impl Context for ScriptExplain<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl ExplainContext for ScriptExplain<'_> {
    fn inferred(&self) -> Inferred {
        self.inferred
    }
    fn reason(&self) -> Reason {
        self.reason
    }
    fn bounds_before(&self, col: Col) -> Option<Bounds> {
        self.before.get(col).copied()
    }
    fn because(&mut self, col: Col, end: Bound) -> Result<(), ActionError> {
        if col.index() >= self.num_cols {
            return Err(ActionError::UnknownColumn { col });
        }
        self.named.push((col, end));
        Ok(())
    }
}

/// The presolve context of a [`Script`]: the scripted global bounds; tightenings name only involved columns,
/// constraints are checked, and what the handler did is recorded in [`Record::presolved`].
#[derive(Debug)]
pub struct ScriptPresolve<'a> {
    progress: Progress,
    bounds: &'a TiSlice<Col, Bounds>,
    involved: &'a Involved,
    num_cols: usize,
    round: PresolveRound,
}

impl Context for ScriptPresolve<'_> {
    fn progress(&self) -> Progress {
        self.progress
    }
}

impl PresolveContext for ScriptPresolve<'_> {
    fn bounds(&self, col: Col) -> Option<Bounds> {
        self.bounds.get(col).copied()
    }
    fn tighten(&mut self, col: Col, bounds: Bounds) -> Result<(), ActionError> {
        self.involved.check(col, self.num_cols)?;
        self.round.tightenings.push((col, bounds));
        Ok(())
    }
    fn add_constraint(&mut self, constraint: Constraint) -> Result<(), ActionError> {
        self.round
            .constraints
            .push(check_constraint(&constraint, self.num_cols)?);
        Ok(())
    }
    fn infeasible(&mut self) {
        self.round.infeasible = true;
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
