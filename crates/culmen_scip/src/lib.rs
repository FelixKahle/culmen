// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The SCIP backend of culmen: SCIP 10, built from source through `scip-sys`, with typed statuses and options.
//!
//! [`Scip`] implements [`Backend`]: load a [`Program`], solve it under [`Settings`], and read the generic
//! [`Outcome`] with SCIP's own [`ScipStatus`] beside it. It solves LPs, MILPs, convex QPs and MIQPs, takes a start,
//! and delivers every generic callback event, lazy constraints and user cuts included, through plugins it includes
//! in each instance.
//!
//! ```
//! use culmen_core::{Linear, LinExpr, Model, Outcome, Solver, Var};
//! use culmen_scip::Scip;
//!
//! // A knapsack.
//! let (value, weight) = ([10.0, 13.0, 7.0, 8.0], [5.0, 6.0, 4.0, 3.0]);
//! let mut m = Model::new();
//! let take = m.add_vars(4, |i| Var::binary().name(format!("take[{i}]")));
//! m.add_constraint(LinExpr::sum((0..4).map(|i| weight[i] * take[i])).leq(10.0));
//! m.maximize(LinExpr::sum((0..4).map(|i| value[i] * take[i])));
//!
//! let mut solver = Solver::new(Scip::new(), m.build()?)?;
//! let solved = solver.solve()?;
//! let Outcome::Optimal(best) = solved.outcome else { panic!("expected an optimum, got {:?}", solved.outcome) };
//! assert_eq!(best.objective, 21.0);
//! assert_eq!(best.values[take[1]].round(), 1.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Options and settings
//!
//! Each solve starts from the program as loaded and from the last solve's best solutions: it frees the last solve's
//! transformed problem, resets SCIP's parameters to their defaults, keeps SCIP from catching `SIGINT`
//! (`misc/catchctrlc`), turns off SCIP's reformulation of products of binary variables in nonlinear constraints
//! (`constraints/nonlinear/reformbinprods`; see [`Scip::CLASSES`] for why), then applies the backend's
//! [`ScipOptions`] and then the solve's [`Settings`], so a per-solve setting always wins and an option naming either
//! parameter overrides the backend's choice. SCIP's log stays off standard output unless [`Settings::console`] asks
//! for it.
//!
//! Freeing the transformed problem keeps up to `limits/maxorigsol` (10 by default) of the last solve's best solutions
//! as candidates, which the next solve checks again, against its own lazy constraints too, and may start from, as it
//! would a start; the last solve's parameters decide, and `misc/transsolsorig` set to false keeps none. So a second
//! solve of the same program, under a node or time limit, can end with the first one's optimum. An edit loads the
//! program into a new instance, which holds no solution of before, and so does the [`Solver`](culmen_core::Solver)
//! after a solve that priced a column.
//!
//! Every field of [`Settings`] maps to SCIP:
//!
//! | `Settings` field | SCIP |
//! |---|---|
//! | `time_limit` | `limits/time`, in seconds, from the start of the solve while `misc/resetstat` keeps its default |
//! | `relative_gap` | `limits/gap` |
//! | `absolute_gap` | `limits/absgap` |
//! | `threads` | nothing to set: the solve is sequential, so it uses one thread, within any limit |
//! | `seed` | `randomization/randomseedshift`, at most `i32::MAX` |
//! | `presolve` | `false`: `SCIPsetPresolving` off, after the options' parameters; `true`: the options' presolving emphasis unless it is `Off`, before their parameters, which it leaves as set |
//! | `node_limit` | `limits/nodes`, at most `i64::MAX` |
//! | `solution_limit` | `limits/bestsol`, improving solutions as `Settings` counts them, at most `i32::MAX` |
//! | `console` | SCIP's message handler quiet or not |
//!
//! A solve reports a best solution, an optimum or a reached gap only after SCIP's own check
//! (`SCIPcheckSolOrig`) finds that solution feasible in the original problem, to SCIP's feasibility tolerance; when
//! it does not, as when presolve rewrote a constraint unsoundly, the solve fails with [`ScipError::InfeasibleBest`]
//! instead. That check skips the rows of a solve that prices, which are modifiable, so such a solve also has SCIP
//! check the solution against its transformed problem (`SCIPcheckSol`), whose rows hold the priced columns. There
//! presolve has moved the constant of each column it fixed or shifted into the row's sides, and the relative
//! feasibility tolerance (`numerics/feastol`) scales with the moved side; so a priced solve's point may pass a row of
//! the program by more than the tolerance times that row's own side, up to the tolerance times the side SCIP holds.
//!
//! A value beyond what SCIP's parameter takes is refused with [`ScipError::OutOfRange`] before any parameter is set.
//! SCIP prints the error messages of a call it refuses, such as a parameter it does not know, to standard error
//! through its process-wide error printer, which the backend leaves as it is.
//!
//! # Callbacks
//!
//! [`Scip::EVENTS`] is every observational event but the native one, each from a SCIP plugin, with a context of this
//! crate's, and SCIP has every search capability ([`LazyConstraints`], [`UserCuts`], [`Heuristics`], [`Prices`]):
//!
//! | Event or handler | SCIP plugin | Context |
//! |---|---|---|
//! | `LOG` | the instance's message handler: each info message and warning, a whole line | [`ScipLog`] |
//! | `POLL`, `PROGRESS` | an event handler: each presolve round, focused node, solved node LP and solved node | [`ScipPoll`], [`ScipProgress`] |
//! | `FEASIBLE`, `INCUMBENT` | an event handler: each solution SCIP stores, a new best one as the incumbent too | [`ScipSolution`] |
//! | [`LazyHandler`](culmen_core::LazyHandler) | a constraint handler without constraints, checked and enforced after every other | [`ScipLazy`] |
//! | [`CutHandler`](culmen_core::CutHandler) | a separator on each node's LP solution | [`ScipCut`] |
//! | [`HeuristicHandler`](culmen_core::HeuristicHandler) | a primal heuristic before and after each node | [`ScipHeuristic`] |
//! | [`Pricer`](culmen_core::Pricer) | a variable pricer, delayed, on each optimal or infeasible node LP | [`ScipPrice`] |
//!
//! Every value a context reports is the program's: a candidate, a relaxation and a solution hold one value per
//! column, read through SCIP's original variables whatever presolve did to them, and every objective value and
//! bound is in the program's sense with its offset and quadratic part.
//!
//! What a solve gives up for the search handlers is decided by which handlers its [`Hooks`] hold, never by the
//! events a callback asks for, so a callback that only watches costs SCIP no reduction. A lazy constraint holds for
//! the rest of the solve, so every later candidate satisfies it; a solve with a lazy-constraint handler therefore
//! locks every variable against dual reductions and turns off `misc/usesymmetry`, `misc/allowstrongdualreds` and
//! `misc/allowweakdualreds`, which reason about the constraints SCIP holds rather than those a handler adds later. A
//! solve with a cut or a heuristic handler calls its separator or heuristic at every node, also when an emphasis
//! turned SCIP's own off.
//!
//! Only a solve with a pricer activates SCIP's pricer and marks every program row
//! modifiable, as SCIP requires of a row a pricer adds variables to; any other solve leaves the pricer inactive and
//! no row modifiable, so SCIP gives up nothing for it. A pricing solve also turns off the LP solver's presolving
//! (`lp/presolving`), whose duals of a degenerate node LP need not price out the columns at their bounds, SCIP's
//! aggregation separator (`separating/aggregation/freq`), which cuts with the objective cutoff over the columns held
//! as if no other column could enter, and the LP solver's objective limit (`lp/disablecutoff = 1`), so every node LP
//! handed to the pricer is optimal or infeasible, whatever the options set. SCIP does not price a program with a
//! quadratic objective: such a solve fails with [`ScipError::PricingQuadratic`] before SCIP runs. Every column a
//! pricer adds is a variable of SCIP's transformed problem with no original counterpart, so a heuristic solution
//! of a pricing solve is a transformed one ([`ScipHeuristic`]), and the backend keeps the priced variables until it
//! frees that problem, so the best solution has a value for each.
//!
//! ```
//! use culmen_core::{Col, LazyConstraints, LazyContext, LazyHandler, Linear, Model, Solver, Var};
//! use culmen_scip::Scip;
//!
//! /// Forbids taking both items, a conflict only the handler knows.
//! struct NotBoth(Col, Col);
//!
//! impl<B: LazyConstraints> LazyHandler<B> for NotBoth {
//!     fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
//!         let candidate = ctx.candidate();
//!         if candidate[self.0] + candidate[self.1] > 1.5 {
//!             ctx.add_lazy((self.0 + self.1).leq(1)).expect("a valid constraint");
//!         }
//!     }
//! }
//!
//! let mut m = Model::new();
//! let x = m.add_var(Var::binary());
//! let y = m.add_var(Var::binary());
//! m.maximize(3.0 * x + 2.0 * y);
//! let mut solver = Solver::new(Scip::new(), m.build()?)?;
//! let mut not_both = NotBoth(x, y);
//! let solved = solver.run().lazy(&mut not_both).solve()?;
//! assert_eq!(solved.outcome.objective(), Some(3.0));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A panic in a callback or handler never unwinds into SCIP: the plugin catches it and asks SCIP to stop, and the panic resumes
//! once SCIP has returned. The backend stays usable.
//!
//! # Threads
//!
//! [`Scip`] is `Send`, so it can be moved to a worker thread to solve there. It is not `Sync`: SCIP is built
//! thread-safe in the sense that each instance keeps its state to itself, but one instance promises nothing to two
//! threads calling it at once, not even to read it.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

mod callback;
mod columns;
mod error;
mod instance;
mod options;
mod status;

use core::fmt;
use std::panic::resume_unwind;
use std::time::Instant;

pub use callback::{
    LazySource, ScipCut, ScipEvent, ScipHeuristic, ScipLazy, ScipLog, ScipPoll, ScipPrice, ScipProgress, ScipSolution,
};
use culmen_core::{
    Backend, Events, Heuristics, Hooks, LazyConstraints, ModelClasses, Outcome, Prices, Program, Settings, Solved,
    Start, Starts, Unsupported, UserCuts,
};
pub use error::{Number, RetCode, ScipError};
pub use options::{Emphasis, ParamValue, ScipOptions, ScipParam};
pub use status::{ScipStatus, Status};

use instance::{Group, Instance};

/// The SCIP backend: the options every solve applies and, once a program is loaded, the SCIP instance holding it.
///
/// ```
/// use culmen_core::{Backend, Events, ModelClass};
/// use culmen_scip::{Emphasis, Scip, ScipOptions};
///
/// let scip = Scip::with_options(ScipOptions { heuristics: Some(Emphasis::Off), ..ScipOptions::default() });
/// assert_eq!(scip.options().heuristics, Some(Emphasis::Off));
/// assert!(Scip::CLASSES.contains(ModelClass::Miqp));
/// assert_eq!(Scip::EVENTS, Events::ALL - Events::NATIVE);
/// ```
#[derive(Debug, Default)]
pub struct Scip {
    options: ScipOptions,
    /// The instance holding the loaded program; `None` before a load and after a failed one.
    instance: Option<Instance>,
}

impl Scip {
    /// A backend with SCIP's default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(ScipOptions::default())
    }

    /// A backend with the given options.
    #[must_use]
    pub fn with_options(options: impl Into<ScipOptions>) -> Self {
        Self {
            options: options.into(),
            instance: None,
        }
    }
}

/// Refuses a program of a class or with a feature this backend does not declare, which a
/// [`Solver`](culmen_core::Solver) never hands it.
///
/// # Errors
///
/// [`ScipError::Unsupported`] for the class, else [`ScipError::UnsupportedFeature`] for the first feature.
fn refuse_unsupported(program: &Program) -> Result<(), ScipError> {
    let class = program.class();
    if !Scip::CLASSES.contains(class) {
        return Err(ScipError::Unsupported { class });
    }
    match program.features().difference(Scip::FEATURES).iter().next() {
        Some(feature) => Err(ScipError::UnsupportedFeature { feature }),
        None => Ok(()),
    }
}

/// Resets SCIP's parameters, keeps SCIP from catching `SIGINT`, turns off the reformulation of binary products in
/// nonlinear constraints, and applies `options` and then the solve's settings; asking to presolve only skips an `Off`
/// presolving emphasis of the options, so a presolving parameter they set by name stays as set.
///
/// # Errors
///
/// [`ScipError::OutOfRange`] for the first setting beyond SCIP's parameter, before any parameter is set;
/// [`ScipError::Call`] when SCIP fails to reset or refuses an emphasis; [`ScipError::Param`] and the other parameter
/// errors for the first parameter refused.
fn configure(instance: &mut Instance, options: &ScipOptions, settings: &Settings) -> Result<(), ScipError> {
    let params = settings_params(settings)?;
    instance.reset_params()?;
    instance.set_param("misc/catchctrlc", &ParamValue::Bool(false))?;
    // The epigraph of a quadratic objective is the only nonlinear constraint the backend creates. SCIP's presolve
    // rewrites a product of two binary variables in it through a clique of its clique table (`x·y = x` from
    // `x <= y`), and that table can hold a clique probing derived under the incumbent's cutoff, which no constraint
    // enforces; the rewritten epigraph then admits points whose `t` lies below `q(x)`, and SCIP reports one of them as
    // optimal (it prints "best solution is not feasible in original problem"). Without the rewrite the epigraph stays
    // exact and the clique only prunes.
    instance.set_param("constraints/nonlinear/reformbinprods", &ParamValue::Bool(false))?;
    let emphases = [
        (Group::Heuristics, options.heuristics),
        (Group::Presolving, presolving(options.presolving, settings.presolve)),
        (Group::Separating, options.separating),
    ];
    for (group, emphasis) in emphases {
        if let Some(emphasis) = emphasis {
            instance.set_emphasis(group, emphasis)?;
        }
    }
    for ScipParam { name, value } in &options.params {
        instance.set_param(name, value)?;
    }
    if settings.presolve == Some(false) {
        instance.set_emphasis(Group::Presolving, Emphasis::Off)?;
    }
    for (name, value) in params {
        instance.set_param(name, &value)?;
    }
    instance.set_console(settings.console);
    Ok(())
}

/// Sets what the handlers `hooks` hold need, after the options and settings, whatever events the callback asks for:
/// for a lazy-constraint handler, no symmetry handling and no dual reductions, which reason about the constraints SCIP
/// holds and not those a handler adds later; for a cut or a heuristic handler, the culmen separator or heuristic called
/// at every node, even when an emphasis turned the separators or heuristics off; when `pricing`, the culmen pricer
/// active, every row modifiable, the LP solver's presolving off, whose duals of a degenerate node LP need not price its
/// columns out, SCIP's aggregation separator off, which cuts with the objective cutoff over the columns held, and the
/// LP solver's objective limit off, so every node LP priced is solved to optimality; for a solve without pricing, the
/// pricer inactive and no row modifiable.
///
/// # Errors
///
/// [`ScipError::Param`] when SCIP refuses a parameter, [`ScipError::Call`] when it refuses another call.
fn configure_hooks(instance: &mut Instance, hooks: &Hooks<'_, Scip>, pricing: bool) -> Result<(), ScipError> {
    if hooks.wants_lazy() {
        instance.set_param("misc/usesymmetry", &ParamValue::Int(0))?;
        instance.set_param("misc/allowstrongdualreds", &ParamValue::Bool(false))?;
        instance.set_param("misc/allowweakdualreds", &ParamValue::Bool(false))?;
    }
    if hooks.wants_cut() {
        instance.set_param("separating/culmen/freq", &ParamValue::Int(1))?;
    }
    if hooks.wants_heuristic() {
        instance.set_param("heuristics/culmen/freq", &ParamValue::Int(1))?;
    }
    if pricing {
        // SoPlex's simplifier can solve a degenerate node LP within its presolve, without a simplex iteration, and
        // hand back a dual that does not price a column at its bound out: for a row that forces its columns to zero,
        // a dual of 0 although a held column then has a reduced cost of the wrong sign at its bound. Every column the
        // row would admit then looks useless to the callback, and pricing ends early with a wrong optimum.
        instance.set_param("lp/presolving", &ParamValue::Bool(false))?;
        // SCIP's aggregation separator aggregates from the objective cutoff `c·x <= cutoff` over the columns the
        // problem holds, without regard to a pricer, and so derives cuts, as bounds that fix a priced column at zero,
        // which a column priced later would refute. Its other aggregations skip modifiable rows, which every program
        // row is in a solve that prices, so it has little else to aggregate.
        instance.set_param("separating/aggregation/freq", &ParamValue::Int(-1))?;
        // With the LP solver's objective limit on, SCIP stops a node LP at the cutoff (`OBJLIMIT`) and still prices
        // it, whose duals are not those of an optimal LP. SCIP's default (2) turns the limit off while a pricer is
        // active; an option may have set another value, so a pricing solve turns it off outright.
        instance.set_param("lp/disablecutoff", &ParamValue::Int(1))?;
    }
    instance.set_pricing(pricing)
}

/// The presolving emphasis set before the options' parameters: the options' own, but none in place of `Off` when the
/// settings ask to presolve, so SCIP's default presolving runs. Asking not to presolve sets `Off` after the parameters
/// instead, so it wins over them.
fn presolving(options: Option<Emphasis>, presolve: Option<bool>) -> Option<Emphasis> {
    options.filter(|&emphasis| !(presolve == Some(true) && emphasis == Emphasis::Off))
}

/// The SCIP parameters a solve's settings set, beside its presolving and console.
///
/// # Errors
///
/// [`ScipError::OutOfRange`] when the seed, the node limit or the solution limit exceeds SCIP's parameter.
fn settings_params(settings: &Settings) -> Result<Vec<(&'static str, ParamValue)>, ScipError> {
    let mut set = Vec::new();
    if let Some(limit) = settings.time_limit {
        set.push(("limits/time", ParamValue::Real(limit.as_secs_f64())));
    }
    if let Some(gap) = settings.relative_gap {
        set.push(("limits/gap", ParamValue::Real(gap.get())));
    }
    if let Some(gap) = settings.absolute_gap {
        set.push(("limits/absgap", ParamValue::Real(gap.get())));
    }
    if let Some(seed) = settings.seed {
        let seed = int("randomization/randomseedshift", seed.into())?;
        set.push(("randomization/randomseedshift", ParamValue::Int(seed)));
    }
    if let Some(nodes) = settings.node_limit {
        set.push(("limits/nodes", ParamValue::Longint(longint("limits/nodes", nodes)?)));
    }
    if let Some(solutions) = settings.solution_limit {
        set.push((
            "limits/bestsol",
            ParamValue::Int(int("limits/bestsol", solutions.get())?),
        ));
    }
    Ok(set)
}

/// An exact integer setting as SCIP's `int` parameter `name`.
///
/// # Errors
///
/// [`ScipError::OutOfRange`] when `value` exceeds `i32::MAX`.
fn int(name: &'static str, value: u64) -> Result<i32, ScipError> {
    i32::try_from(value).map_err(|_| ScipError::OutOfRange {
        name,
        value,
        max: i32::MAX.unsigned_abs().into(),
    })
}

/// An exact integer setting as SCIP's `SCIP_Longint` parameter `name`.
///
/// # Errors
///
/// [`ScipError::OutOfRange`] when `value` exceeds `i64::MAX`.
fn longint(name: &'static str, value: u64) -> Result<i64, ScipError> {
    i64::try_from(value).map_err(|_| ScipError::OutOfRange {
        name,
        value,
        max: i64::MAX.unsigned_abs(),
    })
}

impl Backend for Scip {
    type Error = ScipError;
    type Status = ScipStatus;
    type Log<'a> = ScipLog<'a>;
    type Progress<'a> = ScipProgress;
    type Poll<'a> = ScipPoll<'a>;
    type Solution<'a> = ScipSolution<'a>;
    type Native<'a> = Unsupported;
    type Options = ScipOptions;

    /// Every observational event, each from a plugin the instance includes (the module documentation lists them); no
    /// native event.
    const EVENTS: Events = Events::ALL.difference(Events::NATIVE);

    /// Every class: a quadratic objective `q(x)` becomes an epigraph variable `t` with cost one, bounded by the
    /// quadratic constraint `q(x) - t <= 0` and `t >= 0` when minimizing, `q(x) - t >= 0` and `t <= 0` when
    /// maximizing, where a convex `q` is never negative and a concave one never positive. SCIP solves the constraint
    /// as a convex nonlinear constraint by cutting planes. Each solve turns off SCIP's presolve rewrite of binary
    /// products in it (`constraints/nonlinear/reformbinprods`), which rewrites a product through a clique that may hold
    /// only for solutions better than the incumbent and that no constraint enforces, so the rewritten constraint can
    /// admit a point below the epigraph and SCIP can report a wrong optimum for an MIQP whose integer columns presolve
    /// makes binary. A solution's objective takes `q(x)` itself in the place of `t`, which meets it only to SCIP's
    /// feasibility tolerance; so an optimum's objective is close to the optimal value, and its point, where the
    /// objective is flat, less close to the optimal point.
    const CLASSES: ModelClasses = ModelClasses::ALL;

    /// Builds the program into a new SCIP instance, one variable per column and one linear constraint per row (a
    /// ranged row has both sides, a free row neither), and a quadratic objective's epigraph, replacing the instance
    /// before.
    ///
    /// # Errors
    ///
    /// [`ScipError::Unsupported`] or [`ScipError::UnsupportedFeature`] for a program this backend does not declare;
    /// [`ScipError::BeyondInfinity`] for a finite number SCIP would read as infinite; [`ScipError::ColName`] or
    /// [`ScipError::RowName`] for a name holding a NUL byte; [`ScipError::Call`] when SCIP refuses a call. The
    /// backend then holds no program.
    fn options(&self) -> &ScipOptions {
        &self.options
    }

    fn options_mut(&mut self) -> &mut ScipOptions {
        &mut self.options
    }

    fn load(&mut self, program: &Program) -> Result<(), ScipError> {
        self.instance = None;
        refuse_unsupported(program)?;
        self.instance = Some(Instance::load(program)?);
        Ok(())
    }

    /// Solves the loaded program under the backend's options and `settings`, from its start and the last solve's best
    /// solutions as the module documentation says, delivering to the callback of `hooks` the events it asks for,
    /// calling its handlers, and handing every column its pricer adds to the sink it gives.
    ///
    /// # Errors
    ///
    /// [`ScipError::NotLoaded`] before a load; [`ScipError::PricingQuadratic`] for a solve with a pricer on a program
    /// with a quadratic objective, before SCIP is called; [`ScipError::BranchEarly`] for a pricer ending pricing early
    /// where SCIP cannot branch; the errors of
    /// the options and settings the module documentation lists; [`ScipError::Call`] when SCIP fails, or a call a plugin
    /// makes for the callback does; [`ScipError::Failed`] when SCIP ends with a status that proves nothing;
    /// [`ScipError::InfeasibleBest`] when SCIP's own check finds its best solution infeasible in the original problem,
    /// which is then not reported.
    ///
    /// # Panics
    ///
    /// When a callback or handler panics, once SCIP has stopped.
    fn solve(&mut self, settings: &Settings, mut hooks: Hooks<'_, Self>) -> Result<Solved<ScipStatus>, ScipError> {
        let instance = self.instance.as_mut().ok_or(ScipError::NotLoaded)?;
        let priced = hooks.take_priced();
        let pricing = priced.is_some();
        if pricing && instance.quadratic() {
            return Err(ScipError::PricingQuadratic);
        }
        configure(instance, &self.options, settings)?;
        configure_hooks(instance, &hooks, pricing)?;
        let start = Instant::now();
        let ran = instance.solve(hooks, priced, start);
        let elapsed = start.elapsed();
        if let Some(payload) = ran.panic {
            resume_unwind(payload);
        }
        ran.result?;
        let status = instance.status();
        let best = instance.best();
        if let Some(best) = &best
            && !instance.best_feasible(pricing)?
        {
            return Err(ScipError::InfeasibleBest {
                status: status.status,
                objective: best.objective,
            });
        }
        let outcome: Outcome = status::outcome(&status, best)?;
        Ok(Solved {
            outcome,
            status,
            elapsed,
        })
    }
}

/// Lazy constraints through a constraint handler, checked and enforced after every other.
impl LazyConstraints for Scip {
    type Lazy<'a> = ScipLazy<'a>;
}

/// User cuts through a separator on each node's LP solution.
impl UserCuts for Scip {
    type Cut<'a> = ScipCut<'a>;
}

/// Heuristic solutions through a primal heuristic before and after each node.
impl Heuristics for Scip {
    type Heuristic<'a> = ScipHeuristic<'a>;
}

/// Pricing through a variable pricer, delayed, on each optimal or infeasible node LP; not for a program with a
/// quadratic objective ([`ScipError::PricingQuadratic`]).
impl Prices for Scip {
    type Price<'a> = ScipPrice<'a>;
}

/// Starts: a full start as a SCIP solution, a sparse one as a partial solution SCIP completes.
impl Starts for Scip {
    /// Hands SCIP the start for the next solve, which SCIP checks, and completes when it is sparse, as that solve
    /// starts. An infeasible start, or a sparse one SCIP cannot complete, is dropped then, as SCIP drops any.
    ///
    /// # Errors
    ///
    /// [`ScipError::NotLoaded`] before a load, [`ScipError::Start`] for a start that does not fit the program,
    /// [`ScipError::Call`] when SCIP refuses a call.
    fn set_start(&mut self, start: &Start) -> Result<(), ScipError> {
        let instance = self.instance.as_mut().ok_or(ScipError::NotLoaded)?;
        start.check(instance.num_cols()).map_err(|_| ScipError::Start)?;
        instance.add_start(start)
    }
}

/// A SCIP version, ordered as versions are; displayed as `major.minor.patch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Version {
    /// The major version.
    pub major: u32,
    /// The minor version.
    pub minor: u32,
    /// The patch version, SCIP's technical version.
    pub patch: u32,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The version of the SCIP library linked in.
///
/// ```
/// let version = culmen_scip::version();
/// assert_eq!(version.major, 10);
/// assert_eq!(version.to_string(), format!("10.{}.{}", version.minor, version.patch));
/// ```
#[must_use]
pub fn version() -> Version {
    // SAFETY: the version getters take no arguments and return plain integers.
    let (major, minor, patch) = unsafe {
        (
            scip_sys::SCIPmajorVersion(),
            scip_sys::SCIPminorVersion(),
            scip_sys::SCIPtechVersion(),
        )
    };
    // SCIP's version numbers are never negative, so their magnitudes are the numbers.
    Version {
        major: major.unsigned_abs(),
        minor: minor.unsigned_abs(),
        patch: patch.unsigned_abs(),
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU64;
    use core::time::Duration;

    use culmen_core::{Bounds, Priced, ProgramColumn, ProgramPlan, absolute_gap, bounds, relative_gap, threads};

    use super::*;

    #[test]
    fn test_settings_set_scips_parameters_by_name_and_a_thread_count_sets_none() {
        let settings = Settings {
            time_limit: Some(Duration::from_millis(1500)),
            relative_gap: Some(relative_gap!(0.01)),
            absolute_gap: Some(absolute_gap!(0.5)),
            threads: Some(threads!(4)),
            seed: Some(7),
            presolve: Some(false),
            node_limit: Some(100),
            solution_limit: NonZeroU64::new(3),
            console: true,
        };
        assert_eq!(
            settings_params(&settings),
            Ok(vec![
                ("limits/time", ParamValue::Real(1.5)),
                ("limits/gap", ParamValue::Real(0.01)),
                ("limits/absgap", ParamValue::Real(0.5)),
                ("randomization/randomseedshift", ParamValue::Int(7)),
                ("limits/nodes", ParamValue::Longint(100)),
                ("limits/bestsol", ParamValue::Int(3)),
            ])
        );
        assert_eq!(settings_params(&Settings::new()), Ok(vec![]));
    }

    /// Asks for every event and does nothing with them.
    struct Watches;

    impl culmen_core::Callback<Scip> for Watches {}

    /// Prices nothing; never called on a program whose columns price out.
    struct Idle;

    impl culmen_core::Pricer<Scip> for Idle {
        fn price(&mut self, _: &mut ScipPrice<'_>) {}
    }

    /// Adds nothing, counting its calls.
    #[derive(Default)]
    struct Calls(usize);

    impl culmen_core::LazyHandler<Scip> for Calls {
        fn lazy(&mut self, _: &mut ScipLazy<'_>) {
            self.0 += 1;
            assert!(self.0 < 10_000, "Calls: the lazy-constraint handler ran without end");
        }
    }

    impl culmen_core::CutHandler<Scip> for Calls {
        fn cut(&mut self, _: &mut ScipCut<'_>) {
            self.0 += 1;
            assert!(self.0 < 10_000, "Calls: the cut handler ran without end");
        }
    }

    impl culmen_core::HeuristicHandler<Scip> for Calls {
        fn heuristic(&mut self, _: &mut ScipHeuristic<'_>) {
            self.0 += 1;
            assert!(self.0 < 10_000, "Calls: the heuristic handler ran without end");
        }
    }

    /// `x >= 1` and `2x <= 4` over one continuous column of cost one, loaded into SCIP.
    fn one_column() -> (Scip, Program) {
        let mut plan = ProgramPlan::new();
        let x = plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
        plan.row(bounds!(1.0, f64::INFINITY), [(x, 1.0)]);
        plan.row(bounds!(f64::NEG_INFINITY, 4.0), [(x, 2.0)]);
        let program = plan.build().unwrap();
        let mut scip = Scip::new();
        scip.load(&program).unwrap();
        (scip, program)
    }

    fn limited() -> Settings {
        Settings {
            node_limit: Some(100),
            time_limit: Some(Duration::from_secs(10)),
            ..Settings::new()
        }
    }

    #[test]
    fn test_a_solve_without_price_leaves_no_active_pricer_and_no_modifiable_row() {
        let (mut scip, program) = one_column();
        for (callback, pricer, state) in [
            (true, false, (0, 0)),
            (false, true, (1, 2)),
            (false, false, (0, 0)),
            (true, true, (1, 2)),
            (true, false, (0, 0)),
        ] {
            let mut priced = Priced::new(&program, Scip::CLASSES);
            let (mut watches, mut idle) = (Watches, Idle);
            let mut hooks = Hooks::new();
            if callback {
                hooks = hooks.register_callback(&mut watches);
            }
            if pricer {
                hooks = hooks.register_pricer(&mut idle, &mut priced);
            }
            let solved = scip.solve(&limited(), hooks).unwrap();
            assert_eq!(solved.outcome.objective(), Some(1.0), "{callback} {pricer}");
            assert_eq!(priced.num_cols(), 1);
            let instance = scip.instance.as_ref().unwrap();
            assert_eq!(instance.pricing_state(), state, "{callback} {pricer}");
        }
    }

    /// The parameters a search handler sets: symmetry handling, the two dual reductions, and how often the culmen
    /// separator and heuristic run.
    type SearchParams = (Option<i32>, Option<bool>, Option<bool>, Option<i32>, Option<i32>);

    fn search_params(scip: &Scip) -> SearchParams {
        let instance = scip.instance.as_ref().unwrap();
        (
            instance.int_param(c"misc/usesymmetry"),
            instance.bool_param(c"misc/allowstrongdualreds"),
            instance.bool_param(c"misc/allowweakdualreds"),
            instance.int_param(c"separating/culmen/freq"),
            instance.int_param(c"heuristics/culmen/freq"),
        )
    }

    #[test]
    fn test_a_callback_asking_for_every_event_leaves_the_reductions_the_handlers_turn_off() {
        // The emphases turn SCIP's separators and heuristics off, the culmen ones included, so that only a handler
        // turns those back on.
        let (_, program) = one_column();
        let mut scip = Scip::with_options(ScipOptions {
            separating: Some(Emphasis::Off),
            heuristics: Some(Emphasis::Off),
            ..ScipOptions::default()
        });
        scip.load(&program).unwrap();
        scip.solve(&limited(), Hooks::new()).unwrap();
        let plain = search_params(&scip);
        assert_ne!(plain.0, Some(0), "{plain:?}");
        assert_eq!((plain.1, plain.2), (Some(true), Some(true)));
        assert!(plain.3 != Some(1) && plain.4 != Some(1), "{plain:?}");

        // A callback that only watches, asking for every event, changes none of them.
        let mut watches = Watches;
        assert_eq!(culmen_core::Callback::<Scip>::events(&watches), Events::ALL);
        scip.solve(&limited(), Hooks::new().register_callback(&mut watches))
            .unwrap();
        assert_eq!(search_params(&scip), plain);

        // Each handler changes its own.
        let mut lazy = Calls::default();
        scip.solve(&limited(), Hooks::new().register_lazy(&mut lazy)).unwrap();
        assert_eq!(
            search_params(&scip),
            (Some(0), Some(false), Some(false), plain.3, plain.4)
        );
        let (mut cut, mut heuristic) = (Calls::default(), Calls::default());
        let hooks = Hooks::new().register_cut(&mut cut).register_heuristic(&mut heuristic);
        scip.solve(&limited(), hooks).unwrap();
        assert_eq!(search_params(&scip), (plain.0, plain.1, plain.2, Some(1), Some(1)));
    }

    /// Prices one column without entries, of cost one, at its first round.
    #[derive(Default)]
    struct AddsOne(usize);

    impl culmen_core::Pricer<Scip> for AddsOne {
        fn price(&mut self, ctx: &mut ScipPrice<'_>) {
            self.0 += 1;
            assert!(self.0 < 1_000, "AddsOne: the pricer ran without end");
            if self.0 == 1 {
                let column = ProgramColumn::continuous(1.0, bounds!(0.0, 1.0));
                culmen_core::PriceContext::add_col(
                    &mut *ctx,
                    culmen_core::NewCol {
                        column,
                        entries: Vec::new(),
                    },
                )
                .unwrap();
            }
        }
    }

    /// Maximize `x` over `x` in `[0, 1]` with the row `x <= 1`, loaded into SCIP.
    fn maximize_one() -> (Scip, Program) {
        let mut plan = ProgramPlan::new();
        plan.sense = culmen_core::Sense::Maximize;
        let x = plan.col(ProgramColumn::continuous(1.0, bounds!(0.0, 1.0)));
        plan.row(bounds!(f64::NEG_INFINITY, 1.0), [(x, 1.0)]);
        let program = plan.build().unwrap();
        let mut scip = Scip::new();
        scip.load(&program).unwrap();
        (scip, program)
    }

    #[test]
    fn test_a_priced_column_is_locked_both_ways_by_the_lazy_handler_and_unlocked_with_its_problem() {
        let (mut scip, program) = maximize_one();
        let mut priced = Priced::new(&program, Scip::CLASSES);
        let (mut pricer, mut lazy) = (AddsOne::default(), Calls::default());
        let hooks = Hooks::new()
            .register_lazy(&mut lazy)
            .register_pricer(&mut pricer, &mut priced);
        let solved = scip.solve(&limited(), hooks).unwrap();
        assert_eq!(solved.outcome.objective(), Some(2.0));
        let instance = scip.instance.as_ref().unwrap();
        let var = instance.priced_vars()[0];
        // The column has no entry, so every lock on it is the lazy handler's: it may add a constraint on it.
        assert_eq!(instance.model_locks(var), (1, 1));
        assert_eq!(instance.kept_locks(var), (1, 1));
        // The next solve frees the transformed problem, and the handler's unlock with it.
        scip.solve(&limited(), Hooks::new()).unwrap();
        assert_eq!(scip.instance.as_ref().unwrap().kept_locks(var), (0, 0));
    }

    /// Adds `x <= 1/2` for every candidate, counting its calls.
    #[derive(Default)]
    struct HalvesX(usize);

    impl culmen_core::LazyHandler<Scip> for HalvesX {
        fn lazy(&mut self, ctx: &mut ScipLazy<'_>) {
            self.0 += 1;
            assert!(self.0 < 1_000, "HalvesX: the lazy-constraint handler ran without end");
            let half = culmen_core::Linear::leq(culmen_core::Col::new(0), 0.5);
            culmen_core::LazyContext::add_lazy(ctx, half).unwrap();
        }
    }

    #[test]
    fn test_a_lazy_constraint_scip_refuses_in_enforcement_fails_the_solve_and_the_instance_solves_again() {
        let (mut scip, _) = maximize_one();
        scip.instance.as_ref().unwrap().refuse_lazy(true);
        let mut lazy = HalvesX::default();
        let failed = scip.solve(&limited(), Hooks::new().register_lazy(&mut lazy));
        assert_eq!(
            failed.map(|solved| solved.outcome),
            Err(ScipError::Call {
                call: "SCIPaddCons",
                code: RetCode::Error,
            })
        );
        assert!(lazy.0 >= 1);
        // SCIP finished the failed solve cleanly, so the instance solves again.
        scip.instance.as_ref().unwrap().refuse_lazy(false);
        let solved = scip.solve(&limited(), Hooks::new()).unwrap();
        assert_eq!(solved.outcome.objective(), Some(1.0));
    }

    #[test]
    fn test_presolving_on_keeps_the_presolving_parameters_the_options_set_by_name() {
        let mut scip = Scip::with_options(ScipOptions::default().param("presolving/domcol/maxrounds", 0));
        let mut plan = ProgramPlan::new();
        plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
        scip.load(&plan.build().unwrap()).unwrap();
        let instance = scip.instance.as_mut().unwrap();
        for presolve in [None, Some(true)] {
            let settings = Settings {
                presolve,
                ..Settings::new()
            };
            configure(instance, &scip.options, &settings).unwrap();
            assert_eq!(
                instance.int_param(c"presolving/domcol/maxrounds"),
                Some(0),
                "{presolve:?}"
            );
        }
        // Off wins over the options, and an option that turned presolving off is turned back on.
        let off = Settings {
            presolve: Some(false),
            ..Settings::new()
        };
        configure(instance, &scip.options, &off).unwrap();
        assert_eq!(instance.int_param(c"presolving/maxrounds"), Some(0));
        let on = Settings {
            presolve: Some(true),
            ..Settings::new()
        };
        let options = ScipOptions {
            presolving: Some(Emphasis::Off),
            ..ScipOptions::default()
        };
        configure(instance, &options, &on).unwrap();
        assert_eq!(instance.int_param(c"presolving/maxrounds"), Some(-1));
    }

    #[test]
    fn test_presolving_on_drops_only_an_off_emphasis_of_the_options() {
        for options in [
            None,
            Some(Emphasis::Default),
            Some(Emphasis::Aggressive),
            Some(Emphasis::Fast),
            Some(Emphasis::Off),
        ] {
            assert_eq!(presolving(options, None), options, "{options:?}");
            assert_eq!(presolving(options, Some(false)), options, "{options:?}");
        }
        assert_eq!(presolving(None, Some(true)), None);
        assert_eq!(presolving(Some(Emphasis::Off), Some(true)), None);
        assert_eq!(presolving(Some(Emphasis::Default), Some(true)), Some(Emphasis::Default));
        assert_eq!(
            presolving(Some(Emphasis::Aggressive), Some(true)),
            Some(Emphasis::Aggressive)
        );
        assert_eq!(presolving(Some(Emphasis::Fast), Some(true)), Some(Emphasis::Fast));
    }
}
