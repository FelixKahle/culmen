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
//! use culmen_core::{Linear, LinExpr, Model, Outcome, Request, Solver, Var};
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
//! let solved = solver.solve(Request::new())?;
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
//! (`misc/catchctrlc`), turns off SCIP's reformulation of products of binary variables in nonlinear constraints (`constraints/nonlinear/reformbinprods`;
//! see [`Scip::CLASSES`] for why), then applies the backend's [`ScipOptions`] and then the solve's [`Settings`], so a
//! per-solve setting always wins and an option naming either parameter overrides the backend's choice. SCIP's log
//! stays off standard output unless [`Settings::console`] asks for it.
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
//! | `presolve` | `SCIPsetPresolving`: off, or the presolving emphasis of the options when it is not off, else default |
//! | `node_limit` | `limits/nodes`, at most `i64::MAX` |
//! | `solution_limit` | `limits/solutions`, at most `i32::MAX` |
//! | `console` | SCIP's message handler quiet or not |
//!
//! A solve reports a best solution, an optimum or a reached gap only after SCIP's own check
//! (`SCIPcheckSolOrig`) finds that solution feasible in the original problem, to SCIP's feasibility tolerance; when
//! it does not, as when presolve rewrote a constraint unsoundly, the solve fails with [`ScipError::InfeasibleBest`]
//! instead. That check skips the rows of a solve that prices, which are modifiable, so such a solve also has SCIP
//! check the solution against its transformed problem (`SCIPcheckSol`), whose rows hold the priced columns.
//!
//! A value beyond what SCIP's parameter takes is refused with [`ScipError::OutOfRange`] before any parameter is set.
//! SCIP prints the error messages of a call it refuses, such as a parameter it does not know, to standard error
//! through its process-wide error printer, which the backend leaves as it is.
//!
//! # Callbacks
//!
//! [`Scip::EVENTS`] is every generic event, each from a SCIP plugin, with a context of this crate's:
//!
//! | Event | SCIP plugin | Context |
//! |---|---|---|
//! | `LOG` | the instance's message handler: each info message and warning, a whole line | [`ScipLog`] |
//! | `POLL`, `PROGRESS` | an event handler: each presolve round, focused node, solved node LP and solved node | [`ScipPoll`], [`ScipProgress`] |
//! | `FEASIBLE`, `INCUMBENT` | an event handler: each solution SCIP stores, a new best one as the incumbent too | [`ScipSolution`] |
//! | `LAZY` | a constraint handler without constraints, checked and enforced after every other | [`ScipLazy`] |
//! | `CUT` | a separator on each node's LP solution | [`ScipCut`] |
//! | `HEURISTIC` | a primal heuristic before and after each node | [`ScipHeuristic`] |
//! | `PRICE` | a variable pricer, delayed, on each optimal or infeasible node LP | [`ScipPrice`] |
//!
//! Every value a context reports is the program's: a candidate, a relaxation and a solution hold one value per
//! column, read through SCIP's original variables whatever presolve did to them, and every objective value and
//! bound is in the program's sense with its offset and quadratic part. A lazy constraint holds for the rest of the
//! solve, so every later candidate satisfies it; a solve asking for [`Events::LAZY`] therefore locks every variable
//! against dual reductions and turns off `misc/usesymmetry`, `misc/allowstrongdualreds` and
//! `misc/allowweakdualreds`, which reason about the constraints SCIP holds rather than those a callback adds later.
//! A solve asking for [`Events::CUT`] or [`Events::HEURISTIC`] calls its separator or heuristic at every node, also
//! when an emphasis turned SCIP's own off.
//!
//! Pricing is opt-in: only a solve whose callback asks for [`Events::PRICE`] activates the pricer and marks every
//! program row modifiable, as SCIP requires of a row a pricer adds variables to; any other solve leaves the pricer
//! inactive and no row modifiable, so SCIP gives up nothing for it. A pricing solve also turns off the LP solver's
//! presolving (`lp/presolving`), whose duals of a degenerate node LP need not price out the columns at their bounds,
//! and SCIP's aggregation separator (`separating/aggregation/freq`), which cuts with the objective cutoff over the
//! columns held as if no other column could enter. SCIP does not price a program with a quadratic
//! objective: such a solve fails with [`ScipError::PricingQuadratic`] before SCIP runs. Every column a callback prices
//! is a variable of SCIP's transformed problem with no original counterpart, so a heuristic solution of a pricing
//! solve is a transformed one ([`ScipHeuristic`]), and the backend keeps the priced variables until it frees that
//! problem, so the best solution has a value for each.
//!
//! ```
//! use culmen_core::{Backend, Callback, Col, Events, LazyContext, Linear, Model, Request, Solver, Var};
//! use culmen_scip::Scip;
//!
//! /// Forbids taking both items, a conflict only the callback knows.
//! struct NotBoth(Col, Col);
//!
//! impl<B: Backend> Callback<B> for NotBoth {
//!     fn events(&self) -> Events {
//!         Events::LAZY
//!     }
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
//! let solved = solver.solve_with(Request::new(), &mut NotBoth(x, y))?;
//! assert_eq!(solved.outcome.objective(), Some(3.0));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A panic in a callback never unwinds into SCIP: the plugin catches it and asks SCIP to stop, and the panic resumes
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
    Backend, Callback, Events, ModelClasses, Outcome, Priced, Program, ProgramFeature, ProgramFeatures, Settings,
    Solved, Start, Unsupported,
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

    /// The options every solve applies.
    #[must_use]
    pub const fn options(&self) -> &ScipOptions {
        &self.options
    }

    /// The options every solve applies, to change between solves.
    pub const fn options_mut(&mut self) -> &mut ScipOptions {
        &mut self.options
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
/// nonlinear constraints, and applies `options` and then the solve's settings.
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
        (Group::Presolving, options.presolving),
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
    if let Some(presolve) = settings.presolve {
        instance.set_emphasis(Group::Presolving, presolving(options.presolving, presolve))?;
    }
    for (name, value) in params {
        instance.set_param(name, &value)?;
    }
    instance.set_console(settings.console);
    Ok(())
}

/// Sets what the events `wanted` need, after the options and settings: for lazy constraints, no symmetry handling and
/// no dual reductions, which reason about the constraints SCIP holds and not those a callback adds later; for cuts and
/// heuristic solutions, the culmen separator and heuristic called at every node, even when an emphasis turned the
/// separators or heuristics off; for pricing, the culmen pricer active, every row modifiable, the LP solver's
/// presolving off, whose duals of a degenerate node LP need not price its columns out, and SCIP's aggregation
/// separator off, which cuts with the objective cutoff over the columns held; for a solve without pricing, the pricer
/// inactive and no row modifiable.
///
/// # Errors
///
/// [`ScipError::Param`] when SCIP refuses a parameter, [`ScipError::Call`] when it refuses another call.
fn configure_events(instance: &mut Instance, wanted: Events) -> Result<(), ScipError> {
    if wanted.contains(Events::LAZY) {
        instance.set_param("misc/usesymmetry", &ParamValue::Int(0))?;
        instance.set_param("misc/allowstrongdualreds", &ParamValue::Bool(false))?;
        instance.set_param("misc/allowweakdualreds", &ParamValue::Bool(false))?;
    }
    if wanted.contains(Events::CUT) {
        instance.set_param("separating/culmen/freq", &ParamValue::Int(1))?;
    }
    if wanted.contains(Events::HEURISTIC) {
        instance.set_param("heuristics/culmen/freq", &ParamValue::Int(1))?;
    }
    let pricing = wanted.contains(Events::PRICE);
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
    }
    instance.set_pricing(pricing)
}

/// The presolving emphasis of a solve whose settings ask to presolve or not: off, or the options' emphasis when it
/// presolves, else SCIP's default.
fn presolving(options: Option<Emphasis>, presolve: bool) -> Emphasis {
    match options {
        _ if !presolve => Emphasis::Off,
        Some(emphasis @ (Emphasis::Aggressive | Emphasis::Fast)) => emphasis,
        Some(Emphasis::Default | Emphasis::Off) | None => Emphasis::Default,
    }
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
            "limits/solutions",
            ParamValue::Int(int("limits/solutions", solutions.get())?),
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
    type Lazy<'a> = ScipLazy<'a>;
    type Cut<'a> = ScipCut<'a>;
    type Heuristic<'a> = ScipHeuristic<'a>;
    type Price<'a> = ScipPrice<'a>;
    type Native<'a> = Unsupported;

    /// Every generic event, each from a plugin the instance includes (the module documentation lists them); no
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

    /// Starts: a full start as a SCIP solution, a sparse one as a partial solution SCIP completes.
    const FEATURES: ProgramFeatures = ProgramFeatures::of(&[ProgramFeature::StartSolution]);

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
    fn load(&mut self, program: &Program) -> Result<(), ScipError> {
        self.instance = None;
        refuse_unsupported(program)?;
        self.instance = Some(Instance::load(program)?);
        Ok(())
    }

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

    /// Solves the loaded program under the backend's options and `settings`, from its start and the last solve's best
    /// solutions as the module documentation says, delivering to `callback` the events it asks for and handing every
    /// column it prices to `priced`.
    ///
    /// # Errors
    ///
    /// [`ScipError::NotLoaded`] before a load; [`ScipError::PricingQuadratic`] for a callback asking for
    /// [`Events::PRICE`] on a program with a quadratic objective, before SCIP is called; [`ScipError::BranchEarly`] for
    /// a pricing callback ending pricing early where SCIP cannot branch; the errors of the options and settings the
    /// module documentation lists; [`ScipError::Call`] when SCIP fails, or a call a plugin makes for the callback does;
    /// [`ScipError::Failed`] when SCIP ends with a status that proves nothing; [`ScipError::InfeasibleBest`] when
    /// SCIP's own check finds its best solution infeasible in the original problem, which is then not reported.
    ///
    /// # Panics
    ///
    /// When `callback` panics, once SCIP has stopped.
    fn solve(
        &mut self,
        settings: &Settings,
        callback: &mut dyn Callback<Self>,
        priced: &mut Priced,
    ) -> Result<Solved<ScipStatus>, ScipError> {
        let instance = self.instance.as_mut().ok_or(ScipError::NotLoaded)?;
        let wanted = callback.events().intersection(Self::EVENTS);
        if wanted.contains(Events::PRICE) && instance.quadratic() {
            return Err(ScipError::PricingQuadratic);
        }
        configure(instance, &self.options, settings)?;
        configure_events(instance, wanted)?;
        let start = Instant::now();
        let ran = instance.solve(callback, wanted, start, priced);
        let elapsed = start.elapsed();
        if let Some(payload) = ran.panic {
            resume_unwind(payload);
        }
        ran.result?;
        let status = instance.status();
        let best = instance.best();
        if let Some(best) = &best
            && !instance.best_feasible(wanted.contains(Events::PRICE))?
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

    use culmen_core::{Bounds, ProgramColumn, ProgramPlan, absolute_gap, bounds, relative_gap, threads};

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
                ("limits/solutions", ParamValue::Int(3)),
            ])
        );
        assert_eq!(settings_params(&Settings::new()), Ok(vec![]));
    }

    /// Asks for the events it holds and does nothing with them.
    struct Asks(Events);

    impl Callback<Scip> for Asks {
        fn events(&self) -> Events {
            self.0
        }
    }

    #[test]
    fn test_a_solve_without_price_leaves_no_active_pricer_and_no_modifiable_row() {
        let mut plan = ProgramPlan::new();
        let x = plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
        plan.row(bounds!(1.0, f64::INFINITY), [(x, 1.0)]);
        plan.row(bounds!(f64::NEG_INFINITY, 4.0), [(x, 2.0)]);
        let program = plan.build().unwrap();
        let mut scip = Scip::new();
        scip.load(&program).unwrap();
        let settings = Settings {
            node_limit: Some(100),
            time_limit: Some(Duration::from_secs(10)),
            ..Settings::new()
        };
        let all_but_price = Events::ALL - Events::PRICE;
        for (wanted, state) in [
            (all_but_price, (0, 0)),
            (Events::PRICE, (1, 2)),
            (Events::NONE, (0, 0)),
            (Events::PRICE | Events::LAZY, (1, 2)),
            (all_but_price, (0, 0)),
        ] {
            let mut priced = Priced::new(&program, Scip::CLASSES);
            let solved = scip.solve(&settings, &mut Asks(wanted), &mut priced).unwrap();
            assert_eq!(solved.outcome.objective(), Some(1.0), "{wanted}");
            assert_eq!(priced.num_cols(), 1);
            let instance = scip.instance.as_ref().unwrap();
            assert_eq!(instance.pricing_state(), state, "{wanted}");
        }
    }

    #[test]
    fn test_presolving_off_wins_and_on_keeps_an_emphasis_that_presolves() {
        for options in [
            None,
            Some(Emphasis::Default),
            Some(Emphasis::Aggressive),
            Some(Emphasis::Fast),
            Some(Emphasis::Off),
        ] {
            assert_eq!(presolving(options, false), Emphasis::Off, "{options:?}");
        }
        assert_eq!(presolving(None, true), Emphasis::Default);
        assert_eq!(presolving(Some(Emphasis::Off), true), Emphasis::Default);
        assert_eq!(presolving(Some(Emphasis::Default), true), Emphasis::Default);
        assert_eq!(presolving(Some(Emphasis::Aggressive), true), Emphasis::Aggressive);
        assert_eq!(presolving(Some(Emphasis::Fast), true), Emphasis::Fast);
    }
}
