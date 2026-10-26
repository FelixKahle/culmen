# HiGHS

`culmen_highs` is the HiGHS backend: HiGHS 1.15 through the `highs-sys` crate, with every HiGHS callback as a typed,
safe context, HiGHS's documented options as typed fields, and every HiGHS status. With the default `highs` feature
it is `culmen::highs`, and its backend type is also `culmen::Highs` and in the prelude.

## The backend

`Highs` owns one HiGHS instance and the options every solve applies.

| Item | Does |
|---|---|
| `Highs::new()` | A backend with HiGHS's default options. |
| `Highs::with_options(options)` | A backend with the given `HighsOptions`. |
| `highs.options()`, `highs.options_mut()` | The options, to read or to change between solves. |
| `Highs::EVENTS` | `LOG`, `PROGRESS`, `POLL`, `FEASIBLE`, `INCUMBENT`, `HEURISTIC` and `NATIVE`. |
| `Highs::CLASSES` | `Lp`, `Milp` and `Qp`; see [Quadratic programs](#quadratic-programs). |
| `culmen_highs::version()` | The version of the linked HiGHS; it displays as `1.15.x`. |

HiGHS is built from the sources bundled with `highs-sys`, with cmake and libclang, so the version is fixed by the
crate and needs no system install. The `Highs` backend is `Send` and can be moved to another thread to solve
there ([Threads](#threads)).

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::highs::{HighsOptions, LpSolver, Switch};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = HighsOptions {
        presolve: Some(Switch::Off),
        mip_heuristic_effort: Some(0.2),
        time_limit: Some(Duration::from_secs(10)),
        ..HighsOptions::default()
    };

    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(4.0));
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + 3.0 * y).leq(6.0));
    m.maximize(3.0 * x + 2.0 * y);

    let mut solver = Solver::new(Highs::with_options(options), m.build()?)?;
    let first = solver.solve(Request::new())?;

    // The options belong to the backend; change them between solves.
    solver.backend_mut().options_mut().solver = Some(LpSolver::Simplex);
    let second = solver.solve(Request::new())?;
    assert_eq!(first.outcome.solution().map(|s| s.objective.round()), Some(13.0));
    assert_eq!(second.outcome.solution().map(|s| s.objective.round()), Some(13.0));
    Ok(())
}
```

## Options

`HighsOptions` is plain data. Each documented HiGHS option is a field named as HiGHS names it, `None` for HiGHS's
default: `presolve`, `solver`, `parallel`, `run_crossover`, `time_limit`, `ranging`, the tolerances
(`primal_feasibility_tolerance`, `dual_feasibility_tolerance`, `optimality_tolerance`, `kkt_tolerance`, …),
`objective_bound` and `objective_target`, `random_seed`, `threads`, the simplex options (`simplex_strategy`,
`simplex_scale_strategy`, the edge weights, the iteration and update limits), the logging options (`output_flag`,
`log_to_console`, `log_file`), the MIP options (`mip_rel_gap`, `mip_abs_gap`, the node, leaf, stall and solution
limits, the cut-pool limits, `mip_heuristic_effort` and each heuristic's switch, `mip_feasibility_tolerance`,
`mip_lp_solver`, `mip_ipm_solver`, …), the interior-point and PDLP limits and tolerances, `solve_relaxation`,
`allow_unbounded_or_infeasible` and `presolve_reduction_limit`. The rustdoc of `HighsOptions` lists every field.

A field whose HiGHS option takes one of a few values is an enum:

| Type | Variants | Options |
|---|---|---|
| `Switch` | `Off`, `Choose`, `On` | `presolve`, `parallel`, `run_crossover` |
| `OnOff` | `Off`, `On` | `ranging` |
| `LpSolver` | `Choose`, `Simplex`, `Ipm`, `Ipx`, `Hipo`, `Pdlp`, `Hipdlp` | `solver` |
| `MipLpSolver` | `Choose`, `Simplex`, `Ipm`, `Ipx`, `Hipo` | `mip_lp_solver` |
| `MipIpmSolver` | `Choose`, `Ipm`, `Ipx`, `Hipo` | `mip_ipm_solver` |
| `SimplexStrategy` | `Choose`, `Dual`, `DualSip`, `DualPami`, `Primal` | `simplex_strategy` |
| `ScaleStrategy` | `Off`, `Choose`, `Equilibration`, `ForcedEquilibration`, `MaxValue` | `simplex_scale_strategy` |
| `EdgeWeight` | `Choose`, `Dantzig`, `Devex`, `SteepestEdge` | `simplex_dual_edge_weight_strategy`, `simplex_primal_edge_weight_strategy` |

**Options by name.** Any other HiGHS option is set by its name with `HighsOptions::option(name, value)`, which
appends a `RawOption { name, value }` to the `raw` field. The value is an `OptionValue`, one of HiGHS's four kinds
(`Bool`, `Int`, `Double`, `String`), converted from a `bool`, `i32`, `f64`, `&str` or `String`. HiGHS checks the
name and the value when the solve applies them; an unknown name or a value of the wrong kind or range is a
[`HighsError::Option`](#errors).

```rust
use culmen::highs::HighsOptions;

let options = HighsOptions::default()
    .option("mip_pool_soft_limit", 5000)
    .option("log_file", "highs.log");
assert_eq!(options.raw.len(), 2);
```

## How options and settings layer

Every solve sets HiGHS up from scratch, in this order:

1. HiGHS's options are reset to its defaults.
2. HiGHS's log is turned on and routed to the callback only: `output_flag` on, `log_to_console` off. Standard
   output stays quiet.
3. The backend's `HighsOptions` apply: the typed fields first, then the options by name, in the order added.
4. The solve's [`Settings`](solving.md#settings) apply.

So a per-solve setting always wins over the backend's options, and an option set by name wins over a typed field
for the same option. Nothing set on HiGHS carries over from one solve to the next except through `HighsOptions`.
The [Backends](backends.md#settings) page lists the HiGHS option each `Settings` field sets.

HiGHS measures `time_limit` on a clock the backend restarts before every run, so each solve gets the full limit.
A `Settings::seed` or `Settings::threads` beyond HiGHS's 32-bit integers is refused with
[`HighsError::OutOfRange`](#errors).

## Threads

- **`Highs` is `Send`**: move it, or the whole `Solver`, to a worker thread to solve there.
- **HiGHS keeps one task scheduler per thread**, started by the first run on that thread at that run's thread
  count. A later run on the same thread asking for another number of threads fails with
  [`HighsError::Run`](#errors). Leaving `Settings::threads` unset never conflicts; a solve on a fresh thread may
  pick any count.
- **Dropping a `Highs`** shuts down the scheduler of the dropping thread.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_vars(4, |_| Var::binary());
    m.add_constraint(LinExpr::sum(x.iter().copied()).leq(2));
    m.maximize(LinExpr::sum(x.iter().copied()));
    let mut solver = Solver::new(Highs::new(), m.build()?)?;

    let worker = std::thread::spawn(move || solver.solve(Request::new().settings_from(Settings { threads: Some(threads!(2)), ..Settings::new() })));
    let solved = worker.join().map_err(|_| "the solve thread panicked")??;
    assert_eq!(solved.outcome.solution().map(|s| s.objective), Some(2.0));
    Ok(())
}
```

## Statuses and outcomes

`Solved::status` is a `HighsStatus`, HiGHS's own account of the run beside the generic outcome:

| Field | Holds |
|---|---|
| `run` | What `Highs_run` returned: a `CallStatus`, `Ok`, `Warning` or `Error`. |
| `model` | The `ModelStatus`: every value of HiGHS's `HighsModelStatus`, and `Other(i32)` for a value HiGHS 1.15 does not define. |
| `primal`, `dual` | Whether a primal or dual solution exists: a `SolutionStatus`, `None`, `Infeasible` or `Feasible`. |
| `mip_gap` | HiGHS's relative MIP gap; `None` for an LP or when HiGHS reports none. |
| `mip_nodes`, `simplex_iterations`, `ipm_iterations`, `qp_iterations` | The work done. |
| `primal_dual_objective_error` | HiGHS's relative primal-dual objective error, `\|P - D\| / (1 + \|P\| + \|D\|)`; `None` when it computed none, as for a MIP. |
| `last_error` | The last error line of HiGHS's log, if it wrote one. |

`ModelStatus` displays in HiGHS's own wording, lower-cased: `"optimal"`, `"time limit reached"`,
`"interrupted by user"`.

```rust
use culmen::prelude::*;
use culmen::highs::{CallStatus, ModelStatus, SolutionStatus};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + y).leq(4.0));
    m.add_constraint((x + 3.0 * y).leq(6.0));
    m.maximize(3.0 * x + 2.0 * y + 10);

    let solved = Solver::new(Highs::new(), m.build()?)?.solve(Request::new())?;
    let status = &solved.status;
    assert_eq!((status.run, status.model), (CallStatus::Ok, ModelStatus::Optimal));
    assert_eq!(status.primal, SolutionStatus::Feasible);
    assert_eq!(status.mip_gap, None);
    assert_eq!(status.model.to_string(), "optimal");
    assert!((solved.outcome.solution().ok_or("no solution")?.objective - 22.0).abs() < 1e-9);
    Ok(())
}
```

The generic [`Outcome`](solving.md#solved-and-outcome) is read from these; the [Backends](backends.md#outcomes-and-statuses)
page lists the outcome of every `ModelStatus`. Two of them are decided by more than the status:

- **`Optimal` or `GapReached`.** HiGHS reports a MIP as optimal once a requested gap is reached. The backend reports
  `GapReached` when the gap between the solution and the bound is wider than HiGHS's defaults (a relative gap of
  `1e-4` and an absolute gap of `1e-6`), and `Optimal` otherwise.
- **`NodeLimit` or `SolutionLimit`.** HiGHS reports one status for its node, leaf and improving-solution limits. The
  backend reports `StopReason::NodeLimit` when a node limit is set (in `Settings` or `HighsOptions`) and the node
  count reached it, and `StopReason::SolutionLimit` otherwise.

An empty model, one without columns, is `Optimal` at its offset.

## Quadratic programs

HiGHS 1.15 solves LPs, MILPs and convex QPs, so `Highs::CLASSES` is `{LP, MILP, QP}`. It refuses a QP with integer
columns when it runs ("Cannot solve MIQP problems with HiGHS"), so a MIQP is refused by
[`Solver::new`](solving.md#model-classes-and-refusal) with `LoadError::Unsupported` before HiGHS sees it.

The backend passes the quadratic part as HiGHS reads it, the lower triangle of the Hessian `H` of `½ xᵀHx` column by
column ([The quadratic part](program.md#the-quadratic-part)): the model's `x * x` reaches HiGHS as `Hₓₓ = 2`, and
`x * y` as `Hᵧₓ = 1`. Every `Program` passed the convexity check when it was built, so HiGHS's own refusal of a Hessian
with a diagonal of the wrong sign never arises.

```rust
use culmen::prelude::*;
use culmen::ModelClass;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The point of x + y <= 1 nearest (1, 2): (0, 1), at squared distance 2.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::free());
    m.add_constraint((x + y).leq(1));
    m.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2));
    let program = m.build()?;
    assert_eq!(program.class(), ModelClass::Qp);

    let solved = Solver::new(Highs::new(), program)?.solve(Request::new())?;
    let Outcome::Optimal(best) = solved.outcome else { return Err("no optimum".into()) };
    assert!((best.objective - 2.0).abs() < 1e-6);
    assert!((best.values[x] - 0.0).abs() < 1e-6 && (best.values[y] - 1.0).abs() < 1e-6);
    assert!(solved.status.qp_iterations > 0);
    Ok(())
}
```

HiGHS's QP solver differs from its LP and MIP solvers in what a solve can see and in how far its answer can be
trusted:

- **Only the log.** It calls no interrupt, solution or MIP callback, so a QP solve delivers `LOG` alone: a
  [`Stop`](callbacks.md#stop) is never polled and an observer gets no progress. The time limit of the
  [`Settings`](solving.md#settings) still applies.
- **A row is always passed.** On a QP without rows HiGHS 1.15 starts from the unconstrained minimizer clamped into
  the bounds and can report that point optimal after no iteration when it is not. The backend passes such a QP
  with one free row without entries, which constrains nothing and makes HiGHS find its starting point with its LP
  solver.
- **An unsound ending is an error.** On a small share of QPs, after its iterates turned NaN, HiGHS 1.15 reports an
  optimum its own primal and dual objectives contradict, at a point that need not be feasible, or reports
  unbounded a QP whose every column is bounded. The backend fails such a solve with
  [`HighsError::UnsoundQp`](#errors): an optimum whose relative primal-dual objective error exceeds
  `Highs::QP_OBJECTIVE_ERROR`, `1e-5` (correct optima stay below `1e-6`), or unboundedness with every column
  bounded. An `Optimal` outcome of a QP is therefore
  one HiGHS's own measures support.
- **It may not terminate.** On a rare QP HiGHS 1.15's QP solver iterates without end, its memory growing. A
  [time limit](solving.md#settings) ends such a solve as `Stopped { reason: TimeLimit, .. }`; set one for any QP
  whose solve must end.

## Errors

`HighsError` says why loading or solving failed:

| Variant | When |
|---|---|
| `NotLoaded` | A solve was asked for before a program was loaded. |
| `TooLarge { what, count }` | The program has more columns, rows, entries or Hessian entries (`what`, a `Dimension`) than HiGHS's 32-bit indices hold. |
| `Load { message }` | HiGHS refused the program. |
| `Option { name, value }` | HiGHS refused an option: an unknown name, or a value of the wrong kind or out of range. |
| `OptionName { name }` | An option name holds a NUL byte, which HiGHS's C strings cannot carry. |
| `OptionString { name }` | A string option's value holds a NUL byte. |
| `OutOfRange { name, value }` | A seed or thread count is above `i32::MAX`, beyond HiGHS's integers. |
| `Run { status, message }` | The run failed, as a thread-count conflict makes it; `message` is HiGHS's last error line. |
| `UnsoundQp(UnsoundQp)` | HiGHS's QP solver reported an ending its own measures or the model contradict: `ObjectiveError { error }` or `UnboundedInABox` ([Quadratic programs](#quadratic-programs)). |
| `Analysis(Analysis)` | HiGHS refused or failed an analysis, as a basis it finds singular ([Analysis](analysis.md#highs)). |

```rust
use culmen::prelude::*;
use culmen::highs::{HighsError, HighsOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);
    let options = HighsOptions { log_file: Some("a\0b".to_owned()), ..HighsOptions::default() };
    let mut solver = Solver::new(Highs::with_options(options), m.build()?)?;

    let Err(SolveError::Backend(error)) = solver.solve(Request::new()) else { panic!("the option is refused") };
    assert_eq!(error, HighsError::OptionString { name: "log_file".to_owned() });
    assert_eq!(error.to_string(), "HiGHS: the value of option `log_file` holds a NUL byte");
    Ok(())
}
```

A run that ends without a solution is not an error but an outcome; a model status that means HiGHS failed
(`LoadError`, `ModelError`, `PresolveError`, `SolveError`, `PostsolveError`, `NotSet`) is `HighsError::Run`.

## Callback contexts

A `Callback<Highs>` receives HiGHS's own context types. Each implements the neutral
[context trait](callbacks.md#contexts) of its event and offers what HiGHS reports beyond it:

| Event | Context | HiGHS callback | Beyond the trait |
|---|---|---|---|
| `LOG` | `HighsLog` | `kHighsCallbackLogging` | `log_type()`: a `HighsLogType`, `Info`, `Detailed`, `Verbose`, `Warning` or `Error` |
| `PROGRESS` | `HighsMipProgress` | `kHighsCallbackMipLogging`, when HiGHS writes a MIP log line | `mip()`: the `MipData` |
| `POLL` | `HighsPoll` | the simplex, interior-point and MIP interrupt callbacks | `source()`: a `PollSource`, `Simplex { iterations }`, `Ipm { iterations }` or `Mip(MipData)` |
| `FEASIBLE` | `HighsSolution` | `kHighsCallbackMipSolution` | `mip()` |
| `INCUMBENT` | `HighsSolution` | `kHighsCallbackMipImprovingSolution` | `mip()` |
| `HEURISTIC` | `HighsHeuristic` | `kHighsCallbackMipUserSolution` | `origin()`, a `QueryOrigin`; `mip()` |
| `NATIVE` | `HighsNative` | `kHighsCallbackMipGetCutPool` | the [cut pool](#the-cut-pool) |
| `LAZY`, `CUT` | `Unsupported` | none | never delivered ([Limitations](#limitations)) |

`MipData` is what every MIP callback reports, in the model's objective space, offset included:

| Field | Holds |
|---|---|
| `running_time` | The MIP solver's own clock, which starts after presolve. |
| `objective` | The objective HiGHS reports with the call; `None` before there is one. |
| `primal_bound`, `dual_bound` | The best objective found and the best bound proved; `None` before there is one. |
| `gap` | HiGHS's relative gap; `None` while it is infinite. |
| `nodes`, `lp_iterations` | Branch-and-bound nodes, and LP iterations over all nodes. |

A context's `progress()` carries the elapsed time since the solve began; a poll's phase is `Simplex`, `Barrier` or
`Mip` after its source. A solution HiGHS reports with another number of values than the program has columns is not
delivered.

The interrupt and logging callbacks are always installed, whatever the callback asks for: the polls so that a panic
can stop the run, the log so that a failed run can name HiGHS's last error line. Calls are serialised behind one
lock; a log line HiGHS writes while `supply_partial` repairs a solution re-enters the same run and is not delivered,
since the callback is busy. A solve of another `Highs` started from inside a callback gets its own callbacks.

```rust
use culmen::prelude::*;
use culmen::highs::{HighsLog, HighsLogType, HighsPoll, PollSource};

#[derive(Default)]
struct Inspect {
    warnings: Vec<String>,
    mip_polls: usize,
    best_bound: Option<f64>,
}

impl Callback<Highs> for Inspect {
    fn events(&self) -> Events {
        Events::LOG | Events::POLL
    }
    fn log(&mut self, ctx: &mut HighsLog<'_>) {
        if ctx.log_type() == HighsLogType::Warning {
            self.warnings.push(culmen::LogContext::text(ctx).to_owned());
        }
    }
    fn poll(&mut self, ctx: &mut HighsPoll<'_>) {
        if let PollSource::Mip(mip) = ctx.source() {
            self.mip_polls += 1;
            self.best_bound = mip.dual_bound.or(self.best_bound);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (value, weight) = ([10.0, 13.0, 7.0, 8.0, 9.0, 4.0], [5.0, 6.0, 4.0, 3.0, 5.0, 2.0]);
    let mut m = Model::new();
    let take = m.add_vars(6, |_| Var::binary());
    m.add_constraint(LinExpr::sum((0..6).map(|i| weight[i] * take[i])).leq(12.0));
    m.maximize(LinExpr::sum((0..6).map(|i| value[i] * take[i])));

    let mut inspect = Inspect::default();
    let solved = Solver::new(Highs::new(), m.build()?)?.solve_with(Request::new(), &mut inspect)?;
    assert_eq!(solved.outcome.solution().map(|s| s.objective), Some(25.0));
    assert!(inspect.warnings.is_empty());
    Ok(())
}
```

A method may name the concrete context type, `&mut HighsPoll<'_>`, or the associated one,
`&mut <Highs as Backend>::Poll<'_>`; they are the same type.

## Heuristic solutions

At `HEURISTIC`, HiGHS asks for a solution of the callback's own. `HighsHeuristic::origin()` says where in the
search it asks: `QueryOrigin::AfterSetup`, once the MIP solver is set up; `BeforeDive`; `EvaluateRootNode(point)`,
during root-node evaluation, `point` a `RootNodePoint` naming HiGHS's numbered point; or `Other(i32)` for a value
HiGHS 1.15 does not define.

- `ctx.supply(&values)` offers a value for every column. The values are checked first: one finite value per column,
  or `ActionError::Length` or `ActionError::NotFinite`. HiGHS then checks the solution against the model and keeps
  it as its incumbent if it is feasible and better.
- `ctx.supply_partial(&entries)` offers values for some columns, and HiGHS completes them into a feasible solution
  if it can. It returns `ActionError::Refused` when HiGHS refuses the entries or finds no feasible completion.

HiGHS calls neither solution callback for a solution it takes this way, so the backend reports it itself: once a
MIP poll shows the supplied solution's objective as HiGHS's best, it arrives as `INCUMBENT`, and so reaches the
[observer](callbacks.md#observer) too.

```rust
use culmen::prelude::*;
use culmen::highs::{HighsHeuristic, HighsOptions, QueryOrigin, Switch};
use culmen::{ActionError, HeuristicContext, TiVec};

struct Start {
    values: Option<TiVec<Col, f64>>,
    supplied: Vec<Result<(), ActionError>>,
}

impl Callback<Highs> for Start {
    fn events(&self) -> Events {
        Events::HEURISTIC
    }
    fn heuristic(&mut self, ctx: &mut HighsHeuristic<'_>) {
        if ctx.origin() == QueryOrigin::AfterSetup
            && let Some(values) = self.values.take()
        {
            self.supplied.push(ctx.supply(&values));
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (value, weight) = ([10.0, 13.0, 7.0, 8.0, 9.0, 4.0], [5.0, 6.0, 4.0, 3.0, 5.0, 2.0]);
    let mut m = Model::new();
    let take = m.add_vars(6, |_| Var::binary());
    m.add_constraint(LinExpr::sum((0..6).map(|i| weight[i] * take[i])).leq(12.0));
    m.maximize(LinExpr::sum((0..6).map(|i| value[i] * take[i])));

    // A known feasible start: items 1 and 3, worth 21.
    let mut start: TiVec<Col, f64> = vec![0.0; 6].into();
    start[take[1]] = 1.0;
    start[take[3]] = 1.0;

    let options = HighsOptions { presolve: Some(Switch::Off), ..HighsOptions::default() };
    let mut solver = Solver::new(Highs::with_options(options), m.build()?)?;
    let mut callback = Start { values: Some(start), supplied: Vec::new() };
    let solved = solver.solve_with(Request::new(), &mut callback)?;
    assert_eq!(callback.supplied, [Ok(())]);
    assert_eq!(solved.outcome.solution().map(|s| s.objective), Some(25.0));
    Ok(())
}
```

## The cut pool

At `NATIVE`, HiGHS reports its cut pool after the root node, and again after each restart, as
`HighsNative::CutPool(pool)`. The pool is copied out of HiGHS, so it may be kept:

| Field | Holds |
|---|---|
| `num_cols` | The number of columns of the presolved problem the cuts range over. |
| `cuts` | Each `Cut { lower, upper, entries }`: `lower <= Σ value · column <= upper`, an open end infinite. |
| `mip` | The `MipData` at the time. |

The cuts range over the presolved problem HiGHS's MIP solver works on, not over the program you loaded, so their columns
are `ReducedCol`, a distinct handle type: a cut cannot be read as a row of your program by mistake. `HighsNative` is
`#[non_exhaustive]`, so a `match` on it needs a `_` arm.

## Limitations

- **No lazy constraints.** HiGHS 1.15 declares `kHighsCallbackMipDefineLazyConstraints` and never calls it, so
  `Events::LAZY` is not in `Highs::EVENTS` and its context is `Unsupported`.
- **No user cuts.** HiGHS has no callback for them; `Events::CUT` is `Unsupported` too.
- **An interrupt is honoured only at a poll.** Only `HighsPoll` has `interrupt`, and only `HighsHeuristic` accepts
  a solution.
- **One thread count per thread.** See [Threads](#threads).
- **32-bit indices.** A program with more than `i32::MAX` columns, rows, entries or Hessian entries is refused with
  `HighsError::TooLarge`.
- **No MIQP, and a QP solve sees only the log.** See [Quadratic programs](#quadratic-programs).
- **Options do not persist inside HiGHS.** Each solve resets them; set them through `HighsOptions`.

## See also

- [Callbacks](callbacks.md): the neutral events and contexts.
- [Analysis](analysis.md#highs): duals, basis, ranging, rays, IIS and feasibility relaxation on HiGHS.
- [Solving](solving.md): requests, settings and outcomes.
- [Backends](backends.md): how HiGHS's callbacks, options and statuses map onto culmen's seam, beside Gurobi and SCIP.
- [Design](design.md#where-unsafe-lives): why the trampoline is sound.
- [Documentation index](README.md).
