# HiGHS

`culmen_highs` is the HiGHS backend: HiGHS 1.16 through the `highs-sys` crate, with every HiGHS callback as a typed,
safe context, HiGHS's commonly used options as typed fields, and every HiGHS status. With the default `highs` feature
it is `culmen::highs`, and its backend type is also `culmen::Highs` and in the prelude.

## The backend

`Highs` owns one HiGHS instance and the options every solve applies.

| Item | Does |
|---|---|
| `Highs::new()` | A backend with HiGHS's default options. |
| `Highs::with_options(options)` | A backend with the given `HighsOptions`. |
| `options()`, `options_mut()` (`Configurable`, `Options = HighsOptions`) | The options, to read or to change between solves; `solver.options_mut()` reaches them through a `Solver`. |
| `Highs::EVENTS` | Every observational event: `LOG`, `PROGRESS`, `POLL`, `FEASIBLE`, `INCUMBENT` and `NATIVE`. |
| Search capabilities | `Heuristics` alone ([Heuristic solutions](#heuristic-solutions)); no `LazyConstraints`, `UserCuts` or `Prices` ([Limitations](#limitations)). |
| `Highs::CLASSES` | `Lp`, `Milp` and `Qp`; see [Quadratic programs](#quadratic-programs). |
| `culmen_highs::version()` | The version of the linked HiGHS; it displays as `1.16.x`. |

HiGHS is built from the sources bundled with `highs-sys`, with cmake and libclang, so the version is fixed by the
crate and needs no system install. The `Highs` backend is `Send` and can be moved to another thread to solve
there ([Threads](#threads)).

```rust
use culmen::prelude::*;
use culmen::highs::{HighsOptions, LpSolver, Switch};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = HighsOptions {
        parallel: Some(Switch::Off),
        mip_heuristic_effort: Some(0.2),
        ..HighsOptions::default()
    };

    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(4.0));
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + 3.0 * y).leq(6.0));
    m.maximize(3.0 * x + 2.0 * y);

    let mut solver = Solver::new(Highs::with_options(options), m.build()?)?;
    let first = solver.solve()?;

    // The options belong to the backend; change them between solves.
    solver.options_mut().solver = Some(LpSolver::Simplex);
    let second = solver.solve()?;
    assert_eq!(first.outcome.objective().map(f64::round), Some(13.0));
    assert_eq!(second.outcome.objective().map(f64::round), Some(13.0));
    Ok(())
}
```

## Options

`HighsOptions` is plain data. Each documented HiGHS option is a field named as HiGHS names it, `None` for HiGHS's
default: `presolve_mode`, `solver`, `parallel`, `run_crossover`, `ranging`, the tolerances
(`primal_feasibility_tolerance`, `dual_feasibility_tolerance`, `optimality_tolerance`, `kkt_tolerance`, …),
`objective_bound` and `objective_target`, the simplex options (`simplex_strategy`, `simplex_scale_strategy`, the edge
weights, the iteration and update limits, `simplex_max_concurrency`), the logging options (`output_flag`, `log_file`),
the MIP options (the leaf and stall limits, the cut-pool limits, `mip_heuristic_effort` and each heuristic's switch,
`mip_feasibility_tolerance`, `mip_lp_solver`, `mip_ipm_solver`, …), the interior-point and PDLP limits and
tolerances, the active-set QP solver's `qp_iteration_limit` and `qp_nullspace_limit`, `allow_unbounded_or_infeasible`
and `presolve_reduction_limit`. The rustdoc of `HighsOptions` lists every field. A `u64` field is a limit, and one
beyond HiGHS's integers is HiGHS's largest integer, its "no limit"; a `u32` field is an exact integer, such as
`simplex_max_concurrency`, and one beyond HiGHS's integers is refused.

What [`Settings`](solving.md#settings) types has no field: the time limit, the gaps, the thread count, the seed,
presolve, the node and solution limits and the console are set per solve, with the invariants their types hold. A
standing value across solves is an option by name, such as `.option("presolve", "off")` or
`.option("time_limit", 30.0)`, which holds for every solve whose settings leave it unset.

A switch HiGHS takes as `"on"` or `"off"`, as `ranging`, is a `bool`. A field whose HiGHS option takes one of a
few values is an enum:

| Type | Variants | Options |
|---|---|---|
| `Switch` | `Off`, `Choose`, `On` | `parallel`, `run_crossover` |
| `PresolveMode` | `Choose`, `InitialSweep`, `Light`, `Full` | `presolve_mode` |
| `LpSolver` | `Choose`, `Simplex`, `Ipm`, `Ipx`, `Hipo`, `Pdlp`, `Hipdlp`, `QpAsm` | `solver` |
| `MipLpSolver` | `Choose`, `Simplex`, `Ipm`, `Ipx`, `Hipo` | `mip_lp_solver` |
| `MipIpmSolver` | `Choose`, `Ipm`, `Ipx`, `Hipo` | `mip_ipm_solver` |
| `SimplexStrategy` | `Choose`, `Dual`, `DualSip`, `DualPami`, `Primal` | `simplex_strategy` |
| `ScaleStrategy` | `Off`, `Choose`, `Equilibration`, `ForcedEquilibration`, `MaxValue` | `simplex_scale_strategy` |
| `EdgeWeight` | `Choose`, `Dantzig`, `Devex`, `SteepestEdge` | `simplex_dual_edge_weight_strategy`, `simplex_primal_edge_weight_strategy` |

**Options by name.** Any other HiGHS option is set by its name with `HighsOptions::option(name, value)`, which
appends a `HighsOption { name, value }` to the `options` field. The value is an `OptionValue`, one of HiGHS's four kinds
(`Bool`, `Int`, `Double`, `String`), converted from a `bool`, `i32`, `f64`, `&str` or `String`. HiGHS checks the
name and the value when the solve applies them; an unknown name or a value of the wrong kind or range is a
[`HighsError::Option`](#errors). One option is refused for a MIP however it is set: `solve_relaxation`, which has
HiGHS solve the MIP's continuous relaxation and would report that optimum as the program's, fails the solve with
[`HighsError::Relaxation`](#errors) before HiGHS runs. Relax the program instead, with
[`Solver::set_col_integrality`](solving.md#editing-a-loaded-program).

```rust
use culmen::highs::HighsOptions;

let options = HighsOptions::default()
    .option("mip_pool_soft_limit", 5000)
    .option("log_file", "highs.log");
assert_eq!(options.options.len(), 2);
```

## How options and settings layer

Every solve sets HiGHS up from scratch, in this order:

1. HiGHS's options are reset to its defaults.
2. HiGHS's log is turned on and routed to the callback only: `output_flag` on, `log_to_console` off. Standard
   output stays quiet.
3. The backend's `HighsOptions` apply: the typed fields first, then the options by name, in the order added.
4. The solve's [`Settings`](solving.md#settings) apply.

So a per-solve setting always wins over the backend's options, and an option set by name wins over a typed field
for the same option or over the default a setting left unset. Nothing set on HiGHS carries over from one solve to the next except through `HighsOptions`.
The [Backends](backends.md#settings) page lists the HiGHS option each `Settings` field sets.

HiGHS measures `time_limit` on a clock the backend restarts before every run, so each solve gets the full limit. A
`Settings::seed` or `Settings::threads` beyond HiGHS's 32-bit integers is refused with
[`HighsError::OutOfRange`](#errors), as is a `u32` field of `HighsOptions`.

## Threads

- **`Highs` is `Send`**: move it, or the whole `Solver`, to a worker thread to solve there.
- **HiGHS keeps one task scheduler per thread**, started by the first run on that thread at that run's thread count. A
  later run on the same thread asking for another number of threads fails with [`HighsError::Run`](#errors). Leaving
  `Settings::threads` unset never conflicts; a solve on a fresh thread may pick any count.
- **Dropping a `Highs`** shuts down the scheduler of the dropping thread, so a later run there starts a new one at
  its own count.
- **A MIP uses every thread it is given.** Since HiGHS 1.16 the MIP solver explores its tree in parallel by default,
  on one thread per physical core the program may run on; `Settings::threads` caps it, and the `parallel` option set
  off keeps the tree search on one thread.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_vars(4, |_| Var::binary());
    m.add_constraint(LinExpr::sum(x.iter().copied()).leq(2));
    m.maximize(LinExpr::sum(x.iter().copied()));
    let mut solver = Solver::new(Highs::new(), m.build()?)?;

    let two = Settings { threads: Some(threads!(2)), ..Settings::new() };
    let worker = std::thread::spawn(move || solver.run().settings(two).solve());
    let solved = worker.join().map_err(|_| "the solve thread panicked")??;
    assert_eq!(solved.outcome.objective(), Some(2.0));
    Ok(())
}
```

## Statuses and outcomes

`Solved::status` is a `HighsStatus`, HiGHS's own account of the run beside the generic outcome:

| Field | Holds |
|---|---|
| `run` | What `Highs_run` returned: a `CallStatus`, `Ok`, `Warning` or `Error`. |
| `model` | The `ModelStatus`: every value of HiGHS's `HighsModelStatus`, and `Other(i32)` for a value HiGHS 1.16 does not define. |
| `primal`, `dual` | Whether a primal or dual solution exists: a `SolutionStatus`, `None`, `Infeasible` or `Feasible`. |
| `mip_gap` | HiGHS's relative MIP gap; `None` for an LP or when HiGHS reports none. |
| `simplex_iterations`, `ipm_iterations`, `qp_iterations` | The iterations of each solver, a MIP's simplex iterations over all its nodes. `Solved::iterations` sums every solver's, crossover and PDLP included, and `Solved::nodes` counts a MIP's nodes (`None` for a program without integer columns). |
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

    let solved = Solver::new(Highs::new(), m.build()?)?.solve()?;
    let status = &solved.status;
    assert_eq!((status.run, status.model), (CallStatus::Ok, ModelStatus::Optimal));
    assert_eq!(status.primal, SolutionStatus::Feasible);
    assert_eq!(status.mip_gap, None);
    assert_eq!(status.model.to_string(), "optimal");
    assert!((solved.outcome.objective().ok_or("no solution")? - 22.0).abs() < 1e-9);
    Ok(())
}
```

The generic [`Outcome`](solving.md#solved-and-outcome) is read from these; the
[Backends](backends.md#outcomes-and-statuses) page lists the outcome of every `ModelStatus`. Two of them are decided by
more than the status:

- **`Optimal` or `GapReached`.** HiGHS reports a MIP as optimal once a requested gap is reached. The backend reports
  `GapReached` when the gap between the solution and the bound is wider than HiGHS's defaults (a relative gap of
  `1e-4` and an absolute gap of `1e-6`), and `Optimal` otherwise.
- **`NodeLimit` or `SolutionLimit`.** HiGHS reports one status for its node, leaf and improving-solution limits. The
  backend reports `StopReason::NodeLimit` when the node limit HiGHS held for the run (`mip_max_nodes`, set in
  `Settings` or by name) was reached, and `StopReason::SolutionLimit` otherwise.

An empty model, one without columns, is `Optimal` at its offset when every row admits an activity of zero, and
`Infeasible` when a row such as `0 = 1` does not.

## Quadratic programs

HiGHS 1.16 solves LPs, MILPs and convex QPs, so `Highs::CLASSES` is `{LP, MILP, QP}`. It refuses a QP with integer
columns when it runs ("Cannot solve MIQP problems with HiGHS"), so a MIQP is refused by
[`Solver::new`](solving.md#model-classes-and-refusal) with `LoadError::UnsupportedClass` before HiGHS sees it; SCIP
[solves it](scip.md#quadratic-programs).

The backend passes the quadratic part as HiGHS reads it, the lower triangle of the Hessian `H` of `½ xᵀHx` column by
column ([The quadratic part](program.md#the-quadratic-part)): the model's `x * x` reaches HiGHS as `Hₓₓ = 2`, and
`x * y` as `Hᵧₓ = 1`. Every `Program` passed the convexity check when it was built, so HiGHS's own refusal of a
Hessian with a diagonal of the wrong sign never arises.

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

    let solved = Solver::new(Highs::new(), program)?.solve()?;
    let Outcome::Optimal(best) = solved.outcome else { return Err("no optimum".into()) };
    assert!((best.objective - 2.0).abs() < 1e-6);
    assert!((best.values[x] - 0.0).abs() < 1e-6 && (best.values[y] - 1.0).abs() < 1e-6);
    assert!(solved.status.qp_iterations > 0);
    Ok(())
}
```

HiGHS's QP solver differs from its LP and MIP solvers in what a solve can see and in how far its answer can be
trusted:

- **A poll, a first feasible point and the log.** HiGHS 1.16's active-set QP solver polls for an interrupt once its
  starting point is found and then at every iteration, and reports that starting point once. A QP solve therefore
  delivers `POLL`, with `HighsPollSource::Qp { iterations }`, so a [`Stop`](callbacks.md#stop) ends it as
  `Stopped { reason: Interrupted, .. }`; it delivers `FEASIBLE` once, for the starting point, valued by the backend
  since HiGHS reports no objective with it; and `LOG`. It delivers no `PROGRESS` and no `INCUMBENT`, so an observer
  sees no progress and no improvement. The time limit of the [`Settings`](solving.md#settings) applies as before.
- **A row is always passed.** On a QP without rows HiGHS 1.16 starts from the unconstrained minimizer clamped into
  the bounds, and 1.15 reported that point optimal after no iteration when it was not. The backend passes such a QP
  with one free row without entries, which constrains nothing and makes HiGHS find its starting point with its LP
  solver.
- **An unsound ending is an error.** On a small share of QPs HiGHS reports an optimum at a point holding NaN, an optimum
  its own primal and dual objectives contradict, at a point that need not be feasible, or reports unbounded a QP whose
  every column is bounded: 1.15 did all three after its iterates turned NaN, and 1.16 still reports the second for some
  edited QPs, though it solves every one of 5000 random QPs of the tests. The backend fails such a solve with
  [`HighsError::UnsoundQp`](#errors): an optimum at a point holding NaN, an optimum whose relative primal-dual objective
  error exceeds `HighsOptions::qp_objective_error`, `relative_gap!(1e-5)` unless set otherwise (correct optima stay
  below `1e-6`; a badly scaled QP may need a looser threshold), or unboundedness with every column bounded. An `Optimal` outcome of a QP is therefore one HiGHS's own measures support.
- **It may not terminate.** On a rare QP HiGHS 1.15's QP solver iterated without end, its memory growing; no test
  has seen 1.16 do so. A [time limit](solving.md#settings) ends such a solve as `Stopped { reason: TimeLimit, .. }`,
  and a `Stop` as `Stopped { reason: Interrupted, .. }`; set one for any QP whose solve must end.

## Semi columns, several objectives and starts

`Highs::FEATURES` is every [feature](program.md#features), and `Highs` implements `Starts`:

- **Semi columns** reach HiGHS as `kHighsVarTypeSemiContinuous` and `kHighsVarTypeSemiInteger`. HiGHS 1.16 lowers a
  semi column's upper bound above `1e5` to `1e5`, or refuses the model when its lower bound is above a tenth of that,
  and fails a run whose solution sits at the lowered bound (`assessSemiVariables` in `HighsLpUtils.cpp`). The backend
  refuses such a column at load instead, with `HighsError::SemiUpper { col, upper }`; `Highs::SEMI_UPPER_LIMIT` is
  the limit.
- **Several objectives** reach HiGHS as linear objectives, the program's own first, through
  `Highs_passLinearObjectives`; each solve sets `blend_multi_objectives` by the combination. Lexicographically, HiGHS
  optimizes by descending priority, each objective minimized for a positive weight and maximized otherwise, and after
  each stage adds the row keeping it within `min(abs, rel · |f*|)` of its optimum `f*`, leaving out a negative
  tolerance: the backend gives each objective the weight `±1` by its sense, its tolerances, `-1` for an absent one, and
  a priority by its place. Blended, HiGHS minimizes `Σ weight · objective`: each weight is the culmen weight signed by
  its sense. HiGHS refuses a further objective beside a Hessian, as culmen does.

  A multi-objective solve leaves HiGHS's own objective value at zero after a lexicographic solve and at the blend's
  otherwise (`returnFromLexicographicOptimization` in `HighsInterface.cpp`), and its MIP bound is that of the last
  stage or the blend: the backend reports the program's own objective, computed from the solution, and no bound, so a
  MIP's outcome is `Optimal` or `Stopped`. Objective values HiGHS reports to callbacks during such a solve are those
  of the stage it is solving or of the blend.
- **A start**, given to `Run::start`, reaches HiGHS through `Highs_setSparseSolution` before the run, a full one with
  every column. HiGHS refuses no value for lying outside its column's bounds (`Highs::setSolution` in `Highs.cpp`), so
  a semi column's zero is kept, and leaves the other columns of a sparse start open; for a MIP it completes them by a
  sub-MIP over the columns left open. A feasible start of a MIP is the first incumbent HiGHS reports. HiGHS also starts from the
  last solve's solution when a solve gives none, or a sparse start without entries.

## Model files

`culmen_highs::files` reads and writes MPS and LP files with HiGHS's own readers and writers. `files::read(path)` and
`files::write(&program, path)` pick the `ModelFormat`, `Mps` or `Lp`, by the extension, `.mps` or `.lp` in any case;
`files::read_as` and `files::write_as` take it explicitly, through a temporary file of the right extension when the
path has another. Reading gives a checked `Program`, every column and row named as in the file and in the file's
order; writing then reading gives the same program up to names and that order, and the same optimum. The one
exception is a ranged row in an LP file, which HiGHS writes, and so reads back, as two rows named with `lo` and `up`
appended.

```rust
use culmen::prelude::*;
use culmen::highs::files::{self, FileError, ModelFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().upper(4).name("x"));
    let y = m.add_var(Var::semi_continuous().bounds(1, 3).name("y"));
    m.add_constraint(("cap", (x + 2 * y).leq(6)));
    m.maximize(3 * x + 2 * y);
    let program = m.build()?;

    let path = std::env::temp_dir().join(format!("culmen-highs-guide-{}.mps", std::process::id()));
    files::write(&program, &path)?;
    let read = files::read(&path)?;
    std::fs::remove_file(&path)?;
    assert_eq!(read.columns()[x].name.as_deref(), Some("x"));

    let best = |p: Program| -> Result<Outcome, Box<dyn std::error::Error>> {
        Ok(Solver::new(Highs::new(), p)?.solve()?.outcome)
    };
    assert_eq!(best(read)?, best(program)?);

    assert!(matches!(files::read("plan.json"), Err(FileError::Extension { .. })));
    assert_eq!(ModelFormat::of_path("plan.LP".as_ref()), Some(ModelFormat::Lp));
    Ok(())
}
```

`FileError` says why:

| Variant | When |
|---|---|
| `Extension { path }` | The extension names no format. |
| `Path { path }` | The path is not UTF-8 or holds a NUL byte. |
| `Io { path, kind }` | The file system failed, as for a missing file. |
| `Read { path, format }` | HiGHS could not parse the file in the format. |
| `Write { path, format }` | HiGHS could not write the file. |
| `ColBounds { col, source }`, `RowBounds { row, source }` | Bounds in the file are no interval. |
| `VarType { col, raw }` | A column has a HiGHS variable type with no `Integrality`. |
| `Program(ProgramError)` | The content fails a program's checks, such as a non-convex objective. |
| `UnsupportedFeature { format, feature }` | The format, as HiGHS writes it, loses a feature: further objectives, in either format, since neither reader takes them. |
| `Name { name }` | A name holds a NUL byte. |
| `Highs(HighsError)` | HiGHS refused the program. |

## Editing a loaded program

`Highs` changes its model in place when a [`Solver` edits its program](solving.md#editing-a-loaded-program), so HiGHS
keeps the basis of its last solve:

| Edit | HiGHS calls |
|---|---|
| Column bounds | `Highs_changeColsBoundsBySet` |
| Costs | `Highs_changeColsCostBySet`; with several objectives, `Highs_passLinearObjectives` again, since HiGHS does not read the costs into them |
| Integrality | `Highs_changeColsIntegralityBySet` |
| Row bounds | `Highs_changeRowsBoundsBySet` |
| New rows | `Highs_addRows` |
| New columns | `Highs_addCols`, then `Highs_changeColsIntegralityByRange` for any that is not continuous; with several objectives, `Highs_passLinearObjectives` again, since HiGHS does not extend them |
| Sense, offset | `Highs_changeObjectiveSense`, `Highs_changeObjectiveOffset`; with several objectives, `Highs_passLinearObjectives` again |
| A replaced objective (`set_objective`) whose quadratic part stays as it was | The sense, the offset and every cost as above |

Three edits load the program whole instead: a replaced objective whose quadratic part changed, rows added to a QP
without rows, which HiGHS holds with a free row ([Quadratic programs](#quadratic-programs)), and any kind of edit a
later culmen adds before this backend knows it. A new Hessian could be passed in place, but a QP without rows needs
the free row added or dropped with it, so the load keeps that one rule in one place. A
semi column whose upper bound would exceed `Highs::SEMI_UPPER_LIMIT` is refused with `HighsError::SemiUpper` before
HiGHS sees the edit, but the refusal comes from the backend: the `Solver` then loads the program before the edit
again, which costs a full load, and the last solve's result is no longer read. A call HiGHS refuses is
`HighsError::Edit { call }`, naming it; HiGHS 1.15 could fail halfway through adding columns, after it extended
some of its arrays, so the `Solver` then loads the program before the edit again too.

Appending columns costs time in the size of the whole matrix on both sides: the program inserts each new column's
entries at the ends of their rows, and the backend gathers the new columns' entries from every row. So append the
columns of one round of pricing in one `add_cols` call, not one `add_col` each.

What warms the next solve is the LP basis. After a cut or a bound change, the dual simplex starts from the last
optimal basis: on the LP relaxation of a 300-column assignment problem with presolve off, a violated cut costs 2
simplex iterations where a fresh solve takes 103, and fixing five columns at zero 2 where a fresh solve takes 97. A
MIP gains less: HiGHS drops its solution and its MIP search on any change of the model, so a MIP solve after an edit
searches afresh, its root LP warmed by the basis; to start it from the last incumbent, hand that as a
[start](solving.md#starts). Every solve after an edit also loses "the last solve's solution" HiGHS otherwise starts
from.

## Errors

`HighsError` says why loading or solving failed:

| Variant | When |
|---|---|
| `NotLoaded` | A solve was asked for before a program was loaded. |
| `TooLarge { what, count }` | The program has more columns, rows, entries or Hessian entries (`what`, a `Dimension`) than HiGHS's 32-bit indices hold. |
| `Load` | HiGHS refused the program, as it refuses a matrix value at or above its `large_matrix_value` (`1e15`); its log is off outside a run, so the reason is not kept. |
| `Option { name, value }` | HiGHS refused an option: an unknown name, or a value of the wrong kind or out of range. |
| `SemiUpper { col, upper }` | A semi column's upper bound exceeds `Highs::SEMI_UPPER_LIMIT`; see [Semi columns, several objectives and starts](#semi-columns-several-objectives-and-starts). |
| `Start` | HiGHS refused a start. |
| `Edit { call }` | HiGHS refused an edit of its model; `call` names the C function ([Editing a loaded program](#editing-a-loaded-program)). |
| `OptionName { name }` | An option name holds a NUL byte, which HiGHS's C strings cannot carry. |
| `OptionString { name }` | A string option's value holds a NUL byte. |
| `OutOfRange { name, value, max }` | An exact integer setting or option (a seed, a thread count, a `u32` field of `HighsOptions`) is above `max`, `i32::MAX`, beyond HiGHS's integers. |
| `Run { status, message }` | The run failed, as a thread-count conflict makes it; `message` is HiGHS's last error line. |
| `UnsoundQp(UnsoundQp)` | HiGHS's QP solver reported an ending its own measures or the model contradict: `NotANumber`, `ObjectiveError { error }` or `UnboundedInABox` ([Quadratic programs](#quadratic-programs)). |
| `Analysis(Analysis)` | HiGHS refused or failed an analysis, as a basis it finds singular ([Analysis](analysis.md#highs)). |
| `AnalysisFeature { analysis, feature }` | HiGHS 1.16 gets the analysis wrong for a program with the feature, as a feasibility relaxation with semi columns ([Analysis](analysis.md#highs)). |
| `Relaxation` | The options set `solve_relaxation` for a MIP, which would report the relaxation's optimum as the program's ([Options](#options)). |

```rust
use culmen::prelude::*;
use culmen::highs::{HighsError, HighsOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);
    let options = HighsOptions { log_file: Some("a\0b".to_owned()), ..HighsOptions::default() };
    let mut solver = Solver::new(Highs::with_options(options), m.build()?)?;

    let Err(SolveError::Backend(error)) = solver.solve() else { panic!("the option is refused") };
    assert_eq!(error, HighsError::OptionString { name: "log_file".to_owned() });
    assert_eq!(error.to_string(), "HiGHS: the value of option `log_file` holds a NUL byte");
    Ok(())
}
```

A run that ends without a solution is not an error but an outcome; a model status that means HiGHS failed
(`LoadError`, `ModelError`, `PresolveError`, `SolveError`, `PostsolveError`, `NotSet`) is `HighsError::Run`.

## Callback contexts

A `Callback<Highs>` and a `HeuristicHandler<Highs>` receive HiGHS's own context types. Each implements the neutral
[context trait](callbacks.md#contexts) of its event and offers what HiGHS reports beyond it:

| Event or handler | Context | HiGHS callback | Beyond the trait |
|---|---|---|---|
| `LOG` | `HighsLog` | `kHighsCallbackLogging` | `log_type()`: a `HighsLogType`, `Info`, `Detailed`, `Verbose`, `Warning` or `Error` |
| `PROGRESS` | `HighsProgress` | `kHighsCallbackMipLogging`, when HiGHS writes a MIP log line | `mip()`: the `HighsMipData` |
| `POLL` | `HighsPoll` | the simplex, interior-point, MIP and QP interrupt callbacks | `source()`: a `HighsPollSource`, `Simplex { iterations }`, `Ipm { iterations }`, `Mip(HighsMipData)` or `Qp { iterations }` |
| `FEASIBLE` | `HighsSolution` | `kHighsCallbackMipSolution`, and `kHighsCallbackQpFirstFeasiblePoint` once per QP solve | `mip()`: the `HighsMipData`, `None` for the QP's point |
| `INCUMBENT` | `HighsSolution` | `kHighsCallbackMipImprovingSolution` | `mip()` |
| `HeuristicHandler` | `HighsHeuristic` | `kHighsCallbackMipUserSolution`, started only for a solve with a heuristic handler | `origin()`, a `HighsQueryOrigin`; `mip()`; `supply_partial(&entries)` |
| `NATIVE` | `HighsNative` | `kHighsCallbackMipGetCutPool` | the [cut pool](#the-cut-pool) |
| lazy constraints, cuts, pricing | none | none | `Highs` implements none of their capabilities, so their handlers do not compile ([Limitations](#limitations)) |

`HighsMipData` is what every MIP callback reports, in the model's objective space, offset included:

| Field | Holds |
|---|---|
| `running_time` | The MIP solver's own clock, which starts after presolve. |
| `objective` | The objective HiGHS reports with the call; `None` before there is one. |
| `primal_bound`, `dual_bound` | The best objective found and the best bound proved; `None` before there is one. |
| `gap` | HiGHS's relative gap; `None` while it is infinite. |
| `nodes`, `lp_iterations` | Branch-and-bound nodes, and LP iterations over all nodes. |

A context's `progress()` carries the elapsed time since the solve began; a poll's phase is `Simplex`, `Barrier`,
`Mip` or, for the QP solver, `Other` after its source. A solution HiGHS reports with another number of values than the
program has columns is not delivered.

The interrupt and logging callbacks are always installed, whatever the callback asks for: the polls so that a panic
can stop the run, the log so that a failed run can name HiGHS's last error line. Calls are serialised behind one
lock; a log line HiGHS writes while `supply_partial` repairs a solution re-enters the same run and is not delivered,
since the heuristic handler is busy. A solve of another `Highs` started from inside a callback gets its own callbacks.

```rust
use culmen::prelude::*;
use culmen::highs::{HighsLog, HighsLogType, HighsPoll, HighsPollSource};

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
        if let HighsPollSource::Mip(mip) = ctx.source() {
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
    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let solved = solver.run().callback(&mut inspect).solve()?;
    assert_eq!(solved.outcome.objective(), Some(25.0));
    assert!(inspect.warnings.is_empty());
    Ok(())
}
```

A method may name the concrete context type, `&mut HighsPoll<'_>`, or the associated one, `&mut <Highs as
Backend>::Poll<'_>`; they are the same type.

## Heuristic solutions

HiGHS implements `Heuristics`: at its user-solution callback it asks a registered `HeuristicHandler<Highs>` for a
solution of its own. `HighsHeuristic::origin()` says where in the
search it asks: `HighsQueryOrigin::AfterSetup`, once the MIP solver is set up; `BeforeDive`; `EvaluateRootNode(point)`,
during root-node evaluation, `point` a `HighsRootNodePoint` naming HiGHS's numbered point; or `Other(i32)` for a value
HiGHS 1.16 does not define.

- `ctx.supply(&values)` offers a value for every column. The values are checked first: one finite value per column,
  or `ActionError::Length` or `ActionError::NonFinite`. HiGHS then checks the solution against the model and keeps
  it as its incumbent if it is feasible and better.
- `ctx.supply_partial(&entries)`, an inherent method of `HighsHeuristic` since not every solver completes a partial
  solution, offers values for some columns, and HiGHS completes them into a feasible solution if it can. It returns `ActionError::Refused` for no entries, for a column named twice, for a value outside its
  column's bounds or integrality, and when HiGHS finds no feasible completion. A refused call leaves what an earlier
  call at the same query supplied.

HiGHS calls neither solution callback for a solution it takes this way, so the backend reports it itself: once a
MIP poll shows the supplied solution's objective as HiGHS's best, it arrives as `INCUMBENT`, and so reaches the
[observer](callbacks.md#observer) too.

```rust
use culmen::prelude::*;
use culmen::highs::{HighsHeuristic, HighsQueryOrigin};
use culmen::{ActionError, TiVec};

struct Start {
    values: Option<TiVec<Col, f64>>,
    supplied: Vec<Result<(), ActionError>>,
}

impl HeuristicHandler<Highs> for Start {
    fn heuristic(&mut self, ctx: &mut HighsHeuristic<'_>) {
        if ctx.origin() == HighsQueryOrigin::AfterSetup
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

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let mut handler = Start { values: Some(start), supplied: Vec::new() };
    let no_presolve = Settings { presolve: Some(false), ..Settings::new() };
    let solved = solver.run().settings(no_presolve).heuristic(&mut handler).solve()?;
    assert_eq!(handler.supplied, [Ok(())]);
    assert_eq!(solved.outcome.objective(), Some(25.0));
    Ok(())
}
```

## The cut pool

At `NATIVE`, HiGHS reports its cut pool after the root node, and again after each restart, as
`HighsNative::CutPool(pool)`. The pool is copied out of HiGHS, so it may be kept:

| Field | Holds |
|---|---|
| `num_cols` | The number of columns of the presolved problem the cuts range over. |
| `cuts` | Each `HighsCut { lower, upper, entries }`: `lower <= Σ value · column <= upper`, an open end infinite. |
| `mip` | The `HighsMipData` at the time. |

The cuts range over the presolved problem HiGHS's MIP solver works on, not over the program you loaded, so their columns
are `ReducedCol`, a distinct handle type: a cut cannot be read as a row of your program by mistake. `HighsNative` is
`#[non_exhaustive]`, so a `match` on it needs a `_` arm.

## Limitations

- **No lazy constraints.** HiGHS 1.16 declares `kHighsCallbackMipDefineLazyConstraints` and never calls it, so
  `Highs` does not implement `LazyConstraints`, and registering a `LazyHandler` for it does not compile.
  [SCIP](scip.md#lazy-constraints) takes them.
- **No user cuts.** HiGHS has no callback for them, so `Highs` does not implement `UserCuts`. [SCIP](scip.md#cuts)
  takes them.
- **No pricing.** HiGHS has no pricing callback, so `Highs` does not implement `Prices` and a `Pricer` for it does not
  compile: price in an outer loop by [editing](#editing-a-loaded-program) the program, as the `column_generation` example does, or at every
  node on [SCIP](scip.md#pricing).
- **An interrupt is honoured only at a poll.** Only `HighsPoll` has `interrupt`, and only `HighsHeuristic` accepts
  a solution.
- **One thread count per thread.** See [Threads](#threads).
- **32-bit indices.** A program with more than `i32::MAX` columns, rows, entries or Hessian entries is refused with
  `HighsError::TooLarge`.
- **No MIQP, and a QP solve sees no progress or improvement.** See [Quadratic programs](#quadratic-programs); SCIP
  solves MIQPs ([SCIP](scip.md#quadratic-programs)).
- **Options do not persist inside HiGHS.** Each solve resets them; set them through `HighsOptions`.
- **Semi columns up to `1e5`, and several objectives linear only.** See [Semi columns, several objectives and
  starts](#semi-columns-several-objectives-and-starts).

## See also

- [Callbacks](callbacks.md): the neutral events and contexts.
- [Analysis](analysis.md#highs): duals, basis, ranging, rays, IIS and feasibility relaxation on HiGHS.
- [Solving](solving.md): the run of a solve, settings and outcomes.
- [SCIP](scip.md): the second backend, for MIQPs, lazy constraints, cuts and pricing.
- [Backends](backends.md): how HiGHS's callbacks, options and statuses map onto culmen's seam, beside Gurobi and SCIP.
- [Design](design.md#where-unsafe-lives): why the trampoline is sound.
- [Documentation index](README.md).
