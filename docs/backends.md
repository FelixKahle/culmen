# Backends

culmen_core does not know any solver. A backend is a crate that implements `culmen_core::Backend` for its solver:
it declares the model classes it solves, loads a `Program`, applies the generic `Settings` over its own typed options,
drives the solver's callbacks as culmen's generic events, and maps the solver's statuses to the generic `Outcome`,
keeping its own status beside it.
HiGHS (`culmen_highs`) and SCIP (`culmen_scip`) are the two backends. This page shows that the seam fits the serious
MIP solvers: each core concept is listed with what HiGHS 1.16 and SCIP 10 offer for it as their backends use it, and
what Gurobi 12 would offer a backend written later.

## The seam

```rust,ignore
pub trait Backend: Sized {
    type Error: core::error::Error + Send + Sync + 'static;
    type Status: Debug + Clone + PartialEq + Send + 'static; // the solver's own status, kept for diagnostics
    type Log<'a>: LogContext;             // Events::LOG
    type Progress<'a>: ProgressContext;   // Events::PROGRESS
    type Poll<'a>: PollContext;           // Events::POLL
    type Solution<'a>: SolutionContext;   // Events::FEASIBLE and Events::INCUMBENT
    type Lazy<'a>: LazyContext;           // Events::LAZY
    type Cut<'a>: CutContext;             // Events::CUT
    type Heuristic<'a>: HeuristicContext; // Events::HEURISTIC
    type Price<'a>: PriceContext;         // Events::PRICE
    type Native<'a>: Debug;               // Events::NATIVE: whatever only this solver reports
    const EVENTS: Events;                 // what this backend delivers
    const CLASSES: ModelClasses;          // the model classes this backend solves
    const ANALYSES: Analyses = Analyses::NONE; // what it reports after a solve: duals, basis, ranging, …
    const FEATURES: ProgramFeatures = ProgramFeatures::NONE; // semi columns, objectives, starts it takes
    fn load(&mut self, program: &Program) -> Result<(), Self::Error>;
    fn edit(&mut self, program: &Program, edit: &Edit) -> Result<(), Self::Error> { self.load(program) }
    fn set_start(&mut self, start: &Start) -> Result<(), Self::Error> { Ok(()) } // only with StartSolution
    fn solve(&mut self, settings: &Settings, callback: &mut dyn Callback<Self>, priced: &mut Priced)
        -> Result<Solved<Self::Status>, Self::Error>;
}
```

- **Capabilities are typed and queryable.** A backend lists the events it delivers in `Backend::EVENTS`. For an event
  it cannot deliver it names `culmen_core::Unsupported` as the context type: an enum with no values, so a callback
  method for that event compiles but can never run. HiGHS, for instance, names `Unsupported` for `Lazy` and `Cut`.
- **Model classes are declared.** A backend lists the classes it solves (`ModelClass::Lp`, `Milp`, `Qp`, `Miqp`)
  in `Backend::CLASSES`. `Solver::new` reads it before `load`, for every backend alike, and refuses any other class
  with `LoadError::Unsupported { class, supported }`, so `load` never re-checks the class. Features beyond the class
  (semi columns, multiple objectives, starts) are declared alike in `Backend::FEATURES` and refused with
  `LoadError::UnsupportedFeature` or, for a start, `SolveError::UnsupportedFeature`.
- **Analyses are declared.** A backend lists what it reports after a solve in `Backend::ANALYSES` and implements the
  matching methods (`duals`, `basis`, `set_basis`, `ranging`, `primal_ray`, `dual_ray`, `iis`, `relax`), whose
  default bodies refuse. The `Solver` checks the declaration, the program's class, the fit of a basis or weights
  and the solve read before it asks, for every backend alike ([Analysis](analysis.md)).
- **Actions are typed per event.** Only `PollContext` has `interrupt`, only `LazyContext` has `add_lazy`, only
  `CutContext` has `add_cut`, only `HeuristicContext` has `supply` and `supply_partial`, only `PriceContext` has
  `add_col`, `branch_early` and `report_bound`. An action that a solver
  honours only at certain points exists only on the context of those points.
- **Extras stay with the backend.** Each context type is the backend's own struct. What the solver reports beyond
  the neutral trait is an inherent method on it (`HighsPoll::source`, `HighsSolution::mip`) or a native event
  (`HighsNative::CutPool`). Generic code (`impl<B: Backend> Callback<B> for …`) sees the traits, and backend code
  (`impl Callback<Highs> for …`) sees everything.
- **The convenient path sits on top.** `Watch` (made from a `Request` of settings, a `Stop` and an `Observer`) is an
  ordinary `impl<B: Backend> Callback<B>`: it polls the stop at `POLL`, throttles `PROGRESS`/`POLL` into
  `Observer::progress`, reports `INCUMBENT` as `Observer::improved`, and joins `LOG` chunks into lines. It works
  unchanged on every backend, and pairs with a backend-specific callback as `(watch, mine)`.
- **Edits are a hook with a correct default.** A `Solver` that [edits its program](solving.md#editing-a-loaded-program)
  calls `edit` with the edited program and an `Edit` naming what changed: the columns whose bounds, costs or
  integrality changed, the rows whose bounds changed, or the rows or columns appended. Only a `Solver` makes an
  `Edit`, and its variants are non-exhaustive, so a backend matches it (`Edit::ColBounds { cols, .. }`) and never
  forges one. It calls `edit` only right
  after the backend held the program before the edit, and only with a class and features the backend declares. The
  default loads the program whole; a backend overrides it to change its model in place and keep its warm start, and
  falls back to `load` for any kind of `Edit` it does not handle, since the enum is non-exhaustive. When `edit`
  fails, the backend may hold anything: the `Solver` loads the program before the edit again.
- **Priced columns go to a sink the caller owns.** `solve` takes a `Priced`, made for the loaded program. A backend
  that delivers `PRICE` hands every column a callback adds to `Priced::push`, which checks it (rows the program has,
  finite numbers, bounds that hold zero, no semi column, a class the backend solves) and returns its id, the next
  after the program's columns and every one priced before. The core owns the checks and the numbering, so every
  backend refuses the same columns the same way, and a backend that declared `PRICE` but forgot the sink would not
  compile. A `Solver` appends the columns to its program however `solve` ends, unwinding included, and loads the
  grown program at its next call; code that drives a backend directly makes its own `Priced` and takes the columns
  with `Priced::into_cols`. A backend that does not price ignores the parameter.
- **The contract** (on `Backend`): solve the last program `load` or `edit` accepted; report every objective and
  bound with the program's quadratic part and offset in the model's sense, and every point with one value per
  column, the ones priced so far included; call only events in `EVENTS ∩ callback.events()`, only during `solve`, never two at once; a panic in a
  callback stops the solve and unwinds out of `solve`, never aborts. A backend that declares `PRICE` delivers it
  only for a solved node LP, optimal (`PriceMode::ReducedCost`) or infeasible (`PriceMode::Farkas`), ends pricing at
  a node only after a round that added nothing and did not branch early, and keeps every added column for the rest
  of the solve.

`Script`, the in-memory backend in culmen_core, delivers every generic event from a scripted list. Its tests drive
lazy constraints, cuts, heuristic solutions, pricing, native events, the observer, stops and panics without any
solver.
That is evidence that nothing in the core assumes HiGHS.

## Model classes

| Class | HiGHS 1.16 (`culmen_highs`) | Gurobi 12 | SCIP 10 (`culmen_scip`) |
|---|---|---|---|
| `Lp` | yes | yes | yes |
| `Milp` | yes | yes | yes |
| `Qp` | yes: `Highs_passModel` with the lower triangle of `H` in `½ xᵀHx`, column-wise | yes: `GRBaddqpterms`, whose `qval` is the coefficient of `xᵢ · xⱼ` as culmen stores it | yes: an epigraph variable `t` with cost one, bounded by the quadratic constraint `q(x) − t ≤ 0` (`≥ 0` when maximizing), `SCIPcreateConsBasicQuadraticNonlinear` |
| `Miqp` | no: `Highs_run` refuses it ("Cannot solve MIQP problems with HiGHS") | yes | yes, alike, with `constraints/nonlinear/reformbinprods` off |

A `Program`'s quadratic part is canonical and convex in its sense
([The program as data](program.md#the-quadratic-part)), so a backend converts it without checking it again: HiGHS
takes `Hᵢᵢ = 2q` on the diagonal and `q` below it; Gurobi takes the entries as they are; SCIP takes them as the terms of
its quadratic constraint. Quadratic constraints, which
Gurobi and SCIP solve and HiGHS does not, would be further classes that a backend declares or leaves out.

`Script` declares every class, since it replays its script rather than solving.

## Features

| Feature | HiGHS 1.16 (`culmen_highs`) | Gurobi 12 | SCIP 10 (`culmen_scip`) |
|---|---|---|---|
| `SemiContinuous` | yes: `kHighsVarTypeSemiContinuous`, upper bound at most `1e5` (`Highs::SEMI_UPPER_LIMIT`) | yes: `GRB_SEMICONT` | no: SCIP has no semi variable type, and `culmen_scip` does not model one (a binary indicator and two rows would) |
| `SemiInteger` | yes: `kHighsVarTypeSemiInteger`, alike | yes: `GRB_SEMIINT` | no, alike |
| `MultipleObjectives` | yes: `Highs_passLinearObjectives`, lexicographic by priority or blended by `blend_multi_objectives`; linear only | yes: `GRBsetobjectiven`, priorities, weights and tolerances | no: SCIP has one objective |
| `StartSolution` | yes: `Highs_setSparseSolution`, HiGHS completing the columns a sparse start leaves out | yes: the `Start` attribute, `GRB_UNDEFINED` for a partial one | yes: `SCIPcreateSol` and `SCIPaddSolFree`; a sparse start `SCIPcreatePartialSol`, completed by SCIP's `completesol` heuristic |

`Script` declares every feature and records each start in `record.starts`.

## Events

| Core concept | HiGHS 1.16 (`culmen_highs`) | Gurobi 12 | SCIP 10 (`culmen_scip`) |
|---|---|---|---|
| `LOG`: a log chunk, its level | `kHighsCallbackLogging`: message, `log_type` (only field set) | `GRB_CB_MESSAGE`: `GRB_CB_MSG_STRING` | the instance's message handler (`SCIPmessagehdlrCreate`), buffered to whole lines: info messages as `Info`, warnings as `Warning` |
| `PROGRESS`: the solver's own pace | `kHighsCallbackMipLogging` (fires when a MIP log line is printed) | `GRB_CB_MIP`: `MIP_OBJBST`, `MIP_OBJBND`, `MIP_NODCNT`, `MIP_ITRCNT`; `GRB_CB_SIMPLEX`, `GRB_CB_BARRIER` for LPs | the event handler at each poll below, with `SCIPgetPrimalbound`, `SCIPgetDualbound`, `SCIPgetNTotalNodes`, `SCIPgetNLPIterations` |
| `POLL`: frequent, may interrupt | `kHighsCallbackSimplexInterrupt`, `kHighsCallbackIpmInterrupt`, `kHighsCallbackMipInterrupt`: `user_interrupt = 1` | `GRB_CB_POLLING` and every other `where`: `GRBterminate` | an event handler on `SCIP_EVENTTYPE_PRESOLVEROUND`, `NODEFOCUSED`, `LPSOLVED`, `NODESOLVED`: `SCIPinterruptSolve` |
| `Progress` fields | `running_time`, `mip_primal_bound`, `mip_dual_bound`, `mip_node_count`, `mip_total_lp_iterations`, `simplex_iteration_count`, `ipm_iteration_count` | `cbGet` codes above, `GRB_CB_RUNTIME` | the wall clock of the solve and the getters above, each in the stages SCIP reports it |
| `FEASIBLE`: any new solution | `kHighsCallbackMipSolution`: `mip_solution`, `objective_function_value` | `GRB_CB_MIPSOL`: `MIPSOL_SOL`, `MIPSOL_OBJ` | the event handler on `SCIP_EVENTTYPE_SOLFOUND`, and the solutions stored before it caught events (starts, the last solve's best) |
| `INCUMBENT`: a better solution | `kHighsCallbackMipImprovingSolution`; an accepted user solution is reported at the next MIP poll, as HiGHS skips its solution callbacks for it | `GRB_CB_MIPSOL` when `MIPSOL_OBJ` beats `MIPSOL_OBJBST` | `SCIP_EVENTTYPE_BESTSOLFOUND` |
| `LAZY`: cut off a candidate | none: `kHighsCallbackMipDefineLazyConstraints` exists in 1.16 but is never called, so `Unsupported` | `GRB_CB_MIPSOL` (and `MIPNODE`): `GRBcblazy`, with `LazyConstraints = 1` | a constraint handler without constraints, checked and enforced last: `CONSCHECK` rejects, `CONSENFOLP` / `CONSENFOPS` / `CONSENFORELAX` add global linear constraints (`SCIPaddCons`, `CONSADDED`) |
| `CUT`: cut off a node relaxation | none, so `Unsupported` | `GRB_CB_MIPNODE` with `MIPNODE_STATUS == OPTIMAL`: `GRBcbget(MIPNODE_REL)`, `GRBcbcut` | a separator (`SEPAEXECLP`): `SCIPgetSolVal` of the LP, `SCIPaddRow` and `SCIPaddPoolCut` |
| `HEURISTIC`: supply a solution | `kHighsCallbackMipUserSolution`: `user_solution` + `user_has_solution`; partial via `Highs_setCallbackSparseSolution` + `Highs_repairCallbackSolution` | `GRB_CB_MIPNODE` (and `MIP`): `GRBcbsolution` (partial values allowed), `GRBcbusesolution` | a primal heuristic (`HEUREXEC`) before and after each node: `SCIPcreateOrigSol` (`SCIPcreateSol` in a solve that prices), `SCIPtrySolFree`; no partial solution during the search, so `supply_partial` is refused |
| `PRICE`: answer a node LP's duals with columns | none, so `Unsupported` | none: Gurobi 12 has no pricing callback | a pricer plugin, delayed and active only in a solve that asks for `PRICE`, whose rows are then modifiable (`SCIPincludePricerBasic`, `SCIPactivatePricer`, `SCIPsetConsModifiable`): `PRICERREDCOST` and `PRICERFARKAS` read `SCIPgetDualsolLinear` or `SCIPgetDualfarkasLinear`, then `SCIPcreateVar`, `SCIPaddPricedVar` and `SCIPaddCoefLinear`; refused on a program with a quadratic objective |
| `NATIVE`: solver-only events | `kHighsCallbackMipGetCutPool` as `HighsNative::CutPool` (presolved-space cuts) | `GRB_CB_PRESOLVE`, `GRB_CB_MULTIOBJ`, `GRB_CB_IIS`, and `MIPNODE` node data beyond the relaxation | none in `culmen_scip`; any other event type (`SCIP_EVENTTYPE_*`), branching or node-selection data would fit |

HiGHS's threading is handled inside culmen_highs. The MIP kinds come from its serial search loop. Logging may come
from task-pool threads. The trampoline serialises all calls behind a mutex, so callbacks are `Send` and the stop is
`Sync`. Gurobi calls back on the thread that called `GRBoptimize`, and SCIP plugins run on the solving thread, so a
backend for either may call the callback directly, as `Script` does; the bounds on `Callback` and `Stop` already
allow a solver that calls back from other threads.

## Settings

| `Settings` field | HiGHS 1.16 option | Gurobi 12 parameter | SCIP 10 parameter (`culmen_scip`) |
|---|---|---|---|
| `time_limit` | `time_limit` (clock zeroed per run) | `TimeLimit` | `limits/time` (SCIP's clock restarts with each solve) |
| `relative_gap` | `mip_rel_gap` | `MIPGap` | `limits/gap` |
| `absolute_gap` | `mip_abs_gap` | `MIPGapAbs` | `limits/absgap` |
| `threads` | `threads` (per-thread scheduler; a conflicting count is an error, and one above `i32::MAX` is refused) | `Threads` | none: `SCIPsolve` is sequential, and SCIP's concurrent solve would run copies without culmen's plugins, so the setting has no effect |
| `seed` | `random_seed` (one above `i32::MAX` is refused, not clamped) | `Seed` | `randomization/randomseedshift` (one above `i32::MAX` is refused) |
| `presolve` | `presolve` = `"on"` / `"off"` | `Presolve` = `-1` / `0` | `SCIPsetPresolving`: `OFF`, or the options' presolving emphasis, else `DEFAULT` |
| `node_limit` | `mip_max_nodes` | `NodeLimit` | `limits/nodes` (one above `i64::MAX` is refused) |
| `solution_limit` | `mip_max_improving_sols` | `SolutionLimit` | `limits/solutions` (one above `i32::MAX` is refused) |
| `console` | `log_to_console` (the log always reaches the callback) | `LogToConsole` (with `OutputFlag = 1`) | the message handler writes to standard output and standard error too (the log always reaches the callback) |

Every backend also has typed options for the rest (`culmen_highs::HighsOptions` covers HiGHS's documented
options as typed fields) and a raw name/value escape hatch (`HighsOptions::option`); `culmen_scip::ScipOptions` has
SCIP's three emphasis settings as typed fields and every other parameter by name (`ScipOptions::param`, set with
`SCIPsetIntParam` and friends). A Gurobi backend would do the same with `GRBsetintparam` / `GRBsetdblparam` /
`GRBsetstrparam`.
`Settings` apply after the backend's options, so a per-solve setting always wins.

## Outcomes and statuses

| `Outcome` | HiGHS 1.16 `ModelStatus` | Gurobi 12 status | SCIP 10 `SCIP_STATUS` (`culmen_scip`) |
|---|---|---|---|
| `Optimal` | `Optimal` with a gap within HiGHS's defaults; `ModelEmpty` | `GRB_OPTIMAL` within the default gap | `OPTIMAL` |
| `GapReached` | `Optimal` with a gap wider than the defaults (a requested gap was reached) | `GRB_OPTIMAL` with a gap wider than the defaults | `GAPLIMIT` |
| `Stopped { TimeLimit }` | `TimeLimit` | `GRB_TIME_LIMIT` | `TIMELIMIT` |
| `Stopped { Interrupted }` | `Interrupt` | `GRB_INTERRUPTED` | `USERINTERRUPT` |
| `Stopped { NodeLimit }` | `SolutionLimit` with the node limit reached | `GRB_NODE_LIMIT` | `NODELIMIT`, `TOTALNODELIMIT`, `STALLNODELIMIT` |
| `Stopped { SolutionLimit }` | `SolutionLimit` otherwise (leaves, improving solutions) | `GRB_SOLUTION_LIMIT` | `SOLLIMIT`, `BESTSOLLIMIT` |
| `Stopped { IterationLimit }` | `IterationLimit` | `GRB_ITERATION_LIMIT` | none (LP iteration limits stop the LP only) |
| `Stopped { ObjectiveLimit }` | `ObjectiveBound`, `ObjectiveTarget` | `GRB_CUTOFF`, `GRB_USER_OBJ_LIMIT` | `PRIMALLIMIT`, `DUALLIMIT` |
| `Stopped { MemoryLimit }` | `MemoryLimit` | `GRB_MEM_LIMIT` | `MEMLIMIT` |
| `Stopped { Other }` | `Unknown`, `HighsInterrupt` | `GRB_SUBOPTIMAL`, `GRB_WORK_LIMIT` | `RESTARTLIMIT` |
| `Infeasible` | `Infeasible` | `GRB_INFEASIBLE` | `INFEASIBLE` |
| `Unbounded` | `Unbounded` | `GRB_UNBOUNDED` | `UNBOUNDED` |
| `InfeasibleOrUnbounded` | `UnboundedOrInfeasible` | `GRB_INF_OR_UNBD` | `INFORUNBD` |
| `Err(backend error)` | `NotSet`, `LoadError`, `ModelError`, `PresolveError`, `SolveError`, `PostsolveError`, or a failed run; for a QP also `Optimal` at a point holding NaN or with a relative primal-dual objective error above `1e-5`, and `Unbounded` with every column bounded (`HighsError::UnsoundQp`) | `GRB_NUMERIC`, `GRB_LOADED` after optimize, an error code | `UNKNOWN` after solve, `OPTIMAL` or `GAPLIMIT` without a solution or bound (`ScipError::Failed`), a best solution SCIP's own `SCIPcheckSolOrig` finds infeasible (`ScipError::InfeasibleBest`), a `SCIP_RETCODE` error |

A `Stopped` outcome carries the best solution when the solver holds a feasible one, and the best bound for a MIP.
`Solved::status` keeps the native status for diagnostics. For HiGHS that is `HighsStatus`: run status, model status,
primal and dual solution status, MIP gap, nodes, iterations, and the last error line of the log; for SCIP it is
`ScipStatus`: SCIP's status, dual bound, nodes, LP iterations and solutions found.

## What a Gurobi backend would be

A `culmen_gurobi` crate would hold an `Env`/`Model` pair over `grb-sys`. `load` calls `GRBnewmodel` with the `Program`'s
arrays. `solve` sets parameters from its own typed `GurobiOptions`, then from `Settings`, installs one
`GRBsetcallbackfunc` trampoline for the run, and maps `where` codes to events as in the table above. Its context
types would be `GurobiMessage`, `GurobiMip`, `GurobiMipSol` and `GurobiMipNode`. `GurobiMipNode` would implement
`CutContext` and `HeuristicContext` and offer `cbGet` codes as inherent methods. `EVENTS` would be `Events::ALL`
without `PRICE`, and `CLASSES` would be `ModelClasses::ALL`, the quadratic part passed with `GRBaddqpterms`.
None of this needs a change to culmen_core. SCIP needed more plumbing, since its callbacks are plugins (an event
handler, a message handler, a constraint handler, a separator, a heuristic and a pricer, each included in the
`SCIP*`), but each plugin maps onto the event kinds above; `culmen_scip` includes them in every instance and delivers
every generic event that way ([SCIP](scip.md#callback-events)). `PRICE` is the one event whose seam the pricer needed in culmen_core: the sink
`Priced` beside the callback, since the columns a solve prices outlive it.
