# Backends

culmen_core does not know any solver. A backend is a crate that implements `culmen_core::Backend` for its solver:
it declares the model classes it solves, loads a `Program`, applies the generic `Settings` over its own typed options,
drives the solver's callbacks as culmen's generic events, and maps the solver's statuses to the generic `Outcome`,
keeping its own status beside it.
HiGHS (`culmen_highs`) is the first backend. This page shows that the seam fits the other serious MIP solvers too:
each core concept is listed with what HiGHS 1.15 offers for it, and what Gurobi 12 and SCIP would offer a backend
written later.

## The seam

```rust
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
    type Native<'a>: Debug;               // Events::NATIVE: whatever only this solver reports
    const EVENTS: Events;                 // what this backend delivers
    const CLASSES: ModelClasses;          // the model classes this backend solves
    const ANALYSES: Analyses = Analyses::NONE; // what it reports after a solve: duals, basis, ranging, …
    const FEATURES: ProgramFeatures = ProgramFeatures::NONE; // semi columns, objectives, starts it takes
    fn load(&mut self, program: &Program) -> Result<(), Self::Error>;
    fn set_start(&mut self, start: &Start) -> Result<(), Self::Error> { Ok(()) } // only with StartSolution
    fn solve(&mut self, settings: &Settings, callback: &mut dyn Callback<Self>)
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
  `CutContext` has `add_cut`, only `HeuristicContext` has `supply` and `supply_partial`. An action that a solver
  honours only at certain points exists only on the context of those points.
- **Extras stay with the backend.** Each context type is the backend's own struct. What the solver reports beyond
  the neutral trait is an inherent method on it (`HighsPoll::source`, `HighsSolution::mip`) or a native event
  (`HighsNative::CutPool`). Generic code (`impl<B: Backend> Callback<B> for …`) sees the traits, and backend code
  (`impl Callback<Highs> for …`) sees everything.
- **The convenient path sits on top.** `Watch` (made from a `Request` of settings, a `Stop` and an `Observer`) is an
  ordinary `impl<B: Backend> Callback<B>`: it polls the stop at `POLL`, throttles `PROGRESS`/`POLL` into
  `Observer::progress`, reports `INCUMBENT` as `Observer::improved`, and joins `LOG` chunks into lines. It works
  unchanged on every backend, and pairs with a backend-specific callback as `(watch, mine)`.
- **The contract** (on `Backend`): solve the last loaded program; report every objective and bound with the program's
  quadratic part and offset in the model's sense; call only events in `EVENTS ∩ callback.events()`, only during `solve`,
  never two at once; a panic in a callback stops the solve and unwinds out of `solve`, never aborts.

`Script`, the in-memory backend in culmen_core, delivers every generic event from a scripted list. Its tests drive
lazy constraints, cuts, heuristic solutions, native events, the observer, stops and panics without any solver.
That is evidence that nothing in the core assumes HiGHS.

## Model classes

| Class | HiGHS 1.15 (`culmen_highs`) | Gurobi 12 | SCIP |
|---|---|---|---|
| `Lp` | yes | yes | yes |
| `Milp` | yes | yes | yes |
| `Qp` | yes: `Highs_passModel` with the lower triangle of `H` in `½ xᵀHx`, column-wise | yes: `GRBaddqpterms`, whose `qval` is the coefficient of `xᵢ · xⱼ` as culmen stores it | yes: a linear objective over an epigraph variable bounded by a quadratic constraint |
| `Miqp` | no: `Highs_run` refuses it ("Cannot solve MIQP problems with HiGHS") | yes | yes |

A `Program`'s quadratic part is canonical and convex in its sense
([The program as data](program.md#the-quadratic-part)), so a backend converts it without checking it again: HiGHS
takes `Hᵢᵢ = 2q` on the diagonal and `q` below it; Gurobi takes the entries as they are. Quadratic constraints, which
Gurobi and SCIP solve and HiGHS does not, would be further classes that a backend declares or leaves out.

`Script` declares every class, since it replays its script rather than solving.

## Features

| Feature | HiGHS 1.15 (`culmen_highs`) | Gurobi 12 | SCIP |
|---|---|---|---|
| `SemiContinuous` | yes: `kHighsVarTypeSemiContinuous`, upper bound at most `1e5` (`Highs::SEMI_UPPER_LIMIT`) | yes: `GRB_SEMICONT` | yes: a binary indicator and two linear rows |
| `SemiInteger` | yes: `kHighsVarTypeSemiInteger`, alike | yes: `GRB_SEMIINT` | yes, alike with an integer column |
| `MultipleObjectives` | yes: `Highs_passLinearObjectives`, lexicographic by priority or blended by `blend_multi_objectives`; linear only | yes: `GRBsetobjectiven`, priorities, weights and tolerances | no native support |
| `StartSolution` | yes: `Highs_setSolution`, a sparse start padded with HiGHS's undefined value | yes: the `Start` attribute, `GRB_UNDEFINED` for a partial one | yes: `SCIPcreateSol` and `SCIPaddSolFree`, partial solutions completed |

`Script` declares every feature and records each start in `record.starts`.

## Events

| Core concept | HiGHS 1.15 (`culmen_highs`) | Gurobi 12 | SCIP |
|---|---|---|---|
| `LOG`: a log chunk, its level | `kHighsCallbackLogging`: message, `log_type` (only field set) | `GRB_CB_MESSAGE`: `GRB_CB_MSG_STRING` | a message handler (`SCIPmessagehdlrCreate`) |
| `PROGRESS`: the solver's own pace | `kHighsCallbackMipLogging` (fires when a MIP log line is printed) | `GRB_CB_MIP`: `MIP_OBJBST`, `MIP_OBJBND`, `MIP_NODCNT`, `MIP_ITRCNT`; `GRB_CB_SIMPLEX`, `GRB_CB_BARRIER` for LPs | event handler on `SCIP_EVENTTYPE_NODESOLVED` with `SCIPgetPrimalbound`, `SCIPgetDualbound`, `SCIPgetNNodes` |
| `POLL`: frequent, may interrupt | `kHighsCallbackSimplexInterrupt`, `kHighsCallbackIpmInterrupt`, `kHighsCallbackMipInterrupt`: `user_interrupt = 1` | `GRB_CB_POLLING` and every other `where`: `GRBterminate` | event handler on `SCIP_EVENTTYPE_LPSOLVED` / `NODEFOCUSED`: `SCIPinterruptSolve` |
| `Progress` fields | `running_time`, `mip_primal_bound`, `mip_dual_bound`, `mip_node_count`, `mip_total_lp_iterations`, `simplex_iteration_count`, `ipm_iteration_count` | `cbGet` codes above, `GRB_CB_RUNTIME` | `SCIPgetSolvingTime` and the getters above |
| `FEASIBLE`: any new solution | `kHighsCallbackMipSolution`: `mip_solution`, `objective_function_value` | `GRB_CB_MIPSOL`: `MIPSOL_SOL`, `MIPSOL_OBJ` | event handler on `SCIP_EVENTTYPE_SOLFOUND` |
| `INCUMBENT`: a better solution | `kHighsCallbackMipImprovingSolution`; an accepted user solution is reported at the next MIP poll, as HiGHS skips its solution callbacks for it | `GRB_CB_MIPSOL` when `MIPSOL_OBJ` beats `MIPSOL_OBJBST` | `SCIP_EVENTTYPE_BESTSOLFOUND` |
| `LAZY`: cut off a candidate | none: `kHighsCallbackMipDefineLazyConstraints` exists in 1.15 but is never called, so `Unsupported` | `GRB_CB_MIPSOL` (and `MIPNODE`): `GRBcblazy`, with `LazyConstraints = 1` | a constraint handler: `CONSCHECK` / `CONSENFOLP` adding rows |
| `CUT`: cut off a node relaxation | none, so `Unsupported` | `GRB_CB_MIPNODE` with `MIPNODE_STATUS == OPTIMAL`: `GRBcbget(MIPNODE_REL)`, `GRBcbcut` | a separator (`SEPAEXECLP`): `SCIPgetSolVal` of the LP, `SCIPaddRow` |
| `HEURISTIC`: supply a solution | `kHighsCallbackMipUserSolution`: `user_solution` + `user_has_solution`; partial via `Highs_setCallbackSparseSolution` + `Highs_repairCallbackSolution` | `GRB_CB_MIPNODE` (and `MIP`): `GRBcbsolution` (partial values allowed), `GRBcbusesolution` | a primal heuristic plugin (`HEUREXEC`): `SCIPcreateSol`, `SCIPtrySolFree` |
| `NATIVE`: solver-only events | `kHighsCallbackMipGetCutPool` as `HighsNative::CutPool` (presolved-space cuts) | `GRB_CB_PRESOLVE`, `GRB_CB_MULTIOBJ`, `GRB_CB_IIS`, and `MIPNODE` node data beyond the relaxation | any other event type (`SCIP_EVENTTYPE_*`), branching or node-selection data |

HiGHS's threading is handled inside culmen_highs. The MIP kinds come from its serial search loop. Logging may come
from task-pool threads. The trampoline serialises all calls behind a mutex, so callbacks are `Send` and the stop is
`Sync`. Gurobi calls back on the thread that called `GRBoptimize`, and SCIP plugins run on the solving thread, so a
backend for either may call the callback directly, as `Script` does; the bounds on `Callback` and `Stop` already
allow a solver that calls back from other threads.

## Settings

| `Settings` field | HiGHS 1.15 option | Gurobi 12 parameter | SCIP parameter |
|---|---|---|---|
| `time_limit` | `time_limit` (clock zeroed per run) | `TimeLimit` | `limits/time` |
| `relative_gap` | `mip_rel_gap` | `MIPGap` | `limits/gap` |
| `absolute_gap` | `mip_abs_gap` | `MIPGapAbs` | `limits/absgap` |
| `threads` | `threads` (per-thread scheduler; a conflicting count is an error, and one above `i32::MAX` is refused) | `Threads` | `parallel/maxnthreads` (concurrent solve) or `lp/threads` |
| `seed` | `random_seed` (one above `i32::MAX` is refused, not clamped) | `Seed` | `randomization/randomseedshift` |
| `presolve` | `presolve` = `"on"` / `"off"` | `Presolve` = `-1` / `0` | `presolving/maxrounds` = `-1` / `0` |
| `node_limit` | `mip_max_nodes` | `NodeLimit` | `limits/nodes` |
| `solution_limit` | `mip_max_improving_sols` | `SolutionLimit` | `limits/solutions` |
| `console` | `log_to_console` (the log always reaches the callback) | `LogToConsole` (with `OutputFlag = 1`) | message handler output to stdout |

Every backend also has typed options for the rest (`culmen_highs::HighsOptions` covers HiGHS's documented
options as typed fields) and a raw name/value escape hatch (`HighsOptions::option`). A Gurobi backend would do the
same with `GRBsetintparam` / `GRBsetdblparam` / `GRBsetstrparam`, and SCIP with `SCIPsetIntParam` and friends.
`Settings` apply after the backend's options, so a per-solve setting always wins.

## Outcomes and statuses

| `Outcome` | HiGHS 1.15 `ModelStatus` | Gurobi 12 status | SCIP `SCIP_STATUS` |
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
| `Err(backend error)` | `NotSet`, `LoadError`, `ModelError`, `PresolveError`, `SolveError`, `PostsolveError`, or a failed run; for a QP also `Optimal` at a point holding NaN or with a relative primal-dual objective error above `1e-5`, and `Unbounded` with every column bounded (`HighsError::UnsoundQp`) | `GRB_NUMERIC`, `GRB_LOADED` after optimize, an error code | `UNKNOWN` after solve, a `SCIP_RETCODE` error |

A `Stopped` outcome carries the best solution when the solver holds a feasible one, and the best bound for a MIP.
`Solved::status` keeps the native status for diagnostics. For HiGHS that is `HighsStatus`: run status, model status,
primal and dual solution status, MIP gap, nodes, iterations, and the last error line of the log.

## What a Gurobi backend would be

A `culmen_gurobi` crate would hold an `Env`/`Model` pair over `grb-sys`. `load` calls `GRBnewmodel` with the `Program`'s
arrays. `solve` sets parameters from its own typed `GurobiOptions`, then from `Settings`, installs one
`GRBsetcallbackfunc` trampoline for the run, and maps `where` codes to events as in the table above. Its context
types would be `GurobiMessage`, `GurobiMip`, `GurobiMipSol` and `GurobiMipNode`. `GurobiMipNode` would implement
`CutContext` and `HeuristicContext` and offer `cbGet` codes as inherent methods. `EVENTS` would be `Events::ALL`,
and `CLASSES` would be `ModelClasses::ALL`, the quadratic part passed with `GRBaddqpterms`.
None of this needs a change to culmen_core. SCIP needs more plumbing, since its callbacks are plugins (an event
handler, a message handler, a constraint handler, a separator and a heuristic, each registered on the `SCIP*`), but
each plugin maps onto exactly one event kind above.
