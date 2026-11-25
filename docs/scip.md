# SCIP

`culmen_scip` is the SCIP backend: SCIP 10 with SoPlex as its LP solver, built from source through the `scip-sys`
crate, with every SCIP status, SCIP's emphasis settings and parameters as typed options, and every generic callback
event delivered through SCIP plugins: lazy constraints, user cuts, heuristic solutions and pricing included. With the
`scip` feature it is `culmen::scip`, and its backend type is also `culmen::Scip` and in the prelude.

SCIP is the backend for what HiGHS cannot do: a convex quadratic objective over integer columns (MIQP), constraints
known only to a callback, cuts of your own, and column generation at every node of the search (branch-and-price).

## The backend

`Scip` owns the options every solve applies and, once a program is loaded, one SCIP instance holding it.

| Item | Does |
|---|---|
| `Scip::new()` | A backend with SCIP's default parameters. |
| `Scip::with_options(options)` | A backend with the given `ScipOptions`. |
| `scip.options()`, `scip.options_mut()` | The options, to read or to change between solves. |
| `Scip::EVENTS` | Every event but `NATIVE`: `LOG`, `PROGRESS`, `POLL`, `FEASIBLE`, `INCUMBENT`, `LAZY`, `CUT` and `HEURISTIC`. |
| `Scip::PRICES` | `true`: SCIP prices for a callback whose `prices()` is true ([Pricing](#pricing)). |
| `Scip::CLASSES` | `Lp`, `Milp`, `Qp` and `Miqp`; see [Quadratic programs](#quadratic-programs). |
| `Scip::FEATURES` | `StartSolution` only; see [Starts](#starts). |
| `Scip::ANALYSES` | None; see [Limitations](#limitations). |
| `culmen_scip::version()` | The version of the linked SCIP; it displays as `10.0.x`. |

```rust
use culmen::prelude::*;
use culmen::scip::{Emphasis, ScipOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A knapsack.
    let (value, weight) = ([10.0, 13.0, 7.0, 8.0], [5.0, 6.0, 4.0, 3.0]);
    let mut m = Model::new();
    let take = m.add_vars(4, |i| Var::binary().name(format!("take[{i}]")));
    m.add_constraint(LinExpr::sum((0..4).map(|i| weight[i] * take[i])).leq(10.0));
    m.maximize(LinExpr::sum((0..4).map(|i| value[i] * take[i])));

    let options = ScipOptions { heuristics: Some(Emphasis::Fast), ..ScipOptions::default() };
    let mut solver = Solver::new(Scip::with_options(options), m.build()?)?;
    let first = solver.solve(Request::new())?;

    // The options belong to the backend; change them between solves.
    solver.backend_mut().options_mut().separating = Some(Emphasis::Off);
    let second = solver.solve(Request::new())?;
    assert_eq!(first.outcome.objective(), Some(21.0));
    assert_eq!(second.outcome.objective(), Some(21.0));
    assert_eq!(culmen::scip::version().major, 10);
    Ok(())
}
```

## Building SCIP

No system SCIP is needed, and none is used. The build script of `scip-sys` (its `from-source` feature, which
`culmen_scip` enables) does this the first time it runs:

1. It **downloads the SCIP Optimization Suite 10.0.2 sources from GitHub**, so the build needs network access to
   `github.com`. A build without network fails at this step.
2. It compiles SCIP and SoPlex with **cmake** and a **C and C++ compiler** into static libraries, which are linked into
   your binary, so nothing SCIP needs is looked for at run time.
3. It generates the Rust bindings with **bindgen**, which needs **libclang**.

On Debian or Ubuntu, `apt-get install cmake libclang-dev build-essential` provides what the build needs; on macOS, the
Xcode command line tools and `brew install cmake llvm`.

The build is long and large, and Cargo runs it once per **feature resolution** of `scip-sys`: each set of features
Cargo resolves for it and its build dependencies (a `cargo build` and a `cargo test` that enable different features of
shared dependencies, a debug and a release profile) has its own build directory and builds SCIP again. Each costs:

| | Each feature resolution |
|---|---|
| Time | about two and a half minutes with four build jobs on a recent laptop, the download included (a clean `cargo build -p culmen_scip` took 2 min 17 s, 1.6 GB of memory at its peak); longer on a small CI runner |
| Disk | about 500 MB under `target/<profile>/build/scip-sys/`: 120 MB of sources, 250 MB of cmake build tree, 95 MB of libraries, 16 MB of SCIP's own binaries; that clean build's whole `target` took 1 GB |

A build later reuses each directory, so only the first build of each resolution pays. This is why `scip` is not a
default feature of `culmen`: most models are served by HiGHS, whose build is shorter and needs no network.

```toml
[dependencies]
culmen = { git = "https://github.com/FelixKahle/culmen", features = ["scip"] }
```

## Options

`ScipOptions` is plain data:

| Field | SCIP | Default |
|---|---|---|
| `heuristics` | `SCIPsetHeuristics`: how hard SCIP's primal heuristics work, an `Emphasis` | `None`, SCIP's own |
| `presolving` | `SCIPsetPresolving`: how hard SCIP presolves | `None` |
| `separating` | `SCIPsetSeparating`: how hard SCIP separates its cutting planes | `None` |
| `params` | parameters set by SCIP's names, in order, after the emphases | empty |

`Emphasis` is SCIP's `SCIP_PARAMSETTING`: `Default`, `Aggressive`, `Fast` or `Off`. **Parameters by name** are added
with `ScipOptions::param(name, value)`, which appends a `ScipParam { name, value }`. The value is a `ParamValue`, one of
SCIP's six types (`Bool`, `Int`, `Longint`, `Real`, `Char`, `String`), converted from a `bool`, `i32`, `i64`, `f64`,
`char`, `&str` or `String`. SCIP checks the name and the value when the solve sets them; an unknown name or a value of
the wrong type or range is a [`ScipError::Param`](#errors) naming the parameter and SCIP's return code.

```rust
use culmen::scip::{Emphasis, ScipOptions};

let options = ScipOptions { presolving: Some(Emphasis::Aggressive), ..ScipOptions::default() }
    .param("limits/maxorigsol", 5)
    .param("lp/initalgorithm", 'd')
    .param("display/verblevel", 5);
assert_eq!(options.params.len(), 3);
```

## How options and settings layer

Every solve sets SCIP up from its defaults, in this order:

1. The last solve's transformed problem is freed, so SCIP holds the program as loaded, and every parameter is reset to
   SCIP's default.
2. Two parameters are set on every solve:
   - `misc/catchctrlc` to `false`: SCIP would otherwise install its own `SIGINT` handler for the solve and take a
     Ctrl-C meant for your program. A solve is stopped through a [`Stop`](callbacks.md#stop) instead.
   - `constraints/nonlinear/reformbinprods` to `false`: SCIP's presolve rewrites a product of two binary columns in
     the epigraph of a quadratic objective through a clique of its clique table, and that table can hold a clique
     derived under the incumbent's cutoff that no constraint enforces; the rewritten epigraph then admits points
     below the objective, and SCIP could report a wrong optimum for an MIQP. Without the rewrite the epigraph stays
     exact ([Quadratic programs](#quadratic-programs)).
3. The backend's `ScipOptions` apply: the three emphases, then the parameters by name in the order added.
4. The solve's [`Settings`](solving.md#settings) apply.
5. What the events the callback asks for need is set last ([Callback events](#callback-events)): for `LAZY`, symmetry
   handling and dual reductions off (`misc/usesymmetry`, `misc/allowstrongdualreds`, `misc/allowweakdualreds`); for
   `CUT` and `HEURISTIC`, culmen's separator or heuristic called at every node; for a callback that prices, the pricer active, every
   row modifiable, `lp/presolving` off, `separating/aggregation/freq` at `-1` and `lp/disablecutoff` at `1`.

So a per-solve setting always wins over the backend's options, and an option naming one of the two parameters of
step 2 overrides the backend's choice. The parameters of step 5 are what those events need to be correct, so they
override an option or a setting that names them. SCIP's log stays off standard output unless `Settings::console`
asks for it.

| `Settings` field | SCIP |
|---|---|
| `time_limit` | `limits/time`, in seconds; SCIP's clock restarts with each solve while `misc/resetstat` keeps its default |
| `relative_gap` | `limits/gap` |
| `absolute_gap` | `limits/absgap` |
| `threads` | nothing; see [Threads](#threads) |
| `seed` | `randomization/randomseedshift`; a seed above `i32::MAX` is refused |
| `presolve` | `SCIPsetPresolving`: `Off` for `false`; for `true`, the options' `presolving` emphasis when it presolves, else `Default` |
| `node_limit` | `limits/nodes`; a limit above `i64::MAX` is refused |
| `solution_limit` | `limits/solutions`; a limit above `i32::MAX` is refused |
| `console` | SCIP's messages reach standard output and standard error, beside the callback |

A value beyond what SCIP's parameter takes is refused with [`ScipError::OutOfRange`](#errors) before any parameter is
set.

**A second solve starts from the first one's best solutions.** Freeing the transformed problem keeps up to
`limits/maxorigsol` (10 by default) of the last solve's best solutions as candidates, which the next solve checks
again, against its own lazy constraints too, and may start from, as it would a [start](#starts); so a second solve of
the same program, under a node or time limit, can end with the first one's optimum. The last solve's parameters
decide, and `misc/transsolsorig` set to `false` keeps none. HiGHS, too, starts from its last solution. An
[edit](solving.md#editing-a-loaded-program) loads the program into a new instance, which holds no solution of before,
and so does the reload after a solve that priced columns.

## Threads

- **`Scip` is `Send`**: move it, or the whole `Solver`, to a worker thread to solve there. It is not `Sync`: one
  instance promises nothing to two threads calling it at once. SCIP is built thread-safe (`SCIP_THREADSAFE`), so
  instances on different threads do not interfere.
- **A solve runs on one thread.** culmen calls `SCIPsolve`, SCIP's sequential solve, and every plugin runs on the
  thread that called it, so `Settings::threads` has nothing to set and is taken without effect. SCIP's concurrent
  solve races copies of the problem in several threads, and those copies hold neither culmen's plugins nor its
  callback, so the backend does not use it.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_vars(4, |_| Var::binary());
    m.add_constraint(LinExpr::sum(x.iter().copied()).leq(2));
    m.maximize(LinExpr::sum(x.iter().copied()));
    let mut solver = Solver::new(Scip::new(), m.build()?)?;

    let settings = Settings { threads: Some(threads!(4)), ..Settings::new() };
    let worker = std::thread::spawn(move || solver.solve(Request::new().settings_from(settings)));
    let solved = worker.join().map_err(|_| "the solve thread panicked")??;
    assert_eq!(solved.outcome.objective(), Some(2.0));
    Ok(())
}
```

## Statuses and outcomes

`Solved::status` is a `ScipStatus`, SCIP's own account of the solve beside the generic outcome:

| Field | Holds |
|---|---|
| `status` | SCIP's `Status`: every value of `SCIP_STATUS` in SCIP 10, and `Other(u32)` for a value SCIP 10 does not define. |
| `dual_bound` | The best bound SCIP proved, offset included, in the program's sense; `None` when it proved none. |
| `nodes` | The branch-and-bound nodes, across restarts. |
| `lp_iterations` | The LP iterations. |
| `solutions_found` | The feasible solutions SCIP found, its heuristics', the starts and the carried solutions included. |

`Status` displays in SCIP's own wording (`SCIPstatusName`): `"optimal solution found"`, `"gap limit reached"`.

| `Status` | `Outcome` |
|---|---|
| `Optimal` | `Optimal`, within SCIP's gap limits (`limits/gap` is 0 by default) |
| `GapLimit` | `GapReached` with the solution and the dual bound |
| `Infeasible`, `Unbounded`, `InfeasibleOrUnbounded` | the outcome of the same name |
| `UserInterrupt` | `Stopped { Interrupted }` |
| `NodeLimit`, `TotalNodeLimit`, `StallNodeLimit` | `Stopped { NodeLimit }` |
| `TimeLimit` | `Stopped { TimeLimit }` |
| `MemoryLimit` | `Stopped { MemoryLimit }` |
| `SolutionLimit`, `BestSolutionLimit` | `Stopped { SolutionLimit }` |
| `PrimalLimit`, `DualLimit` | `Stopped { ObjectiveLimit }` |
| `Terminate`, `RestartLimit` | `Stopped { Other }` |
| `Unknown`, `Other(_)`, or `Optimal` / `GapLimit` without a solution or bound | `Err(ScipError::Failed)` |

A `Stopped` outcome carries SCIP's best solution when it holds one, and its dual bound. A solve reports a best
solution, an optimum or a reached gap only after SCIP's own check (`SCIPcheckSolOrig`) finds that solution feasible in
the original problem, to SCIP's feasibility tolerance; when it does not, as when presolve rewrote a constraint
unsoundly, the solve fails with `ScipError::InfeasibleBest` instead of reporting a wrong answer. A solve that priced
columns also checks it against SCIP's transformed problem (`SCIPcheckSol`), whose rows hold the priced columns.

```rust
use culmen::prelude::*;
use culmen::scip::Status;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + y).leq(4.0));
    m.add_constraint((x + 3.0 * y).leq(6.0));
    m.maximize(3.0 * x + 2.0 * y + 10);

    let solved = Solver::new(Scip::new(), m.build()?)?.solve(Request::new())?;
    let status = &solved.status;
    assert_eq!(status.status, Status::Optimal);
    assert_eq!(status.status.to_string(), "optimal solution found");
    assert!((status.dual_bound.ok_or("no bound")? - 22.0).abs() < 1e-9);
    assert!((solved.outcome.objective().ok_or("no solution")? - 22.0).abs() < 1e-9);
    Ok(())
}
```

An empty program, one without columns, is `Optimal` at its offset when every row admits an activity of zero, and
`Infeasible` when a row such as `0 = 1` does not.

## Quadratic programs

`Scip::CLASSES` is every class: LP, MILP, QP and MIQP. SCIP solves no quadratic objective natively, so the backend
writes a quadratic objective `q(x)` as an **epigraph**: a variable `t` in its place, with cost one, bounded by the
quadratic constraint `q(x) - t <= 0` and `t >= 0` when minimizing, `q(x) - t >= 0` and `t <= 0` when maximizing, where
a convex `q` is never negative and a concave one never positive. SCIP solves that constraint as a convex nonlinear
constraint, by cutting planes, with the `reformbinprods` rewrite off ([How options and settings
layer](#how-options-and-settings-layer)).

**Accuracy.** A solution's objective takes `q(x)` itself in place of `t`, and the cutting planes meet the constraint
only to SCIP's feasibility tolerance, `1e-6` by default. So an optimum's objective is close to the optimal value, within
about that tolerance, and its point, where the objective is flat, less close to the optimal point: within about the
tolerance's square root, `1e-3`. Tighten `numerics/feastol` through `ScipOptions::param` for a closer point.

```rust
use culmen::prelude::*;
use culmen::ModelClass;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The point of x + y <= 1 nearest (1, 2) with x whole: (0, 1), at squared distance 2.
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(-5, 5));
    let y = m.add_var(Var::free());
    m.add_constraint((x + y).leq(1));
    m.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2));
    let program = m.build()?;
    assert_eq!(program.class(), ModelClass::Miqp);

    let solved = Solver::new(Scip::new(), program)?.solve(Request::new())?;
    let Outcome::Optimal(best) = solved.outcome else { return Err("no optimum".into()) };
    assert!((best.objective - 2.0).abs() < 1e-6);
    assert!(best.values[x].abs() < 1e-3 && (best.values[y] - 1.0).abs() < 1e-3);
    Ok(())
}
```

## Starts

`Scip::FEATURES` is `StartSolution` alone. A [start](solving.md#starts) reaches SCIP before the solve: a full start as
a SCIP solution (`SCIPcreateSol`, `SCIPaddSolFree`), a sparse one as a partial solution (`SCIPcreatePartialSol`) that
SCIP completes when the solve begins. SCIP checks a start as it begins; an infeasible start, or a sparse one it cannot
complete, is dropped then, without an error, as SCIP drops any. A feasible start is the solve's first incumbent. SCIP
completes a sparse start with its `completesol` heuristic, a sub-SCIP over the columns left open, so a `heuristics`
emphasis of `Off` keeps a sparse start from being completed, and it is dropped.

Semi columns and several objectives are refused by [`Solver::new`](solving.md#model-classes-and-refusal) with
`LoadError::UnsupportedFeature` before SCIP sees the program.

```rust
use culmen::prelude::*;
use culmen::scip::{Emphasis, ScipOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (value, weight) = ([1.5, 1.4, 1.3, 1.2, 1.1, 1.0], 2.0);
    let mut m = Model::new();
    let take = m.add_vars(6, |_| Var::binary());
    m.add_constraint(LinExpr::sum(take.iter().map(|&x| weight * x)).leq(5.0));
    m.maximize(LinExpr::sum(take.iter().zip(value).map(|(&x, v)| v * x)));

    // Without heuristics, cuts and presolve, one node finds no solution of its own.
    let plain = ScipOptions { heuristics: Some(Emphasis::Off), separating: Some(Emphasis::Off), ..ScipOptions::default() };
    let one_node = Settings { node_limit: Some(1), presolve: Some(false), ..Settings::new() };
    let mut solver = Solver::new(Scip::with_options(plain), m.build()?)?;

    // A start taking the last two items, worth 2.1, is the best one node holds.
    let last_two = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let started = solver.solve(Request::new().settings_from(one_node.clone()).start(last_two))?;
    let Outcome::Stopped { best: Some(best), .. } = started.outcome else { return Err("no start".into()) };
    assert!((best.objective - 2.1).abs() < 1e-9);

    // A solve to the optimum, 2.9, and then one node again: it ends with the optimum the last solve left.
    assert!((solver.solve(Request::new())?.outcome.objective().ok_or("no optimum")? - 2.9).abs() < 1e-9);
    let again = solver.solve(Request::new().settings_from(one_node))?;
    assert!((again.outcome.objective().ok_or("no solution")? - 2.9).abs() < 1e-9);
    Ok(())
}
```

## Callback events

A `Callback<Scip>` receives SCIP's own context types. Each implements the neutral [context trait](callbacks.md#contexts)
of its event; each event comes from a SCIP plugin that culmen includes in every instance:

| Event | SCIP plugin | Context | Beyond the trait |
|---|---|---|---|
| `LOG` | the instance's message handler: each info message and warning, a whole line | `ScipLog` | |
| `POLL` | an event handler on each presolve round, focused node, solved node LP and solved node (`PRESOLVEROUND`, `NODEFOCUSED`, `LPSOLVED`, `NODESOLVED`); `interrupt` calls `SCIPinterruptSolve` | `ScipPoll` | `event()`: a `ScipEvent` |
| `PROGRESS` | the same event handler, at each poll | `ScipProgress` | `event()` |
| `FEASIBLE`, `INCUMBENT` | the event handler on each solution SCIP stores (`POORSOLFOUND`, `BESTSOLFOUND`), a new best one as the incumbent too; the starts and carried solutions SCIP stored before it, as it catches its events | `ScipSolution` | |
| `LAZY` | a constraint handler without constraints, checked and enforced after every other (`CONSCHECK`, `CONSENFOLP`, `CONSENFOPS`, `CONSENFORELAX`) | `ScipLazy` | `source()`: a `LazySource`, `Check`, `Lp`, `Pseudo` or `Relaxation` |
| `CUT` | a separator on each node's LP solution (`SEPAEXECLP`) | `ScipCut` | |
| `HEURISTIC` | a primal heuristic before and after each node (`HEUREXEC`) | `ScipHeuristic` | |
| pricing (`prices()`) | a variable pricer, delayed, on each optimal or infeasible node LP (`PRICERREDCOST`, `PRICERFARKAS`) | `ScipPrice` | `local_bounds(col)`: the node's bounds of a column |
| `NATIVE` | none | `Unsupported` | never delivered |

Every value a context reports is the program's: a candidate, a relaxation and a solution hold one value per column,
read through SCIP's original variables whatever presolve did to them, and every objective value and bound is in the
program's sense with its offset and quadratic part. Progress carries the wall-clock time since the solve began, the
primal and dual bound, the nodes and the LP iterations, each in the stages SCIP reports it.

A plugin acts only for the events a solve's callback asks for, so a solve that asks for nothing runs SCIP as it would
without culmen: asking for `LAZY` or pricing changes how SCIP solves (see each section below), asking for `CUT` or
`HEURISTIC` calls culmen's separator or heuristic at every node, and the other events cost a call per event. SCIP calls every plugin on the thread running the solve, one call at a time; a plugin
called while the callback is busy with an outer event (a solution supplied from `heuristic` is checked and stored at
once, which calls the constraint handler and the event handler) finds it busy: a log line or a solution is then
delivered as soon as the outer call returns, and a poll or a progress report is dropped. Everything a context changes
in SCIP (a lazy constraint, a cut, a priced column) is handed to SCIP once the callback returns.

A panic in a callback never unwinds into SCIP: the plugin catches it and asks SCIP to stop (`SCIPinterruptSolve`),
every later delivery is skipped, and the panic resumes once SCIP has returned. The backend stays usable.

## Lazy constraints

At `LAZY`, the callback sees every candidate SCIP would accept: each solution SCIP checks before storing it, from a
heuristic, a start, a sub-SCIP or a supplied solution (`LazySource::Check`), and each node's LP solution (`Lp`), pseudo
solution (`Pseudo`) or relaxation (`Relaxation`) that every other constraint handler found feasible, so an integer
one. `add_lazy` adds a constraint the candidate may violate:

- A constraint holds for the rest of the solve. Every later candidate satisfies every lazy constraint added before, to
  SCIP's feasibility tolerance; one that violates a constraint kept before is rejected without asking the callback.
- At an enforcement, SCIP's problem takes the constraint as a global linear constraint (`SCIPaddCons`), which SCIP then
  enforces, separates and checks with everything else, and SCIP solves the node again. SCIP allows no constraint to be
  added while it checks a solution, so a constraint added at `Check` waits in the handler, which rejects every later
  candidate violating it, until the next enforcement adds it.
- A constraint added again, the same terms and bounds as one kept, is not added twice.

A solve asking for `LAZY` gives up what reasons only about the constraints SCIP holds: every variable is locked against
dual reductions, symmetry handling is off (`misc/usesymmetry = 0`) and so are SCIP's strong and weak dual reductions
(`misc/allowstrongdualreds`, `misc/allowweakdualreds`). Without that, presolve could fix a column by an argument a
constraint added later refutes. SCIP's sub-SCIPs, which copy the problem without the callback, still run; their
solutions are checked by the main SCIP and so meet the callback.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::LazyContext;
use culmen::scip::{LazySource, ScipLazy};

/// Forbids taking both items, a conflict only the callback knows.
struct NotBoth {
    pair: (Col, Col),
    calls: usize,
    checked: usize,
}

impl Callback<Scip> for NotBoth {
    fn events(&self) -> Events {
        Events::LAZY
    }
    fn lazy(&mut self, ctx: &mut ScipLazy<'_>) {
        self.calls += 1;
        assert!(self.calls < 1000, "a lazy callback that never settles");
        if ctx.source() == LazySource::Check {
            self.checked += 1;
        }
        let (x, y) = self.pair;
        let candidate = ctx.candidate();
        if candidate[x] + candidate[y] > 1.5 {
            ctx.add_lazy((x + y).leq(1)).expect("a valid constraint");
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    let z = m.add_var(Var::binary());
    m.add_constraint((x + y + z).leq(2));
    m.maximize(3.0 * x + 2.0 * y + z);

    let mut solver = Solver::new(Scip::new(), m.build()?)?;
    let request = Request::new()
        .settings_from(Settings { node_limit: Some(1000), ..Settings::new() })
        .time_limit(Duration::from_secs(10));
    let mut not_both = NotBoth { pair: (x, y), calls: 0, checked: 0 };
    let solved = solver.solve_with(request, &mut not_both)?;
    let best = solved.outcome.solution().ok_or("no solution")?;
    assert_eq!(best.objective, 4.0); // x and z
    assert!(best.values[x] + best.values[y] <= 1.0 + 1e-9);
    Ok(())
}
```

The `lazy_constraints` example of the `culmen` crate solves a small travelling-salesman problem this way, cutting off
subtours.

## Cuts

At `CUT`, the callback sees each node's LP solution, `relaxation()`, before SCIP's own separators. `add_cut` adds a
constraint that every integer-feasible point of the program satisfies: each becomes a global row over the columns'
variables; a cut that cuts the LP solution off by SCIP's efficacy goes into the LP at once (`SCIPaddRow`), and every cut
goes into SCIP's global cut pool (`SCIPaddPoolCut`), which SCIP separates again at later nodes. SCIP takes a cut as
valid and may cut off every point it excludes; that the cut is valid is the callback's promise, which nothing checks.
A solve asking for `CUT` calls the separator at every node, also when the `separating` emphasis turned SCIP's own off.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::CutContext;
use culmen::scip::{Emphasis, ScipOptions};

/// A cover cut: no three of the four items fit, so at most two are taken.
struct Cover {
    items: Vec<Col>,
    calls: usize,
    added: usize,
}

impl<B: Backend> Callback<B> for Cover {
    fn events(&self) -> Events {
        Events::CUT
    }
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        self.calls += 1;
        assert!(self.calls < 1000, "a separator that never settles");
        let relaxation = ctx.relaxation();
        if self.items.iter().map(|&x| relaxation[x]).sum::<f64>() > 2.0 + 1e-6 {
            ctx.add_cut(LinExpr::sum(self.items.iter().copied()).leq(2)).expect("a valid cut");
            self.added += 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let take = m.add_vars(4, |_| Var::binary());
    m.add_constraint(LinExpr::sum(take.iter().map(|&x| 4.0 * x)).leq(11.0));
    m.maximize(LinExpr::sum(take.iter().zip([5.0, 4.0, 3.0, 2.0]).map(|(&x, v)| v * x)));

    // SCIP's own cuts and presolve off, so the root LP takes 2.75 items and the callback's cut is the one that helps.
    let options = ScipOptions { separating: Some(Emphasis::Off), ..ScipOptions::default() };
    let mut solver = Solver::new(Scip::with_options(options), m.build()?)?;
    let request = Request::new()
        .settings_from(Settings { presolve: Some(false), node_limit: Some(1000), ..Settings::new() })
        .time_limit(Duration::from_secs(10));
    let mut cover = Cover { items: take.clone(), calls: 0, added: 0 };
    let solved = solver.solve_with(request, &mut cover)?;
    assert_eq!(solved.outcome.objective(), Some(9.0));
    assert!(cover.added >= 1);
    Ok(())
}
```

## Heuristic solutions

At `HEURISTIC`, SCIP asks for a solution of the callback's own, before and after each node, ahead of SCIP's own
heuristics. A solve asking for it calls the heuristic at every node, also when the `heuristics` emphasis turned SCIP's
own off.

- `ctx.supply(&values)` offers a value for every column. The values are checked first: one finite value per column,
  or `ActionError::Length` or `ActionError::NotFinite`. The solution becomes an original solution of SCIP's
  (`SCIPcreateOrigSol`), the epigraph's variable set to the quadratic part's value, and SCIP tries it at once
  (`SCIPtrySolFree`): it checks it against the program, bounds and integrality included, and stores it if it is
  feasible and among the solutions it keeps (`limits/maxsol`). A solution SCIP does not store is
  `ActionError::Refused`. In a solve that also asks for `LAZY`, the solution is tried once the heuristic callback
  returns, after the lazy-constraint callback judged it, and `supply` returns `Ok` when it took it to try.
- `ctx.supply_partial(&entries)` is refused with `ActionError::Refused` once the entries pass the core checks: SCIP
  completes a partial solution only as a start, before a solve begins. Hand a partial solution over as a sparse
  [start](#starts) instead.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{ActionError, HeuristicContext, TiVec};

struct Offer {
    values: Option<TiVec<Col, f64>>,
    supplied: Vec<Result<(), ActionError>>,
}

impl<B: Backend> Callback<B> for Offer {
    fn events(&self) -> Events {
        Events::HEURISTIC
    }
    fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>) {
        if let Some(values) = self.values.take() {
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

    // A known feasible solution: items 1 and 3, worth 21.
    let mut known: TiVec<Col, f64> = vec![0.0; 6].into();
    known[take[1]] = 1.0;
    known[take[3]] = 1.0;

    // Without presolve, which would solve so small a program before the first node, so no heuristic would run.
    let mut solver = Solver::new(Scip::new(), m.build()?)?;
    let mut offer = Offer { values: Some(known), supplied: Vec::new() };
    let request = Request::new()
        .settings_from(Settings { presolve: Some(false), ..Settings::new() })
        .time_limit(Duration::from_secs(10));
    let solved = solver.solve_with(request, &mut offer)?;
    assert_eq!(offer.supplied, [Ok(())]);
    assert_eq!(solved.outcome.objective(), Some(25.0));
    Ok(())
}
```

## Pricing

[Pricing](callbacks.md#pricing), for a callback whose `prices()` is true, runs at every node: SCIP solves each node LP over the columns it holds, prices
those itself, and then hands the callback the node LP's duals, optimal (`PriceMode::ReducedCost`) or infeasible
(`PriceMode::Farkas`), until a round adds no column. Branching on the master's integer columns, as SCIP chooses, makes
it branch-and-price.

How SCIP does it:

- **The pricer.** Every instance includes a variable pricer, delayed (SCIP prices the variables it holds before it asks
  the callback) and inactive. A solve whose callback prices activates it (`SCIPactivatePricer`) and marks
  every program row's constraint modifiable (`SCIPsetConsModifiable`), as SCIP requires of a row a pricer adds
  variables to; any other solve deactivates it and leaves every row unmodifiable, so it costs a solve without pricing
  nothing.
- **The duals.** SCIP minimizes `s·c` for the program's cost `c`, with `s = 1` when the program minimizes and `s = -1`
  when it maximizes, so its dual of a row is `s` times the program's: the context hands out `s · SCIPgetDualsolLinear`,
  in the sign convention of [`Duals`](analysis.md). A Farkas multiplier (`SCIPgetDualfarkasLinear`) does not depend on
  the sense and is handed out as SCIP reports it. `tolerance()` is SCIP's dual feasibility tolerance
  (`numerics/dualfeastol`).
- **The columns.** Each column the callback adds is checked against what SCIP can hold (a number SCIP reads as
  infinite, a name with a NUL byte, as `ActionError::Refused`) and by the solve's `Priced`, which numbers it. Once the
  callback returns it becomes a variable of SCIP's transformed problem (`SCIPcreateVar` with objective `s·c`,
  `SCIPaddPricedVar`) with its coefficients in the transformed rows (`SCIPaddCoefLinear`). SCIP never deletes it.
- **Ending a round.** A round that returns is SCIP's `SUCCESS`, with the bound `report_bound` gave, if any, as the
  node's lower bound. `branch_early()` in a reduced-cost round stops pricing at the node (`stopearly`), and SCIP
  branches; in a Farkas round SCIP takes the round as not run. SCIP can branch only on an unfixed integer column: an
  early stop where there is none, as in a program without integer columns, fails the solve with
  `ScipError::BranchEarly`. A round SCIP calls while the callback is busy with an outer event, or after it panicked,
  is not run. A stop during pricing ends the node's pricing loop, and the node LP then counts as no bound.
- **What a pricing solve gives up.** SCIP itself turns off what needs every column known: restarts, presolve and bound
  tightening on the modifiable rows, the presolvers and propagators that need complete columns, the objective's
  integrality and scaling. The backend also turns off the LP solver's presolving (`lp/presolving`), since SoPlex's
  simplifier can return duals of a degenerate node LP that do not price a column at its bound out, which would end
  pricing with a wrong optimum; SCIP's aggregation separator (`separating/aggregation/freq = -1`), which cuts with
  the objective cutoff over the columns held as if no other column could enter; and the LP solver's objective limit
  (`lp/disablecutoff = 1`), which would stop a node LP at the cutoff and hand its non-optimal duals to the callback.
- **Other events beside it.** A priced column has entries only in program rows, so it enters every lazy constraint
  and cut with coefficient zero: those must hold for every column the callback may price. A heuristic solution of a
  pricing solve is a transformed solution (`SCIPcreateSol`) holding the priced columns, refused when presolve fixed or
  aggregated a program column so that it cannot hold the values given. A lazy constraint may name a priced column.
- **No quadratic objective.** A callback that prices is refused on a program with a quadratic objective with
  `ScipError::PricingQuadratic` before SCIP runs. Asking for every event (`Events::ALL`) does not price, so such a
  callback solves a quadratic program.

**What branch-and-price is here.** It is column generation at every node with **SCIP's default branching on the
master's integer columns**. Each node LP is the full master LP under the bounds branching set on some held columns,
which SCIP prices itself within those bounds, so the search is **exact when the pricing is exact and never adds a
column the problem already holds**: a column branching fixed, added again, escapes the branch, and the children then
overlap. Branching on master columns is correct but weak: the branch `x ≤ ⌊v⌋` barely changes the LP, and the tree can
grow large. There is no branching event, so **no Ryan–Foster or other branching on the pricing problem**; a master that
needs it is out of reach of this first version. `ScipPrice::local_bounds(col)` gives the node's bounds of a column, for
a pricer that wants to skip what branching implies.

Since a pricing solve leaves the objective's integrality unknown to SCIP, a master whose columns all cost whole
amounts is proved optimal sooner with an absolute gap just below one: a solution less than one unit from the bound is
then optimal.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{NewCol, PriceContext, ProgramColumn};

/// Adds every pattern of the pool, not added yet, that improves the node LP. The pool holds every pattern the
/// master lacks, so the pricing is exact and never adds a held column.
struct Pool {
    patterns: Vec<NewCol>,
    added: Vec<Option<Col>>,
    rounds: usize,
}

impl<B: Backend> Callback<B> for Pool {
    fn events(&self) -> Events {
        Events::NONE
    }
    fn prices(&self) -> bool {
        true
    }
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        self.rounds += 1;
        assert!(self.rounds < 10_000, "a pricer that never settles");
        for (pattern, added) in self.patterns.iter().zip(&mut self.added) {
            if added.is_none() && ctx.improves(pattern) {
                *added = Some(ctx.add_col(pattern.clone()).expect("a valid pattern"));
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cut rolls of width 10 into 7 pieces of width 3, 2 of width 4 and 3 of width 5, one pattern per width so far.
    let mut m = Model::new();
    let [a, d, f] = [(); 3].map(|_| m.add_var(Var::integer()));
    let threes = m.add_constraint((3.0 * a).geq(7));
    let fours = m.add_constraint((2.0 * d).geq(2));
    let fives = m.add_constraint((2.0 * f).geq(3));
    m.minimize(a + d + f);

    // The other patterns of three, four and five pieces a roll of 10 holds.
    let pattern = |pieces: [f64; 3]| NewCol {
        column: ProgramColumn::integer(1.0, Bounds::NON_NEGATIVE),
        entries: [threes, fours, fives].into_iter().zip(pieces).filter(|&(_, k)| k > 0.0).collect(),
    };
    let others = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    let mut pool = Pool { patterns: others.map(pattern).to_vec(), added: vec![None; 8], rounds: 0 };

    // Every pattern costs one roll, so less than one roll from the bound is optimal.
    let mut solver = Solver::new(Scip::new(), m.build()?)?;
    let request = Request::new()
        .settings_from(Settings { absolute_gap: Some(absolute_gap!(0.999)), node_limit: Some(1000), ..Settings::new() })
        .time_limit(Duration::from_secs(10));
    let solved = solver.solve_with(request, &mut pool)?;

    // Five rolls: the LP bound is 4.5, and three threes, two threes and a four, a three and a five, and two fives
    // cover the demand.
    let best = solved.outcome.solution().ok_or("no solution")?;
    assert_eq!(best.objective.round(), 5.0);
    assert_eq!(best.values.len(), solver.program().num_cols());
    assert!(solver.program().max_violation(&best.values) < 1e-6);
    Ok(())
}
```

The `branch_and_price` example of the `culmen` crate prices cutting patterns at every node with a knapsack solved by
a second `Scip`, which excludes every pattern the master holds.

## Errors

`ScipError` says why loading or solving failed:

| Variant | When |
|---|---|
| `NotLoaded` | A solve or a start was asked for before a program was loaded. |
| `Call { call, code }` | A call of SCIP's C API returned an error; `call` names the C function, `code` is a `RetCode`. |
| `Param { name, value, code }` | SCIP refused a parameter: an unknown name, a value of the wrong type or out of its range. |
| `ParamName { name }`, `ParamString { name }` | A parameter's name or string value holds a NUL byte. |
| `ParamChar { name, value }` | A character parameter's value is not ASCII. |
| `OutOfRange { name, value, max }` | A seed, node limit or solution limit exceeds SCIP's parameter. |
| `ColName { col }`, `RowName { row }` | A column's or row's name holds a NUL byte. |
| `BeyondInfinity { number, value, infinity }` | A finite number of the program is at least SCIP's infinity (`1e20`) in magnitude; `number` says where it stands. |
| `RowTooLong { row, len }`, `QuadraticTooLong { len }` | A row or the quadratic part has more entries than SCIP's `int` counts. |
| `Unsupported { class }`, `UnsupportedFeature { feature }` | A program the backend does not take, which a `Solver` never hands it. |
| `Start` | A start that does not fit the program, which a `Solver` never hands over. |
| `Failed { status }` | SCIP ended with a status that proves nothing, or claimed an optimum or a gap without a solution or a bound. |
| `PricingQuadratic` | A callback that prices was given a program with a quadratic objective. |
| `BranchEarly` | A pricing callback ended pricing early where SCIP has no integer column left to branch on. |
| `InfeasibleBest { status, objective }` | SCIP's own check finds its best solution infeasible in the original problem; the solution is not reported. |

`RetCode` is every error value of `SCIP_RETCODE` (`NoMemory`, `LpError`, `InvalidCall`, `ParameterUnknown`, …) and
`Other(i32)` for one SCIP 10 does not define.

```rust
use culmen::prelude::*;
use culmen::scip::ScipError;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);
    let mut solver = Solver::new(Scip::new(), m.build()?)?;

    let settings = Settings { seed: Some(u32::MAX), ..Settings::new() };
    let Err(SolveError::Backend(error)) = solver.solve(Request::new().settings_from(settings)) else {
        panic!("the seed is refused")
    };
    let max = i32::MAX.unsigned_abs().into();
    let value = u32::MAX.into();
    assert_eq!(error, ScipError::OutOfRange { name: "randomization/randomseedshift", value, max });
    assert_eq!(solver.solve(Request::new())?.outcome.objective(), Some(1.0));
    Ok(())
}
```

**SCIP's own error messages go to standard error.** SCIP prints the reason for a call it refuses, such as an unknown
parameter name, through its process-wide error printer, which the backend leaves as it is; the typed error carries
the call or parameter and SCIP's return code, and the message is on standard error.

## Limitations

- **No semi columns and no several objectives.** `Scip::FEATURES` is `StartSolution` alone, so `Solver::new` refuses a
  program with a semi-continuous or semi-integer column or with further objectives
  ([Starts](#starts)).
- **No analyses.** `Scip::ANALYSES` is empty: duals, basis, ranging, rays, IIS and feasibility relaxation are
  `AnalysisError::Unsupported` ([Analysis](analysis.md)).
- **No partial heuristic solutions.** `supply_partial` is always refused; a partial solution is a sparse start
  ([Heuristic solutions](#heuristic-solutions)).
- **No native events and no branching event.** `NATIVE` is `Unsupported`, and there is no event to branch on the
  pricing problem, so branch-and-price branches on master columns only ([Pricing](#pricing)).
- **No pricing on a quadratic objective.** `ScipError::PricingQuadratic` ([Pricing](#pricing)).
- **An interrupt acts after the current LP.** SCIP's LP solver runs without events, so a stop asked for during one LP
  solve, by a `Stop` or a poll's `interrupt`, acts when that LP ends; a long root LP delays it.
- **One thread per solve.** `Settings::threads` has no effect ([Threads](#threads)).
- **Results are exact to SCIP's tolerances.** A solution is feasible to SCIP's feasibility tolerance (`numerics/feastol`,
  `1e-6`), and SCIP's own check of the best solution uses the same tolerance; a quadratic objective's point is close to
  the optimal point only to about its square root ([Quadratic programs](#quadratic-programs)).
- **SCIP's error messages go to standard error** ([Errors](#errors)).
- **A second solve may start from the first one's solutions** ([How options and settings
  layer](#how-options-and-settings-layer)); set `misc/transsolsorig` to `false` for a cold re-solve.
- **No model files.** The backend reads and writes no MPS or LP files; [LP files](program.md#lp-files) are written by
  culmen itself, and HiGHS's [model files](highs.md#model-files) are behind the `highs` feature.
- **A long first build that needs network access** ([Building SCIP](#building-scip)).

## See also

- [Callbacks](callbacks.md): the neutral events and contexts, and pricing.
- [Solving](solving.md): requests, settings, starts and outcomes.
- [HiGHS](highs.md): the default backend.
- [Backends](backends.md): how SCIP's plugins, parameters and statuses map onto culmen's seam, beside HiGHS and Gurobi.
- [Documentation index](README.md).
