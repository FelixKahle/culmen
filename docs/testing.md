# Testing

Code that drives a solve (a service that stops a solve on request, an observer that streams progress, a handler that
adds lazy constraints) is tested best without a solver: deterministically, fast, and with every event a real solver
might send. culmen offers three things for this: the `Script` backend, the `culmen_testing` crate, and serialization
of the plain data.

## The `Script` backend

`culmen::script::Script` is an in-memory [backend](backends.md) that delivers a fixed list of `Step`s to the callback and the
[search handlers](callbacks.md#search-handlers) and then ends with a fixed `Outcome`. It supports every event and has
every search capability (`LazyConstraints`, `UserCuts`, `Heuristics`, `Prices`), so it drives the whole callback
layer, and it needs no native code; it runs under Miri.

| Constructor | Script |
|---|---|
| `Script::new(steps, end)` | Delivers `steps` in order, then ends with the outcome `end`. |
| `Script::optimal(objective, values)` | No steps; ends optimal with the given solution. |

Each step is one event, in the generic terms of [`Callback`](callbacks.md#callback) and the handlers:

| Step | Event or handler | Context offers |
|---|---|---|
| `Step::Log { elapsed, level, text }` | `Log` | the chunk; line breaks in `text` end lines |
| `Step::Progress(progress)` | `Progress` | the progress |
| `Step::Poll(progress)` | `Poll` | `interrupt()` |
| `Step::Feasible(found)` | `Feasible` | the solution |
| `Step::Incumbent(found)` | `Incumbent` | the solution |
| `Step::Lazy(found)` | `LazyHandler` | the candidate; `add_lazy` |
| `Step::Cut { progress, relaxation }` | `CutHandler` | the relaxation; `add_cut` |
| `Step::Heuristic(progress)` | `HeuristicHandler` | `supply`, and `supply_partial` as an inherent method of `ScriptHeuristic` |
| `Step::Price { progress, mode, duals }` | `Pricer` | a `ScriptPrice`: the mode, the program's sense, the duals, `tolerance()` of `ScriptPrice::TOLERANCE`; `add_col`, `branch_early`, `report_bound` |
| `Step::Note(text)` | `Native` | a `ScriptNote` with `text()` |

A `Found` is `{ progress, objective, values }`. `step.event()` names a step's observational event kind, `None` for
the four search steps.

The script behaves as a solver would:

- An observational step reaches the callback only when the callback asks for its kind, and a search step reaches
  only a handler of its kind registered on the solve's `Run`, as with any backend. The outcome, the elapsed time and
  the work (`Solved::nodes` and `iterations`, the last counts a step's progress carried) count every step, whatever
  the callback asks for and whichever handlers are registered.
- A poll that interrupts ends the solve at once as `Outcome::Stopped` with `StopReason::Interrupted`, the last
  incumbent delivered before as `best` and the last bound reported as `bound`.
- Actions are checked as a solver's would be ([`ActionError`](callbacks.md#contexts)) and recorded in
  `script.record`: the `settings` of each solve, the `lazy` constraints and `cuts` added, the solutions
  `supplied`, the `starts` handed over, and one `PriceRound` per price step in `priced`, across every solve.
- A priced column is checked and numbered by the solve's `Priced`, so every later step of the solve counts it: a
  lazy constraint may name it, and a supplied solution holds a value for it. A `PriceRound` holds the ids it `added`,
  the last `bound` reported, and whether the round ended `early`; the columns themselves are the program's
  (`solver.program()`), which a `Solver` grows when the solve ends. A bound reported in Farkas mode is refused.
- `Solved::status` is a `ScriptStatus`: `Finished`, or `Interrupted { step }` with the position of the interrupting
  step. Solving before loading is `ScriptError::NotLoaded`.

Through a `Solver`, read the record with `solver.backend().record`.

```rust
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Found, Phase};
use culmen::script::{Script, ScriptStatus, Step};

/// The code under test: solve with a time limit and a stop, and return the best objective found.
fn best_within<B: Backend>(
    solver: &mut Solver<B>,
    limit: Duration,
    stop: &AtomicBool,
) -> Result<Option<f64>, SolveError<B::Error>> {
    let solved = solver.run().time_limit(limit).stop(stop).solve()?;
    Ok(solved.outcome.objective())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);
    let program = m.build()?;

    // Finished: the scripted outcome comes back, and the settings reached the backend.
    let mut solver = Solver::new(Script::optimal(1.0, vec![1.0]), program.clone())?;
    let running = AtomicBool::new(false);
    assert_eq!(best_within(&mut solver, Duration::from_secs(5), &running)?, Some(1.0));
    assert_eq!(solver.backend().record.settings[0].time_limit, Some(Duration::from_secs(5)));

    // Stopped: the incumbent found before the poll is the answer.
    let at = Progress::at(Duration::from_secs(1), Phase::Mip);
    let steps = [Step::Incumbent(Found { progress: at, objective: 0.0, values: vec![0.0].into() }), Step::Poll(at)];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), program)?;
    let stopped = AtomicBool::new(true);
    let solved_best = best_within(&mut solver, Duration::from_secs(5), &stopped)?;
    assert_eq!(solved_best, Some(0.0));
    let again = solver.run().stop(&stopped).solve()?;
    assert_eq!(again.status, ScriptStatus::Interrupted { step: 1 });
    Ok(())
}
```

## Testing callbacks

A callback written generically, `impl<B: Backend> Callback<B>`, and a handler written generically,
`impl<B: LazyConstraints> LazyHandler<B>` and so on, run on the `Script` exactly as on a solver, so their reactions to
each event can be checked against the record. Here a lazy-constraint handler forbids taking both of two items:

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{ActionError, AddedRow, Found, Phase};
use culmen::script::{Script, Step};

struct NotBoth {
    x: Col,
    y: Col,
    refused: Vec<ActionError>,
}

impl<B: LazyConstraints> LazyHandler<B> for NotBoth {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        let candidate = ctx.candidate();
        if candidate[self.x] + candidate[self.y] > 1.5
            && let Err(error) = ctx.add_lazy((self.x + self.y).leq(1))
        {
            self.refused.push(error);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    m.maximize(x + y);

    let at = Progress::at(Duration::from_secs(1), Phase::Mip);
    let both = Found { progress: at, objective: 2.0, values: vec![1.0, 1.0].into() };
    let one = Found { progress: at, objective: 1.0, values: vec![1.0, 0.0].into() };
    let script = Script::new([Step::Lazy(both), Step::Lazy(one)], Outcome::Infeasible);
    let mut solver = Solver::new(script, m.build()?)?;

    let mut not_both = NotBoth { x, y, refused: Vec::new() };
    solver.run().lazy(&mut not_both).solve()?;
    assert!(not_both.refused.is_empty());
    assert_eq!(
        solver.backend().record.lazy,
        [AddedRow { terms: vec![(x, 1.0), (y, 1.0)], bounds: bounds!(f64::NEG_INFINITY, 1.0) }]
    );
    Ok(())
}
```

An `AddedRow` is a constraint as the backend received it: its terms as given and its bounds with the constraint's
constant moved into them. Observers, stops, pairs of callbacks and handlers, and panics are tested the same way.

A pricer is tested with price steps that hand it the duals a solver would: the record shows each round, and the
program the columns it added.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{NewCol, Phase, PriceMode};
use culmen::script::{Script, Step};

/// Adds the pattern when it improves, else reports a bound and stops pricing.
struct Pattern(NewCol);

impl<B: Prices> Pricer<B> for Pattern {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        if ctx.improves(&self.0) {
            let _ = ctx.add_col(self.0.clone());
        } else if let PriceMode::ReducedCost { objective } = ctx.mode() {
            let _ = ctx.report_bound(objective);
            let _ = ctx.branch_early();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cover one unit of demand; the column x costs 3, a pattern costing 1 is withheld.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    let demand = m.add_constraint(x.geq(1));
    m.minimize(3.0 * x);
    let pattern = NewCol::continuous().cost(1).entry(demand, 1);

    let at = Progress::at(Duration::ZERO, Phase::Simplex);
    let round = |objective: f64, dual: f64| Step::Price {
        progress: at,
        mode: PriceMode::ReducedCost { objective },
        duals: vec![dual].into(),
    };
    // At the dual 3 the pattern prices at 1 - 3 = -2 and is added; at the dual 1 it prices at 0.
    let script = Script::new([round(3.0, 3.0), round(1.0, 1.0)], Outcome::Infeasible);
    let mut solver = Solver::new(script, m.build()?)?;
    solver.run().pricer(&mut Pattern(pattern)).solve()?;

    let rounds = &solver.backend().record.priced;
    assert_eq!(rounds[0].added.clone().collect::<Vec<_>>(), [Col::new(1)]);
    assert_eq!((rounds[1].added.len(), rounds[1].bound, rounds[1].early), (0, Some(1.0), true));
    assert_eq!(solver.program().columns()[Col::new(1)].cost, 1.0);
    Ok(())
}
```

## `culmen_testing`

`culmen_testing` is the workspace's shared test infrastructure. It is used only as a dev-dependency and is never
published (`publish = false`); outside this workspace, depend on it by path under `[dev-dependencies]`.

| Module | Holds |
|---|---|
| `reference` | `solve(&program)`, an exact solver for tiny linear problems; `close(a, b, tolerance)`, a relative comparison; `edit_plan(..)`, the rules of [editing a loaded program](solving.md#editing-a-loaded-program) judged without culmen_core's checks. |
| `strategy` | `proptest` strategies: `small_lp()` and `small_semi()`, tiny bounded LPs and MILPs (the second with semi columns) that `reference::solve` can enumerate; `small_qp()`, small convex QPs; `priced_split()`, a program split into the columns a solve starts with and those a pricer may add; `edits(&program)` and `with_edits(programs)`, edits of a program as data. |
| `tolerance` | `Tolerances`, a solver's feasibility, integrality, dual and gap tolerances as it defines them (`Tolerances::SCIP`, `Tolerances::HIGHS`); `loosened(&program)`, the program every point it may call feasible rounds into; `window(&program, &reference)`, the objective values its optimum may take: from the exact optimum of the loosened program on the better side to the exact optimum less its dual tolerance and gap on the worse; `stray(&program, &values)`, the first row or column a point passes by more than the tolerances allow. A property test compares a backend's optimum with this window rather than with the exact optimum at a fixed tolerance, since how far a tolerance moves the objective depends on the instance. |
| `op` | `EditOp`, one edit of a `Solver`'s program as data, which `apply` replays. |
| `mirror` | `Mirror<B>`, a backend that wraps another, records every load and edit, checks the edit contract, and fails or panics on demand; it has each capability its inner backend has, and relays the callback and the handlers to it through `Hooks::relay`. |
| `instances` | `assignment(products, locations, seed, offset)`, a generalized assignment hard enough that a solver searches for seconds; it returns the `program` and its variables `x`. |

`reference::solve` enumerates every integer point of the integer columns' box and every vertex of the continuous
part, solving each in exact rational arithmetic, so it shares none of a floating-point solver's reasoning. It
returns `Reference::Infeasible` or `Reference::Optimal(value)`, the exact optimum; `objective()` reads it as the
nearest `f64`. The objective must be linear, every column needs finite bounds, and the integer box at most a million
points; it panics otherwise.

```rust
use culmen::prelude::*;
use culmen_testing::reference::{self, Reference};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(0, 3));
    let y = m.add_var(Var::continuous().bounds(0, 2));
    m.add_constraint((x + y).leq(4.0));
    m.maximize(2.0 * x + y);
    let program = m.build()?;

    let exact = reference::solve(&program);
    assert_ne!(exact, Reference::Infeasible);
    assert_eq!(exact.objective(), Some(7.0));

    let solved = Solver::new(Highs::new(), program)?.solve()?;
    let found = solved.outcome.objective().ok_or("no solution")?;
    assert!(reference::close(found, 7.0, 1e-9));
    Ok(())
}
```

With `small_lp()`, a property test states that a backend agrees with the reference on every generated problem, as
`culmen_highs`'s and `culmen_scip`'s own tests do; `culmen_scip`'s also price the columns `priced_split()` withholds
back in and compare the result with the reference's optimum of the whole program.

## Serialization

With the `serde` feature, the plain data derives `Serialize` and `Deserialize`:

- The program: `Program`, `ProgramPlan` (its `quadratic` terms included), `ProgramColumn`, `ProgramRow`, `NewCol`,
  `Bounds`, `Integrality`, `Sense`, `Col`, `Row`, `ModelClass`, `ModelClasses` (as a list of classes),
  `ProgramFeature`, `ProgramFeatures`, and the further objectives: `Objectives`, `LinearObjective`, `ObjectiveId`,
  `Stage`, `Weighted`, `Tolerance`, `Weight`.
- The solve: `Settings`, `RelativeGap`, `AbsoluteGap`, `Start`, `Solved`, `Outcome`, `Solution`, `StopReason`,
  `Progress`, `Phase`, `LogLevel`.
- The analyses: `Analysis`, `Duals`, `Basis`, `BasisStatus`, `Ranging`, `Ranged`, `Limit`, `Line`, `Iis`,
  `Side`, `Weights`, `Penalty`, `Relaxed`, `Violation`, `Bound`.
- With `highs`: `HighsOptions` and its option types, `OptionValue`, `HighsStatus`, `ModelStatus`, `SolutionStatus`,
  `CallStatus`, `Version` and `files::ModelFormat`.
- With `scip`: `ScipOptions`, `Emphasis`, `ScipParam`, `ParamValue`, `ScipStatus`, `Status`, `RetCode` and `Version`.

The model (`Model`, `Var`, `LinExpr`, `QuadExpr`, `Constraint`, the families) is not serializable: serialize the
`Program` it builds.

A checked value is checked again as it is read, so a document cannot smuggle in what the constructors refuse:

- A `Program` is written as its plan and read back through `ProgramPlan::build`, the convexity check of its quadratic
  part included; a defect is the [`ProgramError`](program.md#checks-and-programerror) as a deserialization error. A plan
  without the `quadratic` field, as written before quadratic objectives, reads as linear.
- `Bounds` is written as `{"lower": …, "upper": …}`, an infinite end as `null`, since JSON has no infinity; it is
  read back through `Bounds::new`, an end that is `null` or left out read as infinite.
- A number that may be infinite or NaN (a `Limit`'s `value` and `objective`, a `Violation`'s `amount`, a real
  option of `HighsOptions` or a real `ParamValue`) is written as a number when finite and as the string `"inf"`,
  `"-inf"` or `"NaN"` otherwise, so it reads back as it was (`culmen::serde_real`).
- `RelativeGap` and `AbsoluteGap` are written as plain numbers and read back through their checked constructors;
  `Col` and `Row` are plain numbers. A `Settings` refuses a thread count or a solution limit of zero, as their
  `NonZero` types do.

```rust,ignore
// Needs `serde_json`, which the culmen crate does not depend on.
use culmen::prelude::*;
use culmen::RelativeGap;

let json = serde_json::to_string(&bounds!(0.0, f64::INFINITY))?;
assert_eq!(json, r#"{"lower":0.0,"upper":null}"#);
assert!(serde_json::from_str::<Bounds>(r#"{"lower":2.0,"upper":1.0}"#).is_err());
let error = serde_json::from_str::<RelativeGap>("-0.5").unwrap_err();
assert!(error.to_string().starts_with("RelativeGap: below zero"));
```

## See also

- [Callbacks](callbacks.md): what the `Script` delivers.
- [Backends](backends.md): the seam the `Script` implements, and the contract every backend keeps.
- [The program as data](program.md): what serializes, and how it is checked.
- [Documentation index](README.md).
