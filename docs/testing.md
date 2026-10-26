# Testing

Code that drives a solve (a service that stops a solve on request, an observer that streams progress, a callback that
adds lazy constraints) is tested best without a solver: deterministically, fast, and with every event a real solver
might send. culmen offers three things for this: the `Script` backend, the `culmen_testing` crate, and serialization
of the plain data.

## The `Script` backend

`culmen::Script` is an in-memory [backend](backends.md) that delivers a fixed list of `Step`s to the callback and
then ends with a fixed `Outcome`. It supports every event, so it drives the whole callback layer, and it needs no
native code; it runs under Miri.

| Constructor | Script |
|---|---|
| `Script::new(steps, end)` | Delivers `steps` in order, then ends with the outcome `end`. |
| `Script::optimal(objective, values)` | No steps; ends optimal with the given solution. |

Each step is one event, in the generic terms of [`Callback`](callbacks.md#callback):

| Step | Event | Context offers |
|---|---|---|
| `Step::Log { elapsed, level, text }` | `LOG` | the chunk; line breaks in `text` end lines |
| `Step::Progress(progress)` | `PROGRESS` | the progress |
| `Step::Poll(progress)` | `POLL` | `interrupt()` |
| `Step::Feasible(found)` | `FEASIBLE` | the solution |
| `Step::Incumbent(found)` | `INCUMBENT` | the solution |
| `Step::Lazy(found)` | `LAZY` | the candidate; `add_lazy` |
| `Step::Cut { progress, relaxation }` | `CUT` | the relaxation; `add_cut` |
| `Step::Heuristic(progress)` | `HEURISTIC` | `supply`, `supply_partial` |
| `Step::Note(text)` | `NATIVE` | a `ScriptNote` with `text()` |

A `Found` is `{ progress, objective, values }`. `step.event()` names a step's event kind.

The script behaves as a solver would:

- A step reaches the callback only when the callback asks for its kind, as with any backend. The outcome and the
  elapsed time count every step, whatever the callback asks for.
- A poll that interrupts ends the solve at once as `Outcome::Stopped` with `StopReason::Interrupted`, the last
  incumbent delivered before as `best` and the last bound reported as `bound`.
- Actions are checked as a solver's would be ([`ActionError`](callbacks.md#contexts)) and recorded in
  `script.record`: the `settings` of each solve, the `lazy` constraints and `cuts` added, and the solutions
  `supplied`, across every solve.
- `Solved::status` is a `ScriptStatus`: `Finished`, or `Interrupted { step }` with the position of the interrupting
  step. Solving before loading is `ScriptError::NotLoaded`.

Through a `Solver`, read the record with `solver.backend().record`.

```rust
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Found, Phase, Script, ScriptStatus, Step};

/// The code under test: solve with a time limit and a stop, and return the best objective found.
fn best_within<B: Backend>(solver: &mut Solver<B>, limit: Duration, stop: &AtomicBool) -> Result<Option<f64>, B::Error> {
    let solved = solver.solve(Request::new().time_limit(limit).stop(stop))?;
    Ok(solved.outcome.solution().map(|s| s.objective))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);
    let lp = m.build()?;

    // Finished: the scripted outcome comes back, and the settings reached the backend.
    let mut solver = Solver::new(Script::optimal(1.0, vec![1.0]), lp.clone())?;
    let running = AtomicBool::new(false);
    assert_eq!(best_within(&mut solver, Duration::from_secs(5), &running)?, Some(1.0));
    assert_eq!(solver.backend().record.settings[0].time_limit, Some(Duration::from_secs(5)));

    // Stopped: the incumbent found before the poll is the answer.
    let at = Progress::at(Duration::from_secs(1), Phase::Mip);
    let steps = [Step::Incumbent(Found { progress: at, objective: 0.0, values: vec![0.0].into() }), Step::Poll(at)];
    let mut solver = Solver::new(Script::new(steps, Outcome::Infeasible), lp)?;
    let stopped = AtomicBool::new(true);
    let solved_best = best_within(&mut solver, Duration::from_secs(5), &stopped)?;
    assert_eq!(solved_best, Some(0.0));
    let again = solver.solve(Request::new().stop(&stopped))?;
    assert_eq!(again.status, ScriptStatus::Interrupted { step: 1 });
    Ok(())
}
```

## Testing callbacks

A callback written generically, `impl<B: Backend> Callback<B>`, runs on the `Script` exactly as on a solver, so its
reactions to each event can be checked against the record. Here a lazy-constraint callback forbids taking both of
two items:

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{ActionError, AddedRow, Found, LazyContext, Phase, Script, Step};

struct NotBoth {
    x: Col,
    y: Col,
    refused: Vec<ActionError>,
}

impl<B: Backend> Callback<B> for NotBoth {
    fn events(&self) -> Events {
        Events::LAZY
    }
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
    solver.solve_with(Request::new(), &mut not_both)?;
    assert!(not_both.refused.is_empty());
    assert_eq!(
        solver.backend().record.lazy,
        [AddedRow { terms: vec![(x, 1.0), (y, 1.0)], bounds: bounds!(f64::NEG_INFINITY, 1.0) }]
    );
    Ok(())
}
```

An `AddedRow` is a constraint as the backend received it: its terms as given and its bounds with the constraint's
constant moved into them. Observers, stops, pairs of callbacks and panics are tested the same way.

## `culmen_testing`

`culmen_testing` is the workspace's shared test infrastructure. It is used only as a dev-dependency and is never
published (`publish = false`); outside this workspace, depend on it by path under `[dev-dependencies]`.

| Module | Holds |
|---|---|
| `reference` | `solve(&lp)`, an exact solver for tiny problems, and `close(a, b, tolerance)`, a relative comparison. |
| `strategy` | `small_lp()`, a `proptest` strategy of tiny bounded LPs and MILPs that `reference::solve` can enumerate. |
| `instances` | `assignment(products, locations, seed, offset)`, a generalized assignment hard enough that a solver searches for seconds; it returns the `lp` and its variables `x`. |

`reference::solve` enumerates every integer point of the integer columns' box and every vertex of the continuous
part, solving each in exact rational arithmetic, so it shares none of a floating-point solver's reasoning. It
returns `Reference::Infeasible` or `Reference::Optimal(value)`, the exact optimum; `objective()` reads it as the
nearest `f64`. Every column needs finite bounds, and the integer box at most a million points; it panics otherwise.

```rust
use culmen::prelude::*;
use culmen_testing::reference::{self, Reference};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(0, 3));
    let y = m.add_var(Var::continuous().bounds(0, 2));
    m.add_constraint((x + y).leq(4.0));
    m.maximize(2.0 * x + y);
    let lp = m.build()?;

    let exact = reference::solve(&lp);
    assert_ne!(exact, Reference::Infeasible);
    assert_eq!(exact.objective(), Some(7.0));

    let solved = Solver::new(Highs::new(), lp)?.solve(Request::new())?;
    let found = solved.outcome.solution().ok_or("no solution")?.objective;
    assert!(reference::close(found, 7.0, 1e-9));
    Ok(())
}
```

With `small_lp()`, a property test states that a backend agrees with the reference on every generated problem, as
`culmen_highs`'s own tests do.

## Serialization

With the `serde` feature, the plain data derives `Serialize` and `Deserialize`:

- The LP: `Lp`, `LpPlan` (its `quadratic` terms included), `LpColumn`, `LpRow`, `Bounds`, `Integrality`, `Sense`,
  `Col`, `Row`, `ModelClass`, `ModelClasses` (as a list of classes).
- The solve: `Settings`, `RelativeGap`, `AbsoluteGap`, `Solved`, `Outcome`, `Solution`, `StopReason`, `Progress`,
  `Phase`, `LogLevel`.
- With `highs`: `HighsOptions` and its option types, `OptionValue`, `HighsStatus`, `ModelStatus`, `SolutionStatus`,
  `CallStatus`.

The model (`Model`, `Var`, `LinExpr`, `QuadExpr`, `Constraint`, the families) is not serializable: serialize the
`Lp` it builds.

A checked value is checked again as it is read, so a document cannot smuggle in what the constructors refuse:

- An `Lp` is written as its plan and read back through `LpPlan::build`, the convexity check of its quadratic part
  included; a defect is the [`LpError`](lp.md#checks-and-lperror) as a deserialization error. A plan without the
  `quadratic` field, as written before quadratic objectives, reads as linear.
- `Bounds` is written as `{"lower": …, "upper": …}`, an infinite end as `null`, since JSON has no infinity; it is
  read back through `Bounds::new`, an end that is `null` or left out read as infinite.
- `RelativeGap` and `AbsoluteGap` are written as plain numbers and read back through their checked constructors;
  `Col` and `Row` are plain numbers.

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
- [The LP as data](lp.md): what serializes, and how it is checked.
- [Documentation index](README.md).
