# Callbacks

While a solver runs, it reports what it does: log text, progress, polls at which it may be interrupted, solutions it
finds, and chances to hand it solutions or constraints. culmen delivers these as typed events. This page covers both
ways to receive them: the convenient one, a stop and an observer carried by the [`Request`](solving.md#request), and
the complete one, a `Callback` with a typed context per event.

## Two levels

| Level | You write | You get | Works on |
|---|---|---|---|
| Convenient | a [`Stop`](#stop) and an [`Observer`](#observer), passed with `Request::stop` and `Request::observer` | progress at a steady pace, every new best solution, whole log lines, a cooperative stop | every backend, unchanged |
| Complete | a [`Callback<B>`](#callback), passed with `Solver::solve_with` | every event the backend delivers, each with its context and actions | every backend through the neutral traits, or one backend through its own context types |

Both run in the same solve: `solve_with(request, &mut callback)` runs the request's stop and observer first and the
callback after them, for every event.

## `Stop`

A `Stop` is a cooperative stop request, polled while the solve runs. When it answers `true` at a poll, the solver is
asked to stop as soon as it can; the solve then ends as `Outcome::Stopped` with `StopReason::Interrupted`, carrying
the best solution found and the best bound.

`Stop` is implemented for `AtomicBool`, for `Arc<S>` and `&S` of any stop, and for any `Fn() -> bool + Sync` through
`stop_fn`. A tokio `CancellationToken` adapts in one line: `stop_fn(move || token.is_cancelled())`.

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Found, Phase, Script, Step};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);

    // A scripted backend stands in for a solver: a solution at one second, then a poll.
    let at = Progress::at(Duration::from_secs(1), Phase::Mip);
    let found = Found { progress: at, objective: 1.0, values: vec![1.0].into() };
    let script = Script::new([Step::Incumbent(found), Step::Poll(at)], Outcome::Infeasible);
    let mut solver = Solver::new(script, m.build()?)?;

    // Set from another thread, a signal handler or a UI; here, before the solve.
    let cancel = Arc::new(AtomicBool::new(false));
    cancel.store(true, Ordering::Release);

    let solved = solver.solve(Request::new().stop(&cancel))?;
    let Outcome::Stopped { reason: StopReason::Interrupted, best: Some(best), .. } = solved.outcome else {
        return Err("expected an interrupted solve".into());
    };
    assert_eq!(best.values[x], 1.0);

    let never = stop_fn(|| false);
    let solved = solver.solve(Request::new().stop(&never))?;
    assert_eq!(solved.outcome, Outcome::Infeasible);
    Ok(())
}
```

A stop is polled only at the backend's poll events, so it acts within one poll interval, not at once. A backend that
delivers no poll events cannot be stopped this way; HiGHS delivers them from the simplex, interior-point and MIP
solvers.

## `Observer`

An `Observer` watches a solve. Each method does nothing by default; implement the ones you need.

| Method | Called with | When |
|---|---|---|
| `progress(&Progress)` | where the solve stands | at most about once per `Request::progress_interval`, the first time at once |
| `improved(&Improvement)` | a new best solution | every new incumbent |
| `log(&LogLine)` | a complete log line, without its line break | every line of the solver's log |

A `Progress` holds `elapsed`, the `phase` (`Presolve`, `Simplex`, `Barrier`, `Mip` or `Other`), and, as the
solver reports them, the best `objective`, the best `bound`, the `nodes` explored and the LP `iterations`; a value
the solver does not report in this phase is `None`. Every value is in the model's objective space, offset included.
`progress.gap()` computes the relative gap `|objective - bound| / |objective|`, `None` without both values.

An `Improvement` holds the `progress` when the solution was found, its `objective` and its `values`, read with the
model's handles. A `LogLine` holds the `elapsed` time, the `level` (`Detail`, `Info`, `Warning`, `Error`, in that
order), and the `text`. The solver writes its log in chunks; the observer receives whole lines.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Found, LogLevel, Phase, Script, Step};

#[derive(Default)]
struct Seen {
    progress: Vec<Duration>,
    improved: Vec<f64>,
    lines: Vec<String>,
}

impl Observer for Seen {
    fn progress(&mut self, p: &Progress) {
        self.progress.push(p.elapsed);
    }
    fn improved(&mut self, i: &Improvement<'_>) {
        self.improved.push(i.objective);
    }
    fn log(&mut self, line: &LogLine<'_>) {
        self.lines.push(line.text.to_owned());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);

    let at = |ms| Progress::at(Duration::from_millis(ms), Phase::Mip);
    let chunk = |text: &str| Step::Log { elapsed: Duration::ZERO, level: LogLevel::Info, text: text.to_owned() };
    let steps = vec![
        chunk("Presolve"),
        chunk(" done\nSolving\n"),
        Step::Poll(at(0)),
        Step::Poll(at(500)),
        Step::Incumbent(Found { progress: at(600), objective: 1.0, values: vec![1.0].into() }),
        Step::Poll(at(1200)),
    ];
    let end = Outcome::Optimal(Solution { objective: 1.0, values: vec![1.0].into() });
    let mut solver = Solver::new(Script::new(steps, end), m.build()?)?;

    let mut seen = Seen::default();
    solver.solve(Request::new().observer(&mut seen))?;
    assert_eq!(seen.lines, ["Presolve done", "Solving"]);
    assert_eq!(seen.progress, [Duration::ZERO, Duration::from_millis(1200)]); // one report per second at most
    assert_eq!(seen.improved, [1.0]);
    Ok(())
}
```

On HiGHS the same observer receives HiGHS's log, and its console output stays quiet unless `Request::console(true)`
asks for it.

## `Callback`

`Callback<B>` has one method per event kind, each taking the backend's context for it, and `events()`, the set of
events it wants. Every method does nothing by default, and `events()` defaults to every event. A backend calls a
method only for an event in both its own `Backend::EVENTS` and the callback's `events()`, so asking for fewer events
costs the solver less.

| Method | Event | Context |
|---|---|---|
| `log` | `Events::LOG` | `B::Log<'_>`: [`LogContext`](#contexts) |
| `progress` | `Events::PROGRESS` | `B::Progress<'_>`: `ProgressContext` |
| `poll` | `Events::POLL` | `B::Poll<'_>`: `PollContext` |
| `feasible` | `Events::FEASIBLE` | `B::Solution<'_>`: `SolutionContext` |
| `incumbent` | `Events::INCUMBENT` | `B::Solution<'_>`: `SolutionContext` |
| `lazy` | `Events::LAZY` | `B::Lazy<'_>`: `LazyContext` |
| `cut` | `Events::CUT` | `B::Cut<'_>`: `CutContext` |
| `heuristic` | `Events::HEURISTIC` | `B::Heuristic<'_>`: `HeuristicContext` |
| `native` | `Events::NATIVE` | `B::Native<'_>`: whatever only this backend reports |

Write it once for every backend, `impl<B: Backend> Callback<B>`, and it sees each context through its neutral
trait. Write it for one backend, `impl Callback<Highs>`, and it sees that backend's own context types, with
everything the solver offers beyond the traits ([HiGHS callback contexts](highs.md#callback-contexts)).

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Phase, PollContext, Script, ScriptStatus, Step};

/// Stops any backend's solve once it has run for a fixed time.
struct Deadline(Duration);

impl<B: Backend> Callback<B> for Deadline {
    fn events(&self) -> Events {
        Events::POLL
    }
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        if ctx.progress().elapsed >= self.0 {
            ctx.interrupt();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    m.add_var(Var::binary());
    let polls = (1..=3).map(|s| Step::Poll(Progress::at(Duration::from_secs(s), Phase::Mip)));
    let mut solver = Solver::new(Script::new(polls.collect::<Vec<_>>(), Outcome::Infeasible), m.build()?)?;

    let solved = solver.solve_with(Request::new(), &mut Deadline(Duration::from_secs(2)))?;
    assert_eq!(solved.status, ScriptStatus::Interrupted { step: 1 });
    Ok(())
}
```

## `Events`

`Events` is a set of event kinds: `NONE`, `LOG`, `PROGRESS`, `POLL`, `FEASIBLE`, `INCUMBENT`, `LAZY`, `CUT`,
`HEURISTIC`, `NATIVE`, and `ALL`. Combine sets with `|` and `&` (or `union` and `intersection`), test them with
`contains`, and iterate the single events with `iter()`.

```rust
use culmen::prelude::*;

let wanted = Events::LOG | Events::POLL;
assert!(wanted.contains(Events::POLL));
assert!(!wanted.contains(Events::CUT));
assert_eq!(wanted.len(), 2);
assert_eq!(format!("{wanted:?}"), "Events(LOG | POLL)");
assert!(Highs::EVENTS.contains(Events::HEURISTIC));
```

`FEASIBLE` is any solution the solver found; `INCUMBENT` is one better than every one before, so the same solution
may arrive as both.

## Contexts

Each event's context reads where the solve stands and offers only the actions the solver honours at that point:

| Trait | Reads | Does |
|---|---|---|
| `LogContext` | `text()`, the chunk as written, line breaks included; `level()`; `progress()` | nothing |
| `ProgressContext` | `progress()` | nothing |
| `PollContext` | `progress()` | `interrupt()`: ask the solver to stop as soon as it can |
| `SolutionContext` | `progress()`, `objective()`, `values()` | nothing |
| `LazyContext` | `progress()`, `objective()`, `candidate()` | `add_lazy(constraint)`: a constraint the candidate may violate, kept for the rest of the search |
| `CutContext` | `progress()`, `relaxation()` | `add_cut(cut)`: a constraint every integer-feasible point satisfies |
| `HeuristicContext` | `progress()` | `supply(values)`: a full solution; `supply_partial(entries)`: some columns' values, for the solver to complete |

Values are `TiSlice<Col, f64>` read with the model's handles, and constraints are the same
[`Constraint`](modelling.md#comparisons) values a model adds, so `ctx.add_lazy(x.leq(y))` reads as the model does.
The traits are at the crate root, `culmen::PollContext` and so on; bring the one you call into scope.

An action returns `Err(ActionError)` when it is refused:

| Variant | Why |
|---|---|
| `Length { expected, found }` | A full solution holds the wrong number of values. |
| `UnknownColumn { col }` | An entry names a column the problem does not have. |
| `NotFinite { col }` | A value or coefficient is NaN or infinite. |
| `Bounds(InvalidBounds)` | A constraint's bounds are invalid after its constant moved into them. |
| `Refused(reason)` | The solver refused the action, for the reason given. |

## Pairing callbacks

Two callbacks run together as a pair `(a, b)`: each runs only for the events it asked for, `a` first, and the pair
asks for the union. `&mut C` is a callback wherever `C` is, so a pair can borrow both and leave them readable
afterwards. `()` is the callback that asks for nothing.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Phase, PollContext, Script, Step};

#[derive(Default)]
struct CountPolls(usize);

impl<B: Backend> Callback<B> for CountPolls {
    fn events(&self) -> Events {
        Events::POLL
    }
    fn poll(&mut self, _: &mut B::Poll<'_>) {
        self.0 += 1;
    }
}

struct StopAfter(Duration);

impl<B: Backend> Callback<B> for StopAfter {
    fn events(&self) -> Events {
        Events::POLL
    }
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        if ctx.progress().elapsed >= self.0 {
            ctx.interrupt();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    m.add_var(Var::binary());
    let polls: Vec<Step> = (1..=5).map(|s| Step::Poll(Progress::at(Duration::from_secs(s), Phase::Mip))).collect();
    let mut solver = Solver::new(Script::new(polls, Outcome::Infeasible), m.build()?)?;

    let (mut count, mut stop) = (CountPolls::default(), StopAfter(Duration::from_secs(3)));
    solver.solve_with(Request::new(), &mut (&mut count, &mut stop))?;
    assert_eq!(count.0, 3);
    Ok(())
}
```

The request's stop and observer are themselves a callback, `Watch`, which `solve_with` pairs with yours as
`(watch, callback)`. `Request::into_watch()` hands out that `Watch`, the settings and the start, for code that drives a
`Backend` directly; `Watch::finish` reports a last log line that never got its line break.

## `Unsupported`

A backend that cannot deliver an event leaves it out of `Backend::EVENTS` and names `Unsupported` as its context
type. `Unsupported` is an enum with no values: a callback method for that event compiles, but it can never be
called, since no context can be made. Check a backend's capabilities at compile time or run time with
`B::EVENTS.contains(..)`.

```rust
use culmen::prelude::*;

assert!(!Highs::EVENTS.contains(Events::LAZY));
assert!(!Highs::EVENTS.contains(Events::CUT));
assert!(culmen::Script::EVENTS.contains(Events::ALL));
```

HiGHS 1.15 offers neither lazy constraints nor user cuts; see [Limitations](highs.md#limitations).

## Panics and threads

- **A panic in an observer, a stop or a callback never aborts the process.** It stops the solve as soon as the
  solver allows and then continues unwinding out of `solve` or `solve_with`. The backend stays usable: the next solve
  on the same `Solver` runs normally. Catch it with `std::panic::catch_unwind` around the solve, if you need to.
- **Calls never overlap.** A backend calls one callback method at a time, and only while `solve` runs.
- **They may come from another thread.** A backend may call back from a thread other than the one that called
  `solve`, so `Callback` and `Observer` are `Send` and `Stop` is `Sync`. HiGHS, for one, may write log lines from
  its worker threads; it serialises every call behind a lock.

## See also

- [Solving](solving.md): the request that carries the stop and the observer.
- [HiGHS](highs.md#callback-contexts): what each HiGHS context offers beyond the traits.
- [Testing](testing.md#testing-callbacks): driving callbacks with the `Script` backend.
- [Backends](backends.md): how events map onto HiGHS, Gurobi and SCIP.
- [Documentation index](README.md).
