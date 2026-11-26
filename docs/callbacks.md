# Callbacks

While a solver runs, it reports what it does: log text, progress, polls at which it may be interrupted, and solutions
it finds. A solver that can also takes part in its search: it accepts constraints and cuts, solutions of yours, and
columns. culmen delivers the first kind as typed observational events and the second to search handlers, one trait
per kind. This page covers the convenient way to watch a solve, a stop and an observer carried by the
[`Request`](solving.md#request), the complete one, a `Callback` with a typed context per event, and the handlers.

## Two levels

| Level | You write | You get | Works on |
|---|---|---|---|
| Convenient | a [`Stop`](#stop) and an [`Observer`](#observer), passed with `Request::stop` and `Request::observer` | progress at a steady pace, every new best solution, whole log lines, a cooperative stop | every backend, unchanged |
| Complete | a [`Callback<B>`](#callback), registered on the solve's [`Hooks`](#search-handlers) and passed with `Solver::solve_with` | every observational event the backend delivers, each with its context | every backend through the neutral traits, or one backend through its own context types |
| Search | a [search handler](#search-handlers) (`LazyHandler`, `CutHandler`, `HeuristicHandler`, `Pricer`), registered on the same `Hooks` | candidates to cut off, relaxations to cut, chances to supply solutions, node LPs to price | only a backend with the capability; on any other the registration does not compile |

All run in the same solve: `solve_with(request, hooks)` runs the request's stop and observer first and the callback
after them, for every event, and the handlers at their own points of the search.

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
delivers no poll events cannot be stopped this way; HiGHS delivers them from the simplex, interior-point, QP and MIP
solvers, [SCIP](scip.md#callback-events) at each presolve round, focused node, solved node LP and solved node, so a
stop asked for during one of SCIP's LP solves acts when that LP ends.

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

On HiGHS the same observer receives HiGHS's log, and its console output stays quiet unless `Settings::console`
asks for it.

## `Callback`

`Callback<B>` has one method per observational event kind, each taking the backend's context for it, and `events()`,
the set of events it wants. Every method does nothing by default and `events()` defaults to every event. A backend
calls a method only for an event in both its own `Backend::EVENTS` and the callback's `events()`, so asking for fewer
events costs the solver less. A callback only watches: what changes the search is a [search
handler](#search-handlers), so a callback asking for every event costs a solver no reduction it would otherwise make.

| Method | Event | Context |
|---|---|---|
| `log` | `Events::LOG` | `B::Log<'_>`: [`LogContext`](#contexts) |
| `progress` | `Events::PROGRESS` | `B::Progress<'_>`: `ProgressContext` |
| `poll` | `Events::POLL` | `B::Poll<'_>`: `PollContext` |
| `feasible` | `Events::FEASIBLE` | `B::Solution<'_>`: `SolutionContext` |
| `incumbent` | `Events::INCUMBENT` | `B::Solution<'_>`: `SolutionContext` |
| `native` | `Events::NATIVE` | `B::Native<'_>`: whatever only this backend reports |

Write it once for every backend, `impl<B: Backend> Callback<B>`, and it sees each context through its neutral
trait. Write it for one backend, `impl Callback<Highs>`, and it sees that backend's own context types, with
everything the solver offers beyond the traits ([HiGHS callback contexts](highs.md#callback-contexts),
[SCIP callback events](scip.md#callback-events)). A callback reaches the solve registered on its `Hooks`:
`Hooks::new().register_callback(&mut callback)`.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Phase, Script, ScriptStatus, Step};

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

    let mut deadline = Deadline(Duration::from_secs(2));
    let solved = solver.solve_with(Request::new(), Hooks::new().register_callback(&mut deadline))?;
    assert_eq!(solved.status, ScriptStatus::Interrupted { step: 1 });
    Ok(())
}
```

## `Events`

`Events` is a set of observational event kinds: `NONE`, `LOG`, `PROGRESS`, `POLL`, `FEASIBLE`, `INCUMBENT`,
`NATIVE`, and `ALL`. Combine sets with `|`, `&` and `-` (or `union`, `intersection` and `difference`), take every
other event with `!` (`complement`), test them with `contains`, and iterate the single events with `iter()`;
`Display` prints `{LOG, POLL}`. The search events are not in the set: no set of events registers a handler.

```rust
use culmen::prelude::*;

let wanted = Events::LOG | Events::POLL;
assert!(wanted.contains(Events::POLL));
assert!(!wanted.contains(Events::NATIVE));
assert_eq!(wanted.len(), 2);
assert_eq!(format!("{wanted:?}"), "Events(LOG | POLL)");
assert_eq!((Events::ALL - Events::LOG).len(), 5);
assert_eq!(Highs::EVENTS, Events::ALL);
```

`FEASIBLE` is any solution the solver found; `INCUMBENT` is one better than every one before, so the same solution
may arrive as both.

## Search handlers

What changes a solver's search is a capability of the backend, a trait it implements or does not, and each capability
takes one kind of handler:

| Capability | Handler, its one method | Context | HiGHS | SCIP | `Script` |
|---|---|---|---|---|---|
| `LazyConstraints` | `LazyHandler<B>`: `lazy` | `B::Lazy<'_>`: [`LazyContext`](#contexts) | no | yes | yes |
| `UserCuts` | `CutHandler<B>`: `cut` | `B::Cut<'_>`: `CutContext` | no | yes | yes |
| `Heuristics` | `HeuristicHandler<B>`: `heuristic` | `B::Heuristic<'_>`: `HeuristicContext` | yes | yes | yes |
| `Prices` | `Pricer<B>`: `price` | `B::Price<'_>`: `PriceContext` | no | yes | yes |

A handler is registered on the `Hooks` of one solve, beside the callback: `register_lazy`, `register_cut`,
`register_heuristic` and `register_pricer`, each replacing the handler of its kind registered before, and
`register_lazy_and_cut` for one value in both roles, a separator that cuts off integer and fractional points alike.
Each registration method exists only for a backend with the capability, so a handler the backend would ignore does
not compile: a lazy-constraint handler for HiGHS, whose answer would be the optimum of the relaxation without the
constraints, is an error at the call site, not a wrong optimum. Generic code states what it needs,
`fn solve_tour<B: LazyConstraints>(solver: &mut Solver<B>)`, and runs on every backend that has it.

A backend calls a handler only when one of its kind is registered, and gives up nothing for a kind no handler is
registered for: [SCIP](scip.md#callback-events) turns off its dual reductions for a lazy-constraint handler and
activates its pricer for a pricer, never for a callback that asks for events.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Found, Phase, Script, Step};

/// Forbids taking both items, a conflict only the handler knows, on any backend that takes lazy constraints.
struct NotBoth(Col, Col);

impl<B: LazyConstraints> LazyHandler<B> for NotBoth {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        let candidate = ctx.candidate();
        if candidate[self.0] + candidate[self.1] > 1.5 {
            ctx.add_lazy((self.0 + self.1).leq(1)).expect("a valid constraint");
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    let y = m.add_var(Var::binary());
    m.maximize(3.0 * x + 2.0 * y);

    // A scripted backend offers the candidate a solver would: both items.
    let both = Found { progress: Progress::at(Duration::ZERO, Phase::Mip), objective: 5.0, values: vec![1.0, 1.0].into() };
    let mut solver = Solver::new(Script::new([Step::Lazy(both)], Outcome::Infeasible), m.build()?)?;
    let mut not_both = NotBoth(x, y);
    solver.solve_with(Request::new(), Hooks::new().register_lazy(&mut not_both))?;
    assert_eq!(solver.backend().record.lazy.len(), 1);
    Ok(())
}
```

The same registration for HiGHS does not compile:

```compile_fail,E0599
use culmen::prelude::*;

struct NotBoth(Col, Col);

impl<B: LazyConstraints> LazyHandler<B> for NotBoth {
    fn lazy(&mut self, _: &mut B::Lazy<'_>) {}
}

let mut m = Model::new();
let x = m.add_var(Var::binary());
let y = m.add_var(Var::binary());
let mut solver = Solver::new(Highs::new(), m.build().unwrap()).unwrap();
// error[E0599]: the method `register_lazy` exists for struct `Hooks<'_, Highs>`, but its trait bounds were not
// satisfied: `Highs` does not take lazy constraints.
solver.solve_with(Request::new(), Hooks::new().register_lazy(&mut NotBoth(x, y)));
```

## Contexts

Each event's context reads where the solve stands and offers only the actions the solver honours at that point.
Every context is a `Context`, whose one method is `progress()`, so code that needs only where the solve stands, such
as a deadline check, takes any context as `&impl Context`.

| Trait | Reads | Does |
|---|---|---|
| `Context` | `progress()` | nothing |
| `LogContext` | `text()`, the chunk as written, line breaks included; `level()` | nothing |
| `ProgressContext` | the progress | nothing |
| `PollContext` | the progress | `interrupt()`: ask the solver to stop as soon as it can |
| `SolutionContext` | `objective()`, `values()` | nothing |
| `LazyContext` | `objective()`, `candidate()` | `add_lazy(constraint)`: a constraint the candidate may violate, kept for the rest of the search |
| `CutContext` | `relaxation()` | `add_cut(cut)`: a constraint every integer-feasible point satisfies |
| `HeuristicContext` | the progress | `supply(values)`: a full solution |
| `PriceContext` | `mode()`, `sense()`, `duals()`, `tolerance()`; `reduced_cost(column)` and `improves(column)` computed from them | `add_col(column)`: a column to add to the search, returning its id; `branch_early()`: stop pricing at this node; `report_bound(bound)`: a bound on this node's subtree |

A solver that completes a partial solution offers that on its own heuristic context, as an inherent method, since
not every solver honours one: `HighsHeuristic::supply_partial` and the `Script`'s `supply_partial`. SCIP completes a
partial solution only as a [start](solving.md#starts).

Values are `TiSlice<Col, f64>` read with the model's handles, one per column the problem holds when the event is
delivered, the columns priced earlier in the solve included, and constraints are the same
[`Constraint`](modelling.md#comparisons) values a model adds, so `ctx.add_lazy(x.leq(y))` reads as the model does.
The traits are at the crate root, `culmen::PollContext` and so on, and in `culmen::prelude`, so a callback written
with `use culmen::prelude::*;` calls their methods without importing them one by one.

An action returns `Err(ActionError)` when it is refused:

| Variant | Why |
|---|---|
| `Length { expected, found }` | A full solution holds the wrong number of values. |
| `UnknownColumn { col }` | An entry names a column the problem does not have. |
| `NotFinite { col }` | A value or coefficient is NaN or infinite. |
| `Bounds(InvalidBounds)` | A constraint's bounds are invalid after its constant moved into them; `Error::source` is the `InvalidBounds`. |
| `UnknownRow { row }` | An entry of a new column names a row the program does not have. |
| `NotFiniteEntry { row, col }` | A coefficient of a new column is NaN or infinite, as given or once its repeated entries were added up. |
| `ExcludesZero { col, bounds }` | A priced column's bounds do not hold zero. |
| `Integrality { col, integrality }` | A priced column is semi, or of an integrality the backend's classes do not allow beside the program's. |
| `NotFiniteBound { bound }` | A reported bound is NaN or infinite. |
| `Refused(reason)` | The solver refused the action, for the reason given. |

## Pricing

Pricing is column generation inside the solve, branch-and-price on a solver whose search branches: after the
solver has solved a node LP, it hands the pricer the LP's row multipliers and asks for columns that would improve
it. The pricer adds them with `add_col`, as `NewCol`s with entries in the program's rows, and the solver solves the
node LP again and asks again, until a round adds nothing. A pricer is a `Pricer<B>`, registered with
`Hooks::register_pricer` for a backend that implements `Prices`; no set of events turns pricing on, so a callback
never prices by accident: a solver that prices gives up reductions that need every column known up front, and SCIP
refuses to price a program with a quadratic objective.

- **Two modes.** In `PriceMode::ReducedCost { objective }` the node LP is optimal, and `duals()` holds each row's
  dual in the sign convention of [`Duals`](analysis.md): the rate at which the LP optimum changes as the row's active
  bound rises. A column's reduced cost is `cost − Σ duals[r]·a_r`. In `PriceMode::Farkas` the node LP is infeasible,
  `duals()` holds the Farkas multipliers, and the cost counts as zero, so the reduced cost is `−Σ duals[r]·a_r`.
  `reduced_cost(&column)` computes it, and `improves(&column)` says whether the column could improve the LP, or make
  it feasible, beyond `tolerance()`, by the column's bounds and the program's sense: a column rising from zero
  improves a minimisation (and every Farkas round) when its reduced cost is below `−tolerance`, a maximisation when
  it is above `tolerance`. A pricer that ignores Farkas rounds leaves an infeasible node unrepaired, and the solver
  then counts it infeasible.
- **The signs, for both senses.** The convention is the same formula in both senses, so the signs follow from it:

  | | Minimize | Maximize | Farkas, either sense |
  |---|---|---|---|
  | dual of a row at its lower bound (`≥`) | `≥ 0` | `≤ 0` | the multiplier as the solver reports it |
  | dual of a row at its upper bound (`≤`) | `≤ 0` | `≥ 0` | |
  | reduced cost `d` | `cost − Σ duals[r]·a_r` | `cost − Σ duals[r]·a_r` | `−Σ duals[r]·a_r` |
  | a column in `[0, u]`, `u > 0`, improves when | `d < −tolerance` | `d > tolerance` | `d < −tolerance` |

  A column whose bounds hold zero strictly inside (`lower < 0 < upper`) improves when `|d|` exceeds the tolerance in
  the direction it can move, and one with `upper = 0` when the inequality is the other way round; `improves` encodes
  all of it. In Farkas mode the multipliers certify `fᵀAx ≥ fᵀb`, `b` the active side of each row, violated by
  every point within the bounds of the columns held; a column with `Σ f_r·a_r > 0` that can rise from zero weakens
  that certificate, which is the reduced-cost rule of the phase-one minimisation with every cost zero.
- **What is refused.** A priced column's bounds hold zero, so a column left out is the same as one fixed at zero
  (`ExcludesZero`); it is continuous or integer, of a class the backend solves (`Integrality`); its cost and entries
  are finite and name the program's rows. A refused column takes no id.
- **Ids.** `add_col` returns the column's id, the next after the program's columns and every one priced before: the
  first is `Col(n)` for a program of `n` columns. Every point a later event delivers counts it, and when the solve
  ends the `Solver`'s program holds it ([A solve that prices](solving.md#a-solve-that-prices)).
- **What the pricer guarantees.**
  1. *Exact pricing.* Whenever some column the problem does not hold yet improves the node LP, in either mode, the
     pricer adds at least one improving column, or calls `branch_early()`, which ends pricing at the node and
     leaves the LP value no bound unless `report_bound(bound)` gives one (refused in Farkas mode). Only then is a node
     LP's value a bound, and only then is `Outcome::Optimal` a proof.
  2. *No re-adding.* The solver prices the columns it holds itself, within their bounds at the node, so the pricer
     never adds one the problem already holds. A copy is not refused, since two identical columns may be meant (two
     identical trucks), but a column branching fixed, added again, escapes the branching, and the search may then
     revisit what it ruled out.
  3. *Robust rows.* A priced column has entries only in the program's rows, so it enters every lazy constraint and
     cut with coefficient zero, and their duals do not change its reduced cost: every lazy constraint and cut must
     hold for every column the pricer may price, each entering it with zero. A bound given to `report_bound` holds
     for the node's whole subtree over every column the program may price.
- **What a backend that implements `Prices` guarantees.** It calls `price` only after a node LP is solved: optimal in
  `ReducedCost` mode, infeasible in `Farkas` mode, with every program row's multiplier in the convention above. It
  ends pricing at a node only after a round that added no column and did not branch early, and counts the node LP as
  a bound only then; a round that branched early counts only through `report_bound`. Every column added enters the
  search with its cost, bounds, integrality and entries, stays for the rest of the solve, and is part of every point
  reported afterwards. A solve without a pricer runs as if the backend could not price.
- **Branch-and-price.** On a backend whose search branches, pricing at every node with exact pricing is exact
  branch-and-price under the backend's own branching. [SCIP](scip.md#pricing) branches on the master's integer
  columns, as it chooses; culmen has no branching event, so branching on the pricing problem (Ryan–Foster) is out of
  reach.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{NewCol, Phase, PriceMode, ProgramColumn, Script, Step};

/// Adds every pattern of the pool, not added yet, that improves the node LP.
struct Pool {
    patterns: Vec<NewCol>,
    added: Vec<Option<Col>>,
}

impl<B: Prices> Pricer<B> for Pool {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        for (pattern, added) in self.patterns.iter().zip(&mut self.added) {
            if added.is_none() && ctx.improves(pattern) {
                *added = ctx.add_col(pattern.clone()).ok();
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cut rolls of width 10 into 7 pieces of width 3, 2 of width 4 and 3 of width 5, one pattern per width so far.
    let mut m = Model::new();
    let [a, d, f] = [(); 3].map(|_| m.add_var(Var::continuous()));
    let threes = m.add_constraint((3.0 * a).geq(7));
    let fours = m.add_constraint((2.0 * d).geq(2));
    let fives = m.add_constraint((2.0 * f).geq(3));
    m.minimize(a + d + f);
    let pattern = |entries| NewCol { column: ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE), entries };
    let b = pattern(vec![(threes, 2.0), (fours, 1.0)]); // two 3s and a 4
    let e = pattern(vec![(fours, 1.0), (fives, 1.0)]); // a 4 and a 5

    // A scripted backend hands over the duals a solver would: B prices at 1 - 2/3 - 1/2 = -1/6, E at exactly 0.
    let at = Progress::at(Duration::ZERO, Phase::Simplex);
    let round = |objective: f64, duals: Vec<f64>| Step::Price {
        progress: at,
        mode: PriceMode::ReducedCost { objective },
        duals: duals.into(),
    };
    let first = round(29.0 / 6.0, vec![1.0 / 3.0, 0.5, 0.5]);
    let second = round(4.5, vec![1.0 / 3.0, 1.0 / 3.0, 0.5]);
    let end = Outcome::Optimal(Solution { objective: 4.5, values: vec![1.0, 0.0, 1.5, 2.0].into() });
    let mut solver = Solver::new(Script::new([first, second], end), m.build()?)?;

    let mut pool = Pool { patterns: vec![b, e], added: vec![None, None] };
    let solved = solver.solve_with(Request::new(), Hooks::new().register_pricer(&mut pool))?;
    assert_eq!(pool.added, [Some(Col::new(3)), None]);
    assert_eq!(solver.program().num_cols(), 4);
    let best = solved.outcome.solution().ok_or("no solution")?;
    assert!(solver.program().max_violation(&best.values) < 1e-9);
    Ok(())
}
```

## Pairing callbacks

Two callbacks run together as a pair `(a, b)`: each runs only for the events it asked for, `a` first, and the pair
asks for the union. `&mut C` is a callback wherever `C` is, so a pair can borrow both and leave them readable
afterwards. `()` is the callback that asks for nothing. Handlers pair the same way: `(a, b)` is a handler of a kind
wherever both are, both run, `a` first, on one context, so at a price step the columns `b` adds follow those `a`
added. One value in two roles that the backend calls apart is registered once in each, or, for lazy constraints and
cuts, with `register_lazy_and_cut`; two handlers that share state hold it behind a `&Mutex`, since a backend never
calls two of them at once.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{Phase, Script, Step};

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
    solver.solve_with(Request::new(), Hooks::new().register_callback(&mut (&mut count, &mut stop)))?;
    assert_eq!(count.0, 3);
    Ok(())
}
```

The request's stop and observer are themselves a callback, `Watch`, which `solve_with` runs before yours, as the pair
`(watch, callback)` would. `Request::into_watch()` hands out that `Watch` and the settings, for code that drives a
`Backend` directly and registers `&mut (watch, callback)` on its `Hooks`; `Watch::finish` reports a last log line that
never got its line break.

## `Unsupported`

A backend that cannot deliver an observational event leaves it out of `Backend::EVENTS` and names `Unsupported` as
its context type. `Unsupported` is an enum with no values: a callback method for that event compiles, but it can never
be called, since no context can be made. SCIP names it for its native event. A search capability a backend lacks
needs no such type: the backend does not implement the trait, and a handler for it does not compile
([Search handlers](#search-handlers)).

```rust
use culmen::prelude::*;

assert_eq!(Highs::EVENTS, Events::ALL);
assert!(culmen::Script::EVENTS.contains(Events::ALL));

/// Compiles only for a backend that takes lazy constraints.
fn takes_lazy<B: LazyConstraints>() {}
takes_lazy::<culmen::Script>();
```

HiGHS 1.16 offers no lazy constraints, user cuts or pricing; see [Limitations](highs.md#limitations).
[SCIP](scip.md#callback-events), with the `scip` feature, delivers every observational event but a native one and
has every search capability, through plugins it includes in each instance; it refuses pricing on a program with a
quadratic objective with a typed error before it solves.

## Panics and threads

- **A panic in an observer, a stop, a callback or a handler never aborts the process.** It stops the solve as soon as the
  solver allows and then continues unwinding out of `solve` or `solve_with`. The backend stays usable: the next solve
  on the same `Solver` runs normally. Catch it with `std::panic::catch_unwind` around the solve, if you need to.
- **Calls never overlap.** A backend calls one callback or handler method at a time, and only while `solve` runs.
- **They may come from another thread.** A backend may call back from a thread other than the one that called
  `solve`, so `Callback`, every handler and `Observer` are `Send` and `Stop` is `Sync`. HiGHS, for one, may write log lines from
  its worker threads; it serialises every call behind a lock.

## See also

- [Solving](solving.md): the request that carries the stop and the observer.
- [HiGHS](highs.md#callback-contexts): what each HiGHS context offers beyond the traits.
- [SCIP](scip.md#callback-events): the SCIP plugin behind each event, lazy constraints, cuts, heuristic solutions and
  pricing.
- [Testing](testing.md#testing-callbacks): driving callbacks and handlers with the `Script` backend.
- [Backends](backends.md): how events map onto HiGHS, Gurobi and SCIP.
- [Documentation index](README.md).
