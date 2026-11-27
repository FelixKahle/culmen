# Solving

A solve takes three values: a `Solver`, which holds a backend with a program loaded; a `Run`, which configures one
solve's settings, stop, observer, start and handlers; and the `Solved` it returns, which holds the generic `Outcome`
beside the backend's own status. This page covers all three. Watching and stopping a solve are on
[Callbacks](callbacks.md).

## `Solver`

`Solver::new(backend, program)` loads `program` into `backend`. It fails with a
[`LoadError`](#model-classes-and-refusal): `UnsupportedClass` when the backend does not solve the program's class and
`UnsupportedFeature` when it does not take a feature the program uses, both before the backend sees it, or `Backend`
with the backend's own error when the backend refuses the program. A `Solver` can then solve as often as needed,
each solve configured on its own.

| Method | Does |
|---|---|
| `solve()` | Solves under the backend's defaults, with no stop, observer, callback, start or handler. |
| `run()` | A [`Run`](#run) to configure one solve: settings, a stop, an observer, a [`Callback`](callbacks.md#callback), a [start](#starts) and the [search handlers](callbacks.md#search-handlers), ended by `.solve()`. |
| `program()` | The loaded program, grown by every column a solve priced. |
| `backend()` | The backend, to read its own state. |
| `options()`, `options_mut()` | The backend's own options, for a backend that is `Configurable` (`HighsOptions`, `ScipOptions`; the `Script` has none), to read or to change between solves; the backend itself is not reachable mutably, so the program it holds stays the solver's. |
| `into_parts()` | The backend and the program, taken apart; the backend may be behind the program after a failed edit or a solve that priced. |
| `set_col_bounds(..)`, `set_col_costs(..)`, `set_col_integrality(..)`, `set_row_bounds(..)`, `add_row(..)`, `add_rows(..)`, `add_col(..)`, `add_cols(..)` | Edit the loaded program, to solve it again warm ([Editing a loaded program](#editing-a-loaded-program)). |
| `duals()`, `basis()`, `set_basis(..)`, `ranging()`, `primal_ray()`, `dual_ray()`, `iis()`, `feasibility_relaxation(..)` | What a solve reports beyond its outcome, each for a backend that implements its analysis trait ([Analysis](analysis.md)). |

The backend is chosen by type: `Solver<Highs>` solves with HiGHS, `Solver<Scip>` with SCIP ([SCIP](scip.md), the
`scip` feature), `Solver<Script>` replays a script ([Testing](testing.md#the-script-backend)), and a backend of your
own plugs in by implementing [`Backend`](backends.md).

```rust
use std::time::Duration;

use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_vars(3, |i| Var::binary().name(format!("x[{i}]")));
    m.add_constraint(LinExpr::sum([2.0 * x[0], 3.0 * x[1], 4.0 * x[2]]).leq(5.0));
    m.maximize(LinExpr::sum([3.0 * x[0], 4.0 * x[1], 5.0 * x[2]]));

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let first = solver.solve()?;
    let second = solver.run().time_limit(Duration::from_secs(5)).solve()?;
    assert_eq!(first.outcome, second.outcome);
    assert_eq!(solver.program().num_cols(), 3);
    Ok(())
}
```

## Model classes and refusal

Every program has a [class](program.md#model-classes), `program.class()`: `Lp`, `Milp`, `Qp` or `Miqp`. Every backend
declares the classes it solves as `Backend::CLASSES`, a `ModelClasses` set, and `Solver::new` checks the program's class
against it before it calls the backend's `load`. The check is the same for every backend, and a backend's `load` never
sees a class it did not declare.

| Backend | `CLASSES` |
|---|---|
| `Highs` | `{LP, MILP, QP}`: HiGHS 1.16 refuses to solve a QP with integer columns. |
| `Scip` | Every class: a quadratic objective becomes an epigraph ([SCIP](scip.md#quadratic-programs)). |
| `Script` | Every class: it replays its script rather than solving. |

The same holds for [features](program.md#features): every backend declares the features it takes as
`Backend::FEATURES`, a `ProgramFeatures` set (none unless declared), and `Solver::new` refuses a program using any
other before `load`. `Highs` takes every feature; `Scip` takes none, so it refuses semi columns and further
objectives; `Script` takes every feature.

The refusal is typed: `LoadError<E>` is `UnsupportedClass { class, supported }`, naming the program's class and the backend's
set, `UnsupportedFeature { feature, supported }`, naming the first feature the backend lacks and its set, or
`Backend(E)` with the backend's own error. It displays all three, and `Error::source` is the backend's error.

```rust
use culmen::prelude::*;
use culmen::{LoadError, ModelClass};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A QP with an integer column: a MIQP, which HiGHS 1.16 cannot solve.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let n = m.add_var(Var::integer().upper(3));
    m.minimize((x - 0.5) * (x - 0.5) + n);
    let program = m.build()?;
    assert_eq!(program.class(), ModelClass::Miqp);

    let error = Solver::new(Highs::new(), program).unwrap_err();
    assert_eq!(error, LoadError::UnsupportedClass { class: ModelClass::Miqp, supported: Highs::CLASSES });
    assert_eq!(error.to_string(), "Solver: the backend solves {LP, MILP, QP}, not a MIQP");

    // Without the integer column it is a QP, which HiGHS solves.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    m.minimize((x - 0.5) * (x - 0.5));
    let solved = Solver::new(Highs::new(), m.build()?)?.solve()?;
    let best = solved.outcome.solution().ok_or("no solution")?;
    assert!((best.values[x] - 0.5).abs() < 1e-6);
    Ok(())
}
```

## `Run`

`solver.run()` starts a `Run`: the backend's defaults, no stop, no observer, no callback, no start and no handler. It
is built by chaining and ended by `.solve()`, which returns what `solver.solve()` does. Each method replaces what it
sets, so the last call of a method is the one that counts:

| Method | Sets | For |
|---|---|---|
| `.settings(settings)` | Every setting at once, from a [`Settings`](#settings), its time limit included: a time limit set before is replaced. | every backend |
| `.time_limit(duration)` | The wall-clock time the solve may take, and nothing else: the other settings stay, and a later `.settings(..)` replaces it. | every backend |
| `.stop(&stop)` | The [`Stop`](callbacks.md#stop) polled while the solve runs. | every backend |
| `.observer(&mut observer)` | The [`Observer`](callbacks.md#observer) of progress, improving solutions and log lines. | every backend |
| `.progress_interval(duration)` | The least time between two progress reports; `Watch::DEFAULT_PROGRESS_INTERVAL`, one second, unless set. | every backend |
| `.callback(&mut callback)` | The [`Callback`](callbacks.md#callback), run after the stop and the observer, with the backend's full context types. | every backend |
| `.start(start)` | The [start](#starts) the search begins from. | `B: Starts` |
| `.lazy(&mut h)`, `.cut(&mut h)`, `.lazy_and_cut(&mut h)`, `.heuristic(&mut h)`, `.pricer(&mut p)` | The [search handlers](callbacks.md#search-handlers). | `B: LazyConstraints`, `UserCuts`, both, `Heuristics`, `Prices` |

A method of the last two rows exists only for a backend with the capability, so a start or a handler the backend
would ignore does not compile. The run borrows its stop, observer, callback and handlers, so all of them outlive the
solve and can be read afterwards.

## Starts

A `Start` is a solution the solver may start from: `Start::Full(values)`, one value per column, or
`Start::Sparse(entries)`, values for some columns, the solver choosing the rest. A MIP solver takes a feasible start
as its first incumbent and may complete a partial or infeasible one; an LP solver may start from it. A start need not
be feasible, but it must fit the program: `start.check(num_cols)` states it.

A backend that takes a start implements `Starts`, as HiGHS, SCIP and the `Script` do, and only for such a backend
does a `Run` have `.start(start)`. A start is a full `TiVec<Col, f64>` or `Vec<f64>`, or a sparse `Vec<(Col, f64)>`,
and `.solve()` checks it before the backend sees it. A start applies to its one solve; a backend
may also start a solve from the last solve's best solution, as HiGHS
([HiGHS](highs.md#semi-columns-several-objectives-and-starts)) and SCIP ([SCIP](scip.md#how-options-and-settings-layer))
do until an edit.

`.solve()` refuses a start typed, as `SolveError`, and the last solve stays readable:

| Variant | When |
|---|---|
| `Start(StartError::Length { expected, found })` | A full start holds another number of values than the program has columns. |
| `Start(StartError::UnknownCol { col })` | A sparse start names a column the program does not have. |
| `Start(StartError::RepeatedCol { col })` | A sparse start names a column twice. |
| `Start(StartError::NonFinite { col, value })` | A value is NaN or infinite. |

```rust
use culmen::prelude::*;
use culmen::{Improvement, StartError};

#[derive(Default)]
struct First(Option<f64>);

impl Observer for First {
    fn improved(&mut self, i: &Improvement<'_>) {
        self.0.get_or_insert(i.objective);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (value, weight) = ([10.0, 13.0, 7.0, 8.0], [5.0, 6.0, 4.0, 3.0]);
    let mut m = Model::new();
    let take = m.add_vars(4, |_| Var::binary());
    m.add_constraint(LinExpr::sum((0..4).map(|i| weight[i] * take[i])).leq(10.0));
    m.maximize(LinExpr::sum((0..4).map(|i| value[i] * take[i])));
    let mut solver = Solver::new(Highs::new(), m.build()?)?;

    // Items 1 and 3, worth 21, are the optimum: HiGHS takes them as its first incumbent.
    let mut first = First::default();
    let start = vec![(take[0], 0.0), (take[1], 1.0), (take[2], 0.0), (take[3], 1.0)];
    solver.run().observer(&mut first).start(start).solve()?;
    assert_eq!(first.0, Some(21.0));

    let error = solver.run().start(vec![(take[1], 1.0), (take[1], 0.0)]).solve().unwrap_err();
    assert_eq!(error, SolveError::Start(StartError::RepeatedCol { col: take[1] }));
    Ok(())
}
```

## `Settings`

`Settings` is the plain data behind a solve's limits, in terms every backend maps to its own parameters. `None`
leaves a setting to the backend's own options, which in turn default to the solver's. Settings take precedence over
the backend's options for the one solve they are passed to.

| Field | Type | Meaning |
|---|---|---|
| `time_limit` | `Option<Duration>` | The wall-clock time the solve may take. |
| `relative_gap` | `Option<RelativeGap>` | The relative gap between best solution and best bound at which a MIP counts as solved. |
| `absolute_gap` | `Option<AbsoluteGap>` | The absolute gap, in objective units, at which a MIP counts as solved. |
| `threads` | `Option<NonZeroUsize>` | The number of threads the solver may use. |
| `seed` | `Option<u32>` | The seed of the solver's random choices, for reproducible runs. |
| `presolve` | `Option<bool>` | Whether the solver presolves first. |
| `node_limit` | `Option<u64>` | The number of branch-and-bound nodes after which a MIP stops; zero is passed to the solver as zero, which stops at its first check of the limit. |
| `solution_limit` | `Option<NonZeroU64>` | The number of improving solutions (each better than every one before, the first included) after which a MIP stops; `solutions!(n)` writes a literal one. |
| `console` | `bool` | Whether the solver also prints its log to standard output; the log reaches callbacks either way. |

A `Run` sets them all with `.settings(..)`, and the time limit alone with `.time_limit(..)`:

```rust
use std::time::Duration;

use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::binary());
    m.maximize(x);
    let mut solver = Solver::new(Highs::new(), m.build()?)?;

    let settings = Settings {
        relative_gap: Some(relative_gap!(0.01)),
        threads: Some(threads!(4)),
        seed: Some(7),
        node_limit: Some(10_000),
        solution_limit: Some(solutions!(5)),
        ..Settings::default()
    };
    let solved = solver.run().settings(settings).time_limit(Duration::from_secs(30)).solve()?;
    assert_eq!(solved.outcome.objective(), Some(1.0));
    Ok(())
}
```

How each setting maps onto HiGHS is on [HiGHS](highs.md#how-options-and-settings-layer), onto SCIP on
[SCIP](scip.md#how-options-and-settings-layer); the [Backends](backends.md#settings) page maps them onto HiGHS, Gurobi
and SCIP side by side.

## Checked literals

A gap is a finite number, zero or above, and a thread count or a solution limit is not zero. The types say so:
`RelativeGap`, `AbsoluteGap`, `NonZeroUsize` and `NonZeroU64` cannot hold anything else. Their literals are written
with macros that check them when the code is compiled, so a bad literal never reaches a running program:

```rust
use culmen::prelude::*;

let gap = relative_gap!(0.01);
let units = absolute_gap!(2.5);
let four = threads!(4);
let first = solutions!(1);
assert_eq!((gap.get(), units.get(), four.get(), first.get()), (0.01, 2.5, 4, 1));
assert_eq!(gap.to_string(), "RelativeGap(0.01)");
```

```compile_fail,E0080
use culmen::prelude::*;

let gap = relative_gap!(-0.01); // RelativeGap: below zero
```

```compile_fail,E0080
use culmen::prelude::*;

let none = threads!(0); // threads: zero
```

A number read at run time is checked with `RelativeGap::new`, `AbsoluteGap::new`, their `TryFrom<f64>`, or
`NonZeroUsize::new` and `NonZeroU64::new`, which refuse it with an error you can report:

```rust
use culmen::{InvalidRelativeGap, RelativeGap};

let read_from_config = -0.5;
assert_eq!(RelativeGap::try_from(read_from_config), Err(InvalidRelativeGap::Negative));
assert_eq!(RelativeGap::new(f64::NAN), Err(InvalidRelativeGap::NonFinite));
assert_eq!(InvalidRelativeGap::Negative.to_string(), "RelativeGap: below zero");
```

`f64::from(gap)` reads a gap back as a number, as `gap.get()` does.

## `Solved` and `Outcome`

A solve returns `Result<Solved<B::Status>, SolveError<B::Error>>`. `Solved` is plain data:

| Field | Holds |
|---|---|
| `outcome` | How the solve ended, as an `Outcome`. |
| `status` | The backend's own status, for diagnostics; for HiGHS a [`HighsStatus`](highs.md#statuses-and-outcomes), for SCIP a [`ScipStatus`](scip.md#statuses-and-outcomes). |
| `elapsed` | The wall-clock time of the solve. |
| `nodes` | The branch-and-bound nodes the solve explored; `None` when it ran no branch and bound or the backend does not count them. |
| `iterations` | The iterations of the LP and QP algorithms the solve ran, summed over every node; `None` when the backend does not count them. |

`Outcome` holds exactly what each ending proves:

| Variant | Means | Holds |
|---|---|---|
| `Optimal(solution)` | Optimal within the solver's default tolerances. | The solution. |
| `GapReached { solution, bound }` | Stopped because the requested relative or absolute gap was reached. | The best solution and the best bound proved. |
| `Stopped { reason, best, bound }` | Stopped before proving optimality. | Why, the best solution if any, the best bound if any. |
| `Infeasible` | No point satisfies the constraints. | Nothing. |
| `Unbounded` | The objective improves without limit. | Nothing. |
| `InfeasibleOrUnbounded` | One of the two, and the solver did not tell which. | Nothing. |

A `Solution` is `{ objective, values }`: the objective value, quadratic part and offset included, and one value per
column. Every objective value and bound includes the program's quadratic part and offset and is in the model's own
sense. For a program with [several objectives](program.md#several-objectives), `objective` is the program's own
objective, the first; `program.objective_values(&solution.values)` gives every objective's value, and a backend
reports no bound, since its bound is that of the last stage or of the blend. `outcome.solution()` returns the
best solution of any variant, `None` when there is none, `outcome.objective()` its objective value, and
`outcome.into_solution()` takes it by value. `outcome.bound()` is the best bound proved: the optimum for `Optimal`,
the `bound` of `GapReached` and `Stopped`, and `None` when none was proved; with `objective()` it gives the gap a solve
left.

```rust
use culmen::prelude::*;

fn report(outcome: &Outcome) -> String {
    match outcome {
        Outcome::Optimal(best) => format!("optimal {}", best.objective),
        Outcome::GapReached { solution, bound } => format!("{} within the gap of {bound}", solution.objective),
        Outcome::Stopped { reason, best: Some(best), .. } => format!("stopped ({reason:?}) at {}", best.objective),
        Outcome::Stopped { reason, best: None, .. } => format!("stopped ({reason:?}) without a solution"),
        Outcome::Infeasible | Outcome::Unbounded | Outcome::InfeasibleOrUnbounded => "no solution".to_owned(),
    }
}

let stopped = Outcome::Stopped { reason: StopReason::TimeLimit, best: None, bound: Some(10.0) };
assert_eq!(report(&stopped), "stopped (TimeLimit) without a solution");
assert_eq!((stopped.solution(), stopped.bound()), (None, Some(10.0)));
```

## `StopReason`

`StopReason` says why a `Stopped` solve stopped: `TimeLimit`, `Interrupted` (a [stop](callbacks.md#stop) or a
callback asked), `NodeLimit`, `SolutionLimit`, `IterationLimit`, `ObjectiveLimit` (a bound or target set in the
backend's options), `MemoryLimit`, or `Other`, when the solver did not name a reason; then the backend's status says
more. The enum is `#[non_exhaustive]`: a later version may add reasons, so a `match` on it needs a `_` arm.

## Reading a solution

A solution is read with the handles the model returned:

- `solution.values[x]` for one variable, since `values` is a `TiVec<Col, f64>`.
- `family.values(&solution.values)` for a whole [`VarMap`](modelling.md#keyed-families), as `(&key, value)` pairs.
- `expr.value(&solution.values)` for any [expression](modelling.md#expressions).
- `program.row(r).activity(&solution.values)` for a constraint's activity, `program.activities(&solution.values)`
  for every row's, and `program.max_violation(&solution.values)` to check the solution against the program
  ([Checking a point](program.md#checking-a-point)).

Integer variables come back as floating-point numbers within the solver's integrality tolerance; round them
(`value.round()`) or compare against `0.5` for binaries.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let take = m.add_var_map(["cork", "glass", "tin"], |item| Var::binary().name(format!("take[{item}]")));
    let weight = 3.0 * take["cork"] + 4.0 * take["glass"] + 2.0 * take["tin"];
    let cap = m.add_constraint(("capacity", weight.clone().leq(6.0)));
    m.maximize(5.0 * take["cork"] + 6.0 * take["glass"] + 3.0 * take["tin"]);

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let solved = solver.solve()?;
    let best = solved.outcome.solution().ok_or("no solution")?;

    let taken: Vec<&str> = take.values(&best.values).filter(|&(_, v)| v > 0.5).map(|(&item, _)| item).collect();
    assert_eq!(taken, ["glass", "tin"]);
    assert_eq!(weight.value(&best.values).round(), 6.0);
    assert_eq!(solver.program().row(cap).activity(&best.values).round(), 6.0);
    assert!(solver.program().max_violation(&best.values) < 1e-6);
    Ok(())
}
```

## Editing a loaded program

A `Solver` edits the program it holds, so the next solve starts from where the last one ended: a cut added to an LP
relaxation, a column priced in by column generation, a variable fixed in a dive. Each edit is one call:

| Method | Changes |
|---|---|
| `set_col_bounds([(col, bounds), ..])` | Columns' bounds; `bounds` is anything that converts into `Bounds`. |
| `set_col_costs([(col, cost), ..])` | Columns' costs in the program's own objective. |
| `set_col_integrality([(col, integrality), ..])` | Columns' integrality. |
| `set_row_bounds([(row, bounds), ..])` | Rows' bounds; `..`, `Bounds::FREE`, switches a row off. |
| `add_row(row)`, `add_rows(rows)` | Appends rows and returns them: anything [`Model::add_constraint`](modelling.md#constraints) takes, a `Constraint`, a `(name, constraint)` pair or a `ProgramRow`. |
| `add_col(column)`, `add_cols(columns)` | Appends columns and returns them: a `NewCol`, built as `NewCol::continuous().cost(2).entry(row, 1)` with its entries in existing rows, or a bare `ProgramColumn` in no row. |

Every edit keeps these rules:

- **All or none.** When an edit returns `Ok`, `solver.program()` is the edited program and the backend holds it. When
  it returns `Err`, both are as they were.
- **Checked as `build` checks.** Every id an edit names is the program's; costs and coefficients are finite, as
  given and after repeated terms add up, and terms that cancel are dropped; a constraint's constant moves into its
  bounds as [`Model::build`](modelling.md#building-and-modelerror) moves it; a semi column keeps bounds a semi column
  may have; and the backend solves the class and takes the features the edited program would have. A refusal is an
  `EditError` naming the fault, decided before the backend sees the edit.
- **The last value given holds.** A setter given one id twice keeps the last value; an edit that gives nothing
  changes nothing and never reaches the backend.
- **Ids keep their meaning.** Rows and columns are appended after every one before and never deleted. To take a row
  out, free its bounds; to take a column out, fix it at zero.
- **The last solve is no longer read.** An edit clears the result the [analyses](analysis.md) read, so `duals()`
  after an edit is `NotSolved` until the next solve; an edit refused by its checks keeps it.

A backend changes its model in place through [`Backend::edit`](backends.md#the-seam) and keeps its warm start; one that
does not loads the edited program whole, which is always correct and starts cold. When the backend refuses an edit,
`EditError::Backend`, the `Solver` restores its program and loads it into the backend again; should that fail too, the
next call that runs the backend loads it first. [HiGHS](highs.md#editing-a-loaded-program) changes its model in place;
[SCIP](scip.md#how-options-and-settings-layer) loads the edited program into a new instance.

```rust
use culmen::prelude::*;
use culmen::{EditError, NewCol};

/// The optimal objective of the next solve.
fn best(solver: &mut Solver<Highs>) -> Result<f64, Box<dyn std::error::Error>> {
    let solved = solver.solve()?;
    Ok(solved.outcome.objective().ok_or("no solution")?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Maximize x + y over the unit square.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1.0));
    let y = m.add_var(Var::continuous().upper(1.0));
    m.maximize(x + y);
    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    assert_eq!(best(&mut solver)?, 2.0);

    // A cut, then a column worth 2 that the cut holds too.
    let cut = solver.add_row(("cut", (x + y).leq(1.5)))?;
    assert_eq!(best(&mut solver)?, 1.5);
    let z = solver.add_col(NewCol::continuous().cost(2).bounds(bounds!(0.0, 1.0)).entry(cut, 1))?;
    assert_eq!(best(&mut solver)?, 2.5);

    // The cut switched off, and y fixed at zero.
    solver.set_row_bounds([(cut, ..)])?;
    solver.set_col_bounds([(y, bounds!(0.0, 0.0))])?;
    assert_eq!(best(&mut solver)?, 3.0);

    // A refused edit changes nothing.
    let refused = solver.set_col_costs([(z, f64::INFINITY)]);
    assert_eq!(refused, Err(EditError::NonFiniteCost { col: z, value: f64::INFINITY }));
    assert_eq!(solver.program().columns()[z].cost, 2.0);
    Ok(())
}
```

The examples `cutting_planes` and `column_generation` of the `culmen` crate edit a program round after round.

## A solve that prices

A [pricer](callbacks.md#pricing), registered with `Run::pricer` on a backend that implements `Prices`
([SCIP](scip.md#pricing)), adds columns during the solve, column generation at every node of the search. Those columns
are the program's from then on:

- **The program grows.** Each priced column continues the program's numbering: the first is `Col(n)` for a program
  of `n` columns, and `add_col` returns that id. When the solve ends, however it ends (returning `Ok`, failing, or
  unwinding from a panic), `solver.program()` holds every column priced, in the order of their ids, with its cost,
  bounds, integrality and entries as checked, and the solution reads them with their ids: `best.values.len()` is
  `solver.program().num_cols()`.
- **The backend reloads.** The backend held the program before the solve, so the next call that runs it loads the
  grown program first; a failure there is that call's `Backend` error.
- **The analyses find no solve to read.** After a solve that priced at least one column, `duals()` and the other
  analyses that read a solve are `NotSolved` until the next solve that prices nothing.
- **The next solve starts with the columns.** It may start from a `Start` naming them, and edits work on the grown
  program as on any other.

```rust
use std::time::Duration;

use culmen::prelude::*;
use culmen::{NewCol, Phase, PriceMode, Script, Step};

/// Prices one column in at every price step.
struct OneMore;

impl<B: Prices> Pricer<B> for OneMore {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        let row = Row::new(0);
        let column = NewCol::continuous().cost(1).entry(row, 1);
        assert_eq!(ctx.add_col(column), Ok(Col::new(1)));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    m.add_constraint(x.geq(1));
    m.minimize(2.0 * x);

    let step = Step::Price {
        progress: Progress::at(Duration::ZERO, Phase::Simplex),
        mode: PriceMode::ReducedCost { objective: 2.0 },
        duals: vec![2.0].into(),
    };
    let end = Outcome::Optimal(Solution { objective: 1.0, values: vec![0.0, 1.0].into() });
    let mut solver = Solver::new(Script::new([step], end), m.build()?)?;
    let solved = solver.run().pricer(&mut OneMore).solve()?;

    assert_eq!(solver.program().num_cols(), 2);
    assert_eq!(solved.outcome.solution().map(|s| s.values.len()), Some(2));
    Ok(())
}
```

The `branch_and_price` example of the `culmen` crate prices cutting patterns at every node on SCIP.

## Errors

`Solver::new` fails with a [`LoadError<B::Error>`](#model-classes-and-refusal): a class the backend does not solve, a
feature it does not take, or the backend's own error when it refuses the program. A solve fails with a
`SolveError<B::Error>`: a [start](#starts) that does not fit the program, decided before the backend sees it, or
`SolveError::Backend` with the backend's own error, a setting it cannot
apply, or a failed run. An edit fails with an [`EditError<B::Error>`](#editing-a-loaded-program): an id the program does
not have, a number that is not finite, invalid bounds, a class or feature the backend lacks, all decided before the
backend sees the edit, or `EditError::Backend` when the backend refuses it. For HiGHS the backend's error is
[`HighsError`](highs.md#errors), for SCIP [`ScipError`](scip.md#errors). A solve that ends without a solution is not
an error; it is an `Outcome` such as `Infeasible` or `Stopped`.

`solve` panics when an observer, a callback or a handler panics: the panic continues out of the call once the solver
has stopped ([Panics and threads](callbacks.md#panics-and-threads)).

## See also

- [Callbacks](callbacks.md): stops, observers, callbacks and search handlers.
- [Analysis](analysis.md): duals, the basis, ranging, rays, IIS and feasibility relaxation after a solve.
- [HiGHS](highs.md): the HiGHS backend's options, statuses and errors.
- [SCIP](scip.md): the SCIP backend's options, statuses and errors, and pricing at every node.
- [Testing](testing.md): solving with a scripted backend.
- [Documentation index](README.md).
