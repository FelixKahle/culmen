# Solving

A solve takes three values: a `Solver`, which holds a backend with a program loaded; a `Request`, which carries one
solve's settings, stop and observer; and the `Solved` it returns, which holds the generic `Outcome` beside the
backend's own status. This page covers all three. Watching and stopping a solve are on
[Callbacks](callbacks.md).

## `Solver`

`Solver::new(backend, program)` loads `program` into `backend`. It fails with a
[`LoadError`](#model-classes-and-refusal): `Unsupported` when the backend does not solve the program's class, before the
backend sees it, or `Backend` with the backend's own error when the backend refuses the program. A `Solver` can then
solve as often as needed, each time with its own request.

| Method | Does |
|---|---|
| `solve(request)` | Solves under the request's settings, polling its stop and reporting to its observer. |
| `solve_with(request, &mut callback)` | The same, and also runs a [`Callback`](callbacks.md#callback) with the backend's full context types. |
| `program()` | The loaded program. |
| `backend()`, `backend_mut()` | The backend, to read its state or change its own options between solves. |
| `into_parts()` | The backend and the program, taken apart. |

The backend is chosen by type: `Solver<Highs>` solves with HiGHS, `Solver<Script>` replays a script
([Testing](testing.md#the-script-backend)), and a backend of your own plugs in by implementing
[`Backend`](backends.md).

```rust
use std::time::Duration;

use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_vars(3, |i| Var::binary().name(format!("x[{i}]")));
    m.add_constraint(LinExpr::sum([2.0 * x[0], 3.0 * x[1], 4.0 * x[2]]).leq(5.0));
    m.maximize(LinExpr::sum([3.0 * x[0], 4.0 * x[1], 5.0 * x[2]]));

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let first = solver.solve(Request::new())?;
    let second = solver.solve(Request::new().time_limit(Duration::from_secs(5)))?;
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
| `Highs` | `{LP, MILP, QP}`: HiGHS 1.15 refuses to solve a QP with integer columns. |
| `Script` | Every class: it replays its script rather than solving. |

The refusal is typed: `LoadError<E>` is `Unsupported { class, supported }`, naming the program's class and the backend's
set, or `Backend(E)` with the backend's own error. It displays both, and `Error::source` is the backend's error.

```rust
use culmen::prelude::*;
use culmen::{LoadError, ModelClass};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A QP with an integer column: a MIQP, which HiGHS 1.15 cannot solve.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let n = m.add_var(Var::integer().upper(3));
    m.minimize((x - 0.5) * (x - 0.5) + n);
    let program = m.build()?;
    assert_eq!(program.class(), ModelClass::Miqp);

    let error = Solver::new(Highs::new(), program).unwrap_err();
    assert_eq!(error, LoadError::Unsupported { class: ModelClass::Miqp, supported: Highs::CLASSES });
    assert_eq!(error.to_string(), "Solver: the backend solves {LP, MILP, QP}, not a MIQP");

    // Without the integer column it is a QP, which HiGHS solves.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    m.minimize((x - 0.5) * (x - 0.5));
    let solved = Solver::new(Highs::new(), m.build()?)?.solve(Request::new())?;
    let best = solved.outcome.solution().ok_or("no solution")?;
    assert!((best.values[x] - 0.5).abs() < 1e-6);
    Ok(())
}
```

## `Request`

A `Request` is built by chaining. Each method replaces what it sets:

| Method | Sets |
|---|---|
| `Request::new()` | The solver's defaults, no stop, no observer. |
| `.time_limit(duration)` | The wall-clock time the solve may take. |
| `.settings_from(settings)` | Every setting at once, from a [`Settings`](#settings); it replaces a time limit set before. |
| `.stop(&stop)` | The [`Stop`](callbacks.md#stop) polled while the solve runs. |
| `.observer(&mut observer)` | The [`Observer`](callbacks.md#observer) of progress, improving solutions and log lines. |
| `.progress_interval(duration)` | The least time between two progress reports; `Request::DEFAULT_PROGRESS_INTERVAL`, one second, unless set. |

`request.settings()` reads the settings back. The request borrows its stop and observer, so both outlive the solve
and can be read afterwards.

## `Settings`

`Settings` is the plain data behind a request's limits, in terms every backend maps to its own parameters. `None`
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
| `node_limit` | `Option<u64>` | The number of branch-and-bound nodes after which a MIP stops. |
| `solution_limit` | `Option<u64>` | The number of feasible solutions after which a MIP stops. |
| `console` | `bool` | Whether the solver also prints its log to standard output; the log reaches callbacks either way. |

The request sets the time limit itself; every other setting goes through `settings_from`:

```rust
use std::time::Duration;

use culmen::prelude::*;

let settings = Settings {
    time_limit: Some(Duration::from_secs(30)),
    relative_gap: Some(relative_gap!(0.01)),
    threads: Some(threads!(4)),
    seed: Some(7),
    node_limit: Some(10_000),
    ..Settings::default()
};
let request = Request::new().settings_from(settings.clone());
assert_eq!(request.settings(), &settings);
```

How each setting maps onto HiGHS is on [HiGHS](highs.md#how-options-and-settings-layer); the
[Backends](backends.md#settings) page maps them onto HiGHS, Gurobi and SCIP side by side.

## Checked literals

A gap is a finite number, zero or above, and a thread count is not zero. The types say so: `RelativeGap`,
`AbsoluteGap` and `NonZeroUsize` cannot hold anything else. Their literals are written with macros that check them
when the code is compiled, so a bad literal never reaches a running program:

```rust
use culmen::prelude::*;

let gap = relative_gap!(0.01);
let units = absolute_gap!(2.5);
let four = threads!(4);
assert_eq!((gap.get(), units.get(), four.get()), (0.01, 2.5, 4));
assert_eq!(gap.to_string(), "RelativeGap(0.01)");
```

```compile_fail
use culmen::prelude::*;

let gap = relative_gap!(-0.01); // RelativeGap: below zero
```

```compile_fail
use culmen::prelude::*;

let none = threads!(0); // threads: zero
```

A number read at run time is checked with `RelativeGap::new`, `AbsoluteGap::new`, their `TryFrom<f64>`, or
`NonZeroUsize::new`, which refuse it with an error you can report:

```rust
use culmen::{InvalidRelativeGap, RelativeGap};

let read_from_config = -0.5;
assert_eq!(RelativeGap::try_from(read_from_config), Err(InvalidRelativeGap::Negative));
assert_eq!(RelativeGap::new(f64::NAN), Err(InvalidRelativeGap::NotFinite));
assert_eq!(InvalidRelativeGap::Negative.to_string(), "RelativeGap: below zero");
```

`f64::from(gap)` reads a gap back as a number, as `gap.get()` does.

## `Solved` and `Outcome`

A solve returns `Result<Solved<B::Status>, B::Error>`. `Solved` is plain data:

| Field | Holds |
|---|---|
| `outcome` | How the solve ended, as an `Outcome`. |
| `status` | The backend's own status, for diagnostics; for HiGHS a [`HighsStatus`](highs.md#statuses-and-outcomes). |
| `elapsed` | The wall-clock time of the solve. |

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
sense. `outcome.solution()` returns the
best solution of any variant, `None` when there is none, and `outcome.into_solution()` takes it by value.

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
assert_eq!(stopped.solution(), None);
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
- `program.row(r).activity(&solution.values)` for a constraint's activity, and `program.max_violation(&solution.values)`
  to check the solution against the program ([Checking a point](program.md#checking-a-point)).

Integer variables come back as floating-point numbers within the solver's integrality tolerance; round them
(`value.round()`) or compare against `0.5` for binaries.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let take = m.add_var_map(["cork", "glass", "tin"], |item| Var::binary().name(format!("take[{item}]")));
    let weight = 3.0 * take["cork"] + 4.0 * take["glass"] + 2.0 * take["tin"];
    let cap = m.add_named_constraint("capacity", weight.clone().leq(6.0));
    m.maximize(5.0 * take["cork"] + 6.0 * take["glass"] + 3.0 * take["tin"]);

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let solved = solver.solve(Request::new())?;
    let best = solved.outcome.solution().ok_or("no solution")?;

    let taken: Vec<&str> = take.values(&best.values).filter(|&(_, v)| v > 0.5).map(|(&item, _)| item).collect();
    assert_eq!(taken, ["glass", "tin"]);
    assert_eq!(weight.value(&best.values).round(), 6.0);
    assert_eq!(solver.program().row(cap).activity(&best.values).round(), 6.0);
    assert!(solver.program().max_violation(&best.values) < 1e-6);
    Ok(())
}
```

## Errors

`Solver::new` fails with a [`LoadError<B::Error>`](#model-classes-and-refusal): a class the backend does not solve,
or the backend's own error when it refuses the program. `solve` and `solve_with` fail with the backend's own error type,
`B::Error`: a setting it cannot apply, or a failed run. For HiGHS that is [`HighsError`](highs.md#errors). A solve
that ends without a solution is not an error; it is an `Outcome` such as `Infeasible` or `Stopped`.

`solve` and `solve_with` panic when an observer or callback panics: the panic continues out of the call once the
solver has stopped ([Panics and threads](callbacks.md#panics-and-threads)).

## See also

- [Callbacks](callbacks.md): stops, observers and callbacks.
- [HiGHS](highs.md): the HiGHS backend's options, statuses and errors.
- [Testing](testing.md): solving with a scripted backend.
- [Documentation index](README.md).
