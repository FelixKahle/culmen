# Design

This page states the principles the API follows and why. Each one is a guarantee you can rely on when you use
culmen, and a rule a contribution keeps.

## The program is checked data

A [`Program`](program.md#program) is a value, built once. Every number in it was checked when it was built: costs,
coefficients and the offset are finite, bounds are valid intervals, and each row holds each column at most once.
Nothing downstream checks again, and nothing downstream can meet a program that was not checked: `Program` has no
public fields, its only constructors check, and deserialization goes through the same check.

The one program that changes is the one a `Solver` holds, and only through the `Solver`'s
[edits](solving.md#editing-a-loaded-program). Each edit runs the checks `build` would run on the edited program before
it changes anything, and is all or none, so `solver.program()` is always a program `build` accepts; ids are only ever
appended, so a handle never changes its meaning. Everywhere else a program, once built, stays as it is.

The check happens in one place, `build`, and its error names what is at fault. A model's error names the variable or
constraint by the name you gave it (`` Model: constraint `big` has the non-finite coefficient inf on variable `x` ``),
because the model runs the program's checks in its own terms; every model `build` accepts gives a program those checks
accept.

Holding no solver state, a program can be cloned, compared, serialized, written as an [LP file](program.md#lp-files),
and loaded into any backend.

## Typed handles

`Col` and `Row` are distinct `Copy` types, not `usize`. A solution is a `TiVec<Col, f64>`, indexed by `Col` only, so
reading a solution with a row, or a row table with a column, does not compile. The handles a model returns are the
columns and rows of the program it builds, so there is no mapping to keep between "model variables" and "solver
columns": `solution.values[x]` is the value of the variable `x`.

The same holds beyond the program. HiGHS's cut pool ranges over the presolved problem, whose columns are not the
program's; they are `ReducedCol`, a third type, so a cut cannot be read as a row of your program by mistake. SCIP's
contexts read every value through SCIP's original variables, so they hand out the program's columns whatever presolve
did, and a column priced during a solve is a `Col` that continues the program's numbering.

Keyed families give handles domain keys (`x[(product, location)]`), and a family is plain data you can iterate,
look up and read a solution through.

## Literals are checked when compiled

A value with an invariant has a type that holds it: `Bounds` is a valid interval, `RelativeGap` and `AbsoluteGap`
are finite and not negative, a thread count is a `NonZeroUsize` and a solution limit a `NonZeroU64`. A literal of such
a type is written with a macro, `bounds!(0.0, 40.0)`, `relative_gap!(0.01)`, `threads!(4)`, `solutions!(5)`, which
checks it when the code is compiled, so a typo in a literal is a compile error, not a run-time `unwrap` panic:

```compile_fail,E0080
use culmen::prelude::*;

let gap = relative_gap!(-0.01); // RelativeGap: below zero
```

Numbers read at run time go through the fallible constructors (`Bounds::new`, `RelativeGap::try_from`), whose errors
say what was refused. Code never writes `.unwrap()` to make a literal.

## Plain data and sealed values

A type without an invariant is plain data with `pub` fields and no getters: `ProgramPlan`, `ProgramColumn`,
`Constraint`, `LinExpr`, `Settings`, `Solution`, `Progress`, `HighsOptions`, `HighsStatus`. You build them with struct
literals and `..Default::default()`, read and change their fields directly, and match on them.

A type with an invariant keeps its fields private behind a checked constructor: `Bounds`, `Program`, the gaps. Their
fields cannot be written from outside, not even all at once:

```compile_fail,E0451
use culmen::Bounds;

let empty = Bounds { lower: 2.0, upper: 1.0 }; // the fields are private
```

Constructors take what converts into their input without loss, so a caller passes `0` where an `f64` is wanted, a
`Bounds` where a `Var` is wanted, and an owned or borrowed value alike. A conversion that can fail is never implicit;
it is a `TryFrom` or a `new` returning `Result`, so the refusal is visible where the value is made.

## Which types may grow: `#[non_exhaustive]`

Whether a public type is `#[non_exhaustive]` follows from what a user does with it, the same in every crate:

- **A struct with `pub` fields is never non-exhaustive.** Plain data is built with a struct literal and
  `..Default::default()` and taken apart by destructuring, and Rust forbids both outside the crate for a
  non-exhaustive struct. A new field is therefore a breaking change, made only in a breaking release. A struct with
  private fields needs no attribute.
- **An error enum is non-exhaustive**, and so is every enum that only an error carries as its detail
  (`InvalidBounds`, `Misfit`, `Table`, `ExprPlace`, `UnsoundQp`, `Dimension`, `Number`, …). A check or a backend
  learns new refusals; a caller matches the ones it handles and passes the rest on. Inside its crate the match stays
  exhaustive, so an error's `Display` remains one exhaustive table.
- **An enum that lists an open set of kinds is non-exhaustive**: a solver's own statuses, return codes, option values
  and log or poll kinds (`ModelStatus`, `CallStatus`, `LpSolver`, `PollSource`, `Status`, `RetCode`, `Emphasis`,
  `ParamValue`, …), which grow with the solver's versions, and culmen's own lists of kinds that a further feature
  adds to (`ModelClass`, `ProgramFeature`, `Analysis`, `Phase`, `StopReason`, `Edit`, `Step`, `ScipEvent`,
  `LazySource`, `HighsNative`, `ModelFormat`). Code that must handle every kind is told so by a declared set
  instead (`Backend::CLASSES`, `FEATURES`), which refuses what it does not list.
- **An enum whose variants its meaning closes stays exhaustive**, so that code which must handle every case is
  checked to: `Sense`, `Integrality`, `Outcome`, `Objectives`, `Combination`, `PriceMode`, `Start`, `NewRow`,
  `Side`, `Bound`, `Line`, `BasisStatus`, `LogLevel`, `ScriptStatus`, `Switch`, and `Unsupported`, which
  has no variant. A new variant there changes what every match must do, so it is a breaking change by design.

`culmen_testing` is not published and is used only inside the workspace, where a change and every match it breaks
land in one commit; it follows the rules for structs and leaves its enums exhaustive.

## The solver-neutral seam

`culmen_core` knows no solver. A solver plugs in by implementing [`Backend`](backends.md#the-seam), an open trait:

- **Observational events are typed and declared.** A backend names one context type per observational event kind
  and lists what it delivers in `Backend::EVENTS`. An event it cannot deliver takes `Unsupported`, an enum with no
  values, so a callback for it compiles everywhere and never runs on that backend. A callback only watches, so asking
  for every event costs a solver no reduction.
- **What changes the search is a capability.** Lazy constraints, user cuts, heuristic solutions and pricing are traits
  a backend implements or leaves out (`LazyConstraints`, `UserCuts`, `Heuristics`, `Prices`), each naming the context
  of its handler (`LazyHandler`, `CutHandler`, `HeuristicHandler`, `Pricer`). A handler is registered on the `Run`
  of a solve only for a backend that has the capability, so a lazy constraint a solver would ignore, which would turn
  a relaxation's optimum into a wrong answer, does not compile, and a backend gives up a reduction for a handler only
  when one is registered.
- **Model classes are declared.** A backend lists the classes of program it solves in `Backend::CLASSES`, and
  `Solver::new` refuses any other class with a typed `LoadError::Unsupported` before the backend sees the program, the
  same way for every backend. A class is a property of the program, which edits change, so it stays a run-time check.
- **Capabilities of the backend are traits.** Taking a start and each analysis after a solve are traits a backend
  implements or leaves out (`Starts`, `DualAnalysis`, …), and the `Solver` methods that need one exist only for a
  backend that has it: `solver.duals()` on a backend without duals, or a start handed to a backend that takes none,
  does not compile.
- **Actions live where the solver honours them.** Only a poll context can interrupt, only a lazy context can add a
  lazy constraint, only a heuristic context can supply a solution, only a price context can add a column. Calling an
  action at a point where the solver would ignore it does not compile, and an action only some solvers honour, as
  completing a partial heuristic solution, is an inherent method of their contexts.
- **Extras stay with their backend.** Each context type is the backend's own; what one solver reports beyond the
  neutral traits is an inherent method or a native event. Generic code sees the traits; code for one backend sees
  everything.
- **The convenient path is built on the seam, not beside it.** The stop and the observer of a `Run` run as `Watch`,
  an ordinary generic callback that the `Solver` runs before yours, so they work on every backend unchanged.
- **Settings are neutral, options are the backend's.** `Settings` holds what every solver understands; a backend's
  own options hold the rest; settings apply after options, per solve.
- **Outcomes are neutral, statuses are kept.** Each solve maps the solver's status onto `Outcome`, which holds
  exactly what the ending proves, and keeps the native status beside it for diagnostics.

[Backends](backends.md) shows how HiGHS, Gurobi and SCIP map onto every part of this seam.

## A solve never aborts

A solve is something an application runs inside a service, so a defect in user code during a solve must not take
the process down. A panic in an observer, a stop, a callback or a handler stops the solve as soon as the solver allows, and
then continues unwinding out of `solve` on the caller's thread, where it can be caught. The backend stays usable.
No profile sets `panic = "abort"`.

The stop is cooperative: it is polled at the solver's poll points, and the solve ends with what it found, as an
ordinary `Outcome::Stopped` with the best solution and bound.

## Where `unsafe` lives

`culmen_core` forbids `unsafe` (`#![forbid(unsafe_code)]`); the facade and the macros contain none. All `unsafe` is in
the two backends, `culmen_highs` and `culmen_scip`, where the solvers' C APIs make it necessary, and every `unsafe`
block there carries a `// SAFETY:` comment saying why it is sound. The parts that matter most, for HiGHS:

- **One instance, one owner.** A HiGHS instance is created with the `Highs` value that owns it and destroyed when
  that value drops, and every call that changes it borrows it mutably, so no two calls ever overlap on one instance.
  Destroying an instance also shuts down the HiGHS task scheduler of the destroying thread, so an instance dropped
  while its thread is inside a run or a callback waits until that run ends and is destroyed then, on that thread.
- **Callback data lives exactly as long as the run.** The data HiGHS's callback reaches lives on the stack of
  `solve`; it is installed just before HiGHS runs and uninstalled just after, before it is dropped, so no pointer
  outlives the run.
- **Calls are serialised.** HiGHS may log from worker threads, so everything a callback touches sits behind one
  lock, the callback is `Send` and the stop `Sync`; a call that re-enters the run on the same thread is not
  delivered.
- **Panics never cross the C boundary.** A panic in a callback is caught inside the trampoline, turns every later
  poll of the run into an interrupt, and is resumed after HiGHS has returned and the callback is uninstalled.
- **Only fields HiGHS sets are read.** For each callback kind, the trampoline reads only the fields HiGHS sets for
  it, and arrays only within the lengths HiGHS reports.

The rustdoc of `culmen_highs::callback` states these arguments in full. SCIP's backend keeps the same rules with
plugins in place of a trampoline:

- **One instance, one owner.** A SCIP instance and every variable and constraint it created are owned by one value,
  released and freed when it drops, on every path; its methods that change SCIP borrow it mutably.
- **The run lives exactly as long as `SCIPsolve`.** Each plugin is included once with a pointer to a slot on the heap
  that outlives the instance; a solve installs its run, which lives on the stack of `solve`, in that slot for the call
  of `SCIPsolve` alone, and a guard clears it on every path before the run is dropped.
- **Calls never overlap.** SCIP calls every plugin on the solving thread, one at a time; a plugin called inside
  another's call finds the hooks borrowed and does not deliver, and every change a context asks of SCIP is applied
  after the handler returns, outside any borrow.
- **Panics never cross the C boundary.** A panic in a callback is caught in the plugin, which asks SCIP to stop and
  returns a result its callback allows; the panic resumes once `SCIPsolve` has returned.

The documentation of `culmen_scip`'s private `callback` module (`crates/culmen_scip/src/callback/mod.rs`) states these
arguments in full.

## See also

- [The program as data](program.md) and [Modelling](modelling.md): the checked values in practice.
- [Callbacks](callbacks.md): the seam from a user's side.
- [Backends](backends.md): the seam from a backend author's side.
- [Documentation index](README.md).
