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
`NewCol`, `Constraint`, `LinExpr`, `Settings`, `Solution`, `Progress`, `HighsOptions`, `HighsStatus`. You build them
with struct literals and `..Default::default()`, read and change their fields directly, and match on them. A column is
also built by chaining from its integrality (`ProgramColumn::integer().cost(1)`, `NewCol::continuous().entry(row,
2)`), as a `Var` is, since a literal would spell out the defaults its integrality already names.

A type with an invariant keeps its fields private behind a checked constructor: `Bounds`, `Program`, the gaps. Their
fields cannot be written from outside, not even all at once:

```compile_fail,E0451
use culmen::Bounds;

let empty = Bounds { lower: 2.0, upper: 1.0 }; // the fields are private
```

Constructors take what converts into their input without loss, so a caller passes `0` where an `f64` is wanted, a
`Bounds` where a `Var` is wanted, and an owned or borrowed value alike. A conversion that can fail is never implicit;
it is a `TryFrom` or a `new` returning `Result`, so the refusal is visible where the value is made.

## One address per item

Each public item has one path, chosen by who names it. The crate root holds what code that models and solves names:
the model and the program, `Solver` and `Run`, settings and outcomes, the callback and handler traits, the context
traits a handler calls, the capability traits a generic bound names (`LazyConstraints`, `DualAnalysis`, …), the
analyses' data and the errors. `Backend` is there too, since generic consumer code bounds on it. `culmen::script` holds
the scripted test backend and what only it needs; `culmen::backend` holds what only code implementing or driving a
backend names (`Hooks`, `Relay`, `Edit`, `Priced`, `Watch`, `Found`, `Unsupported`, the `check_*` helpers). Each
solver's own types are at the root of its crate, `culmen::highs` or `culmen::scip`, and a submodule holds only free
functions whose names read right with it (`culmen::highs::files::read`), and the types those functions take and
return stay at the crate root (`culmen::highs::ModelFormat`). Nothing is re-exported a second time for
convenience; the prelude, a selection of root items and the backend types, is the one exception.

## Public types are exhaustive

Every public enum and every public struct is exhaustive, in every crate: none is marked non-exhaustive, and none
will be.

- **A consumer's `match` is checked.** A `match` on an error, a status, an `Edit` or a `StopReason` names every
  variant and needs no wildcard arm, so the compiler tells the code that matches when a variant appears. A wildcard
  arm written only to satisfy a non-exhaustive enum would hide that variant, and a backend that loaded the program
  whole for an `Edit` it did not know would hide it as a silent loss of its warm start.
- **Growth is a breaking release.** A new variant, a new field or a new error is a breaking change, made only in a
  breaking release, which for culmen at 0.x is a minor version bump: a `0.5` dependency never sees a variant `0.5`
  did not have. Plain data with `pub` fields keeps its struct literal and `..Default::default()` outside its crate,
  and every error's `Display` stays one exhaustive table.
- **What must be closed at run time is a declared set.** A backend lists the classes and features it takes
  (`Backend::CLASSES`, `FEATURES`), and the `Solver` refuses any other with a typed error, so a class a later version
  adds is refused by a backend that does not declare it rather than reaching it.

### A solver's own codes

An enum that mirrors a solver's codes (`ModelStatus`, `CallStatus`, `SolutionStatus`, `HighsLogType`,
`HighsQueryOrigin`, `Status`, `RetCode`) lists every value of the solver version its `-sys` crate builds, with the
header that defines them named in its docs. Converting a C integer into it is still a partial function, so the
conversion is total and honest:

- **A value the pinned headers rule out is a typed error.** The conversion returns the backend's error carrying the
  raw value, `HighsError::Unlisted { code, raw }` or `ScipError::Unlisted { code, raw }`, where `code` names the
  enumeration (`HighsCode`, `ScipCode`). It never panics, and it never maps the value onto a listed variant: an
  unknown model status is not a failure, an unknown solution status is not "none", an unknown call status is not an
  error code. A value HiGHS passes to a callback is held like a panic, stops the run at the next poll, and fails the
  solve with that error once HiGHS returns; inside an action, which can only return an `ActionError`, it is the
  `Refused` reason, which names the value.
- **A value that is open-ended in normal operation is a variant.** Where the solver itself documents values beyond
  a fixed list, the enum carries them in an explicit variant holding the raw value, whose docs say when it occurs.
  No enum of HiGHS 1.16 or SCIP 10 that culmen reads needs one today.
- **An enum that only goes into the solver** (`LpSolver`, `PresolveMode`, `Emphasis`, `ParamValue`, …) has no
  conversion from the solver, and is exhaustive like any other.

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
  `Solver::new` refuses any other class with a typed `LoadError::UnsupportedClass` before the backend sees the program,
  the same way for every backend. A class is a property of the program, which edits change, so it stays a run-time
  check.
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

A solve is something an application runs inside a service, so a defect in user code during a solve must not take the
process down. A panic in an observer, a stop, a callback or a handler stops the solve as soon as the solver allows, and
then continues unwinding out of `solve` on the caller's thread, where it can be caught. The backend stays usable. No
profile sets `panic = "abort"`.

The stop is cooperative: it is polled at the solver's poll points, and the solve ends with what it found, as an
ordinary `Outcome::Stopped` with the best solution and bound.

## Where `unsafe` lives

`culmen_core` forbids `unsafe` (`#![forbid(unsafe_code)]`); the facade and the macros contain none. All `unsafe` is in
the two backends, `culmen_highs` and `culmen_scip`, where the solvers' C APIs make it necessary, and every `unsafe`
block there carries a `// SAFETY:` comment saying why it is sound. The parts that matter most, for HiGHS:

- **One instance, one owner.** A HiGHS instance is created with the `Highs` value that owns it and destroyed when
  that value drops, and every call that changes it borrows it mutably, so no two calls ever overlap on one instance.
  Destroying an instance also shuts down the HiGHS task scheduler of the destroying thread, so an instance dropped
  while its thread is inside a run or a callback waits until a run ends on a thread that is then outside every run
  and callback, and is destroyed there, which may be another thread than the one that dropped it.
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

The documentation of `culmen_highs`'s private `callback` module (`crates/culmen_highs/src/callback.rs`) states these
arguments in full. SCIP's backend keeps the same rules with plugins in place of a trampoline:

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
