# Design

This page states the principles the API follows and why. Each one is a guarantee you can rely on when you use
culmen, and a rule a contribution keeps.

## The LP is checked data

An [`Lp`](lp.md#lp) is a value, built once and never changed. Every number in it was checked when it was built:
costs, coefficients and the offset are finite, bounds are valid intervals, and each row holds each column at most
once. Nothing downstream checks again, and nothing downstream can meet an LP that was not checked: `Lp` has no
public fields, its only constructors check, and deserialization goes through the same check.

The check happens in one place, `build`, and its error names what is at fault. A model's error names the variable
or constraint by the name you gave it (`` Model: constraint `big` has the non-finite coefficient inf on variable `x` ``),
because the model runs the LP's checks in its own terms; every model `build` accepts gives an LP the LP's own checks
accept.

Holding no solver state, an LP can be cloned, compared, serialized, written as an [LP file](lp.md#lp-files), and
loaded into any backend.

## Typed handles

`Col` and `Row` are distinct `Copy` types, not `usize`. A solution is a `TiVec<Col, f64>`, indexed by `Col` only, so
reading a solution with a row, or a row table with a column, does not compile. The handles a model returns are the
columns and rows of the LP it builds, so there is no mapping to keep between "model variables" and "solver
columns": `solution.values[x]` is the value of the variable `x`.

The same holds beyond the LP. HiGHS's cut pool ranges over the presolved problem, whose columns are not the LP's;
they are `ReducedCol`, a third type, so a cut cannot be read as a row of your LP by mistake.

Keyed families give handles domain keys (`x[(product, location)]`), and a family is plain data you can iterate,
look up and read a solution through.

## Literals are checked when compiled

A value with an invariant has a type that holds it: `Bounds` is a valid interval, `RelativeGap` and `AbsoluteGap`
are finite and not negative, a thread count is a `NonZeroUsize`. A literal of such a type is written with a macro,
`bounds!(0.0, 40.0)`, `relative_gap!(0.01)`, `threads!(4)`, which checks it when the code is compiled, so a typo in
a literal is a compile error, not a run-time `unwrap` panic:

```compile_fail
use culmen::prelude::*;

let gap = relative_gap!(-0.01); // RelativeGap: below zero
```

Numbers read at run time go through the fallible constructors (`Bounds::new`, `RelativeGap::try_from`), whose errors
say what was refused. Code never writes `.unwrap()` to make a literal.

## Plain data and sealed values

A type without an invariant is plain data with `pub` fields and no getters: `LpPlan`, `LpColumn`, `Constraint`,
`LinExpr`, `Settings`, `Solution`, `Progress`, `HighsOptions`, `HighsStatus`. You build them with struct literals and
`..Default::default()`, read and change their fields directly, and match on them.

A type with an invariant keeps its fields private behind a checked constructor: `Bounds`, `Lp`, the gaps. Their
fields cannot be written from outside, not even all at once:

```compile_fail
use culmen::Bounds;

let empty = Bounds { lower: 2.0, upper: 1.0 }; // the fields are private
```

Constructors take what converts into their input without loss, so a caller passes `0` where an `f64` is wanted, a
`Bounds` where a `Var` is wanted, and an owned or borrowed value alike. A conversion that can fail is never implicit;
it is a `TryFrom` or a `new` returning `Result`, so the refusal is visible where the value is made.

## The solver-neutral seam

`culmen_core` knows no solver. A solver plugs in by implementing [`Backend`](backends.md#the-seam), an open trait:

- **Events are typed and declared.** A backend names one context type per event kind and lists what it delivers
  in `Backend::EVENTS`. An event it cannot deliver takes `Unsupported`, an enum with no values, so a callback for it
  compiles everywhere and never runs on that backend.
- **Model classes are declared.** A backend lists the classes of program it solves in `Backend::CLASSES`, and
  `Solver::new` refuses any other class with a typed `LoadError::Unsupported` before the backend sees the LP, the
  same way for every backend.
- **Actions live where the solver honours them.** Only a poll context can interrupt, only a lazy context can add a
  lazy constraint, only a heuristic context can supply a solution. Calling an action at a point where the solver
  would ignore it does not compile.
- **Extras stay with their backend.** Each context type is the backend's own; what one solver reports beyond the
  neutral traits is an inherent method or a native event. Generic code sees the traits; code for one backend sees
  everything.
- **The convenient path is built on the seam, not beside it.** The request's stop and observer run as `Watch`, an
  ordinary generic callback, so they work on every backend unchanged.
- **Settings are neutral, options are the backend's.** `Settings` holds what every solver understands; a backend's
  own options hold the rest; settings apply after options, per solve.
- **Outcomes are neutral, statuses are kept.** Each solve maps the solver's status onto `Outcome`, which holds
  exactly what the ending proves, and keeps the native status beside it for diagnostics.

[Backends](backends.md) shows how HiGHS, Gurobi and SCIP map onto every part of this seam.

## A solve never aborts

A solve is something an application runs inside a service, so a defect in user code during a solve must not take
the process down. A panic in an observer, a stop or a callback stops the solve as soon as the solver allows, and
then continues unwinding out of `solve` on the caller's thread, where it can be caught. The backend stays usable.
No profile sets `panic = "abort"`.

The stop is cooperative: it is polled at the solver's poll points, and the solve ends with what it found, as an
ordinary `Outcome::Stopped` with the best solution and bound.

## Where `unsafe` lives

`culmen_core` forbids `unsafe` (`#![forbid(unsafe_code)]`); the facade and the macros contain none. All `unsafe` is in
`culmen_highs`, where HiGHS's C API makes it necessary, and every `unsafe` block there carries a `// SAFETY:`
comment saying why it is sound. The parts that matter most:

- **One instance, one owner.** A HiGHS instance is created and destroyed with the `Highs` value that owns it, and
  every call that changes it borrows it mutably, so no two calls ever overlap on one instance.
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

The rustdoc of `culmen_highs::callback` states these arguments in full.

## See also

- [The LP as data](lp.md) and [Modelling](modelling.md): the checked values in practice.
- [Callbacks](callbacks.md): the seam from a user's side.
- [Backends](backends.md): the seam from a backend author's side.
- [Documentation index](README.md).
