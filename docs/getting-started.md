# Getting started

This page takes you from an empty project to a solved model: what to install, which features and backend to pick, a
first model written two ways, and how to read its solution.

## Requirements

- Rust 1.99 or later, edition 2024.
- For the default `highs` feature: **cmake** and **libclang**. The HiGHS backend builds HiGHS 1.16 from the
  sources bundled with the `highs-sys` crate, so no system HiGHS is needed, but its C++ build needs both tools.
  The first build takes a minute or two; later builds reuse it.
- For the `scip` feature: **cmake**, a **C and C++ compiler**, **libclang**, and **network access** to `github.com`
  while it builds. The SCIP backend downloads the SCIP Optimization Suite 10.0.2 sources and compiles SCIP and SoPlex
  from them, so no system SCIP is needed. The first build takes a few minutes and about 500 MB of disk, once per
  profile and feature set ([Building SCIP](scip.md#building-scip)).

Without the `highs` and `scip` features nothing native is built, and culmen is pure Rust.

## Adding culmen

culmen is on crates.io:

```toml
[dependencies]
culmen = { version = "0.4", features = ["macros"] }
# the main branch instead: culmen = { git = "https://github.com/FelixKahle/culmen", features = ["macros"] }
```

Depend on `culmen` only. It re-exports everything a model and a solve need; the other crates are its parts.

## Features

| Feature | Default | What it adds |
|---|---|---|
| `highs` | on | The HiGHS backend: `culmen::Highs`, and the whole backend crate as `culmen::highs`. |
| `scip` | off | The SCIP backend: `culmen::Scip`, and the whole backend crate as `culmen::scip` (see [SCIP](scip.md)). |
| `macros` | off | `variable!`, `constraint!`, `objective!` and `sum!` (see [Macros](macros.md)). |
| `serde` | off | `Serialize` and `Deserialize` for the plain data, and for the options and statuses of each backend enabled (see [Serialization](testing.md#serialization)). |

With `default-features = false` you get the modelling layer, the program and the solve API without a solver: enough to
build and inspect models, write LP files, or drive a backend of your own.

`culmen::prelude` holds what a model and a solve need: `Model`, `Var`, `LinExpr`, `QuadExpr`, `Linear`, `Quadratic`,
`Constraint`, `Objective`, `Tolerance`, `Bounds`, `Col`, `Row`, `Program`, `Sense`, `Solver`, `Settings`, `Start`,
`Outcome`, `Solution`, `Solved`, `SolveError`, `StopReason`, `Observer`, `Progress`, `Improvement`, `LogLine`,
`Stop`, `Callback`, `Backend`, `Events`, `IndexSet`, `VarMap`, `ConstraintMap`, the functions
`product`, `set` and `stop_fn`, the literal macros `bounds!`, `relative_gap!`, `absolute_gap!`, `weight!`, `threads!`
and `solutions!`, the [context traits](callbacks.md#contexts) a callback or handler calls (`Context`, `LogContext`,
`PollContext`, `LazyContext` and the rest) with `Unsupported`, the [search capabilities and their
handlers](callbacks.md#search-handlers) (`LazyConstraints` and `LazyHandler`, `UserCuts` and `CutHandler`,
`Heuristics` and `HeuristicHandler`, `Prices` and `Pricer`), and, with their features, `Highs`, `Scip` and the
modelling macros. Everything else
is at the crate root, `culmen::…`.

## Choosing a backend

The backend is a type: `Solver::new(Highs::new(), program)` or `Solver::new(Scip::new(), program)`. The model, the
solve and the outcome are the same for both, so switching is one word.

| Need | HiGHS (`highs`, default) | SCIP (`scip`) |
|---|---|---|
| LPs, MILPs, convex QPs | yes | yes; a quadratic objective through an epigraph, exact to SCIP's tolerance |
| A convex quadratic objective over integer columns (MIQP) | no | yes |
| Lazy constraints, user cuts | no: a handler for either does not compile | yes |
| Pricing at every node (branch-and-price) | no, a pricer does not compile; price in an outer loop with [edits](solving.md#editing-a-loaded-program) | yes |
| Duals, basis, ranging, rays, IIS, feasibility relaxation | yes | no |
| Semi-continuous and semi-integer columns, several objectives | yes | no |
| Reading and writing MPS and LP files with the solver's own readers | yes | no; culmen writes [LP files](program.md#lp-files) itself |
| Threads | a parallel MIP search | one thread per solve |
| Build | HiGHS's bundled sources, a minute or two | SCIP's sources downloaded from GitHub, a few minutes and 500 MB |

Start with HiGHS. Enable `scip` for what only SCIP does, and both may be enabled at once: a program solves on either.

## A first model

A 0/1 knapsack: take the items of most value whose weight fits a capacity of 10.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let items = [("tent", 11.0, 7.0), ("stove", 6.0, 4.0), ("rope", 3.0, 2.0), ("lamp", 4.0, 3.0)];

    // One binary variable per item, keyed by its name.
    let mut m = Model::new();
    let take = m.add_var_map(items.iter().map(|&(name, ..)| name), |name| {
        Var::binary().name(format!("take[{name}]"))
    });

    // Σ weight · take <= 10, and maximize Σ value · take.
    let weight = LinExpr::sum(items.iter().map(|&(name, _, weight)| weight * take[name]));
    m.add_constraint(("capacity", weight.leq(10.0)));
    m.maximize(LinExpr::sum(items.iter().map(|&(name, value, _)| value * take[name])));

    // Build the checked program, load it into HiGHS, and solve with HiGHS's defaults.
    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let solved = solver.solve()?;

    let Outcome::Optimal(best) = solved.outcome else {
        return Err(format!("no optimum: {:?}", solved.outcome).into());
    };
    assert_eq!(best.objective, 15.0);
    for (name, value) in take.values(&best.values) {
        if value > 0.5 {
            println!("take {name}");
        }
    }
    assert!(best.values[take["tent"]] > 0.5 && best.values[take["lamp"]] > 0.5);
    Ok(())
}
```

What happened, step by step:

1. `Model::add_var_map` added one variable per key and returned a [`VarMap`](modelling.md#keyed-families): the
   column handle of each item, read as `take["tent"]`.
2. `weight * take[name]` is a [`Term`](modelling.md#expressions); `LinExpr::sum` adds terms into an expression;
   `.leq(10.0)` turns it into a [`Constraint`](modelling.md#comparisons).
3. `m.build()` checked every number and produced a [`Program`](program.md#program). A defect there is a
   [`ModelError`](modelling.md#building-and-modelerror) naming the variable or constraint at fault.
4. `Solver::new` loaded the program into the backend; `solve` ran it under the backend's defaults (a
   [`Run`](solving.md#run) sets limits, a stop and an observer) and returned
   [`Solved`](solving.md#solved-and-outcome): the generic `Outcome`, HiGHS's own status, and the time.
5. The solution is read with the handles the model returned: `best.values[take["tent"]]`, or the whole family at
   once with `take.values(&best.values)`.

## The same model with macros

With the `macros` feature, the same model reads closer to the mathematics. It builds the same program.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let names = ["tent", "stove", "rope", "lamp"];
    let value = [11.0, 6.0, 3.0, 4.0];
    let weight = [7.0, 4.0, 2.0, 3.0];

    let mut m = Model::new();
    variable!(m, take[i in 0..names.len()], Bin);
    constraint!(m, capacity, sum!(weight[i] * take[i] for i in 0..names.len()) <= 10);
    objective!(m, Max, sum!(value[i] * take[i] for i in 0..names.len()));

    assert_eq!(m.var_name(take[0]), Some("take[0]"));
    assert_eq!(m.constraint_name(capacity), Some("capacity"));

    let solved = Solver::new(Highs::new(), m.build()?)?.solve()?;
    assert_eq!(solved.outcome.objective(), Some(15.0));
    Ok(())
}
```

Each macro expands to the method calls of the first model; [Macros](macros.md#expansions) shows each form beside
the calls it writes.

## Errors

Every error type implements `std::error::Error`, so `?` into `Box<dyn std::error::Error>` works across all of them,
as above. The ones you meet first:

| Error | Raised by | Page |
|---|---|---|
| `ModelError` | `Model::build` | [Modelling](modelling.md#building-and-modelerror) |
| `ProgramError` | `ProgramPlan::build` | [The program as data](program.md#checks-and-programerror) |
| `InvalidBounds` | `Bounds::new` and its conversions | [The program as data](program.md#bounds) |
| `LoadError` | `Solver::new`: a model class the backend does not solve, a feature it does not take, or the backend's own error | [Solving](solving.md#model-classes-and-refusal) |
| `HighsError` | `Solver::solve` with HiGHS, and inside `LoadError` from `Solver::new` | [HiGHS](highs.md#errors) |
| `ScipError` | `Solver::solve` with SCIP, and inside `LoadError` from `Solver::new` | [SCIP](scip.md#errors) |
| `ActionError` | actions inside a callback | [Callbacks](callbacks.md#contexts) |

Each error's message starts with what it concerns (`Model:`, `Program:`, `Bounds:`, `Solver:`, `HiGHS:`, `SCIP:`,
`callback action:`, `model file:`), and a model error names the variable or constraint at fault by the name you gave
it.

## Examples

The `culmen` crate ships eight examples:

- `cargo run -p culmen --example knapsack`: the knapsack above, with five items.
- `cargo run -p culmen --release --example warehouse -- 20`: a warehouse slotting problem of 60 products and 30
  locations, with live progress, stopped after 20 seconds or on Ctrl-C. Add `--log` to print HiGHS's log too.
- `cargo run -p culmen --example cutting_planes`: cover cuts added to a knapsack's LP relaxation, each solved again
  from the last basis ([Editing a loaded program](solving.md#editing-a-loaded-program)).
- `cargo run -p culmen --example column_generation`: cutting stock by column generation, a pattern appended to the
  master each round.
- `cargo run -p culmen --example branch_and_price --features scip`: the same cutting stock priced at every node of
  SCIP's search ([Pricing](scip.md#pricing)).
- `cargo run -p culmen --example lazy_constraints --features scip`: a round trip through eight towns, its subtours cut
  off by lazy constraints on SCIP ([Lazy constraints](scip.md#lazy-constraints)).

`knapsack` and `warehouse` each have a `_macros` twin, run with `--features macros`, that writes the same model with
the macros.

## See also

- [Modelling](modelling.md) for everything a model can hold.
- [Solving](solving.md) for limits, gaps and outcomes.
- [Callbacks](callbacks.md) for watching and stopping a solve.
- [Documentation index](README.md).
