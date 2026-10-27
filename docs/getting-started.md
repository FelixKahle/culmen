# Getting started

This page takes you from an empty project to a solved model: what to install, which features to pick, a first
model written two ways, and how to read its solution.

## Requirements

- Rust 1.99 or later, edition 2024.
- For the default `highs` feature: **cmake** and **libclang**. The HiGHS backend builds HiGHS 1.15 from the
  sources bundled with the `highs-sys` crate, so no system HiGHS is needed, but its C++ build needs both tools.
  The first build takes a minute or two; later builds reuse it.

Without the `highs` feature nothing native is built, and culmen is pure Rust.

## Adding culmen

culmen is not yet on crates.io. Depend on it by path or by git:

```toml
[dependencies]
culmen = { path = "../culmen/crates/culmen", features = ["macros"] }
# or: culmen = { git = "<repository url>", features = ["macros"] }
```

Depend on `culmen` only. It re-exports everything a model and a solve need; the other crates are its parts.

## Features

| Feature | Default | What it adds |
|---|---|---|
| `highs` | on | The HiGHS backend: `culmen::Highs`, and the whole backend crate as `culmen::highs`. |
| `macros` | off | `variable!`, `constraint!`, `objective!` and `sum!` (see [Macros](macros.md)). |
| `serde` | off | `Serialize` and `Deserialize` for the plain data, and for the HiGHS options and statuses with `highs` (see [Serialization](testing.md#serialization)). |

With `default-features = false` you get the modelling layer, the program and the solve API without a solver: enough to
build and inspect models, write LP files, or drive a backend of your own.

`culmen::prelude` holds what a model and a solve need: `Model`, `Var`, `LinExpr`, `Linear`, `Constraint`, `Bounds`,
`Col`, `Row`, `Program`, `Sense`, `Solver`, `Request`, `Settings`, `Outcome`, `Solution`, `Solved`, `StopReason`,
`Observer`, `Progress`, `Improvement`, `LogLine`, `Stop`, `Callback`, `Backend`, `Events`, `IndexSet`, `VarMap`,
`ConstraintMap`, the functions `product`, `set` and `stop_fn`, the literal macros `bounds!`, `relative_gap!`,
`absolute_gap!` and `threads!`, and, with their features, `Highs` and the modelling macros. Everything else is at the
crate root, `culmen::…`.

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
    m.add_named_constraint("capacity", weight.leq(10.0));
    m.maximize(LinExpr::sum(items.iter().map(|&(name, value, _)| value * take[name])));

    // Build the checked program, load it into HiGHS, and solve with HiGHS's defaults.
    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let solved = solver.solve(Request::new())?;

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
4. `Solver::new` loaded the program into the backend; `solve` ran it under a [`Request`](solving.md#request) and
   returned [`Solved`](solving.md#solved-and-outcome): the generic `Outcome`, HiGHS's own status, and the time.
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

    let solved = Solver::new(Highs::new(), m.build()?)?.solve(Request::new())?;
    assert_eq!(solved.outcome.solution().map(|s| s.objective), Some(15.0));
    Ok(())
}
```

Each macro expands to the methods of the first version; [Macros](macros.md#expansions) shows each form beside the
calls it writes.

## Errors

Every error type implements `std::error::Error`, so `?` into `Box<dyn std::error::Error>` works across all of them,
as above. The ones you meet first:

| Error | Raised by | Page |
|---|---|---|
| `ModelError` | `Model::build` | [Modelling](modelling.md#building-and-modelerror) |
| `ProgramError` | `ProgramPlan::build` | [The program as data](program.md#checks-and-programerror) |
| `InvalidBounds` | `Bounds::new` and its conversions | [The program as data](program.md#bounds) |
| `LoadError` | `Solver::new`: a model class the backend does not solve, or the backend's own error | [Solving](solving.md#model-classes-and-refusal) |
| `HighsError` | `Solver::solve` with HiGHS, and inside `LoadError` from `Solver::new` | [HiGHS](highs.md#errors) |
| `ActionError` | actions inside a callback | [Callbacks](callbacks.md#contexts) |

Each error's message starts with the name of the type it concerns (`Model:`, `Program:`, `Bounds:`, `Solver:`,
`HiGHS:`), and a model error names the variable or constraint at fault by the name you gave it.

## Examples

The `culmen` crate ships six examples:

- `cargo run -p culmen --example knapsack`: the knapsack above, with five items.
- `cargo run -p culmen --release --example warehouse -- 20`: a warehouse slotting problem of 60 products and 30
  locations, with live progress, stopped after 20 seconds or on Ctrl-C. Add `--log` to print HiGHS's log too.
- `cargo run -p culmen --example cutting_planes`: cover cuts added to a knapsack's LP relaxation, each solved again
  from the last basis ([Editing a loaded program](solving.md#editing-a-loaded-program)).
- `cargo run -p culmen --example column_generation`: cutting stock by column generation, a pattern appended to the
  master each round.

`knapsack` and `warehouse` each have a `_macros` twin, run with `--features macros`, that writes the same model with
the macros.

## See also

- [Modelling](modelling.md) for everything a model can hold.
- [Solving](solving.md) for limits, gaps and outcomes.
- [Callbacks](callbacks.md) for watching and stopping a solve.
- [Documentation index](README.md).
