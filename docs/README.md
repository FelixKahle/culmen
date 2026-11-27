# culmen documentation

culmen is a Rust library for linear, mixed-integer and convex quadratic programming. A model is written the way the
mathematics reads, built into a checked program, and solved on a backend chosen by type, with a live view of the search
and a cooperative stop. HiGHS 1.16 is the default backend; SCIP 10, behind the `scip` feature, is the second.

These pages are the user guide. The API reference is the rustdoc of each crate (`cargo doc --open -p culmen`).
Every Rust snippet on these pages is compiled and run as a doctest of the `culmen` crate, so each one shows the
library as it is. The two exceptions are marked `ignore`: the outline of the `Backend` trait in
[Backends](backends.md), which leaves out its imports and default bodies, and the JSON example in
[Serialization](testing.md#serialization), which needs `serde_json`.

## The crates

| Crate | What it holds |
|---|---|
| `culmen` | The crate to depend on: re-exports `culmen_core` at its root, the HiGHS backend as `culmen::Highs` and `culmen::highs`, the SCIP backend as `culmen::Scip` and `culmen::scip`, the macros, and a `prelude`. |
| `culmen_core` | The program as data, the modelling layer, the backend seam and the solve API. It has no `unsafe` and no solver. |
| `culmen_highs` | The HiGHS backend. With `culmen_scip`, the only crates with `unsafe`. |
| `culmen_scip` | The SCIP backend, SCIP built from source. |
| `culmen_macros` | `variable!`, `constraint!`, `objective!` and `sum!`, which expand to `culmen_core`'s methods. |
| `culmen_testing` | Dev-only test infrastructure: an exact reference solver, property-test strategies, hard instances. |

Dependencies point down: `culmen` depends on the four below it, `culmen_highs` and `culmen_scip` on `culmen_core`,
and `culmen_core` on no other culmen crate. `culmen_macros` depends on no culmen crate at all; its expansions name
the paths of `culmen` or `culmen_core`, whichever the calling crate depends on.

## Reading path

1. [Getting started](getting-started.md): install, pick features, solve a first model.
2. [Modelling](modelling.md): variables, expressions, constraints, objectives and keyed families with plain methods.
3. [Macros](macros.md): the same models written with `variable!`, `constraint!`, `objective!` and `sum!`.
4. [Solving](solving.md): the solver, the run that configures a solve and its settings, outcomes and reading solutions.
5. [Callbacks](callbacks.md): watching a solve, stopping it, reaching every event the solver reports, and the
   handlers that take part in its search.
6. [Analysis](analysis.md): duals, the basis and warm starts, ranging, rays, IIS and feasibility relaxation.
7. [HiGHS](highs.md): the backend's options, threads, statuses and callback contexts, and what HiGHS cannot do.
8. [SCIP](scip.md): building SCIP, its options and statuses, MIQPs, lazy constraints, cuts, heuristic solutions and
   pricing at every node, and what SCIP's backend cannot do.

Then, as needed:

- [The program as data](program.md) for building a program directly, inspecting what a model built, or writing an LP
  file.
- [Testing](testing.md) for testing code that drives a solve without a solver, and for serialization.
- [Design](design.md) for the principles behind the API.
- [Backends](backends.md) for backend authors: the `Backend` seam, and how HiGHS, Gurobi and SCIP map onto it.

## Contents

| Page | Sections |
|---|---|
| [Getting started](getting-started.md) | [Requirements](getting-started.md#requirements), [Adding culmen](getting-started.md#adding-culmen), [Features](getting-started.md#features), [Choosing a backend](getting-started.md#choosing-a-backend), [A first model](getting-started.md#a-first-model), [The same model with macros](getting-started.md#the-same-model-with-macros), [Errors](getting-started.md#errors), [Examples](getting-started.md#examples) |
| [Modelling](modelling.md) | [Models and handles](modelling.md#models-and-handles), [Variables](modelling.md#variables), [Expressions](modelling.md#expressions), [Quadratic expressions](modelling.md#quadratic-expressions), [Comparisons](modelling.md#comparisons), [Constraints](modelling.md#constraints), [The objective](modelling.md#the-objective), [Several objectives](modelling.md#several-objectives), [Keyed families](modelling.md#keyed-families), [Index sets](modelling.md#index-sets), [Names](modelling.md#names), [Building and `ModelError`](modelling.md#building-and-modelerror) |
| [Macros](macros.md) | [Enabling the macros](macros.md#enabling-the-macros), [`variable!`](macros.md#variable), [`constraint!`](macros.md#constraint), [`objective!`](macros.md#objective), [`sum!`](macros.md#sum), [Indexing inside a macro](macros.md#indexing-inside-a-macro), [Domains](macros.md#domains), [Expansions](macros.md#expansions), [Compile errors](macros.md#compile-errors) |
| [The program as data](program.md) | [`ProgramPlan`](program.md#programplan), [`Program`](program.md#program), [`Bounds`](program.md#bounds), [Integrality and sense](program.md#integrality-and-sense), [`Col` and `Row`](program.md#col-and-row), [Row views and iterators](program.md#row-views-and-iterators), [The quadratic part](program.md#the-quadratic-part), [Model classes](program.md#model-classes), [Features](program.md#features), [Several objectives](program.md#several-objectives), [Checking a point](program.md#checking-a-point), [LP files](program.md#lp-files), [Checks and `ProgramError`](program.md#checks-and-programerror) |
| [Solving](solving.md) | [`Solver`](solving.md#solver), [Model classes and refusal](solving.md#model-classes-and-refusal), [`Run`](solving.md#run), [Starts](solving.md#starts), [`Settings`](solving.md#settings), [Checked literals](solving.md#checked-literals), [`Solved` and `Outcome`](solving.md#solved-and-outcome), [`StopReason`](solving.md#stopreason), [Reading a solution](solving.md#reading-a-solution), [Editing a loaded program](solving.md#editing-a-loaded-program), [A solve that prices](solving.md#a-solve-that-prices), [Errors](solving.md#errors) |
| [Callbacks](callbacks.md) | [Two levels](callbacks.md#two-levels), [`Stop`](callbacks.md#stop), [`Observer`](callbacks.md#observer), [`Callback`](callbacks.md#callback), [`Events`](callbacks.md#events), [Search handlers](callbacks.md#search-handlers), [Contexts](callbacks.md#contexts), [Pricing](callbacks.md#pricing), [Pairing callbacks](callbacks.md#pairing-callbacks), [`Unsupported`](callbacks.md#unsupported), [Panics and threads](callbacks.md#panics-and-threads) |
| [Analysis](analysis.md) | [Analyses and refusal](analysis.md#analyses-and-refusal), [Duals and reduced costs](analysis.md#duals-and-reduced-costs), [The basis and warm starts](analysis.md#the-basis-and-warm-starts), [Ranging](analysis.md#ranging), [Rays](analysis.md#rays), [Irreducible infeasible subsystems](analysis.md#irreducible-infeasible-subsystems), [Feasibility relaxation](analysis.md#feasibility-relaxation), [HiGHS](analysis.md#highs) |
| [HiGHS](highs.md) | [The backend](highs.md#the-backend), [Options](highs.md#options), [How options and settings layer](highs.md#how-options-and-settings-layer), [Threads](highs.md#threads), [Statuses and outcomes](highs.md#statuses-and-outcomes), [Quadratic programs](highs.md#quadratic-programs), [Semi columns, several objectives and starts](highs.md#semi-columns-several-objectives-and-starts), [Model files](highs.md#model-files), [Editing a loaded program](highs.md#editing-a-loaded-program), [Errors](highs.md#errors), [Callback contexts](highs.md#callback-contexts), [Heuristic solutions](highs.md#heuristic-solutions), [The cut pool](highs.md#the-cut-pool), [Limitations](highs.md#limitations) |
| [SCIP](scip.md) | [The backend](scip.md#the-backend), [Building SCIP](scip.md#building-scip), [Options](scip.md#options), [How options and settings layer](scip.md#how-options-and-settings-layer), [Threads](scip.md#threads), [Statuses and outcomes](scip.md#statuses-and-outcomes), [Quadratic programs](scip.md#quadratic-programs), [Starts](scip.md#starts), [Callback events](scip.md#callback-events), [Lazy constraints](scip.md#lazy-constraints), [Cuts](scip.md#cuts), [Heuristic solutions](scip.md#heuristic-solutions), [Pricing](scip.md#pricing), [Errors](scip.md#errors), [Limitations](scip.md#limitations) |
| [Testing](testing.md) | [The `Script` backend](testing.md#the-script-backend), [Testing callbacks](testing.md#testing-callbacks), [`culmen_testing`](testing.md#culmen_testing), [Serialization](testing.md#serialization) |
| [Design](design.md) | [The program is checked data](design.md#the-program-is-checked-data), [Typed handles](design.md#typed-handles), [Literals are checked when compiled](design.md#literals-are-checked-when-compiled), [Plain data and sealed values](design.md#plain-data-and-sealed-values), [Which types may grow: `#[non_exhaustive]`](design.md#which-types-may-grow-non_exhaustive), [The solver-neutral seam](design.md#the-solver-neutral-seam), [A solve never aborts](design.md#a-solve-never-aborts), [Where `unsafe` lives](design.md#where-unsafe-lives) |
| [Backends](backends.md) | [The seam](backends.md#the-seam), [Model classes](backends.md#model-classes), [Features](backends.md#features), [Events](backends.md#events), [Settings](backends.md#settings), [Outcomes and statuses](backends.md#outcomes-and-statuses), [What a Gurobi backend would be](backends.md#what-a-gurobi-backend-would-be) |
