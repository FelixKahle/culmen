# Changelog

Every release of culmen and what a user of the one before meets in it. culmen is at 0.x, so a minor release may
break the API; every public type is exhaustive, so a new variant or field arrives only in such a release.

## 0.5.0

A breaking release. The solve API, the callbacks and the capability model are new: what a backend can do is a trait
it implements, so asking a backend for what it cannot do no longer compiles. The sections below follow the order a
0.4.0 user meets the changes in; each removed or renamed item names what replaces it.

### Solving

- **`Request` is gone. A solve is configured on `Solver::run()`, which returns a `Run`.**
  - `solver.solve(Request::new())` is `solver.solve()` (the backend's defaults) or `solver.run()…solve()`.
  - `Request::settings_from(s)` is `Run::settings(s)`. `time_limit`, `stop`, `observer`, `progress_interval` and
    `start` keep their names on `Run`.
  - `Solver::solve_with(request, callback)` is `solver.run().callback(&mut callback).solve()`.
  - `Request::into_watch`, `settings()` and `start_solution()` are gone. Code that drives a backend directly builds
    `culmen::backend::Watch::new(stop, observer, interval)` and `Hooks`.
- **Starts are a capability.** `Run::start` exists only for a backend that implements `Starts`.
- **Backend options are a capability.** `Highs::options()`/`options_mut()` and `Scip::options()`/`options_mut()` are
  methods of the `Configurable` trait (import `culmen::Configurable` to call them on a bare backend);
  `Solver::options()` and `options_mut()` reach them on a solver. `Solver::backend_mut()` is gone.
- `Backend::solve(&mut self, settings, hooks: Hooks<'_, Self>)` replaces the old signature (for backend authors).
  `Hooks` carries the callback, the handlers and the pricing sink; `Relay` and `Hooks::relay` serve a backend that
  wraps another.
- `Solved` carries the neutral work counters `nodes` and `iterations`.

### Callbacks and handlers

- **Search handlers are separate traits, registered on `Run`.** The `lazy`, `cut`, `heuristic` and `price` methods
  of `Callback` and the backend's `PRICES` flag are gone. Implement `LazyHandler<B>`, `CutHandler<B>`,
  `HeuristicHandler<B>` or `Pricer<B>` and register it with `Run::lazy`, `cut`, `lazy_and_cut`, `heuristic` or
  `pricer`. Each exists only for a backend implementing `LazyConstraints`, `UserCuts`, `Heuristics` or `Prices`, so a
  handler a backend would ignore no longer compiles.
- **The `culmen::context` module is gone.** The context traits are at the crate root and in the prelude: `Context`
  (new, the supertrait with `progress()`), `LogContext`, `ProgressContext`, `PollContext`, `SolutionContext`,
  `LazyContext`, `CutContext`, `HeuristicContext`, `PriceContext` and `PriceMode`.
- `HeuristicContext::supply_partial` is gone from the trait: completing a partial solution is an inherent method of
  the contexts that honour it, `HighsHeuristic::supply_partial` and `ScriptHeuristic::supply_partial`.
- **`Events` is a set of the new `Event` enum.** `Event::Log | Event::Poll` replaces the flag constants
  (`Events::LOG`, …); `Events::of(&[..])` is the `const` form, and `union`, `intersection`, `difference`,
  `complement`, `contains`, `is_empty`, `len` and `iter` stay beside the operators `|`, `&`, `-` and `!`.
  `Step::event()` returns `Option<Event>`.
- **Moved to `culmen::backend`** (for backend authors): `Unsupported` (no longer in the prelude), `Watch`, `Edit`,
  `AddedRow`, `Priced`, `check_solution`, `check_entries`, `check_constraint`, `Hooks` and `Relay`.
- **Moved to `culmen::script`:** `Script`, `Step`, `Found`, `Record`, `PriceRound`, `ScriptError`, `ScriptStatus`
  and the `Script*` contexts, which were at the root.

### Analyses

- **Analyses are capability traits.** `Analyses`, `AnalysesIter` and `Backend::ANALYSES` are gone. `Solver::duals`,
  `basis`, `set_basis`, `ranging`, `primal_ray`, `dual_ray`, `iis` and `feasibility_relaxation` exist only for a
  backend implementing `DualAnalysis`, `BasisAnalysis`, `RangingAnalysis`, `RayAnalysis`, `IisAnalysis` or
  `RelaxationAnalysis`. `AnalysisError::map_backend` is gone.
- `Weights::uniform(program, cols, rows)` is `Weights::hard`, `soft_rows`, `soft_bounds` or `soft`.

### Options and settings

- **`HighsOptions` lost every field `Settings` covers.** `presolve`, `time_limit`, `random_seed`, `threads`,
  `log_to_console`, `mip_max_nodes`, `mip_max_improving_sols`, `mip_rel_gap`, `mip_abs_gap`,
  `primal_feasibility_tolerance`, `mip_feasibility_tolerance` and `dual_feasibility_tolerance` are set per solve
  through `Settings`, or by HiGHS's name with `HighsOptions::option`. `solve_relaxation` is gone: it is refused with
  `HighsError::Relaxation`, and relaxing a MIP is `Solver::set_col_integrality`. `ranging` takes a `bool`.
- `OnOff` is `Switch`, one on/off type for every option; `RawOption` is `HighsOption`.
- `Highs::QP_OBJECTIVE_ERROR` is the option `HighsOptions::qp_objective_error`, with the same default.
- **New settings:** `Settings::feasibility_tolerance` (HiGHS `primal_feasibility_tolerance` and
  `mip_feasibility_tolerance`, SCIP `numerics/feastol`) and `Settings::optimality_tolerance` (HiGHS
  `dual_feasibility_tolerance`, SCIP `numerics/dualfeastol`), each a `FeasibilityTolerance` from `1e-10` to `1e-3`
  written with `feasibility_tolerance!(1e-6)`. Settings neither backend honours with one meaning are left out: an
  integrality tolerance of its own (HiGHS's `mip_feasibility_tolerance` also bounds a MIP's rows, and SCIP has no
  parameter beside `numerics/feastol`), an iteration limit (SCIP's `lp/iterlim` limits each LP solve, HiGHS's
  `simplex_iteration_limit` the whole), and an objective cutoff (HiGHS's `objective_bound` stops an LP's dual
  simplex and is a MIP's limit in its internal minimizing sense, SCIP's `SCIPsetObjlimit` discards solutions in the
  program's sense).
- `ParamValue` takes a `char` only through `TryFrom<char>`, refusing one that is not ASCII with the new `NotAscii`;
  `ParamValue::Char(c)` writes one directly.

### Errors

- `LoadError::Unsupported` is `UnsupportedClass`, beside the new `UnsupportedFeature`.
- `StartError::UnknownColumn` is `UnknownCol`, `RepeatedColumn` is `RepeatedCol`, `NotFinite` is `NonFinite`.
- `HighsError` gains `Unlisted { code: HighsCode, raw, call }`, for a value HiGHS 1.16 does not define and the call
  that gave it, and `Relaxation`. `ScipError` gains `Unlisted { code: ScipCode, raw, call }`; `RetCode::Other(i32)` is
  gone. A status or code outside the pinned headers is an error, never mapped onto a listed variant.
- `ModelError` gains `MixedObjectives` and `RankAlone`; `EditError` gains `ColEnd` and `RowEnd`.
- Every public enum and struct is exhaustive: remove wildcard arms. A future variant arrives only in a minor (0.x)
  release.

### Modelling and the program

- `Model::add_named_constraint(name, c)` is `Model::add_constraint((name, c))`. `add_constraint` takes anything
  `Into<NewRow>`: a `Constraint`, a `(name, Constraint)` pair or a `ProgramRow`; `add_constraint_map`'s closure
  returns the same.
- `Model::add_objective(Objective)` with `Objective::tolerance(..)`/`weight(..)` and `Model::set_combination(..)`
  are `Model::add_stage(objective, tolerance)` (lexicographic) or `Model::add_weighted(objective, weight)` (blended).
  A model mixing both is refused with `ModelError::MixedObjectives`, and a weight or tolerance of the model's own
  objective without a further objective with `ModelError::RankAlone`.
- `Model::add_vars` and `add_var_map` closures may return `Bounds` as well as `Var`.
- **`Var::bounds` takes a `Bounds`**, as a program column's and a new column's do: `Var::integer().bounds(0, 40)` is
  `Var::integer().bounds(bounds!(0.0, 40.0))`, checked when compiled. `.lower(..)` and `.upper(..)` stay for numbers
  known only at run time.
- **`ProgramColumn` and the new `NewCol` are built by chaining.** `ProgramColumn::continuous(cost, bounds)` is
  `ProgramColumn::continuous().cost(c).bounds(b)`; the same for `integer`, `binary` (no cost argument now),
  `semi_continuous` and `semi_integer`, and `free()` is new.
- `ProgramPlan::row(bounds, terms)` is `ProgramPlan::row(impl Into<ProgramRow>)`, as `plan.row((bounds, [(x, 1.0)]))`.
- `Program::write_lp(&mut impl fmt::Write)` is `write_lp(impl io::Write)`, and `LpFileError::Write` carries the
  `io::Error`.
- `Program::is_mip()` is `program.class().is_discrete()`; `ModelClass::is_integer()` is `ModelClass::is_discrete()`
  (semi columns count as discrete).
- New on `Program`: `coefficient(row, col)`, `col_entries(col)`, `col_named(name)` and `row_named(name)`. New on
  `Cols` and `Rows`: `start`, `end`, `is_empty` and `contains`, and they iterate again when borrowed.
- `NewRow` moved from the solve module to the model module (still `culmen::NewRow`).
- `ProgramFeature::StartSolution` and `ProgramFeature::is_per_solve()` are gone: a start is the `Starts`
  capability.
- New on `Outcome`: `bound()`, the proven bound.

### Editing a loaded program

- **New:** `Solver::set_sense`, `set_offset` and `set_objective` edit the program's own objective;
  `Solver::set_col_lower`, `set_col_upper`, `set_row_lower` and `set_row_upper` change one end of a bound, keeping
  the other; `Solver::set_coefficients([(row, col, value)])` changes entries of the constraint matrix, zero taking one
  out (gurobipy's `chgCoeff`), as the new `Edit::Coefficients`, in place on HiGHS and by a reload on SCIP.

### Module paths

- **`culmen::Highs` and `culmen::Scip` at the root are gone.** Use `culmen::prelude::Highs` or `culmen::highs::Highs`,
  and `culmen::scip::Scip` alike.
- **The `culmen_highs::callback` module is private.** Its types are at the crate root, with the `Highs` prefix:
  `callback::MipData` is `HighsMipData`, `callback::HighsMipProgress` is `HighsProgress`, `callback::PollSource` is
  `HighsPollSource`, `callback::QueryOrigin` is `HighsQueryOrigin`, `callback::RootNodePoint` is `HighsRootNodePoint`,
  `callback::Cut` is `HighsCut` and `callback::CutPool` is `HighsCutPool`.
- `culmen::highs::files::{ModelFormat, FileError}` are `culmen::highs::{ModelFormat, FileError}`; `files` keeps its
  functions.
- SCIP's `LazySource` is `ScipLazySource`.
- The prelude is a selection of what a model, a solve and a handler name; it states what it leaves out.

### Behaviour

- With several objectives on HiGHS, the callbacks' solutions carry the program's own objective value, the generic
  `Progress` carries no objective and no bound (HiGHS's stage or blend figures stay in `HighsMipData`), and a supplied
  heuristic solution HiGHS takes is reported as the incumbent.
- `rust-version` is 1.100.

### Fixed since 0.4.0

- HiGHS: with several objectives, callbacks see the program's own objective and a supplied incumbent is reported.
- HiGHS: info, option, solution and name reads convert HiGHS's status totally, an unlisted one failing the read.
- HiGHS: `solve_relaxation` is refused for a MIP instead of reporting the relaxation's optimum as the program's.
- SCIP: pricing is switched on by a registered or relayed pricer, not by a relayed sink.
- SCIP: a `char` parameter value converts with `TryFrom`, refusing a character that is not ASCII.
- SCIP: a cut or a lazy constraint over fixed columns that add up beyond SCIP's infinity is refused; the check of a
  lazy constraint over fixed columns walks SCIP's representation itself.
- SCIP: a cut SCIP refuses, a lazy constraint SCIP refuses in enforcement, a failing constraint enforcement and a
  pricer's error stop SCIP instead of returning an error code; a Farkas round refuses an early stop.
- SCIP: the lazy handler locks every priced column both ways and unlocks it with its problem.
- SCIP: a priced solve's point is checked against the sides SCIP's presolve moved.
- Core: the basis is refused once a ray was asked of the backend.
- Core: a weight or tolerance of the model's own objective without a further objective is refused.
- Core: the LP writer labels lines named alike with a few tries each, not one per line before.
- Core: replacing a program's objective asserts one cost per column, as its docs say.

### Not in 0.5.0

Each of these would change an exhaustive type and so waits for a later minor release: a pool of the solutions a
solve found, removing rows or columns (ids are only appended), quadratic constraints, SOS and indicator constraints,
and general constraints (min, max, abs).
