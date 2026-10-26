# Analysis

A solve's `Outcome` holds the primal solution. The `Solver` also reports what comes with it: the duals and reduced
costs of an optimal LP or QP, the simplex basis (read after a solve, set before the next as a warm start), cost and
bound ranging, a primal ray of an unbounded LP and a Farkas certificate of an infeasible one, an irreducible
infeasible subsystem, and a feasibility relaxation. Every result is plain data read with the program's own `Col` and
`Row` handles.

## Analyses and refusal

Each kind is an `Analysis`, and a backend declares the ones it provides as `Backend::ANALYSES`, an `Analyses` set.
Each analysis is defined only for some [classes](program.md#model-classes) of program: `Analysis::classes()`.

| `Analysis` | `Solver` methods | Classes | Reads |
|---|---|---|---|
| `Duals` | `duals()` | LP, QP | an optimal solve |
| `Basis` | `basis()`, `set_basis(&basis)` | LP | any solve (`basis`), none (`set_basis`) |
| `Ranging` | `ranging()` | LP | an optimal solve |
| `Rays` | `primal_ray()`, `dual_ray()` | LP | any solve |
| `Iis` | `iis()` | LP, QP | none |
| `FeasibilityRelaxation` | `feasibility_relaxation(&weights)` | LP, MILP | none |

A MIP has no duals, basis or ranging: they are statements about a continuous program, and the LP left after fixing
the integers says nothing about the MIP. An IIS of a MILP's relaxation is not an IIS of the MILP, so `iis` takes only
continuous programs; a QP's constraints are linear, so its IIS is theirs. A feasibility relaxation minimizes a linear
violation under the program's integrality, so a MILP has one; a QP, whose quadratic part has no place in it, is
refused.

The `Solver` checks every request the same way for every backend, before the backend is asked, and refuses with a
typed `AnalysisError<B::Error>`, in this order:

| Variant | When |
|---|---|
| `Unsupported { analysis, supported }` | The backend does not declare the analysis. |
| `Class { analysis, class }` | The analysis is not defined for the program's class. |
| `Misfit(Misfit)` | A basis or weights do not fit the program: `Length { table, expected, found }`, `BasicCount { expected, found }`, or `Status { line, status }` for a line put at a bound it does not have. |
| `NotSolved { analysis }` | It reads a solve and there is none: no solve since the program was loaded, the last one failed, or a later `set_basis`, `iis` or `feasibility_relaxation` replaced its result. |
| `NotOptimal { analysis }` | It reads an optimal solve and the last one did not end `Optimal`. |
| `Backend(E)` | The backend failed; for HiGHS, `HighsError::Analysis(analysis)`. |

| Backend | `ANALYSES` |
|---|---|
| `Highs` | Every analysis. |
| `Script` | None: it replays events and has no duals to report honestly. |

```rust
use culmen::prelude::*;
use culmen::{Analyses, Analysis, AnalysisError, ModelClass, Script};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().upper(3.0));
    m.add_constraint(x.leq(2.5));
    m.maximize(x);
    let program = m.build()?;

    let mut solver = Solver::new(Highs::new(), program.clone())?;
    solver.solve(Request::new())?;
    let refused = solver.duals().unwrap_err();
    assert_eq!(refused, AnalysisError::Class { analysis: Analysis::Duals, class: ModelClass::Milp });
    assert_eq!(refused.to_string(), "Solver: duals are defined for {LP, QP}, not a MILP");

    let script = Solver::new(Script::optimal(2.0, vec![2.0]), program)?;
    let refused = script.basis().unwrap_err();
    assert_eq!(refused, AnalysisError::Unsupported { analysis: Analysis::Basis, supported: Analyses::NONE });
    Ok(())
}
```

## Duals and reduced costs

`duals()` returns `Duals { rows, cols }`: a `TiVec<Row, f64>` of row duals and a `TiVec<Col, f64>` of reduced costs,
read as `duals.rows[r]` and `duals.cols[x]`.

The sign convention is HiGHS's, and it reads the same in either sense: **every dual is the rate at which the optimal
objective changes as the bound it belongs to rises.** At the optimum `x`, with costs `c`, the gradient `∇q(x)` of the
quadratic part and the matrix `A`,

```text
c + ∇q(x) = Aᵀ·rows + cols
```

A row or column strictly inside its bounds has dual zero. Minimizing, a line at its lower bound has a dual of zero or
above and a line at its upper bound one of zero or below; maximizing reverses both. For an LP the optimal objective is
then the offset plus each dual times the bound its line is at.

A row's activity is not a dual: it is read from the solution with `program.activities(&values)`, a
`TiVec<Row, f64>`, or one row at a time with `program.row(r).activity(&values)`.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Maximize 3x + 2y subject to x + y <= 4 and x + 3y <= 7, with x <= 3: x = 3 at its bound, y = 1.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(3.0));
    let y = m.add_var(Var::continuous());
    let r1 = m.add_constraint((x + y).leq(4.0));
    let r2 = m.add_constraint((x + 3.0 * y).leq(7.0));
    m.maximize(3.0 * x + 2.0 * y);

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let best = solver.solve(Request::new())?.outcome.into_solution().ok_or("no optimum")?;
    let duals = solver.duals()?;
    // c_y = 2 = rows[r1] with y basic; c_x = 3 = rows[r1] + cols[x]; r2 is slack.
    assert_eq!((duals.rows[r1], duals.rows[r2]), (2.0, 0.0));
    assert_eq!((duals.cols[x], duals.cols[y]), (1.0, 0.0));
    // Strong duality: 4·2 + 3·1 = 11.
    assert_eq!(4.0 * duals.rows[r1] + 3.0 * duals.cols[x], best.objective);
    let activities = solver.program().activities(&best.values);
    assert_eq!((activities[r1], activities[r2]), (4.0, 6.0));
    Ok(())
}
```

## The basis and warm starts

`basis()` returns the `Basis` the last solve ended with, `None` when the backend holds none (as after an
interior-point solve without crossover). A `Basis` holds one `BasisStatus` per column and per row: `Lower`, `Basic`,
`Upper`, `Zero` (a free line nonbasic at zero) or `NonBasic`, HiGHS's `kHighsBasisStatus*`. A row's status refers to
its activity.

`set_basis(&basis)` sets the basis the next solve starts from. `Basis::check` runs first and refuses, as
`AnalysisError::Misfit`, a basis with the wrong number of statuses, without one basic line per row, or that puts a
line at a bound it does not have, as a basis of another program would. `Basis::slack(&program)` is the basis with
every row basic and every column at a bound it has.

```rust
use culmen::prelude::*;
use culmen::BasisStatus;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_vars(3, |_| Var::continuous().upper(4.0));
    m.add_constraint((x[0] + x[1] + x[2]).leq(6.0));
    m.add_constraint((x[0] - x[1]).geq(-1.0));
    m.maximize(2.0 * x[0] + 3.0 * x[1] + x[2]);
    let program = m.build()?;
    let no_presolve = Settings { presolve: Some(false), ..Settings::new() };

    let mut cold = Solver::new(Highs::new(), program.clone())?;
    cold.solve(Request::new().settings_from(no_presolve.clone()))?;
    let basis = cold.basis()?.ok_or("no basis")?;
    assert_eq!(basis.rows.iter().chain(&basis.cols).filter(|&&s| s == BasisStatus::Basic).count(), 2);

    let mut warm = Solver::new(Highs::new(), program)?;
    warm.set_basis(&basis)?;
    let solved = warm.solve(Request::new().settings_from(no_presolve))?;
    assert_eq!(solved.status.simplex_iterations, 0, "the optimal basis needs no pivot");
    Ok(())
}
```

## Ranging

`ranging()` reports, for an optimal LP, how far each cost and bound may move, one at a time, before the optimal basis
changes. `Ranging` holds `costs` and `col_bounds` per column and `row_bounds` per row, each a `Ranged { down, up }`
of two `Limit`s:

| `Limit` field | Holds |
|---|---|
| `value` | The cost or value at which the basis changes; infinite when it never does in that direction. |
| `objective` | The optimal objective there, offset included. |
| `entering`, `leaving` | The `Line` (a `Line::Col` or a `Line::Row`) entering and leaving the basis there; `None` when none does. |

```rust
use culmen::prelude::*;
use culmen::Line;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize -3x - 2y subject to x + y <= 4, x + 3y <= 7, x <= 3: x = 3 at its bound, y = 1 basic.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(3.0));
    let y = m.add_var(Var::continuous());
    let r1 = m.add_constraint((x + y).leq(4.0));
    let r2 = m.add_constraint((x + 3.0 * y).leq(7.0));
    m.minimize(-3.0 * x - 2.0 * y);

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    solver.solve(Request::new())?;
    let cost = solver.ranging()?.costs[y];
    // y's cost may fall to -3, where x leaves its bound and r2 tightens, and rise to 0, where r1 loosens.
    assert_eq!((cost.down.value, cost.down.objective), (-3.0, -12.0));
    assert_eq!((cost.down.entering, cost.down.leaving), (Some(Line::Col(x)), Some(Line::Row(r2))));
    assert_eq!((cost.up.value, cost.up.entering), (0.0, Some(Line::Row(r1))));
    Ok(())
}
```

## Rays

`primal_ray()` returns a direction `d` of unboundedness: every line's recession allows it (`d ≥ 0` along a line with
only a finite lower bound, `d ≤ 0` with only an upper one, zero with both, for columns and for row activities `Ad`),
and the objective improves along it. `dual_ray()` returns a Farkas certificate of infeasibility: row multipliers `y`
for which the largest `yᵀAx` over the columns' bounds is below the least `yᵀr` over the rows' bounds. Each is `None`
unless the last solve ended as it proves (`Unbounded` or `Infeasible`, or `InfeasibleOrUnbounded` for either), or
when the backend finds none.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // x + y <= 1 and x + y >= 3 contradict each other.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(10.0));
    let y = m.add_var(Var::continuous().upper(10.0));
    let below = m.add_constraint((x + y).leq(1.0));
    let above = m.add_constraint((x + y).geq(3.0));
    m.minimize(x + y);

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    assert_eq!(solver.solve(Request::new())?.outcome, Outcome::Infeasible);
    let ray = solver.dual_ray()?.ok_or("no certificate")?;
    // (-1, 1): -(x + y) + (x + y) = 0 over the box, but -r1 + r2 >= -1 + 3 = 2 over the rows' bounds.
    assert_eq!((ray[below], ray[above]), (-1.0, 1.0));
    assert_eq!(solver.primal_ray()?, None);
    Ok(())
}
```

## Irreducible infeasible subsystems

`iis()` returns an `Iis` of an infeasible program's constraints, `None` when they are feasible. It needs no solve. An
`Iis` holds `rows` and `cols`, each a `BTreeMap` from the handle to the `Side` of its bounds in the subsystem:
`Lower`, `Upper` or `Both`. The subsystem (the listed rows with the listed sides, the listed sides of the listed
columns' bounds, every other bound dropped) is infeasible, and dropping any one entry makes it feasible. HiGHS
reports `Both` for a column with two finite bounds even when one side suffices.

```rust
use std::collections::BTreeMap;

use culmen::prelude::*;
use culmen::Side;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::free().upper(1.0));
    let y = m.add_var(Var::free().upper(2.0));
    let need = m.add_constraint((x + y).geq(5.0));
    m.add_constraint((x - y).leq(10.0));
    m.minimize(x);

    let mut solver = Solver::new(Highs::new(), m.build()?)?;
    let iis = solver.iis()?.ok_or("feasible")?;
    assert_eq!(iis.rows, BTreeMap::from([(need, Side::Lower)]));
    assert_eq!(iis.cols, BTreeMap::from([(x, Side::Upper), (y, Side::Upper)]));
    Ok(())
}
```

## Feasibility relaxation

`feasibility_relaxation(&weights)` finds a point of least weighted violation, `None` when no point meets the bounds
and rows kept hard. `Weights` holds an `Option<Penalty>` per column lower bound (`lower`), per column upper bound
(`upper`) and per row (`rows`); `None` keeps it hard, and a row's penalty prices a violation of either of its bounds.
`Weights::uniform(&program, cols, rows)` gives every column bound one price and every row another. A `Penalty` is a
finite number, zero or above: a literal is `penalty!(2.0)`, checked when the code is compiled, and a number read at
run time is checked by `Penalty::new`.

The result is `Relaxed { values }`, one value per column. `relaxed.violations(&program)` yields each
`Violation { line, bound, amount }` the point commits, columns first, and `weights.penalty(violations)` prices them.
A MILP keeps its integrality.

```rust
use culmen::prelude::*;
use culmen::{Bound, Line, Weights, penalty};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Demand of at least 5 against a capacity of 3: they cannot both hold.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(10.0));
    let demand = m.add_constraint(x.geq(5.0));
    let capacity = m.add_constraint(x.leq(3.0));
    m.minimize(x);
    let program = m.build()?;

    // Missing demand costs 1 per unit, exceeding capacity 3.
    let mut weights = Weights::uniform(&program, None, Some(penalty!(1.0)));
    weights.rows[capacity] = Some(penalty!(3.0));
    let mut solver = Solver::new(Highs::new(), program.clone())?;
    let relaxed = solver.feasibility_relaxation(&weights)?.ok_or("hard constraints conflict")?;

    let violations: Vec<_> = relaxed.violations(&program).collect();
    assert_eq!(violations.len(), 1);
    assert_eq!((violations[0].line, violations[0].bound), (Line::Row(demand), Bound::Lower));
    assert!((violations[0].amount - 2.0).abs() < 1e-9);
    assert!((weights.penalty(violations) - 2.0).abs() < 1e-9);
    Ok(())
}
```

## HiGHS

`Highs` provides every analysis through HiGHS's own calls: `Highs_getSolution` for duals, `Highs_getBasis` and
`Highs_setBasis`, `Highs_getRanging`, `Highs_getPrimalRay` and `Highs_getDualRay`, `Highs_getIis` with its full IIS
strategy, and `Highs_feasibilityRelaxation`. An IIS and a relaxation reset HiGHS's options to the backend's
`HighsOptions` first, as a solve does. What the backend adds to HiGHS 1.15:

- **An LP without rows has a primal ray.** HiGHS reports none; the backend returns the unit direction of the first
  column whose cost improves toward an infinite bound.
- **A relaxation that cannot meet its hard constraints is `None`.** HiGHS does not report whether its elastic program
  was feasible, so the backend checks the point it returns against the hard bounds, rows and integrality.
- **A QP without rows has no row duals.** HiGHS is given one free row for such a QP (see
  [Quadratic programs](highs.md#quadratic-programs)); it is not one of the program's rows.
- **A ray can replace the basis.** To find a ray HiGHS may solve the LP again, without presolve or with zero costs.

## See also

- [Solving](solving.md): the solver, requests and outcomes.
- [HiGHS](highs.md): the backend's options, statuses and errors.
- [Backends](backends.md): the `Backend` seam.
- [Documentation index](README.md).
