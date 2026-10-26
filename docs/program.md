# The program as data

A `Program` is a linear, mixed-integer or convex quadratic program: minimize or maximize
`cost · x + Σ q · xᵢ · xⱼ + offset` over `x`, with every column in its bounds, the integer columns whole, every semi
column zero or in its bounds, and every row's activity `Σ coefficient · column` in its bounds; with further linear
objectives, it optimizes them lexicographically or blended. Its constraints are linear. It is an
immutable value, built once and checked then, and it is what a [backend](solving.md#solver) loads. A
[model](modelling.md) builds one; this page builds them directly and reads them back.

## `ProgramPlan`

A `ProgramPlan` is a program as plain data, with `pub` fields, checked into a `Program` by `ProgramPlan::build`:

| Field | Holds |
|---|---|
| `sense` | `Sense::Minimize` (the default) or `Sense::Maximize`. |
| `offset` | The constant added to the objective. |
| `columns` | A `TiVec<Col, ProgramColumn>`: each column's `cost`, `bounds`, `integrality` and `name`. |
| `rows` | A `TiVec<Row, ProgramRow>`: each row's `bounds`, `terms` as `(Col, f64)`, and `name`. |
| `quadratic` | The objective's quadratic terms as `(Col, Col, f64)`, each `(i, j, q)` adding `q · xᵢ · xⱼ`; empty for a linear objective. See [The quadratic part](#the-quadratic-part). |
| `objectives` | Further linear objectives and how they combine, `Objectives::Single` for none. See [Several objectives](#several-objectives). |

`plan.col(column)` pushes a column and returns its `Col`; `plan.row(bounds, terms)` pushes an unnamed row and
returns its `Row`. `ProgramColumn::continuous(cost, bounds)`, `ProgramColumn::integer(cost, bounds)`,
`ProgramColumn::binary(cost)`, `ProgramColumn::semi_continuous(cost, bounds)` and
`ProgramColumn::semi_integer(cost, bounds)` make columns without a name; the fields are public, so set `name` directly.

```rust
use culmen::{Bounds, ProgramColumn, ProgramPlan, Sense, bounds};

fn main() -> Result<(), culmen::ProgramError> {
    // Maximize 3x + 2y subject to x + y <= 4 and x + 3y <= 6, with x, y >= 0.
    let mut plan = ProgramPlan::new();
    plan.sense = Sense::Maximize;
    let x = plan.col(ProgramColumn { name: Some("x".into()), ..ProgramColumn::continuous(3.0, Bounds::NON_NEGATIVE) });
    let y = plan.col(ProgramColumn::continuous(2.0, Bounds::NON_NEGATIVE));
    let total = plan.row(bounds!(f64::NEG_INFINITY, 4.0), [(x, 1.0), (y, 1.0)]);
    plan.row(bounds!(f64::NEG_INFINITY, 6.0), [(x, 1.0), (y, 3.0), (x, 0.5), (x, -0.5)]);
    plan.rows[total].name = Some("total".into());

    let program = plan.build()?;
    assert_eq!((program.num_cols(), program.num_rows(), program.num_entries()), (2, 2, 4));
    assert_eq!(program.row(total).name(), Some("total"));
    Ok(())
}
```

A row's terms may repeat a column; their coefficients are added up in the order given, and an entry that sums to
zero is dropped. Every row of a `Program` holds each column at most once, with a non-zero coefficient, sorted by column.

`Program` and `ProgramPlan` convert both ways: `ProgramPlan::from(program)` takes a program apart into its plan, and
`Program::try_from(plan)` is `plan.build()`.

## `Program`

A `Program` is read through methods; it has no public fields and no constructor that skips the checks.

| Method | Gives |
|---|---|
| `sense()`, `offset()` | The objective's sense and constant. |
| `num_cols()`, `num_rows()`, `num_entries()` | The counts; entries are the non-zero coefficients. |
| `is_mip()` | Whether any column is integer, semi-continuous or semi-integer. |
| `quadratic()` | The objective's [quadratic part](#the-quadratic-part), a `QuadObjective`. |
| `objectives()` | The further objectives and how they combine, an [`Objectives`](#several-objectives). |
| `class()` | The [`ModelClass`](#model-classes): `Lp`, `Milp`, `Qp` or `Miqp`. |
| `features()` | The [`ProgramFeatures`](#features) it uses beyond its class. |
| `cols()`, `rows()` | Every `Col`, every `Row`, in order. |
| `columns()` | The columns' `ProgramColumn` data, indexed by `Col`: `program.columns()[x]`. |
| `row_bounds()` | The rows' bounds, indexed by `Row`. |
| `row(row)` | A [`RowView`](#row-views-and-iterators) of one row: its bounds, name and entries. |
| `matrix()` | The whole matrix in compressed row form, a `RowMatrix { starts, cols, values }`. |
| `objective_value(values)`, `objective_values(values)`, `max_violation(values)` | See [Checking a point](#checking-a-point). |
| `to_lp_string()`, `write_lp(out)` | See [LP files](#lp-files). |

The methods that take a handle panic when the program has no such column or row; they never read anything undefined.

## `Bounds`

`Bounds` is a closed interval `lower ..= upper` that a column or a row activity must lie in. Either end may be
infinite, and the interval is never empty: neither end is NaN, `lower <= upper`, the lower end is never `+inf` and
the upper end never `-inf`. No `Bounds` value can break this, since every way to make one checks it.

| Way to make one | Checked |
|---|---|
| `bounds!(lower, upper)` | When the code is compiled; an invalid literal is a compile error. |
| `Bounds::FREE`, `Bounds::NON_NEGATIVE`, `Bounds::BINARY` | By construction: `(-inf, +inf)`, `[0, +inf)`, `[0, 1]`. |
| `Bounds::new(lower, upper)`, `fixed(v)`, `at_least(l)`, `at_most(u)` | At run time; `Err(InvalidBounds)` when refused. |
| `Bounds::try_from(l..=u)`, `try_from(l..)`, `try_from(..=u)` | At run time, as the matching constructor. |
| `Bounds::from(..)` | Always `Bounds::FREE`. |

Write literals with `bounds!`, which works in a `const` too, and use the checked constructors for numbers read at
run time:

```rust
use culmen::{Bounds, InvalidBounds, bounds};

const STOCK: Bounds = bounds!(0.0, 40.0);
assert_eq!((STOCK.lower(), STOCK.upper()), (0.0, 40.0));
assert_eq!(Bounds::try_from(0.0..=40.0), Ok(STOCK));
assert_eq!(Bounds::try_from(0.0..), Ok(Bounds::NON_NEGATIVE));
assert_eq!(Bounds::from(..), Bounds::FREE);
assert_eq!(Bounds::new(2.0, 1.0), Err(InvalidBounds::Empty { lower: 2.0, upper: 1.0 }));

assert!(STOCK.contains(12.0));
assert_eq!(STOCK.violation(41.5), 1.5);
assert_eq!(STOCK.shifted(-10.0), Ok(bounds!(-10.0, 30.0)));
```

```compile_fail
use culmen::{Bounds, bounds};

const EMPTY: Bounds = bounds!(2.0, 1.0); // the lower end exceeds the upper end
```

`lower()` and `upper()` read the ends, `contains(v)` tests a value, `violation(v)` is how far `v` lies outside (zero
inside), and `shifted(d)` moves both ends, refusing a non-finite shift or a finite end that overflows.

`Bounds` displays in interval notation, an infinite end open; the formatter's options apply to each finite end:

```rust
use culmen::{Bounds, bounds};

assert_eq!(bounds!(0.0, 40.0).to_string(), "[0, 40]");
assert_eq!(Bounds::NON_NEGATIVE.to_string(), "[0, +inf)");
assert_eq!(bounds!(f64::NEG_INFINITY, 1.0).to_string(), "(-inf, 1]");
assert_eq!(Bounds::FREE.to_string(), "(-inf, +inf)");
assert_eq!(format!("{:.1}", bounds!(0.5, 2.0)), "[0.5, 2.0]");
```

`InvalidBounds` says why an interval was refused:

| Variant | Message |
|---|---|
| `NotANumber` | `Bounds: an end is NaN` |
| `LowerIsPlusInfinity` | `Bounds: the lower end is +inf` |
| `UpperIsMinusInfinity` | `Bounds: the upper end is -inf` |
| `Empty { lower, upper }` | `Bounds: the lower end {lower} exceeds the upper end {upper}` |
| `Overflow` | `Bounds: shifting a finite end overflowed it` |
| `NonFiniteShift { shift }` | `Bounds: the shift {shift} is not finite` |

## Integrality and sense

`Integrality` is `Continuous` (the default), `Integer`, `SemiContinuous` or `SemiInteger`. A binary column is an
integer column with `Bounds::BINARY`; the program stores no kind of its own for it, since the bounds already say it. A
semi column is zero or in its bounds, whole when `SemiInteger` and not zero; a program holds one only with bounds
`lower ..= upper`, `0 < lower` and `upper` finite, so zero lies outside them. `is_integral()` (`Integer`,
`SemiInteger`), `is_semi()` and `is_discrete()` (anything but `Continuous`) read the kinds back, and
`violation(bounds, value)` measures how far a value lies from the values a column takes.

`Sense` is `Minimize` (the default) or `Maximize`. `sense.better(a, b)` says whether objective value `a` is strictly
better than `b` under it.

## `Col` and `Row`

`Col` and `Row` are typed, `Copy` handles: the position of a column or a row, counting from zero. Because they are
distinct types, a table indexed by columns cannot be read with a row. `Col::new(i)` and `Row::new(i)` make a handle
from a position, `index()` reads it back, and both convert to and from `usize`.

Handles come from `Model::add_var`, `Model::add_constraint`, `ProgramPlan::col` and `ProgramPlan::row`; write `Col::new`
yourself only where the position itself is the point. A handle from one program used on another reads that program's
column or row of the same position; a read that cannot answer for the position panics or returns `None`, as its docs
say.

Per-column data is a `TiVec<Col, f64>` or a borrowed `TiSlice<Col, f64>`, re-exported as `culmen::TiVec` and
`culmen::TiSlice` from `typed-index-collections`. They index by `Col` only: `values[x]`.

## Row views and iterators

`program.row(r)` is a `RowView`: the row's `bounds()`, `name()`, `len()`, `is_empty()`, its `entries()` as
`(Col, f64)` sorted by column, and `activity(values)`, the row's `Σ coefficient · value` at a point. A `RowView` is
also `IntoIterator` over its entries.

```rust
use culmen::{ProgramColumn, ProgramPlan, TiVec, bounds};

fn main() -> Result<(), culmen::ProgramError> {
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::binary(1.0));
    let y = plan.col(ProgramColumn::binary(1.0));
    let r = plan.row(bounds!(0.0, 1.0), [(y, 2.0), (x, 1.0)]);
    let program = plan.build()?;

    let row = program.row(r);
    assert_eq!((row.bounds(), row.name(), row.len()), (bounds!(0.0, 1.0), None, 2));
    assert_eq!(row.into_iter().collect::<Vec<_>>(), [(x, 1.0), (y, 2.0)]);
    assert_eq!(program.row(r).entries().rev().next(), Some((y, 2.0)));

    let point: TiVec<culmen::Col, f64> = vec![1.0, 1.0].into();
    assert_eq!(row.activity(&point), 3.0);
    assert_eq!(program.cols().collect::<Vec<_>>(), [x, y]);
    assert_eq!(program.rows().len(), 1);
    Ok(())
}
```

The iterators (`Cols`, `Rows`, `RowEntries`, and the family iterators of [Modelling](modelling.md#keyed-families))
are named types that run both ways and know their exact length.

## The quadratic part

`program.quadratic()` is a `QuadObjective`, the objective's `Σ q · xᵢ · xⱼ` in one canonical form: entries `(i, j, q)`
with `i <= j`, each pair at most once, every coefficient finite and non-zero, sorted by `(i, j)`. The model's `x * x`
is the entry `(x, x, 1)` and `2.0 * x * y` the entry `(x, y, 2)`. A plan's terms may name a pair in either order and
repeat it; the coefficients of a pair are added up in the order given, and a pair that sums to zero is dropped.

Solvers take the quadratic part as `½ xᵀHx` with a symmetric Hessian `H`. In that form `Hᵢᵢ = 2q` on the diagonal and
`Hᵢⱼ = Hⱼᵢ = q` off it; `quadratic().hessian()` gives its lower triangle column by column, `(column, row, value)` with
`row >= column`, which is the layout HiGHS reads.

| Method | Gives |
|---|---|
| `entries()`, `&q` in a `for` loop | The entries `(i, j, q)`, as a `QuadEntries` iterator. |
| `hessian()` | The lower triangle of `H`, as a `HessianEntries` iterator. |
| `len()`, `is_empty()` | The number of entries. |
| `value(values)` | `Σ q · xᵢ · xⱼ` at a point. |

```rust
use culmen::{Bounds, ProgramColumn, ProgramPlan, TiSlice};

fn main() -> Result<(), culmen::ProgramError> {
    // x² + 2xy + 3y², written with the cross term split in both orders.
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(0.0, Bounds::FREE));
    let y = plan.col(ProgramColumn::continuous(0.0, Bounds::FREE));
    plan.quadratic = vec![(y, x, 1.0), (x, x, 1.0), (x, y, 1.0), (y, y, 3.0)];
    let program = plan.build()?;

    let q = program.quadratic();
    assert_eq!(q.entries().collect::<Vec<_>>(), [(x, x, 1.0), (x, y, 2.0), (y, y, 3.0)]);
    assert_eq!(q.hessian().collect::<Vec<_>>(), [(x, x, 2.0), (x, y, 2.0), (y, y, 6.0)]);
    assert_eq!(program.objective_value(TiSlice::from_ref(&[1.0, 1.0])), 6.0);
    Ok(())
}
```

The class of program is *convex* quadratic. Convexity is a property of real numbers that floating point cannot
decide exactly, so `build` checks it with a stated tolerance: `H` splits into blocks of columns its entries join,
and for every block `s · H + δ · I` must have a Cholesky factor, where `s` is `1` to minimize and `-1` to maximize and
`δ` is `Program::CONVEXITY_TOLERANCE` (`1e-9`) times the block's largest `|Hᵢⱼ|`. Every `Program` therefore has an
objective convex to minimize, or concave to maximize, up to that tolerance; a semidefinite one passes. Each block costs
a dense factorization cubic in its number of columns, so a separable objective, a sum of squares of single columns,
costs next to nothing.

## Model classes

`program.class()` is a `ModelClass`, the kind of program it is: whether it has an integer column, and whether its
objective has a quadratic part. A semi-continuous or semi-integer column counts as integer, since its values are not
one interval; further objectives do not change the class, and are a [feature](#features).

| Class | Integer columns | Objective |
|---|---|---|
| `ModelClass::Lp` | none | linear |
| `ModelClass::Milp` | some | linear |
| `ModelClass::Qp` | none | convex quadratic |
| `ModelClass::Miqp` | some | convex quadratic |

`is_integer()` and `is_quadratic()` read the two properties back, and a class displays as its abbreviation: `LP`,
`MILP`, `QP`, `MIQP`. A backend declares the classes it solves as a `ModelClasses` set, and
[`Solver::new`](solving.md#model-classes-and-refusal) refuses any other. `ModelClass` is non-exhaustive: quadratic
constraints would come as further classes, which a backend that does not declare them refuses.

```rust
use culmen::{ModelClass, ModelClasses};

let linear = ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]);
let classes = linear | ModelClass::Qp.into();
assert!(classes.contains(ModelClass::Qp) && !classes.contains(ModelClass::Miqp));
assert_eq!(classes & linear, linear);
assert_eq!(classes.iter().collect::<Vec<_>>(), [ModelClass::Lp, ModelClass::Milp, ModelClass::Qp]);
assert_eq!(classes.to_string(), "{LP, MILP, QP}");
assert_eq!(format!("{classes:?}"), "ModelClasses(LP | MILP | QP)");
assert_eq!(ModelClasses::ALL.len(), 4);
```

`ModelClasses` is a small set: `NONE`, `ALL`, `of(&[..])` in a `const`, `From<ModelClass>`, `contains`, `union` and
`intersection` (also `|`, `&`, `|=`, `&=`), `len`, `is_empty`, an `iter()` that runs both ways, `FromIterator` and
`Extend` of classes.

## Features

`program.features()` is a `ProgramFeatures` set of what the program uses beyond its class, each a `ProgramFeature`:
`SemiContinuous` and `SemiInteger` for a column of that kind, `MultipleObjectives` for a further objective. A fourth,
`StartSolution`, belongs to a solve's [start](solving.md#starts), never to a program. A backend declares the features
it takes in `Backend::FEATURES`, and [`Solver::new`](solving.md#model-classes-and-refusal) refuses a program using
any other. `ProgramFeature` is non-exhaustive; `ProgramFeatures` is a set like `ModelClasses`, with `difference` and
`contains_all` besides.

```rust
use culmen::prelude::*;
use culmen::{ProgramFeature, ProgramFeatures};

fn main() -> Result<(), culmen::ModelError> {
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(1, 4));
    m.maximize(x);
    m.add_objective(Objective::minimize(x));
    let features = m.build()?.features();
    assert_eq!(features, ProgramFeatures::of(&[ProgramFeature::SemiContinuous, ProgramFeature::MultipleObjectives]));
    assert_eq!(features.to_string(), "{semi-continuous columns, multiple objectives}");
    assert!(!features.contains(ProgramFeature::StartSolution));
    Ok(())
}
```

## Several objectives

`program.objectives()` is an `Objectives`, how the program's own objective combines with further `LinearObjective`s,
each a `sense`, `terms` and an `offset`:

| Variant | Meaning |
|---|---|
| `Single` | The program's own objective alone. |
| `Lexicographic { first, then }` | The own objective first, then each `Stage { objective, tolerance }` in order, each over the points keeping every earlier objective within its `Tolerance` of the optimum its stage found. |
| `Blended { first, with }` | One objective minimized: `Σ s · w · f` over the own objective with weight `first` and each `Weighted { objective, weight }`, `s` one to minimize and minus one to maximize. |

A `Tolerance { absolute, relative }` allows `min(absolute, relative · |f*|)` from an optimum `f*`, an absent part
left out, both absent no constraint at all; `Tolerance::EXACT` allows nothing. A `Weight` is a finite number above
zero, written `weight!(2.0)` for a literal. `objectives.further()` iterates the further objectives; `ObjectiveId`
numbers them from one, the own objective being `ObjectiveId::FIRST`.

In a `Program`, every further objective's terms are sorted, merged and stripped of zeros, its coefficients and offset
finite, and the own objective is linear: `build` refuses a quadratic part beside further objectives. Without a further
objective, either combination is `Single`.

```rust
use culmen::{Bounds, LinearObjective, Objectives, ProgramColumn, ProgramPlan, Sense, Stage, TiSlice, Tolerance};

fn main() -> Result<(), culmen::ProgramError> {
    let mut plan = ProgramPlan::new();
    plan.sense = Sense::Maximize;
    let x = plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
    let y = plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
    let second = LinearObjective { sense: Sense::Minimize, terms: vec![(y, 1.0), (y, 1.0)], offset: 0.0 };
    plan.objectives = Objectives::Lexicographic {
        first: Tolerance::EXACT,
        then: vec![Stage { objective: second, tolerance: Tolerance::EXACT }],
    };
    let program = plan.build()?;
    assert_eq!(program.objectives().further().next().unwrap().terms, [(y, 2.0)]);
    assert_eq!(program.objective_values(TiSlice::from_ref(&[3.0, 1.0])).raw, [4.0, 2.0]);
    let _ = x;
    Ok(())
}
```

## Checking a point

`program.objective_value(values)` is the objective at a point, quadratic part and offset included.
`program.objective_values(values)` is every objective's value, the own one first, indexed by `ObjectiveId`.
`program.max_violation(values)` is the largest amount by which the point violates a column bound, a row, integrality
or a semi column's choice between zero and its bounds, and zero for a feasible point. Both panic when `values` holds fewer values than the program has columns. They are the
way to check a solution against the program independently of any solver:

```rust
use culmen::prelude::*;

fn main() -> Result<(), culmen::ModelError> {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(0, 3));
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + y).leq(4.0));
    m.maximize(2.0 * x + y + 1.0);
    let program = m.build()?;

    let good: culmen::TiVec<Col, f64> = vec![3.0, 1.0].into();
    assert_eq!((program.objective_value(&good), program.max_violation(&good)), (8.0, 0.0));
    let bad: culmen::TiVec<Col, f64> = vec![2.5, 2.0].into();
    assert_eq!(program.max_violation(&bad), 0.5);
    Ok(())
}
```

## LP files

`program.to_lp_string()` writes the program in the CPLEX LP file format, a `Result<String, LpFileError>`, and `program.write_lp(&mut out)` writes it into
any `fmt::Write`. The output is for reading by eye or by another solver while debugging:

- Names are reduced to the characters the format allows. An unnamed column is written `x<index>` and an unnamed row
  `r<index>`; distinct names that reduce to the same text are not told apart.
- A ranged row becomes two rows suffixed `_lo` and `_hi`; a row with no finite bound is left out.
- Integer and semi-integer columns are listed under `General`, semi-continuous and semi-integer ones under
  `Semi-Continuous`: a column in both is semi-integer, as CPLEX and HiGHS read it. Bounds equal to the format's
  default `0 ..` are not written.
- A program with further objectives is refused with `LpFileError::Unsupported { feature: MultipleObjectives }`, before
  anything is written: CPLEX's multi-objective sections combine tolerances otherwise than `Tolerance` does, and
  HiGHS cannot read them. `LpFileError::Write` carries the writer's own error.
- A quadratic part is written doubled in the format's brackets halved: `x² + 2xy` as `[ 2 x ^ 2 + 2 x * y ] / 2`.
  HiGHS reads it back as the same objective.

```rust
use culmen::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(0, 5).name("x"));
    let y = m.add_var(Var::free().name("y"));
    m.add_named_constraint("band", (x - y).between(-1, 1));
    m.maximize(3.0 * x + y);
    let text = m.build()?.to_lp_string()?;
    assert_eq!(
        text,
        "Maximize\n obj: 3 x + 1 y\nSubject To\n band_lo: 1 x - 1 y >= -1\n band_hi: 1 x - 1 y <= 1\n\
         Bounds\n 0 <= x <= 5\n y free\nGeneral\n x\nEnd\n"
    );

    m.maximize(3.0 * x - x * x - 0.5 * x * y - y * y);
    let text = m.build()?.to_lp_string()?;
    assert!(text.starts_with("Maximize\n obj: 3 x + [ -2 x ^ 2 - 1 x * y - 2 y ^ 2 ] / 2\n"));
    Ok(())
}
```

## Checks and `ProgramError`

`ProgramPlan::build` accepts a plan when its offset and costs are finite, every term names a column of the plan with a
finite coefficient, in every row and every further objective the coefficients of each column, added up in the order
given, are finite, every further objective's offset is finite, every semi column's bounds exclude zero
(`0 < lower`, `upper` finite), every quadratic term names columns of the plan with a finite coefficient, the
coefficients of each pair, added up in the order given, are finite, the quadratic part passes the
[convexity check](#the-quadratic-part), and it is empty beside further objectives. Bounds and integrality need no
other check: their types already guarantee them. Otherwise it returns a `ProgramError`
naming the first defect:

| Variant | Message |
|---|---|
| `NonFiniteOffset { value }` | `Program: the objective offset {value} is not finite` |
| `NonFiniteCost { col, value }` | `Program: the cost {value} of {col:?} is not finite` |
| `NonFiniteCoefficient { row, col, value }` | `Program: the coefficient {value} of {col:?} in {row:?} is not finite` |
| `UnknownColumn { row, col }` | `Program: {row:?} names {col:?}, which the plan does not have` |
| `UnknownQuadColumn { col }` | `Program: a quadratic term names {col:?}, which the plan does not have` |
| `NonFiniteQuadCoefficient { first, second, value }` | `Program: the quadratic coefficient {value} of {first:?} · {second:?} is not finite` |
| `NonConvexObjective { sense, col }` | `Program: the quadratic objective is not convex in the block of {col:?}`, or `not concave` to maximize |
| `SemiBounds { col, bounds }` | `Program: the semi column {col:?} has the bounds {bounds}, not 0 < lower <= upper < +inf` |
| `NonFiniteObjectiveOffset { objective, value }` | `Program: the offset {value} of {objective:?} is not finite` |
| `NonFiniteObjectiveCoefficient { objective, col, value }` | `Program: the coefficient {value} of {col:?} in {objective:?} is not finite` |
| `UnknownObjectiveColumn { objective, col }` | `Program: {objective:?} names {col:?}, which the plan does not have` |
| `QuadraticWithObjectives` | `Program: the objective has a quadratic part beside further objectives` |

```rust
use culmen::{Col, ProgramColumn, ProgramError, ProgramPlan, Row, bounds};

let mut plan = ProgramPlan::new();
let x = plan.col(ProgramColumn::binary(1.0));
plan.row(bounds!(0.0, 1.0), [(x, 1.0), (Col::new(7), 1.0)]);
assert_eq!(plan.build(), Err(ProgramError::UnknownColumn { row: Row::new(0), col: Col::new(7) }));
```

A model reports the same defects as a [`ModelError`](modelling.md#building-and-modelerror), in the model's own names.
With the `serde` feature a `Program` is read back through its plan and checked the same way; see
[Serialization](testing.md#serialization).

## See also

- [Modelling](modelling.md): the usual way to build a `Program`.
- [Solving](solving.md): loading a `Program` into a backend.
- [Design](design.md#the-program-is-checked-data): why the program is a checked value.
- [Documentation index](README.md).
