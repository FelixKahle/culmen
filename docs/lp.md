# The LP as data

An `Lp` is a linear, mixed-integer or convex quadratic program: minimize or maximize
`cost · x + Σ q · xᵢ · xⱼ + offset` over `x`, with every column in its bounds, the integer columns whole, and every
row's activity `Σ coefficient · column` in its bounds. Its constraints are linear. It is an
immutable value, built once and checked then, and it is what a [backend](solving.md#solver) loads. A
[model](modelling.md) builds one; this page builds them directly and reads them back.

## `LpPlan`

An `LpPlan` is an LP as plain data, with `pub` fields, checked into an `Lp` by `LpPlan::build`:

| Field | Holds |
|---|---|
| `sense` | `Sense::Minimize` (the default) or `Sense::Maximize`. |
| `offset` | The constant added to the objective. |
| `columns` | A `TiVec<Col, LpColumn>`: each column's `cost`, `bounds`, `integrality` and `name`. |
| `rows` | A `TiVec<Row, LpRow>`: each row's `bounds`, `terms` as `(Col, f64)`, and `name`. |
| `quadratic` | The objective's quadratic terms as `(Col, Col, f64)`, each `(i, j, q)` adding `q · xᵢ · xⱼ`; empty for a linear objective. See [The quadratic part](#the-quadratic-part). |

`plan.col(column)` pushes a column and returns its `Col`; `plan.row(bounds, terms)` pushes an unnamed row and
returns its `Row`. `LpColumn::continuous(cost, bounds)`, `LpColumn::integer(cost, bounds)` and
`LpColumn::binary(cost)` make columns without a name; the fields are public, so set `name` directly.

```rust
use culmen::{Bounds, LpColumn, LpPlan, Sense, bounds};

fn main() -> Result<(), culmen::LpError> {
    // Maximize 3x + 2y subject to x + y <= 4 and x + 3y <= 6, with x, y >= 0.
    let mut plan = LpPlan::new();
    plan.sense = Sense::Maximize;
    let x = plan.col(LpColumn { name: Some("x".into()), ..LpColumn::continuous(3.0, Bounds::NON_NEGATIVE) });
    let y = plan.col(LpColumn::continuous(2.0, Bounds::NON_NEGATIVE));
    let total = plan.row(bounds!(f64::NEG_INFINITY, 4.0), [(x, 1.0), (y, 1.0)]);
    plan.row(bounds!(f64::NEG_INFINITY, 6.0), [(x, 1.0), (y, 3.0), (x, 0.5), (x, -0.5)]);
    plan.rows[total].name = Some("total".into());

    let lp = plan.build()?;
    assert_eq!((lp.num_cols(), lp.num_rows(), lp.num_entries()), (2, 2, 4));
    assert_eq!(lp.row(total).name(), Some("total"));
    Ok(())
}
```

A row's terms may repeat a column; their coefficients are added up in the order given, and an entry that sums to
zero is dropped. Every row of an `Lp` holds each column at most once, with a non-zero coefficient, sorted by column.

`Lp` and `LpPlan` convert both ways: `LpPlan::from(lp)` takes an LP apart into its plan, and `Lp::try_from(plan)` is
`plan.build()`.

## `Lp`

An `Lp` is read through methods; it has no public fields and no constructor that skips the checks.

| Method | Gives |
|---|---|
| `sense()`, `offset()` | The objective's sense and constant. |
| `num_cols()`, `num_rows()`, `num_entries()` | The counts; entries are the non-zero coefficients. |
| `is_mip()` | Whether any column is integer. |
| `quadratic()` | The objective's [quadratic part](#the-quadratic-part), a `QuadObjective`. |
| `class()` | The [`ModelClass`](#model-classes): `Lp`, `Milp`, `Qp` or `Miqp`. |
| `cols()`, `rows()` | Every `Col`, every `Row`, in order. |
| `columns()` | The columns' `LpColumn` data, indexed by `Col`: `lp.columns()[x]`. |
| `row_bounds()` | The rows' bounds, indexed by `Row`. |
| `row(row)` | A [`RowView`](#row-views-and-iterators) of one row: its bounds, name and entries. |
| `matrix()` | The whole matrix in compressed row form, a `RowMatrix { starts, cols, values }`. |
| `objective_value(values)`, `max_violation(values)` | See [Checking a point](#checking-a-point). |
| `to_lp_string()`, `write_lp(out)` | See [LP files](#lp-files). |

The methods that take a handle panic when the LP has no such column or row; they never read anything undefined.

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

`Integrality` is `Continuous` (the default) or `Integer`. A binary column is an integer column with
`Bounds::BINARY`; the LP stores no third kind, since the bounds already say it.

`Sense` is `Minimize` (the default) or `Maximize`. `sense.better(a, b)` says whether objective value `a` is strictly
better than `b` under it.

## `Col` and `Row`

`Col` and `Row` are typed, `Copy` handles: the position of a column or a row, counting from zero. Because they are
distinct types, a table indexed by columns cannot be read with a row. `Col::new(i)` and `Row::new(i)` make a handle
from a position, `index()` reads it back, and both convert to and from `usize`.

Handles come from `Model::add_var`, `Model::add_constraint`, `LpPlan::col` and `LpPlan::row`; write `Col::new`
yourself only where the position itself is the point. A handle from one LP used on another reads that LP's column or
row of the same position; a read that cannot answer for the position panics or returns `None`, as its docs say.

Per-column data is a `TiVec<Col, f64>` or a borrowed `TiSlice<Col, f64>`, re-exported as `culmen::TiVec` and
`culmen::TiSlice` from `typed-index-collections`. They index by `Col` only: `values[x]`.

## Row views and iterators

`lp.row(r)` is a `RowView`: the row's `bounds()`, `name()`, `len()`, `is_empty()`, its `entries()` as
`(Col, f64)` sorted by column, and `activity(values)`, the row's `Σ coefficient · value` at a point. A `RowView` is
also `IntoIterator` over its entries.

```rust
use culmen::{LpColumn, LpPlan, TiVec, bounds};

fn main() -> Result<(), culmen::LpError> {
    let mut plan = LpPlan::new();
    let x = plan.col(LpColumn::binary(1.0));
    let y = plan.col(LpColumn::binary(1.0));
    let r = plan.row(bounds!(0.0, 1.0), [(y, 2.0), (x, 1.0)]);
    let lp = plan.build()?;

    let row = lp.row(r);
    assert_eq!((row.bounds(), row.name(), row.len()), (bounds!(0.0, 1.0), None, 2));
    assert_eq!(row.into_iter().collect::<Vec<_>>(), [(x, 1.0), (y, 2.0)]);
    assert_eq!(lp.row(r).entries().rev().next(), Some((y, 2.0)));

    let point: TiVec<culmen::Col, f64> = vec![1.0, 1.0].into();
    assert_eq!(row.activity(&point), 3.0);
    assert_eq!(lp.cols().collect::<Vec<_>>(), [x, y]);
    assert_eq!(lp.rows().len(), 1);
    Ok(())
}
```

The iterators (`Cols`, `Rows`, `RowEntries`, and the family iterators of [Modelling](modelling.md#keyed-families))
are named types that run both ways and know their exact length.

## The quadratic part

`lp.quadratic()` is a `QuadObjective`, the objective's `Σ q · xᵢ · xⱼ` in one canonical form: entries `(i, j, q)`
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
use culmen::{Bounds, LpColumn, LpPlan, TiSlice};

fn main() -> Result<(), culmen::LpError> {
    // x² + 2xy + 3y², written with the cross term split in both orders.
    let mut plan = LpPlan::new();
    let x = plan.col(LpColumn::continuous(0.0, Bounds::FREE));
    let y = plan.col(LpColumn::continuous(0.0, Bounds::FREE));
    plan.quadratic = vec![(y, x, 1.0), (x, x, 1.0), (x, y, 1.0), (y, y, 3.0)];
    let lp = plan.build()?;

    let q = lp.quadratic();
    assert_eq!(q.entries().collect::<Vec<_>>(), [(x, x, 1.0), (x, y, 2.0), (y, y, 3.0)]);
    assert_eq!(q.hessian().collect::<Vec<_>>(), [(x, x, 2.0), (x, y, 2.0), (y, y, 6.0)]);
    assert_eq!(lp.objective_value(TiSlice::from_ref(&[1.0, 1.0])), 6.0);
    Ok(())
}
```

The class of program is *convex* quadratic. Convexity is a property of real numbers that floating point cannot
decide exactly, so `build` checks it with a stated tolerance: `H` splits into blocks of columns its entries join,
and for every block `s · H + δ · I` must have a Cholesky factor, where `s` is `1` to minimize and `-1` to maximize and
`δ` is `Lp::CONVEXITY_TOLERANCE` (`1e-9`) times the block's largest `|Hᵢⱼ|`. Every `Lp` therefore has an objective
convex to minimize, or concave to maximize, up to that tolerance; a semidefinite one passes. Each block costs a dense
factorization cubic in its number of columns, so a separable objective, a sum of squares of single columns, costs
next to nothing.

## Model classes

`lp.class()` is a `ModelClass`, the kind of program the LP is: whether it has an integer column, and whether its
objective has a quadratic part.

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

## Checking a point

`lp.objective_value(values)` is the objective at a point, quadratic part and offset included.
`lp.max_violation(values)` is the largest amount by which the point violates a column bound, a row, or integrality,
and zero for a feasible point. Both panic when `values` holds fewer values than the LP has columns. They are the way
to check a solution against the LP independently of any solver:

```rust
use culmen::prelude::*;

fn main() -> Result<(), culmen::ModelError> {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(0, 3));
    let y = m.add_var(Var::continuous());
    m.add_constraint((x + y).leq(4.0));
    m.maximize(2.0 * x + y + 1.0);
    let lp = m.build()?;

    let good: culmen::TiVec<Col, f64> = vec![3.0, 1.0].into();
    assert_eq!((lp.objective_value(&good), lp.max_violation(&good)), (8.0, 0.0));
    let bad: culmen::TiVec<Col, f64> = vec![2.5, 2.0].into();
    assert_eq!(lp.max_violation(&bad), 0.5);
    Ok(())
}
```

## LP files

`lp.to_lp_string()` writes the LP in the CPLEX LP file format, and `lp.write_lp(&mut out)` writes it into any
`fmt::Write`. The output is for reading by eye or by another solver while debugging:

- Names are reduced to the characters the format allows. An unnamed column is written `x<index>` and an unnamed row
  `r<index>`; distinct names that reduce to the same text are not told apart.
- A ranged row becomes two rows suffixed `_lo` and `_hi`; a row with no finite bound is left out.
- Integer columns are listed under `General`; bounds equal to the format's default `0 ..` are not written.
- A quadratic part is written doubled in the format's brackets halved: `x² + 2xy` as `[ 2 x ^ 2 + 2 x * y ] / 2`.
  HiGHS reads it back as the same objective.

```rust
use culmen::prelude::*;

fn main() -> Result<(), culmen::ModelError> {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(0, 5).name("x"));
    let y = m.add_var(Var::free().name("y"));
    m.add_named_constraint("band", (x - y).between(-1, 1));
    m.maximize(3.0 * x + y);
    let text = m.build()?.to_lp_string();
    assert_eq!(
        text,
        "Maximize\n obj: 3 x + 1 y\nSubject To\n band_lo: 1 x - 1 y >= -1\n band_hi: 1 x - 1 y <= 1\n\
         Bounds\n 0 <= x <= 5\n y free\nGeneral\n x\nEnd\n"
    );

    m.maximize(3.0 * x - x * x - 0.5 * x * y - y * y);
    let text = m.build()?.to_lp_string();
    assert!(text.starts_with("Maximize\n obj: 3 x + [ -2 x ^ 2 - 1 x * y - 2 y ^ 2 ] / 2\n"));
    Ok(())
}
```

## Checks and `LpError`

`LpPlan::build` accepts a plan when its offset and costs are finite, every term names a column of the plan with a
finite coefficient, in every row the coefficients of each column, added up in the order given, are finite, every
quadratic term names columns of the plan with a finite coefficient, the coefficients of each pair, added up in the
order given, are finite, and the quadratic part passes the [convexity check](#the-quadratic-part).
Bounds and integrality need no check there: their types already guarantee them. Otherwise it returns an `LpError`
naming the first defect:

| Variant | Message |
|---|---|
| `NonFiniteOffset { value }` | `Lp: the objective offset {value} is not finite` |
| `NonFiniteCost { col, value }` | `Lp: the cost {value} of {col:?} is not finite` |
| `NonFiniteCoefficient { row, col, value }` | `Lp: the coefficient {value} of {col:?} in {row:?} is not finite` |
| `UnknownColumn { row, col }` | `Lp: {row:?} names {col:?}, which the plan does not have` |
| `UnknownQuadColumn { col }` | `Lp: a quadratic term names {col:?}, which the plan does not have` |
| `NonFiniteQuadCoefficient { first, second, value }` | `Lp: the quadratic coefficient {value} of {first:?} · {second:?} is not finite` |
| `NonConvexObjective { sense, col }` | `Lp: the quadratic objective is not convex in the block of {col:?}`, or `not concave` to maximize |

```rust
use culmen::{Col, LpColumn, LpError, LpPlan, Row, bounds};

let mut plan = LpPlan::new();
let x = plan.col(LpColumn::binary(1.0));
plan.row(bounds!(0.0, 1.0), [(x, 1.0), (Col::new(7), 1.0)]);
assert_eq!(plan.build(), Err(LpError::UnknownColumn { row: Row::new(0), col: Col::new(7) }));
```

A model reports the same defects as a [`ModelError`](modelling.md#building-and-modelerror), in the model's own names.
With the `serde` feature an `Lp` is read back through its plan and checked the same way; see
[Serialization](testing.md#serialization).

## See also

- [Modelling](modelling.md): the usual way to build an `Lp`.
- [Solving](solving.md): loading an `Lp` into a backend.
- [Design](design.md#the-lp-is-checked-data): why the LP is a checked value.
- [Documentation index](README.md).
