# Modelling

A `Model` holds variables, linear constraints and a linear or convex quadratic objective written with handles and
expressions, and builds them into a checked [`Program`](program.md#program). This page covers every part of a model with
the plain methods and operators; [Macros](macros.md) writes the same calls more briefly.

## Models and handles

`Model::new()` is an empty model that minimizes zero. Adding a variable returns its column, a
[`Col`](program.md#col-and-row); adding a constraint returns its row, a [`Row`](program.md#col-and-row). Both are `Copy`
handles, and they are the columns and rows of the program the model builds, so a solution is read with the handle the
model returned.

```rust
use culmen::prelude::*;

fn main() -> Result<(), culmen::ModelError> {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(4.0).name("x"));
    let ys = m.add_vars(3, |i| Var::binary().name(format!("y[{i}]")));
    let r = m.add_named_constraint("link", (x - 4.0 * ys[0]).leq(0.0));
    m.maximize(x + ys[1]);

    assert_eq!((m.num_vars(), m.num_constraints()), (4, 1));
    let program = m.build()?;
    assert_eq!(program.row(r).name(), Some("link"));
    assert_eq!(program.columns()[x].bounds, bounds!(0.0, 4.0));
    Ok(())
}
```

`add_vars(count, f)` adds `count` variables, the `i`-th made by `f(i)`, and returns their columns in order.

## Variables

A `Var` describes one variable: its integrality, bounds and name. It is built by chaining and checked only when the
model is built.

| Constructor | Integrality | Bounds |
|---|---|---|
| `Var::continuous()` | continuous | `[0, +inf)` |
| `Var::integer()` | integer | `[0, +inf)` |
| `Var::binary()` | integer | `[0, 1]` |
| `Var::free()` | continuous | `(-inf, +inf)` |
| `Var::semi_continuous()` | zero, or any value in its bounds | none until set: `0 < lower <= upper < +inf` |
| `Var::semi_integer()` | zero, or a whole value in its bounds | none until set: `0 < lower <= upper < +inf` |

| Method | Effect |
|---|---|
| `.lower(l)` | Sets the lower bound; `f64::NEG_INFINITY` for none. |
| `.upper(u)` | Sets the upper bound; `f64::INFINITY` for none. |
| `.bounds(l, u)` | Sets both. |
| `.name(n)` | Names the variable in errors, logs and LP files. |

Bounds take anything that converts into `f64` losslessly, so `Var::integer().bounds(0, 40)` needs no `.0`.
A [`Bounds`](program.md#bounds) value converts into a continuous `Var` in those bounds, so `add_var` takes it directly:

```rust
use culmen::prelude::*;

let mut m = Model::new();
let stock = m.add_var(Var::integer().bounds(0, 40).name("stock"));
let flow = m.add_var(bounds!(0.0, 12.5));
m.add_var(Var::binary());
m.add_var(Var::free());
assert_eq!(m.var_name(stock), Some("stock"));
assert_eq!(m.var_name(flow), None);
```

A variable whose bounds are invalid (a NaN, a lower bound above the upper, a lower bound of `+inf`) is accepted by
`add_var` and refused by `build`, which names it; see [Building and `ModelError`](#building-and-modelerror).

### Semi-continuous and semi-integer variables

A semi variable is zero or in its bounds: a machine that is off, or runs between its minimum and maximum load. It
needs both bounds, with zero outside them: `build` refuses a semi variable whose bounds are not
`lower ..= upper` with `0 < lower` and `upper` finite, with `ModelError::SemiBounds`. A semi-integer variable is
also whole when it is not zero. A model with one is mixed-integer: its program is a MILP (a MIQP with a quadratic
objective), and it uses the [feature](program.md#features) `SemiContinuous` or `SemiInteger`.

```rust
use culmen::prelude::*;
use culmen::{Integrality, ModelClass, ModelError};

fn main() -> Result<(), ModelError> {
    let mut m = Model::new();
    // Off, or between 20 and 80 units.
    let load = m.add_var(Var::semi_continuous().bounds(20, 80).name("load"));
    // Off, or 2 to 6 whole trucks.
    let trucks = m.add_var(Var::semi_integer().bounds(2, 6).name("trucks"));
    m.add_constraint((load + 10 * trucks).geq(15));
    m.minimize(2 * load + 5 * trucks);
    let program = m.build()?;
    assert_eq!(program.class(), ModelClass::Milp);
    assert_eq!(program.columns()[load].integrality, Integrality::SemiContinuous);

    m.add_var(Var::semi_continuous().upper(9).name("unbounded below"));
    assert!(matches!(m.build(), Err(ModelError::SemiBounds { .. })));
    let _ = trucks;
    Ok(())
}
```

## Expressions

Three types make up an expression:

- `Col`, a variable.
- `Term`, a column scaled by a coefficient: `Term { col, coefficient }`. Scaling a column makes one without
  allocating.
- `LinExpr`, the expression `Σ coefficient · col + constant`: plain data with `pub terms: Vec<Term>` and
  `pub constant: f64`. Terms may repeat a column; they are added up when the model is built.

The operators combine them:

| Expression | Gives |
|---|---|
| `2.0 * x`, `x * 2.0`, `3 * x`, `x * 3`, `x / 2.0`, `-x` (also on a `Term`) | `Term` |
| `a + b`, `a - b` with `a` a `Col`, `Term` or `LinExpr` and `b` any of those or a number | `LinExpr` |
| `5.0 + x`, `5.0 - x`, `1 - x` (a number on the left) | `LinExpr` |
| `e * 2.0`, `2.0 * e`, `e * 2`, `2 * e`, `e / 2.0`, `-e` with `e` a `LinExpr` | `LinExpr` |
| `e += b`, `e -= b` with `b` any linear item; `e *= 2.0` | in place |

Numbers are `f64` or `i32`. To sum many items, use `LinExpr::sum` or `Iterator::sum`; both take any iterator of
linear items and build a single expression:

```rust
use culmen::prelude::*;

let mut m = Model::new();
let xs = m.add_vars(3, |_| Var::continuous());
let weights = [4.0, 5.0, 6.0];

let load = LinExpr::sum(xs.iter().zip(weights).map(|(&x, w)| w * x));
let same: LinExpr = xs.iter().zip(weights).map(|(&x, w)| w * x).sum();
assert_eq!(load, same);

let mut e = 2.0 * xs[0] + xs[1] - 3;
e += &load; // a borrowed expression adds without being consumed
e.add_term(xs[2], 0.5);
assert_eq!(e.constant, -3.0);
assert_eq!(e.terms.len(), 6);

let point: culmen::TiVec<Col, f64> = vec![1.0, 2.0, 4.0].into();
assert_eq!(load.value(&point), 4.0 + 10.0 + 24.0);
```

`LinExpr::new()` is the expression zero. `From` converts a `Col`, a `Term` or an `f64` into a `LinExpr`, and a `Col`
into a `Term`. `LinExpr::value(&values)` evaluates the expression at a point, such as a solution's values; it panics
when a term's column has no value there.

The trait behind all of this is `Linear`: anything that adds to an expression. `Col`, `Term`, `LinExpr`, `&LinExpr`,
`f64` and `i32` implement it, and every operator, sum and comparison takes `impl Linear`.

## Quadratic expressions

A product of two linear items is quadratic, for an objective:

- `QuadTerm`, a product of two columns scaled by a coefficient: `QuadTerm { first, second, coefficient }`, meaning
  `coefficient · first · second`. `x * y`, `x * x` and `2.0 * x * y` make one without allocating.
- `QuadExpr`, the expression `Σ coefficient · first · second + linear`: plain data with `pub terms: Vec<QuadTerm>`
  and `pub linear: LinExpr`. Pairs may repeat, in either order; `x * y` and `y * x` add up when the model is built.

| Expression | Gives |
|---|---|
| `x * y`, `x * (2.0 * y)`, `(2.0 * x) * y`, `2.0 * x * y` with `x`, `y` a `Col` or `Term` | `QuadTerm` |
| `q * 2.0`, `2.0 * q`, `q * 2`, `2 * q`, `q / 2.0`, `-q` with `q` a `QuadTerm` | `QuadTerm` |
| `a * b` with one side a `LinExpr` and the other a `Col`, `Term` or `LinExpr`, as in `(x - 1) * (x - 1)` | `QuadExpr`, expanded |
| `a + b`, `a - b` with either side a `QuadTerm` or `QuadExpr` and the other any linear or quadratic item | `QuadExpr` |
| `e * 2.0`, `2.0 * e`, `e * 2`, `2 * e`, `e / 2.0`, `-e` with `e` a `QuadExpr` | `QuadExpr` |
| `e += b`, `e -= b` with `b` any linear or quadratic item; `e *= 2.0` | in place |

`QuadExpr::sum`, `Iterator::sum` and `collect` add up quadratic items and `extend` adds them to an expression,
`QuadExpr::add_term(first, second, coefficient)` adds one product, `QuadExpr::value(&values)` evaluates the expression at a point, and `From` converts a `QuadTerm`, a
`LinExpr`, a `Col`, a `Term` or an `f64` into a `QuadExpr`. The trait behind them is `Quadratic`, which every
`Linear` item, `QuadTerm`, `QuadExpr` and `&QuadExpr` implement; the objective setters take `impl Quadratic`.

```rust
use culmen::prelude::*;
use culmen::QuadExpr;

let mut m = Model::new();
let x = m.add_var(Var::free());
let y = m.add_var(Var::free());

let e = x * x + 2.0 * x * y + y * y - 4.0 * x + 3.0;
assert_eq!((e.terms.len(), e.linear.terms.len(), e.linear.constant), (3, 1, 3.0));

let distance = (x - 1) * (x - 1) + (y - 2) * (y - 2);
let point: culmen::TiVec<Col, f64> = vec![4.0, 6.0].into();
assert_eq!(distance.value(&point), 9.0 + 16.0);

let squares: QuadExpr = [x, y].into_iter().map(|v| v * v).sum();
assert_eq!(squares.terms.len(), 2);
```

Constraints stay linear: `leq`, `geq`, `equals` and `between` are methods of `Linear`, which no quadratic item
implements, so a quadratic constraint does not compile.

## Comparisons

`Linear` also turns an expression into a `Constraint`:

| Method | Constraint |
|---|---|
| `a.leq(b)` | `a <= b` |
| `a.geq(b)` | `a >= b` |
| `a.equals(b)` | `a == b` |
| `e.between(lo, hi)` | `lo <= e <= hi` |

`a` and `b` are any linear items, so both sides may hold variables. A `Constraint` is plain data:
`Constraint { expr, lower, upper }` with `lower <= expr <= upper`. `leq`, `geq` and `equals` store `a - b` as the
expression; when the model is built, the expression's constant moves into the bounds, so `x + 1 <= 3` and `x <= 2`
build the same row.

```rust
use culmen::prelude::*;

let mut m = Model::new();
let x = m.add_var(Var::continuous());
let y = m.add_var(Var::continuous());

let c = (x + 2.0 * y).leq(10.0);
assert_eq!((c.lower, c.upper), (f64::NEG_INFINITY, 0.0));
assert_eq!(c.expr.constant, -10.0);

let balance = x.equals(y);
assert_eq!((balance.lower, balance.upper), (0.0, 0.0));
let band = (x - y).between(-1, 1);
assert_eq!((band.lower, band.upper), (-1.0, 1.0));
```

## Constraints

| Method | Adds |
|---|---|
| `m.add_constraint(c)` | An unnamed constraint; returns its `Row`. |
| `m.add_named_constraint(name, c)` | A named constraint; returns its `Row`. |
| `m.add_constraint_map(keys, f)` | One named constraint per key; see [Keyed families](#keyed-families). |

## The objective

`m.set_objective(sense, expr)` sets the objective, replacing any before. `m.minimize(expr)` and `m.maximize(expr)`
are the same with the sense written in the name. The expression is linear or quadratic. Its constant becomes the
program's [offset](program.md#program), which every reported objective value and bound includes. A column that appears
several times in the objective has its coefficients added up, and so has a pair of columns in a product.

```rust
use culmen::prelude::*;

fn main() -> Result<(), culmen::ModelError> {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous());
    m.set_objective(Sense::Maximize, 3.0 * x + 1.5 + x);
    let program = m.build()?;
    assert_eq!((program.sense(), program.offset(), program.columns()[x].cost), (Sense::Maximize, 1.5, 4.0));
    Ok(())
}
```

A quadratic objective must be convex to minimize and concave to maximize: `build` checks it, block by block of
variables the products join, and refuses any other with `ModelError::NonConvexObjective`, which names the first
variable of the first block at fault. The check factors each block's Hessian with a small tolerance, as
[the program's convexity check](program.md#the-quadratic-part) states, so a semidefinite objective such as `(x + y)²`
passes.

```rust
use culmen::prelude::*;
use culmen::{ModelClass, ModelError};

fn main() -> Result<(), ModelError> {
    let mut m = Model::new();
    let x = m.add_var(Var::free().name("x"));
    let y = m.add_var(Var::free().name("y"));
    m.minimize(x * x + 2.0 * x * y + y * y - 4.0 * x);
    let program = m.build()?;
    assert_eq!(program.class(), ModelClass::Qp);
    assert_eq!(program.quadratic().entries().collect::<Vec<_>>(), [(x, x, 1.0), (x, y, 2.0), (y, y, 1.0)]);

    m.maximize(5.0 - (x - 3) * (x - 3) - y * y);
    assert!(m.build().is_ok(), "a concave objective is maximized");

    m.minimize(x * y);
    let error = m.build().unwrap_err();
    assert_eq!(
        error.to_string(),
        "Model: the objective is not convex, as minimizing needs, in the block of variable `x`"
    );
    Ok(())
}
```

## Several objectives

`m.add_objective(objective)` adds a further linear objective and returns its `ObjectiveId`; the model's own
objective, set by `minimize` and its kin, is `ObjectiveId::FIRST`, and further ones follow in the order added. An
`Objective` is built like a `Var`: `Objective::minimize(expr)` or `Objective::maximize(expr)`, then `.tolerance(..)`
and `.weight(..)`. How the objectives combine is `m.set_combination(..)`:

- `Combination::Lexicographic` (the default): the model's own objective first, then each further one in order,
  each over the points that keep every earlier objective within its `Tolerance` of the optimum its stage found. The
  order is the priority. A `Tolerance` allows `min(absolute, relative · |f*|)` from an optimum `f*`, leaving out an
  absent part; `Tolerance::EXACT`, the default, allows nothing. `m.set_objective_tolerance(..)` sets the own
  objective's.
- `Combination::Blended`: one objective, `Σ s · w · f` over every objective `f` with its `Weight` `w` and `s` one to
  minimize and minus one to maximize, is minimized. `Weight::ONE` is the default; `m.set_objective_weight(..)` sets
  the own objective's.

The objectives are linear: beside a further objective, `build` refuses a quadratic term in the own objective with
`ModelError::QuadraticWithObjectives`. The program records them as its [`Objectives`](program.md#several-objectives),
and a solution reports the own objective in `Solution::objective`; `program.objective_values(&values)` gives every
objective's value, indexed by `ObjectiveId`.

```rust
use culmen::prelude::*;
use culmen::{Combination, ObjectiveId};

fn main() -> Result<(), culmen::ModelError> {
    // Serve at least 10 units at the least cost; among the cheapest plans, the least emissions.
    let mut m = Model::new();
    let coal = m.add_var(Var::continuous().name("coal"));
    let gas = m.add_var(Var::continuous().upper(6).name("gas"));
    m.add_constraint((coal + gas).geq(10));
    m.minimize(2 * coal + 2 * gas);
    let co2 = m.add_objective(Objective::minimize(3 * coal + gas));
    let program = m.build()?;
    let values = program.objective_values(culmen::TiSlice::from_ref(&[4.0, 6.0]));
    assert_eq!((values[ObjectiveId::FIRST], values[co2]), (20.0, 18.0));

    // Allow the cost to rise by up to 1 for the emissions' sake, or blend the two at a price of 0.5 per unit.
    m.set_objective_tolerance(Tolerance { absolute: Some(absolute_gap!(1.0)), relative: None });
    m.set_combination(Combination::Blended);
    m.set_objective_weight(weight!(2.0));
    assert!(m.build().is_ok());
    Ok(())
}
```

## Keyed families

Most models index variables and constraints by domain keys: products, locations, pairs of them. A family keeps the
handle of each key.

- `m.add_var_map(keys, |key| var)` adds one variable per key and returns a `VarMap<K>`.
- `m.add_constraint_map(keys, |key| (name, constraint))` adds one named constraint per key and returns a
  `ConstraintMap<K>`.

`keys` is any iterator: filter it to leave keys out, and use `product(&a, &b)` for every pair of two
[index sets](#index-sets), the first outermost. A key may be any `Hash + Eq + Clone` type: a number, a `&str`, a
tuple. Both methods panic when the keys repeat one.

```rust
use culmen::prelude::*;

fn main() -> Result<(), culmen::ModelError> {
    // Assign each product to one location within capacity, at the least walking cost; product 2 does not fit
    // location 0.
    let (products, locations) = (0..3, 0..2);
    let volume = [4.0, 3.0, 5.0];
    let capacity = [8.0, 6.0];
    let walk = [[1.0, 3.0], [2.0, 1.0], [4.0, 2.0]];
    let fits = |p: usize, l: usize| p != 2 || l != 0;

    let mut m = Model::new();
    let x = m.add_var_map(product(&products, &locations).filter(|&(p, l)| fits(p, l)), |(p, l)| {
        Var::binary().name(format!("x[{p},{l}]"))
    });
    let once = m.add_constraint_map(products.clone(), |&p| {
        let placed = LinExpr::sum(locations.clone().filter_map(|l| x.get(&(p, l))));
        (format!("once[{p}]"), placed.equals(1))
    });
    m.add_constraint_map(locations.clone(), |&l| {
        let load = products.clone().filter_map(|p| Some(volume[p] * x.get(&(p, l))?));
        (format!("cap[{l}]"), LinExpr::sum(load).leq(capacity[l]))
    });
    m.minimize(LinExpr::sum(x.iter().map(|(&(p, l), x)| walk[p][l] * x)));

    assert_eq!(x.len(), 5);
    assert_eq!(x.get(&(2, 0)), None);
    assert_eq!(m.var_name(x[(1, 0)]), Some("x[1,0]"));
    assert_eq!(m.constraint_name(once[2]), Some("once[2]"));
    let program = m.build()?;
    assert_eq!((program.num_cols(), program.num_rows()), (5, 5));
    Ok(())
}
```

`VarMap<K>` and `ConstraintMap<K>` are `Keyed<K, Col>` and `Keyed<K, Row>`. A family keeps its keys in insertion
order and reads them back:

| Method | Gives |
|---|---|
| `family[key]` | The handle; panics when the family does not hold the key. |
| `family.get(&key)` | The handle, `None` when the family does not hold the key. |
| `family.len()`, `family.is_empty()` | The number of keys. |
| `family.iter()`, `&family` in a `for` loop | `(&key, handle)` pairs. |
| `family.keys()`, `family.handles()` | The keys, the handles. |
| `vars.values(&solution.values)` | `(&key, value)` pairs at a point; `VarMap` only. |

The iterators run both ways and know their exact length. `values` panics when a variable of the family has no value
in the point. `Keyed::new()` and `Keyed::insert(key, handle)` build a family by hand; `insert` panics on a key
already held.

## Index sets

An `IndexSet` is a set of indices that a family or a [macro loop](macros.md#domains) ranges over, iterable as often
as a nested loop needs. It yields its members by value, cloned out of a collection, so ranging over a `Vec<usize>`
gives `usize` and leaves the vector usable.

`IndexSet` is implemented for `Range`, `RangeInclusive`, slices, arrays, `Vec`, `VecDeque`, `BTreeSet`, and a
reference to any of them. Any cloneable iterator becomes one through `set`, which restarts it from a clone on every
pass:

```rust
use culmen::prelude::*;

let evens = set((0..10).filter(|i| i % 2 == 0));
assert_eq!(evens.members().count(), 5);
assert_eq!(evens.members().count(), 5);

let pairs: Vec<_> = product(&(0..2), &["north", "south"]).collect();
assert_eq!(pairs, [(0, "north"), (0, "south"), (1, "north"), (1, "south")]);
```

`product` pairs two sets. For three or more indices, give `add_var_map` any iterator of tuples, such as a
`flat_map` over the sets, or write the family with the [`variable!`](macros.md#variable) macro, which takes any
number of indices.

## Names

A name is optional on every variable and constraint. It appears in [`ModelError`](#building-and-modelerror)
messages, in solver logs, and in [LP files](program.md#lp-files); an unnamed variable is shown by its column, `Col(3)`,
and an unnamed constraint by its row. `m.var_name(col)` and `m.constraint_name(row)` read a name back, `None` when
there is none or the model has no such handle; `program.columns()[col].name` and `program.row(row).name()` read it from
the built program. Names need not be unique.

## Building and `ModelError`

`m.build()` checks every number and handle and returns the [`Program`](program.md#program), or a `ModelError` naming the
first variable or constraint at fault. The model is left as it was, so it can be changed and built again. Every model
`build` accepts gives a valid program: the checks are the program's own, made in the model's terms.

| Variant | When |
|---|---|
| `VarBounds { var, source }` | A variable's bounds are invalid; `source` is the [`InvalidBounds`](program.md#bounds). |
| `ObjectiveCoefficient { var, value }` | A variable's objective coefficient is NaN or infinite, as given or added up. |
| `ObjectiveConstant { value }` | The objective's constant is NaN or infinite. |
| `Coefficient { constraint, var, value }` | A constraint's coefficient is NaN or infinite, as given or added up. |
| `ConstraintBounds { constraint, source }` | A constraint's bounds are invalid after its constant moved into them. |
| `UnknownVar { place, col }` | An expression names a column this model did not hand out, in the objective or a constraint. |
| `ObjectiveQuadCoefficient { first, second, value }` | A quadratic objective coefficient is NaN or infinite, as given or added up over its pair. |
| `NonConvexObjective { sense, var }` | The quadratic objective is not convex to minimize, or not concave to maximize. |
| `SemiBounds { var, bounds }` | A semi variable's bounds are not `lower ..= upper` with `0 < lower` and `upper` finite. |
| `FurtherCoefficient { objective, var, value }` | A further objective's coefficient is NaN or infinite, as given or added up. |
| `FurtherConstant { objective, value }` | A further objective's constant is NaN or infinite. |
| `QuadraticWithObjectives` | The own objective has a quadratic term beside further objectives. |

`var` is a `ModelVar { col, name }` and `constraint` a `ModelConstraint { row, name }`; each displays as its name
in backticks, or as its handle when it has none. `place` is an `ExprPlace`: `Objective`, `Constraint(..)` or `Further(ObjectiveId)`.

```rust
use culmen::prelude::*;
use culmen::ModelError;

let mut m = Model::new();
m.add_var(Var::integer().bounds(5, 2).name("stock"));
let error = m.build().unwrap_err();
assert!(matches!(error, ModelError::VarBounds { .. }));
assert_eq!(
    error.to_string(),
    "Model: variable `stock` has invalid bounds: Bounds: the lower end 5 exceeds the upper end 2"
);

let mut m = Model::new();
let x = m.add_var(Var::continuous().name("x"));
m.add_named_constraint("big", (1e308 * x + 1e308 * x).leq(1.0));
let error = m.build().unwrap_err();
assert_eq!(error.to_string(), "Model: constraint `big` has the non-finite coefficient inf on variable `x`");

// A handle from another model.
let mut other = Model::new();
other.add_var(Var::continuous());
let foreign = other.add_var(Var::continuous());
let mut m = Model::new();
m.minimize(foreign);
let error = m.build().unwrap_err();
assert_eq!(error.to_string(), "Model: the objective names Col(1), which this model did not hand out");
```

`ModelError::source()` returns the `InvalidBounds` behind `VarBounds` and `ConstraintBounds`.

## See also

- [Macros](macros.md): the same model with `variable!`, `constraint!`, `objective!` and `sum!`.
- [The program as data](program.md): what `build` produces.
- [Solving](solving.md#reading-a-solution): reading a solution with the model's handles.
- [Documentation index](README.md).
