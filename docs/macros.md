# Macros

`variable!`, `constraint!`, `objective!` and `sum!` write a model the way the mathematics reads. Each expands to the
plain methods of [Modelling](modelling.md), so a model written with them and one written with the methods build the
same program, and the two styles mix freely in one model.

## Enabling the macros

Turn on the `macros` feature of `culmen`:

```toml
[dependencies]
culmen = { path = "../culmen/crates/culmen", features = ["macros"] }
```

The macros are then in `culmen::prelude` and at `culmen::variable` and so on. They work from a crate that depends on
`culmen` or on `culmen_core`, under any name the manifest gives the dependency.

```rust
use culmen::prelude::*;

fn main() -> Result<(), culmen::ModelError> {
    let (products, locations) = (0..4, 0..2);
    let volume = [4.0, 3.0, 5.0, 2.0];
    let cap = [8.0, 7.0];
    let walk = [[1.0, 3.0], [2.0, 1.0], [4.0, 2.0], [1.0, 1.0]];

    let mut m = Model::new();
    variable!(m, x[p in products, l in locations], Bin);
    constraint!(m, once[p in products], sum!(x[p, l] for l in locations) == 1);
    constraint!(m, capacity[l in locations], sum!(volume[p] * x[p, l] for p in products) <= cap[l]);
    objective!(m, Min, sum!(walk[p][l] * x[p, l] for p in products, l in locations));

    assert_eq!(m.var_name(x[(3, 1)]), Some("x[3,1]"));
    assert_eq!(m.constraint_name(capacity[1]), Some("capacity[1]"));
    let program = m.build()?;
    assert_eq!((program.num_cols(), program.num_rows()), (8, 6));
    Ok(())
}
```

## `variable!`

`variable!(model, spec)` or `variable!(model, spec, kind)` declares one variable or a family, binds it to a local of
the declared name, and names it after that name.

The spec is the declaration with its bounds:

| Spec | Bounds |
|---|---|
| `x` | the kind's default |
| `x >= lb` or `lb <= x` | lower bound `lb` |
| `x <= ub` or `ub >= x` | upper bound `ub` |
| `x == v` | fixed at `v` |
| `lb <= x <= ub` or `ub >= x >= lb` | both |

The declaration is a name, `x`, or a family, `x[i in I, j in J if cond]`. The kind is `Cont` (the default), `Int`,
`Bin` or `Free`, also spelled `Continuous`, `Integer` and `Binary`. An unwritten bound keeps the kind's default: zero
below for `Cont` and `Int`, nothing for `Free`. A `Bin` takes no bounds; a bounded integer is `Int` with the bounds.

```rust
use culmen::prelude::*;

let ub = [3.0, 4.0];
let mut m = Model::new();
variable!(m, a);                                // continuous, [0, +inf)
variable!(m, b >= -2.0);                        // continuous, [-2, +inf)
variable!(m, c <= 5, Int);                      // integer, [0, 5]
variable!(m, d == 2.5, Free);                   // fixed at 2.5
variable!(m, -1 <= e <= 1, Free);               // continuous, [-1, 1]
variable!(m, g, Binary);                        // binary
variable!(m, 0 <= h[i in 0..2] <= ub[i], Cont); // a family; each bound may use the index

assert_eq!(m.var_name(c), Some("c"));
assert_eq!(m.var_name(h[1]), Some("h[1]"));
assert_eq!(m.num_vars(), 8);
let _ = (a, b, d, e, g);
```

Each entry of a family is named `name[i,j]`, each index shown through its `Display`. A family's index takes any
number of loops, each `pattern in domain`, and an optional filter `if cond` after the last one; a pattern is a name or
a tuple of names, such as `(a, b) in arcs`. The keys of the family are the patterns' values as a tuple, `(i, j)`, or
the value alone for one loop.

## `constraint!`

`constraint!` writes a relation, `lhs <= rhs`, `lhs >= rhs`, `lhs == rhs`, or a range `lo <= e <= hi` (also
`hi >= e >= lo`). Both sides may hold variables. It takes four forms:

| Form | Does |
|---|---|
| `constraint!(relation)` | Makes the `Constraint` value. |
| `constraint!(model, relation)` | Adds it unnamed. |
| `constraint!(model, name, relation)` | Adds it named `"name"` and binds its `Row` to the local `name`. |
| `constraint!(model, name[i in I, …], relation)` | Adds a family, one named entry per key, bound as a `ConstraintMap`. |

```rust
use culmen::prelude::*;

let mut m = Model::new();
variable!(m, x);
variable!(m, y);

assert_eq!(constraint!(x + y <= 4), (x + y).leq(4));
assert_eq!(constraint!(2 * x == y - 1), (2 * x).equals(y - 1));
assert_eq!(constraint!(-1 <= x - y <= 1), (x - y).between(-1, 1));

constraint!(m, x + 2.0 * y >= 1.0);
constraint!(m, budget, x + y <= 10.0);
constraint!(m, pair[i in 0..3, j in 0..3 if i < j], (i as f64) * x + (j as f64) * y <= 6.0);
assert_eq!(m.constraint_name(budget), Some("budget"));
assert_eq!(pair.len(), 3);
assert_eq!(m.constraint_name(pair[(0, 2)]), Some("pair[0,2]"));
```

A named constraint's handle is bound but need not be read; the macro marks it used.

## `objective!`

`objective!(model, Min, expr)` or `objective!(model, Max, expr)`, also spelled `Minimize` and `Maximize`, sets the
objective, replacing any before. The expression may be [quadratic](modelling.md#quadratic-expressions), convex to
minimize or concave to maximize:

```rust
use culmen::prelude::*;

fn main() -> Result<(), culmen::ModelError> {
    // Least squares on a budget: track the targets, spending at most 4 in all.
    let (items, target) = (0..3, [1.0, 2.0, 3.0]);
    let mut m = Model::new();
    variable!(m, x[i in items]);
    constraint!(m, budget, sum!(x[i] for i in items) <= 4);
    objective!(m, Min, sum!((x[i] - target[i]) * (x[i] - target[i]) for i in items));
    assert_eq!(m.build()?.quadratic().len(), 3);
    Ok(())
}
```

## `sum!`

`sum!(body for i in I, j in J if cond)` is the sum of `body` over a loop nest, the first loop outermost, keeping the
combinations the filter accepts. A later domain may use an earlier index, as in `j in i + 1..n`. A body may be any
linear item, a variable, a term, an expression or a number, and the sum is then a `LinExpr`, which a constraint
takes; or a quadratic item, a `QuadTerm` or `QuadExpr`, and the sum is a `QuadExpr`, for an objective. The trait
`Summand` picks the sum's type from the body's.

```rust
use culmen::prelude::*;

let arcs = vec![(0_usize, 1_usize), (1, 2), (0, 2)];
let cost = [[0.0, 1.0, 4.0], [0.0, 0.0, 2.0], [0.0; 3]];
let mut m = Model::new();
variable!(m, flow[(a, b) in arcs]);

let total = sum!(cost[a][b] * flow[a, b] for (a, b) in arcs);
assert_eq!(total, LinExpr::sum(arcs.iter().map(|&(a, b)| cost[a][b] * flow[(a, b)])));

let from_zero = sum!(flow[a, b] for (a, b) in arcs if a == 0);
assert_eq!(from_zero, flow[(0, 1)] + flow[(0, 2)]);

let pairs = sum!(1 for i in 0..4, _j in i + 1..4);
assert_eq!(pairs.constant, 6.0);
assert_eq!(m.var_name(flow[(1, 2)]), Some("flow[1,2]"));

let squares: culmen::QuadExpr = sum!(flow[a, b] * flow[a, b] for (a, b) in arcs);
assert_eq!(squares.terms.len(), 3);
```

Inside `sum!` a pattern may be `_j` or `_`, since nothing is named after it; a family's index must name every key.

## Indexing inside a macro

Inside every macro, `x[i, j]` reads the family entry keyed `(i, j)`: the macro writes it as
`x[(i.clone(), j.clone())]`. The indices are cloned, so a `String` key stays usable after the lookup. A subscript
with one index, `x[i]` or `cap[l]`, is left as written; it indexes a family with single keys and an array alike.
A comma inside a subscript is a syntax error for every other type, so the rewrite changes nothing else.

```rust
use culmen::prelude::*;

let products = vec!["cork".to_owned(), "glass".to_owned()];
let sites = ["north", "south"];
let mut m = Model::new();
variable!(m, x[p in products, s in sites], Bin);
constraint!(m, once[p in products], sum!(x[p, s] for s in sites) == 1);
objective!(m, Max, sum!(x[p, s] for p in products, s in sites));

assert_eq!(m.var_name(x[("glass".to_owned(), "south")]), Some("x[glass,south]"));
assert_eq!(m.constraint_name(once["cork".to_owned()]), Some("once[cork]"));
```

Outside a macro, index a family with the tuple itself: `x[(3, 1)]`.

## Domains

Each domain is evaluated where its loop starts, as a `for` loop's is, and iterated as an
[`IndexSet`](modelling.md#index-sets), which yields members by value. `p in products` binds `p: usize` over a
`Vec<usize>`, a `&Vec<usize>`, an array or a range alike, and leaves `products` usable afterwards. Wrap any other
cloneable iterator with `set`:

```rust
use culmen::prelude::*;

let mut m = Model::new();
variable!(m, x[i in set((0..10).filter(|i| i % 3 == 0))], Int);
assert_eq!(x.keys().copied().collect::<Vec<_>>(), [0, 3, 6, 9]);
```

## Expansions

Each form beside the calls it writes:

| Macro form | Plain API |
|---|---|
| `variable!(m, x)` | `let x = m.add_var(Var::continuous().name("x"));` |
| `variable!(m, 0 <= y <= 9, Int)` | `let y = m.add_var(Var::integer().lower(0).upper(9).name("y"));` |
| `variable!(m, z == 2.5, Free)` | `let z = m.add_var(Var::free().bounds(2.5, 2.5).name("z"));` |
| `variable!(m, x[i in I, j in J if f(i, j)], Bin)` | `let x = m.add_var_map(keys, \|&(i, j)\| Var::binary().name(format!("x[{i},{j}]")));` with `keys` the filtered pairs |
| `constraint!(e <= f)` | `Linear::leq(e, f)`; also `geq`, `equals` |
| `constraint!(lo <= e <= hi)` | `Linear::between(e, lo, hi)` |
| `constraint!(m, e == f)` | `m.add_constraint(Linear::equals(e, f))` |
| `constraint!(m, c, e >= f)` | `let c = m.add_named_constraint("c", Linear::geq(e, f));` |
| `constraint!(m, c[i in I], e <= f)` | `let c = m.add_constraint_map(keys, \|&i\| (format!("c[{i}]"), Linear::leq(e, f)));` |
| `objective!(m, Min, e)` | `m.set_objective(Sense::Minimize, e)` |
| `sum!(e for i in I, j in J if g)` | a `LinExpr` (linear `e`) or `QuadExpr` (quadratic `e`), each `e` added with `Summand::add_into` in nested `for` loops |
| `x[i, j]` inside a macro | `x[(i.clone(), j.clone())]` |

A family's keys are collected into a `Vec` by the same loop nest before `add_var_map` or `add_constraint_map` runs,
so a filter sees every index bound.

## Compile errors

A malformed macro call is a compile error at the token at fault, with a message saying what the macro expected:

| Message | Cause |
|---|---|
| ``variable!(model, spec[, kind])`` | `variable!` with too few or too many arguments. |
| ``a kind: `Cont`, `Int`, `Bin` or `Free` `` | An unknown kind. |
| ``bounds read `x >= lb`, `x <= ub`, `x == v` or `lb <= x <= ub` `` | A spec whose relations do not read as one of the forms. |
| ``a binary's bounds are [0, 1]: for others, `Int` with the bounds`` | A `Bin` with bounds. |
| ``a name, such as `x` or `x[i in I]` `` | A declaration that does not start with a name. |
| ``an index reads `name in domain` `` | A loop without `in`, or with an empty side. |
| ``the filter `if cond` comes last, after every `name in domain` `` | A filter before the last loop. |
| ``an index is a name or a tuple of names, such as `i` or `(i, j)` `` | A family pattern with a path, a literal or a constructor. |
| ``a family's index names its key: with `_`, its entries' names would collide`` | `_` in a family's index. |
| ``an index binds its key by value: no `ref` or `mut` `` | `ref` or `mut` in a family's index. |
| ``a constraint reads `lhs <= rhs`, `lhs >= rhs`, `lhs == rhs` or `lo <= e <= hi` `` | No relation, or relations that do not read as one. |
| ``a sense: `Min` or `Max` `` | An unknown objective sense. |
| ``sum!(body for i in I, …)`` | `sum!` without `for`. |

```compile_fail
use culmen::prelude::*;

let mut m = Model::new();
variable!(m, 0 <= open <= 1, Bin); // a binary's bounds are [0, 1]
```

```compile_fail
use culmen::prelude::*;

let mut m = Model::new();
variable!(m, x[_ in 0..3]); // a family's index names its key
```

## See also

- [Modelling](modelling.md): the methods the macros expand to, and `ModelError`.
- [Getting started](getting-started.md#the-same-model-with-macros): a first model written with the macros.
- [Documentation index](README.md).
