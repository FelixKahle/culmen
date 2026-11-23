<!-- Draft, not part of the guide: the design of the pricing event, written 2026-11-23 and not yet reviewed or built. Remove or fold into docs/ before the SCIP pull request. -->

# Design: a PRICE event for branch-and-price in culmen

I changed nothing in the repository and compiled no probe. Every claim about SCIP below is marked in one of two ways:
- **[read: file]** means I read it in the SCIP 10.0.2 sources that scip-sys vendors, under `target/debug/build/scip-sys/*/out/scipoptsuite-10.0.2/scip/src/scip/` and `scip/doc/xternal.c`.
- **[assumed; probe Pn]** means it still has to be checked. The probes are listed in §8.5.

## 0. Recommendations at a glance

| # | Decision | Rejected alternative |
|---|---|---|
| 1 | **One event, `Events::PRICE`, with a mode.** It has one `Callback::price`, one `PriceContext` trait and one `Backend::Price<'a>` type. `PriceMode::ReducedCost { objective }` or `PriceMode::Farkas` tells the two cases apart. | Two events (PRICE and FARKAS): twice the types, and a user who forgets Farkas gets a wrong "infeasible" without any warning. |
| 2 | **Duals of program rows only**, as `&TiSlice<Row, f64>`, in the sign convention of `Duals`. Farkas mode is defined as reduced-cost pricing of the phase-one minimisation. Two provided methods, `reduced_cost` and `improves`, hold the sign logic once. | Also handing out the duals of lazy rows and cuts under a third id type. They are irrelevant, because a priced column has coefficient 0 there (§2.3). |
| 3 | **The callback adds the existing `NewCol`.** Its entries can only name program rows, which the `Row` type already enforces. Its bounds must contain 0. Semi columns are refused. | A narrower record type: it would duplicate `NewCol` without adding an invariant. |
| 4 | **Priced columns are real `Col`s.** They continue the program's numbering. `add_col` returns the id. After the solve the `Solver` appends them to its `Program` on every path (Ok, Err, unwind) and reloads the backend at the next call. | A separate table of priced columns in the solution: two id spaces, `values.len() != num_cols`, and the next solve would not start with them. |
| 5 | **The backend receives a sink, `&mut Priced`, as a new parameter of `Backend::solve`.** The core owns the checks and the numbering of priced columns. | A `Backend::take_priced()` method with a default body: a backend that declares PRICE and forgets to implement it would still compile and silently lose columns. |
| 6 | **No branching event in v1.** Version 1 is exact branch-and-price under SCIP's default branching on master variables, as long as the pricer never re-adds a column the problem already holds. Correct, but often weak. Ryan–Foster branching is named as a later step. | A branching event now: it would need a branching seam that does not exist yet. |

## 1. The event and its place in the seam

### 1.1 Events

The new constant goes before `NATIVE`, so `NATIVE` stays last in the iteration and `Display` order. The bit values are private.

```rust
impl Events {
    /// A node LP whose duals a callback may answer with new columns: pricing.
    pub const PRICE: Self = Self(1 << 8);
    /// An event only the backend knows, delivered with the backend's own context type.
    pub const NATIVE: Self = Self(1 << 9);
    /// Every event.
    pub const ALL: Self = Self((1 << 10) - 1);
}
```

### 1.2 Mode and context trait (`culmen_core::solve::context`)

```rust
/// What the node LP is when the solver asks for columns, and so what the context's duals mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PriceMode {
    /// The node LP is optimal over the columns it holds; a column with an improving reduced cost may improve it.
    ReducedCost {
        /// The node LP's objective value in the program's sense, offset and quadratic part included.
        objective: f64,
    },
    /// The node LP is infeasible over the columns it holds; a column the Farkas multipliers favour may make it
    /// feasible.
    Farkas,
}

/// A node LP the callback may improve, or make feasible, by adding columns: pricing.
pub trait PriceContext {
    /// Where the solve stands.
    fn progress(&self) -> Progress;
    /// Whether the node LP is optimal, with its objective, or infeasible.
    fn mode(&self) -> PriceMode;
    /// Whether the program minimizes or maximizes.
    fn sense(&self) -> Sense;
    /// The node LP's multiplier of every program row, in the convention [`PriceContext::reduced_cost`] states.
    fn duals(&self) -> &TiSlice<Row, f64>;
    /// The magnitude up to which the solver counts a reduced cost as zero.
    fn tolerance(&self) -> f64;
    /// Adds a column to the search and returns its id, which continues the program's columns.
    ///
    /// # Errors
    ///
    /// [`ActionError`] when an entry names a row the program lacks, a number is not finite, the bounds do not hold
    /// zero, the column is semi, or the solver refuses it.
    fn add_col(&mut self, column: NewCol) -> Result<Col, ActionError>;
    /// Ends pricing at this node after this round and branches; the node LP's value is then no bound unless
    /// [`PriceContext::report_bound`] gives one.
    fn branch_early(&mut self);
    /// Reports a bound on the optimal objective in this node's subtree, over every column the program may price, in
    /// the program's sense with its offset: a lower bound when minimizing, an upper one when maximizing.
    ///
    /// # Errors
    ///
    /// [`ActionError::NotFinite`] for a NaN or infinite bound; [`ActionError::Refused`] in Farkas mode.
    fn report_bound(&mut self, bound: f64) -> Result<(), ActionError>;

    /// The column's reduced cost at this node: `cost − Σ duals[r]·a_r` in reduced-cost mode, and `−Σ duals[r]·a_r` in
    /// Farkas mode, where the cost counts as zero.
    ///
    /// # Panics
    ///
    /// When an entry names a row beyond [`PriceContext::duals`].
    fn reduced_cost(&self, column: &NewCol) -> f64 { /* provided, in culmen_core */ }

    /// Whether adding the column could improve the node LP, or make it feasible, beyond the solver's tolerance.
    ///
    /// # Panics
    ///
    /// As [`PriceContext::reduced_cost`].
    fn improves(&self, column: &NewCol) -> bool { /* provided */ }
}
```

`improves` is written once, in the core. Let `d` be `reduced_cost`, and take it in minimisation form: `t = d` for Farkas mode or a minimisation, `t = −d` for a maximisation. Then `improves` returns:

```
(upper > 0 && t < −tol) || (lower < 0 && t > tol)
```

where `upper` and `lower` are `column.column.bounds`. `Unsupported` implements the trait with `match *self {}`.

### 1.3 Backend and Callback (`backend.rs`)

```rust
pub trait Backend: Sized {
    // ...
    /// The context of a [`Events::PRICE`] event.
    type Price<'a>: PriceContext;

    /// Solves the loaded program, calling `callback` as the search goes and handing every column a callback prices in
    /// to `priced`.
    fn solve(
        &mut self,
        settings: &Settings,
        callback: &mut dyn Callback<Self>,
        priced: &mut Priced,
    ) -> Result<Solved<Self::Status>, Self::Error>;
}

pub trait Callback<B: Backend>: Send {
    // ...
    /// A node LP's duals, to be answered with columns that improve it.
    fn price(&mut self, ctx: &mut B::Price<'_>) { let _ = ctx; }
}
```

The `&mut C` forwarding and the `(X, Y)` pair both get a `price` arm, like the other events. The two callbacks of a pair share one context, so their columns are numbered one after the other. `Watch` ignores PRICE.

### 1.4 The sink (`solve/priced.rs`, new)

```rust
/// The columns one solve prices in, numbered after the program's columns: what a backend's price context hands every
/// added column to, so that the [`Solver`] can append them to its program.
#[derive(Debug)]
pub struct Priced { /* num_rows, base num_cols, classes, quadratic: bool, discrete: bool, cols: Vec<NewCol> */ }

impl Priced {
    /// No column yet, for a solve of `program` by a backend that solves `classes`.
    pub fn new(program: &Program, classes: ModelClasses) -> Self;
    /// Checks a column as [`PriceContext::add_col`] states, and appends it; its id, after the program's columns and
    /// every one priced before.
    ///
    /// # Errors
    ///
    /// As [`PriceContext::add_col`], except a solver's own refusal.
    pub fn push(&mut self, column: NewCol) -> Result<Col, ActionError>;
    /// The number of columns: the program's and those priced so far.
    pub fn num_cols(&self) -> usize;
    /// The ids priced so far, in order.
    pub fn cols(&self) -> Cols;
    /// A priced column as checked (entries sorted by row, merged, zeros dropped); `None` for an id it did not hand out.
    pub fn get(&self, col: Col) -> Option<&NewCol>;
    /// The columns priced, in the order of their ids.
    pub fn into_cols(self) -> Vec<NewCol>;
}
```

`push` reuses `NewCol::checked` and `bad_column`, which already exist as `pub(crate)` in the core. It adds two checks:
- the bounds must contain 0;
- the integrality must not be semi, and the program's class with this column must stay in `classes`.

### 1.5 Script

```rust
pub enum Step {
    // ...
    /// A node LP offered for pricing.
    Price {
        /// Where the solve stands.
        progress: Progress,
        /// Whether the node LP is optimal, with its objective, or infeasible.
        mode: PriceMode,
        /// The multiplier of every program row.
        duals: TiVec<Row, f64>,
    },
}

/// What a callback did at one [`Step::Price`].
#[derive(Clone, Debug, PartialEq)]
pub struct PriceRound {
    /// The columns it added; the program holds them after the solve.
    pub added: Cols,
    /// The bound it reported, `None` when it reported none.
    pub bound: Option<f64>,
    /// Whether it ended pricing early.
    pub early: bool,
}

pub struct Record { /* ... */ /// One entry per price step the callback ran, in order.
    pub priced: Vec<PriceRound> }

/// The price context of a [`Script`]: columns are checked by the solve's [`Priced`] and the round recorded.
#[derive(Debug)]
pub struct ScriptPrice<'a> { /* step data, sense, &mut Priced, &mut PriceRound */ }

impl ScriptPrice<'_> {
    /// The tolerance the script's context reports.
    pub const TOLERANCE: f64 = 1e-9;
}
```

The Record does not keep the columns themselves: the program holds them (one fact, one place). The Script's `load` also keeps the program's `sense`. Its running column count becomes `priced.num_cols()`, so a later LAZY, CUT or HEURISTIC step checks lengths against n+k.

## 2. What the context hands out

### 2.1 Reduced-cost mode

`duals()[r]` is the rate at which the node LP optimum changes as row r's active bound rises. This is the convention of `Duals` and `Solver::duals()` (docs/analysis.md, HiGHS's convention). The formula `c = Aᵀ·rows + cols` is the same in both senses.

- **Minimisation:** a row at its lower bound has a dual ≥ 0, a row at its upper bound a dual ≤ 0. A column with cost c and entries a_r, bounds [0, u] with u > 0, prices out (improves) when **c − Σ y_r a_r < −tol**.
- **Maximisation:** both signs reverse. The same column improves when **c − Σ y_r a_r > +tol**.
- **General bounds that contain 0:** a column with lower < 0 < upper improves when |d| > tol. A column with upper = 0 improves when the inequality is the other way round. `improves` encodes all of this.

How SCIP maps onto it:
- SCIP's transformed problem minimises `objsense · c` [read: var.c:4527, `(SCIP_Real)objsense * origvar->obj`]. Objective scaling is skipped when a pricer is active [read: prob.c:1950], and SCIP computes `redcost = obj − Σ val·dualsol` [read: lp.c:4056–4062].
- So **y_culmen = s · SCIPgetDualsolLinear(transcons)**, with s = +1 when minimising and −1 when maximising.
- The new variable gets `SCIPcreateVar(obj = s·c)`. This matches the warning in scip_var.h ("the original objective function value of variables created during the solving process has to be multiplied by -1") [read: scip_var.h:84–87].
- The full sign chain is [assumed; probe P2].

### 2.2 Farkas mode

`duals()[r]` is SCIP's Farkas multiplier f_r, unchanged. It is independent of the sense, because only the objective is negated in the transformation [read: var.c:4527].

The certificate is `fᵀAx ≥ fᵀb`, where b is the lhs where f_r > 0 and the rhs where f_r < 0. It is violated by every x within its bounds [read: type_pricer.h, PRICERFARKAS].

A column helps when `(Σ f_r a_r) · x'` > 0, with x' = ub if the coefficient is ≥ 0 and lb otherwise [read: lp.c:4356–4370; pricestore.c:440–462].

culmen states this as: in Farkas mode the cost counts as zero and the sense counts as minimise, so `reduced_cost = −Σ f_r a_r` and the column improves when that is < −tol (for ub > 0). SCIP's xternal.c says the same: "Farkas pricing is the same as the regular pricing with a zero objective function" [read].

### 2.3 Rows that are not in the Program

There are three kinds: lazy constraints, user cuts, and the epigraph constraint of a QP together with its linearisations. They have no `Row`, so **they are excluded**.

This is exact, not an approximation. A priced column's entries can only name program `Row`s, so its coefficient in every other row is 0. Those rows' duals therefore contribute nothing to its reduced cost or Farkas coefficient: Σ over all LP rows equals Σ over program rows.

What correctness then needs is a **user guarantee**: every lazy constraint and cut must be valid for the full program with every column it may price, each unseen column entering it with coefficient zero. These are "robust" rows. A non-robust cut, such as a Chvátal–Gomory rank-1 cut on master columns that would need lifting for new columns, cannot be expressed and is the user's error.

SCIP's own aggregation-based cuts ignore modifiable rows:
- [read: cuts.c:3277 and 3399, `SCIProwIsModifiable(row)` returns early; cuts.c:7621 says "The weights of modifiable rows are set to 0.0, because these rows cannot participate in an MIR cut"];
- [read: sepa_aggregation.c:427–428 and 1264; sepa_zerohalf.c:662];
- that Gomory goes through the same path is [assumed; covered by the property test with separators on, P8].

### 2.4 Objective and progress

The objective comes from `SCIPgetLPObjval` mapped back with `SCIPretransformObj` [read: scip_lp.h:197; scip_sol.h:853], so it is in model space with the offset. With an epigraph, t stands for q(x), as it does in `Columns::read`. Progress is as in the other contexts.

### 2.5 Local bounds

Local bounds are **not** in the trait (see §6). `ScipPrice` offers them as an inherent method:

```rust
impl ScipPrice<'_> {
    /// The node's local bounds of a column, as branching and propagation left them; `None` for an id the problem lacks.
    pub fn local_bounds(&self, col: Col) -> Option<Bounds>;
}
```

They come from SCIPvarGetLbLocal and SCIPvarGetUbLocal [read: pub_var.h:876], on the transformed variable.

Deferred: the node LP's primal values, a `relaxation()` method. Pricing does not need them.

## 3. What the callback adds, and what is refused

The callback adds the existing plain-data `NewCol { column: ProgramColumn, entries: Vec<(Row, f64)> }`:
- cost, bounds, integrality (`Continuous` or `Integer`) and name;
- entries in program rows only, which the type already guarantees;
- a cost that is linear only: `NewCol` carries no quadratic terms and no further-objective terms, so those are 0.

New flat variants of `ActionError` (the enum is not `#[non_exhaustive]`; see §9):

```rust
/// An entry names a row the program does not have.
UnknownRow { row: Row },
/// A coefficient of a new column is NaN or infinite, as given or once its repeated entries were added up.
NotFiniteEntry { row: Row, col: Col },
/// A priced column's bounds do not hold zero, so leaving it out would not be the same as fixing it at zero.
ExcludesZero { col: Col, bounds: Bounds },
/// A priced column's integrality is semi, or one the backend's classes do not allow.
Integrality { col: Col, integrality: Integrality },
```

A NaN or infinite cost reuses `NotFinite { col }`. Each variant gets one `Display` sentence; for example `ExcludesZero` prints "callback action: the priced column {col:?} has the bounds {bounds}, which do not hold zero".

Backend refusals go through `Refused(reason)`. For SCIP these are a number beyond SCIP's infinity, a name with a NUL byte, and `report_bound` in Farkas mode.

Why the bounds must hold zero: a withheld column is equivalent to one fixed at 0. That is what makes the restricted master a restriction of the full program, and so what makes "no improving column" mean "LP optimal".

## 4. Identity of priced columns

**Decision: option (a).**

1. Inside a solve, `Priced::push` hands out `Col(n + k)`, where n is the program's column count when the solve started and k counts the columns priced so far. `add_col` returns that id; the rule "a method that pushes a row returns its id" holds.
2. Every point a context delivers after that has length n + k at that moment: FEASIBLE and INCUMBENT values, a LAZY candidate, a CUT relaxation, the length `HeuristicContext::supply` expects. A column priced later is not part of a point delivered earlier; this is documented on each context trait.
   - In SCIP, a priced variable has no original counterpart, so in an original solution it reads as 0 [read: sol.c:1954–1957, "the variable has no original counterpart: in the original solution, it has a value of zero"].
   - A variable created after a transformed solution was stored also reads as 0 [assumed; P5].
3. The final `Solved` holds values for all n + K columns, so `best.values.len() == solver.program().num_cols()` holds after the solve.
4. In `Solver::solve_with`:
   - before the solve, `let mut priced = Priced::new(&self.program, B::CLASSES)`, passed to `backend.solve`;
   - a drop guard (or a `catch_unwind` followed by `resume_unwind`; both are safe Rust) runs on **every path**: Ok, Err, and unwinding out of a panicking callback;
   - if `priced` is not empty, the guard does `self.program.push_cols(priced.into_cols())`, `self.synced = false` and `self.ended = None`.
   - Ids keep their meaning even when the solve failed after handing them out (design.md: "a handle never changes its meaning").
   - The backend is not edited at once. The existing `sync()` loads the grown program at the next call that runs the backend, and a failure there surfaces through the existing `SolveError::Backend` or `EditError::Backend`.
   - For SCIP this is cheap: it has no `edit` override and solves from scratch anyway. It is also correct: the new instance holds the priced columns as ordinary original variables, and no stale retransformed solutions are carried over.
5. Analyses after a solve that priced at least one column return `NotSolved`. No backend has both PRICE and analyses today (SCIP declares no `ANALYSES`).
   - Rejected: requiring the backend to answer analyses for the grown program. Open question 3.
6. `Solver::into_parts`: the documentation becomes "the backend holds the program unless an edit failed and so did its reload, a panic interrupted them, or the last solve priced columns".
7. Script: the same `Solver` path applies. A backend driven directly (`Request::into_watch()`) creates its own `Priced` and appends `into_cols()` itself; this gets documented.
8. Rejected (b), a separate table: `Solution::values` would no longer be indexed by `Col` alone, mid-solve contexts would need a second index type, and the next solve would not start with the columns.

## 5. Guarantees

### 5.1 The user guarantees

1. **Exact pricing.**
   - At every PRICE event in reduced-cost mode: if any column of the full program that the problem does not yet hold improves, the callback adds at least one improving column, or calls `branch_early`.
   - The same holds in Farkas mode.
   - Only then is the node LP value a valid bound, and only then is `Outcome::Optimal` a proof.
2. **No re-adding.** The callback never adds a column the problem already holds, unless it means a second copy. The solver prices the columns it holds itself, within their node bounds.
3. Priced columns' bounds contain 0. Lazy constraints and cuts are robust (§2.3). `report_bound` is valid for the whole subtree over the full program.

### 5.2 A backend that declares PRICE guarantees

- **When PRICE is delivered:** only after a node LP is solved. In reduced-cost mode the LP is optimal; in Farkas mode it is infeasible.
- **What the context holds:** every program row's multiplier, in the convention of §2.
- **The loop:** pricing at a node ends only when a round adds no column and nobody asked to branch early. Only then does the node LP count as a bound. A round that stopped early counts only through `report_bound`.
- **Lifetime of a column:** every column added enters the search with its cost, bounds, integrality and entries, and stays for the rest of the solve. It is never deleted.
- **Reporting:** every column goes through `Priced::push`, and every point the backend reports afterwards includes it.
- **Requests that do not ask for PRICE:** a solve whose callback does not ask for PRICE behaves exactly as before.

### 5.3 SCIP: what the backend sets when PRICE is wanted, and what SCIP does by itself

**Plugin, included once per instance:**
- `SCIPincludePricerBasic(name "culmen", delay = TRUE, redcost, farkas, data = slot)`, with no copy callback.
- delay = TRUE means SCIP first prices the existing variables, so the callback can assume they do not improve [read: xternal.c PRICER_DELAY; scip_pricer.h].

**Per solve, in the problem stage, inside `configure_events`:**
- **Pricer:** `SCIPactivatePricer` when PRICE is wanted, `SCIPdeactivatePricer` otherwise. Both are allowed in the PROBLEM stage [read: scip_pricer.h:258–288].
- **Constraints:** `SCIPsetConsModifiable(origcons, wanted)` on every program row's constraint. This is allowed in PROBLEM [read: scip_cons.h:971–985]. SCIP requires it ("all constraints for which additional variables are generated by a pricer have to be flagged as modifiable") [read: xternal.c].
- **Effect on modifiable linear constraints:**
  - never presolved: cons_linear.c:16286 "we can only presolve linear constraints, that are not modifiable" [read];
  - never upgraded: cons_linear.c:18694 [read];
  - never used to tighten bounds: cons_linear.c:6727 and 7006, "we cannot tighten variables' bounds, if the constraint may be not complete" [read].
  - So they stay linear constraints, and `SCIPgetDualsolLinear` applies to them.
  - That the transformed copy inherits the flag, and that `SCIPgetTransformedCons` is non-null for every row after presolve: [assumed; P1].

**What SCIP does by itself once a pricer is active, so culmen sets nothing for it:**
- **Restarts** are disallowed: `restartAllowed` requires `nactivepricers == 0` [read: solve.c:4316, 4321].
- **Presolvers** stop where they need complete columns or rows:
  - [read: presol_dualcomp.c:539, presol_dualagg.c:523, presol_stuffing.c:373, presol_redvub.c:595];
  - a matrix built "only if complete" is not built [read: matrix.c:752];
  - domcol, dualinfer and tworowbnd pass TRUE there [read: presol_domcol.c:2089 and the others; that this argument is `onlyifcomplete` is assumed from its position, probe P1];
  - components is skipped [read: cons_components.c:2157].
- **Propagators:** pseudoobj, redcost and rootredcost skip unless forced [read: prop_pseudoobj.c:3470, prop_redcost.c:643, prop_rootredcost.c:619].
- **Symmetry:** prop_symmetry checks for active pricers [read only that it references them; that it disables itself is assumed, P1].
- **Objective:** no integrality detection and no scaling [read: prob.c:1763, 1950].
- **LP cutoff:** disabled automatically while not all columns are in the LP [read: lp.c:2782].
- **Sub-SCIP copies** are invalid, because the pricer is not copied [read: scip_copy.c:2690–2691; xternal.c PRICERCOPY]. Solutions found by sub-SCIPs are still primal-valid, since unseen columns are 0 [read: xternal.c].
- **Interrupts:** a stop interrupts the pricing loop and the node LP is then not used [read: solve.c:2289–2297].

**Dual reductions:** no extra setting is needed for PRICE.
- The reductions SCIP still does reason per variable over its own locks, and new columns do not change an existing variable's locks. Dual fixing stays valid.
- With LAZY, the existing settings stay (`misc/usesymmetry = 0`, dual reductions off).

**Deletion of priced variables:** none happens. `pricing/delvars` and `pricing/delvarsroot` default to FALSE [read: set.c:383–385], and a variable is deletable only once marked [read: var.c:2461, `deletable = FALSE` at creation]. culmen never marks one.

**Result codes:**

| Mode | culmen action | SCIP result |
|---|---|---|
| Reduced cost | ran normally, columns or none | `SCIP_SUCCESS`, `lowerbound` = `SCIPtransformObj(report_bound)` or −∞ |
| Reduced cost | `branch_early` | `SCIP_SUCCESS` with `*stopearly = TRUE` |
| Reduced cost | callback busy (nested call) or panicked | `SCIP_DIDNOTRUN` |
| Farkas | ran normally | `SCIP_SUCCESS` |
| Farkas | `branch_early`, busy or panicked | `SCIP_DIDNOTRUN` |

Source and status of these choices:
- SCIP uses the pricer's bound only on `SCIP_SUCCESS`: `SCIPnodeUpdateLowerbound` is called only in the SUCCESS branch [read: solve.c:2335–2352].
- `stopearly` ends pricing and marks the node aborted [read: solve.c:2341–2346, 2392–2397].
- That SUCCESS + stopearly with no new variable is accepted: [assumed; P4].
- `SCIPtransformObj` [read: scip_sol.h:832].
- An aborted node stores its LP solution and branches on the pseudo solution [read: solve.c:4653–4700].
- The Farkas callback has only SUCCESS or DIDNOTRUN [read: type_pricer.h]. Without a Farkas callback SCIP counts SUCCESS, so the node would be wrongly infeasible [read: pricer.c SCIPpricerFarkas, `if( pricer->pricerfarkas == NULL ) return SCIP_OKAY;` with result SUCCESS]. So culmen must always install one.

**Applying a column:** after the callback returns, outside the `RefCell` borrow, like lazy constraints:
- `SCIPcreateVar(name, lb, ub, s·c, CONTINUOUS|INTEGER, initial = FALSE, removable = TRUE, …)`;
- `SCIPaddPricedVar(var, score = |d|)`; SCIPaddPricedVar is SOLVING-stage only [read: scip_prob.h:833–845];
- `SCIPaddCoefLinear(transcons[r], var, a)` for each entry [read: xternal.c; cons_linear.h:224–226];
- the instance keeps one reference to each priced variable until the next `free_transform`, so that `best()` can read them after `SCIPsolve`.

**Cost when PRICE is not requested:** none. The pricer is included but inactive, and SCIP calls only active pricers [read: solve.c:2321, loop over `nactivepricers`]. Constraints are not modifiable.

## 6. Branching: what v1 really is

- culmen v1 does **branch-and-price with SCIP's default variable branching on integer master columns**.
- It is **exact** when:
  - the user's pricing is exact over the columns the problem does not yet hold;
  - the user never re-adds a column the problem already holds;
  - SCIP prices existing columns itself, within their local bounds, before calling the pricer (delay = TRUE).
- Then each node LP is the full master LP under the bounds branching set on some columns. The branching partitions the problem, and integral leaves are feasible for the full master.
- What is wrong is **regenerating** a column that branching fixed. The duplicate escapes the branch, the children overlap, and the search may not terminate.
- Variable branching on master columns is still **weak**. The down branch x_p ≤ ⌊v⌋ barely changes the LP, and trees can be huge.
- **Pricing respects branching for free.** Branching bounds existing columns, which the solver prices itself; unseen columns are untouched. So the trait does not need local bounds. `ScipPrice::local_bounds` is offered for pricers that want them, for example to skip patterns implied by fixings.
- **Ryan–Foster and other constraint branching** need a branching event: a branching rule plugin plus pricing-subproblem modifications visible to the callback. This is named as a later step and not designed here.
- **What the documentation says:** "column generation at every node with exact pricing; branching on master columns as SCIP chooses; correct, often slow on set-partitioning masters".

## 7. Interaction with the rest

- **LAZY with PRICE.**
  - Lazy rows stay non-modifiable linear constraints, so priced columns enter them with coefficient 0 (user guarantee, §2.3).
  - A lazy constraint may name a priced `Col`. `transformed()` passes priced variables through, since they are transformed already.
  - The lazy handler's `lock_all` locks only original variables. Priced variables get their locks from the linear constraints they enter. With LAZY, dual reductions are off anyway.
- **CUT with PRICE.** Cut rows are created non-modifiable (`SCIPcreateEmptyRowSepa(…, modifiable = 0)`); the robustness contract applies.
- **HEURISTIC with PRICE.**
  - `supply` must use a transformed solution (`SCIPcreateSol`, `SCIPsetSolVal` on transformed and priced variables). An original solution would silently drop priced values (sol.c reads them as 0, [read]).
  - Behaviour when presolve fixed or aggregated an original variable: [assumed; P6].
  - `supply_partial` stays refused.
- **QP/MIQP.**
  - Priced columns never enter q(x), and the epigraph's linearisations get coefficient 0, so the reasoning holds.
  - Untested: the reference enumerator does not do QPs, and the interaction with cons_nonlinear is [assumed; P7].
  - Recommendation: SCIP refuses PRICE on a program with a quadratic part in v1, with `ScipError::PricingQuadratic` before `SCIPsolve` (open question 2).
- **Starts.** Checked against the program before the solve, so they may name columns priced in earlier solves (now ordinary columns), not columns priced in this one.
- **Edits after a solve** work as usual on the grown program.
- **Settings:** no new field. `pricing/maxvars` and the like go through `ScipOptions::params`. `presolve` still applies, and the tests run with presolve both on and off.
- **Analyses:** `NotSolved` after a solve that priced (§4.5).
- **Probing:** no default plugin prices during probing. `SCIPsolveProbingLPWithPricing` is referenced only in scip_probing.c [read: grep over every .c file].

## 8. Tests, written from the rules before the implementation

### 8.1 Fixture 1: cutting stock, minimisation, unique duals in every round

**Instance.** Roll width 10; pieces of width 3, 4 and 5 with demands 7, 2 and 3.
- Rows: `Row(0)`: Σ 3-pieces ≥ 7; `Row(1)`: Σ 4-pieces ≥ 2; `Row(2)`: Σ 5-pieces ≥ 3.
- Every column costs 1 with bounds [0, ∞).
- There are 11 feasible patterns (a3, a4, a5): (1,0,0), (2,0,0), (3,0,0)=A, (0,1,0), (1,1,0), (2,1,0)=B, (0,2,0)=D, (0,0,1), (1,0,1)=C, (0,1,1)=E, (0,0,2)=F.
- Start: A=`Col(0)`, D=`Col(1)`, F=`Col(2)`. The other 8 patterns are withheld. The pricer scans the withheld ones and adds those that `improves`.

**Round 1.** Mode `ReducedCost { objective: 29/6 }`.
- Primal: xA = 7/3, xD = 1, xF = 3/2 (each row tight, each basic variable > 0, so nondegenerate).
- Duals (1/3, 1/2, 1/2).
- Reduced costs of the withheld patterns:

| Pattern | Reduced cost |
|---|---|
| (1,0,0) | 2/3 |
| (2,0,0) | 1/3 |
| (0,1,0) | 1/2 |
| (1,1,0) | 1/6 |
| **B** | **−1/6** |
| (0,0,1) | 1/2 |
| C | 1/6 |
| E | **0** |

- Only B improves. E sits exactly on the tolerance boundary and must not be added. `add_col(B)` returns `Col(3)`.

**Round 2.** Mode `ReducedCost { objective: 9/2 }`.
- Primal: xA = 1, xB = 2, xF = 3/2, xD = 0 (nondegenerate).
- Duals (1/3, 1/3, 1/2). These follow from the basic columns: 3y₃ = 1, 2y₃ + y₄ = 1, 2y₅ = 1.
- Reduced costs: (1,0,0) 2/3; (2,0,0) 1/3; (0,1,0) 2/3; (1,1,0) 1/3; (0,0,1) 1/2; C 1/6; E 1/6; existing D 1/3.
- Nothing improves.

**LP result.** Optimum 9/2 with values (1, 0, 3/2, 2) in `Col` order. The program has 4 columns.

**Integer variant** (every column `Integer`): the optimum is **5**.
- Lower bound: ⌈4.5⌉ = 5, because every cost is 1, so the objective is integral.
- Achieved by A + 2B + F + C: 3-pieces 3+4+1 = 8 ≥ 7, 4-pieces 2 ≥ 2, 5-pieces 2+1 = 3 ≥ 3.
- Assert only the optimum and that the final solution is feasible in the full 11-pattern program. Which columns get priced depends on SCIP's branching.

### 8.2 Fixture 2: maximisation, a Farkas round, then a reduced-cost round with the max sign

**Instance.** Maximise 3x1 + 2x2 + 7x3.
- `Row(0)`: x1 + x2 + 2x3 ≤ 4.
- `Row(1)`: x1 + x3 ≥ 1.
- All bounds [0, 10], continuous.
- Start: x2 = `Col(0)`. Withheld: x1, then x3. The pricer adds the **first** improving withheld column in that order.

**Round 1: `Farkas`.** The restricted master is infeasible (0 ≥ 1).
- Any valid multiplier vector has f₀ ≤ 0 (Row 0 has only a rhs) and f₁ ≥ 0 (Row 1 has only a lhs).
- Violation needs max over x2 ∈ [0, 10] of f₀·x2 = 0 < 4f₀ + f₁, so **f₁ > −4f₀ ≥ 0**.
- Hence `reduced_cost(x1) = −(f₀ + f₁) < 0` and `reduced_cost(x3) = −(2f₀ + f₁) < 0`, for every valid ray SCIP may return.
- Assert f₁ > 0, f₀ ≤ 0, `improves(x1)`. Then x1 is added as `Col(1)`.
- The exact values are not unique and must not be asserted.

**Round 2: `ReducedCost { objective: 12 }`.**
- Primal x1 = 4, x2 = 0. The basis is {x1, slack of Row 1 = 3}: nondegenerate, so the duals are unique.
- Duals **(3, 0)**. For a maximisation, a row at its upper bound has a dual ≥ 0.
- Reduced costs: x1 = 0; x2 = 2 − 3 = −1, which is optimal for max at its lower bound; withheld x3 = 7 − 2·3 = **+1 > 0, so it improves (max)**. x3 is added as `Col(2)`.

**Round 3: `ReducedCost { objective: 14 }`.**
- Primal x3 = 2, x1 = x2 = 0.
- Duals (7/2, 0). Reduced costs: x1 −1/2, x2 −3/2, x3 0.
- Nothing is withheld; nothing is added.

**Result.** Optimum 14, values (0, 0, 2) in `Col` order (x2, x1, x3).

In SCIP's transformed space this shows the mapping: `SCIPgetDualsolLinear(Row 0) = −3` in round 2, and x3 is created with objective −7 (P2).

### 8.3 Property test (`culmen_scip/tests/price.rs`, infrastructure in `culmen_testing::strategy`)

**Strategy** `priced_split()`. It generates a full program and a mask of kept columns:
- 1–3 rows with ≤, ≥, = or ranged bounds and small integer coefficients in −2..2;
- 2–6 columns with costs in −3..3, bounds [0, u] with u ∈ {1, 2, 3}, each continuous or integer;
- either sense.

The integer box stays ≤ 4⁶, within `reference::solve`'s limit. The start program keeps all rows and the kept columns. The withheld columns become `NewCol`s with the same `Row` ids.

**Reference.** `culmen_testing::reference::solve(&full)`: exact enumeration of the full program with every column up front, in rational arithmetic. It never prices, so it shares nothing with the implementation.

**Under test.** `Solver::new(Scip::new(), start)`, then `solve_with` a test pricer that adds every withheld, not-yet-added column for which `ctx.improves`. A second variant adds only the single most-improving column. Each runs with presolve on and off.

**Assertions:**
1. The result is infeasible exactly when the reference is.
2. The objective is close to the reference optimum (relative 1e-6).
3. `best.values.len() == solver.program().num_cols() == kept + added`.
4. Every added `Col` equals the withheld column it stands for: cost, bounds, integrality and entries.
5. Mapping the values back to the full program (withheld and not added → 0) gives `full.max_violation ≤ 1e-6` and an objective equal to the reference optimum.
6. **An invariant that needs no model:** every `Progress::bound` reported at POLL never beats the reference optimum, ≤ opt + tol when minimising.

A pricer that ignores Farkas mode or flips a sign fails 2 or 6. This exercises MILP with SCIP's variable branching too.

### 8.4 Script-level tests (`culmen_core/tests/price.rs`)

1. **Numbering.** `add_col` returns `Col(n)`. After the solve, `program().columns()[Col(n)]` equals the checked column. `record.priced == [PriceRound { added: n..n+1, bound: None, early: false }]`.
2. **Lengths.** A LAZY step after a PRICE step accepts `Col(n)` and refuses `Col(n+1)` with `UnknownColumn`.
3. **Refusals.**

| Input | Expected error |
|---|---|
| entry naming `Row(m)` | `UnknownRow` |
| NaN cost | `NotFinite` |
| NaN entry | `NotFiniteEntry` |
| bounds [1, 2] | `ExcludesZero` |
| semi-continuous | `Integrality` |

4. **Fixture values through the provided methods.** Fixture 1, round 1: `reduced_cost(B) == −1/6`, `improves(B)`, `!improves(E)`. Fixture 2, round 2 (max): `reduced_cost(x3) == 1`, `improves`. Farkas with f = (−0.2, 1): x1 improves, x2 (−f₀ = 0.2) does not.
5. **Panic.** A panic after `add_col`: caught with `catch_unwind`, then the program holds the column and the next solve reloads. Script's `load` then sees n+1 columns.
6. **Interrupt.** A poll interrupt after pricing: `Stopped`, and the columns are kept.
7. **Pair.** A pair `(a, b)` both pricing: ids n and n+1 in that order.
8. **Bound and early stop.** `report_bound` in Farkas mode is refused. `branch_early` and `report_bound` are recorded.
9. **Events docs.** The `Events` Display and len doctests are updated.

### 8.5 SCIP-level tests and probes

**Tests:**
- fixtures 1 (LP and IP) and 2, checking the duals of the first rounds to 1e-9;
- a panic in `price`;
- an interrupt during pricing;
- a solve without PRICE leaves `SCIPgetNActivePricers == 0` and the constraints not modifiable (unit test);
- Fixture 1 IP with a lazy constraint xB ≤ 1 (robust). The optimum stays 5, achieved by B + E + F + 2A, and every reported solution meets xB ≤ 1.

**Probes**, each a throwaway program over scip-sys:

| Probe | What it settles |
|---|---|
| P1 | Fixture 1 with modifiable rows and presolve on: the transformed constraints are non-null, modifiable is inherited, duals are (1/3, 1/2, 1/2) then (1/3, 1/3, 1/2). The `SCIPmatrixCreate` argument meaning and prop_symmetry disabling itself, by reading. |
| P2 | Fixture 2 dual signs in transformed space. |
| P3 | Farkas callback on Fixture 2: `SCIPgetLPSolstat == INFEASIBLE`, f₁ > 0, f₀ ≤ 0. |
| P4 | SUCCESS + stopearly with no variable and a lower bound: accepted, and the node bound is updated. |
| P5 | `SCIPgetSolVal` of a priced variable created after a solution was stored reads 0. |
| P6 | A transformed `SCIPcreateSol` with priced variables when presolve fixed an original variable. |
| P7 | QP epigraph with pricing. |
| P8 | Gomory cuts stay valid with modifiable rows; covered by the property test with separators on. |

## 9. Stages, files and breaking changes

### Stage 1: the seam and Script, no solver

Files:
- `culmen_core/src/solve/{events.rs, context.rs, backend.rs, mod.rs}`, new `solve/priced.rs`, `script.rs`, `lib.rs` re-exports;
- `culmen_highs/src/lib.rs`: `Price = Unsupported`, new `solve` signature;
- `culmen_scip/src/lib.rs`: `Price = Unsupported`, `EVENTS = ALL − NATIVE − PRICE` for now, new signature;
- `culmen_testing/src/mirror.rs`;
- the test backends in `culmen_core/tests/{features, analysis, edit, quadratic}.rs`;
- docs `callbacks.md`, `backends.md` (a PRICE row: HiGHS none; Gurobi none, as far as I know, since Gurobi 12 has no pricing callback; SCIP pricer), `testing.md`, `solving.md`.

Tests: §8.4.

### Stage 2: the SCIP pricer for LP and MILP

Files:
- new `culmen_scip/src/callback/price.rs`: plugin, `ScipPrice`, applying columns, result codes;
- `callback/mod.rs`: include, `Run` holding priced variables and the cache of transformed constraints, module table;
- `columns.rs`: a priced tail, read for all n+k;
- `instance/mod.rs`: per-solve modifiable flag, activate and deactivate, owning priced variables until `free_transform`;
- `lib.rs`: `configure_events`, `EVENTS = ALL − NATIVE`, docs;
- `error.rs`.

Tests: fixtures 1 and 2, panic, interrupt, inactive when not wanted. Probes P1–P5 first.

### Stage 3: interplay and the property test

Files: `callback/heuristic.rs` (transformed solutions when pricing), `lazy.rs` and `cut.rs` (pass-through for priced variables), the QP refusal in `lib.rs`, `culmen_testing/src/strategy.rs`.

Tests: §8.3 and the LAZY + PRICE fixture. Probes P6–P8.

### Stage 4: example and documentation

New `culmen/examples/branch_and_price.rs`: Fixture 1 grown to the PIECES of `column_generation.rs`. Plus the SCIP chapter in the docs.

### Breaking changes (justified, since the workspace is at 0.4.0 and unreleased)

1. **`Backend::solve` gains `priced: &mut Priced`.** Every backend and every direct driver changes. A sink owned by the caller is the only design where a forgotten implementation cannot compile, and where the `Solver` sees the columns on unwind.
2. **`Backend::Price<'a>` is a required associated type.** Associated type defaults are unstable, so every backend has to name it.
3. **`Events::ALL` gains PRICE.** `ALL.len()` goes from 9 to 10, the docs' `(Events::ALL - Events::LOG).len()` goes from 8 to 9, and NATIVE's private bit moves.
4. **`ActionError` gains four variants.** Exhaustive matches break.
5. **`Step` gains `Price`, and `Record` gains `priced`.** Exhaustive matches and struct literals break.
6. **Behaviour changes:**
   - `Solver::program()` grows after a solve that priced;
   - analyses are `NotSolved` then;
   - the backend reloads at the next call;
   - the `into_parts` documentation changes.
7. **`Scip::EVENTS`** drops PRICE in stage 1 and gains it in stage 2. The doctest `Scip::EVENTS == Events::ALL - Events::NATIVE` holds again from stage 2 on.

## 10. Open questions for you

1. **Default `Callback::events()`.** Today it is `ALL`. With PRICE in `ALL`, every callback that does not narrow `events()` would put SCIP into pricing mode: modifiable rows, no restarts, many presolvers off. That is correct, because the default `price` adds nothing and so claims none exists, but it is slower. *Recommendation:* make the default `Events::ALL - Events::PRICE` and document that pricing must be asked for.
2. **PRICE on QP/MIQP in SCIP.** *Recommendation:* refuse it in v1 with `ScipError::PricingQuadratic`, until P7 and a QP reference exist.
3. **Analyses after a pricing solve.** *Recommendation:* `NotSolved` in v1, rather than requiring backends to answer for the grown program.
4. **Branching event (Ryan–Foster).** *Recommendation:* a later, separate design. v1 ships as exact branch-and-price under variable branching, with its weakness documented.
5. **Refuse exact duplicate columns?** That would catch the "regenerated a branched column" bug. *Recommendation:* no. Identical columns can be legitimate (two identical trucks), and refusing them changes the model. Document the rule in `add_col` instead.

