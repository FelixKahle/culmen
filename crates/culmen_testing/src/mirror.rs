// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A backend that wraps another: it records every load and edit a `Solver` asks of it, checks the edit contract of
//! `Backend::edit`, keeps the program it holds, and fails or panics on demand.

use core::fmt;

use culmen_core::backend::{Edit, Hooks, Relay};
use culmen_core::{
    Backend, Basis, BasisAnalysis, Callback, Col, Configurable, DualAnalysis, Duals, Events, Heuristics, Iis,
    IisAnalysis, LazyConstraints, ModelClasses, Prices, Program, ProgramFeatures, ProgramPlan, Ranging,
    RangingAnalysis, RayAnalysis, RelaxationAnalysis, Relaxed, Row, Settings, Solved, Start, Starts, TiVec, UserCuts,
    Weights,
};

/// A backend that forwards to `inner`, recording what it is asked to hold and failing as [`Faults`] say.
///
/// [`Mirror::held`] is the program the mirror holds: the last one a load or an edit accepted, `None` before any and
/// after a failure, injected or not. A test asserts `held() == Some(solver.program())` to check that the solver and
/// its backend agree. Its [`Options`](Configurable::Options) are its [`Faults`], so a test sets the failures to inject
/// between calls with `solver.options_mut()`; the inner backend's own options are set before the mirror wraps it.
#[derive(Debug)]
pub struct Mirror<B> {
    /// The wrapped backend.
    pub inner: B,
    /// Every load and edit, in order, with the program each handed over.
    pub calls: Vec<Call>,
    /// The failures still to inject.
    pub faults: Faults,
    held: Option<Program>,
}

/// Failures a [`Mirror`] injects, each counting down as it fires.
///
/// The counts are independent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Faults {
    /// How many of the next edits fail after the inner backend applied them, as a solver failing halfway would.
    pub refuse_edits: usize,
    /// How many of the next loads fail before the inner backend sees them.
    pub refuse_loads: usize,
    /// How many of the next edits panic after the inner backend applied them.
    pub panic_edits: usize,
}

/// One call a [`Mirror`] received.
#[derive(Clone, Debug, PartialEq)]
pub enum Call {
    /// A load of this program.
    Load(Program),
    /// An edit into this program.
    Edit(Edited, Program),
}

/// An [`Edit`] as owned data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Edited {
    /// [`Edit::ColBounds`].
    ColBounds(Vec<Col>),
    /// [`Edit::ColCosts`].
    ColCosts(Vec<Col>),
    /// [`Edit::ColIntegrality`].
    ColIntegrality(Vec<Col>),
    /// [`Edit::RowBounds`].
    RowBounds(Vec<Row>),
    /// [`Edit::NewRows`].
    NewRows(Vec<Row>),
    /// [`Edit::NewCols`].
    NewCols(Vec<Col>),
    /// [`Edit::Sense`].
    Sense,
    /// [`Edit::Offset`].
    Offset,
    /// [`Edit::Objective`].
    Objective,
}

impl Edited {
    /// The owned form of `edit`.
    fn of(edit: &Edit) -> Self {
        match edit {
            Edit::ColBounds { cols } => Self::ColBounds(cols.clone()),
            Edit::ColCosts { cols } => Self::ColCosts(cols.clone()),
            Edit::ColIntegrality { cols } => Self::ColIntegrality(cols.clone()),
            Edit::RowBounds { rows } => Self::RowBounds(rows.clone()),
            Edit::NewRows { rows } => Self::NewRows(rows.into_iter().collect()),
            Edit::NewCols { cols } => Self::NewCols(cols.into_iter().collect()),
            Edit::Sense => Self::Sense,
            Edit::Offset => Self::Offset,
            Edit::Objective => Self::Objective,
        }
    }
}

/// Why a [`Mirror`] failed: its inner backend failed, or a fault was injected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MirrorError<E> {
    /// The inner backend failed.
    Inner(E),
    /// A [`Faults`] count fired.
    Injected,
}

impl<E: fmt::Display> fmt::Display for MirrorError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inner(error) => write!(f, "Mirror: {error}"),
            Self::Injected => f.write_str("Mirror: an injected failure"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for MirrorError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Inner(error) => Some(error),
            Self::Injected => None,
        }
    }
}

impl<B> Mirror<B> {
    /// A mirror of `inner` that holds nothing and injects no fault.
    pub const fn new(inner: B) -> Self {
        Self {
            inner,
            calls: Vec::new(),
            faults: Faults {
                refuse_edits: 0,
                refuse_loads: 0,
                panic_edits: 0,
            },
            held: None,
        }
    }

    /// The program the mirror holds: the last a load or an edit accepted; `None` before any and after a failure.
    pub const fn held(&self) -> Option<&Program> {
        self.held.as_ref()
    }
}

/// Counts a fault down, and whether it fired.
const fn fire(count: &mut usize) -> bool {
    let fired = *count > 0;
    if fired {
        *count -= 1;
    }
    fired
}

/// Whether `after` is `before` changed only as `edit` names, with every slice sorted and each id once.
fn changed_only_as_named(before: &Program, after: &Program, edit: &Edited) -> bool {
    let (mut expected, got) = (ProgramPlan::from(before.clone()), ProgramPlan::from(after.clone()));
    let ascending = |ids: &[usize]| ids.windows(2).all(|w| w[0] < w[1]);
    let named = |cols: &[Col]| {
        ascending(&cols.iter().map(|c| c.index()).collect::<Vec<_>>())
            && cols.iter().all(|c| c.index() < got.columns.len())
    };
    match edit {
        Edited::ColBounds(cols) if named(cols) => cols
            .iter()
            .for_each(|&c| expected.columns[c].bounds = got.columns[c].bounds),
        Edited::ColCosts(cols) if named(cols) => cols
            .iter()
            .for_each(|&c| expected.columns[c].cost = got.columns[c].cost),
        Edited::ColIntegrality(cols) if named(cols) => {
            cols.iter()
                .for_each(|&c| expected.columns[c].integrality = got.columns[c].integrality);
        }
        Edited::RowBounds(rows) if ascending(&rows.iter().map(|r| r.index()).collect::<Vec<_>>()) => {
            if rows.iter().any(|r| r.index() >= got.rows.len()) {
                return false;
            }
            rows.iter().for_each(|&r| expected.rows[r].bounds = got.rows[r].bounds);
        }
        Edited::NewRows(rows) => {
            let start = expected.rows.len();
            if rows.iter().map(|r| r.index()).ne(start..got.rows.len()) {
                return false;
            }
            rows.iter().for_each(|&r| expected.rows.push(got.rows[r].clone()));
        }
        Edited::NewCols(cols) => return appended_columns(&mut expected, &got, cols) && expected == got,
        Edited::Sense => expected.sense = got.sense,
        Edited::Offset => expected.offset = got.offset,
        Edited::Objective => {
            expected.sense = got.sense;
            expected.offset = got.offset;
            expected.quadratic.clone_from(&got.quadratic);
            if expected.columns.len() != got.columns.len() {
                return false;
            }
            for (column, new) in expected.columns.iter_mut().zip(&got.columns) {
                column.cost = new.cost;
            }
        }
        // A guard above failed: the ids are unsorted, repeated or beyond the program.
        _ => return false,
    }
    expected == got
}

/// Appends to `expected` the columns `cols` of `got` and their terms in `expected`'s rows, and whether `cols` are
/// exactly the columns `got` has beyond `expected`'s.
fn appended_columns(expected: &mut ProgramPlan, got: &ProgramPlan, cols: &[Col]) -> bool {
    let start = expected.columns.len();
    if cols.iter().map(|c| c.index()).ne(start..got.columns.len()) || expected.rows.len() != got.rows.len() {
        return false;
    }
    cols.iter().for_each(|&c| expected.columns.push(got.columns[c].clone()));
    for (row, terms) in expected.rows.iter_mut().zip(got.rows.iter().map(|r| &r.terms)) {
        row.terms.extend(terms.iter().filter(|(col, _)| col.index() >= start));
    }
    true
}

/// A mirror's hooks, relayed to the inner backend: the context types are the same.
struct Forward<'f, 'h, B: Backend>(&'f mut Hooks<'h, Mirror<B>>);

impl<B: Backend> fmt::Debug for Forward<'_, '_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Forward").field(&self.0).finish()
    }
}

impl<B: Backend> Callback<B> for Forward<'_, '_, B> {
    fn events(&self) -> Events {
        self.0.events()
    }
    fn log(&mut self, ctx: &mut B::Log<'_>) {
        self.0.log(ctx);
    }
    fn progress(&mut self, ctx: &mut B::Progress<'_>) {
        self.0.progress(ctx);
    }
    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        self.0.poll(ctx);
    }
    fn feasible(&mut self, ctx: &mut B::Solution<'_>) {
        self.0.feasible(ctx);
    }
    fn incumbent(&mut self, ctx: &mut B::Solution<'_>) {
        self.0.incumbent(ctx);
    }
    fn native(&mut self, ctx: &mut B::Native<'_>) {
        self.0.native(ctx);
    }
}

/// Each search event goes to the mirror's own hooks, which have the capability exactly when `B` has it.
impl<B: Backend> Relay<B> for Forward<'_, '_, B> {
    fn wants_lazy(&self) -> bool
    where
        B: LazyConstraints,
    {
        self.0.wants_lazy()
    }
    fn lazy(&mut self, ctx: &mut <B as LazyConstraints>::Lazy<'_>)
    where
        B: LazyConstraints,
    {
        self.0.lazy(ctx);
    }
    fn wants_cut(&self) -> bool
    where
        B: UserCuts,
    {
        self.0.wants_cut()
    }
    fn cut(&mut self, ctx: &mut <B as UserCuts>::Cut<'_>)
    where
        B: UserCuts,
    {
        self.0.cut(ctx);
    }
    fn wants_heuristic(&self) -> bool
    where
        B: Heuristics,
    {
        self.0.wants_heuristic()
    }
    fn heuristic(&mut self, ctx: &mut <B as Heuristics>::Heuristic<'_>)
    where
        B: Heuristics,
    {
        self.0.heuristic(ctx);
    }
    fn wants_price(&self) -> bool
    where
        B: Prices,
    {
        self.0.wants_price()
    }
    fn price(&mut self, ctx: &mut <B as Prices>::Price<'_>)
    where
        B: Prices,
    {
        self.0.price(ctx);
    }
}

/// Every declaration is the inner backend's; every call is forwarded, recorded and checked.
impl<B: Backend> Backend for Mirror<B> {
    type Error = MirrorError<B::Error>;
    type Status = B::Status;
    type Log<'a> = B::Log<'a>;
    type Progress<'a> = B::Progress<'a>;
    type Poll<'a> = B::Poll<'a>;
    type Solution<'a> = B::Solution<'a>;
    type Native<'a> = B::Native<'a>;

    const EVENTS: Events = B::EVENTS;
    const CLASSES: ModelClasses = B::CLASSES;
    const FEATURES: ProgramFeatures = B::FEATURES;

    fn load(&mut self, program: &Program) -> Result<(), Self::Error> {
        self.calls.push(Call::Load(program.clone()));
        self.held = None;
        if fire(&mut self.faults.refuse_loads) {
            return Err(MirrorError::Injected);
        }
        self.inner.load(program).map_err(MirrorError::Inner)?;
        self.held = Some(program.clone());
        Ok(())
    }

    /// Checks the contract, forwards the edit, and injects the faults due.
    ///
    /// # Panics
    ///
    /// When the mirror holds no program, when `program` is not the held program changed only as `edit` names with
    /// sorted ids each once, when its class or features are beyond the declarations, and when a
    /// [`Faults::panic_edits`] count fires.
    fn edit(&mut self, program: &Program, edit: &Edit) -> Result<(), Self::Error> {
        let edited = Edited::of(edit);
        self.calls.push(Call::Edit(edited.clone(), program.clone()));
        let before = self.held.take().expect("Mirror: an edit while holding no program");
        assert!(
            changed_only_as_named(&before, program, &edited),
            "Mirror: {edited:?} changed more than it names, or names ids out of order"
        );
        assert!(
            B::CLASSES.contains(program.class()),
            "Mirror: an edit into an undeclared class"
        );
        assert!(
            B::FEATURES.contains_all(program.features()),
            "Mirror: an edit into an undeclared feature"
        );
        self.inner.edit(program, edit).map_err(MirrorError::Inner)?;
        if fire(&mut self.faults.panic_edits) {
            panic!("Mirror: an injected panic");
        }
        if fire(&mut self.faults.refuse_edits) {
            return Err(MirrorError::Injected);
        }
        self.held = Some(program.clone());
        Ok(())
    }

    /// Relays the hooks to the inner backend, the sink of a pricer with them.
    fn solve(&mut self, settings: &Settings, mut hooks: Hooks<'_, Self>) -> Result<Solved<Self::Status>, Self::Error> {
        let priced = hooks.take_priced();
        let mut forward = Forward(&mut hooks);
        self.inner
            .solve(settings, Hooks::relay(&mut forward, priced))
            .map_err(MirrorError::Inner)
    }
}

/// The faults are the mirror's options, so a test changes them between calls through the `Solver`.
impl<B: Backend> Configurable for Mirror<B> {
    type Options = Faults;

    fn options(&self) -> &Faults {
        &self.faults
    }

    fn options_mut(&mut self) -> &mut Faults {
        &mut self.faults
    }
}

/// Every search capability is the inner backend's, with its contexts; [`Mirror::solve`](Backend::solve) relays the
/// handlers.
impl<B: LazyConstraints> LazyConstraints for Mirror<B> {
    type Lazy<'a> = B::Lazy<'a>;
}

impl<B: UserCuts> UserCuts for Mirror<B> {
    type Cut<'a> = B::Cut<'a>;
}

impl<B: Heuristics> Heuristics for Mirror<B> {
    type Heuristic<'a> = B::Heuristic<'a>;
}

impl<B: Prices> Prices for Mirror<B> {
    type Price<'a> = B::Price<'a>;
}

/// Every other capability is the inner backend's, forwarded with its errors as [`MirrorError::Inner`].
impl<B: Starts> Starts for Mirror<B> {
    fn set_start(&mut self, start: &Start) -> Result<(), Self::Error> {
        self.inner.set_start(start).map_err(MirrorError::Inner)
    }
}

impl<B: DualAnalysis> DualAnalysis for Mirror<B> {
    fn duals(&self, program: &Program) -> Result<Duals, Self::Error> {
        self.inner.duals(program).map_err(MirrorError::Inner)
    }
}

impl<B: BasisAnalysis> BasisAnalysis for Mirror<B> {
    fn basis(&self, program: &Program) -> Result<Option<Basis>, Self::Error> {
        self.inner.basis(program).map_err(MirrorError::Inner)
    }

    fn set_basis(&mut self, program: &Program, basis: &Basis) -> Result<(), Self::Error> {
        self.inner.set_basis(program, basis).map_err(MirrorError::Inner)
    }
}

impl<B: RangingAnalysis> RangingAnalysis for Mirror<B> {
    fn ranging(&mut self, program: &Program) -> Result<Ranging, Self::Error> {
        self.inner.ranging(program).map_err(MirrorError::Inner)
    }
}

impl<B: RayAnalysis> RayAnalysis for Mirror<B> {
    fn primal_ray(&mut self, program: &Program) -> Result<Option<TiVec<Col, f64>>, Self::Error> {
        self.inner.primal_ray(program).map_err(MirrorError::Inner)
    }

    fn dual_ray(&mut self, program: &Program) -> Result<Option<TiVec<Row, f64>>, Self::Error> {
        self.inner.dual_ray(program).map_err(MirrorError::Inner)
    }
}

impl<B: IisAnalysis> IisAnalysis for Mirror<B> {
    fn iis(&mut self, program: &Program) -> Result<Option<Iis>, Self::Error> {
        self.inner.iis(program).map_err(MirrorError::Inner)
    }
}

impl<B: RelaxationAnalysis> RelaxationAnalysis for Mirror<B> {
    fn relax(&mut self, program: &Program, weights: &Weights) -> Result<Option<Relaxed>, Self::Error> {
        self.inner.relax(program, weights).map_err(MirrorError::Inner)
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use culmen_core::backend::Found;
    use culmen_core::script::{Script, Step};
    use culmen_core::{EditError, LazyContext, LazyHandler, Linear, Model, Outcome, Phase, Progress, Solver, Var};

    use super::*;

    /// Minimize `x + y` over `x`, `y` in `[0, 4]` and the row `x + y >= 1`, and the same with `y` costing 3.
    fn programs() -> (Program, Program, Col) {
        let mut m = Model::new();
        let x = m.add_var(Var::continuous().upper(4));
        let y = m.add_var(Var::continuous().upper(4));
        m.add_constraint((x + y).geq(1));
        m.minimize(x + y);
        let before = m.build().unwrap();
        m.minimize(x + 3.0 * y);
        (before, m.build().unwrap(), y)
    }

    #[test]
    fn test_a_mirror_holds_what_it_loaded_and_what_it_was_edited_into() {
        let (before, after, y) = programs();
        let mut solver = Solver::new(Mirror::new(Script::optimal(0.0, vec![])), before.clone()).unwrap();
        assert_eq!(solver.backend().held(), Some(&before));
        solver.set_col_costs([(y, 3.0)]).unwrap();
        assert_eq!(solver.backend().held(), Some(&after));
        assert_eq!(
            solver.backend().calls,
            [Call::Load(before), Call::Edit(Edited::ColCosts(vec![y]), after)]
        );
    }

    #[test]
    fn test_a_mirror_forwards_a_start_to_its_inner_backend() {
        let (before, _, y) = programs();
        let mut solver = Solver::new(Mirror::new(Script::optimal(0.0, vec![])), before).unwrap();
        solver.run().start(vec![(y, 1.0)]).solve().unwrap();
        assert_eq!(solver.backend().inner.record.starts, [Start::Sparse(vec![(y, 1.0)])]);
    }

    /// Cuts off every candidate by a constraint on its first column, counting its calls.
    struct FirstAtZero(usize);

    impl<B: LazyConstraints> LazyHandler<B> for FirstAtZero {
        fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
            self.0 += 1;
            ctx.add_lazy(Col::new(0).leq(0)).unwrap();
        }
    }

    /// Counts the polls it sees.
    struct Polls(usize);

    impl<B: Backend> Callback<B> for Polls {
        fn poll(&mut self, _: &mut B::Poll<'_>) {
            self.0 += 1;
        }
    }

    #[test]
    fn test_a_mirror_relays_its_callback_and_handlers_to_its_inner_backend() {
        let (before, ..) = programs();
        let at = Progress::at(Duration::ZERO, Phase::Other);
        let candidate = Found {
            progress: at,
            objective: 0.0,
            values: vec![1.0, 0.0].into(),
        };
        let script = Script::new([Step::Poll(at), Step::Lazy(candidate)], Outcome::Infeasible);
        let mut solver = Solver::new(Mirror::new(script), before).unwrap();
        let (mut polls, mut lazy) = (Polls(0), FirstAtZero(0));
        solver.run().callback(&mut polls).lazy(&mut lazy).solve().unwrap();
        assert_eq!((polls.0, lazy.0), (1, 1));
        assert_eq!(solver.backend().inner.record.lazy.len(), 1);

        // Without a handler, the inner backend sees none either.
        solver.solve().unwrap();
        assert_eq!(solver.backend().inner.record.lazy.len(), 1);
    }

    #[test]
    fn test_a_mirror_finds_an_edit_that_changed_what_it_does_not_name() {
        // Only a solver makes an `Edit`, and it never names one wrongly, so the check is tested on its own.
        let (before, after, y) = programs();
        assert!(!changed_only_as_named(
            &before,
            &after,
            &Edited::ColBounds(vec![Col::new(0)])
        ));
        assert!(changed_only_as_named(&before, &after, &Edited::ColCosts(vec![y])));
    }

    #[test]
    fn test_a_mirror_injects_its_faults_and_then_holds_nothing() {
        let (before, after, y) = programs();
        let mut mirror = Mirror::new(Script::optimal(0.0, vec![]));
        mirror.faults.refuse_loads = 1;
        assert_eq!(mirror.load(&before), Err(MirrorError::Injected));
        assert_eq!(mirror.held(), None);
        let mut solver = Solver::new(mirror, before.clone()).unwrap();
        *solver.options_mut() = Faults {
            refuse_edits: 1,
            refuse_loads: 1,
            ..Faults::default()
        };
        assert_eq!(
            solver.set_col_costs([(y, 3.0)]),
            Err(EditError::Backend(MirrorError::Injected))
        );
        assert_eq!(solver.backend().held(), None);
        assert_eq!(solver.backend().faults, Faults::default());
        assert!(
            solver
                .backend()
                .calls
                .contains(&Call::Edit(Edited::ColCosts(vec![y]), after))
        );
    }
}
