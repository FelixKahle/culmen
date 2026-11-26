// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constraint handlers driven by the scripted backend, and the rules every backend's node record checks.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};

use proptest::prelude::*;

use culmen_core::{
    ActionError, AddedRow, Bound, Bounds, CheckContext, CheckedChild, Child, Col, ConstraintHandler, CustomConstraints,
    CutContext, CutHandler, Declaration, EnforceContext, ExplainContext, Explained, Found, Inferred, Involved,
    LazyConstraints, LazyContext, LazyHandler, Linear, Lock, Locks, Model, NodeAction, NodeActions, NodeContext,
    Outcome, Phase, PresolveContext, PresolveRound, Progress, Reason, Script, Solver, Step, TiVec, UserCuts, Var,
    Verdict, bounds,
};

/// The most calls a test handler takes of one method; a scripted solve makes a handful.
const MAX_CALLS: u32 = 100;

/// Counts one call, and fails a handler that runs without end.
fn count(calls: &mut u32) {
    *calls += 1;
    assert!(
        *calls <= MAX_CALLS,
        "a test handler was called more than {MAX_CALLS} times"
    );
}

fn at() -> Progress {
    Progress::at(Duration::ZERO, Phase::Mip)
}

fn found(values: &[f64]) -> Found {
    Found {
        progress: at(),
        objective: 0.0,
        values: values.to_vec().into(),
    }
}

/// `n` columns, each in `bounds`.
fn domain(n: usize, bounds: Bounds) -> TiVec<Col, Bounds> {
    vec![bounds; n].into()
}

/// A solver over `n` integer columns in `[0, 3]`, replaying `steps`.
fn solver(n: usize, steps: Vec<Step>) -> (Solver<Script>, Vec<Col>) {
    let mut m = Model::new();
    let cols = m.add_vars(n, |_| Var::integer().bounds(0.0, 3.0));
    m.minimize(cols[0]);
    let solver = Solver::new(Script::new(steps, Outcome::Infeasible), m.build().unwrap()).unwrap();
    (solver, cols.to_vec())
}

/// `x` only together with `y`: rejects `x > y` and resolves it with the constraint `x <= y`.
struct OnlyWith {
    x: Col,
    y: Col,
    checks: u32,
    enforces: u32,
}

impl<B: CustomConstraints> ConstraintHandler<B> for OnlyWith {
    fn check(&mut self, ctx: &mut B::Check<'_>) -> Verdict {
        count(&mut self.checks);
        let point = ctx.candidate();
        if point[self.x] > point[self.y] + 0.5 {
            Verdict::Reject
        } else {
            Verdict::Accept
        }
    }

    fn enforce(&mut self, ctx: &mut B::Enforce<'_>) {
        count(&mut self.enforces);
        let point = ctx.candidate();
        assert!(point[self.x] > point[self.y], "enforce sees only a rejected point");
        ctx.add_constraint(self.x.leq(self.y)).unwrap();
    }
}

#[test]
fn test_an_enforce_step_checks_first_and_enforces_only_a_rejected_point() {
    let node = domain(2, bounds!(0.0, 1.0));
    let steps = vec![
        Step::Check(found(&[0.0, 1.0])),
        Step::Enforce {
            found: found(&[1.0, 1.0]),
            bounds: node.clone(),
        },
        Step::Enforce {
            found: found(&[1.0, 0.0]),
            bounds: node,
        },
    ];
    let (mut solver, cols) = solver(2, steps);
    let (x, y) = (cols[0], cols[1]);
    let mut rule = OnlyWith {
        x,
        y,
        checks: 0,
        enforces: 0,
    };
    solver.run().constraint(&mut rule).solve().unwrap();

    assert_eq!((rule.checks, rule.enforces), (3, 1));
    let record = &solver.backend().record;
    assert_eq!(record.declarations, [Declaration::default()]);
    assert_eq!(record.verdicts, [Verdict::Accept, Verdict::Accept, Verdict::Reject]);
    assert_eq!(record.enforced.len(), 1);
    let x_minus_y = AddedRow {
        terms: vec![(x, 1.0), (y, -1.0)],
        bounds: bounds!(f64::NEG_INFINITY, 0.0),
    };
    assert_eq!(record.enforced[0].constraints(), [x_minus_y]);
    assert_eq!(record.enforced[0].action(), Some(NodeAction::Reduce));
}

/// Involves only `x`; at a node it reads the point and bounds, then tries each action and keeps what it was told.
struct Probe {
    x: Col,
    y: Col,
    calls: u32,
    answers: Vec<Result<(), ActionError>>,
}

impl ConstraintHandler<Script> for Probe {
    fn declaration(&self) -> Declaration {
        Declaration {
            locks: Locks::Only(vec![(self.x, Lock::Both)]),
            ..Declaration::default()
        }
    }

    fn check(&mut self, _: &mut <Script as CustomConstraints>::Check<'_>) -> Verdict {
        count(&mut self.calls);
        Verdict::Reject
    }

    fn enforce(&mut self, ctx: &mut <Script as CustomConstraints>::Enforce<'_>) {
        count(&mut self.calls);
        assert_eq!(ctx.candidate().raw, [2.0, 1.0]);
        assert_eq!(ctx.objective(), 0.0);
        assert_eq!(ctx.bounds(self.x), Some(bounds!(0.0, 3.0)));
        assert_eq!(ctx.bounds(Col::new(5)), None);
        let unknown = Col::new(9);
        self.answers = vec![
            ctx.tighten(self.x, bounds!(1.0, 3.0), Some(Reason(7))),
            ctx.tighten(self.y, bounds!(1.0, 3.0), None),
            ctx.tighten(unknown, bounds!(1.0, 3.0), None),
            ctx.add_constraint(unknown.leq(1)),
            ctx.add_constraint((self.x + self.y).geq(4)),
            ctx.branch(vec![Child {
                bounds: vec![(self.x, bounds!(0.0, 1.0))],
                ..Child::default()
            }]),
        ];
    }
}

#[test]
fn test_an_enforcement_reads_the_scripted_node_and_its_refused_actions_record_nothing() {
    let steps = vec![Step::Enforce {
        found: found(&[2.0, 1.0]),
        bounds: domain(2, bounds!(0.0, 3.0)),
    }];
    let (mut solver, cols) = solver(2, steps);
    let (x, y) = (cols[0], cols[1]);
    let mut probe = Probe {
        x,
        y,
        calls: 0,
        answers: Vec::new(),
    };
    solver.run().constraint(&mut probe).solve().unwrap();

    let unknown = Col::new(9);
    assert_eq!(
        probe.answers,
        [
            Ok(()),
            Err(ActionError::NotInvolved { col: y }),
            Err(ActionError::UnknownColumn { col: unknown }),
            Err(ActionError::UnknownColumn { col: unknown }),
            Ok(()),
            Err(ActionError::Exclusive {
                action: NodeAction::Branch,
                after: NodeAction::Reduce
            }),
        ]
    );
    let actions = &solver.backend().record.enforced[0];
    assert_eq!(actions.tightenings(), [(x, bounds!(1.0, 3.0), Some(Reason(7)))]);
    let at_least_four = AddedRow {
        terms: vec![(x, 1.0), (y, 1.0)],
        bounds: bounds!(4.0, f64::INFINITY),
    };
    assert_eq!(actions.constraints(), [at_least_four]);
    assert_eq!(actions.children(), None);
    assert_eq!(actions.cutoff(), None);
}

/// Branches every rejected point into the children it is given, then tries more, keeping what it was told.
struct Brancher {
    children: Vec<Vec<Child>>,
    calls: u32,
    answers: Vec<Result<(), ActionError>>,
}

impl<B: CustomConstraints> ConstraintHandler<B> for Brancher {
    fn check(&mut self, _: &mut B::Check<'_>) -> Verdict {
        count(&mut self.calls);
        Verdict::Reject
    }

    fn enforce(&mut self, ctx: &mut B::Enforce<'_>) {
        count(&mut self.calls);
        let x = Col::new(0);
        let children = self.children.remove(0);
        self.answers.push(ctx.branch(children));
        let fallback = Child {
            bounds: vec![(x, bounds!(0.0, 1.0))],
            ..Child::default()
        };
        self.answers.push(ctx.branch(vec![fallback]));
        self.answers.push(ctx.tighten(x, bounds!(0.0, 1.0), None));
    }
}

#[test]
fn test_a_branching_records_its_children_checked_against_the_node() {
    let node = domain(2, bounds!(0.0, 3.0));
    let step = || Step::Enforce {
        found: found(&[0.0, 0.0]),
        bounds: node.clone(),
    };
    let (mut solver, cols) = solver(2, vec![step(), step(), step()]);
    let (x, y) = (cols[0], cols[1]);
    let narrow = |col, bounds| Child {
        bounds: vec![(col, bounds)],
        ..Child::default()
    };
    let given = vec![
        narrow(x, bounds!(0.0, 1.0)),
        Child {
            bounds: vec![(x, bounds!(2.0, 5.0)), (x, bounds!(-1.0, 2.5))],
            ..Child::default()
        },
        Child {
            bounds: vec![(y, bounds!(0.0, 3.0))],
            constraints: vec![(x + y).geq(1)],
        },
        narrow(x, bounds!(5.0, 6.0)),
    ];
    let mut brancher = Brancher {
        children: vec![
            given,
            vec![narrow(x, bounds!(0.0, 1.0)), narrow(y, bounds!(-1.0, 4.0))],
            Vec::new(),
        ],
        calls: 0,
        answers: Vec::new(),
    };
    solver.run().constraint(&mut brancher).solve().unwrap();

    let exclusive = |after| {
        Err(ActionError::Exclusive {
            action: NodeAction::Branch,
            after,
        })
    };
    let reduce_after_branch = || {
        Err(ActionError::Exclusive {
            action: NodeAction::Reduce,
            after: NodeAction::Branch,
        })
    };
    assert_eq!(
        brancher.answers,
        [
            Ok(()),
            exclusive(NodeAction::Branch),
            reduce_after_branch(),
            Err(ActionError::NoProgress { child: 1 }),
            Ok(()),
            reduce_after_branch(),
            Err(ActionError::NoChildren),
            Ok(()),
            reduce_after_branch(),
        ]
    );
    let enforced = &solver.backend().record.enforced;
    // The fourth child misses the node, so it is empty and left out.
    let expected = [
        CheckedChild {
            bounds: vec![(x, bounds!(0.0, 1.0))],
            constraints: vec![],
        },
        CheckedChild {
            bounds: vec![(x, bounds!(2.0, 2.5))],
            constraints: vec![],
        },
        CheckedChild {
            bounds: vec![],
            constraints: vec![AddedRow {
                terms: vec![(x, 1.0), (y, 1.0)],
                bounds: bounds!(1.0, f64::INFINITY),
            }],
        },
    ];
    assert_eq!(enforced[0].children(), Some(&expected[..]));
    // A refused branching records nothing, so the fallback of each later call is the one recorded.
    for actions in &enforced[1..] {
        assert_eq!(actions.children(), Some(&expected[..1]));
        assert_eq!(actions.action(), Some(NodeAction::Branch));
    }
}

/// Tightens with a reason, cuts the node off naming bounds, and tries more afterwards.
struct CutAfterTightening {
    x: Col,
    calls: u32,
    answers: Vec<Result<(), ActionError>>,
}

impl<B: CustomConstraints> ConstraintHandler<B> for CutAfterTightening {
    fn check(&mut self, _: &mut B::Check<'_>) -> Verdict {
        count(&mut self.calls);
        Verdict::Accept
    }

    fn propagate(&mut self, ctx: &mut B::Propagate<'_>) {
        count(&mut self.calls);
        assert_eq!(ctx.bounds(self.x), Some(bounds!(0.0, 3.0)));
        self.answers = vec![
            ctx.tighten(self.x, bounds!(0.0, 2.0), Some(Reason(2))),
            ctx.cut_off_because(&[(Col::new(4), Bound::Lower)]),
            ctx.cut_off_because(&[(self.x, Bound::Upper)]),
            ctx.tighten(self.x, bounds!(0.0, 1.0), None),
            ctx.cut_off_because(&[]),
        ];
        ctx.cut_off();
    }
}

#[test]
fn test_a_propagation_may_cut_off_after_tightening_and_nothing_follows_the_cutoff() {
    let steps = vec![Step::Propagate {
        progress: at(),
        bounds: domain(2, bounds!(0.0, 3.0)),
    }];
    let (mut solver, cols) = solver(2, steps);
    let x = cols[0];
    let mut rule = CutAfterTightening {
        x,
        calls: 0,
        answers: Vec::new(),
    };
    solver.run().constraint(&mut rule).solve().unwrap();

    let after_cutoff = |action| {
        Err(ActionError::Exclusive {
            action,
            after: NodeAction::CutOff,
        })
    };
    assert_eq!(
        rule.answers,
        [
            Ok(()),
            Err(ActionError::UnknownColumn { col: Col::new(4) }),
            Ok(()),
            after_cutoff(NodeAction::Reduce),
            after_cutoff(NodeAction::CutOff),
        ]
    );
    let propagated = &solver.backend().record.propagated[0];
    assert_eq!(propagated.tightenings(), [(x, bounds!(0.0, 2.0), Some(Reason(2)))]);
    assert_eq!(propagated.cutoff(), Some(&[(x, Bound::Upper)][..]));
    assert_eq!(propagated.action(), Some(NodeAction::CutOff));
}

/// The column at position `reason` was fixed, which pushed the others off its value: explained by its two bounds.
struct FixedPushes {
    cols: Vec<Col>,
    calls: u32,
    refused: Option<ActionError>,
}

impl<B: CustomConstraints> ConstraintHandler<B> for FixedPushes {
    fn check(&mut self, _: &mut B::Check<'_>) -> Verdict {
        count(&mut self.calls);
        Verdict::Accept
    }

    fn explain(&mut self, ctx: &mut B::Explain<'_>) -> Explained {
        count(&mut self.calls);
        let Some(&a) = usize::try_from(ctx.reason().0).ok().and_then(|i| self.cols.get(i)) else {
            return Explained::Unknown;
        };
        assert_eq!(ctx.bounds_before(a), Some(bounds!(1.0, 1.0)));
        assert_eq!(ctx.bounds_before(Col::new(8)), None);
        assert_eq!(ctx.inferred().end, Bound::Lower);
        self.refused = ctx.because(Col::new(8), Bound::Lower).err();
        match ctx.because(a, Bound::Lower).and_then(|()| ctx.because(a, Bound::Upper)) {
            Ok(()) => Explained::Complete,
            Err(_) => Explained::Unknown,
        }
    }
}

#[test]
fn test_an_explanation_names_bounds_before_the_tightening_and_is_recorded_with_its_answer() {
    let explain = |reason| Step::Explain {
        progress: at(),
        inferred: Inferred {
            col: Col::new(1),
            end: Bound::Lower,
            value: 2.0,
        },
        reason: Reason(reason),
        before: vec![bounds!(1.0, 1.0), bounds!(1.0, 3.0)].into(),
    };
    let (mut solver, cols) = solver(2, vec![explain(0), explain(5)]);
    let x = cols[0];
    let mut rule = FixedPushes {
        cols,
        calls: 0,
        refused: None,
    };
    solver.run().constraint(&mut rule).solve().unwrap();

    assert_eq!(rule.refused, Some(ActionError::UnknownColumn { col: Col::new(8) }));
    assert_eq!(
        solver.backend().record.explained,
        [
            (Explained::Complete, vec![(x, Bound::Lower), (x, Bound::Upper)]),
            (Explained::Unknown, vec![]),
        ]
    );
}

/// No two of `cols` take the same value; presolve removes a fixed column's value from the lower ends of the others,
/// adds the sum every all-different point reaches, and finds two columns fixed at one value infeasible.
struct AllDifferent {
    cols: Vec<Col>,
    outside: Col,
    calls: u32,
    refused: Option<ActionError>,
}

impl<B: CustomConstraints> ConstraintHandler<B> for AllDifferent {
    fn declaration(&self) -> Declaration {
        let locks = self.cols.iter().map(|&col| (col, Lock::Both)).collect();
        Declaration {
            locks: Locks::Only(locks),
            ..Declaration::default()
        }
    }

    fn check(&mut self, _: &mut B::Check<'_>) -> Verdict {
        count(&mut self.calls);
        Verdict::Accept
    }

    fn presolve(&mut self, ctx: &mut B::Presolve<'_>) {
        count(&mut self.calls);
        self.refused = ctx.tighten(self.outside, bounds!(1.0, 3.0)).err();
        for &a in &self.cols {
            let fixed = ctx.bounds(a).unwrap();
            if fixed.lower() != fixed.upper() {
                continue;
            }
            for &b in self.cols.iter().filter(|&&b| b != a) {
                let other = ctx.bounds(b).unwrap();
                if other.lower() == fixed.lower() && other.upper() == fixed.lower() {
                    ctx.infeasible();
                    return;
                }
                if other.lower() == fixed.lower() {
                    let raised = Bounds::new(fixed.lower() + 1.0, other.upper()).unwrap();
                    ctx.tighten(b, raised).unwrap();
                }
            }
        }
        let sum: culmen_core::LinExpr = self.cols.iter().copied().sum();
        ctx.add_constraint(sum.geq(6)).unwrap();
    }
}

#[test]
fn test_a_presolve_tightens_the_global_bounds_of_involved_columns_and_declares_infeasibility() {
    // Four columns in [0, 3] with the first fixed at 0, and a fifth the rule ignores.
    let mut global = domain(5, bounds!(0.0, 3.0));
    global[Col::new(0)] = bounds!(0.0, 0.0);
    let mut clash = global.clone();
    clash[Col::new(1)] = bounds!(0.0, 0.0);
    let steps = vec![
        Step::Presolve {
            progress: at(),
            bounds: global,
        },
        Step::Presolve {
            progress: at(),
            bounds: clash,
        },
    ];
    let (mut solver, cols) = solver(5, steps);
    let mut rule = AllDifferent {
        cols: cols[..4].to_vec(),
        outside: cols[4],
        calls: 0,
        refused: None,
    };
    solver.run().constraint(&mut rule).solve().unwrap();

    assert_eq!(rule.refused, Some(ActionError::NotInvolved { col: cols[4] }));
    let presolved = &solver.backend().record.presolved;
    assert_eq!(
        presolved[0].tightenings,
        [
            (cols[1], bounds!(1.0, 3.0)),
            (cols[2], bounds!(1.0, 3.0)),
            (cols[3], bounds!(1.0, 3.0)),
        ]
    );
    assert_eq!(presolved[0].constraints.len(), 1);
    assert_eq!(presolved[0].constraints[0].bounds, bounds!(6.0, f64::INFINITY));
    assert!(!presolved[0].infeasible);
    assert!(presolved[1].infeasible);
    assert!(presolved[1].tightenings.is_empty());
}

/// Rejects every point and leaves every other method at its default.
struct RejectAll(u32);

impl<B: CustomConstraints> ConstraintHandler<B> for RejectAll {
    fn check(&mut self, _: &mut B::Check<'_>) -> Verdict {
        count(&mut self.0);
        Verdict::Reject
    }
}

fn every_constraint_step() -> Vec<Step> {
    let node = domain(1, bounds!(0.0, 3.0));
    vec![
        Step::Check(found(&[1.0])),
        Step::Enforce {
            found: found(&[1.0]),
            bounds: node.clone(),
        },
        Step::Propagate {
            progress: at(),
            bounds: node.clone(),
        },
        Step::Explain {
            progress: at(),
            inferred: Inferred {
                col: Col::new(0),
                end: Bound::Upper,
                value: 2.0,
            },
            reason: Reason(0),
            before: node.clone(),
        },
        Step::Presolve {
            progress: at(),
            bounds: node,
        },
    ]
}

#[test]
fn test_the_provided_methods_default_to_the_sound_answers() {
    let (mut solver, _) = solver(1, every_constraint_step());
    let mut rule = RejectAll(0);
    solver.run().constraint(&mut rule).solve().unwrap();

    assert_eq!(rule.0, 2);
    let record = &solver.backend().record;
    assert_eq!(record.declarations, [Declaration::default()]);
    assert_eq!(record.verdicts, [Verdict::Reject, Verdict::Reject]);
    assert_eq!(record.enforced.len(), 1);
    assert_eq!(record.enforced[0].action(), None);
    assert_eq!(record.propagated[0].action(), None);
    assert_eq!(record.explained, [(Explained::Unknown, vec![])]);
    assert_eq!(record.presolved, [PresolveRound::default()]);
}

#[test]
fn test_constraint_steps_reach_nothing_without_a_handler_and_are_no_observational_event() {
    assert!(every_constraint_step().iter().all(|step| step.event().is_none()));
    let (mut solver, _) = solver(1, every_constraint_step());
    solver.solve().unwrap();
    let record = &solver.backend().record;
    assert!(record.declarations.is_empty() && record.verdicts.is_empty());
    assert!(record.enforced.is_empty() && record.propagated.is_empty());
    assert!(record.explained.is_empty() && record.presolved.is_empty());
}

#[test]
fn test_a_later_constraint_registration_replaces_the_earlier() {
    let (mut solver, _) = solver(1, every_constraint_step());
    let (mut first, mut second) = (RejectAll(0), RejectAll(0));
    solver
        .run()
        .constraint(&mut first)
        .constraint(&mut second)
        .solve()
        .unwrap();
    assert_eq!((first.0, second.0), (0, 2));
}

#[test]
fn test_a_panicking_check_unwinds_out_of_the_solve_and_the_solver_solves_again() {
    struct Panics;
    impl<B: CustomConstraints> ConstraintHandler<B> for Panics {
        fn check(&mut self, _: &mut B::Check<'_>) -> Verdict {
            panic!("a defect in the rule");
        }
    }
    let (mut solver, _) = solver(1, every_constraint_step());
    let unwound = catch_unwind(AssertUnwindSafe(|| solver.run().constraint(&mut Panics).solve()));
    assert!(unwound.is_err());
    assert_eq!(solver.solve().unwrap().outcome, Outcome::Infeasible);
}

/// Counts what reaches it in each role.
#[derive(Default)]
struct Roles {
    checks: u32,
    cuts: u32,
    lazies: u32,
}

impl<B: CustomConstraints> ConstraintHandler<B> for Roles {
    fn check(&mut self, _: &mut B::Check<'_>) -> Verdict {
        count(&mut self.checks);
        Verdict::Accept
    }
}

impl<B: UserCuts> CutHandler<B> for Roles {
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        count(&mut self.cuts);
        assert_eq!(ctx.relaxation().len(), 1);
    }
}

impl<B: LazyConstraints> LazyHandler<B> for Roles {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        count(&mut self.lazies);
        assert_eq!(ctx.candidate().len(), 1);
    }
}

fn roles_steps() -> Vec<Step> {
    vec![
        Step::Check(found(&[1.0])),
        Step::Cut {
            progress: at(),
            relaxation: vec![0.5].into(),
        },
        Step::Lazy(found(&[1.0])),
    ]
}

/// What reached each handler, as `(checks, cuts, lazies)`.
fn seen(roles: &Roles) -> (u32, u32, u32) {
    (roles.checks, roles.cuts, roles.lazies)
}

#[test]
fn test_one_handler_in_constraint_and_cut_roles_takes_both_until_a_later_registration_takes_one() {
    let (mut solver, _) = solver(1, roles_steps());
    let mut both = Roles::default();
    solver.run().constraint_and_cut(&mut both).solve().unwrap();
    assert_eq!(seen(&both), (1, 1, 0));

    let (mut both, mut cuts) = (Roles::default(), Roles::default());
    solver
        .run()
        .constraint_and_cut(&mut both)
        .cut(&mut cuts)
        .solve()
        .unwrap();
    assert_eq!((seen(&both), seen(&cuts)), ((1, 0, 0), (0, 1, 0)));

    // Of two handlers in two roles each, the one registered last cuts.
    let (mut lazy, mut constraint) = (Roles::default(), Roles::default());
    solver
        .run()
        .lazy_and_cut(&mut lazy)
        .constraint_and_cut(&mut constraint)
        .solve()
        .unwrap();
    assert_eq!((seen(&lazy), seen(&constraint)), ((0, 0, 1), (1, 1, 0)));

    let (mut lazy, mut constraint) = (Roles::default(), Roles::default());
    solver
        .run()
        .constraint_and_cut(&mut constraint)
        .lazy_and_cut(&mut lazy)
        .solve()
        .unwrap();
    assert_eq!((seen(&lazy), seen(&constraint)), ((0, 1, 1), (1, 0, 0)));
}

/// One action on a node record, its arguments drawn so that each kind of refusal occurs.
#[derive(Clone, Debug)]
enum Act {
    /// The constraint `col <= 1`.
    Constraint(u8),
    Tighten(u8, Option<u32>),
    /// One child per entry: `None` narrows nothing, `Some(col)` narrows `col` to `[0, 1]`.
    Branch(Vec<Option<u8>>),
    CutOff,
    CutOffBecause(Vec<u8>),
}

/// Columns 0 and 1 are involved, column 2 is known but not involved, column 3 is unknown; the node is `[0, 3]`.
const NUM_COLS: usize = 3;

fn act() -> impl Strategy<Value = Act> {
    let col = 0u8..4;
    prop_oneof![
        col.clone().prop_map(Act::Constraint),
        (col.clone(), proptest::option::of(0u32..3)).prop_map(|(c, r)| Act::Tighten(c, r)),
        proptest::collection::vec(proptest::option::weighted(0.9, col.clone()), 0..3).prop_map(Act::Branch),
        Just(Act::CutOff),
        proptest::collection::vec(col, 0..3).prop_map(Act::CutOffBecause),
    ]
}

/// What the record should hold, kept from the rules' prose: kinds exclude each other, a cutoff ends the call, a
/// refused action records nothing.
#[derive(Default)]
struct Expected {
    constraints: Vec<AddedRow>,
    tightenings: Vec<(Col, Bounds, Option<Reason>)>,
    children: Option<Vec<CheckedChild>>,
    cutoff: Option<Vec<(Col, Bound)>>,
}

impl Expected {
    /// The kind recorded that excludes an action of kind `action`, if any.
    fn excluding(&self, action: NodeAction) -> Option<NodeAction> {
        let reduced = !self.constraints.is_empty() || !self.tightenings.is_empty();
        if self.cutoff.is_some() {
            Some(NodeAction::CutOff)
        } else if self.children.is_some() && action != NodeAction::CutOff {
            Some(NodeAction::Branch)
        } else if reduced && action == NodeAction::Branch {
            Some(NodeAction::Reduce)
        } else {
            None
        }
    }

    /// The refusal of a column for writing, if any.
    fn unwritable(col: u8) -> Option<ActionError> {
        let col = Col::new(usize::from(col));
        match col.index() {
            0 | 1 => None,
            2 => Some(ActionError::NotInvolved { col }),
            _ => Some(ActionError::UnknownColumn { col }),
        }
    }

    /// Takes `act` in and returns what the record should answer.
    fn take(&mut self, act: &Act) -> Result<(), ActionError> {
        let kind = match act {
            Act::Constraint(_) | Act::Tighten(..) => NodeAction::Reduce,
            Act::Branch(_) => NodeAction::Branch,
            Act::CutOff | Act::CutOffBecause(_) => NodeAction::CutOff,
        };
        if let Act::CutOff = act {
            self.cutoff.get_or_insert_with(Vec::new);
            return Ok(());
        }
        if let Some(after) = self.excluding(kind) {
            return Err(ActionError::Exclusive { action: kind, after });
        }
        match act {
            Act::Constraint(c) => {
                let col = Col::new(usize::from(*c));
                if col.index() >= NUM_COLS {
                    return Err(ActionError::UnknownColumn { col });
                }
                self.constraints.push(AddedRow {
                    terms: vec![(col, 1.0)],
                    bounds: bounds!(f64::NEG_INFINITY, 1.0),
                });
            }
            Act::Tighten(c, reason) => {
                if let Some(error) = Self::unwritable(*c) {
                    return Err(error);
                }
                self.tightenings
                    .push((Col::new(usize::from(*c)), bounds!(1.0, 2.0), reason.map(Reason)));
            }
            Act::Branch(children) => self.branch(children)?,
            Act::CutOffBecause(cols) => {
                if let Some(&c) = cols.iter().find(|&&c| usize::from(c) >= NUM_COLS) {
                    let col = Col::new(usize::from(c));
                    return Err(ActionError::UnknownColumn { col });
                }
                let reasons = cols.iter().map(|&c| (Col::new(usize::from(c)), Bound::Lower)).collect();
                self.cutoff = Some(reasons);
            }
            Act::CutOff => {}
        }
        Ok(())
    }

    fn branch(&mut self, children: &[Option<u8>]) -> Result<(), ActionError> {
        if children.is_empty() {
            return Err(ActionError::NoChildren);
        }
        let mut checked = Vec::new();
        for (index, child) in children.iter().enumerate() {
            let Some(c) = child else {
                return Err(ActionError::NoProgress { child: index });
            };
            if let Some(error) = Self::unwritable(*c) {
                return Err(error);
            }
            checked.push(CheckedChild {
                bounds: vec![(Col::new(usize::from(*c)), bounds!(0.0, 1.0))],
                constraints: vec![],
            });
        }
        self.children = Some(checked);
        Ok(())
    }
}

/// Takes `act` into the record under test.
fn record(actions: &mut NodeActions, act: &Act) -> Result<(), ActionError> {
    let col = |c: u8| Col::new(usize::from(c));
    let node = |col: Col| (col.index() < NUM_COLS).then_some(bounds!(0.0, 3.0));
    match act {
        Act::Constraint(c) => actions.add_constraint(&col(*c).leq(1)),
        Act::Tighten(c, reason) => actions.tighten(col(*c), bounds!(1.0, 2.0), reason.map(Reason)),
        Act::Branch(children) => {
            let children = children
                .iter()
                .map(|child| Child {
                    bounds: child.map(|c| vec![(col(c), bounds!(0.0, 1.0))]).unwrap_or_default(),
                    ..Child::default()
                })
                .collect();
            actions.branch(children, &node)
        }
        Act::CutOff => {
            actions.cut_off();
            Ok(())
        }
        Act::CutOffBecause(cols) => {
            let reasons: Vec<_> = cols.iter().map(|&c| (col(c), Bound::Lower)).collect();
            actions.cut_off_because(&reasons)
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Every sequence of actions is recorded as given, or refused at the first action the rules exclude, with
    /// nothing recorded for it.
    #[test]
    fn test_node_actions_record_every_action_the_rules_allow_and_nothing_of_a_refused_one(
        acts in proptest::collection::vec(act(), 0..8),
    ) {
        let involved = Involved::new(&Locks::Only(vec![(Col::new(0), Lock::Up), (Col::new(1), Lock::Down)]), NUM_COLS);
        let mut actions = NodeActions::new(&involved, NUM_COLS);
        let mut expected = Expected::default();
        for act in &acts {
            prop_assert_eq!(record(&mut actions, act), expected.take(act), "{:?}", act);
            prop_assert_eq!(actions.constraints(), &expected.constraints[..]);
            prop_assert_eq!(actions.tightenings(), &expected.tightenings[..]);
            prop_assert_eq!(actions.children(), expected.children.as_deref());
            prop_assert_eq!(actions.cutoff(), expected.cutoff.as_deref());
        }
    }
}

#[test]
fn test_the_kind_that_decides_a_node_is_the_cutoff_then_the_branching_then_the_reductions() {
    let involved = Involved::new(&Locks::All, 1);
    let x = Col::new(0);
    let mut actions = NodeActions::new(&involved, 1);
    assert_eq!(actions.action(), None);
    actions.tighten(x, bounds!(0.0, 1.0), None).unwrap();
    assert_eq!(actions.action(), Some(NodeAction::Reduce));
    actions.cut_off();
    assert_eq!(actions.action(), Some(NodeAction::CutOff));

    let mut actions = NodeActions::new(&involved, 1);
    let child = Child {
        bounds: vec![(x, bounds!(0.0, 1.0))],
        ..Child::default()
    };
    actions.branch(vec![child], &|_| Some(bounds!(0.0, 3.0))).unwrap();
    assert_eq!(actions.action(), Some(NodeAction::Branch));
}

#[test]
fn test_action_errors_of_a_node_print_what_was_refused() {
    let exclusive = ActionError::Exclusive {
        action: NodeAction::Branch,
        after: NodeAction::Reduce,
    };
    assert_eq!(
        exclusive.to_string(),
        "callback action: branching is refused after reducing in the same call"
    );
    assert_eq!(
        ActionError::NoProgress { child: 2 }.to_string(),
        "callback action: child 2 of the branching narrows no bound of the node and adds no constraint"
    );
    assert_eq!(
        ActionError::NoChildren.to_string(),
        "callback action: a branching needs at least one child"
    );
    assert_eq!(
        ActionError::NotInvolved { col: Col::new(1) }.to_string(),
        "callback action: Col(1) is not a column the constraint handler's declaration involves"
    );
    assert_eq!(NodeAction::CutOff.to_string(), "cutting off");
}
