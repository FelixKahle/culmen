// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The solver-neutral analysis layer: the checks every request passes before a backend is asked, the checked
//! penalty, basis fitting, violations against a simple reference, and serde round trips.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::cell::RefCell;
use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};

use culmen_core::{
    Analysis, AnalysisError, Backend, Basis, BasisAnalysis, BasisStatus, Bound, Col, DualAnalysis, Duals, Iis,
    IisAnalysis, InvalidPenalty, Limit, Line, Linear, Misfit, Model, ModelClass, ModelClasses, Outcome, Penalty,
    Program, ProgramPlan, Ranged, Ranging, RangingAnalysis, RayAnalysis, RelaxationAnalysis, Relaxed, Row, ScriptError,
    Settings, Solution, Solved, Solver, Table, TiVec, Unsupported, Var, Violation, Weights, penalty,
};
use proptest::prelude::*;

/// A backend that ends every solve with a fixed outcome, or panics out of it when `panics`, and answers every
/// analysis with zeros, recording which it was asked for.
#[derive(Debug)]
struct Canned {
    outcome: Outcome,
    /// Its one option: whether a solve panics, as one whose callback panicked would.
    panics: bool,
    asked: RefCell<Vec<Analysis>>,
}

impl Canned {
    fn new(outcome: Outcome) -> Self {
        Self {
            outcome,
            panics: false,
            asked: RefCell::new(Vec::new()),
        }
    }

    fn ask(&self, analysis: Analysis) {
        self.asked.borrow_mut().push(analysis);
    }
}

impl Backend for Canned {
    type Error = ScriptError;
    type Status = ();
    type Log<'a> = Unsupported;
    type Progress<'a> = Unsupported;
    type Poll<'a> = Unsupported;
    type Solution<'a> = Unsupported;
    type Native<'a> = Unsupported;

    const EVENTS: culmen_core::Events = culmen_core::Events::NONE;
    const CLASSES: ModelClasses = ModelClasses::ALL;

    fn load(&mut self, _: &Program) -> Result<(), ScriptError> {
        Ok(())
    }

    fn solve(&mut self, _: &Settings, _: culmen_core::Hooks<'_, Self>) -> Result<Solved<()>, ScriptError> {
        assert!(!self.panics, "Canned: a callback panicked");
        Ok(Solved {
            outcome: self.outcome.clone(),
            status: (),
            elapsed: Duration::ZERO,
            nodes: None,
            iterations: None,
        })
    }
}

impl culmen_core::Configurable for Canned {
    type Options = bool;

    fn options(&self) -> &bool {
        &self.panics
    }

    fn options_mut(&mut self) -> &mut bool {
        &mut self.panics
    }
}

impl DualAnalysis for Canned {
    fn duals(&self, program: &Program) -> Result<Duals, ScriptError> {
        self.ask(Analysis::Duals);
        Ok(Duals {
            rows: program.rows().map(|_| 0.0).collect(),
            cols: program.cols().map(|_| 0.0).collect(),
        })
    }
}

impl BasisAnalysis for Canned {
    fn basis(&self, program: &Program) -> Result<Option<Basis>, ScriptError> {
        self.ask(Analysis::Basis);
        Ok(Some(Basis::slack(program)))
    }

    fn set_basis(&mut self, _: &Program, _: &Basis) -> Result<(), ScriptError> {
        self.ask(Analysis::Basis);
        Ok(())
    }
}

impl RangingAnalysis for Canned {
    fn ranging(&mut self, program: &Program) -> Result<Ranging, ScriptError> {
        self.ask(Analysis::Ranging);
        let never = Limit {
            value: f64::INFINITY,
            objective: f64::INFINITY,
            entering: None,
            leaving: None,
        };
        let ranged = Ranged { down: never, up: never };
        Ok(Ranging {
            costs: program.cols().map(|_| ranged).collect(),
            col_bounds: program.cols().map(|_| ranged).collect(),
            row_bounds: program.rows().map(|_| ranged).collect(),
        })
    }
}

impl RayAnalysis for Canned {
    fn primal_ray(&mut self, program: &Program) -> Result<Option<TiVec<Col, f64>>, ScriptError> {
        self.ask(Analysis::Rays);
        Ok(Some(program.cols().map(|_| 0.0).collect()))
    }

    fn dual_ray(&mut self, program: &Program) -> Result<Option<TiVec<Row, f64>>, ScriptError> {
        self.ask(Analysis::Rays);
        Ok(Some(program.rows().map(|_| 0.0).collect()))
    }
}

impl IisAnalysis for Canned {
    fn iis(&mut self, _: &Program) -> Result<Option<Iis>, ScriptError> {
        self.ask(Analysis::Iis);
        Ok(Some(Iis::default()))
    }
}

impl RelaxationAnalysis for Canned {
    fn relax(&mut self, program: &Program, _: &Weights) -> Result<Option<Relaxed>, ScriptError> {
        self.ask(Analysis::FeasibilityRelaxation);
        Ok(Some(Relaxed {
            values: program.cols().map(|_| 0.0).collect(),
        }))
    }
}

/// `x + y <= 4` over continuous `x, y >= 0`, maximizing `x`; integer columns make it a MILP.
fn lp(integer: bool) -> Program {
    let mut m = Model::new();
    let var = || if integer { Var::integer() } else { Var::continuous() };
    let x = m.add_var(var());
    let y = m.add_var(var());
    m.add_constraint((x + y).leq(4.0));
    m.maximize(x);
    m.build().unwrap()
}

fn optimal() -> Outcome {
    Outcome::Optimal(Solution {
        objective: 4.0,
        values: vec![4.0, 0.0].into(),
    })
}

fn canned(outcome: Outcome, integer: bool) -> Solver<Canned> {
    Solver::new(Canned::new(outcome), lp(integer)).unwrap()
}

fn asked(solver: &Solver<Canned>) -> Vec<Analysis> {
    solver.backend().asked.borrow().clone()
}

#[test]
fn test_an_analysis_undefined_for_the_class_is_refused_before_the_backend_is_asked() {
    let mut solver = canned(optimal(), true);
    solver.solve().unwrap();
    let class = ModelClass::Milp;
    assert_eq!(
        solver.duals(),
        Err(AnalysisError::Class {
            analysis: Analysis::Duals,
            class
        })
    );
    assert_eq!(
        solver.basis(),
        Err(AnalysisError::Class {
            analysis: Analysis::Basis,
            class
        })
    );
    assert_eq!(
        solver.primal_ray(),
        Err(AnalysisError::Class {
            analysis: Analysis::Rays,
            class
        })
    );
    assert_eq!(
        solver.iis(),
        Err(AnalysisError::Class {
            analysis: Analysis::Iis,
            class
        })
    );
    assert!(
        solver
            .feasibility_relaxation(&Weights::uniform(solver.program(), None, None))
            .is_ok()
    );
    assert_eq!(asked(&solver), vec![Analysis::FeasibilityRelaxation]);
}

#[test]
fn test_analyses_that_read_a_solve_wait_for_one() {
    let mut solver = canned(optimal(), false);
    let not_solved = |analysis| AnalysisError::NotSolved { analysis };
    assert_eq!(solver.duals(), Err(not_solved(Analysis::Duals)));
    assert_eq!(solver.basis(), Err(not_solved(Analysis::Basis)));
    assert_eq!(solver.ranging().unwrap_err(), not_solved(Analysis::Ranging));
    assert_eq!(solver.primal_ray(), Err(not_solved(Analysis::Rays)));
    assert_eq!(solver.dual_ray(), Err(not_solved(Analysis::Rays)));
    assert_eq!(asked(&solver), vec![]);
    // An IIS and a relaxation need no solve.
    assert_eq!(solver.iis(), Ok(Some(Iis::default())));
    assert_eq!(asked(&solver), vec![Analysis::Iis]);
}

#[test]
fn test_duals_and_ranging_need_an_optimal_solve() {
    let mut solver = canned(Outcome::Infeasible, false);
    solver.solve().unwrap();
    assert_eq!(
        solver.duals(),
        Err(AnalysisError::NotOptimal {
            analysis: Analysis::Duals
        })
    );
    assert_eq!(
        solver.ranging().unwrap_err(),
        AnalysisError::NotOptimal {
            analysis: Analysis::Ranging
        }
    );
    assert!(solver.basis().unwrap().is_some(), "a basis needs any solve");
}

#[test]
fn test_ranging_after_an_optimal_solve_reaches_the_backend() {
    let mut solver = canned(optimal(), false);
    solver.solve().unwrap();
    let ranging = solver.ranging().unwrap();
    assert_eq!((ranging.costs.len(), ranging.row_bounds.len()), (2, 1));
    assert_eq!(asked(&solver), vec![Analysis::Ranging]);
}

#[test]
fn test_rays_are_asked_for_only_after_the_ending_they_prove() {
    let cases = [
        (optimal(), (false, false)),
        (Outcome::Unbounded, (true, false)),
        (Outcome::Infeasible, (false, true)),
        (Outcome::InfeasibleOrUnbounded, (true, true)),
    ];
    for (outcome, (primal, dual)) in cases {
        let mut solver = canned(outcome.clone(), false);
        solver.solve().unwrap();
        assert_eq!(solver.primal_ray().unwrap().is_some(), primal, "{outcome:?}");
        assert_eq!(solver.dual_ray().unwrap().is_some(), dual, "{outcome:?}");
        assert_eq!(
            asked(&solver).len(),
            usize::from(primal) + usize::from(dual),
            "{outcome:?}"
        );
    }
}

#[test]
fn test_a_basis_iis_or_relaxation_ends_the_reading_of_the_last_solve() {
    let mut solver = canned(optimal(), false);
    let weights = Weights::uniform(solver.program(), Some(penalty!(1.0)), None);
    solver.solve().unwrap();
    assert_eq!(solver.iis(), Ok(None), "an optimum proves the constraints feasible");
    assert!(solver.duals().is_ok(), "an IIS answered from the last solve keeps it");

    solver.set_basis(&Basis::slack(&lp(false))).unwrap();
    assert_eq!(
        solver.duals(),
        Err(AnalysisError::NotSolved {
            analysis: Analysis::Duals
        })
    );
    solver.solve().unwrap();
    solver.feasibility_relaxation(&weights).unwrap();
    assert_eq!(
        solver.duals(),
        Err(AnalysisError::NotSolved {
            analysis: Analysis::Duals
        })
    );
    solver.solve().unwrap();
    assert!(solver.duals().is_ok());
    assert_eq!(
        asked(&solver),
        vec![
            Analysis::Duals,
            Analysis::Basis,
            Analysis::FeasibilityRelaxation,
            Analysis::Duals
        ]
    );
}

#[test]
fn test_a_solve_that_panics_leaves_no_solve_to_read() {
    let mut solver = canned(optimal(), false);
    solver.solve().unwrap();
    assert!(solver.duals().is_ok());
    *solver.options_mut() = true;
    let solved = catch_unwind(AssertUnwindSafe(|| solver.solve()));
    assert!(solved.is_err());
    assert_eq!(
        solver.duals(),
        Err(AnalysisError::NotSolved {
            analysis: Analysis::Duals
        }),
        "the backend no longer holds the optimum the solve before ended with"
    );
    assert_eq!(asked(&solver), vec![Analysis::Duals]);
}

#[test]
fn test_weights_and_bases_that_do_not_fit_are_refused() {
    let mut solver = canned(optimal(), false);
    let mut weights = Weights::uniform(solver.program(), None, None);
    weights.upper.push(None);
    assert_eq!(
        solver.feasibility_relaxation(&weights),
        Err(AnalysisError::Misfit(Misfit::Length {
            table: Table::UpperWeights,
            expected: 2,
            found: 3
        }))
    );
    let mut basis = Basis::slack(solver.program());
    basis.cols[Col::new(1)] = BasisStatus::Upper;
    assert_eq!(
        solver.set_basis(&basis),
        Err(AnalysisError::Misfit(Misfit::Status {
            line: Line::Col(Col::new(1)),
            status: BasisStatus::Upper
        }))
    );
    assert_eq!(asked(&solver), vec![]);
}

#[test]
fn test_errors_read_as_sentences() {
    type E = AnalysisError<ScriptError>;
    let messages = [
        (
            E::NotSolved {
                analysis: Analysis::Duals,
            },
            "Solver: duals need a solve, and there is none to read",
        ),
        (
            E::NotOptimal {
                analysis: Analysis::Ranging,
            },
            "Solver: ranging needs an optimal solve",
        ),
        (
            E::Class {
                analysis: Analysis::Iis,
                class: ModelClass::Milp,
            },
            "Solver: IIS is defined for {LP, QP}, not a MILP",
        ),
        (
            E::Backend(ScriptError::NotLoaded),
            "Solver: the backend failed the analysis: Script: no program is loaded",
        ),
        (
            E::Misfit(Misfit::BasicCount { expected: 2, found: 0 }),
            "Solver: the basis has 0 basic lines for 2 rows",
        ),
        (
            E::Misfit(Misfit::Status {
                line: Line::Row(Row::new(1)),
                status: BasisStatus::Upper,
            }),
            "Solver: the basis puts row 1 at its upper bound, which is infinite",
        ),
        (
            E::Misfit(Misfit::Length {
                table: Table::LowerWeights,
                expected: 3,
                found: 1,
            }),
            "Solver: the weights have 1 lower-bound weights for 3 columns",
        ),
    ];
    for (error, text) in messages {
        assert_eq!(error.to_string(), text);
    }
}

#[test]
fn test_the_classes_of_each_analysis() {
    let of = |classes: &[ModelClass]| ModelClasses::of(classes);
    assert_eq!(Analysis::Duals.classes(), of(&[ModelClass::Lp, ModelClass::Qp]));
    assert_eq!(Analysis::Basis.classes(), of(&[ModelClass::Lp]));
    assert_eq!(Analysis::Ranging.classes(), of(&[ModelClass::Lp]));
    assert_eq!(Analysis::Rays.classes(), of(&[ModelClass::Lp]));
    assert_eq!(Analysis::Iis.classes(), of(&[ModelClass::Lp, ModelClass::Qp]));
    assert_eq!(
        Analysis::FeasibilityRelaxation.classes(),
        of(&[ModelClass::Lp, ModelClass::Milp])
    );
}

#[test]
fn test_basis_check_fixtures() {
    // x free, y in [0, 1], one row x + y <= 4.
    let mut m = Model::new();
    let x = m.add_var(Var::free());
    let y = m.add_var(Var::continuous().upper(1.0));
    let r = m.add_constraint((x + y).leq(4.0));
    let program = m.build().unwrap();
    let slack = Basis::slack(&program);
    assert_eq!(slack.cols.raw, vec![BasisStatus::Zero, BasisStatus::Lower]);
    assert_eq!(slack.rows.raw, vec![BasisStatus::Basic]);
    let mut basis = slack.clone();
    basis.cols[x] = BasisStatus::Basic;
    basis.rows[r] = BasisStatus::Upper;
    assert_eq!(
        basis.check(&program),
        Ok(()),
        "the row at its finite upper bound, x basic"
    );
    basis.rows[r] = BasisStatus::Lower;
    assert_eq!(
        basis.check(&program),
        Err(Misfit::Status {
            line: Line::Row(r),
            status: BasisStatus::Lower
        })
    );
    basis.rows[r] = BasisStatus::NonBasic;
    basis.cols[y] = BasisStatus::Upper;
    assert_eq!(basis.check(&program), Ok(()));
}

#[test]
fn test_violations_and_their_penalty_on_a_fixture() {
    // x in [0, 2], y in [1, 3], x + y <= 3: at (3, 0) x is 1 above, y 1 below, the row 0 off.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(2.0));
    let y = m.add_var(Var::continuous().bounds(1.0, 3.0));
    let r = m.add_constraint((x + y).leq(3.0));
    let program = m.build().unwrap();
    let relaxed = Relaxed {
        values: vec![3.0, 0.0].into(),
    };
    let violations: Vec<Violation> = relaxed.violations(&program).collect();
    assert_eq!(
        violations,
        vec![
            Violation {
                line: Line::Col(x),
                bound: Bound::Upper,
                amount: 1.0
            },
            Violation {
                line: Line::Col(y),
                bound: Bound::Lower,
                amount: 1.0
            },
        ]
    );
    let mut weights = Weights::uniform(&program, Some(penalty!(2.0)), Some(penalty!(5.0)));
    weights.lower[y] = None;
    assert_eq!(weights.penalty(violations), 2.0, "y's hard lower bound counts nothing");
    assert_eq!(weights.price(Line::Row(r), Bound::Lower), Some(penalty!(5.0)));
}

/// The violations of `values` in `program`, by the definition: each column, then each row with its activity summed
/// from the plan's terms, below its lower end or above its upper end.
fn reference_violations(program: &Program, values: &[f64]) -> Vec<(Line, Bound, f64)> {
    let plan = ProgramPlan::from(program.clone());
    let lines = plan
        .columns
        .iter_enumerated()
        .map(|(c, column)| (Line::Col(c), column.bounds, values[c.index()]))
        .chain(plan.rows.iter().enumerate().map(|(r, row)| {
            let activity = row.terms.iter().map(|&(c, a)| a * values[c.index()]).sum::<f64>();
            (Line::Row(Row::new(r)), row.bounds, activity)
        }));
    lines
        .filter_map(|(line, bounds, value)| {
            if value < bounds.lower() {
                Some((line, Bound::Lower, bounds.lower() - value))
            } else if value > bounds.upper() {
                Some((line, Bound::Upper, value - bounds.upper()))
            } else {
                None
            }
        })
        .collect()
}

/// A small program: one to four columns with integer bounds, some open, and up to three rows of small integer
/// coefficients with integer bounds, some open.
fn small_program() -> impl Strategy<Value = Program> {
    let end = || prop_oneof![Just(None), (-4i32..=4).prop_map(Some)];
    let bounds = move || {
        (end(), end(), 0i32..=4).prop_map(|(lower, upper, width)| {
            let lower = lower.map_or(f64::NEG_INFINITY, f64::from);
            let upper = upper.map_or(f64::INFINITY, |u| lower.max(f64::from(u)) + f64::from(width));
            culmen_core::Bounds::new(lower, upper).unwrap()
        })
    };
    (1usize..=4)
        .prop_flat_map(move |n| {
            let row = (bounds(), prop::collection::vec((0..n, -3i32..=3), 1..=3));
            (prop::collection::vec(bounds(), n), prop::collection::vec(row, 0..=3))
        })
        .prop_map(|(columns, rows)| {
            let mut plan = ProgramPlan::new();
            columns.into_iter().for_each(|b| {
                plan.col(culmen_core::ProgramColumn::continuous().cost(1.0).bounds(b));
            });
            rows.into_iter().for_each(|(b, terms)| {
                plan.row((b, terms.into_iter().map(|(j, a)| (Col::new(j), f64::from(a)))));
            });
            plan.build().unwrap()
        })
}

proptest! {
    /// A penalty is accepted exactly when it is finite and not below zero.
    #[test]
    fn test_a_penalty_is_a_finite_non_negative_number(value in prop_oneof![any::<f64>(), -2.0..2.0f64]) {
        let accepted = Penalty::new(value);
        prop_assert_eq!(accepted.is_ok(), value.is_finite() && value >= 0.0);
        if let Ok(penalty) = accepted {
            prop_assert_eq!(f64::from(penalty), value);
        }
    }

    /// The violations of a random point agree with the definition, and their penalty with the weighted sum.
    #[test]
    fn test_violations_agree_with_the_definition(
        program in small_program(),
        values in prop::collection::vec(-6i32..=6, 4),
        weight in 0i32..=3,
    ) {
        let values: Vec<f64> = values.into_iter().take(program.num_cols()).map(f64::from).collect();
        let relaxed = Relaxed { values: values.clone().into() };
        let got: Vec<(Line, Bound, f64)> = relaxed.violations(&program).map(|v| (v.line, v.bound, v.amount)).collect();
        let want = reference_violations(&program, &values);
        prop_assert_eq!(&got, &want);
        let price = Penalty::new(f64::from(weight)).unwrap();
        let weights = Weights::uniform(&program, Some(price), None);
        let cols: f64 = want.iter().filter(|v| matches!(v.0, Line::Col(_))).map(|v| v.2 * f64::from(weight)).sum();
        prop_assert_eq!(weights.penalty(relaxed.violations(&program)), cols);
    }

    /// The slack basis fits every program.
    #[test]
    fn test_the_slack_basis_fits(program in small_program()) {
        prop_assert_eq!(Basis::slack(&program).check(&program), Ok(()));
    }
}

#[test]
fn test_a_penalty_literal_and_its_refusals() {
    const ONE: Penalty = penalty!(1.0);
    assert_eq!(ONE.get(), 1.0);
    assert_eq!(Penalty::try_from(f64::INFINITY), Err(InvalidPenalty::NonFinite));
    assert_eq!(InvalidPenalty::Negative.to_string(), "Penalty: below zero");
}

#[cfg(feature = "serde")]
mod serde_round_trips {
    use core::fmt::Debug;
    use std::collections::BTreeMap;

    use culmen_core::{Limit, Ranged, Ranging, Side};
    use serde::Serialize;
    use serde::de::DeserializeOwned;

    use super::*;

    fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
        let json = serde_json::to_string(value).unwrap();
        let back: T = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, value, "{json}");
    }

    #[test]
    fn test_analysis_data_round_trips() {
        let program = lp(false);
        round_trip(&Analysis::FeasibilityRelaxation);
        round_trip(&Duals {
            rows: vec![-2.0].into(),
            cols: vec![0.5, 0.0].into(),
        });
        round_trip(&Basis::slack(&program));
        round_trip(&BasisStatus::NonBasic);
        let limit = Limit {
            value: -3.0,
            objective: -12.0,
            entering: Some(Line::Col(Col::new(0))),
            leaving: None,
        };
        let ranged = Ranged {
            down: limit,
            up: Limit {
                value: f64::MAX,
                entering: Some(Line::Row(Row::new(1))),
                ..limit
            },
        };
        round_trip(&Ranging {
            costs: vec![ranged].into(),
            col_bounds: vec![ranged].into(),
            row_bounds: vec![].into(),
        });
        round_trip(&Iis {
            cols: BTreeMap::from([(Col::new(3), Side::Upper)]),
            rows: BTreeMap::from([(Row::new(0), Side::Both), (Row::new(2), Side::Lower)]),
        });
        round_trip(&penalty!(0.25));
        round_trip(&Weights::uniform(&program, Some(penalty!(2.0)), None));
        round_trip(&Relaxed {
            values: vec![1.0, -0.5].into(),
        });
        round_trip(&Violation {
            line: Line::Row(Row::new(1)),
            bound: Bound::Upper,
            amount: 2.0,
        });
    }

    /// The values a real ranging holds: a limit never reached in one direction is infinite, and so is the objective
    /// where the program turns unbounded; a point holding an infinity or NaN is violated by as much.
    #[test]
    fn test_infinite_limits_and_violations_round_trip_through_json() {
        let unreached = Limit {
            value: f64::NEG_INFINITY,
            objective: f64::INFINITY,
            entering: None,
            leaving: None,
        };
        round_trip(&Ranging {
            costs: vec![Ranged {
                down: unreached,
                up: Limit {
                    value: 2.5,
                    objective: -7.0,
                    entering: Some(Line::Col(Col::new(1))),
                    leaving: Some(Line::Row(Row::new(0))),
                },
            }]
            .into(),
            col_bounds: vec![].into(),
            row_bounds: vec![Ranged {
                down: unreached,
                up: unreached,
            }]
            .into(),
        });
        round_trip(&Violation {
            line: Line::Col(Col::new(0)),
            bound: Bound::Lower,
            amount: f64::INFINITY,
        });
        let nan = Violation {
            line: Line::Col(Col::new(0)),
            bound: Bound::Lower,
            amount: f64::NAN,
        };
        let back: Violation = serde_json::from_str(&serde_json::to_string(&nan).unwrap()).unwrap();
        assert!(back.amount.is_nan());
    }

    #[test]
    fn test_a_negative_penalty_is_refused_as_it_is_read() {
        assert!(serde_json::from_str::<Penalty>("-1.0").is_err());
        assert!(serde_json::from_str::<Penalty>("1.5").is_ok());
    }
}
