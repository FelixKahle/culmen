// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a program and a solve take beyond their class: semi columns, further objectives and starts, their checks, the
//! features a program uses, and their refusal by a backend that lacks them.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use core::fmt;
use core::time::Duration;
use std::collections::HashSet;

use culmen_core::{
    Backend, Bound, Bounds, Callback, Col, Combination, Events, ExprPlace, Integrality, Line, Linear, LinearObjective,
    LoadError, LpFileError, Model, ModelClass, ModelClasses, ModelError, Objective, ObjectiveId, Objectives, Outcome,
    Program, ProgramColumn, ProgramError, ProgramFeature, ProgramFeatures, ProgramPlan, Relaxed, Request, Script,
    Sense, Settings, SolveError, Solved, Solver, Stage, Start, StartError, TiSlice, TiVec, Tolerance, Unsupported, Var,
    Violation, Weight, Weighted, absolute_gap, bounds, relative_gap, weight,
};
use proptest::prelude::*;

/// Minimize `x + y` over `x + y >= 1` with `x` semi-continuous in `[2, 10]` and `y` semi-integer in `[1.5, 4]`.
fn semi() -> (Model, Col, Col) {
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(2, 10).name("x"));
    let y = m.add_var(Var::semi_integer().bounds(1.5, 4).name("y"));
    m.add_constraint((x + y).geq(1));
    m.minimize(x + y);
    (m, x, y)
}

#[test]
fn test_a_semi_program_is_mixed_integer_and_uses_its_features() {
    let (mut m, x, y) = semi();
    let program = m.build().unwrap();
    assert_eq!(program.class(), ModelClass::Milp);
    assert!(program.is_mip());
    assert_eq!(
        program.features(),
        ProgramFeatures::of(&[ProgramFeature::SemiContinuous, ProgramFeature::SemiInteger])
    );
    assert_eq!(program.columns()[x].integrality, Integrality::SemiContinuous);
    assert_eq!(program.columns()[y].integrality, Integrality::SemiInteger);

    m.minimize(x * x + y);
    assert_eq!(m.build().unwrap().class(), ModelClass::Miqp);
}

#[test]
fn test_a_semi_column_is_zero_or_in_its_bounds() {
    let program = semi().0.build().unwrap();
    let violation = |x: f64, y: f64| program.max_violation(TiSlice::from_ref(&[x, y]));
    // Feasible: x off and y at 2; x at 2 and y off; both on.
    assert_eq!(violation(0.0, 2.0), 0.0);
    assert_eq!(violation(2.0, 0.0), 0.0);
    assert_eq!(violation(10.0, 4.0), 0.0);
    // x between zero and its lower bound is nearer to zero: 0.5 off zero, or 0.5 below 2 for 1.5.
    assert_eq!(violation(0.5, 2.0), 0.5);
    assert_eq!(violation(1.5, 2.0), 0.5);
    // y must be whole when on: 2.25 is 0.25 off 2; 1.5 is in the bounds yet 0.5 off a whole number.
    assert_eq!(violation(0.0, 2.25), 0.25);
    assert_eq!(violation(0.0, 1.5), 0.5);
    // Above the upper bound, the distance to it.
    assert_eq!(violation(11.0, 0.0), 1.0);
}

#[test]
fn test_a_semi_column_needs_bounds_that_exclude_zero() {
    for (lower, upper) in [(0.0, 4.0), (-1.0, 4.0), (1.0, f64::INFINITY)] {
        let bounds = Bounds::new(lower, upper).unwrap();
        let mut plan = ProgramPlan::new();
        let x = plan.col(ProgramColumn::semi_continuous(1.0, bounds));
        assert_eq!(plan.build(), Err(ProgramError::SemiBounds { col: x, bounds }));

        let mut m = Model::new();
        let y = m.add_var(Var::semi_integer().bounds(lower, upper).name("y"));
        let Err(ModelError::SemiBounds { var, bounds: refused }) = m.build() else {
            panic!("the semi variable in {bounds} is refused")
        };
        assert_eq!((var.col, var.name.as_deref(), refused), (y, Some("y"), bounds));
    }
    let mut plan = ProgramPlan::new();
    plan.col(ProgramColumn::semi_integer(0.0, bounds!(1e-9, 1e9)));
    assert!(plan.build().is_ok());
    assert_eq!(
        ProgramError::SemiBounds {
            col: Col::new(0),
            bounds: Bounds::NON_NEGATIVE
        }
        .to_string(),
        "Program: the semi column Col(0) has the bounds [0, +inf), not 0 < lower <= upper < +inf"
    );
}

/// Maximize `x + y` over `x + y <= 10`, `x <= 6`; then minimize `2y - 1`.
fn two_stage() -> (Model, Col, Col, ObjectiveId) {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(6).name("x"));
    let y = m.add_var(Var::continuous().upper(10).name("y"));
    m.add_constraint((x + y).leq(10));
    m.maximize(x + y);
    let second = m.add_objective(Objective::minimize(y + y - 1).weight(weight!(2.0)));
    (m, x, y, second)
}

#[test]
fn test_a_model_builds_its_further_objectives_lexicographic_or_blended() {
    let (mut m, x, y, second) = two_stage();
    assert_eq!(second, ObjectiveId::new(1));
    let further = LinearObjective {
        sense: Sense::Minimize,
        terms: vec![(y, 2.0)],
        offset: -1.0,
    };
    let loose = Tolerance {
        absolute: Some(absolute_gap!(2.0)),
        relative: Some(relative_gap!(0.1)),
    };
    m.set_objective_tolerance(loose);
    let program = m.build().unwrap();
    assert_eq!(
        program.objectives(),
        &Objectives::Lexicographic {
            first: loose,
            then: vec![Stage {
                objective: further.clone(),
                tolerance: Tolerance::EXACT
            }],
        }
    );
    assert_eq!(program.class(), ModelClass::Lp);
    assert_eq!(program.features(), ProgramFeature::MultipleObjectives.into());
    assert_eq!(
        program.objective_values(TiSlice::from_ref(&[6.0, 4.0])),
        TiVec::<ObjectiveId, f64>::from(vec![10.0, 7.0])
    );

    m.set_combination(Combination::Blended);
    m.set_objective_weight(weight!(0.5));
    assert_eq!(
        m.build().unwrap().objectives(),
        &Objectives::Blended {
            first: weight!(0.5),
            with: vec![Weighted {
                objective: further,
                weight: weight!(2.0)
            }],
        }
    );
    let _ = x;
}

#[test]
fn test_further_objectives_are_canonical_in_a_program() {
    let mut plan = ProgramPlan::new();
    let x = plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
    let y = plan.col(ProgramColumn::continuous(0.0, Bounds::NON_NEGATIVE));
    plan.objectives = Objectives::Blended {
        first: Weight::ONE,
        with: vec![Weighted {
            objective: LinearObjective {
                sense: Sense::Maximize,
                terms: vec![(y, 1.0), (x, 2.0), (y, -1.0), (x, 0.5)],
                offset: 3.0,
            },
            weight: weight!(3.0),
        }],
    };
    let program = plan.clone().build().unwrap();
    let Objectives::Blended { with, .. } = program.objectives() else {
        panic!("blended")
    };
    assert_eq!(with[0].objective.terms, [(x, 2.5)]);
    assert_eq!(program.objectives().further().count(), 1);

    // Without a further objective, either combination is a single objective.
    plan.objectives = Objectives::Lexicographic {
        first: Tolerance::EXACT,
        then: Vec::new(),
    };
    let program = plan.build().unwrap();
    assert_eq!(program.objectives(), &Objectives::Single);
    assert_eq!(program.features(), ProgramFeatures::NONE);
}

#[test]
fn test_further_objectives_are_checked() {
    let further = |terms: Vec<(Col, f64)>, offset: f64| {
        let mut plan = ProgramPlan::new();
        plan.col(ProgramColumn::continuous(1.0, Bounds::NON_NEGATIVE));
        plan.objectives = Objectives::Lexicographic {
            first: Tolerance::EXACT,
            then: vec![Stage {
                objective: LinearObjective {
                    sense: Sense::Minimize,
                    terms,
                    offset,
                },
                tolerance: Tolerance::EXACT,
            }],
        };
        plan
    };
    let (x, id) = (Col::new(0), ObjectiveId::new(1));
    assert_eq!(
        further(vec![(Col::new(1), 1.0)], 0.0).build(),
        Err(ProgramError::UnknownObjectiveColumn {
            objective: id,
            col: Col::new(1)
        })
    );
    assert_eq!(
        further(vec![(x, 1e308), (x, 1e308)], 0.0).build(),
        Err(ProgramError::NonFiniteObjectiveCoefficient {
            objective: id,
            col: x,
            value: f64::INFINITY
        })
    );
    assert_eq!(
        further(vec![(x, 1.0)], f64::INFINITY).build(),
        Err(ProgramError::NonFiniteObjectiveOffset {
            objective: id,
            value: f64::INFINITY
        })
    );
    let mut quadratic = further(vec![(x, 1.0)], 0.0);
    quadratic.quadratic = vec![(x, x, 1.0)];
    assert_eq!(quadratic.clone().build(), Err(ProgramError::QuadraticWithObjectives));
    // A quadratic part that cancels to nothing is no quadratic part.
    quadratic.quadratic = vec![(x, x, 1.0), (x, x, -1.0)];
    assert!(quadratic.build().is_ok());
}

#[test]
fn test_a_model_names_the_further_objective_at_fault() {
    let (mut m, x, _, _) = two_stage();
    let third = m.add_objective(Objective::maximize(f64::NAN * x));
    let Err(ModelError::FurtherCoefficient { objective, var, value }) = m.build() else {
        panic!("a NaN coefficient is refused")
    };
    assert_eq!((objective, var.col, var.name.as_deref()), (third, x, Some("x")));
    assert!(value.is_nan());

    let (mut m, x, _, _) = two_stage();
    let third = m.add_objective(Objective::maximize(x + f64::INFINITY));
    assert_eq!(
        m.build().unwrap_err(),
        ModelError::FurtherConstant {
            objective: third,
            value: f64::INFINITY
        }
    );

    let (mut m, _, _, _) = two_stage();
    let third = m.add_objective(Objective::maximize(Col::new(9)));
    assert_eq!(
        m.build().unwrap_err(),
        ModelError::UnknownVar {
            place: ExprPlace::Further(third),
            col: Col::new(9)
        }
    );

    let (mut m, x, y, _) = two_stage();
    m.maximize(x + y - x * x);
    assert_eq!(m.build().unwrap_err(), ModelError::QuadraticWithObjectives);
}

#[test]
fn test_the_lp_file_lists_semi_columns_and_refuses_several_objectives() {
    let text = semi().0.build().unwrap().to_lp_string().unwrap();
    assert_eq!(
        text,
        "Minimize\n obj: 1 x + 1 y\nSubject To\n r0: 1 x + 1 y >= 1\nBounds\n 2 <= x <= 10\n 1.5 <= y <= 4\n\
         General\n y\nSemi-Continuous\n x\n y\nEnd\n"
    );
    let several = two_stage().0.build().unwrap();
    let refused = LpFileError::Unsupported {
        feature: ProgramFeature::MultipleObjectives,
    };
    assert_eq!(several.to_lp_string(), Err(refused));
    let mut out = String::new();
    assert_eq!(several.write_lp(&mut out), Err(refused));
    assert_eq!(out, "", "nothing is written");
    assert_eq!(
        refused.to_string(),
        "LP file: a program with multiple objectives cannot be written"
    );
}

#[test]
fn test_feature_sets_are_sets() {
    let semi = ProgramFeatures::of(&[ProgramFeature::SemiContinuous, ProgramFeature::SemiInteger]);
    let mut all = semi;
    all |= ProgramFeature::MultipleObjectives.into();
    all.extend([ProgramFeature::StartSolution]);
    assert_eq!(all, ProgramFeatures::ALL);
    assert_eq!(all.len(), 4);
    assert!(all.contains_all(semi) && !semi.contains_all(all));
    assert_eq!(all & semi, semi);
    assert_eq!(
        all.difference(semi).iter().rev().collect::<Vec<_>>(),
        [ProgramFeature::StartSolution, ProgramFeature::MultipleObjectives]
    );
    assert_eq!(all.iter().len(), 4);
    assert!(ProgramFeatures::NONE.is_empty());
    assert_eq!(ProgramFeatures::default(), ProgramFeatures::NONE);
    assert_eq!(format!("{:?}", ProgramFeatures::NONE), "ProgramFeatures()");
    assert_eq!(ProgramFeatures::NONE.to_string(), "{}");
    assert_eq!(
        all.to_string(),
        "{semi-continuous columns, semi-integer columns, multiple objectives, a start solution}"
    );
}

/// A backend that takes no feature: it solves LPs and MILPs and declares nothing beyond them.
#[derive(Debug, Default)]
struct Plain {
    loads: usize,
}

/// Why [`Plain`] refused; it never does.
#[derive(Debug, PartialEq)]
struct Never;

impl fmt::Display for Never {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Plain: never")
    }
}

impl core::error::Error for Never {}

impl Backend for Plain {
    type Error = Never;
    type Status = ();
    type Log<'a> = Unsupported;
    type Progress<'a> = Unsupported;
    type Poll<'a> = Unsupported;
    type Solution<'a> = Unsupported;
    type Lazy<'a> = Unsupported;
    type Cut<'a> = Unsupported;
    type Heuristic<'a> = Unsupported;
    type Native<'a> = Unsupported;

    const EVENTS: Events = Events::NONE;
    const CLASSES: ModelClasses = ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]);

    fn load(&mut self, _: &Program) -> Result<(), Never> {
        self.loads += 1;
        Ok(())
    }

    fn solve(&mut self, _: &Settings, _: &mut dyn Callback<Self>) -> Result<Solved<()>, Never> {
        Ok(Solved {
            outcome: Outcome::Infeasible,
            status: (),
            elapsed: Duration::ZERO,
        })
    }
}

#[test]
fn test_a_backend_without_a_feature_refuses_it_before_it_sees_it() {
    let refused = |program: Program| Solver::new(Plain::default(), program).unwrap_err();
    assert_eq!(
        refused(semi().0.build().unwrap()),
        LoadError::UnsupportedFeature {
            feature: ProgramFeature::SemiContinuous,
            supported: ProgramFeatures::NONE
        }
    );
    let error = refused(two_stage().0.build().unwrap());
    assert_eq!(
        error,
        LoadError::UnsupportedFeature {
            feature: ProgramFeature::MultipleObjectives,
            supported: ProgramFeatures::NONE
        }
    );
    assert_eq!(
        error.to_string(),
        "Solver: the backend takes {}, not multiple objectives"
    );

    let mut plain = Model::new();
    let z = plain.add_var(Var::binary());
    plain.maximize(z);
    let mut solver = Solver::new(Plain::default(), plain.build().unwrap()).unwrap();
    assert_eq!(solver.backend().loads, 1);
    assert_eq!(
        solver.solve(Request::new().start(vec![(z, 1.0)])).unwrap_err(),
        SolveError::UnsupportedFeature {
            feature: ProgramFeature::StartSolution,
            supported: ProgramFeatures::NONE
        }
    );
    assert!(solver.solve(Request::new()).is_ok());
}

#[test]
fn test_a_start_is_checked_against_the_program_and_handed_over() {
    let (m, x, y) = semi();
    let mut solver = Solver::new(Script::optimal(2.0, vec![2.0, 0.0]), m.build().unwrap()).unwrap();
    let full = TiVec::<Col, f64>::from(vec![2.0, 0.0]);
    solver.solve(Request::new().start(full.clone())).unwrap();
    solver.solve(Request::new().start(vec![(y, 2.0)])).unwrap();
    solver.solve(Request::new()).unwrap();
    assert_eq!(
        solver.backend().record.starts,
        [Start::Full(full), Start::Sparse(vec![(y, 2.0)])]
    );

    let refused = |solver: &mut Solver<Script>, start: Start| match solver.solve(Request::new().start(start)) {
        Err(SolveError::Start(error)) => error,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        refused(&mut solver, TiVec::from(vec![1.0]).into()),
        StartError::Length { expected: 2, found: 1 }
    );
    assert_eq!(
        refused(&mut solver, vec![(Col::new(2), 1.0)].into()),
        StartError::UnknownColumn { col: Col::new(2) }
    );
    assert_eq!(
        refused(&mut solver, vec![(x, 2.0), (x, 3.0)].into()),
        StartError::RepeatedColumn { col: x }
    );
    assert_eq!(
        refused(&mut solver, vec![(y, f64::INFINITY)].into()),
        StartError::NotFinite {
            col: y,
            value: f64::INFINITY
        }
    );
    assert_eq!(
        solver.backend().record.starts.len(),
        2,
        "no refused start reaches the backend"
    );
    assert_eq!(
        StartError::RepeatedColumn { col: x }.to_string(),
        "Start: Col(0) is given twice"
    );
}

#[test]
fn test_tolerance_and_weight_hold_their_invariants() {
    assert_eq!(Weight::new(f64::NAN), Err(culmen_core::InvalidWeight::NotFinite));
    assert_eq!(Weight::new(-1.0), Err(culmen_core::InvalidWeight::NotPositive));
    assert_eq!(Weight::try_from(0.25).map(f64::from), Ok(0.25));
    let absolute_only = Tolerance {
        absolute: Some(absolute_gap!(3.0)),
        relative: None,
    };
    assert_eq!(absolute_only.allowance(-1e9), 3.0);
    let relative_only = Tolerance {
        absolute: None,
        relative: Some(relative_gap!(0.5)),
    };
    assert_eq!(relative_only.allowance(-4.0), 2.0);
}

/// A program of up to five columns, each continuous, integer, semi-continuous or semi-integer, with up to two further
/// objectives.
fn featured() -> impl Strategy<Value = ProgramPlan> {
    let column = (0u8..4).prop_map(|kind| {
        let bounds = bounds!(1.0, 3.0);
        match kind {
            0 => ProgramColumn::continuous(1.0, bounds),
            1 => ProgramColumn::integer(1.0, bounds),
            2 => ProgramColumn::semi_continuous(1.0, bounds),
            _ => ProgramColumn::semi_integer(1.0, bounds),
        }
    });
    (prop::collection::vec(column, 0..5), 0usize..3, any::<bool>()).prop_map(|(columns, further, blended)| {
        let objective = || LinearObjective {
            sense: Sense::Maximize,
            terms: Vec::new(),
            offset: 1.0,
        };
        let objectives = if blended {
            Objectives::Blended {
                first: Weight::ONE,
                with: (0..further)
                    .map(|_| Weighted {
                        objective: objective(),
                        weight: Weight::ONE,
                    })
                    .collect(),
            }
        } else {
            Objectives::Lexicographic {
                first: Tolerance::EXACT,
                then: (0..further)
                    .map(|_| Stage {
                        objective: objective(),
                        tolerance: Tolerance::EXACT,
                    })
                    .collect(),
            }
        };
        ProgramPlan {
            columns: columns.into(),
            objectives,
            ..ProgramPlan::new()
        }
    })
}

/// A start of up to four entries over up to three columns, some full, values finite or not.
fn start() -> impl Strategy<Value = (Start, usize)> {
    let value = prop_oneof![4 => (-2i32..=2).prop_map(f64::from), 1 => Just(f64::NAN)];
    let sparse = prop::collection::vec((0usize..4, value.clone()), 0..4)
        .prop_map(|entries| Start::Sparse(entries.into_iter().map(|(i, v)| (Col::new(i), v)).collect()));
    let full = prop::collection::vec(value, 0..4).prop_map(|values| Start::Full(values.into()));
    (prop_oneof![sparse, full], 0usize..4)
}

proptest! {
    /// The features of a program are the kinds of its semi columns and whether it has a further objective, read
    /// from the plan it was built from.
    #[test]
    fn test_features_are_what_the_plan_holds(plan in featured()) {
        let kinds: HashSet<Integrality> = plan.columns.iter().map(|c| c.integrality).collect();
        let mut expected = Vec::new();
        if kinds.contains(&Integrality::SemiContinuous) {
            expected.push(ProgramFeature::SemiContinuous);
        }
        if kinds.contains(&Integrality::SemiInteger) {
            expected.push(ProgramFeature::SemiInteger);
        }
        let further = match &plan.objectives {
            Objectives::Single => 0,
            Objectives::Lexicographic { then, .. } => then.len(),
            Objectives::Blended { with, .. } => with.len(),
        };
        if further > 0 {
            expected.push(ProgramFeature::MultipleObjectives);
        }
        let program = plan.build().unwrap();
        prop_assert_eq!(program.features().iter().collect::<Vec<_>>(), expected);
        prop_assert_eq!(program.objective_values(TiSlice::from_ref(&[2.0; 5][..program.num_cols()])).len(), 1 + further);
    }

    /// A start fits a program of `num_cols` columns exactly when, read from its prose: a full start has `num_cols`
    /// values, every entry's column is below `num_cols` and named once, and every value is finite.
    #[test]
    fn test_a_start_fits_as_its_check_states((start, num_cols) in start()) {
        let entries: Vec<(Col, f64)> = start.entries().collect();
        let mut columns: Vec<Col> = entries.iter().map(|&(c, _)| c).collect();
        columns.sort();
        let once = columns.windows(2).all(|w| w[0] != w[1]);
        let length = match &start {
            Start::Full(values) => values.len() == num_cols,
            Start::Sparse(_) => true,
        };
        let fits = length && once && entries.iter().all(|&(c, v)| c.index() < num_cols && v.is_finite());
        prop_assert_eq!(start.check(num_cols).is_ok(), fits);
    }
}

#[cfg(feature = "serde")]
mod serde_round_trips {
    use core::fmt::Debug;

    use serde::Serialize;
    use serde::de::DeserializeOwned;

    use super::*;

    fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
        let json = serde_json::to_string(value).unwrap();
        let back: T = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, value, "{json}");
    }

    #[test]
    fn test_the_new_plain_data_round_trips() {
        round_trip(&Integrality::SemiContinuous);
        round_trip(&Integrality::SemiInteger);
        round_trip(&ProgramFeature::StartSolution);
        round_trip(&ProgramFeatures::of(&[
            ProgramFeature::SemiInteger,
            ProgramFeature::MultipleObjectives,
        ]));
        round_trip(&ObjectiveId::new(2));
        round_trip(&weight!(1.5));
        round_trip(&Tolerance {
            absolute: None,
            relative: Some(relative_gap!(0.2)),
        });
        round_trip(&Start::Full(vec![1.0, 0.0].into()));
        round_trip(&Start::Sparse(vec![(Col::new(1), 2.5)]));
        round_trip(&semi().0.build().unwrap());
        round_trip(&ProgramPlan::from(semi().0.build().unwrap()));
        let (mut m, ..) = two_stage();
        round_trip(&m.build().unwrap());
        m.set_combination(Combination::Blended);
        round_trip(&m.build().unwrap());
        round_trip(&ProgramPlan::from(m.build().unwrap()));
    }

    #[test]
    fn test_checked_values_are_checked_as_they_are_read() {
        assert!(serde_json::from_str::<Weight>("0.0").is_err());
        assert!(serde_json::from_str::<Weight>("-2.0").is_err());
        // A semi column with zero in its bounds is refused as the program is read.
        let mut json = serde_json::to_value(semi().0.build().unwrap()).unwrap();
        json["columns"][0]["bounds"]["lower"] = serde_json::json!(0.0);
        assert!(serde_json::from_value::<Program>(json).is_err());
        // A program written before further objectives reads with one.
        let mut json = serde_json::to_value(semi().0.build().unwrap()).unwrap();
        json.as_object_mut().unwrap().remove("objectives");
        let program: Program = serde_json::from_value(json).unwrap();
        assert_eq!(program.objectives(), &Objectives::Single);
    }
}

#[test]
fn test_a_semi_column_at_zero_violates_none_of_its_bounds() {
    let (m, x, y) = semi();
    let program = m.build().unwrap();
    let violations = |values: Vec<f64>| {
        Relaxed { values: values.into() }
            .violations(&program)
            .collect::<Vec<_>>()
    };
    // x off and y on at 3: no bound and no row is violated.
    assert_eq!(violations(vec![0.0, 3.0]), []);
    // x at 0.5 is 0.5 off zero and 1.5 below its lower bound; y at 0.75 is 0.75 off zero: each the nearer.
    assert_eq!(
        violations(vec![0.5, 0.75]),
        [
            Violation {
                line: Line::Col(x),
                bound: Bound::Lower,
                amount: 0.5
            },
            Violation {
                line: Line::Col(y),
                bound: Bound::Lower,
                amount: 0.75
            },
        ]
    );
    // Above the upper bound, zero is no nearer.
    assert_eq!(
        violations(vec![12.0, 0.0]),
        [Violation {
            line: Line::Col(x),
            bound: Bound::Upper,
            amount: 2.0
        }]
    );
}
