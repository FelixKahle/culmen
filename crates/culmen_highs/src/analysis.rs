// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The analyses of [`Highs`]: HiGHS's duals, basis, ranging, rays, IIS and feasibility relaxation in culmen's types.
//!
//! The [`Solver`](culmen_core::Solver) has checked each request before it reaches these functions; they map HiGHS's
//! numbering to typed lines and drop the free row a QP without rows is given (see `instance`).

use culmen_core::{
    Analysis, AnalysisError, Basis, BasisStatus, Bound, Col, Duals, Iis, Limit, Line, Program, ProgramFeature,
    ProgramFeatures, Ranged, Ranging, Relaxed, Row, Sense, Settings, Side, TiVec, Weights,
};
use highs_sys::{
    HighsInt, kHighsBasisStatusBasic, kHighsBasisStatusLower, kHighsBasisStatusNonbasic, kHighsBasisStatusUpper,
    kHighsBasisStatusZero, kHighsIisBoundBoxed, kHighsIisBoundLower, kHighsIisBoundUpper,
    kHighsIisStrategyFromLpRowPriority,
};

use crate::instance::{Penalties, RangingRecord, RawIis};
use crate::{Highs, HighsError, ModelStatus, OptionValue, configure};

/// The result of an analysis, or why it failed.
type Analyzed<T> = Result<T, AnalysisError<HighsError>>;

/// The failure of `analysis` inside HiGHS.
const fn failed(analysis: Analysis) -> AnalysisError<HighsError> {
    AnalysisError::Backend(HighsError::Analysis(analysis))
}

/// The duals of the last run, in culmen's convention, which is HiGHS's.
///
/// # Errors
///
/// When HiGHS reports no dual solution.
pub(crate) fn duals(highs: &Highs, program: &Program) -> Analyzed<Duals> {
    if highs.instance.info_int(c"dual_solution_status") != Some(2) {
        return Err(failed(Analysis::Duals));
    }
    let (cols, mut rows) = highs.instance.duals();
    rows.truncate(program.num_rows());
    Ok(Duals {
        rows: rows.into(),
        cols: cols.into(),
    })
}

/// The basis of the last run, `None` when HiGHS holds no valid one.
///
/// # Errors
///
/// When HiGHS reports a status outside `kHighsBasisStatus*`.
pub(crate) fn basis(highs: &Highs, program: &Program) -> Analyzed<Option<Basis>> {
    let Some((cols, mut rows)) = highs.instance.basis() else {
        return Ok(None);
    };
    rows.truncate(program.num_rows());
    let typed = |raw: Vec<HighsInt>| raw.into_iter().map(status).collect::<Option<Vec<_>>>();
    match (typed(cols), typed(rows)) {
        (Some(cols), Some(rows)) => Ok(Some(Basis {
            cols: cols.into(),
            rows: rows.into(),
        })),
        _ => Err(failed(Analysis::Basis)),
    }
}

/// Sets the basis the next run starts from.
///
/// # Errors
///
/// When HiGHS refuses the basis, as a singular one.
///
/// # Panics
///
/// When `basis` does not hold one status per column and per row of HiGHS's model, which the solver's check rules out.
pub(crate) fn set_basis(highs: &mut Highs, basis: &Basis) -> Analyzed<()> {
    let cols: Vec<HighsInt> = basis.cols.iter().map(|&s| raw_status(s)).collect();
    let rows: Vec<HighsInt> = basis.rows.iter().map(|&s| raw_status(s)).collect();
    match highs.instance.set_basis(&cols, &rows) {
        crate::CallStatus::Error => Err(failed(Analysis::Basis)),
        crate::CallStatus::Ok | crate::CallStatus::Warning => Ok(()),
    }
}

/// HiGHS's number of each basis status.
const STATUSES: [(HighsInt, BasisStatus); 5] = [
    (kHighsBasisStatusLower, BasisStatus::Lower),
    (kHighsBasisStatusBasic, BasisStatus::Basic),
    (kHighsBasisStatusUpper, BasisStatus::Upper),
    (kHighsBasisStatusZero, BasisStatus::Zero),
    (kHighsBasisStatusNonbasic, BasisStatus::NonBasic),
];

/// The basis status of HiGHS's number for it, `None` for a number outside `kHighsBasisStatus*`.
fn status(raw: HighsInt) -> Option<BasisStatus> {
    STATUSES
        .iter()
        .find(|&&(number, _)| number == raw)
        .map(|&(_, status)| status)
}

/// HiGHS's number for a basis status.
const fn raw_status(status: BasisStatus) -> HighsInt {
    match status {
        BasisStatus::Lower => kHighsBasisStatusLower,
        BasisStatus::Basic => kHighsBasisStatusBasic,
        BasisStatus::Upper => kHighsBasisStatusUpper,
        BasisStatus::Zero => kHighsBasisStatusZero,
        BasisStatus::NonBasic => kHighsBasisStatusNonbasic,
    }
}

/// The ranging of the last run's optimal basis.
///
/// # Errors
///
/// When HiGHS refuses, as without a simplex basis.
pub(crate) fn ranging(highs: &mut Highs, program: &Program) -> Analyzed<Ranging> {
    let raw = highs.instance.ranging().ok_or(failed(Analysis::Ranging))?;
    let n = program.num_cols();
    let pair = |down: &RangingRecord, up: &RangingRecord| -> Vec<Ranged> {
        (0..down.value.len())
            .map(|i| Ranged {
                down: limit(down, i, n),
                up: limit(up, i, n),
            })
            .collect()
    };
    Ok(Ranging {
        costs: pair(&raw.cost_down, &raw.cost_up).into(),
        col_bounds: pair(&raw.col_bound_down, &raw.col_bound_up).into(),
        row_bounds: pair(&raw.row_bound_down, &raw.row_bound_up).into(),
    })
}

/// Entry `i` of a ranging record, its variables numbered as HiGHS numbers them over `n` columns.
///
/// # Panics
///
/// When the record has no entry `i`.
fn limit(record: &RangingRecord, i: usize, n: usize) -> Limit {
    Limit {
        value: record.value[i],
        objective: record.objective[i],
        entering: line(record.entering[i], n),
        leaving: line(record.leaving[i], n),
    }
}

/// HiGHS's variable `index` over `n` columns: a column below `n`, a row from `n` on; `None` for a negative index.
fn line(index: HighsInt, n: usize) -> Option<Line> {
    let index = usize::try_from(index).ok()?;
    Some(if index < n {
        Line::Col(Col::new(index))
    } else {
        Line::Row(Row::new(index - n))
    })
}

/// A primal ray of the last run's unbounded LP.
///
/// HiGHS 1.15 reports no primal ray for an LP without rows; there the LP is unbounded along one column whose cost
/// improves the objective toward an infinite bound, which is the ray.
///
/// # Errors
///
/// When HiGHS fails to compute one.
pub(crate) fn primal_ray(highs: &mut Highs, program: &Program) -> Analyzed<Option<TiVec<Col, f64>>> {
    if program.num_rows() == 0 {
        return Ok(column_ray(program));
    }
    let ray = highs.instance.primal_ray().map_err(|_| failed(Analysis::Rays))?;
    Ok(ray.map(TiVec::from))
}

/// The ray of an LP without rows: the unit direction of the first column along which the objective improves without
/// a bound; `None` when there is none.
fn column_ray(program: &Program) -> Option<TiVec<Col, f64>> {
    let improving = |cost: f64| match program.sense() {
        Sense::Minimize => -cost,
        Sense::Maximize => cost,
    };
    let (col, step) = program.columns().iter_enumerated().find_map(|(col, c)| {
        let step = improving(c.cost).signum();
        let open = if step > 0.0 { c.bounds.upper() } else { c.bounds.lower() };
        (c.cost != 0.0 && open.is_infinite()).then_some((col, step))
    })?;
    Some(program.cols().map(|c| if c == col { step } else { 0.0 }).collect())
}

/// A dual ray of the last run's infeasible LP. HiGHS orients it as [`Solver::dual_ray`](culmen_core::Solver::dual_ray)
/// states it, in either sense.
///
/// # Errors
///
/// When HiGHS fails to compute one.
pub(crate) fn dual_ray(highs: &mut Highs) -> Analyzed<Option<TiVec<Row, f64>>> {
    let ray = highs.instance.dual_ray().map_err(|_| failed(Analysis::Rays))?;
    Ok(ray.map(TiVec::from))
}

/// An IIS of the loaded program, computed with HiGHS's options reset to the backend's and its full IIS strategy.
///
/// # Errors
///
/// When an option is refused, HiGHS fails, or it finds no IIS of a program it found infeasible.
pub(crate) fn iis(highs: &mut Highs, program: &Program) -> Analyzed<Option<Iis>> {
    configure(&mut highs.instance, &highs.options, &Settings::new()).map_err(AnalysisError::Backend)?;
    highs
        .instance
        .set_option("iis_strategy", &OptionValue::Int(kHighsIisStrategyFromLpRowPriority))
        .map_err(AnalysisError::Backend)?;
    let raw = highs.instance.iis().ok_or(failed(Analysis::Iis))?;
    let iis = typed_iis(&raw, program);
    match (
        iis.cols.is_empty() && iis.rows.is_empty(),
        highs.instance.model_status(),
    ) {
        (false, _) => Ok(Some(iis)),
        (true, ModelStatus::Infeasible) => Err(failed(Analysis::Iis)),
        (true, _) => Ok(None),
    }
}

/// HiGHS's IIS in culmen's types: free columns, which bring no bound, and lines the program lacks are left out.
fn typed_iis(raw: &RawIis, program: &Program) -> Iis {
    let entries = |entries: &[(HighsInt, HighsInt)], len: usize| {
        entries
            .iter()
            .filter_map(move |&(index, bound)| {
                let index = usize::try_from(index).ok().filter(|&i| i < len)?;
                Some((index, side(bound)?))
            })
            .collect::<Vec<_>>()
    };
    Iis {
        cols: entries(&raw.cols, program.num_cols())
            .into_iter()
            .map(|(i, s)| (Col::new(i), s))
            .collect(),
        rows: entries(&raw.rows, program.num_rows())
            .into_iter()
            .map(|(i, s)| (Row::new(i), s))
            .collect(),
    }
}

/// The side of each of HiGHS's `kHighsIisBound*` that brings a bound.
const SIDES: [(HighsInt, Side); 3] = [
    (kHighsIisBoundLower, Side::Lower),
    (kHighsIisBoundUpper, Side::Upper),
    (kHighsIisBoundBoxed, Side::Both),
];

/// The side of HiGHS's `kHighsIisBound*`; `None` for a free line, which brings no bound.
fn side(raw: HighsInt) -> Option<Side> {
    SIDES.iter().find(|&&(number, _)| number == raw).map(|&(_, side)| side)
}

/// The feasibility relaxation of the loaded program under `weights`, solved with HiGHS's options reset to the
/// backend's; `None` when no point meets the hard bounds and rows.
///
/// HiGHS 1.15 does not report whether its elastic program was feasible, so the point it returns is checked against
/// the hard bounds, rows and integrality instead. Its elastic program keeps a semi column within its bounds, never at
/// zero (it solves the program without `Highs::run`, which alone reformulates semi columns), so its point can miss the
/// least violation by any amount; a program with semi columns is refused.
///
/// # Errors
///
/// [`HighsError::AnalysisFeature`] for a program with semi columns; when an option is refused or HiGHS fails.
pub(crate) fn relax(highs: &mut Highs, program: &Program, weights: &Weights) -> Analyzed<Option<Relaxed>> {
    let semi = ProgramFeatures::of(&[ProgramFeature::SemiContinuous, ProgramFeature::SemiInteger]);
    if let Some(feature) = program.features().intersection(semi).iter().next() {
        return Err(AnalysisError::Backend(HighsError::AnalysisFeature {
            analysis: Analysis::FeasibilityRelaxation,
            feature,
        }));
    }
    configure(&mut highs.instance, &highs.options, &Settings::new()).map_err(AnalysisError::Backend)?;
    let penalties = penalties(weights, highs.instance.num_row());
    let values = highs
        .instance
        .feasibility_relaxation(&penalties)
        .ok_or(failed(Analysis::FeasibilityRelaxation))?;
    let relaxed = Relaxed { values: values.into() };
    Ok(meets_hard(&relaxed, program, weights).then_some(relaxed))
}

/// The penalties HiGHS takes for `weights`, `-1` for a hard bound or row, over `num_rows` rows of HiGHS's model.
fn penalties(weights: &Weights, num_rows: usize) -> Penalties {
    let raw = |p: &Option<culmen_core::Penalty>| p.map_or(-1.0, culmen_core::Penalty::get);
    Penalties {
        lower: weights.lower.iter().map(raw).collect(),
        upper: weights.upper.iter().map(raw).collect(),
        rows: weights
            .rows
            .iter()
            .map(raw)
            .chain(core::iter::repeat(-1.0))
            .take(num_rows)
            .collect(),
    }
}

/// HiGHS's primal feasibility tolerance, relative to the size of a bound above one.
const TOLERANCE: f64 = 1e-6;

/// Whether the point meets every hard bound and row of `program`, and its integrality, within [`TOLERANCE`].
///
/// # Panics
///
/// When the point holds fewer values than the program has columns, or the weights lack an entry of the program.
fn meets_hard(relaxed: &Relaxed, program: &Program, weights: &Weights) -> bool {
    let hard_met = relaxed
        .violations(program)
        .filter(|v| weights.price(v.line, v.bound).is_none())
        .all(|v| v.amount <= TOLERANCE * Bound::of(v.bound, v.line.bounds(program)).abs().max(1.0));
    let whole = program
        .columns()
        .iter_enumerated()
        .filter(|(_, c)| c.integrality.is_integral())
        .all(|(col, _)| (relaxed.values[col] - relaxed.values[col].round()).abs() <= TOLERANCE);
    hard_met && whole
}
