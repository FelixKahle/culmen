// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a solver that meets its program only to tolerances may return, stated against the exact reference.
//!
//! A solver's point may pass each row side and column bound by its feasibility tolerance and lie off a whole number by
//! its integrality tolerance, and its search may stop short of the optimum by its dual feasibility tolerance and its
//! gaps. So its optimum is not the exact optimum `opt` of the program but lies in a window: no better than the exact
//! optimum of the program loosened by what the tolerances allow ([`Tolerances::loosened`]), and no worse than `opt` by
//! more than the optimality slack. The window and the loosened program are computed here from the tolerances'
//! definitions and the exact reference, never by asking the solver.

use culmen_core::{Bounds, Col, Integrality, Program, ProgramPlan, Sense, TiSlice};

use crate::reference::{Reference, solve};

/// The relative rounding of the floating-point sums a solver and these checks compute: activities and objectives of
/// tiny programs are sums of a dozen products of small numbers, each exact to `2⁻⁵³` of its size, so `1e-12` of the
/// largest magnitude involved covers their rounding with a wide margin.
const ROUNDING: f64 = 1e-12;

/// A solver's tolerances, each as the solver defines it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    /// How far a row's activity or a column's value may pass a finite side `s` and still meet it: this times
    /// `max(1, |s|, |value|)` when `relative`, this itself when not.
    pub feasibility: f64,
    /// Whether `feasibility` scales with the side and the value.
    pub relative: bool,
    /// How far an integer column's value, or a semi column's value standing for zero, may lie from a whole number or
    /// from zero.
    pub integrality: f64,
    /// How far a reduced cost may have the wrong sign at a basis the solver takes as optimal.
    pub dual: f64,
    /// The difference between the best solution's objective and the bound at which the search may stop.
    pub absolute_gap: f64,
    /// The same difference relative to the objective's magnitude.
    pub relative_gap: f64,
}

impl Tolerances {
    /// SCIP 10's defaults with `limits/gap` and `limits/absgap` at zero, their defaults.
    ///
    /// - `numerics/feastol` = 1e-6 is relative: SCIP holds `activity` to a side `s` when
    ///   `(s - activity) / max(1, |s|, |activity|) <= feastol` (`SCIPsetIsFeasLT` over `SCIPrelDiff`), for a linear
    ///   row and for a column bound alike, and an integer value integral when it lies within `feastol` of a whole
    ///   number (`SCIPsetIsFeasIntegral`).
    /// - `numerics/dualfeastol` = 1e-7 bounds the reduced costs of an LP SCIP takes as optimal, and is the tolerance
    ///   below which a priced column does not improve.
    /// - A node whose bound is within `numerics/epsilon` = 1e-9 of the incumbent is pruned, and a gap of zero is met
    ///   within the same epsilon.
    pub const SCIP: Self = Self {
        feasibility: 1e-6,
        relative: true,
        integrality: 1e-6,
        dual: 1e-7,
        absolute_gap: 1e-9,
        relative_gap: 1e-9,
    };

    /// HiGHS 1.16's MIP defaults: `mip_feasibility_tolerance` = 1e-6, absolute, for rows, bounds and integrality;
    /// `dual_feasibility_tolerance` = 1e-7; `mip_abs_gap` = 1e-6 and `mip_rel_gap` = 1e-4, either of which ends the
    /// search. A test that requests another gap overrides the field.
    pub const HIGHS: Self = Self {
        feasibility: 1e-6,
        relative: false,
        integrality: 1e-6,
        dual: 1e-7,
        absolute_gap: 1e-6,
        relative_gap: 1e-4,
    };

    /// How far a value may pass the finite side `side`, whatever the value.
    ///
    /// Relative, `(s - v) <= t · max(1, |s|, |v|)` allows at most `t / (1 - t) · max(1, |s|)`: when `|v|` is the
    /// largest, `v` lies beyond a negative `s` and the condition is `v >= s / (1 - t)`.
    fn slack(&self, side: f64) -> f64 {
        if self.relative {
            self.feasibility / (1.0 - self.feasibility) * side.abs().max(1.0)
        } else {
            self.feasibility
        }
    }

    /// `bounds` with each finite end moved outward by its slack and `extra`, rounded outward.
    ///
    /// # Panics
    ///
    /// When the widened ends do not make bounds, which widening valid bounds never causes.
    fn widened(&self, bounds: Bounds, extra: f64) -> Bounds {
        let (lower, upper) = (bounds.lower(), bounds.upper());
        let lower = if lower.is_finite() {
            (lower - self.slack(lower) - extra).next_down()
        } else {
            lower
        };
        let upper = if upper.is_finite() {
            (upper + self.slack(upper) + extra).next_up()
        } else {
            upper
        };
        Bounds::new(lower, upper).expect("widened bounds are bounds")
    }

    /// The program every point the solver may call feasible rounds into: each row side and column bound moved
    /// outward by its feasibility slack, each row's sides further by the integrality tolerance times the absolute
    /// coefficients of its columns that are not continuous, and integrality kept.
    ///
    /// Rounding such a point's integer columns to the nearest whole number, and a semi column's value within the
    /// integrality tolerance of zero to zero, moves each row's activity by at most that further amount and keeps an
    /// integer column within its loosened bounds, so the rounded point is feasible here.
    ///
    /// # Panics
    ///
    /// When a semi column's lower bound lies within its slack of zero, so the loosened column holds zero in its
    /// bounds and does not build.
    #[must_use]
    pub fn loosened(&self, program: &Program) -> Program {
        let mut plan = ProgramPlan::from(program.clone());
        for row in &mut plan.rows {
            let rounding: f64 = row
                .terms
                .iter()
                .filter(|&&(col, _)| program.columns()[col].integrality != Integrality::Continuous)
                .map(|&(_, a)| a.abs() * self.integrality)
                .sum();
            row.bounds = self.widened(row.bounds, rounding);
        }
        for column in &mut plan.columns {
            column.bounds = self.widened(column.bounds, 0.0);
        }
        plan.build()
            .expect("a semi column's lower bound beyond its slack, so the loosened program builds")
    }

    /// The window the solver's optimal objective lies in, `None` when `optimum`, the exact reference of `program`,
    /// is infeasible.
    ///
    /// Better than the optimum, it reaches the exact optimum of [`Tolerances::loosened`] plus the integrality
    /// tolerance times the absolute costs of the columns that are not continuous, which is what rounding the point
    /// into the loosened program changes its objective by. Worse than the optimum, it reaches the dual tolerance
    /// times the sum of the columns' ranges, which is the most a basis with reduced costs that far off the sign
    /// loses, plus the larger gap: the absolute gap, or the relative one times `2 · max(1, |opt|)`, which bounds its
    /// denominator, the smaller of the incumbent's and the bound's magnitudes. Both ends allow the rounding of the
    /// objective's sum.
    ///
    /// # Panics
    ///
    /// As [`Tolerances::loosened`] and [`solve`]; and when the program is feasible and its loosening is not, which
    /// loosening never causes.
    #[must_use]
    pub fn window(&self, program: &Program, optimum: &Reference) -> Option<Window> {
        let opt = optimum.objective()?;
        let loose = solve(&self.loosened(program))
            .objective()
            .expect("the loosening of a feasible program is feasible");
        let columns = program.columns();
        let rounded: f64 = columns
            .iter()
            .filter(|c| c.integrality != Integrality::Continuous)
            .map(|c| c.cost.abs() * self.integrality)
            .sum();
        let ranges: f64 = columns
            .iter()
            .map(|c| c.bounds.upper().max(0.0) - c.bounds.lower().min(0.0))
            .sum();
        let gap = self.absolute_gap.max(self.relative_gap * 2.0 * opt.abs().max(1.0));
        let rounding = ROUNDING * opt.abs().max(loose.abs()).max(1.0);
        let (better, worse) = (rounded + rounding, self.dual * ranges + gap + rounding);
        Some(match program.sense() {
            Sense::Minimize => Window {
                least: loose - better,
                greatest: opt + worse,
            },
            Sense::Maximize => Window {
                least: opt - worse,
                greatest: loose + better,
            },
        })
    }

    /// Why `values` is not a point the solver may call feasible for `program`, naming the first column or row it
    /// passes by more than these tolerances allow; `None` when it is one.
    ///
    /// # Panics
    ///
    /// When `values` holds fewer values than the program has columns.
    #[must_use]
    pub fn stray(&self, program: &Program, values: &TiSlice<Col, f64>) -> Option<String> {
        self.stray_col(program, values)
            .or_else(|| self.stray_row(program, values, false))
    }

    /// As [`Tolerances::stray`], for a point the solver checked only against its presolved rows, as SCIP checks the
    /// best solution of a solve that prices.
    ///
    /// SCIP holds a row of a solve that prices to `numerics/feastol` (1e-6, relative) only in its transformed problem:
    /// its check in the original problem skips modifiable rows (`SCIPsolCheckOrig`), and every row of such a solve is
    /// modifiable. There presolve has replaced each column it fixed or shifted by the constant it stands for
    /// (`applyFixings` in `cons_linear.c`: a fixed column's value, or the lower bound of an integer column it made
    /// binary) and moved that constant into the row's sides, leaving the row otherwise unscaled. A side `s` becomes
    /// `s - k`, the activity moves by the same `k`, and the relative tolerance then allows
    /// `t / (1 - t) · max(1, |s - k|)`, which can exceed `t / (1 - t) · max(1, |s|)`. So each finite side is loosened
    /// further by `t / (1 - t) · K`, where `K = Σ |a_j| · m_j` bounds `|k|`, with `m_j` the largest magnitude of
    /// column `j`'s finite bounds and its value. An absolute tolerance does not change with the side.
    ///
    /// # Panics
    ///
    /// As [`Tolerances::stray`].
    #[must_use]
    pub fn stray_presolved(&self, program: &Program, values: &TiSlice<Col, f64>) -> Option<String> {
        self.stray_col(program, values)
            .or_else(|| self.stray_row(program, values, true))
    }

    /// The first column of `values` that passes its bounds or integrality by more than these tolerances allow.
    ///
    /// # Panics
    ///
    /// As [`Tolerances::stray`].
    fn stray_col(&self, program: &Program, values: &TiSlice<Col, f64>) -> Option<String> {
        for (col, column) in program.columns().iter_enumerated() {
            let value = values[col];
            let within = self
                .widened(column.bounds, ROUNDING * value.abs().max(1.0))
                .violation(value)
                == 0.0;
            let whole = (value - value.round()).abs() <= self.integrality;
            let met = match column.integrality {
                Integrality::Continuous => within,
                Integrality::Integer => within && whole,
                Integrality::SemiContinuous => within || value.abs() <= self.integrality,
                Integrality::SemiInteger => (within && whole) || value.abs() <= self.integrality,
            };
            if !met {
                return Some(format!("{col:?} at {value} passes {column:?}"));
            }
        }
        None
    }

    /// The first row whose activity at `values` passes its sides by more than these tolerances allow, each side
    /// loosened further as [`Tolerances::stray_presolved`] states when `presolved`.
    ///
    /// # Panics
    ///
    /// As [`Tolerances::stray`].
    fn stray_row(&self, program: &Program, values: &TiSlice<Col, f64>, presolved: bool) -> Option<String> {
        let reach = |col: Col| {
            let bounds = program.columns()[col].bounds;
            [bounds.lower(), bounds.upper(), values[col]]
                .into_iter()
                .filter(|v| v.is_finite())
                .fold(0.0, |m: f64, v| m.max(v.abs()))
        };
        for row in program.rows() {
            let view = program.row(row);
            let size: f64 = view.entries().map(|(col, a)| (a * values[col]).abs()).sum();
            let moved = if presolved && self.relative {
                let k: f64 = view.entries().map(|(col, a)| a.abs() * reach(col)).sum();
                self.feasibility / (1.0 - self.feasibility) * k
            } else {
                0.0
            };
            let activity = view.activity(values);
            if self
                .widened(view.bounds(), ROUNDING * size.max(1.0) + moved)
                .violation(activity)
                > 0.0
            {
                return Some(format!("{row:?} at {activity} passes {:?}", view.bounds()));
            }
        }
        None
    }
}

/// The objective values a solver's optimum may take, from `least` to `greatest`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    /// The least objective value in the window.
    pub least: f64,
    /// The greatest objective value in the window.
    pub greatest: f64,
}

impl Window {
    /// Whether the window holds `objective`.
    #[must_use]
    pub fn contains(&self, objective: f64) -> bool {
        (self.least..=self.greatest).contains(&objective)
    }

    /// Whether `bound`, a bound on the optimum under `sense`, is no tighter than the window's worse end allows: a
    /// valid bound of a minimization lies at most at `greatest`, of a maximization at least at `least`.
    #[must_use]
    pub fn admits_bound(&self, sense: Sense, bound: f64) -> bool {
        match sense {
            Sense::Minimize => bound <= self.greatest,
            Sense::Maximize => bound >= self.least,
        }
    }
}

#[cfg(test)]
mod tests {
    use culmen_core::{Linear, Model, Var};

    use super::*;

    #[test]
    fn test_a_relative_tolerance_loosens_a_side_by_its_size() {
        // Maximize x over x <= 4 with x in [0, 10]: SCIP may take x up to 4 + 4e-6 / (1 - 1e-6), and may stop short
        // of 4 by the dual tolerance over the range 10, 1e-6, and the relative epsilon gap 1e-9 · 2 · 4.
        let mut m = Model::new();
        let x = m.add_var(Var::continuous().upper(10));
        m.add_constraint(x.leq(4));
        m.maximize(x);
        let program = m.build().unwrap();
        let window = Tolerances::SCIP.window(&program, &solve(&program)).unwrap();
        let loose = 4.0 + 4e-6 / (1.0 - 1e-6);
        assert!((window.greatest - loose).abs() <= 1e-11, "{window:?}");
        assert!((window.least - (4.0 - 1e-6 - 8e-9)).abs() <= 1e-11, "{window:?}");
        let stray = |v: f64| Tolerances::SCIP.stray(&program, TiSlice::from_ref(&[v][..]));
        assert_eq!(stray(4.000_003_9), None);
        assert!(stray(4.000_004_1).is_some());
        assert!(window.admits_bound(Sense::Maximize, 4.0) && !window.admits_bound(Sense::Maximize, 3.99999));
    }

    #[test]
    fn test_an_integer_column_off_a_whole_number_loosens_its_rows_and_the_objective() {
        // Minimize -x - 2y over x + y <= 1, x integer in [0, 1], y continuous in [0, 1]: the optimum is y = 1 at -2.
        // HiGHS may hold x at 1e-6 with the row at 1 + 1e-6; rounded, x is 0, and the loosened program, whose row
        // reaches 1 + 2e-6 for its own 1e-6 and x's rounding, puts y at its loosened bound 1 + 1e-6, worth -2 - 2e-6,
        // to which x's cost of magnitude 1 adds 1e-6. The worse end is the dual 1e-7 over the ranges 2 and the
        // absolute gap 1e-6.
        let mut m = Model::new();
        let x = m.add_var(Var::integer().upper(1));
        let y = m.add_var(Var::continuous().upper(1));
        m.add_constraint((x + y).leq(1));
        m.minimize(-1.0 * x - 2.0 * y);
        let program = m.build().unwrap();
        let tolerances = Tolerances {
            relative_gap: 0.0,
            ..Tolerances::HIGHS
        };
        let window = tolerances.window(&program, &solve(&program)).unwrap();
        assert!((window.least - (-2.0 - 3e-6)).abs() <= 1e-11, "{window:?}");
        assert!((window.greatest - (-2.0 + 2e-7 + 1e-6)).abs() <= 1e-11, "{window:?}");
        let stray = |values: &[f64]| tolerances.stray(&program, TiSlice::from_ref(values));
        assert_eq!(stray(&[1e-6, 1.0]), None);
        assert!(
            stray(&[1e-6, 1.0 + 1e-6]).is_some(),
            "the row at 1 + 2e-6 passes its side by 2e-6"
        );
        assert!(stray(&[0.5, 0.5]).is_some());
        assert_eq!(tolerances.window(&program, &Reference::Infeasible), None);
    }

    #[test]
    fn test_a_presolved_row_is_held_relative_to_its_moved_side() {
        // 2x + 2y <= 2 with x in [-1, 0] and y in [0, 2]. Presolve fixing x at -1 leaves 2y <= 4, which SCIP holds to
        // 1e-6 relative to 4: y = 2 + 1.9e-6 puts 2y at 4 + 3.8e-6, within 4e-6, and the row of the program at
        // 2 + 3.8e-6, beyond its own 2e-6. The presolved check loosens the side by 1e-6 / (1 - 1e-6) times
        // 2·1 + 2·2 = 6 more, to about 8e-6, and still refuses 2 + 1e-5.
        let mut m = Model::new();
        let x = m.add_var(Var::continuous().lower(-1).upper(0));
        let y = m.add_var(Var::continuous().upper(2));
        m.add_constraint((2.0 * x + 2.0 * y).leq(2));
        m.minimize(x);
        let program = m.build().unwrap();
        let point = |values: &[f64]| TiSlice::from_ref(values).to_owned();
        let shifted = point(&[-1.0, 2.0 + 1.9e-6]);
        assert!(Tolerances::SCIP.stray(&program, &shifted).is_some());
        assert_eq!(Tolerances::SCIP.stray_presolved(&program, &shifted), None);
        let beyond = point(&[-1.0 + 5e-6, 2.0]);
        assert!(Tolerances::SCIP.stray_presolved(&program, &beyond).is_some());
        let absolute = Tolerances::HIGHS;
        assert_eq!(
            absolute.stray(&program, &shifted).is_some(),
            absolute.stray_presolved(&program, &shifted).is_some()
        );
    }
}
