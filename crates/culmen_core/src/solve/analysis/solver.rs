// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The analyses as methods of a [`Solver`], each in an `impl` block bounded by its capability trait and checked the
//! same way for every backend before the backend is asked.

use typed_index_collections::TiVec;

use super::{Analysis, AnalysisError, Basis, Duals, Iis, Ranging, Relaxed, Weights};
use crate::{
    Backend, BasisAnalysis, Col, DualAnalysis, IisAnalysis, Outcome, RangingAnalysis, RayAnalysis, RelaxationAnalysis,
    Row, Solver,
};

/// How the last solve ended, as far as the analyses read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ending {
    /// [`Outcome::Optimal`].
    Optimal,
    /// [`Outcome::Infeasible`].
    Infeasible,
    /// [`Outcome::Unbounded`].
    Unbounded,
    /// [`Outcome::InfeasibleOrUnbounded`].
    Either,
    /// Any other outcome.
    Other,
}

impl Ending {
    /// How `outcome` ended.
    pub(crate) const fn of(outcome: &Outcome) -> Self {
        match outcome {
            Outcome::Optimal(_) => Self::Optimal,
            Outcome::Infeasible => Self::Infeasible,
            Outcome::Unbounded => Self::Unbounded,
            Outcome::InfeasibleOrUnbounded => Self::Either,
            Outcome::GapReached { .. } | Outcome::Stopped { .. } => Self::Other,
        }
    }
}

/// How the last solve ended, and whether the backend still holds that solve's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ended {
    pub(crate) ending: Ending,
    /// Whether the backend's state is still that solve's; `false` once a ray asked the backend, which may run its
    /// solver again to find one. The rays read only `ending` and stay answered.
    pub(crate) held: bool,
}

impl Ended {
    /// A solve that ended with `outcome`, whose result the backend holds.
    pub(crate) const fn of(outcome: &Outcome) -> Self {
        Self {
            ending: Ending::of(outcome),
            held: true,
        }
    }
}

/// The result of an analysis, or why it was refused.
type Analyzed<T, B> = Result<T, AnalysisError<<B as Backend>::Error>>;

impl<B: DualAnalysis> Solver<B> {
    /// The row duals and reduced costs of the last solve, which ended optimal, in the sign convention of [`Duals`].
    ///
    /// A row's activity is read from the solution with [`Program::activities`](crate::Program::activities).
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] for a MIP, [`AnalysisError::Feature`] for several objectives,
    /// [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::NotOptimal`] when it did not end
    /// optimal, [`AnalysisError::Backend`] when the backend fails.
    pub fn duals(&self) -> Analyzed<Duals, B> {
        self.optimal(Analysis::Duals)?;
        self.backend.duals(&self.program).map_err(AnalysisError::Backend)
    }
}

impl<B: BasisAnalysis> Solver<B> {
    /// The basis the last solve ended with; `None` when the backend holds none, as after an interior-point solve
    /// without crossover.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] for any program but an LP, [`AnalysisError::Feature`] for several objectives,
    /// [`AnalysisError::NotSolved`] without a solve to read, as after a ray the backend was asked for, which may have
    /// run it again; [`AnalysisError::Backend`] when the backend fails.
    pub fn basis(&self) -> Analyzed<Option<Basis>, B> {
        self.held(Analysis::Basis)?;
        self.backend.basis(&self.program).map_err(AnalysisError::Backend)
    }

    /// Sets the basis the next solve starts from, after [`Basis::check`] found that it fits the program. The last
    /// solve's result is no longer read.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] for any program but an LP, [`AnalysisError::Feature`] for several objectives,
    /// [`AnalysisError::Misfit`] when the basis does not fit the program, [`AnalysisError::Backend`] when the backend
    /// refuses it, or the program it reloads after a failed edit.
    pub fn set_basis(&mut self, basis: &Basis) -> Analyzed<(), B> {
        self.admit(Analysis::Basis)?;
        basis.check(&self.program)?;
        self.sync().map_err(AnalysisError::Backend)?;
        self.ended = None;
        self.backend
            .set_basis(&self.program, basis)
            .map_err(AnalysisError::Backend)
    }
}

impl<B: RangingAnalysis> Solver<B> {
    /// How far each cost and bound may move before the last solve's optimal basis changes.
    ///
    /// It takes `&mut self` because the backend may compute ranging by work on its own state
    /// ([`RangingAnalysis::ranging`]).
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] for any program but an LP, [`AnalysisError::Feature`] for several objectives,
    /// [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::NotOptimal`] when it did not end
    /// optimal, [`AnalysisError::Backend`] when the backend fails.
    pub fn ranging(&mut self) -> Analyzed<Ranging, B> {
        self.optimal(Analysis::Ranging)?;
        self.backend.ranging(&self.program).map_err(AnalysisError::Backend)
    }
}

impl<B: RayAnalysis> Solver<B> {
    /// A direction of unboundedness of the last solve's LP: a `d` that every row's and column's recession allows
    /// (`d ≥ 0` where only a lower bound is finite, `d ≤ 0` where only an upper one is, `d = 0` where both are) and
    /// along which the objective improves. `None` unless the last solve ended
    /// [`Unbounded`](Outcome::Unbounded) or [`InfeasibleOrUnbounded`](Outcome::InfeasibleOrUnbounded), or when the
    /// backend finds none.
    ///
    /// It takes `&mut self` because the backend may run again to find the ray ([`RayAnalysis::primal_ray`]); once it
    /// was asked, [`Solver::basis`] no longer reads the last solve, and the rays still do.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] for any program but an LP, [`AnalysisError::Feature`] for several objectives,
    /// [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::Backend`] when the backend fails.
    pub fn primal_ray(&mut self) -> Analyzed<Option<TiVec<Col, f64>>, B> {
        match self.solved(Analysis::Rays)? {
            Ending::Unbounded | Ending::Either => {
                self.release();
                self.backend.primal_ray(&self.program).map_err(AnalysisError::Backend)
            }
            Ending::Optimal | Ending::Infeasible | Ending::Other => Ok(None),
        }
    }

    /// A Farkas certificate of the last solve's infeasibility: row multipliers `y` for which the largest value of
    /// `yᵀAx` over the columns' bounds is below the least value of `yᵀr` over the rows' bounds `r`, with every
    /// infinite bound met by a zero. `None` unless the last solve ended [`Infeasible`](Outcome::Infeasible) or
    /// [`InfeasibleOrUnbounded`](Outcome::InfeasibleOrUnbounded), or when the backend finds none.
    ///
    /// It takes `&mut self` because the backend may run again to find the ray ([`RayAnalysis::dual_ray`]); once it
    /// was asked, [`Solver::basis`] no longer reads the last solve, and the rays still do.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] for any program but an LP, [`AnalysisError::Feature`] for several objectives,
    /// [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::Backend`] when the backend fails.
    pub fn dual_ray(&mut self) -> Analyzed<Option<TiVec<Row, f64>>, B> {
        match self.solved(Analysis::Rays)? {
            Ending::Infeasible | Ending::Either => {
                self.release();
                self.backend.dual_ray(&self.program).map_err(AnalysisError::Backend)
            }
            Ending::Optimal | Ending::Unbounded | Ending::Other => Ok(None),
        }
    }
}

impl<B: IisAnalysis> Solver<B> {
    /// An irreducible infeasible subsystem of the program's constraints; `None` when they are feasible. It needs no
    /// solve before, and the last solve's result is no longer read after, unless that solve proved the constraints
    /// feasible.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] for a MIP, [`AnalysisError::Backend`] when the backend fails, or refuses the program
    /// it reloads after a failed edit.
    pub fn iis(&mut self) -> Analyzed<Option<Iis>, B> {
        self.admit(Analysis::Iis)?;
        if matches!(self.ended.map(|e| e.ending), Some(Ending::Optimal | Ending::Unbounded)) {
            return Ok(None);
        }
        self.sync().map_err(AnalysisError::Backend)?;
        self.ended = None;
        self.backend.iis(&self.program).map_err(AnalysisError::Backend)
    }
}

impl<B: RelaxationAnalysis> Solver<B> {
    /// A point of least weighted violation of the program's bounds and rows under `weights`; `None` when no point
    /// meets the hard ones. It needs no solve before, and the last solve's result is no longer read after.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] for a QP, [`AnalysisError::Misfit`] when the weights do not fit the program,
    /// [`AnalysisError::Backend`] when the backend fails, or refuses the program it reloads after a failed edit.
    pub fn feasibility_relaxation(&mut self, weights: &Weights) -> Analyzed<Option<Relaxed>, B> {
        self.admit(Analysis::FeasibilityRelaxation)?;
        weights.check(&self.program)?;
        self.sync().map_err(AnalysisError::Backend)?;
        self.ended = None;
        self.backend
            .relax(&self.program, weights)
            .map_err(AnalysisError::Backend)
    }
}

impl<B: Backend> Solver<B> {
    /// Checks that `analysis` is defined for the program's class and features.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Class`] or [`AnalysisError::Feature`].
    fn admit(&self, analysis: Analysis) -> Analyzed<(), B> {
        let class = self.program.class();
        if !analysis.classes().contains(class) {
            Err(AnalysisError::Class { analysis, class })
        } else if let Some(feature) = self.program.features().intersection(analysis.refused()).iter().next() {
            Err(AnalysisError::Feature { analysis, feature })
        } else {
            Ok(())
        }
    }

    /// Admits `analysis` and returns how the solve it reads ended.
    ///
    /// # Errors
    ///
    /// As [`Solver::admit`], and [`AnalysisError::NotSolved`] without a solve to read.
    fn solved(&self, analysis: Analysis) -> Analyzed<Ending, B> {
        self.admit(analysis)?;
        self.ended
            .map(|e| e.ending)
            .ok_or(AnalysisError::NotSolved { analysis })
    }

    /// Admits `analysis`, which reads the backend's state, and returns how the solve it reads ended.
    ///
    /// # Errors
    ///
    /// As [`Solver::solved`], and [`AnalysisError::NotSolved`] once a ray was asked of the backend.
    fn held(&self, analysis: Analysis) -> Analyzed<Ending, B> {
        self.admit(analysis)?;
        self.ended
            .filter(|e| e.held)
            .map(|e| e.ending)
            .ok_or(AnalysisError::NotSolved { analysis })
    }

    /// Records that the backend may no longer hold the last solve's result, as it may run again for a ray.
    fn release(&mut self) {
        if let Some(ended) = &mut self.ended {
            ended.held = false;
        }
    }

    /// Admits `analysis`, which reads the backend's state, and checks that the solve it reads ended optimal.
    ///
    /// # Errors
    ///
    /// As [`Solver::held`], and [`AnalysisError::NotOptimal`] when it did not end optimal.
    fn optimal(&self, analysis: Analysis) -> Analyzed<(), B> {
        match self.held(analysis)? {
            Ending::Optimal => Ok(()),
            Ending::Infeasible | Ending::Unbounded | Ending::Either | Ending::Other => {
                Err(AnalysisError::NotOptimal { analysis })
            }
        }
    }
}
