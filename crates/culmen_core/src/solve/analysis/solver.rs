// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The analyses as methods of a [`Solver`], each checked the same way for every backend before the backend is asked.

use typed_index_collections::TiVec;

use super::{Analysis, AnalysisError, Basis, Duals, Iis, Ranging, Relaxed, Weights};
use crate::{Backend, Col, Outcome, Row, Solver};

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

/// The result of an analysis, or why it was refused.
type Analyzed<T, B> = Result<T, AnalysisError<<B as Backend>::Error>>;

impl<B: Backend> Solver<B> {
    /// The row duals and reduced costs of the last solve, which ended optimal, in the sign convention of [`Duals`].
    ///
    /// A row's activity is read from the solution with [`Program::activities`](crate::Program::activities).
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] when the backend lacks [`Analysis::Duals`], [`AnalysisError::Class`] for a
    /// MIP, [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::NotOptimal`] when it did not end
    /// optimal, [`AnalysisError::Backend`] when the backend fails.
    pub fn duals(&self) -> Analyzed<Duals, B> {
        self.optimal(Analysis::Duals)?;
        self.backend.duals(&self.program)
    }

    /// The basis the last solve ended with; `None` when the backend holds none, as after an interior-point solve
    /// without crossover.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] when the backend lacks [`Analysis::Basis`], [`AnalysisError::Class`] for any
    /// program but an LP, [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::Backend`] when the
    /// backend fails.
    pub fn basis(&self) -> Analyzed<Option<Basis>, B> {
        self.solved(Analysis::Basis)?;
        self.backend.basis(&self.program)
    }

    /// Sets the basis the next solve starts from, after [`Basis::check`] found that it fits the program. The last
    /// solve's result is no longer read.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] when the backend lacks [`Analysis::Basis`], [`AnalysisError::Class`] for any
    /// program but an LP, [`AnalysisError::Misfit`] when the basis does not fit the program,
    /// [`AnalysisError::Backend`] when the backend refuses it.
    pub fn set_basis(&mut self, basis: &Basis) -> Analyzed<(), B> {
        self.admit(Analysis::Basis)?;
        basis.check(&self.program)?;
        self.ended = None;
        self.backend.set_basis(&self.program, basis)
    }

    /// How far each cost and bound may move before the last solve's optimal basis changes.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] when the backend lacks [`Analysis::Ranging`], [`AnalysisError::Class`] for
    /// any program but an LP, [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::NotOptimal`]
    /// when it did not end optimal, [`AnalysisError::Backend`] when the backend fails.
    pub fn ranging(&mut self) -> Analyzed<Ranging, B> {
        self.optimal(Analysis::Ranging)?;
        self.backend.ranging(&self.program)
    }

    /// A direction of unboundedness of the last solve's LP: a `d` that every row's and column's recession allows
    /// (`d ≥ 0` where only a lower bound is finite, `d ≤ 0` where only an upper one is, `d = 0` where both are) and
    /// along which the objective improves. `None` unless the last solve ended
    /// [`Unbounded`](Outcome::Unbounded) or [`InfeasibleOrUnbounded`](Outcome::InfeasibleOrUnbounded), or when the
    /// backend finds none.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] when the backend lacks [`Analysis::Rays`], [`AnalysisError::Class`] for any
    /// program but an LP, [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::Backend`] when the
    /// backend fails.
    pub fn primal_ray(&mut self) -> Analyzed<Option<TiVec<Col, f64>>, B> {
        match self.solved(Analysis::Rays)? {
            Ending::Unbounded | Ending::Either => self.backend.primal_ray(&self.program),
            Ending::Optimal | Ending::Infeasible | Ending::Other => Ok(None),
        }
    }

    /// A Farkas certificate of the last solve's infeasibility: row multipliers `y` for which the largest value of
    /// `yᵀAx` over the columns' bounds is below the least value of `yᵀr` over the rows' bounds `r`, with every
    /// infinite bound met by a zero. `None` unless the last solve ended [`Infeasible`](Outcome::Infeasible) or
    /// [`InfeasibleOrUnbounded`](Outcome::InfeasibleOrUnbounded), or when the backend finds none.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] when the backend lacks [`Analysis::Rays`], [`AnalysisError::Class`] for any
    /// program but an LP, [`AnalysisError::NotSolved`] without a solve to read, [`AnalysisError::Backend`] when the
    /// backend fails.
    pub fn dual_ray(&mut self) -> Analyzed<Option<TiVec<Row, f64>>, B> {
        match self.solved(Analysis::Rays)? {
            Ending::Infeasible | Ending::Either => self.backend.dual_ray(&self.program),
            Ending::Optimal | Ending::Unbounded | Ending::Other => Ok(None),
        }
    }

    /// An irreducible infeasible subsystem of the program's constraints; `None` when they are feasible. It needs no
    /// solve before, and the last solve's result is no longer read after, unless that solve proved the constraints
    /// feasible.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] when the backend lacks [`Analysis::Iis`], [`AnalysisError::Class`] for a MIP,
    /// [`AnalysisError::Backend`] when the backend fails.
    pub fn iis(&mut self) -> Analyzed<Option<Iis>, B> {
        self.admit(Analysis::Iis)?;
        if matches!(self.ended, Some(Ending::Optimal | Ending::Unbounded)) {
            return Ok(None);
        }
        self.ended = None;
        self.backend.iis(&self.program)
    }

    /// A point of least weighted violation of the program's bounds and rows under `weights`; `None` when no point
    /// meets the hard ones. It needs no solve before, and the last solve's result is no longer read after.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`] when the backend lacks [`Analysis::FeasibilityRelaxation`],
    /// [`AnalysisError::Class`] for a QP, [`AnalysisError::Misfit`] when the weights do not fit the program,
    /// [`AnalysisError::Backend`] when the backend fails.
    pub fn feasibility_relaxation(&mut self, weights: &Weights) -> Analyzed<Option<Relaxed>, B> {
        self.admit(Analysis::FeasibilityRelaxation)?;
        weights.check(&self.program)?;
        self.ended = None;
        self.backend.relax(&self.program, weights)
    }

    /// Checks that the backend provides `analysis` and that it is defined for the program's class and features.
    ///
    /// # Errors
    ///
    /// [`AnalysisError::Unsupported`], [`AnalysisError::Class`] or [`AnalysisError::Feature`].
    fn admit(&self, analysis: Analysis) -> Analyzed<(), B> {
        let class = self.program.class();
        if !B::ANALYSES.contains(analysis) {
            Err(AnalysisError::Unsupported {
                analysis,
                supported: B::ANALYSES,
            })
        } else if !analysis.classes().contains(class) {
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
        self.ended.ok_or(AnalysisError::NotSolved { analysis })
    }

    /// Admits `analysis` and checks that the solve it reads ended optimal.
    ///
    /// # Errors
    ///
    /// As [`Solver::solved`], and [`AnalysisError::NotOptimal`] when it did not end optimal.
    fn optimal(&self, analysis: Analysis) -> Analyzed<(), B> {
        match self.solved(analysis)? {
            Ending::Optimal => Ok(()),
            Ending::Infeasible | Ending::Unbounded | Ending::Either | Ending::Other => {
                Err(AnalysisError::NotOptimal { analysis })
            }
        }
    }
}
