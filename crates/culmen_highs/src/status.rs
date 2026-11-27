// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HiGHS's statuses, all of them, and what a run reports beside the outcome.

use core::fmt;

use highs_sys::{
    kHighsModelStatusHighsInterrupt, kHighsModelStatusInfeasible, kHighsModelStatusInterrupt,
    kHighsModelStatusIterationLimit, kHighsModelStatusLoadError, kHighsModelStatusMemoryLimit,
    kHighsModelStatusModelEmpty, kHighsModelStatusModelError, kHighsModelStatusNotset, kHighsModelStatusObjectiveBound,
    kHighsModelStatusObjectiveTarget, kHighsModelStatusOptimal, kHighsModelStatusPostsolveError,
    kHighsModelStatusPresolveError, kHighsModelStatusSolutionLimit, kHighsModelStatusSolveError,
    kHighsModelStatusTimeLimit, kHighsModelStatusUnbounded, kHighsModelStatusUnboundedOrInfeasible,
    kHighsModelStatusUnknown, kHighsSolutionStatusFeasible, kHighsSolutionStatusInfeasible, kHighsStatusOk,
    kHighsStatusWarning,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// HiGHS's model status at the end of a run: every value of `HighsModelStatus` in HiGHS 1.16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ModelStatus {
    /// No status was set.
    NotSet,
    /// The model could not be loaded.
    LoadError,
    /// The model is invalid.
    ModelError,
    /// Presolve failed.
    PresolveError,
    /// The solver failed.
    SolveError,
    /// Postsolve failed.
    PostsolveError,
    /// The model has no columns.
    ModelEmpty,
    /// Optimal within HiGHS's tolerances and gaps.
    Optimal,
    /// Infeasible.
    Infeasible,
    /// Infeasible or unbounded.
    UnboundedOrInfeasible,
    /// Unbounded.
    Unbounded,
    /// The dual simplex reached the objective bound.
    ObjectiveBound,
    /// A MIP reached the objective target.
    ObjectiveTarget,
    /// The time limit was reached.
    TimeLimit,
    /// An iteration limit was reached.
    IterationLimit,
    /// The status is unknown.
    Unknown,
    /// A node, leaf or improving-solution limit was reached.
    SolutionLimit,
    /// A callback interrupted the run.
    Interrupt,
    /// The memory limit was reached.
    MemoryLimit,
    /// HiGHS interrupted itself, as a concurrent run does.
    HighsInterrupt,
    /// A value HiGHS 1.16 does not define.
    Other(i32),
}

impl ModelStatus {
    /// The status of HiGHS's number for it.
    #[expect(
        non_upper_case_globals,
        reason = "the patterns are HiGHS's constants under their C names"
    )]
    pub(crate) const fn from_raw(raw: i32) -> Self {
        match raw {
            kHighsModelStatusNotset => Self::NotSet,
            kHighsModelStatusLoadError => Self::LoadError,
            kHighsModelStatusModelError => Self::ModelError,
            kHighsModelStatusPresolveError => Self::PresolveError,
            kHighsModelStatusSolveError => Self::SolveError,
            kHighsModelStatusPostsolveError => Self::PostsolveError,
            kHighsModelStatusModelEmpty => Self::ModelEmpty,
            kHighsModelStatusOptimal => Self::Optimal,
            kHighsModelStatusInfeasible => Self::Infeasible,
            kHighsModelStatusUnboundedOrInfeasible => Self::UnboundedOrInfeasible,
            kHighsModelStatusUnbounded => Self::Unbounded,
            kHighsModelStatusObjectiveBound => Self::ObjectiveBound,
            kHighsModelStatusObjectiveTarget => Self::ObjectiveTarget,
            kHighsModelStatusTimeLimit => Self::TimeLimit,
            kHighsModelStatusIterationLimit => Self::IterationLimit,
            kHighsModelStatusUnknown => Self::Unknown,
            kHighsModelStatusSolutionLimit => Self::SolutionLimit,
            kHighsModelStatusInterrupt => Self::Interrupt,
            kHighsModelStatusMemoryLimit => Self::MemoryLimit,
            kHighsModelStatusHighsInterrupt => Self::HighsInterrupt,
            other => Self::Other(other),
        }
    }
}

/// HiGHS's own wording of the status, lower-cased.
///
/// ```
/// use culmen_highs::ModelStatus;
/// assert_eq!(ModelStatus::TimeLimit.to_string(), "time limit reached");
/// assert_eq!(ModelStatus::Other(42).to_string(), "model status 42");
/// ```
impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::NotSet => "not set",
            Self::LoadError => "load error",
            Self::ModelError => "model error",
            Self::PresolveError => "presolve error",
            Self::SolveError => "solve error",
            Self::PostsolveError => "postsolve error",
            Self::ModelEmpty => "empty",
            Self::Optimal => "optimal",
            Self::Infeasible => "infeasible",
            Self::UnboundedOrInfeasible => "primal infeasible or unbounded",
            Self::Unbounded => "unbounded",
            Self::ObjectiveBound => "bound on objective reached",
            Self::ObjectiveTarget => "target for objective reached",
            Self::TimeLimit => "time limit reached",
            Self::IterationLimit => "iteration limit reached",
            Self::Unknown => "unknown",
            Self::SolutionLimit => "solution limit reached",
            Self::Interrupt => "interrupted by user",
            Self::MemoryLimit => "memory limit reached",
            Self::HighsInterrupt => "interrupted by HiGHS",
            Self::Other(raw) => return write!(f, "model status {raw}"),
        };
        f.write_str(text)
    }
}

/// Whether a primal or dual solution exists: HiGHS's `kHighsSolutionStatus*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum SolutionStatus {
    /// There is none.
    None,
    /// There is one, and it is infeasible.
    Infeasible,
    /// There is one, and it is feasible.
    Feasible,
}

impl SolutionStatus {
    /// The status of HiGHS's number for it; an unknown number reads as none.
    #[expect(
        non_upper_case_globals,
        reason = "the patterns are HiGHS's constants under their C names"
    )]
    pub(crate) const fn from_raw(raw: i32) -> Self {
        match raw {
            kHighsSolutionStatusInfeasible => Self::Infeasible,
            kHighsSolutionStatusFeasible => Self::Feasible,
            _ => Self::None,
        }
    }
}

/// What a HiGHS call returned: HiGHS's `kHighsStatus*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum CallStatus {
    /// The call succeeded.
    Ok,
    /// The call succeeded with a warning.
    Warning,
    /// The call failed.
    Error,
}

impl CallStatus {
    /// The status of HiGHS's number for it; an unknown number reads as an error.
    #[expect(
        non_upper_case_globals,
        reason = "the patterns are HiGHS's constants under their C names"
    )]
    pub(crate) const fn from_raw(raw: i32) -> Self {
        match raw {
            kHighsStatusOk => Self::Ok,
            kHighsStatusWarning => Self::Warning,
            _ => Self::Error,
        }
    }
}

/// HiGHS's own account of a run, beside the generic [`Outcome`](culmen_core::Outcome).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HighsStatus {
    /// What `Highs_run` returned.
    pub run: CallStatus,
    /// The model status.
    pub model: ModelStatus,
    /// Whether a primal solution exists.
    pub primal: SolutionStatus,
    /// Whether a dual solution exists.
    pub dual: SolutionStatus,
    /// The MIP's relative gap as HiGHS computed it; `None` for an LP or when HiGHS reports none.
    pub mip_gap: Option<f64>,
    /// The number of simplex iterations, a MIP's over all its nodes; [`Solved::iterations`](culmen_core::Solved) sums
    /// every solver's.
    pub simplex_iterations: u64,
    /// The number of interior-point iterations.
    pub ipm_iterations: u64,
    /// The number of iterations of the QP solver.
    pub qp_iterations: u64,
    /// The relative difference of the primal and dual objective values, `|P - D| / (1 + |P| + |D|)`; `None` when
    /// HiGHS computed none, as for a MIP.
    ///
    /// A QP that HiGHS reports optimal with an error above the options'
    /// [`qp_objective_error`](crate::HighsOptions::qp_objective_error) fails with
    /// [`HighsError::UnsoundQp`](crate::HighsError::UnsoundQp).
    pub primal_dual_objective_error: Option<f64>,
    /// The last error line of HiGHS's log, if it wrote one.
    pub last_error: Option<String>,
}
