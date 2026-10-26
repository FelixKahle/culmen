// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HiGHS's statuses, all of them, and what a run reports beside the outcome.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// HiGHS's model status at the end of a run: every value of `HighsModelStatus` in HiGHS 1.15.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    /// A value HiGHS 1.15 does not define.
    Other(i32),
}

impl ModelStatus {
    /// The status of HiGHS's number for it.
    pub(crate) const fn from_raw(raw: i32) -> Self {
        match raw {
            0 => Self::NotSet,
            1 => Self::LoadError,
            2 => Self::ModelError,
            3 => Self::PresolveError,
            4 => Self::SolveError,
            5 => Self::PostsolveError,
            6 => Self::ModelEmpty,
            7 => Self::Optimal,
            8 => Self::Infeasible,
            9 => Self::UnboundedOrInfeasible,
            10 => Self::Unbounded,
            11 => Self::ObjectiveBound,
            12 => Self::ObjectiveTarget,
            13 => Self::TimeLimit,
            14 => Self::IterationLimit,
            15 => Self::Unknown,
            16 => Self::SolutionLimit,
            17 => Self::Interrupt,
            18 => Self::MemoryLimit,
            19 => Self::HighsInterrupt,
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
    pub(crate) const fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Infeasible,
            2 => Self::Feasible,
            _ => Self::None,
        }
    }
}

/// What a HiGHS call returned: HiGHS's `kHighsStatus*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    pub(crate) const fn from_raw(raw: i32) -> Self {
        match raw {
            0 => Self::Ok,
            1 => Self::Warning,
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
    /// The number of branch-and-bound nodes.
    pub mip_nodes: u64,
    /// The number of simplex iterations.
    pub simplex_iterations: u64,
    /// The number of interior-point iterations.
    pub ipm_iterations: u64,
    /// The last error line of HiGHS's log, if it wrote one.
    pub last_error: Option<String>,
}
