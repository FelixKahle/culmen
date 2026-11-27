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
    kHighsModelStatusUnknown, kHighsSolutionStatusFeasible, kHighsSolutionStatusInfeasible, kHighsSolutionStatusNone,
    kHighsStatusError, kHighsStatusOk, kHighsStatusWarning,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{HighsCode, HighsError};

/// HiGHS's model status at the end of a run: every `kHighsModelStatus*` value of `highs_c_api.h` in HiGHS 1.16.
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
}

impl ModelStatus {
    /// The status of HiGHS's number for it.
    ///
    /// # Errors
    ///
    /// [`HighsError::Unlisted`] for a number no `kHighsModelStatus*` constant has.
    #[expect(
        non_upper_case_globals,
        reason = "the patterns are HiGHS's constants under their C names"
    )]
    pub(crate) const fn from_raw(raw: i32, call: &'static str) -> Result<Self, HighsError> {
        Ok(match raw {
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
            _ => return Err(unlisted(HighsCode::ModelStatus, raw, call)),
        })
    }
}

/// HiGHS's own wording of the status, lower-cased.
///
/// ```
/// use culmen_highs::ModelStatus;
/// assert_eq!(ModelStatus::TimeLimit.to_string(), "time limit reached");
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
        };
        f.write_str(text)
    }
}

/// Whether a primal or dual solution exists: every `kHighsSolutionStatus*` value of `highs_c_api.h` in HiGHS 1.16.
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
    /// The status of HiGHS's number for it.
    ///
    /// # Errors
    ///
    /// [`HighsError::Unlisted`] for a number no `kHighsSolutionStatus*` constant has.
    #[expect(
        non_upper_case_globals,
        reason = "the patterns are HiGHS's constants under their C names"
    )]
    pub(crate) const fn from_raw(raw: i32, call: &'static str) -> Result<Self, HighsError> {
        Ok(match raw {
            kHighsSolutionStatusNone => Self::None,
            kHighsSolutionStatusInfeasible => Self::Infeasible,
            kHighsSolutionStatusFeasible => Self::Feasible,
            _ => return Err(unlisted(HighsCode::SolutionStatus, raw, call)),
        })
    }
}

/// What a HiGHS call returned: every `kHighsStatus*` value of `highs_c_api.h` in HiGHS 1.16.
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
    /// The status of HiGHS's number for it.
    ///
    /// # Errors
    ///
    /// [`HighsError::Unlisted`] for a number no `kHighsStatus*` constant has.
    #[expect(
        non_upper_case_globals,
        reason = "the patterns are HiGHS's constants under their C names"
    )]
    pub(crate) const fn from_raw(raw: i32, call: &'static str) -> Result<Self, HighsError> {
        Ok(match raw {
            kHighsStatusError => Self::Error,
            kHighsStatusOk => Self::Ok,
            kHighsStatusWarning => Self::Warning,
            _ => return Err(unlisted(HighsCode::CallStatus, raw, call)),
        })
    }

    /// Whether a call of the C API that returned `raw` succeeded, with or without a warning.
    ///
    /// # Errors
    ///
    /// [`HighsError::Unlisted`] for a number no `kHighsStatus*` constant has.
    pub(crate) fn succeeded(raw: i32, call: &'static str) -> Result<bool, HighsError> {
        match Self::from_raw(raw, call) {
            Ok(Self::Error) => Ok(false),
            Ok(Self::Ok | Self::Warning) => Ok(true),
            Err(error) => Err(error),
        }
    }
}

/// The error for `raw`, a value of `code` that HiGHS 1.16 does not define, which `call` gave.
pub(crate) const fn unlisted(code: HighsCode, raw: i32, call: &'static str) -> HighsError {
    HighsError::Unlisted { code, raw, call }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_model_status_of_highs_has_its_own_variant_and_any_other_number_is_an_error() {
        let statuses: Vec<ModelStatus> = (0..=19)
            .map(|raw| ModelStatus::from_raw(raw, "test").unwrap())
            .collect();
        let distinct: std::collections::HashSet<ModelStatus> = statuses.iter().copied().collect();
        assert_eq!(distinct.len(), 20, "{statuses:?}");
        assert_eq!(
            ModelStatus::from_raw(kHighsModelStatusOptimal, "test"),
            Ok(ModelStatus::Optimal)
        );
        for raw in [-1, 20, i32::MIN] {
            assert_eq!(
                ModelStatus::from_raw(raw, "test"),
                Err(HighsError::Unlisted {
                    code: HighsCode::ModelStatus,
                    raw,
                    call: "test"
                })
            );
        }
    }

    #[test]
    fn test_every_solution_status_of_highs_has_its_own_variant_and_any_other_number_is_an_error() {
        assert_eq!(
            SolutionStatus::from_raw(kHighsSolutionStatusNone, "test"),
            Ok(SolutionStatus::None)
        );
        assert_eq!(
            SolutionStatus::from_raw(kHighsSolutionStatusInfeasible, "test"),
            Ok(SolutionStatus::Infeasible)
        );
        assert_eq!(
            SolutionStatus::from_raw(kHighsSolutionStatusFeasible, "test"),
            Ok(SolutionStatus::Feasible)
        );
        for raw in [-1, 3] {
            assert_eq!(
                SolutionStatus::from_raw(raw, "test"),
                Err(HighsError::Unlisted {
                    code: HighsCode::SolutionStatus,
                    raw,
                    call: "test"
                })
            );
        }
    }

    #[test]
    fn test_every_call_status_of_highs_has_its_own_variant_and_any_other_number_is_an_error() {
        assert_eq!(CallStatus::from_raw(kHighsStatusError, "test"), Ok(CallStatus::Error));
        assert_eq!(CallStatus::from_raw(kHighsStatusOk, "test"), Ok(CallStatus::Ok));
        assert_eq!(
            CallStatus::from_raw(kHighsStatusWarning, "test"),
            Ok(CallStatus::Warning)
        );
        assert_eq!(CallStatus::succeeded(kHighsStatusError, "test"), Ok(false));
        assert_eq!(CallStatus::succeeded(kHighsStatusOk, "test"), Ok(true));
        assert_eq!(CallStatus::succeeded(kHighsStatusWarning, "test"), Ok(true));
        for raw in [-2, 2] {
            let error = HighsError::Unlisted {
                code: HighsCode::CallStatus,
                raw,
                call: "test",
            };
            assert_eq!(CallStatus::from_raw(raw, "test"), Err(error.clone()));
            assert_eq!(CallStatus::succeeded(raw, "test"), Err(error));
        }
    }
}
