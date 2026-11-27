// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SCIP's solving status, all of it, what a solve reports beside the outcome, and the outcome each status proves.

use core::fmt;

use culmen_core::{Outcome, Solution, StopReason};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ScipError;

/// SCIP's solving status at the end of a solve: every value of `SCIP_STATUS` in SCIP 10.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Status {
    /// The status is not known (`SCIP_STATUS_UNKNOWN`).
    Unknown,
    /// Solved to optimality within SCIP's gap limits (`SCIP_STATUS_OPTIMAL`).
    Optimal,
    /// Proved infeasible (`SCIP_STATUS_INFEASIBLE`).
    Infeasible,
    /// Proved unbounded (`SCIP_STATUS_UNBOUNDED`).
    Unbounded,
    /// Proved infeasible or unbounded (`SCIP_STATUS_INFORUNBD`).
    InfeasibleOrUnbounded,
    /// Interrupted by the user, through `SCIPinterruptSolve` or `SIGINT` (`SCIP_STATUS_USERINTERRUPT`).
    UserInterrupt,
    /// The process received `SIGTERM` (`SCIP_STATUS_TERMINATE`).
    Terminate,
    /// The node limit was reached (`SCIP_STATUS_NODELIMIT`).
    NodeLimit,
    /// The node limit across restarts was reached (`SCIP_STATUS_TOTALNODELIMIT`).
    TotalNodeLimit,
    /// The limit on nodes without a better primal bound was reached (`SCIP_STATUS_STALLNODELIMIT`).
    StallNodeLimit,
    /// The time limit was reached (`SCIP_STATUS_TIMELIMIT`).
    TimeLimit,
    /// The memory limit was reached (`SCIP_STATUS_MEMLIMIT`).
    MemoryLimit,
    /// The relative or absolute gap limit was reached (`SCIP_STATUS_GAPLIMIT`).
    GapLimit,
    /// The primal limit was reached (`SCIP_STATUS_PRIMALLIMIT`).
    PrimalLimit,
    /// The dual limit was reached (`SCIP_STATUS_DUALLIMIT`).
    DualLimit,
    /// The limit on solutions found was reached (`SCIP_STATUS_SOLLIMIT`).
    SolutionLimit,
    /// The limit on improving solutions was reached (`SCIP_STATUS_BESTSOLLIMIT`).
    BestSolutionLimit,
    /// The restart limit was reached (`SCIP_STATUS_RESTARTLIMIT`).
    RestartLimit,
    /// A value SCIP 10 does not define.
    Other(u32),
}

impl Status {
    /// The status of SCIP's number for it.
    pub(crate) const fn from_raw(raw: scip_sys::SCIP_STATUS) -> Self {
        match raw {
            0 => Self::Unknown,
            1 => Self::Optimal,
            2 => Self::Infeasible,
            3 => Self::Unbounded,
            4 => Self::InfeasibleOrUnbounded,
            10 => Self::UserInterrupt,
            11 => Self::Terminate,
            20 => Self::NodeLimit,
            21 => Self::TotalNodeLimit,
            22 => Self::StallNodeLimit,
            23 => Self::TimeLimit,
            24 => Self::MemoryLimit,
            25 => Self::GapLimit,
            26 => Self::PrimalLimit,
            27 => Self::DualLimit,
            28 => Self::SolutionLimit,
            29 => Self::BestSolutionLimit,
            30 => Self::RestartLimit,
            other => Self::Other(other),
        }
    }

    /// The reason a solve that ended with this status stopped early, `None` for a status that is no early stop.
    const fn stop_reason(self) -> Option<StopReason> {
        Some(match self {
            Self::UserInterrupt => StopReason::Interrupted,
            Self::NodeLimit | Self::TotalNodeLimit | Self::StallNodeLimit => StopReason::NodeLimit,
            Self::TimeLimit => StopReason::TimeLimit,
            Self::MemoryLimit => StopReason::MemoryLimit,
            Self::SolutionLimit | Self::BestSolutionLimit => StopReason::SolutionLimit,
            Self::PrimalLimit | Self::DualLimit => StopReason::ObjectiveLimit,
            Self::Terminate | Self::RestartLimit => StopReason::Other,
            Self::Unknown
            | Self::Optimal
            | Self::Infeasible
            | Self::Unbounded
            | Self::InfeasibleOrUnbounded
            | Self::GapLimit
            | Self::Other(_) => return None,
        })
    }
}

/// SCIP's own wording of the status, `SCIPstatusName`.
///
/// ```
/// use culmen_scip::Status;
/// assert_eq!(Status::GapLimit.to_string(), "gap limit reached");
/// assert_eq!(Status::Other(42).to_string(), "status 42");
/// ```
impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Unknown => "unknown",
            Self::Optimal => "optimal solution found",
            Self::Infeasible => "infeasible",
            Self::Unbounded => "unbounded",
            Self::InfeasibleOrUnbounded => "infeasible or unbounded",
            Self::UserInterrupt => "user interrupt",
            Self::Terminate => "termination signal received",
            Self::NodeLimit => "node limit reached",
            Self::TotalNodeLimit => "total node limit reached",
            Self::StallNodeLimit => "stall node limit reached",
            Self::TimeLimit => "time limit reached",
            Self::MemoryLimit => "memory limit reached",
            Self::GapLimit => "gap limit reached",
            Self::PrimalLimit => "primal limit reached",
            Self::DualLimit => "dual limit reached",
            Self::SolutionLimit => "solution limit reached",
            Self::BestSolutionLimit => "solution improvement limit reached",
            Self::RestartLimit => "restart limit reached",
            Self::Other(raw) => return write!(f, "status {raw}"),
        };
        f.write_str(text)
    }
}

/// SCIP's own account of a solve, beside the generic [`Outcome`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScipStatus {
    /// SCIP's solving status.
    pub status: Status,
    /// The number of feasible solutions SCIP found, its heuristics' and the starts it accepted included.
    pub solutions_found: u64,
}

/// The generic outcome of a solve SCIP ended with `status`, its best solution `best` and its best bound `bound`.
///
/// # Errors
///
/// [`ScipError::Failed`] for [`Status::Unknown`] and an undefined status, for [`Status::Optimal`] without a
/// solution, and for [`Status::GapLimit`] without a solution or a bound.
pub(crate) fn outcome(status: Status, best: Option<Solution>, bound: Option<f64>) -> Result<Outcome, ScipError> {
    let failed = || ScipError::Failed { status };
    match status {
        Status::Optimal => best.map(Outcome::Optimal).ok_or_else(failed),
        Status::GapLimit => match (best, bound) {
            (Some(solution), Some(bound)) => Ok(Outcome::GapReached { solution, bound }),
            _ => Err(failed()),
        },
        Status::Infeasible => Ok(Outcome::Infeasible),
        Status::Unbounded => Ok(Outcome::Unbounded),
        Status::InfeasibleOrUnbounded => Ok(Outcome::InfeasibleOrUnbounded),
        other => match other.stop_reason() {
            Some(reason) => Ok(Outcome::Stopped { reason, best, bound }),
            None => Err(failed()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every status of SCIP 10, with its number.
    const ALL: [(u32, Status); 18] = [
        (scip_sys::SCIP_Status_SCIP_STATUS_UNKNOWN, Status::Unknown),
        (scip_sys::SCIP_Status_SCIP_STATUS_OPTIMAL, Status::Optimal),
        (scip_sys::SCIP_Status_SCIP_STATUS_INFEASIBLE, Status::Infeasible),
        (scip_sys::SCIP_Status_SCIP_STATUS_UNBOUNDED, Status::Unbounded),
        (
            scip_sys::SCIP_Status_SCIP_STATUS_INFORUNBD,
            Status::InfeasibleOrUnbounded,
        ),
        (scip_sys::SCIP_Status_SCIP_STATUS_USERINTERRUPT, Status::UserInterrupt),
        (scip_sys::SCIP_Status_SCIP_STATUS_TERMINATE, Status::Terminate),
        (scip_sys::SCIP_Status_SCIP_STATUS_NODELIMIT, Status::NodeLimit),
        (scip_sys::SCIP_Status_SCIP_STATUS_TOTALNODELIMIT, Status::TotalNodeLimit),
        (scip_sys::SCIP_Status_SCIP_STATUS_STALLNODELIMIT, Status::StallNodeLimit),
        (scip_sys::SCIP_Status_SCIP_STATUS_TIMELIMIT, Status::TimeLimit),
        (scip_sys::SCIP_Status_SCIP_STATUS_MEMLIMIT, Status::MemoryLimit),
        (scip_sys::SCIP_Status_SCIP_STATUS_GAPLIMIT, Status::GapLimit),
        (scip_sys::SCIP_Status_SCIP_STATUS_PRIMALLIMIT, Status::PrimalLimit),
        (scip_sys::SCIP_Status_SCIP_STATUS_DUALLIMIT, Status::DualLimit),
        (scip_sys::SCIP_Status_SCIP_STATUS_SOLLIMIT, Status::SolutionLimit),
        (
            scip_sys::SCIP_Status_SCIP_STATUS_BESTSOLLIMIT,
            Status::BestSolutionLimit,
        ),
        (scip_sys::SCIP_Status_SCIP_STATUS_RESTARTLIMIT, Status::RestartLimit),
    ];

    fn point() -> Solution {
        Solution {
            objective: 3.0,
            values: vec![1.0, 2.0].into(),
        }
    }

    #[test]
    fn test_every_status_of_scip_10_reads_as_itself() {
        for (raw, expected) in ALL {
            assert_eq!(Status::from_raw(raw), expected, "{raw}");
        }
        assert_eq!(Status::from_raw(5), Status::Other(5));
        assert_eq!(Status::from_raw(31), Status::Other(31));
    }

    #[test]
    fn test_every_status_has_the_outcome_it_proves_with_and_without_a_solution() {
        use StopReason as R;
        let stopped = |reason| {
            [
                Ok(Outcome::Stopped {
                    reason,
                    best: Some(point()),
                    bound: Some(2.5),
                }),
                Ok(Outcome::Stopped {
                    reason,
                    best: None,
                    bound: Some(2.5),
                }),
            ]
        };
        let failed = |status| [Err(ScipError::Failed { status }), Err(ScipError::Failed { status })];
        for (_, code) in ALL {
            let expected = match code {
                Status::Unknown => failed(code),
                Status::Optimal => [Ok(Outcome::Optimal(point())), Err(ScipError::Failed { status: code })],
                Status::GapLimit => [
                    Ok(Outcome::GapReached {
                        solution: point(),
                        bound: 2.5,
                    }),
                    Err(ScipError::Failed { status: code }),
                ],
                Status::Infeasible => [Ok(Outcome::Infeasible), Ok(Outcome::Infeasible)],
                Status::Unbounded => [Ok(Outcome::Unbounded), Ok(Outcome::Unbounded)],
                Status::InfeasibleOrUnbounded => {
                    [Ok(Outcome::InfeasibleOrUnbounded), Ok(Outcome::InfeasibleOrUnbounded)]
                }
                Status::UserInterrupt => stopped(R::Interrupted),
                Status::NodeLimit | Status::TotalNodeLimit | Status::StallNodeLimit => stopped(R::NodeLimit),
                Status::TimeLimit => stopped(R::TimeLimit),
                Status::MemoryLimit => stopped(R::MemoryLimit),
                Status::SolutionLimit | Status::BestSolutionLimit => stopped(R::SolutionLimit),
                Status::PrimalLimit | Status::DualLimit => stopped(R::ObjectiveLimit),
                Status::Terminate | Status::RestartLimit => stopped(R::Other),
                Status::Other(_) => unreachable!("every status of SCIP 10 is named"),
            };
            let got = [outcome(code, Some(point()), Some(2.5)), outcome(code, None, Some(2.5))];
            assert_eq!(got, expected, "{code:?}");
        }
        assert_eq!(
            outcome(Status::Other(5), Some(point()), None),
            Err(ScipError::Failed {
                status: Status::Other(5)
            })
        );
    }

    #[test]
    fn test_a_reached_gap_without_a_bound_proves_nothing_and_a_stop_keeps_a_missing_bound() {
        assert_eq!(
            outcome(Status::GapLimit, Some(point()), None),
            Err(ScipError::Failed {
                status: Status::GapLimit
            })
        );
        assert_eq!(
            outcome(Status::TimeLimit, None, None),
            Ok(Outcome::Stopped {
                reason: StopReason::TimeLimit,
                best: None,
                bound: None
            })
        );
    }
}
