// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a solve ends with, in terms every backend maps its statuses to.

use core::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::TiVec;

use crate::Col;

/// A feasible point and its objective value, offset included.
///
/// Values are read with the handles the model or plan returned: `solution.values[x]`, or a whole family at once
/// with [`Keyed::values`](crate::Keyed::values).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Solution {
    /// The objective value, offset included.
    pub objective: f64,
    /// The value of every column, indexed by [`Col`].
    pub values: TiVec<Col, f64>,
}

/// How a solve ended. Each variant holds exactly what that ending proves.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Outcome {
    /// The solution is optimal within the solver's default tolerances.
    Optimal(Solution),
    /// The solve stopped because the requested relative or absolute gap was reached.
    GapReached {
        /// The best solution found.
        solution: Solution,
        /// The best objective bound proved, offset included.
        bound: f64,
    },
    /// The solve stopped before proving optimality.
    Stopped {
        /// Why it stopped.
        reason: StopReason,
        /// The best feasible solution found, `None` when none was.
        best: Option<Solution>,
        /// The best objective bound proved, offset included; `None` when none was.
        bound: Option<f64>,
    },
    /// No point satisfies the constraints.
    Infeasible,
    /// The objective improves without limit over feasible points.
    Unbounded,
    /// The problem is infeasible or unbounded, and the solver did not tell which.
    InfeasibleOrUnbounded,
}

impl Outcome {
    /// The best solution found, `None` when there is none.
    #[must_use]
    pub const fn solution(&self) -> Option<&Solution> {
        match self {
            Self::Optimal(solution) | Self::GapReached { solution, .. } => Some(solution),
            Self::Stopped { best, .. } => best.as_ref(),
            Self::Infeasible | Self::Unbounded | Self::InfeasibleOrUnbounded => None,
        }
    }

    /// The best solution found, by value.
    #[must_use]
    pub fn into_solution(self) -> Option<Solution> {
        match self {
            Self::Optimal(solution) | Self::GapReached { solution, .. } => Some(solution),
            Self::Stopped { best, .. } => best,
            Self::Infeasible | Self::Unbounded | Self::InfeasibleOrUnbounded => None,
        }
    }
}

/// Why a solve stopped before proving optimality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum StopReason {
    /// The time limit ran out.
    TimeLimit,
    /// A callback interrupted the solve, as the [`Stop`](crate::Stop) of a [`Request`](crate::Request) does.
    Interrupted,
    /// The node limit was reached.
    NodeLimit,
    /// The solution limit was reached.
    SolutionLimit,
    /// An iteration limit of the LP solver was reached.
    IterationLimit,
    /// The objective reached a bound or target set in the backend's options.
    ObjectiveLimit,
    /// The solver ran out of memory.
    MemoryLimit,
    /// The solver stopped for a reason it did not name; the backend's status says more.
    Other,
}

/// A finished solve: its outcome, the backend's own status for diagnostics, and the time it took.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Solved<S> {
    /// How the solve ended.
    pub outcome: Outcome,
    /// The backend's own status, as the solver reported it.
    pub status: S,
    /// The wall-clock time of the solve.
    pub elapsed: Duration,
}
