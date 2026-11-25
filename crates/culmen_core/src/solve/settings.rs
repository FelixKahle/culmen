// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The settings every backend understands, applied per solve.

use core::num::{NonZeroU64, NonZeroUsize};
use core::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{AbsoluteGap, RelativeGap};

/// What a solve may spend and when it may stop early, in terms every backend maps to its own parameters.
///
/// Plain data; `None` leaves a setting to the backend's own options, which in turn default to the solver's.
/// Settings given here take precedence over a backend's options for the one solve they are passed to.
///
/// ```
/// use core::time::Duration;
/// use culmen_core::{Settings, relative_gap, threads};
///
/// let settings = Settings {
///     time_limit: Some(Duration::from_secs(30)),
///     relative_gap: Some(relative_gap!(0.01)),
///     threads: Some(threads!(4)),
///     ..Settings::default()
/// };
/// assert_eq!(settings.relative_gap.unwrap().get(), 0.01);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Settings {
    /// The wall-clock time the solve may take.
    pub time_limit: Option<Duration>,
    /// The relative gap between the best solution and the best bound at which a MIP counts as solved.
    pub relative_gap: Option<RelativeGap>,
    /// The absolute gap, in objective units, at which a MIP counts as solved.
    pub absolute_gap: Option<AbsoluteGap>,
    /// The number of threads the solver may use.
    pub threads: Option<NonZeroUsize>,
    /// The seed of the solver's random choices, for reproducible runs.
    pub seed: Option<u32>,
    /// Whether the solver presolves the problem first.
    pub presolve: Option<bool>,
    /// The number of branch-and-bound nodes after which a MIP stops.
    pub node_limit: Option<u64>,
    /// The number of improving solutions after which a MIP stops: solutions each better than every one the solve
    /// found before, the first included. At least one, since a MIP that stops before its first solution has nothing
    /// to stop for; a backend that cannot count improving solutions refuses the setting.
    pub solution_limit: Option<NonZeroU64>,
    /// Whether the solver also prints its log to standard output; the log reaches callbacks either way.
    pub console: bool,
}

impl Settings {
    /// Every setting left to the backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            time_limit: None,
            relative_gap: None,
            absolute_gap: None,
            threads: None,
            seed: None,
            presolve: None,
            node_limit: None,
            solution_limit: None,
            console: false,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::new()
    }
}

/// A literal solution limit, a [`NonZeroU64`] checked when the code is compiled.
///
/// ```
/// use culmen_core::{Settings, solutions};
///
/// let settings = Settings { solution_limit: Some(solutions!(3)), ..Settings::new() };
/// assert_eq!(settings.solution_limit.map(|n| n.get()), Some(3));
/// ```
#[macro_export]
macro_rules! solutions {
    ($value:expr $(,)?) => {
        const {
            match ::core::num::NonZeroU64::new($value) {
                ::core::option::Option::Some(n) => n,
                ::core::option::Option::None => panic!("solutions: zero"),
            }
        }
    };
}

/// A literal thread count, a [`NonZeroUsize`] checked when the code is compiled.
///
/// ```
/// let four = culmen_core::threads!(4);
/// assert_eq!(four.get(), 4);
/// ```
#[macro_export]
macro_rules! threads {
    ($value:expr $(,)?) => {
        const {
            match ::core::num::NonZeroUsize::new($value) {
                ::core::option::Option::Some(n) => n,
                ::core::option::Option::None => panic!("threads: zero"),
            }
        }
    };
}
