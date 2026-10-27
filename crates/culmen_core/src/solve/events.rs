// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kinds of callback events, a set of them, and the progress every event carries.

use core::fmt;
use core::iter::FusedIterator;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};
use core::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A set of callback event kinds: what a [`Backend`](crate::Backend) supports ([`Backend::EVENTS`]) and what a
/// [`Callback`](crate::Callback) asks for ([`Callback::events`](crate::Callback::events)).
///
/// A backend delivers an event only when it is in both sets, so a callback that asks for fewer events costs the
/// solver less.
///
/// [`Backend::EVENTS`]: crate::Backend::EVENTS
///
/// ```
/// use culmen_core::Events;
/// let wanted = Events::LOG | Events::POLL;
/// assert!(wanted.contains(Events::POLL));
/// assert!(!wanted.contains(Events::CUT));
/// assert_eq!(wanted & Events::POLL, Events::POLL);
/// assert_eq!(wanted.iter().collect::<Events>(), wanted);
/// assert_eq!(format!("{wanted:?}"), "Events(LOG | POLL)");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Events(u16);

impl Events {
    /// No event.
    pub const NONE: Self = Self(0);
    /// A chunk of the solver's log.
    pub const LOG: Self = Self(1);
    /// A progress report the solver issues at its own pace, such as a line of its MIP log.
    pub const PROGRESS: Self = Self(1 << 1);
    /// A frequent poll at which a callback may interrupt the solve.
    pub const POLL: Self = Self(1 << 2);
    /// Any feasible solution the solver found.
    pub const FEASIBLE: Self = Self(1 << 3);
    /// A new incumbent: a feasible solution better than every one before.
    pub const INCUMBENT: Self = Self(1 << 4);
    /// A candidate solution a callback may cut off with lazy constraints.
    pub const LAZY: Self = Self(1 << 5);
    /// A node relaxation a callback may cut off with user cuts.
    pub const CUT: Self = Self(1 << 6);
    /// A point at which a callback may hand the solver a heuristic solution.
    pub const HEURISTIC: Self = Self(1 << 7);
    /// An event only the backend knows, delivered with the backend's own context type.
    pub const NATIVE: Self = Self(1 << 8);
    /// Every event.
    pub const ALL: Self = Self((1 << 9) - 1);

    const NAMES: [(Self, &'static str); 9] = [
        (Self::LOG, "LOG"),
        (Self::PROGRESS, "PROGRESS"),
        (Self::POLL, "POLL"),
        (Self::FEASIBLE, "FEASIBLE"),
        (Self::INCUMBENT, "INCUMBENT"),
        (Self::LAZY, "LAZY"),
        (Self::CUT, "CUT"),
        (Self::HEURISTIC, "HEURISTIC"),
        (Self::NATIVE, "NATIVE"),
    ];

    /// The events in either set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The events in both sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Whether every event of `other` is in this set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The number of events in the set.
    #[must_use]
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Each event of the set on its own, in the order of the constants above.
    pub const fn iter(self) -> EventsIter {
        EventsIter(self)
    }

    /// The name of a single event, as its constant is named; `None` for any other set.
    const fn name(self) -> Option<&'static str> {
        let mut i = 0;
        while i < Self::NAMES.len() {
            if Self::NAMES[i].0.0 == self.0 {
                return Some(Self::NAMES[i].1);
            }
            i += 1;
        }
        None
    }
}

impl BitOr for Events {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for Events {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

impl BitAnd for Events {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        self.intersection(other)
    }
}

impl BitAndAssign for Events {
    fn bitand_assign(&mut self, other: Self) {
        *self = self.intersection(other);
    }
}

impl FromIterator<Self> for Events {
    fn from_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().fold(Self::NONE, Self::union)
    }
}

impl Extend<Self> for Events {
    fn extend<I: IntoIterator<Item = Self>>(&mut self, iter: I) {
        *self = iter.into_iter().fold(*self, Self::union);
    }
}

impl IntoIterator for Events {
    type Item = Self;
    type IntoIter = EventsIter;

    fn into_iter(self) -> EventsIter {
        self.iter()
    }
}

impl fmt::Debug for Events {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Events(")?;
        for (i, event) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(" | ")?;
            }
            f.write_str(event.name().unwrap_or("?"))?;
        }
        f.write_str(")")
    }
}

/// The single events of an [`Events`] set, lowest first; made by [`Events::iter`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct EventsIter(Events);

impl Iterator for EventsIter {
    type Item = Events;

    fn next(&mut self) -> Option<Events> {
        let rest = (self.0).0;
        if rest == 0 {
            return None;
        }
        let lowest = rest.isolate_lowest_one();
        (self.0).0 = rest & !lowest;
        Some(Events(lowest))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }
}

impl DoubleEndedIterator for EventsIter {
    fn next_back(&mut self) -> Option<Events> {
        let rest = (self.0).0;
        if rest == 0 {
            return None;
        }
        let highest = rest.isolate_highest_one();
        (self.0).0 = rest & !highest;
        Some(Events(highest))
    }
}

impl ExactSizeIterator for EventsIter {}

impl FusedIterator for EventsIter {}

/// The phase of a solve an event comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Phase {
    /// Presolve.
    Presolve,
    /// A simplex method on an LP.
    Simplex,
    /// An interior-point method on an LP.
    Barrier,
    /// Branch and bound on a MIP.
    Mip,
    /// A phase the backend does not name in these terms.
    Other,
}

/// Where a solve stands: its time, phase, best objective, best bound and work done.
///
/// Every value is in the model's objective space, offset included; a value the solver does not report in this
/// phase is `None`. The gap is computed by [`Progress::gap`], not stored.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Progress {
    /// The wall-clock time since the solve began.
    pub elapsed: Duration,
    /// The phase reporting.
    pub phase: Phase,
    /// The objective value of the best feasible solution; `None` before one is found.
    pub objective: Option<f64>,
    /// The best objective bound proved; `None` before one is.
    pub bound: Option<f64>,
    /// The number of branch-and-bound nodes explored; `None` outside branch and bound.
    pub nodes: Option<u64>,
    /// The number of LP iterations done; `None` when the solver does not report them here.
    pub iterations: Option<u64>,
}

impl Progress {
    /// A report of the elapsed time and phase alone.
    #[must_use]
    pub const fn at(elapsed: Duration, phase: Phase) -> Self {
        Self {
            elapsed,
            phase,
            objective: None,
            bound: None,
            nodes: None,
            iterations: None,
        }
    }

    /// The relative gap `|objective - bound| / |objective|`, `None` without both; zero when both are zero, and
    /// infinite when only the objective is.
    ///
    /// ```
    /// use core::time::Duration;
    /// use culmen_core::{Phase, Progress};
    /// let p = Progress { objective: Some(100.0), bound: Some(90.0), ..Progress::at(Duration::ZERO, Phase::Mip) };
    /// assert_eq!(p.gap(), Some(0.1));
    /// ```
    #[must_use]
    pub fn gap(&self) -> Option<f64> {
        let (objective, bound) = (self.objective?, self.bound?);
        let distance = (objective - bound).abs();
        Some(if distance == 0.0 {
            0.0
        } else {
            distance / objective.abs()
        })
    }
}

/// How much a log chunk matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LogLevel {
    /// Detail beyond the solver's normal log.
    Detail,
    /// The solver's normal log.
    Info,
    /// Something may be wrong.
    Warning,
    /// Something is wrong.
    Error,
}
