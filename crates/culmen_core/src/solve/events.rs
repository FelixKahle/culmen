// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kinds of callback events, a set of them, and the progress every event carries.

use core::fmt;
use core::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A kind of observational callback event: what a [`Callback`](crate::Callback) method is called for.
///
/// The events that change the search (lazy constraints, user cuts, heuristic solutions, pricing) are not events of
/// this kind: each is a handler registered on the [`Run`](crate::Run) of a solve, for a backend with its capability.
///
/// ```
/// use culmen_core::Event;
/// assert_eq!(Event::Incumbent.to_string(), "incumbent");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Event {
    /// A chunk of the solver's log.
    Log,
    /// A progress report the solver issues at its own pace, such as a line of its MIP log.
    Progress,
    /// A frequent poll at which a callback may interrupt the solve.
    Poll,
    /// Any feasible solution the solver found.
    Feasible,
    /// A new incumbent: a feasible solution better than every one before.
    Incumbent,
    /// An event only the backend knows, delivered with the backend's own context type.
    Native,
}

impl Event {
    /// Every event, in the order of their bits in [`Events`].
    const ALL: [Self; 6] = [
        Self::Log,
        Self::Progress,
        Self::Poll,
        Self::Feasible,
        Self::Incumbent,
        Self::Native,
    ];

    /// The event's bit in an [`Events`].
    const fn bit(self) -> u8 {
        1 << self as u8
    }

    /// The event in one word.
    const fn name(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Progress => "progress",
            Self::Poll => "poll",
            Self::Feasible => "feasible",
            Self::Incumbent => "incumbent",
            Self::Native => "native",
        }
    }
}

/// The event in one lower-case word, such as `incumbent`.
impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of [`Event`]s: what a [`Backend`](crate::Backend) delivers ([`Backend::EVENTS`]) and what a
/// [`Callback`](crate::Callback) asks for ([`Callback::events`](crate::Callback::events)).
///
/// A backend delivers an event only when it is in both sets, so a callback that asks for fewer events costs the
/// solver less. No event that changes the search is in this set, so neither [`Events::ALL`] nor a complement can turn
/// one on.
///
/// [`Backend::EVENTS`]: crate::Backend::EVENTS
///
/// ```
/// use culmen_core::{Event, Events};
///
/// const WATCH: Events = Events::of(&[Event::Log, Event::Poll]);
/// let wanted = Event::Log | Event::Poll;
/// assert_eq!(wanted, WATCH);
/// assert!(wanted.contains(Event::Poll));
/// assert!(!wanted.contains(Event::Native));
/// assert_eq!(wanted - Event::Log, Event::Poll.into());
/// assert_eq!(!wanted, Events::ALL - wanted);
/// assert_eq!(wanted.iter().collect::<Events>(), wanted);
/// assert_eq!(format!("{wanted:?}"), "Events(Log | Poll)");
/// assert_eq!(wanted.to_string(), "{log, poll}");
/// assert_eq!(Events::ALL.len(), 6);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(from = "Vec<Event>", into = "Vec<Event>")
)]
pub struct Events(u8);

/// The events of an [`Events`] set, in the order of [`Event`]'s variants; made by [`Events::iter`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct EventsIter(Events);

crate::set::element_set!(Events, EventsIter, Event, u8, "events", "{:?}");

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
