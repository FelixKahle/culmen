// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The convenient path: a stop, an observer of progress, solutions and log lines, and the callback running them.

use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;
use std::sync::Arc;

use typed_index_collections::TiSlice;

use super::context::{Context, LogContext, PollContext, SolutionContext};
use crate::{Backend, Callback, Col, Event, Events, LogLevel, Progress};

/// A cooperative stop request, polled while a solve runs.
///
/// Implemented for `AtomicBool` and shared references to any stop; any `Fn() -> bool` becomes one through
/// [`stop_fn`], which adapts a cancellation token in one line:
///
/// ```
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicBool, Ordering};
/// use culmen_core::{Stop, stop_fn};
///
/// let cancelled = Arc::new(AtomicBool::new(false));
/// assert!(!cancelled.should_stop());
/// cancelled.store(true, Ordering::Release);
/// assert!(cancelled.should_stop());
///
/// // A tokio `CancellationToken` adapts alike: `stop_fn(move || token.is_cancelled())`.
/// let polled = stop_fn(|| true);
/// assert!(polled.should_stop());
/// ```
pub trait Stop: Sync {
    /// Whether the solve should stop now.
    fn should_stop(&self) -> bool;
}

impl Stop for AtomicBool {
    fn should_stop(&self) -> bool {
        self.load(Ordering::Acquire)
    }
}

impl<S: Stop + ?Sized> Stop for &S {
    fn should_stop(&self) -> bool {
        (**self).should_stop()
    }
}

impl<S: Stop + ?Sized + Send> Stop for Arc<S> {
    fn should_stop(&self) -> bool {
        (**self).should_stop()
    }
}

/// A closure as a [`Stop`]; made by [`stop_fn`].
#[derive(Clone, Copy)]
pub struct StopFn<F>(F);

/// A closure as a [`Stop`].
#[must_use]
pub const fn stop_fn<F: Fn() -> bool + Sync>(f: F) -> StopFn<F> {
    StopFn(f)
}

impl<F: Fn() -> bool + Sync> Stop for StopFn<F> {
    fn should_stop(&self) -> bool {
        (self.0)()
    }
}

impl<F> fmt::Debug for StopFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StopFn(..)")
    }
}

/// A new incumbent: a feasible solution better than every one before.
#[derive(Clone, Copy, Debug)]
pub struct Improvement<'a> {
    /// Where the solve stood when the solution was found.
    pub progress: Progress,
    /// The solution's objective value, offset included.
    pub objective: f64,
    /// The solution's value of every column, read with the model's handles.
    pub values: &'a TiSlice<Col, f64>,
}

/// A complete line of the solver's log, without its line break.
#[derive(Clone, Copy, Debug)]
pub struct LogLine<'a> {
    /// The wall-clock time since the solve began.
    pub elapsed: Duration,
    /// How much the line matters.
    pub level: LogLevel,
    /// The line's text.
    pub text: &'a str,
}

/// What an application watches during a solve: progress at a steady pace, every improving solution, every log
/// line. Every method does nothing by default.
///
/// ```
/// use culmen_core::{Improvement, LogLine, Observer, Progress};
///
/// #[derive(Default)]
/// struct Feed(Vec<String>);
///
/// impl Observer for Feed {
///     fn progress(&mut self, p: &Progress) {
///         self.0.push(format!("{:?} best {:?} bound {:?} gap {:?}", p.elapsed, p.objective, p.bound, p.gap()));
///     }
///     fn improved(&mut self, i: &Improvement<'_>) {
///         self.0.push(format!("new best {}", i.objective));
///     }
///     fn log(&mut self, line: &LogLine<'_>) {
///         self.0.push(line.text.to_owned());
///     }
/// }
/// ```
pub trait Observer: Send {
    /// Where the solve stands, at most about once per [`Run::progress_interval`](crate::Run::progress_interval).
    fn progress(&mut self, progress: &Progress) {
        let _ = progress;
    }
    /// A new incumbent.
    fn improved(&mut self, improvement: &Improvement<'_>) {
        let _ = improvement;
    }
    /// A complete log line.
    fn log(&mut self, line: &LogLine<'_>) {
        let _ = line;
    }
}

impl Observer for () {}

impl<O: Observer + ?Sized> Observer for &mut O {
    fn progress(&mut self, progress: &Progress) {
        (**self).progress(progress);
    }
    fn improved(&mut self, improvement: &Improvement<'_>) {
        (**self).improved(improvement);
    }
    fn log(&mut self, line: &LogLine<'_>) {
        (**self).log(line);
    }
}

/// A stop and an observer as a [`Callback`] for every backend: what a [`Run`](crate::Run) runs before its callback.
///
/// It polls the stop at every [`Event::Poll`], reports progress from polls and progress events at most once per
/// interval, reports each incumbent, and joins log chunks into lines, each without its `\n` or `\r\n`, even when the
/// two halves of a `\r\n` arrive in different chunks. Code that drives a [`Backend`] directly registers it paired
/// with a callback of its own, `(watch, mine)`, to keep this convenience while reaching every event, and calls
/// [`Watch::finish`] once the solve returns.
///
/// ```
/// use core::time::Duration;
/// use std::sync::atomic::AtomicBool;
/// use culmen_core::{Backend, Hooks, Model, Outcome, Phase, Progress, Script, Settings, Step, StopReason, Var, Watch};
///
/// let mut m = Model::new();
/// m.add_var(Var::binary());
/// let mut script = Script::new([Step::Poll(Progress::at(Duration::ZERO, Phase::Mip))], Outcome::Infeasible);
/// script.load(&m.build()?)?;
///
/// let stop = AtomicBool::new(true);
/// let mut watch = Watch::new(Some(&stop), None, Watch::DEFAULT_PROGRESS_INTERVAL);
/// let solved = script.solve(&Settings::new(), Hooks::new().register_callback(&mut watch))?;
/// watch.finish();
/// assert!(matches!(solved.outcome, Outcome::Stopped { reason: StopReason::Interrupted, .. }));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Watch<'a> {
    stop: Option<&'a dyn Stop>,
    observer: Option<&'a mut dyn Observer>,
    interval: Duration,
    last_progress: Option<Duration>,
    line: String,
    line_level: LogLevel,
    line_elapsed: Duration,
}

impl<'a> Watch<'a> {
    /// The pace of progress reports when none is set.
    pub const DEFAULT_PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

    /// Polls `stop`, when given, and reports to `observer`, when given, at most one progress report per `interval`.
    #[must_use]
    pub const fn new(stop: Option<&'a dyn Stop>, observer: Option<&'a mut dyn Observer>, interval: Duration) -> Self {
        Self {
            stop,
            observer,
            interval,
            last_progress: None,
            line: String::new(),
            line_level: LogLevel::Info,
            line_elapsed: Duration::ZERO,
        }
    }

    /// Reports the part of a log line still waiting for its line break, if any but carriage returns; a second call
    /// reports nothing.
    pub fn finish(&mut self) {
        if self.line.trim_end_matches('\r').is_empty() {
            self.line.clear();
        } else {
            self.emit_line();
        }
    }

    /// Reports progress unless the last report was less than an interval before; an interval that reaches past
    /// the longest duration after the last report is never over.
    fn report(&mut self, progress: &Progress) {
        let due = match self.last_progress {
            None => true,
            Some(last) => last
                .checked_add(self.interval)
                .is_some_and(|next| progress.elapsed >= next),
        };
        if let (true, Some(observer)) = (due, self.observer.as_deref_mut()) {
            self.last_progress = Some(progress.elapsed);
            observer.progress(progress);
        }
    }

    /// Reports the line gathered so far without the carriage returns that end it, which may have arrived in an
    /// earlier chunk than its line break, and empties it.
    fn emit_line(&mut self) {
        let line = core::mem::take(&mut self.line);
        if let Some(observer) = self.observer.as_deref_mut() {
            observer.log(&LogLine {
                elapsed: self.line_elapsed,
                level: self.line_level,
                text: line.trim_end_matches('\r'),
            });
        }
    }

    /// Appends a log chunk and reports every line it completes.
    fn append(&mut self, text: &str, level: LogLevel, elapsed: Duration) {
        if self.line.is_empty() {
            self.line_level = level;
            self.line_elapsed = elapsed;
        }
        let mut rest = text;
        while let Some(end) = rest.find('\n') {
            self.line.push_str(&rest[..end]);
            self.emit_line();
            self.line_level = level;
            self.line_elapsed = elapsed;
            rest = &rest[end + 1..];
        }
        self.line.push_str(rest);
    }
}

impl<B: Backend> Callback<B> for Watch<'_> {
    fn events(&self) -> Events {
        let watched = if self.observer.is_some() {
            Event::Log | Event::Progress | Event::Poll | Event::Incumbent
        } else {
            Events::NONE
        };
        if self.stop.is_some() {
            watched | Event::Poll
        } else {
            watched
        }
    }

    fn log(&mut self, ctx: &mut B::Log<'_>) {
        self.append(ctx.text(), ctx.level(), ctx.progress().elapsed);
    }

    fn progress(&mut self, ctx: &mut B::Progress<'_>) {
        self.report(&ctx.progress());
    }

    fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        if self.stop.is_some_and(|stop| stop.should_stop()) {
            ctx.interrupt();
        }
        self.report(&ctx.progress());
    }

    fn incumbent(&mut self, ctx: &mut B::Solution<'_>) {
        if let Some(observer) = self.observer.as_deref_mut() {
            observer.improved(&Improvement {
                progress: ctx.progress(),
                objective: ctx.objective(),
                values: ctx.values(),
            });
        }
    }
}

impl fmt::Debug for Watch<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Watch")
            .field("stop", &self.stop.is_some())
            .field("observer", &self.observer.is_some())
            .field("interval", &self.interval)
            .field("last_progress", &self.last_progress)
            .finish_non_exhaustive()
    }
}
