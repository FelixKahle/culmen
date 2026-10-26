// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every HiGHS callback kind as a typed context, and the trampoline HiGHS calls.
//!
//! # Which kind becomes which event
//!
//! | HiGHS 1.15 callback | culmen event | context |
//! |---|---|---|
//! | `kHighsCallbackLogging` | [`Events::LOG`] | [`HighsLog`] |
//! | `kHighsCallbackSimplexInterrupt` | [`Events::POLL`] | [`HighsPoll`], [`PollSource::Simplex`] |
//! | `kHighsCallbackIpmInterrupt` | [`Events::POLL`] | [`HighsPoll`], [`PollSource::Ipm`] |
//! | `kHighsCallbackMipInterrupt` | [`Events::POLL`] | [`HighsPoll`], [`PollSource::Mip`] |
//! | `kHighsCallbackMipLogging` | [`Events::PROGRESS`] | [`HighsMipProgress`] |
//! | `kHighsCallbackMipSolution` | [`Events::FEASIBLE`] | [`HighsSolution`] |
//! | `kHighsCallbackMipImprovingSolution` | [`Events::INCUMBENT`] | [`HighsSolution`] |
//! | `kHighsCallbackMipUserSolution` | [`Events::HEURISTIC`] | [`HighsHeuristic`] |
//! | `kHighsCallbackMipGetCutPool` | [`Events::NATIVE`] | [`HighsNative::CutPool`] |
//! | `kHighsCallbackMipDefineLazyConstraints` | none | HiGHS 1.15 declares it and never calls it |
//!
//! HiGHS has no user cuts and never calls its lazy-constraint callback, so [`Events::LAZY`] and [`Events::CUT`]
//! take [`Unsupported`](culmen_core::Unsupported). An interrupt is honoured only at the three interrupt kinds,
//! which is the only context with [`PollContext::interrupt`]; a solution is taken only at the user-solution kind,
//! the only context with [`HeuristicContext::supply`].
//!
//! # Threads, and why the trampoline is sound
//!
//! HiGHS calls the MIP kinds from its serial search loop: `HighsMipSolverData::checkLimits` calls the interrupt only
//! while no parallel section holds the lock (`!parallelLockActive()`), and the solution, logging, cut-pool and
//! user-solution kinds come from `addIncumbent`, `printDisplayLine`, `evaluateRootNode` and `queryExternalSolution`,
//! all on the thread running `Highs_run`. The simplex and IPM polls come from the solvers' main loops. The logging
//! kind, though, fires from `highsLogUser`, which code inside HiGHS's task pool may reach on a worker thread. So
//! the trampoline does not rely on a single thread: everything a callback touches sits behind one `Mutex`, the
//! callback is `Send` and the stop `Sync`, and every call takes the lock, so calls are serialised whatever thread
//! makes them. A call that re-enters the same run on a thread already inside its callback, as the log lines HiGHS
//! writes while [`HighsHeuristic::supply_partial`] repairs a solution do, is not delivered, since the callback is
//! busy; a solve of another instance started from inside a callback gets its own callbacks.
//!
//! The per-run data lives on the stack of [`Highs::solve`](crate::Highs) and is registered for that run only: the
//! callback is installed just before `Highs_run` and uninstalled (every kind stopped, the data pointer null) just
//! after it returns, before the data is dropped, so no pointer outlives the run. A panic in a callback is caught in
//! the trampoline, never unwinds into HiGHS, turns every later interrupt poll of the run into an interrupt, and is
//! resumed once `Highs_run` has returned and the callback is uninstalled.
//!
//! Values HiGHS reports are read only where HiGHS sets them for the kind: a log line's data carries only its log
//! type, so the trampoline reads nothing else of it, and the input record of a log line is never touched.

use core::any::Any;
use core::cell::Cell;
use core::ffi::{c_char, c_int, c_void};
use core::ptr;
use core::time::Duration;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use culmen_core::context::{check_entries, check_solution};
use culmen_core::{
    ActionError, Callback, Col, Events, HeuristicContext, LogContext, LogLevel, Phase, PollContext, Progress,
    ProgressContext, SolutionContext, TiSlice,
};
use highs_sys::{
    Highs_repairCallbackSolution, Highs_setCallbackSparseSolution, HighsCallbackDataIn, HighsCallbackDataOut, HighsInt,
    kHighsCallbackCallbackMipUserSolution, kHighsCallbackIpmInterrupt, kHighsCallbackLogging,
    kHighsCallbackMipGetCutPool, kHighsCallbackMipImprovingSolution, kHighsCallbackMipInterrupt,
    kHighsCallbackMipLogging, kHighsCallbackMipSolution, kHighsCallbackSimplexInterrupt, kHighsStatusError,
};

use crate::instance::text;
use crate::{Highs, Loaded};

/// A caught panic's payload, resumed once the run has returned.
pub(crate) type Payload = Box<dyn Any + Send>;

thread_local! {
    /// The run whose callback this thread is inside, null outside every callback: a call HiGHS makes back into the
    /// trampoline for that same run is told apart from a call on another thread, or for another run nested inside
    /// the callback.
    static INSIDE: Cell<*const c_void> = const { Cell::new(ptr::null()) };
}

// The trampoline shares the hook between whatever threads HiGHS calls back on.
const _: () = {
    const fn shared<T: Sync>() {}
    shared::<Mutex<Hook<'static>>>();
};

/// A HiGHS callback kind the trampoline handles: every kind HiGHS 1.15 calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `kHighsCallbackLogging`.
    Logging,
    /// An interrupt poll of the solver named.
    Poll(Poller),
    /// `kHighsCallbackMipLogging`.
    MipLogging,
    /// `kHighsCallbackMipSolution`.
    MipSolution,
    /// `kHighsCallbackMipImprovingSolution`.
    MipImprovingSolution,
    /// `kHighsCallbackMipUserSolution`.
    MipUserSolution,
    /// `kHighsCallbackMipGetCutPool`.
    MipGetCutPool,
}

/// The solver of an interrupt poll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Poller {
    /// `kHighsCallbackSimplexInterrupt`.
    Simplex,
    /// `kHighsCallbackIpmInterrupt`.
    Ipm,
    /// `kHighsCallbackMipInterrupt`.
    Mip,
}

impl Kind {
    /// The kind of HiGHS's number, `None` for one HiGHS 1.15 never calls.
    #[expect(
        non_upper_case_globals,
        reason = "the patterns are HiGHS's constants under their C names"
    )]
    const fn from_raw(raw: c_int) -> Option<Self> {
        Some(match raw {
            kHighsCallbackLogging => Self::Logging,
            kHighsCallbackSimplexInterrupt => Self::Poll(Poller::Simplex),
            kHighsCallbackIpmInterrupt => Self::Poll(Poller::Ipm),
            kHighsCallbackMipInterrupt => Self::Poll(Poller::Mip),
            kHighsCallbackMipLogging => Self::MipLogging,
            kHighsCallbackMipSolution => Self::MipSolution,
            kHighsCallbackMipImprovingSolution => Self::MipImprovingSolution,
            kHighsCallbackCallbackMipUserSolution => Self::MipUserSolution,
            kHighsCallbackMipGetCutPool => Self::MipGetCutPool,
            _ => return None,
        })
    }

    /// HiGHS's number for the kind.
    const fn raw(self) -> c_int {
        match self {
            Self::Logging => kHighsCallbackLogging,
            Self::Poll(Poller::Simplex) => kHighsCallbackSimplexInterrupt,
            Self::Poll(Poller::Ipm) => kHighsCallbackIpmInterrupt,
            Self::Poll(Poller::Mip) => kHighsCallbackMipInterrupt,
            Self::MipLogging => kHighsCallbackMipLogging,
            Self::MipSolution => kHighsCallbackMipSolution,
            Self::MipImprovingSolution => kHighsCallbackMipImprovingSolution,
            Self::MipUserSolution => kHighsCallbackCallbackMipUserSolution,
            Self::MipGetCutPool => kHighsCallbackMipGetCutPool,
        }
    }
}

/// The HiGHS callback kinds to start for the events `wanted`: the interrupt polls always, so a panic anywhere can
/// stop the run, and logging always, so a failed run can name HiGHS's last error.
pub(crate) fn kinds(wanted: Events) -> Vec<c_int> {
    let always = [
        Kind::Logging,
        Kind::Poll(Poller::Simplex),
        Kind::Poll(Poller::Ipm),
        Kind::Poll(Poller::Mip),
    ];
    let by_event = [
        (Events::PROGRESS, Kind::MipLogging),
        (Events::FEASIBLE, Kind::MipSolution),
        (Events::INCUMBENT, Kind::MipImprovingSolution),
        (Events::HEURISTIC, Kind::MipUserSolution),
        (Events::NATIVE, Kind::MipGetCutPool),
    ];
    let chosen = by_event
        .into_iter()
        .filter(|&(event, _)| wanted.contains(event))
        .map(|(_, kind)| kind);
    always.into_iter().chain(chosen).map(Kind::raw).collect()
}

/// What the trampoline reaches during one run, behind the run's `Mutex`.
pub(crate) struct Hook<'a> {
    /// The run's callback.
    pub(crate) callback: &'a mut dyn Callback<Highs>,
    /// The events the callback asked for that HiGHS delivers.
    pub(crate) wanted: Events,
    /// When the run began.
    pub(crate) start: Instant,
    /// The loaded program, to size and value solutions.
    pub(crate) program: &'a Loaded,
    /// A panic a callback raised, held until the run returns.
    pub(crate) panic: Option<Payload>,
    /// The last error line of HiGHS's log.
    pub(crate) last_error: Option<String>,
    /// The last solution a callback supplied, until HiGHS takes it or a better incumbent arrives.
    pub(crate) offered: Option<Offer>,
    /// The objective of the last incumbent reported to the callback.
    pub(crate) incumbent: Option<f64>,
}

/// A solution a callback supplied, valued.
#[derive(Debug)]
pub(crate) struct Offer {
    objective: f64,
    values: Vec<f64>,
}

/// The function HiGHS calls for every started callback kind.
///
/// # Safety
///
/// HiGHS calls it as `HighsCCallbackType`, with `user` the data installed for the run: null, or a
/// `Mutex<Hook>` that lives until the callback is uninstalled. For every kind `out` points to a record HiGHS keeps
/// valid for the call, with the fields of that kind set; `data_in` points to the input record of the call, valid and
/// initialised for every kind except logging.
pub(crate) unsafe extern "C" fn trampoline(
    kind: c_int,
    message: *const c_char,
    out: *const HighsCallbackDataOut,
    data_in: *mut HighsCallbackDataIn,
    user: *mut c_void,
) {
    if user.is_null() || out.is_null() || INSIDE.get() == user.cast_const() {
        return;
    }
    let Some(kind) = Kind::from_raw(kind) else { return };
    debug_assert!(
        user.cast::<Mutex<Hook<'_>>>().is_aligned(),
        "trampoline: the callback data is not a run's hook"
    );
    // SAFETY: `user` is not null, so it is the run's `Mutex<Hook>`, alive until the callback is uninstalled after
    // `Highs_run` returns; a shared reference to a `Mutex` is what any thread may hold.
    let shared = unsafe { &*user.cast::<Mutex<Hook<'_>>>() };
    let mut hook = shared.lock().unwrap_or_else(PoisonError::into_inner);
    let outer = INSIDE.replace(user.cast_const());
    let call = Call {
        kind,
        message,
        out,
        data_in,
    };
    let mut interrupt = false;
    if hook.panic.is_none() {
        let hook = &mut *hook;
        let caught = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the caller's contract for this call holds for `call`.
            unsafe { dispatch(hook, &call, &mut interrupt) };
        }));
        if let Err(payload) = caught {
            hook.panic = Some(payload);
        }
    }
    let stop = interrupt || hook.panic.is_some();
    INSIDE.set(outer);
    drop(hook);
    if stop && matches!(kind, Kind::Poll(_)) {
        debug_assert!(
            !data_in.is_null(),
            "trampoline: HiGHS passed no input record to an interrupt poll"
        );
        // SAFETY: at an interrupt kind `data_in` is the call's valid input record, which HiGHS reads after the
        // callback returns.
        unsafe { (&raw mut (*data_in).user_interrupt).write(1) };
    }
}

/// One call of the trampoline, as HiGHS made it.
struct Call {
    kind: Kind,
    message: *const c_char,
    out: *const HighsCallbackDataOut,
    data_in: *mut HighsCallbackDataIn,
}

/// Reads one field of the callback record.
macro_rules! field {
    ($call:expr, $name:ident) => {{
        debug_assert!(!$call.out.is_null(), "field: HiGHS passed no output record");
        // SAFETY: `out` is valid for the call, and the field is one HiGHS sets for this kind.
        unsafe { (&raw const (*$call.out).$name).read() }
    }};
}

/// Builds the context of the call's kind and hands it to the callback.
///
/// # Safety
///
/// `call` is a call of the trampoline whose contract holds.
unsafe fn dispatch(hook: &mut Hook<'_>, call: &Call, interrupt: &mut bool) {
    let elapsed = hook.start.elapsed();
    match call.kind {
        // SAFETY: the caller's contract holds for `call`, the logging kind.
        Kind::Logging => unsafe { log(hook, call, elapsed) },
        Kind::Poll(poller) => poll(hook, call, poller, elapsed, interrupt),
        Kind::MipLogging => hook.callback.progress(&mut HighsMipProgress {
            elapsed,
            mip: mip_data(call),
        }),
        // SAFETY: the caller's contract holds for `call`, a solution kind.
        Kind::MipSolution | Kind::MipImprovingSolution => unsafe { solution(hook, call, elapsed) },
        // SAFETY: the caller's contract holds for `call`, the user-solution kind.
        Kind::MipUserSolution => unsafe { heuristic(hook, call, elapsed) },
        Kind::MipGetCutPool => {
            // SAFETY: the caller's contract holds for `call`, the cut-pool kind.
            let pool = unsafe { cut_pool(call) };
            hook.callback.native(&mut HighsNative::CutPool(pool));
        }
    }
}

/// Hands an interrupt poll to the callback; a MIP poll first reports a supplied solution HiGHS took.
fn poll(hook: &mut Hook<'_>, call: &Call, poller: Poller, elapsed: Duration, interrupt: &mut bool) {
    let source = match poller {
        Poller::Simplex => PollSource::Simplex {
            iterations: count(field!(call, simplex_iteration_count)),
        },
        Poller::Ipm => PollSource::Ipm {
            iterations: count(field!(call, ipm_iteration_count)),
        },
        Poller::Mip => {
            report_offer(hook, call, elapsed);
            PollSource::Mip(mip_data(call))
        }
    };
    if hook.wanted.contains(Events::POLL) {
        hook.callback.poll(&mut HighsPoll {
            elapsed,
            source,
            interrupt,
        });
    }
}

/// A count HiGHS reports, zero for a negative one.
fn count<T: TryInto<u64>>(value: T) -> u64 {
    value.try_into().unwrap_or(0)
}

/// A length or position HiGHS reports, zero for a negative one.
fn position(value: HighsInt) -> usize {
    usize::try_from(value).unwrap_or(0)
}

/// Hands a log chunk to the callback, keeping the last error line.
///
/// # Safety
///
/// `call` is the logging kind of a call whose contract holds.
unsafe fn log(hook: &mut Hook<'_>, call: &Call, elapsed: Duration) {
    // SAFETY: HiGHS passes the log line as a NUL-terminated string valid for the call.
    let text = unsafe { text(call.message) };
    let log_type = HighsLogType::from_raw(field!(call, log_type));
    if log_type == HighsLogType::Error {
        hook.last_error = Some(text.trim_end().to_owned());
    }
    if hook.wanted.contains(Events::LOG) {
        hook.callback.log(&mut HighsLog {
            text: &text,
            log_type,
            elapsed,
        });
    }
}

/// Hands HiGHS's request for a user solution to the callback.
///
/// # Safety
///
/// `call` is the user-solution kind of a call whose contract holds.
unsafe fn heuristic(hook: &mut Hook<'_>, call: &Call, elapsed: Duration) {
    debug_assert!(
        !call.data_in.is_null() && call.data_in.is_aligned(),
        "heuristic: HiGHS passed no usable input record"
    );
    // SAFETY: at the user-solution kind `data_in` is the call's valid, initialised input record, which HiGHS
    // neither reads nor writes until the callback returns.
    let record = UserSolution(unsafe { &mut *call.data_in });
    let mut ctx = HighsHeuristic {
        program: hook.program,
        offered: &mut hook.offered,
        elapsed,
        mip: mip_data(call),
        origin: QueryOrigin::from_raw(field!(call, external_solution_query_origin)),
        record,
    };
    hook.callback.heuristic(&mut ctx);
}

/// Hands a feasible or improving solution to the callback; a record whose size is not the column count is
/// skipped.
///
/// # Safety
///
/// `call` is a solution kind of a call whose contract holds.
unsafe fn solution(hook: &mut Hook<'_>, call: &Call, elapsed: Duration) {
    let (values, size) = (field!(call, mip_solution), position(field!(call, mip_solution_size)));
    // SAFETY: HiGHS's record holds `size` values at `mip_solution`, valid for the call, or a null pointer.
    let values = unsafe { slice(values, size) };
    if values.len() != hook.program.num_cols() {
        return;
    }
    let mut ctx = HighsSolution {
        elapsed,
        objective: field!(call, objective_function_value),
        values: TiSlice::from_ref(values),
        mip: mip_data(call),
    };
    if call.kind == Kind::MipSolution {
        hook.callback.feasible(&mut ctx);
    } else {
        hook.incumbent = Some(ctx.objective);
        hook.callback.incumbent(&mut ctx);
    }
}

/// Reports a supplied solution HiGHS took as its incumbent: HiGHS calls neither solution kind for a user solution,
/// so the trampoline reports it at the first MIP poll whose best objective is the supplied solution's.
fn report_offer(hook: &mut Hook<'_>, call: &Call, elapsed: Duration) {
    let Some(primal) = finite(field!(call, mip_primal_bound)) else {
        return;
    };
    let Some(offer) = hook.offered.as_ref() else { return };
    let taken = (primal - offer.objective).abs() <= 1e-9 * primal.abs().max(1.0);
    let improves = hook
        .incumbent
        .is_none_or(|best| hook.program.sense.better(primal, best));
    if !improves {
        hook.offered = None;
        return;
    }
    if !taken {
        return;
    }
    let Some(offer) = hook.offered.take() else { return };
    hook.incumbent = Some(offer.objective);
    if hook.wanted.contains(Events::INCUMBENT) {
        let values = TiSlice::from_ref(offer.values.as_slice());
        let mut ctx = HighsSolution {
            elapsed,
            objective: offer.objective,
            values,
            mip: mip_data(call),
        };
        hook.callback.incumbent(&mut ctx);
    }
}

/// Copies the cut pool HiGHS reports.
///
/// # Safety
///
/// `call` is the cut-pool kind of a call whose contract holds.
unsafe fn cut_pool(call: &Call) -> CutPool {
    // SAFETY: the caller's contract holds for `call`, the cut-pool kind.
    let arrays = unsafe { CutArrays::of(call) };
    CutPool {
        num_cols: position(field!(call, cutpool_num_col)),
        cuts: arrays.cuts().collect(),
        mip: mip_data(call),
    }
}

/// HiGHS's cut pool as its record lays it out: cut `i` spans the entries from `starts[i]` to `starts[i + 1]`.
///
/// Each array is as long as HiGHS's counts say, or empty where HiGHS passed none; the reads below never index past
/// an array, whatever the starts say.
struct CutArrays<'a> {
    starts: &'a [HighsInt],
    index: &'a [HighsInt],
    value: &'a [f64],
    lower: &'a [f64],
    upper: &'a [f64],
}

impl<'a> CutArrays<'a> {
    /// The arrays of a cut-pool call.
    ///
    /// # Safety
    ///
    /// `call` is the cut-pool kind of a call whose contract holds, and the call lasts for `'a`.
    unsafe fn of(call: &'a Call) -> Self {
        let (num_cuts, num_nz) = (
            position(field!(call, cutpool_num_cut)),
            position(field!(call, cutpool_num_nz)),
        );
        let (starts, index, value) = (
            field!(call, cutpool_start),
            field!(call, cutpool_index),
            field!(call, cutpool_value),
        );
        let (lower, upper) = (field!(call, cutpool_lower), field!(call, cutpool_upper));
        // SAFETY: each array holds as many values as HiGHS's counts say, valid for the call, or is null when empty;
        // the starts hold one more than there are cuts.
        unsafe {
            Self {
                starts: slice(starts, num_cuts + 1),
                index: slice(index, num_nz),
                value: slice(value, num_nz),
                lower: slice(lower, num_cuts),
                upper: slice(upper, num_cuts),
            }
        }
    }

    /// Every cut both bound arrays hold, each with the entries its starts delimit within the entry arrays.
    fn cuts(&self) -> impl Iterator<Item = Cut> + '_ {
        let bounds = self.lower.iter().zip(self.upper);
        bounds.enumerate().map(|(i, (&lower, &upper))| Cut {
            lower,
            upper,
            entries: self.entries(i),
        })
    }

    /// The entries of cut `i`, cut short where the arrays end.
    fn entries(&self, i: usize) -> Vec<(ReducedCol, f64)> {
        let start = |i: usize| self.starts.get(i).copied().map_or(0, position);
        let end = start(i + 1).min(self.index.len()).min(self.value.len());
        let range = start(i).min(end)..end;
        let columns = self.index[range.clone()].iter().map(|&c| ReducedCol(position(c)));
        columns.zip(self.value[range].iter().copied()).collect()
    }
}

/// The `len` values at `data`, or nothing when `data` is null or `len` zero.
///
/// # Safety
///
/// `data` is null, or points to `len` initialised values valid for `'a`.
unsafe fn slice<'a, T>(data: *const T, len: usize) -> &'a [T] {
    if data.is_null() || len == 0 {
        return &[];
    }
    debug_assert!(data.is_aligned(), "slice: HiGHS's array is misaligned");
    // SAFETY: the caller vouches for `len` values at `data`, valid for `'a`.
    unsafe { core::slice::from_raw_parts(data, len) }
}

/// The MIP figures every MIP kind reports.
fn mip_data(call: &Call) -> MipData {
    MipData {
        running_time: Duration::try_from_secs_f64(field!(call, running_time)).unwrap_or(Duration::ZERO),
        objective: finite(field!(call, objective_function_value)),
        primal_bound: finite(field!(call, mip_primal_bound)),
        dual_bound: finite(field!(call, mip_dual_bound)),
        gap: finite(field!(call, mip_gap)).filter(|&g| g >= 0.0),
        nodes: count(field!(call, mip_node_count)),
        lp_iterations: count(field!(call, mip_total_lp_iterations)),
    }
}

/// `value` when finite: HiGHS reports "none yet" as an infinity.
fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// HiGHS's log type of a log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HighsLogType {
    /// `kInfo`: the normal log.
    Info,
    /// `kDetailed`: more detail.
    Detailed,
    /// `kVerbose`: the most detail.
    Verbose,
    /// `kWarning`.
    Warning,
    /// `kError`.
    Error,
}

impl HighsLogType {
    /// The type of HiGHS's number for it; an unknown number reads as info.
    pub(crate) const fn from_raw(raw: c_int) -> Self {
        match raw {
            2 => Self::Detailed,
            3 => Self::Verbose,
            4 => Self::Warning,
            5 => Self::Error,
            _ => Self::Info,
        }
    }

    /// The generic level.
    #[must_use]
    pub const fn level(self) -> LogLevel {
        match self {
            Self::Info => LogLevel::Info,
            Self::Detailed | Self::Verbose => LogLevel::Detail,
            Self::Warning => LogLevel::Warning,
            Self::Error => LogLevel::Error,
        }
    }
}

/// A chunk of HiGHS's log: `kHighsCallbackLogging`.
#[derive(Debug)]
pub struct HighsLog<'a> {
    text: &'a str,
    log_type: HighsLogType,
    elapsed: Duration,
}

impl HighsLog<'_> {
    /// HiGHS's own log type.
    #[must_use]
    pub const fn log_type(&self) -> HighsLogType {
        self.log_type
    }
}

impl LogContext for HighsLog<'_> {
    fn text(&self) -> &str {
        self.text
    }
    fn level(&self) -> LogLevel {
        self.log_type.level()
    }
    fn progress(&self) -> Progress {
        Progress::at(self.elapsed, Phase::Other)
    }
}

/// What every MIP callback kind reports, in the model's objective space, offset included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MipData {
    /// The MIP solver's own clock, which starts after presolve.
    pub running_time: Duration,
    /// The objective value HiGHS reports with the call: the incumbent's, or the reported solution's at a solution
    /// kind; `None` before there is one.
    pub objective: Option<f64>,
    /// The best objective value found; `None` before a solution is.
    pub primal_bound: Option<f64>,
    /// The best bound proved; `None` before one is.
    pub dual_bound: Option<f64>,
    /// HiGHS's relative gap; `None` while it is infinite.
    pub gap: Option<f64>,
    /// The number of branch-and-bound nodes.
    pub nodes: u64,
    /// The number of LP iterations over all nodes.
    pub lp_iterations: u64,
}

impl MipData {
    fn progress(&self, elapsed: Duration) -> Progress {
        Progress {
            elapsed,
            phase: Phase::Mip,
            objective: self.primal_bound,
            bound: self.dual_bound,
            nodes: Some(self.nodes),
            iterations: Some(self.lp_iterations),
        }
    }
}

/// A line of HiGHS's MIP log was written: `kHighsCallbackMipLogging`.
#[derive(Debug)]
pub struct HighsMipProgress {
    elapsed: Duration,
    mip: MipData,
}

impl HighsMipProgress {
    /// HiGHS's MIP figures.
    #[must_use]
    pub const fn mip(&self) -> &MipData {
        &self.mip
    }
}

impl ProgressContext for HighsMipProgress {
    fn progress(&self) -> Progress {
        self.mip.progress(self.elapsed)
    }
}

/// Which solver polls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PollSource {
    /// The simplex method: `kHighsCallbackSimplexInterrupt`.
    Simplex {
        /// Simplex iterations so far.
        iterations: u64,
    },
    /// The interior-point method: `kHighsCallbackIpmInterrupt`.
    Ipm {
        /// Interior-point iterations so far.
        iterations: u64,
    },
    /// The MIP solver: `kHighsCallbackMipInterrupt`.
    Mip(MipData),
}

/// An interrupt poll: `kHighsCallbackSimplexInterrupt`, `kHighsCallbackIpmInterrupt` or
/// `kHighsCallbackMipInterrupt`.
#[derive(Debug)]
pub struct HighsPoll<'a> {
    elapsed: Duration,
    source: PollSource,
    interrupt: &'a mut bool,
}

impl HighsPoll<'_> {
    /// Which solver polls, with its figures.
    #[must_use]
    pub const fn source(&self) -> &PollSource {
        &self.source
    }
}

impl PollContext for HighsPoll<'_> {
    fn progress(&self) -> Progress {
        match self.source {
            PollSource::Simplex { iterations } => Progress {
                iterations: Some(iterations),
                ..Progress::at(self.elapsed, Phase::Simplex)
            },
            PollSource::Ipm { iterations } => Progress {
                iterations: Some(iterations),
                ..Progress::at(self.elapsed, Phase::Barrier)
            },
            PollSource::Mip(mip) => mip.progress(self.elapsed),
        }
    }
    fn interrupt(&mut self) {
        *self.interrupt = true;
    }
}

/// A feasible or improving solution: `kHighsCallbackMipSolution` or `kHighsCallbackMipImprovingSolution`.
#[derive(Debug)]
pub struct HighsSolution<'a> {
    elapsed: Duration,
    objective: f64,
    values: &'a TiSlice<Col, f64>,
    mip: MipData,
}

impl HighsSolution<'_> {
    /// HiGHS's MIP figures.
    #[must_use]
    pub const fn mip(&self) -> &MipData {
        &self.mip
    }
}

impl SolutionContext for HighsSolution<'_> {
    fn progress(&self) -> Progress {
        self.mip.progress(self.elapsed)
    }
    fn objective(&self) -> f64 {
        self.objective
    }
    fn values(&self) -> &TiSlice<Col, f64> {
        self.values
    }
}

/// Where in the search HiGHS asks for a user solution: HiGHS's `ExternalMipSolutionQueryOrigin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueryOrigin {
    /// After the MIP solver is set up.
    AfterSetup,
    /// Before a dive.
    BeforeDive,
    /// During root-node evaluation.
    EvaluateRootNode(RootNodePoint),
    /// A value HiGHS 1.15 does not define.
    Other(i32),
}

/// Where in the root-node evaluation HiGHS asks for a user solution: HiGHS's
/// `kExternalMipSolutionQueryOriginEvaluateRootNode0` to `4`, in the order a root-node evaluation reaches them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RootNodePoint {
    /// Before the root LP is solved: point 0.
    BeforeLp,
    /// After each root separation round that found cuts: point 1.
    SeparationRound,
    /// After root separation ends, before the cut pool is reported and the root heuristics run: point 2.
    AfterSeparation,
    /// After the separation round that follows the RENS heuristic, when that round runs: point 3.
    AfterRens,
    /// At the end of the root-node evaluation, before the search: point 4.
    End,
}

impl QueryOrigin {
    /// The origin of HiGHS's number for it.
    pub(crate) const fn from_raw(raw: i32) -> Self {
        match raw {
            0 => Self::AfterSetup,
            1 => Self::BeforeDive,
            2 => Self::EvaluateRootNode(RootNodePoint::BeforeLp),
            3 => Self::EvaluateRootNode(RootNodePoint::SeparationRound),
            4 => Self::EvaluateRootNode(RootNodePoint::AfterSeparation),
            5 => Self::EvaluateRootNode(RootNodePoint::AfterRens),
            6 => Self::EvaluateRootNode(RootNodePoint::End),
            other => Self::Other(other),
        }
    }
}

/// HiGHS asks for a solution of the callback's own: `kHighsCallbackMipUserSolution`.
///
/// A solution supplied here is checked by HiGHS against the model and kept as the incumbent if it is feasible and
/// better.
#[derive(Debug)]
pub struct HighsHeuristic<'a> {
    program: &'a Loaded,
    offered: &'a mut Option<Offer>,
    elapsed: Duration,
    mip: MipData,
    origin: QueryOrigin,
    record: UserSolution<'a>,
}

impl HighsHeuristic<'_> {
    /// HiGHS's MIP figures.
    #[must_use]
    pub const fn mip(&self) -> &MipData {
        &self.mip
    }

    /// Where in the search HiGHS asks.
    #[must_use]
    pub const fn origin(&self) -> QueryOrigin {
        self.origin
    }
}

impl HighsHeuristic<'_> {
    /// Keeps a supplied solution, valued, to report once HiGHS takes it.
    fn offer(&mut self, values: Vec<f64>) {
        let objective = self.program.objective(&values);
        *self.offered = Some(Offer { objective, values });
    }
}

impl HeuristicContext for HighsHeuristic<'_> {
    fn progress(&self) -> Progress {
        self.mip.progress(self.elapsed)
    }

    fn supply(&mut self, values: &TiSlice<Col, f64>) -> Result<(), ActionError> {
        check_solution(values, self.program.num_cols())?;
        let buffer = self.record.buffer();
        if buffer.len() != values.len() {
            return Err(ActionError::Refused(format!(
                "HiGHS offers room for {} values",
                buffer.len()
            )));
        }
        buffer.copy_from_slice(values.as_ref());
        self.record.mark_written();
        self.offer(values.raw.to_vec());
        Ok(())
    }

    fn supply_partial(&mut self, entries: &[(Col, f64)]) -> Result<(), ActionError> {
        let num_cols = self.program.num_cols();
        check_entries(entries, num_cols)?;
        self.record.set_sparse(entries)?;
        self.record.repair()?;
        // HiGHS copies the C record's flag back over its own after the callback, so the flag is set here too; the
        // repair wrote a value for every column into the record's buffer.
        self.record.mark_written();
        let buffer = self.record.buffer();
        if let Some(values) = (buffer.len() == num_cols).then(|| buffer.to_vec()) {
            self.offer(values);
        }
        Ok(())
    }
}

/// HiGHS's input record at the user-solution kind: the buffer HiGHS sized for one value per column of its model,
/// the flag that tells HiGHS the buffer holds a solution, and the handle to HiGHS's own input behind them.
///
/// The record is HiGHS's for the call alone and nothing else reads or writes it until the callback returns.
#[derive(Debug)]
struct UserSolution<'a>(&'a mut HighsCallbackDataIn);

impl UserSolution<'_> {
    /// The number of columns of HiGHS's model, the size of the buffer.
    fn num_cols(&self) -> usize {
        position(self.0.user_solution_size)
    }

    /// The buffer for one value per column; empty when HiGHS gave none.
    fn buffer(&mut self) -> &mut [f64] {
        let (data, len) = (self.0.user_solution, self.num_cols());
        if data.is_null() || len == 0 {
            return &mut [];
        }
        debug_assert!(data.is_aligned(), "UserSolution: HiGHS's solution buffer is misaligned");
        // SAFETY: at the user-solution kind HiGHS points `user_solution` at its own buffer of `user_solution_size`
        // initialised values, which lives for the call and which nothing else touches while `self` is borrowed.
        unsafe { core::slice::from_raw_parts_mut(data, len) }
    }

    /// Tells HiGHS the buffer holds a solution.
    fn mark_written(&mut self) {
        self.0.user_has_solution = 1;
    }

    /// Hands HiGHS a partial solution, its columns all of HiGHS's model.
    ///
    /// # Errors
    ///
    /// [`ActionError::Refused`] when a column is not one of HiGHS's model, or HiGHS refuses the entries.
    fn set_sparse(&mut self, entries: &[(Col, f64)]) -> Result<(), ActionError> {
        let refused = || ActionError::Refused("HiGHS refused the entries".into());
        // HiGHS checks a column only against `> num_col`, so a column at the count itself is refused here.
        let num_cols = self.num_cols();
        let known = |col: Col| HighsInt::try_from(col.index()).ok().filter(|_| col.index() < num_cols);
        let index = entries.iter().map(|&(col, _)| known(col)).collect::<Option<Vec<_>>>();
        let index = index.ok_or_else(refused)?;
        let value: Vec<f64> = entries.iter().map(|&(_, v)| v).collect();
        let count = HighsInt::try_from(entries.len()).map_err(|_| refused())?;
        debug_assert!(
            index.len() == entries.len() && value.len() == entries.len(),
            "UserSolution: the index and value arrays differ in length from the count"
        );
        debug_assert!(
            index.iter().all(|&c| usize::try_from(c).is_ok_and(|c| c < num_cols)),
            "UserSolution: an index is not a column of HiGHS's model"
        );
        // SAFETY: the record is the call's input record, whose internal pointer HiGHS set to its own input; `index`
        // and `value` hold `count` entries over columns of HiGHS's model. HiGHS may log, which re-enters the
        // trampoline on this thread and is not delivered.
        let status =
            unsafe { Highs_setCallbackSparseSolution(&raw mut *self.0, count, index.as_ptr(), value.as_ptr()) };
        if status == kHighsStatusError {
            Err(refused())
        } else {
            Ok(())
        }
    }

    /// Has HiGHS complete the partial solution set before into the buffer.
    ///
    /// # Errors
    ///
    /// [`ActionError::Refused`] when HiGHS finds no feasible completion.
    fn repair(&mut self) -> Result<(), ActionError> {
        debug_assert!(
            self.0.user_solution_size >= 0,
            "UserSolution: HiGHS's buffer size is negative"
        );
        // SAFETY: the record is the call's input record, whose internal pointer HiGHS set to its own input. HiGHS
        // may log, which re-enters the trampoline on this thread and is not delivered.
        let status = unsafe { Highs_repairCallbackSolution(&raw mut *self.0) };
        if status == kHighsStatusError {
            return Err(ActionError::Refused("HiGHS found no feasible completion".into()));
        }
        Ok(())
    }
}

/// A column of the presolved problem HiGHS's MIP solver works on, which need not be a column of the loaded program.
///
/// ```
/// use culmen_highs::ReducedCol;
/// assert_eq!(ReducedCol::new(3).index(), 3);
/// assert_eq!(usize::from(ReducedCol::from(3)), 3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReducedCol(usize);

impl ReducedCol {
    /// The presolved column at a position, counting from zero.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// The column's position in the presolved problem, counting from zero.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for ReducedCol {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl From<ReducedCol> for usize {
    fn from(col: ReducedCol) -> Self {
        col.0
    }
}

/// A cut of HiGHS's cut pool: `lower <= Σ value · column <= upper` over presolved columns.
#[derive(Clone, Debug, PartialEq)]
pub struct Cut {
    /// The lower bound; `-∞` for none.
    pub lower: f64,
    /// The upper bound; `+∞` for none.
    pub upper: f64,
    /// The entries as `(column, coefficient)`.
    pub entries: Vec<(ReducedCol, f64)>,
}

/// HiGHS's cut pool after the root node: `kHighsCallbackMipGetCutPool`, copied out of HiGHS.
#[derive(Clone, Debug, PartialEq)]
pub struct CutPool {
    /// The number of columns of the presolved problem the cuts range over.
    pub num_cols: usize,
    /// The cuts.
    pub cuts: Vec<Cut>,
    /// HiGHS's MIP figures.
    pub mip: MipData,
}

/// What only HiGHS reports, delivered as [`Events::NATIVE`].
#[derive(Debug)]
#[non_exhaustive]
pub enum HighsNative {
    /// The cut pool after the root node.
    CutPool(CutPool),
}

/// The pointer `ptr::null_mut` as the trampoline's data: no run in progress.
pub(crate) const NO_RUN: *mut c_void = ptr::null_mut();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_query_origin_of_highs_has_its_own_variant() {
        let origins: Vec<QueryOrigin> = (0..=7).map(QueryOrigin::from_raw).collect();
        assert_eq!(
            origins,
            [
                QueryOrigin::AfterSetup,
                QueryOrigin::BeforeDive,
                QueryOrigin::EvaluateRootNode(RootNodePoint::BeforeLp),
                QueryOrigin::EvaluateRootNode(RootNodePoint::SeparationRound),
                QueryOrigin::EvaluateRootNode(RootNodePoint::AfterSeparation),
                QueryOrigin::EvaluateRootNode(RootNodePoint::AfterRens),
                QueryOrigin::EvaluateRootNode(RootNodePoint::End),
                QueryOrigin::Other(7),
            ]
        );
    }
}
