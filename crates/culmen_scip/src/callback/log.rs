// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SCIP's log: a message handler of the instance's own, in place of SCIP's default one.
//!
//! The handler buffers SCIP's output up to each line break, so an info message or a warning arrives as a whole line
//! with its `\n`; SCIP cuts a line longer than its buffer, about a thousand bytes, into pieces. Each line reaches
//! standard output (an info message) or standard error (a warning) when [`Settings::console`] asks for it, and the
//! callback as a [`ScipLog`] when it asks for [`Events::LOG`]. Output SCIP writes to a file of its own, as when it
//! writes a problem, goes to that file unchanged. Sub-SCIPs that SCIP's heuristics copy share the handler, so their
//! messages, which they keep to warnings, arrive too. SCIP's error messages go through its process-wide error printer
//! instead, which the backend leaves alone.
//!
//! [`Settings::console`]: culmen_core::Settings::console
//! [`Events::LOG`]: culmen_core::Events::LOG

use core::ffi::{CStr, c_char};
use core::ptr::{self, NonNull};
use core::time::Duration;
use std::io::Write as _;

use culmen_core::{Events, LogContext, LogLevel, Phase, Progress};
use scip_sys::{
    FILE, SCIP, SCIP_MESSAGEHDLR, SCIPmessagehdlrCreate, SCIPmessagehdlrGetData, SCIPmessagehdlrRelease,
    SCIPsetMessagehdlr, fflush, fputs,
};

use super::{Slot, slot};
use crate::ScipError;
use crate::error::check;

/// A line of SCIP's log: an info message or a warning, with its line break.
#[derive(Debug)]
pub struct ScipLog<'a> {
    text: &'a str,
    level: LogLevel,
    elapsed: Duration,
}

impl<'a> ScipLog<'a> {
    pub(super) const fn new(text: &'a str, level: LogLevel, elapsed: Duration) -> Self {
        Self { text, level, elapsed }
    }
}

impl LogContext for ScipLog<'_> {
    fn text(&self) -> &str {
        self.text
    }
    /// [`LogLevel::Info`] for an info message, [`LogLevel::Warning`] for a warning.
    fn level(&self) -> LogLevel {
        self.level
    }
    fn progress(&self) -> Progress {
        Progress::at(self.elapsed, Phase::Other)
    }
}

/// Creates the handler and installs it in the instance in place of SCIP's default one.
///
/// # Errors
///
/// [`ScipError::Call`] when SCIP refuses a call.
///
/// # Safety
///
/// `scip` is a live instance in its init stage, and `slot` its slot, which lives until after SCIP is freed.
pub(super) unsafe fn install(scip: *mut SCIP, slot: NonNull<Slot>) -> Result<(), ScipError> {
    let mut handler: *mut SCIP_MESSAGEHDLR = ptr::null_mut();
    // SAFETY: `handler` is a local to write the new handler to; the callbacks treat their arguments as SCIP passes
    // them, and the data is the slot, which outlives SCIP. The handler buffers up to each line break, writes no log
    // file and is not quiet, so every message reaches the callbacks.
    check("SCIPmessagehdlrCreate", unsafe {
        SCIPmessagehdlrCreate(
            &raw mut handler,
            1,
            ptr::null(),
            0,
            Some(warning),
            Some(dialog),
            Some(info),
            None,
            slot.as_ptr().cast(),
        )
    })?;
    // SAFETY: the instance is live and in its init stage, where it takes a handler and captures it.
    let installed = check("SCIPsetMessagehdlr", unsafe { SCIPsetMessagehdlr(scip, handler) });
    // SAFETY: `handler` is the reference created above, released once; the instance holds its own.
    let released = check("SCIPmessagehdlrRelease", unsafe {
        SCIPmessagehdlrRelease(&raw mut handler)
    });
    installed.and(released)
}

/// An info message: standard output on the console, [`LogLevel::Info`] to the callback.
unsafe extern "C" fn info(handler: *mut SCIP_MESSAGEHDLR, _file: *mut FILE, msg: *const c_char) {
    // SAFETY: SCIP calls the handler with itself and a NUL-terminated message valid for the call.
    unsafe { line(handler, msg, LogLevel::Info) };
}

/// A warning: standard error on the console, [`LogLevel::Warning`] to the callback.
unsafe extern "C" fn warning(handler: *mut SCIP_MESSAGEHDLR, _file: *mut FILE, msg: *const c_char) {
    // SAFETY: as for `info`.
    unsafe { line(handler, msg, LogLevel::Warning) };
}

/// Output to a file SCIP names, written there as SCIP's default handler does.
unsafe extern "C" fn dialog(_handler: *mut SCIP_MESSAGEHDLR, file: *mut FILE, msg: *const c_char) {
    if file.is_null() || msg.is_null() {
        return;
    }
    // SAFETY: SCIP passes an open stream and a NUL-terminated message, both valid for the call.
    unsafe {
        fputs(msg, file);
        fflush(file);
    }
}

/// Hands a message to the console and the callback.
///
/// # Safety
///
/// `handler` is the instance's handler and `msg` null or a NUL-terminated string valid for the call.
unsafe fn line(handler: *mut SCIP_MESSAGEHDLR, msg: *const c_char, level: LogLevel) {
    if msg.is_null() {
        return;
    }
    // SAFETY: the handler's data is its slot.
    let Some(slot) = (unsafe { slot::<_>(SCIPmessagehdlrGetData(handler)) }) else {
        return;
    };
    // SAFETY: the caller vouches for the message.
    let text = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    if slot.console.get() {
        // A console that cannot be written to loses the line; the solve goes on.
        let _ = match level {
            LogLevel::Warning | LogLevel::Error => std::io::stderr().write_all(text.as_bytes()),
            LogLevel::Info | LogLevel::Detail => std::io::stdout().write_all(text.as_bytes()),
        };
    }
    // SAFETY: SCIP writes messages during the solve the run is installed for, or with none installed.
    if let Some(run) = unsafe { slot.run() }
        && run.wanted.contains(Events::LOG)
    {
        run.log(slot.scip, &text, level);
    }
}
