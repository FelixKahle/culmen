// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SCIP's parameters: reset, set by name, and set in groups by emphasis.

use core::ffi::{c_char, c_uint};
use std::ffi::CString;

use scip_sys::{
    SCIP, SCIP_PARAMSETTING, SCIP_RETCODE, SCIPresetParams, SCIPsetBoolParam, SCIPsetCharParam, SCIPsetHeuristics,
    SCIPsetIntParam, SCIPsetLongintParam, SCIPsetMessagehdlrQuiet, SCIPsetPresolving, SCIPsetRealParam,
    SCIPsetSeparating, SCIPsetStringParam,
};

use super::{Instance, TRUE};
use crate::error::check;
use crate::{Emphasis, ParamValue, RetCode, ScipError};

/// A group of SCIP's plugins an [`Emphasis`] applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Group {
    /// The primal heuristics.
    Heuristics,
    /// The presolvers.
    Presolving,
    /// The separators.
    Separating,
}

impl Instance {
    /// Frees the last solve's transformed problem and resets every parameter to SCIP's default, so the next solve
    /// starts from the problem as loaded and SCIP's defaults.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP fails either.
    pub(crate) fn reset_params(&mut self) -> Result<(), ScipError> {
        self.free_transform()?;
        // SAFETY: the instance is live, borrowed exclusively, and in its problem stage.
        check("SCIPresetParams", unsafe { SCIPresetParams(self.raw()) })
    }

    /// Whether SCIP's log reaches standard output.
    pub(crate) fn set_console(&mut self, console: bool) {
        // SAFETY: the instance is live and borrowed exclusively.
        unsafe { SCIPsetMessagehdlrQuiet(self.raw(), c_uint::from(!console)) };
    }

    /// Sets the parameters of a group of plugins to an emphasis.
    ///
    /// # Errors
    ///
    /// [`ScipError::Call`] when SCIP refuses.
    pub(crate) fn set_emphasis(&mut self, group: Group, emphasis: Emphasis) -> Result<(), ScipError> {
        type Setter = unsafe extern "C" fn(*mut SCIP, SCIP_PARAMSETTING, c_uint) -> SCIP_RETCODE;
        let (call, set): (&'static str, Setter) = match group {
            Group::Heuristics => ("SCIPsetHeuristics", SCIPsetHeuristics),
            Group::Presolving => ("SCIPsetPresolving", SCIPsetPresolving),
            Group::Separating => ("SCIPsetSeparating", SCIPsetSeparating),
        };
        // SAFETY: the instance is live, borrowed exclusively, and in its problem stage; `set` is one of SCIP's
        // emphasis setters, which take the instance, a `SCIP_PARAMSETTING` and whether to stay quiet.
        check(call, unsafe { set(self.raw(), emphasis.raw(), TRUE) })
    }

    /// Sets a parameter by SCIP's name.
    ///
    /// # Errors
    ///
    /// [`ScipError::ParamName`] when the name holds a NUL byte, [`ScipError::ParamString`] when a string value does,
    /// [`ScipError::ParamChar`] for a character that is not ASCII, [`ScipError::Param`] when SCIP refuses the
    /// parameter: an unknown name, a value of the wrong type or out of range.
    pub(crate) fn set_param(&mut self, name: &str, value: &ParamValue) -> Result<(), ScipError> {
        let c_name = CString::new(name).map_err(|_| ScipError::ParamName { name: name.to_owned() })?;
        let (raw, name_ptr) = (self.raw(), c_name.as_ptr());
        let code = match value {
            // SAFETY: the instance is live and borrowed exclusively, the name a NUL-terminated string SCIP reads
            // during the call.
            ParamValue::Bool(v) => unsafe { SCIPsetBoolParam(raw, name_ptr, c_uint::from(*v)) },
            // SAFETY: as above.
            ParamValue::Int(v) => unsafe { SCIPsetIntParam(raw, name_ptr, *v) },
            // SAFETY: as above.
            ParamValue::Longint(v) => unsafe { SCIPsetLongintParam(raw, name_ptr, *v) },
            // SAFETY: as above.
            ParamValue::Real(v) => unsafe { SCIPsetRealParam(raw, name_ptr, *v) },
            ParamValue::Char(v) => {
                let byte = u8::try_from(*v)
                    .ok()
                    .filter(u8::is_ascii)
                    .ok_or_else(|| ScipError::ParamChar {
                        name: name.to_owned(),
                        value: *v,
                    })?;
                // An ASCII byte is below 128, so it is the same number as a `c_char`, signed or not.
                let byte = c_char::try_from(byte).unwrap_or_default();
                // SAFETY: as above.
                unsafe { SCIPsetCharParam(raw, name_ptr, byte) }
            }
            ParamValue::String(v) => {
                let c_value = CString::new(v.as_str()).map_err(|_| ScipError::ParamString { name: name.to_owned() })?;
                // SAFETY: as above, and the value is a NUL-terminated string SCIP copies during the call.
                unsafe { SCIPsetStringParam(raw, name_ptr, c_value.as_ptr()) }
            }
        };
        RetCode::from_raw(code).map_or(Ok(()), |code| {
            Err(ScipError::Param {
                name: name.to_owned(),
                value: value.clone(),
                code,
            })
        })
    }
}
