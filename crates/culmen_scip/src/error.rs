// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why the SCIP backend refused or failed, and SCIP's return codes.

use core::fmt;

use culmen_core::{Col, Row};
use scip_sys::SCIP_RETCODE;

use crate::{ParamValue, Status};

/// Why loading or solving with SCIP failed.
#[derive(Clone, Debug, PartialEq)]
pub enum ScipError {
    /// A solve or a start was asked for before a program was loaded.
    NotLoaded,
    /// A call of SCIP's C API returned an error code.
    Call {
        /// The C function of SCIP that failed.
        call: &'static str,
        /// What it returned.
        code: RetCode,
    },
    /// SCIP refused a parameter: an unknown name, a value of the wrong type, or a value out of its range.
    Param {
        /// The parameter's name.
        name: String,
        /// The value refused.
        value: ParamValue,
        /// What SCIP returned.
        code: RetCode,
    },
    /// A parameter name holds a NUL byte, which SCIP's C strings cannot carry.
    ParamName {
        /// The name.
        name: String,
    },
    /// A string parameter's value holds a NUL byte, which SCIP's C strings cannot carry.
    ParamString {
        /// The parameter's name.
        name: String,
    },
    /// A character parameter's value is not ASCII, which SCIP's C `char` cannot carry.
    ParamChar {
        /// The parameter's name.
        name: String,
        /// The value given.
        value: char,
    },
    /// An exact integer setting, such as a seed or a limit, exceeds the largest value of SCIP's parameter.
    OutOfRange {
        /// SCIP's name of the parameter.
        name: &'static str,
        /// The value given.
        value: u64,
        /// The largest value the parameter takes.
        max: u64,
    },
    /// A column's name holds a NUL byte, which SCIP's C strings cannot carry.
    ColName {
        /// The column.
        col: Col,
    },
    /// A row's name holds a NUL byte, which SCIP's C strings cannot carry.
    RowName {
        /// The row.
        row: Row,
    },
    /// A finite number of the program is at least SCIP's infinity in magnitude, so SCIP would read it as infinite.
    BeyondInfinity {
        /// Where the number is.
        number: Number,
        /// The number.
        value: f64,
        /// SCIP's infinity, `numerics/infinity`.
        infinity: f64,
    },
    /// A row has more entries than SCIP's `int` counts.
    RowTooLong {
        /// The row.
        row: Row,
        /// Its number of entries.
        len: usize,
    },
    /// The quadratic part of the objective has more entries than SCIP's `int` counts.
    QuadraticTooLong {
        /// Its number of entries.
        len: usize,
    },
    /// The start does not fit the loaded program; a [`Solver`](culmen_core::Solver) never hands such a start over.
    Start,
    /// SCIP ended a solve with a status that proves nothing (`UNKNOWN`), or claimed an optimum or a reached gap
    /// without a solution or a bound.
    Failed {
        /// SCIP's status at the end of the solve.
        status: Status,
    },
    /// A solve with a [`Pricer`](culmen_core::Pricer) was given a program with a quadratic objective, which this
    /// backend does not price; refused before SCIP is called.
    PricingQuadratic,
    /// A pricer ended pricing early at a node where SCIP has no integer variable left to branch on, as in a
    /// program without integer columns, so SCIP could not branch.
    BranchEarly,
    /// SCIP ended a solve with a best solution that its own check finds infeasible in the original problem
    /// (`SCIPcheckSolOrig`), as when presolve rewrote a constraint unsoundly; the solution is not reported.
    InfeasibleBest {
        /// SCIP's status at the end of the solve.
        status: Status,
        /// The objective SCIP claimed for the solution, offset included, in the program's sense.
        objective: f64,
    },
    /// SCIP returned a value of one of its enumerations that SCIP 10's headers do not define. The pinned SCIP returns
    /// only defined values, so this names a SCIP other than the one `scip-sys` builds.
    Unlisted {
        /// The enumeration the value belongs to.
        code: ScipCode,
        /// The value.
        raw: i64,
        /// The C function of SCIP that returned the value.
        call: &'static str,
    },
}

/// An enumeration of SCIP 10 whose values the backend reads, named in [`ScipError::Unlisted`].
///
/// ```
/// use culmen_scip::{ScipCode, ScipError};
/// let error = ScipError::Unlisted { code: ScipCode::Status, raw: 42, call: "SCIPgetStatus" };
/// assert_eq!(error.to_string(), "SCIP: SCIPgetStatus gave the status 42, which SCIP 10 does not define");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScipCode {
    /// `SCIP_RETCODE` in `scip/type_retcode.h`: `SCIP_OKAY` and [`RetCode`].
    RetCode,
    /// `SCIP_STATUS` in `scip/type_stat.h`: [`Status`].
    Status,
}

impl fmt::Display for ScipCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RetCode => "return code",
            Self::Status => "status",
        })
    }
}

impl fmt::Display for ScipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLoaded => write!(f, "SCIP: no program is loaded"),
            Self::Call { call, code } => write!(f, "SCIP: {call} failed: {code}"),
            Self::Param { name, value, code } => {
                write!(f, "SCIP: parameter `{name}` refused the value {value}: {code}")
            }
            Self::ParamName { name } => write!(f, "SCIP: parameter name {name:?} holds a NUL byte"),
            Self::ParamString { name } => write!(f, "SCIP: the value of parameter `{name}` holds a NUL byte"),
            Self::ParamChar { name, value } => {
                write!(
                    f,
                    "SCIP: the value {value:?} of parameter `{name}` is not an ASCII character"
                )
            }
            Self::OutOfRange { name, value, max } => {
                write!(f, "SCIP: parameter `{name}` takes at most {max}, not {value}")
            }
            Self::ColName { col } => write!(f, "SCIP: the name of column {} holds a NUL byte", col.index()),
            Self::RowName { row } => write!(f, "SCIP: the name of row {} holds a NUL byte", row.index()),
            Self::BeyondInfinity {
                number,
                value,
                infinity,
            } => write!(
                f,
                "SCIP: the {number} is {value}, which SCIP reads as infinite from {infinity} on"
            ),
            Self::RowTooLong { row, len } => {
                write!(
                    f,
                    "SCIP: row {} has {len} entries, more than SCIP's `int` counts",
                    row.index()
                )
            }
            Self::QuadraticTooLong { len } => {
                write!(
                    f,
                    "SCIP: the quadratic objective has {len} entries, more than SCIP's `int` counts"
                )
            }
            Self::Start => write!(f, "SCIP: the start does not fit the loaded program"),
            Self::Failed { status } => write!(f, "SCIP: the solve ended without a result ({status})"),
            Self::PricingQuadratic => write!(
                f,
                "SCIP: the backend does not price columns into a program with a quadratic objective"
            ),
            Self::BranchEarly => write!(
                f,
                "SCIP: a pricer ended pricing early at a node with no integer variable left to branch on"
            ),
            Self::InfeasibleBest { status, objective } => write!(
                f,
                "SCIP: the best solution, of objective {objective}, is infeasible in the original problem by SCIP's \
                 own check, though the solve ended with \"{status}\""
            ),
            Self::Unlisted { code, raw, call } => {
                write!(f, "SCIP: {call} gave the {code} {raw}, which SCIP 10 does not define")
            }
        }
    }
}

impl core::error::Error for ScipError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Call { code, .. } | Self::Param { code, .. } => Some(code),
            Self::NotLoaded
            | Self::ParamName { .. }
            | Self::ParamString { .. }
            | Self::ParamChar { .. }
            | Self::OutOfRange { .. }
            | Self::ColName { .. }
            | Self::RowName { .. }
            | Self::BeyondInfinity { .. }
            | Self::RowTooLong { .. }
            | Self::QuadraticTooLong { .. }
            | Self::Start
            | Self::Failed { .. }
            | Self::PricingQuadratic
            | Self::BranchEarly
            | Self::InfeasibleBest { .. }
            | Self::Unlisted { .. } => None,
        }
    }
}

/// A character a SCIP `char` parameter cannot take, refused by [`ParamValue`]'s `TryFrom<char>`: one that is not
/// ASCII, which SCIP's C `char` cannot carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotAscii(pub char);

impl fmt::Display for NotAscii {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SCIP: the character {:?} is not ASCII, as a char parameter needs",
            self.0
        )
    }
}

impl core::error::Error for NotAscii {}

/// Where a number SCIP would read as infinite stands in the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Number {
    /// A bound of the column.
    ColBound(Col),
    /// The cost of the column.
    Cost(Col),
    /// A bound of the row.
    RowBound(Row),
    /// The coefficient of a column in a row.
    Coefficient {
        /// The row.
        row: Row,
        /// The column.
        col: Col,
    },
    /// The coefficient of a product of two columns, or of a square, in the objective's quadratic part.
    QuadCoefficient {
        /// The first column, never after the second.
        first: Col,
        /// The second column.
        second: Col,
    },
    /// The objective's offset.
    Offset,
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColBound(col) => write!(f, "bound of column {}", col.index()),
            Self::Cost(col) => write!(f, "cost of column {}", col.index()),
            Self::RowBound(row) => write!(f, "bound of row {}", row.index()),
            Self::Coefficient { row, col } => {
                write!(f, "coefficient of column {} in row {}", col.index(), row.index())
            }
            Self::QuadCoefficient { first, second } => write!(
                f,
                "quadratic coefficient of columns {} and {}",
                first.index(),
                second.index()
            ),
            Self::Offset => write!(f, "objective offset"),
        }
    }
}

/// An error code a call of SCIP's C API returned: every error value of `SCIP_RETCODE` in `scip/type_retcode.h` of
/// SCIP 10.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RetCode {
    /// An unspecified error (`SCIP_ERROR`).
    Error,
    /// Insufficient memory (`SCIP_NOMEMORY`).
    NoMemory,
    /// A read error (`SCIP_READERROR`).
    ReadError,
    /// A write error (`SCIP_WRITEERROR`).
    WriteError,
    /// A file was not found (`SCIP_NOFILE`).
    NoFile,
    /// A file could not be created (`SCIP_FILECREATEERROR`).
    FileCreateError,
    /// The LP solver failed (`SCIP_LPERROR`).
    LpError,
    /// No problem exists (`SCIP_NOPROBLEM`).
    NoProblem,
    /// The method cannot be called at this stage of the solve (`SCIP_INVALIDCALL`).
    InvalidCall,
    /// The input data is invalid (`SCIP_INVALIDDATA`).
    InvalidData,
    /// A method returned an invalid result (`SCIP_INVALIDRESULT`).
    InvalidResult,
    /// A required plugin was not found (`SCIP_PLUGINNOTFOUND`).
    PluginNotFound,
    /// No parameter has the name (`SCIP_PARAMETERUNKNOWN`).
    ParameterUnknown,
    /// The parameter is of another type (`SCIP_PARAMETERWRONGTYPE`).
    ParameterWrongType,
    /// The value is invalid for the parameter (`SCIP_PARAMETERWRONGVAL`).
    ParameterWrongValue,
    /// The key already exists in a table (`SCIP_KEYALREADYEXISTING`).
    KeyAlreadyExisting,
    /// The largest branching depth was exceeded (`SCIP_MAXDEPTHLEVEL`).
    MaxDepthLevel,
    /// No branching could be created (`SCIP_BRANCHERROR`).
    BranchError,
    /// The function is not implemented (`SCIP_NOTIMPLEMENTED`).
    NotImplemented,
}

impl RetCode {
    /// The error of SCIP's code for it, `None` for `SCIP_OKAY`.
    ///
    /// # Errors
    ///
    /// [`ScipError::Unlisted`] for a code `SCIP_RETCODE` does not have.
    pub(crate) const fn from_raw(raw: SCIP_RETCODE, call: &'static str) -> Result<Option<Self>, ScipError> {
        Ok(Some(match raw {
            1 => return Ok(None),
            0 => Self::Error,
            -1 => Self::NoMemory,
            -2 => Self::ReadError,
            -3 => Self::WriteError,
            -4 => Self::NoFile,
            -5 => Self::FileCreateError,
            -6 => Self::LpError,
            -7 => Self::NoProblem,
            -8 => Self::InvalidCall,
            -9 => Self::InvalidData,
            -10 => Self::InvalidResult,
            -11 => Self::PluginNotFound,
            -12 => Self::ParameterUnknown,
            -13 => Self::ParameterWrongType,
            -14 => Self::ParameterWrongValue,
            -15 => Self::KeyAlreadyExisting,
            -16 => Self::MaxDepthLevel,
            -17 => Self::BranchError,
            -18 => Self::NotImplemented,
            _ => {
                return Err(ScipError::Unlisted {
                    code: ScipCode::RetCode,
                    raw: raw as i64,
                    call,
                });
            }
        }))
    }
}

/// SCIP's own wording of the code.
///
/// ```
/// use culmen_scip::RetCode;
/// assert_eq!(RetCode::ParameterUnknown.to_string(), "the parameter with the given name was not found");
/// ```
impl fmt::Display for RetCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Error => "unspecified error",
            Self::NoMemory => "insufficient memory error",
            Self::ReadError => "read error",
            Self::WriteError => "write error",
            Self::NoFile => "file not found error",
            Self::FileCreateError => "cannot create file",
            Self::LpError => "error in LP solver",
            Self::NoProblem => "no problem exists",
            Self::InvalidCall => "method cannot be called at this time in solution process",
            Self::InvalidData => "error in input data",
            Self::InvalidResult => "method returned an invalid result code",
            Self::PluginNotFound => "a required plugin was not found",
            Self::ParameterUnknown => "the parameter with the given name was not found",
            Self::ParameterWrongType => "the parameter is not of the expected type",
            Self::ParameterWrongValue => "the value is invalid for the given parameter",
            Self::KeyAlreadyExisting => "the given key is already existing in table",
            Self::MaxDepthLevel => "maximal branching depth level exceeded",
            Self::BranchError => "no branching could be created",
            Self::NotImplemented => "function not implemented",
        };
        f.write_str(text)
    }
}

impl core::error::Error for RetCode {}

/// Checks what a call of SCIP's C API returned.
///
/// # Errors
///
/// [`ScipError::Call`] naming `call` when `code` is an error, [`ScipError::Unlisted`] when it is no value of
/// `SCIP_RETCODE`.
pub(crate) fn check(call: &'static str, code: SCIP_RETCODE) -> Result<(), ScipError> {
    match RetCode::from_raw(code, call)? {
        None => Ok(()),
        Some(code) => Err(ScipError::Call { call, code }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_code_of_scip_10_reads_as_its_error_okay_as_none_and_any_other_as_an_error() {
        assert_eq!(RetCode::from_raw(scip_sys::SCIP_Retcode_SCIP_OKAY, "SCIPcall"), Ok(None));
        let codes = [
            (scip_sys::SCIP_Retcode_SCIP_ERROR, RetCode::Error),
            (scip_sys::SCIP_Retcode_SCIP_NOMEMORY, RetCode::NoMemory),
            (scip_sys::SCIP_Retcode_SCIP_READERROR, RetCode::ReadError),
            (scip_sys::SCIP_Retcode_SCIP_WRITEERROR, RetCode::WriteError),
            (scip_sys::SCIP_Retcode_SCIP_NOFILE, RetCode::NoFile),
            (scip_sys::SCIP_Retcode_SCIP_FILECREATEERROR, RetCode::FileCreateError),
            (scip_sys::SCIP_Retcode_SCIP_LPERROR, RetCode::LpError),
            (scip_sys::SCIP_Retcode_SCIP_NOPROBLEM, RetCode::NoProblem),
            (scip_sys::SCIP_Retcode_SCIP_INVALIDCALL, RetCode::InvalidCall),
            (scip_sys::SCIP_Retcode_SCIP_INVALIDDATA, RetCode::InvalidData),
            (scip_sys::SCIP_Retcode_SCIP_INVALIDRESULT, RetCode::InvalidResult),
            (scip_sys::SCIP_Retcode_SCIP_PLUGINNOTFOUND, RetCode::PluginNotFound),
            (scip_sys::SCIP_Retcode_SCIP_PARAMETERUNKNOWN, RetCode::ParameterUnknown),
            (
                scip_sys::SCIP_Retcode_SCIP_PARAMETERWRONGTYPE,
                RetCode::ParameterWrongType,
            ),
            (
                scip_sys::SCIP_Retcode_SCIP_PARAMETERWRONGVAL,
                RetCode::ParameterWrongValue,
            ),
            (
                scip_sys::SCIP_Retcode_SCIP_KEYALREADYEXISTING,
                RetCode::KeyAlreadyExisting,
            ),
            (scip_sys::SCIP_Retcode_SCIP_MAXDEPTHLEVEL, RetCode::MaxDepthLevel),
            (scip_sys::SCIP_Retcode_SCIP_BRANCHERROR, RetCode::BranchError),
            (scip_sys::SCIP_Retcode_SCIP_NOTIMPLEMENTED, RetCode::NotImplemented),
        ];
        for (raw, code) in codes {
            assert_eq!(RetCode::from_raw(raw, "SCIPcall"), Ok(Some(code)), "{raw}");
            assert_eq!(check("SCIPcall", raw), Err(ScipError::Call { call: "SCIPcall", code }));
        }
        for raw in [-99, -19, 2, i32::MIN] {
            let unlisted = ScipError::Unlisted {
                code: ScipCode::RetCode,
                raw: i64::from(raw),
                call: "SCIPcall",
            };
            assert_eq!(RetCode::from_raw(raw, "SCIPcall"), Err(unlisted.clone()));
            assert_eq!(check("SCIPcall", raw), Err(unlisted));
        }
        assert_eq!(check("SCIPcall", scip_sys::SCIP_Retcode_SCIP_OKAY), Ok(()));
    }

    #[test]
    fn test_a_failed_call_names_the_call_and_has_the_code_as_its_source() {
        use core::error::Error as _;
        let error = ScipError::Call {
            call: "SCIPsolve",
            code: RetCode::LpError,
        };
        assert_eq!(error.to_string(), "SCIP: SCIPsolve failed: error in LP solver");
        assert_eq!(
            error.source().map(ToString::to_string),
            Some("error in LP solver".to_owned())
        );
        assert!(ScipError::NotLoaded.source().is_none());
    }
}
