// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The types of [`files`](crate::files): the formats a model file is read and written in, and why a read or write
//! failed. They sit at the crate root with the backend's other types; the `files` module holds only its functions.

use core::fmt;
use std::io;
use std::path::{Path, PathBuf};

use culmen_core::{Col, InvalidBounds, ProgramError, ProgramFeature, Row};
use highs_sys::HighsInt;

use crate::HighsError;

/// A model file format HiGHS reads and writes.
///
/// ```
/// use culmen_highs::ModelFormat;
/// assert_eq!(ModelFormat::of_path("plant.MPS".as_ref()), Some(ModelFormat::Mps));
/// assert_eq!(ModelFormat::of_path("plant.lp".as_ref()), Some(ModelFormat::Lp));
/// assert_eq!(ModelFormat::of_path("plant.mps.gz".as_ref()), None);
/// assert_eq!(ModelFormat::Lp.to_string(), "LP");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ModelFormat {
    /// The MPS format, fixed or free, with HiGHS's `QUADOBJ` section for a quadratic objective.
    Mps,
    /// The CPLEX LP format.
    Lp,
}

impl ModelFormat {
    /// The format a path's extension names, `.mps` or `.lp` in any case; `None` for any other.
    #[must_use]
    pub fn of_path(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?;
        [Self::Mps, Self::Lp]
            .into_iter()
            .find(|format| extension.eq_ignore_ascii_case(format.extension()))
    }

    /// The extension HiGHS reads the format by, without its dot.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mps => "mps",
            Self::Lp => "lp",
        }
    }
}

/// `MPS` or `LP`.
impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mps => "MPS",
            Self::Lp => "LP",
        })
    }
}

/// Why a model file could not be read or written.
#[derive(Clone, Debug, PartialEq)]
pub enum FileError {
    /// The path's extension names no format: neither `.mps` nor `.lp`.
    Extension {
        /// The path.
        path: PathBuf,
    },
    /// The path is not UTF-8 or holds a NUL byte, so HiGHS cannot take it.
    Path {
        /// The path.
        path: PathBuf,
    },
    /// The file system failed on the path.
    Io {
        /// The path.
        path: PathBuf,
        /// What failed.
        kind: io::ErrorKind,
    },
    /// HiGHS could not read the file in the format: it is missing, unreadable or does not parse.
    Read {
        /// The path.
        path: PathBuf,
        /// The format it was read in.
        format: ModelFormat,
    },
    /// HiGHS could not write the file.
    Write {
        /// The path.
        path: PathBuf,
        /// The format it was written in.
        format: ModelFormat,
    },
    /// A column's bounds in the file are no interval.
    ColBounds {
        /// The column, in the order HiGHS read them.
        col: Col,
        /// What is wrong with its bounds.
        source: InvalidBounds,
    },
    /// A row's bounds in the file are no interval.
    RowBounds {
        /// The row, in the order HiGHS read them.
        row: Row,
        /// What is wrong with its bounds.
        source: InvalidBounds,
    },
    /// A column has a HiGHS variable type culmen has no [`Integrality`](culmen_core::Integrality) for.
    VarType {
        /// The column, in the order HiGHS read them.
        col: Col,
        /// HiGHS's variable type.
        raw: HighsInt,
    },
    /// The file's content fails a program's checks.
    Program(ProgramError),
    /// The format, as HiGHS writes it, loses a feature the program uses.
    UnsupportedFeature {
        /// The format.
        format: ModelFormat,
        /// The feature it loses.
        feature: ProgramFeature,
    },
    /// A name holds a NUL byte, which HiGHS's C strings cannot carry.
    Name {
        /// The name.
        name: String,
    },
    /// HiGHS refused the program.
    Highs(HighsError),
}

impl FileError {
    /// The error of an I/O failure on `path`.
    pub(crate) fn io(path: &Path, error: &io::Error) -> Self {
        Self::Io {
            path: path.to_owned(),
            kind: error.kind(),
        }
    }
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extension { path } => write!(f, "model file: {} names no format", path.display()),
            Self::Path { path } => write!(f, "model file: HiGHS cannot take the path {}", path.display()),
            Self::Io { path, kind } => write!(f, "model file: {} failed: {kind}", path.display()),
            Self::Read { path, format } => write!(f, "model file: HiGHS could not read {} as {format}", path.display()),
            Self::Write { path, format } => {
                write!(f, "model file: HiGHS could not write {} as {format}", path.display())
            }
            Self::ColBounds { col, source } => write!(f, "model file: {col:?} has invalid bounds: {source}"),
            Self::RowBounds { row, source } => write!(f, "model file: {row:?} has invalid bounds: {source}"),
            Self::VarType { col, raw } => write!(f, "model file: {col:?} has the unknown variable type {raw}"),
            Self::Program(error) => write!(f, "model file: the content is no valid program: {error}"),
            Self::UnsupportedFeature { format, feature } => {
                write!(f, "model file: HiGHS's {format} writer loses {feature}")
            }
            Self::Name { name } => write!(f, "model file: the name {name:?} holds a NUL byte"),
            Self::Highs(error) => write!(f, "model file: {error}"),
        }
    }
}

impl core::error::Error for FileError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::ColBounds { source, .. } | Self::RowBounds { source, .. } => Some(source),
            Self::Program(error) => Some(error),
            Self::Highs(error) => Some(error),
            _ => None,
        }
    }
}
