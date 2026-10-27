// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model files: a [`Program`] read from and written to MPS and LP files by HiGHS's own readers and writers.
//!
//! [`read`] and [`write()`] pick the format by the path's extension, `.mps` or `.lp` in any case; [`read_as`] and
//! [`write_as`] take it explicitly, whatever the extension. What is read is checked as any program is, and a program
//! is written only when the format, as HiGHS writes it, keeps all of it.
//!
//! ```
//! use culmen_core::{Linear, Model, Var};
//! use culmen_highs::files::{self, ModelFormat};
//!
//! let mut m = Model::new();
//! let x = m.add_var(Var::integer().upper(4).name("x"));
//! let y = m.add_var(Var::continuous().upper(3).name("y"));
//! m.add_named_constraint("cap", (x + 2 * y).leq(6));
//! m.maximize(3 * x + 2 * y);
//! let program = m.build()?;
//!
//! let path = std::env::temp_dir().join(format!("culmen-doc-{}.mps", std::process::id()));
//! files::write(&program, &path)?;
//! let read = files::read_as(&path, ModelFormat::Mps)?;
//! std::fs::remove_file(&path)?;
//! assert_eq!((read.num_cols(), read.num_rows()), (2, 1));
//! assert_eq!(read.row(culmen_core::Row::new(0)).name(), Some("cap"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::ffi::CString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use culmen_core::{
    Bounds, Col, Integrality, InvalidBounds, Program, ProgramColumn, ProgramError, ProgramFeature, ProgramPlan,
    ProgramRow, Row, Sense, TiVec,
};
use highs_sys::{HighsInt, kHighsMaximumStringLength, kHighsObjSenseMaximize};

use crate::instance::{Instance, Line, Padding, RawModel, var_type};
use crate::{CallStatus, HighsError};

/// A model file format HiGHS reads and writes.
///
/// ```
/// use culmen_highs::files::ModelFormat;
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

/// Reads the model file at `path` in the format its extension names.
///
/// # Errors
///
/// [`FileError::Extension`] when the extension names no format; otherwise as [`read_as`].
pub fn read(path: impl AsRef<Path>) -> Result<Program, FileError> {
    let path = path.as_ref();
    let format = ModelFormat::of_path(path).ok_or_else(|| FileError::Extension { path: path.to_owned() })?;
    read_as(path, format)
}

/// Reads the model file at `path` in `format`, whatever its extension, into a checked [`Program`].
///
/// Every column and row keeps the name the file gives it. HiGHS's readers take no further objectives, so the program
/// has one.
///
/// # Errors
///
/// [`FileError::Path`] for a path HiGHS cannot take, [`FileError::Io`] when the file cannot be read,
/// [`FileError::Read`] when HiGHS cannot parse it; [`FileError::ColumnBounds`], [`FileError::RowBounds`],
/// [`FileError::VarType`] or [`FileError::Program`] when its content is no valid program, such as a non-convex
/// quadratic objective.
pub fn read_as(path: impl AsRef<Path>, format: ModelFormat) -> Result<Program, FileError> {
    let path = path.as_ref();
    let staged = Staged::to_read(path, format)?;
    let length = fs::metadata(staged.path()).map_err(|e| FileError::io(path, &e))?.len();
    let mut instance = Instance::new();
    if instance.read_model(&c_path(staged.path())?) == CallStatus::Error {
        return Err(FileError::Read {
            path: path.to_owned(),
            format,
        });
    }
    let model = instance.model();
    // A name HiGHS read from the file is no longer than the file, and one it made up for an unnamed line is
    // shorter than its own name buffers, `NAME_LENGTH`; the buffer is longer than both.
    let mut buffer = vec![
        0u8;
        usize::try_from(length)
            .unwrap_or(usize::MAX)
            .saturating_add(1)
            .max(NAME_LENGTH)
    ];
    // SAFETY: the buffer is longer than the file and than HiGHS's own name buffers, so longer than any name HiGHS
    // holds.
    let (col_names, row_names) = unsafe {
        (
            names(&instance, Line::Col, model.costs.len(), &mut buffer),
            names(&instance, Line::Row, model.row_lower.len(), &mut buffer),
        )
    };
    program_of(&model, col_names, row_names)
}

/// The length of HiGHS's own name buffers, `kHighsMaximumStringLength`; the least buffer a name is read into.
const NAME_LENGTH: usize = kHighsMaximumStringLength as usize;

/// Writes `program` to the model file at `path` in the format its extension names.
///
/// # Errors
///
/// [`FileError::Extension`] when the extension names no format; otherwise as [`write_as`].
pub fn write(program: &Program, path: impl AsRef<Path>) -> Result<(), FileError> {
    let path = path.as_ref();
    let format = ModelFormat::of_path(path).ok_or_else(|| FileError::Extension { path: path.to_owned() })?;
    write_as(program, path, format)
}

/// Writes `program` to the model file at `path` in `format`, whatever its extension, replacing any file there.
///
/// Columns and rows keep their names where the format allows them; HiGHS names the others. Reading the file back
/// gives the same program up to names and the order of columns and rows.
///
/// # Errors
///
/// [`FileError::Unsupported`] for a program the format, as HiGHS writes it, loses part of: further objectives in
/// either format, a semi-integer column in an LP file (HiGHS lists it as semi-continuous only), and a semi column
/// with equal bounds in an MPS file (HiGHS writes it fixed); [`FileError::Name`] for a name holding a NUL byte;
/// [`FileError::Highs`] when HiGHS refuses the program, [`FileError::Path`] for a path HiGHS cannot take,
/// [`FileError::Write`] when HiGHS cannot write the file and [`FileError::Io`] when the file cannot be moved into
/// place.
pub fn write_as(program: &Program, path: impl AsRef<Path>, format: ModelFormat) -> Result<(), FileError> {
    let path = path.as_ref();
    writable(program, format)?;
    let mut instance = Instance::new();
    instance.pass(program, Padding::Exact).map_err(FileError::Highs)?;
    pass_names(&mut instance, program)?;
    let staged = Staged::to_write(path, format);
    if instance.write_model(&c_path(staged.path())?) == CallStatus::Error {
        return Err(FileError::Write {
            path: path.to_owned(),
            format,
        });
    }
    staged.commit(path)
}

/// Refuses a program `format`, as HiGHS writes it, loses part of.
///
/// # Errors
///
/// [`FileError::Unsupported`] as [`write_as`] states.
fn writable(program: &Program, format: ModelFormat) -> Result<(), FileError> {
    let unsupported = |feature| FileError::Unsupported { format, feature };
    if program.objectives().are_several() {
        return Err(unsupported(ProgramFeature::MultipleObjectives));
    }
    let columns = program.columns();
    let lost = match format {
        ModelFormat::Lp => columns
            .iter()
            .find(|c| c.integrality == Integrality::SemiInteger)
            .map(|c| c.integrality),
        ModelFormat::Mps => columns
            .iter()
            .find(|c| c.integrality.is_semi() && c.bounds.lower() == c.bounds.upper())
            .map(|c| c.integrality),
    };
    match lost {
        Some(Integrality::SemiInteger) => Err(unsupported(ProgramFeature::SemiInteger)),
        Some(_) => Err(unsupported(ProgramFeature::SemiContinuous)),
        None => Ok(()),
    }
}

/// Passes every column's and row's name to HiGHS; an unnamed one stays blank for HiGHS to name.
///
/// # Errors
///
/// [`FileError::Name`] for a name holding a NUL byte, [`FileError::Highs`] when HiGHS refuses one.
fn pass_names(instance: &mut Instance, program: &Program) -> Result<(), FileError> {
    let cols = program.columns().iter().map(|c| (Line::Col, c.name.as_deref()));
    let rows = program.rows().map(|row| (Line::Row, program.row(row).name()));
    let lines = cols
        .enumerate()
        .chain(rows.enumerate())
        .map(|(i, (line, name))| (line, i, name));
    lines
        .filter_map(|(line, i, name)| Some((line, i, name?)))
        .try_for_each(|(line, i, name)| {
            let c_name = CString::new(name).map_err(|_| FileError::Name { name: name.to_owned() })?;
            // `Instance::pass` accepted the program, so its counts fit HiGHS's index type.
            let index = HighsInt::try_from(i).unwrap_or(HighsInt::MAX);
            instance.pass_name(line, index, &c_name).map_err(FileError::Highs)
        })
}

/// The names of the first `count` columns or rows of the model `instance` holds, read through `buffer`.
///
/// # Safety
///
/// Every name of the model `instance` holds is shorter than `buffer`.
unsafe fn names(instance: &Instance, line: Line, count: usize, buffer: &mut [u8]) -> Vec<Option<String>> {
    (0..count)
        .map(|i| {
            // A count of the model HiGHS holds fits its index type.
            let index = HighsInt::try_from(i).unwrap_or(HighsInt::MAX);
            // SAFETY: the caller vouches that every name is shorter than the buffer.
            unsafe { instance.name(line, index, buffer) }
        })
        .collect()
}

/// The checked program of the model HiGHS read, its columns and rows named.
///
/// # Errors
///
/// [`FileError::ColumnBounds`], [`FileError::RowBounds`], [`FileError::VarType`] or [`FileError::Program`].
pub(crate) fn program_of(
    model: &RawModel,
    col_names: Vec<Option<String>>,
    row_names: Vec<Option<String>>,
) -> Result<Program, FileError> {
    let sense = if model.sense == kHighsObjSenseMaximize {
        Sense::Maximize
    } else {
        Sense::Minimize
    };
    let plan = ProgramPlan {
        sense,
        offset: model.offset,
        columns: columns_of(model, col_names)?,
        rows: rows_of(model, row_names)?,
        quadratic: quadratic_of(model),
        ..ProgramPlan::new()
    };
    plan.build().map_err(FileError::Program)
}

/// The model's columns as plan columns.
///
/// # Errors
///
/// [`FileError::ColumnBounds`] or [`FileError::VarType`] for the first column at fault.
fn columns_of(model: &RawModel, names: Vec<Option<String>>) -> Result<TiVec<Col, ProgramColumn>, FileError> {
    let parts = model
        .costs
        .iter()
        .zip(&model.lower)
        .zip(&model.upper)
        .zip(&model.integrality);
    parts
        .zip(names)
        .enumerate()
        .map(|(i, ((((&cost, &lower), &upper), &raw), name))| {
            let col = Col::new(i);
            let bounds = Bounds::new(lower, upper).map_err(|source| FileError::ColumnBounds { col, source })?;
            let integrality = integrality_of(raw).ok_or(FileError::VarType { col, raw })?;
            Ok(ProgramColumn {
                cost,
                bounds,
                integrality,
                name,
            })
        })
        .collect()
}

/// The integrality of HiGHS's variable type, `None` for one culmen has no integrality for.
fn integrality_of(raw: HighsInt) -> Option<Integrality> {
    [
        Integrality::Continuous,
        Integrality::Integer,
        Integrality::SemiContinuous,
        Integrality::SemiInteger,
    ]
    .into_iter()
    .find(|&integrality| var_type(integrality) == raw)
}

/// The model's rows as plan rows.
///
/// # Errors
///
/// [`FileError::RowBounds`] for the first row whose bounds are no interval.
fn rows_of(model: &RawModel, names: Vec<Option<String>>) -> Result<TiVec<Row, ProgramRow>, FileError> {
    let bounds = model.row_lower.iter().zip(&model.row_upper);
    let spans = model.starts.windows(2).map(|w| position(w[0])..position(w[1]));
    bounds
        .zip(spans)
        .zip(names)
        .enumerate()
        .map(|(i, (((&lower, &upper), span), name))| {
            let row = Row::new(i);
            let bounds = Bounds::new(lower, upper).map_err(|source| FileError::RowBounds { row, source })?;
            let terms = model.indices[span.clone()]
                .iter()
                .zip(&model.values[span])
                .map(|(&index, &value)| (Col::new(position(index)), value))
                .collect();
            Ok(ProgramRow { bounds, terms, name })
        })
        .collect()
}

/// The model's quadratic terms from the lower triangle of its Hessian `H` of `½ xᵀHx`: `(c, c, H/2)` on the diagonal,
/// `(c, r, H)` below it; an entry above it, which only a square Hessian holds beside its mirror, is left out.
fn quadratic_of(model: &RawModel) -> Vec<(Col, Col, f64)> {
    let spans = model.q_starts.windows(2).map(|w| position(w[0])..position(w[1]));
    spans
        .enumerate()
        .flat_map(|(c, span)| {
            let entries = model.q_indices[span.clone()].iter().zip(&model.q_values[span]);
            entries.filter_map(move |(&r, &value)| {
                let r = position(r);
                (r >= c).then(|| (Col::new(c), Col::new(r), if r == c { value / 2.0 } else { value }))
            })
        })
        .collect()
}

/// A HiGHS index as a position; a negative one, which HiGHS never writes, as one no program has.
fn position(index: HighsInt) -> usize {
    usize::try_from(index).unwrap_or(usize::MAX)
}

/// A path as the C string HiGHS takes.
///
/// # Errors
///
/// [`FileError::Path`] for a path that is not UTF-8 or holds a NUL byte.
fn c_path(path: &Path) -> Result<CString, FileError> {
    path.to_str()
        .and_then(|text| CString::new(text).ok())
        .ok_or_else(|| FileError::Path { path: path.to_owned() })
}

/// The file HiGHS reads or writes: the path itself when its extension names the format, else a temporary file with
/// the format's extension, which HiGHS picks its reader and writer by, removed when dropped.
#[derive(Debug)]
struct Staged {
    path: PathBuf,
    temporary: bool,
}

/// Numbers the temporary files of this process apart.
static STAGED: AtomicUsize = AtomicUsize::new(0);

impl Staged {
    /// The file to read `path` in `format` from: a temporary copy when the extension names another format.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`] when the copy fails.
    fn to_read(path: &Path, format: ModelFormat) -> Result<Self, FileError> {
        if ModelFormat::of_path(path) == Some(format) {
            return Ok(Self::direct(path));
        }
        let staged = Self::temporary(&std::env::temp_dir(), format);
        fs::copy(path, &staged.path).map_err(|e| FileError::io(path, &e))?;
        Ok(staged)
    }

    /// The file to write `path` in `format` to: a temporary file beside it when the extension names another format.
    fn to_write(path: &Path, format: ModelFormat) -> Self {
        if ModelFormat::of_path(path) == Some(format) {
            return Self::direct(path);
        }
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        Self::temporary(directory, format)
    }

    fn direct(path: &Path) -> Self {
        Self {
            path: path.to_owned(),
            temporary: false,
        }
    }

    /// A new temporary file's path in `directory`, with `format`'s extension.
    fn temporary(directory: &Path, format: ModelFormat) -> Self {
        let n = STAGED.fetch_add(1, Ordering::Relaxed);
        let name = format!(".culmen-{}-{n}.{}", std::process::id(), format.extension());
        Self {
            path: directory.join(name),
            temporary: true,
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    /// Moves a temporary file written in place of `path` there.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`] when the move fails.
    fn commit(mut self, path: &Path) -> Result<(), FileError> {
        if !self.temporary {
            return Ok(());
        }
        fs::rename(&self.path, path).map_err(|e| FileError::io(path, &e))?;
        self.temporary = false;
        Ok(())
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        if self.temporary {
            // A temporary file that cannot be removed is left behind; the read or write it served is unaffected.
            let _ = fs::remove_file(&self.path);
        }
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
    ColumnBounds {
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
    /// A column has a HiGHS variable type culmen has no [`Integrality`] for.
    VarType {
        /// The column, in the order HiGHS read them.
        col: Col,
        /// HiGHS's variable type.
        raw: HighsInt,
    },
    /// The file's content fails a program's checks.
    Program(ProgramError),
    /// The format, as HiGHS writes it, loses a feature the program uses.
    Unsupported {
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
    fn io(path: &Path, error: &io::Error) -> Self {
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
            Self::ColumnBounds { col, source } => write!(f, "model file: {col:?} has invalid bounds: {source}"),
            Self::RowBounds { row, source } => write!(f, "model file: {row:?} has invalid bounds: {source}"),
            Self::VarType { col, raw } => write!(f, "model file: {col:?} has the unknown variable type {raw}"),
            Self::Program(error) => write!(f, "model file: the content is no valid program: {error}"),
            Self::Unsupported { format, feature } => {
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
            Self::ColumnBounds { source, .. } | Self::RowBounds { source, .. } => Some(source),
            Self::Program(error) => Some(error),
            Self::Highs(error) => Some(error),
            _ => None,
        }
    }
}
