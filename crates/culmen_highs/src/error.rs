// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why the HiGHS backend refused or failed.

use core::fmt;

use crate::{ModelStatus, OptionValue};

/// Why loading or solving with HiGHS failed.
#[derive(Clone, Debug, PartialEq)]
pub enum HighsError {
    /// A solve was asked for before an LP was loaded.
    NotLoaded,
    /// The LP has more columns, rows or entries than HiGHS's 32-bit indices hold.
    TooLarge {
        /// What overflowed.
        what: Dimension,
        /// How many there are.
        count: usize,
    },
    /// HiGHS refused the LP.
    Load {
        /// HiGHS's last error line, if it wrote one.
        message: Option<String>,
    },
    /// HiGHS refused an option.
    Option {
        /// The option's name.
        name: String,
        /// The value refused.
        value: OptionValue,
    },
    /// An exact integer setting, such as a seed or a thread count, exceeds HiGHS's largest integer, `i32::MAX`.
    OutOfRange {
        /// HiGHS's name of the option.
        name: &'static str,
        /// The value given.
        value: u64,
    },
    /// An option name holds a NUL byte, which HiGHS's C strings cannot carry.
    OptionName {
        /// The name.
        name: String,
    },
    /// A string option's value holds a NUL byte, which HiGHS's C strings cannot carry.
    OptionString {
        /// The option's name.
        name: String,
    },
    /// The run failed, as a thread whose HiGHS scheduler runs at another width than `threads` asks makes it.
    Run {
        /// The model status at the failure.
        status: ModelStatus,
        /// HiGHS's last error line, if it wrote one.
        message: Option<String>,
    },
}

impl fmt::Display for HighsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = |m: &Option<String>| m.as_deref().map(|m| format!(": {m}")).unwrap_or_default();
        match self {
            Self::NotLoaded => write!(f, "HiGHS: no LP is loaded"),
            Self::TooLarge { what, count } => write!(f, "HiGHS: {count} {what} exceed HiGHS's 32-bit indices"),
            Self::Load { message: m } => write!(f, "HiGHS: the LP was refused{}", message(m)),
            Self::Option { name, value } => write!(f, "HiGHS: option `{name}` refused the value {value}"),
            Self::OutOfRange { name, value } => {
                write!(f, "HiGHS: option `{name}` takes at most {}, not {value}", i32::MAX)
            }
            Self::OptionName { name } => write!(f, "HiGHS: option name {name:?} holds a NUL byte"),
            Self::OptionString { name } => write!(f, "HiGHS: the value of option `{name}` holds a NUL byte"),
            Self::Run { status, message: m } => write!(f, "HiGHS: the run failed ({status}){}", message(m)),
        }
    }
}

impl core::error::Error for HighsError {}

/// What of an LP HiGHS counts in its 32-bit indices.
///
/// ```
/// use culmen_highs::Dimension;
/// assert_eq!(Dimension::Entries.to_string(), "entries");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dimension {
    /// The columns.
    Columns,
    /// The rows.
    Rows,
    /// The matrix's nonzero entries.
    Entries,
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Columns => "columns",
            Self::Rows => "rows",
            Self::Entries => "entries",
        })
    }
}
