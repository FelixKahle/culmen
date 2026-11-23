// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The SCIP backend of culmen, on SCIP 10 built from source through `scip-sys`.

use core::fmt;

/// A SCIP version, ordered as versions are; displayed as `major.minor.patch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Version {
    /// The major version.
    pub major: u32,
    /// The minor version.
    pub minor: u32,
    /// The patch version, SCIP's technical version.
    pub patch: u32,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The version of the SCIP library linked in.
///
/// ```
/// let version = culmen_scip::version();
/// assert_eq!(version.major, 10);
/// assert_eq!(version.to_string(), format!("10.{}.{}", version.minor, version.patch));
/// ```
#[must_use]
pub fn version() -> Version {
    // SAFETY: the version getters take no arguments and return plain integers.
    let (major, minor, patch) = unsafe {
        (
            scip_sys::SCIPmajorVersion(),
            scip_sys::SCIPminorVersion(),
            scip_sys::SCIPtechVersion(),
        )
    };
    // SCIP's version numbers are never negative, so their magnitudes are the numbers.
    Version {
        major: major.unsigned_abs(),
        minor: minor.unsigned_abs(),
        patch: patch.unsigned_abs(),
    }
}
