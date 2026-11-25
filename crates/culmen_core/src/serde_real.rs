// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The serde form of an `f64` that may be infinite or NaN, for `#[serde(with = "culmen_core::serde_real")]`: a
//! finite value is a number, and the others are the strings `"inf"`, `"-inf"` and `"NaN"`, since JSON has no number
//! for them and `serde_json` writes each as `null`, which reads back as no `f64` at all.
//!
//! [`option`] is the same form for an `Option<f64>`, with `None` as the format's none.
//!
//! ```
//! #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
//! struct Reach {
//!     #[serde(with = "culmen_core::serde_real")]
//!     value: f64,
//! }
//!
//! let json = serde_json::to_string(&Reach { value: f64::NEG_INFINITY }).unwrap();
//! assert_eq!(json, r#"{"value":"-inf"}"#);
//! assert_eq!(serde_json::from_str::<Reach>(&json).unwrap(), Reach { value: f64::NEG_INFINITY });
//! assert_eq!(serde_json::from_str::<Reach>(r#"{"value":2}"#).unwrap(), Reach { value: 2.0 });
//! ```

use core::fmt;

use serde::de::{self, Visitor};
use serde::{Deserializer, Serializer};

/// The string of `+∞`.
const INFINITY: &str = "inf";
/// The string of `-∞`.
const NEG_INFINITY: &str = "-inf";
/// The string of NaN.
const NAN: &str = "NaN";

/// Writes `value` as a number when it is finite, else as its string.
///
/// # Errors
///
/// The serializer's own.
pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if value.is_finite() {
        serializer.serialize_f64(*value)
    } else if value.is_nan() {
        serializer.serialize_str(NAN)
    } else if value.is_sign_positive() {
        serializer.serialize_str(INFINITY)
    } else {
        serializer.serialize_str(NEG_INFINITY)
    }
}

/// Reads a number, or one of the strings `"inf"`, `"-inf"` and `"NaN"`.
///
/// # Errors
///
/// Any other string or kind of value.
pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    deserializer.deserialize_any(RealVisitor)
}

/// The form of [`serialize`] and [`deserialize`] for an `Option<f64>`.
pub mod option {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// An `f64` in the form of [`super::serialize`].
    #[derive(Serialize, Deserialize)]
    #[serde(transparent)]
    struct Real(#[serde(with = "super")] f64);

    /// Writes `None` as the format's none, and a value as [`super::serialize`] does.
    ///
    /// # Errors
    ///
    /// The serializer's own.
    pub fn serialize<S: Serializer>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error> {
        value.map(Real).serialize(serializer)
    }

    /// Reads the format's none as `None`, and a value as [`super::deserialize`] does.
    ///
    /// # Errors
    ///
    /// A value [`super::deserialize`] refuses.
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
        Ok(Option::<Real>::deserialize(deserializer)?.map(|Real(value)| value))
    }
}

/// Reads the form of [`serialize`].
struct RealVisitor;

impl Visitor<'_> for RealVisitor {
    type Value = f64;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a number, or one of \"{INFINITY}\", \"{NEG_INFINITY}\" and \"{NAN}\""
        )
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<f64, E> {
        Ok(value)
    }

    // An integer in a document stands for the nearest `f64`.
    fn visit_i64<E: de::Error>(self, value: i64) -> Result<f64, E> {
        Ok(value as f64)
    }

    // An integer in a document stands for the nearest `f64`.
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<f64, E> {
        match value {
            INFINITY => Ok(f64::INFINITY),
            NEG_INFINITY => Ok(f64::NEG_INFINITY),
            NAN => Ok(f64::NAN),
            other => Err(E::invalid_value(de::Unexpected::Str(other), &self)),
        }
    }
}
