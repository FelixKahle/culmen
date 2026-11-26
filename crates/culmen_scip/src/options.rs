// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The options every solve applies: SCIP's emphasis settings and parameters set by SCIP's names.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The options a [`Scip`](crate::Scip) backend applies before every solve, after resetting SCIP's parameters and
/// before the solve's [`Settings`](culmen_core::Settings).
///
/// Plain data: each emphasis left `None` keeps SCIP's default, and [`ScipOptions::params`] are set in order after the
/// emphases, so a parameter set by name overrides what an emphasis chose for it.
///
/// ```
/// use culmen_scip::{Emphasis, ScipOptions};
///
/// let options = ScipOptions { heuristics: Some(Emphasis::Off), ..ScipOptions::default() }
///     .param("limits/maxorigsol", 5)
///     .param("lp/initalgorithm", 'd');
/// assert_eq!(options.params.len(), 2);
/// assert_eq!(options.enforcement_rounds, 10_000);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScipOptions {
    /// How hard SCIP's primal heuristics work (`SCIPsetHeuristics`); `None` keeps SCIP's defaults.
    pub heuristics: Option<Emphasis>,
    /// How hard SCIP presolves (`SCIPsetPresolving`); `None` keeps SCIP's defaults. The solve's
    /// [`Settings::presolve`](culmen_core::Settings::presolve) overrides it.
    pub presolving: Option<Emphasis>,
    /// How hard SCIP separates cutting planes (`SCIPsetSeparating`); `None` keeps SCIP's defaults.
    pub separating: Option<Emphasis>,
    /// Parameters set by SCIP's names, in order, after the emphases.
    pub params: Vec<ScipParam>,
    /// The most times a [`ConstraintHandler`](culmen_core::ConstraintHandler) is asked to enforce at one node before
    /// the solve fails with [`ScipError::EnforcementStalled`](crate::ScipError::EnforcementStalled); 10 000 by
    /// default. A handler that resolves a point by adding constraints is asked again at the node each time, so the
    /// limit bounds a handler that keeps acting without resolving anything.
    pub enforcement_rounds: u64,
}

/// SCIP's defaults, no parameter, and 10 000 enforcement rounds.
impl Default for ScipOptions {
    fn default() -> Self {
        Self {
            heuristics: None,
            presolving: None,
            separating: None,
            params: Vec::new(),
            enforcement_rounds: 10_000,
        }
    }
}

impl ScipOptions {
    /// Adds a parameter by SCIP's name, set after the emphases and every earlier parameter.
    #[must_use]
    pub fn param(mut self, name: impl Into<String>, value: impl Into<ParamValue>) -> Self {
        self.params.push(ScipParam {
            name: name.into(),
            value: value.into(),
        });
        self
    }
}

/// One of SCIP's emphasis settings for a group of plugins: SCIP's `SCIP_PARAMSETTING`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Emphasis {
    /// SCIP's default parameters for the group.
    Default,
    /// More effort than the default.
    Aggressive,
    /// Less effort than the default.
    Fast,
    /// The group switched off.
    Off,
}

impl Emphasis {
    /// SCIP's `SCIP_PARAMSETTING` value of the emphasis.
    pub(crate) const fn raw(self) -> scip_sys::SCIP_PARAMSETTING {
        match self {
            Self::Default => scip_sys::SCIP_ParamSetting_SCIP_PARAMSETTING_DEFAULT,
            Self::Aggressive => scip_sys::SCIP_ParamSetting_SCIP_PARAMSETTING_AGGRESSIVE,
            Self::Fast => scip_sys::SCIP_ParamSetting_SCIP_PARAMSETTING_FAST,
            Self::Off => scip_sys::SCIP_ParamSetting_SCIP_PARAMSETTING_OFF,
        }
    }
}

/// A parameter set by SCIP's name.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScipParam {
    /// SCIP's name of the parameter, such as `limits/maxorigsol` (an `int` parameter) or `limits/nodes` (a
    /// `SCIP_Longint` one, set with an `i64` value).
    pub name: String,
    /// The value set.
    pub value: ParamValue,
}

/// A parameter value of one of SCIP's six types.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ParamValue {
    /// A `SCIP_Bool` parameter.
    Bool(bool),
    /// An `int` parameter.
    Int(i32),
    /// A `SCIP_Longint` parameter.
    Longint(i64),
    /// A `SCIP_Real` parameter; written in the form of [`culmen_core::serde_real`], so an infinity reads back.
    Real(#[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real"))] f64),
    /// A `char` parameter; SCIP takes only ASCII characters.
    Char(char),
    /// A string parameter.
    String(String),
}

impl fmt::Display for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Longint(value) => write!(f, "{value}"),
            Self::Real(value) => write!(f, "{value}"),
            Self::Char(value) => write!(f, "{value:?}"),
            Self::String(value) => write!(f, "{value:?}"),
        }
    }
}

impl From<bool> for ParamValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i32> for ParamValue {
    fn from(value: i32) -> Self {
        Self::Int(value)
    }
}

impl From<i64> for ParamValue {
    fn from(value: i64) -> Self {
        Self::Longint(value)
    }
}

impl From<f64> for ParamValue {
    fn from(value: f64) -> Self {
        Self::Real(value)
    }
}

impl From<char> for ParamValue {
    fn from(value: char) -> Self {
        Self::Char(value)
    }
}

impl From<&str> for ParamValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<String> for ParamValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}
