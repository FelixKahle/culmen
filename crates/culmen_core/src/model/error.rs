// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why a model could not be built, naming what is at fault as the model names it.

use core::fmt;

use crate::{Col, InvalidBounds, Row};

/// A variable of a [`Model`](crate::Model): its column and the name the model gave it.
///
/// Displayed as its name in backticks, or as its column when it has none.
///
/// ```
/// use culmen_core::{Col, ModelVar};
/// let named = ModelVar { col: Col::new(0), name: Some("stock".to_owned()) };
/// assert_eq!(named.to_string(), "`stock`");
/// assert_eq!(ModelVar { col: Col::new(3), name: None }.to_string(), "Col(3)");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelVar {
    /// The variable's column.
    pub col: Col,
    /// The variable's name; `None` when the model gave it none.
    pub name: Option<String>,
}

impl fmt::Display for ModelVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(name) => write!(f, "`{name}`"),
            None => write!(f, "{:?}", self.col),
        }
    }
}

/// A constraint of a [`Model`](crate::Model): its row and the name the model gave it.
///
/// Displayed as its name in backticks, or as its row when it has none.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelConstraint {
    /// The constraint's row.
    pub row: Row,
    /// The constraint's name; `None` when the model gave it none.
    pub name: Option<String>,
}

impl fmt::Display for ModelConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(name) => write!(f, "`{name}`"),
            None => write!(f, "{:?}", self.row),
        }
    }
}

/// Where in a [`Model`](crate::Model) an expression stands.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprPlace {
    /// The objective.
    Objective,
    /// A constraint.
    Constraint(ModelConstraint),
}

impl fmt::Display for ExprPlace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Objective => f.write_str("the objective"),
            Self::Constraint(constraint) => constraint.fmt(f),
        }
    }
}

/// Why a [`Model`](crate::Model) could not be built; each names the variable or constraint at fault.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelError {
    /// A variable's bounds are invalid.
    VarBounds {
        /// The variable.
        var: ModelVar,
        /// What is wrong with its bounds.
        source: InvalidBounds,
    },
    /// A variable's objective coefficient is NaN or infinite, as given or added up.
    ObjectiveCoefficient {
        /// The variable.
        var: ModelVar,
        /// The coefficient.
        value: f64,
    },
    /// The objective's constant is NaN or infinite.
    ObjectiveConstant {
        /// The constant.
        value: f64,
    },
    /// A constraint's coefficient is NaN or infinite, as given or added up.
    Coefficient {
        /// The constraint.
        constraint: ModelConstraint,
        /// The variable of the coefficient.
        var: ModelVar,
        /// The coefficient.
        value: f64,
    },
    /// A constraint's bounds, after its constant moved into them, are invalid.
    ConstraintBounds {
        /// The constraint.
        constraint: ModelConstraint,
        /// What is wrong with its bounds.
        source: InvalidBounds,
    },
    /// An expression names a column this model did not hand out.
    UnknownVar {
        /// Where the expression stands.
        place: ExprPlace,
        /// The column named.
        col: Col,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VarBounds { var, source } => write!(f, "Model: variable {var} has invalid bounds: {source}"),
            Self::ObjectiveCoefficient { var, value } => {
                write!(
                    f,
                    "Model: the objective coefficient {value} of variable {var} is not finite"
                )
            }
            Self::ObjectiveConstant { value } => write!(f, "Model: the objective constant {value} is not finite"),
            Self::Coefficient { constraint, var, value } => {
                write!(
                    f,
                    "Model: constraint {constraint} has the non-finite coefficient {value} on variable {var}"
                )
            }
            Self::ConstraintBounds { constraint, source } => {
                write!(f, "Model: constraint {constraint} has invalid bounds: {source}")
            }
            Self::UnknownVar { place, col } => {
                write!(f, "Model: {place} names {col:?}, which this model did not hand out")
            }
        }
    }
}

impl core::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::VarBounds { source, .. } | Self::ConstraintBounds { source, .. } => Some(source),
            _ => None,
        }
    }
}
