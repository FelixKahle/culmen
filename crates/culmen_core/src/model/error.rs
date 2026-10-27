// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why a model could not be built, naming what is at fault as the model names it.

use core::fmt;

use crate::{Bounds, Col, InvalidBounds, ObjectiveId, Row, Sense};

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
    /// A further objective.
    Further(ObjectiveId),
}

impl fmt::Display for ExprPlace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Objective => f.write_str("the objective"),
            Self::Constraint(constraint) => constraint.fmt(f),
            Self::Further(objective) => write!(f, "the further objective {objective:?}"),
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
    /// A quadratic objective coefficient is NaN or infinite, as given or added up over its pair of variables.
    ObjectiveQuadCoefficient {
        /// The first variable of the product.
        first: ModelVar,
        /// The second variable of the product.
        second: ModelVar,
        /// The coefficient.
        value: f64,
    },
    /// The quadratic objective is not convex to minimize, or not concave to maximize, within
    /// [`Program::CONVEXITY_TOLERANCE`](crate::Program::CONVEXITY_TOLERANCE).
    NonConvexObjective {
        /// The objective's sense.
        sense: Sense,
        /// The first variable of the first block of the objective's Hessian that fails the check.
        var: ModelVar,
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
    /// A semi variable's bounds are not `lower ..= upper` with `0 < lower` and `upper` finite.
    SemiBounds {
        /// The variable.
        var: ModelVar,
        /// Its bounds.
        bounds: Bounds,
    },
    /// A further objective's coefficient is NaN or infinite, as given or added up.
    FurtherCoefficient {
        /// The objective.
        objective: ObjectiveId,
        /// The variable of the coefficient.
        var: ModelVar,
        /// The coefficient.
        value: f64,
    },
    /// A further objective's constant is NaN or infinite.
    FurtherConstant {
        /// The objective.
        objective: ObjectiveId,
        /// The constant.
        value: f64,
    },
    /// The model's own objective has a quadratic term beside further objectives, which need it linear.
    QuadraticWithObjectives,
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
            Self::ObjectiveQuadCoefficient { first, second, value } => {
                write!(
                    f,
                    "Model: the objective coefficient {value} of {first} · {second} is not finite"
                )
            }
            Self::NonConvexObjective {
                sense: Sense::Minimize,
                var,
            } => write!(
                f,
                "Model: the objective is not convex, as minimizing needs, in the block of variable {var}"
            ),
            Self::NonConvexObjective {
                sense: Sense::Maximize,
                var,
            } => write!(
                f,
                "Model: the objective is not concave, as maximizing needs, in the block of variable {var}"
            ),
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
            Self::SemiBounds { var, bounds } => write!(
                f,
                "Model: semi variable {var} has the bounds {bounds}, not 0 < lower <= upper < +inf"
            ),
            Self::FurtherCoefficient { objective, var, value } => write!(
                f,
                "Model: the coefficient {value} of variable {var} in the further objective {objective:?} is not finite"
            ),
            Self::FurtherConstant { objective, value } => write!(
                f,
                "Model: the constant {value} of the further objective {objective:?} is not finite"
            ),
            Self::QuadraticWithObjectives => {
                write!(f, "Model: the objective has a quadratic term beside further objectives")
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
