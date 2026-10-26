// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why a program could not be loaded into a backend.

use core::fmt;

use crate::{ModelClass, ModelClasses, ProgramFeature, ProgramFeatures};

/// Why [`Solver::new`](crate::Solver::new) could not load a program: the backend does not solve its class, does not
/// take a feature it uses, or refused it with its own error `E`.
///
/// The class and feature checks are the same for every backend: they read [`Backend::CLASSES`](crate::Backend::CLASSES)
/// and [`Backend::FEATURES`](crate::Backend::FEATURES) before the backend sees the program.
///
/// ```
/// use culmen_core::{LoadError, Model, ModelClass, ModelClasses, Script, Solver, Var};
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous());
/// m.minimize(x * x);
/// let error = LoadError::<std::io::Error>::Unsupported {
///     class: m.build()?.class(),
///     supported: ModelClasses::of(&[ModelClass::Lp, ModelClass::Milp]),
/// };
/// assert_eq!(error.to_string(), "Solver: the backend solves {LP, MILP}, not a QP");
/// assert!(Solver::new(Script::optimal(0.0, vec![0.0]), m.build()?).is_ok(), "a script replays any class");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadError<E> {
    /// The backend does not solve the program's class.
    Unsupported {
        /// The program's class.
        class: ModelClass,
        /// The classes the backend solves.
        supported: ModelClasses,
    },
    /// The backend does not take a feature the program uses.
    UnsupportedFeature {
        /// The first feature the program uses that the backend does not take, in the order of [`ProgramFeature`].
        feature: ProgramFeature,
        /// The features the backend takes.
        supported: ProgramFeatures,
    },
    /// The backend refused the program.
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for LoadError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { class, supported } => {
                write!(f, "Solver: the backend solves {supported}, not a {class}")
            }
            Self::UnsupportedFeature { feature, supported } => {
                write!(f, "Solver: the backend takes {supported}, not {feature}")
            }
            Self::Backend(error) => write!(f, "Solver: the backend refused the program: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for LoadError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Unsupported { .. } | Self::UnsupportedFeature { .. } => None,
            Self::Backend(error) => Some(error),
        }
    }
}
