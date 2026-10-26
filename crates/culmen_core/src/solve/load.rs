// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why an LP could not be loaded into a backend.

use core::fmt;

use crate::{ModelClass, ModelClasses};

/// Why [`Solver::new`](crate::Solver::new) could not load an LP: the backend does not solve its class, or the
/// backend refused it with its own error `E`.
///
/// The class check is the same for every backend: it reads [`Backend::CLASSES`](crate::Backend::CLASSES) before
/// the backend sees the LP.
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
    /// The backend does not solve the LP's class.
    Unsupported {
        /// The LP's class.
        class: ModelClass,
        /// The classes the backend solves.
        supported: ModelClasses,
    },
    /// The backend refused the LP.
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for LoadError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { class, supported } => {
                write!(f, "Solver: the backend solves {supported}, not a {class}")
            }
            Self::Backend(error) => write!(f, "Solver: the backend refused the LP: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for LoadError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Unsupported { .. } => None,
            Self::Backend(error) => Some(error),
        }
    }
}
