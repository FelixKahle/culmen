// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What only code that implements a backend, or drives one without a [`Solver`](crate::Solver), names: the hooks a
//! solve hands to [`Backend::solve`](crate::Backend::solve) and what a wrapping backend relays through them, the
//! edit [`Backend::edit`](crate::Backend::edit) receives, the sink of the columns a pricer adds, the stop and observer
//! of a solve as one callback, the checks every backend runs on a callback's actions, and the context of an event a
//! backend cannot deliver.
//!
//! [`Backend`](crate::Backend) itself, its capability traits ([`LazyConstraints`](crate::LazyConstraints),
//! [`Starts`](crate::Starts), [`DualAnalysis`](crate::DualAnalysis), …) and the context traits
//! ([`LazyContext`](crate::LazyContext), …) are at the crate root: a consumer names them in the bounds of generic
//! code (`impl<B: LazyConstraints> LazyHandler<B> for …`) and calls the contexts' methods, so they belong to the
//! consumer's API, and a backend author implements them from there.
//!
//! # Capabilities
//!
//! Each capability is a trait a backend implements or does not, and each method that needs one, a
//! [`Solver`](crate::Solver) method or a registration on [`Hooks`], exists only for a backend that has it, so asking
//! a backend for what it cannot do does not compile: a lazy-constraint handler registered for a backend without
//! [`LazyConstraints`](crate::LazyConstraints) is a compile error, not a solve that ignores it and reports the
//! relaxation's optimum. What depends on the program or on the solve (an analysis defined for the program's class, a
//! solve to read, a basis that fits, a column a pricer adds) stays a run-time refusal, checked the same way for every
//! backend.
//!
//! A backend with a search capability calls the registered handler through its [`Hooks`], only while
//! [`Backend::solve`](crate::Backend::solve) runs, never two handlers or a handler and a callback at once, and only
//! when the hooks have a handler of that kind ([`Hooks::wants_lazy`] and its siblings). A solve without a handler of a
//! kind behaves as if the backend lacked the capability, and gives up nothing for it.
//!
//! A [`Solver`](crate::Solver) calls an analysis method only for an analysis defined for the program's class
//! ([`Analysis::classes`](crate::Analysis::classes)) and features ([`Analysis::refused`](crate::Analysis::refused)),
//! with the loaded program, and only in the state its [`Solver`](crate::Solver) method documents:
//! [`DualAnalysis::duals`](crate::DualAnalysis::duals) and
//! [`RangingAnalysis::ranging`](crate::RangingAnalysis::ranging) after a solve that ended optimal,
//! [`BasisAnalysis::basis`](crate::BasisAnalysis::basis) after a solve, the rays after a solve that ended as they
//! prove, a basis or weights that fit the program. Called otherwise, it may fail or give an unspecified result, never
//! undefined behaviour.

pub use crate::solve::{
    AddedRow, Edit, Hooks, Priced, Relay, Unsupported, Watch, check_constraint, check_entries, check_solution,
};
