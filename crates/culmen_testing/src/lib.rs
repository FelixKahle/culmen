// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared test infrastructure of culmen, used only as a dev-dependency: an exact reference solver for tiny
//! problems, property-test strategies that generate them, harder instances for the solver tests, and for editing a
//! loaded program: edits as data, a reference model of their rules, and a recording backend that fails on demand.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

pub mod instances;
pub mod mirror;
pub mod op;
pub mod reference;
pub mod strategy;
