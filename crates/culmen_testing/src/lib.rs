// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared test infrastructure of culmen, used only as a dev-dependency: an exact reference solver for tiny
//! problems, property-test strategies that generate them, and harder instances for the solver tests.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

pub mod instances;
pub mod reference;
pub mod strategy;
