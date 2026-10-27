// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The guarantee no test can show at run time: checked values are only made by their checked constructors.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

#[test]
fn test_checked_values_cannot_be_forged() {
    trybuild::TestCases::new().compile_fail("tests/compile_fail/forge.rs");
}
