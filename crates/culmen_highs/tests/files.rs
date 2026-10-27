// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model files through HiGHS's readers and writers: round trips keep the program up to names and order and keep its
//! optimum; what a format loses is refused; what HiGHS cannot read is a typed error.
#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use culmen_core::{
    Bounds, Integrality, Linear, Model, Objective, Outcome, Program, ProgramError, ProgramFeature, Request, Sense,
    Solver, Var,
};
use culmen_highs::Highs;
use culmen_highs::files::{self, FileError, ModelFormat};
use culmen_testing::reference::close;
use culmen_testing::strategy::{small_lp, small_semi};
use proptest::prelude::*;

/// A path in the temporary directory, distinct for every call in this process; the file is removed on drop.
struct TempPath(PathBuf);

impl TempPath {
    fn new(extension: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!("culmen-files-{}-{n}.{extension}", std::process::id())))
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The optimal objective value HiGHS finds, `None` when the program is infeasible.
fn optimum(program: &Program) -> Option<f64> {
    let solved = Solver::new(Highs::new(), program.clone())
        .unwrap()
        .solve(Request::new())
        .unwrap();
    match solved.outcome {
        Outcome::Optimal(best) => Some(best.objective),
        Outcome::Infeasible | Outcome::InfeasibleOrUnbounded => None,
        other => panic!("{other:?}"),
    }
}

/// The program written in `format` and read back.
fn round_trip(program: &Program, format: ModelFormat) -> Program {
    let path = TempPath::new(format.extension());
    files::write(program, &path.0).unwrap();
    files::read(&path.0).unwrap()
}

/// Each named column's integrality, bounds and cost, by name.
fn columns_by_name(program: &Program) -> BTreeMap<String, (Integrality, Bounds, f64)> {
    program
        .columns()
        .iter()
        .map(|c| (c.name.clone().unwrap(), (c.integrality, c.bounds, c.cost)))
        .collect()
}

/// A MILP with every kind of column, a ranged row, a free column, an offset, maximized.
fn mixed() -> Program {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(-2, 5).name("x"));
    let y = m.add_var(Var::free().name("y"));
    let z = m.add_var(Var::semi_continuous().bounds(2, 10).name("z"));
    let w = m.add_var(Var::semi_integer().bounds(1, 7).name("w"));
    m.add_named_constraint("band", (x - y + z).between(-1, 4));
    m.add_named_constraint("floor", (x + w).geq(1.5));
    m.add_named_constraint("roof", (y + w).leq(9));
    m.maximize(3.0 * x + y - z + 2 * w + 1.5);
    m.build().unwrap()
}

#[test]
fn test_an_mps_round_trip_keeps_the_program() {
    let program = mixed();
    let read = round_trip(&program, ModelFormat::Mps);
    assert_eq!(read.sense(), Sense::Maximize);
    assert_eq!(read.offset(), 1.5);
    assert_eq!(columns_by_name(&read), columns_by_name(&program));
    let rows: Vec<Option<&str>> = read.rows().map(|r| read.row(r).name()).collect();
    assert_eq!(rows, [Some("band"), Some("floor"), Some("roof")]);
    assert_eq!(optimum(&read), optimum(&program));
    assert_eq!(read.features(), program.features());
}

#[test]
fn test_an_lp_round_trip_keeps_the_program_without_semi_integers() {
    let mut m = Model::new();
    let x = m.add_var(Var::integer().bounds(-2, 5).name("x"));
    let y = m.add_var(Var::free().name("y"));
    let z = m.add_var(Var::semi_continuous().bounds(2, 10).name("z"));
    m.add_named_constraint("band", (x - y + z).between(-1, 4));
    m.add_named_constraint("roof", (x + y).leq(9));
    m.maximize(3.0 * x + y - z + 1.5);
    let program = m.build().unwrap();
    let read = round_trip(&program, ModelFormat::Lp);
    assert_eq!(columns_by_name(&read), columns_by_name(&program));
    assert_eq!((read.sense(), read.offset()), (Sense::Maximize, 1.5));
    assert_eq!(optimum(&read), optimum(&program));
}

#[test]
fn test_a_qp_round_trips_in_both_formats() {
    let mut m = Model::new();
    let x = m.add_var(Var::free().name("x"));
    let y = m.add_var(Var::continuous().upper(3).name("y"));
    m.minimize(x * x + x * y + 2.0 * y * y - 3.0 * x + 1.5);
    let program = m.build().unwrap();
    for format in [ModelFormat::Mps, ModelFormat::Lp] {
        let read = round_trip(&program, format);
        assert_eq!(
            read.quadratic().entries().collect::<Vec<_>>(),
            program.quadratic().entries().collect::<Vec<_>>(),
            "{format}"
        );
        assert_eq!(read.num_rows(), 0, "{format}: no padding row is written");
        assert!(
            close(optimum(&read).unwrap(), optimum(&program).unwrap(), 1e-9),
            "{format}"
        );
    }
}

#[test]
fn test_an_explicit_format_ignores_the_extension() {
    let program = mixed();
    let path = TempPath::new("txt");
    files::write_as(&program, &path.0, ModelFormat::Mps).unwrap();
    assert!(std::fs::read_to_string(&path.0).unwrap().contains("ROWS"));
    assert_eq!(files::read(&path.0), Err(FileError::Extension { path: path.0.clone() }));
    let read = files::read_as(&path.0, ModelFormat::Mps).unwrap();
    assert_eq!(optimum(&read), optimum(&program));
    assert_eq!(
        files::read_as(&path.0, ModelFormat::Lp)
            .map(|p| p.num_cols())
            .unwrap_err(),
        FileError::Read {
            path: path.0.clone(),
            format: ModelFormat::Lp
        },
        "an MPS file does not parse as an LP file"
    );
}

#[test]
fn test_what_a_format_loses_is_refused() {
    let semi_integer = mixed();
    let path = TempPath::new("lp");
    assert_eq!(
        files::write(&semi_integer, &path.0),
        Err(FileError::Unsupported {
            format: ModelFormat::Lp,
            feature: ProgramFeature::SemiInteger
        })
    );
    assert!(!path.0.exists(), "nothing is written");

    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(5, 5));
    m.maximize(x);
    let fixed_semi = m.build().unwrap();
    assert_eq!(
        files::write_as(&fixed_semi, &path.0, ModelFormat::Mps),
        Err(FileError::Unsupported {
            format: ModelFormat::Mps,
            feature: ProgramFeature::SemiContinuous
        })
    );
    assert_eq!(
        round_trip(&fixed_semi, ModelFormat::Lp).columns()[x].integrality,
        Integrality::SemiContinuous
    );

    m.add_objective(Objective::minimize(x));
    let several = m.build().unwrap();
    for format in [ModelFormat::Mps, ModelFormat::Lp] {
        assert_eq!(
            files::write_as(&several, &path.0, format),
            Err(FileError::Unsupported {
                format,
                feature: ProgramFeature::MultipleObjectives
            })
        );
    }
}

/// The text HiGHS's own writer gives a one-column model of `var_type` in `[lower, upper]`, maximizing it.
fn highs_writes(var_type: i32, lower: f64, upper: f64, extension: &str) -> String {
    let path = TempPath::new(extension);
    let c_path = CString::new(path.0.to_str().unwrap()).unwrap();
    let (cost, lower, upper, integrality) = ([1.0], [lower], [upper], [var_type]);
    // SAFETY: the instance is created here, used only on this thread, and destroyed before the block ends. The model
    // has one column and no rows: each column array holds one value, the row and matrix arrays are null with zero
    // counts, which HiGHS skips; the path is a NUL-terminated string HiGHS reads during the call.
    let (passed, written) = unsafe {
        let highs = highs_sys::Highs_create();
        debug_assert!(!highs.is_null(), "HiGHS allocates an instance");
        highs_sys::Highs_setBoolOptionValue(highs, c"output_flag".as_ptr(), 0);
        let passed = highs_sys::Highs_passMip(
            highs,
            1,
            0,
            0,
            highs_sys::kHighsMatrixFormatRowwise,
            highs_sys::kHighsObjSenseMaximize,
            0.0,
            cost.as_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            integrality.as_ptr(),
        );
        let written = highs_sys::Highs_writeModel(highs, c_path.as_ptr());
        highs_sys::Highs_destroy(highs);
        (passed, written)
    };
    // HiGHS warns that it names the blank column and row.
    assert_eq!(passed, highs_sys::kHighsStatusOk);
    assert_ne!(written, highs_sys::kHighsStatusError);
    std::fs::read_to_string(&path.0).unwrap()
}

#[test]
fn test_highs_writers_lose_what_the_backend_refuses() {
    // HiGHS 1.15's LP writer lists a semi-integer column under `semi` only, not under `gen` (FilereaderLp.cpp), so
    // it reads back semi-continuous.
    let lp = highs_writes(highs_sys::kHighsVarTypeSemiInteger, 1.0, 7.0, "lp");
    let general = lp.split("gen").nth(1).unwrap().split("semi").next().unwrap();
    assert!(general.trim().is_empty(), "{lp}");
    // Its MPS writer writes a semi column with equal bounds as fixed (`FX`) before it looks at semi columns
    // (HMPSIO.cpp), so it reads back a fixed continuous column.
    let mps = highs_writes(highs_sys::kHighsVarTypeSemiContinuous, 5.0, 5.0, "mps");
    assert!(mps.contains(" FX BOUND") && !mps.contains(" SC BOUND"), "{mps}");
}

#[test]
fn test_unreadable_files_are_typed_errors() {
    let missing = std::env::temp_dir().join("culmen-files-missing.mps");
    assert!(matches!(
        files::read(&missing),
        Err(FileError::Io {
            kind: std::io::ErrorKind::NotFound,
            ..
        })
    ));
    assert_eq!(
        files::read("model.json"),
        Err(FileError::Extension {
            path: "model.json".into()
        })
    );

    let garbage = TempPath::new("lp");
    std::fs::write(&garbage.0, "Maximize\n obj: x +\nSubject To\n c: x <= \nEnd\n").unwrap();
    assert_eq!(
        files::read(&garbage.0),
        Err(FileError::Read {
            path: garbage.0.clone(),
            format: ModelFormat::Lp
        })
    );

    // HiGHS reads a non-convex objective, which no program holds.
    let concave = TempPath::new("lp");
    std::fs::write(
        &concave.0,
        "Minimize\n obj: [ -2 x ^ 2 ] / 2\nSubject To\nBounds\n x <= 1\nEnd\n",
    )
    .unwrap();
    assert!(matches!(
        files::read(&concave.0),
        Err(FileError::Program(ProgramError::NonConvexObjective {
            sense: Sense::Minimize,
            ..
        }))
    ));
    assert_eq!(
        FileError::Extension { path: "a.json".into() }.to_string(),
        "model file: a.json names no format"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// A tiny LP or MILP keeps its optimum through either format.
    #[test]
    fn test_tiny_programs_keep_their_optimum_through_files(program in small_lp(), lp in any::<bool>()) {
        let format = if lp { ModelFormat::Lp } else { ModelFormat::Mps };
        let read = round_trip(&program, format);
        match (optimum(&read), optimum(&program)) {
            (Some(a), Some(b)) => prop_assert!(close(a, b, 1e-9), "{format}: {a} against {b}"),
            (a, b) => prop_assert_eq!(a, b, "{}", format),
        }
    }

    /// A tiny program with semi columns keeps its optimum through an MPS file.
    #[test]
    fn test_tiny_semi_programs_keep_their_optimum_through_mps(program in small_semi()) {
        let fixed = program.columns().iter().any(|c| c.integrality.is_semi() && c.bounds.lower() == c.bounds.upper());
        prop_assume!(!fixed);
        let read = round_trip(&program, ModelFormat::Mps);
        match (optimum(&read), optimum(&program)) {
            (Some(a), Some(b)) => prop_assert!(close(a, b, 1e-9), "{a} against {b}"),
            (a, b) => prop_assert_eq!(a, b),
        }
    }
}
