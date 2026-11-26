// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model files through HiGHS's readers and writers: round trips keep the program up to names and order and keep its
//! optimum, semi columns included; what a format loses is refused; what HiGHS cannot read is a typed error.

#![cfg_attr(
    test,
    allow(clippy::missing_panics_doc, reason = "test assertions are the statement")
)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use culmen_core::{
    Bounds, Integrality, Linear, Model, Objective, Outcome, Program, ProgramError, ProgramFeature, Sense, Solver, Var,
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
    let solved = Solver::new(Highs::new(), program.clone()).unwrap().solve().unwrap();
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
fn test_an_lp_round_trip_keeps_the_program() {
    let program = mixed();
    let read = round_trip(&program, ModelFormat::Lp);
    assert_eq!((read.sense(), read.offset()), (Sense::Maximize, 1.5));
    assert_eq!(columns_by_name(&read), columns_by_name(&program));
    // An LP file holds the ranged row as two rows.
    let rows: Vec<Option<&str>> = read.rows().map(|r| read.row(r).name()).collect();
    assert_eq!(rows, [Some("bandlo"), Some("bandup"), Some("floor"), Some("roof")]);
    assert_eq!(optimum(&read), optimum(&program));
    assert_eq!(read.features(), program.features());
}

/// The program's own LP file, as culmen writes it, read back by HiGHS's reader.
fn read_lp_string(program: &Program) -> Program {
    let path = TempPath::new("lp");
    std::fs::write(&path.0, program.to_lp_string().unwrap()).unwrap();
    files::read_as(&path.0, ModelFormat::Lp).unwrap()
}

#[test]
fn test_culmens_lp_file_tells_every_column_and_row_apart_whatever_their_names() {
    let mut m = Model::new();
    // A named column `x1` beside unnamed column 1, which the writer would call `x1` too.
    let x = m.add_var(Var::continuous().bounds(0, 10).name("x1"));
    let y = m.add_var(Var::continuous().bounds(2, 10));
    // Two columns of one name, and two names that reduce to the same text.
    let a = m.add_var(Var::continuous().bounds(0, 4).name("a"));
    let b = m.add_var(Var::continuous().bounds(0, 4).name("a"));
    let c = m.add_var(Var::continuous().bounds(0, 4).name("c d"));
    let d = m.add_var(Var::continuous().bounds(0, 4).name("c_d"));
    // A named row `r1` beside unnamed row 1, and a ranged row `band` beside a row named `band_lo`.
    m.add_named_constraint("r1", (x - y).geq(1));
    m.add_constraint((a - b).geq(1));
    m.add_named_constraint("band", (c - d).between(1, 3));
    m.add_named_constraint("band_lo", (c + d).leq(5));
    m.minimize(x + 2 * y + a + b + c + d);
    let program = m.build().unwrap();
    let read = read_lp_string(&program);
    assert_eq!(
        read.num_cols(),
        program.num_cols(),
        "{}",
        program.to_lp_string().unwrap()
    );
    assert_eq!(read.num_rows(), program.num_rows() + 1, "the ranged row is two rows");
    // x = 3, y = 2; a = 1, b = 0; c = 1, d = 0.
    assert_eq!(optimum(&program), Some(9.0));
    assert_eq!(optimum(&read), Some(9.0), "{}", program.to_lp_string().unwrap());
}

#[test]
fn test_a_semi_column_with_equal_bounds_round_trips_in_both_formats() {
    let mut m = Model::new();
    let x = m.add_var(Var::semi_continuous().bounds(5, 5).name("x"));
    let y = m.add_var(Var::semi_integer().bounds(3, 3).name("y"));
    m.add_named_constraint("cap", (x + y).leq(4));
    m.maximize(x + 2 * y);
    let program = m.build().unwrap();
    for format in [ModelFormat::Mps, ModelFormat::Lp] {
        let read = round_trip(&program, format);
        assert_eq!(columns_by_name(&read), columns_by_name(&program), "{format}");
        assert_eq!(optimum(&read), optimum(&program), "{format}");
    }
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
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(5));
    m.maximize(x);
    m.add_objective(Objective::minimize(x));
    let several = m.build().unwrap();
    let path = TempPath::new("lp");
    for format in [ModelFormat::Mps, ModelFormat::Lp] {
        assert_eq!(
            files::write_as(&several, &path.0, format),
            Err(FileError::Unsupported {
                format,
                feature: ProgramFeature::MultipleObjectives
            })
        );
        assert!(!path.0.exists(), "{format}: nothing is written");
    }
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

#[test]
fn test_what_highs_cannot_take_or_write_is_a_typed_error() {
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1).name("x"));
    m.maximize(x);
    let program = m.build().unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let not_utf8 = std::env::temp_dir().join(std::ffi::OsStr::from_bytes(b"culmen-\xff.lp"));
        assert_eq!(
            files::write(&program, &not_utf8),
            Err(FileError::Path { path: not_utf8.clone() })
        );
    }
    let nowhere = std::env::temp_dir().join("culmen-no-such-directory").join("model.lp");
    assert_eq!(
        files::write(&program, &nowhere),
        Err(FileError::Write {
            path: nowhere.clone(),
            format: ModelFormat::Lp
        })
    );
    let path = TempPath::new("lp");
    let mut m = Model::new();
    m.add_var(Var::continuous().name("x\0y"));
    assert_eq!(
        files::write(&m.build().unwrap(), &path.0),
        Err(FileError::Name {
            name: "x\0y".to_owned()
        })
    );
    // HiGHS refuses a matrix value at or above its `large_matrix_value`, 1e15.
    let mut m = Model::new();
    let x = m.add_var(Var::continuous().upper(1));
    m.add_constraint((1e16 * x).leq(1));
    assert_eq!(
        files::write(&m.build().unwrap(), &path.0),
        Err(FileError::Highs(culmen_highs::HighsError::Load))
    );
}

#[test]
fn test_bounds_in_a_file_that_are_no_interval_are_refused() {
    let columns = TempPath::new("lp");
    std::fs::write(
        &columns.0,
        "Minimize\n obj: x + y\nSubject To\n c: x + y >= 1\nBounds\n 3 <= x <= 1\nEnd\n",
    )
    .unwrap();
    assert!(
        matches!(
            files::read(&columns.0),
            Err(FileError::ColumnBounds {
                source: culmen_core::InvalidBounds::Empty { .. },
                ..
            })
        ),
        "{:?}",
        files::read(&columns.0)
    );
    // A row's bounds HiGHS reads are always an interval: it refuses an infinite right-hand side as it reads, so
    // `FileError::RowBounds` guards a reader that would not.
    let rows = TempPath::new("lp");
    std::fs::write(&rows.0, "Minimize\n obj: x\nSubject To\n c: x >= inf\nEnd\n").unwrap();
    assert!(matches!(files::read(&rows.0), Err(FileError::Read { .. })));
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

    /// A tiny program with semi columns keeps its optimum through either format.
    #[test]
    fn test_tiny_semi_programs_keep_their_optimum_through_files(program in small_semi(), lp in any::<bool>()) {
        let format = if lp { ModelFormat::Lp } else { ModelFormat::Mps };
        let read = round_trip(&program, format);
        match (optimum(&read), optimum(&program)) {
            (Some(a), Some(b)) => prop_assert!(close(a, b, 1e-9), "{format}: {a} against {b}"),
            (a, b) => prop_assert_eq!(a, b, "{}", format),
        }
    }
}
