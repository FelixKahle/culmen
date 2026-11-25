// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A program in the CPLEX LP file format, for reading by eye or by another solver while debugging.

use core::fmt;

use std::collections::HashSet;

use typed_index_collections::TiVec;

use super::{Col, Integrality, Program, ProgramFeature, Row, Sense};

impl Program {
    /// The program in the CPLEX LP file format.
    ///
    /// The output is meant for debugging: names are reduced to the characters the format allows, a name that is one of
    /// its keywords (`max`, `sos`, `end`, …, in any case) is prefixed `_`, an unnamed column is written `x<index>` and
    /// an unnamed row `r<index>`, a ranged row becomes two rows suffixed `_lo` and `_hi`, and a free row, which the
    /// format cannot state and which constrains nothing, is left out. Every column and every row gets a label of its
    /// own, so the file states the same program whatever the names: a label taken already, by a name given twice, two
    /// names that reduce to the same text, or a generated label equal to a name, is suffixed `~1`, `~2`, … until it is
    /// free, the columns and rows that keep their own name first. A quadratic part is written in the format's brackets
    /// halved, `[ 2 x ^ 2 + 4 x * y ] / 2` for `x² + 2xy`. Integer and semi-integer columns are listed under `General`,
    /// semi-continuous and semi-integer ones under `Semi-Continuous`: a column in both is semi-integer, as CPLEX and
    /// HiGHS read it.
    ///
    /// ```
    /// use culmen_core::{ProgramColumn, ProgramPlan, Sense, bounds};
    ///
    /// let mut plan = ProgramPlan::new();
    /// plan.sense = Sense::Maximize;
    /// let x = plan.col(ProgramColumn::binary(3.0));
    /// plan.row(bounds!(f64::NEG_INFINITY, 1.0), [(x, 2.0)]);
    /// let text = plan.clone().build()?.to_lp_string()?;
    /// assert!(text.starts_with("Maximize\n obj: 3 x0\nSubject To\n r0: 2 x0 <= 1\n"));
    ///
    /// plan.sense = Sense::Minimize;
    /// let y = plan.col(ProgramColumn::continuous(0.0, bounds!(0.0, 1.0)));
    /// plan.quadratic = vec![(x, x, 1.0), (x, y, -1.5), (y, y, 1.0)];
    /// let text = plan.build()?.to_lp_string()?;
    /// assert!(text.starts_with("Minimize\n obj: 3 x0 + [ 2 x0 ^ 2 - 3 x0 * x1 + 2 x1 ^ 2 ] / 2\n"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`LpFileError::Unsupported`] for a program with [`ProgramFeature::MultipleObjectives`]: CPLEX's multi-objective
    /// sections combine tolerances otherwise than [`Tolerance`](super::Tolerance) states, and HiGHS cannot read them.
    pub fn to_lp_string(&self) -> Result<String, LpFileError> {
        let mut out = String::new();
        self.write_lp(&mut out)?;
        Ok(out)
    }

    /// Writes the program in the CPLEX LP file format, as [`Program::to_lp_string`] describes.
    ///
    /// # Errors
    ///
    /// [`LpFileError::Unsupported`] as [`Program::to_lp_string`] states, before anything is written;
    /// [`LpFileError::Write`] for the writer's error.
    pub fn write_lp(&self, out: &mut impl fmt::Write) -> Result<(), LpFileError> {
        if self.objectives.are_several() {
            return Err(LpFileError::Unsupported {
                feature: ProgramFeature::MultipleObjectives,
            });
        }
        let labels = Labels::of(self);
        self.write_objective(out, &labels)?;
        out.write_str("Subject To\n")?;
        self.rows().try_for_each(|row| self.write_row(out, &labels, row))?;
        self.write_bounds(out, &labels)?;
        self.write_section(out, &labels, "General", Integrality::is_integral)?;
        self.write_section(out, &labels, "Semi-Continuous", Integrality::is_semi)?;
        out.write_str("End\n")?;
        Ok(())
    }

    /// Writes a section listing the columns whose integrality `listed` picks, unless it lists none.
    ///
    /// # Errors
    ///
    /// The writer's error.
    fn write_section(
        &self,
        out: &mut impl fmt::Write,
        labels: &Labels,
        title: &str,
        listed: impl Fn(Integrality) -> bool,
    ) -> fmt::Result {
        let mut cols = self
            .cols()
            .filter(|&col| listed(self.columns[col].integrality))
            .peekable();
        if cols.peek().is_none() {
            return Ok(());
        }
        writeln!(out, "{title}")?;
        cols.try_for_each(|col| writeln!(out, " {}", labels.cols[col]))
    }

    /// Writes the sense and the objective: its linear terms, its quadratic part and its offset.
    ///
    /// # Errors
    ///
    /// The writer's error.
    fn write_objective(&self, out: &mut impl fmt::Write, labels: &Labels) -> fmt::Result {
        out.write_str(match self.sense {
            Sense::Minimize => "Minimize\n",
            Sense::Maximize => "Maximize\n",
        })?;
        out.write_str(" obj:")?;
        let costs = self
            .columns
            .iter_enumerated()
            .map(|(col, c)| (col, c.cost))
            .filter(|&(_, c)| c != 0.0);
        let linear = costs.clone().next().is_some();
        if linear || self.quadratic.is_empty() {
            write_terms(out, labels, costs)?;
        }
        if !self.quadratic.is_empty() {
            self.write_quadratic(out, labels, linear)?;
        }
        if self.offset != 0.0 {
            write!(
                out,
                " {} {}",
                if self.offset < 0.0 { '-' } else { '+' },
                self.offset.abs()
            )?;
        }
        out.write_str("\n")
    }

    /// Writes the quadratic part doubled in brackets halved, ` + [ 2 x ^ 2 + 4 x * y ] / 2`, without the leading
    /// `+` when nothing comes before it.
    ///
    /// # Errors
    ///
    /// The writer's error.
    fn write_quadratic(&self, out: &mut impl fmt::Write, labels: &Labels, after_terms: bool) -> fmt::Result {
        out.write_str(if after_terms { " + [" } else { " [" })?;
        for (k, (i, j, q)) in self.quadratic.entries().enumerate() {
            let product = if i == j {
                format!("{} ^ 2", labels.cols[i])
            } else {
                format!("{} * {}", labels.cols[i], labels.cols[j])
            };
            write_signed(out, k == 0, 2.0 * q, &product)?;
        }
        out.write_str(" ] / 2")
    }

    /// Writes one row, as two when it is ranged; a free row is skipped.
    ///
    /// # Errors
    ///
    /// The writer's error.
    fn write_row(&self, out: &mut impl fmt::Write, labels: &Labels, row: Row) -> fmt::Result {
        let bounds = self.row_bounds[row];
        let (lower, upper) = (bounds.lower(), bounds.upper());
        let name = &labels.rows[row];
        let mut line = |suffix: &str, relation: &str, rhs: f64| -> fmt::Result {
            write!(out, " {name}{suffix}:")?;
            write_terms(out, labels, self.row(row).entries())?;
            writeln!(out, " {relation} {rhs}")
        };
        match (lower.is_finite(), upper.is_finite()) {
            (true, true) if lower == upper => line("", "=", lower),
            (true, true) => {
                line("_lo", ">=", lower)?;
                line("_hi", "<=", upper)
            }
            (true, false) => line("", ">=", lower),
            (false, true) => line("", "<=", upper),
            (false, false) => Ok(()),
        }
    }

    /// Writes the bounds section: every column whose bounds are not the format's default `0 ..`.
    ///
    /// # Errors
    ///
    /// The writer's error.
    fn write_bounds(&self, out: &mut impl fmt::Write, labels: &Labels) -> fmt::Result {
        out.write_str("Bounds\n")?;
        for col in self.cols() {
            let bounds = self.columns[col].bounds;
            let name = &labels.cols[col];
            match (bounds.lower(), bounds.upper()) {
                (0.0, f64::INFINITY) => {}
                (f64::NEG_INFINITY, f64::INFINITY) => writeln!(out, " {name} free")?,
                (lower, upper) if lower == upper => writeln!(out, " {name} = {lower}")?,
                (f64::NEG_INFINITY, upper) => writeln!(out, " -inf <= {name} <= {upper}")?,
                (lower, f64::INFINITY) => writeln!(out, " {name} >= {lower}")?,
                (lower, upper) => writeln!(out, " {lower} <= {name} <= {upper}")?,
            }
        }
        Ok(())
    }
}

/// Writes ` c x + d y …`, or ` 0` for no terms.
///
/// # Errors
///
/// The writer's error.
fn write_terms(out: &mut impl fmt::Write, labels: &Labels, terms: impl Iterator<Item = (Col, f64)>) -> fmt::Result {
    let mut first = true;
    for (col, value) in terms {
        write_signed(out, first, value, &labels.cols[col])?;
        first = false;
    }
    if first {
        out.write_str(" 0")?;
    }
    Ok(())
}

/// The label of every column and row in one LP file: no two columns share one, and no two rows write a line of the
/// same name (a ranged row writes its label suffixed `_lo` and `_hi`), nor one named as the objective, `obj`.
#[derive(Debug)]
struct Labels {
    cols: TiVec<Col, String>,
    /// The label of every row, which a ranged row writes suffixed.
    rows: TiVec<Row, String>,
}

impl Labels {
    fn of(program: &Program) -> Self {
        let cols = program.cols().map(|col| {
            let name = program.columns[col].name.as_deref();
            Wanted::new(name, label(name, 'x', col.index()), &[""])
        });
        let rows = program.rows().map(|row| {
            let (lower, upper) = (program.row_bounds[row].lower(), program.row_bounds[row].upper());
            let suffixes: &[&str] = match (lower.is_finite(), upper.is_finite()) {
                (true, true) if lower != upper => &["_lo", "_hi"],
                (false, false) => &[],
                _ => &[""],
            };
            let name = program.row_names[row].as_deref();
            Wanted::new(name, label(name, 'r', row.index()), suffixes)
        });
        Self {
            cols: unique(cols.collect(), &[]).into(),
            rows: unique(rows.collect(), &["obj"]).into(),
        }
    }
}

/// The label a line would take, and the suffixes it writes it with: one line for `[""]`, none for a free row.
#[derive(Debug)]
struct Wanted {
    base: String,
    /// Whether the line has a name of its own, so its base takes precedence over the generated ones.
    named: bool,
    suffixes: &'static [&'static str],
}

impl Wanted {
    fn new(name: Option<&str>, base: String, suffixes: &'static [&'static str]) -> Self {
        Self {
            base,
            named: name.is_some_and(|name| !name.is_empty()),
            suffixes,
        }
    }

    /// Whether `base`, with each of this line's suffixes, is none of `taken`.
    fn fits(&self, base: &str, taken: &HashSet<String>) -> bool {
        self.suffixes
            .iter()
            .all(|suffix| !taken.contains(&format!("{base}{suffix}")))
    }

    /// Takes `base` with each of this line's suffixes.
    fn claim(&self, base: String, taken: &mut HashSet<String>) -> String {
        taken.extend(self.suffixes.iter().map(|suffix| format!("{base}{suffix}")));
        base
    }
}

/// A label for each wanted line, such that no two lines and none of `reserved` write the same text: a named line
/// keeps its base when it is free, in order, and every other line takes the first free of its base, `base~1`,
/// `base~2`, … in order.
///
/// # Panics
///
/// Never: each text taken rules out at most one of a line's candidates, so one of the first `taken + 1` is free.
fn unique(wanted: Vec<Wanted>, reserved: &[&str]) -> Vec<String> {
    let mut taken: HashSet<String> = reserved.iter().map(|&r| r.to_owned()).collect();
    let mut labels: Vec<Option<String>> = wanted
        .iter()
        .map(|line| (line.named && line.fits(&line.base, &taken)).then(|| line.claim(line.base.clone(), &mut taken)))
        .collect();
    for (line, label) in wanted.iter().zip(&mut labels) {
        if label.is_none() {
            let candidates = (0..=taken.len()).map(|k| match k {
                0 => line.base.clone(),
                k => format!("{}~{k}", line.base),
            });
            let mut free = candidates.filter(|candidate| line.fits(candidate, &taken));
            let chosen = free
                .next()
                .expect("Labels: one of the first `taken + 1` candidates is free");
            *label = Some(line.claim(chosen, &mut taken));
        }
    }
    labels.into_iter().flatten().collect()
}

/// Writes ` c what` for the first term of a sum and ` + c what` or ` - c what` for any other, `-c` first when
/// negative.
///
/// # Errors
///
/// The writer's error.
fn write_signed(out: &mut impl fmt::Write, first: bool, value: f64, what: &str) -> fmt::Result {
    match (first, value < 0.0) {
        (true, false) => write!(out, " {value} {what}"),
        (true, true) => write!(out, " -{} {what}", value.abs()),
        (false, false) => write!(out, " + {value} {what}"),
        (false, true) => write!(out, " - {} {what}", value.abs()),
    }
}

/// The words of the LP format a name may not be, in any case: its section headers, `inf`, `free`, and their
/// variants as CPLEX and HiGHS read them.
const KEYWORDS: [&str; 33] = [
    "bin", "binaries", "binary", "bound", "bounds", "end", "free", "gen", "general", "generals", "inf", "infinity",
    "int", "integer", "integers", "max", "maximise", "maximize", "maximum", "min", "minimise", "minimize", "minimum",
    "s.t.", "semi", "semis", "sos", "sos1", "sos2", "st", "st.", "subject", "such",
];

/// A name reduced to the characters the LP format allows, never starting with a digit, a period or an `e`, and
/// never one of its keywords.
fn label(name: Option<&str>, prefix: char, index: usize) -> String {
    let Some(name) = name.filter(|n| !n.is_empty()) else {
        return format!("{prefix}{index}");
    };
    let mut out = String::with_capacity(name.len() + 1);
    let reserved = KEYWORDS.iter().any(|keyword| keyword.eq_ignore_ascii_case(name));
    if reserved || name.starts_with(|c: char| c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E') {
        out.push('_');
    }
    out.extend(name.chars().map(|c| {
        if c.is_ascii_alphanumeric() || "_.!#$%&()/,;?@{}|~".contains(c) {
            c
        } else {
            '_'
        }
    }));
    out
}

/// Why [`Program::write_lp`] wrote no LP file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LpFileError {
    /// The program uses a feature the LP file format as written here cannot state.
    Unsupported {
        /// The feature.
        feature: ProgramFeature,
    },
    /// The writer failed.
    Write(fmt::Error),
}

impl From<fmt::Error> for LpFileError {
    fn from(error: fmt::Error) -> Self {
        Self::Write(error)
    }
}

impl fmt::Display for LpFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { feature } => write!(f, "LP file: a program with {feature} cannot be written"),
            Self::Write(error) => write!(f, "LP file: the writer failed: {error}"),
        }
    }
}

impl core::error::Error for LpFileError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Unsupported { .. } => None,
            Self::Write(error) => Some(error),
        }
    }
}
