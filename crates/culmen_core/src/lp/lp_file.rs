// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An LP in the CPLEX LP file format, for reading by eye or by another solver while debugging.

use core::fmt;

use super::{Col, Integrality, Lp, Row, Sense};

impl Lp {
    /// The LP in the CPLEX LP file format.
    ///
    /// The output is meant for debugging: names are reduced to the characters the format allows, an unnamed column
    /// is written `x<index>` and an unnamed row `r<index>`, and a ranged row becomes two rows suffixed `_lo` and
    /// `_hi`. Distinct names that reduce to the same text are not told apart. A quadratic part is written in the
    /// format's brackets halved, `[ 2 x ^ 2 + 4 x * y ] / 2` for `x² + 2xy`.
    ///
    /// ```
    /// use culmen_core::{LpColumn, LpPlan, Sense, bounds};
    ///
    /// let mut plan = LpPlan::new();
    /// plan.sense = Sense::Maximize;
    /// let x = plan.col(LpColumn::binary(3.0));
    /// plan.row(bounds!(f64::NEG_INFINITY, 1.0), [(x, 2.0)]);
    /// let text = plan.clone().build()?.to_lp_string();
    /// assert!(text.starts_with("Maximize\n obj: 3 x0\nSubject To\n r0: 2 x0 <= 1\n"));
    ///
    /// plan.sense = Sense::Minimize;
    /// let y = plan.col(LpColumn::continuous(0.0, bounds!(0.0, 1.0)));
    /// plan.quadratic = vec![(x, x, 1.0), (x, y, -1.5), (y, y, 1.0)];
    /// let text = plan.build()?.to_lp_string();
    /// assert!(text.starts_with("Minimize\n obj: 3 x0 + [ 2 x0 ^ 2 - 3 x0 * x1 + 2 x1 ^ 2 ] / 2\n"));
    /// # Ok::<(), culmen_core::LpError>(())
    /// ```
    #[must_use]
    pub fn to_lp_string(&self) -> String {
        let mut out = String::new();
        // Writing into a `String` cannot fail.
        let _ = self.write_lp(&mut out);
        out
    }

    /// Writes the LP in the CPLEX LP file format, as [`Lp::to_lp_string`] describes.
    ///
    /// # Errors
    ///
    /// The writer's error.
    pub fn write_lp(&self, out: &mut impl fmt::Write) -> fmt::Result {
        self.write_objective(out)?;
        out.write_str("Subject To\n")?;
        for row in self.rows() {
            self.write_row(out, row)?;
        }
        self.write_bounds(out)?;
        let integers: Vec<Col> = self
            .cols()
            .filter(|&col| self.columns[col].integrality == Integrality::Integer)
            .collect();
        if !integers.is_empty() {
            out.write_str("General\n")?;
            for col in integers {
                writeln!(out, " {}", self.col_label(col))?;
            }
        }
        out.write_str("End\n")
    }

    /// Writes the sense and the objective: its linear terms, its quadratic part and its offset.
    ///
    /// # Errors
    ///
    /// The writer's error.
    fn write_objective(&self, out: &mut impl fmt::Write) -> fmt::Result {
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
            self.write_terms(out, costs)?;
        }
        if !self.quadratic.is_empty() {
            self.write_quadratic(out, linear)?;
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
    fn write_quadratic(&self, out: &mut impl fmt::Write, after_terms: bool) -> fmt::Result {
        out.write_str(if after_terms { " + [" } else { " [" })?;
        for (k, (i, j, q)) in self.quadratic.entries().enumerate() {
            let product = if i == j {
                format!("{} ^ 2", self.col_label(i))
            } else {
                format!("{} * {}", self.col_label(i), self.col_label(j))
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
    fn write_row(&self, out: &mut impl fmt::Write, row: Row) -> fmt::Result {
        let bounds = self.row_bounds[row];
        let (lower, upper) = (bounds.lower(), bounds.upper());
        let name = self.row_label(row);
        let mut line = |suffix: &str, relation: &str, rhs: f64| -> fmt::Result {
            write!(out, " {name}{suffix}:")?;
            self.write_terms(out, self.row(row).entries())?;
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
    fn write_bounds(&self, out: &mut impl fmt::Write) -> fmt::Result {
        out.write_str("Bounds\n")?;
        for col in self.cols() {
            let bounds = self.columns[col].bounds;
            let name = self.col_label(col);
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

    /// Writes ` c x + d y …`, or ` 0` for no terms.
    ///
    /// # Errors
    ///
    /// The writer's error.
    fn write_terms(&self, out: &mut impl fmt::Write, terms: impl Iterator<Item = (Col, f64)>) -> fmt::Result {
        let mut first = true;
        for (col, value) in terms {
            write_signed(out, first, value, &self.col_label(col))?;
            first = false;
        }
        if first {
            out.write_str(" 0")?;
        }
        Ok(())
    }

    fn col_label(&self, col: Col) -> String {
        label(self.columns[col].name.as_deref(), 'x', col.index())
    }

    fn row_label(&self, row: Row) -> String {
        label(self.row_names[row].as_deref(), 'r', row.index())
    }
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

/// A name reduced to the characters the LP format allows, never starting with a digit, a period or an `e`.
fn label(name: Option<&str>, prefix: char, index: usize) -> String {
    let Some(name) = name.filter(|n| !n.is_empty()) else {
        return format!("{prefix}{index}");
    };
    let mut out = String::with_capacity(name.len() + 1);
    if name.starts_with(|c: char| c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E') {
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
