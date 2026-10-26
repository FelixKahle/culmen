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
    /// `_hi`. Distinct names that reduce to the same text are not told apart.
    ///
    /// ```
    /// use culmen_core::{LpColumn, LpPlan, Sense, bounds};
    ///
    /// let mut plan = LpPlan::new();
    /// plan.sense = Sense::Maximize;
    /// let x = plan.col(LpColumn::binary(3.0));
    /// plan.row(bounds!(f64::NEG_INFINITY, 1.0), [(x, 2.0)]);
    /// let text = plan.build()?.to_lp_string();
    /// assert!(text.starts_with("Maximize\n obj: 3 x0\nSubject To\n r0: 2 x0 <= 1\n"));
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
        self.write_terms(out, costs)?;
        if self.offset != 0.0 {
            write!(
                out,
                " {} {}",
                if self.offset < 0.0 { '-' } else { '+' },
                self.offset.abs()
            )?;
        }
        out.write_str("\nSubject To\n")?;
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
            let sign = if value < 0.0 { "-" } else { "+" };
            if first && value >= 0.0 {
                write!(out, " {} {}", value, self.col_label(col))?;
            } else if first {
                write!(out, " -{} {}", value.abs(), self.col_label(col))?;
            } else {
                write!(out, " {sign} {} {}", value.abs(), self.col_label(col))?;
            }
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
