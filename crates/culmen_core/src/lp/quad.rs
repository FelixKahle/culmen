// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The quadratic part of an objective: its canonical entries, its Hessian, and the convexity check.

use core::iter::{Copied, FusedIterator};
use core::slice;
use std::collections::BTreeMap;

use itertools::Itertools;
use nalgebra::{Cholesky, DMatrix};
use typed_index_collections::{TiSlice, TiVec};

use super::{Col, Sense};

/// The quadratic part of an [`Lp`](super::Lp)'s objective, `Σ q · xᵢ · xⱼ`.
///
/// Each entry `(i, j, q)` adds `q · xᵢ · xⱼ`, so the model's `x * x` is the entry `(x, x, 1)` and `2 · x · y` the
/// entry `(x, y, 2)`. The entries are canonical: `i <= j`, each pair at most once, every coefficient finite and
/// non-zero, sorted by `(i, j)`. In the solver's form `½ xᵀHx`, the symmetric Hessian `H` has `Hᵢᵢ = 2q` on the
/// diagonal and `Hᵢⱼ = Hⱼᵢ = q` off it; [`QuadObjective::hessian`] gives its lower triangle.
///
/// An `Lp` holds one only when its objective is convex for minimizing or concave for maximizing, as
/// [`Lp::CONVEXITY_TOLERANCE`](super::Lp::CONVEXITY_TOLERANCE) states.
///
/// ```
/// use culmen_core::{Bounds, LpColumn, LpPlan, TiSlice};
///
/// let mut plan = LpPlan::new();
/// let x = plan.col(LpColumn::continuous(0.0, Bounds::FREE));
/// let y = plan.col(LpColumn::continuous(0.0, Bounds::FREE));
/// plan.quadratic = vec![(y, x, 1.0), (x, x, 1.0), (x, y, 1.0), (y, y, 1.0)];
/// let lp = plan.build()?;
/// let q = lp.quadratic();
/// assert_eq!(q.entries().collect::<Vec<_>>(), [(x, x, 1.0), (x, y, 2.0), (y, y, 1.0)]);
/// assert_eq!(q.hessian().collect::<Vec<_>>(), [(x, x, 2.0), (x, y, 2.0), (y, y, 2.0)]);
/// assert_eq!(q.value(TiSlice::from_ref(&[1.0, 2.0])), 9.0);
/// # Ok::<(), culmen_core::LpError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
pub struct QuadObjective {
    entries: Vec<(Col, Col, f64)>,
}

impl QuadObjective {
    /// The canonical form of `terms`: each pair ordered, merged in the order given, zeros dropped, sorted.
    ///
    /// Merged coefficients are not checked; a sum that overflows stays infinite.
    pub(crate) fn from_terms(terms: impl IntoIterator<Item = (Col, Col, f64)>) -> Self {
        Self {
            entries: merge_pairs(terms).into_iter().filter(|&(_, _, q)| q != 0.0).collect(),
        }
    }

    /// The number of entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the objective has no quadratic part.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries as `(i, j, q)`, adding `q · xᵢ · xⱼ`, with `i <= j` and sorted by `(i, j)`.
    pub fn entries(&self) -> QuadEntries<'_> {
        QuadEntries {
            inner: self.entries.iter().copied(),
        }
    }

    /// The lower triangle of the Hessian `H` of `½ xᵀHx`, column by column: `(column, row, value)` with
    /// `row >= column`, sorted by `(column, row)`, a diagonal value twice its entry's coefficient.
    pub fn hessian(&self) -> HessianEntries<'_> {
        HessianEntries { inner: self.entries() }
    }

    /// The value `Σ q · xᵢ · xⱼ` at a point.
    ///
    /// # Panics
    ///
    /// When `values` has no value for a column of an entry.
    #[must_use]
    pub fn value(&self, values: &TiSlice<Col, f64>) -> f64 {
        self.entries().map(|(i, j, q)| q * values[i] * values[j]).sum()
    }
}

/// `(i, j, q)` with `i <= j`.
fn ordered((i, j, q): (Col, Col, f64)) -> (Col, Col, f64) {
    if i <= j { (i, j, q) } else { (j, i, q) }
}

/// The pairs ordered and sorted, each once with its coefficients added up in the order given; zeros are kept.
pub(crate) fn merge_pairs(terms: impl IntoIterator<Item = (Col, Col, f64)>) -> Vec<(Col, Col, f64)> {
    // A stable sort keeps each pair's terms in the order given, so every caller adds them up alike.
    terms
        .into_iter()
        .map(ordered)
        .sorted_by_key(|&(i, j, _)| (i, j))
        .coalesce(|a, b| {
            if (a.0, a.1) == (b.0, b.1) {
                Ok((a.0, a.1, a.2 + b.2))
            } else {
                Err((a, b))
            }
        })
        .collect()
}

/// The Hessian value of a canonical entry: twice the coefficient on the diagonal.
const fn hessian_value((i, j, q): (Col, Col, f64)) -> f64 {
    if i.index() == j.index() { 2.0 * q } else { q }
}

/// The relative tolerance of the convexity check; see [`Lp::CONVEXITY_TOLERANCE`](super::Lp::CONVEXITY_TOLERANCE).
pub(crate) const CONVEXITY_TOLERANCE: f64 = 1e-9;

/// Checks that canonical, finite entries are convex for minimizing or concave for maximizing, block by block.
///
/// # Errors
///
/// The smallest column of the first block, by its smallest column, whose Hessian fails the check.
///
/// # Panics
///
/// When an entry names a column at or beyond `num_cols`.
pub(crate) fn check_convex(sense: Sense, entries: &QuadObjective, num_cols: usize) -> Result<(), Col> {
    blocks(entries, num_cols)
        .into_iter()
        .find(|(_, block)| !block_is_convex(sense, block))
        .map_or(Ok(()), |(first, _)| Err(first))
}

/// The entries grouped into the connected blocks of the Hessian, each keyed by its smallest column.
///
/// # Panics
///
/// When an entry names a column at or beyond `num_cols`.
fn blocks(entries: &QuadObjective, num_cols: usize) -> BTreeMap<Col, Vec<(Col, Col, f64)>> {
    let mut roots = Roots::new(num_cols);
    entries.entries().for_each(|(i, j, _)| roots.join(i, j));
    entries.entries().fold(BTreeMap::new(), |mut blocks, entry| {
        blocks.entry(roots.find(entry.0)).or_insert_with(Vec::new).push(entry);
        blocks
    })
}

/// Whether one block's Hessian, signed by the sense, plus its tolerance on the diagonal, has a Cholesky factor.
fn block_is_convex(sense: Sense, block: &[(Col, Col, f64)]) -> bool {
    let cols: Vec<Col> = block.iter().flat_map(|&(i, j, _)| [i, j]).sorted().dedup().collect();
    let sign = match sense {
        Sense::Minimize => 1.0,
        Sense::Maximize => -1.0,
    };
    let scale = block.iter().map(|&e| hessian_value(e).abs()).fold(0.0, f64::max);
    let mut matrix = DMatrix::from_diagonal_element(cols.len(), cols.len(), CONVEXITY_TOLERANCE * scale);
    for &entry in block {
        // Every column of the block is in `cols`, so its position is where it sorts.
        let (a, b) = (
            cols.partition_point(|&c| c < entry.0),
            cols.partition_point(|&c| c < entry.1),
        );
        let value = sign * hessian_value(entry);
        matrix[(a, b)] += value;
        if a != b {
            matrix[(b, a)] += value;
        }
    }
    Cholesky::new(matrix).is_some()
}

/// A union-find forest over the columns whose root is always the smallest column of its tree.
struct Roots {
    parent: TiVec<Col, Col>,
}

impl Roots {
    /// Every column its own tree.
    fn new(num_cols: usize) -> Self {
        Self {
            parent: (0..num_cols).map(Col::new).collect(),
        }
    }

    /// The root of `col`'s tree, halving the path on the way.
    ///
    /// # Panics
    ///
    /// When `col` is not a column of the forest.
    fn find(&mut self, mut col: Col) -> Col {
        while self.parent[col] != col {
            self.parent[col] = self.parent[self.parent[col]];
            col = self.parent[col];
        }
        col
    }

    /// Joins the trees of `a` and `b` under the smaller root.
    ///
    /// # Panics
    ///
    /// When `a` or `b` is not a column of the forest.
    fn join(&mut self, a: Col, b: Col) {
        let (a, b) = (self.find(a), self.find(b));
        let (low, high) = if a <= b { (a, b) } else { (b, a) };
        self.parent[high] = low;
    }
}

/// The entries of a [`QuadObjective`] as `(i, j, q)`; made by [`QuadObjective::entries`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct QuadEntries<'a> {
    inner: Copied<slice::Iter<'a, (Col, Col, f64)>>,
}

impl Iterator for QuadEntries<'_> {
    type Item = (Col, Col, f64);

    fn next(&mut self) -> Option<(Col, Col, f64)> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for QuadEntries<'_> {
    fn next_back(&mut self) -> Option<(Col, Col, f64)> {
        self.inner.next_back()
    }
}

impl ExactSizeIterator for QuadEntries<'_> {}

impl FusedIterator for QuadEntries<'_> {}

impl<'a> IntoIterator for &'a QuadObjective {
    type Item = (Col, Col, f64);
    type IntoIter = QuadEntries<'a>;

    fn into_iter(self) -> QuadEntries<'a> {
        self.entries()
    }
}

/// The lower triangle of a [`QuadObjective`]'s Hessian as `(column, row, value)`; made by
/// [`QuadObjective::hessian`].
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct HessianEntries<'a> {
    inner: QuadEntries<'a>,
}

impl Iterator for HessianEntries<'_> {
    type Item = (Col, Col, f64);

    fn next(&mut self) -> Option<(Col, Col, f64)> {
        self.inner.next().map(|e| (e.0, e.1, hessian_value(e)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for HessianEntries<'_> {
    fn next_back(&mut self) -> Option<(Col, Col, f64)> {
        self.inner.next_back().map(|e| (e.0, e.1, hessian_value(e)))
    }
}

impl ExactSizeIterator for HessianEntries<'_> {}

impl FusedIterator for HessianEntries<'_> {}
