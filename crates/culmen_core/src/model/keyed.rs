// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Handles keyed by domain keys, and the index sets families range over.

use core::borrow::Borrow;
use core::fmt;
use core::hash::Hash;
use core::iter::{Cloned, FusedIterator};
use core::ops::{Index, Range, RangeInclusive};
use core::slice;
use std::collections::{BTreeSet, HashMap, VecDeque, btree_set, vec_deque};

use derive_where::derive_where;
use typed_index_collections::TiSlice;

use crate::{Col, Row};

/// Handles keyed by domain keys, in insertion order: a family of variables ([`VarMap`]) or constraints
/// ([`ConstraintMap`]).
///
/// ```
/// use culmen_core::{Model, Var};
///
/// let products = ["cork", "glass"];
/// let mut m = Model::new();
/// let x = m.add_var_map(products, |p| Var::binary().name(format!("take[{p}]")));
/// assert_eq!(m.var_name(x["glass"]), Some("take[glass]"));
/// assert_eq!(x.get(&"tin"), None);
/// assert_eq!(x.keys().collect::<Vec<_>>(), [&"cork", &"glass"]);
/// for (product, take) in &x {
///     assert_eq!(x[*product], take);
/// }
/// ```
#[derive(Clone)]
pub struct Keyed<K, H> {
    entries: Vec<(K, H)>,
    positions: HashMap<K, usize>,
}

/// Variables keyed by domain keys.
pub type VarMap<K> = Keyed<K, Col>;

/// Constraints keyed by domain keys.
pub type ConstraintMap<K> = Keyed<K, Row>;

impl<K, H> Keyed<K, H> {
    /// The number of keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The keys with their handles, in insertion order.
    pub fn iter(&self) -> KeyedIter<'_, K, H> {
        KeyedIter(self.entries.iter())
    }

    /// The keys, in insertion order.
    pub fn keys(&self) -> Keys<'_, K, H> {
        Keys(self.entries.iter())
    }

    /// The handles, in insertion order.
    pub fn handles(&self) -> Handles<'_, K, H> {
        Handles(self.entries.iter())
    }
}

impl<'a, K, H: Copy> IntoIterator for &'a Keyed<K, H> {
    type Item = (&'a K, H);
    type IntoIter = KeyedIter<'a, K, H>;

    fn into_iter(self) -> KeyedIter<'a, K, H> {
        self.iter()
    }
}

/// The keys of a [`Keyed`] family with their handles, in insertion order; made by [`Keyed::iter`].
#[derive_where(Clone)]
#[derive_where(Debug; K: fmt::Debug, H: fmt::Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct KeyedIter<'a, K, H>(slice::Iter<'a, (K, H)>);

impl<'a, K, H: Copy> Iterator for KeyedIter<'a, K, H> {
    type Item = (&'a K, H);

    fn next(&mut self) -> Option<(&'a K, H)> {
        self.0.next().map(|(key, handle)| (key, *handle))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, H: Copy> DoubleEndedIterator for KeyedIter<'_, K, H> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(key, handle)| (key, *handle))
    }
}

impl<K, H: Copy> ExactSizeIterator for KeyedIter<'_, K, H> {}

impl<K, H: Copy> FusedIterator for KeyedIter<'_, K, H> {}

/// The keys of a [`Keyed`] family, in insertion order; made by [`Keyed::keys`].
#[derive_where(Clone)]
#[derive_where(Debug; K: fmt::Debug, H: fmt::Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Keys<'a, K, H>(slice::Iter<'a, (K, H)>);

impl<'a, K, H> Iterator for Keys<'a, K, H> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        self.0.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, H> DoubleEndedIterator for Keys<'_, K, H> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(key, _)| key)
    }
}

impl<K, H> ExactSizeIterator for Keys<'_, K, H> {}

impl<K, H> FusedIterator for Keys<'_, K, H> {}

/// The handles of a [`Keyed`] family, in insertion order; made by [`Keyed::handles`].
#[derive_where(Clone)]
#[derive_where(Debug; K: fmt::Debug, H: fmt::Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Handles<'a, K, H>(slice::Iter<'a, (K, H)>);

impl<K, H: Copy> Iterator for Handles<'_, K, H> {
    type Item = H;

    fn next(&mut self) -> Option<H> {
        self.0.next().map(|(_, handle)| *handle)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, H: Copy> DoubleEndedIterator for Handles<'_, K, H> {
    fn next_back(&mut self) -> Option<H> {
        self.0.next_back().map(|(_, handle)| *handle)
    }
}

impl<K, H: Copy> ExactSizeIterator for Handles<'_, K, H> {}

impl<K, H: Copy> FusedIterator for Handles<'_, K, H> {}

impl<K: Hash + Eq + Clone, H: Copy> Keyed<K, H> {
    /// An empty family.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            positions: HashMap::new(),
        }
    }

    /// Adds a key with its handle.
    ///
    /// # Panics
    ///
    /// When the family already holds `key`: an index set that repeats a member.
    #[track_caller]
    pub fn insert(&mut self, key: K, handle: H) {
        let position = self.entries.len();
        assert!(
            self.positions.insert(key.clone(), position).is_none(),
            "Keyed: a key is inserted twice"
        );
        self.entries.push((key, handle));
    }

    /// The handle of a key, `None` when the family does not hold it.
    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<H>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.positions.get(key).map(|&i| self.entries[i].1)
    }
}

impl<K> Keyed<K, Col> {
    /// Each key with its variable's value at a point, in insertion order.
    ///
    /// ```
    /// # use culmen_core::{Model, Var, Col};
    /// # use typed_index_collections::TiVec;
    /// let mut m = Model::new();
    /// let x = m.add_var_map(0..2, |_| Var::binary());
    /// let values: TiVec<Col, f64> = vec![0.0, 1.0].into();
    /// assert_eq!(x.values(&values).collect::<Vec<_>>(), [(&0, 0.0), (&1, 1.0)]);
    /// ```
    ///
    /// The iterator panics when a variable of the family has no value in `values`.
    pub fn values<'a>(&'a self, values: &'a TiSlice<Col, f64>) -> Values<'a, K> {
        Values {
            entries: self.entries.iter(),
            values,
        }
    }
}

/// The keys of a [`VarMap`] with their variables' values at a point, in insertion order; made by
/// [`Keyed::values`].
///
/// It panics when a variable of the family has no value in the point.
#[derive_where(Clone)]
#[derive_where(Debug; K: fmt::Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Values<'a, K> {
    entries: slice::Iter<'a, (K, Col)>,
    values: &'a TiSlice<Col, f64>,
}

impl<'a, K> Iterator for Values<'a, K> {
    type Item = (&'a K, f64);

    fn next(&mut self) -> Option<(&'a K, f64)> {
        self.entries.next().map(|(key, col)| (key, self.values[*col]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entries.size_hint()
    }
}

impl<K> DoubleEndedIterator for Values<'_, K> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.entries.next_back().map(|(key, col)| (key, self.values[*col]))
    }
}

impl<K> ExactSizeIterator for Values<'_, K> {}

impl<K> FusedIterator for Values<'_, K> {}

impl<K: Hash + Eq + Clone, H: Copy> Default for Keyed<K, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone, H: Copy> Index<K> for Keyed<K, H> {
    type Output = H;

    /// # Panics
    ///
    /// When the family does not hold `key`.
    #[track_caller]
    fn index(&self, key: K) -> &H {
        match self.positions.get(&key) {
            Some(&i) => &self.entries[i].1,
            None => panic!("Keyed: the family does not hold the key"),
        }
    }
}

impl<K: fmt::Debug, H: fmt::Debug> fmt::Debug for Keyed<K, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Keyed").field(&self.entries).finish()
    }
}

/// A set of indices a family ranges over, re-iterable as often as a nested loop needs.
///
/// Implemented for ranges, slices, arrays, `Vec`, `VecDeque`, `BTreeSet` and references to any of them; members
/// are produced by value (cloned from a collection), so `p in products` over a `Vec<usize>` binds `p: usize`.
/// Any cloneable iterator becomes one through [`set`].
pub trait IndexSet {
    /// The type of a member.
    type Item;

    /// The iterator over the members.
    type Members<'a>: Iterator<Item = Self::Item>
    where
        Self: 'a;

    /// The members, in the set's order.
    fn members(&self) -> Self::Members<'_>;
}

impl<T> IndexSet for Range<T>
where
    Range<T>: Iterator<Item = T> + Clone,
{
    type Item = T;
    type Members<'a>
        = Self
    where
        Self: 'a;

    fn members(&self) -> Self {
        self.clone()
    }
}

impl<T> IndexSet for RangeInclusive<T>
where
    RangeInclusive<T>: Iterator<Item = T> + Clone,
{
    type Item = T;
    type Members<'a>
        = Self
    where
        Self: 'a;

    fn members(&self) -> Self {
        self.clone()
    }
}

impl<T: Clone> IndexSet for [T] {
    type Item = T;
    type Members<'a>
        = Cloned<slice::Iter<'a, T>>
    where
        Self: 'a;

    fn members(&self) -> Cloned<slice::Iter<'_, T>> {
        self.iter().cloned()
    }
}

impl<T: Clone, const N: usize> IndexSet for [T; N] {
    type Item = T;
    type Members<'a>
        = Cloned<slice::Iter<'a, T>>
    where
        Self: 'a;

    fn members(&self) -> Cloned<slice::Iter<'_, T>> {
        self.iter().cloned()
    }
}

impl<T: Clone> IndexSet for Vec<T> {
    type Item = T;
    type Members<'a>
        = Cloned<slice::Iter<'a, T>>
    where
        Self: 'a;

    fn members(&self) -> Cloned<slice::Iter<'_, T>> {
        self.iter().cloned()
    }
}

impl<T: Clone> IndexSet for VecDeque<T> {
    type Item = T;
    type Members<'a>
        = Cloned<vec_deque::Iter<'a, T>>
    where
        Self: 'a;

    fn members(&self) -> Cloned<vec_deque::Iter<'_, T>> {
        self.iter().cloned()
    }
}

impl<T: Clone> IndexSet for BTreeSet<T> {
    type Item = T;
    type Members<'a>
        = Cloned<btree_set::Iter<'a, T>>
    where
        Self: 'a;

    fn members(&self) -> Cloned<btree_set::Iter<'_, T>> {
        self.iter().cloned()
    }
}

impl<S: IndexSet + ?Sized> IndexSet for &S {
    type Item = S::Item;
    type Members<'a>
        = S::Members<'a>
    where
        Self: 'a;

    fn members(&self) -> S::Members<'_> {
        (**self).members()
    }
}

/// A cloneable iterator as an [`IndexSet`], restarted from a clone for every pass.
#[derive(Clone, Debug)]
pub struct Members<I>(I);

/// Wraps a cloneable iterator as an [`IndexSet`].
///
/// ```
/// use culmen_core::{IndexSet, set};
/// let evens = set((0..10).filter(|i| i % 2 == 0));
/// assert_eq!(evens.members().count(), 5);
/// assert_eq!(evens.members().count(), 5);
/// ```
#[must_use]
pub fn set<I: IntoIterator>(iter: I) -> Members<I::IntoIter>
where
    I::IntoIter: Clone,
{
    Members(iter.into_iter())
}

impl<I: Iterator + Clone> IndexSet for Members<I> {
    type Item = I::Item;
    type Members<'a>
        = I
    where
        Self: 'a;

    fn members(&self) -> I {
        self.0.clone()
    }
}

/// Every pair `(a, b)` of members of two index sets, `a` outermost: the keys of a two-index family.
///
/// ```
/// use culmen_core::{IndexSet, product};
/// let pairs: Vec<_> = product(&(0..2), &['a', 'b']).collect();
/// assert_eq!(pairs, [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]);
/// ```
pub fn product<'a, A, B>(a: &'a A, b: &'a B) -> Product<'a, A, B>
where
    A: IndexSet + ?Sized,
    B: IndexSet + ?Sized,
    A::Item: Clone,
{
    Product {
        outer: a.members(),
        current: None,
        inner: b,
    }
}

/// Every pair of members of two index sets, the first set's outermost; made by [`product`].
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Product<'a, A: IndexSet + ?Sized + 'a, B: IndexSet + ?Sized + 'a> {
    outer: A::Members<'a>,
    /// The outer member being paired, and the inner members still to pair it with; `None` before the first pair.
    current: Option<(A::Item, B::Members<'a>)>,
    inner: &'a B,
}

impl<A, B> Iterator for Product<'_, A, B>
where
    A: IndexSet + ?Sized,
    B: IndexSet + ?Sized,
    A::Item: Clone,
{
    type Item = (A::Item, B::Item);

    fn next(&mut self) -> Option<(A::Item, B::Item)> {
        loop {
            if let Some((a, members)) = &mut self.current
                && let Some(b) = members.next()
            {
                return Some((a.clone(), b));
            }
            let a = self.outer.next()?;
            self.current = Some((a, self.inner.members()));
        }
    }
}

impl<'a, A, B> FusedIterator for Product<'a, A, B>
where
    A: IndexSet + ?Sized,
    B: IndexSet + ?Sized,
    A::Item: Clone,
    A::Members<'a>: FusedIterator,
{
}

impl<'a, A, B> Clone for Product<'a, A, B>
where
    A: IndexSet + ?Sized,
    B: IndexSet + ?Sized,
    A::Item: Clone,
    A::Members<'a>: Clone,
    B::Members<'a>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            outer: self.outer.clone(),
            current: self.current.clone(),
            inner: self.inner,
        }
    }
}

impl<'a, A, B> fmt::Debug for Product<'a, A, B>
where
    A: IndexSet + ?Sized,
    B: IndexSet + ?Sized + fmt::Debug,
    A::Item: fmt::Debug,
    A::Members<'a>: fmt::Debug,
    B::Members<'a>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Product")
            .field("outer", &self.outer)
            .field("current", &self.current)
            .field("inner", &self.inner)
            .finish()
    }
}
