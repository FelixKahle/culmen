// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The one implementation of a set of the variants of a fieldless enum, for `ModelClasses`, `ProgramFeatures` and
//! `Events`: every such set has the same methods, operators, iterator, conversions and formatting.

/// Implements a set over a fieldless element enum for a `Copy` newtype `$set` over the bits `$bits`, and its iterator
/// `$iter`, a newtype over `$set`.
///
/// The element enum provides `const ALL: [Self; N]`, every variant in the order of their bits, and
/// `const fn bit(self) -> $bits`, its one bit; `$noun` is the plural the docs use, and `$debug` the format of one
/// element in the set's `Debug`, `"{}"` or `"{:?}"`.
macro_rules! element_set {
    ($set:ident, $iter:ident, $element:ident, $bits:ty, $noun:literal, $debug:literal) => {
        impl $set {
            #[doc = concat!("No ", $noun, ".")]
            pub const NONE: Self = Self(0);
            #[doc = concat!("Every one of the ", $noun, ".")]
            pub const ALL: Self = Self::of(&$element::ALL);

            #[doc = concat!("The set of the given ", $noun, ".")]
            #[must_use]
            pub const fn of(elements: &[$element]) -> Self {
                let mut bits = 0;
                let mut i = 0;
                while i < elements.len() {
                    bits |= elements[i].bit();
                    i += 1;
                }
                Self(bits)
            }

            #[doc = concat!("The ", $noun, " in either set.")]
            #[must_use]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            #[doc = concat!("The ", $noun, " in both sets.")]
            #[must_use]
            pub const fn intersection(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }

            #[doc = concat!("The ", $noun, " in this set and not in `other`.")]
            #[must_use]
            pub const fn difference(self, other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            #[doc = concat!("The ", $noun, " not in this set: [`", stringify!($set), "::ALL`] without them.")]
            #[must_use]
            pub const fn complement(self) -> Self {
                Self::ALL.difference(self)
            }

            #[doc = concat!("Whether `element` is in the set.")]
            #[must_use]
            pub const fn contains(self, element: $element) -> bool {
                self.0 & element.bit() != 0
            }

            #[doc = concat!("Whether every one of the ", $noun, " of `other` is in this set.")]
            #[must_use]
            pub const fn contains_all(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// Whether the set is empty.
            #[must_use]
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            #[doc = concat!("The number of ", $noun, " in the set.")]
            #[must_use]
            pub const fn len(self) -> usize {
                self.0.count_ones() as usize
            }

            #[doc = concat!("Each element of the set, in the order of [`", stringify!($element), "`]'s variants.")]
            pub const fn iter(self) -> $iter {
                $iter(self)
            }
        }

        impl From<$element> for $set {
            fn from(element: $element) -> Self {
                Self(element.bit())
            }
        }

        /// The set of both elements, so a set is written `a | b` as its elements are named.
        impl core::ops::BitOr for $element {
            type Output = $set;
            fn bitor(self, other: Self) -> $set {
                $set::from(self).union(other.into())
            }
        }

        impl core::ops::BitOr<$element> for $set {
            type Output = Self;
            fn bitor(self, element: $element) -> Self {
                self.union(element.into())
            }
        }

        impl core::ops::BitOrAssign<$element> for $set {
            fn bitor_assign(&mut self, element: $element) {
                *self = self.union(element.into());
            }
        }

        impl core::ops::Sub<$element> for $set {
            type Output = Self;
            fn sub(self, element: $element) -> Self {
                self.difference(element.into())
            }
        }

        impl core::ops::BitOr for $set {
            type Output = Self;
            fn bitor(self, other: Self) -> Self {
                self.union(other)
            }
        }

        impl core::ops::BitOrAssign for $set {
            fn bitor_assign(&mut self, other: Self) {
                *self = self.union(other);
            }
        }

        impl core::ops::BitAnd for $set {
            type Output = Self;
            fn bitand(self, other: Self) -> Self {
                self.intersection(other)
            }
        }

        impl core::ops::BitAndAssign for $set {
            fn bitand_assign(&mut self, other: Self) {
                *self = self.intersection(other);
            }
        }

        impl core::ops::Sub for $set {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                self.difference(other)
            }
        }

        impl core::ops::SubAssign for $set {
            fn sub_assign(&mut self, other: Self) {
                *self = self.difference(other);
            }
        }

        impl core::ops::Not for $set {
            type Output = Self;
            fn not(self) -> Self {
                self.complement()
            }
        }

        impl FromIterator<$element> for $set {
            fn from_iter<I: IntoIterator<Item = $element>>(iter: I) -> Self {
                iter.into_iter().map(Self::from).fold(Self::NONE, Self::union)
            }
        }

        impl Extend<$element> for $set {
            fn extend<I: IntoIterator<Item = $element>>(&mut self, iter: I) {
                *self = iter.into_iter().map(Self::from).fold(*self, Self::union);
            }
        }

        impl IntoIterator for $set {
            type Item = $element;
            type IntoIter = $iter;

            fn into_iter(self) -> $iter {
                self.iter()
            }
        }

        impl From<Vec<$element>> for $set {
            fn from(elements: Vec<$element>) -> Self {
                elements.into_iter().collect()
            }
        }

        impl From<$set> for Vec<$element> {
            fn from(set: $set) -> Self {
                set.iter().collect()
            }
        }

        impl core::fmt::Debug for $set {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                use itertools::Itertools as _;
                let elements = self.iter().map(|element| format!($debug, element));
                write!(f, "{}({})", stringify!($set), elements.format(" | "))
            }
        }

        /// `{a, b}`, each element as it displays; `{}` for none.
        impl core::fmt::Display for $set {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                use itertools::Itertools as _;
                write!(f, "{{{}}}", self.iter().format(", "))
            }
        }

        impl $iter {
            /// Removes `element` from the remaining set and returns it.
            fn remove(&mut self, element: $element) -> $element {
                (self.0).0 &= !element.bit();
                element
            }
        }

        impl Iterator for $iter {
            type Item = $element;

            fn next(&mut self) -> Option<$element> {
                let rest = self.0;
                $element::ALL
                    .into_iter()
                    .find(|&element| rest.contains(element))
                    .map(|element| self.remove(element))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.0.len(), Some(self.0.len()))
            }
        }

        impl DoubleEndedIterator for $iter {
            fn next_back(&mut self) -> Option<$element> {
                let rest = self.0;
                $element::ALL
                    .into_iter()
                    .rfind(|&element| rest.contains(element))
                    .map(|element| self.remove(element))
            }
        }

        impl ExactSizeIterator for $iter {}

        impl core::iter::FusedIterator for $iter {}

        const _: () = assert!(
            $element::ALL.len() <= <$bits>::BITS as usize,
            concat!(stringify!($set), ": more elements than bits")
        );
    };
}

pub(crate) use element_set;
