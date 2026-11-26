// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rules over integer points that no single linear constraint states, each written from its one-sentence prose, for
//! constraint-handler tests to enforce and the reference models to filter by.
//!
//! A rule is handed a point as the values of the integer columns it is about, by position: position `k` of the slice
//! is the `k`-th entry of the point, and a rule names positions, not columns.

/// A rule over a point's integer values, naming positions in the point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rule {
    /// No two of the positions take the same value.
    AllDifferent(Vec<usize>),
    /// The values at the positions add up to an even number.
    EvenParity(Vec<usize>),
    /// Along the path of positions, no two neighbours are both nonzero.
    NoTwoAdjacent(Vec<usize>),
}

impl Rule {
    /// Whether `point` satisfies the rule.
    ///
    /// # Panics
    ///
    /// When the rule names a position beyond `point`.
    #[must_use]
    pub fn holds(&self, point: &[i64]) -> bool {
        match self {
            Self::AllDifferent(at) => all_different(at, point),
            Self::EvenParity(at) => even_parity(at, point),
            Self::NoTwoAdjacent(path) => no_two_adjacent(path, point),
        }
    }

    /// The positions the rule names, in order.
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        match self {
            Self::AllDifferent(at) | Self::EvenParity(at) | Self::NoTwoAdjacent(at) => at,
        }
    }
}

/// No two of the positions `at` take the same value in `point`.
///
/// # Panics
///
/// When a position is beyond `point`.
#[must_use]
pub fn all_different(at: &[usize], point: &[i64]) -> bool {
    at.iter()
        .enumerate()
        .all(|(i, &a)| at[i + 1..].iter().all(|&b| point[a] != point[b]))
}

/// The values of `point` at the positions `at` add up to an even number.
///
/// # Panics
///
/// When a position is beyond `point`.
#[must_use]
pub fn even_parity(at: &[usize], point: &[i64]) -> bool {
    at.iter().map(|&a| point[a]).sum::<i64>() % 2 == 0
}

/// Along `path`, no two neighbours are both nonzero in `point`.
///
/// # Panics
///
/// When a position is beyond `point`.
#[must_use]
pub fn no_two_adjacent(path: &[usize], point: &[i64]) -> bool {
    path.windows(2).all(|pair| point[pair[0]] == 0 || point[pair[1]] == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_each_rule_judges_hand_picked_points_as_its_prose_says() {
        let all = [0, 1, 2];
        assert!(all_different(&all, &[0, 2, 1]));
        assert!(!all_different(&all, &[1, 0, 1]));
        assert!(all_different(&[0, 2], &[1, 1, 2]));
        assert!(even_parity(&all, &[1, 1, 0]));
        assert!(!even_parity(&all, &[1, 1, 1]));
        assert!(even_parity(&all, &[3, 0, 1]));
        assert!(no_two_adjacent(&all, &[2, 0, 3]));
        assert!(!no_two_adjacent(&all, &[0, 1, 1]));
        // Positions 0 and 2 are not neighbours on the path 0, 1, 2, but are on 0, 2, 1.
        assert!(no_two_adjacent(&all, &[1, 0, 1]));
        assert!(!no_two_adjacent(&[0, 2, 1], &[1, 0, 1]));
        assert!(Rule::EvenParity(vec![0, 1]).holds(&[1, 1]));
        assert_eq!(Rule::AllDifferent(vec![1, 0]).positions(), [1, 0]);
    }
}
