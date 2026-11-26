// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a constraint handler judges, declares and does: the plain data of the
//! [`ConstraintHandler`](crate::ConstraintHandler) seam, and [`NodeActions`], the checked record of a node's actions
//! that every backend's enforcement and propagation contexts share, so every backend refuses the same actions the same
//! way.

use core::fmt;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::solve::context::check_constraint;
use crate::{ActionError, AddedRow, Bound, Bounds, Col, Constraint};

/// The handler's judgement of a point: whether it satisfies the handler's rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Verdict {
    /// The point satisfies the rule.
    Accept,
    /// The point violates the rule.
    Reject,
}

/// Whether an explanation named every bound its inference rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Explained {
    /// Every bound the inference rests on was named with
    /// [`ExplainContext::because`](crate::ExplainContext::because); the solver may learn from it.
    Complete,
    /// The handler cannot explain it; the solver treats the inferred bound as a decision. Always sound.
    Unknown,
}

/// The handler's own tag on an inferred bound, handed back to
/// [`ConstraintHandler::explain`](crate::ConstraintHandler::explain); its meaning is the handler's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Reason(pub u32);

/// The directions in which moving a column can turn a point the rule accepts into one it rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Lock {
    /// Lowering the column can violate the rule.
    Down,
    /// Raising the column can violate the rule.
    Up,
    /// Moving the column either way can violate the rule.
    Both,
}

/// The columns the rule involves, and how.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Locks {
    /// Every column, both ways: the safe default, which promises nothing.
    All,
    /// Only these columns, each in its directions; a column listed twice takes the union. A column not listed is one
    /// the rule ignores: moving it never turns an accepted point into a rejected one. Columns priced during the solve
    /// are always locked both ways.
    Only(Vec<(Col, Lock)>),
}

/// Which points the handler is asked to enforce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Candidates {
    /// Points every other constraint of the solver accepts, integrality included.
    Integral,
    /// Also relaxation points whose integer columns are fractional, before the solver branches on them.
    Fractional,
}

/// What the solver reads of a handler once, when a solve starts.
///
/// ```
/// use culmen_core::{Candidates, Col, Declaration, Lock, Locks};
///
/// assert_eq!(Declaration::default(), Declaration { locks: Locks::All, candidates: Candidates::Integral });
/// let precise = Declaration { locks: Locks::Only(vec![(Col::new(0), Lock::Up)]), ..Declaration::default() };
/// assert_eq!(precise.candidates, Candidates::Integral);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Declaration {
    /// The columns the rule involves; [`Locks::All`] by default.
    pub locks: Locks,
    /// Which points reach [`ConstraintHandler::enforce`](crate::ConstraintHandler::enforce);
    /// [`Candidates::Integral`] by default.
    pub candidates: Candidates,
}

impl Default for Declaration {
    fn default() -> Self {
        Self {
            locks: Locks::All,
            candidates: Candidates::Integral,
        }
    }
}

/// One child of a branching: the node's domain narrowed by `bounds`, with `constraints` holding in its subtree.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Child {
    /// Each column's interval, intersected with its bounds at the node; a column listed twice takes the intersection.
    pub bounds: Vec<(Col, Bounds)>,
    /// Linear constraints that hold in this child's subtree only.
    pub constraints: Vec<Constraint>,
}

/// The bound an explanation answers for.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Inferred {
    /// The column whose bound was tightened.
    pub col: Col,
    /// The end of its domain that was tightened.
    pub end: Bound,
    /// A bound at least as weak as the one tightened, in the column's units; explaining this one suffices.
    pub value: f64,
}

/// A kind of action at a node: what [`ActionError::Exclusive`] names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NodeAction {
    /// Adding constraints and tightening bounds.
    Reduce,
    /// Splitting the node into children.
    Branch,
    /// Closing the node.
    CutOff,
}

/// `reducing`, `branching` or `cutting off`.
impl fmt::Display for NodeAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Reduce => "reducing",
            Self::Branch => "branching",
            Self::CutOff => "cutting off",
        })
    }
}

/// The columns a handler may write: those its [`Declaration::locks`] involve, and every column priced during the
/// solve.
///
/// For backend authors: made once per solve, from the declaration and the number of columns of the program the solve
/// began with, and handed to each [`NodeActions`] and to the checks of a presolve context. A column [`Locks::Only`]
/// lists beyond the program is a priced one, involved anyway.
///
/// ```
/// use culmen_core::{Col, Involved, Lock, Locks};
///
/// let involved = Involved::new(&Locks::Only(vec![(Col::new(1), Lock::Down)]), 3);
/// assert!(!involved.contains(Col::new(0)));
/// assert!(involved.contains(Col::new(1)));
/// assert!(involved.contains(Col::new(3))); // priced during the solve
/// assert!(Involved::new(&Locks::All, 3).contains(Col::new(0)));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Involved {
    /// Whether each program column is listed; `None` when every column is involved. Shared, so each node's record
    /// takes it without copying.
    listed: Option<Arc<[bool]>>,
}

impl Involved {
    /// The columns `locks` involve, for a program of `program_cols` columns.
    #[must_use]
    pub fn new(locks: &Locks, program_cols: usize) -> Self {
        let listed = match locks {
            Locks::All => None,
            Locks::Only(cols) => {
                let mut listed = vec![false; program_cols];
                for &(col, _) in cols {
                    if let Some(entry) = listed.get_mut(col.index()) {
                        *entry = true;
                    }
                }
                Some(listed.into())
            }
        };
        Self { listed }
    }

    /// Whether a handler may write `col`: a column the locks list, or one beyond the program's, which a pricer added.
    #[must_use]
    pub fn contains(&self, col: Col) -> bool {
        self.listed
            .as_deref()
            .is_none_or(|listed| listed.get(col.index()).copied().unwrap_or(true))
    }

    /// Refuses a column a handler may not write, of a problem holding `num_cols` columns.
    ///
    /// # Errors
    ///
    /// [`ActionError::UnknownColumn`] for a column the problem does not hold, [`ActionError::NotInvolved`] for one
    /// this set does not.
    pub fn check(&self, col: Col, num_cols: usize) -> Result<(), ActionError> {
        if col.index() >= num_cols {
            Err(ActionError::UnknownColumn { col })
        } else if self.contains(col) {
            Ok(())
        } else {
            Err(ActionError::NotInvolved { col })
        }
    }
}

/// A child of a branching as [`NodeActions::branch`] checked it, not empty.
///
/// A row of [`NodeActions`], which is its seal: only a branching it accepted hands one out.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedChild {
    /// Each column the child narrows strictly, once, with its interval in the child: its bounds at the node
    /// intersected with every interval the child gives it. A column the child leaves as the node has it is not listed.
    pub bounds: Vec<(Col, Bounds)>,
    /// The child's constraints, as checked, in the order given.
    pub constraints: Vec<AddedRow>,
}

/// The actions a handler took in one call at a node, checked as it takes them: what every backend's enforcement and
/// propagation contexts record, and what the backend applies after the handler returns.
///
/// For backend authors, so every backend checks a handler's actions by one set of rules, which make the result a
/// backend reports for the call follow from what was recorded:
///
/// - **Exclusive kinds.** Reducing ([`NodeActions::add_constraint`], [`NodeActions::tighten`]), branching
///   ([`NodeActions::branch`]) and cutting off ([`NodeActions::cut_off`], [`NodeActions::cut_off_because`]) exclude
///   each other within one call. Reductions may follow each other; a branching is one call and follows nothing; a
///   cutoff may follow reductions or a branching and closes the node whatever they were, and nothing follows it. An
///   action refused by this rule is [`ActionError::Exclusive`].
/// - **All or nothing.** An action that is refused, by this rule or for its arguments, records nothing; one that is
///   accepted is recorded exactly as given, a branching as its checked children.
/// - **Known and involved columns.** Every column named is one the problem holds; a tightening and a child's bound
///   name only columns the [`Involved`] set holds ([`ActionError::NotInvolved`]).
/// - **Children that make progress.** A branching has at least one child ([`ActionError::NoChildren`]), and each child
///   narrows a bound of the node strictly or adds a constraint ([`ActionError::NoProgress`]). A child whose interval
///   for some column misses the node's bounds is empty: it is checked like the others, and left out of
///   [`NodeActions::children`].
///
/// ```
/// use culmen_core::{ActionError, Bounds, Child, Col, Involved, Locks, NodeAction, NodeActions, bounds};
///
/// let (x, y) = (Col::new(0), Col::new(1));
/// let node = |_: Col| Some(bounds!(0.0, 3.0));
/// let mut actions = NodeActions::new(&Involved::new(&Locks::All, 2), 2);
/// actions.tighten(x, bounds!(1.0, 3.0), None)?;
/// let split = vec![Child { bounds: vec![(y, bounds!(0.0, 1.0))], ..Child::default() }];
/// assert_eq!(
///     actions.branch(split, &node),
///     Err(ActionError::Exclusive { action: NodeAction::Branch, after: NodeAction::Reduce })
/// );
/// assert_eq!(actions.tightenings(), [(x, bounds!(1.0, 3.0), None)]);
/// assert_eq!(actions.children(), None);
/// assert_eq!(actions.action(), Some(NodeAction::Reduce));
/// # Ok::<(), ActionError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct NodeActions {
    involved: Involved,
    /// The number of columns the problem holds at the node, the priced ones included.
    num_cols: usize,
    constraints: Vec<AddedRow>,
    tightenings: Vec<(Col, Bounds, Option<Reason>)>,
    /// The non-empty children; `None` before a branching.
    children: Option<Vec<CheckedChild>>,
    /// The bounds the cutoff names; `None` before a cutoff.
    cutoff: Option<Vec<(Col, Bound)>>,
}

impl NodeActions {
    /// A record of nothing yet, for a node of a problem holding `num_cols` columns, writing only columns `involved`
    /// holds.
    #[must_use]
    pub fn new(involved: &Involved, num_cols: usize) -> Self {
        Self {
            involved: involved.clone(),
            num_cols,
            constraints: Vec::new(),
            tightenings: Vec::new(),
            children: None,
            cutoff: None,
        }
    }

    /// The kind of action that decides the node: a cutoff when one is recorded, otherwise the branching or the
    /// reductions recorded; `None` when nothing is.
    #[must_use]
    pub const fn action(&self) -> Option<NodeAction> {
        if self.cutoff.is_some() {
            Some(NodeAction::CutOff)
        } else if self.children.is_some() {
            Some(NodeAction::Branch)
        } else if !self.constraints.is_empty() || !self.tightenings.is_empty() {
            Some(NodeAction::Reduce)
        } else {
            None
        }
    }

    /// The constraints added, as checked, in order.
    #[must_use]
    pub fn constraints(&self) -> &[AddedRow] {
        &self.constraints
    }

    /// The tightenings, as given, in order: the column, the interval it is narrowed to, and the handler's reason.
    #[must_use]
    pub fn tightenings(&self) -> &[(Col, Bounds, Option<Reason>)] {
        &self.tightenings
    }

    /// The branching's children that are not empty, in order; `None` when no branching is recorded, and empty when
    /// every child is.
    #[must_use]
    pub fn children(&self) -> Option<&[CheckedChild]> {
        self.children.as_deref()
    }

    /// The bounds the cutoff names, as given, empty for a plain [`NodeActions::cut_off`]; `None` when the node is not
    /// cut off.
    #[must_use]
    pub fn cutoff(&self) -> Option<&[(Col, Bound)]> {
        self.cutoff.as_deref()
    }

    /// Refuses an action of kind `action` that the actions recorded exclude.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after a cutoff, after a branching, and for a branching after reductions.
    fn admit(&self, action: NodeAction) -> Result<(), ActionError> {
        match (self.action(), action) {
            (None, _) | (Some(NodeAction::Reduce), NodeAction::Reduce | NodeAction::CutOff) => Ok(()),
            (Some(NodeAction::Branch), NodeAction::CutOff) => Ok(()),
            (Some(after), action) => Err(ActionError::Exclusive { action, after }),
        }
    }

    /// Refuses a column the problem does not hold at the node.
    ///
    /// # Errors
    ///
    /// [`ActionError::UnknownColumn`].
    const fn known(&self, col: Col) -> Result<(), ActionError> {
        if col.index() < self.num_cols {
            Ok(())
        } else {
            Err(ActionError::UnknownColumn { col })
        }
    }

    /// Records a constraint that holds for the rest of the solve.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after a branching or a cutoff; the errors of
    /// [`check_constraint`](crate::context::check_constraint).
    pub fn add_constraint(&mut self, constraint: &Constraint) -> Result<(), ActionError> {
        self.admit(NodeAction::Reduce)?;
        let row = check_constraint(constraint, self.num_cols)?;
        self.constraints.push(row);
        Ok(())
    }

    /// Records narrowing `col` to `bounds` at the node, with the handler's `reason`.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after a branching or a cutoff; [`ActionError::UnknownColumn`] or
    /// [`ActionError::NotInvolved`] for `col`.
    pub fn tighten(&mut self, col: Col, bounds: Bounds, reason: Option<Reason>) -> Result<(), ActionError> {
        self.admit(NodeAction::Reduce)?;
        self.involved.check(col, self.num_cols)?;
        self.tightenings.push((col, bounds, reason));
        Ok(())
    }

    /// Records a branching into `children`, each checked against the node's bounds `node` gives, `None` for a
    /// column the node lacks.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after any other action; [`ActionError::NoChildren`]; for the first child at
    /// fault, [`ActionError::UnknownColumn`] or [`ActionError::NotInvolved`] for a bound, the errors of
    /// [`check_constraint`](crate::context::check_constraint) for a constraint, or [`ActionError::NoProgress`].
    pub fn branch(&mut self, children: Vec<Child>, node: &dyn Fn(Col) -> Option<Bounds>) -> Result<(), ActionError> {
        self.admit(NodeAction::Branch)?;
        if children.is_empty() {
            return Err(ActionError::NoChildren);
        }
        let mut checked = Vec::with_capacity(children.len());
        for (index, child) in children.iter().enumerate() {
            if let Some(child) = self.check_child(index, child, node)? {
                checked.push(child);
            }
        }
        self.children = Some(checked);
        Ok(())
    }

    /// The child at position `index` as checked, `None` when it is empty.
    ///
    /// # Errors
    ///
    /// As [`NodeActions::branch`] for one child.
    fn check_child(
        &self,
        index: usize,
        child: &Child,
        node: &dyn Fn(Col) -> Option<Bounds>,
    ) -> Result<Option<CheckedChild>, ActionError> {
        let mut domain: Vec<(Col, Bounds)> = Vec::with_capacity(child.bounds.len());
        let mut empty = false;
        for &(col, interval) in &child.bounds {
            self.involved.check(col, self.num_cols)?;
            let at_node = node(col).ok_or(ActionError::UnknownColumn { col })?;
            let place = domain.iter().position(|&(c, _)| c == col);
            let current = place.map_or(at_node, |i| domain[i].1);
            match (intersect(current, interval), place) {
                (None, _) => empty = true,
                (Some(narrowed), Some(i)) => domain[i].1 = narrowed,
                (Some(narrowed), None) => domain.push((col, narrowed)),
            }
        }
        let constraints = child
            .constraints
            .iter()
            .map(|constraint| check_constraint(constraint, self.num_cols))
            .collect::<Result<Vec<_>, _>>()?;
        // An interval is never wider than the node's bounds it was intersected with, so one that differs is narrower.
        domain.retain(|&(col, bounds)| node(col) != Some(bounds));
        if !empty && domain.is_empty() && constraints.is_empty() {
            return Err(ActionError::NoProgress { child: index });
        }
        Ok((!empty).then_some(CheckedChild {
            bounds: domain,
            constraints,
        }))
    }

    /// Records that the node holds no point the program and the rule accept, naming no bounds; nothing follows it. A
    /// second cutoff changes nothing.
    pub fn cut_off(&mut self) {
        self.cutoff.get_or_insert_with(Vec::new);
    }

    /// Records that the node holds no point the program and the rule accept, because of the bounds `reasons` name at
    /// the node; nothing follows it.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after a cutoff; [`ActionError::UnknownColumn`] for a column of `reasons`.
    pub fn cut_off_because(&mut self, reasons: &[(Col, Bound)]) -> Result<(), ActionError> {
        self.admit(NodeAction::CutOff)?;
        for &(col, _) in reasons {
            self.known(col)?;
        }
        self.cutoff = Some(reasons.to_vec());
        Ok(())
    }
}

/// The intersection of two intervals; `None` when it is empty.
fn intersect(a: Bounds, b: Bounds) -> Option<Bounds> {
    Bounds::new(a.lower().max(b.lower()), a.upper().min(b.upper())).ok()
}
