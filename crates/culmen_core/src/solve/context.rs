// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The solver-neutral callback contexts: what each event lets a callback read and do.
//!
//! A [`Backend`](crate::Backend) names one concrete type per observational event kind, and a backend with a search
//! capability ([`LazyConstraints`](crate::LazyConstraints), [`UserCuts`](crate::UserCuts),
//! [`Heuristics`](crate::Heuristics), [`Prices`](crate::Prices), [`CustomConstraints`](crate::CustomConstraints))
//! names the contexts of its handler, each implementing the matching trait here. Every context is a [`Context`]: it reports where the solve stands. Code generic over
//! backends uses the traits; code written for one backend gets that backend's type, with whatever the solver offers
//! beyond the traits as its inherent methods. A backend that cannot deliver an observational event names
//! [`Unsupported`], which has no values, so a callback for that event can never run.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use typed_index_collections::TiSlice;

use crate::program::{BadTerm, bad_term};
use crate::{
    Bound, Bounds, Child, Col, Constraint, Inferred, Integrality, InvalidBounds, LogLevel, NewCol, NodeAction,
    Progress, Reason, Row, Sense,
};

/// What every context reports: where the solve stands when the event is delivered.
///
/// Every context trait below has it as a supertrait, so code that only needs the progress, such as a deadline check,
/// takes any context as `&impl Context`.
pub trait Context {
    /// Where the solve stands; for a log chunk often only the elapsed time.
    fn progress(&self) -> Progress;
}

/// A chunk of the solver's log. A line may arrive in several chunks; [`Watch`](crate::Watch) joins them.
pub trait LogContext: Context {
    /// The text as the solver wrote it, line breaks included.
    fn text(&self) -> &str;
    /// How much the chunk matters.
    fn level(&self) -> LogLevel;
}

/// A progress report the solver issues at its own pace; its [`Context::progress`] is the report.
pub trait ProgressContext: Context {}

/// A frequent poll at which the callback may stop the solve.
pub trait PollContext: Context {
    /// Asks the solver to stop as soon as it can. The solve then ends as
    /// [`StopReason::Interrupted`](crate::StopReason::Interrupted), with the best solution found.
    fn interrupt(&mut self);
}

/// A feasible solution the solver found: any one, or a new incumbent.
pub trait SolutionContext: Context {
    /// The solution's objective value, offset included.
    fn objective(&self) -> f64;
    /// The solution's value of every column the problem holds when the event is delivered, the columns priced so far
    /// in this solve included.
    fn values(&self) -> &TiSlice<Col, f64>;
}

/// A candidate solution the callback may cut off with lazy constraints before the solver accepts it.
pub trait LazyContext: Context {
    /// The candidate's objective value, offset included.
    fn objective(&self) -> f64;
    /// The candidate's value of every column the problem holds when the event is delivered, the columns priced so
    /// far in this solve included.
    fn candidate(&self) -> &TiSlice<Col, f64>;
    /// Adds a constraint the candidate may violate; it holds for the rest of the search.
    ///
    /// # Errors
    ///
    /// [`ActionError`] when the constraint names an unknown column, has a non-finite number, or the solver
    /// refuses it.
    fn add_lazy(&mut self, constraint: Constraint) -> Result<(), ActionError>;
}

/// A node relaxation the callback may cut off with user cuts.
pub trait CutContext: Context {
    /// The node relaxation's value of every column the problem holds when the event is delivered, the columns priced
    /// so far in this solve included.
    fn relaxation(&self) -> &TiSlice<Col, f64>;
    /// Adds a cut: a constraint every integer-feasible point satisfies.
    ///
    /// # Errors
    ///
    /// [`ActionError`] when the cut names an unknown column, has a non-finite number, or the solver refuses it.
    fn add_cut(&mut self, cut: Constraint) -> Result<(), ActionError>;
}

/// A point at which the callback may hand the solver a heuristic solution.
///
/// A solver that completes a partial solution offers it as an inherent method of its own context, since not every
/// solver honours one.
pub trait HeuristicContext: Context {
    /// Offers a value for every column the problem holds when the event is delivered, the columns priced so far in
    /// this solve included. The solver checks it and keeps it if it is feasible and better.
    ///
    /// # Errors
    ///
    /// [`ActionError`] when `values` does not hold one finite value per column, or the solver refuses it.
    fn supply(&mut self, values: &TiSlice<Col, f64>) -> Result<(), ActionError>;
}

/// What the node LP is when the solver asks for columns, and so what a [`PriceContext`]'s duals mean.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PriceMode {
    /// The node LP is optimal over the columns it holds; a column with an improving reduced cost may improve it.
    ReducedCost {
        /// The node LP's objective value in the program's sense, offset and quadratic part included.
        objective: f64,
    },
    /// The node LP is infeasible over the columns it holds; a column the Farkas multipliers favour may make it
    /// feasible.
    Farkas,
}

/// A node LP the callback may improve, or make feasible, by adding columns: pricing.
///
/// The duals are those of the program's rows. A priced column has entries only in program rows, so its coefficient
/// in every lazy constraint and cut is zero, and their duals do not change its reduced cost; every lazy constraint
/// and cut must therefore hold for every column the callback may price, each entering it with coefficient zero.
///
/// The node LP's value is a bound only when pricing is exact: whenever a column the problem does not hold yet
/// improves the node LP, or makes it feasible, the callback adds at least one such column or calls
/// [`PriceContext::branch_early`]. The solver prices the columns it holds itself, so a callback never adds a column
/// the problem already holds unless it means a second copy.
pub trait PriceContext: Context {
    /// Whether the node LP is optimal, with its objective, or infeasible.
    fn mode(&self) -> PriceMode;
    /// Whether the program minimizes or maximizes.
    fn sense(&self) -> Sense;
    /// The node LP's multiplier of every program row, in the convention [`PriceContext::reduced_cost`] states.
    ///
    /// In [`PriceMode::ReducedCost`] it is the rate at which the node LP's optimum changes as the row's active bound
    /// rises, the sign convention of [`Duals`](crate::Duals); in [`PriceMode::Farkas`] it is the row's Farkas
    /// multiplier, whatever the sense.
    fn duals(&self) -> &TiSlice<Row, f64>;
    /// The magnitude up to which the solver counts a reduced cost as zero.
    fn tolerance(&self) -> f64;
    /// Adds a column to the search and returns its id, which follows every column the problem holds, the ones priced
    /// before included. The column stays for the rest of the solve, and the [`Solver`](crate::Solver) appends it to
    /// its program when the solve ends.
    ///
    /// A column equal to one the problem holds is not refused: it is a second copy, which may be meant, as two
    /// identical trucks are. A column branching has fixed, added again, escapes that branching, so a callback never
    /// regenerates one.
    ///
    /// # Errors
    ///
    /// [`ActionError::NotFinite`] for a cost that is NaN or infinite, [`ActionError::Integrality`] for a semi column
    /// or an integrality the backend's classes do not allow beside the program's, [`ActionError::ExcludesZero`] for
    /// bounds that do not hold zero, [`ActionError::UnknownRow`] or [`ActionError::NotFiniteEntry`] for an entry at
    /// fault; [`ActionError::Refused`] when the solver refuses it.
    fn add_col(&mut self, column: NewCol) -> Result<Col, ActionError>;
    /// Ends pricing at this node after this round and has the solver branch; the node LP's value is then no bound
    /// unless [`PriceContext::report_bound`] gives one.
    fn branch_early(&mut self);
    /// Reports a bound on the optimal objective in this node's subtree, over every column the program may price, in
    /// the program's sense with its offset: a lower bound when minimizing, an upper one when maximizing. The last
    /// bound reported in a round counts.
    ///
    /// # Errors
    ///
    /// [`ActionError::NotFiniteBound`] for a NaN or infinite bound; [`ActionError::Refused`] in
    /// [`PriceMode::Farkas`], or when the solver refuses it.
    fn report_bound(&mut self, bound: f64) -> Result<(), ActionError>;

    /// The column's reduced cost at this node: `cost − Σ duals[r]·a_r` in [`PriceMode::ReducedCost`], and
    /// `−Σ duals[r]·a_r` in [`PriceMode::Farkas`], where the cost counts as zero.
    ///
    /// # Panics
    ///
    /// When an entry names a row beyond [`PriceContext::duals`].
    fn reduced_cost(&self, column: &NewCol) -> f64 {
        let duals = self.duals();
        let priced: f64 = column.entries.iter().map(|&(row, a)| duals[row] * a).sum();
        match self.mode() {
            PriceMode::ReducedCost { .. } => column.column.cost - priced,
            PriceMode::Farkas => -priced,
        }
    }

    /// Whether adding the column could improve the node LP, or make it feasible, beyond [`PriceContext::tolerance`].
    ///
    /// With `d` the [`PriceContext::reduced_cost`], taken as `t = d` in [`PriceMode::Farkas`] or when minimizing and
    /// `t = −d` when maximizing, it is whether the column may rise from zero with `t < −tolerance`, or fall below
    /// zero with `t > tolerance`.
    ///
    /// # Panics
    ///
    /// As [`PriceContext::reduced_cost`].
    fn improves(&self, column: &NewCol) -> bool {
        let d = self.reduced_cost(column);
        let t = match (self.mode(), self.sense()) {
            (PriceMode::ReducedCost { .. }, Sense::Maximize) => -d,
            (PriceMode::ReducedCost { .. }, Sense::Minimize) | (PriceMode::Farkas, _) => d,
        };
        let (tolerance, bounds) = (self.tolerance(), column.column.bounds);
        (bounds.upper() > 0.0 && t < -tolerance) || (bounds.lower() < 0.0 && t > tolerance)
    }
}

/// A point a [`ConstraintHandler`](crate::ConstraintHandler) judges.
pub trait CheckContext: Context {
    /// The point's objective value, offset included.
    fn objective(&self) -> f64;
    /// The point's value of every column the problem holds when the event is delivered, the columns priced so far in
    /// this solve included.
    fn candidate(&self) -> &TiSlice<Col, f64>;
}

/// A node of the search: its domain, and the actions that narrow it or close it.
///
/// Bounds it reports are those when the call began; the actions take effect after the handler returns. Reducing,
/// branching and cutting off exclude each other within one call, and a cutoff ends the call: an action they exclude
/// is refused with [`ActionError::Exclusive`].
pub trait NodeContext: Context {
    /// The column's bounds at this node when the call began; `None` for a column the problem lacks.
    fn bounds(&self, col: Col) -> Option<Bounds>;
    /// Narrows the column to `bounds` at this node and below; at the root, for the rest of the solve. With a
    /// reason, the solver may later ask [`ConstraintHandler::explain`](crate::ConstraintHandler::explain) which bounds
    /// it rested on.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after a branching or a cutoff; [`ActionError::UnknownColumn`],
    /// [`ActionError::NotInvolved`] for a column the declaration does not involve; [`ActionError::Refused`] when the
    /// solver cannot change the column.
    fn tighten(&mut self, col: Col, bounds: Bounds, reason: Option<Reason>) -> Result<(), ActionError>;
    /// Closes the node: it holds no point the program and the rule accept. Ends the call's actions; a second cutoff
    /// changes nothing.
    fn cut_off(&mut self);
    /// Closes the node, naming the bounds that make it infeasible, so the solver may learn a conflict from them:
    /// `(col, end)` is the column's `end` bound at this node. Ends the call's actions.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after a cutoff; [`ActionError::UnknownColumn`].
    fn cut_off_because(&mut self, reasons: &[(Col, Bound)]) -> Result<(), ActionError>;
}

/// A point at a node that [`ConstraintHandler::check`](crate::ConstraintHandler::check) rejected, to resolve.
pub trait EnforceContext: CheckContext + NodeContext {
    /// Adds a constraint every point the program and the rule accept satisfies; it holds for the rest of the solve.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after a branching or a cutoff; the errors of [`check_constraint`];
    /// [`ActionError::Refused`] when the solver refuses it.
    fn add_constraint(&mut self, constraint: Constraint) -> Result<(), ActionError>;
    /// Splits the node into `children`, whose union holds every point of the node the program and the rule accept,
    /// and each of which narrows a bound of the node or adds a constraint.
    ///
    /// # Errors
    ///
    /// [`ActionError::Exclusive`] after any other action; [`ActionError::NoChildren`]; for the first child at fault,
    /// [`ActionError::UnknownColumn`] or [`ActionError::NotInvolved`] for a bound, the errors of
    /// [`check_constraint`] for a constraint, or [`ActionError::NoProgress`].
    fn branch(&mut self, children: Vec<Child>) -> Result<(), ActionError>;
}

/// A node's domain to narrow, without a point.
pub trait PropagateContext: NodeContext {}

/// A bound a [`ConstraintHandler`](crate::ConstraintHandler) tightened with a [`Reason`], to explain.
pub trait ExplainContext: Context {
    /// The bound to explain.
    fn inferred(&self) -> Inferred;
    /// The tag the handler gave the tightening.
    fn reason(&self) -> Reason;
    /// The column's bounds just before the tightening; `None` for a column the problem lacks.
    fn bounds_before(&self, col: Col) -> Option<Bounds>;
    /// Names the column's `end` bound, as it was just before the tightening, as part of the explanation. The names
    /// count only when the handler answers [`Explained::Complete`](crate::Explained::Complete).
    ///
    /// # Errors
    ///
    /// [`ActionError::UnknownColumn`].
    fn because(&mut self, col: Col, end: Bound) -> Result<(), ActionError>;
}

/// The program's global domain before the search, for a [`ConstraintHandler`](crate::ConstraintHandler) to narrow.
pub trait PresolveContext: Context {
    /// The column's global bounds now; `None` for a column the problem lacks.
    fn bounds(&self, col: Col) -> Option<Bounds>;
    /// Narrows the column's global bounds; every point the program and the rule accept must keep satisfying them.
    ///
    /// # Errors
    ///
    /// [`ActionError::UnknownColumn`], [`ActionError::NotInvolved`] for a column the declaration does not involve;
    /// [`ActionError::Refused`] when the solver cannot change the column.
    fn tighten(&mut self, col: Col, bounds: Bounds) -> Result<(), ActionError>;
    /// Adds a constraint every point the program and the rule accept satisfies.
    ///
    /// # Errors
    ///
    /// The errors of [`check_constraint`]; [`ActionError::Refused`] when the solver refuses it.
    fn add_constraint(&mut self, constraint: Constraint) -> Result<(), ActionError>;
    /// Declares that no point satisfies the program and the rule.
    fn infeasible(&mut self);
}

/// The context of an observational event a backend cannot deliver, and its native event when it has none. It has no
/// values, so a callback for it never runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unsupported {}

impl Context for Unsupported {
    fn progress(&self) -> Progress {
        match *self {}
    }
}

impl LogContext for Unsupported {
    fn text(&self) -> &str {
        match *self {}
    }
    fn level(&self) -> LogLevel {
        match *self {}
    }
}

impl ProgressContext for Unsupported {}

impl PollContext for Unsupported {
    fn interrupt(&mut self) {
        match *self {}
    }
}

impl SolutionContext for Unsupported {
    fn objective(&self) -> f64 {
        match *self {}
    }
    fn values(&self) -> &TiSlice<Col, f64> {
        match *self {}
    }
}

/// Why a callback's action was refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ActionError {
    /// A full solution holds the wrong number of values.
    Length {
        /// The number of columns.
        expected: usize,
        /// The number of values given.
        found: usize,
    },
    /// An entry names a column the problem does not have.
    UnknownColumn {
        /// The column named.
        col: Col,
    },
    /// A value or coefficient is NaN or infinite.
    NotFinite {
        /// The column it belongs to.
        col: Col,
    },
    /// A constraint's bounds, after its constant moved into them, are invalid.
    Bounds(InvalidBounds),
    /// An entry of a new column names a row the program does not have.
    UnknownRow {
        /// The row named.
        row: Row,
    },
    /// A coefficient of a new column is NaN or infinite, as given or once its repeated entries were added up.
    NotFiniteEntry {
        /// The row of the coefficient.
        row: Row,
        /// The new column.
        col: Col,
    },
    /// A priced column's bounds do not hold zero, so leaving it out would not be the same as fixing it at zero.
    ExcludesZero {
        /// The new column.
        col: Col,
        /// Its bounds.
        bounds: Bounds,
    },
    /// A priced column's integrality is semi, or one the backend's classes do not allow beside the program's.
    Integrality {
        /// The new column.
        col: Col,
        /// Its integrality.
        integrality: Integrality,
    },
    /// A reported bound is NaN or infinite.
    NotFiniteBound {
        /// The bound reported.
        bound: f64,
    },
    /// A tightening or a child's bound names a column the constraint handler's
    /// [`Declaration`](crate::Declaration) does not involve.
    NotInvolved {
        /// The column named.
        col: Col,
    },
    /// An action the actions recorded before it in the same call exclude: reducing (constraints, tightenings),
    /// branching and cutting off exclude each other, a branching is one call, and a cutoff ends the call.
    Exclusive {
        /// The kind of the action refused.
        action: NodeAction,
        /// The kind recorded before that excludes it.
        after: NodeAction,
    },
    /// A branching without children.
    NoChildren,
    /// A child of a branching that narrows no bound of the node and adds no constraint.
    NoProgress {
        /// The child's position among the children given.
        child: usize,
    },
    /// The solver refused the action; the text is its reason.
    Refused(String),
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { expected, found } => {
                write!(f, "callback action: {found} values given for {expected} columns")
            }
            Self::UnknownColumn { col } => write!(f, "callback action: {col:?} is not a column of the problem"),
            Self::NotFinite { col } => write!(f, "callback action: the number for {col:?} is not finite"),
            Self::Bounds(error) => write!(f, "callback action: {error}"),
            Self::UnknownRow { row } => write!(f, "callback action: {row:?} is not a row of the program"),
            Self::NotFiniteEntry { row, col } => {
                write!(
                    f,
                    "callback action: the coefficient of {col:?} in {row:?} is not finite"
                )
            }
            Self::ExcludesZero { col, bounds } => write!(
                f,
                "callback action: the priced column {col:?} has the bounds {bounds}, which do not hold zero"
            ),
            Self::Integrality { col, integrality } => write!(
                f,
                "callback action: the priced column {col:?} is {integrality:?}, which pricing or the backend does not \
                 allow"
            ),
            Self::NotFiniteBound { bound } => write!(f, "callback action: the bound {bound} is not finite"),
            Self::NotInvolved { col } => write!(
                f,
                "callback action: {col:?} is not a column the constraint handler's declaration involves"
            ),
            Self::Exclusive { action, after } => {
                write!(f, "callback action: {action} is refused after {after} in the same call")
            }
            Self::NoChildren => write!(f, "callback action: a branching needs at least one child"),
            Self::NoProgress { child } => write!(
                f,
                "callback action: child {child} of the branching narrows no bound of the node and adds no constraint"
            ),
            Self::Refused(reason) => write!(f, "callback action: the solver refused it: {reason}"),
        }
    }
}

/// The source of [`ActionError::Bounds`] is its [`InvalidBounds`]; every other variant has none.
impl core::error::Error for ActionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Bounds(error) => Some(error),
            Self::Length { .. }
            | Self::UnknownColumn { .. }
            | Self::NotFinite { .. }
            | Self::UnknownRow { .. }
            | Self::NotFiniteEntry { .. }
            | Self::ExcludesZero { .. }
            | Self::Integrality { .. }
            | Self::NotFiniteBound { .. }
            | Self::NotInvolved { .. }
            | Self::Exclusive { .. }
            | Self::NoChildren
            | Self::NoProgress { .. }
            | Self::Refused(_) => None,
        }
    }
}

/// Checks a full solution against the number of columns: one finite value each.
///
/// For backend authors, so every backend refuses the same solutions the same way.
///
/// # Errors
///
/// [`ActionError::Length`] or [`ActionError::NotFinite`].
pub fn check_solution(values: &TiSlice<Col, f64>, num_cols: usize) -> Result<(), ActionError> {
    if values.len() != num_cols {
        return Err(ActionError::Length {
            expected: num_cols,
            found: values.len(),
        });
    }
    match values.iter_enumerated().find(|(_, v)| !v.is_finite()) {
        Some((col, _)) => Err(ActionError::NotFinite { col }),
        None => Ok(()),
    }
}

/// Checks partial solution entries against the number of columns: known columns, finite values.
///
/// # Errors
///
/// [`ActionError::UnknownColumn`] or [`ActionError::NotFinite`].
pub fn check_entries(entries: &[(Col, f64)], num_cols: usize) -> Result<(), ActionError> {
    for &(col, value) in entries {
        if col.index() >= num_cols {
            return Err(ActionError::UnknownColumn { col });
        }
        if !value.is_finite() {
            return Err(ActionError::NotFinite { col });
        }
    }
    Ok(())
}

/// A constraint a callback added, as checked by [`check_constraint`]: `Σ coefficient · column` in the bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct AddedRow {
    /// The terms as `(column, coefficient)`, repeated columns kept as given.
    pub terms: Vec<(Col, f64)>,
    /// The interval the terms' activity must lie in, the constraint's constant moved into it.
    pub bounds: Bounds,
}

/// Checks a constraint against the number of columns and returns it as a row: known columns, finite coefficients as
/// given and added up per column in the order given, as a program's row holds them, and valid bounds once its
/// constant moved into them.
///
/// # Errors
///
/// [`ActionError::UnknownColumn`], [`ActionError::NotFinite`] or [`ActionError::Bounds`].
pub fn check_constraint(constraint: &Constraint, num_cols: usize) -> Result<AddedRow, ActionError> {
    let terms = constraint.row_terms();
    match bad_term(&terms, num_cols) {
        Some(BadTerm::Unknown(col)) => return Err(ActionError::UnknownColumn { col }),
        Some(BadTerm::NonFinite(col, _)) => return Err(ActionError::NotFinite { col }),
        None => {}
    }
    let bounds = constraint.row_bounds().map_err(ActionError::Bounds)?;
    Ok(AddedRow { terms, bounds })
}
