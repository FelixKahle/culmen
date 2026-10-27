// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quadratic expressions for objectives, their operators, and the sums `sum!` builds.

use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use typed_index_collections::TiSlice;

use super::expr::{LinExpr, Linear, Term};
use crate::Col;

/// A product of two columns scaled by a coefficient: `coefficient · first · second`.
///
/// `x * y`, `2.0 * x * y` and `x * x` make one without allocating. The order of the columns carries no meaning:
/// `x * y` and `y * x` add up when the model is built.
///
/// ```
/// use culmen_core::{Model, QuadTerm, Var};
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous());
/// let y = m.add_var(Var::continuous());
/// assert_eq!(2.0 * x * y, QuadTerm { first: x, second: y, coefficient: 2.0 });
/// assert_eq!(-(x * x) / 4.0, QuadTerm { first: x, second: x, coefficient: -0.25 });
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadTerm {
    /// The first column.
    pub first: Col,
    /// The second column; the same as the first for a square.
    pub second: Col,
    /// Its coefficient.
    pub coefficient: f64,
}

/// A quadratic expression `Σ coefficient · first · second + linear`, for objectives.
///
/// Plain data: terms may repeat a pair and are added up when the model is built, where the objective must be
/// convex to minimize and concave to maximize. Expressions come from the operators on [`QuadTerm`] and
/// `QuadExpr` with any [`Quadratic`] item, from products of linear items (`(x - 1) * (x - 1)`), from
/// [`QuadExpr::sum`] or `Iterator::sum`, and from [`QuadExpr::add_term`]. Constraints stay linear: the comparisons
/// of [`Linear`] take no quadratic item.
///
/// ```
/// use culmen_core::{Model, QuadExpr, TiSlice, Var};
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::free());
/// let y = m.add_var(Var::free());
/// let e = x * x + 2.0 * x * y + y * y - 4.0 * x + 3.0;
/// assert_eq!((e.terms.len(), e.linear.terms.len(), e.linear.constant), (3, 1, 3.0));
/// let squares: QuadExpr = [x, y].into_iter().map(|v| v * v).sum();
/// assert_eq!(squares.value(TiSlice::from_ref(&[1.0, 2.0])), 5.0);
/// assert_eq!(((x - 1) * (x - 1)).value(TiSlice::from_ref(&[3.0, 0.0])), 4.0);
/// ```
///
/// A quadratic constraint does not compile:
///
/// ```compile_fail
/// use culmen_core::{Linear, Model, Var};
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous());
/// m.add_constraint((x * x).leq(1.0));
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
pub struct QuadExpr {
    /// The quadratic terms, in the order they were added.
    pub terms: Vec<QuadTerm>,
    /// The linear part, constant included.
    pub linear: LinExpr,
}

impl QuadExpr {
    /// The expression zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            terms: Vec::new(),
            linear: LinExpr::new(),
        }
    }

    /// The sum of quadratic or linear items.
    ///
    /// ```
    /// use culmen_core::{Model, QuadExpr, Var};
    /// let mut m = Model::new();
    /// let xs = m.add_vars(3, |_| Var::continuous());
    /// let energy = QuadExpr::sum(xs.iter().map(|&x| 0.5 * x * x));
    /// assert_eq!(energy.terms.len(), 3);
    /// ```
    #[must_use]
    pub fn sum<T: Quadratic>(items: impl IntoIterator<Item = T>) -> Self {
        items.into_iter().fold(Self::new(), |mut expr, item| {
            item.add_quad_to(&mut expr, 1.0);
            expr
        })
    }

    /// Adds `coefficient · first · second` and returns the expression for chaining.
    pub fn add_term(&mut self, first: Col, second: Col, coefficient: f64) -> &mut Self {
        self.terms.push(QuadTerm {
            first,
            second,
            coefficient,
        });
        self
    }

    /// The expression's value at a point.
    ///
    /// # Panics
    ///
    /// When a term's column has no value in `values`.
    #[must_use]
    pub fn value(&self, values: &TiSlice<Col, f64>) -> f64 {
        let quadratic = self
            .terms
            .iter()
            .map(|t| t.coefficient * values[t.first] * values[t.second])
            .sum::<f64>();
        quadratic + self.linear.value(values)
    }
}

/// Anything that adds to a quadratic expression: every [`Linear`] item, [`QuadTerm`], [`QuadExpr`] and
/// `&QuadExpr`.
///
/// It is the bound the objective setters ([`Model::minimize`](crate::Model::minimize) and its kin), the quadratic
/// operators and [`QuadExpr::sum`] take, so an objective is linear or quadratic alike.
pub trait Quadratic: Sized {
    /// Adds `scale · self` to `expr`.
    fn add_quad_to(self, expr: &mut QuadExpr, scale: f64);

    /// The expression as an owned [`QuadExpr`].
    fn into_quad_expr(self) -> QuadExpr {
        let mut expr = QuadExpr::new();
        self.add_quad_to(&mut expr, 1.0);
        expr
    }
}

impl<T: Linear> Quadratic for T {
    fn add_quad_to(self, expr: &mut QuadExpr, scale: f64) {
        self.add_to(&mut expr.linear, scale);
    }

    fn into_quad_expr(self) -> QuadExpr {
        QuadExpr {
            terms: Vec::new(),
            linear: self.into_expr(),
        }
    }
}

impl Quadratic for QuadTerm {
    fn add_quad_to(self, expr: &mut QuadExpr, scale: f64) {
        expr.terms.push(self * scale);
    }
}

impl Quadratic for QuadExpr {
    fn add_quad_to(self, expr: &mut QuadExpr, scale: f64) {
        if scale == 1.0 && expr.terms.is_empty() {
            expr.terms = self.terms;
            self.linear.add_to(&mut expr.linear, 1.0);
        } else {
            (&self).add_quad_to(expr, scale);
        }
    }

    fn into_quad_expr(self) -> QuadExpr {
        self
    }
}

impl Quadratic for &QuadExpr {
    fn add_quad_to(self, expr: &mut QuadExpr, scale: f64) {
        expr.terms.extend(self.terms.iter().map(|&t| t * scale));
        (&self.linear).add_to(&mut expr.linear, scale);
    }
}

/// An item `sum!` adds up: a [`Linear`] item into a [`LinExpr`], a quadratic item into a [`QuadExpr`].
///
/// A sum takes the type of its items, so a sum of linear items stays linear and can be constrained.
///
/// ```
/// use culmen_core::{LinExpr, Model, QuadExpr, Summand, Var};
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous());
/// let mut load = LinExpr::new();
/// (3.0 * x).add_into(&mut load);
/// let mut energy = QuadExpr::new();
/// (x * x).add_into(&mut energy);
/// assert_eq!((load.terms.len(), energy.terms.len()), (1, 1));
/// ```
pub trait Summand: Sized {
    /// The expression a sum of such items is.
    type Sum: Default;

    /// Adds the item to `sum`.
    fn add_into(self, sum: &mut Self::Sum);
}

impl<T: Linear> Summand for T {
    type Sum = LinExpr;

    fn add_into(self, sum: &mut LinExpr) {
        self.add_to(sum, 1.0);
    }
}

impl Summand for QuadTerm {
    type Sum = QuadExpr;

    fn add_into(self, sum: &mut QuadExpr) {
        self.add_quad_to(sum, 1.0);
    }
}

impl Summand for QuadExpr {
    type Sum = Self;

    fn add_into(self, sum: &mut Self) {
        self.add_quad_to(sum, 1.0);
    }
}

impl Summand for &QuadExpr {
    type Sum = QuadExpr;

    fn add_into(self, sum: &mut QuadExpr) {
        self.add_quad_to(sum, 1.0);
    }
}

/// The product of two scaled columns.
const fn term_product(a: Term, b: Term) -> QuadTerm {
    QuadTerm {
        first: a.col,
        second: b.col,
        coefficient: a.coefficient * b.coefficient,
    }
}

/// The product of two linear expressions, expanded: `(Σ a + c)(Σ b + d) = Σ a·b + d·Σ a + c·Σ b + c·d`.
fn product(a: LinExpr, b: LinExpr) -> QuadExpr {
    let terms = a
        .terms
        .iter()
        .cartesian_product(&b.terms)
        .map(|(&s, &t)| term_product(s, t))
        .collect();
    let mut linear = LinExpr::from(a.constant * b.constant);
    for (expr, scale) in [(&a, b.constant), (&b, a.constant)] {
        if scale != 0.0 {
            linear.terms.extend(expr.terms.iter().map(|&t| t * scale));
        }
    }
    QuadExpr { terms, linear }
}

/// `lhs + sign · rhs` as one expression.
fn combine(lhs: impl Quadratic, rhs: impl Quadratic, sign: f64) -> QuadExpr {
    let mut expr = lhs.into_quad_expr();
    rhs.add_quad_to(&mut expr, sign);
    expr
}

impl Mul for Col {
    type Output = QuadTerm;
    fn mul(self, rhs: Self) -> QuadTerm {
        term_product(self.into(), rhs.into())
    }
}

impl Mul<Term> for Col {
    type Output = QuadTerm;
    fn mul(self, rhs: Term) -> QuadTerm {
        term_product(self.into(), rhs)
    }
}

impl Mul<Col> for Term {
    type Output = QuadTerm;
    fn mul(self, rhs: Col) -> QuadTerm {
        term_product(self, rhs.into())
    }
}

impl Mul for Term {
    type Output = QuadTerm;
    fn mul(self, rhs: Self) -> QuadTerm {
        term_product(self, rhs)
    }
}

impl Mul for LinExpr {
    type Output = QuadExpr;
    fn mul(self, rhs: Self) -> QuadExpr {
        product(self, rhs)
    }
}

impl Mul<Col> for LinExpr {
    type Output = QuadExpr;
    fn mul(self, rhs: Col) -> QuadExpr {
        product(self, rhs.into())
    }
}

impl Mul<LinExpr> for Col {
    type Output = QuadExpr;
    fn mul(self, rhs: LinExpr) -> QuadExpr {
        product(self.into(), rhs)
    }
}

impl Mul<Term> for LinExpr {
    type Output = QuadExpr;
    fn mul(self, rhs: Term) -> QuadExpr {
        product(self, rhs.into())
    }
}

impl Mul<LinExpr> for Term {
    type Output = QuadExpr;
    fn mul(self, rhs: LinExpr) -> QuadExpr {
        product(self.into(), rhs)
    }
}

impl Mul<f64> for QuadTerm {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            coefficient: self.coefficient * rhs,
            ..self
        }
    }
}

impl Mul<i32> for QuadTerm {
    type Output = Self;
    fn mul(self, rhs: i32) -> Self {
        self * f64::from(rhs)
    }
}

impl Mul<QuadTerm> for f64 {
    type Output = QuadTerm;
    fn mul(self, rhs: QuadTerm) -> QuadTerm {
        rhs * self
    }
}

impl Mul<QuadTerm> for i32 {
    type Output = QuadTerm;
    fn mul(self, rhs: QuadTerm) -> QuadTerm {
        rhs * f64::from(self)
    }
}

impl Div<f64> for QuadTerm {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            coefficient: self.coefficient / rhs,
            ..self
        }
    }
}

impl Neg for QuadTerm {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl MulAssign<f64> for QuadExpr {
    fn mul_assign(&mut self, rhs: f64) {
        self.terms.iter_mut().for_each(|t| t.coefficient *= rhs);
        self.linear *= rhs;
    }
}

impl Mul<f64> for QuadExpr {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        self *= rhs;
        self
    }
}

impl Mul<i32> for QuadExpr {
    type Output = Self;
    fn mul(self, rhs: i32) -> Self {
        self * f64::from(rhs)
    }
}

impl Mul<QuadExpr> for f64 {
    type Output = QuadExpr;
    fn mul(self, rhs: QuadExpr) -> QuadExpr {
        rhs * self
    }
}

impl Mul<QuadExpr> for i32 {
    type Output = QuadExpr;
    fn mul(self, rhs: QuadExpr) -> QuadExpr {
        rhs * f64::from(self)
    }
}

impl Div<f64> for QuadExpr {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            terms: self.terms.into_iter().map(|t| t / rhs).collect(),
            linear: self.linear / rhs,
        }
    }
}

impl Neg for QuadExpr {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl<T: Quadratic> Add<T> for QuadTerm {
    type Output = QuadExpr;
    fn add(self, rhs: T) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl<T: Quadratic> Sub<T> for QuadTerm {
    type Output = QuadExpr;
    fn sub(self, rhs: T) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl<T: Quadratic> Add<T> for QuadExpr {
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        combine(self, rhs, 1.0)
    }
}

impl<T: Quadratic> Sub<T> for QuadExpr {
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        combine(self, rhs, -1.0)
    }
}

impl<T: Quadratic> AddAssign<T> for QuadExpr {
    fn add_assign(&mut self, rhs: T) {
        rhs.add_quad_to(self, 1.0);
    }
}

impl<T: Quadratic> SubAssign<T> for QuadExpr {
    fn sub_assign(&mut self, rhs: T) {
        rhs.add_quad_to(self, -1.0);
    }
}

impl<T: Quadratic> Sum<T> for QuadExpr {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        Self::sum(iter)
    }
}

/// The sum of the items, as [`QuadExpr::sum`].
impl<T: Quadratic> FromIterator<T> for QuadExpr {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::sum(iter)
    }
}

/// Adds each item, as `+=` does.
impl<T: Quadratic> Extend<T> for QuadExpr {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        iter.into_iter().for_each(|item| item.add_quad_to(self, 1.0));
    }
}

impl Add<QuadTerm> for Col {
    type Output = QuadExpr;
    fn add(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadTerm> for Col {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadExpr> for Col {
    type Output = QuadExpr;
    fn add(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadExpr> for Col {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadTerm> for Term {
    type Output = QuadExpr;
    fn add(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadTerm> for Term {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadExpr> for Term {
    type Output = QuadExpr;
    fn add(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadExpr> for Term {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadTerm> for LinExpr {
    type Output = QuadExpr;
    fn add(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadTerm> for LinExpr {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadExpr> for LinExpr {
    type Output = QuadExpr;
    fn add(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadExpr> for LinExpr {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadTerm> for f64 {
    type Output = QuadExpr;
    fn add(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadTerm> for f64 {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadExpr> for f64 {
    type Output = QuadExpr;
    fn add(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadExpr> for f64 {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadTerm> for i32 {
    type Output = QuadExpr;
    fn add(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadTerm> for i32 {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadTerm) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl Add<QuadExpr> for i32 {
    type Output = QuadExpr;
    fn add(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, 1.0)
    }
}

impl Sub<QuadExpr> for i32 {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        combine(self, rhs, -1.0)
    }
}

impl From<QuadTerm> for QuadExpr {
    fn from(term: QuadTerm) -> Self {
        term.into_quad_expr()
    }
}

impl From<LinExpr> for QuadExpr {
    fn from(linear: LinExpr) -> Self {
        linear.into_quad_expr()
    }
}

impl From<Col> for QuadExpr {
    fn from(col: Col) -> Self {
        col.into_quad_expr()
    }
}

impl From<Term> for QuadExpr {
    fn from(term: Term) -> Self {
        term.into_quad_expr()
    }
}

impl From<f64> for QuadExpr {
    fn from(constant: f64) -> Self {
        constant.into_quad_expr()
    }
}
