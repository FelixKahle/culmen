// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Linear expressions, their operators, and the constraints they compare into.

use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use typed_index_collections::TiSlice;

use crate::Col;

/// A column scaled by a coefficient: `coefficient · col`.
///
/// `2.0 * x` and `x * 2.0` make one without allocating, so a sum of many products builds a single expression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    /// The column.
    pub col: Col,
    /// Its coefficient.
    pub coefficient: f64,
}

/// A linear expression `Σ coefficient · col + constant`.
///
/// Plain data: terms may repeat a column and are added up when the model is built. Expressions come from the
/// operators on [`Col`], [`Term`], `LinExpr`, `f64` and `i32`, from [`LinExpr::sum`] or `Iterator::sum`, and from
/// [`LinExpr::add_term`]:
///
/// ```
/// use culmen_core::{LinExpr, Model, Var};
///
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous());
/// let y = m.add_var(Var::continuous());
/// let e = 2.0 * x + y - 3;
/// assert_eq!(e.terms.len(), 2);
/// assert_eq!(e.constant, -3.0);
/// let total: LinExpr = [x, y].into_iter().sum();
/// assert_eq!(total.terms.len(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LinExpr {
    /// The terms, in the order they were added.
    pub terms: Vec<Term>,
    /// The constant.
    pub constant: f64,
}

impl LinExpr {
    /// The expression zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            terms: Vec::new(),
            constant: 0.0,
        }
    }

    /// The sum of linear items: columns, terms, expressions or numbers.
    ///
    /// ```
    /// use culmen_core::{LinExpr, Model, Var};
    /// let mut m = Model::new();
    /// let xs = m.add_vars(3, |_| Var::binary());
    /// let weights = [4.0, 5.0, 6.0];
    /// let load = LinExpr::sum(xs.iter().zip(weights).map(|(&x, w)| w * x));
    /// assert_eq!(load.terms.len(), 3);
    /// ```
    #[must_use]
    pub fn sum<T: Linear>(items: impl IntoIterator<Item = T>) -> Self {
        let mut expr = Self::new();
        for item in items {
            item.add_to(&mut expr, 1.0);
        }
        expr
    }

    /// Adds `coefficient · col` and returns the expression for chaining.
    pub fn add_term(&mut self, col: Col, coefficient: f64) -> &mut Self {
        self.terms.push(Term { col, coefficient });
        self
    }

    /// The expression's value at a point.
    ///
    /// # Panics
    ///
    /// When a term's column has no value in `values`.
    #[must_use]
    pub fn value(&self, values: &TiSlice<Col, f64>) -> f64 {
        self.constant + self.terms.iter().map(|t| t.coefficient * values[t.col]).sum::<f64>()
    }
}

/// Anything that adds to a linear expression: [`Col`], [`Term`], [`LinExpr`], `&LinExpr`, `f64` and `i32`.
///
/// It is the bound the operators, [`LinExpr::sum`] and the comparisons take, and its comparison methods turn an
/// expression into a [`Constraint`] without the macros:
///
/// ```
/// use culmen_core::{Linear, Model, Var};
/// let mut m = Model::new();
/// let x = m.add_var(Var::continuous());
/// let y = m.add_var(Var::continuous());
/// let c = (x + 2.0 * y).leq(10.0);
/// assert_eq!((c.lower, c.upper), (f64::NEG_INFINITY, 0.0));
/// assert_eq!(c.expr.constant, -10.0);
/// let r = (x - y).between(-1.0, 1.0);
/// assert_eq!((r.lower, r.upper), (-1.0, 1.0));
/// ```
pub trait Linear: Sized {
    /// Adds `scale · self` to `expr`.
    fn add_to(self, expr: &mut LinExpr, scale: f64);

    /// The expression as an owned [`LinExpr`].
    fn into_expr(self) -> LinExpr {
        let mut expr = LinExpr::new();
        self.add_to(&mut expr, 1.0);
        expr
    }

    /// The constraint `self <= rhs`.
    fn leq(self, rhs: impl Linear) -> Constraint {
        Constraint {
            expr: difference(self, rhs),
            lower: f64::NEG_INFINITY,
            upper: 0.0,
        }
    }

    /// The constraint `self >= rhs`.
    fn geq(self, rhs: impl Linear) -> Constraint {
        Constraint {
            expr: difference(self, rhs),
            lower: 0.0,
            upper: f64::INFINITY,
        }
    }

    /// The constraint `self == rhs`.
    fn equals(self, rhs: impl Linear) -> Constraint {
        Constraint {
            expr: difference(self, rhs),
            lower: 0.0,
            upper: 0.0,
        }
    }

    /// The constraint `lower <= self <= upper`.
    fn between(self, lower: impl Into<f64>, upper: impl Into<f64>) -> Constraint {
        Constraint {
            expr: self.into_expr(),
            lower: lower.into(),
            upper: upper.into(),
        }
    }
}

/// `lhs - rhs` as one expression.
fn difference(lhs: impl Linear, rhs: impl Linear) -> LinExpr {
    let mut expr = LinExpr::new();
    lhs.add_to(&mut expr, 1.0);
    rhs.add_to(&mut expr, -1.0);
    expr
}

impl Linear for Col {
    fn add_to(self, expr: &mut LinExpr, scale: f64) {
        expr.terms.push(Term {
            col: self,
            coefficient: scale,
        });
    }
}

impl Linear for Term {
    fn add_to(self, expr: &mut LinExpr, scale: f64) {
        expr.terms.push(Term {
            col: self.col,
            coefficient: scale * self.coefficient,
        });
    }
}

impl Linear for LinExpr {
    fn add_to(self, expr: &mut LinExpr, scale: f64) {
        if scale == 1.0 && expr.terms.is_empty() {
            let constant = expr.constant;
            *expr = self;
            expr.constant += constant;
        } else {
            (&self).add_to(expr, scale);
        }
    }

    fn into_expr(self) -> LinExpr {
        self
    }
}

impl Linear for &LinExpr {
    fn add_to(self, expr: &mut LinExpr, scale: f64) {
        expr.terms.extend(self.terms.iter().map(|t| Term {
            col: t.col,
            coefficient: scale * t.coefficient,
        }));
        expr.constant += scale * self.constant;
    }
}

impl Linear for f64 {
    fn add_to(self, expr: &mut LinExpr, scale: f64) {
        expr.constant += scale * self;
    }
}

impl Linear for i32 {
    fn add_to(self, expr: &mut LinExpr, scale: f64) {
        expr.constant += scale * f64::from(self);
    }
}

impl<T: Linear> Sum<T> for LinExpr {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        Self::sum(iter)
    }
}

impl<T: Linear> AddAssign<T> for LinExpr {
    fn add_assign(&mut self, rhs: T) {
        rhs.add_to(self, 1.0);
    }
}

impl<T: Linear> SubAssign<T> for LinExpr {
    fn sub_assign(&mut self, rhs: T) {
        rhs.add_to(self, -1.0);
    }
}

impl MulAssign<f64> for LinExpr {
    fn mul_assign(&mut self, rhs: f64) {
        for term in &mut self.terms {
            term.coefficient *= rhs;
        }
        self.constant *= rhs;
    }
}

/// `Add` and `Sub` of any [`Linear`] on a linear type, giving a [`LinExpr`].
macro_rules! linear_sum_ops {
    ($($ty:ty),*) => {$(
        impl<T: Linear> Add<T> for $ty {
            type Output = LinExpr;
            fn add(self, rhs: T) -> LinExpr {
                let mut expr = self.into_expr();
                rhs.add_to(&mut expr, 1.0);
                expr
            }
        }

        impl<T: Linear> Sub<T> for $ty {
            type Output = LinExpr;
            fn sub(self, rhs: T) -> LinExpr {
                let mut expr = self.into_expr();
                rhs.add_to(&mut expr, -1.0);
                expr
            }
        }
    )*};
}

linear_sum_ops!(Col, Term, LinExpr);

/// A number on the left of `+` or `-`, with a linear type on the right.
macro_rules! scalar_sum_ops {
    ($scalar:ty => $($ty:ty),*) => {$(
        impl Add<$ty> for $scalar {
            type Output = LinExpr;
            fn add(self, rhs: $ty) -> LinExpr {
                rhs + self
            }
        }

        impl Sub<$ty> for $scalar {
            type Output = LinExpr;
            fn sub(self, rhs: $ty) -> LinExpr {
                let mut expr = LinExpr::new();
                self.add_to(&mut expr, 1.0);
                rhs.add_to(&mut expr, -1.0);
                expr
            }
        }
    )*};
}

scalar_sum_ops!(f64 => Col, Term, LinExpr);
scalar_sum_ops!(i32 => Col, Term, LinExpr);

/// Scaling a column or a term, by a number on either side, gives a [`Term`].
macro_rules! term_scale_ops {
    ($($ty:ty),*) => {$(
        impl Mul<f64> for $ty {
            type Output = Term;
            fn mul(self, rhs: f64) -> Term {
                let t = Term::from(self);
                Term { col: t.col, coefficient: t.coefficient * rhs }
            }
        }

        impl Mul<i32> for $ty {
            type Output = Term;
            fn mul(self, rhs: i32) -> Term {
                self * f64::from(rhs)
            }
        }

        impl Mul<$ty> for f64 {
            type Output = Term;
            fn mul(self, rhs: $ty) -> Term {
                rhs * self
            }
        }

        impl Mul<$ty> for i32 {
            type Output = Term;
            fn mul(self, rhs: $ty) -> Term {
                rhs * f64::from(self)
            }
        }

        impl Div<f64> for $ty {
            type Output = Term;
            fn div(self, rhs: f64) -> Term {
                let t = Term::from(self);
                Term { col: t.col, coefficient: t.coefficient / rhs }
            }
        }

        impl Neg for $ty {
            type Output = Term;
            fn neg(self) -> Term {
                self * -1.0
            }
        }
    )*};
}

term_scale_ops!(Col, Term);

impl From<Col> for Term {
    fn from(col: Col) -> Self {
        Self { col, coefficient: 1.0 }
    }
}

impl Mul<f64> for LinExpr {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        self *= rhs;
        self
    }
}

impl Mul<LinExpr> for f64 {
    type Output = LinExpr;
    fn mul(self, rhs: LinExpr) -> LinExpr {
        rhs * self
    }
}

impl Mul<i32> for LinExpr {
    type Output = Self;
    fn mul(self, rhs: i32) -> Self {
        self * f64::from(rhs)
    }
}

impl Mul<LinExpr> for i32 {
    type Output = LinExpr;
    fn mul(self, rhs: LinExpr) -> LinExpr {
        rhs * f64::from(self)
    }
}

impl Div<f64> for LinExpr {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self {
        for term in &mut self.terms {
            term.coefficient /= rhs;
        }
        self.constant /= rhs;
        self
    }
}

impl Neg for LinExpr {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl From<Col> for LinExpr {
    fn from(col: Col) -> Self {
        col.into_expr()
    }
}

impl From<Term> for LinExpr {
    fn from(term: Term) -> Self {
        term.into_expr()
    }
}

impl From<f64> for LinExpr {
    fn from(constant: f64) -> Self {
        Self {
            terms: Vec::new(),
            constant,
        }
    }
}

/// A linear constraint `lower <= expr <= upper`, written by [`Linear::leq`], [`Linear::geq`], [`Linear::equals`],
/// [`Linear::between`] or the `constraint!` macro.
///
/// Plain data: the numbers are checked when the model is built, and the expression's constant moves into the
/// bounds there, so `x + 1 <= 3` and `x <= 2` build the same row.
#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    /// The constrained expression.
    pub expr: LinExpr,
    /// The least value the expression may take; `-∞` for none.
    pub lower: f64,
    /// The greatest value the expression may take; `+∞` for none.
    pub upper: f64,
}
