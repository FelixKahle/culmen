// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The modelling macros of culmen: a model written the way the mathematics reads, each macro expanding to the public
//! methods of culmen's modelling layer, so a model written with them and one written with the methods build the same
//! `Lp`.
//!
//! Use them through the `culmen` facade with its `macros` feature (or beside `culmen_core`); they are documented
//! there with the methods each form writes. In brief:
//!
//! - `variable!(m, x)`, `variable!(m, 0 <= y[i in I] <= cap[i], Int)`, `variable!(m, z[p in P, l in L if fits(p,
//!   l)], Bin)`: one variable or a family, bound to a local of the same name and named after it.
//! - `constraint!(lhs <= rhs)`, `constraint!(m, lhs <= rhs)`, `constraint!(m, name, lhs == rhs)`,
//!   `constraint!(m, name[i in I], lo <= body <= hi)`: a constraint as a value, added, added by name, or a family.
//! - `objective!(m, Min, expr)` and `objective!(m, Max, expr)`, linear or quadratic.
//! - `sum!(body for i in I, j in J if cond)`: a sum over a loop nest, linear or quadratic as its body is.
//!
//! Inside every macro, `x[i, j]` is `x[(i.clone(), j.clone())]`: a family indexed by a tuple key. A comma inside a
//! subscript is a syntax error for every other type, so nothing else changes.

mod family;
mod home;
mod tokens;

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;

use family::{Head, head, is_head, loops, nest};
use tokens::{Relation, first_span, rewrite_indices, split_commas, split_keyword, split_relations};

/// Declares a variable or a family of them, binds it to a local of the same name and names it after it.
///
/// `variable!(model, spec[, kind])`. The spec is the name, `x` or `x[i in I, j in J if cond]`, with the bounds it
/// is given: `x`, `x >= lb`, `x <= ub`, `x == value`, `lb <= x <= ub` or `ub >= x >= lb`. The kind is `Cont` (the
/// default), `Int`, `Bin` or `Free`, also spelled `Continuous`, `Integer` and `Binary`. Expands to
/// `Model::add_var(Var::…)` or `Model::add_var_map(keys, |key| Var::…)`.
#[proc_macro]
pub fn variable(input: TokenStream) -> TokenStream {
    expand(input, variable_tokens)
}

/// A constraint as a value, `constraint!(lhs <= rhs)`; added, `constraint!(model, lhs <= rhs)`; added by name and
/// bound to a local, `constraint!(model, name, lhs <= rhs)`; or a family, `constraint!(model, name[i in I], body)`.
/// The relation is `<=`, `>=`, `==`, or a range `lo <= expr <= hi`. Expands to `Linear::leq`, `Linear::geq`,
/// `Linear::equals` or `Linear::between`, and `Model::add_constraint`, `Model::add_named_constraint` or
/// `Model::add_constraint_map`.
#[proc_macro]
pub fn constraint(input: TokenStream) -> TokenStream {
    expand(input, constraint_tokens)
}

/// The objective: `objective!(model, Min, expr)` or `objective!(model, Max, expr)`, also spelled `Minimize` and
/// `Maximize`; `expr` is linear or quadratic. Expands to `Model::set_objective`.
#[proc_macro]
pub fn objective(input: TokenStream) -> TokenStream {
    expand(input, objective_tokens)
}

/// The sum of `body` over a loop nest: `sum!(body for i in I, j in J if cond)`. Expands to `for` loops adding each
/// body with `Summand::add_into` to a sum that starts empty: a `LinExpr` for a linear body, a `QuadExpr` for a
/// quadratic one.
#[proc_macro]
pub fn sum(input: TokenStream) -> TokenStream {
    expand(input, sum_tokens)
}

/// The expansion `tokens` makes of `input`, handed the path of the layer, or the error at the token at fault.
fn expand(
    input: TokenStream,
    tokens: impl FnOnce(&TokenStream2, TokenStream2) -> syn::Result<TokenStream2>,
) -> TokenStream {
    home::path()
        .and_then(|root| tokens(&root, input.into()))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// What values a variable takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Continuous,
    Integer,
    Binary,
    Free,
}

/// The kind a trailing argument names.
///
/// # Errors
///
/// For anything but one of the kinds' names.
fn kind(tokens: &TokenStream2) -> syn::Result<Kind> {
    let text = tokens.to_string();
    match text.as_str() {
        "Cont" | "Continuous" => Ok(Kind::Continuous),
        "Int" | "Integer" => Ok(Kind::Integer),
        "Bin" | "Binary" => Ok(Kind::Binary),
        "Free" => Ok(Kind::Free),
        _ => Err(syn::Error::new(
            first_span(tokens),
            "a kind: `Cont`, `Int`, `Bin` or `Free`",
        )),
    }
}

/// The bounds a spec gives, as written.
#[derive(Default)]
struct Bounds {
    lower: Option<TokenStream2>,
    upper: Option<TokenStream2>,
    fixed: Option<TokenStream2>,
}

/// A variable spec split into the declaration and its bounds.
///
/// # Errors
///
/// For a spec whose relations do not read as one of the forms `variable!` takes.
fn spec(tokens: TokenStream2) -> syn::Result<(TokenStream2, Bounds)> {
    let (parts, relations) = split_relations(tokens);
    let rel: Vec<Relation> = relations.iter().map(|&(r, _)| r).collect();
    let span = relations.first().map_or_else(Span::call_site, |&(_, span)| span);
    let bound = |b: &TokenStream2| Some(rewrite_indices(b.clone()));
    let (declared, bounds) = match (rel.as_slice(), parts.as_slice()) {
        ([], [x]) => (x, Bounds::default()),
        ([Relation::Geq], [x, b]) | ([Relation::Leq], [b, x]) if is_head(x) => (
            x,
            Bounds {
                lower: bound(b),
                ..Bounds::default()
            },
        ),
        ([Relation::Leq], [x, b]) | ([Relation::Geq], [b, x]) if is_head(x) => (
            x,
            Bounds {
                upper: bound(b),
                ..Bounds::default()
            },
        ),
        ([Relation::Eq], [x, b]) if is_head(x) => (
            x,
            Bounds {
                fixed: bound(b),
                ..Bounds::default()
            },
        ),
        ([Relation::Leq, Relation::Leq], [l, x, u]) | ([Relation::Geq, Relation::Geq], [u, x, l]) => (
            x,
            Bounds {
                lower: bound(l),
                upper: bound(u),
                fixed: None,
            },
        ),
        _ => {
            return Err(syn::Error::new(
                span,
                "bounds read `x >= lb`, `x <= ub`, `x == v` or `lb <= x <= ub`",
            ));
        }
    };
    Ok((declared.clone(), bounds))
}

/// The `Var` a kind and bounds describe.
fn var_expr(root: &TokenStream2, kind: Kind, bounds: &Bounds) -> TokenStream2 {
    let base = match kind {
        Kind::Continuous => quote!(#root::Var::continuous()),
        Kind::Integer => quote!(#root::Var::integer()),
        Kind::Binary => quote!(#root::Var::binary()),
        Kind::Free => quote!(#root::Var::free()),
    };
    if let Some(value) = &bounds.fixed {
        let v = Ident::new("value", Span::mixed_site());
        return quote!({ let #v = #value; #base.bounds(#v, #v) });
    }
    let lower = bounds.lower.as_ref().map(|l| quote!(.lower(#l)));
    let upper = bounds.upper.as_ref().map(|u| quote!(.upper(#u)));
    quote!(#base #lower #upper)
}

/// `variable!`.
fn variable_tokens(root: &TokenStream2, input: TokenStream2) -> syn::Result<TokenStream2> {
    let args = split_commas(input);
    let (model, spec_tokens, kind_tokens) = match args.as_slice() {
        [model, spec] => (model, spec, None),
        [model, spec, kind] => (model, spec, Some(kind)),
        _ => return Err(syn::Error::new(Span::call_site(), "variable!(model, spec[, kind])")),
    };
    let kind = kind_tokens.map_or(Ok(Kind::Continuous), kind)?;
    let (declared, bounds) = spec(spec_tokens.clone())?;
    if kind == Kind::Binary && (bounds.lower.is_some() || bounds.upper.is_some() || bounds.fixed.is_some()) {
        let span = kind_tokens.map_or_else(Span::call_site, first_span);
        return Err(syn::Error::new(
            span,
            "a binary's bounds are [0, 1]: for others, `Int` with the bounds",
        ));
    }
    let Head { name, index } = head(declared)?;
    let var = var_expr(root, kind, &bounds);
    let made = match index {
        Some(index) => {
            let entry = Ident::new("name", Span::mixed_site());
            index.family(
                root,
                model,
                &Ident::new("add_var_map", Span::call_site()),
                &name,
                &quote!(#var.name(#entry)),
            )
        }
        None => {
            let text = name.to_string();
            quote!((#model).add_var(#var.name(#text)))
        }
    };
    Ok(quote!(let #name = #made;))
}

/// A constraint value from a relation's tokens.
///
/// # Errors
///
/// For tokens that are not one relation or one range.
fn relation(root: &TokenStream2, tokens: TokenStream2) -> syn::Result<TokenStream2> {
    let span = first_span(&tokens);
    let (parts, relations) = split_relations(rewrite_indices(tokens));
    let rel: Vec<Relation> = relations.iter().map(|&(r, _)| r).collect();
    let linear = quote!(#root::Linear);
    match (rel.as_slice(), parts.as_slice()) {
        ([Relation::Leq], [l, r]) => Ok(quote!(#linear::leq(#l, #r))),
        ([Relation::Geq], [l, r]) => Ok(quote!(#linear::geq(#l, #r))),
        ([Relation::Eq], [l, r]) => Ok(quote!(#linear::equals(#l, #r))),
        ([Relation::Leq, Relation::Leq], [lo, e, hi]) => Ok(quote!(#linear::between(#e, #lo, #hi))),
        ([Relation::Geq, Relation::Geq], [hi, e, lo]) => Ok(quote!(#linear::between(#e, #lo, #hi))),
        _ => Err(syn::Error::new(
            span,
            "a constraint reads `lhs <= rhs`, `lhs >= rhs`, `lhs == rhs` or `lo <= e <= hi`",
        )),
    }
}

/// `constraint!`.
fn constraint_tokens(root: &TokenStream2, input: TokenStream2) -> syn::Result<TokenStream2> {
    let args = split_commas(input);
    match args.as_slice() {
        [body] => relation(root, body.clone()),
        [model, body] => {
            let c = relation(root, body.clone())?;
            Ok(quote!((#model).add_constraint(#c)))
        }
        [model, declared, body] => {
            let c = relation(root, body.clone())?;
            let Head { name, index } = head(declared.clone())?;
            let made = match index {
                Some(index) => {
                    let entry = Ident::new("name", Span::mixed_site());
                    let method = Ident::new("add_constraint_map", Span::call_site());
                    index.family(root, model, &method, &name, &quote!((#entry, #c)))
                }
                None => {
                    let text = name.to_string();
                    quote!((#model).add_named_constraint(#text, #c))
                }
            };
            // A constraint's handle is a convenience most models never read, so its binding is marked as read.
            Ok(quote!(let #name = #made; let _ = &#name;))
        }
        _ => Err(syn::Error::new(
            Span::call_site(),
            "constraint!([model, [name,]] relation)",
        )),
    }
}

/// `objective!`.
fn objective_tokens(root: &TokenStream2, input: TokenStream2) -> syn::Result<TokenStream2> {
    let args = split_commas(input);
    let [model, sense, expr] = args.as_slice() else {
        return Err(syn::Error::new(Span::call_site(), "objective!(model, Min | Max, expr)"));
    };
    let sense = match sense.to_string().as_str() {
        "Min" | "Minimize" => quote!(#root::Sense::Minimize),
        "Max" | "Maximize" => quote!(#root::Sense::Maximize),
        _ => return Err(syn::Error::new(first_span(sense), "a sense: `Min` or `Max`")),
    };
    let expr = rewrite_indices(expr.clone());
    Ok(quote!((#model).set_objective(#sense, #expr)))
}

/// `sum!`.
fn sum_tokens(root: &TokenStream2, input: TokenStream2) -> syn::Result<TokenStream2> {
    let (body, Some((_, nest_tokens))) = split_keyword(input, "for") else {
        return Err(syn::Error::new(Span::call_site(), "sum!(body for i in I, …)"));
    };
    if body.is_empty() {
        return Err(syn::Error::new(Span::call_site(), "sum! needs a body before `for`"));
    }
    let body = rewrite_indices(body);
    let total = Ident::new("total", Span::mixed_site());
    let loops = nest(
        root,
        &loops(nest_tokens)?,
        &quote!(#root::Summand::add_into(#body, &mut #total);),
    );
    // The sum's type is the body's `Summand::Sum`, which the call in the loop settles.
    Ok(quote!({
        let mut #total = ::core::default::Default::default();
        #loops
        #total
    }))
}
