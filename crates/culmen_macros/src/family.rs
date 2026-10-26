// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarations and loop nests: a name with its optional index, `x` or `x[i in I, j in J if c]`, read; and the
//! loops that collect a family's keys or a sum's terms written out.
//!
//! Every domain is evaluated where its loop starts, as a `for` loop's is, and iterated as an `IndexSet`, which
//! yields its members by value: `p in products` binds `p: usize` over a `Vec<usize>`, a `&Vec<usize>` or a range
//! alike, and leaves `products` usable by the next loop.

use proc_macro2::{Delimiter, Ident, Span, TokenStream, TokenTree};
use quote::{format_ident, quote};

use crate::tokens::{bound_names, first_span, rewrite_indices, split_commas, split_filter, split_keyword, tuple_of};

/// One loop of an index or a sum: `pattern in domain`.
pub(crate) struct Loop {
    /// The pattern each item is bound to.
    pub(crate) pattern: TokenStream,
    /// The domain the items come from, its multi-index accesses rewritten.
    pub(crate) domain: TokenStream,
}

/// A loop nest as written, `p in P, q in Q if cond`: the loops, outermost first, and the filter after the last.
pub(crate) struct Loops {
    /// The loops, outermost first.
    pub(crate) loops: Vec<Loop>,
    /// `if cond`, the condition, its multi-index accesses rewritten.
    pub(crate) filter: Option<TokenStream>,
}

/// Reads a loop nest: `pattern in domain`, separated by commas, the last optionally followed by `if cond`.
///
/// # Errors
///
/// For a loop with no `in`, or an empty pattern or domain; a filter after any loop but the last; an `if` with no
/// condition.
pub(crate) fn loops(tokens: TokenStream) -> syn::Result<Loops> {
    let parts = split_commas(tokens);
    let last = parts.len() - 1;
    let (mut loops, mut filter) = (Vec::with_capacity(parts.len()), None);
    for (at, part) in parts.into_iter().enumerate() {
        let span = first_span(&part);
        let (pattern, Some((_, rest))) = split_keyword(part, "in") else {
            return Err(syn::Error::new(span, "an index reads `name in domain`"));
        };
        let (domain, kept) = split_filter(rest);
        if pattern.is_empty() || domain.is_empty() {
            return Err(syn::Error::new(span, "an index reads `name in domain`"));
        }
        if let Some((if_span, cond)) = kept {
            if at != last {
                return Err(syn::Error::new(
                    if_span,
                    "the filter `if cond` comes last, after every `name in domain`",
                ));
            }
            if cond.is_empty() {
                return Err(syn::Error::new(if_span, "`if` needs a condition"));
            }
            filter = Some(rewrite_indices(cond));
        }
        loops.push(Loop {
            pattern,
            domain: rewrite_indices(domain),
        });
    }
    Ok(Loops { loops, filter })
}

/// A declared name, with the loop nest of its index if it declares a family.
pub(crate) struct Head {
    /// The name, bound to the result and given to it.
    pub(crate) name: Ident,
    /// The index, for a family.
    pub(crate) index: Option<Index>,
}

/// The index of a family: its loop nest, every pattern naming its keys, and the names bound.
pub(crate) struct Index {
    /// The loops and the filter.
    nest: Loops,
    /// The names every pattern binds, in order: what an entry's name shows.
    names: Vec<Ident>,
}

/// Reads `tokens` as a declaration: a name, optionally followed by an index in brackets.
///
/// # Errors
///
/// For anything else, an index that does not read, or a pattern that does not name its keys.
pub(crate) fn head(tokens: TokenStream) -> syn::Result<Head> {
    let mut tokens = tokens.into_iter();
    let name = match tokens.next() {
        Some(TokenTree::Ident(name)) => name,
        Some(other) => return Err(syn::Error::new(other.span(), "a name, such as `x` or `x[i in I]`")),
        None => return Err(syn::Error::new(Span::call_site(), "a name, such as `x` or `x[i in I]`")),
    };
    let index = match tokens.next() {
        None => None,
        Some(TokenTree::Group(group)) if group.delimiter() == Delimiter::Bracket => {
            let nest = loops(group.stream())?;
            let mut names = Vec::new();
            for each in &nest.loops {
                names.extend(bound_names(&each.pattern)?);
            }
            Some(Index { nest, names })
        }
        Some(other) => {
            return Err(syn::Error::new(
                other.span(),
                "after the name, only an index `[i in I, …]`",
            ));
        }
    };
    if let Some(extra) = tokens.next() {
        return Err(syn::Error::new(extra.span(), "unexpected after the declaration"));
    }
    Ok(Head { name, index })
}

/// Whether `tokens` read as a declaration: a name, optionally followed by brackets holding an `in`. A subscript
/// holds none, so `cap[i]` is a bound and `x[i in I]` a declaration.
pub(crate) fn is_head(tokens: &TokenStream) -> bool {
    let tokens: Vec<TokenTree> = tokens.clone().into_iter().collect();
    match tokens.as_slice() {
        [TokenTree::Ident(_)] => true,
        [TokenTree::Ident(_), TokenTree::Group(group)] => {
            group.delimiter() == Delimiter::Bracket && split_keyword(group.stream(), "in").1.is_some()
        }
        _ => false,
    }
}

/// A domain's members: the domain evaluated and iterated as an `IndexSet`.
fn items(root: &TokenStream, domain: &TokenStream) -> TokenStream {
    quote!(#root::IndexSet::members(&(#domain)))
}

/// The loops that run `body` once per item combination the filter keeps, each pattern bound for the loops and the
/// filter inside it.
pub(crate) fn nest(root: &TokenStream, loops: &Loops, body: &TokenStream) -> TokenStream {
    let kept = loops.filter.as_ref().map_or_else(|| quote!(true), |cond| quote!(#cond));
    let mut nest = quote!(if #kept { #body });
    for each in loops.loops.iter().rev() {
        let (pattern, items) = (&each.pattern, items(root, &each.domain));
        nest = quote!(for #pattern in #items { #nest });
    }
    nest
}

impl Index {
    /// The statements that collect every key into a vector `keys`, and that vector's name.
    fn collect(&self, root: &TokenStream) -> (TokenStream, Ident) {
        let span = Span::mixed_site();
        let keys = Ident::new("keys", span);
        let patterns: Vec<TokenStream> = self.nest.loops.iter().map(|each| each.pattern.clone()).collect();
        let key = tuple_of(
            &patterns
                .iter()
                .map(|p| quote!(::core::clone::Clone::clone(&#p)))
                .collect::<Vec<_>>(),
        );
        let body = quote!(#keys.push(#key););
        let loops = nest(root, &self.nest, &body);
        (quote!(let mut #keys = ::std::vec::Vec::new(); #loops), keys)
    }

    /// The pattern a whole key binds: the loops' patterns, alone or as a tuple.
    fn pattern(&self) -> TokenStream {
        tuple_of(
            &self
                .nest
                .loops
                .iter()
                .map(|each| each.pattern.clone())
                .collect::<Vec<_>>(),
        )
    }

    /// An entry's name, `base[i,j]`, each index shown through its `Display`.
    fn name(&self, base: &Ident) -> TokenStream {
        let template = format!("{}[{}]", base, vec!["{}"; self.names.len()].join(","));
        let names = &self.names;
        quote!(::std::format!(#template, #(#names),*))
    }

    /// The family `model.method(keys, |key| make)`, where `make` sees the key's pattern bound and the entry's name
    /// as `name`.
    pub(crate) fn family(
        &self,
        root: &TokenStream,
        model: &TokenStream,
        method: &Ident,
        base: &Ident,
        make: &TokenStream,
    ) -> TokenStream {
        let span = Span::mixed_site();
        let (key, name) = (Ident::new("key", span), format_ident!("name", span = span));
        let (collect, keys) = self.collect(root);
        let (pattern, entry_name) = (self.pattern(), self.name(base));
        quote! {{
            #collect
            (#model).#method(#keys, |#key| {
                let #pattern = ::core::clone::Clone::clone(#key);
                let #name = #entry_name;
                #make
            })
        }}
    }
}

#[cfg(test)]
mod tests {
    use proc_macro2::{Ident, Span};
    use quote::quote;

    use super::{head, is_head, loops, nest};

    #[test]
    fn test_a_declaration_reads_a_name_and_an_optional_index() {
        let plain = head(quote!(x)).unwrap();
        assert!(plain.name == "x" && plain.index.is_none());
        let family = head(quote!(x[i in 0..n, (a, b) in arcs if a != b])).unwrap();
        let index = family.index.unwrap();
        assert_eq!(index.nest.loops.len(), 2);
        assert_eq!(index.pattern().to_string(), "(i , (a , b))");
        assert_eq!(index.nest.filter.as_ref().unwrap().to_string(), "a != b");
        assert_eq!(
            index.names.iter().map(ToString::to_string).collect::<Vec<_>>(),
            ["i", "a", "b"]
        );
        assert_eq!(
            index.name(&Ident::new("x", Span::call_site())).to_string(),
            ":: std :: format ! (\"x[{},{},{}]\" , i , a , b)"
        );
        assert!(head(quote!(x y)).is_err() && head(quote!(1)).is_err() && head(quote!(x(i))).is_err());
        assert!(head(quote!(x[_ in 0..3])).is_err(), "a family's index names its keys");
        assert!(is_head(&quote!(x[i in I])) && is_head(&quote!(x)) && !is_head(&quote!(x + 1)) && !is_head(&quote!(3)));
        assert!(!is_head(&quote!(cap[i])), "a subscript holds no `in`");
    }

    #[test]
    fn test_loops_take_any_pattern_and_one_filter_after_the_last() {
        let read = loops(quote!(_ in 0..3, (a, &b) in pairs if a < b)).unwrap();
        let patterns: Vec<String> = read.loops.iter().map(|each| each.pattern.to_string()).collect();
        assert_eq!(patterns, ["_", "(a , & b)"]);
        assert_eq!(read.filter.unwrap().to_string(), "a < b");
        assert!(
            loops(quote!(i in 0..3 if i > 0, j in 0..3)).is_err(),
            "a filter before the last loop"
        );
        assert!(loops(quote!(i in 0..3 if)).is_err());
        assert!(loops(quote!(i)).is_err());
        assert!(loops(quote!(in 0..3)).is_err());
    }

    #[test]
    fn test_a_nest_iterates_each_domain_as_an_index_set_and_keeps_what_the_filter_keeps() {
        let nest = nest(
            &quote!(::culmen),
            &loops(quote!(i in v, j in i..n if i != j)).unwrap(),
            &quote!(go();),
        );
        let nest = nest.to_string();
        assert!(
            nest.starts_with("for i in :: culmen :: IndexSet :: members (& (v))"),
            "{nest}"
        );
        assert!(
            nest.contains("for j in :: culmen :: IndexSet :: members (& (i .. n))"),
            "{nest}"
        );
        assert!(nest.contains("if i != j { go () ; }"), "{nest}");
    }
}
