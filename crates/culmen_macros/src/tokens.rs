// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The token work every macro shares: splitting at a top-level comma, relation or keyword, reading which names a
//! pattern binds, and rewriting the brackets of a multi-index access.
//!
//! "Top level" means outside every group: a comma or a `<=` inside parentheses, brackets or braces belongs to the
//! expression it sits in, never to the macro's own grammar.

use proc_macro2::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::quote;

/// A comparison of the grammar: `<=`, `>=` or `==`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Relation {
    /// `<=`.
    Leq,
    /// `>=`.
    Geq,
    /// `==`.
    Eq,
}

/// `tokens` split at every top-level comma; a trailing comma leaves no empty part, and a part that is one invisible
/// group is read as what it holds.
///
/// A comma between a turbofish's angle brackets (`f::<A, B>`) or between a closure's bars (`|a, b|`) is not at the
/// top level, though neither is a group.
pub(crate) fn split_commas(tokens: TokenStream) -> Vec<TokenStream> {
    let mut parts = vec![Vec::new()];
    let mut nesting = Nesting::default();
    let mut before: Option<TokenTree> = None;
    let mut joint: Option<char> = None;
    for token in tokens {
        let outside = nesting.is_outside();
        nesting.step(&token, before.as_ref(), joint);
        match &token {
            TokenTree::Punct(punct) if punct.as_char() == ',' && outside => parts.push(Vec::new()),
            _ => parts
                .last_mut()
                .unwrap_or_else(|| unreachable!("one part at least"))
                .push(token.clone()),
        }
        joint = match &token {
            TokenTree::Punct(punct) if punct.spacing() == Spacing::Joint => Some(punct.as_char()),
            _ => None,
        };
        before = Some(token);
    }
    if parts.last().is_some_and(Vec::is_empty) && parts.len() > 1 {
        let _ = parts.pop();
    }
    parts
        .into_iter()
        .map(|part| unwrap_invisible(part.into_iter().collect()))
        .collect()
}

/// What `tokens` hold, where they are one invisible group, at any depth: what another macro's `$e:expr` hands a
/// macro, with the fragment's tokens inside.
///
/// The group keeps a fragment one operand, which a part never needs: a part is the whole of a macro's argument.
fn unwrap_invisible(tokens: TokenStream) -> TokenStream {
    let mut iter = tokens.clone().into_iter();
    match (iter.next(), iter.next()) {
        (Some(TokenTree::Group(group)), None) if group.delimiter() == Delimiter::None => {
            unwrap_invisible(group.stream())
        }
        _ => tokens,
    }
}

/// The span of the first token of `tokens`, or the call site for none.
pub(crate) fn first_span(tokens: &TokenStream) -> Span {
    tokens
        .clone()
        .into_iter()
        .next()
        .map_or_else(Span::call_site, |token| token.span())
}

/// Where a scan is among the brackets that are not groups: a turbofish's angle brackets and a closure's bars.
#[derive(Default)]
struct Nesting {
    /// How many turbofish angle brackets are open.
    angles: usize,
    /// Whether a closure's parameters are open.
    bars: bool,
    /// Whether the last punctuation was `::`, after which a `<` opens a turbofish.
    path: bool,
}

impl Nesting {
    /// Whether the scan is outside every turbofish and closure parameter list.
    const fn is_outside(&self) -> bool {
        self.angles == 0 && !self.bars
    }

    /// Steps over `token`, which follows `before`, the punctuation joint to it being `joint`.
    fn step(&mut self, token: &TokenTree, before: Option<&TokenTree>, joint: Option<char>) {
        let TokenTree::Punct(punct) = token else {
            self.path = false;
            return;
        };
        let after_path = self.path;
        self.path = punct.as_char() == ':' && joint == Some(':');
        match punct.as_char() {
            '<' if after_path || self.angles > 0 => self.angles += 1,
            // The `>` of an `->` inside a turbofish closes nothing.
            '>' if self.angles > 0 && joint != Some('-') => self.angles -= 1,
            '|' if self.bars => self.bars = false,
            // A bar where an operand ends is a bitwise or; where an expression starts, it opens a closure's
            // parameters, and a joint `||` is a closure with none.
            '|' if self.angles == 0
                && starts_expression(before)
                && punct.spacing() == Spacing::Alone
                && joint != Some('|') =>
            {
                self.bars = true;
            }
            _ => {}
        }
    }
}

/// Whether a token after `before` starts an expression: after nothing, a punctuation, or `move`, `return` and the
/// like; not after an operand, a name, a literal or a group.
fn starts_expression(before: Option<&TokenTree>) -> bool {
    match before {
        None | Some(TokenTree::Punct(_)) => true,
        Some(TokenTree::Ident(ident)) => ["move", "return", "in", "else", "break"]
            .iter()
            .any(|word| ident == word),
        Some(TokenTree::Group(_) | TokenTree::Literal(_)) => false,
    }
}

/// `tokens` split at every top-level relation, `<=`, `>=` or `==`, with the relations in order: one part more than
/// relations.
pub(crate) fn split_relations(tokens: TokenStream) -> (Vec<TokenStream>, Vec<(Relation, Span)>) {
    let tokens: Vec<TokenTree> = tokens.into_iter().collect();
    let mut parts = vec![Vec::new()];
    let mut relations = Vec::new();
    let mut at = 0;
    while at < tokens.len() {
        if let Some(relation) = relation_at(&tokens, at) {
            relations.push((relation, tokens[at].span()));
            parts.push(Vec::new());
            at += 2;
            continue;
        }
        parts
            .last_mut()
            .unwrap_or_else(|| unreachable!("one part at least"))
            .push(tokens[at].clone());
        at += 1;
    }
    (parts.into_iter().map(TokenStream::from_iter).collect(), relations)
}

/// The relation that starts at `tokens[at]`: a joint `<`, `>` or `=` followed by an `=`, and not itself the end of a
/// longer operator (`<<=`, `>>=`). The `=` may be joint to what follows, as in `x<=-1`, and a `?` may be joint to the
/// relation, as in `x?<=1`.
fn relation_at(tokens: &[TokenTree], at: usize) -> Option<Relation> {
    let (TokenTree::Punct(first), Some(TokenTree::Punct(second))) = (&tokens[at], tokens.get(at + 1)) else {
        return None;
    };
    if first.spacing() != Spacing::Joint || second.as_char() != '=' {
        return None;
    }
    if at > 0
        && let TokenTree::Punct(before) = &tokens[at - 1]
        && joins_into(before)
    {
        return None;
    }
    match first.as_char() {
        '<' => Some(Relation::Leq),
        '>' => Some(Relation::Geq),
        '=' => Some(Relation::Eq),
        _ => None,
    }
}

/// Whether `before`, the punctuation right before a relation, joins it into a longer operator: any joint
/// punctuation but `?`, which ends an operand and starts no operator.
fn joins_into(before: &Punct) -> bool {
    before.spacing() == Spacing::Joint && before.as_char() != '?'
}

/// `tokens` split at the first top-level identifier `keyword`: what comes before it, and what after, if it occurs.
pub(crate) fn split_keyword(tokens: TokenStream, keyword: &str) -> (TokenStream, Option<(Span, TokenStream)>) {
    let mut before = Vec::new();
    let mut tokens = tokens.into_iter();
    while let Some(token) = tokens.next() {
        if let TokenTree::Ident(ident) = &token
            && ident == keyword
        {
            return (before.into_iter().collect(), Some((ident.span(), tokens.collect())));
        }
        before.push(token);
    }
    (before.into_iter().collect(), None)
}

/// A loop's domain split from the filter after it, `0..n if i > 0`: the domain, and the `if` with its condition.
///
/// The filter is the first top-level `if` where an operand ends: an `if` where an expression starts, the domain's
/// first token, one after an operator or after `else`, is an `if` expression of the domain's own.
pub(crate) fn split_filter(domain: TokenStream) -> (TokenStream, Option<(Span, TokenStream)>) {
    let mut before: Vec<TokenTree> = Vec::new();
    let mut tokens = domain.into_iter();
    while let Some(token) = tokens.next() {
        if let TokenTree::Ident(ident) = &token
            && ident == "if"
            && !starts_expression(before.last())
        {
            return (before.into_iter().collect(), Some((ident.span(), tokens.collect())));
        }
        before.push(token);
    }
    (before.into_iter().collect(), None)
}

/// Every identifier a pattern binds, in order: the names of `i` and `(a, b)`.
///
/// # Errors
///
/// For a pattern the macros do not take: one with `_`, `ref`, `mut`, `@`, `|`, a path or a literal, any of which
/// would not read back as the key it binds, or would leave part of it out of the entry's name.
pub(crate) fn bound_names(pattern: &TokenStream) -> syn::Result<Vec<Ident>> {
    let mut names = Vec::new();
    collect_names(pattern.clone(), &mut names)?;
    Ok(names)
}

/// Adds the identifiers `tokens` binds to `names`, refusing what a key pattern cannot hold.
fn collect_names(tokens: TokenStream, names: &mut Vec<Ident>) -> syn::Result<()> {
    let mut tokens = tokens.into_iter().peekable();
    while let Some(token) = tokens.next() {
        // A name followed by a group or a path is a constructor or a path, `Some(i)`, `Key { a }`, not a binding.
        if let TokenTree::Ident(ident) = &token
            && matches!(tokens.peek(), Some(TokenTree::Group(_) | TokenTree::Punct(_)))
            && !matches!(tokens.peek(), Some(TokenTree::Punct(punct)) if punct.as_char() == ',')
        {
            return Err(syn::Error::new(
                ident.span(),
                "an index is a name or a tuple of names, such as `i` or `(i, j)`",
            ));
        }
        match token {
            TokenTree::Ident(ident) if ident == "ref" || ident == "mut" => {
                return Err(syn::Error::new(
                    ident.span(),
                    "an index binds its key by value: no `ref` or `mut`",
                ));
            }
            TokenTree::Ident(ident) if ident == "_" => {
                return Err(syn::Error::new(
                    ident.span(),
                    "a family's index names its key: with `_`, its entries' names would collide",
                ));
            }
            TokenTree::Ident(ident) => names.push(ident),
            TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis => {
                collect_names(group.stream(), names)?;
            }
            TokenTree::Punct(punct) if punct.as_char() == ',' => {}
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "an index is a name or a tuple of names, such as `i` or `(i, j)`",
                ));
            }
        }
    }
    Ok(())
}

/// The keywords after which a `[` opens an array or a slice pattern, not an index.
const NOT_INDEXED: [&str; 19] = [
    "for", "in", "if", "else", "while", "loop", "match", "let", "return", "move", "as", "where", "mut", "ref", "break",
    "continue", "const", "static", "yield",
];

/// `tokens` with every multi-index access `x[i, j]` written `x[(i.clone(), j.clone())]` (through `Clone::clone`), at
/// any depth: an index by a tuple of clones, which a family takes by value and which leaves the indices usable after
/// it. A comma inside a subscript is a syntax error for every other type, so only family accesses change;
/// `cost[i]` and `[a, b]` stay as they are.
pub(crate) fn rewrite_indices(tokens: TokenStream) -> TokenStream {
    let mut out: Vec<TokenTree> = Vec::new();
    for token in tokens {
        let TokenTree::Group(group) = token else {
            out.push(token);
            continue;
        };
        let inner = rewrite_indices(group.stream());
        let subscript = group.delimiter() == Delimiter::Bracket && out.last().is_some_and(indexable);
        let parts = split_commas(inner.clone());
        let inner = if subscript && parts.len() > 1 {
            let cloned: Vec<TokenStream> = parts
                .into_iter()
                .map(|part| quote!(::core::clone::Clone::clone(&(#part))))
                .collect();
            tuple_of(&cloned)
        } else {
            inner
        };
        let mut rewritten = Group::new(group.delimiter(), inner);
        rewritten.set_span(group.span());
        out.push(TokenTree::Group(rewritten));
    }
    out.into_iter().collect()
}

/// Whether a `[` right after `token` is a subscript: after a name that is not a keyword, a closing group, another
/// macro's fragment (an invisible group) or a literal.
fn indexable(token: &TokenTree) -> bool {
    match token {
        TokenTree::Ident(ident) => !NOT_INDEXED.iter().any(|keyword| ident == keyword),
        TokenTree::Group(group) => {
            matches!(
                group.delimiter(),
                Delimiter::Parenthesis | Delimiter::Bracket | Delimiter::None
            )
        }
        TokenTree::Literal(_) => true,
        TokenTree::Punct(_) => false,
    }
}

/// `names` as a pattern or an expression: the one name alone, a tuple of them otherwise.
pub(crate) fn tuple_of(items: &[TokenStream]) -> TokenStream {
    if let [only] = items {
        return only.clone();
    }
    let mut inner = TokenStream::new();
    for (at, item) in items.iter().enumerate() {
        if at > 0 {
            inner.extend([TokenTree::Punct(Punct::new(',', Spacing::Alone))]);
        }
        inner.extend(item.clone());
    }
    TokenTree::Group(Group::new(Delimiter::Parenthesis, inner)).into()
}

#[cfg(test)]
mod tests {
    use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
    use quote::quote;

    use super::{Relation, bound_names, rewrite_indices, split_commas, split_filter, split_keyword, split_relations};

    /// `tokens` as another macro's `$e:expr` hands them on: one invisible group.
    fn invisible(tokens: TokenStream) -> TokenTree {
        TokenTree::Group(Group::new(Delimiter::None, tokens))
    }

    /// The text of each part.
    fn texts(parts: &[TokenStream]) -> Vec<String> {
        parts.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn test_commas_split_only_at_the_top_level() {
        assert_eq!(
            texts(&split_commas(quote!(m, x[i, j], f(a, b),))),
            ["m", "x [i , j]", "f (a , b)"]
        );
        assert_eq!(texts(&split_commas(quote!())), [""]);
        assert_eq!(
            texts(&split_commas(quote!(m, id::<1, 2>(x) <= 1))),
            ["m", "id :: < 1 , 2 > (x) <= 1"]
        );
        assert_eq!(
            texts(&split_commas(quote!(m, f(|a, b| a + b), move |c, d| c))),
            ["m", "f (| a , b | a + b)", "move | c , d | c"]
        );
        assert_eq!(
            texts(&split_commas(quote!(a | b, c))),
            ["a | b", "c"],
            "a bitwise or is no closure"
        );
        assert_eq!(
            texts(&split_commas(quote!(x < y, z > w))),
            ["x < y", "z > w"],
            "a comparison is no turbofish"
        );
    }

    #[test]
    fn test_a_part_that_is_one_invisible_group_reads_as_what_it_holds() {
        let fragment = invisible(invisible(quote!(x <= 1)).into());
        let parts = split_commas(quote!(m, #fragment));
        assert_eq!(texts(&parts), ["m", "x <= 1"]);
        let (_, relations) = split_relations(parts[1].clone());
        assert_eq!(relations.len(), 1, "the relation is at the top level once unwrapped");
        let operand = invisible(quote!(a + b));
        assert_eq!(
            split_commas(quote!(#operand * 2)).len(),
            1,
            "a group beside other tokens stays one operand"
        );
        let access = invisible(quote!(x));
        let rewritten = rewrite_indices(quote!(#access[i, j])).to_string();
        assert!(
            rewritten.starts_with("x [(:: core :: clone"),
            "a fragment is indexable: {rewritten}"
        );
    }

    #[test]
    fn test_a_filter_is_an_if_where_an_operand_ends() {
        let split = |tokens| {
            let (domain, filter) = split_filter(tokens);
            (domain.to_string(), filter.map(|(_, cond)| cond.to_string()))
        };
        assert_eq!(
            split(quote!(0..n if i > 0)),
            ("0 .. n".to_owned(), Some("i > 0".to_owned()))
        );
        assert_eq!(
            split(quote!(v.iter() if ok)),
            ("v . iter ()".to_owned(), Some("ok".to_owned()))
        );
        assert_eq!(
            split(quote!(if a { x } else { y })),
            ("if a { x } else { y }".to_owned(), None),
            "a domain's own if"
        );
        assert_eq!(
            split(quote!(if a { x } else if b { y } else { z } if k)),
            ("if a { x } else if b { y } else { z }".to_owned(), Some("k".to_owned()))
        );
        assert_eq!(
            split(quote!(0..if a { 3 } else { 4 })).1,
            None,
            "an if after an operator starts an operand"
        );
    }

    #[test]
    fn test_relations_split_at_the_top_level_and_not_inside_longer_operators() {
        let (parts, relations) = split_relations(quote!(1 <= x + f(a <= b) <= 3));
        assert_eq!(texts(&parts), ["1", "x + f (a <= b)", "3"]);
        assert_eq!(
            relations.iter().map(|&(relation, _)| relation).collect::<Vec<_>>(),
            [Relation::Leq, Relation::Leq]
        );
        let (parts, relations) = split_relations(quote!(x == y));
        assert_eq!(
            (texts(&parts), relations[0].0),
            (vec!["x".to_owned(), "y".to_owned()], Relation::Eq)
        );
        let (_, relations) = split_relations(quote!(a >= b));
        assert_eq!(relations[0].0, Relation::Geq);
        let (parts, _) = split_relations("x<=-1".parse().unwrap());
        assert_eq!(texts(&parts), ["x", "- 1"], "an operator joint to the next sign");
        let (parts, relations) = split_relations(quote!(a != b && c << d < e));
        assert!(relations.is_empty() && parts.len() == 1, "no relation of the grammar");
        let (parts, relations) = split_relations("x?<=1".parse().unwrap());
        assert_eq!(
            (texts(&parts), relations[0].0),
            (vec!["x ?".to_owned(), "1".to_owned()], Relation::Leq),
            "after a `?`"
        );
        let (parts, relations) = split_relations("a<<=b".parse().unwrap());
        assert!(
            relations.is_empty() && parts.len() == 1,
            "the tail of `<<=` is no relation"
        );
    }

    #[test]
    fn test_a_keyword_splits_at_its_first_top_level_occurrence() {
        let (before, after) = split_keyword(quote!(x[i, j] for i in 0..n, j in (if a { b } else { c })), "for");
        assert_eq!(before.to_string(), "x [i , j]");
        assert_eq!(
            after.unwrap().1.to_string(),
            "i in 0 .. n , j in (if a { b } else { c })"
        );
        assert!(split_keyword(quote!(a b c), "if").1.is_none());
    }

    #[test]
    fn test_a_pattern_binds_names_and_refuses_what_a_key_cannot_hold() {
        let names =
            |tokens| bound_names(&tokens).map(|names| names.iter().map(ToString::to_string).collect::<Vec<_>>());
        assert_eq!(names(quote!(i)).unwrap(), ["i"]);
        assert!(
            names(quote!((a, (b, _)))).is_err(),
            "a key is named whole, or its entries' names collide"
        );
        assert!(names(quote!(_)).is_err());
        assert!(names(quote!(ref i)).is_err());
        assert!(names(quote!(Some(i))).is_err());
        assert!(names(quote!(1)).is_err());
        assert!(names(quote!(Key { a })).is_err());
        assert!(names(quote!(a::b)).is_err());
    }

    #[test]
    fn test_multi_index_brackets_become_tuples_and_others_stay() {
        let clone = |e: &str| format!(":: core :: clone :: Clone :: clone (& ({e}))");
        assert_eq!(
            rewrite_indices(quote!(x[i, j] + cost[i] * y[(a, b)])).to_string(),
            format!("x [({} , {})] + cost [i] * y [(a , b)]", clone("i"), clone("j"))
        );
        assert_eq!(
            rewrite_indices(quote!(f(z[1, 2])[0, 1])).to_string(),
            format!(
                "f (z [({} , {})]) [({} , {})]",
                clone("1"),
                clone("2"),
                clone("0"),
                clone("1")
            )
        );
        assert_eq!(
            rewrite_indices(quote!(let v = [a, b]; for [p, q] in w {})).to_string(),
            "let v = [a , b] ; for [p , q] in w { }"
        );
        assert_eq!(
            rewrite_indices(quote!(sum!(x[i, j] for i in 0..2))).to_string(),
            format!("sum ! (x [({} , {})] for i in 0 .. 2)", clone("i"), clone("j"))
        );
    }
}
