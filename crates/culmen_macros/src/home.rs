// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where culmen's modelling layer is reachable from the crate a macro runs in.
//!
//! A consumer reaches the layer through whichever crate it depends on: the `culmen` facade, or `culmen_core`, which
//! the facade re-exports at its root. A macro sees only tokens, so it reads the calling crate's manifest, under
//! whatever name that manifest gives the dependency.

use proc_macro_crate::{Error, FoundCrate, crate_name};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

/// The crates that reach the layer, each with the path from its root to it, in the order they are tried: the most
/// direct first.
const HOMES: [(&str, &[&str]); 2] = [("culmen", &[]), ("culmen_core", &[])];

/// What a macro answers when the calling crate depends on none of `HOMES`.
const NOT_FOUND: &str = "the modelling macros build a culmen model, reached through a dependency on `culmen` or \
                         `culmen_core`; this crate's manifest names neither";

/// The path each of `HOMES` names, as the tests expect it.
#[cfg(test)]
const PATHS: [&str; 2] = ["::culmen", "::culmen_core"];

/// The path of the home, from the extern prelude of the crate the macro runs in, as its manifest says.
///
/// # Errors
///
/// If the manifest names none of the crates in `HOMES`.
pub(crate) fn path() -> syn::Result<TokenStream> {
    path_with(crate_name)
}

/// The path of the home, with `find` answering under which name the calling crate knows a package.
///
/// Inside a home crate's own tests, examples and doc tests the crate is found as itself, and is named by its own
/// name, which each of those targets has in its extern prelude. When there is no manifest to read, as in a build that
/// does not run through cargo, the path is the first home's.
///
/// # Errors
///
/// If `find` finds none of the crates in `HOMES`.
fn path_with(mut find: impl FnMut(&str) -> Result<FoundCrate, Error>) -> syn::Result<TokenStream> {
    for (package, inner) in HOMES {
        match find(package) {
            Ok(FoundCrate::Name(name)) => return Ok(path_in(&name, inner)),
            Ok(FoundCrate::Itself) => return Ok(path_in(package, inner)),
            Err(Error::CrateNotFound { .. }) => {}
            Err(_) => {
                let (first, inner) = HOMES[0];
                return Ok(path_in(first, inner));
            }
        }
    }
    Err(syn::Error::new(Span::call_site(), NOT_FOUND))
}

/// The path `::root::inner…`.
fn path_in(root: &str, inner: &[&str]) -> TokenStream {
    let root = Ident::new(root, Span::call_site());
    let inner = inner.iter().map(|module| Ident::new(module, Span::call_site()));
    quote!(::#root #(::#inner)*)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use proc_macro_crate::{Error, FoundCrate};

    use super::{HOMES, NOT_FOUND, PATHS, path_with};

    /// The path found when the calling crate knows exactly `known`, each as `(package, name in the manifest)`.
    fn path(known: &[(&str, &str)]) -> String {
        let find = |package: &str| match known.iter().find(|&&(known, _)| known == package) {
            Some(&(_, "self")) => Ok(FoundCrate::Itself),
            Some(&(_, name)) => Ok(FoundCrate::Name(name.to_owned())),
            None => Err(Error::CrateNotFound {
                crate_name: package.to_owned(),
                path: PathBuf::from("Cargo.toml"),
            }),
        };
        path_with(find).map_or_else(|error| error.to_string(), |path| path.to_string().replace(' ', ""))
    }

    #[test]
    fn test_each_home_alone_is_named_by_its_path() {
        for ((package, _), expected) in HOMES.iter().zip(PATHS) {
            assert_eq!(path(&[(package, package)]), expected);
        }
    }

    #[test]
    fn test_the_most_direct_home_is_named() {
        let all: Vec<(&str, &str)> = HOMES.iter().rev().map(|&(package, _)| (package, package)).collect();
        assert_eq!(path(&all), PATHS[0]);
    }

    #[test]
    fn test_a_renamed_dependency_is_named_as_renamed() {
        let (last, inner) = HOMES[HOMES.len() - 1];
        let renamed: Vec<&str> = ["::renamed"].into_iter().chain(inner.iter().copied()).collect();
        assert_eq!(path(&[(last, "renamed")]), renamed.join("::"));
    }

    #[test]
    fn test_a_home_itself_is_named_by_its_name() {
        assert_eq!(path(&[(HOMES[0].0, "self")]), PATHS[0]);
    }

    #[test]
    fn test_no_manifest_falls_back_to_the_first_home() {
        let path = path_with(|_| Err(Error::CargoManifestDirNotSet)).map(|path| path.to_string().replace(' ', ""));
        assert_eq!(path.ok().as_deref(), Some(PATHS[0]));
    }

    #[test]
    fn test_no_home_is_refused() {
        assert_eq!(path(&[]), NOT_FOUND);
    }
}
