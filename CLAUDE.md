# CLAUDE.md — Rust style

These rules apply to all Rust code in this repository. Follow them by default. Depart from one only when the user
explicitly asks you to, and if an existing file breaks one, raise it instead of silently working around it.

## Structure

- Dependencies between crates or modules point down. A lower layer never reaches up into a higher one, and there are
  no cycles. If a change seems to need an upward edge, the code belongs somewhere else, so move it.
- Data that crosses a process boundary (inputs, outputs, persisted state) is plain data and derives `serde`
  `Serialize`/`Deserialize`.
- An operational setting (a timeout, a grace period, an address, a project or service name) is not a global
  constant. It is read once at startup from what the platform provides, or from an environment variable when the
  platform provides nothing, into one typed configuration that is passed down. A built-in value is only the
  documented default of such a setting.
- Make invalid states unrepresentable through types. Give each kind of id its own type. Use typed index collections
  so that a table indexed by one id cannot be indexed by another. Use newtypes instead of bare numbers.
- One fact, one place. A record carries an id or a value only if none of its other fields implies it; what a field
  implies is a read on the table that holds it, never a second field. A derived table's row names the key it rests on
  and nothing that key already states. When planning a type, list every field with what implies it; a field implied
  by another is not built.
- Library functions are around forty lines; one that grows past it is split by responsibility, not by length. A test
  may be longer when it is one scenario read top to bottom; it is not split into helpers for length alone. A flat
  error enum's `Display` is one exhaustive table of sentences, however long, since splitting it would lose
  exhaustiveness or nest the enum.
- The public API is idiomatic: `Option`, `Result` and plain enums. A hot-path representation (sentinels, niche
  packing, no `Option` in hot enum variants, struct-of-arrays) is allowed only behind a private boundary, where one
  type owns the invariant and the public surface converts at the edge. A hot representation never appears in a
  public signature.

## API

- Literals are checked when the code is compiled, not when it runs. A value with an invariant (a positive amount, a
  rate, a name) has a `const fn` constructor that panics in a `const` context, wrapped in a macro (`volume!(750)`,
  `ratio!(1 / 100)`, `name!("cork")`), so new code, READMEs and doctests never write `.parse().unwrap()` or
  `Positive::new(..).unwrap()` for a literal. The fallible constructor stays for data read from documents. An
  existing test moves to the macros when it is edited for another reason, never in a sweep of its own.
- A constructor shown in a README or doctest is written the way a user would write the call. A call that reads
  clumsily there is an API defect, not a documentation one.
- A method that pushes a row returns the row's id, and every example and test uses that id; a hand-written index
  (`TaskId::new(0)`) appears only where the index itself is the subject.
- A record without an invariant is plain data with `pub` fields, as `std::process::Output` is: no getter that
  restates a field, no field-only constructor, no `with_*` setter. A record with a cross-field invariant keeps its
  fields private behind a checked constructor, and a row of a sealed table has `pub` fields because the table is the
  seal. An abstraction stays only where it carries an invariant or a seam; one that forwards, mirrors or restates goes.

- The caller supplies, the API converts. A public constructor, builder or setter takes `impl Into<T>` (or the matching
  std conversion trait) wherever a lossless, infallible conversion into `T` exists, as std does. A builder that holds
  its input behind an `Arc` takes `impl Into<Arc<T>>`, so the caller passes the value owned or shared and never writes
  `Arc::new` to hand it over.
- Only infallible conversions are taken this way. A value that can be refused keeps its checked constructor or
  `TryFrom`, so the refusal is visible where the value is made.
- Keep the generic shell thin: convert once, then call a non-generic body, so the function is compiled once.
- Do not take `impl Into<T>` where it would force the caller to write a type annotation.

### Checked/unchecked pairs

- The pair is `f` and `f_unchecked`. The signatures are identical in every respect except the return wrapper: the
  same generics, the same parameter types (including id types and `&`/`&mut`), and the same `Self`. `f` returns
  `Option<T>` or `Result<T, E>` where `f_unchecked` returns `T`. No other difference is allowed.
- `f_unchecked` is `unsafe fn` with a `# Safety` section when violating the precondition can cause UB. Otherwise it is
  a safe fn whose docs state that a violated precondition gives an unspecified result but never UB.
- The precondition is stated once, in `f`'s docs, and `f_unchecked` refers to it. `f_unchecked` carries a
  `debug_assert!` of that precondition.
- A property test asserts `f(x) == Some(f_unchecked(x))` for every `x` meeting the precondition, and that `f(x)` is
  `None`/`Err` for every `x` that does not.

## Dependencies

- Prefer std first, then an established crate, and write it yourself only as a last resort. Never hand-write
  something an established crate already does correctly, especially the error-prone parts such as parsing, printing,
  normalization, rounding and exact arithmetic. Your own newtype adds only its invariants on top of the crate.
- Add a dependency in the same change that first uses it, never earlier. Remove a dependency once nothing uses it.
- In a workspace, declare every external dependency once under `[workspace.dependencies]`. Members take it with
  `workspace = true` and never pin their own version.
- Every dependency entry gets a comment explaining what it is for and why a crate is used instead of own code.
- Use `default-features = false` wherever the crate supports it, and enable only the features you actually use. If a
  feature is enabled or disabled for a non-obvious reason, the comment says why.
- Dev-only crates (test, bench, and compile-fail tools) go in `[dev-dependencies]` and nowhere else.

### First choices

Use these before writing the equivalent code yourself:

| Need | Crate |
|---|---|
| Anything generic over a number: bounds, sentinels, casts, byte images | `num-traits`. This is mandatory, never hand-written. |
| Iterator adaptors std lacks: merging sorted sequences, finding repeats, tuple windows, joined `Display` | `itertools` (`default-features = false, features = ["use_std"]`). Look here before writing a loop. |
| gcd, and integer operations std leaves unstable (for example signed ceiling division) | `num-integer` |
| Exact ratios | `num-rational` |
| Exact fixed-point decimals | `rust_decimal`. Keep its serde feature off when the newtype carries its own serde form. |
| Vectors indexed by a typed id | `typed-index-collections` |
| Standard traits on generic types without bounds on phantom parameters | `derive-where` |
| Forwarding a newtype's arithmetic, `Display`, `FromStr` and conversions to its single field | `derive_more`, enabling only the derives you need |
| Forwarding `Zero`, `ToPrimitive` and `FromPrimitive` through a newtype | `num-derive` |
| Serialization | `serde` with `derive` |

### Derives

- Derives from the crates in the table above (`derive_more`, `derive-where`, `num-derive`, `serde`) are allowed and
  preferred over hand-written forwarding. Format strings in `derive_more` are fine.
- Write by hand anything that must report a panic at the caller's line, and mark it `#[track_caller]`. This covers
  negation, scalar products, `Sum`, and operations that panic on overflow.
- Write by hand any impl that forwards to an associated type, because a derive cannot see it.
- The newtype's own name must appear in its errors and in its `Debug`/`Display` output. Forwarding must not hide it.

## Documentation and `unsafe`

- Every public item gets a doc comment (`missing_docs` enforces it). A private item gets one only when its name and
  type do not already say what it guarantees; never write one that restates them.
- Every function that can panic has a `# Panics` section, and every function returning `Result` has an `# Errors`
  section, private ones included. Clippy checks only the public ones, so the private ones are a review matter.
- Every `unsafe fn` has a `# Safety` section, and every `unsafe` block has a `// SAFETY:` comment explaining why it
  is sound. Never write either of these where nothing is unsafe.
- Every type implements `Debug`.
- Use `pub(crate)` or narrower for anything that is not part of the crate's API.
- Doc comments state what an item guarantees, not how it is implemented. Ordinary comments state why, in plain
  declarative sentences.
- The first sentence of a doc is the guarantee; a field's doc gives its unit and what its absence means. A sentence
  that restates the type or the name says nothing and is removed; say that fields are independent only where a reader
  could take them for coupled.

These lints enforce the rules above. If the manifest lacks them, add them: use `[workspace.lints.*]` plus
`[lints] workspace = true` in each member, or `[lints.*]` for a single crate.

```toml
[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "deny"
missing_docs = "deny"
missing_debug_implementations = "deny"
unreachable_pub = "warn"

[workspace.lints.clippy]
allow_attributes_without_reason = "deny"
undocumented_unsafe_blocks = "deny"
missing_safety_doc = "deny"
missing_panics_doc = "deny"
missing_errors_doc = "deny"
unnecessary_safety_comment = "deny"
unnecessary_safety_doc = "deny"

[workspace.lints.rustdoc]
broken_intra_doc_links = "deny"
missing_crate_level_docs = "deny"
bare_urls = "deny"
invalid_codeblock_attributes = "deny"
invalid_rust_codeblocks = "deny"
```

Code you write must pass with these lints on. Do not use `#[allow]` to get a build through. Where a lint is a false
positive or a numeric `cast_*` is deliberate, use the narrowest possible `#[expect(lint, reason = "...")]`, so it
warns once it goes stale. The test-module `missing_panics_doc` allow below is the one permitted `#[allow]`, and it
carries a `reason`.

## Testing

- Every test function's name starts with `test_`, property tests included.
- Write property tests (`proptest`) that check the code against deliberately simple reference models. A reference
  model must not share the reasoning of the code it checks. For example, compute exact products with `num-bigint`
  instead of repeating the overflow logic under test.
- For code that applies rules over state (a derivation, a walk, a ledger), the reference model and the hand-worked
  fixtures are written first, from the rules' prose, by someone who does not write the implementation and does not
  read it. The implementation is then written to pass them. A model that re-derives the result step by step is not a
  model; prefer invariants that hold without any model (conservation at every hub, a level never below its floor,
  every event opened by a move) and hand-written expected outputs for small cases.
- Every defect fix comes with a test that fails without the fix, and the report shows that failure (run the new test
  against the code before the fix, or keep the saved property-test seed that found it). A fix with no failing test
  is not a fix of a known defect.
- Every serializable type gets a serde round-trip test through `serde_json`.
- A test function documents no panic: its assertions are its statement. Integration-test binaries and `#[cfg(test)]`
  modules carry a crate-level
  `#![cfg_attr(test, allow(clippy::missing_panics_doc, reason = "test assertions are the statement"))]`, the one
  `allow` this file permits; every library function, private ones included, keeps its `# Panics` and `# Errors`
  sections.
- What the types forbid is stated in the docs and relied on. A compile-fail test (`trybuild`) is written only where
  a guarantee cannot be shown any other way and a mistake there would be silent — a built value forged outside its
  builder, a sealed trait implemented outside the crate — and there is one fixture per such guarantee in the whole
  workspace, never one per type. No mutation testing. Both cost build time that outweighs their value here.
- Benchmarks (`criterion`) fix the amount of work, never the time, and compare configurations inside one binary.
- In a workspace, shared test infrastructure (property-test configuration, bench data, test-only types, reference
  models) lives in one dedicated testing crate. That crate is used only as a dev-dependency.

## Build profiles

If the manifest lacks these, add them:

```toml
# Own code stays debuggable in dev builds while dependencies are optimised.
[profile.dev]
opt-level = 1

[profile.dev.package."*"]
opt-level = 3

[profile.release]
lto = "fat"
codegen-units = 1
debug = "line-tables-only"

[profile.bench]
inherits = "release"
```

- Never set `panic = "abort"` in any profile. A worker thread that panics must unwind through a drop guard that tells
  the other threads to stop before the panic reaches the caller, so write multithreaded code with such guards.

## Manifest conventions

- Use `edition = "2024"`. Workspaces use `resolver = "3"`.
- Pin `rust-version` to the stable toolchain in use.
- Set `publish = false` unless the crate is meant for crates.io.
- In a workspace, shared package metadata lives in `[workspace.package]` and members inherit it.
- The root manifest opens with a short comment saying what the project is and how its crates relate.
