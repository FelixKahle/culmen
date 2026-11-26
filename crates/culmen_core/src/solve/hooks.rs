// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The handlers that change a search, and the hooks that carry them and the callback to a backend.

use core::fmt;

use crate::{Backend, Callback, Events, Heuristics, LazyConstraints, Priced, Prices, UserCuts};

/// What a solve asks about each candidate solution: lazy constraints that cut it off before the solver accepts it.
///
/// Registered with [`Run::lazy`](crate::Run::lazy), only for a backend that takes lazy constraints. Two handlers run
/// together as a pair `(a, b)`, `a` first, on one context.
///
/// ```
/// use culmen_core::{Col, LazyConstraints, LazyContext, LazyHandler, Linear};
///
/// /// Forbids taking both items, a conflict only the handler knows.
/// struct NotBoth(Col, Col);
///
/// impl<B: LazyConstraints> LazyHandler<B> for NotBoth {
///     fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
///         let candidate = ctx.candidate();
///         if candidate[self.0] + candidate[self.1] > 1.5 {
///             ctx.add_lazy((self.0 + self.1).leq(1)).expect("a valid constraint");
///         }
///     }
/// }
/// ```
pub trait LazyHandler<B: LazyConstraints>: Send {
    /// A candidate solution, to be cut off by lazy constraints.
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>);
}

/// What a solve asks about node relaxations: user cuts that cut them off.
///
/// Registered with [`Run::cut`](crate::Run::cut), only for a backend that takes user cuts. Two handlers run together
/// as a pair `(a, b)`, `a` first, on one context.
pub trait CutHandler<B: UserCuts>: Send {
    /// A node relaxation, to be cut off by user cuts.
    fn cut(&mut self, ctx: &mut B::Cut<'_>);
}

/// What a solve asks for heuristic solutions of the handler's own.
///
/// Registered with [`Run::heuristic`](crate::Run::heuristic), only for a backend that takes heuristic solutions. Two
/// handlers run together as a pair `(a, b)`, `a` first, on one context.
pub trait HeuristicHandler<B: Heuristics>: Send {
    /// A chance to hand the solver a heuristic solution.
    fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>);
}

/// What a solve asks at every node LP: columns that improve it, or make it feasible.
///
/// Registered with [`Run::pricer`](crate::Run::pricer), only for a backend that prices. Two pricers run together as a
/// pair `(a, b)`, `a` first; they share one context, so the columns of the second follow those of the first.
pub trait Pricer<B: Prices>: Send {
    /// A node LP's duals, to be answered with columns that improve it.
    fn price(&mut self, ctx: &mut B::Price<'_>);
}

impl<B: LazyConstraints, H: LazyHandler<B> + ?Sized> LazyHandler<B> for &mut H {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        (**self).lazy(ctx);
    }
}

impl<B: LazyConstraints, X: LazyHandler<B>, Y: LazyHandler<B>> LazyHandler<B> for (X, Y) {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        self.0.lazy(ctx);
        self.1.lazy(ctx);
    }
}

impl<B: UserCuts, H: CutHandler<B> + ?Sized> CutHandler<B> for &mut H {
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        (**self).cut(ctx);
    }
}

impl<B: UserCuts, X: CutHandler<B>, Y: CutHandler<B>> CutHandler<B> for (X, Y) {
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        self.0.cut(ctx);
        self.1.cut(ctx);
    }
}

impl<B: Heuristics, H: HeuristicHandler<B> + ?Sized> HeuristicHandler<B> for &mut H {
    fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>) {
        (**self).heuristic(ctx);
    }
}

impl<B: Heuristics, X: HeuristicHandler<B>, Y: HeuristicHandler<B>> HeuristicHandler<B> for (X, Y) {
    fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>) {
        self.0.heuristic(ctx);
        self.1.heuristic(ctx);
    }
}

impl<B: Prices, P: Pricer<B> + ?Sized> Pricer<B> for &mut P {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        (**self).price(ctx);
    }
}

impl<B: Prices, X: Pricer<B>, Y: Pricer<B>> Pricer<B> for (X, Y) {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        self.0.price(ctx);
        self.1.price(ctx);
    }
}

/// A backend's hooks handed on to a backend it runs inside, whose contexts it names as its own: what a wrapping
/// backend, such as one that records or checks the calls it forwards, registers on the inner backend's [`Hooks`]
/// with [`Hooks::relay`].
///
/// Each search method carries its capability as a `where` clause on `B`, so one relay serves every `B` and forwards a
/// handler of a kind exactly when `B` has that capability; a wrapper whose capabilities are its inner backend's
/// writes each method as a call of the same method on its own [`Hooks`]. Its [`Callback`] methods forward the
/// observational events alike.
pub trait Relay<B: Backend>: Callback<B> {
    /// Whether the relayed hooks have a lazy-constraint handler.
    fn wants_lazy(&self) -> bool
    where
        B: LazyConstraints;
    /// Hands a candidate solution to the relayed hooks.
    fn lazy(&mut self, ctx: &mut <B as LazyConstraints>::Lazy<'_>)
    where
        B: LazyConstraints;
    /// Whether the relayed hooks have a cut handler.
    fn wants_cut(&self) -> bool
    where
        B: UserCuts;
    /// Hands a node relaxation to the relayed hooks.
    fn cut(&mut self, ctx: &mut <B as UserCuts>::Cut<'_>)
    where
        B: UserCuts;
    /// Whether the relayed hooks have a heuristic handler.
    fn wants_heuristic(&self) -> bool
    where
        B: Heuristics;
    /// Hands a heuristic point to the relayed hooks.
    fn heuristic(&mut self, ctx: &mut <B as Heuristics>::Heuristic<'_>)
    where
        B: Heuristics;
    /// Whether the relayed hooks have a pricer.
    fn wants_price(&self) -> bool
    where
        B: Prices;
    /// Hands a pricing round to the relayed hooks.
    fn price(&mut self, ctx: &mut <B as Prices>::Price<'_>)
    where
        B: Prices;
}

// The slots below hold a registered handler for every `B`, while a handler trait names `B`'s context only when `B`
// has the capability: each slot's method carries the capability as a `where` clause on `B`, which keeps the slot
// dyn-compatible for every backend, and is implemented only where the capability holds.

trait LazySlot<B: Backend>: Send {
    fn lazy(&mut self, ctx: &mut <B as LazyConstraints>::Lazy<'_>)
    where
        B: LazyConstraints;
}

impl<B: LazyConstraints, H: LazyHandler<B>> LazySlot<B> for H {
    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        LazyHandler::lazy(self, ctx);
    }
}

trait CutSlot<B: Backend>: Send {
    fn cut(&mut self, ctx: &mut <B as UserCuts>::Cut<'_>)
    where
        B: UserCuts;
}

impl<B: UserCuts, H: CutHandler<B>> CutSlot<B> for H {
    fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        CutHandler::cut(self, ctx);
    }
}

/// One handler in both roles, lazy constraints and cuts.
trait LazyCutSlot<B: Backend>: Send {
    fn lazy(&mut self, ctx: &mut <B as LazyConstraints>::Lazy<'_>)
    where
        B: LazyConstraints;
    fn cut(&mut self, ctx: &mut <B as UserCuts>::Cut<'_>)
    where
        B: UserCuts;
}

impl<B: LazyConstraints + UserCuts, H: LazyHandler<B> + CutHandler<B>> LazyCutSlot<B> for H {
    fn lazy(&mut self, ctx: &mut <B as LazyConstraints>::Lazy<'_>) {
        LazyHandler::lazy(self, ctx);
    }
    fn cut(&mut self, ctx: &mut <B as UserCuts>::Cut<'_>) {
        CutHandler::cut(self, ctx);
    }
}

trait HeuristicSlot<B: Backend>: Send {
    fn heuristic(&mut self, ctx: &mut <B as Heuristics>::Heuristic<'_>)
    where
        B: Heuristics;
}

impl<B: Heuristics, H: HeuristicHandler<B>> HeuristicSlot<B> for H {
    fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>) {
        HeuristicHandler::heuristic(self, ctx);
    }
}

trait PriceSlot<B: Backend>: Send {
    fn price(&mut self, ctx: &mut <B as Prices>::Price<'_>)
    where
        B: Prices;
}

impl<B: Prices, P: Pricer<B>> PriceSlot<B> for P {
    fn price(&mut self, ctx: &mut B::Price<'_>) {
        Pricer::price(self, ctx);
    }
}

/// What one solve runs as the search goes: a callback for the observational events and a handler for each kind of
/// search event the backend can take, handed to [`Backend::solve`] by value.
///
/// A [`Solver`](crate::Solver) builds the hooks a backend receives from what its [`Run`](crate::Run) registered,
/// adding its own stop and observer, which run before the callback, and the sink of the columns a pricer adds. Code
/// that drives a backend directly builds them with [`Hooks::new`] and the registration methods; their names keep the
/// `register_` prefix because the short names are the dispatch methods a backend calls. Each registration method
/// exists only for a backend with the capability, so a handler the backend would ignore does not compile:
///
/// ```compile_fail,E0599
/// use culmen_core::{Col, Hooks, LazyConstraints, LazyContext, LazyHandler, Linear, Unsupported};
/// # use culmen_core::{Backend, Events, ModelClasses, Program, Settings, Solved};
/// # #[derive(Debug)] struct Plain;
/// # impl Backend for Plain {
/// #     type Error = std::io::Error; type Status = (); type Log<'a> = Unsupported; type Progress<'a> = Unsupported;
/// #     type Poll<'a> = Unsupported; type Solution<'a> = Unsupported; type Native<'a> = Unsupported;
/// #     type Options = ();
/// #     const EVENTS: Events = Events::NONE; const CLASSES: ModelClasses = ModelClasses::ALL;
/// #     fn options(&self) -> &() { &() }
/// #     fn options_mut(&mut self) -> &mut () { Box::leak(Box::new(())) }
/// #     fn load(&mut self, _: &Program) -> Result<(), Self::Error> { Ok(()) }
/// #     fn solve(&mut self, _: &Settings, _: Hooks<'_, Self>) -> Result<Solved<()>, Self::Error> { unimplemented!() }
/// # }
/// struct Never;
/// impl<B: LazyConstraints> LazyHandler<B> for Never {
///     fn lazy(&mut self, _: &mut B::Lazy<'_>) {}
/// }
/// // `Plain` does not take lazy constraints.
/// let hooks = Hooks::<Plain>::new().register_lazy(&mut Never);
/// ```
///
/// A backend reads what to deliver from the hooks: [`Hooks::events`] for the observational events and
/// [`Hooks::wants_lazy`], [`Hooks::wants_cut`], [`Hooks::wants_heuristic`] and [`Hooks::wants_price`] for the
/// search events, each a property of what was registered, so a callback asking for every observational event costs
/// the solver no reduction.
#[must_use]
pub struct Hooks<'h, B: Backend> {
    /// The solver's own stop and observer, run before `callback`.
    watch: Option<&'h mut dyn Callback<B>>,
    callback: Option<&'h mut dyn Callback<B>>,
    lazy: Option<&'h mut dyn LazySlot<B>>,
    cut: Option<&'h mut dyn CutSlot<B>>,
    /// A handler in both roles; [`Hooks::lazy`] and [`Hooks::cut`] reach it when `lazy` or `cut` is not set.
    lazy_and_cut: Option<&'h mut dyn LazyCutSlot<B>>,
    heuristic: Option<&'h mut dyn HeuristicSlot<B>>,
    pricer: Option<&'h mut dyn PriceSlot<B>>,
    /// The hooks of a wrapping backend, reached for every kind nothing else here is registered for.
    relay: Option<&'h mut dyn Relay<B>>,
    /// The sink of the columns the pricer adds; taken out by the backend.
    priced: Option<&'h mut Priced>,
}

impl<'h, B: Backend> Hooks<'h, B> {
    /// No callback and no handler: a solve that delivers nothing.
    pub const fn new() -> Self {
        Self {
            watch: None,
            callback: None,
            lazy: None,
            cut: None,
            lazy_and_cut: None,
            heuristic: None,
            pricer: None,
            relay: None,
            priced: None,
        }
    }

    /// Runs `callback` for the observational events it asks for, replacing the callback registered before.
    pub fn register_callback(mut self, callback: &'h mut impl Callback<B>) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Hooks that hand every event and every search event to `relay`, and the columns a relayed pricer adds to
    /// `priced`: what a backend that runs another one inside it hands that one.
    pub fn relay(relay: &'h mut impl Relay<B>, priced: Option<&'h mut Priced>) -> Self {
        Self {
            relay: Some(relay),
            priced,
            ..Self::new()
        }
    }

    /// The hooks with the solver's own `watch`, run before the callback, and the sink `priced` in place of any
    /// given before, for a solve that lasts no longer than both.
    pub(crate) fn install<'x>(self, watch: &'x mut dyn Callback<B>, priced: &'x mut Priced) -> Hooks<'x, B>
    where
        'h: 'x,
    {
        // Each field is moved on its own, so that its trait object takes the shorter lifetime.
        Hooks {
            watch: Some(watch),
            callback: self.callback.map(|c| -> &'x mut dyn Callback<B> { c }),
            lazy: self.lazy.map(|h| -> &'x mut dyn LazySlot<B> { h }),
            cut: self.cut.map(|h| -> &'x mut dyn CutSlot<B> { h }),
            lazy_and_cut: self.lazy_and_cut.map(|h| -> &'x mut dyn LazyCutSlot<B> { h }),
            heuristic: self.heuristic.map(|h| -> &'x mut dyn HeuristicSlot<B> { h }),
            pricer: self.pricer.map(|h| -> &'x mut dyn PriceSlot<B> { h }),
            relay: self.relay.map(|h| -> &'x mut dyn Relay<B> { h }),
            priced: Some(priced),
        }
    }

    /// The observational events the callbacks ask for that the backend delivers.
    #[must_use]
    pub fn events(&self) -> Events {
        let asked = [
            self.watch.as_deref().map(|c| c.events()),
            self.callback.as_deref().map(|c| c.events()),
            self.relay.as_deref().map(|c| c.events()),
        ];
        asked.into_iter().flatten().fold(Events::NONE, Events::union) & B::EVENTS
    }

    /// Runs `call` on each callback that asks for `event`: the solver's own first, then the registered one, then the
    /// relayed hooks.
    fn each(&mut self, event: Events, mut call: impl FnMut(&mut dyn Callback<B>)) {
        if let Some(watch) = self.watch.as_deref_mut()
            && watch.events().contains(event)
        {
            call(watch);
        }
        if let Some(callback) = self.callback.as_deref_mut()
            && callback.events().contains(event)
        {
            call(callback);
        }
        if let Some(relay) = self.relay.as_deref_mut()
            && relay.events().contains(event)
        {
            call(relay);
        }
    }

    /// Hands a log chunk to each callback that asks for [`Events::LOG`].
    pub fn log(&mut self, ctx: &mut B::Log<'_>) {
        self.each(Events::LOG, |c| c.log(ctx));
    }

    /// Hands a progress report to each callback that asks for [`Events::PROGRESS`].
    pub fn progress(&mut self, ctx: &mut B::Progress<'_>) {
        self.each(Events::PROGRESS, |c| c.progress(ctx));
    }

    /// Hands a poll to each callback that asks for [`Events::POLL`].
    pub fn poll(&mut self, ctx: &mut B::Poll<'_>) {
        self.each(Events::POLL, |c| c.poll(ctx));
    }

    /// Hands a feasible solution to each callback that asks for [`Events::FEASIBLE`].
    pub fn feasible(&mut self, ctx: &mut B::Solution<'_>) {
        self.each(Events::FEASIBLE, |c| c.feasible(ctx));
    }

    /// Hands a new incumbent to each callback that asks for [`Events::INCUMBENT`].
    pub fn incumbent(&mut self, ctx: &mut B::Solution<'_>) {
        self.each(Events::INCUMBENT, |c| c.incumbent(ctx));
    }

    /// Hands a native event to each callback that asks for [`Events::NATIVE`].
    pub fn native(&mut self, ctx: &mut B::Native<'_>) {
        self.each(Events::NATIVE, |c| c.native(ctx));
    }

    /// The sink of the columns the pricer adds, made for the loaded program: `Some` when a pricer is registered, or
    /// relayed with a sink, the first time it is asked for, and `None` after and for hooks without a pricer. A
    /// backend that prices takes it once, before it solves, and hands every column the pricer adds to it.
    pub fn take_priced(&mut self) -> Option<&'h mut Priced> {
        if self.pricer.is_none() && self.relay.is_none() {
            return None;
        }
        self.priced.take()
    }
}

impl<'h, B: LazyConstraints> Hooks<'h, B> {
    /// Runs `handler` for every candidate solution, replacing the lazy-constraint handler registered before.
    pub fn register_lazy(mut self, handler: &'h mut impl LazyHandler<B>) -> Self {
        self.lazy = Some(handler);
        self
    }

    /// Whether a lazy-constraint handler is registered.
    #[must_use]
    pub fn wants_lazy(&self) -> bool {
        self.lazy.is_some() || self.lazy_and_cut.is_some() || self.relay.as_deref().is_some_and(|r| r.wants_lazy())
    }

    /// Hands a candidate solution to the lazy-constraint handler, if one is registered.
    pub fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        if let Some(handler) = self.lazy.as_deref_mut() {
            handler.lazy(ctx);
        } else if let Some(handler) = self.lazy_and_cut.as_deref_mut() {
            handler.lazy(ctx);
        } else if let Some(relay) = self.relay.as_deref_mut()
            && relay.wants_lazy()
        {
            relay.lazy(ctx);
        }
    }
}

impl<'h, B: UserCuts> Hooks<'h, B> {
    /// Runs `handler` for node relaxations, replacing the cut handler registered before.
    pub fn register_cut(mut self, handler: &'h mut impl CutHandler<B>) -> Self {
        self.cut = Some(handler);
        self
    }

    /// Whether a cut handler is registered.
    #[must_use]
    pub fn wants_cut(&self) -> bool {
        self.cut.is_some() || self.lazy_and_cut.is_some() || self.relay.as_deref().is_some_and(|r| r.wants_cut())
    }

    /// Hands a node relaxation to the cut handler, if one is registered.
    pub fn cut(&mut self, ctx: &mut B::Cut<'_>) {
        if let Some(handler) = self.cut.as_deref_mut() {
            handler.cut(ctx);
        } else if let Some(handler) = self.lazy_and_cut.as_deref_mut() {
            handler.cut(ctx);
        } else if let Some(relay) = self.relay.as_deref_mut()
            && relay.wants_cut()
        {
            relay.cut(ctx);
        }
    }
}

impl<'h, B: LazyConstraints + UserCuts> Hooks<'h, B> {
    /// Runs one `handler` both for every candidate solution and for node relaxations, as a separator that cuts off
    /// integer and fractional points alike, replacing the lazy-constraint and cut handlers registered before.
    pub fn register_lazy_and_cut(mut self, handler: &'h mut (impl LazyHandler<B> + CutHandler<B>)) -> Self {
        self.lazy = None;
        self.cut = None;
        self.lazy_and_cut = Some(handler);
        self
    }
}

impl<'h, B: Heuristics> Hooks<'h, B> {
    /// Runs `handler` at the points the backend takes heuristic solutions, replacing the heuristic handler registered
    /// before.
    pub fn register_heuristic(mut self, handler: &'h mut impl HeuristicHandler<B>) -> Self {
        self.heuristic = Some(handler);
        self
    }

    /// Whether a heuristic handler is registered.
    #[must_use]
    pub fn wants_heuristic(&self) -> bool {
        self.heuristic.is_some() || self.relay.as_deref().is_some_and(|r| r.wants_heuristic())
    }

    /// Hands a heuristic point to the heuristic handler, if one is registered.
    pub fn heuristic(&mut self, ctx: &mut B::Heuristic<'_>) {
        if let Some(handler) = self.heuristic.as_deref_mut() {
            handler.heuristic(ctx);
        } else if let Some(relay) = self.relay.as_deref_mut()
            && relay.wants_heuristic()
        {
            relay.heuristic(ctx);
        }
    }
}

impl<'h, B: Prices> Hooks<'h, B> {
    /// Runs `pricer` at every node LP and hands the columns it adds to `priced`, made for the loaded program,
    /// replacing the pricer and the sink registered before.
    pub fn register_pricer(mut self, pricer: &'h mut impl Pricer<B>, priced: &'h mut Priced) -> Self {
        self.pricer = Some(pricer);
        self.priced = Some(priced);
        self
    }

    /// Runs `pricer` at every node LP without a sink yet: the [`Solver`](crate::Solver) gives its own with
    /// [`Hooks::install`] before the backend sees the hooks.
    pub(crate) fn solver_pricer(mut self, pricer: &'h mut impl Pricer<B>) -> Self {
        self.pricer = Some(pricer);
        self
    }

    /// Whether a pricer is registered.
    #[must_use]
    pub fn wants_price(&self) -> bool {
        self.pricer.is_some() || self.relay.as_deref().is_some_and(|r| r.wants_price())
    }

    /// Hands a pricing round to the pricer, if one is registered.
    pub fn price(&mut self, ctx: &mut B::Price<'_>) {
        if let Some(pricer) = self.pricer.as_deref_mut() {
            pricer.price(ctx);
        } else if let Some(relay) = self.relay.as_deref_mut()
            && relay.wants_price()
        {
            relay.price(ctx);
        }
    }
}

impl<B: Backend> Default for Hooks<'_, B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> fmt::Debug for Hooks<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hooks")
            .field("watch", &self.watch.is_some())
            .field("callback", &self.callback.is_some())
            .field("lazy", &self.lazy.is_some())
            .field("cut", &self.cut.is_some())
            .field("lazy_and_cut", &self.lazy_and_cut.is_some())
            .field("heuristic", &self.heuristic.is_some())
            .field("pricer", &self.pricer.is_some())
            .field("relay", &self.relay.is_some())
            .field("priced", &self.priced.is_some())
            .finish()
    }
}
