// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HiGHS's options: the commonly used ones typed, any other by name.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use culmen_core::{RelativeGap, relative_gap};

use crate::HighsError;

/// An option value of one of HiGHS's four kinds.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OptionValue {
    /// A `bool` option.
    Bool(bool),
    /// An integer option.
    Int(i32),
    /// A `double` option; written in the form of [`culmen_core::serde_real`], so an infinity reads back.
    Double(#[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real"))] f64),
    /// A string option.
    String(String),
}

impl fmt::Display for OptionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Double(value) => write!(f, "{value}"),
            Self::String(value) => write!(f, "{value:?}"),
        }
    }
}

impl From<bool> for OptionValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i32> for OptionValue {
    fn from(value: i32) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for OptionValue {
    fn from(value: f64) -> Self {
        Self::Double(value)
    }
}

impl From<&str> for OptionValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<String> for OptionValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

/// A typed option's value as HiGHS takes it.
trait ToOption {
    /// The value of the option `name` as HiGHS takes it.
    ///
    /// # Errors
    ///
    /// [`HighsError::OutOfRange`] for an exact integer beyond HiGHS's integers.
    fn to_option(&self, name: &'static str) -> Result<OptionValue, HighsError>;
}

impl ToOption for bool {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        Ok(OptionValue::Bool(*self))
    }
}

impl ToOption for i32 {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        Ok(OptionValue::Int(*self))
    }
}

impl ToOption for u32 {
    /// An exact integer, such as a seed: one beyond HiGHS's integers is refused, since no other value means the
    /// same.
    fn to_option(&self, name: &'static str) -> Result<OptionValue, HighsError> {
        exact(name, u64::from(*self))
    }
}

impl ToOption for u64 {
    /// A limit: one beyond HiGHS's integers is HiGHS's own "no limit", its largest integer.
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        Ok(limit(*self))
    }
}

impl ToOption for f64 {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        Ok(OptionValue::Double(*self))
    }
}

impl ToOption for String {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        Ok(OptionValue::String(self.clone()))
    }
}

/// A limit as HiGHS's integer: one beyond HiGHS's integers is HiGHS's own "no limit", its largest integer.
pub(crate) fn limit(value: u64) -> OptionValue {
    OptionValue::Int(i32::try_from(value).unwrap_or(i32::MAX))
}

/// An exact integer, such as a seed or a thread count, as the value of HiGHS's integer option `name`.
///
/// # Errors
///
/// [`HighsError::OutOfRange`] when `value` exceeds HiGHS's largest integer.
pub(crate) fn exact(name: &'static str, value: u64) -> Result<OptionValue, HighsError> {
    i32::try_from(value)
        .map(OptionValue::Int)
        .map_err(|_| HighsError::OutOfRange {
            name,
            value,
            max: i32::MAX.unsigned_abs().into(),
        })
}

/// Off, on, or HiGHS's choice: `presolve`, `parallel`, `run_crossover`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Switch {
    /// Never.
    Off,
    /// HiGHS decides.
    Choose,
    /// Always.
    On,
}

impl ToOption for Switch {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        let value = match self {
            Self::Off => "off",
            Self::Choose => "choose",
            Self::On => "on",
        };
        Ok(OptionValue::from(value))
    }
}

/// A switch HiGHS takes as the string `"on"` or `"off"`, as `ranging`, from a `bool`.
#[expect(
    clippy::unnecessary_wraps,
    reason = "a typed option's conversion returns the result every other conversion returns"
)]
fn on_off(value: &bool, _: &'static str) -> Result<OptionValue, HighsError> {
    Ok(OptionValue::from(if *value { "on" } else { "off" }))
}

/// The LP or QP solver: `solver`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LpSolver {
    /// HiGHS decides.
    Choose,
    /// The simplex method.
    Simplex,
    /// HiGHS's choice of interior-point solver.
    Ipm,
    /// The IPX interior-point solver.
    Ipx,
    /// The HiPO interior-point solver, which HiGHS takes only when it finds its HiPO extension library at run time;
    /// without it a solve asking for HiPO fails with [`HighsError::Option`].
    Hipo,
    /// The PDLP first-order solver.
    Pdlp,
    /// The HiPDLP first-order solver.
    Hipdlp,
    /// The active-set QP solver, HiGHS's solver for a QP.
    QpAsm,
}

impl ToOption for LpSolver {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        let value = match self {
            Self::Choose => "choose",
            Self::Simplex => "simplex",
            Self::Ipm => "ipm",
            Self::Ipx => "ipx",
            Self::Hipo => "hipo",
            Self::Pdlp => "pdlp",
            Self::Hipdlp => "hipdlp",
            Self::QpAsm => "qpasm",
        };
        Ok(OptionValue::from(value))
    }
}

/// The LP solver inside the MIP solver: `mip_lp_solver`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MipLpSolver {
    /// HiGHS decides.
    Choose,
    /// The simplex method.
    Simplex,
    /// HiGHS's choice of interior-point solver.
    Ipm,
    /// The IPX interior-point solver.
    Ipx,
    /// The HiPO interior-point solver, which HiGHS takes only when it finds its HiPO extension library at run time;
    /// without it a solve asking for HiPO fails with [`HighsError::Option`].
    Hipo,
}

impl ToOption for MipLpSolver {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        let value = match self {
            Self::Choose => "choose",
            Self::Simplex => "simplex",
            Self::Ipm => "ipm",
            Self::Ipx => "ipx",
            Self::Hipo => "hipo",
        };
        Ok(OptionValue::from(value))
    }
}

/// The interior-point solver inside the MIP solver: `mip_ipm_solver`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MipIpmSolver {
    /// HiGHS decides.
    Choose,
    /// HiGHS's choice of interior-point solver.
    Ipm,
    /// The IPX interior-point solver.
    Ipx,
    /// The HiPO interior-point solver, which HiGHS takes only when it finds its HiPO extension library at run time;
    /// without it a solve asking for HiPO fails with [`HighsError::Option`].
    Hipo,
}

impl ToOption for MipIpmSolver {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        let value = match self {
            Self::Choose => "choose",
            Self::Ipm => "ipm",
            Self::Ipx => "ipx",
            Self::Hipo => "hipo",
        };
        Ok(OptionValue::from(value))
    }
}

/// The simplex strategy: `simplex_strategy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SimplexStrategy {
    /// HiGHS decides.
    Choose,
    /// Dual simplex, serial.
    Dual,
    /// Dual simplex, SIP parallelism.
    DualSip,
    /// Dual simplex, PAMI parallelism.
    DualPami,
    /// Primal simplex.
    Primal,
}

impl ToOption for SimplexStrategy {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        let value = match self {
            Self::Choose => 0,
            Self::Dual => 1,
            Self::DualSip => 2,
            Self::DualPami => 3,
            Self::Primal => 4,
        };
        Ok(OptionValue::Int(value))
    }
}

/// The simplex scaling strategy: `simplex_scale_strategy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ScaleStrategy {
    /// No scaling.
    Off,
    /// HiGHS decides.
    Choose,
    /// Equilibration.
    Equilibration,
    /// Forced equilibration.
    ForcedEquilibration,
    /// Max-value scaling.
    MaxValue,
}

impl ToOption for ScaleStrategy {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        let value = match self {
            Self::Off => 0,
            Self::Choose => 1,
            Self::Equilibration => 2,
            Self::ForcedEquilibration => 3,
            Self::MaxValue => 4,
        };
        Ok(OptionValue::Int(value))
    }
}

/// A simplex edge-weight strategy: `simplex_dual_edge_weight_strategy`, `simplex_primal_edge_weight_strategy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EdgeWeight {
    /// HiGHS decides.
    Choose,
    /// Dantzig pricing.
    Dantzig,
    /// Devex pricing.
    Devex,
    /// Steepest-edge pricing.
    SteepestEdge,
}

impl ToOption for EdgeWeight {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        let value = match self {
            Self::Choose => -1,
            Self::Dantzig => 0,
            Self::Devex => 1,
            Self::SteepestEdge => 2,
        };
        Ok(OptionValue::Int(value))
    }
}

/// Which presolve rules HiGHS allows: `presolve_mode`, HiGHS's `PresolveMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PresolveMode {
    /// HiGHS decides: the light rules for a PDLP solver, every rule otherwise.
    Choose,
    /// Only the rules of presolve's initial sweep.
    InitialSweep,
    /// The light rules.
    Light,
    /// Every rule.
    Full,
}

impl ToOption for PresolveMode {
    fn to_option(&self, _: &'static str) -> Result<OptionValue, HighsError> {
        let value = match self {
            Self::Choose => 0,
            Self::InitialSweep => 1,
            Self::Light => 2,
            Self::Full => 3,
        };
        Ok(OptionValue::Int(value))
    }
}

/// Declares the typed options: the struct, its field docs, and the list of `(name, value)` it sets.
macro_rules! options {
    (@value $value:ident, $name:literal) => {
        $value.to_option($name)
    };
    (@value $value:ident, $name:literal, $convert:path) => {
        $convert($value, $name)
    };
    ($($(#[$doc:meta])* $field:ident: $ty:ty = $name:literal $(as $convert:path)?,)*) => {
        /// HiGHS's options for every solve of a [`Highs`](crate::Highs) backend.
        ///
        /// The commonly used options are typed fields, each named as HiGHS names it and `None` for HiGHS's default;
        /// any other option is set by name through [`HighsOptions::option`], after the typed ones. The
        /// [`Settings`](culmen_core::Settings) of a solve apply after both, so a per-solve limit always wins. What
        /// `Settings` types (the time limit, the gaps, the feasibility and optimality tolerances, threads, the seed,
        /// presolve, the node and solution limits, the console) has no field here; set by HiGHS's name, such an option
        /// holds for every solve whose settings leave it unset.
        ///
        /// An `f64` field is written in the form of [`culmen_core::serde_real`], so an infinity reads back.
        ///
        /// A `u64` field is a limit or a count, and one beyond HiGHS's integers is HiGHS's largest integer: for a
        /// limit its "no limit", for a count or threshold the largest HiGHS takes. A `u32` field is an exact integer,
        /// and one beyond HiGHS's integers fails the solve with [`HighsError::OutOfRange`].
        ///
        /// ```
        /// use culmen_highs::{HighsOptions, LpSolver, OptionValue};
        ///
        /// let options = HighsOptions {
        ///     solver: Some(LpSolver::Simplex),
        ///     mip_heuristic_effort: Some(0.2),
        ///     ..HighsOptions::default()
        /// }
        /// .option("mip_pool_soft_limit", 5000);
        /// assert_eq!(options.options[0].name, "mip_pool_soft_limit");
        /// assert_eq!(options.options[0].value, OptionValue::Int(5000));
        /// ```
        #[derive(Clone, Debug, PartialEq)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        pub struct HighsOptions {
            $($(#[$doc])* #[doc = concat!("\n\nHiGHS's `", $name, "`.")] pub $field: Option<$ty>,)*
            /// Further options by HiGHS's name, set in order after the typed ones.
            pub options: Vec<HighsOption>,
            /// The largest relative primal-dual objective error, `|P - D| / (1 + |P| + |D|)`, at which a QP optimum
            /// HiGHS reports is taken as optimal; above it the solve fails with
            /// [`HighsError::UnsoundQp`](crate::HighsError::UnsoundQp). The backend's own threshold, not a HiGHS
            /// option; `1e-5` by default.
            ///
            /// HiGHS's QP solver can report optimality at a point whose primal and dual objectives disagree by tens of
            /// percent, and that point need not even be feasible: 1.15 did so after its iterates turned NaN, and 1.16
            /// still does on a small share of edited QPs. On random convex QPs its correct optima stay below `1e-6`; a
            /// badly scaled QP may need a looser threshold.
            pub qp_objective_error: RelativeGap,
        }

        /// Every HiGHS option at HiGHS's default, and the default `qp_objective_error`.
        impl Default for HighsOptions {
            fn default() -> Self {
                Self {
                    $($field: None,)*
                    options: Vec::new(),
                    qp_objective_error: relative_gap!(1e-5),
                }
            }
        }

        impl HighsOptions {
            /// Every typed option set here, as `(name, value)`.
            ///
            /// # Errors
            ///
            /// [`HighsError::OutOfRange`] for the first typed option whose value HiGHS's integers cannot hold.
            pub(crate) fn assignments(&self) -> Result<Vec<(&'static str, OptionValue)>, HighsError> {
                let mut set = Vec::new();
                $(if let Some(value) = &self.$field {
                    set.push(($name, options!(@value value, $name $(, $convert)?)?));
                })*
                Ok(set)
            }
        }
    };
}

options! {
    /// Which presolve rules HiGHS allows when it presolves.
    presolve_mode: PresolveMode = "presolve_mode",
    /// The LP or QP solver.
    solver: LpSolver = "solver",
    /// Whether to use parallelism.
    parallel: Switch = "parallel",
    /// Whether to run crossover after an interior-point solve.
    run_crossover: Switch = "run_crossover",
    /// Whether to compute cost, bound and right-hand-side ranging.
    ranging: bool = "ranging" as on_off,
    /// The magnitude at which a cost counts as infinite.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    infinite_cost: f64 = "infinite_cost",
    /// The magnitude at which a bound counts as infinite.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    infinite_bound: f64 = "infinite_bound",
    /// The magnitude at or below which a matrix entry counts as zero.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    small_matrix_value: f64 = "small_matrix_value",
    /// The magnitude at or above which a matrix entry counts as infinite.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    large_matrix_value: f64 = "large_matrix_value",
    /// One tolerance for every feasibility and optimality measure.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    kkt_tolerance: f64 = "kkt_tolerance",
    /// The primal residual tolerance.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    primal_residual_tolerance: f64 = "primal_residual_tolerance",
    /// The dual residual tolerance.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    dual_residual_tolerance: f64 = "dual_residual_tolerance",
    /// The optimality tolerance.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    optimality_tolerance: f64 = "optimality_tolerance",
    /// The objective bound at which the dual simplex stops.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    objective_bound: f64 = "objective_bound",
    /// The objective value at which a MIP stops.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    objective_target: f64 = "objective_target",
    /// The exponent of a power-of-two scaling of the objective.
    user_objective_scale: i32 = "user_objective_scale",
    /// The exponent of a power-of-two scaling of the bounds.
    user_bound_scale: i32 = "user_bound_scale",
    /// The simplex strategy.
    simplex_strategy: SimplexStrategy = "simplex_strategy",
    /// The simplex scaling strategy.
    simplex_scale_strategy: ScaleStrategy = "simplex_scale_strategy",
    /// The dual simplex edge weights.
    simplex_dual_edge_weight_strategy: EdgeWeight = "simplex_dual_edge_weight_strategy",
    /// The primal simplex edge weights.
    simplex_primal_edge_weight_strategy: EdgeWeight = "simplex_primal_edge_weight_strategy",
    /// The simplex iteration limit.
    simplex_iteration_limit: u64 = "simplex_iteration_limit",
    /// The limit on simplex UPDATE operations before refactorization.
    simplex_update_limit: u64 = "simplex_update_limit",
    /// The most concurrency of parallel simplex.
    simplex_max_concurrency: u32 = "simplex_max_concurrency",
    /// Whether HiGHS writes a log at all. Every solve turns it on before applying these options, so `false` here
    /// silences log events and leaves [`HighsStatus::last_error`](crate::HighsStatus::last_error) empty.
    output_flag: bool = "output_flag",
    /// A file the log is written to.
    log_file: String = "log_file",
    /// Whether the MIP solver detects symmetry.
    mip_detect_symmetry: bool = "mip_detect_symmetry",
    /// Whether the MIP solver may restart.
    mip_allow_restart: bool = "mip_allow_restart",
    /// The limit on nodes whose estimate exceeds the cutoff.
    mip_max_stall_nodes: u64 = "mip_max_stall_nodes",
    /// The node limit when completing a partial start.
    mip_max_start_nodes: u64 = "mip_max_start_nodes",
    /// The MIP leaf-node limit.
    mip_max_leaves: u64 = "mip_max_leaves",
    /// Whether MIP presolve runs only at the root.
    mip_root_presolve_only: bool = "mip_root_presolve_only",
    /// The age at which dynamic LP rows leave the MIP relaxation.
    mip_lp_age_limit: u32 = "mip_lp_age_limit",
    /// The age at which cut-pool rows are deleted.
    mip_pool_age_limit: u32 = "mip_pool_age_limit",
    /// The soft limit on cut-pool rows.
    mip_pool_soft_limit: u64 = "mip_pool_soft_limit",
    /// The observations before pseudo costs count as reliable.
    mip_pscost_minreliable: u64 = "mip_pscost_minreliable",
    /// The clique-table size at which conflict-graph queries run in parallel.
    mip_min_cliquetable_entries_for_parallelism: u64 = "mip_min_cliquetable_entries_for_parallelism",
    /// The share of effort spent on heuristics, from 0 to 1.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    mip_heuristic_effort: f64 = "mip_heuristic_effort",
    /// Whether the feasibility-jump heuristic runs.
    mip_heuristic_run_feasibility_jump: bool = "mip_heuristic_run_feasibility_jump",
    /// Whether the RINS heuristic runs.
    mip_heuristic_run_rins: bool = "mip_heuristic_run_rins",
    /// Whether the RENS heuristic runs.
    mip_heuristic_run_rens: bool = "mip_heuristic_run_rens",
    /// Whether the root reduced-cost heuristic runs.
    mip_heuristic_run_root_reduced_cost: bool = "mip_heuristic_run_root_reduced_cost",
    /// Whether the ZI-round heuristic runs.
    mip_heuristic_run_zi_round: bool = "mip_heuristic_run_zi_round",
    /// Whether the shifting heuristic runs.
    mip_heuristic_run_shifting: bool = "mip_heuristic_run_shifting",
    /// Whether cuts are separated at nodes other than the root.
    mip_allow_cut_separation_at_nodes: bool = "mip_allow_cut_separation_at_nodes",
    /// The least time between two lines of the MIP log, in seconds.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    mip_min_logging_interval: f64 = "mip_min_logging_interval",
    /// The LP solver inside the MIP solver.
    mip_lp_solver: MipLpSolver = "mip_lp_solver",
    /// The interior-point solver inside the MIP solver.
    mip_ipm_solver: MipIpmSolver = "mip_ipm_solver",
    /// The interior-point optimality tolerance.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    ipm_optimality_tolerance: f64 = "ipm_optimality_tolerance",
    /// The interior-point iteration limit.
    ipm_iteration_limit: u64 = "ipm_iteration_limit",
    /// The PDLP iteration limit.
    pdlp_iteration_limit: u64 = "pdlp_iteration_limit",
    /// The PDLP optimality tolerance.
    #[cfg_attr(feature = "serde", serde(with = "culmen_core::serde_real::option"))]
    pdlp_optimality_tolerance: f64 = "pdlp_optimality_tolerance",
    /// The iteration limit of the active-set QP solver.
    qp_iteration_limit: u64 = "qp_iteration_limit",
    /// The largest null space the active-set QP solver works with.
    qp_nullspace_limit: u64 = "qp_nullspace_limit",
    /// Whether HiGHS may end as infeasible-or-unbounded without trying to tell which; it ends so either way when it
    /// cannot tell.
    allow_unbounded_or_infeasible: bool = "allow_unbounded_or_infeasible",
    /// The limit on presolve reductions; HiGHS's -1 is no limit.
    presolve_reduction_limit: i32 = "presolve_reduction_limit",
}

impl HighsOptions {
    /// Adds an option by HiGHS's name, set after the typed ones and every earlier one.
    #[must_use]
    pub fn option(mut self, name: impl Into<String>, value: impl Into<OptionValue>) -> Self {
        self.options.push(HighsOption {
            name: name.into(),
            value: value.into(),
        });
        self
    }
}

/// An option set by HiGHS's name, beside the typed fields of [`HighsOptions`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HighsOption {
    /// HiGHS's name of the option.
    pub name: String,
    /// The value set.
    pub value: OptionValue,
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::instance::Instance;

    /// HiGHS's largest integer, `kHighsIInf`, as HiGHS's documentation states it.
    const HIGHS_LARGEST: u64 = 2_147_483_647;

    #[test]
    fn test_exact_integers_beyond_highs_integers_are_refused_and_limits_are_none() {
        let with = |concurrency, age| {
            HighsOptions {
                simplex_max_concurrency: Some(concurrency),
                mip_lp_age_limit: Some(age),
                mip_max_leaves: Some(u64::MAX),
                ..HighsOptions::default()
            }
            .assignments()
        };
        assert_eq!(
            with(7, 2),
            Ok(vec![
                ("simplex_max_concurrency", OptionValue::Int(7)),
                ("mip_max_leaves", OptionValue::Int(2_147_483_647)),
                ("mip_lp_age_limit", OptionValue::Int(2)),
            ])
        );
        assert_eq!(
            with(u32::MAX, 2),
            Err(HighsError::OutOfRange {
                name: "simplex_max_concurrency",
                value: 4_294_967_295,
                max: 2_147_483_647,
            })
        );
        assert_eq!(
            with(7, 2_147_483_648),
            Err(HighsError::OutOfRange {
                name: "mip_lp_age_limit",
                value: 2_147_483_648,
                max: 2_147_483_647,
            })
        );
    }

    #[test]
    fn test_ranging_is_highs_string_switch() {
        let with = |ranging| {
            HighsOptions {
                ranging: Some(ranging),
                ..HighsOptions::default()
            }
            .assignments()
        };
        assert_eq!(with(true), Ok(vec![("ranging", OptionValue::from("on"))]));
        assert_eq!(with(false), Ok(vec![("ranging", OptionValue::from("off"))]));
    }

    #[test]
    fn test_highs_takes_every_choice_of_every_typed_option() {
        let mut every = Vec::new();
        for s in [Switch::Off, Switch::Choose, Switch::On] {
            every.push(HighsOptions {
                parallel: Some(s),
                run_crossover: Some(s),
                ..HighsOptions::default()
            });
        }
        for p in [
            PresolveMode::Choose,
            PresolveMode::InitialSweep,
            PresolveMode::Light,
            PresolveMode::Full,
        ] {
            every.push(HighsOptions {
                presolve_mode: Some(p),
                ..HighsOptions::default()
            });
        }
        for r in [false, true] {
            every.push(HighsOptions {
                ranging: Some(r),
                ..HighsOptions::default()
            });
        }
        for s in [
            LpSolver::Choose,
            LpSolver::Simplex,
            LpSolver::Ipm,
            LpSolver::Ipx,
            LpSolver::Hipo,
            LpSolver::Pdlp,
            LpSolver::Hipdlp,
            LpSolver::QpAsm,
        ] {
            every.push(HighsOptions {
                solver: Some(s),
                ..HighsOptions::default()
            });
        }
        for s in [
            MipLpSolver::Choose,
            MipLpSolver::Simplex,
            MipLpSolver::Ipm,
            MipLpSolver::Ipx,
            MipLpSolver::Hipo,
        ] {
            every.push(HighsOptions {
                mip_lp_solver: Some(s),
                ..HighsOptions::default()
            });
        }
        for s in [
            MipIpmSolver::Choose,
            MipIpmSolver::Ipm,
            MipIpmSolver::Ipx,
            MipIpmSolver::Hipo,
        ] {
            every.push(HighsOptions {
                mip_ipm_solver: Some(s),
                ..HighsOptions::default()
            });
        }
        for s in [
            SimplexStrategy::Choose,
            SimplexStrategy::Dual,
            SimplexStrategy::DualSip,
            SimplexStrategy::DualPami,
            SimplexStrategy::Primal,
        ] {
            every.push(HighsOptions {
                simplex_strategy: Some(s),
                ..HighsOptions::default()
            });
        }
        for s in [
            ScaleStrategy::Off,
            ScaleStrategy::Choose,
            ScaleStrategy::Equilibration,
            ScaleStrategy::ForcedEquilibration,
            ScaleStrategy::MaxValue,
        ] {
            every.push(HighsOptions {
                simplex_scale_strategy: Some(s),
                ..HighsOptions::default()
            });
        }
        for w in [
            EdgeWeight::Choose,
            EdgeWeight::Dantzig,
            EdgeWeight::Devex,
            EdgeWeight::SteepestEdge,
        ] {
            every.push(HighsOptions {
                simplex_dual_edge_weight_strategy: Some(w),
                simplex_primal_edge_weight_strategy: Some(w),
                ..HighsOptions::default()
            });
        }
        every.push(HighsOptions {
            qp_iteration_limit: Some(1000),
            qp_nullspace_limit: Some(u64::MAX),
            ..HighsOptions::default()
        });
        // HiGHS takes HiPO only when it finds HiPO's extension library on the machine it runs on.
        let hipo = OptionValue::from("hipo");
        let mut instance = Instance::new();
        for options in every {
            for (name, value) in options.assignments().unwrap() {
                if value != hipo {
                    assert_eq!(instance.set_option(name, &value), Ok(()), "{name} = {value}");
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_an_exact_integer_is_kept_or_refused(value in prop_oneof![0..=HIGHS_LARGEST + 2, any::<u64>()]) {
            let kept = exact("random_seed", value);
            if value <= HIGHS_LARGEST {
                prop_assert_eq!(kept, Ok(OptionValue::Int(i32::try_from(value).unwrap())));
            } else {
                prop_assert_eq!(kept, Err(HighsError::OutOfRange { name: "random_seed", value, max: HIGHS_LARGEST }));
            }
        }

        #[test]
        fn test_a_limit_beyond_highs_integers_is_no_limit(value in prop_oneof![0..=HIGHS_LARGEST + 2, any::<u64>()]) {
            let expected = i32::try_from(value.min(HIGHS_LARGEST)).unwrap();
            prop_assert_eq!(limit(value), OptionValue::Int(expected));
        }
    }
}
