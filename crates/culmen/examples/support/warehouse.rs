// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What both warehouse examples share: the data, the live view and the solve. Each example writes the model in its
//! own style.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use culmen::prelude::*;

/// A warehouse slotting instance: products with a pick rate and a volume, some of them frozen; locations with a
/// capacity and a walking distance, some of them cooled. A frozen product only fits a cooled location.
pub(crate) struct Warehouse {
    /// Every product, by index.
    pub(crate) products: Vec<usize>,
    /// Every location, by index.
    pub(crate) locations: Vec<usize>,
    /// Picks per day of each product.
    pub(crate) picks: Vec<f64>,
    /// Volume of each product, in litres.
    pub(crate) volume: Vec<f64>,
    /// Whether each product must be kept frozen.
    pub(crate) frozen: Vec<bool>,
    /// Walking distance to each location, in metres.
    pub(crate) distance: Vec<f64>,
    /// Capacity of each location, in litres.
    pub(crate) capacity: Vec<f64>,
    /// Whether each location is cooled.
    pub(crate) cooled: Vec<bool>,
}

impl Warehouse {
    /// A deterministic instance of 60 products and 30 locations.
    pub(crate) fn generate() -> Self {
        let (num_products, num_locations) = (60, 30);
        let mut state = 12_345_u64;
        let mut below = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n
        };
        let picks: Vec<f64> = (0..num_products).map(|_| (1 + below(50)) as f64).collect();
        let volume: Vec<f64> = (0..num_products).map(|_| (5 + below(30)) as f64).collect();
        let frozen: Vec<bool> = (0..num_products).map(|p| p % 6 == 0).collect();
        let distance: Vec<f64> = (0..num_locations).map(|l| 5.0 + 3.0 * l as f64).collect();
        let cooled: Vec<bool> = (0..num_locations).map(|l| l % 3 == 0).collect();
        let total: f64 = volume.iter().sum();
        let capacity = vec![(total / num_locations as f64 * 1.15).round(); num_locations];
        Self {
            products: (0..num_products).collect(),
            locations: (0..num_locations).collect(),
            picks,
            volume,
            frozen,
            distance,
            capacity,
            cooled,
        }
    }

    /// Whether product `p` may be stored at location `l`.
    pub(crate) fn fits(&self, p: usize, l: usize) -> bool {
        !self.frozen[p] || self.cooled[l]
    }

    /// The daily walking cost of storing `p` at `l`: there and back for every pick.
    pub(crate) fn walk(&self, p: usize, l: usize) -> f64 {
        self.picks[p] * 2.0 * self.distance[l]
    }
}

/// Prints the search as it goes: progress once a second, every new best solution, and HiGHS's log when asked.
struct Live {
    show_log: bool,
}

impl Observer for Live {
    fn progress(&mut self, p: &Progress) {
        let shown = |v: Option<f64>| v.map_or_else(|| "-".to_owned(), |v| format!("{v:.0}"));
        let gap = p.gap().map_or_else(|| "-".to_owned(), |g| format!("{:.2}%", 100.0 * g));
        let nodes = p.nodes.map_or_else(String::new, |n| format!(" nodes {n}"));
        println!(
            "[{:>5.1}s] best {:>8} bound {:>8} gap {gap:>8}{nodes}",
            p.elapsed.as_secs_f64(),
            shown(p.objective),
            shown(p.bound)
        );
    }

    fn improved(&mut self, i: &Improvement<'_>) {
        println!(
            "[{:>5.1}s] new best solution: {:.0}",
            i.progress.elapsed.as_secs_f64(),
            i.objective
        );
    }

    fn log(&mut self, line: &LogLine<'_>) {
        if self.show_log {
            println!("  highs | {}", line.text);
        }
    }
}

/// Solves the model, stopping on Ctrl-C or after the seconds given as the first argument (default 10), and prints
/// the assignment.
pub(crate) fn solve(
    warehouse: &Warehouse,
    program: Program,
    x: &VarMap<(usize, usize)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let seconds: u64 = std::env::args().nth(1).map_or(Ok(10), |s| s.parse())?;
    let show_log = std::env::args().any(|a| a == "--log");
    let stop = Arc::new(AtomicBool::new(false));
    let on_ctrl_c = Arc::clone(&stop);
    ctrlc::set_handler(move || on_ctrl_c.store(true, Ordering::Release))?;

    println!(
        "{} variables, {} constraints; stopping after {seconds} s or on Ctrl-C",
        program.num_cols(),
        program.num_rows()
    );
    let mut solver = Solver::new(Highs::new(), program)?;
    let mut live = Live { show_log };
    let request = Request::new()
        .time_limit(Duration::from_secs(seconds))
        .stop(&stop)
        .observer(&mut live);
    let solved = solver.solve(request)?;

    let (best, ending) = match &solved.outcome {
        Outcome::Optimal(best) => (best, "optimal".to_owned()),
        Outcome::GapReached { solution, bound } => (solution, format!("gap reached, bound {bound:.0}")),
        Outcome::Stopped {
            reason,
            best: Some(best),
            bound,
        } => (best, format!("stopped: {reason:?}, bound {bound:?}")),
        other => return Err(format!("no solution: {other:?}").into()),
    };
    println!(
        "{ending} after {:.1} s; walking {:.0} m per day",
        solved.elapsed.as_secs_f64(),
        best.objective
    );
    for (&(p, l), value) in x.values(&best.values) {
        if value > 0.5 && p < 10 {
            let kind = if warehouse.frozen[p] { "frozen" } else { "ambient" };
            println!("  product {p:>2} ({kind}) -> location {l:>2}");
        }
    }
    Ok(())
}
