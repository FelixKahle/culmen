// Copyright (c) 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A travelling salesman's shortest round trip through eight towns, with subtours cut off by lazy constraints on
//! SCIP. The model says only that every town has two roads; a round trip through all eight is one more condition, a
//! constraint for every set of towns, too many to write down. The callback (`Events::LAZY`) receives every
//! candidate SCIP would accept, finds its subtours, and adds for each one the constraint that some two roads leave
//! it. SCIP keeps each constraint for the rest of the search, so the optimum it proves is a round trip.
//!
//! `cargo run -p culmen --example lazy_constraints --features scip`

use std::collections::BTreeMap;
use std::time::Duration;

use culmen::LazyContext;
use culmen::prelude::*;

/// Each town's name and position, in kilometres.
const TOWNS: [(&str, f64, f64); 8] = [
    ("Ash", 0.0, 0.0),
    ("Birch", 30.0, 10.0),
    ("Cedar", 10.0, 40.0),
    ("Dale", 40.0, 50.0),
    ("Elm", 100.0, 10.0),
    ("Fern", 130.0, 0.0),
    ("Glen", 110.0, 50.0),
    ("Holt", 140.0, 40.0),
];

/// More candidates than this search can need; a defect ends here instead of looping.
const MAX_CANDIDATES: usize = 10_000;

/// The road between two towns, in whole kilometres.
fn distance(a: usize, b: usize) -> f64 {
    let ((_, ax, ay), (_, bx, by)) = (TOWNS[a], TOWNS[b]);
    (ax - bx).hypot(ay - by).round()
}

/// Cuts off every candidate made of more than one round trip.
struct Subtours {
    /// The road between each pair of towns `a < b`.
    road: BTreeMap<(usize, usize), Col>,
    candidates: usize,
    added: usize,
}

impl Subtours {
    /// The round trips the roads taken make, each as its towns in the order travelled. A walk that ends before it
    /// returns to its first town, which only a candidate breaking the two-roads rows can make, counts as a trip too:
    /// every set of towns but all of them must be left by two roads, so its constraint is valid all the same.
    fn tours(&self, taken: impl Fn(Col) -> bool) -> Vec<Vec<usize>> {
        let mut seen = [false; TOWNS.len()];
        let mut tours = Vec::new();
        for start in 0..TOWNS.len() {
            if seen[start] {
                continue;
            }
            seen[start] = true;
            let mut tour = vec![start];
            // Each step goes to a town not seen yet, so the walk ends within as many steps as there are towns.
            while let Some(next) = (0..TOWNS.len()).find(|&next| {
                let town = tour[tour.len() - 1];
                !seen[next]
                    && self
                        .road
                        .get(&(town.min(next), town.max(next)))
                        .is_some_and(|&x| taken(x))
            }) {
                seen[next] = true;
                tour.push(next);
            }
            tours.push(tour);
        }
        tours
    }
}

impl<B: Backend> Callback<B> for Subtours {
    fn events(&self) -> Events {
        Events::LAZY
    }

    fn lazy(&mut self, ctx: &mut B::Lazy<'_>) {
        self.candidates += 1;
        assert!(self.candidates <= MAX_CANDIDATES, "{} candidates", self.candidates);
        let candidate = ctx.candidate();
        let tours = self.tours(|x| candidate[x] > 0.5);
        if tours.len() == 1 {
            return;
        }
        for tour in tours {
            // At least two roads leave every set of towns a round trip passes through and leaves again.
            let leaving = self
                .road
                .iter()
                .filter(|&(&(a, b), _)| tour.contains(&a) != tour.contains(&b));
            ctx.add_lazy(LinExpr::sum(leaving.map(|(_, &x)| x)).geq(2))
                .expect("a valid constraint");
            self.added += 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Model::new();
    let pairs = (0..TOWNS.len()).flat_map(|a| (a + 1..TOWNS.len()).map(move |b| (a, b)));
    let road: BTreeMap<(usize, usize), Col> = pairs
        .map(|(a, b)| {
            (
                (a, b),
                m.add_var(Var::binary().name(format!("road[{},{}]", TOWNS[a].0, TOWNS[b].0))),
            )
        })
        .collect();
    for (town, &(name, ..)) in TOWNS.iter().enumerate() {
        let touching = road.iter().filter(|&(&(a, b), _)| a == town || b == town);
        m.add_named_constraint(
            format!("two[{name}]"),
            LinExpr::sum(touching.map(|(_, &x)| x)).equals(2),
        );
    }
    m.minimize(LinExpr::sum(road.iter().map(|(&(a, b), &x)| distance(a, b) * x)));
    let mut solver = Solver::new(Scip::new(), m.build()?)?;

    let request = Request::new()
        .settings_from(Settings {
            node_limit: Some(10_000),
            ..Settings::new()
        })
        .time_limit(Duration::from_secs(30));
    let mut subtours = Subtours {
        road,
        candidates: 0,
        added: 0,
    };
    let solved = solver.solve_with(request, &mut subtours)?;
    let Outcome::Optimal(best) = solved.outcome else {
        return Err(format!("no optimum: {:?}", solved.outcome).into());
    };

    let tours = subtours.tours(|x| best.values[x] > 0.5);
    let [tour] = tours.as_slice() else {
        return Err(format!("not one round trip: {tours:?}").into());
    };
    let names: Vec<&str> = tour.iter().map(|&town| TOWNS[town].0).collect();
    println!(
        "{} subtour constraints added over {} candidates",
        subtours.added, subtours.candidates
    );
    println!("round trip {}: {} km", names.join(" - "), best.objective);
    Ok(())
}
