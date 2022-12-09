#![feature(test)]

extern crate test;

use test::black_box;

use std::time::{Duration, Instant};

use clap::Parser;

use ndarray::*;
use ndarray_linalg::*;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    num_runs: usize,

    /// Number of times to greet
    #[arg(short, long)]
    dataset_size: usize,
}

fn init_arrays(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut a = Array2::zeros((n, n));
    let mut b = Array1::zeros(n);
    for i in 0..n {
        b[i] = (i as f64 + 1.0) / (n as f64) / 2.0 + 4.0;
    }
    for i in 0..n {
        for j in 0..(i + 1) {
            a[(i, j)] = (-(j as f64) % (n as f64)) / (n as f64) + 1.0;
        }
    }
    (a, b)
}

/*
// Solve `Ax=b`
fn solve(a: Array2<f64>, b: Array1<f64>) -> Result<(), error::LinalgError> {
    let x = a.solve(&b)?;
    black_box(x);
    Ok(())
}
*/

// Solve `Ax=b` for many b with fixed A
fn factorize(a: Array2<f64>, b: Array1<f64>) -> Result<(), error::LinalgError> {
    let f = a.factorize_into()?; // LU factorize A (A is consumed)
    let x = f.solve_into(b)?; // solve Ax=b using factorized L, U
    black_box(x);
    Ok(())
}

fn time<F>(f: F, n: usize) -> Duration
where
    F: Fn(Array2<f64>, Array1<f64>) -> Result<(), error::LinalgError>,
{
    let (a, b) = init_arrays(n);
    let start = Instant::now();
    f(a, b).unwrap();
    Instant::now() - start
}

fn main() {
    let args = Args::parse();

    let mut duration = Duration::ZERO;
    for _ in 0..args.num_runs {
        duration += time(factorize, args.dataset_size);
    }
    duration /= args.num_runs as u32;
    println!("{}", duration.as_secs_f32());
}
