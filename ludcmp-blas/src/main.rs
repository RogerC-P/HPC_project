#![feature(test)]

extern crate test;

use test::black_box;

use std::time::{Instant, Duration};

use ndarray::*;
use ndarray_linalg::*;

const N: usize = 2048;
const ITERATIONS: usize = 10;

fn init_arrays(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut a = Array2::zeros((n, n));
    let mut b = Array1::zeros(n);
    for i in 0..n {
        b[i] = (i as f64 + 1.0) / (n as f64) / 2.0 + 4.0;
    }
    for i in 0..n {
        for j in 0..(i+1) {
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
    F: Fn(Array2<f64>, Array1<f64>) -> Result<(), error::LinalgError>
{
    let (a, b) = init_arrays(n);
    let start = Instant::now();
    f(a, b).unwrap();
    Instant::now() - start
}

fn main() {
    let mut duration = Duration::ZERO;
    for _ in 0..ITERATIONS {
        duration += time(factorize, N);
    }
    duration /= ITERATIONS as u32;
    println!("lu factorize duration: {} ms", duration.as_millis());
}