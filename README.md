# Polybench Parallelization

## Checking the implementations

Use the script `scripts/checkImpls.py` to check the outputs of our polybench implementations.

## Timing the implementations

Use the script `scripts/timeImpls.py` to time our polybench implementations.

Our BLAS implementation is written in Rust. It is found in the directory `ludcmp-blas`. Run it with `cargo run --release`.
