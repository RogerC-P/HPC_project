# Polybench Parallelization

## GEMM Implementations

|Implementation Name|Link|Notes|
|---|---|---|
|`gemm-mpi`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemm/gemm-mpi.c) [Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemm/customdatasizes/gemm.h)||
|`gemm-openmp`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemm/gemm-openmp.c) [Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemm/customdatasizes/gemm.h)||
|`compilation scripts and other`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/tree/master/scripts/gemm)||

## LUDCMP Implementations

|Implementation Name|Link|Notes|
|---|---|---|
|`ludcmp-polybench`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp.c)|Base implementation from PolyBench|
|`ludcmp-blas`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp-blas.c)|Uses LAPACK routines, can be used with OpenBLAS or Intel MKL|
|`ludcmp-blocking`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp-blocking.c)||
|`ludcmp-blocking-openmp`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp-blocking-openmp.c)||
|`ludcmp-blocking-openmp-fma`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp-blocking-openmp-fma.c)||
