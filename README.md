# Polybench Parallelization

## GEMM Implementations

|Implementation Name|Link|Notes|
|---|---|---|
|`gemm-openmp`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/georg/shared/gemm.h)|
|`gemm-mkl`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/mkl/linear-algebra/blas/gemm/gemm.c)||
|`gemm-openblas`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/openblas/linear-algebra/blas/gemm/gemm.c)||
|`gemm-mpi`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemm/gemm-mpi.c) [Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/blas/gemm/customdatasizes/gemm.h)||

## LUDCMP Implementations

|Implementation Name|Link|Notes|
|---|---|---|
|`ludcmp-polybench`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp.c)|Base implementation from PolyBench|
|`ludcmp-blas`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp-blas.c)|Uses LAPACK routines, can be used with OpenBLAS or Intel MKL|
|`ludcmp-blocking`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp-blocking.c)||
|`ludcmp-blocking-openmp`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp-blocking-openmp.c)||
|`ludcmp-blocking-openmp-fma`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/master/linear-algebra/solvers/ludcmp/ludcmp-blocking-openmp-fma.c)||
|`ludcmp-mpi`|[Open](https://github.com/fabianboesiger/PolyBenchC-4.2.1/blob/georg/shared/lu.h)|
