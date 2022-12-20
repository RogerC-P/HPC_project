/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include <gemm.h>

#include <cblas.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"

/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		double *alpha,
		double *beta,
		double *C,
		double *A,
		double *B)
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i * nj + j] = (double) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i * nk + j] = (double) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i * nj + j] = (double) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj, double *C)
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i * nj + j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 double alpha,
		 double beta,
		 double *C, double *A, double *B)
{
#pragma scop
  #pragma omp parallel
  gemm(ni, nj, nk, alpha, A, nk, B, nj, beta, C, nj);
#pragma endscop
}


static
void kernel_gemm_blas(int ni, int nj, int nk,
		 double alpha,
		 double beta,
		 double *C, double *A, double *B)
{
#pragma scop
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              ni, nj, nk,
              alpha, A, nk,
              B, nj,
              beta, C, nj);
#pragma endscop
}

static
void kernel_gemm_original(int ni, int nj, int nk,
		 double alpha,
		 double beta,
		 double *C, double *A, double *B)
{
  int i, j, k;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++)
      C[i * nj + j] *= beta;
    for (k = 0; k < nk; k++) {
      for (j = 0; j < nj; j++)
        C[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
    }
  }
#pragma endscop

}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  double alpha;
  double beta;

  double *A;
  double *B;
  double *C;

  int n0 = (NI < (1 << 10)) ? NI : (1 << 10);

  for (int n = n0; n <= NI; n <<= 1) {
    printf("size: %d\n", n);

    A = (double *) malloc(n * n * sizeof(double));
    B = (double *) malloc(n * n * sizeof(double));
    C = (double *) malloc(n * n * sizeof(double));

    for (int k = 0; k < NUM_RUNS; k++) {
      /* Initialize array(s). */
      init_array (n, n, n, &alpha, &beta,
            C, A, B);

      /* Start timer. */
      polybench_start_instruments;

      /* Run kernel. */
      kernel_gemm_blas (n, n, n,
             alpha, beta,
             C, A, B);

      // /* Stop and print timer. */
      polybench_stop_instruments;
      polybench_print_instruments;
      /* Prevent dead-code elimination. All live-out data must be printed
         by the function call in argument. */
      polybench_prevent_dce(print_array(n, n,  C));
    }

    free(C);
    free(A);
    free(B);
  }

  /* Be clean. */
  return 0;
}
