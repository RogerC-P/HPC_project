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

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

#define BI 20
#define BJ 40
#define BK 12

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
#pragma scop
  __m256d valpha = _mm256_set1_pd(alpha);

  #pragma omp parallel for
  for (int i = 0; i < _PB_NI - BI + 1; i += BI) {
    int j;
    for (j = 0; j < _PB_NJ - BJ + 1; j += BJ) {
      for (int u = i; u < i + BI; u++) {
        for (int v = j; v < j + BI; v++) {
          C[u][v] *= beta;
        }
      }

      int k;
      for (k = 0; k < _PB_NK - BK + 1; k += BK) {
        for (int u = i; u < i + BI; u += 4) {
          for (int v = j; v < j + BJ; v += 8) {
            __m256d ab00 = _mm256_set1_pd(0.0);
            __m256d ab01 = _mm256_set1_pd(0.0);
            __m256d ab02 = _mm256_set1_pd(0.0);
            __m256d ab03 = _mm256_set1_pd(0.0);

            __m256d ab10 = _mm256_set1_pd(0.0);
            __m256d ab11 = _mm256_set1_pd(0.0);
            __m256d ab12 = _mm256_set1_pd(0.0);
            __m256d ab13 = _mm256_set1_pd(0.0);

            for (int w = k; w < k + BK; w++) {
              __m256d a0 = _mm256_set1_pd(A[u + 0][w]);
              __m256d a1 = _mm256_set1_pd(A[u + 1][w]);
              __m256d a2 = _mm256_set1_pd(A[u + 2][w]);
              __m256d a3 = _mm256_set1_pd(A[u + 3][w]);

              __m256d b0 = _mm256_load_pd(&B[w][v + 0]);
              __m256d b1 = _mm256_load_pd(&B[w][v + 4]);

              ab00 = _mm256_fmadd_pd(a0, b0, ab00);
              ab01 = _mm256_fmadd_pd(a1, b0, ab01);
              ab02 = _mm256_fmadd_pd(a2, b0, ab02);
              ab03 = _mm256_fmadd_pd(a3, b0, ab03);

              ab10 = _mm256_fmadd_pd(a0, b1, ab10);
              ab11 = _mm256_fmadd_pd(a1, b1, ab11);
              ab12 = _mm256_fmadd_pd(a2, b1, ab12);
              ab13 = _mm256_fmadd_pd(a3, b1, ab13);
            }

            __m256d c00 = _mm256_load_pd(&C[u + 0][v + 0]);
            __m256d c01 = _mm256_load_pd(&C[u + 1][v + 0]);
            __m256d c02 = _mm256_load_pd(&C[u + 2][v + 0]);
            __m256d c03 = _mm256_load_pd(&C[u + 3][v + 0]);

            __m256d c10 = _mm256_load_pd(&C[u + 0][v + 4]);
            __m256d c11 = _mm256_load_pd(&C[u + 1][v + 4]);
            __m256d c12 = _mm256_load_pd(&C[u + 2][v + 4]);
            __m256d c13 = _mm256_load_pd(&C[u + 3][v + 4]);

            c00 = _mm256_fmadd_pd(valpha, ab00, c00);
            c01 = _mm256_fmadd_pd(valpha, ab01, c01);
            c02 = _mm256_fmadd_pd(valpha, ab02, c02);
            c03 = _mm256_fmadd_pd(valpha, ab03, c03);

            c10 = _mm256_fmadd_pd(valpha, ab10, c10);
            c11 = _mm256_fmadd_pd(valpha, ab11, c11);
            c12 = _mm256_fmadd_pd(valpha, ab12, c12);
            c13 = _mm256_fmadd_pd(valpha, ab13, c13);

            _mm256_store_pd(&C[u + 0][v + 0], c00);
            _mm256_store_pd(&C[u + 1][v + 0], c01);
            _mm256_store_pd(&C[u + 2][v + 0], c02);
            _mm256_store_pd(&C[u + 3][v + 0], c03);

            _mm256_store_pd(&C[u + 0][v + 4], c10);
            _mm256_store_pd(&C[u + 1][v + 4], c11);
            _mm256_store_pd(&C[u + 2][v + 4], c12);
            _mm256_store_pd(&C[u + 3][v + 4], c13);
          }
        }
      }

      for (int u = i; u < i + BI; u += 4) {
        for (int v = j; v < j + BJ; v += 8) {
          __m256d ab00 = _mm256_set1_pd(0.0);
          __m256d ab01 = _mm256_set1_pd(0.0);
          __m256d ab02 = _mm256_set1_pd(0.0);
          __m256d ab03 = _mm256_set1_pd(0.0);

          __m256d ab10 = _mm256_set1_pd(0.0);
          __m256d ab11 = _mm256_set1_pd(0.0);
          __m256d ab12 = _mm256_set1_pd(0.0);
          __m256d ab13 = _mm256_set1_pd(0.0);

          for (int w = k; w < _PB_NK; w++) {
            __m256d a0 = _mm256_set1_pd(A[u + 0][w]);
            __m256d a1 = _mm256_set1_pd(A[u + 1][w]);
            __m256d a2 = _mm256_set1_pd(A[u + 2][w]);
            __m256d a3 = _mm256_set1_pd(A[u + 3][w]);

            __m256d b0 = _mm256_load_pd(&B[w][v + 0]);
            __m256d b1 = _mm256_load_pd(&B[w][v + 4]);

            ab00 = _mm256_fmadd_pd(a0, b0, ab00);
            ab01 = _mm256_fmadd_pd(a1, b0, ab01);
            ab02 = _mm256_fmadd_pd(a2, b0, ab02);
            ab03 = _mm256_fmadd_pd(a3, b0, ab03);

            ab10 = _mm256_fmadd_pd(a0, b1, ab10);
            ab11 = _mm256_fmadd_pd(a1, b1, ab11);
            ab12 = _mm256_fmadd_pd(a2, b1, ab12);
            ab13 = _mm256_fmadd_pd(a3, b1, ab13);
          }

          __m256d c00 = _mm256_load_pd(&C[u + 0][v + 0]);
          __m256d c01 = _mm256_load_pd(&C[u + 1][v + 0]);
          __m256d c02 = _mm256_load_pd(&C[u + 2][v + 0]);
          __m256d c03 = _mm256_load_pd(&C[u + 3][v + 0]);

          __m256d c10 = _mm256_load_pd(&C[u + 0][v + 4]);
          __m256d c11 = _mm256_load_pd(&C[u + 1][v + 4]);
          __m256d c12 = _mm256_load_pd(&C[u + 2][v + 4]);
          __m256d c13 = _mm256_load_pd(&C[u + 3][v + 4]);

          c00 = _mm256_fmadd_pd(valpha, ab00, c00);
          c01 = _mm256_fmadd_pd(valpha, ab01, c01);
          c02 = _mm256_fmadd_pd(valpha, ab02, c02);
          c03 = _mm256_fmadd_pd(valpha, ab03, c03);

          c10 = _mm256_fmadd_pd(valpha, ab10, c10);
          c11 = _mm256_fmadd_pd(valpha, ab11, c11);
          c12 = _mm256_fmadd_pd(valpha, ab12, c12);
          c13 = _mm256_fmadd_pd(valpha, ab13, c13);

          _mm256_store_pd(&C[u + 0][v + 0], c00);
          _mm256_store_pd(&C[u + 1][v + 0], c01);
          _mm256_store_pd(&C[u + 2][v + 0], c02);
          _mm256_store_pd(&C[u + 3][v + 0], c03);

          _mm256_store_pd(&C[u + 0][v + 4], c10);
          _mm256_store_pd(&C[u + 1][v + 4], c11);
          _mm256_store_pd(&C[u + 2][v + 4], c12);
          _mm256_store_pd(&C[u + 3][v + 4], c13);
        }
      }
    }

    for (; j < _PB_NJ; j++) {
      for (int u = i; u < i + BI; u++) {
        C[u][j] *= beta;
      }

      for (int k = 0; k < _PB_NK; k++) {
        for (int u = i; u < i + BI; u++) {
          C[u][j] += alpha * A[u][k] * B[k][j];
        }
      }
    }
  }

  int i = BI * (_PB_NI / BI);
  for (; i < _PB_NI; i++) {
    for (int j = 0; j < _PB_NJ; j++) {
      C[i][j] *= beta;
      for (int k = 0; k < _PB_NK; k++) {
        C[i][j] += alpha * A[i][k] * B[k][j];
      }
    }
  }
#pragma endscop

}

static
void kernel_gemm_original(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
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
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++)
	C[i][j] *= beta;
    for (k = 0; k < _PB_NK; k++) {
       for (j = 0; j < _PB_NJ; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
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

  polybench_program_total_flops = (double) ni * (double) nj * (2 * (double) nk - 1);

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
