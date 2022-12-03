/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* lu.c: this file is part of PolyBench/C */

#include <mpi.h>
#include <pthread.h>

#include <gemm.h>
#define PARALLEL_GEMM
#include <gemm.h>

/* Include polybench common header. */
#include <polybench.h>

void print(int n, double *A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) printf("%.2lf ", A[i * n + j]);
    printf("\n");
  }
}

void swap(double **a, double **b) {
  void *tmp = *a;
  *a = *b;
  *b = tmp;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void lu(int n, double *A) {
#pragma scop
  int block_size = 80;

  double *q = malloc(block_size * sizeof(double));

  #pragma omp parallel
  for (int l = 0; l < n - block_size + 1; l += block_size) {
    if (l > 0) {
      int l = block_size;
      pgemm(n - l, n - l, block_size,
            -1, A + l * n + (l - block_size), n,
            A + (l - block_size) * n + l, n,
            1, A + l * n + l, n);
    }

    // #pragma omp master
    // for (int k = l; k < l + block_size; k++) {
    //   double A_kk = A[k * n + k];
    //   for (int i = k + 1; i < l + block_size; i++) A[i * n + k] /= A_kk;

    //   for (int i = k + 1; i < l + block_size; i++) {
    //     for (int j = k + 1; j < l + block_size; j++) {
    //       A[i * n + j] -= A[i * n + k] * A[k * n + j];
    //     }
    //   }
    // }

    // #pragma omp for
    // for (int u = l + block_size; u < n - block_size + 1; u += block_size) {
    //   for (int k = l; k < l + block_size; k++) {
    //     for (int i = k + 1; i < l + block_size; i++) {
    //       for (int j = u; j < u + block_size; j++) {
    //         A[i * n + j] -= A[i * n + k] * A[k * n + j];
    //       }
    //     }
    //   }
    // }

    // #pragma omp master
    // for (int k = l; k < l + block_size; k++)
    //   q[k - l] = 1 / A[k * n + k];

    // #pragma omp for
    // for (int i = l + block_size; i < n; i++) {
    //   for (int k = l; k < l + block_size; k++) {
    //     A[i * n + k] *= q[k - l];

    //     double Aik = A[i * n + k];
    //     for (int j = k + 1; j < l + block_size; j++) {
    //       A[i * n + j] -= Aik * A[k * n + j];
    //     }
    //   }
    // }
    #pragma omp barrier   
  }

  int l = n - (block_size - n % block_size);
  for (int k = l; k < n; k++) {
    double A_kk = A[k * n + k];
    for (int i = k + 1; i < n; i++) A[i * n + k] /= A_kk;

    for (int i = k + 1; i < n; i++) {
      for (int j = k + 1; j < n; j++) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
      }
    }
  }

  free(q);

  #pragma endscop
}
