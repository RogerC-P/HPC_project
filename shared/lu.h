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

#define BLOCK_SIZE 80

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void lu(int n, double *A) {
#pragma scop

  #pragma omp parallel
  {
    double q[BLOCK_SIZE];

    for (int l = 0; l < n - BLOCK_SIZE + 1; l += BLOCK_SIZE) {
      if (l > 0) {
        pgemm(n - l, n - l, BLOCK_SIZE,
              -1, A + l * n + (l - BLOCK_SIZE), n,
              A + (l - BLOCK_SIZE) * n + l, n,
              1, A + l * n + l, n);
      }

      #pragma omp master
      for (int k = l; k < l + BLOCK_SIZE; k++) {
        double A_kk = A[k * n + k];
        for (int i = k + 1; i < l + BLOCK_SIZE; i++) A[i * n + k] /= A_kk;

        for (int i = k + 1; i < l + BLOCK_SIZE; i++) {
          for (int j = k + 1; j < l + BLOCK_SIZE; j++) {
            A[i * n + j] -= A[i * n + k] * A[k * n + j];
          }
        }
      }

      #pragma omp barrier

      #pragma omp for
      for (int u = l + BLOCK_SIZE; u < n - BLOCK_SIZE + 1; u += BLOCK_SIZE) {
        for (int k = l; k < l + BLOCK_SIZE; k++) {
          for (int i = k + 1; i < l + BLOCK_SIZE; i++) {
            for (int j = u; j < u + BLOCK_SIZE; j++) {
              A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
          }
        }
      }

      for (int k = l; k < l + BLOCK_SIZE; k++)
        q[k - l] = 1 / A[k * n + k];

      #pragma omp for
      for (int i = l + BLOCK_SIZE; i < n; i++) {
        for (int k = l; k < l + BLOCK_SIZE; k++) {
          A[i * n + k] *= q[k - l];

          double Aik = A[i * n + k];
          for (int j = k + 1; j < l + BLOCK_SIZE; j++) {
            A[i * n + j] -= Aik * A[k * n + j];
          }
        }
      }
      #pragma omp barrier   
    }
  }

  if (n % BLOCK_SIZE != 0) {
    int l = n - (BLOCK_SIZE - n % BLOCK_SIZE);
    for (int k = l; k < n; k++) {
      double A_kk = A[k * n + k];
      for (int i = k + 1; i < n; i++) A[i * n + k] /= A_kk;

      for (int i = k + 1; i < n; i++) {
        for (int j = k + 1; j < n; j++) {
          A[i * n + j] -= A[i * n + k] * A[k * n + j];
        }
      }
    }
  }

  #pragma endscop
}
