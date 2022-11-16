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

#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define BI 20
#define BJ 40
#define BK 12

#define BLOCK_MULT(k_end) do { \
  __m256d ab00 = _mm256_set1_pd(0.0); \
  __m256d ab01 = _mm256_set1_pd(0.0); \
  __m256d ab02 = _mm256_set1_pd(0.0); \
  __m256d ab03 = _mm256_set1_pd(0.0); \
 \
  __m256d ab10 = _mm256_set1_pd(0.0); \
  __m256d ab11 = _mm256_set1_pd(0.0); \
  __m256d ab12 = _mm256_set1_pd(0.0); \
  __m256d ab13 = _mm256_set1_pd(0.0); \
 \
  for (int w = k; w < k_end; w++) { \
    __m256d a0 = _mm256_set1_pd(A[(u + 0) * lda + w]); \
    __m256d a1 = _mm256_set1_pd(A[(u + 1) * lda + w]); \
    __m256d a2 = _mm256_set1_pd(A[(u + 2) * lda + w]); \
    __m256d a3 = _mm256_set1_pd(A[(u + 3) * lda + w]); \
 \
    __m256d b0 = _mm256_loadu_pd(&B[w * ldb + v + 0]); \
    __m256d b1 = _mm256_loadu_pd(&B[w * ldb + v + 4]); \
 \
    ab00 = _mm256_fmadd_pd(a0, b0, ab00); \
    ab01 = _mm256_fmadd_pd(a1, b0, ab01); \
    ab02 = _mm256_fmadd_pd(a2, b0, ab02); \
    ab03 = _mm256_fmadd_pd(a3, b0, ab03); \
 \
    ab10 = _mm256_fmadd_pd(a0, b1, ab10); \
    ab11 = _mm256_fmadd_pd(a1, b1, ab11); \
    ab12 = _mm256_fmadd_pd(a2, b1, ab12); \
    ab13 = _mm256_fmadd_pd(a3, b1, ab13); \
  } \
 \
  __m256d c00 = _mm256_loadu_pd(&C[(u + 0) * ldc + v + 0]); \
  __m256d c01 = _mm256_loadu_pd(&C[(u + 1) * ldc + v + 0]); \
  __m256d c02 = _mm256_loadu_pd(&C[(u + 2) * ldc + v + 0]); \
  __m256d c03 = _mm256_loadu_pd(&C[(u + 3) * ldc + v + 0]); \
 \
  __m256d c10 = _mm256_loadu_pd(&C[(u + 0) * ldc + v + 4]); \
  __m256d c11 = _mm256_loadu_pd(&C[(u + 1) * ldc + v + 4]); \
  __m256d c12 = _mm256_loadu_pd(&C[(u + 2) * ldc + v + 4]); \
  __m256d c13 = _mm256_loadu_pd(&C[(u + 3) * ldc + v + 4]); \
 \
  c00 = _mm256_fmadd_pd(valpha, ab00, c00); \
  c01 = _mm256_fmadd_pd(valpha, ab01, c01); \
  c02 = _mm256_fmadd_pd(valpha, ab02, c02); \
  c03 = _mm256_fmadd_pd(valpha, ab03, c03); \
 \
  c10 = _mm256_fmadd_pd(valpha, ab10, c10); \
  c11 = _mm256_fmadd_pd(valpha, ab11, c11); \
  c12 = _mm256_fmadd_pd(valpha, ab12, c12); \
  c13 = _mm256_fmadd_pd(valpha, ab13, c13); \
 \
  _mm256_storeu_pd(&C[(u + 0) * ldc + v + 0], c00); \
  _mm256_storeu_pd(&C[(u + 1) * ldc + v + 0], c01); \
  _mm256_storeu_pd(&C[(u + 2) * ldc + v + 0], c02); \
  _mm256_storeu_pd(&C[(u + 3) * ldc + v + 0], c03); \
 \
  _mm256_storeu_pd(&C[(u + 0) * ldc + v + 4], c10); \
  _mm256_storeu_pd(&C[(u + 1) * ldc + v + 4], c11); \
  _mm256_storeu_pd(&C[(u + 2) * ldc + v + 4], c12); \
  _mm256_storeu_pd(&C[(u + 3) * ldc + v + 4], c13); \
} while (0);

#ifdef PARALLEL_GEMM
void pgemm
#else
void gemm
#endif
    (int m, int n, int k,
		 double alpha, double *A, int lda,
     double *B, int ldb,
		 double beta, double *C, int ldc)
{
  __m256d valpha = _mm256_set1_pd(alpha);

#ifdef PARALLEL_GEMM
  #pragma omp for schedule(static, 4)
#endif
  for (int i = 0; i < m - BI + 1; i += BI) {
    if (i < 0) continue;

    int j;
    for (j = 0; j < n - BJ + 1; j += BJ) {
      for (int u = i; u < i + BI; u++) {
        for (int v = j; v < j + BJ; v++) {
          C[u * ldc + v] *= beta;
        }
      }

      int l;
      for (l = 0; l < k - BK + 1; l += BK) {
        for (int u = i; u < i + BI; u += 4) {
          for (int v = j; v < j + BJ; v += 8) {
            BLOCK_MULT(l + BK);
          }
        }
      }

      for (int u = i; u < i + BI; u += 4) {
        for (int v = j; v < j + BJ; v += 8) {
          BLOCK_MULT(l);
        }
      }
    }

    for (; j < n; j++) {
      for (int u = i; u < i + BI; u++) {
        C[u * ldc + j] *= beta;
      }

      for (int l = 0; l < k; l++) {
        for (int u = i; u < i + BI; u++) {
          C[u * ldc + j] += alpha * A[u * lda + l] * B[l * ldb + j];
        }
      }
    }
  }

#ifdef PARALLEL_GEMM
  #pragma omp master
  {
    #pragma omp task
    {
#else
  {
    {
#endif
      int i = BI * (m / BI);
      for (; i < m; i++) {
        for (int j = 0; j < n; j++) {
          C[i * ldc + j] *= beta;
          for (int l = 0; l < k; l++) {
            C[i * ldc + j] += alpha * A[i * lda + l] * B[l * ldb + j];
          }
        }
      }
    }
  }
}
