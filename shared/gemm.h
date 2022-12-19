#include <immintrin.h>

#ifndef NB
#define NB 128
#endif

inline __attribute__((always_inline)) void block_mul(double beta, double *C, int ldc) {
  for (int i = 0; i < NB; i++) {
    for (int j = 0; j < NB; j++) {
      C[i * ldc + j] *= beta;
    }
  }
}

#define RI 4
#define RJ 4

inline __attribute__((always_inline)) void micro_mm(
    double alpha, double *A, int lda,
    double *B, int ldb,
    double *C, int ldc)
{
  __m256d valpha = _mm256_set1_pd(alpha);

  __m256d sums[RI][RJ] = {_mm256_set1_pd(0)};

  for (int k = 0; k < NB; k++) {
    for (int j = 0; j < RJ; j++) {
      __m256d b = _mm256_loadu_pd(&B[k * ldb + j * 4]);
      for (int i = 0; i < RI; i++) {
        __m256d a = _mm256_set1_pd(A[i * lda + k]);
        sums[i][j] = _mm256_fmadd_pd(a, b, sums[i][j]);
      }
    }
  }

  for (int i = 0; i < RI; i++) {
    for (int j = 0; j < RJ; j++) {
      __m256d c = _mm256_loadu_pd(&C[i * ldc + 4 * j]);
      c = _mm256_fmadd_pd(valpha, sums[i][j], c);
      _mm256_storeu_pd(&C[i * ldc + 4 * j], c);
    }
  }
}

inline __attribute__((always_inline)) void mini_mm(
    double alpha, double *A, int lda,
    double *B, int ldb,
    double *C, int ldc)
{
  for (int i = 0; i < NB - RI + 1; i += RI) {
    for (int j = 0; j < NB - 4 * RJ + 1; j += 4 * RJ) {
      micro_mm(alpha, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
    }
  }
}

void mm(
    int ni, int nj, int nk,
    double alpha, double *A, int lda,
    double *B, int ldb,
    double beta, double *C, int ldc)
{
  #pragma omp for
  for (int i = 0; i < ni - NB + 1; i += NB) {
    for (int j = 0; j < nj - NB + 1; j += NB) {
      block_mul(beta, &C[i * ldc + j], ldc);

      for (int k = 0; k < nk - NB + 1; k += NB) {
        mini_mm(
          alpha, &A[i * lda + k], lda,
          &B[k * ldb + j], ldb,
          &C[i * ldc + j], ldc
        );
      }
    }
  }

  #pragma omp for
  for (int i = 0; i < ni; i++) {
    for (int j = NB * (nj / NB); j < nj; j++) {
      C[i * lda + j] *= beta;
      for (int k = 0; k < nk; k++) {
        C[i * ldc + j] += alpha * A[i * lda + k] * B[k * ldb + j];
      }
    }
  }

  #pragma omp for
  for (int i = NB * (ni / NB); i < ni; i++) {
    for (int j = 0; j < NB * (nj / NB); j++) {
      C[i * lda + j] *= beta;
      for (int k = 0; k < nk; k++) {
        C[i * ldc + j] += alpha * A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

#define LDA_MULTIPLE 57

void pad_matrix(int ni, int nj, double *A, int lda, double **new_A, int *new_lda) {
  *new_lda = LDA_MULTIPLE * (nj / LDA_MULTIPLE + 1);
  *new_A = (double *) malloc(ni * (*new_lda) * sizeof(double));

  if (!new_A) {
    printf("Ran out of memory :(\n");
    exit(-1);
  }

  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) (*new_A)[i * (*new_lda) + j] = A[i * lda + j];
  }
}

void padded_mm(
    int ni, int nj, int nk,
    double alpha, double *A, int lda,
    double *B, int ldb,
    double beta, double *C, int ldc)
{
  double *padded_A;
  int padded_lda;

  double *padded_B;
  int padded_ldb;

  double *padded_C;
  int padded_ldc;

  #pragma omp single \
    copyprivate(padded_A, padded_lda, padded_B, padded_ldb, padded_C, padded_ldc)
  {
    pad_matrix(ni, nk, A, lda, &padded_A, &padded_lda);
    pad_matrix(nk, nj, B, ldb, &padded_B, &padded_ldb);
    pad_matrix(ni, nj, C, ldc, &padded_C, &padded_ldc);
  }

  #pragma omp barrier

  mm(ni, nj, nk,
      alpha, padded_A, padded_lda,
      padded_B, padded_ldb,
      beta, padded_C, padded_ldc);

  #pragma omp barrier

  #pragma omp master
  {
    for (int i = 0; i < ni; i++) {
      for (int j = 0; j < nj; j++) {
        C[i * ldc + j] = padded_C[i * padded_ldc + j];
      }
    }

    free(padded_A);
    free(padded_B);
    free(padded_C);
  }
}

void gemm(
    int ni, int nj, int nk,
    double alpha, double *A, int lda,
    double *B, int ldb,
    double beta, double *C, int ldc)
{
  if (0 && ni < 512 && nk < 512) {
    mm(ni, nj, nk,
        alpha, A, lda,
        B, ldb,
        beta, C, ldc);
  } else {
    padded_mm(ni, nj, nk,
        alpha, A, lda,
        B, ldb,
        beta, C, ldc);
  }
}
