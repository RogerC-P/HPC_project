/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

#include "ludcmp.h"


int world_size;
int rank;

/* Array initialization. */
static
void init_array (int n,
		 double *A,
		 double *b, double *x, double *y)
{
  int i, j;
  double fn = (double) n;

  for (i = 0; i < n; i++) {
    x[i] = 0;
    y[i] = 0;
    b[i] = (i+1)/fn/2.0 + 100;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      A[i * n + j] = ((double) (i+1)*(j+1)) / n;
      if (i == j) A[i * n + i] = n * n;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n, double *x)

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   double *A,
		   double *b, double *x, double *y)
{
  int i, j, k;

  double w;

#pragma scop
	int ipiv[n];
 	LAPACKE_dgetrf(CblasRowMajor, n, n, (double *) A, n, ipiv);
  LAPACKE_dgetrs(CblasRowMajor, 'N', n, 1, (const double *) A, n, ipiv, b, 1);  
  memcpy(x, b, n * sizeof(double));
#pragma endscop
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp_original(int n,
		   double *A,
		   double *b, double *x, double *y)
{
  int i, j, k;

  double w;

#pragma scop
  for (i = 0; i < n; i++) {
    for (j = 0; j <i; j++) {
       w = A[i * n + j];
       for (k = 0; k < j; k++) {
          w -= A[i * n + k] * A[k * n + j];
       }
        A[i * n + j] = w / A[j * n + j];
    }
   for (j = i; j < n; j++) {
       w = A[i * n + j];
       for (k = 0; k < i; k++) {
          w -= A[i * n + k] * A[k * n + j];
       }
       A[i * n + j] = w;
    }
  }

  for (i = 0; i < n; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i * n + j] * y[j];
     y[i] = w;
  }

   for (i = n-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < n; j++)
        w -= A[i * n + j] * x[j];
     x[i] = w / A[i * n + i];
  }
#pragma endscop

}

int main(int argc, char** argv)
{
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Variable declaration/allocation. */
  double *A;
  double *b;
  double *x;
  double *y;

  int n0 = (N < (1 << 10)) ? N : (1 << 10);

  for (int n = n0; n <= N; n <<= 1) {
    if (rank == 0) printf("size %d:\n", n);

    A = (double *) malloc(n * n * sizeof(double));
    b = (double *) malloc(n * sizeof(double));
    x = (double *) malloc(n * sizeof(double));
    y = (double *) malloc(n * sizeof(double));

    for (int i = 0; i < NUM_RUNS; i++) {

      /* Initialize array(s). */
      init_array (n, A, b, x, y);

      MPI_Barrier(MPI_COMM_WORLD);

      if (rank == 0) {
        /* Start timer. */
        polybench_start_instruments;
      }

      /* Run kernel. */
      kernel_ludcmp (n, A, b, x, y);


      if (rank == 0) {
        /* Stop and print timer. */
        polybench_stop_instruments;
        polybench_print_instruments;

        /* Prevent dead-code elimination. All live-out data must be printed
           by the function call in argument. */
        polybench_prevent_dce(print_array(n, x));
      }
    }

    if (rank == 0) printf("###\n");

    /* Be clean. */
    free(A);
    free(b);
    free(x);
    free(y);
  }

  MPI_Finalize();

  return 0;
}
