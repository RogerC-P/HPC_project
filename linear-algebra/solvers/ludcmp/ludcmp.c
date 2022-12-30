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

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"
#include "lu.h"

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
  lu(n, A);

  if (rank == 0) {
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
  }
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

void benchmark(int argc, char** argv, int n) {
  if (rank == 0) printf("size: %d\n", n);
  
  double *A = (double *) malloc(n * n * sizeof(double));
  double *b = (double *) malloc(n * sizeof(double));
  double *x = (double *) malloc(n * sizeof(double));
  double *y = (double *) malloc(n * sizeof(double));
  
  for (int k = 0; k < NUM_RUNS; k++) {
    /* Initialize array(s). */
    init_array (n, A, b, x, y);

    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Start timer. */
    if (rank == 0) polybench_start_instruments;
    
    /* Run kernel. */
    kernel_ludcmp (n, A, b, x, y);
    
    // /* Stop and print timer. */
    if (rank == 0) {
      polybench_stop_instruments;
      polybench_print_instruments;
      /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
      polybench_prevent_dce(print_array(n, x));
    }
  }
  
  free(A);
  free(b);
  free(x);
  free(y);
}

int main(int argc, char** argv)
{
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Weak scaling
  int p = NUM_PROCESSORS;
  int i = 0;
  while (p > 1) {
    p /= 2;
    i += 1;
  }

  int ns[6] = {5000, 6300, 7938, 10000, 12600, 15874 };
  benchmark(argc, argv, ns[i]);

#ifdef STRONG_SCALING
  int n0 = (NI < (1 << 10)) ? NI : (1 << 10);
  for (int n = n0; n <= NI; n <<= 1) {
    benchmark(argc, argv, n);
    if (rank == 0) printf("###\n");
  }
#endif

  MPI_Finalize();

  return 0;
}
