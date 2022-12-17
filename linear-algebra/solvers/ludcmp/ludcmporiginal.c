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

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"

int world_size;
int rank;

static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++) {
    x[i] = 0;
    y[i] = 0;
    b[i] = (i+1)/fn/2.0 + 100;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / n;
      if (i == j) A[i][i] = n * n;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

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
void kernel_ludcmp_original(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;

  DATA_TYPE w;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <i; j++) {
       w = A[i][j];
       for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   for (j = i; j < _PB_N; j++) {
       w = A[i][j];
       for (k = 0; k < i; k++) {
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  for (i = 0; i < _PB_N; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   for (i = _PB_N-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < _PB_N; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }
#pragma endscop

}

int main(int argc, char** argv)
{
  printf("ludcmp\n####\n");

  int n0 = (N < (1 << 10)) ? N : (1 << 10);
  for (int n = n0; n <= N; n <<= 1) {
    printf("size %d:\n", n);

    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

    for (int i = 0; i < N_RUNS; i++) {
    /* Initialize array(s). */
      init_array (n,
            POLYBENCH_ARRAY(A),
            POLYBENCH_ARRAY(b),
            POLYBENCH_ARRAY(x),
            POLYBENCH_ARRAY(y));

      /* Start timer. */
      polybench_start_instruments;

      /* Run kernel. */
      kernel_ludcmp_original (n,
         POLYBENCH_ARRAY(A),
         POLYBENCH_ARRAY(b),
         POLYBENCH_ARRAY(x),
         POLYBENCH_ARRAY(y));

      /* Stop and print timer. */
      polybench_stop_instruments;
      polybench_print_instruments;

      /* Prevent dead-code elimination. All live-out data must be printed
         by the function call in argument. */
      polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

    }

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(b);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);

    printf("###\n");
  }

  return 0;
}
