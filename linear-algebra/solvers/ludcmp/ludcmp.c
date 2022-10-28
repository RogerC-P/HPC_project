/**
 * ludcmp.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <mpi.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "ludcmp.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;

  for (i = 0; i <= n; i++)
    {
      x[i] = i + 1;
      y[i] = (i+1)/n/2.0 + 1;
      b[i] = (i+1)/n/2.0 + 42;
      for (j = 0; j <= n; j++) {
	      A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / n;
        if (i == j) A[i][j] = n;
      }
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N+1,n+1))

{
  int i;

  for (i = 0; i <= n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, x[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  DATA_TYPE w;

#pragma scop
  int n_chunks = 4;

  assert(n % n_chunks == 0);
  int chunk_size = n / n_chunks;

  int chunks_per_rank = n_chunks / world_size;

  int psizes[2] = {0, 0};
  MPI_Dims_create(world_size, 2, psizes);

  MPI_Datatype *dist_types = (MPI_Datatype *) malloc(world_size * sizeof(MPI_Datatype));
  for (int i = 0; i < world_size; i++) {
    int sizes[2] = {n, n};
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
    int dargs[2] = {chunk_size, chunk_size};

    MPI_Type_create_darray(world_size, i, 2,
                           sizes, distribs, dargs, psizes,
                           MPI_ORDER_C, MPI_DOUBLE, &dist_types[i]);
    MPI_Type_commit(&dist_types[i]);
  }

  MPI_Request *send_requests;
  if (rank == 0) {
    send_requests = (MPI_Request *) malloc(world_size * sizeof(MPI_Request));
    for (int i = 0; i < 1; i++) {
      MPI_Isend(&A[0][0], 1, dist_types[i],
                i, 0, MPI_COMM_WORLD, &send_requests[i]);
    }
  }

  int dist_size;
  MPI_Type_size(dist_types[rank], &dist_size);
  dist_size /= sizeof(double);

  double *D = (double *) malloc(dist_size * sizeof(double));

  MPI_Request recv_request;
  MPI_Irecv(D, dist_size, MPI_DOUBLE,
            0, 0, MPI_COMM_WORLD, &recv_request);

  MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

  if (rank == 0) {
    MPI_Waitall(1, send_requests, MPI_STATUSES_IGNORE);
  }

  int m = n / psizes[0];

  if (rank == 0) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        printf("%.2lf ", D[i * m + j]);
      }
      printf("\n");
    }
  }

  for (int k = 0; k < _PB_N; k++) {
    for (int i = k + 1; i <= _PB_N; i++) A[i][k] /= A[k][k];

    for (int i = k + 1; i < _PB_N; i++) {
      for (int j = k + 1; j < _PB_N; j++) {
        A[i][j] -= A[i][k] * A[k][j];
      }
    }
  }

  // Forward, Backward Substitution
  b[0] = 1.0;
  y[0] = b[0];
  for (int i = 1; i < _PB_N; i++) {
    w = b[i];
    for (int j = 0; j < i; j++)
      w = w - A[i][j] * y[j];
    y[i] = w;
  }

  x[_PB_N] = y[_PB_N] / A[_PB_N][_PB_N];

  for (int i = 0; i < _PB_N - 1; i++) {
    w = y[_PB_N - 1 - (i)];
    for (int j = _PB_N - i; j < _PB_N; j++)
      w = w - A[_PB_N - 1 - i][j] * x[j];
    x[_PB_N - 1 - i] = w / A[_PB_N - 1 - (i)][_PB_N - 1-(i)];
  }
#pragma endscop
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
  for (i = 0; i < _PB_N - 1; i++) {
    for (j = i+1; j < _PB_N; j++) {
      w = A[j][i];
      for (k = 0; k < i; k++)
        w = w - A[j][k] * A[k][i];
      A[j][i] = w / A[i][i];
    }
    for (j = i+1; j < _PB_N; j++) {
      w = A[i+1][j];
      for (k = 0; k <= i; k++)
        w = w - A[i+1][k] * A[k][j];
      A[i+1][j] = w;
    }
  }

  // Forward, Backward Substitution
  b[0] = 1.0;
  y[0] = b[0];
  for (i = 1; i <= _PB_N; i++) {
    w = b[i];
    for (j = 0; j < i; j++)
      w = w - A[i][j] * y[j];
    y[i] = w;
  }

  x[_PB_N] = y[_PB_N] / A[_PB_N][_PB_N];

  for (i = 0; i <= _PB_N - 1; i++) {
    w = y[_PB_N - 1 - (i)];
    for (j = _PB_N - i; j <= _PB_N; j++)
      w = w - A[_PB_N - 1 - i][j] * x[j];
    x[_PB_N - 1 - i] = w / A[_PB_N - 1 - (i)][_PB_N - 1-(i)];
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Hello from rank %d\n", rank);

  if (rank == 0) {
    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

    /* Initialize array(s). */
    init_array (n,
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(b),
          POLYBENCH_ARRAY(x),
          POLYBENCH_ARRAY(y));
  }

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_ludcmp (n,
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

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  MPI_Finalize();

  return 0;
}
