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

int world_size;
int rank;

/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;

  for (i = 0; i < n; i++) {
    x[i] = i + 1;
    y[i] = (i+1)/n/2.0 + 1;
    b[i] = (i+1)/n/2.0 + 42;
    for (j = 0; j < n; j++) {
      A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / n;
      if (i == j) A[i][j] = n * n;
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
  int block_size = 64;

  int psizes[2] = {0, 0};
  MPI_Dims_create(world_size, 2, psizes);

  MPI_Datatype *dist_types = (MPI_Datatype *) malloc(world_size * sizeof(MPI_Datatype));
  for (int i = 0; i < world_size; i++) {
    int sizes[2] = {n, n};
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
    int dargs[2] = {block_size, block_size};

    MPI_Type_create_darray(world_size, i, 2,
                           sizes, distribs, dargs, psizes,
                           MPI_ORDER_C, MPI_DOUBLE, &dist_types[i]);
    MPI_Type_commit(&dist_types[i]);
  }

  MPI_Request *send_requests;
  if (rank == 0) {
    send_requests = (MPI_Request *) malloc(world_size * sizeof(MPI_Request));
    for (int i = 0; i < world_size; i++) {
      MPI_Isend(&A[0][0], 1, dist_types[i],
                i, 0, MPI_COMM_WORLD, &send_requests[i]);
    }
  }

  int dist_size;
  MPI_Type_size(dist_types[rank], &dist_size);
  dist_size /= sizeof(double);

  double *B = (double *) malloc(dist_size * sizeof(double));

  MPI_Request recv_request;
  MPI_Irecv(B, dist_size, MPI_DOUBLE,
            0, 0, MPI_COMM_WORLD, &recv_request);

  MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

  if (rank == 0) {
    MPI_Waitall(world_size, send_requests, MPI_STATUSES_IGNORE);
  }

  int m = n / psizes[0];

  int row_idx = rank / psizes[0];
  int col_idx = rank % psizes[1];

  MPI_Comm row_comm;
  MPI_Comm_split(MPI_COMM_WORLD, col_idx, rank, &row_comm);

  MPI_Comm col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, row_idx, rank, &col_comm);

  int row_rank;
  MPI_Comm_rank(row_comm, &row_rank);

  int col_rank;
  MPI_Comm_rank(col_comm, &col_rank);

  double *L_k = (double *) malloc(m * sizeof(double));
  double *U_k = (double *) malloc(m * sizeof(double));

  int chunk_size = block_size * psizes[0];

  for (int k = 0; k < N; k++) {
    int chunk = k / chunk_size;
    int chunk_offset = k % chunk_size;

    int block_idx = chunk_offset / block_size;

    int row_offset = chunk * block_size;
    if (block_idx == row_rank) row_offset += chunk_offset % block_size;
    else if (block_idx > row_rank) row_offset += block_size;

    int col_offset = chunk * block_size;
    if (block_idx == col_rank) col_offset += chunk_offset % block_size;
    else if (block_idx > col_rank) col_offset += block_size;

    int row_size = m - col_offset;

    if (row_rank == block_idx) {
      memcpy(U_k + col_offset, &B[row_offset * m + col_offset], row_size * sizeof(double));
      row_offset += 1;
    }

    MPI_Bcast(U_k + col_offset, row_size, MPI_DOUBLE, block_idx, row_comm);

    int col_size = m - row_offset;

    if (col_rank == block_idx) {
      for (int j = row_offset; j < row_offset + col_size; j++) {
        B[j * m + col_offset] /= U_k[col_offset];
        L_k[j] = B[j * m + col_offset];
      }

      col_offset += 1;
    }

    MPI_Bcast(L_k + row_offset, col_size, MPI_DOUBLE, block_idx, col_comm);

    for (int i = row_offset; i < m; i++) {
      for (int j = col_offset; j < m; j++) {
        B[i * m + j] -= U_k[i] * L_k[j];
      }
    }
  }

  free(U_k);
  free(L_k);

  MPI_Request *recv_requests;
  if (rank == 0) {
    recv_requests = (MPI_Request *) malloc(world_size * sizeof(MPI_Request));
    for (int i = 0; i < world_size; i++) {
      MPI_Irecv(&A[0][0], 1, dist_types[i],
                i, 0, MPI_COMM_WORLD, &recv_requests[i]);
    }
  }

  MPI_Request send_request;
  MPI_Isend(B, dist_size, MPI_DOUBLE,
            0, 0, MPI_COMM_WORLD, &send_request);

  if (rank == 0) {
    MPI_Waitall(world_size, recv_requests, MPI_STATUSES_IGNORE);
  }

  MPI_Wait(&send_request, MPI_STATUS_IGNORE);

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
  for (i = 1; i < _PB_N; i++) {
    w = b[i];
    for (j = 0; j < i; j++)
      w = w - A[i][j] * y[j];
    y[i] = w;
  }

  x[_PB_N] = y[_PB_N] / A[_PB_N][_PB_N];

  for (i = 0; i < _PB_N - 1; i++) {
    w = y[_PB_N - 1 - (i)];
    for (j = _PB_N - i; j < _PB_N; j++)
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

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n_rank = (rank == 0) ? n : 0;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n_rank, n_rank);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n_rank);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n_rank);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n_rank);

  if (rank == 0) {
    /* Initialize array(s). */
    init_array (n,
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(b),
          POLYBENCH_ARRAY(x),
          POLYBENCH_ARRAY(y));
  }

  if (rank == 0) {
    /* Start timer. */
    polybench_start_instruments;
  }

  /* Run kernel. */
  kernel_ludcmp (n,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(b),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y));

  if (rank == 0) {
    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    polybench_prevent_dce(print_array(n - 1, POLYBENCH_ARRAY(x)));
  }

  MPI_Finalize();

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}
