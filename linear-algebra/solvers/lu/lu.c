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

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lu.h"

int world_size;
int rank;

/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  // This is really slow
  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  // int r,s,t;
  // POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  // for (r = 0; r < n; ++r)
  //   for (s = 0; s < n; ++s)
  //     (POLYBENCH_ARRAY(B))[r][s] = 0;
  // for (t = 0; t < n; ++t)
  //   for (r = 0; r < n; ++r)
  //     for (s = 0; s < n; ++s)
	// (POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
  //   for (r = 0; r < n; ++r)
  //     for (s = 0; s < n; ++s)
	// A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  // POLYBENCH_FREE_ARRAY(B);

}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_lu(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  DATA_TYPE w;

#pragma scop
  int block_size = 4;

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

  int dist_size;
  MPI_Type_size(dist_types[rank], &dist_size);

  double *B = (double *) malloc(dist_size * sizeof(double));

  int position = 0;
  MPI_Pack(&A[0][0], 1, dist_types[rank], B, dist_size, &position, MPI_COMM_WORLD) ;

  // MPI_Request *send_requests;
  // if (rank == 0) {
  //   send_requests = (MPI_Request *) malloc(world_size * sizeof(MPI_Request));
  //   for (int i = 0; i < world_size; i++) {
  //     MPI_Isend(&A[0][0], 1, dist_types[i],
  //               i, 0, MPI_COMM_WORLD, &send_requests[i]);
  //   }
  // }

  // MPI_Request recv_request;
  // MPI_Irecv(B, dist_size, MPI_DOUBLE,
  //           0, 0, MPI_COMM_WORLD, &recv_request);

  // MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

  // if (rank == 0) {
  //   MPI_Waitall(world_size, send_requests, MPI_STATUSES_IGNORE);
  // }

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
    int old_col_offset = col_offset;

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
        B[i * m + j] -= L_k[i] * U_k[j];
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
  MPI_Isend(B, dist_size / sizeof(double), MPI_DOUBLE,
            0, 0, MPI_COMM_WORLD, &send_request);

  if (rank == 0) {
    MPI_Waitall(world_size, recv_requests, MPI_STATUSES_IGNORE);
  }

  MPI_Wait(&send_request, MPI_STATUS_IGNORE);
  #pragma endscop
}


/* Original kernel */
static
void kernel_lu_original(int n,
	       DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <i; j++) {
       for (k = 0; k < j; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
        A[i][j] /= A[j][j];
    }
   for (j = i; j < _PB_N; j++) {
       for (k = 0; k < i; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
    }
  }
#pragma endscop
}

#ifndef REUSE_LU_KERNEL
int main(int argc, char** argv)
{
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  if (rank == 0) {
    /* Start timer. */
    polybench_start_instruments;
  }

  /* Run kernel. */
  kernel_lu (n, POLYBENCH_ARRAY(A));

  if (rank == 0) {
    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
  }

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  MPI_Finalize();

  return 0;
}
#endif
