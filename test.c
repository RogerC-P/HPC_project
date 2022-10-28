/**
 * ludcmp.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#include <mpi.h>

#define N 8

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel(double A[N][N]) {
  double w;

  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Hello from rank %d\n", rank);

  int n_chunks = 4;

  assert(N % n_chunks == 0);
  int chunk_size = N / n_chunks;

  int chunks_per_rank = n_chunks / world_size;

  int psizes[2] = {0, 0};
  MPI_Dims_create(world_size, 2, psizes);

  MPI_Datatype *dist_types = (MPI_Datatype *) malloc(world_size * sizeof(MPI_Datatype));
  for (int i = 0; i < world_size; i++) {
    int sizes[2] = {N, N};
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
    for (int i = 0; i < world_size; i++) {
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
    MPI_Waitall(world_size, send_requests, MPI_STATUSES_IGNORE);
  }

  int m = N / psizes[0];

  if (rank == 1) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        printf("%.2lf ", D[i * m + j]);
      }
      printf("\n");
    }
  }

  MPI_Finalize();
}

int main(int argc, char** argv) {
  double (*A)[N][N] = (double(*)[N][N]) malloc ((N * N) * sizeof(double));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) (*A)[i][j] = i + j;
  }

  kernel(*A);

  return 0;
}
