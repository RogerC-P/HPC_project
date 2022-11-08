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

#include <gemm.h>

/* Include polybench common header. */
#include <polybench.h>

void print(int n, double *A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) printf("%.2lf ", A[i * n + j]);
    printf("\n");
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void lu(int n, double *A)
{
#pragma scop
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int block_size = 200;

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

  double *B = (double *) malloc(dist_size);

  int position = 0;
  MPI_Pack(A, 1, dist_types[rank], B, dist_size, &position, MPI_COMM_WORLD) ;

  int m = n / psizes[0];

  int row_idx = rank / psizes[0];
  int col_idx = rank % psizes[0];

  MPI_Comm row_comm;
  MPI_Comm_split(MPI_COMM_WORLD, row_idx, rank, &row_comm);

  MPI_Comm col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, col_idx, rank, &col_comm);

  int row_rank;
  MPI_Comm_rank(row_comm, &row_rank);

  int col_rank;
  MPI_Comm_rank(col_comm, &col_rank);

  int chunk_size = block_size * psizes[0];

  double *LU_k = (double *) malloc(block_size * block_size * sizeof(double));

  double prev_row_size;
  double *U_p = (double *) malloc(block_size * m * sizeof(double));
  double *L_p = (double *) malloc(m * block_size * sizeof(double));

  double *U_k = (double *) malloc(block_size * m * sizeof(double));
  double *L_k = (double *) malloc(m * block_size * sizeof(double));

  int ro(int bk) {
    int chunk_idx = bk / psizes[0];
    int block_idx = bk % psizes[0];

    int result = chunk_idx * block_size;
    if (row_rank < block_idx) result += block_size;
    return result;
  }

  int co(int bk) {
    int chunk_idx = bk / psizes[0];
    int block_idx = bk % psizes[0];

    int result = chunk_idx * block_size;
    if (col_rank < block_idx) result += block_size;
    return result;
  }

  void swap(double **a, double **b) {
    void *tmp = *a;
    *a = *b;
    *b = tmp;
  }

  int n_blocks = n / block_size;

  for (int bk = 0; bk < n_blocks; bk++) {
    int block_idx = bk % psizes[0];

    #pragma omp parallel sections
    {
      #pragma omp section
      {
        if (bk > 0) {
          if (row_rank == block_idx && col_rank == block_idx) {
            gemm(block_size, block_size, block_size,
                 -1, L_p, block_size,
                 U_p, m - ro(bk),
                 1, B + co(bk) * m + ro(bk), m);
          }

          if (col_rank == block_idx) {
            gemm(block_size, m - ro(bk + 1), block_size,
                 -1, L_p, block_size,
                 U_p + (ro(bk + 1) - ro(bk)), m - ro(bk),
                 1, B + co(bk) * m + ro(bk + 1), m);
          }

          if (row_rank == block_idx) {
            gemm(m - co(bk + 1), block_size, block_size,
                 -1, L_p + (co(bk + 1) - co(bk)) * block_size, block_size,
                 U_p, m - ro(bk),
                 1, B + co(bk + 1) * m + ro(bk), m);
          }
        }

        if (row_rank == block_idx && col_rank == block_idx) {
          for (int k = ro(bk); k < ro(bk) + block_size; k++) {
            double B_kk = B[k * m + k];
            for (int i = k + 1; i < co(bk) + block_size; i++) B[i * m + k] /= B_kk;

            for (int i = k + 1; i < co(bk) + block_size; i++) {
              for (int j = k + 1; j < ro(bk) + block_size; j++) {
                B[i * m + j] -= B[i * m + k] * B[k * m + j];
              }
            }
          }

          for (int i = co(bk); i < co(bk) + block_size; i++) {
            for (int j = ro(bk); j < ro(bk) + block_size; j++)
              LU_k[(i - co(bk)) * block_size + (j - ro(bk))] = B[i * m + j];
          }
        }

        MPI_Request row_request, col_request;
        if (row_rank == block_idx) {
          MPI_Ibcast(LU_k, block_size * block_size, MPI_DOUBLE, block_idx, col_comm, &col_request);
        }

        if (col_rank == block_idx) {
          MPI_Ibcast(LU_k, block_size * block_size, MPI_DOUBLE, block_idx, row_comm, &row_request);
        }

        if (row_rank == block_idx) MPI_Wait(&col_request, MPI_STATUS_IGNORE);
        if (col_rank == block_idx) MPI_Wait(&row_request, MPI_STATUS_IGNORE);

        if (col_rank == block_idx) {
          for (int k = co(bk); k < co(bk) + block_size; k++) {
            for (int i = k + 1; i < co(bk) + block_size; i++) {
              for (int j = ro(bk + 1); j < m; j++) {
                B[i * m + j] -= LU_k[(i - co(bk)) * block_size + (k - co(bk))] * B[k * m + j];
              }
            }
          }

          for (int i = co(bk); i < co(bk) + block_size; i++) {
            for (int j = ro(bk + 1); j < m; j++) {
              U_k[(i - co(bk)) * (m - ro(bk + 1)) + (j - ro(bk + 1))] = B[i * m + j];
            }
          }
        }

        if (row_rank == block_idx) {
          for (int k = ro(bk); k < ro(bk) + block_size; k++) {
            for (int i = co(bk + 1); i < m; i++)
              B[i * m + k] /= LU_k[(k - ro(bk)) * block_size + (k - ro(bk))];

            for (int i = co(bk + 1); i < m; i++) {
              for (int j = k + 1; j < ro(bk) + block_size; j++) {
                B[i * m + j] -= B[i * m + k] * LU_k[(k - ro(bk)) * block_size + (j - ro(bk))];
              }
            }
          }

          for (int i = co(bk + 1); i < m; i++) {
            for (int j = ro(bk); j < ro(bk) + block_size; j++) {
              L_k[(i - co(bk + 1)) * block_size + (j - ro(bk))] = B[i * m + j];
            }
          }
        }

        MPI_Ibcast(L_k, (m - co(bk + 1)) * block_size, MPI_DOUBLE,
                   block_idx, row_comm, &row_request);

        MPI_Ibcast(U_k, block_size * (m - ro(bk + 1)), MPI_DOUBLE,
                   block_idx, col_comm, &col_request);

        MPI_Wait(&row_request, MPI_STATUS_IGNORE);
        MPI_Wait(&col_request, MPI_STATUS_IGNORE);
      }

      #pragma omp section
      {
        if (bk > 0) {
          gemm(m - co(bk + 1), m - ro(bk + 1), block_size,
               -1, L_p + (co(bk + 1) - co(bk)) * block_size, block_size,
               U_p + (ro(bk + 1) - ro(bk)), m - ro(bk),
               1, B + co(bk + 1) * m + ro(bk + 1), m);
        }
      }
    }

    swap(&L_k, &L_p);
    swap(&U_k, &U_p);
  }

  MPI_Request *recv_requests;
  if (rank == 0) {
    recv_requests = (MPI_Request *) malloc(world_size * sizeof(MPI_Request));
    for (int i = 0; i < world_size; i++) {
      MPI_Irecv(A, 1, dist_types[i],
                i, 0, MPI_COMM_WORLD, &recv_requests[i]);
    }
  }

  MPI_Request send_request;
  MPI_Isend(B, dist_size / sizeof(double), MPI_DOUBLE,
            0, 0, MPI_COMM_WORLD, &send_request);

  if (rank == 0) {
    MPI_Waitall(world_size, recv_requests, MPI_STATUSES_IGNORE);
    free(recv_requests);
  }

  MPI_Wait(&send_request, MPI_STATUS_IGNORE);

  free(B);

  free(LU_k);
  free(U_k);
  free(L_k);

  free(U_p);
  free(L_p);
  #pragma endscop
}
