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
#include <pthread.h>
#include <math.h>

#include <gemm.h>

/* Include polybench common header. */
#include <polybench.h>

void print(int n, int m, double *A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) printf("%.4lf ", A[i * m + j]);
    printf("\n");
  }
}

void swap(double **a, double **b) {
  void *tmp = *a;
  *a = *b;
  *b = tmp;
}

int min(int a, int b) {
	return (a < b) ? a : b;
}


#ifndef LU_BLOCK_SIZE
#define LU_BLOCK_SIZE (4)
#endif

int compute_axis_size(int n, int psize, int idx) {
	int n_blocks = (n + LU_BLOCK_SIZE - 1) / LU_BLOCK_SIZE;

	int result = 0;
	for (int i = 0; i < n_blocks; i++) {
		int block_size = min(LU_BLOCK_SIZE, n - i * LU_BLOCK_SIZE);
		if (i % psize == idx) result += block_size;
	}

	return result;
}

#define SMALL_BLOCK_SIZE ((GEMM_BLOCK_SIZE < LU_BLOCK_SIZE) ? GEMM_BLOCK_SIZE : LU_BLOCK_SIZE)

void lu(int n, double *A) {
#pragma scop
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int psizes[2] = {0, 0};
  MPI_Dims_create(world_size, 2, psizes);

  int chunk_size = LU_BLOCK_SIZE * psizes[0];

  int n_old = n;

  MPI_Datatype *dist_types = (MPI_Datatype *) malloc(world_size * sizeof(MPI_Datatype));

  for (int i = 0; i < world_size; i++) {
    int sizes[2] = {n, n};
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
    int dargs[2] = {LU_BLOCK_SIZE, LU_BLOCK_SIZE};

    MPI_Type_create_darray(world_size, i, 2,
                           sizes, distribs, dargs, psizes,
                           MPI_ORDER_C, MPI_DOUBLE, &dist_types[i]);
    MPI_Type_commit(&dist_types[i]);
  }

  int dist_size;
  MPI_Type_size(dist_types[rank], &dist_size);

  double *B = (double *) malloc(dist_size);

  if (!B) {
    printf("Error: Out of memory :(\n");
    exit(-1);
  }

  int position = 0;
  MPI_Pack(A, 1, dist_types[rank], B, dist_size, &position, MPI_COMM_WORLD) ;

	int n_blocks = (n + LU_BLOCK_SIZE - 1) / LU_BLOCK_SIZE;

  int row_idx = rank / psizes[0];
  int col_idx = rank % psizes[0];

	int mr = compute_axis_size(n, psizes[0], col_idx);
	int mc = compute_axis_size(n, psizes[0], row_idx);

  int ldb = mr;

  MPI_Comm row_comm;
  MPI_Comm_split(MPI_COMM_WORLD, row_idx, rank, &row_comm);

  MPI_Comm col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, col_idx, rank, &col_comm);

  int row_rank;
  MPI_Comm_rank(row_comm, &row_rank);

  int col_rank;
  MPI_Comm_rank(col_comm, &col_rank);

  int ro(int bk) {
    int chunk_idx = bk / psizes[0];
    int block_idx = bk % psizes[0];

    int result = chunk_idx * LU_BLOCK_SIZE;
    if (row_rank < block_idx) result += LU_BLOCK_SIZE;
    return min(mr, result);
  }

  int co(int bk) {
    int chunk_idx = bk / psizes[0];
    int block_idx = bk % psizes[0];

    int result = chunk_idx * LU_BLOCK_SIZE;
    if (col_rank < block_idx) result += LU_BLOCK_SIZE;
    return min(mc, result);
  }

  int ldl = LU_BLOCK_SIZE;

  double *U_p = (double *) malloc(LU_BLOCK_SIZE * mc * sizeof(double));
  double *L_p = (double *) malloc(mr * ldl * sizeof(double));

  double *LU_k = (double *) malloc(LU_BLOCK_SIZE * LU_BLOCK_SIZE * sizeof(double));
  double *U_k = (double *) malloc(LU_BLOCK_SIZE * mc * sizeof(double));
  double *L_k = (double *) malloc(mr * ldl * sizeof(double));

  double *q = (double *) malloc(LU_BLOCK_SIZE * sizeof(double));

  MPI_Request *row_request = (MPI_Request *) malloc(sizeof(MPI_Request));
  MPI_Request *col_request = (MPI_Request *) malloc(sizeof(MPI_Request));

  #pragma omp parallel
  for (int bk = 0; bk < n_blocks; bk++) {
    int block_idx = bk % psizes[0];

    int ro_k = ro(bk);
    int ro_n = ro(bk + 1);

    int co_k = co(bk);
    int co_n = co(bk + 1);

    if (bk > 0 && row_rank == block_idx && col_rank == block_idx) {
			int d = min(LU_BLOCK_SIZE, mr - ro_k);
      gemm(d, d, LU_BLOCK_SIZE,
           -1, L_p, ldl,
           U_p, mr - ro_k,
           1, B + co_k * ldb + ro_k, ldb);
    }

    #pragma omp barrier

    #pragma omp master
    if (row_rank == block_idx || col_rank == block_idx){
      if (row_rank == block_idx && col_rank == block_idx) {
        for (int k = ro_k; k < min(ro_k + LU_BLOCK_SIZE, mr); k++) {
          double B_kk = B[k * ldb + k];
          for (int i = k + 1; i < min(co_k + LU_BLOCK_SIZE, mc); i++) B[i * ldb + k] /= B_kk;

          for (int i = k + 1; i < min(co_k + LU_BLOCK_SIZE, mc); i++) {
            for (int j = k + 1; j < min(ro_k + LU_BLOCK_SIZE, mr); j++) {
              B[i * ldb + j] -= B[i * ldb + k] * B[k * ldb + j];
            }
          }
        }

        for (int i = co_k; i < min(co_k + LU_BLOCK_SIZE, mc); i++) {
          for (int j = ro_k; j < min(ro_k + LU_BLOCK_SIZE, mr); j++)
            LU_k[(i - co_k) * LU_BLOCK_SIZE + (j - ro_k)] = B[i * ldb + j];
        }
      }
    }

		// if (bk == n_blocks - 1) break;

    #pragma omp master
    if (row_rank == block_idx || col_rank == block_idx){
      if (row_rank == block_idx) {
        MPI_Ibcast(LU_k, LU_BLOCK_SIZE * LU_BLOCK_SIZE, MPI_DOUBLE, block_idx, col_comm, col_request);
      }

      if (col_rank == block_idx) {
        MPI_Ibcast(LU_k, LU_BLOCK_SIZE * LU_BLOCK_SIZE, MPI_DOUBLE, block_idx, row_comm, row_request);
      }
    }

    if (bk > 0) {
      if (col_rank == block_idx) {
        gemm(LU_BLOCK_SIZE, mr - ro_n, LU_BLOCK_SIZE,
              -1, L_p, ldl,
              U_p + (ro_n - ro_k), mr - ro_k,
              1, B + co_k * ldb + ro_n, ldb);
      }

      if (row_rank == block_idx) {
        gemm(mc - co_n, LU_BLOCK_SIZE, LU_BLOCK_SIZE,
              -1, L_p + (co_n - co_k) * ldl, ldl,
              U_p, mr - ro_k,
              1, B + co_n * ldb + ro_k, ldb);
      }
    }

    #pragma omp master
    {
      if (row_rank == block_idx) MPI_Wait(col_request, MPI_STATUS_IGNORE);
      if (col_rank == block_idx) MPI_Wait(row_request, MPI_STATUS_IGNORE);
    }

    #pragma omp barrier

    if (col_rank == block_idx) {
      #pragma omp for
      for (int u = ro_n; u < mr; u += SMALL_BLOCK_SIZE) {
        for (int k = co_k; k < min(co_k + LU_BLOCK_SIZE, mc); k++) {
          for (int i = k + 1; i < min(co_k + LU_BLOCK_SIZE, mc); i++) {
            for (int j = u; j < min(u + SMALL_BLOCK_SIZE, mr); j++) {
              B[i * ldb + j] -= LU_k[(i - co_k) * LU_BLOCK_SIZE + (k - co_k)] * B[k * ldb + j];
            }
          }
        }

        for (int i = co_k; i < min(co_k + LU_BLOCK_SIZE, mc); i++) {
          for (int j = u; j < min(u + SMALL_BLOCK_SIZE, mr); j++) {
            U_k[(i - co_k) * (mr - ro_n) + (j - ro_n)] = B[i * ldb + j];
          }
        }
      }
    }

    if (row_rank == block_idx) {
      #pragma omp master
      for (int k = ro_k; k < min(ro_k + LU_BLOCK_SIZE, mr); k++)
        q[k - ro_k] = 1 / LU_k[(k - ro_k) * LU_BLOCK_SIZE + (k - ro_k)];

      #pragma omp barrier

      #pragma omp for schedule(static, SMALL_BLOCK_SIZE)
      for (int i = co_n; i < mc; i++) {
        for (int k = ro_k; k < min(ro_k + LU_BLOCK_SIZE, mr); k++) {
          B[i * ldb + k] *= q[k - ro_k];

          double Bik = B[i * ldb + k];
          for (int j = k + 1; j < min(ro_k + LU_BLOCK_SIZE, mr); j++) {
            B[i * ldb + j] -= Bik * LU_k[(k - ro_k) * LU_BLOCK_SIZE + (j - ro_k)];
          }
        }

        for (int j = ro_k; j < min(ro_k + LU_BLOCK_SIZE, mr); j++) {
          L_k[(i - co_n) * ldl + (j - ro_k)] = B[i * ldb + j];
        }
      }
    }

    #pragma omp barrier

    #pragma omp master
    {
      MPI_Ibcast(L_k, (mc - co_n) * ldl, MPI_DOUBLE,
                 block_idx, row_comm, row_request);

      MPI_Ibcast(U_k, LU_BLOCK_SIZE * (mr - ro_n), MPI_DOUBLE,
                 block_idx, col_comm, col_request);
    }


    if (bk > 0) {
      gemm(mc - co_n, mr - ro_n, LU_BLOCK_SIZE,
            -1, L_p + (co_n - co_k) * ldl, ldl,
            U_p + (ro_n - ro_k), mr - ro_k,
            1, B + co_n * ldb + ro_n, ldb);
    }

    #pragma omp barrier

    #pragma omp master
    {
      MPI_Wait(row_request, MPI_STATUS_IGNORE);
      MPI_Wait(col_request, MPI_STATUS_IGNORE);

      swap(&U_p, &U_k);
      swap(&L_p, &L_k);
    }

    #pragma omp barrier
  }

  free(q);

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

  free(dist_types);

  free(B);

  free(LU_k);
  free(U_k);
  free(L_k);

  free(U_p);
  free(L_p);
  #pragma endscop
}
