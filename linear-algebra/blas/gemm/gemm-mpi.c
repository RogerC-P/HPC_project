/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

#include <mpi.h>
#include <immintrin.h>

/* Include benchmark-specific header. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  //*alpha = 1.0;
  //*beta = 1.0;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  //FILE *fp = fopen("/tmp/mpi.txt","w+");

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");      
	    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
      
      //if (j == 0) {
      //  fprintf(fp, "\n");
      //}
      //fprintf(fp, "%f ", C[i][j]);
    }
  //  fclose(fp);
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


int getI(int rank) {
  return rank / NJ;
}

int getJ(int rank) {
  return rank % NJ;
}

int getStart(int rank, int segmentLenght) {
  return rank * segmentLenght;
}

int getEnd(int start, int segmentLenght, int rank) {
  return start + segmentLenght - 1;
}

void getStartEnd(int ni, int rank, int mpiSize, int* start, int* end) {
  int nrOfPoints  = ni;
  double segmentLenght = nrOfPoints / (double)mpiSize;
  
  int isEven;
  if (segmentLenght == (int) segmentLenght) {
    isEven = 0;
  } else {
    isEven = 1;
  }

  if (isEven != 0 && rank == mpiSize - 1) {
    int prevEnd = getEnd(getStart(rank -1 , segmentLenght), segmentLenght, rank);
    *start = prevEnd + 1;
    *end = nrOfPoints - 1;
  } else {
    *start = getStart(rank, segmentLenght);
    *end = getEnd(*start, segmentLenght, rank);
  }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), int rank, int mpiSize)
{

  

  
  int startJ = 0;
  int endJ = nj;

  int startI, endI;
  getStartEnd(ni, rank, mpiSize, &startI, &endI);
  
  

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  


int BI = 20;
int BJ = 40;
int BK = 12;

  __m256d valpha = _mm256_set1_pd(alpha);
  __m256d vbeta = _mm256_set1_pd(beta);
  
  int i;
  for (i = startI; i <= endI - BI + 1; i += BI) {
    int j;
    for (j = 0; j < _PB_NJ - BJ + 1; j += BJ) {      
      for (int u = i; u < i + BI; u++) {
        int v;
        for (v = j; v < j + BJ; v+=4) {
          //C[i][j] *= beta;
            __m256d vc = _mm256_load_pd(&C[u][v]);
            __m256d vMultiResult = _mm256_mul_pd(vbeta, vc);                          
            _mm256_store_pd(&C[u][v], vMultiResult);
        }        
      }


      int k;
      for (k = 0; k < _PB_NK - BK + 1; k += BK) {
        // start doing stuff within a block
        for (int ui = i; ui < i + BI; ui += 4) {
          for (int vj = j; vj < j + BJ; vj += 8) {
            __m256d ab00 = _mm256_set1_pd(0.0);
            __m256d ab01 = _mm256_set1_pd(0.0);
            __m256d ab02 = _mm256_set1_pd(0.0);
            __m256d ab03 = _mm256_set1_pd(0.0);

            __m256d ab10 = _mm256_set1_pd(0.0);
            __m256d ab11 = _mm256_set1_pd(0.0);
            __m256d ab12 = _mm256_set1_pd(0.0);
            __m256d ab13 = _mm256_set1_pd(0.0);

            for (int wk = k; wk < k + BK; wk++) {
              __m256d a0 = _mm256_set1_pd(A[ui + 0][wk]);
              __m256d a1 = _mm256_set1_pd(A[ui + 1][wk]);
              __m256d a2 = _mm256_set1_pd(A[ui + 2][wk]);
              __m256d a3 = _mm256_set1_pd(A[ui + 3][wk]);

              __m256d b0 = _mm256_loadu_pd(&B[wk][vj + 0]);
              __m256d b1 = _mm256_loadu_pd(&B[wk][vj + 4]);

              ab00 = _mm256_fmadd_pd(a0, b0, ab00);
              ab01 = _mm256_fmadd_pd(a1, b0, ab01);
              ab02 = _mm256_fmadd_pd(a2, b0, ab02);
              ab03 = _mm256_fmadd_pd(a3, b0, ab03);

              ab10 = _mm256_fmadd_pd(a0, b1, ab10);
              ab11 = _mm256_fmadd_pd(a1, b1, ab11);
              ab12 = _mm256_fmadd_pd(a2, b1, ab12);
              ab13 = _mm256_fmadd_pd(a3, b1, ab13);
            }
            __m256d c00 = _mm256_loadu_pd(&C[ui + 0][vj + 0]);
            __m256d c01 = _mm256_loadu_pd(&C[ui + 1][vj + 0]);
            __m256d c02 = _mm256_loadu_pd(&C[ui + 2][vj + 0]);
            __m256d c03 = _mm256_loadu_pd(&C[ui + 3][vj + 0]);

            __m256d c10 = _mm256_loadu_pd(&C[ui + 0][vj + 4]);
            __m256d c11 = _mm256_loadu_pd(&C[ui + 1][vj + 4]);
            __m256d c12 = _mm256_loadu_pd(&C[ui + 2][vj + 4]);
            __m256d c13 = _mm256_loadu_pd(&C[ui + 3][vj + 4]);

            c00 = _mm256_fmadd_pd(valpha, ab00, c00);
            c01 = _mm256_fmadd_pd(valpha, ab01, c01);
            c02 = _mm256_fmadd_pd(valpha, ab02, c02);
            c03 = _mm256_fmadd_pd(valpha, ab03, c03);

            c10 = _mm256_fmadd_pd(valpha, ab10, c10);
            c11 = _mm256_fmadd_pd(valpha, ab11, c11);
            c12 = _mm256_fmadd_pd(valpha, ab12, c12);
            c13 = _mm256_fmadd_pd(valpha, ab13, c13);

            _mm256_storeu_pd(&C[ui + 0][vj + 0], c00);
            _mm256_storeu_pd(&C[ui + 1][vj + 0], c01);
            _mm256_storeu_pd(&C[ui + 2][vj + 0], c02);
            _mm256_storeu_pd(&C[ui + 3][vj + 0], c03);

            _mm256_storeu_pd(&C[ui + 0][vj + 4], c10);
            _mm256_storeu_pd(&C[ui + 1][vj + 4], c11);
            _mm256_storeu_pd(&C[ui + 2][vj + 4], c12);
            _mm256_storeu_pd(&C[ui + 3][vj + 4], c13);
          }
        }
      }

      if (k < _PB_NK) {
        for (int u = i; u < i + BI; u++) {
          for (int v = j; v < j + BJ; v++) {
            double ab = 0;
            for (int w = k; w < _PB_NK; w++) {
              ab += A[u][w] * B[w][v];
            }
            C[u][v] += alpha * ab;
          }
        }
      }
    }

    
    for (; j < _PB_NJ; j++) {
      for (int u = i; u < i + BI; u++) {
        C[u][j] *= beta;
      }

      for (int k = 0; k < _PB_NK; k++) {
        for (int u = i; u < i + BI; u++) {
          C[u][j] += alpha * A[u][k] * B[k][j];
        }
      }
    }
    
  }

  for (; i <= endI; i++) {
    for (int j = 0; j < _PB_NJ; j++) {
      C[i][j] *= beta;
      for (int k = 0; k < _PB_NK; k++) {
        C[i][j] += alpha * A[i][k] * B[k][j];
      }
    }
  }
  
#pragma endscop
}

void gatherResults(int ni, int nj,
      DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
      int rank,
		 int size, 
     int argc, char** argv, double ijResult[], int resultSize) {
  
  
  
  int startI, endI;
  getStartEnd(ni, rank, size, &startI, &endI);
  int resultIndex = -1;
  if (rank != 0) {
    for (int i = startI; i <= endI; ++i) {
       for (int j = 0; j < nj; ++j) {
          ijResult[++resultIndex] = C[i][j];
       }
    }
    MPI_Send(ijResult, resultSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  
  if (rank == 0) {
  
    for (int worker = 1; worker < size; worker++) {
      
      MPI_Recv(ijResult, resultSize, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      
      getStartEnd(ni, worker, size, &startI, &endI);
      

      int resultIndex = -1;
      for (int i = startI; i <= endI; ++i) {
        for (int j = 0; j < nj; ++j) {
            C[i][j] = ijResult[++resultIndex];
        }
      }
    }
    
  }
  
}



int main(int argc, char** argv)
{

int rank, size;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if (size > NI) {
      printf("Hola Hola! size must be <= NI. was %d > %d  Exiting", size, NI);
      MPI_Finalize();
  
      return 1;
    }

  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  int startI, endI;
  getStartEnd(ni, size - 1, size, &startI, &endI);
  int resultSize = (endI - startI + 1) * nj;
  double *ijResult = malloc (sizeof (double) * resultSize);
  /* Start timer. */
  if (rank == 0) {
    polybench_start_instruments;
  }

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B),
         rank, size);

  /* Stop and print timer. */
  // wait for all workers to finish
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    // stop time only when all workers are done
    polybench_stop_instruments;
  }

  //polybench_print_instruments;
  
  // gather results from all workers and print them out
  // gathering is not timed
  gatherResults(ni, nj, POLYBENCH_ARRAY(C), rank, size, argc, argv, ijResult, resultSize);

  if (rank == 0) {
    polybench_print_instruments;
    polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(C)));
  }

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  MPI_Finalize();
  return 0;
}
