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


double **allocate_array(int row_dim, int col_dim) 
{
  double **result;
  int i;

  result=(double **)malloc(row_dim*sizeof(double *));  
  result[0]=(double *)malloc(row_dim*col_dim*sizeof(double));
  
  for(i=1; i<row_dim; i++)
	result[i]=result[i-1]+col_dim;
  return result;
}

void deallocate_array(double **array, int row_dim) 
{
  int i;
  
  for(i=1; i<row_dim; i++)
	array[i]=NULL;
  free(array[0]);
  free(array);
}


static
void write_array(int ni, int nj, double **C, char *destination)
{
  int i, j;
  int write = 0;
  if (write == 1) {
    FILE *fp = fopen(destination,"w+");

    
    for (i = 0; i < ni; i++)
      for (j = 0; j < nj; j++) {
        //if (i == 0) printf ("\n");      
        //printf ("%f ", C[i][j]);
        
        if (j == 0) {
          fprintf(fp, "\n");
          fprintf(fp, "---%d-------------------------------\n", i);
        }
        fprintf(fp, "%f ", C[i][j]);
      }
    fclose(fp);
  }
}

/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		double **C,
		double **A,
		double **B)
{
  int i, j;

  
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
void print_array(int ni, int nj, double **C)
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) printf ("\n");      
	    printf ("%f ", C[i][j]);
  
  }
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
		 double **C,
		 double **A,
		 double **B, int rank, int mpiSize)
{

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ

int startI, endI;
getStartEnd(ni, rank, mpiSize, &startI, &endI);


endI = endI - startI;
int nrOfElements = endI;
startI = 0;
// scatter only neccesary parts of A C to workers
int *displsA = (int *)malloc(mpiSize * sizeof(int));
int *displsC = (int *)malloc(mpiSize * sizeof(int));
int *scountsA = (int *)malloc(mpiSize * sizeof(int));
int *scountsC = (int *)malloc(mpiSize * sizeof(int));

int offsetA = 0;
int offsetC = 0;
if (rank == 0) {
  for (int i = 0; i < mpiSize; i++) {
      int startI, endI;
      getStartEnd(ni, i, mpiSize, &startI, &endI);
      endI = endI - startI;
      int nrOfElements = endI;
      displsA[i] = offsetA;
      int scountA = (nrOfElements + 1) * nk;
      offsetA += scountA;
      scountsA[i] = scountA;

      displsC[i] = offsetC;
      int scountC = (nrOfElements + 1) * nj;
      offsetC += scountC;
      scountsC[i] = scountC;
  }
}

MPI_Scatterv(*A, scountsA, displsA, MPI_DOUBLE,
				   *A, (nrOfElements + 1) * nk, MPI_DOUBLE,
				   0, MPI_COMM_WORLD);
MPI_Scatterv(*C, scountsC, displsC, MPI_DOUBLE,
				   *C, (nrOfElements + 1) * nj, MPI_DOUBLE,
				   0, MPI_COMM_WORLD);

// copy entire B to everybody
MPI_Bcast(*B, nk * nj, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  
  
// kick off computations
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
            __m256d vc = _mm256_loadu_pd(&C[u][v]);
            __m256d vMultiResult = _mm256_mul_pd(vbeta, vc);                          
            _mm256_storeu_pd(&C[u][v], vMultiResult);
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
  
  

// gather results

MPI_Gatherv(*C, (nrOfElements + 1) * nj, MPI_DOUBLE,
               *C, scountsC, displsC, MPI_DOUBLE, 0, MPI_COMM_WORLD);

}

int calcISize(int startI, int endI) {
  return endI - startI + 1;
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

  int startI, endI;
  getStartEnd(ni, rank, size, &startI, &endI);
  int iSize = calcISize(startI, endI);
  int iSizeToAllocate;
  if (rank == 0) {
    iSizeToAllocate = ni;
  } else {
    iSizeToAllocate = iSize;
  }

  /* Variable declaration/allocation. */
  DATA_TYPE alpha = 1.5;
  DATA_TYPE beta = 1.2;
  double **A, **B, **C;

  
  C=allocate_array(iSizeToAllocate, nj);
  A=allocate_array(iSizeToAllocate, nk);
  B=allocate_array(nk, nj);
  
  /* Initialize array(s). */
  if (rank == 0) {
    init_array (ni, nj, nk,
	      C,
	      A,
	      B);
    
  }

  /* Start timer. */
  if (rank == 0) {
    polybench_start_instruments;
  }

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       C,
	       A,
	       B,
         rank, size);

  /* Stop and print timer. */
  if (rank == 0) {
    // stop time only when all workers are done
    polybench_stop_instruments;
  }

  

  if (rank == 0) {
    polybench_print_instruments;
    polybench_prevent_dce(print_array(ni, nj, C));
  }

  /* Be clean. */
  deallocate_array(C, iSizeToAllocate);
  deallocate_array(A, iSizeToAllocate);
  deallocate_array(B, nk);

  MPI_Finalize();
  return 0;
}
