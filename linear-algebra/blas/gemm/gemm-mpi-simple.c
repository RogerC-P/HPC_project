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
#pragma scop
  
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

  
  for (int i = startI; i <= endI; i++) {

    
    for (int j = 0; j < nj; j++) {
	    C[i][j] *= beta;
    }
    
    for (int k = 0; k < _PB_NK; k++) {
      for (int j = 0; j < nj; j++) {
        C[i][j] += alpha * A[i][k] * B[k][j];
      }
    }
  }

  MPI_Gatherv(*C, (nrOfElements + 1) * nj, MPI_DOUBLE,
               *C, scountsC, displsC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#pragma endscop
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

  /* Variable declaration/allocation. */
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

  //polybench_print_instruments;
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
