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

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");      
	    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
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

void getStartEnd(int segmentLenght, int isEven, int rank, int mpiSize, int nrOfPoints, int* start, int* end) {
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
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), int rank, int mpiSize, 
     double ijResult[], int start, int end)
{
  


//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  int resultIndex = -1;
  for(int point = start; point <= end; ++point) {
    ++resultIndex;
    int i = getI(point);
    int j = getJ(point);
    /* C := alpha*A*B + beta*C */
    ijResult[resultIndex] = C[i][j] * beta;
    for (int k = 0; k < nk; k++) {
      ijResult[resultIndex] += A[i][k] * B[k][j] * alpha;  
    }
  }
#pragma endscop
}

void gatherResults(int ni, int nj,
      DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
      int rank,
		 int size, 
     double myIjResult[], int resultSize,
     int argc, char** argv,
     int segmentLenght, int isEven) {
  
  if (rank != 0) {
    MPI_Send(myIjResult, resultSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  int start, end;
  int nrOfPoints = ni * nj;
  if (rank == 0) {
    
    for (int worker = 0; worker < size; worker++) {
      if (worker != 0) {
        MPI_Recv(myIjResult, resultSize, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      
      getStartEnd(segmentLenght, isEven, worker, size, nrOfPoints, &start, &end);
      

      int resultIndex = -1;
      for(int point = start; point <= end; ++point) {
        ++resultIndex;
        int workerI = getI(point);
        int workerJ = getJ(point);
        double result = myIjResult[resultIndex];
        
        C[workerI][workerJ] = result;  
      }
      
    }
    polybench_print_instruments;
    polybench_prevent_dce(print_array(ni, nj, C));
  }
}


int main(int argc, char** argv)
{

int rank, size;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NI * NJ) {
      printf("Hola Hola! size must be <= NI * NJ. was %d > %d  Exiting", size, NI * NJ);
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


  int resultSize, start, end;
  
  // some bunch of initialiations. will assume not to time it.
  int nrOfPoints  = ni * nj;
  double segmentLenght = nrOfPoints / (double)size;
  
  int isEven;
  if (segmentLenght == (int) segmentLenght) {
    isEven = 0;
  } else {
    isEven = 1;
  }

  getStartEnd(segmentLenght, isEven, rank, size, nrOfPoints, &start, &end);

  int lastRankStart, lastRankEnd;
  getStartEnd(segmentLenght, isEven, size - 1, size, nrOfPoints, &lastRankStart, &lastRankEnd);
  resultSize = lastRankEnd - lastRankStart + 1;
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
         rank, size, ijResult, start, end);

  /* Stop and print timer. */
  // wait for all workers to finish
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    // stop time only when all workers are done
    polybench_stop_instruments;
  }
  
  // gather results from all workers and print them out
  // gathering is not timed
  gatherResults(ni, nj, POLYBENCH_ARRAY(C), rank, size, ijResult, resultSize, argc, argv, segmentLenght, isEven);


  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  MPI_Finalize();
  return 0;
}
