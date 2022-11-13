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


int getI(int rank, int n) {
  return rank % n;
}

int getJ(int rank, int n) {
  return rank / n;
}

int getStart(int rank, double segmentLenght) {
  return rank * segmentLenght;
}

int getEnd(int start, double segmentLenght, int isEven, int rank, int mpiSize) {
  int end = start + segmentLenght - 1;
  if (rank == mpiSize - 1 && isEven != 0) {
    end += 1;
  }
  return end;
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
  int i, j, k;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
int resultSize;
  double segmentLenght = (ni * nj) / (double)mpiSize;

  // range
  
  int isEven;
  if (segmentLenght == (int) segmentLenght) {
    isEven = 0;
  } else {
    isEven = 1;
  }

  int start = getStart(rank, segmentLenght);
  int end = getEnd(start, segmentLenght, isEven, rank, mpiSize);

  if (isEven != 0) {
    resultSize = end - start + 2;
  } else {
    resultSize = end - start + 1;
  }

  

  double ijResult[resultSize];


  int resultIndex = -1;
  for(int point = start; point <= end; ++point) {

    ++resultIndex;
    i = getI(point, ni);
    j = getJ(point, nj);
    k = (i + j) % nk;

    /* C := alpha*A*B + beta*C */

    ijResult[resultIndex] = C[i][j] * beta;
    double a, b, c;
    for (int iteration = 0; iteration < ni; iteration++) {

        a = A[i][k];
        b = B[k][j];
        c = ijResult[resultIndex];
        c += a * b * alpha;
        ijResult[resultIndex] = c;
        k = (k + 1) % nk;
        if (rank == 0) {
          C[i][j] = c;
        }
    }
  }
  if (rank != 0) {
    MPI_Send(&ijResult, resultSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    for (int worker = 1; worker < mpiSize; worker++) {
      MPI_Recv(&ijResult, resultSize, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      int start = getStart(worker, segmentLenght);
      int end = getEnd(start, segmentLenght, isEven, worker, mpiSize);

      int resultIndex = -1;
      for(int point = start; point <= end; ++point) {
        ++resultIndex;
        int workerI = getI(point, ni);
        int workerJ = getJ(point, nj);
        double result = ijResult[resultIndex];
        
        C[workerI][workerJ] = result;  
      }

    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{

int rank, size;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > NI * NI) {
      printf("Hola Hola! size must be <= N * N. was %d > %d  Exiting", size, NI * NI);
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

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B),
         rank, size);

  /* Stop and print timer. */
  polybench_stop_instruments;
  

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */

  if (rank == 0) {
    polybench_print_instruments;
    polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));
    //print_array(ni, nj,  POLYBENCH_ARRAY(C));
    
    
  }
  

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  MPI_Finalize();
  return 0;
}
