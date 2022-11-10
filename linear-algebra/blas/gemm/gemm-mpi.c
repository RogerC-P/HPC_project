/**
 * gemm.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>
#include <mpi.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
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

  *alpha = 32412;
  *beta = 2123;
  //*alpha = 1;
  //*beta = 1;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((DATA_TYPE) i*j) / ni;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), int whichFile)
{
  int i, j;

  FILE *fileP;
  if (whichFile == 0) {
    fileP = fopen("/tmp/resultsParallel.txt","w");
  } else if (whichFile == 1) {
    fileP = fopen("/tmp/inputA.txt","w");
  } else if (whichFile == 2) {
    fileP = fopen("/tmp/inputB.txt","w");
  }
  
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      if (j == 0) {
        fprintf (fileP, "\n");
      }
	    fprintf (fileP, DATA_PRINTF_MODIFIER, C[i][j]);
	    //if ((i * ni + j) % 20 == 0) fprintf (fileP, "\n");
    }
  fprintf (fileP, "\n");
  fclose(fileP);
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
     int mpiSize,
     int rank)
{
  int i, j, k;

double segmentLenght = _PB_NK / (double)mpiSize;

// range
int start = rank * segmentLenght;
int end = start + segmentLenght - 1;
int isEven;
if (segmentLenght == (int) segmentLenght) {
  isEven = 0;
} else {
  isEven = 1;
}

if (rank == mpiSize - 1 && isEven != 0) {
  end += 1;
}

int ijCoordinates[2];
ijCoordinates[0] = -1;
ijCoordinates[1] = -1;
double ijResults[mpiSize];
for (int i = 0; i < mpiSize; ++i) {
  ijResults[i] = 0.0;
}

#pragma scop
  /* C := alpha*A*B + beta*C */

  // master
  if (rank == 0) {
    for (i = 0; i < _PB_NI; i++) {
      for (j = 0; j < _PB_NJ; j++) {
	      C[i][j] *= beta;
	      
        ijCoordinates[0] = i;
        ijCoordinates[1] = j;
        // send calculation coordinates to workers
        for (int workerRank = 1; workerRank < mpiSize; ++workerRank) {
          MPI_Send(&ijCoordinates, 2, MPI_INT, workerRank, 0, MPI_COMM_WORLD);
        }

        double a, b, c;
        c = 0.0;
        // calculate own segment
        for (k = start; k <= end; ++k) {          
          a = A[i][k];
          b = B[k][j];
          c += alpha * a * b;
          C[i][j] = c;
        }

        double finalC = C[i][j];
        double workerResults;
        // receive segment results from workers
        for (int workerRank = 1; workerRank < mpiSize; ++workerRank) {
          MPI_Recv(&ijResults, mpiSize, MPI_DOUBLE, workerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          workerResults = ijResults[workerRank];
          finalC += workerResults;
          C[i][j] = finalC;
        }
      }
    }

    // send finish signal to workers
    ijCoordinates[0] = _PB_NI;
    ijCoordinates[1] = _PB_NJ;
    for (int workerRank = 1; workerRank < mpiSize; ++workerRank) {
      MPI_Send(&ijCoordinates, 2, MPI_INT, workerRank, 0, MPI_COMM_WORLD);
    }
  }

  // workers
  if (rank != 0) {
    while (ijCoordinates[0] < _PB_NI && ijCoordinates[1] < _PB_NJ) {
      // receive coordinates from master and do calcs
      MPI_Recv(&ijCoordinates, 2,
                 MPI_INT, 0, 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      ijResults[rank] = 0.0;
      double a, b, c;
      c = 0.0;
      for (k = start; k <= end; ++k) {
        int i = ijCoordinates[0];
        int j = ijCoordinates[1];
        a = A[i][k];
        b = B[k][j];
        c += alpha * a * b;
        ijResults[rank] = c;
      }
      // send results to master
      MPI_Send(&ijResults, mpiSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }
  
#pragma endscop

}


int main(int argc, char** argv) {
    int rank, size;

    //int i = 0;
    //while (i == 0) {
    //  sleep(1);
    //}

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    printf("my rank is %i \n", rank);

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

  printf("initialize array\n");

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  printf("start benchmark\n");
  /* Start timer. */
  polybench_start_instruments;

  

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B), size, rank);


  
  /* Stop and print timer. */
  polybench_stop_instruments;
  

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
     
  //polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));
  
  if (rank == 0) {
    printf("benchmark finished \n");
    polybench_print_instruments;
    print_array(ni, nj,  POLYBENCH_ARRAY(C), 0);
    print_array(ni, nj,  POLYBENCH_ARRAY(A), 1);
    print_array(ni, nj,  POLYBENCH_ARRAY(B), 2);
  }
  
  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);




    MPI_Finalize();
    return 0;



}
