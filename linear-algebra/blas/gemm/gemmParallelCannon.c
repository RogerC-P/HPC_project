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

// assumes mpiSize == ni == nj
#pragma scop


i = rank % ni;
j = rank / ni;
k = (i + j) % ni;

/* C := alpha*A*B + beta*C */

double ijResult[3];
ijResult[0] = i;
ijResult[1] = j;
ijResult[2] = C[i][j] * beta;

double a, b, c;
for (int iteration = 0; iteration < ni; iteration++) {

    a = A[i][k];
    b = B[k][j];
    c = ijResult[2];
    c += a * b * alpha;
    ijResult[2] = c;
    k = (k + 1) % ni;
}

if (rank != 0) {
  MPI_Send(&ijResult, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

if (rank == 0) {
  for (int worker = 1; worker < mpiSize; worker++) {
    MPI_Recv(&ijResult, 3, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int workerI = ijResult[0];
    int workerJ = ijResult[1];
    double result = ijResult[2];
    
    C[workerI][workerJ] = result;
  }
}

#pragma endscop

}


int main(int argc, char** argv) {
    int rank, size;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


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
