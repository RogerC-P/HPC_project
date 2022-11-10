
// For implementation details, refer to the following paper:
// https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5171403

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"

#include <omp.h>
#include <immintrin.h>
#include <mpi.h>


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


DATA_TYPE min(DATA_TYPE x, DATA_TYPE y) {
  if (x < y) {
    return x;
  } else {
    return y;
  }
}


void invert_unity_lower_triangular_matrix_opt_avx(int d, DATA_TYPE L[d][d]) {
    DATA_TYPE b[d][d];
    memset(b, 0, sizeof(b));

    for (int i = 0; i < d; i++) {
        b[i][i] = 1; // diagonal
    }

    for (int i = 1; i < d; i++) {
        for (int j = 0; j < i; j++) {
            DATA_TYPE sum1 = 0.0;
            DATA_TYPE sum2 = 0.0;
            DATA_TYPE sum3 = 0.0;
            DATA_TYPE sum4 = 0.0;

            int krest = j;
            for (int k = krest; k+4 <= i; k+=4) {
                sum1 += L[i][k+0] * b[k+0][j];
                sum2 += L[i][k+1] * b[k+1][j];
                sum3 += L[i][k+2] * b[k+2][j];
                sum4 += L[i][k+3] * b[k+3][j];
                krest = k + 4;

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 8; 
                #endif
            }
            for (int k = krest; k < i; k++) {
                sum1 += L[i][k] * b[k][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            sum1 += sum2;
            sum3 += sum4;
            sum1 += sum3;
            b[i][j] = -sum1;
        }
    }

    memcpy(L, b, sizeof(b));
}

void invert_upper_triangular_matrix_opt_avx(int d, DATA_TYPE U[d][d]) {
    DATA_TYPE c[d][d];
    memset(c, 0, sizeof(c));

    for (int i = d-1; i >= 0; i--) {
        c[i][i] = 1 / U[i][i]; // diagonal

        #ifdef COUNT_FLOPS
        FLOP_COUNTER += 1; 
        #endif

        for (int j = d-1; j >= i + 1; j--) {
            DATA_TYPE sum = 0.0;
            for (int k = i+1; k <= j; k++) {
                sum += U[i][k] * c[k][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            c[i][j] = -sum / U[i][i];

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 1; 
            #endif
        }
    }

        memcpy(U, c, sizeof(c));
}

void block_lu_factorization_recursive_opt_avx_rank_0(
    int n,
    int o,
    int s,
    DATA_TYPE l[s][s],
    DATA_TYPE A[n][n],
    DATA_TYPE L[n][n],
    DATA_TYPE U[n][n]
) {
    
    // Compute the inverse in-place.
    invert_unity_lower_triangular_matrix_opt_avx(s, l);

    // Step 2: Compute U_12 = L_11^(-1) A_12
    #pragma omp parallel for
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < n - o - s; j++) {
            DATA_TYPE sum1 = 0.0;
            DATA_TYPE sum2 = 0.0;
            DATA_TYPE sum3 = 0.0;
            DATA_TYPE sum4 = 0.0;

            int krest = 0;
            for (int k = krest; k+4 <= s; k+=4) {
                sum1 += l[i][k+0] * A[o + k+0][o + s + j];
                sum2 += l[i][k+1] * A[o + k+1][o + s + j];
                sum3 += l[i][k+2] * A[o + k+2][o + s + j];
                sum4 += l[i][k+3] * A[o + k+3][o + s + j];
                krest = k + 4;

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 8; 
                #endif
            }
            for (int k = krest; k < s; k++) {
                sum1 += l[i][k] * A[o + k][o + s + j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            sum1 += sum2;
            sum3 += sum4;
            sum1 += sum3;
            U[o + i][o + s + j] = sum1;
        }
    }
}

void block_lu_factorization_recursive_opt_avx_rank_1(
    int n,
    DATA_TYPE A[n][n],
    DATA_TYPE L[n][n],
    DATA_TYPE U[n][n]
) {
    int s = min(16, n);
    int o = 0;

    DATA_TYPE l[s][s]; // Equivalent to L_11 in paper 
    DATA_TYPE u[s][s]; // Equivalent to U_11 in paper
    memset(l, 0, sizeof(l));
    memset(u, 0, sizeof(u));


    while (1) {
        MPI_Recv(&o, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (o == -1) {
            return;
        }

        MPI_Recv(u, s * s, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute the inverse in-place.
        invert_upper_triangular_matrix_opt_avx(s, u);

        MPI_Recv(A, n * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(L, n * n, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Step 3: Compute L_21 = A_21 U_11^(-1)
        #pragma omp parallel for
        for (int i = 0; i < n - o - s; i++) {
            for (int j = 0; j < s; j++) {
                DATA_TYPE sum1 = 0.0;
                DATA_TYPE sum2 = 0.0;
                DATA_TYPE sum3 = 0.0;
                DATA_TYPE sum4 = 0.0;

                int krest = 0;
                for (int k = krest; k+4 <= s; k+=4) {
                    sum1 += A[o + s + i][o + k+0] * u[k+0][j];
                    sum2 += A[o + s + i][o + k+1] * u[k+1][j];
                    sum3 += A[o + s + i][o + k+2] * u[k+2][j];
                    sum4 += A[o + s + i][o + k+3] * u[k+3][j];
                    krest = k + 4;

                    #ifdef COUNT_FLOPS
                    FLOP_COUNTER += 8; 
                    #endif
                }
                for (int k = krest; k < s; k++) {
                    sum1 += A[o + s + i][o + k] * u[k][j];

                    #ifdef COUNT_FLOPS
                    FLOP_COUNTER += 2; 
                    #endif
                }
                sum1 += sum2;
                sum3 += sum4;
                sum1 += sum3;
                L[o  + s + i][o + j] = sum1;
            }
        }

        MPI_Send(L, n * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
}

// Equation 4 in the paper linked above
// LU factorization according to Doolitte's method
void block_lu_factorization_recursive_opt_avx(
    int n,
    int o, // offset of submatrix (starting index for both x and y)
    int s, // max size of submatrix (exclusive)
    DATA_TYPE A[n][n],
    DATA_TYPE L[n][n],
    DATA_TYPE U[n][n]
) {
#ifdef DEBUG
    assert(s > 0);
    assert(n >= o + s);
#endif

    DATA_TYPE l[s][s]; // Equivalent to L_11 in paper 
    DATA_TYPE u[s][s]; // Equivalent to U_11 in paper
    memset(l, 0, sizeof(l));
    memset(u, 0, sizeof(u));

    // Step 1: Compute l, u
    DATA_TYPE a[s][s]; 
    memset(a, 0, sizeof(a));
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            a[i][j] = A[i + o][j + o];
        }
    }

    // Set diagonal of L
    for (int i = 0; i < s; i++) {
        l[i][i] = 1;
    }
    // See equation 4
    for (int j = 0; j < s; j++) {
        u[0][j] = A[o][o + j];
    }
    for (int i = 0; i < s; i++) {
        l[i][0] = A[o + i][o] / u[0][0];

        #ifdef COUNT_FLOPS
        FLOP_COUNTER += 1; 
        #endif
    }
    for (int k = 1; k < s; k++) {
        for (int j = k; j < s; j++) {
            DATA_TYPE sum1 = 0.0;
            DATA_TYPE sum2 = 0.0;
            DATA_TYPE sum3 = 0.0;
            DATA_TYPE sum4 = 0.0;

            int mrest = 0;
            for (int m = mrest; m+4 <= k; m+=4) {
                sum1 += l[k][m+0] * u[m+0][j];
                sum2 += l[k][m+1] * u[m+1][j];
                sum3 += l[k][m+2] * u[m+2][j];
                sum4 += l[k][m+3] * u[m+3][j];
                mrest = m + 4;

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 8; 
                #endif
            }
            for (int m = mrest; m < k; m++) {
                sum1 += l[k][m] * u[m][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            sum1 += sum2;
            sum3 += sum4;
            sum1 += sum3;
            u[k][j] = A[o + k][o + j] - sum1;

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 4; 
            #endif
        }

        for (int i = k + 1; i < s; i++) {
            DATA_TYPE sum1 = 0.0;
            DATA_TYPE sum2 = 0.0;
            DATA_TYPE sum3 = 0.0;
            DATA_TYPE sum4 = 0.0;

            int mrest = 0;
            for (int m = mrest; m+4 <= k; m+=4) {
                sum1 += l[i][m+0] * u[m+0][k];
                sum2 += l[i][m+1] * u[m+1][k];
                sum3 += l[i][m+2] * u[m+2][k];
                sum4 += l[i][m+3] * u[m+3][k];
                mrest = m + 4;

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 8; 
                #endif
            }
            for (int m = mrest; m < k; m++) {
                sum1 += l[i][m] * u[m][k];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            sum1 += sum2;
            sum3 += sum4;
            sum1 += sum3;
            l[i][k] = (A[o + i][o + k] - sum1) / u[k][k];

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 5; 
            #endif
        }
    }

    // Store into resulting L U matrices.
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            L[o + i][o + j] = l[i][j];
            U[o + i][o + j] = u[i][j];
        }
    }


    MPI_Send(&o, 1, MPI_INT, 1, 4, MPI_COMM_WORLD);
    MPI_Send(u, s * s, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    MPI_Send(A, n * n, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
    MPI_Send(L, n * n, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);

    block_lu_factorization_recursive_opt_avx_rank_0(n, o, s, l, A, L, U);

    MPI_Recv(L, n * n, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    
    // Compute A_22'
    #pragma omp parallel for
    for (int i = o; i < n; i++) {
        int jrest = o;
        for (int j = jrest; j+8 <= n; j+=8) {
            __m256d sumv1 = _mm256_setzero_pd();
            __m256d sumv2 = _mm256_setzero_pd();
            __m256d sumv3 = _mm256_setzero_pd();
            __m256d sumv4 = _mm256_setzero_pd();
            __m256d sumv5 = _mm256_setzero_pd();
            __m256d sumv6 = _mm256_setzero_pd();
            __m256d sumv7 = _mm256_setzero_pd();
            __m256d sumv8 = _mm256_setzero_pd();

            int krest = 0;
            for (int k = krest; k+4 <= s; k+=4) {
                __m256d LL = _mm256_loadu_pd(&L[i][o+k]); 

                __m256d L1 = _mm256_set1_pd(((double *) &LL)[0]); 
                __m256d L2 = _mm256_set1_pd(((double *) &LL)[1]); 
                __m256d L3 = _mm256_set1_pd(((double *) &LL)[2]); 
                __m256d L4 = _mm256_set1_pd(((double *) &LL)[3]); 
                
                __m256d U1 = _mm256_loadu_pd(&U[o+k+0][j]);
                __m256d U2 = _mm256_loadu_pd(&U[o+k+1][j]);
                __m256d U3 = _mm256_loadu_pd(&U[o+k+2][j]);
                __m256d U4 = _mm256_loadu_pd(&U[o+k+3][j]);

                __m256d U5 = _mm256_loadu_pd(&U[o+k+0][j+4]);
                __m256d U6 = _mm256_loadu_pd(&U[o+k+1][j+4]);
                __m256d U7 = _mm256_loadu_pd(&U[o+k+2][j+4]);
                __m256d U8 = _mm256_loadu_pd(&U[o+k+3][j+4]);

                sumv1 = _mm256_fmadd_pd(L1, U1, sumv1);
                sumv2 = _mm256_fmadd_pd(L2, U2, sumv2);
                sumv3 = _mm256_fmadd_pd(L3, U3, sumv3);
                sumv4 = _mm256_fmadd_pd(L4, U4, sumv4);

                sumv5 = _mm256_fmadd_pd(L1, U5, sumv5);
                sumv6 = _mm256_fmadd_pd(L2, U6, sumv6);
                sumv7 = _mm256_fmadd_pd(L3, U7, sumv7);
                sumv8 = _mm256_fmadd_pd(L4, U8, sumv8);
                krest = k + 8;

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 64; 
                #endif
            }

            sumv1 = _mm256_add_pd(sumv1, sumv2);
            sumv3 = _mm256_add_pd(sumv3, sumv4);

            sumv1 = _mm256_add_pd(sumv1, sumv3);

            sumv5 = _mm256_add_pd(sumv5, sumv6);
            sumv7 = _mm256_add_pd(sumv7, sumv8);

            sumv5 = _mm256_add_pd(sumv5, sumv7);

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 24; 
            #endif

            __m256d A1 = _mm256_loadu_pd(&A[i][j]);
            __m256d A2 = _mm256_loadu_pd(&A[i][j+4]);

            for (int k = krest; k < s; k++) {
                __m256d L1 = _mm256_set1_pd(L[i][o + k]); 
                __m256d U1 = _mm256_loadu_pd(&U[o+k][j]);
                sumv1 = _mm256_fmadd_pd(L1, U1, sumv1);

                __m256d U2 = _mm256_loadu_pd(&U[o+k][j+4]);
                sumv5 = _mm256_fmadd_pd(L1, U2, sumv5);

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 16; 
                #endif
            }

            A1 = _mm256_sub_pd(A1, sumv1);
            _mm256_storeu_pd(&A[i][j], A1);
            
            A2 = _mm256_sub_pd(A2, sumv5);
            _mm256_storeu_pd(&A[i][j+4], A2);
            
            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 8; 
            #endif

            jrest = j + 8;
        }

        for (int j = jrest; j < n; j++) {
            DATA_TYPE sum1 = 0.0;
            DATA_TYPE sum2 = 0.0;
            DATA_TYPE sum3 = 0.0;
            DATA_TYPE sum4 = 0.0;

            int k = 0;
            for (; k+4 <= s; k+=4) {
                sum1 += L[i][o + k+0] * U[o + k+0][j];
                sum2 += L[i][o + k+1] * U[o + k+1][j];
                sum3 += L[i][o + k+2] * U[o + k+2][j];
                sum4 += L[i][o + k+3] * U[o + k+3][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 8; 
                #endif
            }
            for (; k < s; k++) {
                sum1 += L[i][o + k] * U[o + k][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            sum1 += sum2;
            sum3 += sum4;
            sum1 += sum3;
            A[i][j] = A[i][j] - sum1;
            
            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 4; 
            #endif
        }
    }

    // Step 4: Recursively compute L_22, U_22
    int next_o = o + s;
    int next_s = min(s, n - next_o);

    if (next_s > 0) {
        block_lu_factorization_recursive_opt_avx(n, next_o, next_s, A, L, U);
    } else {
        int exit = -1;
        MPI_Send(&exit, 1, MPI_INT, 1, 4, MPI_COMM_WORLD);
    }
}

// Solves Ax=b for x
// Modifies A, x, and b.
void block_lu_factorization_opt_avx_double(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n)) {

    int s = min(16, n);
    //DATA_TYPE L[n][n];
    //DATA_TYPE U[n][n];
    DATA_TYPE (*L)[n] = calloc(n * n, sizeof(DATA_TYPE));
    DATA_TYPE (*U)[n] = calloc(n * n, sizeof(DATA_TYPE));

    block_lu_factorization_recursive_opt_avx(n, 0, s, A, L, U);

    // Solve Ly = b for y (forward substitution)
    for (int i = 0; i < n; i++) {
        __m256d sum1 = _mm256_set1_pd(0.0);
        __m256d sum2 = _mm256_set1_pd(0.0);

        int jrest = 0;
        for (int j = jrest; j+8<= i; j+=8) {
            __m256d L1 = _mm256_loadu_pd(&L[i][j]);
            __m256d y1 = _mm256_loadu_pd(&y[j]);
            __m256d L2 = _mm256_loadu_pd(&L[i][j+4]);
            __m256d y2 = _mm256_loadu_pd(&y[j+4]);
            sum1 = _mm256_fmadd_pd(L1, y1, sum1);
            sum2 = _mm256_fmadd_pd(L2, y2, sum2);
            jrest = j + 8;

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 16; 
            #endif
        }

        sum1 = _mm256_add_pd(sum1, sum2);

        double *sum1_vec_arr = (double *) &sum1;
        double sum = sum1_vec_arr[0]
            + sum1_vec_arr[1]
            + sum1_vec_arr[2]
            + sum1_vec_arr[3];

        #ifdef COUNT_FLOPS
            FLOP_COUNTER += 7; 
        #endif

        for (int j = jrest; j < i; j++) {
            sum += L[i][j] * y[j];

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 2; 
            #endif
        }

        y[i] = (b[i] - sum) / L[i][i];

        #ifdef COUNT_FLOPS
        FLOP_COUNTER += 1; 
        #endif
    }
    // Solve Ux = y for x (back substitution)
    for (int i = n - 1; i >= 0; i--) {
        DATA_TYPE sum1 = 0.0;
        DATA_TYPE sum2 = 0.0;
        DATA_TYPE sum3 = 0.0;
        DATA_TYPE sum4 = 0.0;

        int jrest = i + 1;
        for (int j = jrest; j+4 <= n; j+=4) {
            sum1 += U[i][j+0] * x[j+0];
            sum1 += U[i][j+1] * x[j+1];
            sum1 += U[i][j+2] * x[j+2];
            sum1 += U[i][j+3] * x[j+3];
            jrest = j + 4;

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 8; 
            #endif
        }
        for (int j = jrest; j < n; j++) {
            sum1 += U[i][j] * x[j];

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 2; 
            #endif
        }
        sum1 += sum2;
        sum3 += sum4;
        sum1 += sum3;

        x[i] = (y[i] - sum1) / U[i][i];

        #ifdef COUNT_FLOPS
        FLOP_COUNTER += 5; 
        #endif
    }
    

    free(L);
    free(U);
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i,j,k;
  DATA_TYPE w;

  #pragma scop
  block_lu_factorization_opt_avx_double(n, A, b, x, y);
  #pragma endscop
}


int main(int argc, char** argv)
{
      
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(U, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

    if (rank == 0) {

        /* Initialize array(s). */
        init_array (n,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(b),
                POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y));

        /* Start timer. */
        polybench_start_instruments;

        /* Run kernel. */
        kernel_ludcmp (n,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(b),
                POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y));

        /* Stop and print timer. */
        polybench_stop_instruments;
        polybench_print_instruments;

        /* Prevent dead-code elimination. All live-out data must be printed
            by the function call in argument. */
        polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

    } else if (rank == 1) {
        block_lu_factorization_recursive_opt_avx_rank_1(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(U));
    }

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);


  MPI_Finalize();

  return 0;
}
