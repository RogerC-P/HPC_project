
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

/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,NN,NN,n,n),
		 DATA_TYPE POLYBENCH_1D(b,NN,n),
		 DATA_TYPE POLYBENCH_1D(x,NN,n),
		 DATA_TYPE POLYBENCH_1D(y,NN,n))
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
  /*
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NN, NN, n, n);
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
    */
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,NN,n))

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

void invert_unity_lower_triangular_matrix(int d, DATA_TYPE L[d][d]) {
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

            int k = j;
            for (; k+4 <= i; k+=4) {
                sum1 += L[i][k+0] * b[k+0][j];
                sum2 += L[i][k+1] * b[k+1][j];
                sum3 += L[i][k+2] * b[k+2][j];
                sum4 += L[i][k+3] * b[k+3][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 8; 
                #endif
            }
            for (; k < i; k++) {
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

void invert_upper_triangular_matrix(int d, DATA_TYPE U[d][d]) {
    DATA_TYPE c[d][d];
    memset(c, 0, sizeof(c));

    for (int i = d-1; i >= 0; i--) {
        c[i][i] = 1 / U[i][i]; // diagonal

        #ifdef COUNT_FLOPS
        FLOP_COUNTER += 1; 
        #endif

        for (int j = d-1; j >= i + 1; j--) {
            DATA_TYPE sum1 = 0.0;
            DATA_TYPE sum2 = 0.0;
            DATA_TYPE sum3 = 0.0;
            DATA_TYPE sum4 = 0.0;
            int k = i+1;
            for (; k+4 <= j; k+=4) {
                sum1 += U[i][k] * c[k][j];
                sum2 += U[i][k+1] * c[k+1][j];
                sum3 += U[i][k+2] * c[k+2][j];
                sum4 += U[i][k+3] * c[k+3][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            for (; k <= j; k++) {
                sum1 += U[i][k] * c[k][j];
 
                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            sum1 += sum2;
            sum3 += sum4;
            sum1 += sum3;
            c[i][j] = -sum1 / U[i][i];

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 1; 
            #endif
        }
    }

        memcpy(U, c, sizeof(c));
}

// Equation 4 in the paper linked above
// LU factorization according to Doolitte's method
void block_lu_factorization_recursive(
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
    // Step 1: Compute l, u
    DATA_TYPE l[s][s]; // Equivalent to L_11 in paper
    DATA_TYPE u[s][s]; // Equivalent to U_11 in paper
    memset(l, 0, sizeof(l));
    memset(u, 0, sizeof(u));

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
            DATA_TYPE sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += l[k][m] * u[m][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            u[k][j] = A[o + k][o + j] - sum;

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 1; 
            #endif
        }
        for (int i = k + 1; i < s; i++) {
            DATA_TYPE sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += l[i][m] * u[m][k];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            l[i][k] = (A[o + i][o + k] - sum) / u[k][k];

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 2; 
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

    // Compute the inverse in-place.
    //invert_unity_lower_triangular_matrix(s, l);
    //invert_upper_triangular_matrix(s, u);

    
    // Compute the inverse in-place.
    invert_unity_lower_triangular_matrix(s, l);
    invert_upper_triangular_matrix(s, u);


    // Step 2: Compute U_12 = L_11^(-1) A_12
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < n - o - s; j++) {
            DATA_TYPE sum = 0.0;
            for (int k = 0; k < s; k++) {
                sum += l[i][k] * A[o + k][o + s + j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            U[o + i][o + s + j] = sum;
        }
    }

    
    // Step 3: Compute L_21 = A_21 U_11^(-1)
    for (int i = 0; i < n - o - s; i++) {
        for (int j = 0; j < s; j++) {
            DATA_TYPE sum = 0.0;
            for (int k = 0; k < s; k++) {
                sum += A[o + s + i][o + k] * u[k][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            L[o  + s + i][o + j] = sum;
        }
    }
        

    // Compute A_22'
    for (int i = o; i < n; i++) {
        for (int j = o; j < n; j++) {
            DATA_TYPE sum = 0.0;
            for (int k = 0; k < s; k++) {
                sum += L[i][o + k] * U[o + k][j];

                #ifdef COUNT_FLOPS
                FLOP_COUNTER += 2; 
                #endif
            }
            A[i][j] = A[i][j] - sum;
            
            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 1; 
            #endif
        }
    }

    // Step 4: Recursively compute L_22, U_22
    int next_o = o + s;
    int next_s = min(s, n - next_o);
    if (next_s > 0) {
        block_lu_factorization_recursive(n, next_o, next_s, A, L, U);
    }
}

// Solves Ax=b for x
// Modifies A, x, and b.
void block_lu_factorization(int n,
		   DATA_TYPE POLYBENCH_2D(A,NN,NN,n,n),
		   DATA_TYPE POLYBENCH_1D(b,NN,n),
		   DATA_TYPE POLYBENCH_1D(x,NN,n),
		   DATA_TYPE POLYBENCH_1D(y,NN,n)
) {

    int s = min(16, n);
    DATA_TYPE (*L)[n] = calloc(n * n, sizeof(DATA_TYPE));
    DATA_TYPE (*U)[n] = calloc(n * n, sizeof(DATA_TYPE));

    block_lu_factorization_recursive(n, 0, s, A, L, U);

    // Solve Ly = b for y (forward substitution)
    for (int i = 0; i < n; i++) {
        DATA_TYPE sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 2; 
            #endif
        }
        y[i] = (b[i] - sum) / L[i][i];

        #ifdef COUNT_FLOPS
        FLOP_COUNTER += 2; 
        #endif
    }
    // Solve Ux = y for x (back substitution)
    for (int i = n - 1; i >= 0; i--) {
        DATA_TYPE sum = 0.0;
        for (int j = n - 1; j > i; j--) {
            sum += U[i][j] * x[j];

            #ifdef COUNT_FLOPS
            FLOP_COUNTER += 2; 
            #endif
        }

        x[i] = (y[i] - sum) / U[i][i];

        #ifdef COUNT_FLOPS
        FLOP_COUNTER += 2; 
        #endif
    }

    free(L);
    free(U);
}




/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,NN,NN,n,n),
		   DATA_TYPE POLYBENCH_1D(b,NN,n),
		   DATA_TYPE POLYBENCH_1D(x,NN,n),
		   DATA_TYPE POLYBENCH_1D(y,NN,n))
{
  int i,j,k;
  DATA_TYPE w;

  #pragma scop
  block_lu_factorization(n, A, b, x, y);

  /*
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <i; j++) {
      w = A[i][j];
      for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
      }
      A[i][j] = w / A[j][j];
    }
    for (j = i; j < _PB_N; j++) {
      w = A[i][j];
      for (k = 0; k < i; k++) {
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w;
    }
  }

  for (i = 0; i < _PB_N; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   for (i = _PB_N-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < _PB_N; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }
  */
  #pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = NN;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NN, NN, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, NN, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, NN, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, NN, n);


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

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}
