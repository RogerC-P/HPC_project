/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _GEMM_H
# define _GEMM_H


# if !defined(NI) && !defined(NJ) && !defined(NK)
/* Define sample dataset sizes. */
#  ifdef MINI_DATASET
#define NI 20
#define NJ 25
#define NK 30
#  endif

#  ifdef SMALL_DATASET
#   define NI 60
#   define NJ 70
#   define NK 80
#  endif

#  ifdef MEDIUM_DATASET
#define NI 200
#define NJ 220
#define NK 240
#  endif

#  ifdef LARGE_DATASET
#   define NI 1000
#   define NJ 1100
#   define NK 1200
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NI 16384
#   define NJ 16384
#   define NK 16384
#  endif

#  ifdef BIGGER_DATASET
#   define NI 4096
#   define NJ 4096
#   define NK 4096
#  endif

#  ifdef DATASET_5000
#   define NI 5000
#   define NJ 5000
#   define NK 5000
#  endif

#  ifdef DATASET_6300
#   define NI 6300
#   define NJ 6300
#   define NK 6300
#  endif

#  ifdef DATASET_7938
#   define NI 7938
#   define NJ 7938
#   define NK 7938
#  endif

#  ifdef DATASET_10000
#   define NI 10000
#   define NJ 10000
#   define NK 10000
#  endif

#  ifdef DATASET_12600
#   define NI 12600
#   define NJ 12600
#   define NK 12600
#  endif

#  ifdef DATASET_15874
#   define NI 15874
#   define NJ 15874
#   define NK 15874
#  endif


#  ifdef DATASET_1024
#   define NI 1024
#   define NJ 1024
#   define NK 1024
#  endif

#  ifdef DATASET_2048
#   define NI 2048
#   define NJ 2048
#   define NK 2048
#  endif

#  ifdef DATASET_4096
#   define NI 4096
#   define NJ 4096
#   define NK 4096
#  endif

#  ifdef DATASET_8192
#   define NI 8192
#   define NJ 8192
#   define NK 8192
#  endif

#endif /* !(NI NJ NK) */

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)
# define _PB_NK POLYBENCH_LOOP_BOUND(NK,nk)


/* Default data type */
# if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#  define DATA_TYPE_IS_DOUBLE
# endif

#ifdef DATA_TYPE_IS_INT
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
#endif

#ifdef DATA_TYPE_IS_FLOAT
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2f "
#  define SCALAR_VAL(x) x##f
#  define SQRT_FUN(x) sqrtf(x)
#  define EXP_FUN(x) expf(x)
#  define POW_FUN(x,y) powf(x,y)
# endif

#ifdef DATA_TYPE_IS_DOUBLE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
# endif

#endif /* !_GEMM_H */
