/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3 -mavx
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include<stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 256
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* 4 rows of x multiply 1 column of y */
static void mult4x1 (int K, int inc, double *x, double *y, double *frm)
{
/* frm represents four row multiplication
 *  frm[0] = x[0] * y + frm[0]
 *  frm[1] = x[1] * y + frm[1]
 *  frm[2] = x[2] * y + frm[2]
 *  frm[3] = x[3] * y + frm[3]
 *  each time calculate four rows within the same columns of x
 */
  register double frm_reg_0 = 0.0;
  register double frm_reg_1 = 0.0;
  register double frm_reg_2 = 0.0;
  register double frm_reg_3 = 0.0;
  register double y_reg     = 0.0;
  frm_reg_0  = frm[0];
  frm_reg_1  = frm[1];
  frm_reg_2  = frm[2];
  frm_reg_3  = frm[3];

  for(int k = 0; k < K; ++k)
  {
    y_reg      = y[k];
    frm_reg_0 += y_reg * x[k*inc];
    frm_reg_1 += y_reg * x[1+k*inc];
    frm_reg_2 += y_reg * x[2+k*inc];
    frm_reg_3 += y_reg * x[3+k*inc];
  }

  frm[0] = frm_reg_0;
  frm[1] = frm_reg_1;
  frm[2] = frm_reg_2;
  frm[3] = frm_reg_3;
}

/* 1 row of x multiplies 4 columns of y */
static void mult1x4 (int K, int inc, double *x, double *y, double *fcm)
{
  register double fcm_reg_0 = 0.0;
  register double fcm_reg_1 = 0.0;
  register double fcm_reg_2 = 0.0;
  register double fcm_reg_3 = 0.0;
  register double x_reg     = 0.0;
  fcm_reg_0  = fcm[0];
  fcm_reg_1  = fcm[inc];
  fcm_reg_2  = fcm[2*inc];
  fcm_reg_3  = fcm[3*inc];
  double *x_ptr  = &x[0];
  double *y0_ptr = &y[0];
  double *y1_ptr = &y[inc];
  double *y2_ptr = &y[2*inc];
  double *y3_ptr = &y[3*inc];

  for(int k = 0; k < K; k+=4)
  {
    x_reg      = *x_ptr;
    fcm_reg_0 += x_reg * *y0_ptr;
    fcm_reg_1 += x_reg * *y1_ptr;
    fcm_reg_2 += x_reg * *y2_ptr;
    fcm_reg_3 += x_reg * *y3_ptr;
    x_reg      = *(x_ptr+inc);
    fcm_reg_0 += x_reg * *(y0_ptr+1);
    fcm_reg_1 += x_reg * *(y1_ptr+1);
    fcm_reg_2 += x_reg * *(y2_ptr+1);
    fcm_reg_3 += x_reg * *(y3_ptr+1);
    x_reg      = *(x_ptr+2*inc);
    fcm_reg_0 += x_reg * *(y0_ptr+2);
    fcm_reg_1 += x_reg * *(y1_ptr+2);
    fcm_reg_2 += x_reg * *(y2_ptr+2);
    fcm_reg_3 += x_reg * *(y3_ptr+2);
    x_reg      = *(x_ptr+3*inc);
    fcm_reg_0 += x_reg * *(y0_ptr+3);
    fcm_reg_1 += x_reg * *(y1_ptr+3);
    fcm_reg_2 += x_reg * *(y2_ptr+3);
    fcm_reg_3 += x_reg * *(y3_ptr+3);
    x_ptr  += 4*inc;
    y0_ptr += 4;
    y1_ptr += 4;
    y2_ptr += 4;
    y3_ptr += 4;
  }

  fcm[0]     = fcm_reg_0;
  fcm[inc]   = fcm_reg_1;
  fcm[2*inc] = fcm_reg_2;
  fcm[3*inc] = fcm_reg_3;
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

/* 4 row of x multiplies 4 columns of y */
static void mult4x4 (int K, int inc, double *x, double *y, double *fsm)
{
  v2df_t fsm_vreg_00_10;
  v2df_t fsm_vreg_20_30;
  v2df_t fsm_vreg_01_11;
  v2df_t fsm_vreg_21_31;
  v2df_t fsm_vreg_02_12;
  v2df_t fsm_vreg_22_32;
  v2df_t fsm_vreg_03_13;
  v2df_t fsm_vreg_23_33;
  fsm_vreg_00_10.v = _mm_setzero_pd();
  fsm_vreg_20_30.v = _mm_setzero_pd();
  fsm_vreg_01_11.v = _mm_setzero_pd();
  fsm_vreg_21_31.v = _mm_setzero_pd();
  fsm_vreg_02_12.v = _mm_setzero_pd();
  fsm_vreg_22_32.v = _mm_setzero_pd();
  fsm_vreg_03_13.v = _mm_setzero_pd();
  fsm_vreg_23_33.v = _mm_setzero_pd();
  fsm_vreg_00_10.v = _mm_load_pd((double *) &fsm[0]);
  fsm_vreg_20_30.v = _mm_load_pd((double *) &fsm[2]);
  fsm_vreg_01_11.v = _mm_load_pd((double *) &fsm[inc]);
  fsm_vreg_21_31.v = _mm_load_pd((double *) &fsm[2+inc]);
  fsm_vreg_02_12.v = _mm_load_pd((double *) &fsm[2*inc]);
  fsm_vreg_22_32.v = _mm_load_pd((double *) &fsm[2+2*inc]);
  fsm_vreg_03_13.v = _mm_load_pd((double *) &fsm[3*inc]);
  fsm_vreg_23_33.v = _mm_load_pd((double *) &fsm[2+3*inc]);
  v2df_t x_vreg_0k_1k;
  v2df_t x_vreg_2k_3k;
  v2df_t y_vreg_k0_k0;
  v2df_t y_vreg_k1_k1;
  v2df_t y_vreg_k2_k2;
  v2df_t y_vreg_k3_k3;  
  x_vreg_0k_1k.v = _mm_setzero_pd();
  x_vreg_2k_3k.v = _mm_setzero_pd();
  y_vreg_k0_k0.v = _mm_setzero_pd();
  y_vreg_k1_k1.v = _mm_setzero_pd();
  y_vreg_k2_k2.v = _mm_setzero_pd();
  y_vreg_k3_k3.v = _mm_setzero_pd();
  double *x_0k_1k_ptr = &x[0];
  double *x_2k_3k_ptr = &x[2];
  double *y_k0_k0_ptr = &y[0];
  double *y_k1_k1_ptr = &y[inc];
  double *y_k2_k2_ptr = &y[2*inc];
  double *y_k3_k3_ptr = &y[3*inc];

  for(int k = 0; k < K; ++k)
  {
    x_vreg_0k_1k.v = _mm_load_pd((double *) x_0k_1k_ptr);
    x_vreg_2k_3k.v = _mm_load_pd((double *) x_2k_3k_ptr);
    y_vreg_k0_k0.v = _mm_loaddup_pd((double *) y_k0_k0_ptr++);
    y_vreg_k1_k1.v = _mm_loaddup_pd((double *) y_k1_k1_ptr++);
    y_vreg_k2_k2.v = _mm_loaddup_pd((double *) y_k2_k2_ptr++);
    y_vreg_k3_k3.v = _mm_loaddup_pd((double *) y_k3_k3_ptr++);
    x_0k_1k_ptr   += 4;
    x_2k_3k_ptr   += 4;
    fsm_vreg_00_10.v += x_vreg_0k_1k.v * y_vreg_k0_k0.v;
    fsm_vreg_01_11.v += x_vreg_0k_1k.v * y_vreg_k1_k1.v;
    fsm_vreg_02_12.v += x_vreg_0k_1k.v * y_vreg_k2_k2.v;
    fsm_vreg_03_13.v += x_vreg_0k_1k.v * y_vreg_k3_k3.v;
    fsm_vreg_20_30.v += x_vreg_2k_3k.v * y_vreg_k0_k0.v;
    fsm_vreg_21_31.v += x_vreg_2k_3k.v * y_vreg_k1_k1.v;
    fsm_vreg_22_32.v += x_vreg_2k_3k.v * y_vreg_k2_k2.v;
    fsm_vreg_23_33.v += x_vreg_2k_3k.v * y_vreg_k3_k3.v;
  }

  fsm[0]       = fsm_vreg_00_10.d[0];
  fsm[inc]     = fsm_vreg_01_11.d[0];
  fsm[2*inc]   = fsm_vreg_02_12.d[0];
  fsm[3*inc]   = fsm_vreg_03_13.d[0];
  fsm[1]       = fsm_vreg_00_10.d[1];
  fsm[1+inc]   = fsm_vreg_01_11.d[1];
  fsm[1+2*inc] = fsm_vreg_02_12.d[1];
  fsm[1+3*inc] = fsm_vreg_03_13.d[1];
  fsm[2]       = fsm_vreg_20_30.d[0];
  fsm[2+inc]   = fsm_vreg_21_31.d[0];
  fsm[2+2*inc] = fsm_vreg_22_32.d[0];
  fsm[2+3*inc] = fsm_vreg_23_33.d[0];
  fsm[3]       = fsm_vreg_20_30.d[1];
  fsm[3+inc]   = fsm_vreg_21_31.d[1];
  fsm[3+2*inc] = fsm_vreg_22_32.d[1];
  fsm[3+3*inc] = fsm_vreg_23_33.d[1];
}

#include<immintrin.h> // AVX intrinsics

typedef union
{
  __m256d v;
  double d[4];
} v4df_t;

static void mult1x1_256 (int K, int inc, double *x, double *y, double *fcm)
{
  double c = fcm[0];
  for (int k = 0; k < K; ++k)
    c += x[k*inc] * y[k];
  fcm[0] = c;
}

/* 1 row of x multiplies 4 columns of y , fcm is the continuous version of C */
/*
static void mult1x4_256 (int K, int inc, double *x, double *y, double *fcm)
{
  mult1x1_256 (K, inc, x, y, fcm);
  mult1x1_256 (K, inc, x, y+inc, fcm+inc);
  mult1x1_256 (K, inc, x, y+2*inc, fcm+2*inc);
  mult1x1_256 (K, inc, x, y+3*inc, fcm+3*inc);
}

static void mult3x4_256 (int K, int inc, double *x, double *y, double *fcm)
{
  mult1x4_256 (K, inc, x, y, fcm);
  mult1x4_256 (K, inc, x+1, y, fcm+1);
  mult1x4_256 (K, inc, x+2, y, fcm+2);
}
*/

static void mult3x3_256 (int K, int inc, double *x, double *y, double * restrict fsm)
{
  /* For each row i of A */
  for (int i = 0; i < 3; ++i)
    /* For each column j of B */
    for (int j = 0; j < 3; ++j)
    {
      /* Compute C(i,j) */
      double cij = fsm[i+j*inc];
      for (int k = 0; k < K; ++k)
        cij += x[i+k*inc] * y[k+j*inc];
      fsm[i+j*inc] = cij;
    } 
}

/* 1 row of x multiplies 4 columns of y with AVX 256 registers */
static void mult1x4_256 (int K, int inc, double *x, double *y, double * restrict fsm)
{
  register v4df_t fsm_vreg_00;
  register v4df_t fsm_vreg_01;
  register v4df_t fsm_vreg_02;
  register v4df_t fsm_vreg_03;
  fsm_vreg_00.v = _mm256_setzero_pd();
  fsm_vreg_01.v = _mm256_setzero_pd();
  fsm_vreg_02.v = _mm256_setzero_pd();
  fsm_vreg_03.v = _mm256_setzero_pd();

  fsm_vreg_00.v = _mm256_loadu_pd((double *) &fsm[0]);
  fsm_vreg_01.v = _mm256_loadu_pd((double *) &fsm[inc]);
  fsm_vreg_02.v = _mm256_loadu_pd((double *) &fsm[2*inc]);
  fsm_vreg_03.v = _mm256_loadu_pd((double *) &fsm[3*inc]);

  v4df_t x_vreg_0k;
  v4df_t y_vreg_k0_k0_k0_k0;
  v4df_t y_vreg_k1_k1_k1_k1;
  v4df_t y_vreg_k2_k2_k2_k2;
  v4df_t y_vreg_k3_k3_k3_k3;
  x_vreg_0k.v = _mm256_setzero_pd();
  y_vreg_k0_k0_k0_k0.v = _mm256_setzero_pd();
  y_vreg_k1_k1_k1_k1.v = _mm256_setzero_pd();
  y_vreg_k2_k2_k2_k2.v = _mm256_setzero_pd();
  y_vreg_k3_k3_k3_k3.v = _mm256_setzero_pd();

  register double *x_0k_ptr = &x[0];
  register double *y_k0_k0_k0_k0_ptr = &y[0];

  for(int k = 0; k < K; ++k)
  {
    x_vreg_0k.v = _mm256_loadu_pd((double *) x_0k_ptr);
    x_0k_ptr += 4;
    y_vreg_k0_k0_k0_k0.v = _mm256_set1_pd(*y_k0_k0_k0_k0_ptr);
    y_vreg_k1_k1_k1_k1.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+1));
    y_vreg_k2_k2_k2_k2.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+2));
    y_vreg_k3_k3_k3_k3.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+3));
    y_k0_k0_k0_k0_ptr += 4;
    fsm_vreg_00.v += x_vreg_0k.v * y_vreg_k0_k0_k0_k0.v;
    fsm_vreg_01.v += x_vreg_0k.v * y_vreg_k1_k1_k1_k1.v;
    fsm_vreg_02.v += x_vreg_0k.v * y_vreg_k2_k2_k2_k2.v;
    fsm_vreg_03.v += x_vreg_0k.v * y_vreg_k3_k3_k3_k3.v;
  }

  fsm[0]       = fsm_vreg_00.d[0];
  fsm[1]       = fsm_vreg_00.d[1];
  fsm[2]       = fsm_vreg_00.d[2];
  fsm[inc]     = fsm_vreg_01.d[0];
  fsm[1+inc]   = fsm_vreg_01.d[1];
  fsm[2+inc]   = fsm_vreg_01.d[2];
  fsm[2*inc]   = fsm_vreg_02.d[0];
  fsm[1+2*inc] = fsm_vreg_02.d[1];
  fsm[2+2*inc] = fsm_vreg_02.d[2];
  fsm[3*inc]   = fsm_vreg_03.d[0];
  fsm[1+3*inc] = fsm_vreg_03.d[1];
  fsm[2+3*inc] = fsm_vreg_03.d[2];

}

/* 3 row of x multiplies 4 columns of y with AVX 256 registers */
static void mult3x4_256 (int K, int inc, double *x, double *y, double * restrict fsm)
{
  register v4df_t fsm_vreg_00_10_20;
  register v4df_t fsm_vreg_01_11_21;
  register v4df_t fsm_vreg_02_12_22;
  register v4df_t fsm_vreg_03_13_23;
  fsm_vreg_00_10_20.v = _mm256_setzero_pd();
  fsm_vreg_01_11_21.v = _mm256_setzero_pd();
  fsm_vreg_02_12_22.v = _mm256_setzero_pd();
  fsm_vreg_03_13_23.v = _mm256_setzero_pd();

  fsm_vreg_00_10_20.v = _mm256_loadu_pd((double *) &fsm[0]);
  fsm_vreg_01_11_21.v = _mm256_loadu_pd((double *) &fsm[inc]);
  fsm_vreg_02_12_22.v = _mm256_loadu_pd((double *) &fsm[2*inc]);
  fsm_vreg_03_13_23.v = _mm256_loadu_pd((double *) &fsm[3*inc]);

  v4df_t x_vreg_0k_1k_2k;
  v4df_t y_vreg_k0_k0_k0_k0;
  v4df_t y_vreg_k1_k1_k1_k1;
  v4df_t y_vreg_k2_k2_k2_k2;
  v4df_t y_vreg_k3_k3_k3_k3;
  x_vreg_0k_1k_2k.v = _mm256_setzero_pd();
  y_vreg_k0_k0_k0_k0.v = _mm256_setzero_pd();
  y_vreg_k1_k1_k1_k1.v = _mm256_setzero_pd();
  y_vreg_k2_k2_k2_k2.v = _mm256_setzero_pd();
  y_vreg_k3_k3_k3_k3.v = _mm256_setzero_pd();

  register double *x_0k_1k_2k_ptr = &x[0];
  register double *y_k0_k0_k0_k0_ptr = &y[0];

  for(int k = 0; k < K; ++k)
  {
    x_vreg_0k_1k_2k.v = _mm256_loadu_pd((double *) x_0k_1k_2k_ptr);
    x_0k_1k_2k_ptr += 4;
    y_vreg_k0_k0_k0_k0.v = _mm256_set1_pd(*y_k0_k0_k0_k0_ptr);
    y_vreg_k1_k1_k1_k1.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+1));
    y_vreg_k2_k2_k2_k2.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+2));
    y_vreg_k3_k3_k3_k3.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+3));
    y_k0_k0_k0_k0_ptr += 4;
    fsm_vreg_00_10_20.v += x_vreg_0k_1k_2k.v * y_vreg_k0_k0_k0_k0.v;
    fsm_vreg_01_11_21.v += x_vreg_0k_1k_2k.v * y_vreg_k1_k1_k1_k1.v;
    fsm_vreg_02_12_22.v += x_vreg_0k_1k_2k.v * y_vreg_k2_k2_k2_k2.v;
    fsm_vreg_03_13_23.v += x_vreg_0k_1k_2k.v * y_vreg_k3_k3_k3_k3.v;
  }

  fsm[0]       = fsm_vreg_00_10_20.d[0];
  fsm[1]       = fsm_vreg_00_10_20.d[1];
  fsm[2]       = fsm_vreg_00_10_20.d[2];
  fsm[inc]     = fsm_vreg_01_11_21.d[0];
  fsm[1+inc]   = fsm_vreg_01_11_21.d[1];
  fsm[2+inc]   = fsm_vreg_01_11_21.d[2];
  fsm[2*inc]   = fsm_vreg_02_12_22.d[0];
  fsm[1+2*inc] = fsm_vreg_02_12_22.d[1];
  fsm[2+2*inc] = fsm_vreg_02_12_22.d[2];
  fsm[3*inc]   = fsm_vreg_03_13_23.d[0];
  fsm[1+3*inc] = fsm_vreg_03_13_23.d[1];
  fsm[2+3*inc] = fsm_vreg_03_13_23.d[2];

}


/* 4 row of x multiplies 4 columns of y with AVX 256 registers */
static void mult4x4_256 (int K, int inc, double *x, double *y, double * restrict fsm)
{
  register v4df_t fsm_vreg_00_10_20_30;
  register v4df_t fsm_vreg_01_11_21_31;
  register v4df_t fsm_vreg_02_12_22_32;
  register v4df_t fsm_vreg_03_13_23_33;
  fsm_vreg_00_10_20_30.v = _mm256_setzero_pd();
  fsm_vreg_01_11_21_31.v = _mm256_setzero_pd();
  fsm_vreg_02_12_22_32.v = _mm256_setzero_pd();
  fsm_vreg_03_13_23_33.v = _mm256_setzero_pd();

  fsm_vreg_00_10_20_30.v = _mm256_loadu_pd((double *) &fsm[0]);
  fsm_vreg_01_11_21_31.v = _mm256_loadu_pd((double *) &fsm[inc]);
  fsm_vreg_02_12_22_32.v = _mm256_loadu_pd((double *) &fsm[2*inc]);
  fsm_vreg_03_13_23_33.v = _mm256_loadu_pd((double *) &fsm[3*inc]);

  v4df_t x_vreg_0k_1k_2k_3k;
  v4df_t y_vreg_k0_k0_k0_k0;
  v4df_t y_vreg_k1_k1_k1_k1;
  v4df_t y_vreg_k2_k2_k2_k2;
  v4df_t y_vreg_k3_k3_k3_k3;
  x_vreg_0k_1k_2k_3k.v = _mm256_setzero_pd();
  y_vreg_k0_k0_k0_k0.v = _mm256_setzero_pd();
  y_vreg_k1_k1_k1_k1.v = _mm256_setzero_pd();
  y_vreg_k2_k2_k2_k2.v = _mm256_setzero_pd();
  y_vreg_k3_k3_k3_k3.v = _mm256_setzero_pd();

  register double *x_0k_1k_2k_3k_ptr = &x[0];
  register double *y_k0_k0_k0_k0_ptr = &y[0];

  for(int k = 0; k < K; ++k)
  {
    x_vreg_0k_1k_2k_3k.v = _mm256_loadu_pd((double *) x_0k_1k_2k_3k_ptr);
    x_0k_1k_2k_3k_ptr += 4;
    y_vreg_k0_k0_k0_k0.v = _mm256_set1_pd(*y_k0_k0_k0_k0_ptr);
    y_vreg_k1_k1_k1_k1.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+1));
    y_vreg_k2_k2_k2_k2.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+2));
    y_vreg_k3_k3_k3_k3.v = _mm256_set1_pd(*(y_k0_k0_k0_k0_ptr+3));
    y_k0_k0_k0_k0_ptr += 4;
    fsm_vreg_00_10_20_30.v += x_vreg_0k_1k_2k_3k.v * y_vreg_k0_k0_k0_k0.v;
    fsm_vreg_01_11_21_31.v += x_vreg_0k_1k_2k_3k.v * y_vreg_k1_k1_k1_k1.v;
    fsm_vreg_02_12_22_32.v += x_vreg_0k_1k_2k_3k.v * y_vreg_k2_k2_k2_k2.v;
    fsm_vreg_03_13_23_33.v += x_vreg_0k_1k_2k_3k.v * y_vreg_k3_k3_k3_k3.v;
  }

  fsm[0]       = fsm_vreg_00_10_20_30.d[0];
  fsm[1]       = fsm_vreg_00_10_20_30.d[1];
  fsm[2]       = fsm_vreg_00_10_20_30.d[2];
  fsm[3]       = fsm_vreg_00_10_20_30.d[3];
  fsm[inc]     = fsm_vreg_01_11_21_31.d[0];
  fsm[1+inc]   = fsm_vreg_01_11_21_31.d[1];
  fsm[2+inc]   = fsm_vreg_01_11_21_31.d[2];
  fsm[3+inc]   = fsm_vreg_01_11_21_31.d[3];
  fsm[2*inc]   = fsm_vreg_02_12_22_32.d[0];
  fsm[1+2*inc] = fsm_vreg_02_12_22_32.d[1];
  fsm[2+2*inc] = fsm_vreg_02_12_22_32.d[2];
  fsm[3+2*inc] = fsm_vreg_02_12_22_32.d[3];
  fsm[3*inc]   = fsm_vreg_03_13_23_33.d[0];
  fsm[1+3*inc] = fsm_vreg_03_13_23_33.d[1];
  fsm[2+3*inc] = fsm_vreg_03_13_23_33.d[2];
  fsm[3+3*inc] = fsm_vreg_03_13_23_33.d[3];

}

/* 8 rows of x multiply 1 column of y */
static void mult8x1 (int K, int inc, double *x, double *y, double *erm)
{
/* similar with above, calculate eight rows at each iteration */
  register double erm_reg_0 = 0.0;
  register double erm_reg_1 = 0.0;
  register double erm_reg_2 = 0.0;
  register double erm_reg_3 = 0.0;
  register double erm_reg_4 = 0.0;
  register double erm_reg_5 = 0.0;
  register double erm_reg_6 = 0.0;
  register double erm_reg_7 = 0.0;
  register double y_reg     = 0.0;
  erm_reg_0  = erm[0];
  erm_reg_1  = erm[1];
  erm_reg_2  = erm[2];
  erm_reg_3  = erm[3];
  erm_reg_4  = erm[4];
  erm_reg_5  = erm[5];
  erm_reg_6  = erm[6];
  erm_reg_7  = erm[7];

  for(int k = 0; k < K; ++k)
  {
    y_reg      = y[k];
    erm_reg_0 += y_reg * x[k*inc];
    erm_reg_1 += y_reg * x[1+k*inc];
    erm_reg_2 += y_reg * x[2+k*inc];
    erm_reg_3 += y_reg * x[3+k*inc];
    erm_reg_4 += y_reg * x[4+k*inc];
    erm_reg_5 += y_reg * x[5+k*inc];
    erm_reg_6 += y_reg * x[6+k*inc];
    erm_reg_7 += y_reg * x[7+k*inc];
  }

  erm[0] = erm_reg_0;
  erm[1] = erm_reg_1;
  erm[2] = erm_reg_2;
  erm[3] = erm_reg_3;
  erm[4] = erm_reg_4;
  erm[5] = erm_reg_5;
  erm[6] = erm_reg_6;
  erm[7] = erm_reg_7;
}

/* 1 row of x multiplies 8 columns of y */
static void mult1x8 (int K, int inc, double *x, double *y, double *ecm)
{
  register double ecm_reg_0 = 0.0;
  register double ecm_reg_1 = 0.0;
  register double ecm_reg_2 = 0.0;
  register double ecm_reg_3 = 0.0;
  register double ecm_reg_4 = 0.0;
  register double ecm_reg_5 = 0.0;
  register double ecm_reg_6 = 0.0;
  register double ecm_reg_7 = 0.0;
  register double x_reg     = 0.0;
  ecm_reg_0  = ecm[0];
  ecm_reg_1  = ecm[inc];
  ecm_reg_2  = ecm[2*inc];
  ecm_reg_3  = ecm[3*inc];
  ecm_reg_4  = ecm[4*inc];
  ecm_reg_5  = ecm[5*inc];
  ecm_reg_6  = ecm[6*inc];
  ecm_reg_7  = ecm[7*inc];
  double *x_ptr  = &x[0];
  double *y0_ptr = &y[0];
  double *y1_ptr = &y[inc];
  double *y2_ptr = &y[2*inc];
  double *y3_ptr = &y[3*inc];
  double *y4_ptr = &y[4*inc];
  double *y5_ptr = &y[5*inc];
  double *y6_ptr = &y[6*inc];
  double *y7_ptr = &y[7*inc];

  for(int k = 0; k < K; ++k)
  {
    x_reg      = *x_ptr;
    x_ptr     += inc;
    ecm_reg_0 += x_reg * *y0_ptr++;
    ecm_reg_1 += x_reg * *y1_ptr++;
    ecm_reg_2 += x_reg * *y2_ptr++;
    ecm_reg_3 += x_reg * *y3_ptr++;
    ecm_reg_4 += x_reg * *y4_ptr++;
    ecm_reg_5 += x_reg * *y5_ptr++;
    ecm_reg_6 += x_reg * *y6_ptr++;
    ecm_reg_7 += x_reg * *y7_ptr++;
  }

  ecm[0]     = ecm_reg_0;
  ecm[inc]   = ecm_reg_1;
  ecm[2*inc] = ecm_reg_2;
  ecm[3*inc] = ecm_reg_3;
  ecm[4*inc] = ecm_reg_4;
  ecm[5*inc] = ecm_reg_5;
  ecm[6*inc] = ecm_reg_6;
  ecm[7*inc] = ecm_reg_7;
}

static void cmem1xk_a (int K, int inc, double* a, double* ca)
{
  for(int j = 0; j < K; ++j){  /* loop over columns of A */
    double 
      *a_ptr = &a[j*inc];

    *ca++ = *a_ptr;
    *ca++ = 0;
    *ca++ = 0;
    *ca++ = 0;
  }
}

static void cmem3xk_a (int K, int inc, double* a, double* ca)
{
  for(int j = 0; j < K; ++j){  /* loop over columns of A */
    double 
      *a_ptr = &a[j*inc];

    *ca++ = *a_ptr;
    *ca++ = *(a_ptr+1);
    *ca++ = *(a_ptr+2);
    *ca++ = 0;
  }
}


static void cmem4xk_a (int K, int inc, double* a, double* ca)
{
  for(int j = 0; j < K; ++j){  /* loop over columns of A */
    double 
      *a_ptr = &a[j*inc];

    *ca++ = *a_ptr;
    *ca++ = *(a_ptr+1);
    *ca++ = *(a_ptr+2);
    *ca++ = *(a_ptr+3);
  }
}

static void cmemkx4_b (int K, int inc, double* b, double* cb)
{
  double *b0_ptr = &b[0],     *b1_ptr = &b[inc],
         *b2_ptr = &b[2*inc], *b3_ptr = &b[3*inc];
  for(int i = 0; i < K; ++i){  /* loop over rows of B */
    *cb++ = *b0_ptr++;
    *cb++ = *b1_ptr++;
    *cb++ = *b2_ptr++;
    *cb++ = *b3_ptr++;
  }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int Mr, int Nr, int Kr, double* A, double* B, double* C)
{

  /* Mr, Nr, Kr are the raw number of range {1, 31, 33, 63, 65, 95, 97, 127, 129, 255} since we set BLOCK_SIZE as 256 */
//  printf("Mr=%d, Nr=%d, Kr=%d\n", Mr, Nr, Kr);
  int M = Mr/4*4;
  int m = Mr%4; // m is 1 or 3
  int N = Nr/4*4;
  int n = Nr%4; // n is 1 or 3
  int K = Kr; // K in mult4x4 increases by 1 each time, therefore K does not need to be aligned by 4 elements. But we do need to consider the alignment if unroll the loop in mult4x4.
//  printf("M=%d,N=%d,K=%d\n",M,N,K);
  int Mc;
  int Nc;
  if(!m) {
    Mc = M;
  } else {
    Mc = M+4;
  }
  if(!n) {
    Nc = N;
  } else {
    Nc = N+4;
  }
  double ConA[Mc*Kr];
  double ConB[Kr*Nc];
  /* For 4 columns j of B */
  for (int j = 0; j < N; j+=4)
  {
    cmemkx4_b(K, lda, B+j*lda, &ConB[K*j]);
    /* For 4 rows i of A */ 
    for (int i = 0; i < M; i+=4) 
    {
      /* Compute C(i,j) */
//      printf("i=%d,j=%d\n",i,j);
      if(j == 0) cmem4xk_a(K, lda, A+i, &ConA[i*K]);
      mult4x4_256(K, lda, &ConA[i*K], &ConB[K*j], C+i+j*lda);
    }
    if (m == 1) {
      cmem1xk_a(K, lda, A+Mr-1, &ConA[(Mc-4)*K]);
      mult1x4_256(K, lda, &ConA[(Mc-4)*K], &ConB[K*j], C+Mr-1+j*lda);
    } else if (m == 3) {
      cmem3xk_a(K, lda, A+Mr-3, &ConA[(Mc-4)*K]);
      mult3x4_256(K, lda, &ConA[(Mc-4)*K], &ConB[K*j], C+Mr-3+j*lda);
    } else {}
  }
  if (n == 1) {
    for (int i = 0; i < Mr; ++i)
      mult1x1_256(K, lda, &A[i], &B[(Nr-1)*lda], C+i+(Nr-1)*lda);
  } else if (n == 3) {
    for (int i = 0; i < Mr; ++i)
    {
      mult1x1_256(K, lda, &A[i], &B[(Nr-1)*lda], C+i+(Nr-1)*lda);
      mult1x1_256(K, lda, &A[i], &B[(Nr-2)*lda], C+i+(Nr-2)*lda);
      mult1x1_256(K, lda, &A[i], &B[(Nr-3)*lda], C+i+(Nr-3)*lda);
    }
  } else {}

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
