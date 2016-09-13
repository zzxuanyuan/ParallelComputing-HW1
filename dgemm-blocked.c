/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
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

/* 4 row of x multiplies 4 columns of y */
static void mult4x4 (int K, int inc, double *x, double *y, double *fsm)
{
  register double fsm_reg_00 = 0.0;
  register double fsm_reg_01 = 0.0;
  register double fsm_reg_02 = 0.0;
  register double fsm_reg_03 = 0.0;
  register double fsm_reg_10 = 0.0;
  register double fsm_reg_11 = 0.0;
  register double fsm_reg_12 = 0.0;
  register double fsm_reg_13 = 0.0;
  register double fsm_reg_20 = 0.0;
  register double fsm_reg_21 = 0.0;
  register double fsm_reg_22 = 0.0;
  register double fsm_reg_23 = 0.0;
  register double fsm_reg_30 = 0.0;
  register double fsm_reg_31 = 0.0;
  register double fsm_reg_32 = 0.0;
  register double fsm_reg_33 = 0.0;
  register double x0_reg     = 0.0;
  register double x1_reg     = 0.0;
  register double x2_reg     = 0.0;
  register double x3_reg     = 0.0;
  fsm_reg_00  = fsm[0];
  fsm_reg_01  = fsm[inc];
  fsm_reg_02  = fsm[2*inc];
  fsm_reg_03  = fsm[3*inc];
  fsm_reg_10  = fsm[1];
  fsm_reg_11  = fsm[1+inc];
  fsm_reg_12  = fsm[1+2*inc];
  fsm_reg_13  = fsm[1+3*inc];
  fsm_reg_20  = fsm[2];
  fsm_reg_21  = fsm[2+inc];
  fsm_reg_22  = fsm[2+2*inc];
  fsm_reg_23  = fsm[2+3*inc];
  fsm_reg_30  = fsm[3];
  fsm_reg_31  = fsm[3+inc];
  fsm_reg_32  = fsm[3+2*inc];
  fsm_reg_33  = fsm[3+3*inc];
  double *x0_ptr = &x[0];
  double *x1_ptr = &x[1];
  double *x2_ptr = &x[2];
  double *x3_ptr = &x[3];
  double *y0_ptr = &y[0];
  double *y1_ptr = &y[inc];
  double *y2_ptr = &y[2*inc];
  double *y3_ptr = &y[3*inc];

  for(int k = 0; k < K; ++k)
  {
    x0_reg      = *x0_ptr;
    x1_reg      = *x1_ptr;
    x2_reg      = *x2_ptr;
    x3_reg      = *x3_ptr;
    x0_ptr     += inc;
    x1_ptr     += inc;
    x2_ptr     += inc;
    x3_ptr     += inc;
    fsm_reg_00 += x0_reg * *y0_ptr;
    fsm_reg_10 += x1_reg * *y0_ptr;
    fsm_reg_20 += x2_reg * *y0_ptr;
    fsm_reg_30 += x3_reg * *y0_ptr;
    fsm_reg_01 += x0_reg * *y1_ptr;
    fsm_reg_11 += x1_reg * *y1_ptr;
    fsm_reg_21 += x2_reg * *y1_ptr;
    fsm_reg_31 += x3_reg * *y1_ptr;
    fsm_reg_02 += x0_reg * *y2_ptr;
    fsm_reg_12 += x1_reg * *y2_ptr;
    fsm_reg_22 += x2_reg * *y2_ptr;
    fsm_reg_32 += x3_reg * *y2_ptr;
    fsm_reg_03 += x0_reg * *y3_ptr;
    fsm_reg_13 += x1_reg * *y3_ptr;
    fsm_reg_23 += x2_reg * *y3_ptr;
    fsm_reg_33 += x3_reg * *y3_ptr;
    y0_ptr     += 1;
    y1_ptr     += 1;
    y2_ptr     += 1;
    y3_ptr     += 1;
  }

  fsm[0]       = fsm_reg_00;
  fsm[inc]     = fsm_reg_01;
  fsm[2*inc]   = fsm_reg_02;
  fsm[3*inc]   = fsm_reg_03;
  fsm[1]       = fsm_reg_10;
  fsm[1+inc]   = fsm_reg_11;
  fsm[1+2*inc] = fsm_reg_12;
  fsm[1+3*inc] = fsm_reg_13;
  fsm[2]       = fsm_reg_20;
  fsm[2+inc]   = fsm_reg_21;
  fsm[2+2*inc] = fsm_reg_22;
  fsm[2+3*inc] = fsm_reg_23;
  fsm[3]       = fsm_reg_30;
  fsm[3+inc]   = fsm_reg_31;
  fsm[3+2*inc] = fsm_reg_32;
  fsm[3+3*inc] = fsm_reg_33;
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

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int j = 0; j < N; j+=4)
    /* For each column j of B */ 
    for (int i = 0; i < M; i+=4) 
    {
      /* Compute C(i,j) */
//      mult1x4(K, lda, A+i, B+j*lda, C+i+j*lda);
      mult4x4(K, lda, A+i, B+j*lda, C+i+j*lda);
//      mult1x8(K, lda, A+i, B+j*lda, C+i+j*lda);
    }
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
