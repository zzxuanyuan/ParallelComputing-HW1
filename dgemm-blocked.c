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

#define X(k) x[(k)*incx]

static void single_mult (int K, double *x, int incx, double *y, double *sm)
{
/* sm represents single multiplication
 *  sm = x * y + sm
 *  x increases by incx which is the stride size, and y increases by 1
 */
  for(int k = 0; k < K; ++k) {
    *sm += X(k) * y[k];
  }
}

static void fourrow_mult (int K, double *x, int incx, double *y, double *frm)
{
/* frm represents four row multiplication
 *  frm[0] = x[0] * y + frm[0]
 *  frm[1] = x[1] * y + frm[1]
 *  frm[2] = x[2] * y + frm[2]
 *  frm[3] = x[3] * y + frm[3]
 *  each time calculate four rows within the same columns of x
 */
  single_mult(K, x,   incx, y, frm  );
  single_mult(K, x+1, incx, y, frm+1);
  single_mult(K, x+2, incx, y, frm+2);
  single_mult(K, x+3, incx, y, frm+3);
}

static void eightrow_mult (int K, double *x, int incx, double *y, double *erm)
{
/* similar with above, calculate eight rows at each iteration */
  single_mult(K, x,   incx, y, erm  );
  single_mult(K, x+1, incx, y, erm+1);
  single_mult(K, x+2, incx, y, erm+2);
  single_mult(K, x+3, incx, y, erm+3);
  single_mult(K, x+4, incx, y, erm+4);
  single_mult(K, x+5, incx, y, erm+5);
  single_mult(K, x+6, incx, y, erm+6);
  single_mult(K, x+7, incx, y, erm+7);
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i+=8)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
//      fourrow_mult(K, A+i, lda, B+j*lda, C+i+j*lda);
      eightrow_mult(K, A+i, lda, B+j*lda, C+i+j*lda);
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
