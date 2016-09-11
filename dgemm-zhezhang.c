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
#define BLOCK_SIZE 128
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  double c0 = 0.0;
  double c1 = 0.0;
  double c2 = 0.0;
  double c3 = 0.0;
  double c4 = 0.0;
  double c5 = 0.0;
  double c6 = 0.0;
  double c7 = 0.0;

  /* For each column j of B */
  for (int j = 0; j < N; ++j)
    /* For each row j of A */ 
    for (int k = 0; k < K; ++k) 
    {
      for (int i = 0; i < M; i+=8)
      {
        /* Compute C(i,j) */
	c0  = C[i+j*lda];
        c1  = C[i+1+j*lda];
	c2  = C[i+2+j*lda];
        c3  = C[i+3+j*lda];
	c4  = C[i+4+j*lda];
        c5  = C[i+5+j*lda];
	c6  = C[i+6+j*lda];
        c7  = C[i+7+j*lda];
	c0 += A[i+k*lda]   * B[k+j*lda];
        c1 += A[i+1+k*lda] * B[k+j*lda];
	c2 += A[i+2+k*lda] * B[k+j*lda];
        c3 += A[i+3+k*lda] * B[k+j*lda];
	c4 += A[i+4+k*lda] * B[k+j*lda];
        c5 += A[i+5+k*lda] * B[k+j*lda];
	c6 += A[i+6+k*lda] * B[k+j*lda];
        c7 += A[i+7+k*lda] * B[k+j*lda];
	C[i+j*lda]   = c0;
        C[i+1+j*lda] = c1;
	C[i+2+j*lda] = c2;
        C[i+3+j*lda] = c3;
	C[i+4+j*lda] = c4;
        C[i+5+j*lda] = c5;
	C[i+6+j*lda] = c6;
        C[i+7+j*lda] = c7;
      }
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
