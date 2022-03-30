// RanUnif.c
// Author: Mark Broadie
// Collaborator: Mikhail Smelyanskiy, Intel

/* See "Random Number Generators: Good Ones Are Hard To Find", */
/*     Park & Miller, CACM 31#10 October 1988 pages 1192-1201. */

/*************************************************************************
* RISC-V Vectorized Version
* Author: Cristóbal Ramírez Lazo
* email: cristobal.ramirez@bsc.es
* Barcelona Supercomputing Center (2020)
*************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "HJM_type.h"


FTYPE RanUnif(long *s);

void RanUnif_vector(long *s, int iFactors, int iN, int BLOCKSIZE, FTYPE **randZ);

FTYPE RanUnif(long *s) {
    // uniform random number generator

    long ix, k1;
    FTYPE dRes;

    ix = *s;
    k1 = ix / 127773L;
    ix = 16807L * (ix - k1 * 127773L) - k1 * 2836L;
    if (ix < 0) ix = ix + 2147483647L;
    *s = ix;
    dRes = (ix * 4.656612875e-10);
    return (dRes);

} // end of RanUnif

#ifdef USE_RISCV_VECTOR

void RanUnif_vector( long *s , int iFactors , int iN ,int  BLOCKSIZE , FTYPE **randZ )
{
  // uniform random number generator
//  unsigned long int gvl = __builtin_epi_vsetvl(BLOCKSIZE, __epi_e64, __epi_m1);
    int limit  = loop_bound(SPECIES_512, BLOCKSIZE);
  _MMR_f64    k1;
  _MMR_f64      zero;
  _MMR_MASK_i64   mask1;
  _MMR_f64    dRes;

  _MMR_f64    cons1     = _MM_SET_f64(127773);
  _MMR_f64    cons2     = _MM_SET_f64(16807);
  _MMR_f64    cons3     = _MM_SET_f64(2836);
  _MMR_f64    cons4     = _MM_SET_f64(2147483647);
  _MMR_f64    cons5     = _MM_SET_f64(4.656612875E-10);
  _MMR_f64    xSeed     = _MM_LOAD_f64(s);

  for (int l=0;l<=iFactors-1;++l){
    for (int j=1;j<=iN-1;++j){
        for (int b = 0; b < limit; b+=SPECIES_512) {
            k1    = _MM_DIV_f64(xSeed,cons1);
            xSeed   = _MM_SUB_f64(_MM_MUL_f64(cons2,_MM_SUB_f64(xSeed,_MM_MUL_f64(k1,cons1))), _MM_MUL_f64(k1,cons3));
            zero    = _MM_SET_f64(0);
            mask1   = _MM_VFLE_f64(xSeed,zero);
            xSeed     = _MM_ADD_f64_MASK(xSeed,mask1,xSeed, cons4);
            dRes    = _MM_MUL_f64( cons5,xSeed);

            _MM_STORE_f64(&randZ[l][BLOCKSIZE*j+b], dRes);
        }

      }
  }
}

#endif