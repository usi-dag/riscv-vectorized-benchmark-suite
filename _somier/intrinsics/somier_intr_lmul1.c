/*************************************************************************
* Somier - RISC-V Vectorized version
* Author: Jesus Labarta
* Barcelona Supercomputing Center
*************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <assert.h>
#include "../somier.h"

#ifdef USE_RISCV_VECTOR
#include "../common/vector_defines.h"
#endif

static int count2=0;

void accel_intr(int n, double (*A)[n][n][n], double (*F)[n][n][n], double M)
{
   int i, j, k;
   int limit = loop_bound(SPECIES_512, n);
   _MMR_f64 vF0, vF1, vF2, vA0, vA1, vA2;
   double invM = 1/M;

   for (i = 0; i<n; i++) {
       for (j = 0; j < n; j++) {
           for (k = 0; k < limit;) {
               _MMR_f64 v_invM = _MM_SET_f64(invM);

               vF0 = _MM_LOAD_f64(&F[0][i][j][k]);
               vA0 = _MM_MUL_f64(vF0, v_invM);
               _MM_STORE_f64(&A[0][i][j][k], vA0);

               vF1 = _MM_LOAD_f64(&F[1][i][j][k]);
               vA1 = _MM_MUL_f64(vF1, v_invM);
               _MM_STORE_f64(&A[1][i][j][k], vA1);

               vF2 = _MM_LOAD_f64(&F[2][i][j][k]);
               vA2 = _MM_MUL_f64(vF2, v_invM);
               _MM_STORE_f64(&A[2][i][j][k], vA2);

               k += SPECIES_512;
           }

           for (; k < n; k++) {
               A[0][i][j][k]= F[0][i][j][k]/M;
               A[1][i][j][k]= F[1][i][j][k]/M;
               A[2][i][j][k]= F[2][i][j][k]/M;
           }
       }
   }
}

//#undef COLAPSED
//#define COLAPSED

void vel_intr(int n, double (*V)[n][n][n], double (*A)[n][n][n], double dt) {
    int i, j, k;
    int limit = loop_bound(SPECIES_512, n);
    _MMR_f64 vV0, vV1, vV2, vA0, vA1, vA2;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < limit; k += SPECIES_512) {
                _MMR_f64 vdt = _MM_SET_f64(dt);

                vV0 = _MM_LOAD_f64(&V[0][i][j][k]);
                vA0 = _MM_LOAD_f64(&A[0][i][j][k]);
                vV0 = _MM_MACC_f64(vV0, vA0, vdt);
                _MM_STORE_f64(&V[0][i][j][k], vV0);

                vV1 = _MM_LOAD_f64(&V[1][i][j][k]);
                vA1 = _MM_LOAD_f64(&A[1][i][j][k]);
                vV1 = _MM_MACC_f64(vV1, vA1, vdt);
                _MM_STORE_f64(&V[1][i][j][k], vV1);
                vV2 = _MM_LOAD_f64(&V[2][i][j][k]);
                vA2 = _MM_LOAD_f64(&A[2][i][j][k]);
                vV2 = _MM_MACC_f64(vV2, vA2, vdt);
                _MM_STORE_f64(&V[2][i][j][k], vV2);
            }

            for (; k < n; k++) {
                V[0][i][j][k] += A[0][i][j][k] * dt;
                V[1][i][j][k] += A[1][i][j][k] * dt;
                V[2][i][j][k] += A[2][i][j][k] * dt;
            }


        }
    }
}

void pos_intr(int n, double (*X)[n][n][n], double (*V)[n][n][n], double dt)
{
   int i, j, k;
    int limit = loop_bound(SPECIES_512, n);
   _MMR_f64 vV0, vV1, vV2, vX0, vX1, vX2;


   for (i = 0; i<n; i++) {
      for (j = 0; j<n; j++) {
         for (k = 0; k<limit;) {
            _MMR_f64 vdt = _MM_SET_f64(dt);

            vX0 = _MM_LOAD_f64( &X[0][i][j][k] );
            vV0 = _MM_LOAD_f64( &V[0][i][j][k] );
            vX0 = _MM_MACC_f64(vX0, vV0, vdt);
            _MM_STORE_f64(&X[0][i][j][k], vX0);

            vX1 = _MM_LOAD_f64( &X[1][i][j][k] );
            vV1 = _MM_LOAD_f64( &V[1][i][j][k] );
            vX1 = _MM_MACC_f64(vX1, vV1, vdt);
            _MM_STORE_f64(&X[1][i][j][k], vX1);
            vX2 = _MM_LOAD_f64( &X[2][i][j][k] );
            vV2 = _MM_LOAD_f64( &V[2][i][j][k] );
            vX2 = _MM_MACC_f64(vX2, vV2, vdt);
            _MM_STORE_f64(&X[2][i][j][k], vX2);

            k+=SPECIES_512;
         }

          for (; k < n; k++) {
              X[0][i][j][k] += V[0][i][j][k] * dt;
              X[1][i][j][k] += V[1][i][j][k] * dt;
              X[2][i][j][k] += V[2][i][j][k] * dt;
          }
      }


   }
   // TODO would need to check that possition des not go beyond the box walls
//   FENCE();
}
//      for (i = 0; i<N; i++)
//         for (j = 0; j<N; j++)
//            for (kk = 0; kk<N; kk+=vl) {

//             int maxk = (kk+vl < N? kk+vl: N);
//               for (k = kk; k<maxk; k++) {
//               X[0][i][j][k] += V[0][i][j][k]*dt;
//               X[1][i][j][k] += V[1][i][j][k]*dt;
//               X[2][i][j][k] += V[2][i][j][k]*dt;
//             }
//            }

