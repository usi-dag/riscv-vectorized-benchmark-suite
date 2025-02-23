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
#include "somier_v.h"

#ifdef USE_RISCV_VECTOR
#include "../common/vector_defines.h"
#endif

static int count = 0;

//reference scalar code
void force_contr_prevec(int n, double (*X)[n][n][n], double (*F)[n][n][n], int i, int j, int neig_i, int neig_j) {
    double dx, dy, dz, dl, spring_F, FX, FY, FZ;

    for (int k = 1; k < n - 1; k++) {

        dx = X[0][neig_i][neig_j][k] - X[0][i][j][k];
        dy = X[1][neig_i][neig_j][k] - X[1][i][j][k];
        dz = X[2][neig_i][neig_j][k] - X[2][i][j][k];
        dl = sqrt(dx * dx + dy * dy + dz * dz);
        spring_F = 0.25 * spring_K * (dl - 1);
        FX = spring_F * dx / dl;
        FY = spring_F * dy / dl;
        FZ = spring_F * dz / dl;
        F[0][i][j][k] += FX;
        F[1][i][j][k] += FY;
        F[2][i][j][k] += FZ;
    }
}

double buffer[1024];

inline void force_contr_vec(int n, double (*X)[n][n][n], double (*F)[n][n][n], int i, int j, int neig_i, int neig_j);

void force_contr_vec(int n, double (*X)[n][n][n], double (*F)[n][n][n], int i, int j, int neig_i, int neig_j) {
//   unsigned long gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
    int limit = loop_bound(SPECIES_512, n - 2);

    _MMR_f64 v_1 = _MM_SET_f64(1.0);
    _MMR_f64 v_spr_K = _MM_SET_f64(0.25 * spring_K);

    for (int k = 1; k < limit;) {

//      gvl = __builtin_epi_vsetvl(n-1 - k, __epi_e64, __epi_m1);

        _MMR_f64 v_x1 = _MM_LOAD_f64(&X[0][neig_i][neig_j][k]);
        _MMR_f64 v_x2 = _MM_LOAD_f64(&X[0][i][j][k]);
        _MMR_f64 v_dx = _MM_SUB_f64(v_x1, v_x2);
        _MMR_f64 v_dx2 = _MM_MUL_f64(v_dx, v_dx);
        _MMR_f64 v_x3 = _MM_LOAD_f64(&X[1][neig_i][neig_j][k]);
        _MMR_f64 v_x4 = _MM_LOAD_f64(&X[1][i][j][k]);
        _MMR_f64 v_dy = _MM_SUB_f64(v_x3, v_x4);
        _MMR_f64 v_dy2 = _MM_MUL_f64(v_dy, v_dy);
        _MMR_f64 v_x5 = _MM_LOAD_f64(&X[2][neig_i][neig_j][k]);
        _MMR_f64 v_x6 = _MM_LOAD_f64(&X[2][i][j][k]);
        _MMR_f64 v_dz = _MM_SUB_f64(v_x5, v_x6);
        _MMR_f64 v_dz2 = _MM_MUL_f64(v_dz, v_dz);
        _MMR_f64 v_dl = _MM_ADD_f64(v_dx2, v_dy2);
        v_dl = _MM_ADD_f64(v_dl, v_dz2);
        v_dl = _MM_SQRT_f64(v_dl);
        _MMR_f64 v_dl1 = _MM_SUB_f64(v_dl, v_1);
        _MMR_f64 v_spr_F = _MM_MUL_f64(v_spr_K, v_dl1);
        _MMR_f64 v_dFX = _MM_DIV_f64(v_dx, v_dl);
        _MMR_f64 v_dFY = _MM_DIV_f64(v_dy, v_dl);
        _MMR_f64 v_dFZ = _MM_DIV_f64(v_dz, v_dl);
        _MMR_f64 v_FX = _MM_LOAD_f64(&F[0][i][j][k]);
        _MMR_f64 v_FY = _MM_LOAD_f64(&F[1][i][j][k]);
        _MMR_f64 v_FZ = _MM_LOAD_f64(&F[2][i][j][k]);
        v_FX = _MM_MACC_f64(v_FX, v_spr_F, v_dFX);
        v_FY = _MM_MACC_f64(v_FY, v_spr_F, v_dFY);
        v_FZ = _MM_MACC_f64(v_FZ, v_spr_F, v_dFZ);
        _MM_STORE_f64(&F[0][i][j][k], v_FX);
        _MM_STORE_f64(&F[1][i][j][k], v_FY);
        _MM_STORE_f64(&F[2][i][j][k], v_FZ);

        k += SPECIES_512;
    }
}

// TODO already bug in the riscv version
void k_force_contr_vec(int n, double (*X)[n][n][n], double (*F)[n][n][n], int i, int j) {
    long gvl = __builtin_epi_vsetvl(n, __epi_e64, __epi_m1);
    int limit = loop_bound(SPECIES_512, n - 2);

    _MMR_f64 v_1 = _MM_SET_f64(1.0);
    _MMR_f64 v_spr_K = _MM_SET_f64(0.25 * spring_K);

    for (int k = 1; k < limit;) {

//      gvl = __builtin_epi_vsetvl(n-1 - k, __epi_e64, __epi_m1);

        // TODO still missing: elements to shift at the boundaries !!!!!

        _MMR_f64 v_x = _MM_LOAD_f64(&X[0][i][j][k]);
        _MMR_f64 v_y = _MM_LOAD_f64(&X[1][i][j][k]);
        _MMR_f64 v_z = _MM_LOAD_f64(&X[2][i][j][k]);
        _MMR_f64 v_FX = _MM_LOAD_f64(&F[0][i][j][k]);
        _MMR_f64 v_FY = _MM_LOAD_f64(&F[1][i][j][k]);
        _MMR_f64 v_FZ = _MM_LOAD_f64(&F[2][i][j][k]);

        _MMR_f64 v_xs = _MM_LOAD_f64(&X[0][i][j][k - 1]);// _MM_VSLIDEDOWN_f64(v_x, 1) ;
        _MMR_f64 v_ys = _MM_LOAD_f64(&X[1][i][j][k - 1]);// _MM_VSLIDEDOWN_f64(v_y, 1) ;
        _MMR_f64 v_zs = _MM_LOAD_f64(&X[2][i][j][k - 1]);// _MM_VSLIDEDOWN_f64(v_z, 1) ;
        _MMR_f64 v_dx = _MM_SUB_f64(v_xs, v_x);
        _MMR_f64 v_dx2 = _MM_MUL_f64(v_dx, v_dx);
        _MMR_f64 v_dy = _MM_SUB_f64(v_ys, v_y);
        _MMR_f64 v_dy2 = _MM_MUL_f64(v_dy, v_dy);
        _MMR_f64 v_dz = _MM_SUB_f64(v_zs, v_z);
        _MMR_f64 v_dz2 = _MM_MUL_f64(v_dz, v_dz);
        _MMR_f64 v_dl = _MM_ADD_f64(v_dx2, v_dy2);
        v_dl = _MM_ADD_f64(v_dl, v_dz2);
        v_dl = _MM_SQRT_f64(v_dl);
        _MMR_f64 v_dl1 = _MM_SUB_f64(v_dl, v_1);
        _MMR_f64 v_spr_F = _MM_MUL_f64(v_spr_K, v_dl1);
        _MMR_f64 v_dFX = _MM_DIV_f64(v_dx, v_dl);
        _MMR_f64 v_dFY = _MM_DIV_f64(v_dy, v_dl);
        _MMR_f64 v_dFZ = _MM_DIV_f64(v_dz, v_dl);
        v_FX = _MM_MACC_f64(v_FX, v_spr_F, v_dFX);
        v_FY = _MM_MACC_f64(v_FY, v_spr_F, v_dFY);
        v_FZ = _MM_MACC_f64(v_FZ, v_spr_F, v_dFZ);

        v_xs = _MM_LOAD_f64(&X[0][i][j][k + 1]); // _MM_VSLIDEUP_f64(v_x, 1) ;
        v_ys = _MM_LOAD_f64(&X[1][i][j][k + 1]); // _MM_VSLIDEUP_f64(v_y, 1) ;
        v_zs = _MM_LOAD_f64(&X[2][i][j][k + 1]); // _MM_VSLIDEUP_f64(v_z, 1) ;
        v_dx = _MM_SUB_f64(v_xs, v_x);
        v_dx2 = _MM_MUL_f64(v_dx, v_dx);
        v_dy = _MM_SUB_f64(v_ys, v_y);
        v_dy2 = _MM_MUL_f64(v_dy, v_dy);
        v_dz = _MM_SUB_f64(v_zs, v_z);
        v_dz2 = _MM_MUL_f64(v_dz, v_dz);
        v_dl = _MM_ADD_f64(v_dx2, v_dy2);
        v_dl = _MM_ADD_f64(v_dl, v_dz2);
        v_dl = _MM_SQRT_f64(v_dl);
        v_dl1 = _MM_SUB_f64(v_dl, v_1);
        v_spr_F = _MM_MUL_f64(v_spr_K, v_dl1);
        v_dFX = _MM_DIV_f64(v_dx, v_dl);
        v_dFY = _MM_DIV_f64(v_dy, v_dl);
        v_dFZ = _MM_DIV_f64(v_dz, v_dl);
        v_FX = _MM_MACC_f64(v_FX, v_spr_F, v_dFX, gvl);
        v_FY = _MM_MACC_f64(v_FY, v_spr_F, v_dFY, gvl);
        v_FZ = _MM_MACC_f64(v_FZ, v_spr_F, v_dFZ, gvl);


        _MM_STORE_f64(&F[0][i][j][k], v_FX);
        _MM_STORE_f64(&F[1][i][j][k], v_FY);
        _MM_STORE_f64(&F[2][i][j][k], v_FZ);

        k += SPECIES_512;
    }
}


void k_force_contr_prevec(int n, double (*X)[n][n][n], double (*F)[n][n][n], int i, int j) {
    double dx, dy, dz, dl, spring_F, FX, FY, FZ;

    for (int k = 1; k < n - 1; k++) {
        dx = X[0][i][j][k - 1] - X[0][i][j][k];
        dy = X[1][i][j][k - 1] - X[1][i][j][k];
        dz = X[2][i][j][k - 1] - X[2][i][j][k];
        dl = sqrt(dx * dx + dy * dy + dz * dz);
        spring_F = 0.25 * spring_K * (dl - 1);
        FX = spring_F * dx / dl;
        FY = spring_F * dy / dl;
        FZ = spring_F * dz / dl;
        F[0][i][j][k] += FX;
        F[1][i][j][k] += FY;
        F[2][i][j][k] += FZ;
        dx = X[0][i][j][k + 1] - X[0][i][j][k];
        dy = X[1][i][j][k + 1] - X[1][i][j][k];
        dz = X[2][i][j][k + 1] - X[2][i][j][k];
        dl = sqrt(dx * dx + dy * dy + dz * dz);
        spring_F = 0.25 * spring_K * (dl - 1);
        FX = spring_F * dx / dl;
        FY = spring_F * dy / dl;
        FZ = spring_F * dz / dl;
        F[0][i][j][k] += FX;
        F[1][i][j][k] += FY;
        F[2][i][j][k] += FZ;
    }

}

__attribute__((noinline)) void emit_event() {
    __asm__("vadd.vi v0,v0,0");
}


void compute_forces_prevec(int n, double (*X)[n][n][n], double (*F)[n][n][n]) {
    int limit = loop_bound(SPECIES_512, n);
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            force_contr_vec(n, X, F, i, j, i, j + 1);
            force_contr_vec(n, X, F, i, j, i - 1, j);
            force_contr_vec(n, X, F, i, j, i + 1, j);
            force_contr_vec(n, X, F, i, j, i, j - 1);
//            force_contr_prevec (n, X, F, i, j, i,   j-1);   //fails if force_contr_vec
            k_force_contr_prevec(n, X, F, i, j);

            for (int k = limit; k < n - 1; k++) {
                force_contribution (n, X, F, i, j, k, i-1, j,   k);
                force_contribution (n, X, F, i, j, k, i+1, j,   k);
                force_contribution (n, X, F, i, j, k, i,   j-1, k);
                force_contribution (n, X, F, i, j, k, i,   j+1, k);
                force_contribution (n, X, F, i, j, k, i,   j,   k-1);
                force_contribution (n, X, F, i, j, k, i,   j,   k+1);
            }
        }
    }
}