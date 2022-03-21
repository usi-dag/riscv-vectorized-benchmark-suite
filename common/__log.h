//
// RISC-V VECTOR LOG FUNCTION Version by Cristóbal Ramírez Lazo, "Barcelona 2019"
// This RISC-V Vector implementation is based on the original code presented by Julien Pommier

/* 
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

//void print_num(_MMR_f64 var) {
//    double val[8];
//    memcpy(val, &var, sizeof(val));
//    printf("Numerical: %f %f %f %f %f %f %f %f \n",
//           val[0], val[1], val[2], val[3], val[4], val[5],
//           val[6], val[7]);
//}
//
//void print_num_int(_MMR_i64 var) {
//    int val[16];
//    memcpy(val, &var, sizeof(val));
//    printf("Numerical int: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i \n",
//           val[0], val[1], val[2], val[3], val[4], val[5],
//           val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13],
//           val[14], val[15]);
//}

inline _MMR_f64 __log_1xf64(_MMR_f64 x) {

    _MMR_i64 _x_i;
    _MMR_i64 imm0;
    _MMR_f64 e;
    _MMR_MASK_i64 invalid_mask = _MM_VFLE_f64(x, _MM_SET_f64(0.0f));
    x = _MM_MAX_f64(x, (_MMR_f64) _MM_SET_i64(0x0010000000000000));  /* cut off denormalized stuff */
    imm0 = _MM_SRL_i64((_MMR_i64) x, _MM_SET_i64(52));
    /* keep only the fractional part */
    _x_i = _MM_AND_i64((_MMR_i64) x, _MM_SET_i64(~0x7ff0000000000000));
    _x_i = _MM_OR_i64(_x_i, (_MMR_i64) _MM_SET_f64(0.5f));
    x = (_MMR_f64) _x_i;
    imm0 = _MM_SUB_i64(imm0, _MM_SET_i64(1023)); // TODO if value negative then casting gives nan
//    printf("imm0 ");
//    print_num_int(imm0);
//  e = _MM_VFCVT_F_X_f64(imm0); // TODO BUG returns all -nan
//  e = (_MMR_f64) imm0;
    int val[16];
    memcpy(val, &imm0, sizeof(val));
//    e = _mm512_set_pd(val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
    e = _MM_LOAD_f64(&val);
//    printf("e ");
//    print_num(e);
    e = _MM_ADD_f64(e, _MM_SET_f64(1.0f));

    _MMR_MASK_i64 mask = _MM_VFLT_f64(x, _MM_SET_f64(0.707106781186547524));
    _MMR_f64 tmp = _MM_MERGE_f64(mask, _MM_SET_f64(0.0f), x); // inverted position of mask

    x = _MM_SUB_f64(x, _MM_SET_f64(1.0f));
    e = _MM_SUB_f64(e, _MM_MERGE_f64(mask, _MM_SET_f64(0.0f), _MM_SET_f64(1.0f))); // inverted position of mask
    x = _MM_ADD_f64(x, tmp);

    _MMR_f64 z = _MM_MUL_f64(x, x);
    _MMR_f64 y;

    y = _MM_MADD_f64(_MM_SET_f64(7.0376836292E-2), x, _MM_SET_f64(-1.1514610310E-1));
    y = _MM_MADD_f64(y, x, _MM_SET_f64(1.1676998740E-1));
    y = _MM_MADD_f64(y, x, _MM_SET_f64(-1.2420140846E-1));
    y = _MM_MADD_f64(y, x, _MM_SET_f64(1.4249322787E-1));
    y = _MM_MADD_f64(y, x, _MM_SET_f64(-1.6668057665E-1));
    y = _MM_MADD_f64(y, x, _MM_SET_f64(2.0000714765E-1));
    y = _MM_MADD_f64(y, x, _MM_SET_f64(-2.4999993993E-1));
    y = _MM_MADD_f64(y, x, _MM_SET_f64(3.3333331174E-1));
    y = _MM_MUL_f64(y, z);

    y = _MM_MACC_f64(y, e, _MM_SET_f64(-2.12194440e-4));
    tmp = _MM_MUL_f64(z, _MM_SET_f64(0.5f));
    y = _MM_SUB_f64(y, tmp);
    tmp = _MM_MUL_f64(e, _MM_SET_f64(0.693359375));
    x = _MM_ADD_f64(x, y);
    x = _MM_ADD_f64(x, tmp);
    x = _MM_MERGE_f64(invalid_mask, x, (_MMR_f64) _MM_SET_i64(0xffffffffffffffff));

    return x;
}

//inline _MMR_f32 __log_2xf32(_MMR_f32 x , unsigned long int gvl) {
//
//_MMR_i32   _x_i;
//_MMR_i32   imm0;
//_MMR_f32  e;
//
//_MMR_MASK_i32 invalid_mask = _MM_VFLE_f32(x,_MM_SET_f32(0.0f,gvl),gvl);
//
//  x = _MM_MAX_f32(x, (_MMR_f32)_MM_SET_i32(0x00800000,gvl));  /* cut off denormalized stuff */
//  imm0 = _MM_SRL_i32((_MMR_i32)x, _MM_SET_i32(23,gvl));
//  /* keep only the fractional part */
//  _x_i = _MM_AND_i32((_MMR_i32)x, _MM_SET_i32(~0x7f800000,gvl));
//  _x_i = _MM_OR_i32(_x_i, (_MMR_i32)_MM_SET_f32(0.5f,gvl));
//  x= (_MMR_f32)_x_i;
//  imm0 = _MM_SUB_i32(imm0 ,_MM_SET_i32(0x7f,gvl) );
//  e = _MM_VFCVT_F_X_f32(imm0,gvl);
//  e = _MM_ADD_f32(e, _MM_SET_f32(1.0f,gvl) ,gvl);
//
//_MMR_MASK_i32 mask = _MM_VFLT_f32(x, _MM_SET_f32(0.707106781186547524,gvl));
//_MMR_f32 tmp  = _MM_MERGE_f32(_MM_SET_f32(0.0f,gvl),x, mask,gvl);
//
//  x = _MM_SUB_f32(x, _MM_SET_f32(1.0f,gvl),gvl);
//  e = _MM_SUB_f32(e, _MM_MERGE_f32(_MM_SET_f32(0.0f,gvl),_MM_SET_f32(1.0f,gvl), mask,gvl),gvl);
//  x = _MM_ADD_f32(x, tmp,gvl);
//
//_MMR_f32 z = _MM_MUL_f32(x,x,gvl);
//_MMR_f32 y;
//
//  y = _MM_MADD_f32(_MM_SET_f32(7.0376836292E-2,gvl),x,_MM_SET_f32(-1.1514610310E-1,gvl),gvl);
//  y = _MM_MADD_f32(y,x,_MM_SET_f32(1.1676998740E-1,gvl),gvl);
//  y = _MM_MADD_f32(y,x,_MM_SET_f32(-1.2420140846E-1,gvl),gvl);
//  y = _MM_MADD_f32(y,x,_MM_SET_f32(1.4249322787E-1,gvl),gvl);
//  y = _MM_MADD_f32(y,x,_MM_SET_f32(-1.6668057665E-1,gvl),gvl);
//  y = _MM_MADD_f32(y,x,_MM_SET_f32(2.0000714765E-1,gvl),gvl);
//  y = _MM_MADD_f32(y,x,_MM_SET_f32(-2.4999993993E-1,gvl),gvl);
//  y = _MM_MADD_f32(y,x,_MM_SET_f32(3.3333331174E-1,gvl),gvl);
//  y = _MM_MUL_f32(y, z,gvl);
//  y = _MM_MACC_f32(y,e,_MM_SET_f32(-2.12194440e-4,gvl),gvl);
//  tmp = _MM_MUL_f32(z, _MM_SET_f32(0.5f,gvl),gvl);
//  y = _MM_SUB_f32(y, tmp,gvl);
//  tmp = _MM_MUL_f32(e, _MM_SET_f32(0.693359375,gvl),gvl);
//  x = _MM_ADD_f32(x, y,gvl);
//  x = _MM_ADD_f32(x, tmp,gvl);
//  x = _MM_MERGE_f32(x,(_MMR_f32)_MM_SET_i32(0xffffffff,gvl), invalid_mask,gvl);
//
//  return x;
//}