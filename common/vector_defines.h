#ifndef VECTOR_DEFINITION
#define VECTOR_DEFINITION

#include <math.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <string.h>


#define _MM_ALIGN64 __attribute__((aligned (64)))

#define MUSTINLINE __attribute__((always_inline))

#define INT32_SPECIES_512 16
#define INT32_SPECIES_256 8
#define INT32_SPECIES_128 4
#define INT32_SPECIES_64 2

#define SPECIES_512 8 // species for double
#define SPECIES_256 4 // species for double
#define SPECIES_128 2 // species for double
#define SPECIES_64 1 // species for double
//---------------------------------------------------------------------------
// UTILITY FUNCTION
inline int loop_bound(int species_size, int length) {
    return floor(abs(length)/species_size) * species_size;
}

//---------------------------------------------------------------------------
// DATA TYPES

//#define _MMR_f64        	__epi_1xf64
#define _MMR_f64        	__m512d
#define _MMR_f32        	__m512

//TODO #define _MMR_2xf64			__epi_2xf64
//TODO #define _MMR_4xf64			__epi_4xf32

#define _MMR_i64        	__m512i  // TODO is there a float vector???
#define _MMR_i32        	__m512i


//---------------------------------------------------------------------------
// INTEGER INTRINSICS

#define _MM_LOAD_i64    	_mm512_loadu_si512 //__builtin_epi_vload_1xi64
#define _MM_LOAD_i32    	_mm512_loadu_si512
//TODO #define _MM_LOAD_i16        __builtin_epi_vload_4xi16
//TODO #define _MM_LOAD_i8        __builtin_epi_vload_8xi8

//TODO #define _MM_LOAD_INDEX_i64 __builtin_epi_vload_indexed_1xi64
//TODO #define _MM_LOAD_INDEX_i32 __builtin_epi_vload_indexed_2xi32
//TODO
//TODO #define _MM_LOAD_STRIDE_i64 __builtin_epi_vload_strided_1xi64
//TODO #define _MM_LOAD_STRIDE_i32 __builtin_epi_vload_strided_2xi32

#define _MM_STORE_i64   	_mm512_storeu_pd
#define _MM_STORE_i32   	_mm512_storeu_si512
//TODO #define _MM_STORE_i16       __builtin_epi_vstore_4xi16
//TODO #define _MM_STORE_i8       __builtin_epi_vstore_8xi8

//TODO #define _MM_STORE_INDEX_i64 __builtin_epi_vstore_indexed_1xi64
//TODO #define _MM_STORE_INDEX_i32 __builtin_epi_vstore_indexed_2xi32

//TODO #define _MM_STORE_STRIDE_i64 __builtin_epi_vstore_strided_1xi64
//TODO #define _MM_STORE_STRIDE_i32 __builtin_epi_vstore_strided_2xi32

#define _MM_ADD_i64     	_mm512_add_epi64 // __builtin_epi_vadd_1xi64
#define _MM_ADD_i32     	_mm512_add_epi32

#define _MM_SUB_i64			  _mm512_sub_epi64 // __builtin_epi_vsub_1xi64
#define _MM_SUB_i32			  _mm512_add_epi32

#define _MM_ADD_i64_MASK  // _mm512_mask_add_pd __builtin_epi_vadd_1xi64_mask
//TODO #define _MM_ADD_i32_MASK  __builtin_epi_vadd_2xi32_mask

// #define _MM_MUL_i64       _mm512_mullo_epi64 // __builtin_epi_vmul_1xi64 // TODO no mul for epi64
//TODO #define _MM_MUL_i32       __builtin_epi_vmul_2xi32
//TODO #define _MM_MUL_i16       __builtin_epi_vmul_4xi16
//TODO #define _MM_MUL_i8       __builtin_epi_vmul_8xi8

// #define _MM_DIV_i64       _mm512_div_epi64 // __builtin_epi_vdiv_1xi64
//TODO #define _MM_DIV_i32       __builtin_epi_vdiv_2xi32

//TODO #define _MM_REM_i64       __builtin_epi_vrem_1xi64
//TODO #define _MM_REM_i32       __builtin_epi_vrem_2xi32

#define _MM_SET_i64     	_mm512_set1_epi64
#define _MM_SET_i32     	_mm512_set1_epi32

//TODO #define _MM_MIN_i64         __builtin_epi_vmin_1xi64
#define _MM_MIN_i32         _mm512_min_epi32

//TODO #define _MM_MAX_i64         __builtin_epi_vmax_1xi64
//TODO #define _MM_MAX_i32         __builtin_epi_vmax_2xi32

#define _MM_SLL_i64     	_mm512_sllv_epi64 // __builtin_epi_vsll_1xi64
#define _MM_SLL_i32     	_mm512_sllv_epi32

#define _MM_SRL_i64     	_mm512_srav_epi64 // __builtin_epi_vsrl_1xi64
//TODO #define _MM_SRL_i32     	__builtin_epi_vsrl_2xi32

#define _MM_AND_i64     	_mm512_and_si512 //__builtin_epi_vand_1xi64
//TODO #define _MM_AND_i32     	__builtin_epi_vand_2xi32

#define _MM_OR_i64     		_mm512_or_si512 //__builtin_epi_vor_1xi64
//TODO #define _MM_OR_i32     		__builtin_epi_vor_2xi32

#define _MM_XOR_i64     	_mm512_xor_si512 // __builtin_epi_vxor_1xi64
//TODO #define _MM_XOR_i32     	__builtin_epi_vxor_2xi32

//TODO #define _MM_NOT_i64(x)     	_MM_XOR_i64((x),(x), gvl)
//TODO #define _MM_NOT_i32(x)     	_MM_XOR_i32((x),(x), gvl)

//TODO #define _MM_REDSUM_i64   	__builtin_epi_vredsum_1xi64
#define _MM_REDSUM_i32   	_mm512_reduce_add_epi32
#define _MM_MASK_REDSUM_i32 _mm512_mask_reduce_add_epi32
//TODO #define _MM_REDSUM_i16      __builtin_epi_vredsum_4xi16
//TODO #define _MM_REDSUM_i8      __builtin_epi_vredsum_8xi8

#define _MM_MERGE_i64  		_mm512_mask_blend_epi64 // __builtin_epi_vmerge_1xi64
//TODO #define _MM_MERGE_i32  		__builtin_epi_vmerge_2xi32

#define _MM_ABS_i32 _mm512_abs_epi32

#define _MM_LSHIFT_i32 _mm512_slli_epi32

//---------------------------------------------------------------------------
// FLOATING POINT INTRINSICS


#define _MM_LOAD_f64    	_mm512_loadu_pd
#define _MM_LOAD_f32    	_mm512_loadu_ps

#define __MM_MASK_LOAD_f32 _mm512_mask_load_ps

//TODO #define _MM_LOAD_INDEX_f64 __builtin_epi_vload_indexed_1xf64
//TODO #define _MM_LOAD_INDEX_f32 __builtin_epi_vload_indexed_2xf32

//TODO #define _MM_LOAD_STRIDE_f64 __builtin_epi_vload_strided_1xf64
//TODO #define _MM_LOAD_STRIDE_f32 __builtin_epi_vload_strided_2xf32


#define _MM_STORE_f64   	_mm512_storeu_pd // __builtin_epi_vstore_1xf64
#define _MM_STORE_f32   	_mm512_storeu_ps

//TODO #define _MM_STORE_INDEX_f64 __builtin_epi_vstore_indexed_1xf64
//TODO #define _MM_STORE_INDEX_f32 __builtin_epi_vstore_indexed_2xf32

//TODO #define _MM_STORE_STRIDE_f64 __builtin_epi_vstore_strided_1xf64
//TODO #define _MM_STORE_STRIDE_f32 __builtin_epi_vstore_strided_2xf32

#define _MM_MUL_f64     	_mm512_mul_pd
#define _MM_MUL_f32     	_mm512_mul_ps

//TODO #define _MM_MUL_f32     	__builtin_epi_vfmul_2xf32
//TODO #define _MM_MUL_f32     	__builtin_epi_vfmul_2xf32
#define _MM_ADD_f64     	_mm512_add_pd
#define _MM_ADD_f32     	_mm512_add_ps

#define _MM_SUB_f64     	_mm512_sub_pd // __builtin_epi_vfsub_1xf64
#define _MM_SUB_f32     	_mm512_sub_ps

#define _MM_SUB_f64_MASK	_mm512_mask_sub_pd // __builtin_epi_vfsub_1xf64_mask
#define _MM_SUB_f32_MASK	_mm512_mask_sub_ps

#define _MM_ADD_f64_MASK  _mm512_mask_add_pd
//TODO #define _MM_ADD_f32_MASK  __builtin_epi_vfadd_2xf32_mask

#define _MM_DIV_f64     	_mm512_div_pd // __builtin_epi_vfdiv_1xf64
#define _MM_DIV_f32     	_mm512_div_ps

//TODO #define _MM_DIV_2xf64		__builtin_epi_vfdiv_2xf64
//TODO #define _MM_DIV_4xf32     	__builtin_epi_vfdiv_4xf32

#define _MM_SQRT_f64    	_mm512_sqrt_pd
#define _MM_SQRT_f32    	_mm512_sqrt_ps

#define _MM_SET_f64     	_mm512_set1_pd // __builtin_epi_vbroadcast_1xf64
#define _MM_SET_f32     	_mm512_set1_ps

#define _MM_MIN_f64         _mm512_min_pd // __builtin_epi_vfmin_1xf64
#define _MM_MIN_f32         _mm512_min_ps

#define _MM_MAX_f64         _mm512_max_pd
#define _MM_MAX_f32         _mm512_max_ps

//TODO #define _MM_VFSGNJ_f64      __builtin_epi_vfsgnj_1xf64
//TODO #define _MM_VFSGNJ_f32      __builtin_epi_vfsgnj_2xf32

// TODO search if there is a better option
#define _MM_VFSGNJN_f64     _mm512_neg_pd // __builtin_epi_vfsgnjn_1xf64 // negate array
#define _MM_VFSGNJN_f32     _mm512_fnmadd_ps

inline _MMR_f64  _mm512_neg_pd(_MMR_f64 a) {
    return _mm512_fnmadd_pd(a, _MM_SET_f64(1.0), _MM_SET_f64(0.0));
}

#define _MM_VFSGNJX_f64    _mm512_abs_pd // __builtin_epi_vfsgnjx_1xf64
#define _MM_VFSGNJX_f32 	_mm512_abs_ps

#define _MM_MERGE_f64  		_mm512_mask_blend_pd // __builtin_epi_vfmerge_1xf64
#define _MM_MERGE_f32 		_mm512_mask_blend_ps

#define _MM_REDSUM_f64  	_mm512_reduce_add_pd // __builtin_epi_vfredsum_1xf64
#define _MM_REDSUM_f32  	_mm512_reduce_add_ps // __builtin_epi_vfredsum_2xf32

//TODO #define _MM_REDSUM_f64_MASK __builtin_epi_vfredsum_1xf64_mask
//TODO #define _MM_REDSUM_f32_MASK __builtin_epi_vfredsum_2xf32_mask

//TODO #define _MM_REDMIN_f64      __builtin_epi_vfredmin_1xf64
//TODO #define _MM_REDMIN_f32      __builtin_epi_vfredmin_2xf32

// TODO this function is equal to __MM_MADD but parameters in different order
// TODO in future change order to be equal to fmadd
#define _MM_MACC_f64      madd // __builtin_epi_vfmacc_1xf64 // AVX function composition
inline _MMR_f64 madd(_MMR_f64 c, _MMR_f64 a, _MMR_f64 b) {
    return _MM_ADD_f64(c, _MM_MUL_f64(a, b));
}

#define _MM_MACC_f32  		_mm512_fmadd_ps // first param c in riscv is last in intel intrinsic (c, a, b) -> (a, b, c)

#define _MM_MADD_f64  		_mm512_fmadd_pd // __builtin_epi_vfmadd_1xf64
#define _MM_MADD_f32  		_mm512_fmadd_ps

//---------------------------------------------------------------------------
// CONVERSION INTRINSICS

#define _MM_VFCVT_F_X_f64   _mm512_castsi512_pd // __builtin_epi_vfcvt_f_x_1xf64_1xi64
#define _MM_VFCVT_F_X_f32   _mm512_castsi512_ps

#define _MM_VFCVT_X_F_i64   _mm512_castpd_si512 // __builtin_epi_vfcvt_x_f_1xi64_1xf64
#define _MM_VFCVT_X_F_i32   _mm512_castps_si512

//TODO #define _MM_VFWCVT_F_F_f64  __builtin_epi_vfwcvt_f_f_2xf64_2xf32
//TODO #define _MM_VFNCVT_F_F_f32  __builtin_epi_vfncvt_f_f_2xf32_2xf64

//TODO #define _MM_VFWCVT_F_X_f64  __builtin_epi_vfwcvt_f_x_2xf64_2xi32

//TODO #define _MM_VFCVT_f32_i32   __builtin_epi_vfcvt_f_x_2xf32_2xi32

//TODO #define FENCE()   asm volatile( "fence" : : );

//---------------------------------------------------------------------------
// VECTOR ELEMENT MANIPULATION

// int
//TODO #define _MM_VEXTRACT_i64    __builtin_epi_vextract_1xi64
//TODO #define _MM_VEXTRACT_i32    __builtin_epi_vextract_2xi32
//TODO #define _MM_VEXTRACT_i16    __builtin_epi_vextract_4xi16
//TODO #define _MM_VEXTRACT_i8    __builtin_epi_vextract_8xi8

//TODO #define _MM_VSLIDEUP_i32    __builtin_epi_vslideup_2xi32
//TODO #define _MM_VSLIDEUP_i64    __builtin_epi_vslideup_1xi64

//TODO #define _MM_VSLIDE1UP_i32    __builtin_epi_vslide1up_2xi32
//TODO #define _MM_VSLIDE1UP_i64    __builtin_epi_vslide1up_1xi64

//TODO #define _MM_VSLIDEUP_i32_MASK    __builtin_epi_vslideup_2xi32_mask
//TODO #define _MM_VSLIDEUP_i64_MASK    __builtin_epi_vslideup_1xi64_mask

//TODO #define _MM_VSLIDEDOWN_i32    __builtin_epi_vslidedown_2xi32
//TODO #define _MM_VSLIDEDOWN_i64    __builtin_epi_vslidedown_1xi64

//TODO #define _MM_VSLIDE1DOWN_i32    __builtin_epi_vslide1down_2xi32
//TODO #define _MM_VSLIDE1DOWN_i64    __builtin_epi_vslide1down_1xi64

//TODO #define _MM_VSLIDEDOWN_i32_MASK    __builtin_epi_vslidedown_2xi32_mask
//TODO #define _MM_VSLIDEDOWN_i64_MASK    __builtin_epi_vslidedown_1xi64_mask

// fp

//TODO #define _MM_VGETFIRST_f32   __builtin_epi_vgetfirst_2xf32
//TODO #define _MM_VGETFIRST_f64   __builtin_epi_vgetfirst_1xf64

//TODO #define _MM_VSLIDEUP_f32    __builtin_epi_vslideup_2xf32
//TODO #define _MM_VSLIDEUP_f64    __builtin_epi_vslideup_1xf64

//TODO #define _MM_VSLIDE1UP_f32    __builtin_epi_vslide1up_2xf32
//TODO #define _MM_VSLIDE1UP_f64    __builtin_epi_vslide1up_1xf64

//TODO #define _MM_VSLIDEUP_f32_MASK    __builtin_epi_vslideup_2xf32_mask
//TODO #define _MM_VSLIDEUP_f64_MASK    __builtin_epi_vslideup_1xf64_mask

//TODO #define _MM_VSLIDEDOWN_f32    __builtin_epi_vslidedown_2xf32
//TODO #define _MM_VSLIDEDOWN_f64    __builtin_epi_vslidedown_1xf64

//TODO #define _MM_VSLIDE1DOWN_f32    __builtin_epi_vslide1down_2xf32
//TODO #define _MM_VSLIDE1DOWN_f64    __builtin_epi_vslide1down_1xf64

//TODO #define _MM_VSLIDEDOWN_f32_MASK    __builtin_epi_vslidedown_2xf32_mask
//TODO #define _MM_VSLIDEDOWN_f64_MASK    __builtin_epi_vslidedown_1xf64_mask
//---------------------------------------------------------------------------
// MASK DEFINITIONS

#define _MMR_MASK_i64   	__mmask16
#define _MMR_MASK_i32   	__mmask16

//TODO #define _MM_CAST_i1_i64  	__builtin_epi_cast_1xi1_1xi64
//TODO #define _MM_CAST_i1_i32  	__builtin_epi_cast_2xi1_2xi32

//TODO #define _MM_CAST_i64_i1  	__builtin_epi_cast_1xi64_1xi1
//TODO #define _MM_CAST_i32_i1  	__builtin_epi_cast_2xi32_2xi1

// OPERATIONS WITH MASKS

#define _MM_VMFIRST_i64 	firstTrue // __builtin_epi_vmfirst_1xi1
//TODO #define _MM_VMFIRST_i32 	__builtin_epi_vmfirst_2xi1

inline int firstTrue(_MMR_MASK_i64 mask, int dimension) {
    int val[dimension];
    memcpy(val, &mask, dimension);
    for (int i = 0; i < dimension; i++) {
        if (val[i] != 0) {
            return i;
        }
    }

    return -1;
}

#define _MM_VMPOPC_i64 		trueCount // __builtin_epi_vmpopc_1xi1

inline int trueCount(_MMR_MASK_i64 a, int dimension) {
    int res = 0;
    int val[dimension];
    memcpy(val, &a, dimension);
    for (int i = 0; i < dimension; i++) {
         if (val[i] != 0) res++;
    }

    return res;
}
//TODO define _MM_VMPOPC_i32 		__builtin_epi_vmpopc_2xi1

//TODO #define _MM_VMAND_i64 		__builtin_epi_vmand_1xi1
//TODO #define _MM_VMAND_i32 		__builtin_epi_vmand_2xi1

//TODO #define _MM_VMNOR_i64 		__builtin_epi_vmnor_1xi1
//TODO #define _MM_VMNOR_i32 		__builtin_epi_vmnor_2xi1

#define _MM_VMOR_i64 		_mm512_kor // _kor_mask8
//TODO #define _MM_VMOR_i32 		__builtin_epi_vmor_2xi1

#define _MM_VMXOR_i64 		_mm512_kxor // _kxor_mask8
//TODO #define _MM_VMXOR_i32 		__builtin_epi_vmxor_2xi1


#define _MM_VMNOT_i64 _mm512_knot

// OPERATIONS TO CREATE A MASK

// Int

#define _MM_VMSLT_i64     _mm512_lt_epi64_mask // __builtin_epi_vmslt_1xi64
//TODO #define _MM_VMSLT_i32     __builtin_epi_vmslt_2xi32

#define _MM_VMSEQ_i64		_mm512_eq_epi64_mask // __builtin_epi_vmseq_1xi64
#define _MM_VMSEQ_i32		_mm512_cmp_epi32_mask

inline _MMR_MASK_i64 _mm512_lt_epi64_mask(_MMR_i64 a, _MMR_i64 b) {
    return _mm512_cmp_epi64_mask(a, b, _CMP_LT_OQ);
}

inline _MMR_MASK_i64 _mm512_eq_epi64_mask(_MMR_i64 a, _MMR_i64 b) {
    return _mm512_cmp_epi64_mask(a, b, _CMP_EQ_OQ);
}

// Fp
#define _MM_VFEQ_f64        _mm512_mask_eq_pd_mask
#define _MM_VFEQ_f32        _mm512_mask_eq_ps_mask


#define _MM_VFGT_f64        _mm512_gt_pd_mask // __builtin_epi_vmfgt_1xf64
//TODO #define _MM_VFGT_f32        __builtin_epi_vmfgt_2xf32

#define _MM_VFGE_f64        _mm512_mask_ge_pd_mask
//TODO #define _MM_VFGE_f32        __builtin_epi_vmfge_2xf32

#define _MM_VFLT_f64        _mm512_lt_pd_mask // __builtin_epi_vmflt_1xf64
#define _MM_VFLT_f32        _mm512_mask_lt_ps_mask // __mmask16 k1, __m512 a, __m512 b, const int imm8


inline _MMR_MASK_i64 _mm512_gt_pd_mask(_MMR_f64 a, _MMR_f64 b) {
    return _mm512_cmp_pd_mask(a, b, _CMP_GT_OS);
}

inline _MMR_MASK_i64 _mm512_mask_ge_pd_mask(_MMR_MASK_i64 mask, _MMR_f64 a, _MMR_f64 b) {
    return _mm512_mask_cmp_pd_mask(mask, a, b, _CMP_GE_OS);
}

inline _MMR_MASK_i32 _mm512_lt_pd_mask(_MMR_f64 a, _MMR_f64 b) {
    return _mm512_cmp_pd_mask(a, b, _CMP_LT_OS);
}

inline _MMR_MASK_i32 _mm512_mask_lt_ps_mask(_MMR_MASK_i32 mask, _MMR_f32 a, _MMR_f32 b) {
    return _mm512_mask_cmp_ps_mask(mask, a, b, _CMP_LT_OS);
}

inline _MMR_MASK_i64 _mm512_mask_eq_pd_mask(_MMR_MASK_i64 mask, _MMR_f64 a, _MMR_f64 b) {
    return _mm512_mask_cmp_pd_mask(mask, a, b, _CMP_EQ_OQ);
}

inline _MMR_MASK_i32 _mm512_mask_eq_ps_mask(_MMR_MASK_i32 mask, _MMR_f32 a, _MMR_f32 b) {
    return _mm512_mask_cmp_ps_mask(mask, a, b, _CMP_EQ_OQ);
}

#define _MM_VFLE_f64   _mm512_mask_le_pd_mask    //  __builtin_epi_vmfle_1xf64
//TODO #define _MM_VFLE_f32        __builtin_epi_vmfle_2xf32

inline _MMR_MASK_i64 _mm512_mask_le_pd_mask(_MMR_f64 a, _MMR_f64 b) {
    return _mm512_cmp_pd_mask(a, b, _CMP_LE_OS);
}

//---------------------------------------------------------------------------
// ADVANCE RISC-V MATH LIBRARY

#ifndef _MM_LOG
#define _MM_LOG
#include "__log.h"
#define _MM_LOG_f64 		__log_1x64_scalar // __log_1xf64 // _mm512_log_pd
#define _MM_LOG_f32 		_mm512_log_ps
#endif

#ifndef _MM_EXP
#define _MM_EXP
#include "__exp.h"
#define _MM_EXP_f64 __exp_1xf64
#define _MM_EXP_f32 __exp_2xf32		// _mm512_exp_ps only for intel compiler
#endif

#ifndef _MM_COS
#define _MM_COS
#include "__cos.h"
#define _MM_COS_f64 	__cos_1xf64 // _mm512_cos_pd <- only in SVML intel
//TODO #define _MM_COS_f32 		__cos_1xf32
#endif

//---------------------------------------------------------------------------

#endif // VECTOR_DEFINITION
