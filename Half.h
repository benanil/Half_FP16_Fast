
/******************************************************************************************
*  Purpose:                                                                               *
*    Conversion of 16 bit floating point values and 32 bit floating point values          *
*    SSE and AVX used for performance but not required, scalar versions exists            *
*  Good To Know:                                                                          *
*    ...                                                                                  *
*  Author:                                                                                *
*    Anilcan Gulkaya 2024 anilcangulkaya7@gmail.com github @benanil                       *
*    Mike Acton (original implementation): https://cellperformance.beyond3d.com/articles/2006/07/branchfree_implementation_of_h_1.html#half_to_float *
*******************************************************************************************/

// todo better check for arm neon fp16

#ifndef Half_FP16_Fast
#define Half_FP16_Fast

#if defined(__clang__) || defined(__GNUC__)
    #define purefn [[clang::always_inline]] __attribute__((pure)) 
#elif defined(_MSC_VER)
    #define purefn __forceinline __declspec(noalias)
#else
    #define purefn inline __attribute__((always_inline))
#endif

/* Architecture Detection */
// detection code from mini audio
// you can define AX_NO_SSE2 or AX_NO_AVX2 in order to disable this extensions

#if defined(__x86_64__) || defined(_M_X64)
    #define AX_X64
#elif defined(__i386) || defined(_M_IX86)
    #define AX_X86
#elif defined(_M_ARM) || defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64) || defined(_M_ARM64EC) || __arm__ || __aarch64__
    #define AX_ARM
#endif

#if defined(AX_ARM)
    #if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64) || defined(_M_ARM64EC) || defined(__aarch64__))
        #include <arm64_neon.h>
    #else
        #include <arm_neon.h>
    #endif
#endif

// write AX_NO_SSE2 or AX_NO_AVX2 to disable vector instructions

/* Intrinsics Support */
#if (defined(AX_X64) || defined(AX_X86)) && !defined(AX_ARM)
    #if defined(_MSC_VER) && !defined(__clang__)
        #if _MSC_VER >= 1400 && !defined(AX_NO_SSE2)   /* 2005 */
            #define AX_SUPPORT_SSE
        #endif
        #if _MSC_VER >= 1700 && !defined(AX_NO_AVX2)   /* 2012 */
            #define AX_SUPPORT_AVX2
        #endif
    #else
        #if defined(__SSE2__) && !defined(AX_NO_SSE2)
            #define AX_SUPPORT_SSE
        #endif
        #if defined(__AVX2__) && !defined(AX_NO_AVX2)
            #define AX_SUPPORT_AVX2
        #endif
    #endif
    
    /* If at this point we still haven't determined compiler support for the intrinsics just fall back to __has_include. */
    #if !defined(__GNUC__) && !defined(__clang__) && defined(__has_include)
        #if !defined(AX_SUPPORT_SSE) && !defined(AX_NO_SSE2) && __has_include(<emmintrin.h>)
            #define AX_SUPPORT_SSE
        #endif
        #if !defined(AX_SUPPORT_AVX2) && !defined(AX_NO_AVX2) && __has_include(<immintrin.h>)
            #define AX_SUPPORT_AVX2
        #endif
    #endif

    #if defined(AX_SUPPORT_AVX2) || defined(AX_SUPPORT_AVX)
        #include <immintrin.h>
    #elif defined(AX_SUPPORT_SSE)
        #include <emmintrin.h>
    #endif
#endif

#if defined(__ARM_NEON__)
    #define PopCount32(x) vcnt_u8((int8x8_t)x)
#elif defined(AX_SUPPORT_SSE)
    #define PopCount32(x) _mm_popcnt_u32(x)
#elif defined(__GNUC__) || !defined(__MINGW32__)
    #define PopCount32(x) __builtin_popcount(x)
#else
purefn uint32_t PopCount32(uint32_t x) {
    x =  x - ((x >> 1) & 0x55555555);        // add pairs of bits
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);  // quads
    x = (x + (x >> 4)) & 0x0F0F0F0F;        // groups of 8
    return (x * 0x01010101) >> 24;          // horizontal sum of bytes	
}

// standard popcount; from wikipedia
purefn uint64_t PopCount64(uint64_t x) {
    x -= ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + (x >> 2 & 0x3333333333333333ull);
    return ((x + (x >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
}
#endif

#ifdef _MSC_VER
    #define LeadingZeroCount32(x) _lzcnt_u32(x)
#elif defined(__GNUC__) || !defined(__MINGW32__)
    #define LeadingZeroCount32(x) __builtin_clz(x)
#else
    #error you should define pack function
    purefn uint32_t LeadingZeroCount64(uint32_t x)
    {
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return (sizeof(uint32_t) * 8) - PopCount32(x);
    }
#endif

#if defined(AX_SUPPORT_SSE) && !defined(AX_ARM)
/*//////////////////////////////////////////////////////////////////////////*/
/*                                 SSE                                      */
/*//////////////////////////////////////////////////////////////////////////*/

typedef __m128  vec_t;
typedef __m128i vecu_t;

#define VecAdd(a, b) _mm_add_ps(a, b)
#define VecLoad(x)           _mm_loadu_ps(x)
#define VecSet1(x)           _mm_set1_ps(x)
#define VeciFromVec(x) _mm_castps_si128(x)
//------------------------------------------------------------------------
// Veci
#define VeciZero()            _mm_set1_epi32(0)
#define VeciSet1(x)          _mm_set1_epi32(x)
#define VeciSet(x, y, z, w)  _mm_set_epi32(x, y, z, w)
#define VeciSetR(x, y, z, w) _mm_setr_epi32(x, y, z, w)
#define VeciLoadA(x)         _mm_load_epi32(x)
#define VeciLoad(x)          _mm_loadu_epi32(x)
#define VeciLoad64(qword)    _mm_loadu_si64(qword)     /* loads 64bit integer to first 8 bytes of register */

#define VeciSelect1111 _mm_set1_epi32(0xFFFFFFFF)

// int 
#define VeciAdd(a, b) _mm_add_epi32(a, b)
#define VeciSub(a, b) _mm_sub_epi32(a, b)
#define VeciMul(a, b) _mm_mul_epi32(a, b)

// Int
#define VeciNot(a)             _mm_andnot_si128(a, _mm_set1_epi32(0xFFFFFFFF))
#define VeciAnd(a, b)          _mm_and_si128(a, b)
#define VeciOr(a, b)           _mm_or_si128(a, b)
#define VeciXor(a, b)          _mm_xor_si128(a, b)

#define VeciAndNot(a, b)       _mm_andnot_si128(a, b)  /* ~a  & b */
#define VeciSrl(a, b)          _mm_srlv_epi32(a, b)    /*  a >> b */
#define VeciSll(a, b)          _mm_sllv_epi32(a, b)    /*  a << b */
#define VeciSrl32(a, b)        _mm_srli_epi32(a, b)    /*  a >> b */
#define VeciSll32(a, b)        _mm_slli_epi32(a, b)    /*  a << b */
#define VeciToVecf(a)          _mm_castsi128_ps(a)     /*  a << b */

#define VeciUnpackLow(a, b)   _mm_unpacklo_epi32(a, b)  /*  [a.x, a.y, b.x, b.y] */
#define VeciUnpackLow16(a, b) _mm_unpacklo_epi16(a, b)  /*  [a.x, a.y, b.x, b.y] */
#define VeciBlend(a, b, c)    _mm_blendv_epi8(a, b, c)

#elif defined(AX_ARM)
typedef float32x4_t vec_t;
typedef uint32x4_t vecu_t;

#define VecSet1(x)          vdupq_n_f32(x)
#define VecAdd(a, b) vaddq_f32(a, b)
#define VecLoad(x)          vld1q_f32(x)
#define VecToInt(x)    vreinterpretq_u32_f32(x)
#define VeciFromVec(x) vreinterpretq_u32_f32(x)
//------------------------------------------------------------------------
// Veci
#define VeciZero()   vdupq_n_u32(0)
#define VeciSet1(x)  vdupq_n_u32(x)
#define VeciSetR(x, y, z, w)  ARMCreateVecI(x, y, z, w)
#define VeciSet(x, y, z, w)   ARMCreateVecI(w, z, y, x)
#define VeciLoadA(x)          vld1q_u32(x)
#define VeciLoad(x)           vld1q_u32(x)
#define VeciLoad64(qword)     vcombine_u32(vcreate_u32(qword), vcreate_u32(0ull)) /* loads 64bit integer to first 8 bytes of register */

#define VeciAdd(a, b) vaddq_u32(a, b)
#define VeciSub(a, b) vsubq_u32(a, b)
#define VeciMul(a, b) vmulq_u32(a, b)

#define VecFromVeci(x) vreinterpretq_f32_u32(x)
#define VeciFromVec(x) vreinterpretq_u32_f32(x)
// Swizzling Masking
#define VecSelect1111 ARMCreateVecI(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu)
#define VeciSelect1111 ARMCreateVecI(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu)

// Logical
#define VeciNot(a)     vmvnq_u32(a)
#define VeciAnd(a, b)  vandq_u32(a, b)
#define VeciOr(a, b)   vorrq_u32(a, b)
#define VeciXor(a, b)  veorq_u32(a, b)

#define VeciAndNot(a, b) vandq_u32(vmvnq_u32(a), b)  /* ~a & b */
#define VeciSrl(a, b)    vshlq_u32(a, vnegq_s32(b))  /* a >> b */
#define VeciSll(a, b)    vshlq_u32(a, b)             /* a << b */
#define VeciSrl32(a, b)  vshrq_n_u32(a, b)           /* a >> b */
#define VeciSll32(a, b)  vshlq_n_u32(a, b)           /* a << b */
#define VeciToVecf(a)    vreinterpretq_f32_s32(a)    /* Reinterpret int as float */

#define VeciUnpackLow(a, b)   vzip1q_s32(a, b)   /* [a.x, a.y, b.x, b.y] */
#define VeciUnpackLow16(a, b) vzip1q_s16(a, b)   /* [a.x, a.y, b.x, b.y] */
#define VeciBlend(a, b, c)    vbslq_u8(c, b, a)  /* Blend a and b based on mask c */

purefn veci_t ARMCreateVecI(uint x, uint y, uint z, uint w) {
    return vcombine_u32(vcreate_u32(((uint64_t)x) | (((uint64_t)y) << 32)),
                        vcreate_u32(((uint64_t)z) | (((uint64_t)w) << 32)));
}

#endif

#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <float.h>

typedef unsigned char      uint8;
typedef unsigned short     uint16;
typedef unsigned int       uint32;
typedef unsigned long long uint64;
typedef char      int8;
typedef short     int16;
typedef int       int32;
typedef long long int64;

typedef uint8_t  uchar;
typedef uint16_t ushort;
typedef uint32_t uint;

/*//////////////////////////////////////////////////////////////////////////*/
/*                             Half                                         */
/*//////////////////////////////////////////////////////////////////////////*/

typedef ushort half;
#define OneFP16()  (15360)
#define MinusOneFP16() (48128)
#define ZeroFP16()  (0)
#define HalfFP16() (14336) /* fp16 0.5 */
#define Sqrt2FP16() (15784) /* fp16 sqrt(2) */

typedef uint half2;
#define Half2Up()     (OneFP16 << 16u)
#define Half2Down()   (MinusOneFP16 << 16u)
#define Half2Left()   (MinusOneFP16)
#define Half2Right()  (OneFP16)
#define Half2One()    (OneFP16 | (OneFP16 << 16))
#define Half2Zero()   (0)

#define MakeHalf2(x, y) ((x) | ((y) << 16))
#define Half2SetX(v, x) (v &= 0xFFFF0000u, v |= x;)
#define Half2SetY(v, y) (v &= 0x0000FFFFu, v |= y;)

#if defined(_MSC_VER) && AX_CPP_VERSION < AX_CPP17
  #define BitCast(To, _Val) *(const To*)(&_Val);
#else
  #define BitCast(To, _Val) __builtin_bit_cast(To, _Val);
#endif

// todo better check for half support
purefn float ConvertHalfToFloat(half x) 
{
#if defined(AX_SUPPORT_AVX2) 
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(x))); 
#elif defined(__ARM_NEON__)
    return _cvtsh_ss(x); 
#else
    uint h = x;
    uint h_e = h & 0x00007c00u;
    uint h_m = h & 0x000003ffu;
    uint h_s = h & 0x00008000u;
    uint h_e_f_bias = h_e + 0x0001c000u;
    uint h_m_nlz = LeadingZeroCount32(h_m) - (32u - 10u); // Assuming 32-bit integer

    uint f_s  = h_s        << 0x00000010u;
    uint f_e  = h_e_f_bias << 0x0000000du;
    uint f_m  = h_m        << 0x0000000du;
    uint f_em = f_e | f_m;
        
    uint h_f_m_sa = h_m_nlz - 0x00000008u;
    uint f_e_denorm_unpacked = 0x0000007eu - h_f_m_sa;
    
    uint h_f_m       = h_m << h_f_m_sa;
    uint f_m_denorm  = h_f_m & 0x007fffffu;
    uint f_e_denorm  = f_e_denorm_unpacked << 0x00000017u;
    uint f_em_denorm = f_e_denorm | f_m_denorm;
    uint f_em_nan    = 0x7f800000u | f_m;
        
    uint is_e_eqz_msb = h_e - 1u;
    uint is_m_nez_msb = ~h_m + 1u;
    uint is_e_flagged_msb = 0x00007bffu - h_e;
    uint is_zero_msb = ~(is_e_eqz_msb & is_m_nez_msb);
    uint is_inf_msb = ~(is_e_flagged_msb & is_m_nez_msb);
    uint is_denorm_msb = is_m_nez_msb & is_e_eqz_msb;
    uint is_nan_msb = is_e_flagged_msb & is_m_nez_msb;
        
    uint is_zero = is_zero_msb >> 31u;
    uint f_zero_result = f_em & ~is_zero;
    uint msk = is_denorm_msb >> 31;
    uint f_denorm_result = (msk & f_em_denorm) | (~msk & f_zero_result);
    msk = is_inf_msb >> 31;
    uint f_inf_result = (msk & 0x7f800000u) | (~msk & f_denorm_result);
    msk = is_nan_msb >> 31;
    uint f_nan_result = (msk & f_em_nan) | (~msk & f_inf_result);
    uint f_result = f_s | f_nan_result;
    return BitCast(float, f_result);
#endif
}

purefn half ConvertFloatToHalf(float Value) 
{
#if defined(AX_SUPPORT_AVX2)
    return _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(Value), 0), 0);
#elif defined(__ARM_NEON__)
    return _cvtss_sh(Value, 0);
#else
    uint32_t Result; // branch removed version of DirectxMath function
    uint32_t IValue = BitCast(uint32_t, Value);
    uint32_t Sign = (IValue & 0x80000000u) >> 16U;
    IValue = IValue & 0x7FFFFFFFu;      // Hack off the sign
    // if (IValue > 0x47FFEFFFu) { 
    //     return 0x7FFFU | Sign; // The number is too large to be represented as a half.  Saturate to infinity.
    // }
    uint32_t mask = 0u - (IValue < 0x38800000u);
    uint32_t b = IValue + 0xC8000000U;
    uint32_t a = (0x800000u | (IValue & 0x7FFFFFu)) >> (113u - (IValue >> 23u));
    
    IValue = (mask & a) | (~mask & b);
    Result = ((IValue + 0x0FFFu + ((IValue >> 13u) & 1u)) >> 13u) & 0x7FFFu; 
    return (half)(Result | Sign);
#endif
}

inline void ConvertHalf2ToFloat2(float* result, uint32_t h) 
{
    #if defined(AX_SUPPORT_AVX2) || defined(__ARM_NEON__)
    result[0] = ConvertHalfToFloat(h & 0xFFFF);
    result[1] = ConvertHalfToFloat(h >> 16);
    #else
    uint64_t h2 = (uint64_t)(h & 0x0000FFFFull) | (uint64_t(h & 0xFFFF0000ull) << 16ull);

    uint64_t h_e = h2 & 0x00007c0000007c00ull;
    uint64_t h_m = h2 & 0x000003ff000003ffull;
    uint64_t h_s = h2 & 0x0000800000008000ull;
    uint64_t h_e_f_bias = h_e + 0x0001c0000001c000ull;

    uint64_t h_m_nlz = LeadingZeroCount32(h_m & 0xFFFFFFFFull) - (32ull - 10ull);
    h_m_nlz |= (uint64_t)(LeadingZeroCount32((h_m >> 32u) & 0xFFFFFFFFull) - (32ull - 10ull)) << 32ull;

    uint64_t f_s  = h_s        << 0x00000010ull;
    uint64_t f_e  = h_e_f_bias << 0x0000000dull;
    uint64_t f_m  = h_m        << 0x0000000dull;
    uint64_t f_em = f_e | f_m;
        
    uint64_t h_f_m_sa = h_m_nlz - 0x0000000800000008ull;
    uint64_t f_e_denorm_unpacked = 0x0000007e0000007eull - h_f_m_sa;

    uint64_t h_f_m = h_m << (h_f_m_sa & 0xFFFFFFFFull) & 0xFFFFFFFFull;
    h_f_m |= ((h_m >> 32ull) << (h_f_m_sa >> 32ull)) << 32ull;

    uint64_t f_m_denorm  = h_f_m & 0x007fffff007fffffull;
    uint64_t f_e_denorm  = f_e_denorm_unpacked << 0x00000017ull;
    uint64_t f_em_denorm = f_e_denorm | f_m_denorm;
    uint64_t f_em_nan    = 0x7f8000007f800000ull | f_m;
        
    uint64_t is_e_eqz_msb     =  h_e - 1ull - (1ull << 32ull);
    uint64_t is_m_nez_msb     = ~h_m + 1ull + (1ull << 32ull);
    const uint64_t SignMask = 0x8000000080000000ull;
    is_m_nez_msb &= SignMask;

    uint64_t is_e_flagged_msb = 0x00007bff00007bffull - h_e;
    uint64_t is_zero_msb      = ~(is_e_eqz_msb     & is_m_nez_msb);
    uint64_t is_inf_msb       = ~(is_e_flagged_msb & is_m_nez_msb);
    uint64_t is_denorm_msb    = is_m_nez_msb     & is_e_eqz_msb;
    uint64_t is_nan_msb       = is_e_flagged_msb & is_m_nez_msb;
        
    uint64_t is_zero  = (is_zero_msb   & SignMask) >> 31ull;
    uint64_t msk      = (is_denorm_msb & SignMask) >> 31ull;
    
    uint64_t f_zero_result   = f_em & ~is_zero;
    uint64_t f_denorm_result = (msk & f_em_denorm) | (~msk & f_zero_result);
    
    msk  = (is_inf_msb & SignMask) >> 31ull;
    uint64_t f_inf_result = (msk & 0x7f8000007f800000ull) | (~msk & f_denorm_result);
    
    msk  = (is_nan_msb & SignMask) >> 31ull;
    uint64_t f_nan_result = (msk & f_em_nan) | (~msk & f_inf_result);
    uint64_t f_result = f_s | f_nan_result;

    result[0] = BitCast(float, (uint32_t)(f_result & 0xFFFFFFFFu));
    result[1] = BitCast(float, (uint32_t)(f_result >> 32ull));
    #endif
}

inline void ConvertFloat2ToHalf2(void* result, float* float2)
{
    *(half*)result       = ConvertFloatToHalf(float2[0]);
    *((half*)result + 1) = ConvertFloatToHalf(float2[1]);
}

// input half4 is 4x 16 bit integer for example it can be uint64_t
inline void ConvertHalf4ToFloat4(vec_t* result, void* half4) 
{
    #ifdef AX_SUPPORT_AVX2
    *result = _mm_cvtph_ps(_mm_loadu_si64(half4));
    #elif defined(AX_SUPPORT_AVX2) || defined(__ARM_NEON__)
    vecu_t h4 = VeciLoad64(half4);
    h4 = VeciUnpackLow16(h4, VeciZero());   // [half4.xy, half4.xy, half4.zw, half4.zw] 
    
    vecu_t h_e = VeciAnd(h4, VeciSet1(0x00007c00));
    vecu_t h_m = VeciAnd(h4, VeciSet1(0x000003ff));
    vecu_t h_s = VeciAnd(h4, VeciSet1(0x00008000));
    vecu_t h_e_f_bias = VeciAdd(h_e, VeciSet1(0x0001c000));
    vecu_t h_m_nlz    = VeciSub(LeadingZeroCount32_128(h_m), VeciSet1(32ull - 10ull));
    
    vecu_t f_s  = VeciSll32(h_s, 0x00000010);
    vecu_t f_e  = VeciSll32(h_e_f_bias, 0x0000000d);
    vecu_t f_m  = VeciSll32(h_m, 0x0000000d);
    vecu_t f_em = VeciOr(f_e, f_m);
        
    vecu_t h_f_m_sa = VeciSub(h_m_nlz, VeciSet1(0x00000008));
    vecu_t f_e_denorm_unpacked = VeciSub(VeciSet1(0x0000007e), h_f_m_sa);
    
    vecu_t h_f_m        = VeciSll(h_m, h_f_m_sa);
    vecu_t f_m_denorm   = VeciAnd(h_f_m, VeciSet1(0x007fffff));
    vecu_t f_e_denorm   = VeciSll32(f_e_denorm_unpacked, 0x00000017ull);
    vecu_t f_em_denorm  = VeciOr(f_e_denorm, f_m_denorm);
    vecu_t f_em_nan     = VeciOr(VeciSet1(0x7f800000), f_m);
        
    vecu_t is_e_eqz_msb = VeciSub(h_e, VeciSet1(1));
    vecu_t is_m_nez_msb = VeciAdd(VeciAndNot(h_m, VeciSelect1111) , VeciSet1(1));
    
    vecu_t is_e_flagged_msb = VeciSub(VeciSet1(0x00007bff), h_e);
    vecu_t is_zero_msb      = VeciAndNot(is_m_nez_msb, is_e_eqz_msb);
    vecu_t is_inf_msb       = VeciAndNot(is_m_nez_msb, is_e_flagged_msb);
    vecu_t is_denorm_msb    = VeciAnd(is_m_nez_msb, is_e_eqz_msb);
    vecu_t is_nan_msb       = VeciAnd(is_e_flagged_msb, is_m_nez_msb);
        
    vecu_t is_zero  = VeciSrl32(is_zero_msb, 31ull);
    
    vecu_t msk = VeciSrl32(is_denorm_msb, 31ull);
    vecu_t f_zero_result = VeciAndNot(is_zero, f_em);
    vecu_t f_denorm_result = VeciOr(VeciAnd(msk, f_em_denorm), VeciAndNot(msk, f_zero_result));
    
    msk  = VeciSrl32(is_inf_msb, 31ull);
    vecu_t f_inf_result = VeciOr(VeciAnd(msk, VeciSet1(0x7f800000)), VeciAndNot(msk, f_denorm_result));
    
    msk  = VeciSrl32(is_nan_msb, 31ull);
    vecu_t f_nan_result = VeciOr(VeciAnd(msk, f_em_nan), VeciAndNot(msk, f_inf_result));
    vecu_t i_result     = VeciOr(f_s, f_nan_result);
    *result = VeciToVecf(i_result);
    #else
    ConvertHalf2ToFloat2((float*)result, *(uint32_t*)half4);
    ConvertHalf2ToFloat2((float*)result + 2, *((uint32_t*)(half4) + 1));
    #endif
}

inline void ConvertFloat4ToHalf4(half* result, float* float4)
{
    #ifdef AX_SUPPORT_AVX2
    *((long long*)result) = _mm_extract_epi64(_mm_cvtps_ph(_mm_loadu_ps(float4), _MM_FROUND_TO_NEAREST_INT), 0);
    #elif defined(AX_SUPPORT_AVX2) || defined(__ARM_NEON__)
    const vecu_t one                        = VeciSet1( 0x00000001 );
    const vecu_t f_s_mask                   = VeciSet1( 0x80000000 );
    const vecu_t f_e_mask                   = VeciSet1( 0x7f800000 );
    const vecu_t f_m_mask                   = VeciSet1( 0x007fffff );
    const vecu_t f_m_hidden_bit             = VeciSet1( 0x00800000 );
    const vecu_t f_m_round_bit              = VeciSet1( 0x00001000 );
    const vecu_t f_snan_mask                = VeciSet1( 0x7fc00000 );
    const vecu_t f_e_pos                    = VeciSet1( 0x00000017 );
    const vecu_t h_e_pos                    = VeciSet1( 0x0000000a );
    const vecu_t h_e_mask                   = VeciSet1( 0x00007c00 );
    const vecu_t h_snan_mask                = VeciSet1( 0x00007e00 );
    const vecu_t h_e_mask_value             = VeciSet1( 0x0000001f );
    const vecu_t f_h_s_pos_offset           = VeciSet1( 0x00000010 );
    const vecu_t f_h_bias_offset            = VeciSet1( 0x00000070 );
    const vecu_t f_h_m_pos_offset           = VeciSet1( 0x0000000d );
    const vecu_t h_nan_min                  = VeciSet1( 0x00007c01 );
    const vecu_t f_h_e_biased_flag          = VeciSet1( 0x0000008f );
    
    vec_t f4 = VecAdd(VecLoad(float4), VecSet1(0.0001f)); // prevent the problem with 0 value, maybe there is better solution for this but idk
    vecu_t f = VecToInt(f4);
    vecu_t f_s                        = VeciAnd( f,               f_s_mask         );
    vecu_t f_e                        = VeciAnd( f,               f_e_mask         );
    vecu_t h_s                        = VeciSrl( f_s,             f_h_s_pos_offset );
    vecu_t f_m                        = VeciAnd( f,               f_m_mask         );
    vecu_t f_e_amount                 = VeciSrl( f_e,             f_e_pos          );
    vecu_t f_e_half_bias              = VeciSub( f_e_amount,      f_h_bias_offset  );
    vecu_t f_snan                     = VeciAnd( f,               f_snan_mask      );
    vecu_t f_m_round_mask             = VeciAnd( f_m,             f_m_round_bit    );
    vecu_t f_m_round_offset           = VeciSll( f_m_round_mask,  one              );
    vecu_t f_m_rounded                = VeciAdd( f_m,             f_m_round_offset );
    vecu_t f_m_denorm_sa              = VeciSub( one,             f_e_half_bias    );
    vecu_t f_m_with_hidden            = VeciOr(  f_m_rounded,     f_m_hidden_bit   );
    vecu_t f_m_denorm                 = VeciSrl( f_m_with_hidden, f_m_denorm_sa    );
    vecu_t h_m_denorm                 = VeciSrl( f_m_denorm,      f_h_m_pos_offset );
    vecu_t f_m_rounded_overflow       = VeciAnd( f_m_rounded,     f_m_hidden_bit   );
    vecu_t m_nan                      = VeciSrl( f_m,             f_h_m_pos_offset );
    vecu_t h_em_nan                   = VeciOr(  h_e_mask,        m_nan            );
    vecu_t h_e_norm_overflow_offset   = VeciAdd( f_e_half_bias,   one);
    vecu_t h_e_norm_overflow          = VeciSll( h_e_norm_overflow_offset, h_e_pos          );
    vecu_t h_e_norm                   = VeciSll( f_e_half_bias,            h_e_pos          );
    vecu_t h_m_norm                   = VeciSrl( f_m_rounded,              f_h_m_pos_offset );
    vecu_t h_em_norm                  = VeciOr(  h_e_norm,                 h_m_norm         );
    vecu_t is_h_ndenorm_msb           = VeciSub( f_h_bias_offset,   f_e_amount    );
    vecu_t is_f_e_flagged_msb         = VeciSub( f_h_e_biased_flag, f_e_half_bias );
    vecu_t is_h_denorm_msb            = VeciNot( is_h_ndenorm_msb );
    vecu_t is_f_m_eqz_msb             = VeciSub( f_m   , one);
    vecu_t is_h_nan_eqz_msb           = VeciSub( m_nan , one);
    vecu_t is_f_inf_msb               = VeciAnd( is_f_e_flagged_msb, is_f_m_eqz_msb   );
    vecu_t is_f_nan_underflow_msb     = VeciAnd( is_f_e_flagged_msb, is_h_nan_eqz_msb );
    vecu_t is_e_overflow_msb          = VeciSub( h_e_mask_value,     f_e_half_bias    );
    vecu_t is_h_inf_msb               = VeciOr(  is_e_overflow_msb,  is_f_inf_msb     );
    vecu_t is_f_nsnan_msb             = VeciSub( f_snan,             f_snan_mask      );
    vecu_t is_m_norm_overflow_msb     = VeciNeg( f_m_rounded_overflow );
    vecu_t is_f_snan_msb              = VeciNot( is_f_nsnan_msb );
    vecu_t h_em_overflow_result       = VeciBlend(h_em_norm                , h_e_norm_overflow, VeciSll32(is_m_norm_overflow_msb, 31));
    vecu_t h_em_nan_result            = VeciBlend(h_em_overflow_result     , h_em_nan,          VeciSll32(is_f_e_flagged_msb, 31));
    vecu_t h_em_nan_underflow_result  = VeciBlend(h_em_nan_result          , h_nan_min,         VeciSll32(is_f_nan_underflow_msb, 31));
    vecu_t h_em_inf_result            = VeciBlend(h_em_nan_underflow_result, h_e_mask,          VeciSll32(is_h_inf_msb, 31));
    vecu_t h_em_denorm_result         = VeciBlend(h_em_inf_result          , h_m_denorm,        VeciSll32(is_h_denorm_msb, 31));
    vecu_t h_em_snan_result           = VeciBlend(h_em_denorm_result       , h_snan_mask,       VeciSll32(is_f_snan_msb, 31));
    vecu_t h_result                   = VeciOr( h_s, h_em_snan_result );
        
    #ifdef AX_SUPPORT_SSE
        const int shufleMask = MakeShuffleMask(0, 2, 1, 3);
        __m128i lo = _mm_shufflelo_epi16(h_result, shufleMask);
        __m128i hi = _mm_shufflehi_epi16(lo, shufleMask);
        h_result = _mm_shuffle_epi32(hi, shufleMask);
        *((long long*)result) = _mm_extract_epi64(h_result, 0);
    #else
        // todo test
        // Narrow the 32-bit to 16-bit, effectively extracting the lower 16 bits of each element
        uint16x4_t low16_bits = vmovn_u32(vec);  // Narrow to 16 bits per element
        // Directly cast the `uint16x4_t` to `uint64_t`
        vst1_u64((uint64_t*)result, vreinterpret_u64_u16(low16_bits));
    #endif

    #else // no intrinsics
        ConvertFloat2ToHalf2(result, float4);
        ConvertFloat2ToHalf2(result + 2, float4 + 2);
    #endif
}


#ifndef AX_SUPPORT_AVX2
inline void ConvertHalf8ToFloat8(void* half8)
{
    ConvertHalf4ToFloat4((vec_t*)half8    , half8);
    ConvertHalf4ToFloat4((vec_t*)half8 + 1, (uint64_t*)half8 + 1);
}

inline void ConvertFloat8ToHalf8(vec_t* result, float* float8)
{
    ConvertFloat4ToHalf4((half*)result, float8);
    ConvertFloat4ToHalf4((half*)result + 4, float8 + 4);
}

#else


#define ConvertFloat8ToHalf8(result, float8) _mm_storeu_si128((__m128i*)result, _mm256_cvtps_ph(_mm256_loadu_ps(float8), _MM_FROUND_TO_NEAREST_INT))

#define ConvertHalf8ToFloat8(result, half8)  _mm256_storeu_ps(result,  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)half8)))

#endif // AX_SUPPORT_AVX2


inline void ConvertHalfToFloatN(float* res, const half* x, const int n) 
{   
    for (int i = 0; i < n; i += 8, x += 8, res += 8)
        ConvertHalf8ToFloat8(res, x);
 
    for (int i = 0; i < (n & 7); i++, res++, x++)
        *res = ConvertHalfToFloat(*x);
}

inline void ConvertFloatToHalfN(half* res, const float* x, const int n) 
{   
    for (int i = 0; i < n; i += 8, x += 8, res += 8)
        ConvertFloat8ToHalf8(res, x);
 
    for (int i = 0; i < (n & 7); i++, res++, x++)
        *res = ConvertFloatToHalf(*x);
}

purefn float3 ConvertHalf3ToFloat3(half* h) {
	float3 res; 
    ConvertHalf2ToFloat2(&res.x, *(uint32_t*)h); 
    res.z = ConvertHalfToFloat(h[2]); 
    return res;
}

#endif // Half_FP16_Fast
