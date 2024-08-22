
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

#ifndef Half_FP16_Fast
#define Half_FP16_Fast
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

#if defined(__clang__) || defined(__GNUC__)
    #define purefn [[clang::always_inline]] __attribute__((pure)) 
#elif defined(_MSC_VER)
    #define purefn __forceinline __declspec(noalias)
#else
    #define purefn inline __attribute__((always_inline))
#endif

#ifdef _MSC_VER
    #include <intrin.h>
    #define VECTORCALL __vectorcall
#elif __CLANG__
    #define VECTORCALL [[clang::vectorcall]] 
#elif __GNUC__
    #define VECTORCALL  
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

#if defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ ) || defined(__ARM_NEON__)
    #define AX_SUPPORT_NEON
    #include <arm_fp16.h>
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

purefn vecu_t ARMCreateVecI(uint x, uint y, uint z, uint w) {
    return vcombine_u32(vcreate_u32(((uint64_t)x) | (((uint64_t)y) << 32)),
                        vcreate_u32(((uint64_t)z) | (((uint64_t)w) << 32)));
}

#endif

/*//////////////////////////////////////////////////////////////////////////*/
/*                             Half                                         */
/*//////////////////////////////////////////////////////////////////////////*/

#ifdef AX_SUPPORT_NEON
typedef float16_t half;
#else
typedef ushort half;
#endif

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
#elif defined(AX_SUPPORT_NEON)
    return vgetq_lane_f32(vcvt_f32_f16(vdup_n_f16(x)), 0);
#else
    uint h = x;
    uint h_e = h & 0x00007c00u;
    uint h_m = h & 0x000003ffu;
    uint h_s = h & 0x00008000u;
    uint h_e_f_bias = h_e + 0x0001c000u;

    uint f_s  = h_s        << 0x00000010u;
    uint f_e  = h_e_f_bias << 0x0000000du;
    uint f_m  = h_m        << 0x0000000du;
    uint f_result = f_s | f_e | f_m;
        
    return BitCast(float, f_result);
#endif
}

purefn half ConvertFloatToHalf(float Value) 
{
#if defined(AX_SUPPORT_AVX2)
    return _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(Value), 0), 0);
#elif defined(AX_SUPPORT_NEON)
    return vget_lane_f16(vcvt_f16_f32(vdupq_n_f32(Value)), 0);
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
#if defined(AX_SUPPORT_AVX2)
    _mm_storel_pi((__m64 *)result, _mm_cvtph_ps(_mm_set1_epi32(h)));
#elif defined(AX_SUPPORT_NEON)
    float16x4_t halfVec = vreinterpret_f16_u32(vdup_n_u32(h));
    vst1_f32(result, vget_low_f32(vcvt_f32_f16(halfVec)));
#else
    uint64_t h2 = (uint64_t)(h & 0x0000FFFFull) | (uint64_t(h & 0xFFFF0000ull) << 16ull);

    uint64_t h_e = h2 & 0x00007c0000007c00ull;
    uint64_t h_m = h2 & 0x000003ff000003ffull;
    uint64_t h_s = h2 & 0x0000800000008000ull;
    uint64_t h_e_f_bias = h_e + 0x0001c0000001c000ull;

    uint64_t f_s  = h_s        << 0x00000010ull;
    uint64_t f_e  = h_e_f_bias << 0x0000000dull;
    uint64_t f_m  = h_m        << 0x0000000dull;
    uint64_t f_result = f_s | f_e | f_m;
        
    *(uint32_t*)result       = f_result & 0xFFFFFFFFu;
    *((uint32_t*)result + 1) = f_result >> 32ull;
#endif
}

inline half2 ConvertFloat2ToHalf2(const float* float2)
{
#if defined(AX_SUPPORT_NEON)
    float32x2_t x = vld1_dup_f32(float2);
    float32x4_t x4 = vcombine_f32(x, x);
    return vget_lane_u32(vreinterpret_u32_f16(vcvt_f16_f32(x4)), 0);
#elif defined(AX_SUPPORT_AVX)
    return _mm_extract_epi32(_mm_cvtps_ph(_mm_set_ss(float2), 0), 0);
#endif
    half2 result;
    result  = ConvertFloatToHalf(float2[0]);
    result |= (uint32_t)ConvertFloatToHalf(float2[1]) << 16;
    return result;
}

// input half4 is 4x 16 bit integer for example it can be uint64_t
inline void ConvertHalf4ToFloat4(float* result, const void* half4) 
{
#ifdef AX_SUPPORT_AVX2
    _mm_storeu_ps(result, _mm_cvtph_ps(_mm_loadu_si64(half4)));

#elif defined(AX_SUPPORT_NEON)
    uint32x2_t  x = vld1_dup_u32((const unsigned int*)half4);
    float16x4_t x4 = vreinterpret_f16_u32(x);
    vst1q_f32(result, vcvt_f32_f16(x4));

#elif defined(AX_SUPPORT_SSE)
    vecu_t h4 = VeciLoad64(half4);
    h4 = VeciUnpackLow16(h4, VeciZero());   // [half4.xy, half4.xy, half4.zw, half4.zw] 
    
    vecu_t h_e = VeciAnd(h4, VeciSet1(0x00007c00));
    vecu_t h_m = VeciAnd(h4, VeciSet1(0x000003ff));
    vecu_t h_s = VeciAnd(h4, VeciSet1(0x00008000));
    vecu_t h_e_f_bias = VeciAdd(h_e, VeciSet1(0x0001c000));
    
    vecu_t f_s  = VeciSll32(h_s, 0x00000010);
    vecu_t f_e  = VeciSll32(h_e_f_bias, 0x0000000d);
    vecu_t f_m  = VeciSll32(h_m, 0x0000000d);
    vecu_t f_em = VeciOr(f_e, f_m);

    vecu_t i_result = VeciOr(f_s, f_em);
    VecStore(result, VeciToVecf(i_result));
    
#else // no intrinsics
    ConvertHalf2ToFloat2(result, *(uint32_t*)half4);
    ConvertHalf2ToFloat2(result + 2, *((uint32_t*)(half4) + 1));
#endif
}

// note that no nan, inf and overflow check. only underflow check
void ConvertFloat4ToHalf4(half* result, const float* float4)
{
#ifdef AX_SUPPORT_AVX2

    *((long long*)result) = _mm_extract_epi64(_mm_cvtps_ph(_mm_loadu_ps(float4), _MM_FROUND_TO_NEAREST_INT), 0);

#elif defined(AX_SUPPORT_NEON)

    *(float16x4_t*)result = vcvt_f16_f32(vld1q_f32(float4));

#elif defined(AX_SUPPORT_AVX2) || defined(__ARM_NEON__)

    vecu_t IValue = VeciLoad((const unsigned int*)float4);
    vecu_t Sign = VeciSrl32(VeciAnd(IValue, VeciSet1(0x80000000u)), 16);
    IValue = VeciAnd(IValue, VeciSet1(0x7FFFFFFFu));      // Hack off the sign
    
    vecu_t mask = VeciCmpLt(IValue, VeciSet1(0x38800000u));
    vecu_t b = VeciAdd(IValue, VeciSet1(0xC8000000u));
    vecu_t a = VeciOr(VeciSet1(0x800000u), VeciAnd(IValue, VeciSet1(0x7FFFFFu)));
    a = VeciSrl(a, VeciSub(VeciSet1(113u), VeciSrl32(IValue, 23u)));
    
    IValue = VeciBlend(b, a, mask);

    vecu_t Result = VeciAdd(IValue, VeciSet1(0x0FFFu));
    Result = VeciAdd(Result, VeciAnd(VeciSrl32(IValue, 13u), VeciSet1(1u)));
    Result = VeciSrl32(Result, 13u);
    Result = VeciAnd(Result, VeciSet1(0x7FFFu));
    Result = VeciOr(Result, Sign);

    #ifdef AX_SUPPORT_SSE
        const int shufleMask = MakeShuffleMask(0, 2, 1, 3);
        __m128i lo = _mm_shufflelo_epi16(Result, shufleMask);
        __m128i hi = _mm_shufflehi_epi16(lo, shufleMask);
        Result = _mm_shuffle_epi32(hi, shufleMask);
        *((long long*)result) = _mm_extract_epi64(Result, 0);
    #else
        // todo test
        // Narrow the 32-bit to 16-bit, effectively extracting the lower 16 bits of each element
        uint16x4_t low16_bits = vmovn_u32(Result);  // Narrow to 16 bits per element
        // Directly cast the `uint16x4_t` to `uint64_t`
        vst1_u64((uint64_t*)result, vreinterpret_u64_u16(low16_bits));
    #endif

#else // no intrinsics
        ConvertFloat2ToHalf2(result, float4);
        ConvertFloat2ToHalf2(result + 2, float4 + 2);
#endif
}


#ifdef AX_SUPPORT_AVX2

// convert 8 float and half with one instruction
#define ConvertFloat8ToHalf8(result, float8) _mm_storeu_si128((__m128i*)result, _mm256_cvtps_ph(_mm256_loadu_ps(float8), _MM_FROUND_TO_NEAREST_INT))

#define ConvertHalf8ToFloat8(result, half8)  _mm256_storeu_ps(result,  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)half8)))

#else

inline void ConvertHalf8ToFloat8(float* float8, const void* half8)
{
    ConvertHalf4ToFloat4(float8    , half8);
    ConvertHalf4ToFloat4(float8 + 4, (const uint64_t*)half8 + 1);
}

inline void ConvertFloat8ToHalf8(half* result, const float* float8)
{
    ConvertFloat4ToHalf4((half*)result, float8);
    ConvertFloat4ToHalf4((half*)result + 4, float8 + 4);
}

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

#endif // Half_FP16_Fast
