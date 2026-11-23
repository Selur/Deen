// SPDX-License-Identifier: MIT
// SSE4.1 kernels (16 pixels per step at 8 bit, 8 at 9-16 bit).
#include <smmintrin.h>

#include "kernels.hpp"
#include "rows.hpp"
#include "scalar.hpp"

namespace deen {
namespace {

struct S {
  using V8 = __m128i;
  using V16 = __m128i;
  using V32 = __m128i;
  using VF = __m128;
  static constexpr bool kRange = true;
  static constexpr bool kSplit16 = true;
  static constexpr bool kMadd = true;
  static constexpr int N8 = 16, N16 = 8;

  static V8 load8(const uint8_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static void store8(uint8_t* p, V8 v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }
  static V16 load16(const uint16_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static void store16(uint16_t* p, V16 v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }
  static V32 load32(const uint32_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static void store32(uint32_t* p, V32 v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }
  static V8 zero8() { return _mm_setzero_si128(); }
  static V16 zero16() { return _mm_setzero_si128(); }
  static V32 zero32() { return _mm_setzero_si128(); }
  static V8 set8(int v) { return _mm_set1_epi8(char(v)); }
  static V16 set16(int v) { return _mm_set1_epi16(short(v)); }
  static V32 set32(int v) { return _mm_set1_epi32(v); }

  static V8 absdiff8(V8 a, V8 b) { return _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a)); }
  static V8 le8(V8 a, V8 b) { return _mm_cmpeq_epi8(_mm_subs_epu8(a, b), _mm_setzero_si128()); }
  static V8 select8(V8 m, V8 a, V8 b) { return _mm_blendv_epi8(b, a, m); }
  static V8 and8(V8 a, V8 b) { return _mm_and_si128(a, b); }
  static V8 sub8(V8 a, V8 b) { return _mm_sub_epi8(a, b); }
  static V8 adds8(V8 a, V8 b) { return _mm_adds_epu8(a, b); }
  static V16 lo8(V8 v) { return _mm_unpacklo_epi8(v, _mm_setzero_si128()); }
  static V16 hi8(V8 v) { return _mm_unpackhi_epi8(v, _mm_setzero_si128()); }
  static V8 pack16(V16 a, V16 b) { return _mm_packus_epi16(a, b); }
  static V16 add16(V16 a, V16 b) { return _mm_add_epi16(a, b); }
  static V16 adds16(V16 a, V16 b) { return _mm_adds_epu16(a, b); }
  static V16 mulhi16(V16 a, V16 b) { return _mm_mulhi_epu16(a, b); }
  template <int K>
  static V16 shl16(V16 v) { return _mm_slli_epi16(v, K); }
  template <int K>
  static V16 shr16(V16 v) { return _mm_srli_epi16(v, K); }

  static V16 absdiff16(V16 a, V16 b) { return _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a)); }
  static V16 le16(V16 a, V16 b) { return _mm_cmpeq_epi16(_mm_subs_epu16(a, b), _mm_setzero_si128()); }
  static V16 select16(V16 m, V16 a, V16 b) { return _mm_blendv_epi8(b, a, m); }
  static V16 and16(V16 a, V16 b) { return _mm_and_si128(a, b); }
  static V16 sub16(V16 a, V16 b) { return _mm_sub_epi16(a, b); }
  static V32 lo16(V16 v) { return _mm_unpacklo_epi16(v, _mm_setzero_si128()); }
  static V32 hi16(V16 v) { return _mm_unpackhi_epi16(v, _mm_setzero_si128()); }
  static V16 pack32(V32 a, V32 b) { return _mm_packus_epi32(a, b); }
  static V32 add32(V32 a, V32 b) { return _mm_add_epi32(a, b); }
  template <int K>
  static V32 shr32(V32 v) { return _mm_srli_epi32(v, K); }
  static V32 mullo32lo(V16 a, V16 b) { return _mm_unpacklo_epi16(_mm_mullo_epi16(a, b), _mm_mulhi_epu16(a, b)); }
  static V32 mullo32hi(V16 a, V16 b) { return _mm_unpackhi_epi16(_mm_mullo_epi16(a, b), _mm_mulhi_epu16(a, b)); }

  static V16 lo8x256(V8 v) { return _mm_unpacklo_epi8(_mm_setzero_si128(), v); }
  static V16 hi8x256(V8 v) { return _mm_unpackhi_epi8(_mm_setzero_si128(), v); }
  // pmaddubsw of the interleaved bytes with 1: a + b per 16-bit lane, at most 510.
  static void addpair8(V16& sl, V16& sh, V8 a, V8 b) {
    sl = _mm_add_epi16(sl, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), _mm_set1_epi8(1)));
    sh = _mm_add_epi16(sh, _mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), _mm_set1_epi8(1)));
  }
  static void maddpair16(V32& wl, V32& wh, V16 v1, V16 v2, V32 w) {
    const V16 flip = _mm_set1_epi16(short(0x8000));
    const V16 a = _mm_xor_si128(v1, flip), b = _mm_xor_si128(v2, flip);
    wl = _mm_add_epi32(wl, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), w));
    wh = _mm_add_epi32(wh, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), w));
  }
  static VF setf(float v) { return _mm_set1_ps(v); }
  static V8 subs8(V8 a, V8 b) { return _mm_subs_epu8(a, b); }
  static V8 inrange8(V8 n, V8 lo, V8 hi) { return _mm_cmpeq_epi8(_mm_max_epu8(_mm_min_epu8(n, hi), lo), n); }
  static V16 subs16(V16 a, V16 b) { return _mm_subs_epu16(a, b); }
  static V16 inrange16(V16 n, V16 lo, V16 hi) { return _mm_cmpeq_epi16(_mm_max_epu16(_mm_min_epu16(n, hi), lo), n); }
  template <int K>
  static V32 shl32(V32 v) { return _mm_slli_epi32(v, K); }
  static VF cvtf(V32 v) { return _mm_cvtepi32_ps(v); }
  static VF divf(VF a, VF b) { return _mm_div_ps(a, b); }
  static V32 cvti(VF v) { return _mm_cvttps_epi32(v); }
};

#include "kernels_simd.hpp"

}  // namespace

const Kernels& kernelsSSE41() { return simdKernels(); }

}  // namespace deen
