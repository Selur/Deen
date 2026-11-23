// SPDX-License-Identifier: MIT
// AVX2 kernels (32 pixels per step at 8 bit, 16 at 9-16 bit). Widening and packing work per 128-bit lane, which keeps
// pack(lo(v), hi(v)) == v.
#include <immintrin.h>

#include "kernels.hpp"
#include "scalar.hpp"

namespace deen {
namespace {

struct S {
  using V8 = __m256i;
  using V16 = __m256i;
  using V32 = __m256i;
  static constexpr int N8 = 32, N16 = 16;

  static V8 load8(const uint8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static void store8(uint8_t* p, V8 v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
  static V16 load16(const uint16_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static void store16(uint16_t* p, V16 v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
  static V32 load32(const uint32_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static void store32(uint32_t* p, V32 v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
  static V8 zero8() { return _mm256_setzero_si256(); }
  static V16 zero16() { return _mm256_setzero_si256(); }
  static V32 zero32() { return _mm256_setzero_si256(); }
  static V8 set8(int v) { return _mm256_set1_epi8(char(v)); }
  static V16 set16(int v) { return _mm256_set1_epi16(short(v)); }
  static V32 set32(int v) { return _mm256_set1_epi32(v); }

  static V8 absdiff8(V8 a, V8 b) { return _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a)); }
  static V8 le8(V8 a, V8 b) { return _mm256_cmpeq_epi8(_mm256_subs_epu8(a, b), _mm256_setzero_si256()); }
  static V8 select8(V8 m, V8 a, V8 b) { return _mm256_blendv_epi8(b, a, m); }
  static V8 and8(V8 a, V8 b) { return _mm256_and_si256(a, b); }
  static V8 sub8(V8 a, V8 b) { return _mm256_sub_epi8(a, b); }
  static V8 adds8(V8 a, V8 b) { return _mm256_adds_epu8(a, b); }
  static V16 lo8(V8 v) { return _mm256_unpacklo_epi8(v, _mm256_setzero_si256()); }
  static V16 hi8(V8 v) { return _mm256_unpackhi_epi8(v, _mm256_setzero_si256()); }
  static V8 pack16(V16 a, V16 b) { return _mm256_packus_epi16(a, b); }
  static V16 add16(V16 a, V16 b) { return _mm256_add_epi16(a, b); }
  static V16 adds16(V16 a, V16 b) { return _mm256_adds_epu16(a, b); }
  static V16 mulhi16(V16 a, V16 b) { return _mm256_mulhi_epu16(a, b); }
  template <int K>
  static V16 shl16(V16 v) { return _mm256_slli_epi16(v, K); }
  template <int K>
  static V16 shr16(V16 v) { return _mm256_srli_epi16(v, K); }

  static V16 absdiff16(V16 a, V16 b) { return _mm256_or_si256(_mm256_subs_epu16(a, b), _mm256_subs_epu16(b, a)); }
  static V16 le16(V16 a, V16 b) { return _mm256_cmpeq_epi16(_mm256_subs_epu16(a, b), _mm256_setzero_si256()); }
  static V16 select16(V16 m, V16 a, V16 b) { return _mm256_blendv_epi8(b, a, m); }
  static V16 and16(V16 a, V16 b) { return _mm256_and_si256(a, b); }
  static V16 sub16(V16 a, V16 b) { return _mm256_sub_epi16(a, b); }
  static V32 lo16(V16 v) { return _mm256_unpacklo_epi16(v, _mm256_setzero_si256()); }
  static V32 hi16(V16 v) { return _mm256_unpackhi_epi16(v, _mm256_setzero_si256()); }
  static V16 pack32(V32 a, V32 b) { return _mm256_packus_epi32(a, b); }
  static V32 add32(V32 a, V32 b) { return _mm256_add_epi32(a, b); }
  template <int K>
  static V32 shr32(V32 v) { return _mm256_srli_epi32(v, K); }
  static V32 mullo32lo(V16 a, V16 b) {
    return _mm256_unpacklo_epi16(_mm256_mullo_epi16(a, b), _mm256_mulhi_epu16(a, b));
  }
  static V32 mullo32hi(V16 a, V16 b) {
    return _mm256_unpackhi_epi16(_mm256_mullo_epi16(a, b), _mm256_mulhi_epu16(a, b));
  }
};

#include "kernels_simd.hpp"

}  // namespace

const Kernels& kernelsAVX2() { return simdKernels(); }

}  // namespace deen
