// SPDX-License-Identifier: MIT
// NEON kernels for arm64 (16 pixels per step at 8 bit, 8 at 9-16 bit).
#include <arm_neon.h>

#include "kernels.hpp"
#include "rows.hpp"
#include "scalar.hpp"

namespace deen {
namespace {

struct S {
  using V8 = uint8x16_t;
  using V16 = uint16x8_t;
  using V32 = uint32x4_t;
  using VF = float32x4_t;
  // |c - n| <= t is two instructions (vabd, vcle), fewer than the range compare.
  static constexpr bool kRange = false;
  static constexpr bool kSplit16 = false;
  static constexpr bool kMadd = false;
  static constexpr int N8 = 16, N16 = 8;

  static V8 load8(const uint8_t* p) { return vld1q_u8(p); }
  static void store8(uint8_t* p, V8 v) { vst1q_u8(p, v); }
  static V16 load16(const uint16_t* p) { return vld1q_u16(p); }
  static void store16(uint16_t* p, V16 v) { vst1q_u16(p, v); }
  static V32 load32(const uint32_t* p) { return vld1q_u32(p); }
  static void store32(uint32_t* p, V32 v) { vst1q_u32(p, v); }
  static V8 zero8() { return vdupq_n_u8(0); }
  static V16 zero16() { return vdupq_n_u16(0); }
  static V32 zero32() { return vdupq_n_u32(0); }
  static V8 set8(int v) { return vdupq_n_u8(uint8_t(v)); }
  static V16 set16(int v) { return vdupq_n_u16(uint16_t(v)); }
  static V32 set32(int v) { return vdupq_n_u32(uint32_t(v)); }

  static V8 absdiff8(V8 a, V8 b) { return vabdq_u8(a, b); }
  static V8 le8(V8 a, V8 b) { return vcleq_u8(a, b); }
  static V8 select8(V8 m, V8 a, V8 b) { return vbslq_u8(m, a, b); }
  static V8 and8(V8 a, V8 b) { return vandq_u8(a, b); }
  static V8 sub8(V8 a, V8 b) { return vsubq_u8(a, b); }
  static V8 adds8(V8 a, V8 b) { return vqaddq_u8(a, b); }
  static V16 lo8(V8 v) { return vmovl_u8(vget_low_u8(v)); }
  static V16 hi8(V8 v) { return vmovl_high_u8(v); }
  static V8 pack16(V16 a, V16 b) {
    return vcombine_u8(vqmovun_s16(vreinterpretq_s16_u16(a)), vqmovun_s16(vreinterpretq_s16_u16(b)));
  }
  static V16 add16(V16 a, V16 b) { return vaddq_u16(a, b); }
  static V16 adds16(V16 a, V16 b) { return vqaddq_u16(a, b); }
  static V16 mulhi16(V16 a, V16 b) {
    const uint32x4_t lo = vmull_u16(vget_low_u16(a), vget_low_u16(b));
    const uint32x4_t hi = vmull_high_u16(a, b);
    return vcombine_u16(vshrn_n_u32(lo, 16), vshrn_n_u32(hi, 16));
  }
  template <int K>
  static V16 shl16(V16 v) { return vshlq_n_u16(v, K); }
  template <int K>
  static V16 shr16(V16 v) { return vshrq_n_u16(v, K); }

  static V16 absdiff16(V16 a, V16 b) { return vabdq_u16(a, b); }
  static V16 le16(V16 a, V16 b) { return vcleq_u16(a, b); }
  static V16 select16(V16 m, V16 a, V16 b) { return vbslq_u16(m, a, b); }
  static V16 and16(V16 a, V16 b) { return vandq_u16(a, b); }
  static V16 sub16(V16 a, V16 b) { return vsubq_u16(a, b); }
  static V32 lo16(V16 v) { return vmovl_u16(vget_low_u16(v)); }
  static V32 hi16(V16 v) { return vmovl_high_u16(v); }
  static V16 pack32(V32 a, V32 b) { return vcombine_u16(vqmovn_u32(a), vqmovn_u32(b)); }
  static V32 add32(V32 a, V32 b) { return vaddq_u32(a, b); }
  template <int K>
  static V32 shr32(V32 v) { return vshrq_n_u32(v, K); }
  static V32 mullo32lo(V16 a, V16 b) { return vmull_u16(vget_low_u16(a), vget_low_u16(b)); }
  static V32 mullo32hi(V16 a, V16 b) { return vmull_high_u16(a, b); }

  static V16 lo8x256(V8 v) { return vshll_n_u8(vget_low_u8(v), 8); }
  static V16 hi8x256(V8 v) { return vshll_high_n_u8(v, 8); }
  static void addpair8(V16& sl, V16& sh, V8 a, V8 b) {
    sl = vaddq_u16(sl, vaddl_u8(vget_low_u8(a), vget_low_u8(b)));
    sh = vaddq_u16(sh, vaddl_high_u8(a, b));
  }
  static VF setf(float v) { return vdupq_n_f32(v); }
  template <int K>
  static V32 shl32(V32 v) { return vshlq_n_u32(v, K); }
  static VF cvtf(V32 v) { return vcvtq_f32_u32(v); }
  static VF divf(VF a, VF b) { return vdivq_f32(a, b); }
  static V32 cvti(VF v) { return vcvtq_u32_f32(v); }
};

#include "kernels_simd.hpp"

}  // namespace

const Kernels& kernelsNEON() { return simdKernels(); }

}  // namespace deen
