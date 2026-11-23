// SPDX-License-Identifier: MIT
// Generic SIMD kernels of Deen, written once against a small vector interface S (kernels_sse41.cpp, kernels_avx2.cpp,
// kernels_neon.cpp). Include inside an anonymous namespace after defining S.
//
// Bit-exact to the C reference: 8 bit runs the MMX arithmetic of the original in 16-bit lanes (sums never exceed
// 16 bits), 9-16 bit sums run in 32-bit lanes. The divisions of the c and a modes (9-16 bit) and the reciprocals of the
// 8-bit a modes are float divisions, which are exact here (see exactDiv). Each row runs in vectors from column 0, the
// last vector overlapping the previous one; rows narrower than one vector run the scalar formula. The window rows come
// from RowWindow (rows.hpp).
//
// Interface S: V8 (N8 bytes), V16 (N16 = N8 / 2 uint16), V32 (N16 / 2 uint32), VF (N16 / 2 float). Widening and packing
// may reorder lanes as long as pack16(lo8(v), hi8(v)) == v and pack32(lo16(v), hi16(v)) == v.
//   load8/store8, load16/store16, load32/store32, zero8/zero16/zero32, set8/set16/set32/setf,
//   absdiff8, le8 (0xFF where a <= b), select8 (m ? a : b), and8, sub8, adds8, subs8, inrange8 (0xFF where
//   lo <= n <= hi), lo8/hi8 (V8 -> V16), lo8x256/hi8x256 (the same times 256), addpair8 (sl/sh += lo8/hi8 of
//   a + b), pack16 (unsigned saturation), add16, adds16, subs16, mulhi16, shl16<k>, shr16<k>, absdiff16, le16,
//   inrange16, select16, and16, sub16, lo16/hi16 (V16 -> V32), pack32 (unsigned saturation), add32, shl32<k>,
//   shr32<k>, mullo32lo/mullo32hi (u16 x u16 -> u32 products in lo16/hi16 order), cvtf (u32 < 2^24 -> float), divf
//   (IEEE division), cvti (float -> u32, truncating).
//   kSplit16: true = 9-16 bit c/a sums in two 16-bit parts (3 instructions per tap), false = widened to 32 bit
//   (lo16/hi16 + add32: one widening add per half on NEON).
//   kMadd: true = w modes at 9-16 bit add two taps per maddpair16(wl, wh, v1, v2, w) (signed pmaddwd of v - 2^15
//   interleaved, w = w1 | w2 << 16), false = mullo32lo/hi per tap.
//   kRange: true = a sample passes when lo <= n <= hi with lo = c - t, hi = c + t (saturating, computed once per
//   threshold value and vector; needs subs8, inrange8, subs16, inrange16), false = when |c - n| <= t (absdiff, le,
//   with one threshold vector per tap).

constexpr int kMaxTaps = 3 * 15 * 15;

// S as a dependent type: the if constexpr branches an S does not use (kRange, kMadd) name operations it need not
// define, which only works when the name is looked up at instantiation.
template <int>
struct Dep {
  using T = S;
};

template <typename F>
static inline void forRow(int width, int n, F&& step) {
  int x = 0;
  for (; x + n <= width; x += n) step(x);
  if (x < width) step(width - n);
}

// The thresholds of a plane grouped by value: the window taps share few distinct values (one per source in the c
// and w modes, a handful in the a modes), and the range bounds are computed once per value and vector.
struct ThrClasses {
  int cls[kMaxTaps];  // class of every tap, all sources in a row
  int value[kMaxTaps];
  int count = 0;

  void build(const PlaneArgs& a, int taps) {
    count = 0;
    for (int s = 0, k = 0; s < a.nsrc; s++)
      for (int t = 0; t < taps; t++, k++) {
        int c = 0;
        while (c < count && value[c] != a.thr[s][t]) c++;
        if (c == count) value[count++] = a.thr[s][t];
        cls[k] = c;
      }
  }
};

// floor(x / d) for integers 0 <= x < 2^24 and 1 <= d <= 255 with floor(x / d) < 2^16, as trunc(float(x) / float(d)):
// x and d are exact floats; if x / d is an integer the division is exact, otherwise it lies at least 1/d >= 1/255
// below the next integer, while the float result is off by less than one ulp (2^-8 below 2^16) in any rounding mode.
template <typename V32>
static inline V32 exactDiv(V32 x, typename S::VF d) {
  return S::cvti(S::divf(S::cvtf(x), d));
}

template <int F, bool PS, int NS>
static void filter8(const PlaneArgs& a, const ThrClasses& tc) {
  using V8 = typename S::V8;
  using V16 = typename S::V16;
  constexpr int N = S::N8;
  using SX = typename Dep<F>::T;
  const int r = a.rad, side = 2 * r + 1, taps = side * side;
  const int total = taps * a.nsrc;
  static thread_local V8 tv_[kMaxTaps], lo_[kMaxTaps], hi_[kMaxTaps];
  static thread_local V16 wv_[kMaxTaps];
  V8* const tv = tv_;
  [[maybe_unused]] V8* const lo = lo_;
  [[maybe_unused]] V8* const hi = hi_;
  V16* const wv = wv_;
  // kRange: one entry per threshold class; else one per tap (no class lookup in the tap loop).
  if constexpr (S::kRange)
    for (int c = 0; c < tc.count; c++) tv[c] = S::set8(tc.value[c]);
  else
    for (int k = 0; k < total; k++) tv[k] = S::set8(tc.value[tc.cls[k]]);
  if (F == kWMode)
    for (int s = 0, k = 0; s < a.nsrc; s++)
      for (int t = 0; t < taps; t++, k++) wv[k] = S::set16(a.weight[s][t]);
  const int* const cls = tc.cls;
  const V16 addc = S::set16((total + 1) / 2), recc = S::set16(int(65536 / total));
  const V16 one16 = S::set16(1), w128 = S::set16(128);
  const V8 one8 = S::set8(1);
  const typename S::VF k65536 = S::setf(65536.0f);
  const uint8_t* tp[kMaxTaps];
  RowWindow win(a, 1);

  for (int y = a.y0; y < a.y1; y++) {
    win.moveTo(y);
    const uint8_t* const* rows = win.rows();
    uint8_t* drow = a.dst + y * a.dst_stride;
    if (a.width < N) {
      filterRowScalar<uint8_t>(a, rows, drow, 0, a.width);
      continue;
    }
    // Every tap as a pointer at column 0, in table order (source, dy, dx).
    for (int s = 0, k = 0; s < a.nsrc; s++)
      for (int dy = 0; dy < side; dy++)
        for (int dx = 0; dx < side; dx++, k++) tp[k] = rows[s * side + dy] + dx - r;
    const uint8_t* crow = rows[r];
    forRow(a.width, N, [&](int x) {
      const V8 c = S::load8(crow + x);
      // General thresholds (kRange): the bounds of every class, once per vector.
      if constexpr (S::kRange)
        if (!PS)
          for (int k = 0; k < tc.count; k++) {
            lo[k] = SX::subs8(c, tv[k]);
            hi[k] = S::adds8(c, tv[k]);
          }
      V16 sl = S::zero16(), sh = S::zero16();
      V8 cnt = S::zero8();
      // PS: the taps source by source (NS == a.nsrc), else all taps in one group.
      for (int s = 0; s < (PS ? NS : 1); s++) {
        const int k0 = PS ? s * taps : 0, k1 = PS ? k0 + taps : total;
        // PS: one threshold for all taps of the group (c and w modes), kept in registers.
        const V8 ts = tv[S::kRange ? cls[k0] : k0];
        [[maybe_unused]] V8 los, his;
        if constexpr (S::kRange)
          if (PS) {
            los = SX::subs8(c, ts);
            his = S::adds8(c, ts);
          }
        auto pass = [&](V8 n, int k) {
          if constexpr (S::kRange)
            return PS ? SX::inrange8(n, los, his) : SX::inrange8(n, lo[cls[k]], hi[cls[k]]);
          else
            return S::le8(S::absdiff8(c, n), PS ? ts : tv[k]);
        };
        if (F == kWMode) {
          for (int k = k0; k < k1; k++) {
            const V8 n = S::load8(tp[k] + x);
            const V8 v = S::select8(pass(n, k), n, c);
            sl = S::adds16(sl, S::mulhi16(S::lo8x256(v), wv[k]));
            sh = S::adds16(sh, S::mulhi16(S::hi8x256(v), wv[k]));
          }
        } else {
          // c and a modes: plain sums, two taps per widening, the last (odd) tap of a group alone.
          auto value = [&](V8 n, int k) {
            const V8 m = pass(n, k);
            if (F == kAMode) {
              cnt = S::sub8(cnt, m);
              return S::and8(m, n);
            }
            return S::select8(m, n, c);
          };
          int k = k0;
          for (; k + 1 < k1; k += 2) {
            const V8 v1 = value(S::load8(tp[k] + x), k);
            const V8 v2 = value(S::load8(tp[k + 1] + x), k + 1);
            S::addpair8(sl, sh, v1, v2);
          }
          if (k < k1) {
            const V8 v = value(S::load8(tp[k] + x), k);
            sl = S::add16(sl, S::lo8(v));
            sh = S::add16(sh, S::hi8(v));
          }
        }
      }
      V8 out;
      switch (F) {
        case kCMode:
          out = S::pack16(S::mulhi16(S::add16(sl, addc), recc), S::mulhi16(S::add16(sh, addc), recc));
          break;
        case kAMode: {
          // R(n) = floor(65536 / n) (exact, see exactDiv: below 2^15 for n >= 3), R(1) = 65536 saturates to 0xFFFF.
          const V16 cl = S::lo8(cnt), ch = S::hi8(cnt);
          const V16 rl = S::pack32(S::cvti(S::divf(k65536, S::cvtf(S::lo16(cl)))),
                                   S::cvti(S::divf(k65536, S::cvtf(S::hi16(cl)))));
          const V16 rh = S::pack32(S::cvti(S::divf(k65536, S::cvtf(S::lo16(ch)))),
                                   S::cvti(S::divf(k65536, S::cvtf(S::hi16(ch)))));
          const V16 al = S::template shr16<1>(S::add16(cl, one16)), ah = S::template shr16<1>(S::add16(ch, one16));
          out = S::pack16(S::mulhi16(S::add16(sl, al), rl), S::mulhi16(S::add16(sh, ah), rh));
          break;
        }
        default:
          out = S::adds8(S::pack16(S::template shr16<8>(S::add16(sl, w128)), S::template shr16<8>(S::add16(sh, w128))),
                         one8);
          break;
      }
      S::store8(drow + x, out);
    });
  }
}

template <int F, bool PS, int NS>
static void filter16(const PlaneArgs& a, const ThrClasses& tc) {
  using V16 = typename S::V16;
  using V32 = typename S::V32;
  using VF = typename S::VF;
  constexpr int N = S::N16;
  using SX = typename Dep<F>::T;
  const int r = a.rad, side = 2 * r + 1, taps = side * side;
  const int total = taps * a.nsrc;
  static thread_local V16 tv_[kMaxTaps], lo_[kMaxTaps], hi_[kMaxTaps], wv_[kMaxTaps];
  V16* const tv = tv_;
  [[maybe_unused]] V16* const lo = lo_;
  [[maybe_unused]] V16* const hi = hi_;
  V16* const wv = wv_;
  // kRange: one entry per threshold class; else one per tap (no class lookup in the tap loop).
  if constexpr (S::kRange)
    for (int c = 0; c < tc.count; c++) tv[c] = S::set16(tc.value[c]);
  else
    for (int k = 0; k < total; k++) tv[k] = S::set16(tc.value[tc.cls[k]]);
  if (F == kWMode)
    for (int s = 0, k = 0; s < a.nsrc; s++)
      for (int t = 0; t < taps; t++, k++) wv[k] = S::set16(a.weight[s][t]);
  const int* const cls = tc.cls;
  const V32 round = S::set32(0x8000), addc = S::set32((total + 1) / 2);
  const VF ftotal = S::setf(float(total));
  const V16 zero16 = S::zero16(), one16 = S::set16(1);
  // w modes with every weight below 2^15 (all real tables): two taps per signed multiply-add (S::kMadd) of v - 2^15,
  // the bias sum of 2^15 * w is added back (mod 2^32; the true sum is below 2^32). The taps of a group run in pairs
  // (k, k+1) from its first tap; an odd last tap runs unsigned, without bias.
  static thread_local V32 wp_[kMaxTaps];
  [[maybe_unused]] V32* const wp = wp_;
  [[maybe_unused]] bool madd = false;
  uint32_t biasv = 0;
  if (F == kWMode && S::kMadd) {
    madd = true;
    const int group = PS ? taps : total;
    for (int k = 0; k < total; k++) {
      const uint32_t w = a.weight[k / taps][k % taps];
      if (w >= 0x8000) madd = false;
      if (k % group != group - 1 || group % 2 == 0) biasv += w << 15;
      if (k + 1 < total) wp[k] = S::set32(int(w | (uint32_t(a.weight[(k + 1) / taps][(k + 1) % taps]) << 16)));
    }
  }
  [[maybe_unused]] const V32 bias = S::set32(int(biasv));
  const uint16_t* tp[kMaxTaps];
  RowWindow win(a, 2);

  for (int y = a.y0; y < a.y1; y++) {
    win.moveTo(y);
    const uint8_t* const* rows = win.rows();
    uint8_t* drow = a.dst + y * a.dst_stride;
    if (a.width < N) {
      filterRowScalar<uint16_t>(a, rows, drow, 0, a.width);
      continue;
    }
    for (int s = 0, k = 0; s < a.nsrc; s++)
      for (int dy = 0; dy < side; dy++)
        for (int dx = 0; dx < side; dx++, k++)
          tp[k] = reinterpret_cast<const uint16_t*>(rows[s * side + dy]) + dx - r;
    const uint16_t* crow = reinterpret_cast<const uint16_t*>(rows[r]);
    forRow(a.width, N, [&](int x) {
      const V16 c = S::load16(crow + x);
      // General thresholds (kRange): the bounds of every class, once per vector.
      if constexpr (S::kRange)
        if (!PS)
          for (int k = 0; k < tc.count; k++) {
            lo[k] = SX::subs16(c, tv[k]);
            hi[k] = S::adds16(c, tv[k]);
          }
      // c and a modes, kSplit16: the sums in two 16-bit parts, sa = sum mod 2^16 and sh = sum of (v >> 8). Both the
      // sum of (v >> 8) and the sum of (v & 0xFF) stay below 2^16 (at most 243 samples of 255), so
      // sum = (sh << 8) + ((sa - (sh << 8)) mod 2^16). Without kSplit16 the sums run in wl/wh (32 bit).
      [[maybe_unused]] V16 sa = S::zero16(), sh = S::zero16();
      V16 cnt = S::zero16();
      V32 wl = S::zero32(), wh = S::zero32();  // w modes (and c/a without kSplit16)
      // PS: the taps source by source (NS == a.nsrc), else all taps in one group.
      for (int s = 0; s < (PS ? NS : 1); s++) {
        const int k0 = PS ? s * taps : 0, k1 = PS ? k0 + taps : total;
        // PS: one threshold for all taps of the group (c and w modes), kept in registers.
        const V16 ts = tv[S::kRange ? cls[k0] : k0];
        [[maybe_unused]] V16 los, his;
        if constexpr (S::kRange)
          if (PS) {
            los = SX::subs16(c, ts);
            his = S::adds16(c, ts);
          }
        auto pass = [&](V16 n, int k) {
          if constexpr (S::kRange)
            return PS ? SX::inrange16(n, los, his) : SX::inrange16(n, lo[cls[k]], hi[cls[k]]);
          else
            return S::le16(S::absdiff16(c, n), PS ? ts : tv[k]);
        };
        if (F == kWMode) {
          int k = k0;
          if constexpr (S::kMadd) {
            if (madd)
              for (; k + 1 < k1; k += 2) {
                const V16 n1 = S::load16(tp[k] + x), n2 = S::load16(tp[k + 1] + x);
                SX::maddpair16(wl, wh, S::select16(pass(n1, k), n1, c), S::select16(pass(n2, k + 1), n2, c), wp[k]);
              }
          }
          for (; k < k1; k++) {
            const V16 n = S::load16(tp[k] + x);
            const V16 v = S::select16(pass(n, k), n, c);
            wl = S::add32(wl, S::mullo32lo(v, wv[k]));
            wh = S::add32(wh, S::mullo32hi(v, wv[k]));
          }
        } else {
          for (int k = k0; k < k1; k++) {
            const V16 n = S::load16(tp[k] + x);
            const V16 m = pass(n, k);
            V16 v;
            if (F == kAMode) {
              v = S::and16(m, n);
              cnt = S::sub16(cnt, m);
            } else {
              v = S::select16(m, n, c);
            }
            if constexpr (S::kSplit16) {
              sa = S::add16(sa, v);
              sh = S::add16(sh, S::template shr16<8>(v));
            } else {
              wl = S::add32(wl, S::lo16(v));
              wh = S::add32(wh, S::hi16(v));
            }
          }
        }
      }
      if constexpr (S::kMadd)
        if (F == kWMode && madd) {
          wl = S::add32(wl, bias);
          wh = S::add32(wh, bias);
        }
      V16 out;
      if (F == kWMode) {
        out = S::pack32(S::template shr32<16>(S::add32(wl, round)), S::template shr32<16>(S::add32(wh, round)));
      } else {
        V32 suml = wl, sumh = wh;
        if constexpr (S::kSplit16) {
          const V16 sl8 = S::sub16(sa, S::template shl16<8>(sh));  // sum of (v & 0xFF)
          suml = S::add32(S::template shl32<8>(S::lo16(sh)), S::lo16(sl8));
          sumh = S::add32(S::template shl32<8>(S::hi16(sh)), S::hi16(sl8));
        }
        if (F == kAMode) {
          // (sum + ceil(cnt/2)) / cnt, and sum itself (the centre) for cnt == 1.
          const V16 add = S::select16(S::le16(cnt, one16), zero16, S::template shr16<1>(S::add16(cnt, one16)));
          out = S::pack32(exactDiv(S::add32(suml, S::lo16(add)), S::cvtf(S::lo16(cnt))),
                          exactDiv(S::add32(sumh, S::hi16(add)), S::cvtf(S::hi16(cnt))));
        } else {
          out = S::pack32(exactDiv(S::add32(suml, addc), ftotal), exactDiv(S::add32(sumh, addc), ftotal));
        }
      }
      S::store16(reinterpret_cast<uint16_t*>(drow) + x, out);
    });
  }
}

// PS: every source has one threshold value for all its taps (c and w modes); else per tap (a modes).
template <int F>
static void filterFamily(const PlaneArgs& a) {
  static thread_local ThrClasses tc;
  const int taps = (2 * a.rad + 1) * (2 * a.rad + 1);
  tc.build(a, taps);
  bool ps = true;
  for (int s = 0; s < a.nsrc; s++)
    for (int t = 1; t < taps; t++) ps = ps && tc.cls[s * taps + t] == tc.cls[s * taps];
  const bool d3 = a.nsrc == 3;
  if (a.bits == 8) {
    if (ps)
      d3 ? filter8<F, true, 3>(a, tc) : filter8<F, true, 1>(a, tc);
    else
      d3 ? filter8<F, false, 3>(a, tc) : filter8<F, false, 1>(a, tc);
  } else {
    if (ps)
      d3 ? filter16<F, true, 3>(a, tc) : filter16<F, true, 1>(a, tc);
    else
      d3 ? filter16<F, false, 3>(a, tc) : filter16<F, false, 1>(a, tc);
  }
}

static void filterSimd(const PlaneArgs& a) {
  switch (a.family) {
    case kCMode: return filterFamily<kCMode>(a);
    case kAMode: return filterFamily<kAMode>(a);
    default: return filterFamily<kWMode>(a);
  }
}

static const Kernels& simdKernels() {
  static const Kernels k = {filterSimd};
  return k;
}
