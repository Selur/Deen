// SPDX-License-Identifier: MIT
// Generic SIMD kernels of Deen, written once against a small vector interface S (kernels_sse41.cpp, kernels_avx2.cpp,
// kernels_neon.cpp). Include inside an anonymous namespace after defining S.
//
// Bit-exact to the C reference: 8 bit runs the MMX arithmetic of the original in 16-bit lanes (sums never exceed
// 16 bits), 9-16 bit sums run in 32-bit lanes and the divisions of the c and a modes use the scalar formulas per lane.
// Each row runs in vectors from column 0, the last vector overlapping the previous one; rows narrower than one
// vector run the scalar formula.
//
// Interface S: V8 (N8 bytes), V16 (N16 = N8 / 2 uint16), V32 (N16 / 2 uint32). Widening and packing may reorder lanes
// as long as pack16(lo8(v), hi8(v)) == v and pack32(lo16(v), hi16(v)) == v.
//   load8/store8, load16/store16, load32/store32, zero8/zero16/zero32, set8/set16/set32,
//   absdiff8, le8 (0xFF where a <= b), select8 (m ? a : b), and8, sub8, adds8,
//   lo8/hi8 (V8 -> V16), pack16 (unsigned saturation), add16, adds16, mulhi16, shl16<k>, shr16<k>,
//   absdiff16, le16, select16, and16, sub16, lo16/hi16 (V16 -> V32), pack32 (unsigned saturation), add32,
//   shr32<k>, mullo32lo/mullo32hi (u16 x u16 -> u32 products in lo16/hi16 order).

constexpr int kMaxTaps = 3 * 15 * 15;

template <typename F>
static inline void forRow(int width, int n, F&& step) {
  int x = 0;
  for (; x + n <= width; x += n) step(x);
  if (x < width) step(width - n);
}

template <int F>
static void filter8(const PlaneArgs& a) {
  using V8 = typename S::V8;
  using V16 = typename S::V16;
  constexpr int N = S::N8;
  const int r = a.rad, side = 2 * r + 1, taps = side * side;
  const int total = taps * a.nsrc;
  static thread_local V8 tv[kMaxTaps];
  static thread_local V16 wv[kMaxTaps];
  for (int s = 0, k = 0; s < a.nsrc; s++)
    for (int t = 0; t < taps; t++, k++) {
      tv[k] = S::set8(a.thr[s][t]);
      if (F == kWMode) wv[k] = S::set16(a.weight[s][t]);
    }
  const V16 addc = S::set16((total + 1) / 2), recc = S::set16(int(65536 / total));
  const V16 one16 = S::set16(1), w128 = S::set16(128);
  const V8 one8 = S::set8(1);
  alignas(64) uint16_t tmp[S::N16];
  const uint16_t* rt = recipTable();

  for (int y = a.y0; y < a.y1; y++) {
    if (a.width < N) {
      filterRowScalar<uint8_t>(a, y, 0, a.width);
      continue;
    }
    const uint8_t* rows[3];
    for (int s = 0; s < a.nsrc; s++) rows[s] = a.src[s] + (y - r) * a.stride[s] - r;
    const uint8_t* crow = a.src[0] + y * a.stride[0];
    uint8_t* drow = a.dst + y * a.dst_stride;
    forRow(a.width, N, [&](int x) {
      const V8 c = S::load8(crow + x);
      V16 sl = S::zero16(), sh = S::zero16();
      V8 cnt = S::zero8();
      int k = 0;
      for (int s = 0; s < a.nsrc; s++) {
        const uint8_t* p = rows[s] + x;
        for (int dy = 0; dy < side; dy++, p += a.stride[s]) {
          for (int dx = 0; dx < side; dx++, k++) {
            const V8 n = S::load8(p + dx);
            const V8 m = S::le8(S::absdiff8(c, n), tv[k]);
            switch (F) {
              case kCMode: {
                const V8 v = S::select8(m, n, c);
                sl = S::add16(sl, S::lo8(v));
                sh = S::add16(sh, S::hi8(v));
                break;
              }
              case kAMode: {
                const V8 v = S::and8(m, n);
                cnt = S::sub8(cnt, m);
                sl = S::add16(sl, S::lo8(v));
                sh = S::add16(sh, S::hi8(v));
                break;
              }
              default: {
                const V8 v = S::select8(m, n, c);
                sl = S::adds16(sl, S::mulhi16(S::template shl16<8>(S::lo8(v)), wv[k]));
                sh = S::adds16(sh, S::mulhi16(S::template shl16<8>(S::hi8(v)), wv[k]));
                break;
              }
            }
          }
        }
      }
      V8 out;
      switch (F) {
        case kCMode:
          out = S::pack16(S::mulhi16(S::add16(sl, addc), recc), S::mulhi16(S::add16(sh, addc), recc));
          break;
        case kAMode: {
          const V16 cl = S::lo8(cnt), ch = S::hi8(cnt);
          S::store16(tmp, cl);
          for (int i = 0; i < S::N16; i++) tmp[i] = rt[tmp[i]];
          const V16 rl = S::load16(tmp);
          S::store16(tmp, ch);
          for (int i = 0; i < S::N16; i++) tmp[i] = rt[tmp[i]];
          const V16 rh = S::load16(tmp);
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

template <int F>
static void filter16(const PlaneArgs& a) {
  using V16 = typename S::V16;
  using V32 = typename S::V32;
  constexpr int N = S::N16, N32 = S::N16 / 2;
  const int r = a.rad, side = 2 * r + 1, taps = side * side;
  const uint32_t total = uint32_t(taps * a.nsrc);
  const uint32_t maxv = (1u << a.bits) - 1;
  static thread_local V16 tv[kMaxTaps], wv[kMaxTaps];
  for (int s = 0, k = 0; s < a.nsrc; s++)
    for (int t = 0; t < taps; t++, k++) {
      tv[k] = S::set16(a.thr[s][t]);
      if (F == kWMode) wv[k] = S::set16(a.weight[s][t]);
    }
  const V32 round = S::set32(0x8000);
  alignas(64) uint32_t ts[2][N32], tc[2][N32];

  for (int y = a.y0; y < a.y1; y++) {
    if (a.width < N) {
      filterRowScalar<uint16_t>(a, y, 0, a.width);
      continue;
    }
    const uint8_t* rows[3];
    for (int s = 0; s < a.nsrc; s++) rows[s] = a.src[s] + (y - r) * a.stride[s] - 2 * r;
    const uint8_t* crow = a.src[0] + y * a.stride[0];
    uint8_t* drow = a.dst + y * a.dst_stride;
    forRow(a.width, N, [&](int x) {
      const V16 c = S::load16(reinterpret_cast<const uint16_t*>(crow) + x);
      V32 sl = S::zero32(), sh = S::zero32();
      V16 cnt = S::zero16();
      int k = 0;
      for (int s = 0; s < a.nsrc; s++) {
        const uint8_t* p = rows[s] + 2 * x;
        for (int dy = 0; dy < side; dy++, p += a.stride[s]) {
          const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
          for (int dx = 0; dx < side; dx++, k++) {
            const V16 n = S::load16(q + dx);
            const V16 m = S::le16(S::absdiff16(c, n), tv[k]);
            switch (F) {
              case kCMode: {
                const V16 v = S::select16(m, n, c);
                sl = S::add32(sl, S::lo16(v));
                sh = S::add32(sh, S::hi16(v));
                break;
              }
              case kAMode: {
                const V16 v = S::and16(m, n);
                cnt = S::sub16(cnt, m);
                sl = S::add32(sl, S::lo16(v));
                sh = S::add32(sh, S::hi16(v));
                break;
              }
              default: {
                const V16 v = S::select16(m, n, c);
                sl = S::add32(sl, S::mullo32lo(v, wv[k]));
                sh = S::add32(sh, S::mullo32hi(v, wv[k]));
                break;
              }
            }
          }
        }
      }
      V16 out;
      if (F == kWMode) {
        out = S::pack32(S::template shr32<16>(S::add32(sl, round)), S::template shr32<16>(S::add32(sh, round)));
      } else {
        S::store32(ts[0], sl);
        S::store32(ts[1], sh);
        if (F == kAMode) {
          S::store32(tc[0], S::lo16(cnt));
          S::store32(tc[1], S::hi16(cnt));
          for (int h = 0; h < 2; h++)
            for (int i = 0; i < N32; i++) ts[h][i] = finishA16(ts[h][i], tc[h][i]);
        } else {
          for (int h = 0; h < 2; h++)
            for (int i = 0; i < N32; i++) ts[h][i] = finishDiv16(ts[h][i], total);
        }
        out = S::pack32(S::load32(ts[0]), S::load32(ts[1]));
      }
      (void)maxv;
      S::store16(reinterpret_cast<uint16_t*>(drow) + x, out);
    });
  }
}

static void filterSimd(const PlaneArgs& a) {
  switch (a.family) {
    case kCMode: return a.bits == 8 ? filter8<kCMode>(a) : filter16<kCMode>(a);
    case kAMode: return a.bits == 8 ? filter8<kAMode>(a) : filter16<kAMode>(a);
    default: return a.bits == 8 ? filter8<kWMode>(a) : filter16<kWMode>(a);
  }
}

static const Kernels& simdKernels() {
  static const Kernels k = {filterSimd};
  return k;
}
