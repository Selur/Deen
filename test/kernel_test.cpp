// SPDX-License-Identifier: MIT
// Compares every available SIMD tier with the C reference kernels on random data: widths 1..300, every mode family,
// 2d and 3d, every radius, 8/10/12/16 bit, random thresholds, real and random weights, random, extreme and flat
// content; and the scene metric against the plain formula.
// Run: meson test -C build (or build/kernel_test). Exit code 0 = all tiers identical to C.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "core/deen.hpp"
#include "core/kernels.hpp"

using namespace deen;

namespace {

std::mt19937 rng(20031);

int rnd(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }

// A plane with a random stride; the samples past the width are filled too (the kernels must not read them).
struct Plane {
  std::vector<uint8_t> buf;
  ptrdiff_t stride = 0;
  const uint8_t* origin = nullptr;
};

Plane makePlane(int w, int h, int bits, int kind) {
  Plane p;
  const int bps = bits == 8 ? 1 : 2;
  p.stride = ptrdiff_t(w + rnd(0, 20)) * bps;
  p.buf.assign(size_t(p.stride) * h, 0);
  const int maxv = (1 << bits) - 1;
  const int base = rnd(0, maxv);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < p.stride / bps; x++) {
      int v;
      switch (kind) {
        case 1: v = rnd(0, 1) ? maxv : 0; break;
        case 2: v = std::min(maxv, std::max(0, base + rnd(-3, 3) * (1 << (bits - 8)))); break;
        case 3: v = base; break;
        default: v = rnd(0, maxv); break;
      }
      uint8_t* row = p.buf.data() + y * p.stride;
      if (bps == 1)
        row[x] = uint8_t(v);
      else
        reinterpret_cast<uint16_t*>(row)[x] = uint16_t(v);
    }
  p.origin = p.buf.data();
  return p;
}

int failures = 0, checks = 0;

}  // namespace

int main() {
  std::vector<std::pair<Isa, const char*>> tiers;
  for (int opt = 1; opt <= 3; opt++) {
    Isa isa;
    if (selectIsa(opt, &isa)) tiers.push_back({isa, isaName(isa)});
  }
  std::printf("tiers:");
  for (auto& t : tiers) std::printf(" %s", t.second);
  std::printf("\n");

  const int bitsList[] = {8, 10, 12, 16};
  for (int iter = 0; iter < 1500; iter++) {
    const int family = iter % 3;
    const bool d3 = (iter / 3) % 2 == 1;
    const int rad = rnd(1, d3 ? 4 : 7);
    const int bits = bitsList[rnd(0, 3)];
    const int w = rnd(0, 3) ? rnd(1, 80) : rnd(81, 300), h = rnd(1, 20);
    const int kind = rnd(0, 3);
    const int nsrc = d3 ? 3 : 1;
    const int taps = (2 * rad + 1) * (2 * rad + 1);
    const int shift = bits - 8;
    Plane src[3];
    std::vector<uint16_t> thr[3], wt[3];
    for (int s = 0; s < nsrc; s++) {
      src[s] = makePlane(w, h, bits, kind);
      if (family == kAMode) {
        thr[s] = adaptiveThresholds(rad, rnd(0, 255), float(rnd(0, 100)) / 100.0f);
      } else {
        thr[s].assign(size_t(taps), uint16_t(rnd(0, 255)));
      }
      for (uint16_t& t : thr[s]) t = uint16_t(t << shift);
      if (rnd(0, 7) == 0)
        for (uint16_t& t : thr[s]) t = uint16_t(rnd(0, (255 << shift)));
      if (family == kWMode) {
        wt[s] = distanceWeights(rad, float(rnd(0, 100)) / 100.0f, d3);
        if (d3 && s == 0)
          for (uint16_t& x : wt[s]) x = uint16_t(x * 2);
        if (rnd(0, 7) == 0 && bits == 8)
          for (uint16_t& x : wt[s]) x = uint16_t(rnd(0, 65535));
      }
    }
    const int bps = bits == 8 ? 1 : 2;
    const ptrdiff_t dstride = ptrdiff_t(w + rnd(0, 16)) * bps;
    std::vector<uint8_t> ref(size_t(dstride) * h, 0xAB), got(size_t(dstride) * h, 0xAB);
    PlaneArgs a;
    for (int s = 0; s < nsrc; s++) {
      a.src[s] = src[s].origin;
      a.stride[s] = src[s].stride;
      a.thr[s] = thr[s].data();
      a.weight[s] = family == kWMode ? wt[s].data() : nullptr;
    }
    a.nsrc = nsrc;
    a.dst_stride = dstride;
    a.width = w;
    a.height = h;
    a.y0 = rnd(0, 1) ? 0 : std::min(h, rad);
    a.y1 = std::max(a.y0, rnd(0, 1) ? h : h - rad);
    a.rad = rad;
    a.bits = bits;
    a.family = family;
    a.dst = ref.data();
    kernelsC().filter(a);
    for (auto& t : tiers) {
      std::fill(got.begin(), got.end(), uint8_t(0xAB));
      a.dst = got.data();
      kernelsFor(t.first).filter(a);
      checks++;
      if (got != ref) {
        if (failures < 30)
          std::printf("FAIL %s family=%d 3d=%d rad=%d bits=%d w=%d h=%d kind=%d\n", t.second, family, int(d3), rad,
                      bits, w, h, kind);
        failures++;
      }
    }
  }
  // Scene metric (Engine::sceneDiff, SIMD sums of |a - b|) against the plain formula: random, extreme and maximal
  // differences.
  for (int iter = 0; iter < 300; iter++) {
    Format f;
    f.bits = bitsList[rnd(0, 3)];
    f.width = rnd(0, 3) ? rnd(2, 80) : rnd(81, 600);
    f.height = rnd(2, 8);
    f.planes = rnd(0, 1) ? 3 : 1;
    f.ssw = rnd(0, 1);
    f.ssh = rnd(0, 1);
    Params p;
    p.opt = 0;
    const Engine e(p, f, 3);
    const int bps = f.bits == 8 ? 1 : 2, maxv = (1 << f.bits) - 1, kind = rnd(0, 2);
    std::vector<uint8_t> buf[2][3];
    FrameView v[2];
    double want = 0.0;
    for (int pl = 0; pl < f.planes; pl++) {
      const int w = pl ? f.width >> f.ssw : f.width, h = pl ? f.height >> f.ssh : f.height;
      const ptrdiff_t stride = ptrdiff_t(w + rnd(0, 9)) * bps;
      for (int i = 0; i < 2; i++) {
        buf[i][pl].assign(size_t(stride) * h, 0);
        v[i].data[pl] = buf[i][pl].data();
        v[i].stride[pl] = stride;
      }
      uint64_t sum = 0;
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
          int s[2];
          for (int i = 0; i < 2; i++) {
            s[i] = kind == 0 ? rnd(0, maxv) : kind == 1 ? (rnd(0, 1) ? maxv : 0) : (i ? maxv : 0);
            uint8_t* row = buf[i][pl].data() + y * stride;
            if (bps == 1)
              row[x] = uint8_t(s[i]);
            else
              reinterpret_cast<uint16_t*>(row)[x] = uint16_t(s[i]);
          }
          sum += uint64_t(std::abs(s[0] - s[1]));
        }
      const double scale = f.bits == 8 ? 1.0 : 255.0 / double(maxv);
      want = want + double(sum) / double(int64_t(w) * h) * scale;
    }
    want = f.planes == 3 ? want * 0.3333333333333333 : want / double(f.planes);
    checks++;
    if (e.sceneDiff(v[0], v[1]) != want) {
      if (failures < 30)
        std::printf("FAIL sceneDiff bits=%d w=%d h=%d planes=%d\n", f.bits, f.width, f.height, f.planes);
      failures++;
    }
  }
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
