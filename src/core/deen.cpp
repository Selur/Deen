// SPDX-License-Identifier: MIT
#include "deen.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "scalar.hpp"

namespace deen {

std::vector<uint16_t> adaptiveThresholds(int rad, int thr, float min) {
  std::vector<uint16_t> t;
  const double inv = 1.0 / std::sqrt(double(rad * rad) + double(rad * rad));
  const double a = 1.0 - double(min);
  for (int dy = -rad; dy <= rad; dy++)
    for (int dx = -rad; dx <= rad; dx++) {
      const double q = std::sqrt(double(dx * dx) + double(dy * dy)) * inv;
      const int e = int((a * double(thr)) * q);
      t.push_back(uint16_t((thr - (e & 0xFF)) & 0xFF));
    }
  return t;
}

std::vector<uint16_t> distanceWeights(int rad, float min, bool three_d) {
  const int total = three_d ? 0x4000 : 0x10000;
  const int taps = windowTaps(rad);
  const int base = total / taps;
  const int d = base - int(double(min) * double(base));
  const double q = 1.0 / std::sqrt(double(rad * rad) + double(rad * rad));
  std::vector<uint16_t> w;
  for (int dy = -rad; dy <= rad; dy++)
    for (int dx = -rad; dx <= rad; dx++) {
      const double t = std::sqrt(double(dx * dx) + double(dy * dy));
      w.push_back(uint16_t(base - int((t * q) * double(d))));
    }
  uint32_t sum = 0;
  for (uint16_t v : w) sum += v;
  const int scale = int((int64_t(total) << 8) / int64_t(sum));
  uint32_t sum2 = 0;
  for (uint16_t& v : w) {
    v = uint16_t((int(v) * scale) >> 8);
    sum2 += v;
  }
  uint16_t& centre = w[size_t(taps / 2)];
  centre = uint16_t(centre + uint16_t(uint32_t(total) - sum2));
  return w;
}

namespace {

int parseFamily(const std::string& m, bool* three_d) {
  std::string s = m;
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  if (s.size() != 3 || s[2] != 'd' || (s[1] != '2' && s[1] != '3'))
    throw std::runtime_error("Deen: mode must be c2d, c3d, a2d, a3d, w2d or w3d");
  *three_d = s[1] == '3';
  switch (s[0]) {
    case 'c': return kCMode;
    case 'a': return kAMode;
    case 'w': return kWMode;
    default: throw std::runtime_error("Deen: mode must be c2d, c3d, a2d, a3d, w2d or w3d");
  }
}

void checkRange(const char* name, int v, int lo, int hi) {
  if (v < lo || v > hi)
    throw std::runtime_error(std::string("Deen: ") + name + " must be " + std::to_string(lo) + ".." +
                             std::to_string(hi));
}

// Edge-replicated copy of a plane with `pad` samples on every side.
struct Padded {
  std::vector<uint8_t> buf;
  ptrdiff_t stride = 0;
  const uint8_t* origin = nullptr;
};

void padPlane(const uint8_t* src, ptrdiff_t sstride, int w, int h, int pad, int bps, Padded* out) {
  const ptrdiff_t row_bytes = ptrdiff_t(w + 2 * pad) * bps;
  out->stride = (row_bytes + 63) & ~ptrdiff_t(63);
  out->buf.resize(size_t(out->stride) * size_t(h + 2 * pad) + 64);
  uint8_t* base = out->buf.data();
  for (int y = -pad; y < h + pad; y++) {
    const uint8_t* s = src + ptrdiff_t(std::min(std::max(y, 0), h - 1)) * sstride;
    uint8_t* d = base + ptrdiff_t(y + pad) * out->stride;
    std::memcpy(d + pad * bps, s, size_t(w) * bps);
    if (bps == 1) {
      std::memset(d, s[0], size_t(pad));
      std::memset(d + pad + w, s[w - 1], size_t(pad));
    } else {
      const uint16_t* s16 = reinterpret_cast<const uint16_t*>(s);
      uint16_t* d16 = reinterpret_cast<uint16_t*>(d);
      for (int x = 0; x < pad; x++) {
        d16[x] = s16[0];
        d16[pad + w + x] = s16[w - 1];
      }
    }
  }
  out->origin = base + ptrdiff_t(pad) * out->stride + ptrdiff_t(pad) * bps;
}

}  // namespace

Engine::Engine(const Params& p, const Format& f, int num_frames) : fmt_(f), num_frames_(num_frames) {
  family_ = parseFamily(p.mode, &three_d_);
  checkRange("rad", p.rad, 1, three_d_ ? 4 : 7);
  checkRange("thrY", p.thrY, 0, 255);
  checkRange("thrUV", p.thrUV, 0, 255);
  checkRange("tthY", p.tthY, 0, 255);
  checkRange("tthUV", p.tthUV, 0, 255);
  if (!(p.min >= 0.0f && p.min <= 1.0f)) throw std::runtime_error("Deen: min must be 0.0..1.0");
  rad_ = p.rad;
  scd_ = p.scd;
  borderfix_ = p.borderfix;
  if (!selectIsa(p.opt, &isa_)) throw std::runtime_error("Deen: opt=" + std::to_string(p.opt) + " is not available");
  k_ = &kernelsFor(isa_);

  const int shift = f.bits - 8;
  const int thr[3] = {p.thrY, p.thrUV, p.thrUV}, tth[3] = {p.tthY, p.tthUV, p.tthUV};
  for (int v = 0; v < 2; v++) {
    const bool d3 = v == 1;
    if (d3 && !three_d_) continue;
    const int nsrc = d3 ? 3 : 1;
    const int taps = windowTaps(rad_);
    std::vector<uint16_t> w1, w2;
    if (family_ == kWMode) {
      w2 = distanceWeights(rad_, p.min, d3);
      w1 = w2;
      if (d3)
        for (uint16_t& x : w1) x = uint16_t(x * 2);
    }
    for (int pl = 0; pl < 3; pl++) {
      Tables& t = tables_[v][pl];
      for (int s = 0; s < nsrc; s++) {
        const int base = s == 0 ? thr[pl] : tth[pl];
        std::vector<uint16_t> tab =
            family_ == kAMode ? adaptiveThresholds(rad_, base, p.min) : std::vector<uint16_t>(size_t(taps), uint16_t(base));
        for (uint16_t& x : tab) x = uint16_t(x << shift);
        t.thr[s] = std::move(tab);
        if (family_ == kWMode) t.weight[s] = s == 0 ? w1 : w2;
      }
    }
  }
}

void Engine::sourceRange(int n, int* first, int* last) const {
  *first = n;
  *last = n;
  if (three_d_) {
    *first = std::max(0, n - 1);
    *last = std::min(num_frames_ - 1, n + 1);
  }
}

double Engine::sceneDiff(const FrameView& a, const FrameView& b) const {
  double total = 0.0;
  const double scale = fmt_.bits == 8 ? 1.0 : 255.0 / double((1 << fmt_.bits) - 1);
  for (int pl = 0; pl < fmt_.planes; pl++) {
    const int w = pl ? fmt_.width >> fmt_.ssw : fmt_.width;
    const int h = pl ? fmt_.height >> fmt_.ssh : fmt_.height;
    uint64_t sum = 0;
    for (int y = 0; y < h; y++) {
      const uint8_t* ra = a.data[pl] + y * a.stride[pl];
      const uint8_t* rb = b.data[pl] + y * b.stride[pl];
      uint32_t rs = 0;
      if (fmt_.bits == 8) {
        for (int x = 0; x < w; x++) rs += uint32_t(std::abs(int(ra[x]) - int(rb[x])));
        sum += rs;
      } else {
        const uint16_t* pa = reinterpret_cast<const uint16_t*>(ra);
        const uint16_t* pb = reinterpret_cast<const uint16_t*>(rb);
        uint64_t s64 = 0;
        for (int x = 0; x < w; x++) s64 += uint32_t(std::abs(int(pa[x]) - int(pb[x])));
        sum += s64;
      }
    }
    total = total + double(sum) / double(int64_t(w) * h) * scale;
  }
  return fmt_.planes == 3 ? total * 0.3333333333333333 : total / double(fmt_.planes);
}

bool Engine::forced2d(int n, const FrameView& prev, const FrameView& cur, const FrameView& next) const {
  if (n < 1) return true;
  const double s = double(scd_);
  if (s < sceneDiff(cur, next)) return true;
  return s < sceneDiff(prev, cur);
}

void Engine::filterPlane(int pl, bool use3d, const FrameView* src[3], const FrameOut& out) const {
  const int w = pl ? fmt_.width >> fmt_.ssw : fmt_.width;
  const int h = pl ? fmt_.height >> fmt_.ssh : fmt_.height;
  const int bps = fmt_.bits == 8 ? 1 : 2;
  const int r = rad_;
  const int y0 = borderfix_ ? 0 : r, y1 = borderfix_ ? h : h - r;
  const FrameView& cur = *src[0];

  // Rows Deen leaves unfiltered (borderfix=false) are copies of the current frame.
  for (int y = 0; y < h; y++) {
    if (y >= y0 && y < y1) continue;
    std::memcpy(out.data[pl] + y * out.stride[pl], cur.data[pl] + y * cur.stride[pl], size_t(w) * bps);
  }
  if (y1 <= y0) return;

  const Tables& t = tables_[use3d ? 1 : 0][pl];
  const int nsrc = use3d ? 3 : 1;
  Padded pad[3];
  PlaneArgs a;
  for (int s = 0; s < nsrc; s++) {
    padPlane(src[s]->data[pl], src[s]->stride[pl], w, h, r, bps, &pad[s]);
    a.src[s] = pad[s].origin;
    a.stride[s] = pad[s].stride;
    a.thr[s] = t.thr[s].data();
    a.weight[s] = family_ == kWMode ? t.weight[s].data() : nullptr;
  }
  a.nsrc = nsrc;
  a.dst = out.data[pl];
  a.dst_stride = out.stride[pl];
  a.width = w;
  a.y0 = y0;
  a.y1 = y1;
  a.rad = r;
  a.bits = fmt_.bits;
  a.family = family_;
  k_->filter(a);
}

void Engine::process(int n, const FrameView& prev, const FrameView& cur, const FrameView& next,
                     const FrameOut& out) const {
  const bool use3d = three_d_ && !forced2d(n, prev, cur, next);
  const FrameView* src[3] = {&cur, &prev, &next};
  for (int pl = 0; pl < fmt_.planes; pl++) filterPlane(pl, use3d, src, out);
}

}  // namespace deen
