// SPDX-License-Identifier: MIT
// The per-pixel formulas of ALGORITHM.md. Used by the C kernels and by the SIMD kernels for rows narrower than a vector.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "kernels.hpp"

namespace deen {
// Internal linkage: every kernel library gets its own copy, compiled with its own instruction set flags.
namespace {

// Deen's 8-bit reciprocal: 0xFFFF for 1, floor(65536 / n) otherwise (n = 1..255).
struct RecipTable {
  uint16_t r[256];
  RecipTable() {
    r[0] = 0;
    for (int n = 1; n < 256; n++) r[n] = uint16_t(n == 1 ? 0xFFFFu : 65536u / unsigned(n));
  }
};

inline const uint16_t* recipTable() {
  static const RecipTable t;
  return t.r;
}

inline uint32_t recip8(uint32_t n) { return recipTable()[n]; }

// floor(x / d) for x < 2^24 and 1 <= d < 256: (x * magic) >> 32 with magic = floor(2^32 / d) + 1.
struct DivTable {
  uint64_t magic[256];
  DivTable() {
    magic[0] = 0;
    for (int d = 1; d < 256; d++) magic[d] = (uint64_t(1) << 32) / uint64_t(d) + 1;
  }
};

inline const DivTable& divTable() {
  static const DivTable t;
  return t;
}

inline uint32_t divSmall(uint32_t x, uint32_t d) {
  return d == 1 ? x : uint32_t((uint64_t(x) * divTable().magic[d]) >> 32);
}

inline int windowTaps(int rad) { return (2 * rad + 1) * (2 * rad + 1); }

// Output of the c and a modes from the window sum and count, 8 bit.
inline uint32_t finishC8(uint32_t sum, uint32_t n) { return ((sum + (n + 1) / 2) * (65536u / n)) >> 16; }
inline uint32_t finishA8(uint32_t sum, uint32_t cnt) { return ((sum + ((cnt + 1) >> 1)) * recip8(cnt)) >> 16; }
// w mode, 8 bit: paddw 0x80 (wrapping), psrlw 8, packuswb, paddusb 1.
inline uint32_t finishW8(uint32_t sum) {
  const uint32_t t = ((sum + 128) & 0xFFFFu) >> 8;
  return std::min<uint32_t>(255, std::min<uint32_t>(255, t) + 1);
}
// 9-16 bit: rounded division.
inline uint32_t finishDiv16(uint32_t sum, uint32_t n) { return divSmall(sum + (n + 1) / 2, n); }
// a modes: a count of 1 (only the centre passed) keeps the centre, as R(1) = 0xFFFF does at 8 bit.
inline uint32_t finishA16(uint32_t sum, uint32_t cnt) { return cnt == 1 ? sum : finishDiv16(sum, cnt); }
inline uint32_t finishW16(uint32_t sum, uint32_t maxv) { return std::min<uint32_t>(maxv, uint32_t((uint64_t(sum) + 0x8000) >> 16)); }

// One output sample. rows: the window rows of every source (RowWindow::rows()), pointing at sample 0.
template <typename T>
inline T filterPixel(const PlaneArgs& a, const uint8_t* const* rows, int x) {
  const int r = a.rad, side = 2 * r + 1;
  const uint32_t c = reinterpret_cast<const T*>(rows[r])[x];
  uint32_t sum = 0, cnt = 0;
  for (int s = 0; s < a.nsrc; s++) {
    const uint16_t* thr = a.thr[s];
    const uint16_t* w = a.weight[s];
    int k = 0;
    for (int dy = 0; dy < side; dy++) {
      const T* row = reinterpret_cast<const T*>(rows[s * side + dy]);
      for (int dx = -r; dx <= r; dx++, k++) {
        const uint32_t n = row[x + dx];
        const bool pass = uint32_t(std::abs(int(c) - int(n))) <= thr[k];
        switch (a.family) {
          case kCMode: sum += pass ? n : c; break;
          case kAMode:
            if (pass) {
              sum += n;
              cnt++;
            }
            break;
          default: {
            const uint32_t v = pass ? n : c;
            if (sizeof(T) == 1)
              sum = std::min<uint32_t>(0xFFFFu, sum + ((v * w[k]) >> 8));  // paddusw of (v << 8) pmulhuw w
            else
              sum += v * w[k];
            break;
          }
        }
      }
    }
  }
  const uint32_t n = uint32_t(windowTaps(r) * a.nsrc);
  if (sizeof(T) == 1) {
    switch (a.family) {
      case kCMode: return T(finishC8(sum, n));
      case kAMode: return T(finishA8(sum, cnt));
      default: return T(finishW8(sum));
    }
  }
  switch (a.family) {
    case kCMode: return T(finishDiv16(sum, n));
    case kAMode: return T(finishA16(sum, cnt));
    default: return T(finishW16(sum, (1u << a.bits) - 1));
  }
}

template <typename T>
inline void filterRowScalar(const PlaneArgs& a, const uint8_t* const* rows, uint8_t* drow, int x0, int x1) {
  T* d = reinterpret_cast<T*>(drow);
  for (int x = x0; x < x1; x++) d[x] = filterPixel<T>(a, rows, x);
}

}  // namespace
}  // namespace deen
