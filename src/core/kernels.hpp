// SPDX-License-Identifier: MIT
// Pixel kernels of Deen. Every instruction set gives bit-identical results to the C reference (see ALGORITHM.md).
#pragma once

#include <cstddef>
#include <cstdint>

namespace deen {

enum Family { kCMode = 0, kAMode = 1, kWMode = 2 };

// One plane. Sources are edge-replicated copies with at least `rad` samples of padding on every side; src[i] points
// at image sample (0, 0), so src[i] + y * stride[i] + x * bytes_per_sample is valid for x, y in [-rad, size + rad).
struct PlaneArgs {
  const uint8_t* src[3] = {nullptr, nullptr, nullptr};  // current, previous, next frame
  ptrdiff_t stride[3] = {0, 0, 0};                      // bytes
  int nsrc = 1;                                         // 1 (2d) or 3 (3d)
  uint8_t* dst = nullptr;                               // row 0 of the output plane
  ptrdiff_t dst_stride = 0;
  int width = 0;
  int y0 = 0, y1 = 0;                                   // rows [y0, y1) are filtered
  int rad = 1;
  int bits = 8;                                         // 8..16
  int family = kCMode;
  // Per source (2r+1)^2 entries in window order (dy = -r..r, then dx = -r..r), on the sample scale.
  const uint16_t* thr[3] = {nullptr, nullptr, nullptr};
  const uint16_t* weight[3] = {nullptr, nullptr, nullptr};  // w modes
};

struct Kernels {
  void (*filter)(const PlaneArgs& a);
};

enum class Isa { C, SSE41, AVX2, NEON };

const Kernels& kernelsC();
#if defined(_M_X64) || defined(__x86_64__)
#define DEEN_X86 1
const Kernels& kernelsSSE41();
const Kernels& kernelsAVX2();
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DEEN_NEON 1
const Kernels& kernelsNEON();
#endif

// opt: -1 = best available, 0 = C, 1 = SSE4.1, 2 = AVX2, 3 = NEON. Returns false if not available.
bool selectIsa(int opt, Isa* isa);
const Kernels& kernelsFor(Isa isa);
const char* isaName(Isa isa);

}  // namespace deen
