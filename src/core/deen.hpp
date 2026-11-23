// SPDX-License-Identifier: MIT
// Host-independent part of Deen: parameters, threshold and weight tables, scene change logic, plane loop.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kernels.hpp"

namespace deen {

struct Params {
  std::string mode = "c3d";
  int rad = 1;
  int thrY = 7, thrUV = 9, tthY = 4, tthUV = 6;
  float min = 0.5f;
  float scd = 9.0f;
  bool borderfix = false;
  int opt = -1;
};

struct Format {
  int width = 0, height = 0;
  int bits = 8;
  int planes = 1;        // 1 (Y / Gray) or 3 (YUV)
  int ssw = 0, ssh = 0;  // chroma subsampling (log2)
};

struct FrameView {
  const uint8_t* data[3] = {nullptr, nullptr, nullptr};
  ptrdiff_t stride[3] = {0, 0, 0};
};

struct FrameOut {
  uint8_t* data[3] = {nullptr, nullptr, nullptr};
  ptrdiff_t stride[3] = {0, 0, 0};
};

// Threshold table of the a modes, 8-bit scale: thr - trunc((1 - min) * thr * dist / (rad * sqrt(2))), mod 256.
std::vector<uint16_t> adaptiveThresholds(int rad, int thr, float min);
// Weight table of the w modes: (2r+1)^2 weights summing to total (0x10000 in 2d, 0x4000 in 3d).
std::vector<uint16_t> distanceWeights(int rad, float min, bool three_d);

class Engine {
 public:
  Engine(const Params& p, const Format& f, int num_frames);

  bool threeD() const { return three_d_; }
  // Frames to fetch for output frame n: n-1, n, n+1 (clamped) in the 3d modes, else n.
  void sourceRange(int n, int* first, int* last) const;
  // Mean absolute difference of two frames on the 8-bit scale, average of the planes (Deen's scene metric).
  double sceneDiff(const FrameView& a, const FrameView& b) const;
  // True if frame n runs the 2d variant of a 3d mode: frame 0 and both frames next to a scene change.
  bool forced2d(int n, const FrameView& prev, const FrameView& cur, const FrameView& next) const;
  // prev/next are only read in the 3d modes (pass cur for clamped neighbours).
  void process(int n, const FrameView& prev, const FrameView& cur, const FrameView& next, const FrameOut& out) const;
  Isa isa() const { return isa_; }

 private:
  struct Tables {
    std::vector<uint16_t> thr[3];     // per source
    std::vector<uint16_t> weight[3];
  };
  void filterPlane(int plane, bool use3d, const FrameView* src[3], const FrameOut& out) const;

  Format fmt_;
  int num_frames_;
  int family_ = kCMode;
  bool three_d_ = false;
  int rad_ = 1;
  float scd_ = 9.0f;
  bool borderfix_ = false;
  Isa isa_ = Isa::C;
  const Kernels* k_ = nullptr;
  Tables tables_[2][3];  // [3d][plane]
};

}  // namespace deen
