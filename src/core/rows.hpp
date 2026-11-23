// SPDX-License-Identifier: MIT
// The 2r+1 window rows of every source for one output row, as padded copies with edge replication. Replaces a padded
// copy of the whole plane per frame: the copies live in a small per-thread ring that stays in the cache, every source
// row is copied once per plane and nothing is allocated per frame.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kernels.hpp"

namespace deen {
// Internal linkage: every kernel library gets its own copy (and its own per-thread buffer).
namespace {

class RowWindow {
 public:
  // Row y of a window holds samples [-r, width + r) of source row clamp(y, 0, height - 1).
  RowWindow(const PlaneArgs& a, int bps) : a_(a), bps_(bps), m_(2 * a.rad + 1) {
    // Sample 0 of every copy is 64-byte aligned; the left padding (r samples) sits in the 64 bytes before it.
    row_bytes_ = kLead + ((ptrdiff_t(a.width + a.rad) * bps + 63) & ~ptrdiff_t(63));
    static thread_local std::vector<uint8_t> buf;
    const size_t need = size_t(row_bytes_) * size_t(m_ * a.nsrc) + 64;
    if (buf.size() < need) buf.resize(need);
    base_ = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(buf.data()) + 63) & ~uintptr_t(63));
  }

  // Window of output row y: rows(s)[dy] (dy = 0..2r) is row y - r + dy of source s, pointing at sample 0. Steps of
  // one row copy one new row per source, any other move copies the whole window.
  void moveTo(int y) {
    const int r = a_.rad;
    if (y == cur_ + 1) {
      for (int s = 0; s < a_.nsrc; s++) fill(s, y + r);
    } else {
      for (int s = 0; s < a_.nsrc; s++)
        for (int yy = y - r; yy <= y + r; yy++) fill(s, yy);
    }
    cur_ = y;
    for (int s = 0; s < a_.nsrc; s++)
      for (int dy = 0; dy < m_; dy++) ptr_[s * m_ + dy] = slot(s, y - r + dy);
  }
  // nsrc * (2r+1) row pointers, source by source.
  const uint8_t* const* rows() const { return ptr_; }

 private:
  static constexpr ptrdiff_t kLead = 64;

  uint8_t* slot(int s, int yy) const {
    const int k = ((yy % m_) + m_) % m_;
    return base_ + ptrdiff_t(s * m_ + k) * row_bytes_ + kLead;
  }

  void fill(int s, int yy) {
    const int w = a_.width, r = a_.rad;
    const uint8_t* src = a_.src[s] + ptrdiff_t(std::min(std::max(yy, 0), a_.height - 1)) * a_.stride[s];
    uint8_t* d = slot(s, yy);
    std::memcpy(d, src, size_t(w) * size_t(bps_));
    if (bps_ == 1) {
      std::memset(d - r, src[0], size_t(r));
      std::memset(d + w, src[w - 1], size_t(r));
    } else {
      const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
      uint16_t* d16 = reinterpret_cast<uint16_t*>(d);
      for (int x = 1; x <= r; x++) {
        d16[-x] = s16[0];
        d16[w - 1 + x] = s16[w - 1];
      }
    }
  }

  const PlaneArgs& a_;
  int bps_, m_;
  ptrdiff_t row_bytes_ = 0;
  uint8_t* base_ = nullptr;
  int cur_ = -1000000;
  const uint8_t* ptr_[3 * 15] = {};
};

}  // namespace
}  // namespace deen
