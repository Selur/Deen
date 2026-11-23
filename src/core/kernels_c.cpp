// SPDX-License-Identifier: MIT
// C reference kernels: the formulas of ALGORITHM.md (scalar.hpp), applied to every sample.
#include "kernels.hpp"
#include "rows.hpp"
#include "scalar.hpp"

namespace deen {
namespace {

void filterC(const PlaneArgs& a) {
  RowWindow win(a, a.bits == 8 ? 1 : 2);
  for (int y = a.y0; y < a.y1; y++) {
    win.moveTo(y);
    uint8_t* drow = a.dst + y * a.dst_stride;
    if (a.bits == 8)
      filterRowScalar<uint8_t>(a, win.rows(), drow, 0, a.width);
    else
      filterRowScalar<uint16_t>(a, win.rows(), drow, 0, a.width);
  }
}

}  // namespace

const Kernels& kernelsC() {
  static const Kernels k = {filterC};
  return k;
}

}  // namespace deen
