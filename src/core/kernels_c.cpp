// SPDX-License-Identifier: MIT
// C reference kernels: the formulas of ALGORITHM.md (scalar.hpp), applied to every sample.
#include "kernels.hpp"
#include "scalar.hpp"

namespace deen {
namespace {

void filterC(const PlaneArgs& a) {
  for (int y = a.y0; y < a.y1; y++) {
    if (a.bits == 8)
      filterRowScalar<uint8_t>(a, y, 0, a.width);
    else
      filterRowScalar<uint16_t>(a, y, 0, a.width);
  }
}

}  // namespace

const Kernels& kernelsC() {
  static const Kernels k = {filterC};
  return k;
}

}  // namespace deen
