// SPDX-License-Identifier: MIT
// VapourSynth (API 4) interface: deen.Deen. Gray and YUV, 8-16 bit integer.
#include <memory>
#include <stdexcept>
#include <string>

#include "VSHelper4.h"
#include "VapourSynth4.h"
#include "core/deen.hpp"
#include "version.hpp"

using namespace deen;

namespace {

struct FilterData {
  VSNode* node = nullptr;
  Format fmt;
  std::unique_ptr<Engine> engine;
};

FrameView view(const VSFrame* f, const FilterData* d, const VSAPI* vsapi) {
  FrameView v;
  for (int i = 0; i < d->fmt.planes; i++) {
    v.data[i] = vsapi->getReadPtr(f, i);
    v.stride[i] = vsapi->getStride(f, i);
  }
  return v;
}

const VSFrame* VS_CC getFrame(int n, int reason, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                              const VSAPI* vsapi) {
  auto* d = static_cast<FilterData*>(instance);
  int first = n, last = n;
  d->engine->sourceRange(n, &first, &last);
  if (reason == arInitial) {
    for (int k = first; k <= last; k++) vsapi->requestFrameFilter(k, d->node, ctx);
    return nullptr;
  }
  if (reason != arAllFramesReady) return nullptr;

  const VSFrame* cur = vsapi->getFrameFilter(n, d->node, ctx);
  const VSFrame* prev = first < n ? vsapi->getFrameFilter(first, d->node, ctx) : nullptr;
  const VSFrame* next = last > n ? vsapi->getFrameFilter(last, d->node, ctx) : nullptr;
  VSFrame* dst = vsapi->newVideoFrame(vsapi->getVideoFrameFormat(cur), vsapi->getFrameWidth(cur, 0),
                                      vsapi->getFrameHeight(cur, 0), cur, core);
  FrameOut out;
  for (int i = 0; i < d->fmt.planes; i++) {
    out.data[i] = vsapi->getWritePtr(dst, i);
    out.stride[i] = vsapi->getStride(dst, i);
  }
  const FrameView vc = view(cur, d, vsapi);
  const VSFrame* result = dst;
  try {
    d->engine->process(n, prev ? view(prev, d, vsapi) : vc, vc, next ? view(next, d, vsapi) : vc, out);
  } catch (const std::exception& e) {
    vsapi->setFilterError(e.what(), ctx);
    vsapi->freeFrame(dst);
    result = nullptr;
  }
  vsapi->freeFrame(cur);
  vsapi->freeFrame(prev);
  vsapi->freeFrame(next);
  return result;
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI* vsapi) {
  auto* d = static_cast<FilterData*>(instance);
  vsapi->freeNode(d->node);
  delete d;
}

void VS_CC create(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
  auto d = std::make_unique<FilterData>();
  d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
  auto fail = [&](const std::string& msg) {
    vsapi->mapSetError(out, msg.c_str());
    vsapi->freeNode(d->node);
  };

  int e = 0;
  Params p;
  if (const char* m = vsapi->mapGetData(in, "mode", 0, &e); !e) p.mode = m;
  auto i = [&](const char* key, int& v) {
    const int64_t x = vsapi->mapGetInt(in, key, 0, &e);
    if (!e) v = int(x);
  };
  auto f = [&](const char* key, float& v) {
    const double x = vsapi->mapGetFloat(in, key, 0, &e);
    if (!e) v = float(x);
  };
  i("rad", p.rad);
  i("thrY", p.thrY);
  i("thrUV", p.thrUV);
  i("tthY", p.tthY);
  i("tthUV", p.tthUV);
  f("min", p.min);
  f("scd", p.scd);
  {
    const int64_t x = vsapi->mapGetInt(in, "borderfix", 0, &e);
    if (!e) p.borderfix = x != 0;
  }
  i("opt", p.opt);

  const VSVideoInfo* vi = vsapi->getVideoInfo(d->node);
  if (!vsh::isConstantVideoFormat(vi)) return fail("Deen: only constant format input is supported");
  const VSVideoFormat& vf = vi->format;
  Format& fmt = d->fmt;
  fmt.width = vi->width;
  fmt.height = vi->height;
  fmt.bits = vf.bitsPerSample;
  if (vf.sampleType != stInteger || vf.bitsPerSample > 16)
    return fail(vf.sampleType == stFloat ? "Deen: float is not supported"
                                         : "Deen: unsupported format: use Gray or YUV with 8-16 bit integer samples");
  if (vf.colorFamily == cfYUV) {
    fmt.planes = 3;
    fmt.ssw = vf.subSamplingW;
    fmt.ssh = vf.subSamplingH;
  } else if (vf.colorFamily == cfGray) {
    fmt.planes = 1;
  } else {
    return fail("Deen: unsupported format: use Gray or YUV with 8-16 bit integer samples");
  }

  try {
    d->engine = std::make_unique<Engine>(p, fmt, vi->numFrames);
  } catch (const std::exception& ex) {
    return fail(ex.what());
  }

  const VSFilterDependency deps[] = {{d->node, d->engine->threeD() ? rpGeneral : rpStrictSpatial}};
  vsapi->createVideoFilter(out, "Deen", vi, getFrame, freeFilter, fmParallel, deps, 1, d.release(), core);
}

}  // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
  vspapi->configPlugin("io.github.selur.deen", "deen", "Deen " DEEN_VERSION,
                       VS_MAKE_VERSION(DEEN_VERSION_MAJOR, DEEN_VERSION_MINOR), VAPOURSYNTH_API_VERSION, 0, plugin);
  vspapi->registerFunction("Deen",
                           "clip:vnode;mode:data:opt;rad:int:opt;thrY:int:opt;thrUV:int:opt;tthY:int:opt;"
                           "tthUV:int:opt;min:float:opt;scd:float:opt;borderfix:int:opt;opt:int:opt;",
                           "clip:vnode;", create, nullptr, plugin);
}
