// SPDX-License-Identifier: MIT
// AviSynth+ interface: Deen with the argument order of Deen 1.0 beta 1. Y and planar YUV(A), 8-16 bit.
#include <memory>
#include <stdexcept>
#include <string>

#include "avisynth.h"
#include "core/deen.hpp"
#include "version.hpp"

using namespace deen;

namespace {

class DeenAvs : public GenericVideoFilter {
 public:
  DeenAvs(PClip child, const Params& p, const Format& fmt, IScriptEnvironment* env)
      : GenericVideoFilter(child), fmt_(fmt) {
    try {
      env->CheckVersion(8);
      has_v8_ = true;
    } catch (const AvisynthError&) {
      has_v8_ = false;
    }
    try {
      engine_ = std::make_unique<Engine>(p, fmt_, vi.num_frames);
    } catch (const std::exception& e) {
      env->ThrowError("%s", e.what());
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override {
    n = n < 0 ? 0 : (n >= vi.num_frames ? vi.num_frames - 1 : n);
    int first = n, last = n;
    engine_->sourceRange(n, &first, &last);
    PVideoFrame cur = child->GetFrame(n, env);
    PVideoFrame prev = first < n ? child->GetFrame(first, env) : cur;
    PVideoFrame next = last > n ? child->GetFrame(last, env) : cur;

    PVideoFrame dst = has_v8_ ? env->NewVideoFrameP(vi, &cur) : env->NewVideoFrame(vi);
    FrameOut out;
    static const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    for (int i = 0; i < fmt_.planes; i++) {
      out.data[i] = dst->GetWritePtr(planes[i]);
      out.stride[i] = dst->GetPitch(planes[i]);
    }
    if (vi.IsYUVA())
      env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), cur->GetReadPtr(PLANAR_A),
                  cur->GetPitch(PLANAR_A), cur->GetRowSize(PLANAR_A), cur->GetHeight(PLANAR_A));
    try {
      engine_->process(n, view(prev), view(cur), view(next), out);
    } catch (const std::exception& e) {
      env->ThrowError("%s", e.what());
    }
    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    (void)frame_range;
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

 private:
  FrameView view(const PVideoFrame& f) const {
    FrameView v;
    static const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    for (int i = 0; i < fmt_.planes; i++) {
      v.data[i] = f->GetReadPtr(planes[i]);
      v.stride[i] = f->GetPitch(planes[i]);
    }
    return v;
  }

  Format fmt_;
  bool has_v8_ = false;
  std::unique_ptr<Engine> engine_;
};

Format formatOf(const VideoInfo& vi, IScriptEnvironment* env) {
  Format f;
  f.width = vi.width;
  f.height = vi.height;
  f.bits = vi.BitsPerComponent();
  if (vi.IsY()) {
    f.planes = 1;
  } else if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    f.planes = 3;
    f.ssw = vi.GetPlaneWidthSubsampling(PLANAR_U);
    f.ssh = vi.GetPlaneHeightSubsampling(PLANAR_U);
  } else {
    env->ThrowError("Deen: unsupported format: use Y or planar YUV(A) with 8-16 bit integer samples");
  }
  if (f.bits == 32) env->ThrowError("Deen: 32 bit float is not supported");
  return f;
}

// c[mode]s[rad]i[thrY]i[thrUV]i[tthY]i[tthUV]i[min]f[scd]f[fcf]s[borderfix]b[opt]i
AVSValue __cdecl Create(AVSValue a, void*, IScriptEnvironment* env) {
  PClip child = a[0].AsClip();
  Params p;
  p.mode = a[1].AsString("c3d");
  p.rad = a[2].AsInt(1);
  p.thrY = a[3].AsInt(7);
  p.thrUV = a[4].AsInt(9);
  p.tthY = a[5].AsInt(4);
  p.tthUV = a[6].AsInt(6);
  p.min = a[7].AsFloatf(0.5f);
  p.scd = a[8].AsFloatf(9.0f);
  const char* fcf = a[9].AsString("");
  if (fcf && *fcf) env->ThrowError("Deen: fcf files are not supported");
  p.borderfix = a[10].AsBool(false);
  p.opt = a[11].AsInt(-1);
  const Format fmt = formatOf(child->GetVideoInfo(), env);
  return new DeenAvs(child, p, fmt, env);
}

}  // namespace

#ifdef _WIN32
#define DEEN_AVS_EXPORT extern "C" __declspec(dllexport)
#else
#define DEEN_AVS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

const AVS_Linkage* AVS_linkage = nullptr;

DEEN_AVS_EXPORT const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;
  env->AddFunction("Deen", "c[mode]s[rad]i[thrY]i[thrUV]i[tthY]i[tthUV]i[min]f[scd]f[fcf]s[borderfix]b[opt]i", Create,
                   nullptr);
  return "Deen " DEEN_VERSION;
}
