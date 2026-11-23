# SPDX-License-Identifier: MIT
'''VapourSynth: deen.Deen bit for bit against the numpy model (model.py), every mode, for every instruction set.

Usage: python test_deen.py [plugin path]   (without a path the installed plugin is used)
'''
import itertools
import os
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model  # noqa: E402

core = vs.core
if len(sys.argv) > 1:
    core.std.LoadPlugin(sys.argv[1])

FORMATS = [(vs.YUV420P8, 8), (vs.YUV444P8, 8), (vs.YUV422P10, 10), (vs.GRAY16, 16), (vs.YUV420P16, 16)]


def make_frames(fmt, width, height, count, seed):
    rng = np.random.default_rng(seed)
    f = core.query_video_format(*fmt) if isinstance(fmt, tuple) else core.get_video_format(fmt)
    maxv = (1 << f.bits_per_sample) - 1
    frames = []
    base = rng.integers(0, maxv + 1, size=(height, width)).astype(np.int64)
    for i in range(count):
        planes = []
        for p in range(f.num_planes):
            pw = width >> (f.subsampling_w if p else 0)
            ph = height >> (f.subsampling_h if p else 0)
            smooth = np.clip(base[:ph, :pw] // 3 + maxv // 3, 0, maxv)
            noise = rng.integers(-maxv // 12, maxv // 12 + 1, size=(ph, pw))
            plane = np.clip(smooth + noise + (i * maxv // 40 if i != 3 else maxv // 2), 0, maxv)
            planes.append(plane.astype(np.uint8 if f.bits_per_sample == 8 else np.uint16))
        frames.append(planes)
    return f, frames


def to_clip(f, frames):
    h, w = frames[0][0].shape
    clip = core.std.BlankClip(format=f.id, width=w, height=h, length=len(frames))

    def put(n, f):
        out = f.copy()
        for p in range(out.format.num_planes):
            np.asarray(out[p])[:] = frames[n][p]
        return out
    return core.std.ModifyFrame(clip, clip, put)


def planes_of(frame):
    return [np.asarray(frame[p]) for p in range(frame.format.num_planes)]


def available_opts():
    opts = [0]
    for o in (1, 2, 3):
        try:
            core.deen.Deen(core.std.BlankClip(format=vs.YUV420P8), mode='c2d', opt=o)
            opts.append(o)
        except vs.Error:
            pass
    return opts


def main():
    opts = available_opts()
    print('instruction sets:', opts)
    failures = checks = 0
    cases = []
    for mode in ('c2d', 'c3d', 'a2d', 'a3d', 'w2d', 'w3d'):
        rads = (1, 2, 4, 7) if mode[1] == '2' else (1, 2, 3, 4)
        for rad, borderfix in itertools.product(rads, (False, True)):
            cases.append(dict(mode=mode, rad=rad, borderfix=borderfix))
    cases += [dict(mode='a3d', rad=2, thrY=20, thrUV=12, tthY=9, tthUV=3, min=0.1, scd=6.0),
              dict(mode='w3d', rad=3, min=0.0, scd=40.0), dict(mode='c3d', rad=1, thrY=0, thrUV=255, scd=-1.0),
              dict(mode='a2d', rad=5, min=1.0), dict(mode='w2d', rad=2, min=1.0, thrY=255)]
    for fi, (fmt, bits) in enumerate(FORMATS):
        f, frames = make_frames(fmt, 61 if fi % 2 else 72, 23 if fi % 2 else 18, 6, 100 + fi)
        clip = to_clip(f, frames)
        for case in cases:
            want = [model.deen(frames, n, bits, **case) for n in range(len(frames))]
            for opt in opts:
                out = core.deen.Deen(clip, opt=opt, **case)
                for n in range(len(frames)):
                    got = planes_of(out.get_frame(n))
                    checks += 1
                    if any(not np.array_equal(g, w) for g, w in zip(got, want[n])):
                        failures += 1
                        if failures <= 20:
                            diff = max(int(np.abs(g.astype(int) - w).max()) for g, w in zip(got, want[n]))
                            print(f'FAIL {f.name} opt={opt} frame={n} {case} max diff {diff}')
    # errors
    for bad in (dict(mode='x2d'), dict(mode='c3d', rad=5), dict(mode='c2d', rad=8), dict(thrY=256), dict(min=1.5)):
        try:
            core.deen.Deen(core.std.BlankClip(format=vs.YUV420P8), **bad)
            print('FAIL no error for', bad)
            failures += 1
        except vs.Error:
            pass
    print(f'{checks} frame checks, {failures} failures')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
