# SPDX-License-Identifier: MIT
'''Speed per mode and instruction set on a static noisy 640x352 YUV420P8 clip (one thread and all threads).

Usage: python bench.py [plugin path]
'''
import sys
import time

import numpy as np
import vapoursynth as vs

core = vs.core
if len(sys.argv) > 1:
    core.std.LoadPlugin(sys.argv[1])

W, H, FRAMES = 640, 352, 200
rng = np.random.default_rng(1)
data = [rng.integers(60, 120, size=(H >> (1 if p else 0), W >> (1 if p else 0)), dtype=np.uint8) for p in range(3)]
blank = core.std.BlankClip(format=vs.YUV420P8, width=W, height=H, length=FRAMES)


def fill(n, f):
    out = f.copy()
    for p in range(3):
        np.asarray(out[p])[:] = data[p]
    return out


src = core.std.ModifyFrame(blank, blank, fill)


def fps(clip, threads):
    core.num_threads = threads
    for _ in clip[:4].frames():
        pass
    t = time.perf_counter()
    for _ in clip.frames():
        pass
    return FRAMES / (time.perf_counter() - t)


base1 = fps(src, 1)
print(f'source alone: {base1:.0f} fps (1 thread)')
for mode, rad in (('c2d', 1), ('c3d', 1), ('a2d', 1), ('a2d', 2), ('a2d', 7), ('a3d', 1), ('a3d', 2), ('a3d', 4),
                  ('w2d', 2), ('w3d', 2)):
    row = []
    for opt in (0, 1, 2, 3):
        try:
            clip = core.deen.Deen(src, mode=mode, rad=rad, scd=255.0, opt=opt)
        except vs.Error:
            continue
        row.append(f'opt={opt} {fps(clip, 1):7.1f}')
    print(f'{mode} rad {rad}: ' + '  '.join(row))
