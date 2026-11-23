# SPDX-License-Identifier: MIT
'''numpy model of Deen, written from ALGORITHM.md: every mode, 8-16 bit, edge replication, scene change logic.'''
import builtins
import math
import struct

import numpy as np


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def parse_mode(mode):
    return 'caw'.index(mode[0]), mode[1] == '3'


def adaptive_thresholds(rad, thr, min_):
    m = f32(min_)
    inv = 1.0 / math.sqrt(float(rad * rad) + float(rad * rad))
    out = {}
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            q = math.sqrt(float(dx * dx) + float(dy * dy)) * inv
            e = int(((1.0 - m) * thr) * q)
            out[(dy, dx)] = (thr - (e & 0xFF)) & 0xFF
    return out


def distance_weights(rad, min_, three_d):
    total = 0x4000 if three_d else 0x10000
    taps = (2 * rad + 1) ** 2
    base = total // taps
    d = base - int(f32(min_) * base)
    q = 1.0 / math.sqrt(float(rad * rad) + float(rad * rad))
    w = {}
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            t = math.sqrt(float(dx * dx) + float(dy * dy))
            w[(dy, dx)] = (base - int((t * q) * d)) & 0xFFFF
    scale = (total << 8) // sum(w.values())
    for k in w:
        w[k] = ((w[k] * scale) >> 8) & 0xFFFF
    w[(0, 0)] = (w[(0, 0)] + total - sum(w.values())) & 0xFFFF
    return w


def recip8(cnt):
    return np.where(cnt == 1, 0xFFFF, 65536 // np.maximum(cnt, 1))


def filter_plane(srcs, thr_tabs, w_tabs, family, rad, bits, borderfix):
    '''srcs: current, previous, next plane (2d: only the current one) as 2-D integer arrays.'''
    c = srcs[0].astype(np.int64)
    h, w = c.shape
    s_sum = np.zeros_like(c)
    cnt = np.zeros_like(c)
    for s, plane in enumerate(srcs):
        p = np.pad(plane.astype(np.int64), rad, mode='edge')
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                n = p[rad + dy:rad + dy + h, rad + dx:rad + dx + w]
                ok = np.abs(c - n) <= thr_tabs[s][(dy, dx)]
                if family == 0:
                    s_sum += np.where(ok, n, c)
                elif family == 1:
                    s_sum += np.where(ok, n, 0)
                    cnt += ok
                else:
                    v = np.where(ok, n, c)
                    if bits == 8:
                        s_sum = np.minimum(0xFFFF, s_sum + ((v * w_tabs[s][(dy, dx)]) >> 8))
                    else:
                        s_sum = (s_sum + v * w_tabs[s][(dy, dx)]) & 0xFFFFFFFF
    total = (2 * rad + 1) ** 2 * len(srcs)
    if bits == 8:
        if family == 0:
            out = ((s_sum + (total + 1) // 2) * (65536 // total)) >> 16
        elif family == 1:
            out = ((s_sum + ((cnt + 1) >> 1)) * recip8(cnt)) >> 16
        else:
            out = np.minimum(255, np.minimum(255, ((s_sum + 128) & 0xFFFF) >> 8) + 1)
    else:
        if family == 0:
            out = (s_sum + (total + 1) // 2) // total
        elif family == 1:
            out = np.where(cnt == 1, s_sum, (s_sum + (cnt + 1) // 2) // np.maximum(cnt, 1))
        else:
            out = np.minimum((1 << bits) - 1, (s_sum + 0x8000) >> 16)
    if not borderfix:
        out[:rad] = c[:rad]
        out[h - rad:] = c[h - rad:]
        if h <= 2 * rad:
            out[:] = c
    return out


def scene_diff(a, b, bits):
    '''a, b: lists of planes. Mean absolute difference on the 8-bit scale, average of the planes.'''
    total = 0.0
    scale = 1.0 if bits == 8 else 255.0 / ((1 << bits) - 1)
    for pa, pb in zip(a, b):
        total = total + float(np.abs(pa.astype(np.int64) - pb.astype(np.int64)).sum()) / pa.size * scale
    return total * 0.3333333333333333 if len(a) == 3 else total / len(a)


def deen(frames, n, bits, mode='c3d', rad=1, thrY=7, thrUV=9, tthY=4, tthUV=6, min=0.5, scd=9.0, borderfix=False):
    '''frames: list of frames, each a list of planes. Returns the planes of output frame n.'''
    family, three_d = parse_mode(mode)
    last = len(frames) - 1
    prev, cur, nxt = frames[builtins.max(0, n - 1)], frames[n], frames[builtins.min(last, n + 1)]
    use3d = three_d
    if three_d:
        s = f32(scd)
        if n < 1 or s < scene_diff(cur, nxt, bits) or s < scene_diff(prev, cur, bits):
            use3d = False
    shift = bits - 8
    out = []
    for p in range(len(cur)):
        thr = thrY if p == 0 else thrUV
        tth = tthY if p == 0 else tthUV
        bases = [thr, tth, tth] if use3d else [thr]
        if family == 1:
            tabs = [adaptive_thresholds(rad, b, min) for b in bases]
        else:
            tabs = [{(dy, dx): b for dy in range(-rad, rad + 1) for dx in range(-rad, rad + 1)} for b in bases]
        tabs = [{k: v << shift for k, v in t.items()} for t in tabs]
        wtabs = [None] * len(bases)
        if family == 2:
            w = distance_weights(rad, min, use3d)
            wtabs = [{k: (v * 2) & 0xFFFF for k, v in w.items()}, w, w] if use3d else [w]
        srcs = [cur[p], prev[p], nxt[p]] if use3d else [cur[p]]
        out.append(filter_plane(srcs, tabs, wtabs, family, rad, bits, borderfix))
    return out
