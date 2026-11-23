# SPDX-License-Identifier: MIT
'''Speed of deen.Deen per mode, radius, bit depth, frame size and thread count, optionally against neo_smo.Deen.

The source is three noisy YUV420 frames that differ a little (mean difference about 2, below Deen's default scd of 9,
so the 3d modes really run 3d), built once and looped, so the source costs next to nothing. Each measurement uses a
new filter instance (VapourSynth caches output frames, so measuring the same node twice would partly read its cache)
and runs for at least --seconds after a warm-up; best of --repeat runs (deen and neo_smo alternating).

Usage: python bench.py [--deen Deen.dll] [--neo neo-smo.dll] [--sizes sd,hd,uhd] [--bits 8,16] [--threads 1,0]
                       [--modes c2d,a3d,...] [--rads 1,2,4] [--opt -1] [--seconds 1.0] [--repeat 1]
                       [--json out.json]
  --threads 0 = all threads. Without --neo only deen.Deen runs.
'''
import argparse
import collections
import json
import time

import numpy as np
import vapoursynth as vs

core = vs.core
SIZES = {'sd': (720, 576), '720p': (960, 720), 'hd': (1920, 1080), 'uhd': (3840, 2160)}


def make_source(w, h, bits):
    fmt = vs.YUV420P8 if bits == 8 else core.query_video_format(vs.YUV, vs.INTEGER, bits, 1, 1).id
    rng = np.random.default_rng(1)
    shift = bits - 8
    planes = []
    for p in range(3):
        pw, ph = (w >> 1, h >> 1) if p else (w, h)
        base = rng.integers(60, 120, size=(ph, pw))
        planes.append([((base + rng.integers(-2, 3, size=(ph, pw))) << shift).astype(np.uint8 if bits == 8 else
                                                                                    np.uint16) for _ in range(3)])
    blank = core.std.BlankClip(format=fmt, width=w, height=h, length=3)
    # The three frames are built once; the callback only hands them out, so a source frame that drops out of the
    # VapourSynth cache (large frames, many threads) costs next to nothing to get again.
    frames = []
    for i in range(3):
        f = blank.get_frame(i).copy()
        for p in range(3):
            np.asarray(f[p])[:] = planes[p][i]
        frames.append(f)
    return core.std.ModifyFrame(blank, blank, lambda n, f: frames[n]) * 100000


def fps(make, threads, seconds):
    """Frames per second of a new filter instance make() after a warm-up (thread pool, caches, clock).

    One request per thread is kept in flight (get_frame_async). The warm-up waits for at least one frame per thread,
    the timed part covers at least three frames per thread, and before returning every outstanding request is waited
    for (untimed), so no work of this measurement runs into the next one.
    """
    core.num_threads = max(threads, 0)  # 0 = all threads
    nt = core.num_threads
    clip = make()
    pending = collections.deque()
    nxt = 0

    def request():
        nonlocal nxt
        pending.append(clip.get_frame_async(nxt))
        nxt += 1

    for _ in range(nt):
        request()

    def run(limit, frames):
        n = 0
        t = time.perf_counter()
        while True:
            pending.popleft().result()
            request()
            n += 1
            if n % 4 == 0:
                el = time.perf_counter() - t
                if el >= limit and n >= frames:
                    return n / el
    run(seconds / 3, max(4, nt))
    result = run(seconds, max(12, 3 * nt))
    while pending:
        pending.popleft().result()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deen')
    ap.add_argument('--neo')
    ap.add_argument('--sizes', default='sd,hd,uhd')
    ap.add_argument('--bits', default='8,16')
    ap.add_argument('--threads', default='1,0')
    ap.add_argument('--modes', default='c2d,c3d,a2d,a3d,w2d,w3d')
    ap.add_argument('--rads', default='1,2,4')
    ap.add_argument('--opt', type=int, default=-1)
    ap.add_argument('--seconds', type=float, default=1.0)
    ap.add_argument('--repeat', type=int, default=1, help='best of N, deen and neo_smo alternating')
    ap.add_argument('--json')
    args = ap.parse_args()
    if args.deen:
        core.std.LoadPlugin(args.deen)
    if args.neo and not hasattr(core, 'neo_smo'):  # may be autoloaded already
        core.std.LoadPlugin(args.neo)
    results = []
    for size in args.sizes.split(','):
        w, h = SIZES[size]
        for bits in map(int, args.bits.split(',')):
            src = make_source(w, h, bits)
            for threads in map(int, args.threads.split(',')):
                for mode in args.modes.split(','):
                    for rad in map(int, args.rads.split(',')):
                        if mode.endswith('3d') and rad > 4:
                            continue
                        row = {'size': size, 'bits': bits, 'threads': threads, 'mode': mode, 'rad': rad}
                        def deen():
                            return core.deen.Deen(src, mode=mode, rad=rad, opt=args.opt)
                        neo = None
                        if args.neo:
                            a = dict(mode=mode, radius=rad, threshold=[7, 9, 9], scalep=True)
                            if mode.endswith('3d'):
                                a.update(temporal_threshold=[4, 6, 6], scenechange=round(9 * 2.54))
                            if mode[0] in 'aw':
                                a['minimum'] = 0.5
                            def neo():
                                return core.neo_smo.Deen(src, **a)
                        row['deen'] = row['neo'] = 0.0
                        for _ in range(args.repeat):
                            row['deen'] = max(row['deen'], fps(deen, threads, args.seconds))
                            if neo:
                                row['neo'] = max(row['neo'], fps(neo, threads, args.seconds))
                        line = (f'{size:4} {bits:2} bit {"all" if threads <= 0 else threads:>3} thr {mode} rad {rad}: '
                                f'deen {row["deen"]:8.1f}')
                        if neo:
                            line += f'  neo_smo {row["neo"]:8.1f}  x{row["deen"] / row["neo"]:5.2f}'
                        else:
                            del row['neo']
                        print(line, flush=True)
                        results.append(row)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
