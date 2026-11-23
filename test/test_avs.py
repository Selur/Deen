#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""AviSynth+ interface of Deen bit for bit against the numpy model (model.py) on synthetic clips.

The clips are written as y4m and read with BestSource; the output comes from avs2yuv64 -raw. Covers every mode,
positional arguments in Deen 1.0 beta 1 order, 8/10/16 bit, Y and 4:2:0/4:2:2/4:4:4, and the parameter errors.

Usage (Windows): F:\\Python4Claude\\python.exe test/test_avs.py [--plugin build/Deen.dll] [--avs DIR] [--work DIR]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# y4m tag, bits, subsampling (w, h), planes
FORMATS = {'420p8': ('420jpeg', 8, (1, 1), 3), '422p10': ('422p10', 10, (1, 0), 3),
           '444p16': ('444p16', 16, (0, 0), 3), 'y8': ('mono', 8, (0, 0), 1)}


def fwd(p):
    return os.path.abspath(p).replace('\\', '/')


def make_frames(fmt, width, height, count, seed):
    _, bits, (sw, sh), nplanes = FORMATS[fmt]
    rng = np.random.default_rng(seed)
    maxv = (1 << bits) - 1
    base = rng.integers(0, maxv + 1, size=(height, width)).astype(np.int64)
    frames = []
    for i in range(count):
        planes = []
        for p in range(nplanes):
            pw, ph = width >> (sw if p else 0), height >> (sh if p else 0)
            noise = rng.integers(-maxv // 12, maxv // 12 + 1, size=(ph, pw))
            plane = np.clip(base[:ph, :pw] // 3 + maxv // 3 + noise + (i * maxv // 40 if i != 3 else maxv // 2), 0, maxv)
            planes.append(plane.astype(np.uint16 if bits > 8 else np.uint8))
        frames.append(planes)
    return frames


def write_y4m(path, frames, fmt):
    tag, bits = FORMATS[fmt][:2]
    h, w = frames[0][0].shape
    with open(path, 'wb') as f:
        f.write(f'YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C{tag}\n'.encode())
        for fr in frames:
            f.write(b'FRAME\n')
            for p in fr:
                f.write(p.astype('<u2' if bits > 8 else 'u1').tobytes())


def run(args, script, path):
    with open(path, 'w') as f:
        f.write(script)
    r = subprocess.run([os.path.join(args.avs, 'avs2yuv64.exe'), '-raw', path, '-o', '-'], capture_output=True)
    return r.returncode, r.stdout, r.stderr.decode(errors='replace')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plugin', default=os.path.join(ROOT, 'build', 'Deen.dll'))
    ap.add_argument('--avs', default=r'F:\Hybrid\64bit\Avisynth')
    ap.add_argument('--work', default=os.path.join(HERE, 'work', 'avs'))
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    bs = os.path.join(args.avs, 'avisynthPlugins', 'BestSource.dll')
    head = f'LoadPlugin("{fwd(bs)}")\nLoadPlugin("{fwd(args.plugin)}")\n'

    cases = []
    for mode in ('c2d', 'c3d', 'a2d', 'a3d', 'w2d', 'w3d'):
        for rad in ((1, 3, 7) if mode[1] == '2' else (1, 2, 4)):
            for borderfix in (False, True):
                cases.append(dict(mode=mode, rad=rad, borderfix=borderfix))
    cases.append(dict(mode='a3d', rad=2, thrY=20, thrUV=12, tthY=9, tthUV=3, min=0.1, scd=6.0))
    failures = checks = 0
    for fi, fmt in enumerate(FORMATS):
        frames = make_frames(fmt, 58 + 4 * fi, 20 + 2 * fi, 6, 7 + fi)
        y4m = os.path.join(args.work, f'src_{fmt}.y4m')
        write_y4m(y4m, frames, fmt)
        bits = FORMATS[fmt][1]
        src = f'BSVideoSource("{fwd(y4m)}", cachemode=0)\n'
        for ci, case in enumerate(cases):
            if fmt in ('444p16', 'y8') and ci % 3:
                continue  # every third case is enough for the extra formats
            kw = ', '.join(f'{k}={("true" if v else "false") if isinstance(v, bool) else repr(v).replace(chr(39), chr(34))}'
                           for k, v in case.items())
            code, out, err = run(args, head + src + f'Deen({kw})\n', os.path.join(args.work, 'case.avs'))
            want = b''.join(p.astype('<u2' if bits > 8 else 'u1').tobytes()
                            for n in range(len(frames)) for p in model.deen(frames, n, bits, **case))
            checks += 1
            if code != 0 or out != want:
                failures += 1
                print(f'FAIL {fmt} {case}: exit {code} {err.strip()[-200:]} ({len(out)} vs {len(want)} bytes)')

    # positional arguments in Deen 1.0 beta 1 order: mode, rad, thrY, thrUV, tthY, tthUV, min, scd, fcf, borderfix
    frames = make_frames('420p8', 64, 24, 6, 3)
    y4m = os.path.join(args.work, 'src_pos.y4m')
    write_y4m(y4m, frames, '420p8')
    src = f'BSVideoSource("{fwd(y4m)}", cachemode=0)\n'
    case = dict(mode='a3d', rad=2, thrY=10, thrUV=12, tthY=3, tthUV=5, min=0.25, scd=7.5, borderfix=True)
    code, out, err = run(args, head + src + 'Deen("a3d", 2, 10, 12, 3, 5, 0.25, 7.5, "", true)\n',
                         os.path.join(args.work, 'pos.avs'))
    want = b''.join(p.astype('u1').tobytes() for n in range(len(frames)) for p in model.deen(frames, n, 8, **case))
    checks += 1
    if code != 0 or out != want:
        failures += 1
        print(f'FAIL positional arguments: exit {code} {err.strip()[-200:]}')
    defaults = model.deen(frames, 2, 8)  # c3d rad 1
    code, out, err = run(args, head + src + 'Deen()\n', os.path.join(args.work, 'def.avs'))
    checks += 1
    size = sum(p.size for p in defaults)
    if code != 0 or out[2 * size:3 * size] != b''.join(p.astype('u1').tobytes() for p in defaults):
        failures += 1
        print(f'FAIL defaults: exit {code} {err.strip()[-200:]}')

    for bad in ('mode="x2d"', 'mode="c3d", rad=5', 'rad=8', 'thrY=256', 'min=1.5', 'fcf="a.fcf"'):
        code, out, err = run(args, head + src + f'Deen({bad})\n', os.path.join(args.work, 'bad.avs'))
        checks += 1
        if 'Deen:' not in err:
            failures += 1
            print(f'FAIL no Deen error for {bad}: exit {code} {err.strip()[-200:]}')
    print(f'{checks} checks, {failures} failures')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
