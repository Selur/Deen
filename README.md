# Deen

**A clean-room reimplementation of Marc FD's Deen 1.0 beta 1 (2003) denoiser.** One plugin binary for **AviSynth+**
and **VapourSynth** (API 4), built for Windows x64, Linux x86_64 and macOS arm64, with SSE4.1, AVX2 and NEON code
paths that give exactly the same output as the plain C code.

Deen 1.0 beta 1 was a closed-source 32-bit MMX/ISSE plugin for AviSynth 2.5, YV12 only. It never got a 64-bit,
high bit depth or VapourSynth version, so scripts built on it (HybridFuPP, for example) could not run on current
systems. This port computes the same 8-bit output as the original for every mode and adds 9-16 bit, every planar
YUV subsampling, Y/Gray, multithreading and VapourSynth.

How it works, formula by formula: [ALGORITHM.md](ALGORITHM.md).

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Modes](#modes)
- [Parameters](#parameters)
- [Formats](#formats)
- [Differences to Deen 1.0 beta 1](#differences-to-deen-10-beta-1)
- [Performance](#performance)
- [Build from source](#build-from-source)
- [Credits and license](#credits-and-license)

## Installation

Release files: `Deen-<version>-<platform>.zip` (plugin and docs) and a VapourSynth wheel.

| | AviSynth+ | VapourSynth |
|---|---|---|
| Plugin | `Deen.dll` into the `plugins64` folder, or `LoadPlugin("Deen.dll")` | `pip install vapoursynth_deen-<version>-py3-none-<platform>.whl` (installs into `site-packages/vapoursynth/plugins`), or the plugin file into a plugin folder / `core.std.LoadPlugin()` |
| File | `Deen.dll` (Windows) | `Deen.dll`, `libDeen.so`, `libDeen.dylib` |
| Needs | AviSynth+ (frame properties are kept from interface version 8) | API 4 (R55 or newer) |

The best instruction set is chosen at runtime; any x86-64 CPU works (C fallback without SSE4.1).

## Usage

AviSynth+, with the argument order of Deen 1.0 beta 1, so old scripts work unchanged:

```
LoadPlugin("Deen.dll")
Deen()                                        # c3d, rad 1, thrY 7, thrUV 9, tthY 4, tthUV 6
Deen("a2d", 2, 30, 0, 0, 0, 0)                # adaptive 5x5, luma only (thrUV 0 changes almost nothing)
Deen("a3d", 1, 10, 24, 3, 5, 0.1, 9.0)        # adaptive spatio-temporal
Deen("w3d", 2, borderfix=true)
```

VapourSynth (namespace `deen`, same parameter names):

```python
import vapoursynth as vs
core = vs.core
clip = core.deen.Deen(clip, mode='a3d', rad=1, thrY=10, thrUV=24, tthY=3, tthUV=5, min=0.1, scd=9.0)
```

## Modes

Every mode averages the `(2*rad+1)^2` window around each pixel; the `3d` modes add the same window in the previous
and the next frame. A sample counts when its absolute difference to the centre pixel is at most the threshold
(`thr` in the current frame, `tth` in the neighbour frames).

| Mode | Behaviour |
|---|---|
| `c2d`, `c3d` | samples above the threshold are replaced by the centre pixel, then all samples are averaged (fixed divisor) |
| `a2d`, `a3d` | only the samples within the threshold are averaged; the threshold falls with the distance from the centre, from `thr` to about `thr * min` at the window corners |
| `w2d`, `w3d` | like `c`, but weighted: the weight falls with the distance from the centre, to about `min` of the centre weight at the corners; in `w3d` the current frame has twice the weight of each neighbour frame |

**Scene changes** (`3d` modes): frame 0 and both frames next to a scene change run the `2d` variant. A scene change
is a mean absolute difference between two neighbouring frames above `scd` (8-bit scale, average of the planes).

## Parameters

| Parameter | Type | Default | Range | Meaning |
|---|---|:--:|---|---|
| `mode` | string | `"c3d"` | `c2d`, `c3d`, `a2d`, `a3d`, `w2d`, `w3d` | see above |
| `rad` | int | 1 | 1..7 (2d), 1..4 (3d) | window radius: 1 = 3x3 ... 7 = 15x15 |
| `thrY`, `thrUV` | int | 7, 9 | 0..255 | spatial threshold for luma / chroma (8-bit scale) |
| `tthY`, `tthUV` | int | 4, 6 | 0..255 | temporal threshold for luma / chroma (3d modes) |
| `min` | float | 0.5 | 0.0..1.0 | a and w modes: threshold / weight at the window corners relative to the centre; 1.0 = no falloff |
| `scd` | float | 9.0 | any | scene change threshold (3d modes); negative = every frame runs 2d |
| `fcf` | string | `""` | `""` only | AviSynth+ only, kept for the argument order (see below) |
| `borderfix` | bool | false | | false: the top and bottom `rad` rows stay unfiltered, as in Deen; true: every row is filtered |
| `opt` | int | -1 | -1..3 | instruction set: -1 best, 0 C, 1 SSE4.1, 2 AVX2, 3 NEON; error if not available |

At 9-16 bit the thresholds stay on the 8-bit scale (shifted by `bits - 8`).

## Formats

| | AviSynth+ | VapourSynth |
|---|---|---|
| Supported | Y, planar YUV and YUVA (alpha copied), 8-16 bit, any subsampling | Gray, YUV, 8-16 bit integer |
| Not supported | RGB, 32 bit float, YUY2 | RGB, float |

Thread-safe and independent of request order (AviSynth+ `MT_NICE_FILTER`, VapourSynth `fmParallel`). Frame
properties are kept.

## Differences to Deen 1.0 beta 1

8-bit output is identical to the original for every mode and parameter in the visible picture, except:

- **Edges:** the original read whatever bytes lay next to the picture in memory (row padding of the source frame)
  for the left and right `rad` columns. This port repeats the edge pixel instead (what the original did with
  `borderfix=true`). With `borderfix=false` the top and bottom `rad` rows stay unfiltered, as in the original.
- **borderfix=true in the 3d modes:** the original padded only the current frame and read the previous and next frame
  `rad` rows too low (and past the end of the plane). This port pads all three frames and uses the same rows.
- **Scene changes:** the original kept a counter between calls, so its result depended on the order in which frames
  were requested. This port decides per frame: frame 0, and any frame whose difference to the previous or next frame
  is above `scd`, runs 2d. For linear playback this is what the original did. The difference is computed on the
  visible pixels (the original also summed row padding).
- **`fcf`** (per-frame settings file) is not supported; a non-empty `fcf` is an error.
- **Ranges:** the original stored every integer argument in one byte (256 became 0) and accepted any `min`; this port
  rejects `thr`/`tth` outside 0..255 and `min` outside 0.0..1.0 (values that made the original wrap its tables).
- Not limited to YV12 and ISSE CPUs.

9-16 bit (new): the same formulas with thresholds shifted by `bits - 8` and exact rounded divisions instead of
Deen's 8-bit reciprocal tables: `c` modes `(sum + ceil(n/2)) / n`, `a` modes `(sum + ceil(cnt/2)) / cnt` (a count of
1 keeps the centre, as at 8 bit), `w` modes `(sum of sample * weight + 0x8000) >> 16` (without the 8-bit `+1` bias,
see ALGORITHM.md).

## Performance

640x352 4:2:0 8 bit, one thread, frames per second (VapourSynth R82, synthetic noise, AMD Ryzen 9 7950X), and the
`a` modes against an Expr implementation of the same formulas (akarin.Expr):

| Call | C | SSE4.1 | AVX2 | Expr (akarin) |
|---|---:|---:|---:|---:|
| `c2d`, rad 1 | 293 | 3019 | 4518 | |
| `c3d`, rad 1 | 113 | 1486 | 2705 | |
| `a2d`, rad 1 | 240 | 1973 | 2390 | 1194 |
| `a2d`, rad 2 | 113 | 1296 | 1877 | 669 |
| `a2d`, rad 7 | 16 | 262 | 560 | 48 |
| `a3d`, rad 1 | 94 | 1145 | 1728 | 609 |
| `a3d`, rad 2 | 37 | 601 | 1091 | 169 |
| `a3d`, rad 4 | 14 | 236 | 507 | 43 |
| `w2d`, rad 2 | 100 | 1368 | 2213 | |
| `w3d`, rad 2 | 32 | 535 | 989 | |

Measure with `test/bench.py`.

## Build from source

Meson and a C++17 compiler (MSVC, GCC or Clang):

```
meson setup build            # Windows: meson setup build --vsenv -Db_vscrt=mt   (or run build.bat)
meson compile -C build
meson test -C build          # every SIMD tier against the C reference
```

Wheel: `pip wheel .` (Windows: add `--config-settings=setup-args=--vsenv --config-settings=setup-args=-Db_vscrt=mt`).

Tests in `test/` (Python with numpy):

| Script | Checks |
|---|---|
| `test_deen.py [plugin]` | VapourSynth: bit for bit against the numpy model `model.py` (written from ALGORITHM.md), every mode, radius and borderfix for every instruction set, 8/10/16 bit, errors |
| `test_avs.py` | AviSynth+ (Windows, `avs2yuv64`, BestSource): the same model, Deen 1.0 beta 1 argument order, errors |
| `kernel_test.cpp` | `meson test`: every SIMD tier against C on random data |
| `bench.py` | speed per mode and instruction set |

## Credits and license

- Deen 1.0 beta 1 © 2003 Marc FD: the original filter, its modes and parameters.
- Basis of this port: the Deen 1.0 beta 1 material ChaosKing shared in the doom9 forum
  ([post](https://forum.doom9.net/showthread.php?p=2033800)); thanks to ChaosKing.
- Deen © 2026 Selur: clean-room AviSynth+ / VapourSynth reimplementation, written from behaviour specifications
  only; it contains no code of the original.

Released under the MIT License, see [LICENSE](LICENSE). The vendored VapourSynth headers in `include/vapoursynth`
are LGPL-2.1, see the license file there.
