# Changelog

## 1.0.1 (2026-11-23)

Faster, output unchanged (bit-identical for every mode, radius, borderfix, bit depth and instruction set).

- No padded copy of every source plane per frame: the window rows come from a small per-thread ring of padded rows
  (no allocation, no full-plane copy, stays in the cache). Large gain at HD/UHD and with many threads.
- 9-16 bit: the c and a mode divisions and the 8-bit a mode reciprocals run in SIMD as float divisions (exact for
  these operand ranges) instead of per pixel in scalar code; the c and a mode sums run in 16-bit lanes.
- x86: pass test as one range compare per tap (bounds computed once per threshold value and kept in registers
  per source in the c and w modes), two taps per widening add at 8 bit (pmaddubsw), two taps per multiply-add in
  the 9-16 bit w modes (pmaddwd). NEON keeps |c - n| <= t and 32-bit sums (fewer instructions there), with the
  threshold per source in a register. All: a flat tap list per row.
- Scene change metric (3d modes) with SSE2 sums of absolute differences.
- `test/bench.py`: sizes SD/HD/UHD, 8/16 bit, one and all threads, comparison with neo_smo.Deen.

## 1.0.0 (2026-11-23)

First release: clean-room reimplementation of Deen 1.0 beta 1 (Marc FD, 2003).

- Modes c2d, c3d, a2d, a3d, w2d, w3d with the parameters mode, rad, thrY, thrUV, tthY, tthUV, min, scd, borderfix;
  8-bit output identical to the original in the visible picture.
- AviSynth+ (`Deen`, argument order of Deen 1.0 beta 1) and VapourSynth (`deen.Deen`) from one source.
- 8-16 bit, Y/Gray and every planar YUV subsampling, YUVA (alpha copied) in AviSynth+.
- Edge replication instead of reading row padding; scene changes decided per frame (independent of request order).
- SSE4.1, AVX2 and NEON, bit-identical to C; multithreaded; Windows x64, Linux x86_64, macOS arm64.
- Not supported: fcf files.
