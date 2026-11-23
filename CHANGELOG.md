# Changelog

## 1.0.0 (2026-11-23)

First release: clean-room reimplementation of Deen 1.0 beta 1 (Marc FD, 2003).

- Modes c2d, c3d, a2d, a3d, w2d, w3d with the parameters mode, rad, thrY, thrUV, tthY, tthUV, min, scd, borderfix;
  8-bit output identical to the original in the visible picture.
- AviSynth+ (`Deen`, argument order of Deen 1.0 beta 1) and VapourSynth (`deen.Deen`) from one source.
- 8-16 bit, Y/Gray and every planar YUV subsampling, YUVA (alpha copied) in AviSynth+.
- Edge replication instead of reading row padding; scene changes decided per frame (independent of request order).
- SSE4.1, AVX2 and NEON, bit-identical to C; multithreaded; Windows x64, Linux x86_64, macOS arm64.
- Not supported: fcf files.
