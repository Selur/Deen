# Deen: the formulas

Every formula below is computed exactly as written, in integers unless stated otherwise; the C kernels
(`src/core/scalar.hpp`) are the reference and the SIMD kernels give the same bits. `r` is `rad`, the window is
`dy, dx = -r..r`, `c` is the centre pixel of the current frame, `n` a window sample. Thresholds `thr`/`tth` are the
per-plane values: `thrY`/`tthY` for Y, `thrUV`/`tthUV` for U and V.

## 1. Window and sources

- 2d modes: one source, the current frame, `N = (2r+1)^2` samples.
- 3d modes: three sources in this order: current frame (threshold table from `thr`), previous frame and next frame
  (both from `tth`), `N = 3 * (2r+1)^2` samples. All comparisons use the centre pixel `c` of the current frame.
- The centre itself is an ordinary sample of the current frame: `|c - c| = 0` always passes, so it is counted once.
- A sample passes when `|c - n| <= T(dy, dx)` (inclusive).

## 2. Threshold tables

c and w modes: `T(dy, dx) = thr` (current frame) or `tth` (neighbour frames) everywhere.

a modes, per source (`t` = `thr` or `tth`, `min` a 32-bit float, the rest in double precision, in this order):

```
inv = 1.0 / sqrt(double(r*r) + double(r*r))
q   = sqrt(double(dx*dx) + double(dy*dy)) * inv
e   = trunc(((1.0 - double(min)) * t) * q)
T   = (t - (e & 0xFF)) & 0xFF
```

`T` falls linearly with the distance from the centre: `t` at the centre, about `t * min` at the corners.

At 9-16 bit every table entry is shifted left by `bits - 8`.

## 3. c modes

```
v   = pass ? n : c
sum = sum of v over all N samples
8 bit:    out = ((sum + (N+1)/2) * floor(65536/N)) >> 16
9-16 bit: out = (sum + (N+1)/2) / N
```

## 4. a modes

```
sum = sum of n over the passing samples, cnt = their number (>= 1, the centre always passes)
8 bit:    out = ((sum + ceil(cnt/2)) * R(cnt)) >> 16,   R(1) = 0xFFFF, R(n) = floor(65536/n)
9-16 bit: out = cnt == 1 ? c : (sum + ceil(cnt/2)) / cnt
```

`R(1) = 0xFFFF` makes a count of 1 return `c` exactly at 8 bit (`((c+1) * 0xFFFF) >> 16 = c`). Because `R(n)` is
rounded down, the 8-bit result can be one below the rounded mean; this is the original's arithmetic and kept.

## 5. w modes

Weight table (`total = 0x10000` in 2d, `0x4000` in 3d; integer divisions truncate, `min` a 32-bit float):

```
base = total / N1                      N1 = (2r+1)^2
d    = base - trunc(double(min) * base)
q    = 1.0 / sqrt(double(r*r) + double(r*r))
w    = uint16(base - trunc((sqrt(double(dx*dx) + double(dy*dy)) * q) * d))
scale = (total << 8) / sum(w)
w    = uint16((w * scale) >> 8)        for every entry
w(0,0) += total - sum(w)               the remainder goes to the centre, sum(w) = total
```

3d: the current frame uses `2 * w` (sum `0x8000`), the previous and next frame `w` (sum `0x4000` each), so the
three frames together weigh `0x10000` with the current frame counting twice.

```
v = pass ? n : c
8 bit:    sum = min(0xFFFF, sum + ((v * w) >> 8))      per sample (the original's (v << 8) pmulhuw w)
          out = min(255, ((sum + 128) >> 8) + 1)
9-16 bit: sum = sum of v * w                           (fits 32 bit: at most max * 0x10000)
          out = (sum + 0x8000) >> 16
```

The 8-bit `+1` is the original's compensation for the per-sample truncation (a flat area of value `v` comes out as
`v + 1` for small radii). At 9-16 bit there is no truncation per sample and the sum is rounded once, so no bias is
added.

## 6. Borders

All sources are read with edge replication: a sample outside the plane takes the value of the nearest edge pixel.

- `borderfix=false`: rows `r .. h-r-1` are filtered, rows `0 .. r-1` and `h-r .. h-1` are copies of the current
  frame (if `h <= 2r` the plane is copied). Every column is filtered.
- `borderfix=true`: every row and column is filtered.

## 7. Scene changes (3d modes)

```
diff(a, b) = sum over the planes of (sum |a - b| / pixels of the plane) * s,  then * 0.3333333333333333 for 3 planes
             (or / planes), s = 1 at 8 bit, 255 / (2^bits - 1) above (double precision)
frame n runs 2d  <=>  n == 0  or  scd < diff(n, n+1)  or  scd < diff(n-1, n)
```

`n+1` is clamped to the last frame (the last frame compares with itself). `scd` is a 32-bit float, the comparisons
are strict. A 2d run of a 3d mode uses the 2d tables (w modes: `total = 0x10000`, no doubling).
