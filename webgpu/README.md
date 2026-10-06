# NyquistGPU in the browser (WebGPU)

A self-contained static page that computes the stability charts of the three
time-independent NyquistGPU examples (`gpu/scripts/systems.jl`: `fourth`, `showcase`,
`turning`) **on the viewer's own GPU** via WebGPU, in Float32. No build step, no external
CDN, no account, no server-side computation.

| file | content |
|---|---|
| `index.html` | page, layout, styles |
| `app.js` | UI: example / constants / resolution / axes / box zoom / hover, render scheduler, `?validate=1` and `?bench=1` modes |
| `engine.js` | WebGPU host code: device, pipelines, row-band dispatches, timestamp timing, colouring pass, read-back |
| `examples.js` | the three examples: constants, slider knobs (as `gpu/interactive/server.jl`), axes, and `D(λ, p, c)` in WGSL |
| `march.wgsl` | the per-point `:unwrap` march (port of `gpu/src/NyquistGPU.jl`) |
| `display.wgsl` | colouring (port of `k_display!` / `pixel_colour` of `server.jl`) |
| `validate/ref_counts.jl` | Julia script: reference counts from the repository engine |
| `validate/ref_counts.json` | its output (160 × 90 grids), read by `?validate=1` |

## Run it

The page loads its WGSL files with `fetch`, so it must be served over http(s) (a plain
`file://` open does not work). Any static file server will do:

```
cd webgpu
python -m http.server 8000
```

then open <http://localhost:8000/> in **Chrome or Edge 113+** (Windows, macOS, ChromeOS;
on Linux/Android enable `chrome://flags/#enable-unsafe-webgpu`). Recent Safari and Firefox
releases with WebGPU should work too (untested). If WebGPU is missing the page says so
instead of failing.

* `index.html` — interactive charts. Sliders recompute live (switch off "Recompute while
  dragging sliders" on slow GPUs); drag on the chart to zoom, double-click to reset; the
  hover line shows `Z`, `σ` and the number of `D` evaluations at a point.
* `index.html?validate=1` — computes the 160 × 90 grids of `validate/ref_counts.json`
  (default and an alternative constant set per example) and reports the percentage of
  counts `Z` that differ from the Julia engine (Float32 and Float64 runs), the σ deviation,
  and difference maps.
* `index.html?bench=1` — full-HD (1920 × 1080) timing of the three examples, median of 3
  after a warm-up.

## Share it with colleagues

The folder is static, so any static host works.

* **GitHub Pages.** This repository already publishes its Documenter docs from the
  `gh-pages` branch. Copy the `webgpu/` folder into that branch as `webgpu/` (Documenter's
  `deploydocs` only rewrites its own `dev/` and version folders) and the demo is at
  `https://bachrathyd.github.io/InterpolatedNyquist.jl/webgpu/`. Alternatively publish a
  separate repository (or a branch whose root is this folder) with Pages "deploy from a
  branch".
* **Without touching Pages**: a CDN that serves GitHub files with correct MIME types, e.g.
  `https://raw.githack.com/bachrathyd/InterpolatedNyquist.jl/webgpu-demo/webgpu/index.html`
  (the plain `raw.githubusercontent.com` URL does not work: it serves `text/plain`).
* Or zip the folder; the recipient runs `python -m http.server` in it.

WebGPU needs a secure context: `https://` or `http://localhost` — not a plain-http LAN address.

## What is computed

Per chart point `p` (one GPU invocation, 8 × 8 workgroups) the `:unwrap` march of
`NyquistGPU.jl` (settings of the interactive server's `F32` format, `refine = none`):

* `λ = σ + iω`, `σ = 0`, `ω` from `ω0 = 1e-9` to `ω_max` (default `1e5`, selectable);
  one evaluation of `D` and `dD/dω` per step through complex **dual numbers**
  (`struct CD { v, d }`, hand-written WGSL complex/dual arithmetic incl. `exp`);
* the observed phase increment `angle(D_b / D_a)` is accepted when it agrees with the
  trapezoid prediction `h (θ'_a + θ'_b)/2` within `tol = 0.3` rad; step factor
  `clamp(0.9 (tol/err)^{1/3}, 0.2, 4)`, step cap `h ≤ max(ω, 1)` (`turning`: `h ≤ 0.05` for
  `ω < 5`); sub-resolution transitions decided by the side of the root (flag 2);
* `Z = n/2 − Φ/π` with the leading order `n = 4`;
* the deepest four `|D|` minima (seed rule at `ω0`, Hermite/Illinois minimum inside a
  step, one Newton step, trust radius `4h`) give first-order root estimates; `σ` = the
  rightmost — the colour of a stable point (viridis, `1 − σ/σ_floor`); unstable points are
  red by `Z` (capped at 6), the boundary is white, failed marches grey — the colour scheme
  of `gpu/interactive/server.jl`.

Portability details: WGSL guarantees neither IEEE `Inf`/`NaN` nor accurate `sin`/`cos`
outside `[−π, π]`, so the Julia NaN/Inf sentinels are replaced by explicit validity flags,
and `sin`/`cos`/`atan2` are Cephes single-precision polynomials (π/2 Cody–Waite reduction).

Large charts are split into row bands; each band is its own submission, sized adaptively to
~60 ms of GPU time, with two in flight. No single submission comes near the 2 s Windows
TDR / browser watchdog. The GPU time is the sum of the bands' compute-pass timestamps
(`timestamp-query`, quantised to 0.1 ms by Chrome); without that feature it falls back to
wall-clock busy time (marked "wall").

## Validation and timings (Intel UHD Graphics 630, Chrome, 2026-10-06)

`julia --project=gpu/scripts -t auto webgpu/validate/ref_counts.jl --cpu` (NyquistGPU,
CPU backend, Float32 and Float64), 160 × 90 grid, `ω_max = 1e5`:

| grid | differing `Z` vs Julia F32 | vs Julia F64 | median / 99th pct. \|Δσ\| (stable points) |
|---|---|---|---|
| fourth | 0 / 14400 (0.000 %) | 0.000 % | 4e-8 / 5e-7 |
| fourth@alt (ζ = 0.08, τ = 1) | 0 (0.000 %) | 0.000 % | 3e-8 / 4e-7 |
| showcase | 0 (0.000 %) | 0.000 % | 3e-8 / 3e-7 |
| showcase@alt (c₁ = 0.2, c₂ = 0.1, τ = 1) | 0 (0.000 %) | 0.000 % | 2e-8 / 4e-7 |
| turning | 0 (0.000 %) | 0.000 % | 2e-8 / 2e-7 |
| turning@alt (ζ₁ = 0.05, A₂ = 1, ω₂ = 3) | 0 (0.000 %) | 0.000 % | 2e-8 / 1e-7 |

Full-HD chart (1920 × 1080 = 2.07 M points), default constants, `?bench=1`:

| example | GPU ms / chart | Mpts/s | D evaluations / point |
|---|---|---|---|
| fourth | 116 | 17.8 | 32 |
| showcase | 233 | 8.9 | 53 |
| turning | 659 | 3.1 | 139 |

## Limitations

* Float32 only, first-order σ estimate (no Newton polish, no certification by counting on
  shifted lines, no Float16 mode, no flagged-point re-check) — the σ colouring shows the
  same speckle as the engine's `refine = none` mode deep inside the stable domain.
* Only the three built-in models; a new model needs its `charD` in WGSL in `examples.js`
  (written with the dual helpers `dadd`, `dmul`, `dscale`, `daddr`, `dexp` …).
* The grid is evaluated in full (no adaptive coarse-to-fine `run_adaptive!`).
* `σ` of the march line is fixed at 0; delays enter only through `exp`, and the sin/cos
  argument reduction is exact up to |τω| ≈ 1e5 (beyond, the delay terms are negligible
  against the λ⁴ term for these models).
