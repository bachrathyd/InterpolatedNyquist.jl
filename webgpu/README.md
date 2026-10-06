# NyquistGPU in the browser (WebGPU)

A self-contained static page that computes the stability charts of the paper's case studies
**on the viewer's own GPU** via WebGPU, in Float32, with live sliders on the model constants.
No build step, no external CDN, no account, no server-side computation.

| file | content |
|---|---|
| `index.html` | page, layout, styles |
| `app.js` | UI: example / constants / resolution / axes / box zoom / hover, render scheduler, `?validate=1` and `?bench=1` modes |
| `engine.js` | WebGPU host code: device, pipelines (default and exact-root variant per example), row-band dispatches, timestamp timing, colouring pass, read-back |
| `examples.js` | the examples: constants, slider knobs, paper axes, march settings, `D(λ, p, c)` in WGSL; host (Float64) effective orders of the bar models |
| `march.wgsl` | the per-point `:unwrap` march (port of `gpu/src/NyquistGPU.jl`) and the exact-root polish |
| `display.wgsl` | colouring (port of `k_display!` / `pixel_colour` of `server.jl`) |
| `validate/web_systems.jl` | the gallery examples in NyquistGPU form `D(λ, p, c)` (same operations as the WGSL) |
| `validate/ref_counts.jl` | Julia script: reference counts and dominant roots from the repository engine |
| `validate/ref_counts.json` | its output (160 × 90 grids, default and a second constant set per example), read by `?validate=1` |

## Examples

Grouped as in the selector; axes, ω_max, step caps and orders are those of the paper's
gallery (`paper/scripts/studies/s08_gallery.jl`, `s10_fractional_controller.jl`). The
formula of `D` is shown under the selector, with a note on how the example is counted.

| example | D(λ) | axes | sliders | march |
|---|---|---|---|---|
| A.1 fourth-order delayed oscillator | `c₁λ⁴ + λ² + 2ζλ + 1 + (P + Dλ)e^{−τλ}` | P, D | τ, ζ, c₁ | n = 4, ω_max = 1e5 |
| A.2 delayed oscillator | `λ² + aλ + k + (b + gλ)e^{−τλ}` (paper: k = g = 0, τ = ½) | a, b | τ, k, g | n = 2, ω_max = 1e4 |
| A.3 distributed delay | `λ² + aλ + k + b e^{−τ₀λ}(1 − e^{−τλ})/λ` (paper: τ = 1) | a, b | τ, k, τ₀ | n = 2, ω_max = 1e4; Taylor series of (1 − e^{−x})/x for \|x\| < ½ (removable singularity) |
| showcase 2-DOF DAE (delayed PD) | `a₁₁a₂₂ − a₁₂²` | P, D | τ, c₁, c₂ | n = 4, ω_max = 1e5 |
| A.7 multi-mode turning | `M₁M₂ + w(1 − e^{−2πλ/Ω})(M₂ + A₂M₁)` | Ω, w | ζ₁, A₂, ω₂ | n = 4, h ≤ 0.05 for ω < 5 |
| A.4 neutral | `λ² + aλ²e^{−τλ} + dλ + k + c e^{−τλ}` (τ = 1, d = 0, k = 1) | a, c | τ, d, k | n = 2, ω_max = 200 (fixed), h ≤ π/(2τ) everywhere |
| A.5 high-gain neutral | same, d = 5, k = 0 | a, c | τ, d, k | as A.4 |
| A.6 PDA control | `λ² + 2ζλ + 1 + (P + Dλ + Aλ²)e^{−τλ}` (ζ = 0.05, D = 0.1, τ = 1) | P, A | ζ, D, τ | n = 2, ω_max = 500 (fixed), h ≤ π/(2τ); dashed: \|A\| = 1 |
| A.8 exact transcendental rod (Zhang & Stépán) | `1 − K e^{−rλ}/cosh γ`, `γ = λ√(1 + c/λ)/√(1 + ηλ)` (paper: c = 0) | r = τ/T, K | η (Kelvin–Voigt), c (external damping) | numerator counted, n_eff from the denominator (host), ω_max = 400 (fixed), h ≤ π/(2 r_max) for ω < 40 |
| A.9 same bar, N-element FEM | `det(λ²M + λC + K + F(λ))`, rank-one feedback | r, K | η, N (2–24) | as A.8 |
| A.11 fractional oscillator | `λ^α + cλ^β + k e^{−τλ}` (α = 1.8, β = 0.8, c = 0.5) | k, τ | α, β, c | n = α, ω_max = 1e4, principal branch |
| A.12 fractional PI controller (Gao, Zhai & Liu, μ = 1.5) | `s^μ(Ts^ν + 1) + K e^{−Ls}(k_p s^μ + k_i)` | k_p, k_i | μ, L, K | n = μ + ν, ω_max = 1e4 |

Notes on the harder ones:

* **Neutral systems (A.4–A.6)**: the phase ripple of a neutral system never decays, so a
  larger ω_max buys nothing; the window is fixed at the gallery's 200 / 200 / 500 (the
  default 1e5 fails on these), the order n = 2 is passed exactly, and the step is capped at
  π/(2τ) along the whole line (recomputed when the τ slider moves). Most points carry the
  integer-residual flag (the tail oscillation reaches arcsin|a|/π < ½), as in the paper.
* **Rod (A.8)** — the paper's recipe: the entire numerator `cosh γ − K e^{−rλ}` is
  marched, with the effective order `n_eff = 2Φ_den(ω_max)/π` of the parameter-independent,
  stable denominator `cosh γ`. In Float32 `cosh γ ~ e^{120}` overflows at ω = 400, so both
  are multiplied by the analytic, zero-free `e^{−γ}`: the march sees
  `(1 + e^{−2γ})/2 − K e^{−rλ−γ}` (no overflow, scale-free) and the denominator becomes
  `(1 + e^{−2γ})/2`, whose phase the host measures (Float64, adaptive unwrap, ~5 ms) whenever
  η or c change. The common `e^{−γ}` phase cancels from the count; the Julia check compares
  this with the paper's unscaled form (identical counts at every unflagged point). The top row
  K = 1 has the root λ = 0 on the line.
* **FEM bar (A.9)**: `Q₀ = λ²M + λC + K` is tridiagonal, so `det Q₀` is a continuant and the
  rank-one feedback enters through `(Q₀⁻¹)_{N1} = (−1)^{N+1} o^{N−1}/det Q₀`: the entire
  numerator `det Q₀ + c(λ)(−1)^{N+1}o^{N−1}` costs O(N) per evaluation (no LU), divided by
  `((4h/6)(λ + √3/h)²)^N` against overflow (poles far left, the same divisor for the host
  denominator). It is interactive at 320 × 180 on an integrated GPU.
* **Fractional (A.11, A.12)**: `λ^μ = exp(μ log λ)` (principal branch); only σ = 0 is
  admissible. The |D| minimum at ω₀ is the branch point, not a root, so the seed rule there
  is switched off for these two (`branch0`).
* Not included: the 50 × 50 dense determinant (A.10) and the 100-vehicle CCC ring — the paper
  keeps both on the phase-ODE back-end because the unwrap march is not reliable there.

## Exact root mode

The default colouring is the first-order estimate of the engine (`refine = 0`): one Newton
step from each of the 4 deepest |D| minima found along the line, σ = the rightmost. Where a
long step holds a maximum and a minimum of |D| (its end slopes then do not bracket the
minimum), or where deeper minima crowd out the dominant one, that root is lost and the
stable domain shows dark streaks or speckles (very visible on the showcase).

**"Exact root (refined)"** implements the paper's dominant-root recipe in the shader:

* 8 minima are tracked (with their ω); minima are also located inside steps whose end slopes
  do not bracket them (sign changes of `d|p|²/dt` of the step's Hermite model on 4 sub-intervals);
  minima whose one-step estimate falls outside the trust region are kept as seeds on the line
  (ranked behind every trusted one);
* each seed and the one-step estimate from ω₀ is polished by damped Newton steps in the complex
  plane, `λ ← λ − D/D_λ` with `D_λ = −i D_ω` from the same dual-number evaluation: steps
  limited to ½ max(|λ|, 1), a step kept only if |D| decreases (otherwise halved), at most 8
  evaluations, stop when converged;
* converged = the Newton step is below the Float32 resolution, or below 1e-3·max(1, |λ|/100)
  *and* contracting tenfold (a drift into a branch point is not a root), and the root lies
  within max(1, |λ_seed|/2) of its seed (Newton thrown off a saddle of |D| between two close
  roots); a trusted seed that does not converge keeps its first-order estimate;
* σ = the largest real part; where the count proves the point stable a root right of
  σ + 0.02 is spurious and dropped (also applied to the default first-order colouring — the
  consistency filter of the paper's charts).

Cost: 1.3–2.2 × the default chart (measured below); the polish adds only ~3–40 evaluations
per point, the rest is the heavier march (8 slots, Hermite scan).

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
  hover line shows `Z`, `σ` (marked "refined" in exact mode) and the number of `D`
  evaluations at a point. `?ex=<key>` opens an example (`fourth`, `algebraic`,
  `distributed`, `showcase`, `turning`, `neutral`, `neutral_hg`, `pda`, `rod`, `fem`, `frac`,
  `gao`), `&exact=1` with exact roots — links that can be sent around.
* `index.html?validate=1` — computes the 160 × 90 grids of `validate/ref_counts.json`
  (every example, default and a second constant set) in both modes and reports the counts
  `Z` that differ from the Julia engine (Float32 and Float64 runs; separately those at
  unflagged points), the σ deviation of the default mode, the host-vs-Julia effective
  orders, the exact mode against Float64 reference dominant roots, a speckle measure and
  the GPU times. `&only=rod,fem` restricts it.
* `index.html?bench=1` — every example at its default resolution, default and exact mode
  (median of 3 after a warm-up); `?bench=hd` the same at 1920 × 1080.

Regenerating the reference (Julia 1.12, CPU, ~6 min with 12 threads):

```
julia --project=gpu/scripts -t auto webgpu/validate/ref_counts.jl --cpu [--only rod,fem]
```

## Share it with colleagues

The folder is static, so any static host works; all paths are relative.

* **GitHub Pages.** This repository already publishes its Documenter docs from the
  `gh-pages` branch. Copy the `webgpu/` folder into that branch as `webgpu/` (Documenter's
  `deploydocs` only rewrites its own `dev/` and version folders) and the demo is at
  `https://bachrathyd.github.io/InterpolatedNyquist.jl/webgpu/`.
* **Without touching Pages**: a CDN that serves GitHub files with correct MIME types, e.g.
  `https://raw.githack.com/bachrathyd/InterpolatedNyquist.jl/webgpu-demo/webgpu/index.html`
  (the plain `raw.githubusercontent.com` URL does not work: it serves `text/plain`).
* Or zip the folder; the recipient runs `python -m http.server` in it.

WebGPU needs a secure context: `https://` or `http://localhost` — not a plain-http LAN address.

## What is computed

Per chart point `p` (one GPU invocation, 8 × 8 workgroups) the `:unwrap` march of
`NyquistGPU.jl` (settings of the interactive server's `F32` format, `refine = none`):

* `λ = σ + iω`, `σ = 0`, `ω` from `ω0 = 1e-9` to `ω_max` (per example, selectable where the
  example allows it); one evaluation of `D` and `dD/dω` per step through complex **dual
  numbers** (`struct CD { v, d }`, hand-written WGSL complex/dual arithmetic incl. `exp`,
  `log`, `sqrt`, division);
* the observed phase increment `angle(D_b / D_a)` is accepted when it agrees with the
  trapezoid prediction `h (θ'_a + θ'_b)/2` within `tol = 0.3` rad; step factor
  `clamp(0.9 (tol/err)^{1/3}, 0.2, 4)`, step cap `h ≤ max(ω, 1)` plus the example's cap over
  its resonance band; sub-resolution transitions decided by the side of the root (flag 2);
* `Z = n/2 − Φ/π` with the leading order (or effective order) `n` of the example;
* σ as described above; stable points are coloured by viridis (`1 − σ/σ_floor`), unstable
  points red by `Z` (capped at 6), the boundary white, failed marches grey — the colour scheme
  of `gpu/interactive/server.jl`.

Portability details: WGSL guarantees neither IEEE `Inf`/`NaN` nor accurate `sin`/`cos`
outside `[−π, π]`, so the Julia NaN/Inf sentinels are replaced by explicit validity flags
(infinite step caps are passed as 3e38), and `sin`/`cos`/`atan2` are Cephes single-precision
polynomials (π/2 Cody–Waite reduction).

Large charts are split into row bands; each band is its own submission, sized adaptively to
~60 ms of GPU time, with two in flight. No single submission comes near the 2 s Windows
TDR / browser watchdog. The GPU time is the sum of the bands' compute-pass timestamps
(`timestamp-query`); without that feature it falls back to wall-clock busy time ("wall").

## Validation and timings (Intel UHD Graphics 630 laptop GPU, Chrome, 2026-10-06)

`julia --project=gpu/scripts -t auto webgpu/validate/ref_counts.jl --cpu` (NyquistGPU,
CPU backend, Float32 and Float64), 160 × 90 grid per example, each with its own ω_max and
step caps; default constants and a second constant set ("@alt", sliders moved):

* **Counts: 0 differing `Z` at unflagged points on all 24 grids** (also 0 between the
  default and the exact pipeline). The only differences (rod: 0 vs Julia F32, 15 / 153
  vs Julia F64; fem: 72 / 82 vs F64; neutral: 1) are flagged points with a root on the line
  — the K = 1 row of the bar charts (λ = 0), where `Z_raw` is a half-integer and the rounding
  is a coin flip; Julia's own F32 and F64 runs differ there the same way.
* The scale-free bar forms agree with the paper's formulations (Float64: `cosh γ` with its
  measured phase, n_eff = 98.85; LU form of the FEM determinant, n_eff = 23.88) at every
  unflagged point; host (JS) and Julia effective orders agree to 1e-8.
* Default-mode σ against the Julia F32 run: median |Δσ| 2e-8 – 2e-6, 99th percentile ≤ 3e-5.
* Exact mode against Float64 dominant roots at 16 stable points per grid (certified by
  counting on shifted lines for the retarded and neutral examples): within 1e-3 at 373 of
  378 points. The exceptions: 3 neutral_hg points, where the root chain approaches the
  asymptote Re λ = ln|a|/τ only as ω → ∞ (the certified value is that supremum, the march
  sees roots up to ω_max); one gao@alt point where the exact mode found a root right of the
  reference's (−1.194 + 6.08i, confirmed by a brute-force root search: the reference missed
  it); one frac@alt point without any minimum.
* Speckle (stable points whose clamped σ jumps against its neighbours' median by more than
  5 % of the colour range), full chart at the default resolution, default → exact:
  showcase 1768 → 21, gao 105 → 0, fourth 116 → 0, turning 109 → 52, pda 80 → 35 (the
  remaining ones in exact mode are genuine ridges, e.g. where two real roots collide).

Default resolution, default constants (`?bench=1`, median of 3):

| example | resolution | GPU ms default | evals / pt | GPU ms exact | evals / pt | exact / default |
|---|---|---|---|---|---|---|
| fourth | 960 × 540 | 24 | 32 | 50 | 38 | 2.1 |
| algebraic (A.2) | 960 × 540 | 15 | 22 | 32 | 26 | 2.2 |
| distributed (A.3) | 960 × 540 | 26 | 24 | 47 | 27 | 1.8 |
| showcase | 960 × 540 | 47 | 53 | 88 | 60 | 1.9 |
| turning (A.7) | 960 × 540 | 137 | 139 | 230 | 162 | 1.7 |
| neutral (A.4) | 480 × 270 | 108 | 293 | 161 | 323 | 1.5 |
| neutral_hg (A.5) | 480 × 270 | 105 | 288 | 160 | 318 | 1.5 |
| pda (A.6) | 480 × 270 | 249 | 704 | 357 | 735 | 1.4 |
| rod (A.8) | 480 × 270 | 188 | 312 | 259 | 341 | 1.4 |
| fem (A.9, N = 12) | 320 × 180 | 155 | 313 | 205 | 345 | 1.3 |
| frac (A.11) | 960 × 540 | 32 | 27 | 54 | 39 | 1.7 |
| gao (A.12) | 480 × 270 | 166 | 269 | 252 | 310 | 1.5 |

Full HD (1920 × 1080 = 2.07 M points, `?bench=hd`), the examples that stay below ~1 s there:

| example | GPU ms default | Mpts/s | evals / pt | GPU ms exact | exact / default |
|---|---|---|---|---|---|
| fourth | 99 | 21.1 | 32 | 193 | 2.0 |
| algebraic | 57 | 36.7 | 22 | 125 | 2.2 |
| distributed | 95 | 21.8 | 24 | 182 | 1.9 |
| showcase | 188 | 11.0 | 53 | 321 | 1.7 |
| turning | 519 | 4.0 | 139 | 892 | 1.7 |
| frac | 118 | 17.6 | 27 | 212 | 1.8 |

## Limitations

* Float32 only; no certification of σ by counting on shifted lines (not admissible for the
  fractional and bar examples anyway), no Float16 mode, no flagged-point re-check.
* A new model needs its `charD` in WGSL in `examples.js` (written with the dual helpers).
* The grid is evaluated in full (no adaptive coarse-to-fine `run_adaptive!`).
* `σ` of the march line is fixed at 0; the sin/cos argument reduction is exact up to
  |τω| ≈ 1e5.
