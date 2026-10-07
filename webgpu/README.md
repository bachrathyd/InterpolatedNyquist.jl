# NyquistGPU in the browser (WebGPU)

A self-contained static page that computes the stability charts of the paper's case studies
**on the viewer's own GPU** via WebGPU, in Float32, with live sliders on the model constants.
The characteristic function is typed as text: any expression in λ, every unknown name becomes a
parameter with a slider, any two parameters are the chart axes. The paper's examples are such
texts too. No build step, no external CDN, no account, no server-side computation.

| file | content |
|---|---|
| `index.html` | page, layout, styles |
| `app.js` | UI: equation box, parameter table (sliders, ranges, X / Y axes), march settings, resolution / box zoom / hover, render scheduler, share links, own equation in `localStorage`, `?validate=1`, `?bench=1`, `?bench=compare` |
| `expr.js` | the equation language: tokenizer (whitelist), parser, typed expression DAG with common-subexpression sharing, WGSL generation, Float64 host evaluator, order estimate, setting expressions |
| `engine.js` | WebGPU host code: device, pipelines (cached by the generated code; default and exact-root variant), row-band dispatches, timestamp timing, colouring pass, read-back |
| `examples.js` | the examples as equation texts with parameter defaults / ranges, axes and march settings; the built-in FEM bar (hand-written WGSL, host effective order) |
| `handwritten.js` | the hand-written WGSL of the first version, only for `?bench=compare` |
| `march.wgsl` | the per-point `:unwrap` march (port of `gpu/src/NyquistGPU.jl`) and the exact-root polish |
| `display.wgsl` | colouring (port of `k_display!` / `pixel_colour` of `server.jl`) |
| `validate/web_systems.jl` | the gallery examples in NyquistGPU form `D(λ, p, c)` (same operations as the WGSL) |
| `validate/ref_counts.jl` | Julia script: reference counts and dominant roots from the repository engine |
| `validate/ref_counts.json` | its output (160 × 90 grids, default and a second constant set per example), read by `?validate=1` |

## Your own equation

Pick **Own example** (it starts as A.1 and is kept in this browser's `localStorage`; the page
also works where storage is blocked) or edit any example's text. The box takes

```
# comment
γ = λ*sqrt(1 + c/λ)/sqrt(1 + η*λ)        # helper: usable in later lines, computed once
(1 + exp(-2*γ))/2 - K*exp(-r*λ - γ)       # the last line (or a line  D = ...) is D(λ)
```

* numbers (`1.5e-3`), `+ - * / ^` (`**`, `·`, `−` and superscripts `λ²`, `λ⁻¹` accepted), unary
  minus, parentheses; `exp log (ln) sqrt sin cos tan sinh cosh tanh exprel pow(a, b)`,
  `exprel(x) = (eˣ − 1)/x` (entire; evaluated by its Taylor series near 0 -- for distributed
  delays, `(1 − e^{−τλ})/λ = τ·exprel(−τλ)`); constants `pi` (`π`), `i`; the variable `λ`
  (or `lambda`; the **λ** button inserts it);
* Greek / Unicode names with subscripts (`τ₀`, `ζ₁`, `ω₂`, `k_p`); implicit multiplication is
  an error (`2λ` → "write 2*λ"); non-analytic functions (`abs`, `real`, `imag`, `conj`, `min`,
  `max`, ...) are rejected; a line that ends or starts with an operator, or an open
  parenthesis, continues on the next line; parse errors are shown with a caret under the line.
* Every identifier that is not λ, a function, a constant or a helper is a **parameter** (in the
  order of first appearance, at most 16). Each has a row: value, min, slider, max, and **X / Y**
  buttons. The two axis parameters span the chart over their [min, max] (drag-zoom writes the
  new ranges back); the others are live sliders. A new parameter starts at 1 on [0, 2] (or at
  its old value when a line `name = number` was deleted).
* Typing recompiles 400 ms after the last key (Ctrl+Enter at once). The pipeline is rebuilt
  only when the text or the axis choice changes (cached by the generated code); sliders only
  rewrite the uniform buffer.
* **Order n** (`Z = n/2 − Φ/π`): estimated on the host as the least-squares slope of `ln|D(s)|`
  over 9 log-spaced real `s ∈ [s*/10, s*]`, `s* = 1e8` (reduced while `|D|` is not finite),
  at an interior point of the axis ranges; one decade lower as a check (exponential growth,
  e.g. an advanced term, gives a warning), and at the four chart corners (a warning if n
  changes over the chart). Re-estimated on every slider move (a few hundred host
  evaluations). Fractional powers give a non-integer n, as they should. A real-coefficient
  check `D(λ̄) = conj D(λ)` warns about complex coefficients.
* **Advanced**: n (override), ω_max (default 1e5), tolerance (0.3 rad), ω₀ (1e-9), step cap
  h_max for ω below a band (empty band: the whole line), branch point at λ = 0 (auto: `λ^x`
  with non-integer x, `log λ`, `sqrt λ` -- the |D| minimum at ω₀ is then no root estimate).
  These are expressions of the parameters (`pi/(2*τ)`; an axis parameter stands for
  max |value| over its range) with `pi`, `inf`, `min`, `max`, `abs`. Neutral systems need a
  small explicit ω_max (200–500) and `h_max = pi/(2*τ)` (hint in the panel).
* **Copy link** puts the equation, parameter values and ranges, axes, settings, resolution and
  the exact flag into the URL hash (`#m=` base64url of a small JSON) and restores it on load;
  `?ex=<key>&exact=1` links still work.
* "Generated WGSL" (in Advanced) shows the shader code of the current equation.

**Safety.** The text is tokenized against a whitelist and parsed to an AST; WGSL and the host
evaluator are generated only from the typed DAG built from it. No `eval` / `new Function`; no
user text enters the shader: parameters become `K[j]` / `p.x` / `p.y`, helpers and
subexpressions `t<id>`, numbers are re-printed from their parsed values (checked against the
Float32 range).

**Code generation.** Each DAG node has a type: R (real, λ-free: f32 arithmetic), C (complex
constant: `vec2`), D (dual number in λ: value and d/dω). The operator picks the cheapest form
(`dscale`, `daddr`, `dmul`, `dmulc`, ...); identical subexpressions are one node (a helper
used twice, the `log λ` of two fractional powers, the `e^{−τλ}` of two terms, `λ²` inside `λ⁴`);
integer exponents become repeated squaring, ±½ `sqrt`, other exponents `exp(b·log a)` on the
principal branch; constant subexpressions are folded in Float64; `−(r·x)` puts the sign on the
real factor. The result is straight-line WGSL, e.g. for A.1:

```
fn charD(l: CD, p: vec2<f32>) -> CD {
    let t3 = dmul(l, l);
    let t4 = dmul(t3, t3);
    let t5 = dscale(t4, K[0]);
    ...
    let t20 = dscale(l, t19);       // t19 = -K[4] (τ)
    let t21 = dexp(t20);
    let t22 = dmul(t17, t21);       // t17 = P + Dλ
    return dadd(t13, t22);
}
```

## Examples

Grouped as in the selector; axes, ω_max, step caps and orders are those of the paper's
gallery (`paper/scripts/studies/s08_gallery.jl`, `s10_fractional_controller.jl`). Every
example except the FEM bar is an equation text (`examples.js`), written operation for operation
like the Julia reference, and runs through exactly the same path as a typed equation; selecting
it writes its text, parameter defaults / ranges and settings into the panel (**Reset** restores
them).

| example | text | axes | settings |
|---|---|---|---|
| A.1 fourth-order delayed oscillator | `c₁*λ^4 + λ^2 + 2*ζ*λ + 1 + (P + D*λ)*exp(-τ*λ)` | P, D | ω_max 1e5 |
| A.2 delayed oscillator | `λ^2 + a*λ + k + (b + g*λ)*exp(-τ*λ)` | a, b | ω_max 1e4 |
| A.3 distributed delay | `λ^2 + a*λ + k + b*τ*exp(-τ₀*λ)*exprel(-τ*λ)` | a, b | ω_max 1e4 |
| showcase 2-DOF DAE | helpers `a₁₁`, `a₁₂`, `a₂₂`; `a₁₁*a₂₂ - a₁₂^2` | P, D | ω_max 1e5 |
| A.7 multi-mode turning | helpers `M₁`, `M₂`; `M₁*M₂ + w*(1 - exp(-2*pi/Ω*λ))*(M₂ + A₂*M₁)` | Ω, w | h_max 0.05 for ω < 5 |
| A.4 / A.5 neutral | `λ^2 + a*λ^2*exp(-τ*λ) + d*λ + k + c*exp(-τ*λ)` | a, c | ω_max 200, h_max `pi/(2*τ)` (whole line) |
| A.6 PDA control | `λ^2 + 2*ζ*λ + 1 + (P + D*λ + A*λ^2)*exp(-τ*λ)` | P, A | ω_max 500, h_max `pi/(2*τ)`; dashed \|A\| = 1 |
| A.8 exact rod | `γ = λ*sqrt(1 + c/λ)/sqrt(1 + η*λ)`; `(1 + exp(-2*γ))/2 - K*exp(-r*λ - γ)` | r, K | ω_max 400, h_max `pi/(2*max(r, 1))` for ω < 40 |
| A.9 FEM bar (built-in) | `det(λ²M + λC + K + F(λ))`, N elements (2–24) | r, K | as A.8; n_eff from the host |
| A.11 fractional oscillator | `λ^α + c*λ^β + k*exp(-τ*λ)` | k, τ | ω_max 1e4; branch point (auto) |
| A.12 fractional PI (Gao, Zhai & Liu) | `λ^μ*(T*λ^ν + 1) + K*exp(-L*λ)*(k_p*λ^μ + k_i)` | k_p, k_i | ω_max 1e4; branch point (auto) |

All orders are estimated (none is stored): 4, 2, 2, 4, 4, 2, 2, 2, ≈ 0 (rod), α, μ + ν.

Notes on the harder ones:

* **Neutral systems (A.4–A.6)**: the phase ripple of a neutral system never decays, so a
  larger ω_max buys nothing; the window is the gallery's 200 / 200 / 500 (the default 1e5
  fails on these), and the step is capped at π/(2τ) along the whole line (an expression of
  the τ slider). Most points carry the integer-residual flag (the tail oscillation reaches
  arcsin|a|/π < ½), as in the paper.
* **Rod (A.8)** — the paper's recipe counts the entire numerator `cosh γ − K e^{−rλ}` with the
  effective order of the stable denominator `cosh γ`. In Float32 `cosh γ ~ e^{120}` overflows at
  ω = 400, so the text multiplies it by the analytic, zero-free `e^{−γ}`:
  `(1 + e^{−2γ})/2 − K e^{−rλ−γ}`. Its order estimated on the real axis is 0 (it tends to ½),
  which reproduces the previous host-measured n_eff of the scaled denominator (6e-10, 8e-6 at
  the second constant set) -- identical counts. The top row K = 1 has the root λ = 0 on the line.
* **FEM bar (A.9)** stays built-in (a loop over N elements is no closed expression; the box
  shows a read-only description): `Q₀ = λ²M + λC + K` is tridiagonal, so `det Q₀` is a continuant
  and the rank-one feedback enters through `(Q₀⁻¹)_{N1} = (−1)^{N+1} o^{N−1}/det Q₀`: the entire
  numerator `det Q₀ + c(λ)(−1)^{N+1}o^{N−1}` costs O(N) per evaluation, divided by
  `((4h/6)(λ + √3/h)²)^N` against overflow; its effective order is the host-measured phase of
  that denominator (Float64, adaptive unwrap, ~15 ms per change of η or N).
* **Fractional (A.11, A.12)**: `λ^μ = exp(μ log λ)` (principal branch); only σ = 0 is
  admissible. The |D| minimum at ω₀ is the branch point, not a root: the branch-point switch
  is detected automatically (`log λ` in the DAG).
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
  evaluations at a point. `?ex=<key>` opens an example (`own`, `fourth`, `algebraic`,
  `distributed`, `showcase`, `turning`, `neutral`, `neutral_hg`, `pda`, `rod`, `fem`, `frac`,
  `gao`), `&exact=1` with exact roots; **Copy link** adds the full state (`#m=...`).
* `index.html?validate=1` — computes the 160 × 90 grids of `validate/ref_counts.json`
  (every example, default and a second constant set) in both modes and reports the counts
  `Z` that differ from the Julia engine (Float32 and Float64 runs; separately those at
  unflagged points), the σ deviation of the default mode, the host-vs-Julia effective
  orders, the exact mode against Float64 reference dominant roots, a speckle measure and
  the GPU times. Every example goes through its equation text (parser, generated WGSL,
  estimated order, setting expressions, which are also compared with the reference's march
  settings). `&only=rod,fem` restricts it.
* `index.html?bench=1` — every example at its default resolution, default and exact mode
  (median of 3 after a warm-up); `?bench=hd` the same at 1920 × 1080.
* `index.html?bench=compare` — generated vs the hand-written WGSL of the first version
  (`handwritten.js`), same jobs, default and exact mode, median of 5 interleaved runs, with
  the differing counts (`?bench=compare-hd` at 1920 × 1080).

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

* `λ = σ + iω`, `σ = 0`, `ω` from `ω0 = 1e-9` to `ω_max` (per equation, Advanced); one evaluation of `D` and `dD/dω` per step through complex **dual
  numbers** (`struct CD { v, d }`, WGSL complex/dual arithmetic incl. `exp`, `log`, `sqrt`,
  division, trigonometric / hyperbolic functions and `exprel`, called by the generated code);
* the observed phase increment `angle(D_b / D_a)` is accepted when it agrees with the
  trapezoid prediction `h (θ'_a + θ'_b)/2` within `tol = 0.3` rad; step factor
  `clamp(0.9 (tol/err)^{1/3}, 0.2, 4)`, step cap `h ≤ max(ω, 1)` plus the example's cap over
  its resonance band; sub-resolution transitions decided by the side of the root (flag 2);
* `Z = n/2 − Φ/π` with the leading order `n` (estimated, set, or the FEM bar's effective order);
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

## Validation and timings (Intel UHD Graphics 630 laptop GPU, Chrome, 2026-10-07)

`julia --project=gpu/scripts -t auto webgpu/validate/ref_counts.jl --cpu` (NyquistGPU,
CPU backend, Float32 and Float64), 160 × 90 grid per example, each with its own ω_max and
step caps; default constants and a second constant set ("@alt", sliders moved). The page
runs every grid through the text path (FEM: built-in), `?validate=1`: **PASS**.

* **Counts: 0 differing `Z` on all 24 grids against the Julia Float32 run** (also at flagged
  points; 0 between the default and the exact pipeline). Against Julia Float64 the only
  differences are flagged points with a root on the line (the K = 1 row of the bar charts,
  λ = 0, where `Z_raw` is a half-integer: rod 15 / 153, fem 72 / 82, neutral 1), where Julia's
  own F32 and F64 runs differ the same way.
* The march settings evaluated from the texts' expressions equal the reference's (ω_max,
  h_max, band) on every grid; the estimated orders differ from the reference's by 0 (integer
  orders), 2e-8 (fractional oscillator), 9e-6 (Gao: `μ + ν` approached as `1/(T s^ν)` decays)
  and 6e-10 / 8e-6 (rod: 0 vs the host-measured n_eff of the scaled denominator).
* Default-mode σ against the Julia F32 run: median |Δσ| 5e-9 – 2e-6, 99th percentile ≤ 2e-5.
* Exact mode against Float64 dominant roots at 16 stable points per grid: within 1e-3 at 373
  of 378 points, the same exceptions as before (3 neutral_hg points where the root chain
  approaches the asymptote Re λ = ln|a|/τ only as ω → ∞; one gao@alt point where the exact mode
  found a root right of the reference's, confirmed by a brute-force search; one frac@alt point
  without any minimum).

**Generated vs hand-written WGSL** (`?bench=compare`, default resolution, median of 5
interleaved runs): parity. Identical counts everywhere and bit-identical results (same step
counts at every point) for all examples but Gao, whose text follows the Julia reference's
operation order (`K*exp(-Lλ)*(...)`) rather than the old WGSL's (`K*(exp(-Lλ)*(...))`).

| example | resolution | GPU ms generated | hand-written | ratio | exact: generated | hand-written | ratio |
|---|---|---|---|---|---|---|---|
| fourth | 960 × 540 | 26.1 | 26.7 | 0.98 | 52.0 | 56.1 | 0.93 |
| algebraic | 960 × 540 | 15.9 | 16.1 | 0.99 | 31.7 | 35.1 | 0.90 |
| distributed | 960 × 540 | 28.4 | 27.7 | 1.02 | 48.5 | 52.4 | 0.93 |
| showcase | 960 × 540 | 52.9 | 51.8 | 1.02 | 88.7 | 91.1 | 0.97 |
| turning | 960 × 540 | 154 | 145 | 1.06 | 266 | 257 | 1.03 |
| neutral | 480 × 270 | 117 | 116 | 1.01 | 170 | 169 | 1.01 |
| pda | 480 × 270 | 283 | 281 | 1.01 | 403 | 400 | 1.01 |
| rod | 480 × 270 | 206 | 208 | 0.99 | 284 | 287 | 0.99 |
| frac | 960 × 540 | 34.5 | 35.5 | 0.97 | 60.2 | 60.8 | 0.99 |
| gao | 480 × 270 | 178 | 178 | 1.00 | 283 | 287 | 0.99 |

The things that make this parity: integer powers as repeated squaring, the shared nodes (the
`log λ` of the fractional powers, helpers computed once), real factors as `dscale` / `daddr`
instead of full dual products, constants folded on the host, and the uniform constants copied
into a private array with constant indices (a loop with a dynamic index would push it to
scratch memory).

Default resolution, default parameters (`?bench=1`, median of 3):

| example | resolution | GPU ms default | evals / pt | GPU ms exact | evals / pt | exact / default |
|---|---|---|---|---|---|---|
| fourth | 960 × 540 | 26 | 32 | 52 | 38 | 2.0 |
| algebraic (A.2) | 960 × 540 | 15 | 22 | 34 | 26 | 2.2 |
| distributed (A.3) | 960 × 540 | 27 | 24 | 50 | 27 | 1.8 |
| showcase | 960 × 540 | 51 | 53 | 99 | 60 | 1.9 |
| turning (A.7) | 960 × 540 | 143 | 139 | 254 | 162 | 1.8 |
| neutral (A.4) | 480 × 270 | 117 | 293 | 170 | 323 | 1.5 |
| neutral_hg (A.5) | 480 × 270 | 105 | 288 | 158 | 318 | 1.5 |
| pda (A.6) | 480 × 270 | 276 | 704 | 404 | 735 | 1.5 |
| rod (A.8) | 480 × 270 | 203 | 312 | 287 | 341 | 1.4 |
| fem (A.9, N = 12) | 320 × 180 | 169 | 313 | 228 | 345 | 1.3 |
| frac (A.11) | 960 × 540 | 35 | 27 | 58 | 39 | 1.7 |
| gao (A.12) | 480 × 270 | 173 | 269 | 291 | 310 | 1.7 |

(The evaluations per point are those of the first version; the times are within the
run-to-run spread of this laptop GPU -- the unchanged FEM shader moved by the same 5–10 %.)

## Limitations

* Float32 only; no certification of σ by counting on shifted lines (not admissible for the
  fractional and bar examples anyway), no Float16 mode, no flagged-point re-check.
* The equation must be a closed expression (no loops, determinants, piecewise definitions);
  anything else needs a hand-written `charD` like the FEM bar. At most 16 parameters.
* The order estimate assumes `|D(s)| ~ s^n` on the positive real axis; exponential growth
  (advanced terms) or an order that changes over the chart is only warned about -- set n
  then. The real-coefficient check is a warning, too (the count over ω ≥ 0 assumes it).
* Float32: unscaled forms such as `cosh γ` of the rod overflow (grey "failed" points) and need
  the scaling shown in A.8; a removable singularity needs `exprel` (or another entire form).
* The grid is evaluated in full (no adaptive coarse-to-fine `run_adaptive!`).
* `σ` of the march line is fixed at 0; the sin/cos argument reduction is exact up to
  |τω| ≈ 1e5.
