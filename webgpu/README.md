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
| `stats.js` | optional anonymous usage counts (GoatCounter); off unless configured |
| `handwritten.js` | the hand-written WGSL of the first version, only for `?bench=compare` |
| `march.wgsl` | the per-point `:unwrap` march (port of `gpu/src/NyquistGPU.jl`) and the exact-root polish; 2D grids and 3D grids (z slices stacked as rows) |
| `view3d.js`, `volume.wgsl` | experimental 3D view: the combined field as a 3D texture, ray marching, orbit camera; STL export of the boundary (marching tetrahedra) |
| `display.wgsl` | colouring (port of `k_display!` / `pixel_colour` of `server.jl`) |
| `validate/web_systems.jl` | the gallery examples in NyquistGPU form `D(λ, p, c)` (same operations as the WGSL) |
| `validate/ref_counts.jl` | Julia script: reference counts and dominant roots from the repository engine |
| `validate/independent.jl` | independent check of the shimmy and CTCR counts: the ODE back-end (`:bs3`, Float64, tol 1e-8) on the paper's own formulas |
| `validate/ref_counts.json` | its output (160 × 90 grids, default and a second constant set per example), read by `?validate=1` |

## Your own equation

Pick **Own example** (it starts as A.1 and is kept in this browser's `localStorage`; the page
also works where storage is blocked) or edit any example's text. The box takes

```
# comment
K = -0.75:1                               # parameter range (chart range of an axis)
r = 0.02:0.05:10.5                        # with a step: 210 grid points along this axis
η = 0.002:0.05 @ 0.01                     # range and initial value (default: the midpoint)
c = 0                                     # a fixed constant (a helper): no slider
γ = λ*sqrt(1 + c/λ)/sqrt(1 + η*λ)        # helper: usable in later lines, computed once
(1 + exp(-2*γ))/2 - K*exp(-r*λ - γ)       # the last line (or a line  D = ...) is D(λ)
```

* numbers (`1.5e-3`), `+ - * / ^` (`**`, `·`, `−` and superscripts `λ²`, `λ⁻¹` accepted), unary
  minus, parentheses; `exp log (ln) sqrt sin cos tan sinh cosh tanh exprel pow(a, b)`,
  `exprel(x) = (eˣ − 1)/x` (entire; evaluated by its Taylor series near 0 -- for distributed
  delays, `(1 − e^{−τλ})/λ = τ·exprel(−τλ)`); constants `pi` (`π`), `i`; the variable `λ`
  (or `lambda`; the **λ** button inserts it);
* **Distributed delays: `integral(f, θ, a, b)`** -- θ is the integration variable (any name that
  is not λ, a constant, a function or a helper), `a`, `b` real expressions of the parameters;
  `f` may contain θ, λ and parameters. When `f` is a sum of `p(θ)·exp(μθ)` terms (p a polynomial
  in θ of degree ≤ 3 whose coefficients may contain λ and parameters, μ free of θ, e.g.
  `exp(-λ*θ)` or `exp(-(λ + d)*θ)`) the integral is generated in **closed form**, stable at
  λ = 0 in Float32: with θ = a + hu, h = b − a,
  `∫ θ^k e^{μθ} dθ = h e^{μa} Σ_m C(k,m) b^{k−m} (−h)^m m! φ_{m+1}(μh)`,
  `φ_j(z) = Σ_i z^i/(i+j)!` (φ₁ = exprel; Taylor series for |z| < 2 + j, else the recurrence
  φ_j = (φ_{j−1} − 1/(j−1)!)/z) -- the removable singularity of the closed form never appears.
  Any other kernel falls back to composite 8-point **Gauss–Legendre** with ⌈|λ|·|b − a|/3⌉
  panels (at most 64), with a warning in the panel: slower, and approximate where
  |λ|·|b − a| > 190, so keep ω_max small there. Nested integrals are not supported.
  Examples: A.3 `b*integral(exp(-λ*θ), θ, τ₀, τ₀ + τ)`; the shimmy's contact-patch memory
  `integral((2*(L - 1) + 4*θ)*exp(-λ*θ), θ, 0, 1)`.
* Greek / Unicode names with subscripts (`τ₀`, `ζ₁`, `ω₂`, `k_p`); implicit multiplication is
  an error (`2λ` → "write 2*λ"); non-analytic functions (`abs`, `real`, `imag`, `conj`, `min`,
  `max`, ...) are rejected; a line that ends or starts with an operator, or an open
  parenthesis, continues on the next line; parse errors are shown with a caret under the line.
* Every identifier that is not λ, a function, a constant or a helper is a **parameter** (the
  declared ones in declaration order, then by first appearance; at most 16). Each has a row:
  value, min, slider, max, and **X / Y / Z** buttons. The two axis parameters span the chart over
  their [min, max]; the others are live sliders.
* **Parameter ranges in the text** (Julia-like): `P = 2.1:20` declares the range [2.1, 20],
  `P = 2.1:0.1:10` start:step:stop (on an axis the step sets the grid of the next frame along it,
  here 80 points; the automatic resolution may refine it later), `P = 2.1:20 @ 5` also the
  initial value (default: the midpoint). Bounds are constants (`0:2*pi`). These lines declare
  parameters, they are not helpers; a plain `name = number` stays a fixed constant without a
  slider. Undeclared parameters get [0, 2] at 1 (or their old value when a line `name = number`
  was deleted). **Text and table stay in sync**: editing min / max in the table (or drag-zooming
  the chart) rewrites the declaration line, or adds one; reset axes restores the example's
  lines. Every example declares its paper ranges and defaults this way, so the text alone
  defines it.
* Typing recompiles 400 ms after the last key (Ctrl+Enter at once). The pipeline is rebuilt
  only when the text or the axis choice changes (cached by the generated code); sliders only
  rewrite the uniform buffer.
* **Order n** (`Z = n/2 − Φ/π`): always computed automatically on the host (shown under the
  sliders; an override exists only in Advanced), as the least-squares slope of `ln|D(s)|`
  over 9 log-spaced real `s ∈ [s*/10, s*]`, `s* = 1e8` (reduced while `|D|` is not finite),
  at an interior point of the axis ranges; one decade lower as a check (exponential growth,
  e.g. an advanced term, gives a warning), and at the four chart corners (a warning if n
  changes over the chart). Re-estimated on every slider move (a few hundred host
  evaluations). Fractional powers give a non-integer n, as they should. A real-coefficient
  check `D(λ̄) = conj D(λ)` warns about complex coefficients.
* **ω_max** (a log-scale slider, 10 … 1e6; each example sets its default) and the **tolerance**
  (a slider on its exponent: 10^x rad, x ∈ [−2, 0], default 0.3): the march accepts a step when
  the observed phase change agrees with the trapezoid prediction within this tolerance, so a
  smaller value means shorter steps, more robust counts and more time.
* **Advanced**: n (override), ω₀ (1e-9), step cap
  h_max for ω below a band (empty band: the whole line), branch point at λ = 0 (auto: `λ^x`
  with non-integer x, `log λ`, `sqrt λ` -- the |D| minimum at ω₀ is then no root estimate).
  These are expressions of the parameters (`pi/(2*τ)`; an axis parameter stands for
  max |value| over its range) with `pi`, `inf`, `min`, `max`, `abs`. Neutral systems need a
  small explicit ω_max (200–500) and `h_max = pi/(2*τ)` (hint in the panel).
* **Copy link** puts the equation, parameter values and ranges, axes, settings, resolution and
  the mode into the URL hash (`#m=` base64url of a small JSON) and restores it on load;
  `?ex=<key>` links still work (`&exact=0`: fast mode; `&exact=1`: exact, the default).

## Resolution, watchdog, modes

* **Auto (20 fps)** (default): when the equation, the mode or the dimension changes, probe grids
  10², 20², 40², ... (3D: 10³, 20³, ...) are timed until one takes > 12 ms; the grid is then sized
  so that a frame takes about 50 ms (clamped to 20 × 20 … 3840 × 2160, in 3D 16³ … 128³, aspect
  16:9 or that of a declared step grid), and the estimate follows the measured frame times
  (log-average, 30 % hysteresis). The fixed sizes stay selectable; in 3D they map to n³ with the
  same number of points (39³ … 160³).
* **Time limit**: a field under the resolution selector, default 5 s, 1–600 s, for the fixed
  resolutions (e.g. 60 s for a 4K image or a 202³ grid; progress is shown under the chart
  during long runs); Auto keeps 5 s. It is part of the share link.
* **Watchdog**: a chart is submitted in row bands (the first ~4k points, then ~60 ms each);
  no band is submitted after the time limit, the partial chart stays and the page says "Stopped after 5 s
  … reduce ω_max or the resolution" (the automatic resolution is lowered for the next frame).
  A submitted band cannot be cancelled, so the per-thread step cap is 50 000 (the examples need
  ≤ ~750 evaluations per point); a march that hits it is a failed (grey) point. Measured: a
  neutral equation at ω_max = 1e6, tol = 0.01, 1920 × 1080 is stopped 5.8 s after the request.
* **Mode**: the default is the **exact root (refined)** mode (below); "fast (first-order σ)" is
  the first-order estimate. Auto-resolution times the mode in use.

## 2D navigation

Mouse: the **wheel** zooms around the cursor, the **middle button** drags (pans) the ranges, a
**left-drag** zooms to a box, a **double-click** resets to the declared ranges. Touch: pinch
zooms, a two-finger drag pans (one finger only reads out the point; page scrolling outside the
chart is not touched). The last image is shown transformed at once, the chart is recomputed
continuously at the automatic resolution, the min / max fields follow immediately and the
range lines of the text are rewritten when the interaction pauses (600 ms; rounded to 1e-4 of
the span, a declared step is rescaled to keep the grid size). The share link carries the view.

On a desktop the controls scroll in their own column and the chart stays in view; the divider
between them can be dragged (240 px to 70 % of the window, kept in localStorage, double-click
resets). The example description and the syntax help are collapsed under the selector and the
equation box, so the parameters follow the equation directly. On a phone the layout stays
stacked, without the divider. The **? Help** button in the header (also `#help` or `?help=1`)
opens a short usage guide.

## 3D view (experimental)

Mark a third parameter with **Z**: the chart becomes a brute-force grid over the three axis
ranges (the third parameter is a per-slice uniform, the slices are stacked as rows of the same
march dispatch) and is shown by ray marching (WebGPU render pass, `volume.wgsl`):

* the field `C = max(σ, σ_floor)/|σ_floor|` (≤ −0.02) at stable points and `+0.6` elsewhere is
  uploaded as an `r8unorm` 3D texture (trilinear filtering);
* the unstable region is not drawn; the stability boundary is the zero level of `C`, a
  Lambert-shaded sheet (normal from the gradient) with its own opacity slider ("3D boundary α":
  0 hidden, 1 opaque); the stable interior has the "3D interior α" slider: 0 transparent (only
  the boundary), 1 solid (every stable sample opaque, nothing behind shows through), in between
  a fog with per-sample alpha `1 − (1 − α)^{8(−C)Δt}` in the σ colour map, so the most stable
  region is the densest;
* **Smooth boundary** (checkbox, default on): after each 3D chart the boundary mesh of the STL
  export (below) is built on the CPU in ~12 ms time slices and drawn with
  WebGPU: vertex normals are the area-weighted averages of the face normals (vertices shared
  through their grid edges), two-sided Lambert shading, premultiplied alpha with the boundary-α
  slider, in the order far side of the mesh, ray-marched interior fog, near side; the caps
  that close it at the box faces are left out on screen. Rotation only redraws (no rebuild).
  While a new mesh is being built the last completed one stays on screen with the new fog (the
  rough ray-marched surface is never shown while the option is on); a build is never cancelled,
  charts that arrive meanwhile only replace the pending request, so during a slider drag every
  finished mesh is shown and the next build starts from the newest chart (tested: 40 slider
  steps, 30 charts, 30 mesh swaps, no frame without a mesh).
  Build time (shimmy, this laptop): 52³ 56 ms (59 204 triangles), 64³ 60 ms (92 002), 128³
  0.5–0.6 s (387 042). Off: the ray-marched surface (slow machines, very large grids).
* **Save STL**: the boundary as a smooth, closed triangulated surface: the zero level of the
  Float32 σ field (σ of the rightmost root, read back at full precision: negative at stable,
  positive at unstable points, continuous through 0 at the boundary), marching tetrahedra (6
  per cell, linear interpolation), padded so the surface is closed at the box faces
  (printable), triangles oriented outward (decided on the exact edge midpoints, robust also for
  near-degenerate triangles); binary STL in axis units or, with "unit box", in [0, 1]³. Checked
  on the shimmy at 52³, 64³ and 128³: every directed edge matched by exactly one reverse edge
  (watertight, consistently oriented); the STL reuses the on-screen mesh;
* the examples with a natural third parameter have a **3D view** button (shimmy: Z = ζ, with a
  thin interior and a stronger boundary; CTCR: Z = the cross-talk gain);
* drag rotates (orbit), the wheel zooms, double-click resets the view; the box edges carry the
  axis names and ranges. Auto-resolution gives 40³–70³ at 20 fps on the test laptop.
* Click **Z** again to go back to 2D. Not for the built-in FEM bar (fixed axes).
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
| A.3 distributed delay | `λ^2 + a*λ + k + b*integral(exp(-λ*θ), θ, τ₀, τ₀ + τ)` | a, b | ω_max 1e4 |
| showcase 2-DOF DAE | helpers `a₁₁`, `a₁₂`, `a₂₂`; `a₁₁*a₂₂ - a₁₂^2` | P, D | ω_max 1e5 |
| A.7 multi-mode turning | helpers `M₁`, `M₂`; `M₁*M₂ + w*(1 - exp(-2*pi/Ω*λ))*(M₂ + A₂*M₁)` | Ω, w | h_max 0.05 for ω < 5 |
| A.4 / A.5 neutral | `λ^2 + a*λ^2*exp(-τ*λ) + d*λ + k + c*exp(-τ*λ)` | a, c | ω_max 200, h_max `pi/(2*τ)` (whole line) |
| A.6 PDA control | `λ^2 + 2*ζ*λ + 1 + (P + D*λ + A*λ^2)*exp(-τ*λ)` | P, A | ω_max 500, h_max `pi/(2*τ)`; dashed \|A\| = 1 |
| A.8 exact rod | `γ = λ*sqrt(1 + c/λ)/sqrt(1 + η*λ)`; `(1 + exp(-2*γ))/2 - K*exp(-r*λ - γ)` | r, K | ω_max 400, h_max `pi/(2*max(r, 1))` for ω < 40 |
| A.9 FEM bar (built-in) | `det(λ²M + λC + K + F(λ))`, N elements (2–24) | r, K | as A.8; n_eff from the host |
| A.11 fractional oscillator | `λ^α + c*λ^β + k*exp(-τ*λ)` | k, τ | ω_max 1e4; branch point (auto) |
| shimmy (Takács, Orosz & Stépán 2009) | helpers `A = integral((2*(L - 1) + 4*θ)*exp(-λ*θ), θ, 0, 1)`, `P`, `Q`; `P - Q/(L^2 + 1/3 + Σ*(L^2 + 1 + Σ))` | V, L (3D: ζ) | ω_max 1e4 |
| two delays, CTCR (Sipahi & Olgac 2004) | `λ^2 + 7.1*λ + a₀ + (6*λ + b₁)*exp(-τ₁*λ) + (2*λ + b₂)*exp(-τ₂*λ) + c₁₂*exp(-(τ₁ + τ₂)*λ)` | τ₁, τ₂ (3D: c₁₂) | ω_max 1e4, h_max `pi/(2*(τ₁ + τ₂))` for ω < 50 |
| A.12 fractional PI (Gao, Zhai & Liu) | `λ^μ*(T*λ^ν + 1) + K*exp(-L*λ)*(k_p*λ^μ + k_i)` | k_p, k_i | ω_max 1e4; branch point (auto) |

All orders are estimated (none is stored): 4, 2, 2, 4, 4, 2, 2, 2, ≈ 0 (rod), α, μ + ν, 3 (shimmy), 2 (CTCR).

Notes on the harder ones:

* **Shimmy** (Takács, Orosz & Stépán, Eur. J. Mech. A/Solids 2009, Eq. (31); dimensionless
  towing speed V, caster length L, relaxation length Σ = 1.8, tyre damping ζ): the term
  `2/λ²[(L − 1)λ + 2 − ((L + 1)λ + 2)e^{−λ}]` is the memory of the contact patch,
  `∫₀¹ (2(L − 1) + 4θ) e^{−λθ} dθ` (checked in Float64 against the printed form at 24 points: relative
  difference ≤ 3e-15, except at λ = 1e-6 i, where the printed form itself loses 5 digits). Term B of
  Eq. (31) carries `1/(L − 1 − Σ)` and is multiplied by `(L − 1 − Σ)` outside the braces: the text
  cancels the two, so nothing divides by zero at L = 1 + Σ (D is unchanged elsewhere). Retarded
  (delay 1), n = 3 (estimated); ω_max = 1e4 (the λ³ term dominates from ω ~ 1/V on); no step cap
  is needed. The 3D view over (V, L, ζ) is the Hopf surface of the MDBM figure.
* **Two delays, CTCR** (Sipahi & Olgac, ACC 2004, Eq. (18), with the cross-talk term
  `8 e^{−(τ₁+τ₂)s}`): the dendrites of the stable region in (τ₁, τ₂) ∈ [0, 3]². The two delays add up to
  6, so the phase turns fast at low ω: without a cap the default step control (tol 0.3) jumped
  over a root pair at 12 / 10 of 14 400 points (against the ODE back-end, see below); with
  h ≤ π/(2(τ₁ + τ₂)_max) for ω < 50 there are none. Sliders: c₁₂ (default 8), a₀ = 21.1425,
  b₁ = 14.8, b₂ = 7.3.

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

**"Exact root (refined)"** (the default mode) implements the paper's dominant-root recipe in the shader:

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
  `gao`), `&exact=0` for the fast mode; **Copy link** adds the full state (`#m=...`).
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

## Usage statistics (optional, GoatCounter)

Anonymous counts with [GoatCounter](https://www.goatcounter.com) (no cookies, no personal data),
**off by default**. To switch them on:

1. Create a free GoatCounter account and a site code, e.g. `nyquistgpu`.
2. In `stats.js` set `GOATCOUNTER = 'https://nyquistgpu.goatcounter.com/count'`.
3. Optional, in the GoatCounter site settings: "Allow public counter" (then
   `SHOW_PUBLIC_COUNTER = true` shows an "N visits" line in the footer; hidden on any error) and
   "Ignore IPs" (your own addresses).

With `GOATCOUNTER` empty nothing is loaded or sent. When set, the page loads
`https://gc.zgo.at/count.js` asynchronously and sends only: one page view (the path, never the
query or the hash, which holds a shared equation); `example/<key>` when a built-in example is
opened (once per selection, not per slider move); `custom-equation` once per distinct own or
edited equation of a session (after the text rested 5 s; decided by a local hash, the text is
never sent); `shared-link-opened`; once per session: `3d-view`, `3d-view/<key>` (per example),
`stl-export`, `copy-link`, `help-open`, `exact-off`, `zoom-pan`, `time-limit-raised` (> 5 s),
`fixed-resolution`, `watchdog-stop`, `parse-error`, `3d-smooth-off`, `validate-run`, `bench-run`;
`gpu/<intel|nvidia|amd|apple|qualcomm|arm|other>` and `no-webgpu`. No parameter values, no
equation text, nothing typed. The Help overlay explains it, and a small-print line at the
bottom of the page says that anonymous visit statistics are collected.

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
runs every grid through the text path (FEM: built-in), `?validate=1`: **PASS** (also after the
range-declaration texts, the 3D-capable march and the 50 000 step cap).

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

**Independent check of the two newer examples** (`validate/independent.jl`, 160 × 90 grids,
default and second parameter sets): the page's settings (`:unwrap`, Float32, the page's form of
D) against the ODE back-end (`:bs3`, Float64, tol 1e-8) on the paper's own formula (the shimmy:
Eq. (31) as printed, 2/λ² form and 1/(L − 1 − Σ), from ω₀ = 1e-3; CTCR: ω_max = 1e3): **0 differing
counts** on all four grids (shimmy 77.4 % / 84.5 % stable, CTCR 59.0 % / 44.4 %). In the page
validation both examples also give 0 differing counts against the Julia references; exact mode
32 of 32 shimmy roots, 31 of 32 CTCR roots (one deep stable point, σ = −1.94, without any tracked
minimum, as frac@alt).

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
| shimmy | 960 × 540 | 78 | 33 | 130 | 39 | 1.7 |
| CTCR | 960 × 540 | 603 | 342 | 993 | 378 | 1.6 |

A.3 through `integral(...)`: 28.6 ms (default) / 50.3 ms (exact) at 960 × 540, the same as the
`exprel` text (identical counts in the validation). 3D grids (GPU time, exact / fast): shimmy 64³
123 / 60 ms, 128³ 479 / 307 ms; CTCR 64³ 543 / 359 ms, 128³ 4.2 / 2.7 s. Auto (20 fps) picks
about 40³–52³ for the shimmy and 25³ for the CTCR.

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
* The 3D view is experimental: brute force (no adaptive refinement), at most 128³ in auto mode,
  an 8-bit field (σ resolved to ~1/128 of the colour range), no hover read-out.
* Auto-resolution is a heuristic (latency + throughput model of the measured frames); frames of
  very slow equations (one march ≳ 50 ms) take about twice their single-march latency.
* Float32: unscaled forms such as `cosh γ` of the rod overflow (grey "failed" points) and need
  the scaling shown in A.8; a removable singularity needs `exprel` (or another entire form).
* The grid is evaluated in full (no adaptive coarse-to-fine `run_adaptive!`).
* `σ` of the march line is fixed at 0; the sin/cos argument reduction is exact up to
  |τω| ≈ 1e5.
