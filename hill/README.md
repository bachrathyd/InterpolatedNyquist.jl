# Hill determinant + argument principle for time-periodic delayed systems

Branch `hill-argument-principle` (local, not pushed). Stability of time-periodic delayed
systems without time integration: a regularized Hill determinant evaluated on the
imaginary axis, its phase unwrapped by the discrete march of the package, and the
argument principle on one period strip of the Floquet exponents.

## Method (delayed Mathieu, `hill_core.jl`)
x'' + κx' + (δ + ε cos ω_p t) x = b x(t − τ).  Harmonic k of the Floquet ansatz:
d_k(λ) = q(λ + ikω_p), q(s) = s² + κs + δ − b e^{−sτ}, coupling ε/2 → tridiagonal Hill matrix,
determinant by a continuant recursion (O(N), dual-number friendly, GPU friendly).

* **Contour:** half-strip {Re λ > 0, a < Im λ/ω_p < a + 1}, a = 0.237. The regularized
  determinant → 1 for Re λ → ∞ and is periodic with period iω_p, so the top/bottom edges
  cancel: **Z = −(1/2π) Δarg over ω ∈ [a, a+1]ω_p**. a is generic so that the exponents of
  real multipliers (Im = 0: μ = +1, Im = ½: flip) never lie on the strip edges.
* **Regularization must be pole-free.** Hill's normalization (rows / d_k) creates a pole at
  every LTI root of q; a q-root just left of the axis next to an unstable Floquet exponent
  just right of it is a zero–pole pair straddling the contour, a −2π slip inside one step
  that no end-point check sees (97 wrong points of 28 800, e.g. at the flip exponent of
  (δ, b) = (1.75, −1.5)). Scaling the rows by r_k = (λ + ikω_p + c)² instead keeps Δ̃ → 1 and
  the periodicity and moves all poles to λ = −c − ikω_p: **no pole count is needed** and
  the error disappears (1 of 28 800 differs, no stability difference).
* **Diagonal tail in closed form** (τω_p = 2πj, which also holds for equal-pitch milling,
  τ = tooth period): Π_k d_k/r_k = sinh(π(λ−z₁)/ω_p) sinh(π(λ−z₂)/ω_p)/sinh²(π(λ+c)/ω_p);
  only the coupling is truncated, the determinant then converges like ~N⁻³ and Z_raw is an
  integer to 1e−6.
* **Truncation from the single tolerance:** a priori N = ⌈√(δ⁺ + ε/(2√tol))/ω_p⌉ + 1 (ring
  change (ε/2)²/|d_k d_{k−1}| ≤ tol), a posteriori check of the ring change at probe
  frequencies, N increased until it is below tol.

## Test 1 results (CPU: `test1_mathieu.jl`; GPU: `gpu_hill.jl`)
κ = 0.1, ε = 1, τ = T = 2π, δ ∈ [−1, 5], b ∈ [−1.5, 1.5], tol = 1e−4.
Reference: Floquet multipliers of the RK4-discretized monodromy map (80 and 300 steps agree).

| | |
|---|---|
| counts vs reference (240×120) | 1 of 28 800 differs, stability identical (Hill-normalized variant: 97 wrong) |
| CPU time, 12 threads | 0.42 s (reference 63.5 s, ~150×); 24 phase evaluations per point (median) |
| GPU (RTX PRO 6000, Colab), full HD | Float32 97.7 ms, Float64 556 ms; GPU = CPU counts at 160×80 in both precisions |
| N chosen | 9 everywhere for tol 1e−4 (map: `results/test1_N.png`) |
| boundary types (crossing multiplier angle) | Neimark–Sacker 360 px, μ = +1 83 px, flip 19 px |

Convergence (`results/test1_convergence.csv`, 325 points): wrong counts 37/4/2/0/0… for
N = 1/2/3/4/5…; the one-ring a-posteriori estimate is ~N/2 below the true determinant error
(tail of an ~N⁻³ series), e.g. N = 9: estimate 7e−5, true 2.5e−4.

Images: `results/test1_chart.png` (colour = count, black = reference boundary, blue = mismatch),
`test1_N.png`, `test1_errest.png`, `test1_boundary_type.png` (green μ=+1, red flip, blue NS).

## Tests 2 and 3: milling (`milling_core.jl`, `test2_milling.jl`, `test3_helix.jl`)
1-DOF milling, time scaled by ω_n, w = a_p K_t/(mω_n²):
x'' + 2ζx' + x = −(w/L) Σ_j ∫₀^L g(φ_j) f(φ_j) [x(t) − x(t − τ_j(ζ))] dζ,
f(φ) = sin φ (cos φ + k_r sin φ), g = 1 on the cutting window (jump at tooth entry).
Hill matrix A_kl = δ_kl p(s_k) + w C_{k−l} K_{k−l}(s_l): C_m are the Fourier coefficients of the
cutting function (1/m decay because of the jump), K_m the geometry/delay kernel. Dense, so the
determinant is an LU of the row-scaled (pole-free) (2N+1)² block times the closed-form diagonal tail.

* **Test 2** (Insperger–Stépán benchmark: z = 2, a/D = 0.05 down milling, K_n/K_t = 1/3, ζ = 0.011,
  f_n = 922 Hz; 5000–25000 rpm, a_p ≤ 5 mm). Straight teeth, uniform pitch → tooth period, one point
  delay τ = T, delay factor (1 − e^{−λT}) common to all harmonics → closed-form tail exact.
  Reference: RK4 monodromy on a grid aligned with the cutting window.
  - 200×100 chart: **16 of 20 000 counts differ** from the reference (15 in stability), all on boundary
    pixels; 18.7 s (12 threads) vs 127 s for the reference. All 16 are barely stable points
    (ρ = 0.9994–0.99996 by a finer reference); with N + 10 harmonics 15 of them become right, so they
    are truncation errors that the one-ring estimate (2–8× optimistic) let through; the 16th sits at
    the N = 40 cap (5900 rpm), where the coarse reference was also wrong (`test2_log.txt`)
  - flip (period-doubling, 419 boundary px) and Neimark–Sacker (562 px) lobes both reproduced
  - N chosen 3…40 (median 6): grows with the lobe number, i.e. ∝ 1/(spindle speed) — `test2_N.png`
  - convergence (`test2_convergence.csv`): determinant error ~N⁻³ once N is past the resonant
    harmonics; at 6000 rpm, a_p = 3 mm the count is wrong for N = 4, 6 and right from N = 8, at
    15000–22000 rpm from N = 2; the one-ring estimate is 2–8× below the true error
* **Test 3** (helix 30°/45°, R = 8 mm, same cutting data; 8000–30000 rpm, a_p ≤ 10 mm). The delay of
  each tooth varies linearly over the axial depth, τ_j(ζ) = (p_j + ζ(tanβ_j − tanβ_{j−1})/R)/Ω, the
  angular lag ζ tanβ_j/R shifts the cutting window → spindle period, distributed delays, k-dependent
  delay factors. Kernel: (a) **exact** (closed form: exponential integrals over ζ), (b) **sampled at
  n_s axial points** (finite-point kernel, the general route).
  Reference: RK4 monodromy over the spindle period, 32 axial slices, Hermite-interpolated history.
  - 120×60 chart (exact kernel): 22 s on 8 threads; N 3…22 (median 10)
  - **0 of 450** reference points differ (sub-grid; 93 s for the reference)
  - sampled kernel vs exact: 674 / 252 / 44 / 10 / 3 differing points for n_s = 1 / 2 / 4 / 8 / 16
    (`test3_kernel_approx.csv`) — the finite-point kernel converges to the exact one
  - boundary types: μ = +1 290 px, flip 10, NS 270 (the helix tool behaves very differently from Test 2)

## GPU (NyquistGPU kernel unchanged, `gpu_models.jl`)
D as a function of μ = λ/ω_p(point) so every point marches the same strip μ ∈ [a, a+1]; the kernel's
Z_raw = −Φ/π = 2Z. Milling: dense LU per thread in an `MMatrix` (N ≤ 24 / 26); a priori N calibrated
on the CPU study, N = ⌈2√(1 + wH_max/√tol)/ω_p⌉ + 3. Complex division and log are written without
Base's scaled algorithm (it throws on an exact zero for dual numbers — a kernel exception) but scaled
by the primal magnitude (Float32 underflow otherwise). GPU forms agree with the CPU cores on all test
points (Mathieu 450, Test 2 288, Test 3 128) in Float64 and Float32.

## Interactive
`gpu/colab/NyquistGPU_Hill_Interactive.ipynb` (this branch): the three time-independent examples plus
*delayed Mathieu*, *milling, straight flutes (Test 2)*, *milling, different helix angles (Test 3)* in
the example dropdown, model constants as sliders (κ, ε; ζ, a/D, K_n/K_t; ζ, a/D, β₂).

## Not done yet
* Banded/shifted truncation and Schur-complement ring growth (not needed for Mathieu: N = 9
  everywhere; becomes relevant for milling with many harmonics).
* Schur-complement ring growth (N is re-factorized per ring in the a-posteriori check) and the banded
  (shifted) window: at low spindle speed N reaches 20–40, a band around the resonant harmonics would
  cut that.
* A tighter a-posteriori estimate: the one-ring change underestimates the tail (×N/2 for Mathieu,
  ×2–8 for milling); a tail-sum (Richardson) correction would make it a bound.
