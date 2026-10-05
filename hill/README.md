# Hill determinant + argument principle for time-periodic delayed systems

Branch `hill-argument-principle`. Stability of time-periodic delayed
systems without time integration: a regularized Hill determinant evaluated on the
imaginary axis, its phase unwrapped by the discrete march of the package, and the
argument principle on one period strip of the Floquet exponents. For straight-fluted
milling the infinite Hill determinant is also available in a **compressed form** (all
harmonics in closed form, a 16-node determinant counted along the unit circle of the
Floquet multiplier): full HD charts in 18 ms (Float16) / 70 ms (Float32) on one GPU, see
[Fast milling](#fast-milling-the-compressed-hill-determinant).

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

## Fast milling: the compressed Hill determinant
`fredholm_core.jl` (CPU prototype), `gpu_fast.jl` (GPU forms), `gpu_fast_bench.jl`, `fast_kernel_info.jl`.
Test 2 (straight teeth, uniform pitch, one delay τ = T = 2π/ω, ω = zΩ), still purely in the frequency
domain, but without truncating the harmonics:

* **Low-rank coupling.** A(λ) = D(λ) + α(λ)H, D = diag p(λ + ikω), p(s) = s² + 2ζs + 1,
  α = w(1 − e^{−λT}), H_kl = h_{k−l} the Toeplitz matrix of the cutting function h(ψ). h vanishes outside
  the cutting window, and the jump at tooth entry is what makes h_m decay like 1/m (many harmonics).
  Quadrature of the Fourier integral over the window factors H ≈ UVᴴ, U_kq = c_q e^{−ikψ_q},
  Vᴴ_ql = e^{ilψ_q}, with c_q = W_q h(ψ_q)/2π, where ψ_q are Q nodes in the window. The jump sits at the
  window edge, so the quadrature sees a smooth integrand.
* **Matrix determinant lemma:** det(D + αUVᴴ) = det D · det(I_Q + αM), M = VᴴD⁻¹U,
  M_pq = c_q S(ψ_p − ψ_q). Every harmonic is summed exactly:
  S(Δ) = Σ_{k∈ℤ} e^{ikΔ}/p(λ + ikω), and by partial fractions of 1/p,
  Σ_k e^{ikΔ}/(a + ikω) = T e^{−aΔ/ω}/(1 − e^{−aT}) for Δ ∈ [0, 2π).
  The only approximation left is the Q-node quadrature.
* **Counting on the unit circle of the Floquet multiplier.** det(I_Q + αM) depends on λ only through
  z = e^{λT}. As a function of z it is analytic outside |z| = 1: its poles e^{r_iT} come from the
  stable free-oscillator roots r_i and lie inside the circle. It also tends to 1 as z → ∞ (α → w).
  Its winding along |z| = 1 therefore counts the multipliers outside the circle. This is the
  generalized (MIMO) Nyquist criterion of the regenerative loop. The D factor (the sinh ratio of the
  dense form) has only the stable free-oscillator zeros, so it drops out. A shift λ → λ + iω only
  conjugates M by a diagonal phase matrix, so the determinant is single-valued in z. By conjugate
  symmetry, half the circle suffices:
  Z = −Φ/π over μ = Im λ/ω ∈ [0, ½]. There is no strip offset a, no row scaling and no poles on the path.
* **O(Q) evaluation.** S(Δ) is a sum of two exponentials in Δ, with an extra factor e^{−a_iT} when
  ψ_p < ψ_q. M is therefore a unit-lower-triangular semiseparable matrix plus a rank-2 term, and
  det(I_Q + αM) = det(I₂ + W·R), with R from one forward recursion over the nodes. With equispaced
  (midpoint) nodes the node-to-node factors are constant, so one evaluation needs 2 complex
  exponentials for the window plus about 14 complex FMAs per node, and no division inside the loop
  (`D_mill2r`, a rolled loop).
  - Read as an algorithm, this recursion is a quadrature of the free oscillator's response over the
    cutting window. It never integrates the delayed system and builds no monodromy matrix.
  - The dense Q×Q LU of the same M (`D_mill2q`) gives the same numbers.

**Accuracy.**
* CPU prototype: 30 of 30 test points agree with the RK4 monodromy reference.
* Midpoint Q = 16 against Q = 64: 0 of 3200 counts differ in Float64.
* GPU forms against the Float64 Q = 64 reference (160×80 = 12 800 points):

| Q = 16 | differing counts | flagged |
|---|---|---|
| Float32 | 11 | 0 |
| Float16 evaluation (march in Float32) | 38 | 376 (2.9 %) |
| Float16 + Float32 re-check of the flagged points | 17 | — |

In Float16, 32 of the 38 differing points are flagged. Almost all flags mean that a multiplier lies
closer to |z| = 1 than Float16 resolves, so the count was decided from the root's side; 270 of the
376 flagged points lie on the stability boundary. The re-check repeats exactly these points in Float32.

**Speed** (RTX PRO 6000 Blackwell, Colab G4; 1920×1080 = 2.07 M points; `gpu_fast_bench.jl`,
`fast_kernel_info.jl`):

| | Q = 8 | Q = 16 | registers / spill (Q = 16) |
|---|---|---|---|
| Float32 | 36.9 ms (56 Mpts/s) | 69.6 ms (30 Mpts/s) | 167 / 0.9 KB |
| Float16 evaluation | 10.1 ms (206 Mpts/s) | 17.4 ms (119 Mpts/s) | 120 / 0.3 KB |

The unrolled forms (`D_mill2m`, `D_mill2s`) hit the 255-register limit and spill. The rolled loop does
not. When no root refinement is requested, the kernels are compiled without the refinement code
(`Val(:none)`), which removes its extra inlined copies of D. The interactive server, Q = 16,
first-order estimate, measured as a whole frame (kernel, colouring and the device-to-host copy of the
display image):

| | full HD | 4K | 8K |
|---|---|---|---|
| Float16 | 19–22 ms (46–53 fps) | 64.5 ms (15 fps) | 243 ms (4 fps) |
| Float16 + Float32 re-check | 25.6 ms (39 fps; 63 k points re-checked in 6.5 ms) | 85.4 ms (12 fps) | 315 ms (3 fps) |
| Float32 | 70.0 ms (14 fps) | 248 ms (4 fps) | — |

In the browser (the Colab page; full HD Float16 chart shown as a 960×540 image), the picture is
different, because the transfer through the kernel channel costs more than the GPU work:

| image sent to the browser | requests in flight | time per frame |
|---|---|---|
| PNG | 1 | 117 ms (8.5 fps) |
| JPEG, 114 KB, encoded in 2.4 ms | 1 | 77 ms (13 fps) |
| JPEG (now the default) | 2: the next frame is computed while the last one travels | 41 ms (24 fps) |

For comparison, the dense Hill form of the same chart (`D_mill2`, tol 1e−2) takes about 1 s for
480×270 on the same GPU.

## Interactive
`gpu/colab/NyquistGPU_Hill_Interactive.ipynb` (this branch) adds these time-periodic examples to the
three time-independent ones:
* *milling, straight flutes (Test 2, compressed Hill, fast)*: the default; Float16 / Float16 + re-check /
  Float32 / Float64; full HD.
* *delayed Mathieu*.
* *milling, straight flutes (Test 2, dense Hill, slow)*.
* *milling, different helix angles (Test 3)*.

Model constants are sliders: κ, ε for Mathieu; ζ, a/D, K_n/K_t for Test 2; ζ, a/D, β₂ for Test 3.

## Not done yet
* The compressed determinant for Test 3 (helix: nodes over the window × the axial slices, a
  distributed delay per node, spindle period) and for multi-DOF models (S becomes a matrix sum of 2n
  exponentials, so the semiseparable rank grows to 2n).
* The F32 kernel runs at a few per cent of the GPU's FMA peak: it is bound by latency and occupancy
  (one serial recursion per thread, 167 registers). Splitting the nodes over 2–4 threads per point,
  or evaluating two frequency samples per pass, would give the scheduler independent work.
* Banded/shifted truncation and Schur-complement ring growth (not needed for Mathieu: N = 9
  everywhere; becomes relevant for milling with many harmonics).
* Schur-complement ring growth (N is re-factorized per ring in the a-posteriori check) and the banded
  (shifted) window: at low spindle speed N reaches 20–40, a band around the resonant harmonics would
  cut that.
* A tighter a-posteriori estimate: the one-ring change underestimates the tail (×N/2 for Mathieu,
  ×2–8 for milling); a tail-sum (Richardson) correction would make it a bound.
