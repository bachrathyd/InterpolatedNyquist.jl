# Hill determinant + argument principle for time-periodic delayed systems

Branch `hill-argument-principle`. Stability of time-periodic delayed
systems without time integration: a regularized Hill determinant evaluated on the
imaginary axis, its phase unwrapped by the discrete march of the package, and the
argument principle on one period strip of the Floquet exponents. For straight-fluted
milling the infinite Hill determinant is also available in a **compressed form** (all
harmonics in closed form, a 16-node determinant counted along the unit circle of the
Floquet multiplier): full HD charts in 5 ms (Float16) / 16 ms (Float32) on one GPU and ~100
frames per second in the browser, see
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
  Vᴴ_ql = e^{ilψ_q}, with c_q = W_q h(ψ_q)/2π, where ψ_q are Q nodes in the window. The jump of h sits at
  the window edge, but the kernel S below has a slope kink on the diagonal (S'(0+) − S'(0−) = 2π/ω²):
  that kink, not the jump, limits the plain node rules to O(Q⁻²). Gauss nodes alone converge no
  faster than midpoints; Gauss nodes plus a diagonal kink correction reach O(Q⁻⁴) (math review).
* **Matrix determinant lemma:** det(D + αUVᴴ) = det D · det(I_Q + αM), M = VᴴD⁻¹U,
  M_pq = c_q S(ψ_p − ψ_q). Every harmonic is summed exactly:
  S(Δ) = Σ_{k∈ℤ} e^{ikΔ}/p(λ + ikω), and by partial fractions of 1/p,
  Σ_k e^{ikΔ}/(a + ikω) = T e^{−aΔ/ω}/(1 − e^{−aT}) for Δ ∈ [0, 2π).
  The only approximation left is the Q-node quadrature.
* **Counting on the unit circle of the Floquet multiplier.** det(I_Q + αM) depends on λ only through
  z = e^{λT}. As a function of z it is analytic outside |z| = 1: its poles e^{r_iT} come from the
  stable free-oscillator roots r_i and lie inside the circle, and there is a pole at z = 0 (from α and
  W_s ∝ 1/z), also inside. It tends to 1 as z → ∞ (α → w).
  Its winding along |z| = 1 therefore counts the multipliers outside the circle. This is the
  generalized (MIMO) Nyquist criterion of the regenerative loop. The D factor (the sinh ratio of the
  dense form) has only the stable free-oscillator zeros, so it drops out. A shift λ → λ + iω only
  conjugates M by a diagonal phase matrix, so the determinant is single-valued in z. By conjugate
  symmetry, half the circle suffices:
  Z = −Φ/π over μ = Im λ/ω ∈ [0, ½]. There is no strip offset a, no row scaling and no poles on the path.
* **O(Q) evaluation.** S(Δ) is a sum of two exponentials in Δ, with an extra factor e^{−a_iT} when
  ψ_p < ψ_q. M is therefore a unit-lower-triangular semiseparable matrix plus a rank-2 term, and
  det(I_Q + αM) = det(I₂ + W·R), with R from one forward recursion over the nodes. With equispaced
  (midpoint) nodes the node-to-node factors are constant, so there is no division inside the loop
  (`D_mill2r`, a rolled loop).
  - Read as an algorithm, this recursion is the kick–drift (Strang) propagator of the 2×2 ODE
    x'' + 2ζx' + (1 + α(z)h)x = 0 over the cutting window: as Q → ∞ the compressed F equals
    det(I − Φ_α(T)/z)/det(I − e^{A₀T}/z) to 1e−14 (math review). So it is a z-dependent monodromy-type
    product in disguise; the method stays the frequency-domain determinant counted by the argument
    principle (no time integration of the delayed system, no eigenvalues of a monodromy matrix).
  - The dense Q×Q LU of the same M (`D_mill2q`) gives the same numbers.
* **Normalized recursion** (`D_mill2n`, the fastest form).
  - In the recursion, row s of the running sums appears only divided by u_s = e^{−a_sψ/ω}, and both
    are scaled by the same propagator. With A_st = P_st/u_s the per-node scalings drop out, and with
    them every λ-dependent exponential except E = e^{−λT} = 1/z: F is visibly a function of the
    Floquet multiplier.
  - Of the node position only κ = u₂/u₁ = e^{−2i√(1−ζ²)ψ/ω} is left. It is a unit rotation that does
    not depend on λ, so it needs no dual numbers.
  - Cost per node: about 64 real multiplications instead of about 140. Per evaluation: 1 dual
    exponential instead of 3.
  - The values agree with `D_mill2r` to 6e−16 and the derivatives to 1e−12. Its sweeps give the same
    counts: 11 differences from the Q = 64 reference in Float64 and Float32, 33 in Float16.

**Accuracy.**
* CPU prototype: 30 of 30 test points agree with the RK4 monodromy reference.
* Midpoint Q = 16 against Q = 64: 0 of 3200 counts differ in Float64.
* GPU forms against the Float64 Q = 64 reference (160×80 = 12 800 points):

| Q = 16 | differing counts | flagged |
|---|---|---|
| Float32 | 11 (the same 11 in Float64: quadrature, not rounding) | 0 |
| Float16 evaluation (march in Float32) | 33–38 | 376–400 (~3 %) |
| Float16 + Float32 re-check of the flagged points | 17 | — |

In Float16, almost all differing points are flagged. Almost all flags mean that a multiplier lies
closer to |z| = 1 than Float16 resolves, so the count was decided from the root's side; 270 of the
376 flagged points lie on the stability boundary. The re-check repeats exactly these points in Float32.

**Speed** (RTX PRO 6000 Blackwell, Colab G4; 1920×1080 = 2.07 M points; one thread per point, which
beat the persistent-lane schedules in `fast_schedule_bench.jl`; `fast_kernel_info.jl`):

| kernel | Q = 8 | Q = 16 | registers / local memory (Q = 16) |
|---|---|---|---|
| `D_mill2r`, Float32 | 30.1 ms (69 Mpts/s) | 56.6 ms (37 Mpts/s) | 167 / 944 B |
| `D_mill2r`, Float16 evaluation | 8.6 ms (241 Mpts/s) | 15.0 ms (138 Mpts/s) | 120 / 336 B |
| **`D_mill2n`, Float32** | 9.4 ms (220 Mpts/s) | **15.9 ms (131 Mpts/s)** | 136 / 576 B |
| **`D_mill2n`, Float16 evaluation** | 3.3 ms (634 Mpts/s) | **5.0 ms (415 Mpts/s)** | 104 / 248 B |

These timings use the march steps of the interactive example (h0 = 0.05, hrel = 0.25 instead of 1e−3
and 0.05). The cap of the strip examples is not needed on the circle: same counts on the 12 800 test
points, with 13.9 instead of 18.3 evaluations per point.

The unrolled forms (`D_mill2m`, `D_mill2s`) hit the 255-register limit and spill; the rolled loops do
not. When no root refinement is requested, the kernels are compiled without the refinement code
(`Val(:none)`), which removes its extra inlined copies of D.

The interactive server (`D_mill2n`, Q = 16, first-order estimate), measured as a whole frame: kernel,
colouring, device-to-host copy and JPEG encoding of the display image:

| | full HD | 4K | 8K |
|---|---|---|---|
| Float16 | 7.8 ms (128 fps) | 23.8 ms (42 fps) | 75 ms (13 fps) |
| Float16 + Float32 re-check | 10.1 ms (99 fps) | 28.9 ms (35 fps) | — |
| Float32 | 18.9 ms (53 fps) | 64.8 ms (15 fps) | 220 ms (4.5 fps) |

In the browser (the Colab page; full HD Float16 chart shown as a 960×540 image), the transfer, not
the GPU, first decided the frame rate. Each request through Colab, via the kernel channel or the port
proxy alike, costs a ~55 ms round trip:

| frames sent to the browser | time per frame |
|---|---|
| `D_mill2r`, PNG through the kernel channel, one request per frame | 117 ms (8.5 fps) |
| `D_mill2r`, JPEG (114 KB, 2.4 ms to encode), one request per frame | 77 ms (13 fps) |
| `D_mill2r`, JPEG, two requests in flight | 41 ms (24 fps) |
| `D_mill2r`, JPEG streamed through the port proxy | 17.9 ms (56 fps) |
| **`D_mill2n`, JPEG streamed through the port proxy** (now the default) | **10.1 ms (99 fps)** |

In the stream mode, the page posts each new state, and a small HTTP server in the notebook renders the
newest state as soon as the previous frame is out. It writes the frames into one open response, which
the proxy passes through unbuffered. The frame rate is then that of the GPU plus the JPEG encoding.
The latency from a slider move to its picture is one round trip plus one render, with no queue.

For comparison, the dense Hill form of the same chart (`D_mill2`, tol 1e−2) takes about 1 s for
480×270 on the same GPU.

## Code review, iteration 1 (2026-10-06)
Six specialist reviewers (mathematics, Julia/GPU engineering, GPU performance, numerical linear
algebra, machining dynamics, adaptive sampling) read the code and ran CPU experiments. Iteration 1
implemented their main findings:

* **The kernels were bound by function calls, not arithmetic.** D was compiled as a separate function
  whose arguments and results went through local memory, and so were the complex dual products in its
  loop. `CUDABackend(always_inline = true)` (`gpu/scripts/common.jl`) removes the calls. Together with
  4 × 8 pixel tiles per warp for full charts (`k_tile!`; neighbouring pixels need similar step counts),
  the old `D_mill2r` in Float32 went from 56.6 ms to 3.5 ms at full HD.
* **Per-point `prepare` hook** (`NyquistGPU.prepare(D, p, c)`): run once per point in the march
  precision; the march then calls D(λ, q, c). The kernels now receive the constants in the march
  precision (refinement and certification use them unrounded) and convert them for D. In Float16
  mode, the chart point was rounded to Float16 before (1920 columns collapsed to 1633 distinct spindle
  speeds), and ω overflowed above 32.75 krpm (z = 2) or for f_n ≥ 1092 Hz.
* **`D_mill2p`, the pole-free prepared Test 2 form** (`hill/gpu_fast.jl`):
  - With C21 = κB21 and C12 = κ̄B12, κ drops out (|κ| = 1). The k_s leave the loop.
  - Multiplying F by (1 − W1)(1 − W2) removes every division and the poles at the free-oscillator
    multipliers. The factor has zero winding on |z| = 1 and is positive at z = ±1, so the count and
    the parity rule are unchanged.
  - Per evaluation: one exponential. Per node: about 44 real multiplications.
  - **Light damping:** at ζ = 0.002 it gives the right counts with the large march steps (2/3200
    against a fine reference). The old forms, whose poles sit near the circle there, miscounted
    133/3200.
  - **Float16:** with the constants prepared in Float32, all wrong counts are flagged (24, 0
    unflagged, against 34 with 8 unflagged before). Float16 + re-check equals Float32 (11 differences
    from Q = 64, the quadrature level).
* **`D_mill3c`, Test 3 in the compressed form** (`hill/gpu_helix.jl`):
  - Material nodes: tooth × axial Gauss node × window Gauss node. The regenerative term of a node is
    exactly the same material node on the previous tooth, so there is no interpolation.
  - D(z) = det(I + (I − P(z))S_z C).
  - Reduced once per point to a determinant of size n_w + 2, where n_w is the number of nodes whose
    regeneration arc crosses the time origin. The origin is chosen to minimize it: one tooth's nodes.
  - The reduction uses forward sweeps with running sums (no N × N matrix), with pole-free rows.
  - CPU checks: reduced = dense determinant to 7e−15. Counts against the dense Hill (exact kernel) on
    288 points: 3 differ at Q = 8, n_s = 4 and 1 at n_s = 5. Float32 counts equal Float64. RK4: 4/4.
    About 10 evaluations per point.
* **Bugs fixed:**
  - `recheck_flagged!` skipped points with the lane schedules.
  - The dense Test 3 form had a Float64 literal (Float64 instructions in every GPU evaluation).
  - A map over a 33-entry constants tuple did not compile on the GPU.
  - A complex division in a setup widened to Float64.
* **Offline PTX check without a GPU:** `gpu/scripts/ptx_offline.jl`, `hill/gpu_compile_check.jl`.

RTX PRO 6000, full HD (`fast_kernel_info.jl`, 4 × 8 tiles, workgroup 64):

| kernel, Q = 16 | before | iteration 1 | registers |
|---|---|---|---|
| `D_mill2r` Float32 | 56.6 ms | 3.54 ms | 163 |
| `D_mill2n` Float32 / Float16 | 15.9 / 5.0 ms | 2.28 / 1.94 ms | 145 / 120 |
| **`D_mill2p` Float32 / Float16** | — | **1.30 / 1.29 ms** (1.6 Gpts/s) | 92 / 72 |

Float16 no longer beats Float32 on this GPU: its earlier advantage came from smaller stack frames,
not from Float16 arithmetic.

GPU tour on the RTX PRO 6000 (`hill/gpu_tour_milling.jl`, `gpu/results/hill_tour_G4_iter1.csv`). Check:
192 × 108 points against a Float64 reference of the same model with a finer quadrature or tolerance.

| model | Float32 | Float16 | Float16 + re-check | check: differ of 20 736 |
|---|---|---|---|---|
| Test 2, `D_mill2p` (Q = 16), full HD | 1.49 ms | 1.30 ms | 2.05 ms | 13 / 47 / 13 |
| Test 3, `D_mill3c` (Q = 8, n_s = 4), full HD | 13.1 s | 13.6 s | 14.0 s | 38 / 42 / 37 |
| Test 3, dense `D_mill3` (tol 1e−2), 480 × 270 | 1.0 s (≈ 16 s at full HD) | — | — | 5 |

Your full-HD Test 3 chart took 61.5 s before. The Test 3 forms are now limited by the per-thread
matrix in GPU memory (about 0.5 TFLOPS effective); that is the target of iteration 2.

## Code review, iteration 2 (2026-10-06)
The same six reviewers read the iteration-1 code again (round 2); the GPU reviewer then wrote the
warp-cooperative Test 3 kernel (round 3). Implemented:

* **Test 2: `D_mill2g`** (mathematics). Gauss nodes plus the diagonal kink correction, in the
  pole-free prepared form of `D_mill2p`. The convergence is O(Q⁻⁴):
  - Q = 8 differs from a Q = 32 reference at 1 of 12 800 points (ζ = 0.011) and at 0 (ζ = 0.002);
    `D_mill2p` at Q = 16 differs at 12 and 5.
  - Float16: every wrong count is flagged, and Float16 + re-check equals Float32.
  - The correction adds poles z = g/(1 + g); on the chart all lie inside |z| < 0.04
    (`mill2g_polecheck`).

  It is now the Test 2 form of the app and the tour.
* **Test 3: `D_mill3h`** (numerical linear algebra). The leading block X11 of the reduced
  (n_w + 2)-square determinant does not depend on λ.
  - Once per point, scaled Householder reflectors bring X11 to Hessenberg form.
  - Per evaluation, a bordered Hessenberg elimination makes exactly the pivot choices of the dense
    partial-pivoting LU, in O(n_w²).
  - That is 7.5× fewer operations per evaluation and 4× less work per point. The values equal
    `D_mill3c` to 1e−14.
* **Test 3 setup, v2** (machining dynamics, mathematics):
  - **Axial nodes per point** from the helix lag across the depth:
    n_s = clamp(⌈2 + 1.6 a_p b_max / (2πΩ)⌉, 2, 6), with b = tan β / R.
  - **Kink correction** of Test 2 for every node.
  - **Range** 3–30 krpm × 0–10 mm (the old 8–30 krpm left out the dense low-speed lobes).
  - **Check against Q = 10, n_s ≤ 8:** 3 of 1152 counts differ (16 with the fixed n_s = 4).
* **Robustness** (Julia/GPU engineering):
  - **Impossible counts are flagged** (Z < 0 or Z > zmax). The pole-free determinants are
    polynomials in 1/z of degree ≤ 2Q + 2 (Test 2) and ≤ n_w + 4 (Test 3).
  - **Parity check of circle marches:** Z ≡ [D(1) < 0] + [D(−1) < 0] (mod 2), else flag 16.
  - **End rule at ω_max:** a flip multiplier at μ = ½.
  - **Trust radius qtrust·h** for the one-step root estimates. The old radius max(|λ|, 1) accepted
    far-away roots.
  - **Wrapped nodes of `D_mill3c`/`D_mill3h` are decided by the time order.** With a node exactly at
    the time origin, the angle test and the sort could disagree, and a sweep then read a stale running
    sum (found by the GPU reviewer).
* **Adaptive charts** (adaptive sampling). `run_adaptive!` refines from coarse to fine, over strides
  16, 8, 4, 2, 1.
  - **Certificate:** every march returns ρ = min |D/D′| over its samples, the distance from the
    contour to the nearest root.
  - **Refinement rule:** a cell is refined when its corner counts differ, a corner is flagged, or a
    corner has ρ < L · (cell side).
  - **Slope bound L:** taken from the data as Lscale × the largest |Δρ| per pixel. With
    `Lmode = :local` (the default), each first-pass cell takes it from the edges of every evaluated
    level near it.
  - **Why local:** the reviewer's single first-pass bound missed 1-pixel-wide lobes at low spindle
    speed on a 192 × 108 Test 3 chart (30 of 20 736 pixels; now 0). A secant between first-pass
    nodes cannot see slopes above ρcut/16, and the lobes get denser like 1/rpm².
  - **Supporting changes:**
    - `run_list!` marches index lists with every schedule.
    - `warm = true` starts from the pixels of the previous call (slider moves).
    - The Float16 + re-check format re-checks after every pass.
    - `stepctl = :damped` is an optional step controller.
* **Warp-cooperative `D_mill3h`** (`hill/warp/`, GPU performance). One warp per point:
  - **Prepare:** the 32 lanes share it (nodes, origin search, ranks, forward sweeps with one
    right-hand side per lane, the Householder reduction).
  - **Evaluation:** in the bordered Hessenberg elimination the lanes are the columns of the three
    active rows. Every lane reads the pivot candidates from shared memory and takes the same decision.
  - **Shared memory:** the X matrices and row buffers live there, sized per axial class n_s. On the
    chart n_w = 8 n_s, and n_s = 3 for 76 % of the points, which needs 8.3 KB per warp instead of
    33 KB. The shared memory is dynamic, so one compiled kernel per number format serves every class.
  - **Bit-identical** to `mill3h_prepare` + `D_mill3h` on the CPU emulation: q, all 162 240 X
    entries, 600 D values in Float32 and Float16, and 120 full marches.
  - **Registers set the occupancy.** On the G4 it uses 255 registers per thread, so at most 8 warps
    per SM are resident. Sizing the persistent grid as one resident wave (occupancy API) cut the
    full-HD chart from 893 to 736 ms. A register cap is slower: 787 ms at 168 and 915 ms at 128.

**GPU tour** (`hill/warp/tour_warp3h.jl`, `hill/gpu_tour_milling.jl`; `gpu/results/hill_tour_*_iter2.csv`).
The hardest case is Test 3 at full HD: 1920 × 1080 = 2.07 M points, 3–30 krpm × 0–10 mm, helix 30°/45°.
* **Formats:** F32 = Float32; F16 = D evaluated in Float16 with a Float32 march; F16+ = F16 followed by
  a Float32 re-check of the flagged points.
* **Check:** 192 × 108 points against a Float64 reference with Q = 10, n_s ≤ 8.
* **Adaptive:** compared with the full chart of the same format at every pixel.

Test 3, warp-cooperative kernel (`mill3w`), full HD:

| GPU | F32 | F16 | F16+ | adaptive F32 / F16 / F16+ | check: differ of 20 736 | adaptive: pixels ≠ full chart |
|---|---|---|---|---|---|---|
| RTX PRO 6000 (G4) | 736 ms | 774 ms | 793 ms | 451 ms / 511 ms / 574 ms | 54 / 62 / 53 | 0 / 0 / 0 |
| A100 40 GB | 1.86 s | 2.12 s | 2.18 s | 908 ms / 1.09 s / 1.20 s | 54 / 62 / 53 | 0 / 0 / 0 |
| L4 | 2.38 s | 2.68 s | 2.79 s | 1.28 s / 1.47 s / 1.60 s | 54 / 62 / 53 | 0 / 0 / 0 |
| T4 | 8.11 s | 9.39 s | 9.46 s | 4.14 s / 4.75 s / 5.04 s | 54 / 62 / 53 | 0 / 0 / 0 |

For comparison, the other forms (full HD unless noted; F32 / F16 / F16+):

| GPU | Test 3, one thread per point (`mill3h`) | `mill3h` adaptive | Test 3, dense Hill (`mill3d`), 480 × 270, F32 | Test 2 (`mill2g`, Q = 8) |
|---|---|---|---|---|
| RTX PRO 6000 (G4) | 9.21 s / 9.19 s / 9.31 s | 5.07 s / 5.13 s / 5.60 s | 1.14 s | 1.57 ms / 1.50 ms / 2.32 ms |
| A100 40 GB | 26.16 s / 26.84 s / 27.39 s | 12.78 s / 13.32 s / 14.28 s | 2.25 s | 6.22 ms / 5.35 ms / 7.56 ms |
| L4 | 31.77 s / 31.51 s / 31.99 s | 14.29 s / 14.59 s / 15.44 s | 2.70 s | 5.32 ms / 5.49 ms / 7.31 ms |
| T4 | 120.80 s / 184.69 s / 186.33 s | 50.19 s / 77.74 s / 79.76 s | 6.59 s | 33.06 ms / 27.84 ms / 33.59 ms |

* **The counts do not depend on the GPU.** All four GPUs give the same check numbers (54 / 62 / 53
  for the warp kernel; 54 / 61 / 53 for one thread per point, whose Float16 rounding differs
  slightly). The 54 differences are the quadrature level of Q = 8 against Q = 10.
* **Adaptive refinement** marches 33 % of the pixels and gives exactly the full chart in every
  format on every GPU. It takes half the time of the full chart; the boundary pixels it marches are
  the expensive ones.
* **Float16 does not pay for Test 3:**
  - The prepare step stays in Float32.
  - The warp evaluation is dominated by the pivot logic and shared-memory traffic, not by
    arithmetic.
  - At full HD, 21 Float16 points fail (non-finite), and the re-check of F16+ repairs them.
  - Float32 is the format to use for Test 3. F16 is 5–16 % slower, F16+ 8–18 % slower.
* **Against your dense full-HD Test 3 chart (61.5 s):** the G4 now needs 0.74 s, or 0.45 s
  adaptively, 80–135× faster. On the T4 the warp kernel (8.1 s) also beats the dense form on the G4
  (about 18 s at full HD, from 1.14 s at 480 × 270).
* **Registers:** 255 per thread on the G4, 215–238 on the others, so at most 8 resident warps per SM.
  Shared memory limits only the large classes (n_s ≥ 5) and the T4 (64 KB per SM).

## Interactive
`gpu/colab/NyquistGPU_Hill_Interactive.ipynb` (this branch) adds these time-periodic examples to the
three time-independent ones:
* *milling, straight flutes (Test 2, compressed Hill, fast)*: the default (`D_mill2g`, Q = 8); Float16 /
  Float16 + re-check / Float32 / Float64; full HD.
* *milling, different helix angles (Test 3, compressed Hill, fast)*: `D_mill3h`. On the GPU it uses
  the warp-cooperative kernel (one warp per point, first-order σ estimate). Range 3–30 krpm ×
  0–10 mm; starts at 960 × 540 (full HD: 0.74 s on a G4).
* *delayed Mathieu*.
* *milling, straight flutes (Test 2, dense Hill, slow)*.
* *milling, different helix angles (Test 3)*: the dense Hill form.

Model constants are sliders: κ, ε for Mathieu; ζ, a/D, K_n/K_t for Test 2; ζ, a/D, β₂ for Test 3.

## Not done yet
* **Register pressure of the warp kernel.** It uses 255 registers per thread on the G4 (sm_120) and
  215–218 on the A100 and L4, which allows at most 8 resident warps per SM. A leaner evaluation could
  double the occupancy, for example pivot logic in fewer live values and row buffers only in shared
  memory. A plain register cap does not help: 168 and 128 were slower.
* **Fewer adaptive passes.** A full-HD chart takes 11 passes; the last six come from updates of the
  local slope bounds. Every pass costs a launch per axial class plus a stream compaction.
* The compressed determinant for multi-DOF models: S becomes a matrix sum of 2n exponentials, so the
  semiseparable rank grows to 2n.
* Banded/shifted truncation and Schur-complement ring growth (not needed for Mathieu: N = 9
  everywhere; becomes relevant for milling with many harmonics).
* Schur-complement ring growth (N is re-factorized per ring in the a-posteriori check) and the banded
  (shifted) window: at low spindle speed N reaches 20–40, a band around the resonant harmonics would
  cut that.
* A tighter a-posteriori estimate: the one-ring change underestimates the tail (×N/2 for Mathieu,
  ×2–8 for milling); a tail-sum (Richardson) correction would make it a bound.
