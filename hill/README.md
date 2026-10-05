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

## Not done yet
* Banded/shifted truncation and Schur-complement ring growth (not needed for Mathieu: N = 9
  everywhere; becomes relevant for milling with many harmonics).
* Test 2 (straight-fluted milling, 1 DOF, jump in the force coefficient: full Fourier series
  of h(t) → banded dense Hill matrix, LU instead of the continuant) and Test 3 (helix, kernel
  approximation).
