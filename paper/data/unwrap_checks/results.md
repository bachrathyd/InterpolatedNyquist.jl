# Unwrap back-end: three numerical checks (s16)

Script: `paper/scripts/studies/s16_unwrap_checks.jl`, run as
`julia --project=gpu/validate -t 12 paper/scripts/studies/s16_unwrap_checks.jl`
(i5-10400, 12 threads, about 2.5 to 4 min in total). The raw data is in the CSV files next to this note.
Unwrap uses the package defaults (tol = 0.3 rad, hrel = 1, hmax = Inf, ωband = 0,
maxsteps = 200 000) unless a column says otherwise. Evaluations are the march's own
counter (`_unwrap_march(...)[3]`). ε = |Zraw − round(Zraw)|.

## Check 1: neutral gallery cases (A.4, A.5, A.6), 40×40 grid

Each case uses the gallery ranges and n_power_max = 2. The reference is ODE Vern9 at
reltol = abstol = 1e-9 with the gallery ω_max (200 / 200 / 500). "inside" means
|neutral coef| < 1, which is |a| for A.4/A.5 and |A| for A.6. Inside counts are 1280 / 1280 / 1360 points.

| case | method | ω_max | Z ≠ ODE (all / inside) | failed (Z=−1) | evals median / max | ε median / max | n(ε>1/4) | parity viol. (inside) | wall [s] |
|---|---|---|---|---|---|---|---|---|---|
| A.4 neutral | ODE 1e-9 | 200 | ref | 0 | – | 0.148 / 0.380 | 320 | **32** | 1.3 |
| A.4 neutral | unwrap, gallery ω_max | 200 | 40 / 32 (all on a = c, see note) | 0 | 266 / 877 | 0.148 / 0.380 | 320 | 0 | 0.04–0.4 |
| A.4 neutral | unwrap, default | 1e5 | 508 / 188 | **480** | 127 335 / 200 001 | 0.004 / 0.091 | 0 | 0 | 20.4 |
| A.5 high-gain | ODE 1e-9 | 200 | ref | 0 | – | 0.144 / 0.382 | 320 | 0 | 1.3 |
| A.5 high-gain | unwrap, gallery ω_max | 200 | 0 / 0 | 0 | 259 / 811 | 0.144 / 0.382 | 320 | 0 | 0.04 |
| A.5 high-gain | unwrap, default | 1e5 | 480 / 160 | **480** | 127 329 / 200 001 | 0.004 / 0.091 | 0 | 0 | 16.6 |
| A.6 PDA | ODE 1e-9 | 500 | ref | 0 | – | 0.070 / 0.490 | 280 | 0 | 2.8 |
| A.6 PDA | unwrap, gallery ω_max | 500 | 0 / 0 | 0 | 648 / 1775 | 0.070 / 0.490 | 280 | 0 | 0.13 |
| A.6 PDA | unwrap, default | 1e5 | 560 / 320 | **560** | 127 334 / 200 001 | 0.004 / 0.032 | 0 | 0 | 18.6 |

The ε statistics of the default unwrap rows cover only the points that did not fail.

Notes.
- On A.4 the 40 disagreements all lie on the grid diagonal a = c. There
  D = (λ²+1)(1 + a e^{−λ}), so a root sits exactly on the line at λ = ±i and Z is
  undefined. ODE returns Z = 1, which violates parity: these are its 32 parity
  violations. Its ε at those points ranges from 0.008 to 0.38, so the ε>1/4 flag catches only 7 of the 40.
  Unwrap returns 0 or 2 there with no flag. Off the diagonal, unwrap agrees with the ODE reference at every point.
- On neutral systems the ε>1/4 flag does not mean "root on the line". It fires on every
  point with |coef| ≳ 0.7, because the truncated tail term asin|a|/π exceeds 1/4 there. The same
  points are flagged for ODE and unwrap because both see the same tail.
- With default ω_max = 1e5 the march runs into maxsteps = 200 000 and fails (NaN, Z = −1)
  at every point with |coef| > 1 and at most points with 0.8 ≲ |coef| < 1. Evals, median/max per band of
  |coef| (A.4): < 0.5: 55 772 / 111 538; 0.5–1: 173 870 / 200 001; > 1: all 200 001.
  The gallery-ω_max figures for the same bands are 135 / 351, 356 / 645 and 555 / 877.
- The n_power estimator gives 2.000 at the first grid point of every case.

ω_max scaling of the unwrap evaluation count (A.4, n_power_max = 2; ODE reference at ω_max = 200):

| (a, c) | ODE Z (Zraw) | ω_max = 1e2 | 1e3 | 1e4 | 1e5 |
|---|---|---|---|---|---|
| (0, 0.5): retarded, no neutral term | 2 (2.000) | 17 ev, Z=2 | 21, Z=2 | 24, Z=2 | 27, Z=2 |
| (0.5, −0.5) | 0 (−0.108) | 113, Z=0 | 1 221, Z=0 | 12 679, Z=0 | 127 271, Z=0 (68 ms) |
| (0.95, 0.5) | 2 (1.836) | 334, Z=2 | 2 918, Z=2 | 28 923, Z=2 (Zraw 1.60) | **fail** (200 001) |
| (−1.1, 0.5): essential instab. | 64 (64.36) | 294, Z=32 | 2 894, Z=320 | 29 277, Z=3184 | **fail** (200 001) |

When a ≠ 0 the cost is linear in ω_max, at about 1.2 evals per unit ω for |a| = 0.5 and
about 2.9 for |a| ≈ 1. On the essentially unstable side the count grows as ω_max/π, as expected.

## Check 2: rightmost-root accuracy at the tab:semidisc points (showcase)

Reference: the s07 reference (ODE 1e-9, 10 roots, Newton 15) polished in BigFloat
(256 bit), with |D(λ_ref)| ≤ 5e-77. It was cross-checked against the max-Re of all
10 unwrap minima after BigFloat polishing, and the two agree to all printed digits.
Reference rightmost roots: stable −0.026811+1.9360i, near boundary +0.040619+2.1154i,
unstable +0.210231+1.8707i. The ODE rows use tol 1e-5 and ω_max = 1e4. Unwrap uses its defaults with
n_power_max = 4. The table gives |σ̂ − σ_ref|. :Newton is 4 steps (the s07 setting).
:Combined gave exactly the same σ̂ as :Newton everywhere.

| point | method | raw (:Linear) | :Newton | :Combined | root picked |
|---|---|---|---|---|---|
| stable | ODE, 1 root | 5.1e-3 | 5.2e-3 | 5.2e-3 | −0.032+0.749i (not rightmost) |
| stable | ODE, 10 roots max-Re (table method) | 2.7e-4 | 6.9e-17 | 6.9e-17 | rightmost |
| stable | unwrap, 1 root | 5.1e-3 | 5.2e-3 | 5.2e-3 | −0.032+0.749i (not rightmost) |
| stable | unwrap, 10 roots max-Re | 1.8e-5 | 4.5e-17 | 4.5e-17 | rightmost |
| near boundary | ODE, 1 root | 7.8e-2 | 7.8e-2 | 7.8e-2 | −0.037+0.955i (**wrong sign**) |
| near boundary | ODE, 10 roots max-Re | 2.7e-4 | 0 | 0 | rightmost |
| near boundary | unwrap, 1 root | 7.8e-2 | 7.8e-2 | 7.8e-2 | −0.037+0.955i (**wrong sign**) |
| near boundary | unwrap, 10 roots max-Re | 3.2e-4 | 6.9e-18 | 6.9e-18 | rightmost |
| unstable | ODE, 1 root | 6.8e-2 | 6.2e-2 | 6.2e-2 | +0.149+1.046i (not rightmost) |
| unstable | ODE, 10 roots max-Re | 3.6e-2 | 5.5e-11 | 5.5e-11 | rightmost |
| unstable | unwrap, 1 root | 6.9e-2 | 6.2e-2 | 6.2e-2 | +0.149+1.046i (not rightmost) |
| unstable | unwrap, 10 roots max-Re | 3.5e-2 | 4.6e-11 | 4.6e-11 | rightmost |

The ODE 10-root rows reproduce the "this work, tol 1e-5" rows of tab_semidisc.

## Check 3: boundary stress on both sides of the showcase Hopf boundary (D = 1.5)

P_b = 2.98640021455434912 from the s13 Float64 bisection. The BigFloat crossing parameter
is 2.986400214554348921, which differs by 2e-16. σ_true is the real part of the crossing root
at the exact Float64 P, from a 256-bit Newton solve (ω_c ≈ 1.0578). The true Z is 2 below P_b and 4 above.
Both values were checked with ODE 1e-10 at ±1e-2 and assigned by the sign of σ_true.
Parity: D(0) = P − 1 > 0 and the leading coefficient is positive (D(1e6) > 0), so Z must be even.
Each solver cell shows Z / σ̂ (raw). "sgn" is whether sign σ̂ = sign σ_true, and "par" is whether Z is even.

| ΔP | true Z | σ_true | ODE tol 1e-5: Z / σ̂ / sgn / par | ODE tol 1e-8: Z / σ̂ / sgn / par | unwrap: Z / σ̂ / sgn / par | unwrap evals |
|---|---|---|---|---|---|---|
| −1e-2 | 2 | −6.635e-4 | 2 / −6.64e-4 / ok / ok | 2 / −6.64e-4 / ok / ok | 2 / −6.635e-4 / ok / ok | 72 |
| −1e-4 | 2 | −6.596e-6 | 2 / −6.60e-6 / ok / ok | 2 / −6.60e-6 / ok / ok | 2 / −6.596e-6 / ok / ok | 102 |
| −1e-6 | 2 | −6.596e-8 | **3** / +5.88e-6 / **no** / **viol** | 2 / −6.60e-8 / ok / ok | 2 / −6.596e-8 / ok / ok | 118 |
| −1e-8 | 2 | −6.596e-10 | **3** / +1.15e-6 / **no** / **viol** | 2 / −6.60e-10 / ok / ok | 2 / −6.596e-10 / ok / ok | 131 |
| −1e-10 | 2 | −6.596e-12 | **3** / −8.17e-6 / ok / **viol** | **3** / +2.08e-6 / **no** / **viol** | 2 / −6.596e-12 / ok / ok | 146 |
| −1e-12 | 2 | −6.595e-14 | **3** / +1.83e-11 / **no** / **viol** | **3** / +3.73e-8 / **no** / **viol** | 2 / −6.596e-14 / ok / ok | 165 |
| +1e-2 | 4 | +6.556e-4 | 4 / +6.56e-4 / ok / ok | 4 / +6.56e-4 / ok / ok | 4 / +6.556e-4 / ok / ok | 73 |
| +1e-4 | 4 | +6.595e-6 | 4 / +6.60e-6 / ok / ok | 4 / +6.60e-6 / ok / ok | 4 / +6.595e-6 / ok / ok | 92 |
| +1e-6 | 4 | +6.596e-8 | **3** / +2.60e-4 / ok / **viol** | 4 / +6.60e-8 / ok / ok | 4 / +6.596e-8 / ok / ok | 106 |
| +1e-8 | 4 | +6.596e-10 | **3** / +1.16e-5 / ok / **viol** | **3** / +4.28e-7 / ok / **viol** | 4 / +6.596e-10 / ok / ok | 137 |
| +1e-10 | 4 | +6.596e-12 | **3** / +1.38e-5 / ok / **viol** | **3** / +2.98e-6 / ok / **viol** | 4 / +6.596e-12 / ok / ok | 156 |
| +1e-12 | 4 | +6.598e-14 | **3** / +8.41e-7 / ok / **viol** | **3** / +1.04e-7 / ok / **viol** | 4 / +6.596e-14 / ok / ok | 157 |

## Summary

1. **Neutral systems, gallery ω_max.** Unwrap matches the tight ODE reference (1e-9) at every point
   where Z is defined, in all three neutral cases. The only disagreements, 40 points on A.4, lie on the
   a = c diagonal, where a root sits exactly on the axis (λ = ±i). There ODE returns an odd Z that parity
   rejects, and unwrap silently returns 0 or 2. Unwrap needs a median of 260–650 evaluations per point and
   is roughly 20–30× faster than ODE at 1e-9 (single-shot timings, so noisy).
2. **Neutral systems, default ω_max = 1e5.** Unwrap is slow and fails here. A neutral phase ripple never
   decays, so the evaluation count grows linearly with ω_max: about 1.2–3 evals per unit ω, a median of
   127 000 per point, and 17–20 s per 1600-point chart. Points with |coef| ≳ 0.8, and every essentially
   unstable point, exhaust maxsteps = 200 000 and return Z = −1. That is 480–560 of 1600 points per case.
   On neutral systems ω_max has to be set explicitly, for example to the gallery's 200–500.
3. **ε > 1/4 flag on neutral systems.** The flag carries no root-on-line information there. The tail term
   asin|a|/π flags every point with |coef| > 0.7 for both back-ends, yet misses 33 of the 40 genuine
   root-on-line points.
4. **Rightmost-root estimate.** With one tracked root, ODE and unwrap both return the deepest |D| minimum,
   which is not the rightmost root at any of the three points. Its σ̂ is off by 5e-3 to 8e-2 and has the
   wrong sign at the near-boundary point, where Z = 2 still classifies the point correctly. With 10 tracked
   roots and max Re, unwrap's raw σ̂ is within 1.8e-5, 3.2e-4 and 3.5e-2 of the reference, the same order as
   ODE (2.7e-4, 2.7e-4, 3.6e-2). Newton or Combined polish brings it to 5e-17, 7e-18 and 5e-11. :Combined
   equals :Newton everywhere.
5. **Boundary stress.** Unwrap gets Z right at all 12 offsets on both sides, down to ΔP = ±1e-12, using
   72–165 evaluations. Its σ̂ matches the BigFloat crossing root to 3–4 digits even at |σ| ≈ 7e-14, and its
   sign is always correct. ODE at 1e-5 miscounts (Z = 3) for |ΔP| ≤ 1e-6 on both sides, 8 of 12 points.
   ODE at 1e-8 miscounts 5 of 12 points (−1e-10, −1e-12, +1e-8, +1e-10, +1e-12). At the miscounted points
   ODE's σ̂ has the wrong sign in 5 of 13 cases, all on the Z = 2 side.
6. **Parity rule.** Every ODE miscount in Check 3 (13 of 13) gives an odd Z where parity requires an even
   one, so the rule flags them all. No correct count violated parity.
