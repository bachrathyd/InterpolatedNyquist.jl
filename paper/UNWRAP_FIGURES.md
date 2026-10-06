# Example charts recomputed with the unwrap march

Branch `paper-unwrap-figures` (from `gpu-cuda`). The background grids of the
example charts (s01 showcase, s08 gallery a/b, s09 σ-contours, s10 Gao et al.
reproduction) are now computed with the discrete phase-unwrapping march
(`calculate_unstable_roots_unwrap_p_vec`) wherever it is valid. Each unwrap grid
was cross-checked against the phase-ODE grid computed with the panel's original
settings on the same points. The per-chart results are in
`data/unwrap_check_<study>.csv`, and every differing point is listed in
`data/unwrap_check_<study>_points.csv`.

* `generate_all.jl --ode-charts` reproduces the original phase-ODE grids, and
  `--no-chart-check` skips the ODE cross-check.
* The ODE back-end is set per panel in `UNWRAP` (`s08_gallery.jl`), in
  `SHOWCASE_UW` (`systems.jl`) and in `s10`. Where a panel keeps the ODE
  back-end, the reason is written next to that setting.
* The MDBM boundary traces are **unchanged**. `mdbm_boundary` (systems.jl) calls
  the phase-ODE back-end (`calculate_unstable_roots_direct`) inside its
  objective. The MDBM times on the panels are for the same ODE back-end,
  re-measured on this machine.

**Machine.** New times come from an Intel i5-10400 with 12 threads, the same
machine as `tab_unwrap` / `s13`. The "old" times are the committed values from a
Ryzen 7 5800H with 16 threads. For a fair ratio, the table also gives the ODE
grid timed on the i5 (a single run; the unwrap time is the best of 3).
`data/machine.txt` and `generated/machine.tex` were **not** updated. They still
describe the Ryzen, which ran the other studies.

## Per figure / panel

The count check covers points where the unwrap Z differs from the ODE-grid Z.
Every such point was re-counted with the ODE at rtol = atol = 1e-10 on the
entire form. A point counts as "root on line" when its count changes after
shifting the line by ±1e-6. At such a point the count is undefined, so neither
back-end can be called wrong there.

| figure / panel | back-end now | grid | old time (Ryzen, ODE) | ODE, i5 | **new time (i5)** | points differing from ODE grid | of which: root on line / ODE wrong / unwrap wrong | stable↔unstable flips |
|---|---|---|---|---|---|---|---|---|
| fig_showcase_hybrid (s01), also the fig_sigma_contours background (s09) | unwrap, n = 4, hmax 0.1 for ω < 6 | 90×70 | 1.19 s | 1.50 s | **18 ms** | 1 | 0 / 1 / 0 | 0 |
| gallery A.1 fourth | unwrap, n = 4 | 75×75 | 152 ms | 176 ms | **11 ms** | 0 | – | 0 |
| gallery A.2 algebraic | **ODE** (see below) | 75×75 | 150 ms | – | 208 ms | (unwrap test: 1) | 0 / 1 / 0 | (ODE wrong at 1) |
| gallery A.3 distributed | unwrap, n = 2 | 75×75 | 85 ms | 100 ms | **7 ms** | 1 | 1 / 0 / 0 | 0 |
| gallery A.4 neutral | unwrap, n = 2, hmax π/2 | 75×75 | 360 ms | 460 ms | **46 ms** | 321 | 75 / 246 / 0 | 35, all on the a = c diagonal (root on line) |
| gallery A.5 high-gain neutral | unwrap, n = 2, hmax π/2 | 75×75 | 380 ms | 525 ms | **45 ms** | 225 | 0 / 225 / 0 | 0 |
| gallery A.6 PDA | unwrap, n = 2, hmax π/2 | 75×75 | 1.30 s | 1.34 s | **140 ms** | 725 | 0 / 725 / 0 | 1 (ODE wrong) |
| gallery A.7 turning | unwrap on cleared form, n = 4, hmax 0.05 for ω < 5 | 75×75 | 2.04 s | 2.41 s | **24 ms** | 20 | 0 / 20 / 0 | 0 |
| gallery A.8 beam | unwrap on cosh γ − K e^{−rλ}, n_eff from denominator, hmax π/21 for ω < 40 | 75×75 | 924 ms | 410 ms | **71 ms** | 133 | 39 / 94 / 0 | 1 (ODE wrong) |
| gallery A.9 FEM bar | unwrap on det(Q0+F), n_eff from det Q0, hmax π/21 for ω < 40 | 75×75 | 51.2 s | 45.2 s | **5.94 s** | 117 | 42 / 75 / 0 | 0 |
| gallery CCC ring | **ODE** (see below) | 75×75 | 6.29 s | – | 7.00 s | (unwrap test: 809 wrong) | – | – |
| gallery A.10 50×50 det | **ODE** (see below) | 40×40 | 163 s | – | 182 s | (unwrap test: 36) | 1 / 11 / 26 | 9 (unwrap) |
| gallery A.11 fractional | unwrap, n = 1.8 | 75×75 | 1.23 s | 1.62 s | **11 ms** | 0 | – | 0 |
| fig_sigma_contours (s09) | same grid as s01 | 90×70 | (= s01) | | **18 ms** | as s01 | | |
| fig_fractional_controller μ = 0.4 (s10) | **ODE** (see below) | 80×60 | not recorded | – | 3.95 s | (unwrap test: 0) | – | 0 |
| fig_fractional_controller μ = 1.5 (s10) | unwrap, n = 2.0 | 80×60 | not recorded | 3.88 s | **73 ms** | 0 | – | 0 |

How to read the count columns:
* **Root on line.** Neutral A.4 on the a = c diagonal: there
  D = (λ²+1)(1+a e^{−λ}), so ±i are roots for every a. The ODE returns the odd
  count 1 there, and the march returns 0, 2 or the invalid marker. Beam and
  FEM bar along K = 1: D(0) = cosh 0 − K = 0. Distributed delay: one point.
* **ODE wrong.** The original ODE grids (tolerance 1e-4) miscount these points.
  Most lie where many roots are unstable:
  * neutral: 244 of 246 have Z > 20, in the essential-instability band |a| > 1;
  * high-gain neutral: 224 of 225 have Z > 20;
  * PDA: 623 of 725 have Z > 20;
  * beam: 29 of 94 have Z > 20;
  * FEM bar: 25 of 75 have Z > 20.

  Classification changes caused by ODE errors: 1 on the beam, 1 on the PDA,
  1 on the algebraic panel (in the unwrap test). On every unwrap panel the
  march matches the 1e-10 reference at **every** point that has no root on the
  line.
* **Step caps.** The march options in the table were chosen against a
  reference march with a step capped at 0.005 (on the 10⁴-window panels, only
  over [0, 100]). Without the chain caps the march miscounts:

  | panel | miscounts without the cap |
  |---|---|
  | turning | 17 |
  | beam | 33 |
  | FEM bar | 37 |

  With the caps it miscounts none, except at points with a root on the line.

Median integer residual of the grids, unwrap vs the old ODE grid:

| panel | unwrap | old ODE grid |
|---|---|---|
| fourth | 2e-10 | 9e-6 |
| turning | 6e-6 | 2e-4 |
| beam | 6e-10 | 2e-4 |
| FEM bar | 7e-10 | 2e-4 |
| distributed | 2.4e-5 | 2.3e-5 |
| frac | 1.6e-5 | 5e-5 |
| neutral | 0.146 | 0.138 |
| high-gain neutral | 0.145 | 0.137 |
| PDA | 0.071 | 0.066 |

On the neutral panels the residual is the tail oscillation, and it is unchanged
apart from the march's lack of integration error.

### Panels that keep the phase-ODE back-end, and why

* **CCC ring (100 vehicles).** The normalization poles have order 198 at
  s = −1. They cannot be cleared, because |D| ~ ω¹⁹⁸ overflows Float64. On this
  D the march loses ±2π (always exactly 2 roots):
  * defaults: 809 of 5625 points;
  * best setting found (tol 0.1, hmax π/0.4 up to ω_max): 22 points.
* **50×50 dense determinant (A.10).** D is entire (n = 100). Its 50 lightly
  damped modes are packed into ω ∈ [0.34, 2.97] and |D| reaches about 1e250.
  * Unwrap differed from the ODE grid at 36 of 1600 points. It was wrong at 26
    of them, and 9 of those were stable points shown as unstable. The ODE grid
    was wrong at 11.
  * No setting tried removed every error (tol down to 0.05, hmax down to 0.02
    over ω < 4). The fine references also disagree with each other, and the
    ODE at 1e-10 aborts at some points.
* **Delayed oscillator (A.2), kept for the colouring only.** The counts are
  fine: they match the ODE grid except at 1 point, where the ODE grid is wrong.
  * The march records a dip estimate only inside the trust region
    |D/D'| ≤ max(|λ|, 1), the package behaviour in `src/unwrap_solver.jl`.
  * Where the dominant roots are real and far left, about 300 of 3627 stable
    pixels end up with no root estimate. They show as a boundary-coloured band
    through the stable domain.
* **Gao et al. μ = 0.4 (s10), kept for the colouring only.** The counts are
  identical to the ODE grid at all 4800 points.
  * The only genuine |D| dip is at the branch point. The other tracking slots
    fill with shallow minima of the e^{−0.4s} ripple, whose march estimates
    reach |σ| ~ 1e3.
  * About 19 stable pixels keep only such estimates, which destroys the
    per-panel colour scale (2 % quantile −748 instead of −1.9). μ = 1.5 is not
    affected.

### Colouring (σ field): what changed, and the measured accuracy

The colour of the stable domain comes from the march's tracked |D| minima, not
from the ODE's. It was checked against Newton-refined dominant roots
(30 tracked roots, ODE at 1e-10, 30 Newton steps). Median |σ error| over the
stable pixels:

| panel | unwrap | old ODE grid |
|---|---|---|
| showcase | 5.5e-4 | 5.1e-3 |
| fourth | 4.3e-3 | 1.0e-2 |
| turning | 2.5e-4 | 2.5e-4 |
| beam | 1.1e-4 | 2.1e-4 |
| distributed | 1.7e-3 | 2.5e-3 |
| neutral | 3.1e-3 | 2.4e-3 |
| high-gain neutral | 9.6e-3 | 8.9e-3 |
| PDA | 3.3e-3 | 2.5e-3 |
| frac | equal | equal |
| Gao μ = 1.5 | 6.7e-3 | 4.1e-3 |

The p99 errors are within about 10 % of the ODE's. The maximum errors are equal
or smaller, except on Gao μ = 1.5: max 0.23 against 0.12.

Three script-side measures make this hold:
1. **Showcase step cap for the colouring.** The showcase uses hmax = 0.1 over
   the resonance band ω < 6. Without it the counts are identical, but at about
   1 % of the stable pixels the dominant root is lost from the σ field.
   * Error up to 0.28, against 0.06 for the ODE.
   * The most stable pixel (S) moves, and the σ-contour levels deepen.
2. **Consistency filter.** A stable point (Z = 0) ignores tracked estimates
   with σ_est > 0.02.
3. **Origin estimate.** The ODE back-end's origin estimate is added
   (`origin_estimate`, systems.jl).

Results of these measures:
* Points S and U of the walkthrough are unchanged, so `fig_method_walkthrough`
  and `showcase_summary.csv` → s03 are unaffected.
* The σ-contour levels of fig_sigma_contours (2 % quantile of the stable σ) move
  from (0, −0.051, …, −0.204) to (0, −0.058, …, −0.231). The new σ field
  matches the refined roots better: 2 % quantile −0.256 truth, −0.256 unwrap,
  −0.227 ODE.

## Regenerated outputs

* figures:
  * `fig_showcase_hybrid.pdf`
  * `fig_sigma_contours.pdf`
  * `fig_gallery_a.pdf`
  * `fig_gallery_b.pdf`
  * `fig_fractional_controller.pdf`

  Panel labels now name the back-end, for example "75×75 unwrap: 11 ms".
* data:
  * `gallery_timings.csv`: new grid times, plus `backend` and
    `ode_grid_time_s_same_machine` columns.
  * `showcase_summary.csv`: plus `grid_backend` and
    `grid_time_ode_s_same_machine`.
  * `sigma_contours.csv`
  * `unwrap_check_s01|s08|s10[_points].csv`: new files.
* Not changed: `fig_method_walkthrough`, `fem_*`, `femcmp_*`,
  `fractional_numbers`, `machine.*`. These were reverted after the rerun
  because only timing noise or line endings differed.

## Text that now needs updating (not edited here)

1. `sections/06_showcase.tex:13-22` (caption of fig:showcase): "Colored
   background: combined field of the brute-force sweep". Name the counting
   back-end, e.g. "discrete phase-unwrapping march (18 ms for the 90×70 grid;
   1.5 s with the phase ODE)". The MDBM curve is still the ODE back-end.
2. `sections/06_showcase.tex:298-303` (caption of fig:sigmacontours): "estimated
   spectral gap σ̂ from the coarse sweep". Optionally name the march. The
   contour levels changed slightly (−0.058 … −0.231).
3. `sections/appendix.tex:14-16`: "every panel is computed at settings chosen for
   speed: a 100×100 background grid and a tolerance of 10⁻⁴". There are three
   problems:
   * The grid is 75×75. This mismatch predates this change.
   * Nine of the twelve panels now use the march (tolerance 0.3 rad, plus step
     caps), not an ODE tolerance.
   * "the count of a few boundary-adjacent pixels may be off by one" now holds
     only for the ODE panels (delayed oscillator, CCC, 50×50 det).
4. `sections/appendix.tex:26-30`: "on every retarded panel the median residual is
   ∼10⁻⁴". On the unwrap panels it is 2e-10 to 2.4e-5 (algebraic, ODE:
   1.4e-4). The neutral ≈0.1 statement still holds (0.07–0.15).
5. `sections/appendix.tex:38-40`: "the shading uses the dominant root (five
   tracked minima, all refined, maximal real part)". The code uses 15 tracked
   minima (5 on FEM, CCC and 50×50 det) and does not refine them (`:Linear`);
   this mismatch predates this change. On unwrap panels the minima come from the
   march; see the colouring section above.
6. `sections/appendix.tex:43-44`: "each panel is annotated with the wall-clock
   cost of both stages". This is still true, but the labels now say which
   back-end made the grid. It is worth one sentence that the MDBM stage still
   runs the phase ODE.
7. `sections/appendix.tex:119-122` (A.7 turning): "D is rational … gives an
   effective order n≈0; no polynomial form is ever constructed". The chart now
   counts the cleared quasi-polynomial M₁M₂ + w(1−e^{−τλ})(M₂ + g₂M₁) with
   n = 4. The march needs an entire D; see 03_method.tex:436-441. The step is
   capped at 0.05 for ω < 5 because of the regenerative root chain.
8. `sections/appendix.tex:133-136` (A.8, eq:beam): the chart now counts the
   numerator cosh(γ) − K e^{−λτ}. The phase of the parameter-independent
   denominator cosh(γ) is measured once and enters as an effective order
   n_eff = 2Φ_den/π ≈ 98.85 at ω_max = 400, which reproduces exactly the
   return-difference count. Two mismatches predate this change: the text has
   c = 0.02, a "+" sign and sech; the code has η = 0.01 and 1 − K e^{−rλ}/cosh γ.
9. `sections/appendix.tex:162-175` (A.9): the chart counts the raw determinant
   det(Q₀+F), via one LU for the determinant and the lemma solve. Its
   effective order comes from the measured phase of det Q₀ (n_eff ≈ 23.88), not
   from a fitted leading order. This removes the 0.4 residual the text blames on
   the raw form; the residual is now 7e-10. The return-difference/lemma
   discussion is still correct as an alternative, but the chart no longer uses
   it.
10. `sections/appendix.tex:204-206` (A.10): "even this case yields charts in tens
    of seconds". It stays on the phase ODE and takes 182 s for 40×40 here (163 s
    on the Ryzen); this was already inaccurate before. Also say why the march is
    not used here.
11. `sections/appendix.tex:214-217` (A.11): "the non-integer effective order
    estimated by eq:npow". The chart now passes n = 1.8. The ODE grid, which
    estimates n, gives identical counts.
12. `sections/appendix.tex:91-93` (A.4–A.5): "the leading-order fit delivers the
    effective order n=2". The panels pass n = 2 explicitly; this mismatch
    predates this change.
13. `sections/03_method.tex:264-267`: "across the retarded case studies … the
    median residual is ∼10⁻⁴". With the march, retarded panels have
    1e-5 to 1e-10 (truncated tail only); see item 4.
14. `sections/03_method.tex:434-448` (Limitations): "With these two rules the
    discrete march reproduces the reference counts on every point of the three
    benchmark charts". This now extends to nine gallery panels plus the
    showcase, apart from points with a root on the line. Worth adding:
    * The denominator-phase variant: n_eff = 2Φ_den/π for a stable,
      parameter-independent denominator such as cosh γ or det Q₀.
    * The two systems where the march is not used: CCC (a 198-fold
      normalization pole that cannot be cleared) and the 50×50 determinant
      (50 modes packed into [0.34, 2.97]).
15. `sections/07_performance.tex:236-240`: "with the ODE march kept for … rational
    D whose denominators cannot be cleared conveniently". The gallery now gives
    concrete examples: CCC, the 50×50 determinant, and two charts kept for their
    colouring.
16. `sections/04_charts.tex:104-109` (workflow): this is consistent. Optionally
    mention the step cap π/(2τ_max) and that a stable denominator's phase can
    serve as the effective order.
