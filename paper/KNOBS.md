# Every figure's tunable knobs, in one place

Current values as of this commit. To change a figure, edit the file/line shown,
delete its cache (or pass `--force`), and rerun only that study:

```powershell
cd paper\scripts
julia --project=. -t auto generate_all.jl --only=s08_gallery          # one study
julia --project=. -t auto generate_all.jl --only=s08_gallery --force  # ignore caches
```

**Cache trap:** most knobs are in the cache key, so editing them recomputes
automatically. A few are NOT (noted below); for those pass `--force` or delete
`paper/data/cache/<study>_*.jls` or you will re-plot a stale grid. `--fast`
halves every resolution for a quick look.

Two shared ranges live in `studies/systems.jl:57-58`:
`SHOWCASE_PRANGE = (0.0, 4.0)`, `SHOWCASE_DRANGE = (-0.5, 3.0)` — used by
s01, s02, s04, s06, s09 (the showcase DDAE).

---

## The gallery (s08) — one line per panel

All 12 panels are defined in **`studies/s08_gallery.jl:250-308`**, the `SPECS`
array. Each panel is ONE line you can edit directly. Fields:

- `xr`, `yr` — the two axis RANGES, e.g. `xr = (-2.0, 4.0)`
- `nx`, `ny` — grid RESOLUTION per axis (all `half(NBF)` = 100, or a literal)
- `ω` — omega_max for that panel
- `tol` — reltol/abstol for that panel
- `npow` — leading order if fixed (else omitted → estimated)

| panel (id) | x-range | y-range | grid | ω_max | tol |
|---|---|---|---|---|---|
| fourth | (−2, 4) | (−2, 5) | 100×100 | 1e4 | 1e‑4 |
| algebraic | (−1, 1) | (−1, 1) | 100×100 | 1e4 | 1e‑4 |
| distributed | (0, 2) | (−1, 5) | 100×100 | 1e4 | 1e‑4 |
| neutral | (−0.9, 0.9) | (−2, 2) | 100×100 | 200 | 1e‑4 |
| neutral_hg | (−0.95, 0.95) | (−10, 10) | 100×100 | 200 | 1e‑4 |
| pda | (−1.5, 2.5) | (−1.6, 1.6) | 100×100 | 500 | 1e‑4 |
| turning | (0.08, 1.2) | (0.01, 1.2) | 100×100 | 1e4 | 1e‑4 |
| beam (Zhang–Stépán Fig 8) | (0.02, 10.5) | (−0.75, 1.0) | 100×100 | 400 | 1e‑4 |
| fem (12‑DoF bar) | (0.02, 10.5) | (−0.75, 1.0) | 100×100 | 400 | 1e‑4 |
| ccc (Ge–Orosz Fig 4d) | (−10, 10) | (0.05, 16) | 100×100 | 1e4 | 1e‑5 |
| bigmat (50×50) | (−0.95, 1.0) | (0.05, 1.5) | 40×40 | 200 | 1e‑4 |
| frac | (0, 5) | (0.1, 2) | 100×100 | 1e4 | 1e‑4 |

Gallery-wide knobs, **`studies/s08_gallery.jl:245-248`**:
- `NBF = 100` — the background grid every `half(NBF)` panel uses. Change once, all panels follow.
- `MDBM_N0_G = 7` — MDBM initial mesh per axis.
- `mdbm_levels(nx)` — refinement depth (auto: ~4× the grid). Edit the formula to force a depth.
- `half(n)` — the `--fast` halving rule.

Model constants (change the physics, not the view):
- beam/fem damping `BEAM_ETA = 0.01` — line 63. FEM element count `build_fem(12)` — line 101.
- ring `CCC_N = 100` vehicles — line 172; reaction delay `CCC_SIG = 0.2` — line 174.

---

## The other figures

| figure | file | resolution | ranges | ω_max | tol |
|---|---|---|---|---|---|
| **fig_showcase_hybrid** + **fig_method_walkthrough** | `studies/s01_showcase.jl` | `nP,nD` line 22 = **90×70** | showcase (systems.jl) | `1e4` lines 27, 78‑79 | default (1e‑5) |
| **fig_tolerance_charts** + **fig_error_field** | `studies/s02_tolerance_error.jl` | `nP,nD` line 10 = **100×80** | showcase | `WMAX_TOL` line 39 = **1e4** | `TOLS` line 19 = **[1e‑2,1e‑4,1e‑6,1e‑8]** |
| **fig_convergence** (+ tables) | `studies/s03_convergence.jl` | single point `p_conv` | — | `WMAX_CONV` line 26 = **1e6** | `TOLS` line 52 = **1e‑2…1e‑10** |
| **fig_refinement** | `studies/s04_refinement.jl` | `nP,nD` line 12 = **60×40** | showcase | `WMAX_REF` line 15 = **1e4** | default |
| **fig_solver_zoo** | `studies/s05_solver_zoo.jl` | `nZ` line 17 = **40×40** | (−2,4)×(−2,5) | `ZOO_WMAX` line 10 = **1e4** | `TOLS_ZOO` line 77 = **1e‑2…1e‑11**; time cap `ZOO_TIME_CAP` line 15 = **1 s** |
| **tab_grid_timings / tab_mdbm / tab_presets / tab_extraction** | `studies/s06_grid_timings.jl` | `NGRID` line 15 = **100** | `CASES` lines 27‑29 | `WMAX_CHART` line 25 = **1e4** | presets line 122‑123: accurate **1e‑5**, fast **1e‑3**. MDBM `MDBM_N0=20`,`MDBM_IT=4` lines 16‑17 |
| **fig_sigma_contours** | `studies/s09_sigma_contours.jl` | `nP,nD` line 7 = **90×70** | showcase | `1e4` lines 12, 24 | default; contour levels line 18 |
| **fig_fractional_controller** | `studies/s10_fractional_controller.jl` | `NXY` line 127 = **80×60** | `CASES_G` lines 21‑26 (kp, ki per μ) | `1e4` throughout | default; only σ=0 (branch cut — see appendix) |
| **tab_evals / wmax_scaling** | `studies/s11_diagnostics.jl` | — | — | ω_max ladder 1e6/1e8/1e10 | — |
| **peak_repair.csv + repair macros** | `studies/s12_peak_repair.jl` | dP ladder `DP_LADDER` = ±1e‑2…1e‑10 around the Hopf point at D=1.5 | — | default (1e6) | default; hard sanity gates assert every claim §6.3 quotes |

Notes:
- "showcase" range = `SHOWCASE_PRANGE × SHOWCASE_DRANGE` from systems.jl.
- "default tol" = the package default (reltol=abstol=1e‑5), i.e. the study
  doesn't pass one.
- s03 uses `WMAX_CONV = 1e6` on purpose — it isolates the integrator, so it
  wants the truncation error far below the tightest tolerance. Everything that
  produces a *chart* uses `1e4`.

---

## The colour scale (all charts share it)

`scripts/common.jl`:
- `BILINEAR_CMAP` and its four corner colours `COL_STABLE_FAR/ZERO`,
  `COL_UNSTAB_ZERO/FAR` — lines ~90‑115. `COL_UNSTAB_FAR` is the deep maroon;
  change it to shift the "badly unstable" end.
- `bilinear_metric(...; σ_quantile=0.02)` — the stable-side clip. Lower the
  quantile to use more of the green range on outlier-heavy charts.
- `annotate_panel!` / `pick_annotation_corner` — the on-chart text plate.

---

## Fastest way to iterate on ONE gallery panel

Point the driver at s08 and edit only that panel's line. If you also want to
skip the expensive panels while tuning a cheap one, temporarily comment the
others out of `SPECS`, or set their `nx,ny` to `half(20)`. The per-panel cache
key includes `xr, yr, ω, tol` and the MDBM depth, so any of those changes
recomputes automatically; changing `D` or `nx/ny` alone also recomputes
(they're in the key too).
