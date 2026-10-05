# NyquistGPU: brute-force stability sweeps on NVIDIA GPUs

A small, separate package inside this repository (`gpu/`, branch `gpu-cuda`). The
CPU package `InterpolatedNyquist.jl` is unchanged. NyquistGPU takes **one scalar
characteristic equation** and an **arbitrary list of parameter points** (any
dimension; a 2-D chart is the special case `grid_points(xs, ys)`). For every
point it returns the number of unstable roots and an estimate of the rightmost
root, running the same KernelAbstractions kernels on a CUDA GPU or on the CPU.

```julia
using NyquistGPU, CUDA

D(λ, p, c) = c[1] * λ^4 + λ^2 + 2 * c[2] * λ + 1 + (p[1] + p[2] * λ) * exp(-c[3] * λ)
c = (0.03, 0.02, 0.5)                                   # constants ("sliders")

pts = grid_points(range(-2, 4; length = 1920), range(-2, 5; length = 1080))
plan = plan_sweep(pts; backend = CUDABackend(), T = Float32, n_power = 4)
run!(plan, D, c)                # repeat with new c: no re-upload, no recompile
res = fetch_result(plan)        # Z, Zraw, sigma (rightmost root), omega, steps, evals, flags
recheck!(res, D, pts; c = c, n_power = 4)              # flagged points again in Float64
Z = reshape(res.Z, 1920, 1080)

r = chart(D, (-2, 4), (-2, 5), 400, 300; c = c, n_power = 4)   # 2-D shortcut -> matrices
```

Rules for `D(λ, p, c)`:
- `λ` is complex, `p` is one parameter point (a tuple of any length), `c` is a tuple of constants.
- Write `D` as an **entire** function. A rational D puts a pole next to every lightly damped mode, and a pole–zero pair can wind the phase by −2π inside one step. Multiply out the denominators; stable poles do not change the count.
- Take constants from `c` or use integer literals. A Float64 literal such as `0.5λ` silently promotes a Float32 kernel to Float64, which is about 32–64× slower on GeForce and RTX-PRO cards. `check_eltype(D, p, c, Float32)` reports it.
- `n_power` is the leading order of D (D ~ λⁿ as |λ| → ∞).

## Methods and schedules
| option | meaning |
|---|---|
| `method = :unwrap` (default) | unwraps `arg D(σ+iω)` directly: 1 evaluation per step, step control by the consistency of the observed phase increment with the exact derivatives. **30–130 evaluations per point** on the paper's systems (the integrator needs 650–12 000). |
| `method = :bs3` | the paper's phase ODE with a Bogacki–Shampine 3(2) pair (validated port of the CPU method) |
| `schedule = :pixel` (GPU default) | one thread per point -- fastest on the T4 |
| `schedule = :queue` (CPU default) | persistent threads + atomic work queue; a lane that finishes a point takes the next one inside the same step loop |
| `schedule = :strided` | persistent threads, static coprime-stride scramble of the points (statistical balancing, no atomics) |
| `hmax`, `ωband` | step cap h ≤ hmax for ω < ωband. Needed when chains of roots run close to the axis (e.g. regenerative delay in turning: `hmax ≈ π/(2τ_max)`) |
| `flags` (output) | bit 1: march failed; bit 2: a root closer to the line than the precision resolves was counted by its side (boundary-grazing point); bit 3: integer residual > 0.25 (a root on the line) |

## Layout
```
gpu/
  Project.toml, src/NyquistGPU.jl   the engine (deps: KernelAbstractions, ForwardDiff, Atomix)
  test/runtests.jl                  unit tests (analytic Hayes region, 3-D point lists, flags, schedules)
  scripts/                          GPU environment (CUDA.jl): gpu_check.jl, bench_ladder.jl, maps.jl,
                                    precision_test.jl, kernel_info.jl, example_custom.jl (template),
                                    systems.jl (paper systems)
  validate/                         CPU cross-validation against InterpolatedNyquist.jl:
                                    validate_vs_package.jl, boundary_stress.jl
  mdbm/                             MDBM.jl with GPU batches (needs MDBM branch vectorized-eval):
                                    mdbm_chart.jl -- boundary by bisection vs brute force
  colab/                            NyquistGPU_Colab.ipynb (+ make_notebook.py), plot_fields.py
  results/                          logs/CSVs of the runs
  PLAN.md                           findings, projections, roadmap
```

## Running
```bash
julia --project=gpu -t auto -e "using Pkg; Pkg.test()"
julia --project=gpu/validate -t auto gpu/validate/validate_vs_package.jl 100
julia --project=gpu/validate -t auto gpu/validate/boundary_stress.jl
julia --project=gpu/scripts gpu/scripts/gpu_check.jl
julia --project=gpu/scripts gpu/scripts/bench_ladder.jl --out <dir>
```
The first `julia --project=...` in each environment instantiates it. Run `using Pkg; Pkg.instantiate()` there first if needed. Without a functional CUDA GPU, the scripts fall back to the CPU backend (`--cpu` forces it). On Google Colab, use `colab/NyquistGPU_Colab.ipynb`.
