# Real-time brute-force stability charts: findings and plan

Branch `gpu-cuda` (forked from `gpu-feasibility`), 2026-10-05.

## Goals
1. A chart at the resolution of the paper's examples (~6–10 k points) in **0.01 s**.
2. A full-HD chart (1920×1080 = 2.07 M points) in **~0.25 s**.
3. Eventually a **web page**: change the parameters, the stability chart is recomputed live.

Scope: scalar characteristic equations `D(λ, p, c)` written directly (no
model extraction), NVIDIA only, any parameter-point list (N-d, non-grid).

## What is on the branch
* `gpu/` — the **NyquistGPU** package (KernelAbstractions; one kernel source for
  CUDA and for the CPU backend, so everything is testable on any machine).
  Key call: `plan_sweep(points; backend, T, …)` → `run!(plan, D, c)` →
  `fetch_result(plan)`; `chart(...)` is the 2-D special case, `recheck!` re-runs
  flagged points in Float64. The main package is untouched.
* `gpu/test` — unit tests against **analytic** stability results (Hayes
  equation region, Float32/64, both methods; a 3-D point sweep; scattered
  point clouds with known roots; schedule equivalence; boundary flags).
* `gpu/validate` — cross-validation against `InterpolatedNyquist.jl`
  (100×100 charts of the three paper systems; near-boundary stress test).
* `gpu/scripts` — CUDA environment: `gpu_check.jl` (GPU == CPU backend),
  `bench_ladder.jl` (100² … 1920×1080 × precision × schedule → CSV + fields),
  `example_custom.jl` (template for your own model).
* `gpu/colab/NyquistGPU_Colab.ipynb` — runs all of it on a Colab GPU and
  writes the results into a Google-Drive folder.

## Finding 1: the algorithm matters more than the GPU
The phase `Φ(ω) = arg D(σ+iω)` is an exact differential: the count needs only its
**end-point value**, unwrapped without losing a branch. Nothing has to be
*integrated*. The new `:unwrap` march takes one evaluation per step (D and dD/dω
from one dual-number call). It accepts a step when the observed increment
`angle(D_b/D_a)` agrees with the trapezoid prediction `h(θ'_a+θ'_b)/2` from the
exact derivatives. A skipped near-root transition shows up as a ≈π mismatch and
is refined. The slowly decaying delay ripple, which forces an integrator into
thousands of steps, becomes invisible once its phase amplitude is below the
tolerance. Root tracking: minima of |D| between accepted samples are located on a
cubic Hermite model of D (no extra evaluations), followed by one Newton step, a
trust region, and depth-ranked slots, as in the package.

Measured on this PC (i5-10400, 6 cores/12 threads, CPU backend, Float32), 100×100 charts:

| system | package Vern9 (default) | `:unwrap` Float32 | speed-up (same CPU) | evaluations / point (unwrap vs integrator) | wrong counts vs package (tol 1e-7) |
|---|---|---|---|---|---|
| 4th-order delayed oscillator | 37.0 µs/pt | **0.6 µs/pt** | ~62× | 31 vs 658 | 0 / 10 000 (F32 and F64) |
| showcase 2-DOF DAE | 277.6 µs/pt | **1.1 µs/pt** | ~250× | 53 vs 12 370 | 0 / 10 000 (F32 and F64) |
| two-mode turning | 1037.7 µs/pt | **2.6 µs/pt** | ~400× | 129 vs 6 580 | 0 / 10 000 (F32 and F64) |

(µs/pt = wall time / points, all 12 threads; `results/validation_cpu_100.log`.
Float64 costs only 15–40 % more on the CPU.)

**Goal 1 is already met on the CPU**: the showcase 100×100 chart takes 11 ms on this 6-core desktop, so the paper's 90×70 takes about 7 ms.

**Near-boundary robustness** (`validate/boundary_stress.jl`, showcase Hopf
boundary approached to ±10⁻¹²): `:unwrap` in Float64 counts correctly at every
offset with 72–165 evaluations. The package (Vern9 at 1e-8) miscounts from
10⁻⁸ onward, and `:bs3` from 10⁻¹⁰. In Float32 `:unwrap` is exact down to
~10⁻⁶. Below that the transition is narrower than the smallest Float32 step, so
the branch is decided by the side of the Newton root estimate and the point is
**flagged**. The only Float32 errors observed were on flagged points, so
`recheck!` (Float64, only those points) makes a Float32 GPU sweep safe.

### Caveats found (and handled)
* **Rational D**: a lightly damped mode is a pole next to the axis. When a
  chatter root leaves that mode, the pole–zero pair winds the phase by −2π inside
  one step, which no end-point check can see (turning: 137 wrong of 3600 before
  the fix). **Multiply out denominators** (stable poles do not change the count),
  as done in `scripts/systems.jl`.
* **Root chains near the axis** (the regenerative delay in turning puts roots
  ≈ Ω apart): two unstable chain roots inside one step slip by −2π. Fix: band cap
  `hmax ≈ π/(2τ_max)` for `ω < ωband` (turning: `hmax = 0.05, ωband = 5` → 0 wrong,
  130 evaluations/point). Automating this from the delays is on the roadmap.
* **Julia specialization trap** (fixed): a `::Function` argument that is only
  passed on is not specialized, so on the CPU backend every D call was a dynamic
  dispatch (3.7× slower). D is now wrapped in a callable struct.

## Finding 2: load balancing (your "refill" idea)
Adaptive marches take 30…400 steps per point. With one thread per point, a warp
(32 lanes) runs as long as its slowest lane. The `:queue` schedule keeps a fixed
pool of persistent lanes. The kernel loop body is **one march step**, and a lane
that finishes immediately pulls the next point from an atomic counter, so all lanes
keep executing the same step instructions. Idealized SIMT efficiency from the
measured step counts (`simulate_schedule`):

| system | one thread/point (row-major) | random order | `:strided` | `:queue` |
|---|---|---|---|---|
| 4th-order | 87 % | 69 % | 88 % | **93 %** |
| showcase | 83 % | 66 % | 84 % | **92 %** |
| turning | 73 % | 54 % | 88 % | **89 %** |

Random assignment *alone* hurts, because neighbouring pixels have similar work.
Random assignment *with in-loop refill* (`:strided`) and the queue both recover it.
**On the real T4, however, the simple one-thread-per-point `:pixel` schedule was
the fastest** (the queue 1.3–1.6×, strided ~2× slower): the persistent-lane loop
carries the larger state through every iteration (more registers, less
occupancy), and the 64-bit index arithmetic of the scramble is expensive on a
GPU. The idealized SIMT gain (≤ 15 %) is smaller than these costs -- the default
is now `:pixel` on the GPU in the scripts; the lane schedules stay for experiments.

## Measured on a Colab T4 (2026-10-05)
`gpu_check.jl`: the CUDA kernels reproduce the CPU backend on **every** unflagged
point (3 systems × Float32/64 × 3 schedules, 128×128). Full-HD charts (1920×1080 =
2.07 M points), Float32, one thread per point (`:pixel`, the fastest schedule on
the GPU; the persistent-lane schedules were 1.3–2× slower here):

| chart | T4 Float32 | T4 Float64 | this CPU (i5, 12 thr) |
|---|---|---|---|
| 4th-order 1920×1080 | **30 ms** (69 Mpts/s) | 203 ms | 0.86 s |
| showcase 1920×1080 | 95–107 ms | 450–585 ms | ~2 s |
| turning 1920×1080 | 213–245 ms | 1.3–1.5 s | 4.2 s |
| showcase 100×100 (paper resolution) | **1.1 ms** | 11 ms | 11 ms |

**Both goals are met on the cheapest GPU**: the paper-resolution chart in ~1 ms
(goal: 10 ms) and full HD in 30–245 ms (goal: 250 ms). The 4th-order number
needed one fix found on the GPU: Julia's generic `λ^4` (a non-inlined
power-by-squaring loop) cost 4.6×; literal powers of our dual numbers are now
unrolled. Kernel diagnostics (`kernel_info.jl`): no Float64 instructions in the
Float32 kernels, 83–119 registers/thread (55–75 % occupancy), 230–310 B of
local memory; 2, 4 or 8 root slots cost the same.

## Which GPU?
The kernel is scalar FP32 arithmetic (no tensor cores, almost no memory traffic),
so what counts is FP32 CUDA-core throughput (cores × clock) and the SM count.

| GPU (Colab) | FP32 vs T4 | FP64 | ≈ CU/h | predicted 4th-order full HD (T4: 30 ms) |
|---|---|---|---|---|
| T4 (free tier) | 1× | 1/32 rate | ~1.2 | 30 ms (measured) |
| L4 | ~3.7× | 1/64 | ~1.7 | ~8–10 ms -- **best value** |
| A100 40/80 GB | ~2.4× | **1/2 rate** | ~5.4–7.5 | ~12–15 ms; the Float64 card |
| H100 (if offered) | ~6–8× | 1/2 rate | ? | ~4–6 ms |
| RTX PRO 6000 Blackwell (G4) | **~15×** | 1/64 | ~8.7 | **~2–3 ms -- fastest for Float32** |

Predictions scale the measured T4 numbers by FP32 throughput (±2×). Float64 on
the GPU is only needed for re-checking flagged points, which are few, so the
consumer/workstation FP64 rate does not matter. Your own 5-year-old NVIDIA card
(an RTX 30-series has ~13–35 TFLOPS FP32, 2–4× a T4) runs the same scripts.

## Lower precision: Float16 / Float8
Measured on the T4 (`precision_test.jl`, 4th-order full HD): Float16 runs in
**15.4 ms vs 33.5 ms** for Float32 (2.2×); 236 counts wrong (0.011 %), **all of
them flagged**, 41 239 points (2 %) flagged in total -> Float64 re-check ≈ 3 ms,
net gain ≈ 1.8×. But Float16 ends at 65504: λ⁴ overflows above ω = 16, so the
march must stop at ω_max ≈ 15 -- model-specific, not a general setting (a
generic version would need D rescaled by λⁿ inside the user's expression). On
L4 / RTX-PRO (Ada/Blackwell) non-tensor FP16 runs at the FP32 rate, so even this
gain disappears there; A100/H100 keep 2–4×. **Float8 does not exist as scalar
arithmetic on any GPU** (only inside tensor-core matrix multiplies). Verdict:
no 10× from precision. Side finding: the march to ω_max = 15 is enough for this
model and still exact in Float32 (17 instead of 31 evaluations) -- a
model-aware ω_max is a free knob.

## MDBM: adaptive refinement with your Multi-Dimensional Bisection Method
MDBM.jl already evaluates each stage (initial grid, every refinement, every
neighbour check) as ONE batch of new points (`MemF(::Vector)`). Branch
`vectorized-eval` of MDBM.jl adds `MDBM_Problem(f, axes; vectorized = fv)`: the
batch goes to `fv(points)` in a single call -- one NyquistGPU sweep, one GPU
launch, only the new points and their values cross the bus (a few MB). Also no
threads inside MDBM, so the data race seen with threaded scalar `f` cannot
occur (that race needs a non-thread-safe `f`; MDBM's own threaded loop writes
disjoint slots). `gpu/mdbm/mdbm_chart.jl`, 4th-order model, objective
g = (Z == 0 ? 1 : -1)·|σ_dom|, bracketing by a true sign change
(`interpolationorder = 0`, then `interpolate!` order 1 -- order 1 during
`solve!` also keeps cubes where g only touches zero, which gave spurious curves):

| | evaluated points | stability boundary vs brute force | wall time (CPU) |
|---|---|---|---|
| brute force 1921×1081 | 2 076 601 | -- | 1.17 s |
| MDBM 241×136 + 3 levels | **40 555 (2.0 %)** | **0 px off, 100 % covered** | 1.31 s |

51× fewer evaluations and an exact boundary; the wall time is now dominated by
MDBM's host-side bookkeeping (~0.9 s for ~2 000 cubes / 40 k points), so on the
GPU brute force (30 ms) is still faster for a cheap 2-D model. MDBM pays off
where an evaluation is expensive (big determinants, FEM), in 3-D+ parameter
spaces (boundary ~ n², volume ~ n³), at very high resolution -- and everywhere
once its host side is profiled and optimized (a natural next step in MDBM.jl).
Count boundaries *inside* the unstable region keep the background resolution
(one more objective per count level would refine them too).

## Roadmap
**Phase 0: measure -- done on the T4.** Next: one run on the RTX PRO 6000 (G4)
and on your own NVIDIA PC (`julia --project=gpu/scripts gpu/scripts/bench_ladder.jl`)
to replace the predictions above.

**Phase 1: GPU tuning,** guided by the Phase-0 profile: register pressure (fewer
root slots), workgroup size and lane count, cheaper step control (cbrt/atan),
CUDA fast intrinsics for sin/cos/exp in the tail, an on-device colouring kernel
(no host copy per frame), and batches of constant sets per launch (precomputed
animations / 3-D cubes).

**Phase 2: robustness without tuning.** Derive the band cap automatically from
the delays (`hmax = π/(2τ_max)`). Add spatial and σ-sign consistency checks that
feed `recheck!`. MDBM on the GPU: merge the `vectorized` hook into MDBM.jl, profile
its host-side bookkeeping, add count-level objectives for full colour maps.

**Phase 3: back into the package.** Offer `:unwrap` as a CPU back-end of
`InterpolatedNyquist.jl` (it is 60–400× faster on the CPU alone, and more robust
near boundaries, a strong result for the paper). Expose the GPU path as a package
extension (`calculate_unstable_roots_p_vec(...; backend = CUDABackend())`) or keep
NyquistGPU separate.

**Phase 4: real-time web page.** Port the `:unwrap` kernel to a **WebGPU** compute
shader (WGSL, Float32, which is validated above). It runs on the *visitor's* GPU
(NVIDIA, AMD, Intel, including this PC's UHD 630): no server, no Colab, nothing
to pay per user. The characteristic equation is typed as a formula, compiled in
the browser to WGSL complex-dual arithmetic, with sliders for the constants and
progressive refinement while dragging. It could be hosted on the repo's GitHub
Pages. Risk: WGSL sin/cos/exp precision is implementation-defined and poor for
large arguments, so the shader needs its own range reduction; the count
tolerance (0.3 rad) is forgiving. A Julia GPU server streaming images is the
fallback, with worse latency and a running cost.

## Colab and your Google account
* Since 22 Sep 2026 an eligible **Google AI plan (AI Pro included) unlocks premium Colab**: a monthly allotment of compute units and faster GPUs (reportedly ~200 CU/month for AI Pro; check under *Colab → Resources*). Claim it from the "Colab paid products" link in a notebook. **No separate Colab subscription is needed** for this project.
* Indicative burn rates (third-party measurements): T4 ~1.2 CU/h, L4 ~1.7, A100 40 GB ~5.4, RTX PRO 6000 ~8.7. One full notebook pass (~20 min) costs well under 1 CU on T4/L4. **Always end with *Runtime → Disconnect and delete runtime*.**
* The subscription benefits belong to the Google account that holds AI Pro. Run Colab under that account. If the Drive folder lives in another account (bachrathyd2), share it with that account and add a shortcut in *My Drive*; the notebook's `DRIVE_FOLDER` then points to the shortcut name.
* Colab's native Julia runtime cannot mount Drive conveniently, so the notebook uses the Python runtime and drives Julia through shell commands (Julia via juliaup).
