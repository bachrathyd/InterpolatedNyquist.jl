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
The real GPU numbers come from the Colab benchmark (it times all three).

## Projection for the GPU (to be measured on Colab)
Rough cost model: ~300–500 FP32 instructions per evaluation (complex-dual D,
sincos/exp, march bookkeeping), ~40 % of peak issue rate.

| chart | evaluations | T4 | L4 | A100 | RTX PRO 6000 (G4) | this CPU (measured) |
|---|---|---|---|---|---|---|
| showcase 100×100 | 0.5 M | < 1 ms | < 1 ms | < 1 ms | < 1 ms | 11 ms |
| showcase 1920×1080 | 110 M | ~20–40 ms | ~5–10 ms | ~8–15 ms | ~2–4 ms | ~2 s |
| turning 1920×1080 | 270 M | ~50–100 ms | ~15–25 ms | ~20–35 ms | ~5–10 ms | ~5 s |

These are projections (±3×), to be replaced by measurements. If they hold,
**goal 2 (full HD in 0.25 s) is met even on the free T4**, and full HD runs at
interactive frame rates on L4 / RTX PRO 6000. Float64 needs the A100: consumer and
RTX-PRO cards run FP64 at 1/32–1/64 rate. Float32 + flagged Float64 recheck is the
intended mode.

## Roadmap
**Phase 0: measure (now).** Run `colab/NyquistGPU_Colab.ipynb`, first on a T4/L4,
then once on the RTX PRO 6000 (G4). Also run it on your own NVIDIA PC:
`julia --project=gpu/scripts gpu/scripts/bench_ladder.jl` (same scripts, no Colab).
Decisions: does `gpu_check` pass on real hardware? Measured ms vs the projections?
Which schedule wins?

**Phase 1: GPU tuning,** guided by the Phase-0 profile: register pressure (fewer
root slots), workgroup size and lane count, cheaper step control (cbrt/atan),
CUDA fast intrinsics for sin/cos/exp in the tail, an on-device colouring kernel
(no host copy per frame), and batches of constant sets per launch (precomputed
animations / 3-D cubes).

**Phase 2: robustness without tuning.** Derive the band cap automatically from
the delays (`hmax = π/(2τ_max)`). Add spatial and σ-sign consistency checks that
feed `recheck!`. Add MDBM boundary refinement on the GPU: every MDBM level is just
another point list, which the N-d API supports directly.

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
