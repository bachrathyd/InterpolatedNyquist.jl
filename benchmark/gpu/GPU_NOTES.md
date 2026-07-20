# GPU feasibility for dense brute-force stability charts

**Question.** Would a GPU pay off for dense BF charts, and do we need a custom
GPU kernel? Tested 2026-07-18 on a machine *without* a CUDA driver — everything
below except the final speedup number is measurable anyway, because the kernel
is written in KernelAbstractions.jl and runs bit-identical logic on the CPU
backend today and on `CUDABackend()` unchanged whenever a GPU is present.

Experiment: `gpu_feasibility.jl` — a one-thread-per-pixel adaptive phase march
(Bogacki–Shampine 3(2), FSAL, dual-number-exact integrand, running min-|D|²
with one-Newton-step σ estimate), 100×100 chart of the 4th-order benchmark,
ω_max = 10⁴, tol 10⁻⁵, validated against the package Vern9 sweep.

## Measured answers

1. **The algorithm fits a GPU kernel — yes, and it needs to be a *kernel*.**
   The whole march is isbits scalar state, no allocations, no dynamic
   dispatch; the KA kernel compiles and reproduces the package counts
   **10000/10000** (median integer residual 2.2·10⁻⁵ = the ω_max truncation
   floor, i.e. integration error is subdominant). Because the phase-ODE RHS is
   independent of the state, each RK stage is just an integrand evaluation at
   a shifted frequency — the kernel is a per-thread *adaptive quadrature*,
   simpler than a general ODE ensemble. Array-style lockstep vectorization
   (EnsembleGPUArray-style) would be the wrong shape: pixels take different
   step sequences. One thread = one pixel (the DiffEqGPU `EnsembleGPUKernel`
   pattern) is the right design, and the custom kernel here is ~120 lines.

2. **Warp divergence is mild, thanks to spatial correlation of the work.**
   Steps per pixel: median 218, p90 278, max 357. With warps of 32 consecutive
   row-major pixels the SIMT efficiency is **86 %** (random order, the
   worst case: 71 %). Adaptivity does *not* wreck GPU utilization on charts,
   because neighbouring pixels need similar work except across boundaries.

3. **Float32 suffices on this system — the consumer-GPU question.**
   The FP32 kernel matches the FP64 counts **10000/10000**; the residual is
   unchanged in the median (2.2·10⁻⁵, truncation-dominated) and the max grows
   only to 1.4·10⁻³ — still 350× below the rounding threshold. So a
   GeForce-class card (FP64 crippled to 1/32–1/64 rate) can run the chart in
   FP32. Caveat: FP32 makes the *near-boundary peak-skipping* onset ~10⁸×
   earlier (peak half-widths below ~10⁻⁷ are unresolvable), so boundary-
   grazing pixels should be re-certified on the CPU — see the hybrid below.

4. **Throughput baseline.** On the same 16-thread CPU the KA kernel does
   41 μs/pt vs the package Vern9 sweep's 53 μs/pt (kernel has zero solver
   overhead; BS3 needs more steps than Vern9 but each is 3 evaluations).

## Verdict

- **Worth it only for genuinely dense work**: at ~40–50 μs/pt the CPU already
  gives a 100×100 chart in ~0.05–0.5 s — GPU is pointless there. The cases
  that justify it: 1000×1000+ charts (~40 s CPU wall on this machine), 3-D
  parameter cubes, or charts inside optimization loops. Expected gain on a
  consumer card in FP32: one to two orders of magnitude (compute-bound,
  86 % warp efficiency, ~650 evaluations/pixel of pure scalar arithmetic);
  on a data-center card the same holds in FP64.
- **Recommended architecture** (fits the paper's philosophy): GPU kernel
  sweeps all pixels cheaply; the *self-validating integer residual* — which
  the kernel computes for free — flags the pixels the GPU cannot certify
  (non-integer Z_raw, near-boundary, FP32-skipped peaks); the CPU re-solves
  only those flagged pixels with the full Vern9 + root-tracking machinery.
  The residual makes the hybrid *safe*, which is exactly the property the
  paper argues for.
- **Limits**: only scalar/closed-form D(λ) fits a per-thread kernel well
  (matrix-valued D would need batched LU — a different, much bigger project);
  the σ estimate in the kernel is the single-closest-minimum variant, which
  is fine for near-boundary use but not for the dominant-root coloring
  (65 % naive sign agreement here, the known deep-domain caveat — the
  multi-minimum buffer would port, but wasn't needed for feasibility).

## Round 2: two-system validation + real-time prototype (user request)

`two_system_validation.jl` — the requested 100×100 BF validation on the two
paper systems, kernel vs package Vern9 at the same tol 10⁻⁵, ω_max = 10⁴:

| system | package (16 threads) | kernel FP64 | kernel FP32 | wrong counts | warp-32 eff. |
|---|---|---|---|---|---|
| showcase 2-DOF DAE (ω⁻¹ ripple) | 181 µs/pt | 254 µs/pt | 290 µs/pt | **0/10000 (both)** | 83 % / 50 % random |
| turning two-mode lobes (rational) | 1825 µs/pt | 97 µs/pt | 97 µs/pt | 3/10000 (FP64), 4 (FP32) | 54 % / 40 % random |

- The showcase — the hard case — validates perfectly in FP32 as well; the
  kernel is slower than Vern9 per CPU thread there (BS3 needs ~4100 steps/px
  where 9th order strides), which is precisely the gap thousands of GPU
  threads close.
- The turning disagreements are boundary-adjacent skipped peaks of the
  low-order pair at this tolerance (the class §6.3 of the paper analyses;
  the residual/σ cross-check flags them for CPU re-solve). Warp efficiency
  is lower here because the dense lobe structure decorrelates neighbours.

`realtime_chart.jl` — the "shader-style" live-chart prototype:
sliders (c₁, ζ, τ of the 4th-order oscillator) → KA kernel sweep → heatmap
of the paper's interpolable coloring, with progressive refinement (60×60
while dragging, 140×140 half a second after the sliders rest). Measured
frame rates, CPU backend, 16 threads, FP32, tol 10⁻⁴:

    60×60: 35 ms/frame (28 fps) · 100×100: 47 ms (21 fps) · 140×140: 86 ms (12 fps)

So for ripple-light systems the dream already runs in real time on the CPU;
the GPU backend is what extends it to the showcase/turning class (~10-40×
more work per pixel) and to higher resolutions. The GUI part needs GLMakie;
`--bench` runs headless.

**Real-time architecture notes** for the eventual tool: (i) Julia's JIT is
the enabler for *arbitrary user-supplied parametric D(λ)* — a new model
specializes the kernel once (seconds), then every slider move is a pure
kernel launch, which is exactly the shader experience without transpiling
the model to GLSL; (ii) sliders must map to *non-swept* parameters (the two
chart axes stay fixed per view); (iii) progressive refinement + the free
integer residual as a per-pixel confidence overlay make the fast preset
safe to show live.

## Round 3: the GPU chart version (400×400 real-time target)

`realtime_chart.jl` is now the full GPU implementation:

- **Backend auto-detection**: `CUDA.functional()` → `CUDABackend()`, otherwise
  the multithreaded CPU backend with a clear message. CUDA.jl is in this
  environment and verified to load gracefully on this driverless machine
  (`functional = false` → CPU fallback) — on a machine with an NVIDIA driver
  the same command runs the sweep on the GPU with no code change.
- **400×400 with persistent device buffers** (allocated once per resolution,
  reused every frame; the device→host copy is 640 kB/frame — negligible).
- **Progressive refinement on CPU**: the app measures one full-res frame at
  startup and sizes the drag resolution to ~80 ms (here: 150×150), refining
  to the full 400×400 half a second after the sliders rest. On GPU it
  simply redraws the full resolution live.
- The GUI smoke test passes on this machine (window opens, renders, and
  `screenshot_realtime.png` shows the correct chart: green stable island
  with the smooth in-kernel σ-gradient, integer plateaus outside, the
  divergence line at P = −1 where D(0) = 1 + P vanishes).

Measured frame times (FP32, tol 10⁻⁴, 4th-order oscillator):

| resolution | CPU 16 threads | GPU (projected, 20–100×) |
|---|---|---|
| 100×100 | 46 ms (22 fps) | ≲ 2 ms |
| 200×200 | 157 ms (6 fps) | 2–8 ms |
| **400×400** | **509 ms (2 fps)** | **5–25 ms (40–200 fps)** |

The 400×400 target is precisely the regime that needs the GPU: real-time on
CPU only up to ~120×120, projected comfortably real-time on any mid-range
CUDA card (the projection uses the measured 83–86 % warp efficiency and the
FP32 validation of rounds 1–2). **Final validation step on a GPU machine**:
clone, `julia --project=benchmark/gpu -t auto benchmark/gpu/realtime_chart.jl --bench`
— the printed backend line must say GPU, and the 400×400 row gives the true
frame time.

## Round 4: running on THIS machine's GPU (AMD iGPU via OpenCL) — blocked by the driver

The machine has no NVIDIA card but does have an **AMD Radeon iGPU (gfx90c,
Vega, Ryzen APU)** exposed through OpenCL with FP64. OpenCL.jl v0.10 provides
a KernelAbstractions backend for it, translating Julia kernels to SPIR-V and
— because AMD's Windows driver cannot ingest SPIR-V — onward to OpenCL C via
spirv2clc. `opencl_igpu_test.jl` attempts the validated march kernel there.
Findings, in bisection order:

1. **The toolchain and device work end-to-end for loop-free kernels**: saxpy
   and a transcendental kernel (`exp`, `sin`, `cos` on Float32) compile
   through the SPIR-V → OpenCL C path and run correctly on the iGPU.
2. **Upstream Julia bug found**: the ForwardDiff dual-number path in the
   march kernel produces LLVM IR that the SPIR-V backend mistranslates
   ("Select values must have same type as select instruction", from
   induction-variable shrinking). Workaround that fully fixes translation:
   hand-derived analytic D′ instead of duals (plus intrinsic-friendly
   `cexp`, `sqrt∘sqrt` step control, `floor(x+0.5)` rounding — kept in
   `opencl_igpu_test.jl`). Worth reporting to JuliaGPU/GPUCompiler.
3. **Hard blocker, vendor side**: this machine's AMD OpenCL compiler
   (driver 30.0.13044, Adrenalin ~22.x, 2022) **segfaults
   (`aclWriteToMem` access violation) on ANY kernel containing a loop** —
   even a trivial counted for-loop summing `exp` — while loop-free kernels
   compile fine. The march kernel therefore cannot run on this driver,
   independent of anything on the Julia side.

**Actionable**: updating the AMD Adrenalin driver to a current release (the
OpenCL stack was rebuilt since 2022) has a good chance of unblocking the
iGPU; after updating, rerun
`julia --project=benchmark/gpu -t auto benchmark/gpu/opencl_igpu_test.jl`.
Expectations should stay modest either way — gfx90c is a small integrated
part (~1.8 FP32 TFLOP/s vs ~1 for the 16-thread CPU), so it would validate
the *pipeline*, not deliver the 400×400 real-time target; that still calls
for a discrete card (CUDA path already in place, or the same OpenCL path on
a discrete AMD GPU with a current driver).

## Aside: is `@fastmath` worth it? (`fastmath_probe.jl`) — no

Showcase chart, 100×100, ω_max = 10⁴, tol = 10⁻⁵, 16 threads:

| experiment | plain | `@fastmath` | result identical? |
|---|---|---|---|
| **A.** user's `D(λ)`, package solver | 3.478 s | 4.055 s (**+17 %**) | **bit-identical** (max abs ΔZ̃ = 0) |
| **B.** whole march, flat real scalars (upper bound) | 0.871 s | 0.812 s (**−7 %**) | counts equal; ΔZ̃ up to 2.3·10⁻², residual max 0.498 → 0.499 |

- **A is the realistic case and `@fastmath` is a pessimization there.** The
  results are bit-identical, which *proves* the flags never reached the
  arithmetic: `@fastmath` is syntactic and does not propagate into callees,
  and with `λ::Complex{Dual}` every operator immediately dispatches into
  Base/ForwardDiff methods whose instructions are emitted without fast flags
  (inlining does not add them afterwards). The 17 % loss is the rewrite of
  `exp` to `Base.FastMath.exp_fast`, which for `Complex{Dual}` hits the
  generic fallback and inlines worse.
- **B is the ceiling: ~7 %**, and only reachable by rewriting the solver in
  flat scalar arithmetic — no duals, no `Complex`, analytic dD/dλ, i.e.
  giving up the generic user-supplied `D(λ)` that the whole package is for.
  Note what the same table shows for free: that rewrite is **4× faster**
  (0.871 s vs 3.478 s) *before* any fast-math. The formulation is the lever;
  `@fastmath` is noise on top of it.
- **Correctness argument stands on its own**: `@fastmath` implies
  `nnan`/`ninf`, but the package deliberately *uses* NaN/Inf as signals — the
  invalid-count marker for a root on the contour, the overflow retreat of
  the leading-order probe (|D| → Inf on purpose, A.10), `isfinite` filters
  in the sweeps. Those guards may legally be deleted under fast-math. Even
  in experiment B, where nothing broke, the max integer residual moved
  0.498 → 0.499, i.e. the self-validating diagnostic is perturbed right at
  the ½ rounding threshold.

Verdict: do not add `@fastmath` to the package. The safe subset (FMA
contraction, no NaN/Inf assumptions) is `@muladd`, which OrdinaryDiffEq
already applies inside its Runge–Kutta steppers, so that benefit is present.

## Reproduce

```powershell
julia --project=benchmark/gpu -e "using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()"
julia --project=benchmark/gpu -t auto benchmark/gpu/gpu_feasibility.jl
```

**On an NVIDIA machine — nothing to change.** CUDA.jl is already in this
environment and `realtime_chart.jl` auto-detects it: clone the repo, run the
two commands above, and the title bar reports `[GPU]` with the live
ms/frame. Sanity sequence on the NVIDIA machine:

```powershell
julia --project=benchmark/gpu -t auto benchmark/gpu/realtime_chart.jl --bench   # frame times incl. 400x400
julia --project=benchmark/gpu -t auto benchmark/gpu/realtime_chart.jl           # the live slider app
```

(The older validation scripts `gpu_feasibility.jl` / `two_system_validation.jl`
pin `backend = CPU()` on purpose — they are the CPU-reference studies.)
