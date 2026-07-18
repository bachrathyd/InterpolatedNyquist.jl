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

## Reproduce

```powershell
julia --project=benchmark/gpu -e "using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()"
julia --project=benchmark/gpu -t auto benchmark/gpu/gpu_feasibility.jl
```

On a machine with a CUDA GPU: add `CUDA` to the environment and replace
`backend = CPU()` with `backend = CUDABackend()` in `run_kernel` (arrays move
via the same KernelAbstractions API); nothing else changes.
