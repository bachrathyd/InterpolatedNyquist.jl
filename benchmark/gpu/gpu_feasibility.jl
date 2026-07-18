# GPU feasibility study for the brute-force stability chart.
#
# Question: would a GPU kernel pay off for DENSE charts, and is one needed?
#
# This machine has no CUDA driver, so the experiment is designed to answer
# everything that does NOT require the hardware:
#   1. Does the per-pixel adaptive phase march fit a GPU kernel at all
#      (no allocations, no dynamic dispatch, isbits state, bounded loop)?
#      -> written below as a KernelAbstractions kernel; the SAME code runs on
#         CUDABackend()/ROCBackend() unchanged. Correctness is verified on the
#         CPU backend against the package solver.
#   2. How much SIMT warp divergence would the adaptive march cause?
#      -> measured: per-pixel accepted+rejected step counts, grouped into
#         warps of 32 consecutive pixels; warp efficiency = work/(32*max).
#         This is the number that decides whether one-thread-per-pixel works.
#   3. Is Float32 enough? Consumer GPUs run FP64 at 1/32-1/64 rate, so the
#      answer decides whether a GeForce-class card helps at all.
#      -> the same kernel is run in Float32 and the counts compared.
#
# Run:  julia --project=benchmark/gpu -t auto benchmark/gpu/gpu_feasibility.jl

using KernelAbstractions, ForwardDiff, Statistics, Printf
using InterpolatedNyquist

# ---------------------------------------------------------------------------
# The benchmark system: the 4th-order delayed oscillator of the paper
# (appendix A.1 / solver zoo), swept over the control gains (P, Dg).
# ---------------------------------------------------------------------------
const C1 = 0.03
const ZETA = 0.02
const TAU = 0.5
const NPOW = 4.0                     # leading order, known by inspection

D_fourth(lam, p) = C1 * lam^4 + lam^2 + 2 * ZETA * lam + 1 +
                   (p[1] + p[2] * lam) * exp(-lam * TAU)

# ---------------------------------------------------------------------------
# The kernel: one thread = one pixel = one full adaptive phase march.
#
# Integrator: adaptive Bogacki-Shampine 3(2). Because the phase ODE's RHS is
# independent of the state, every RK stage is just an integrand evaluation at
# a shifted frequency, and FSAL makes it 3 evaluations per step. The integrand
# theta'(w) = Im(D'/D) is obtained exactly from one dual-number evaluation of
# D at lam = i*(w + eps), exactly as in the package. The minimum of |D|^2 and
# the one-Newton-step sigma estimate are tracked on the fly -- demonstrating
# that the rightmost-root by-product ports to a kernel as well.
#
# Everything is isbits (Complex{Dual{...,T,1}} scalars, plain floats), there
# are no allocations and no dynamic dispatch: this compiles for CUDA as-is.
# ---------------------------------------------------------------------------
struct PhaseTag end                  # ForwardDiff tag to avoid tag-check cost

@inline function eval_D(::Type{T}, w, P, Dg) where {T}
    d = ForwardDiff.Dual{PhaseTag}(T(w), one(T))
    lam = Complex(zero(d), d)
    Dv = T(C1) * lam^4 + lam^2 + 2 * T(ZETA) * lam + one(T) +
         (P + Dg * lam) * exp(-lam * T(TAU))
    Dre, Dim = real(Dv), imag(Dv)
    D_re, D_im = ForwardDiff.value(Dre), ForwardDiff.value(Dim)
    Dp_re, Dp_im = ForwardDiff.partials(Dre, 1), ForwardDiff.partials(Dim, 1)
    absD2 = D_re * D_re + D_im * D_im
    # theta' = Im(D'/D) = (Dre*Dim' - Dim*Dre') / |D|^2
    th = (D_re * Dp_im - D_im * Dp_re) / absD2
    # one Newton step: dD/dlam = -i D'  ->  sigma = -Re(D / (-i D'))
    absDp2 = Dp_re * Dp_re + Dp_im * Dp_im
    sig = -(D_re * Dp_im - D_im * Dp_re) / absDp2
    return th, absD2, sig
end

@kernel function phase_march!(Zraw, sigma, nsteps,
        @Const(Pv), @Const(Dv), wmax::T, rtol::T, atol::T) where {T}
    idx = @index(Global)
    P = Pv[idx]; Dg = Dv[idx]

    w = T(1e-9)
    y = zero(T)
    h = T(1e-2)
    steps = Int32(0)
    mind2 = T(Inf); sigbest = zero(T)

    th1, d2, sg = eval_D(T, w, P, Dg)           # FSAL seed
    if d2 < mind2; mind2 = d2; sigbest = sg; end

    maxsteps = Int32(200_000)
    while w < wmax && steps < maxsteps
        h = min(h, wmax - w)
        th2, d2a, sga = eval_D(T, w + h / 2, P, Dg)
        th3, d2b, sgb = eval_D(T, w + 3 * h / 4, P, Dg)
        ynew = y + h * (T(2 / 9) * th1 + T(1 / 3) * th2 + T(4 / 9) * th3)
        th4, d2c, sgc = eval_D(T, w + h, P, Dg)
        zlow = y + h * (T(7 / 24) * th1 + T(1 / 4) * th2 +
                        T(1 / 3) * th3 + T(1 / 8) * th4)
        err = abs(ynew - zlow)
        tol = atol + rtol * abs(ynew)
        steps += Int32(1)
        if err <= tol                            # accept
            w += h; y = ynew; th1 = th4
            if d2a < mind2; mind2 = d2a; sigbest = sga; end
            if d2b < mind2; mind2 = d2b; sigbest = sgb; end
            if d2c < mind2; mind2 = d2c; sigbest = sgc; end
        end
        fac = T(0.9) * (tol / max(err, T(1e-30)))^T(1 / 3)
        h *= min(max(fac, T(0.2)), T(5.0))
    end

    Zraw[idx] = T(NPOW) / 2 - y / T(pi)
    sigma[idx] = sigbest
    nsteps[idx] = steps
end

# ---------------------------------------------------------------------------
# Run on the CPU backend (the identical kernel would run on CUDABackend()).
# ---------------------------------------------------------------------------
function run_kernel(::Type{T}, Ps, Dgs; wmax = 1e4, rtol = 1e-5, atol = 1e-5) where {T}
    backend = CPU()
    n = length(Ps)
    Zraw = KernelAbstractions.zeros(backend, T, n)
    sigma = KernelAbstractions.zeros(backend, T, n)
    nsteps = KernelAbstractions.zeros(backend, Int32, n)
    Pv = adapt_vec(backend, T, Ps); Dv = adapt_vec(backend, T, Dgs)
    k = phase_march!(backend)
    k(Zraw, sigma, nsteps, Pv, Dv, T(wmax), T(rtol), T(atol); ndrange = n)
    KernelAbstractions.synchronize(backend)
    return Array(Zraw), Array(sigma), Array(nsteps)
end
adapt_vec(::CPU, ::Type{T}, x) where {T} = T.(x)

# ---------------------------------------------------------------------------
# The chart: same ranges as the solver zoo (paper s05)
# ---------------------------------------------------------------------------
const NX, NY = 100, 100
Pz = range(-2.0, 4.0; length = NX)
Dz = range(-2.0, 5.0; length = NY)
params = vec([(p, d) for p in Pz, d in Dz])
Ps = first.(params); Dgs = last.(params)

println("=== GPU feasibility study: $(NX)x$(NY) chart, 4th-order benchmark ===\n")

# --- reference: the package solver -----------------------------------------
t_ref = @elapsed begin
    global _, Zref_raw = calculate_unstable_roots_p_vec(D_fourth, params;
        n_roots_to_track = 0, ω_max = 1e4, reltol = 1e-5, abstol = 1e-5,
        n_power_max = NPOW)
end
t_ref = @elapsed begin
    global _, Zref_raw = calculate_unstable_roots_p_vec(D_fourth, params;
        n_roots_to_track = 0, ω_max = 1e4, reltol = 1e-5, abstol = 1e-5,
        n_power_max = NPOW)
end
Zref = round.(Int, Zref_raw)
@printf("package sweep (Vern9, %d threads, warm): %.3f s  (%.1f us/pt)\n",
    Threads.nthreads(), t_ref, 1e6 * t_ref / length(params))

# --- kernel, Float64 --------------------------------------------------------
run_kernel(Float64, Ps[1:4], Dgs[1:4])                     # JIT warm-up
t64 = @elapsed ((Zraw64, sig64, steps64) = run_kernel(Float64, Ps, Dgs))
Z64 = round.(Int, Zraw64)
resid64 = abs.(Zraw64 .- round.(Zraw64))
@printf("KA kernel CPU backend, Float64:          %.3f s  (%.1f us/pt)\n",
    t64, 1e6 * t64 / length(params))
@printf("  counts vs package reference: %d / %d differ\n",
    count(Z64 .!= Zref), length(Zref))
@printf("  integer residual: median %.2e, max %.2e\n",
    median(resid64), maximum(resid64))
@printf("  sigma sign agreement with (Z==0): %.2f %% of pixels\n",
    100 * count((sig64 .< 0) .== (Z64 .== 0)) / length(Z64))

# --- kernel, Float32 --------------------------------------------------------
run_kernel(Float32, Ps[1:4], Dgs[1:4])
t32 = @elapsed ((Zraw32, sig32, steps32) = run_kernel(Float32, Ps, Dgs))
Z32 = round.(Int, Zraw32)
resid32 = abs.(Zraw32 .- round.(Zraw32))
@printf("\nKA kernel CPU backend, Float32:          %.3f s\n", t32)
@printf("  counts vs Float64 kernel: %d / %d differ\n",
    count(Z32 .!= Z64), length(Z64))
@printf("  counts vs package reference: %d / %d differ\n",
    count(Z32 .!= Zref), length(Zref))
@printf("  integer residual: median %.2e, max %.2e\n",
    median(resid32), maximum(resid32))

# --- warp-divergence estimate ----------------------------------------------
# One GPU thread per pixel, warps of 32 consecutive (row-major) pixels: a
# warp retires when its SLOWEST thread finishes, so
#   efficiency = sum(steps) / sum(32 * max_steps_in_warp).
# Spatial correlation of the work (neighbouring pixels need similar step
# counts except across a stability boundary) is what decides this.
function warp_efficiency(steps, warpsize)
    total = sum(Float64, steps)
    padded = 0.0
    for i in 1:warpsize:length(steps)
        w = view(steps, i:min(i + warpsize - 1, length(steps)))
        padded += length(w) * maximum(w)
    end
    return total / padded
end
eff32 = warp_efficiency(steps64, 32)
effrand = warp_efficiency(steps64[sortperm(rand(length(steps64)))], 32)
@printf("\nstep count per pixel: median %d, p90 %d, max %d\n",
    round(Int, median(steps64)), round(Int, quantile(steps64, 0.9)),
    maximum(steps64))
@printf("warp (32-lane) efficiency, row-major order:   %.1f %%\n", 100 * eff32)
@printf("warp (32-lane) efficiency, random order:      %.1f %%  (worst case)\n",
    100 * effrand)

# --- arithmetic-intensity sanity: evaluations per pixel ---------------------
@printf("\nintegrand evaluations per pixel (3/step + seed): median %d\n",
    round(Int, 3 * median(steps64)))
println("""

Interpretation aid (fill in on a real GPU):
  speedup_estimate = (GPU FP64 GFLOP/s / CPU FP64 GFLOP/s) * warp_efficiency
  On consumer (GeForce) cards FP64 runs at 1/32-1/64 rate -- check the
  Float32 count-agreement above to see whether FP32 is usable instead.
""")
