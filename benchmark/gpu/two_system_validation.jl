# Two-system validation of the GPU-shaped kernel, as requested:
#   1. the showcase 2-DOF DAE (reduced quasi-polynomial) -- the HARD case:
#      delayed velocity feedback, omega^-1 phase ripple, long marches
#   2. the multi-DOF turning model -- rational D with structural poles
# 100x100 brute-force grids, kernel (Float64 + Float32) vs package solver.
#
# Run:  julia --project=benchmark/gpu -t auto benchmark/gpu/two_system_validation.jl

using KernelAbstractions, ForwardDiff, Statistics, Printf
using InterpolatedNyquist

struct PhaseTag end

# ---------------------------------------------------------------------------
# Characteristic functions, kernel-side (generic element type T; the swept
# parameters arrive as (p1, p2), extra fixed parameters are inlined)
# ---------------------------------------------------------------------------
# showcase (reduced 2-DOF DAE): m1=1, m2+m3=0.5, k1=-1, k2=1, c1=c2=0.05, tau=0.5
@inline function D_of(::Val{:showcase}, lam, P, Dg, ::Type{T}) where {T}
    a11 = lam^2 + T(0.1) * lam + T(0.0) + (P + Dg * lam) * exp(-T(0.5) * lam)
    a12 = -(T(0.05) * lam + one(T))
    a22 = T(0.5) * lam^2 + T(0.05) * lam + one(T)
    return a11 * a22 - a12 * a12
end
# turning (two-mode regenerative model of the gallery, A.7)
@inline function D_of(::Val{:turning}, lam, Omega, w, ::Type{T}) where {T}
    tau = T(2pi) / Omega
    G = one(T) / (lam^2 + 2 * T(0.02) * lam + one(T)) +
        T(0.45) / (lam^2 + 2 * T(0.03) * T(2.4) * lam + T(2.4)^2)
    return one(T) + w * (1 - exp(-tau * lam)) * G
end

# package-side references (Float64, standard call path)
D_showcase_ref(lam, p) = D_of(Val(:showcase), lam, p[1], p[2], Float64)
D_turning_ref(lam, p) = D_of(Val(:turning), lam, p[1], p[2], Float64)

@inline function theta_sigma(sys, ::Type{T}, w, p1, p2) where {T}
    d = ForwardDiff.Dual{PhaseTag}(T(w), one(T))
    lam = Complex(zero(d), d)
    Dv = D_of(sys, lam, T(p1), T(p2), T)
    Dre, Dim = real(Dv), imag(Dv)
    D_re, D_im = ForwardDiff.value(Dre), ForwardDiff.value(Dim)
    Dp_re, Dp_im = ForwardDiff.partials(Dre, 1), ForwardDiff.partials(Dim, 1)
    absD2 = D_re * D_re + D_im * D_im
    th = (D_re * Dp_im - D_im * Dp_re) / absD2
    absDp2 = Dp_re * Dp_re + Dp_im * Dp_im
    sig = -(D_re * Dp_im - D_im * Dp_re) / max(absDp2, T(1e-300))
    return th, absD2, sig
end

@kernel function phase_march!(Zraw, sigma, nsteps, sys,
        @Const(Pv), @Const(Dv), npow::T, wmax::T, rtol::T, atol::T) where {T}
    idx = @index(Global)
    p1 = Pv[idx]; p2 = Dv[idx]
    w = T(1e-9); y = zero(T); h = T(1e-2)
    steps = Int32(0)
    mind2 = T(Inf); sigbest = zero(T)
    th1, d2, sg = theta_sigma(sys, T, w, p1, p2)
    if d2 < mind2; mind2 = d2; sigbest = sg; end
    maxsteps = Int32(500_000)
    while w < wmax && steps < maxsteps
        h = min(h, wmax - w)
        th2, d2a, sga = theta_sigma(sys, T, w + h / 2, p1, p2)
        th3, d2b, sgb = theta_sigma(sys, T, w + 3 * h / 4, p1, p2)
        ynew = y + h * (T(2 / 9) * th1 + T(1 / 3) * th2 + T(4 / 9) * th3)
        th4, d2c, sgc = theta_sigma(sys, T, w + h, p1, p2)
        zlow = y + h * (T(7 / 24) * th1 + T(1 / 4) * th2 +
                        T(1 / 3) * th3 + T(1 / 8) * th4)
        err = abs(ynew - zlow)
        tol = atol + rtol * abs(ynew)
        steps += Int32(1)
        if err <= tol
            w += h; y = ynew; th1 = th4
            if d2a < mind2; mind2 = d2a; sigbest = sga; end
            if d2b < mind2; mind2 = d2b; sigbest = sgb; end
            if d2c < mind2; mind2 = d2c; sigbest = sgc; end
        end
        fac = T(0.9) * (tol / max(err, T(1e-30)))^T(1 / 3)
        h *= min(max(fac, T(0.2)), T(5.0))
    end
    Zraw[idx] = npow / 2 - y / T(pi)
    sigma[idx] = sigbest
    nsteps[idx] = steps
end

function run_kernel(sys, ::Type{T}, Ps, Dgs, npow; wmax, rtol = 1e-5, atol = 1e-5) where {T}
    backend = CPU()
    n = length(Ps)
    Zraw = KernelAbstractions.zeros(backend, T, n)
    sigma = KernelAbstractions.zeros(backend, T, n)
    nsteps = KernelAbstractions.zeros(backend, Int32, n)
    k = phase_march!(backend)
    k(Zraw, sigma, nsteps, sys, T.(Ps), T.(Dgs),
        T(npow), T(wmax), T(rtol), T(atol); ndrange = n)
    KernelAbstractions.synchronize(backend)
    return Array(Zraw), Array(sigma), Array(nsteps)
end

function warp_efficiency(steps, warpsize)
    total = sum(Float64, steps); padded = 0.0
    for i in 1:warpsize:length(steps)
        w = view(steps, i:min(i + warpsize - 1, length(steps)))
        padded += length(w) * maximum(w)
    end
    return total / padded
end

# ---------------------------------------------------------------------------
function validate(name, sys, Dref, xr, yr, npow, wmax)
    println("\n=== $name : 100x100, wmax=$wmax, tol=1e-5 ===")
    Pz = range(xr...; length = 100); Dz = range(yr...; length = 100)
    params = vec([(p, d) for p in Pz, d in Dz])
    Ps = first.(params); Dgs = last.(params)

    calculate_unstable_roots_p_vec(Dref, params[1:8];       # warm-up
        n_roots_to_track = 0, ω_max = wmax, reltol = 1e-5, abstol = 1e-5,
        n_power_max = npow)
    t_ref = @elapsed ((_, Zref_raw) = calculate_unstable_roots_p_vec(Dref, params;
        n_roots_to_track = 0, ω_max = wmax, reltol = 1e-5, abstol = 1e-5,
        n_power_max = npow))
    Zref = round.(Int, Zref_raw)
    @printf("  package sweep (Vern9, %d threads):  %7.3f s  (%.0f us/pt)\n",
        Threads.nthreads(), t_ref, 1e6 * t_ref / length(params))

    for T in (Float64, Float32)
        run_kernel(sys, T, Ps[1:8], Dgs[1:8], npow; wmax = wmax)  # warm-up
        t = @elapsed ((Zraw, sig, steps) = run_kernel(sys, T, Ps, Dgs, npow; wmax = wmax))
        Z = round.(Int, Zraw)
        resid = abs.(Zraw .- round.(Zraw))
        nd = count(Z .!= Zref)
        @printf("  KA kernel %s: %7.3f s  (%.0f us/pt)  wrong counts %d/%d  resid med %.1e max %.1e\n",
            rpad(string(T), 7), t, 1e6 * t / length(params), nd, length(Z),
            median(resid), maximum(resid))
        if T == Float64
            @printf("  steps/pixel: median %d, p90 %d, max %d;  warp-32 efficiency %.1f %% (row-major) / %.1f %% (random)\n",
                round(Int, median(steps)), round(Int, quantile(steps, 0.9)),
                maximum(steps), 100 * warp_efficiency(steps, 32),
                100 * warp_efficiency(steps[sortperm(rand(length(steps)))], 32))
        end
    end
end

validate("SHOWCASE 2-DOF DAE (reduced)", Val(:showcase), D_showcase_ref,
    (0.5, 3.0), (-0.5, 3.5), 4.0, 1e4)
validate("TURNING two-mode lobes", Val(:turning), D_turning_ref,
    (0.10, 1.2), (0.01, 1.1), 0.0, 1e4)
