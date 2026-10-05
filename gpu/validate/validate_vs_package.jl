# Cross-validation of the NyquistGPU kernels (CPU backend) against the CPU
# package InterpolatedNyquist.jl -- the "small bench" that runs on any machine.
#
# For each paper system on an n×n chart:
#   reference = package Vern9 sweep at tight tolerance (1e-7), 15 tracked roots,
#               dominant σ = max over the tracked roots (the paper's colouring)
#   kernels   = :unwrap and :bs3, Float64 and Float32, :queue schedule
# Reports per variant: wall time, D-evaluations per point, wrong counts (and how
# many of them sit next to a stability boundary), integer residual, and the
# agreement of the dominant-root estimate in the stable domain. Then the
# idealized SIMT efficiency of the three schedules from the measured step counts.
#
# Run (from the repo root):
#   julia --project=gpu/validate -t auto gpu/validate/validate_vs_package.jl [n] [systems...]
# e.g.  ... validate_vs_package.jl 100 fourth showcase turning

using InterpolatedNyquist, NyquistGPU, KernelAbstractions, Statistics, Printf
include(joinpath(@__DIR__, "..", "scripts", "systems.jl"))

const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
const NAMES = length(ARGS) >= 2 ? ARGS[2:end] : ["fourth", "showcase", "turning"]
const OUT = joinpath(@__DIR__, "..", "results")
mkpath(OUT)

"pixels with a differing 8-neighbour in the reference count map"
function boundary_mask(Z)
    nx, ny = size(Z)
    m = falses(nx, ny)
    for j in 1:ny, i in 1:nx, dj in -1:1, di in -1:1
        ii, jj = i + di, j + dj
        (1 <= ii <= nx && 1 <= jj <= ny) || continue
        Z[ii, jj] != Z[i, j] && (m[i, j] = true)
    end
    return m
end

function reference(sys, xs, ys)
    pts = vec([(x, y) for x in xs, y in ys])
    D = ref_D(sys)
    kw = (n_power_max = sys.ref_npow, ω_max = 1e4)
    calculate_unstable_roots_p_vec(D, pts[1:4]; n_roots_to_track = 15, reltol = 1e-7, abstol = 1e-7, kw...)
    t = @elapsed (Z, Zr, _, sl, _) = calculate_unstable_roots_p_vec(D, pts;
        n_roots_to_track = 15, reltol = 1e-7, abstol = 1e-7, kw...)
    σ = map(v -> (m = maximum(filter(isfinite, v); init = -Inf); isfinite(m) ? m : NaN), sl)
    # the package at its DEFAULT settings, for the speed comparison
    calculate_unstable_roots_p_vec(D, pts[1:4]; kw...)
    tdef = @elapsed calculate_unstable_roots_p_vec(D, pts; kw...)
    return reshape(Z, length(xs), length(ys)), reshape(σ, length(xs), length(ys)), t, tdef
end

rows = String[]
push!(rows, "system,variant,T,time_s,us_per_pt,evals_med,evals_p90,evals_max,wrong,wrong_boundary,resid_med,resid_max,sigma_sign_agree,sigma_med_abs_err")

for name in NAMES
    sys = SYSTEMS[name]
    xs = range(sys.xr...; length = N)
    ys = range(sys.yr...; length = N)
    println("\n=== $(sys.title): $(N)x$(N), $(Threads.nthreads()) CPU threads ===")
    Zref, σref, tref, tdef = reference(sys, xs, ys)
    bmask = boundary_mask(Zref)
    @printf("  package Vern9 tol 1e-7 (reference):      %7.3f s  (%6.1f us/pt)\n", tref, 1e6tref / N^2)
    @printf("  package Vern9 default (tol 1e-5, 1 root): %7.3f s  (%6.1f us/pt)\n", tdef, 1e6tdef / N^2)
    push!(rows, @sprintf("%s,package_default,Float64,%.4f,%.2f,,,,,,,,,", name, tdef, 1e6tdef / N^2))
    stable = (Zref .== 0) .& isfinite.(σref)

    for (meth, T) in ((:unwrap, Float64), (:unwrap, Float32), (:bs3, Float64), (:bs3, Float32))
        kw = (c = sys.c, n_power = sys.npow, T = T, method = meth, schedule = :queue, nroots = 8, sys.kw...)
        meth === :bs3 && (kw = merge(kw, (ω_max = 1e4, tol = 1e-5)))
        chart(sys.D, sys.xr, sys.yr, 4, 4; kw...)                       # JIT warm-up
        r = chart(sys.D, sys.xr, sys.yr, N, N; kw...)
        wrong = r.Z .!= Zref
        nb = count(wrong .& bmask)
        res = filter(isfinite, abs.(r.Zraw .- round.(r.Zraw)))
        ok = stable .& isfinite.(r.sigma)
        sgn = count(((r.sigma .< 0) .== (r.Z .== 0)) .& isfinite.(r.sigma)) / max(1, count(isfinite.(r.sigma)))
        σerr = count(ok) > 0 ? median(abs.(r.sigma[ok] .- σref[ok])) : NaN
        @printf("  %-7s %-8s %7.3f s (%6.1f us/pt)  evals med %5d p90 %5d max %6d  wrong %3d (%3d at boundary)  resid med %.1e max %.1e  σ: sign %.1f%%, |Δσ| med %.1e\n",
            meth, T, r.t, 1e6 * r.t / N^2, median(r.evals), quantile(vec(r.evals), 0.9),
            maximum(r.evals), count(wrong), nb, median(res), maximum(res), 100sgn, σerr)
        push!(rows, @sprintf("%s,%s,%s,%.4f,%.2f,%d,%d,%d,%d,%d,%.2e,%.2e,%.4f,%.2e", name, meth, T,
            r.t, 1e6 * r.t / N^2, median(r.evals), quantile(vec(r.evals), 0.9), maximum(r.evals),
            count(wrong), nb, median(res), maximum(res), sgn, σerr))
        if meth === :unwrap && T === Float32
            st = vec(r.steps)
            @printf("    SIMT efficiency (warp 32) from these step counts: one-per-pixel row-major %.0f%%, random %.0f%% | strided %.0f%% | queue %.0f%%  (lanes = 1024)\n",
                100simulate_schedule(st; schedule = :pixel_rowmajor),
                100simulate_schedule(st; schedule = :pixel_random),
                100simulate_schedule(st; schedule = :strided, lanes = 1024),
                100simulate_schedule(st; schedule = :queue, lanes = 1024))
            bad = findall(wrong)
            if !isempty(bad)
                for I in bad[1:min(end, 5)]
                    @printf("    wrong at (%.4f, %.4f): ref Z=%d, kernel Z=%d (Zraw %.3f), ref σ=%.2e, kernel σ=%.2e, flags %d\n",
                        xs[I[1]], ys[I[2]], Zref[I], r.Z[I], r.Zraw[I], σref[I], r.sigma[I], r.flags[I])
                end
            end
            # the hybrid: fast Float32 sweep, then Float64 re-run of the flagged points
            nflag = count(!=(0), r.flags)
            kw64 = merge(kw, (T = Float64,))
            tre = @elapsed recheck!(r, sys.D, grid_points(xs, ys); kw64...)
            @printf("    + Float64 recheck of %d flagged points (%.3f s): wrong %d (%d at boundary)\n",
                nflag, tre, count(r.Z .!= Zref), count((r.Z .!= Zref) .& bmask))
        end
    end
end

open(joinpath(OUT, "validation_cpu_$(N).csv"), "w") do io
    foreach(l -> println(io, l), rows)
end
println("\nwritten: ", joinpath(OUT, "validation_cpu_$(N).csv"))
