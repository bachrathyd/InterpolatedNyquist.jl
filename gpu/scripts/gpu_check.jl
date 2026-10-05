# GPU sanity check: does the CUDA backend reproduce the CPU backend?
#
# For every system, a small chart is computed by the SAME kernels on the CPU
# backend (the reference validated against InterpolatedNyquist.jl, see
# gpu/validate/) and on the GPU, in Float32 and Float64 and with all three
# schedules. Counts must agree except at flagged (boundary-grazing) points;
# Zraw differs only by libm rounding (GPU and CPU sin/exp/atan differ in ulps).
#
# Run:  julia --project=gpu/scripts gpu/scripts/gpu_check.jl  [--n 64]
# Exit code 1 if any unflagged count differs.

include(joinpath(@__DIR__, "common.jl"))

const N = parse(Int, arg("n", "64"))
print_device()
ON_GPU || println("\n(no functional GPU -- this run checks CPU against CPU; on Colab it checks the GPU)")

bad = 0
for name in ("fourth", "showcase", "turning")
    sys = SYSTEMS[name]
    println("\n== $(sys.title), $(N)x$(N)")
    for T in (Float32, Float64)
        base = (c = sys.c, n_power = sys.npow, T = T, nroots = 8, sys.kw...)
        ref = chart(sys.D, sys.xr, sys.yr, N, N; backend = CPU(), schedule = :queue, base...)
        for sched in (:pixel, :strided, :queue)
            kw = (backend = BACKEND, schedule = sched, base...)
            ON_GPU && (kw = merge(kw, (lanes = default_lanes(),)))
            chart(sys.D, sys.xr, sys.yr, 8, 8; kw...)                    # compile
            r = chart(sys.D, sys.xr, sys.yr, N, N; kw...)
            flagged = (r.flags .!= 0) .| (ref.flags .!= 0)
            diff = (r.Z .!= ref.Z) .& .!flagged
            fin = isfinite.(r.Zraw) .& isfinite.(ref.Zraw)
            dz = maximum(abs.(r.Zraw[fin] .- ref.Zraw[fin]); init = 0.0)
            global bad += count(diff)
            @printf("  %-8s %-8s %8.2f ms   count diffs %d (unflagged) / %d flagged   max|ΔZraw| %.1e   evals med %d\n",
                T, sched, 1e3 * r.t, count(diff), count(flagged), dz, median(r.evals))
        end
    end
end
println(bad == 0 ? "\nPASS: GPU and CPU backends agree on every unflagged point." :
                   "\nFAIL: $bad unflagged count differences.")
exit(bad == 0 ? 0 : 1)
