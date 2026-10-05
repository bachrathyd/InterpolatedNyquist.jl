# Does lower precision pay off? Float16 vs Float32 vs Float64 on the 4th-order
# benchmark chart: time, wrong counts against Float64, flagged points.
#
# Float16 has max 65504: λ^4 alone overflows above ω = 16, so every precision
# runs with ω_max = 15 here (enough for this model: the tail phase there is
# ~0.05 rad, far below the ±π/2 that would change a count). The reference is a
# full Float64 sweep with the default ω_max. Float8 has no scalar arithmetic units on any
# GPU (FP8 exists only inside tensor-core matrix multiplies), so it is not testable
# as a scalar march.
#
# Run:  julia --project=gpu/scripts gpu/scripts/precision_test.jl [--res 1920x1080]

include(joinpath(@__DIR__, "common.jl"))
sys = SYSTEMS["fourth"]
nx, ny = parse_res(arg("res", ON_GPU ? "1920x1080" : "256"))
pts = grid_points(range(sys.xr...; length = nx), range(sys.yr...; length = ny))
print_device()
@printf("\n4th-order chart %dx%d, ω_max = 15 (reference: Float64, ω_max = 1e5)\n", nx, ny)

function runT(T; ω_max = 15.0)
    plan = plan_sweep(pts; backend = BACKEND, T = T, n_power = 4, nroots = 4, ω_max = ω_max,
        schedule = ON_GPU ? :pixel : :queue, lanes = default_lanes())
    run!(plan, sys.D, sys.c)
    t = minimum(timed(() -> run!(plan, sys.D, sys.c)) for _ in 1:3)
    return t, fetch_result(plan)
end

t64, r64 = runT(Float64; ω_max = 1e5)
for T in (Float64, Float32, Float16)
    t, r = try
        runT(T)
    catch e
        @printf("  %-8s failed: %s\n", T, sprint(showerror, e)[1:min(end, 200)])
        continue
    end
    wrong = count(r.Z .!= r64.Z)
    wrong_unflagged = count((r.Z .!= r64.Z) .& (r.flags .== 0))
    @printf("  %-8s %9.2f ms  (%6.2f Mpts/s)  evals med %3d   wrong %6d (%.3f %%), %d of them unflagged   flagged %d   failed %d\n",
        T, 1e3t, nx * ny / t / 1e6, round(Int, median(r.evals)), wrong, 100wrong / length(r.Z),
        wrong_unflagged, count(!=(0), r.flags), count(==(-1), r.Z))
end
