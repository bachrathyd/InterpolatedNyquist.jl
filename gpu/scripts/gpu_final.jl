# The real numbers: brute-force full-HD stability chart of the 4th-order benchmark
# model on one GPU (no MDBM). Run it once per GPU type.
#
#   julia --project=gpu/scripts gpu/scripts/gpu_final.jl [--out DIR] [--res 1920x1080] [--reps 10]
#
# Cases (one thread per point, the fastest GPU schedule):
#   Float64 / Float32 with the default ω_max = 1e5 (production accuracy; Float64 is the reference)
#   Float32 with the :queue schedule (for comparison)
#   Float64 / Float32 / Float16 with ω_max = 15 (enough for this model; Float16 needs
#   it -- its range ends at 65504 -- and uses the centred frequency scaling of
#   precision_scaled.jl), plus Float16 + Float32 re-check of its flagged points
#   a slider loop: a constant (the delay) changes every frame, Float32

include(joinpath(@__DIR__, "common.jl"))
const OUT = arg("out", joinpath(@__DIR__, "..", "results"))
const REPS = parse(Int, arg("reps", "10"))
sys = SYSTEMS["fourth"]
nx, ny = parse_res(arg("res", "1920x1080"))
pts = grid_points(range(sys.xr...; length = nx), range(sys.yr...; length = ny))
mkpath(OUT)
print_device()
if ON_GPU
    dev = CUDA.device()
    @printf("compute : sm_%d%d, %.1f GB, clock %d MHz\n", CUDA.capability(dev).major,
        CUDA.capability(dev).minor, CUDA.totalmem(dev) / 2^30,
        CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) ÷ 1000)
end

const Ωc = (15.0^4 * 1.0 / (0.03 * 65504.0 * 6.103515625e-5))^(1 / 8)   # centred scaling for Float16

function bench(; T, Teval = T, D = sys.D, c = sys.c, ω_max = 1e5, Ω = 1.0, schedule = :pixel)
    plan = plan_sweep(pts; backend = BACKEND, T = T, Teval = Teval, n_power = 4, nroots = 4,
        ω_max = ω_max / Ω, ω0 = 1e-9 / Ω, h0 = 1e-2 / Ω, schedule = schedule,
        lanes = default_lanes())
    run!(plan, D, c)                                                    # compile + warm-up
    ts = [timed(() -> run!(plan, D, c)) for _ in 1:REPS]
    tf = timed(() -> (run!(plan, D, c); Array(plan.Zraw); Array(plan.sigma)))
    return median(ts), minimum(ts), tf, fetch_result(plan), plan
end

println("\n4th-order delayed oscillator, $(nx)x$(ny) = $(nx * ny) points, brute force, $(REPS) repetitions")
cases = [
    ("Float64  ω_max=1e5 (reference)", (T = Float64,)),
    ("Float32  ω_max=1e5", (T = Float32,)),
    ("Float32  ω_max=1e5  :queue", (T = Float32, schedule = :queue)),
    ("Float64  ω_max=15", (T = Float64, ω_max = 15.0)),
    ("Float32  ω_max=15", (T = Float32, ω_max = 15.0)),
    ("Float16  ω_max=15  scaled", (T = Float32, Teval = Float16, D = D_fourth_scaled,
        c = fourth_scaled_consts(Ωc), ω_max = 15.0, Ω = Ωc)),
]
rows = String["device,case,kernel_ms_med,kernel_ms_min,frame_ms,mpts_per_s,evals_med,wrong_pct,flagged_pct"]
dev = device_name()
local ref
@printf("%-32s %10s %10s %10s %9s %6s %9s %9s\n", "case", "kernel ms", "min ms", "frame ms",
    "Mpts/s", "evals", "wrong", "flagged")
for (label, kw) in cases
    tmed, tmin, tf, r, plan = bench(; kw...)
    label == cases[1][1] && (global ref = r)
    N = length(r.Z)
    wrong = 100count(r.Z .!= ref.Z) / N
    flagged = 100count(!=(0), r.flags) / N
    @printf("%-32s %10.2f %10.2f %10.2f %9.1f %6d %8.4f%% %8.3f%%\n", label, 1e3tmed, 1e3tmin,
        1e3tf, N / tmed / 1e6, round(Int, median(r.evals)), wrong, flagged)
    push!(rows, join([dev, label, round(1e3tmed; digits = 3), round(1e3tmin; digits = 3),
        round(1e3tf; digits = 3), round(N / tmed / 1e6; digits = 2), median(r.evals), wrong, flagged], ','))
    if occursin("Float16", label)
        # the hybrid: Float16 sweep + re-check of the flagged points at production settings,
        # in Float32 (error-free above) -- Float64 is 1/32-1/64 rate on most GPUs
        rkw = (c = sys.c, backend = BACKEND, T = Float32, n_power = 4, nroots = 4,
               schedule = :pixel, lanes = default_lanes())
        recheck!(fetch_result(plan), sys.D, pts; rkw...)                 # compile the re-check
        t = timed() do
            run!(plan, kw.D, kw.c)
            recheck!(fetch_result(plan), sys.D, pts; rkw...)
        end
        @printf("%-32s %10.2f   (Float16 sweep + Float32 re-check of the flagged points)\n",
            "Float16 + recheck  total", 1e3t)
        push!(rows, join([dev, "Float16 + Float32 recheck total", round(1e3t; digits = 3), "", "", "", "", "", ""], ','))
    end
end
# slider loop: the delay changes every frame, results copied to the host
let plan = plan_sweep(pts; backend = BACKEND, T = Float32, n_power = 4, nroots = 4, lanes = default_lanes())
    run!(plan, sys.D, sys.c)
    nfr = 20
    t = timed() do
        for k in 1:nfr
            run!(plan, sys.D, (sys.c[1], sys.c[2], sys.c[3] * (1 + 0.01k)))
            Array(plan.Zraw); Array(plan.sigma)
        end
    end
    @printf("%-32s %10.2f ms/frame  (%.1f fps)\n", "slider loop Float32 ω_max=1e5", 1e3t / nfr, nfr / t)
    push!(rows, join([dev, "slider loop Float32", round(1e3t / nfr; digits = 3), "", "", "", "", "", ""], ','))
end
tag = replace(dev, r"[^A-Za-z0-9]+" => "_")
open(joinpath(OUT, "gpu_final_$(tag).csv"), "w") do io
    foreach(l -> println(io, l), rows)
end
println("written: ", joinpath(OUT, "gpu_final_$(tag).csv"))
