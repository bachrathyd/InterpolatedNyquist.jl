# GPU "thread" scaling: the chart is marched by exactly `lanes` persistent GPU threads (schedule
# :queue, every thread takes the next point from a counter), lanes = 1, 4, 16, ..., 2^20. The chart
# grows with the lanes (>= 64 points per lane, at least 64x64, at most 2048x2048) and the time is
# normalized to 10^6 points (a 1000x1000 chart): ms_per_Mpt = time / npts * 1e6  (= µs per point × 1000).
#   julia --project=gpu/scripts gpu/scripts/gpu_lane_scaling.jl [--T Float64] [--csv out.csv]
include(joinpath(@__DIR__, "common.jl"))

const T = arg("T", "Float64") == "Float32" ? Float32 : Float64
const CSV = arg("csv", nothing)
const LANES = [4^k for k in 0:parse(Int, arg("maxk", "10"))]   # 1 ... 1 048 576

dev = ON_GPU ? device_name() : "CPU"
rows = String[]
for name in ("fourth", "showcase", "turning"), lanes in LANES
    sys = SYSTEMS[name]
    side = clamp(ceil(Int, sqrt(64 * lanes)), 64, 2048)
    plan = plan_grid(sys.xr, sys.yr, side, side; backend = BACKEND, T = T, n_power = sys.npow,
                     schedule = :queue, lanes = lanes, workgroup = min(256, lanes), sys.kw...)
    run!(plan, sys.D, sys.c)                                      # compile / warm-up
    t1 = timed(() -> run!(plan, sys.D, sys.c))
    t = t1 > 2 ? t1 : minimum(timed(() -> run!(plan, sys.D, sys.c)) for _ in 1:3)
    npts = side * side
    push!(rows, @sprintf("%s,%d,%s,%s,%d,%.4f,%.4f", dev, lanes, name, T, npts, 1e3t, 1e3t / npts * 1e6))
    println(rows[end])
end
if CSV !== nothing
    isfile(CSV) || write(CSV, "device,threads,system,T,npts,ms,ms_per_Mpt\n")
    open(io -> foreach(r -> println(io, r), rows), CSV, "a")
end
