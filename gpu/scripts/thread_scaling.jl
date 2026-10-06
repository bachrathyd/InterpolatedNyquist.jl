# Thread scaling of the chart engine on the three time-independent examples (fourth, showcase,
# turning): one chart, timed with the threads Julia was started with (CPU backend) or on the GPU.
#   julia --project=gpu/scripts -t N gpu/scripts/thread_scaling.jl [--cpu] [--res 200x200] [--T Float64] [--csv out.csv]
# Prints / appends one CSV row per system: device,threads,system,T,nx,ny,ms,mpts_per_s,evals_per_point
include(joinpath(@__DIR__, "common.jl"))

const RES = parse_res(arg("res", "200x200"))
const T = arg("T", "Float64") == "Float32" ? Float32 : Float64
const CSV = arg("csv", nothing)
const REPS = parse(Int, arg("reps", "3"))

dev = ON_GPU ? device_name() : "CPU"
nx, ny = RES
rows = String[]
for name in ("fourth", "showcase", "turning")
    sys = SYSTEMS[name]
    plan = plan_grid(sys.xr, sys.yr, nx, ny; backend = BACKEND, T = T, n_power = sys.npow, sys.kw...)
    run!(plan, sys.D, sys.c)                                    # compile / warm-up
    t = minimum(timed(() -> run!(plan, sys.D, sys.c)) for _ in 1:REPS)
    ev = sum(fetch_result(plan).evals) / (nx * ny)
    push!(rows, @sprintf("%s,%d,%s,%s,%d,%d,%.3f,%.4f,%.1f", dev, Threads.nthreads(), name, T, nx, ny,
                         1e3t, nx * ny / t / 1e6, ev))
    println(rows[end])
end
if CSV !== nothing
    isfile(CSV) || write(CSV, "device,threads,system,T,nx,ny,ms,mpts_per_s,evals_per_point\n")
    open(io -> foreach(r -> println(io, r), rows), CSV, "a")
end
