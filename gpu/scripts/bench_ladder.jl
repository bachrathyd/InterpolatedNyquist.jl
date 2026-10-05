# Benchmark ladder: chart resolution x precision x schedule x method, on the
# GPU (or the CPU backend with --cpu / without CUDA).
#
# Writes to --out (default gpu/results):
#   bench_<device>_<stamp>.csv          one row per configuration
#   bench_<device>_<stamp>.md           the same as a readable table
#   field_<system>_<nx>x<ny>.f32/.i8    colour field + counts of the --fields
#   field_<system>_<nx>x<ny>.json       resolution (for plotting, e.g. in Colab)
#
# Run (Colab / any CUDA machine):
#   julia --project=gpu/scripts gpu/scripts/bench_ladder.jl --out /content/drive/MyDrive/<folder>
# Options (defaults):
#   --systems fourth,showcase,turning   --res 100,256,512,1024,1920x1080 (CPU: 64,128,256)
#   --T Float32,Float64   --schedules pixel,strided,queue   --methods unwrap (add bs3)
#   --bs3max 256 (bs3 only up to this many pixels per side)   --reps 5
#   --fields 1920x1080 (CPU: 256)   --frames 30 (slider-loop frames at the field resolution)

include(joinpath(@__DIR__, "common.jl"))

const OUT = arg("out", joinpath(@__DIR__, "..", "results"))
const SYS = split(arg("systems", "fourth,showcase,turning"), ',')
const RES = parse_res.(split(arg("res", ON_GPU ? "100,256,512,1024,1920x1080" : "64,128,256"), ','))
const TS = Dict("Float32" => Float32, "Float64" => Float64)
const TLIST = [TS[t] for t in split(arg("T", "Float32,Float64"), ',')]
const SCHED = Symbol.(split(arg("schedules", "pixel,strided,queue"), ','))
const METHODS = Symbol.(split(arg("methods", "unwrap"), ','))
const BS3MAX = parse(Int, arg("bs3max", "256"))
const REPS = parse(Int, arg("reps", "5"))
const FIELDS = parse_res(arg("fields", ON_GPU ? "1920x1080" : "256"))
const FRAMES = parse(Int, arg("frames", "30"))
mkpath(OUT)

const DEV = replace(device_name(), r"[^A-Za-z0-9]+" => "_")
const STAMP = Dates.format(now(), "yyyymmdd_HHMMSS")
print_device()
println("output  : ", abspath(OUT))

csv = String["timestamp,device,system,nx,ny,npts,method,T,schedule,lanes,upload_ms,t_med_ms,t_min_ms,mpts_per_s,frame_ms,evals_med,evals_p90,evals_max,flagged,failed,count_diff_vs_f64"]
md = String["| system | res | method | T | schedule | kernel ms (med) | frame ms | Mpts/s | evals med/max | flagged | Δcount vs F64 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]

function bench_config(sys, pts, nx, ny, ref, meth, T, sched)
    name = sys.name
    kw = (backend = BACKEND, T = T, method = meth, schedule = sched, n_power = sys.npow,
          nroots = 8, lanes = default_lanes(), sys.kw...)
    meth === :bs3 && (kw = merge(kw, (ω_max = 1e4, tol = 1e-5)))
    tup = @elapsed (plan = plan_sweep(pts; kw...))
    run!(plan, sys.D, sys.c)                                       # compile / warm-up
    ts = [timed(() -> run!(plan, sys.D, sys.c)) for _ in 1:REPS]
    # a "frame": compute + bring Zraw and sigma to the host
    tf = timed(() -> (run!(plan, sys.D, sys.c); Array(plan.Zraw); Array(plan.sigma)))
    r = fetch_result(plan)
    flagged = (r.flags .!= 0) .| (ref.flags .!= 0)
    dcount = count((r.Z .!= ref.Z) .& .!flagged)
    tmed = median(ts)
    ev = (round(Int, median(r.evals)), round(Int, quantile(r.evals, 0.9)), maximum(r.evals))
    nflag = count(!=(0), r.flags)
    @printf("  %-6s %-8s %-8s kernel %9.3f ms (min %9.3f)  frame %9.3f ms  %8.2f Mpts/s  evals %4d/%5d  flagged %d  Δcount %d
",
        meth, T, sched, 1e3tmed, 1e3minimum(ts), 1e3tf, nx * ny / tmed / 1e6,
        ev[1], ev[3], nflag, dcount)
    push!(csv, join(Any[STAMP, DEV, name, nx, ny, nx * ny, meth, T, sched, plan.lanes,
        round(1e3tup; digits = 3), round(1e3tmed; digits = 4), round(1e3minimum(ts); digits = 4),
        round(nx * ny / tmed / 1e6; digits = 4), round(1e3tf; digits = 4),
        ev..., nflag, count(==(-1), r.Z), dcount], ','))
    push!(md, @sprintf("| %s | %dx%d | %s | %s | %s | %.3f | %.3f | %.2f | %d/%d | %d | %d |",
        name, nx, ny, meth, T, sched, 1e3tmed, 1e3tf, nx * ny / tmed / 1e6,
        ev[1], ev[3], nflag, dcount))
    if (nx, ny) == FIELDS && meth === :unwrap && T === Float32 && sched === :queue
        save_field(OUT, sys, (Z = reshape(r.Z, nx, ny), sigma = reshape(r.sigma, nx, ny)), nx, ny, 1e3tmed)
        # slider loop: the last constant (the delay τ, or 2π for turning, i.e. a
        # spindle-speed scale) changes every frame -- no re-upload, no recompile
        c0 = collect(sys.c)
        tl = timed() do
            for k in 1:FRAMES
                ck = copy(c0)
                ck[end] *= 1 + 0.002k
                run!(plan, sys.D, Tuple(ck))
                Array(plan.Zraw)
            end
        end
        @printf("    slider loop at %dx%d: %.2f ms/frame  (%.1f fps)
", nx, ny,
            1e3tl / FRAMES, FRAMES / tl)
        push!(md, @sprintf("| %s | %dx%d | slider loop | Float32 | queue | | %.3f | | | | |",
            name, nx, ny, 1e3tl / FRAMES))
    end
end

function bench_system(name)
    sys = SYSTEMS[name]
    for (nx, ny) in RES
        xs = range(sys.xr...; length = nx)
        ys = range(sys.yr...; length = ny)
        pts = grid_points(xs, ys)
        @printf("
== %s  %dx%d (%d points)
", sys.title, nx, ny, nx * ny)
        # reference counts on the same device: Float64 :unwrap
        ref = sweep(sys.D, pts; c = sys.c, backend = BACKEND, T = Float64, method = :unwrap,
            schedule = :queue, n_power = sys.npow, nroots = 8, lanes = default_lanes(), sys.kw...)
        for meth in METHODS, T in TLIST, sched in SCHED
            meth === :bs3 && max(nx, ny) > BS3MAX && continue
            bench_config(sys, pts, nx, ny, ref, meth, T, sched)
        end
    end
end

foreach(bench_system, SYS)

base = joinpath(OUT, "bench_$(DEV)_$(STAMP)")
open(io -> foreach(l -> println(io, l), csv), base * ".csv", "w")
open(base * ".md", "w") do io
    println(io, "# NyquistGPU benchmark -- ", device_name(), " -- ", STAMP, "\n")
    foreach(l -> println(io, l), md)
end
println("\nwritten: ", base, ".csv / .md")
