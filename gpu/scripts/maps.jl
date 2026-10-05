# Stability maps for viewing: full-HD charts of the test systems and a
# resolution series of one system (turning lobes by default). Each chart is a
# Float32 sweep followed by a Float64 re-check of its flagged points.
#
# Run:  julia --project=gpu/scripts gpu/scripts/maps.jl [--out DIR] [--systems fourth,turning]
#           [--res 1920x1080] [--series turning] [--series_res 64,128,256,512,1024,1920x1080]
# Plot: python gpu/colab/plot_fields.py DIR   (or plot_fields.render_all(DIR) from Python)

include(joinpath(@__DIR__, "common.jl"))

const OUT = arg("out", joinpath(@__DIR__, "..", "results", "maps"))
const SYS = split(arg("systems", "fourth,turning"), ',')
const RES = parse_res(arg("res", "1920x1080"))
const SERIES = arg("series", "turning")
const SERIES_RES = parse_res.(split(arg("series_res", "64,128,256,512,1024,1920x1080"), ','))
mkpath(OUT)
print_device()
println("output  : ", abspath(OUT))

function make_map(sys, nx, ny; tag = "")
    xs = range(sys.xr...; length = nx)
    ys = range(sys.yr...; length = ny)
    pts = grid_points(xs, ys)
    kw = (backend = BACKEND, T = Float32, n_power = sys.npow, nroots = 4,
          schedule = ON_GPU ? :pixel : :queue, lanes = default_lanes(), sys.kw...)
    plan = plan_sweep(pts; kw...)
    run!(plan, sys.D, sys.c)                                   # compile / warm-up
    t = timed(() -> run!(plan, sys.D, sys.c))
    r = fetch_result(plan)
    nflag = count(!=(0), r.flags)
    trc = @elapsed recheck!(r, sys.D, pts; c = sys.c, merge(kw, (T = Float64,))...)
    @printf("  %-9s %5dx%-5d  sweep %9.2f ms (%6.2f Mpts/s)   %5d flagged, Float64 recheck %7.1f ms\n",
        sys.name, nx, ny, 1e3t, nx * ny / t / 1e6, nflag, 1e3trc)
    save_field(OUT, sys, (Z = reshape(r.Z, nx, ny), sigma = reshape(r.sigma, nx, ny)),
        nx, ny, 1e3t; tag = tag)
end

println("\n== full-resolution maps")
for name in SYS
    make_map(SYSTEMS[name], RES...)
end
if !isempty(SERIES)
    println("\n== resolution series: ", SERIES)
    for (nx, ny) in SERIES_RES
        make_map(SYSTEMS[SERIES], nx, ny; tag = "_series")
    end
end
