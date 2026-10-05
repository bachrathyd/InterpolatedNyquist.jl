# GPU tour of the milling charts: one chart per model and number format, with timings and the
# accuracy of each format against a Float64 reference on a coarse check grid.
#   formats: F32 (Float32), F16 (D evaluated in Float16, march in Float32),
#            F16+ (F16, then the flagged points again in Float32 on the device)
#   julia --project=gpu/scripts hill/gpu_tour_milling.jl [--models mill2n,mill3d] [--res 1920x1080]
#         [--check 192x108] [--csv out.csv]
# Counts: the circle models (mill2n, ...) return Z_raw = Z, the strip models (mill3d) 2Z.
include(joinpath(@__DIR__, "..", "gpu", "scripts", "common.jl"))
include(joinpath(@__DIR__, "gpu_models.jl"))
include(joinpath(@__DIR__, "gpu_fast.jl"))
const NG = NyquistGPU

const CIRCLE = (n_power = 0, ω0 = 1e-9, ω_max = 0.5, h0 = 0.05, hrel = 0.25, nroots = 1)
const STRIP = (n_power = 0, ω0 = A_STRIP, ω_max = A_STRIP + 1, h0 = 1e-3, hrel = 0.1, nroots = 1)

# name => (title, D, c, reference D and c (evaluated in Float64), axes, march, formats, resolution
#          override, workspace, count divisor)
const MODELS = Dict{String, Any}(
    "mill2n" => (title = "Test 2: straight flutes, compressed Hill (D_mill2n, Q = 16)", D = D_mill2n,
                 c = mill2m_consts(Q = 16), Dref = D_mill2s, cref = mill2q_consts(Q = 64),
                 xr = (5.0, 25.0), yr = (0.0, 5.0), kw = CIRCLE, formats = ("F32", "F16", "F16+"),
                 res = nothing, ws = false, zdiv = 1),
    "mill3d" => (title = "Test 3: helix 30/45 deg, dense Hill (D_mill3, tol 1e-2)", D = D_mill3,
                 c = mill3_consts(tol = 1e-2), Dref = D_mill3, cref = mill3_consts(tol = 1e-4),
                 xr = (8.0, 30.0), yr = (0.0, 10.0), kw = STRIP, formats = ("F32",),
                 res = (480, 270), ws = true, zdiv = 2),
)
@isdefined(EXTRA_MODELS) && merge!(MODELS, EXTRA_MODELS)

fmt_types(f) = f == "F32" ? (Float32, Float32) : (Float32, Float16)

function plan_for(m, xr, yr, nx, ny, T, TE)
    kw = (backend = BACKEND, T = T, Teval = TE, m.kw...)
    if m.ws
        kw = (kw..., schedule = :strided, lanes = min(default_lanes(), 1 << 16),
              workspace = (Complex{ForwardDiff.Dual{NG.PhaseTag, T, 1}}, ws_len(m.D)))
    end
    return plan_grid(xr, yr, nx, ny; kw...)
end

counts(Zraw, zdiv) = map(z -> isfinite(z) ? round(Int, z / zdiv) : -1, Zraw)

function tour(names; res, chk, csv)
    gpu = ON_GPU ? CUDA.name(CUDA.device()) : "CPU"
    rows = String[]
    for name in names
        m = MODELS[name]
        nx, ny = something(m.res, res)
        cx, cy = chk
        println("\n== $(m.title): $(nx)x$(ny), check grid $(cx)x$(cy)")
        # Float64 reference on the check grid (same engine, reference form)
        rp = plan_for((m..., D = m.Dref), m.xr, m.yr, cx, cy, Float64, Float64)
        tref = @elapsed run!(rp, m.Dref, m.cref)
        Zref = counts(Array(rp.Zraw), m.zdiv)
        @printf("   reference (Float64): %.1f s, unstable %.1f %%\n", tref, 100count(>(0), Zref) / length(Zref))
        for f in m.formats
            T, TE = fmt_types(f)
            # accuracy on the check grid
            cp = plan_for(m, m.xr, m.yr, cx, cy, T, TE)
            run!(cp, m.D, m.c)
            if f == "F16+"
                crp = plan_for(m, m.xr, m.yr, cx, cy, Float32, Float32)
                recheck_flagged!(cp, crp, m.D, m.c)
            end
            Zc = counts(Array(cp.Zraw), m.zdiv)
            ndiff, nfail = count(Zc .!= Zref), count(<(0), Zc)
            # timing at the chart resolution
            p = plan_for(m, m.xr, m.yr, nx, ny, T, TE)
            rplan = f == "F16+" ? plan_for(m, m.xr, m.yr, nx, ny, Float32, Float32) : nothing
            frame() = begin
                tk = timed(() -> run!(p, m.D, m.c))
                tr = rplan === nothing ? 0.0 : timed(() -> recheck_flagged!(p, rplan, m.D, m.c))
                (tk, tr)
            end
            t1 = frame()                                       # compiles
            reps = sum(t1) > 10 ? 1 : 3
            ts = [frame() for _ in 1:reps]
            tk = median(first.(ts)); tr = median(last.(ts))
            fl = 100 * count(!=(Int8(0)), Array(p.flags)) / (nx * ny)
            @printf("   %-4s kernel %9.2f ms + re-check %8.2f ms = %9.2f ms (%7.2f Mpts/s) | flagged %.2f %% | check: %d of %d differ, %d failed\n",
                f, 1e3tk, 1e3tr, 1e3(tk + tr), nx * ny / (tk + tr) / 1e6, fl, ndiff, cx * cy, nfail)
            push!(rows, @sprintf("%s,%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d", gpu, name, f, nx, ny,
                1e3tk, 1e3tr, 1e3(tk + tr), nx * ny / (tk + tr) / 1e6, fl, ndiff, cx * cy, nfail))
        end
    end
    println("\nCSV")
    println("gpu,model,format,nx,ny,kernel_ms,recheck_ms,total_ms,mpts_per_s,flagged_pct,check_differ,check_n,check_failed")
    foreach(println, rows)
    csv === nothing || write(csv, "gpu,model,format,nx,ny,kernel_ms,recheck_ms,total_ms,mpts_per_s,flagged_pct,check_differ,check_n,check_failed\n" *
                                  join(rows, "\n") * "\n")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    print_device()
    tour(String.(split(arg("models", "mill2n,mill3d"), ','));
         res = parse_res(arg("res", "1920x1080")), chk = parse_res(arg("check", "192x108")),
         csv = arg("csv", nothing))
end
