# GPU tour of the milling charts: one chart per model and number format, with timings and the
# accuracy of each format against a Float64 reference on a coarse check grid.
#   formats: F32 (Float32), F16 (D evaluated in Float16, march in Float32),
#            F16+ (F16, then the flagged points again in Float32 on the device)
#   julia --project=gpu/scripts hill/gpu_tour_milling.jl [--models mill2p,mill3c,mill3d] [--res 1920x1080]
#         [--check 192x108] [--csv out.csv] [--adaptive yes|no]
# Adaptive rows (model "<name>-ad", models with ad = true): the same chart by NyquistGPU.run_adaptive!
# (certified coarse-to-fine refinement; F16+ re-checks the flagged points of every pass); their
# check columns compare with the full chart of the same format at EVERY pixel.
# Counts: the circle models (mill2n, ...) return Z_raw = Z, the strip models (mill3d) 2Z.
include(joinpath(@__DIR__, "..", "gpu", "scripts", "common.jl"))
include(joinpath(@__DIR__, "gpu_models.jl"))
include(joinpath(@__DIR__, "gpu_fast.jl"))
include(joinpath(@__DIR__, "gpu_helix.jl"))
const NG = NyquistGPU

const CIRCLE = (n_power = 0, ω0 = 1e-9, ω_max = 0.5, h0 = 0.05, hrel = 0.25, nroots = 1, circle = true, parity = true)
# impossible counts are flagged: the pole-free compressed determinants are polynomials of degree
# <= 2Q + 2 (Test 2) and <= nw + 4 (Test 3) in 1/z
circle_kw(zmax) = (CIRCLE..., zmax = zmax)
const STRIP = (n_power = 0, ω0 = A_STRIP, ω_max = A_STRIP + 1, h0 = 1e-3, hrel = 0.1, nroots = 1)

# name => (title, D, c, reference D and c (evaluated in Float64), axes, march, formats, resolution
#          override, workspace, count divisor)
const MODELS = Dict{String, Any}(
    "mill2g" => (title = "Test 2: straight flutes, compressed Hill, Gauss + kink (D_mill2g, Q = 8)", D = D_mill2g,
                 c = mill2g_consts(Q = 8), Dref = D_mill2g, cref = mill2g_consts(Q = 32),
                 xr = (5.0, 25.0), yr = (0.0, 5.0), kw = circle_kw(66), formats = ("F32", "F16", "F16+"),
                 res = nothing, ws = false, wslen = nothing, zdiv = 1),
    "mill2p" => (title = "Test 2: straight flutes, compressed Hill, pole-free (D_mill2p, Q = 16)", D = D_mill2p,
                 c = mill2m_consts(Q = 16), Dref = D_mill2s, cref = mill2q_consts(Q = 64),
                 xr = (5.0, 25.0), yr = (0.0, 5.0), kw = circle_kw(130), formats = ("F32", "F16", "F16+"),
                 res = nothing, ws = false, wslen = nothing, zdiv = 1),
    "mill2n" => (title = "Test 2: straight flutes, compressed Hill (D_mill2n, Q = 16)", D = D_mill2n,
                 c = mill2m_consts(Q = 16), Dref = D_mill2s, cref = mill2q_consts(Q = 64),
                 xr = (5.0, 25.0), yr = (0.0, 5.0), kw = circle_kw(130), formats = ("F32", "F16", "F16+"),
                 res = nothing, ws = false, wslen = nothing, zdiv = 1),
    "mill3d" => (title = "Test 3: helix 30/45 deg, dense Hill (D_mill3, tol 1e-2)", D = D_mill3,
                 c = mill3_consts(tol = 1e-2), Dref = D_mill3, cref = mill3_consts(tol = 1e-4),
                 xr = (8.0, 30.0), yr = (0.0, 10.0), kw = STRIP, formats = ("F32",),
                 res = (480, 270), ws = true, wslen = c -> ws_len(D_mill3), zdiv = 2),
    "mill3c" => (title = "Test 3 (hardest case): helix 30/45 deg, compressed Hill (D_mill3c, Q = 8, adaptive n_s <= 6, kink)", D = D_mill3c,
                 c = mill3c_consts(Q = 8), Dref = D_mill3c, cref = mill3c_consts(Q = 10, nsmax = 8),
                 xr = (3.0, 30.0), yr = (0.0, 10.0), kw = circle_kw(84), formats = ("F32", "F16", "F16+"),
                 res = nothing, ws = true, wslen = mill3c_wslen, zdiv = 1, ad = true),
    "mill3h" => (title = "Test 3 (hardest case): helix 30/45 deg, compressed Hill, Hessenberg (D_mill3h, Q = 8, adaptive n_s <= 6, kink)", D = D_mill3h,
                 c = mill3c_consts(Q = 8), Dref = D_mill3h, cref = mill3c_consts(Q = 10, nsmax = 8),
                 xr = (3.0, 30.0), yr = (0.0, 10.0), kw = circle_kw(84), formats = ("F32", "F16", "F16+"),
                 res = nothing, ws = true, wslen = mill3h_wslen, zdiv = 1, ad = true),
)
@isdefined(EXTRA_MODELS) && merge!(MODELS, EXTRA_MODELS)

fmt_types(f) = f == "F32" ? (Float32, Float32) : (Float32, Float16)

# workspace models: lanes within a memory budget, and stride 1 (neighbouring lanes take
# neighbouring points: coherent work and coalesced workspace access)
function plan_for(m, c, xr, yr, nx, ny, T, TE; budget = 3e9)
    kw = (backend = BACKEND, T = T, Teval = TE, m.kw...)
    if m.ws
        E = Complex{ForwardDiff.Dual{NG.PhaseTag, T, 1}}
        wl = m.wslen(c)
        lanes = clamp(floor(Int, budget / (wl * sizeof(E))), 1024, min(default_lanes(), 1 << 16))
        kw = (kw..., schedule = :strided, lanes = lanes, workspace = (E, wl))
    end
    p = plan_grid(xr, yr, nx, ny; kw...)
    m.ws && (p.stride = 1)
    return p
end

counts(Zraw, zdiv) = map(z -> isfinite(z) ? round(Int, z / zdiv) : -1, Zraw)

function tour(names; res, chk, csv, adaptive = true)
    gpu = ON_GPU ? CUDA.name(CUDA.device()) : "CPU"
    rows = String[]
    for name in names
        m = MODELS[name]
        nx, ny = something(m.res, res)
        cx, cy = chk
        println("\n== $(m.title): $(nx)x$(ny), check grid $(cx)x$(cy)")
        # Float64 reference on the check grid (same engine, reference form)
        rp = plan_for((m..., D = m.Dref), m.cref, m.xr, m.yr, cx, cy, Float64, Float64)   # wslen of Dref = of D
        tref = @elapsed run!(rp, m.Dref, m.cref)
        Zref = counts(Array(rp.Zraw), m.zdiv)
        @printf("   reference (Float64): %.1f s, unstable %.1f %%\n", tref, 100count(>(0), Zref) / length(Zref))
        for f in m.formats
            T, TE = fmt_types(f)
            # accuracy on the check grid
            cp = plan_for(m, m.c, m.xr, m.yr, cx, cy, T, TE)
            run!(cp, m.D, m.c)
            if f == "F16+"
                crp = plan_for(m, m.c, m.xr, m.yr, cx, cy, Float32, Float32)
                recheck_flagged!(cp, crp, m.D, m.c)
            end
            Zc = counts(Array(cp.Zraw), m.zdiv)
            ndiff, nfail = count(Zc .!= Zref), count(<(0), Zc)
            # timing at the chart resolution
            p = plan_for(m, m.c, m.xr, m.yr, nx, ny, T, TE)
            rplan = f == "F16+" ? plan_for(m, m.c, m.xr, m.yr, nx, ny, Float32, Float32) : nothing
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
            (adaptive && get(m, :ad, false)) || continue
            # the same chart, adaptively (same plans): compared with the full chart at every pixel
            Zfull = counts(Array(p.Zraw), m.zdiv)
            info = Ref{Any}(nothing)
            aframe() = timed(() -> (info[] = NG.run_adaptive!(p, m.D, m.c; rplan = rplan)))
            ta1 = aframe()                                     # compiles the list kernels
            ta = median([aframe() for _ in 1:(ta1 > 10 ? 1 : 3)])
            Za = counts(Array(p.Zraw), m.zdiv)
            nda, nfa = count(Za .!= Zfull), count(<(0), Za)
            fla = 100 * count(!=(Int8(0)), Array(p.flags)) / (nx * ny)
            @printf("   %-4s adaptive %7.2f ms (%5.1f %% of the pixels marched, passes %s) | flagged %.2f %% | %d of %d pixels differ from the full chart, %d failed
",
                f, 1e3ta, 100 * info[].points / (nx * ny), join(info[].passes, "/"), fla, nda, nx * ny, nfa)
            push!(rows, @sprintf("%s,%s-ad,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d", gpu, name, f, nx, ny,
                1e3ta, 0.0, 1e3ta, nx * ny / ta / 1e6, fla, nda, nx * ny, nfa))
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
    tour(String.(split(arg("models", "mill2g,mill3h,mill3d"), ','));
         res = parse_res(arg("res", "1920x1080")), chk = parse_res(arg("check", "192x108")),
         csv = arg("csv", nothing), adaptive = arg("adaptive", "yes") == "yes")
end
