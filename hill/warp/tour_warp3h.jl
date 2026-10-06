# GPU tour of Test 3 (helix, the hardest milling case) with the warp-cooperative D_mill3h kernel
# (one warp per point, shared memory per axial class n_s), in the format of hill/gpu_tour_milling.jl:
# same check grid and Float64 reference (the tour's mill3h entry: per-thread D_mill3h,
# cref = mill3c_consts(Q = 10, nsmax = 8)), same march settings (circle_kw(84)), same CSV columns;
# model "mill3w" (full chart) and "mill3w-ad" (run_adaptive! driving the warp kernel, compared with
# the full chart at every pixel). Formats F32, F16 (D in Float16, march in Float32), F16+ (flagged
# points again in Float32 by the warp kernel).
#   julia --project=gpu/scripts hill/warp/tour_warp3h.jl [--res 1920x1080] [--check 192x108] [--adaptive yes] [--csv out.csv]
const REPO = get(ENV, "NGPU_REPO", normpath(joinpath(@__DIR__, "..", "..")))   # the repository root
include(joinpath(REPO, "hill", "gpu_tour_milling.jl"))     # MODELS, plan_for, counts, fmt_types (no run)
include(joinpath(@__DIR__, "warp3h.jl"))
include(joinpath(@__DIR__, "warp3h_kernel.jl"))

function tour_warp3h(; res = (1920, 1080), chk = (192, 108), csv = nothing, adaptive = true)
    gpu = ON_GPU ? CUDA.name(CUDA.device()) : "CPU"
    m = MODELS["mill3h"]                                      # c, cref, xr, yr, kw = circle_kw(84)
    c = m.c
    mw = (m..., ws = false)                                   # the warp kernel brings its own slots
    nx, ny = res; cx, cy = chk
    println("\n== Test 3, warp-cooperative D_mill3h (mill3w): $(nx)x$(ny), check grid $(cx)x$(cy); shared per warp (F32): ",
            join(["n_s=$s $(warp3h_shared_bytes(c, Float32, s)) B" for s in 2:Int(c[8])], ", "))
    rp = plan_for((m..., D = m.Dref), m.cref, m.xr, m.yr, cx, cy, Float64, Float64)   # per-thread reference
    tref = @elapsed run!(rp, m.Dref, m.cref)
    Zref = counts(Array(rp.Zraw), 1)
    @printf("   reference (Float64, per-thread D_mill3h, Q = 10, n_s <= 8): %.1f s, unstable %.1f %%\n", tref,
        100count(>(0), Zref) / length(Zref))
    rows = String[]
    for f in ("F32", "F16", "F16+")
        T, TE = fmt_types(f)
        cp = plan_for(mw, c, m.xr, m.yr, cx, cy, T, TE)
        run!(cp, W3h(), c)
        if f == "F16+"
            crp = plan_for(mw, c, m.xr, m.yr, cx, cy, Float32, Float32)
            recheck_flagged!(cp, crp, W3h(), c)
        end
        Zc = counts(Array(cp.Zraw), 1)
        ndiff, nfail = count(Zc .!= Zref), count(<(0), Zc)
        p = plan_for(mw, c, m.xr, m.yr, nx, ny, T, TE)
        rplan = f == "F16+" ? plan_for(mw, c, m.xr, m.yr, nx, ny, Float32, Float32) : nothing
        frame() = (timed(() -> run!(p, W3h(), c)), rplan === nothing ? 0.0 : timed(() -> recheck_flagged!(p, rplan, W3h(), c)))
        t1 = frame()                                         # compiles the class instances
        reps = sum(t1) > 10 ? 1 : 3
        ts = [frame() for _ in 1:reps]
        tk = median(first.(ts)); tr = median(last.(ts))
        fl = 100 * count(!=(Int8(0)), Array(p.flags)) / (nx * ny)
        @printf("   %-4s kernel %9.2f ms + re-check %8.2f ms = %9.2f ms (%7.2f Mpts/s) | flagged %.2f %% | check: %d of %d differ, %d failed\n",
            f, 1e3tk, 1e3tr, 1e3(tk + tr), nx * ny / (tk + tr) / 1e6, fl, ndiff, cx * cy, nfail)
        push!(rows, @sprintf("%s,mill3w,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d", gpu, f, nx, ny,
            1e3tk, 1e3tr, 1e3(tk + tr), nx * ny / (tk + tr) / 1e6, fl, ndiff, cx * cy, nfail))
        adaptive || continue
        Zfull = counts(Array(p.Zraw), 1)
        info = Ref{Any}(nothing)
        aframe() = timed(() -> (info[] = NG.run_adaptive!(p, W3h(), c; rplan = rplan)))
        ta1 = aframe()
        ta = median([aframe() for _ in 1:(ta1 > 10 ? 1 : 3)])
        Za = counts(Array(p.Zraw), 1)
        nda, nfa = count(Za .!= Zfull), count(<(0), Za)
        fla = 100 * count(!=(Int8(0)), Array(p.flags)) / (nx * ny)
        @printf("   %-4s adaptive %7.2f ms (%5.1f %% of the pixels marched, passes %s) | flagged %.2f %% | %d of %d pixels differ from the full chart, %d failed\n",
            f, 1e3ta, 100 * info[].points / (nx * ny), join(info[].passes, "/"), fla, nda, nx * ny, nfa)
        push!(rows, @sprintf("%s,mill3w-ad,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d", gpu, f, nx, ny,
            1e3ta, 0.0, 1e3ta, nx * ny / ta / 1e6, fla, nda, nx * ny, nfa))
    end
    hdr = "gpu,model,format,nx,ny,kernel_ms,recheck_ms,total_ms,mpts_per_s,flagged_pct,check_differ,check_n,check_failed"
    println("\nCSV\n", hdr); foreach(println, rows)
    csv === nothing || write(csv, hdr * "\n" * join(rows, "\n") * "\n")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    print_device()
    ON_GPU || error("needs a CUDA GPU")
    tour_warp3h(; res = parse_res(arg("res", "1920x1080")), chk = parse_res(arg("check", "192x108")),
                csv = arg("csv", nothing), adaptive = arg("adaptive", "yes") == "yes")
end
