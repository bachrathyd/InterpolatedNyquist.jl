# NyquistGPU interactive server: a persistent Julia process that keeps the GPU
# plans alive between frames and answers one request per line on stdin.
#
#   julia --project=gpu/scripts gpu/interactive/server.jl [--cpu]
#
# Protocol (one line each, whitespace-separated key=value pairs):
#   meta                                  -> META {json: examples, formats, device}
#   render ex=fourth fmt=F16 nx=1920 ny=1080 x0=.. x1=.. y0=.. y1=.. c=0.03,0.02,0.5
#          maxw=1600 maxh=900 smin=-0.5 zcap=6 flags=0 bnd=1 out=/dev/shm/nyq_disp.rgb
#                                         -> OK {json: dw, dh, timings, statistics}
#   save out=/dev/shm/nyq_full.rgb        -> OK {json: w, h}   (full resolution, last render)
#   quit
# Images are raw RGB bytes, row-major from the top-left pixel. Errors: ERR message.
# The GPU colours and downsamples the chart itself, so only the display image
# crosses the bus; a full 8K frame is copied only on `save`.

include(joinpath(@__DIR__, "..", "scripts", "common.jl"))
using KernelAbstractions

# --------------------------------------------------------------------------
# examples: the systems of scripts/systems.jl plus the knobs of the UI
# --------------------------------------------------------------------------
const EXAMPLES = Any[
    (key = "fourth", sys = SYSTEMS["fourth"], ω16 = 15.0, smin = -0.6,
     knobs = [(i = 3, name = "delay τ", lo = 0.1, hi = 1.5),
              (i = 2, name = "damping ζ", lo = 0.0, hi = 0.2),
              (i = 1, name = "c₁ (λ⁴ coeff.)", lo = 0.005, hi = 0.1)]),
    (key = "showcase", sys = SYSTEMS["showcase"], ω16 = 15.0, smin = -0.4,
     knobs = [(i = 7, name = "delay τ", lo = 0.1, hi = 1.5),
              (i = 5, name = "damper c₁", lo = 0.0, hi = 0.5),
              (i = 6, name = "damper c₂", lo = 0.0, hi = 0.5)]),
    (key = "turning", sys = SYSTEMS["turning"], ω16 = 15.0, smin = -0.05,
     knobs = [(i = 1, name = "damping ζ₁", lo = 0.005, hi = 0.1),
              (i = 2, name = "mode-2 weight A₂", lo = 0.0, hi = 1.5),
              (i = 4, name = "mode-2 frequency ω₂", lo = 1.2, hi = 4.0)]),
]
# time-periodic examples (Hill determinant + argument principle, hill/gpu_models.jl):
# the model constants of the knobs are turned into the kernel constants by `cfun`
# (Fourier coefficients of the cutting function, ...); D is a function of μ = λ/ω_p(point),
# the march covers one period strip μ ∈ [a, a+1] and returns Z_raw = 2 Z
const HILL_MODELS = joinpath(@__DIR__, "..", "..", "hill", "gpu_models.jl")
if isfile(HILL_MODELS)
    include(HILL_MODELS)
    include(joinpath(@__DIR__, "..", "..", "hill", "gpu_fast.jl"))
    include(joinpath(@__DIR__, "..", "..", "hill", "gpu_helix.jl"))
    ws_len(::typeof(D_mill3c)) = mill3c_wslen(8, 4, 42)
    ws_len(::typeof(D_mill3h)) = mill3h_wslen(8, 4, 42)
    const HKW = (ω0 = A_STRIP, ω_max = A_STRIP + 1, h0 = 1e-3, hrel = 0.05)
    const HKWM = (ω0 = A_STRIP, ω_max = A_STRIP + 1, h0 = 1e-3, hrel = 0.1)   # milling: fewer samples
    const MEMO = Dict{Any, Any}()
    function memo(f, k)                 # constants per slider state (bounded: sliders produce many)
        length(MEMO) > 512 && !haskey(MEMO, k) && empty!(MEMO)
        return get!(() -> f(), MEMO, k)
    end
    # half circle |z| = 1; the step cap of the strip examples is not needed there (same counts
    # on 12 800 test points, 13.9 instead of 18.3 evaluations per point)
    const HKWQ = (ω0 = 1e-9, ω_max = 0.5, h0 = 0.05, hrel = 0.25)
    push!(EXAMPLES,
        (key = "mill2q", ω16 = 0.0, smin = -0.04, hill = true, zdiv = 1, f16 = true, ωp = mill2q_ωp,
         cfun = c -> memo(() -> mill2g_consts(ζ = c[1], aD = c[2], kr = c[3], Q = 8), (:m2g, c)), res = "1920x1080",
         note = "1-DOF milling, straight flutes, z = 2, down milling, f_n = 922 Hz. Infinite Hill determinant (all harmonics, " *
                "closed form) compressed by the matrix determinant lemma to the 16 quadrature nodes of the cutting window; " *
                "its semiseparable structure gives the 16 x 16 determinant in O(16) operations. Counted along the unit circle " *
                "of the Floquet multiplier (half circle by symmetry).",
         sys = (title = "milling, straight flutes (Test 2, compressed Hill, fast)", D = D_mill2g,
                c = (0.011, 0.05, 1 / 3), npow = 0, xr = (5.0, 25.0), yr = (0.0, 5.0), xl = "rpm/1000",
                yl = "a_p [mm]", kw = HKWQ),
         knobs = [(i = 1, name = "damping ζ", lo = 0.002, hi = 0.05), (i = 2, name = "immersion a/D", lo = 0.02, hi = 1.0),
                  (i = 3, name = "K_n/K_t", lo = 0.0, hi = 1.0)]),
        (key = "mill3c", ω16 = 0.0, smin = -0.03, hill = true, zdiv = 1, f16 = true, ωp = mill3c_ωp,
         cfun = c -> memo(() -> mill3c_consts(ζ = c[1], aD = c[2], β2 = c[3], Q = 8, ns = 4), (:m3c, c)),
         res = "960x540",
         note = "1-DOF milling, two flutes with helix 30° and β₂ (R = 8 mm), delays distributed over the axial depth, " *
                "spindle period. Compressed Hill determinant on 2 x 4 x 8 Gauss nodes in the workpiece frame (all harmonics " *
                "in closed form); the regenerative term of every node is the same material node on the previous tooth. " *
                "Reduced once per point to a 34 x 34 determinant in Hessenberg form (O(34²) per evaluation); counted " *
                "along the unit circle of the Floquet multiplier.",
         sys = (title = "milling, different helix angles (Test 3, compressed Hill, fast)", D = D_mill3h,
                c = (0.011, 0.05, 45.0), npow = 0, xr = (8.0, 30.0), yr = (0.0, 10.0), xl = "rpm/1000",
                yl = "a_p [mm]", kw = HKWQ),
         knobs = [(i = 1, name = "damping ζ", lo = 0.002, hi = 0.05), (i = 2, name = "immersion a/D", lo = 0.02, hi = 1.0),
                  (i = 3, name = "helix β₂ [°]", lo = 0.0, hi = 60.0)]),
        (key = "mathieu", ω16 = 0.0, smin = -0.3, hill = true, ωp = mathieu_ωp,
         cfun = c -> mathieu_consts(c[1], c[2]),
         note = "x'' + κx' + (δ + ε cos t)x = b x(t - 2π); Hill determinant, harmonics derived per point",
         sys = (title = "delayed Mathieu (time-periodic, Hill)", D = D_mathieu, c = (0.1, 1.0), npow = 0,
                xr = (-1.0, 5.0), yr = (-1.5, 1.5), xl = "δ", yl = "b", kw = HKW),
         knobs = [(i = 1, name = "damping κ", lo = 0.0, hi = 0.5), (i = 2, name = "excitation ε", lo = 0.0, hi = 3.0)]),
        (key = "mill2", ω16 = 0.0, smin = -0.04, hill = true, ωp = mill2_ωp,
         cfun = c -> memo(() -> mill2_consts(ζ = c[1], aD = c[2], kr = c[3], tol = 1e-2), (:m2, c)), res = "480x270",
         note = "1-DOF milling, straight flutes, z = 2, down milling, f_n = 922 Hz; axes: spindle speed [1000 rpm], depth of cut [mm]",
         sys = (title = "milling, straight flutes (Test 2, dense Hill, slow)", D = D_mill2, c = (0.011, 0.05, 1 / 3), npow = 0,
                xr = (5.0, 25.0), yr = (0.0, 5.0), xl = "rpm/1000", yl = "a_p [mm]", kw = HKWM),
         knobs = [(i = 1, name = "damping ζ", lo = 0.002, hi = 0.05), (i = 2, name = "immersion a/D", lo = 0.02, hi = 1.0),
                  (i = 3, name = "K_n/K_t", lo = 0.0, hi = 1.0)]),
        (key = "mill3", ω16 = 0.0, smin = -0.03, hill = true, ωp = mill3_ωp,
         cfun = c -> memo(() -> mill3_consts(ζ = c[1], aD = c[2], β2 = c[3], tol = 1e-2), (:m3, c)), res = "320x180",
         note = "1-DOF milling, two flutes with helix 30° and β₂ (R = 8 mm): distributed delays, spindle period; heavier (dense 53x53 LU per point)",
         sys = (title = "milling, different helix angles (Test 3, Hill)", D = D_mill3, c = (0.011, 0.05, 45.0), npow = 0,
                xr = (8.0, 30.0), yr = (0.0, 10.0), xl = "rpm/1000", yl = "a_p [mm]", kw = HKWM),
         knobs = [(i = 1, name = "damping ζ", lo = 0.002, hi = 0.05), (i = 2, name = "immersion a/D", lo = 0.02, hi = 1.0),
                  (i = 3, name = "helix β₂ [°]", lo = 0.0, hi = 60.0)]))
end
ishill(e) = hasproperty(e, :hill) && e.hill
kconsts(e, c) = ishill(e) ? e.cfun(c) : c
const EXBYKEY = Dict(e.key => e for e in EXAMPLES)
# format -> (T, Teval, use the Float16 window, re-check flagged points in Float32)
const FORMATS = Dict("F16" => (Float32, Float16, true, false),
                     "F16+" => (Float32, Float16, true, true),
                     "F32" => (Float32, Float32, false, false),
                     "F64" => (Float64, Float64, false, false))
const REFINES = [("count", "exact rightmost root (Newton + counting on shifted lines)"),
                 ("newton", "Newton ×10 polish of the tracked roots"),
                 ("none", "first-order estimate (fastest)")]
const FORMAT_LABELS = [("F16", "Float16 (fastest)"), ("F16+", "Float16 + Float32 re-check of flagged points"),
                       ("F32", "Float32 (exact, ω_max = 1e5)"), ("F64", "Float64 (reference)")]

jstr(s) = "\"" * replace(string(s), "\\" => "\\\\", "\"" => "\\\"") * "\""
jnum(x) = isfinite(x) ? string(Float64(x)) : "null"
jt(x) = isfinite(x) ? string(round(Float64(x); sigdigits = 4)) : "null"     # timings
jvec(v) = "[" * join(jnum.(v), ",") * "]"

function meta_json()
    exs = map(EXAMPLES) do e
        s = e.sys
        knobs = join(["{\"i\":$(k.i),\"name\":$(jstr(k.name)),\"lo\":$(k.lo),\"hi\":$(k.hi)," *
                      "\"value\":$(jnum(s.c[k.i]))}" for k in e.knobs], ",")
        "{\"key\":$(jstr(e.key)),\"title\":$(jstr(s.title)),\"xr\":$(jvec(collect(s.xr)))," *
        "\"yr\":$(jvec(collect(s.yr))),\"xl\":$(jstr(s.xl)),\"yl\":$(jstr(s.yl))," *
        "\"c\":$(jvec(collect(s.c))),\"smin\":$(e.smin),\"knobs\":[$knobs]," *
        "\"hill\":$(ishill(e)),\"f16\":$(hasproperty(e, :f16) && e.f16),\"note\":$(jstr(ishill(e) ? e.note : "")),\"res\":$(jstr(hasproperty(e, :res) ? e.res : ""))}"
    end
    fmts = join(["[$(jstr(k)),$(jstr(l))]" for (k, l) in FORMAT_LABELS], ",")
    refs = join(["[$(jstr(k)),$(jstr(l))]" for (k, l) in REFINES], ",")
    return "{\"device\":$(jstr(device_name())),\"examples\":[$(join(exs, ","))],\"formats\":[$fmts]," *
           "\"refines\":[$refs],\"w16\":$(EXAMPLES[1].ω16)}"
end

# --------------------------------------------------------------------------
# colouring + downsampling on the device
# --------------------------------------------------------------------------
const VIRIDIS = ((68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142), (33, 144, 141),
                 (39, 173, 129), (92, 200, 99), (170, 220, 50), (253, 231, 37))
const REDS = ((252, 187, 161), (251, 106, 74), (203, 24, 29), (103, 0, 13))

@inline function lerp_table(tab, t)
    n = length(tab) - 1
    x = clamp(t, 0.0f0, 1.0f0) * n
    i = min(unsafe_trunc(Int32, x), Int32(n - 1))
    w = x - i
    r = g = b = 0.0f0
    for k in 0:(n - 1)                      # unrolled select: no dynamic tuple indexing
        if k == i
            a, c = tab[k + 1], tab[k + 2]
            r = a[1] + w * (c[1] - a[1]); g = a[2] + w * (c[2] - a[2]); b = a[3] + w * (c[3] - a[3])
        end
    end
    return r, g, b
end

@inline function pixel_colour(z, s, smin, zcap)
    isfinite(z) || return 128.0f0, 128.0f0, 128.0f0              # failed march: grey
    Z = round(Int32, z)
    Z < 0 && return 128.0f0, 128.0f0, 128.0f0
    Z == 0 && return lerp_table(VIRIDIS, isfinite(s) ? Float32(s) / smin * -1 + 1 : 0.0f0)
    return lerp_table(REDS, (min(Z, zcap) - 1) / max(zcap - 1, 1.0f0))
end

# one display pixel = an f×f block of chart points (box filter); the boundary
# (stable and unstable points in the block or its right/lower neighbours) is white
@kernel function k_display!(img, @Const(Zr), @Const(Sg), @Const(Fl), nx, ny, f, dw,
                            smin, zcap, showflags, boundary)
    I = @index(Global, Linear)
    u = (I - 1) % dw
    v = (I - 1) ÷ dw
    r = g = b = 0.0f0
    cnt = 0
    st = false; un = false; fl = false
    for bj in 0:f, bi in 0:f
        i = u * f + bi
        jr = v * f + bj                                     # row counted from the top
        if i < nx && jr < ny
            k = i + (ny - 1 - jr) * nx + 1
            @inbounds z = Zr[k]
            Z = isfinite(z) ? round(Int32, z) : Int32(-1)
            st |= Z == 0
            un |= Z > 0
            if bi < f && bj < f
                @inbounds s = Sg[k]
                rr, gg, bb = pixel_colour(z, s, smin, zcap)
                r += rr; g += gg; b += bb
                cnt += 1
                @inbounds fl |= Fl[k] != 0
            end
        end
    end
    if cnt > 0
        r /= cnt; g /= cnt; b /= cnt
    end
    if boundary && st && un
        r = g = b = 255.0f0
    elseif showflags && fl
        r = 0.5f0 * r + 127.5f0; g = 0.5f0 * g; b = 0.5f0 * b + 127.5f0       # magenta tint
    end
    @inbounds img[1, I] = unsafe_trunc(UInt8, clamp(r, 0.0f0, 255.0f0))
    @inbounds img[2, I] = unsafe_trunc(UInt8, clamp(g, 0.0f0, 255.0f0))
    @inbounds img[3, I] = unsafe_trunc(UInt8, clamp(b, 0.0f0, 255.0f0))
end

# --------------------------------------------------------------------------
# state: one plan (+ the Float32 re-check plan) per resolution / example / format
# --------------------------------------------------------------------------
mutable struct State
    key::Any
    plan::Any
    rplan::Any
    grid::Any
    last::Any        # (nx, ny, smin, zcap, flags, bnd) of the last render
end
const S = State(nothing, nothing, nothing, nothing, nothing)

function release!()
    S.plan = nothing; S.rplan = nothing; S.key = nothing; S.grid = nothing
    GC.gc(true)
    ON_GPU && CUDA.reclaim()
    return nothing
end

function get_plan(e, fmt, nx, ny, xr, yr)
    T, TE, w16, rc = FORMATS[fmt]
    key = (e.key, fmt, nx, ny)
    if S.key != key
        release!()
        kw = (backend = BACKEND, T = T, Teval = TE, n_power = e.sys.npow, nroots = ishill(e) ? 1 : 4,
              ω_max = w16 ? e.ω16 : 1e5, lanes = default_lanes(), e.sys.kw...)
        if ishill(e) && applicable(ws_len, e.sys.D)      # dense Hill matrix: per-lane workspace
            kw = (kw..., schedule = :strided, lanes = min(default_lanes(), 1 << 16),
                  workspace = (Complex{ForwardDiff.Dual{NyquistGPU.PhaseTag, T, 1}}, ws_len(e.sys.D)))
        end
        S.plan = plan_grid(xr, yr, nx, ny; kw...)
        if rc                        # same march (Hill: the same strip / circle), Float32 throughout
            rkw = (kw..., T = Float32, Teval = Float32)
            ishill(e) || (rkw = (rkw..., nroots = 4, schedule = :pixel))
            S.rplan = plan_grid(xr, yr, nx, ny; rkw...)
        end
        if haskey(kw, :workspace)    # neighbouring lanes take neighbouring points (coalesced workspace)
            S.plan.stride = 1
            S.rplan === nothing || (S.rplan.stride = 1)
        end
        S.key = key
        S.grid = (xr, yr)
    elseif S.grid != (xr, yr)
        regrid!(S.plan, xr, yr, nx, ny)
        S.grid = (xr, yr)
    end
    return S.plan, S.rplan
end

function colour!(plan, nx, ny, f, smin, zcap, flags, bnd)
    dw, dh = cld(nx, f), cld(ny, f)
    img = KernelAbstractions.allocate(BACKEND, UInt8, 3, dw * dh)
    k_display!(BACKEND, 256)(img, plan.Zraw, plan.sigma, plan.flags, Int32(nx), Int32(ny),
        Int32(f), Int32(dw), Float32(smin), Float32(zcap), flags, bnd; ndrange = dw * dh)
    KernelAbstractions.synchronize(BACKEND)
    return img, dw, dh
end

ms(t0) = (time_ns() - t0) / 1e6

function render(a)
    e = EXBYKEY[a["ex"]]
    fmt = a["fmt"]
    haskey(FORMATS, fmt) || error("unknown format $fmt")
    ishill(e) && startswith(fmt, "F16") && !(hasproperty(e, :f16) && e.f16) && (fmt = "F32")   # Float16 only where tested
    nx, ny = parse(Int, a["nx"]), parse(Int, a["ny"])
    xr = (parse(Float64, a["x0"]), parse(Float64, a["x1"]))
    yr = (parse(Float64, a["y0"]), parse(Float64, a["y1"]))
    c = Tuple(parse.(Float64, split(a["c"], ',')))
    length(c) == length(e.sys.c) || error("expected $(length(e.sys.c)) constants")
    c = kconsts(e, c)
    maxw, maxh = parse(Int, get(a, "maxw", "1600")), parse(Int, get(a, "maxh", "900"))
    smin = parse(Float64, get(a, "smin", string(e.smin)))
    zcap = parse(Float64, get(a, "zcap", "6"))
    flags = get(a, "flags", "0") == "1"
    bnd = get(a, "bnd", "1") == "1"
    t0 = time_ns()
    plan, rplan = get_plan(e, fmt, nx, ny, xr, yr)
    tplan = ms(t0)
    # march settings that need no reallocation: ω_max (Float16: capped at its window) and
    # the root refinement (first-order / Newton / exact by counting on shifted lines)
    _, _, w16, _ = FORMATS[fmt]
    wreq = parse(Float64, get(a, "wmax", string(w16 ? e.ω16 : 1e5)))
    wmax = w16 ? min(wreq, e.ω16) : wreq
    ref = get(a, "refine", "count")
    nr = ref == "none" ? 0 : (ref == "newton" ? 10 : 5)
    cert = ref == "count"
    if ishill(e)                  # fixed strip; counting on shifted lines would meet the row-scale poles
        cert = false
        ref == "count" && (nr = 10)
        wmax = NaN
        set_march!(plan; refine = nr, certify = false)
    else
        set_march!(plan; ω_max = wmax, refine = nr, certify = cert)
    end
    if rplan !== nothing          # Hill: keep the strip / circle of the example
        ishill(e) ? set_march!(rplan; refine = nr, certify = false) :
                    set_march!(rplan; ω_max = max(wreq, e.ω16), refine = nr, certify = cert)
    end
    t0 = time_ns()
    run!(plan, e.sys.D, c)
    tker = ms(t0)
    t0 = time_ns()
    nflag = count(!=(Int8(0)), plan.flags)
    nre = 0
    if rplan !== nothing
        nre = recheck_flagged!(plan, rplan, e.sys.D, c)
        nflag = count(!=(Int8(0)), plan.flags)
    end
    tre = ms(t0)
    t0 = time_ns()
    if ishill(e)                  # Z_raw = 2Z over the strip; σ from μ- to λ-units (after the re-check)
        zdiv = hasproperty(e, :zdiv) ? e.zdiv : 2
        zdiv == 1 || (plan.Zraw ./= zdiv)
        cT = map(eltype(plan.sigma), c)
        plan.sigma .*= e.ωp.(plan.points, Ref(cT))
        KernelAbstractions.synchronize(BACKEND)
    end
    f = max(1, cld(nx, maxw), cld(ny, maxh))
    img, dw, dh = colour!(plan, nx, ny, f, smin, zcap, flags, bnd)
    tcol = ms(t0)
    t0 = time_ns()
    host = Array(img)
    out = get(a, "out", "/dev/shm/nyq_disp.rgb")
    write(out, host)
    tread = ms(t0)
    S.last = (nx, ny, smin, zcap, flags, bnd)
    n = nx * ny
    return "{\"dw\":$dw,\"dh\":$dh,\"f\":$f,\"n\":$n,\"t_plan\":$(jt(tplan)),\"t_kernel\":$(jt(tker))," *
           "\"t_recheck\":$(jt(tre)),\"n_recheck\":$nre,\"t_colour\":$(jt(tcol)),\"t_read\":$(jt(tread))," *
           "\"mpts\":$(jt(n / tker / 1e3)),\"flagged_pct\":$(jt(100nflag / n)),\"wmax\":$(jnum(wmax))," *
           "\"fmt\":$(jstr(fmt))}"
end

function save(a)
    S.plan === nothing && error("nothing rendered yet")
    nx, ny, smin, zcap, flags, bnd = S.last
    img, w, h = colour!(S.plan, nx, ny, 1, smin, zcap, flags, bnd)
    write(get(a, "out", "/dev/shm/nyq_full.rgb"), Array(img))
    return "{\"w\":$w,\"h\":$h}"
end

parse_args(ws) = Dict(String(first(split(w, '=', limit = 2))) => String(last(split(w, '=', limit = 2)))
                      for w in ws if occursin('=', w))

# warm-up: compile the default example in the default format (others compile on first use)
let e = EXAMPLES[1]
    render(parse_args(["ex=fourth", "fmt=F16", "nx=64", "ny=36", "x0=$(e.sys.xr[1])", "x1=$(e.sys.xr[2])",
        "y0=$(e.sys.yr[1])", "y1=$(e.sys.yr[2])", "c=" * join(e.sys.c, ','), "out=" * tempname()]))
end
println("READY ", device_name())
flush(stdout)

for line in eachline(stdin)
    ws = split(strip(line))
    isempty(ws) && continue
    cmd = ws[1]
    try
        if cmd == "quit"
            break
        elseif cmd == "meta"
            println("META ", meta_json())
        elseif cmd == "render"
            println("OK ", render(parse_args(ws[2:end])))
        elseif cmd == "save"
            println("OK ", save(parse_args(ws[2:end])))
        else
            println("ERR unknown command $cmd")
        end
    catch err
        println("ERR ", replace(sprint(showerror, err), '\n' => ' ')[1:min(end, 400)])
    end
    flush(stdout)
end
