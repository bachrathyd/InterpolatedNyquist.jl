# Reference counts for the WebGPU demo (webgpu/index.html?validate=1).
#
#   julia --project=gpu/scripts -t auto webgpu/validate/ref_counts.jl --cpu [--nx 160 --ny 90]
#         [--only key1,key2]   (recompute some examples, keep the others of the existing json)
#         [--out file.json]
#
# Runs the repository engine (NyquistGPU, :unwrap march, first-order root estimate, no
# refinement / certification -- the settings the web page ports) on the CPU backend for every
# example of the page (gpu/scripts/systems.jl: fourth, showcase, turning; web_systems.jl: the
# appendix gallery A.2-A.6, A.8, A.9, A.11, A.12), with the default constants and with a second
# constant set ("<key>@alt"), in Float32 (the precision of the web page) and Float64, and writes
# webgpu/validate/ref_counts.json:
#   { nx, ny, wmax, device, examples: { name: { sys, xr, yr, c, npow, wmax, kw, Z32, Z64d, S32,
#     F32, cert, roots } } }
# Z32 are the counts Z (-1 where the march failed; Z64d: [[i, Z64]] where Float64 differs), S32 the rightmost-root estimate σ of the
# Float32 run (null where none), F32 its flags, all column-major nx*ny (x fastest, row 0 =
# yr[1]). npow is the leading order passed to the march -- for the bar models the effective
# order 2Φ_den(ω_max)/π of their stable denominator, measured here by a Float64 march.
# roots: [[i, σ, ω], ...] reference dominant roots at 16 stable points (Z64 = 0): Float64,
# 15 tracked minima, Newton-polished (refine = 6), and -- where `cert` -- certified by
# counting on shifted lines (σtol = 1e-6); the page compares its 'exact root' mode with them.
#
# The bar models (rod, fem) are also counted in Float64 in the paper's own formulation
# (s08_gallery.jl: cosh γ - K e^{-rλ} with the measured phase of cosh γ, resp. the LU form of
# the FEM determinant) and the counts compared with the scale-free forms used here.

include(joinpath(@__DIR__, "..", "..", "gpu", "scripts", "common.jl"))
using LinearAlgebra, Random
include(joinpath(@__DIR__, "web_systems.jl"))

const NX = parse(Int, arg("nx", "160"))
const NY = parse(Int, arg("ny", "90"))
const WMAX = 1e5                     # the original three examples: the server's Float32 format
const ONLY = let s = arg("only", "")
    isempty(s) ? nothing : split(s, ',')
end
const KEYS = ["fourth", "showcase", "turning", "algebraic", "distributed", "neutral",
              "neutral_hg", "pda", "rod", "fem", "frac", "gao"]
# a second constant set for the original three (the slider knobs moved off their defaults)
const ALT = Dict("fourth" => (0.03, 0.08, 1.0),                       # ζ, τ changed
                 "showcase" => (1.0, 0.5, -1.0, 1.0, 0.2, 0.1, 1.0),   # c₁, c₂, τ changed
                 "turning" => (0.05, 1.0, 0.03, 3.0, 2π))              # ζ₁, A₂, ω₂ changed

jn(x) = isfinite(x) ? string(round(Float64(x); sigdigits = 7)) : "null"
jv(v) = "[" * join(jn.(v), ",") * "]"
ji(v) = "[" * join(string.(v), ",") * "]"
# the Float64 counts as a list of the points where they differ from Float32: [[i (0-based), Z64], ...]
z64d(z32, z64) = "[" * join(["[$(i - 1),$(z64[i])]" for i in eachindex(z32) if z32[i] != z64[i]], ",") * "]"

"Effective order n_eff = 2Φ_den(ω_max)/π of a stable denominator (Float64 march, one point)."
function neff(Dden, c, wmax, kw)
    r = sweep(Dden, [(0.0, 0.0)]; c = c, backend = CPU(), T = Float64, n_power = 0.0, nroots = 1,
              ω_max = wmax, kw...)
    r.flags[1] & 1 == 0 || error("denominator march failed")
    return -2 * Float64(r.Zraw[1])
end

"Unified view of an example: (D, c, xr, yr, wmax, npow, kw, cert)."
function spec(key, alt)
    if haskey(SYSTEMS, key)
        s = SYSTEMS[key]
        c = alt ? ALT[key] : s.c
        return (D = s.D, c = c, xr = s.xr, yr = s.yr, wmax = WMAX, npow = Float64(s.npow),
                kw = s.kw, cert = true)
    end
    s = WEB[key]
    c = alt ? s.alt : s.c
    kw = s.kw(c)
    np = s.npow === nothing ? neff(s.den, c, s.wmax, kw) : Float64(s.npow(c, s.wmax))
    return (D = s.D, c = c, xr = s.xr, yr = s.yr, wmax = s.wmax, npow = np, kw = kw, cert = s.cert)
end

function counts(sp, T; nroots = 4, D = sp.D, npow = sp.npow, c = sp.c)
    plan = plan_grid(sp.xr, sp.yr, NX, NY; backend = BACKEND, T = T, n_power = npow,
                     nroots = nroots, ω_max = sp.wmax, refine = 0, certify = false, sp.kw...)
    run!(plan, D, c)                                      # compile
    t = timed(() -> run!(plan, D, c))
    return fetch_result(plan), t
end

"Reference dominant roots at 16 stable points (deterministic choice)."
function ref_roots(sp, Z64, F64)
    xs = range(sp.xr[1], sp.xr[2]; length = NX)
    ys = range(sp.yr[1], sp.yr[2]; length = NY)
    cand = findall(i -> Z64[i] == 0 && F64[i] == 0, eachindex(Z64))
    isempty(cand) && return Tuple{Int, Float64, Float64}[]
    idx = sort(shuffle(MersenneTwister(1), cand)[1:min(16, length(cand))])
    pts = [(xs[(i - 1) % NX + 1], ys[(i - 1) ÷ NX + 1]) for i in idx]
    r = sweep(sp.D, pts; c = sp.c, backend = CPU(), T = Float64, n_power = sp.npow,
              nroots = 15, ω_max = sp.wmax, refine = 6, certify = sp.cert, σtol = 1e-6, sp.kw...)
    return [(idx[k] - 1, Float64(r.sigma[k]), Float64(r.omega[k])) for k in eachindex(idx)
            if isfinite(r.sigma[k])]
end

kwjson(kw) = join(["\"$(k == :ωband ? "wband" : string(k))\":$(jn(v))" for (k, v) in pairs(kw)], ",")

# the paper's own formulation of the bar models (Float64): counts must agree
function paper_check(key, sp)
    if key == "rod"
        sp.c[2] == 0 || return
        n = neff(D_rod_paper_den, sp.c, sp.wmax, sp.kw)
        r, _ = counts(sp, Float64; D = D_rod_paper, npow = n)
    elseif key == "fem"
        num, den = make_fem_paper(sp.c[1], round(Int, sp.c[2]))
        n = neff(den, (), sp.wmax, sp.kw)
        r, _ = counts(sp, Float64; D = num, npow = n, c = ())
    else
        return
    end
    r2, _ = counts(sp, Float64)
    nd = count(r.Z .!= r2.Z)
    nu = count((r.Z .!= r2.Z) .& (((r.flags .| r2.flags) .& 6) .== 0))
    @printf("    paper formulation (%s, n_eff = %.6f vs %.6f): %d of %d counts differ, %d of them at unflagged points\n",
            key, n, sp.npow, nd, NX * NY, nu)
end

print_device()
old = Dict{String, String}()
out = arg("out", joinpath(@__DIR__, "ref_counts.json"))
if ONLY !== nothing && isfile(out)
    # keep the other examples of the existing file (crude split on the top-level entries)
    s = read(out, String)
    for m in eachmatch(r"\"([a-z_]+(?:@alt)?)\":\{\"sys\"", s)
        st = m.offset
        depth = 0
        j = findnext('{', s, st)               # the entry's opening brace
        while true
            ch = s[j]
            ch == '{' && (depth += 1)
            ch == '}' && (depth -= 1)
            depth == 0 && break
            j = nextind(s, j)
        end
        old[m.captures[1]] = s[st:j]
    end
end
parts = String[]
for key in KEYS, alt in (false, true)
    name = alt ? key * "@alt" : key
    if ONLY !== nothing && !(key in ONLY)
        haskey(old, name) && push!(parts, old[name])
        continue
    end
    sp = spec(key, alt)
    r32, t32 = counts(sp, Float32)
    r64, t64 = counts(sp, Float64)
    nd = count(r32.Z .!= r64.Z)
    nu = count((r32.Z .!= r64.Z) .& (((r32.flags .| r64.flags) .& 6) .== 0))
    st = r64.Z .== 0
    sq = sort(filter(isfinite, r32.sigma[st .& (r32.Z .== 0)]))
    q(p) = isempty(sq) ? NaN : sq[clamp(round(Int, p * length(sq)), 1, length(sq))]
    @printf("%-15s n=%.4f  F32 %.2f s  F64 %.2f s  F32 vs F64: %d of %d counts differ (%.3f %%; %d unflagged), flagged %d, stable %.1f %%, σ q02 %.3g q50 %.3g, evals/pt %.0f\n",
            name, sp.npow, t32, t64, nd, NX * NY, 100nd / (NX * NY), nu, count(!=(0), r32.flags),
            100 * count(st) / (NX * NY), q(0.02), q(0.5), sum(r32.evals) / length(r32.evals))
    paper_check(key, sp)
    roots = ref_roots(sp, r64.Z, r64.flags)
    @printf("    reference roots: %d, σ ∈ [%.4g, %.4g]%s\n", length(roots),
            isempty(roots) ? NaN : minimum(r[2] for r in roots),
            isempty(roots) ? NaN : maximum(r[2] for r in roots), sp.cert ? " (certified)" : " (Newton)")
    rj = "[" * join(["[$(r[1]),$(jn(r[2])),$(jn(r[3]))]" for r in roots], ",") * "]"
    push!(parts, "\"$name\":{\"sys\":\"$key\",\"xr\":$(jv(collect(sp.xr))),\"yr\":$(jv(collect(sp.yr)))," *
                 "\"c\":$(jv(collect(sp.c))),\"npow\":$(jn(sp.npow)),\"wmax\":$(jn(sp.wmax)),\"kw\":{$(kwjson(sp.kw))}," *
                 "\"cert\":$(sp.cert),\"roots\":$rj," *
                 "\"Z32\":$(ji(r32.Z)),\"Z64d\":$(z64d(r32.Z, r64.Z)),\"S32\":$(jv(r32.sigma))," *
                 "\"F32\":$(ji(Int.(r32.flags)))}")
end
open(out, "w") do io
    print(io, "{\"nx\":$NX,\"ny\":$NY,\"wmax\":$WMAX,\"device\":\"$(device_name())\",",
          "\"examples\":{", join(parts, ","), "}}")
end
println("wrote ", out)
