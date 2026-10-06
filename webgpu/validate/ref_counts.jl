# Reference counts for the WebGPU demo (webgpu/index.html?validate=1).
#
#   julia --project=gpu/scripts -t auto webgpu/validate/ref_counts.jl --cpu [--nx 160 --ny 90]
#
# Runs the repository engine (NyquistGPU, :unwrap march, first-order root estimate, no
# refinement / certification -- the settings the web page ports) on the CPU backend for the
# three time-independent examples of gpu/scripts/systems.jl, with their default constants,
# and with a second constant set ("<key>@alt"), in Float32 (the precision of the web page)
# and Float64, and writes
# webgpu/validate/ref_counts.json:
#   { nx, ny, wmax, examples: { name: { sys, xr, yr, c, npow, kw, Z32, Z64, S32, F32 } } }
# Z* are the counts Z (-1 where the march failed), S32 the rightmost-root estimate σ of the
# Float32 run (null where none), F32 its flags, all column-major nx*ny (x fastest, row 0 = yr[1]).

include(joinpath(@__DIR__, "..", "..", "gpu", "scripts", "common.jl"))

const NX = parse(Int, arg("nx", "160"))
const NY = parse(Int, arg("ny", "90"))
const WMAX = 1e5                     # the server's Float32 format (exact, ω_max = 1e5)
const KEYS = ["fourth", "showcase", "turning"]
# a second constant set per example (the slider knobs of gpu/interactive/server.jl moved off
# their defaults), stored as "<key>@alt"
const ALT = Dict("fourth" => (0.03, 0.08, 1.0),                       # ζ, τ changed
                 "showcase" => (1.0, 0.5, -1.0, 1.0, 0.2, 0.1, 1.0),   # c₁, c₂, τ changed
                 "turning" => (0.05, 1.0, 0.03, 3.0, 2π))              # ζ₁, A₂, ω₂ changed

jn(x) = isfinite(x) ? string(round(Float64(x); sigdigits = 7)) : "null"
jv(v) = "[" * join(jn.(v), ",") * "]"
ji(v) = "[" * join(string.(v), ",") * "]"

function counts(sys, c, T)
    plan = plan_grid(sys.xr, sys.yr, NX, NY; backend = BACKEND, T = T, n_power = sys.npow,
                     nroots = 4, ω_max = WMAX, refine = 0, certify = false, sys.kw...)
    run!(plan, sys.D, c)                                  # compile
    t = timed(() -> run!(plan, sys.D, c))
    return fetch_result(plan), t
end

print_device()
parts = String[]
for key in KEYS, alt in (false, true)
    sys = SYSTEMS[key]
    c = alt ? ALT[key] : sys.c
    r32, t32 = counts(sys, c, Float32)
    r64, t64 = counts(sys, c, Float64)
    name = alt ? key * "@alt" : key
    nd = count(r32.Z .!= r64.Z)
    @printf("%-13s  F32 %.3f s  F64 %.3f s   F32 vs F64: %d of %d counts differ (%.3f %%), F32 flagged %d\n",
            name, t32, t64, nd, NX * NY, 100nd / (NX * NY), count(!=(0), r32.flags))
    kw = join(["\"$k\":$(jn(v))" for (k, v) in pairs(sys.kw)], ",")
    push!(parts, "\"$name\":{\"sys\":\"$key\",\"xr\":$(jv(collect(sys.xr))),\"yr\":$(jv(collect(sys.yr)))," *
                 "\"c\":$(jv(collect(c))),\"npow\":$(sys.npow),\"kw\":{$kw}," *
                 "\"Z32\":$(ji(r32.Z)),\"Z64\":$(ji(r64.Z)),\"S32\":$(jv(r32.sigma))," *
                 "\"F32\":$(ji(Int.(r32.flags)))}")
end
out = joinpath(@__DIR__, "ref_counts.json")
open(out, "w") do io
    print(io, "{\"nx\":$NX,\"ny\":$NY,\"wmax\":$WMAX,\"device\":\"$(device_name())\",",
          "\"examples\":{", join(parts, ","), "}}")
end
println("wrote ", out)
