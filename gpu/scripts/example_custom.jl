# Template: your own characteristic equation on the GPU.
#
#   julia --project=gpu/scripts gpu/scripts/example_custom.jl [--out DIR] [--res 1920x1080]
#
# Copy this file, edit D / C / the ranges, run. On a machine without CUDA it
# runs on the CPU backend (use a small --res there).

include(joinpath(@__DIR__, "common.jl"))

# ---------------------------------------------------------------------------
# 1. The model: D(λ, p, c) -- λ complex, p = one parameter point (any length),
#    c = constants (sliders). Use integer literals or values from c, so that a
#    Float32 kernel stays Float32. D should be ENTIRE (clear denominators).
# ---------------------------------------------------------------------------
# delayed PD control of a damped oscillator: m λ² + c λ + k + (P + Dg λ) e^{-λτ}
D(λ, p, c) = c[1] * λ^2 + c[2] * λ + c[3] + (p[1] + p[2] * λ) * exp(-c[4] * λ)
const C = (1.0, 0.1, 1.0, 1.0)          # m, damping, k, τ
const NPOW = 2                           # leading order of D (λ^2)

# ---------------------------------------------------------------------------
# 2. The points: ANY list of parameter tuples. A chart is the 2-D grid case.
# ---------------------------------------------------------------------------
nx, ny = parse_res(arg("res", ON_GPU ? "1920x1080" : "320x180"))
xs = range(-1.5, 1.0; length = nx)       # P
ys = range(-1.0, 2.0; length = ny)       # Dg
pts = grid_points(xs, ys)                # Vector{NTuple{2,Float64}}, x fastest

# ---------------------------------------------------------------------------
# 3. Sweep (Float32 on the GPU), then re-check flagged points in Float64.
# ---------------------------------------------------------------------------
print_device()
plan = plan_sweep(pts; backend = BACKEND, T = Float32, n_power = NPOW, nroots = 8,
    schedule = :queue, lanes = default_lanes())
run!(plan, D, C)                                        # compile + warm-up
t = timed(() -> run!(plan, D, C))
r = fetch_result(plan)
@printf("\n%d x %d = %d points in %.2f ms  (%.1f Mpts/s), median %d D-evaluations/point\n",
    nx, ny, nx * ny, 1e3t, nx * ny / t / 1e6, round(Int, median(r.evals)))
n = recheck!(r, D, pts; c = C, n_power = NPOW, nroots = 8, backend = BACKEND, T = Float64)
println("re-checked $n flagged points in Float64")
@printf("stable points: %.1f %%\n", 100 * count(==(0), r.Z) / length(r.Z))

# ---------------------------------------------------------------------------
# 4. Save for plotting (gpu/colab/plot_fields.py) -- same format as bench_ladder.jl
# ---------------------------------------------------------------------------
out = arg("out", joinpath(@__DIR__, "..", "results"))
mkpath(out)
base = joinpath(out, "field_custom_$(nx)x$(ny)")
m(v) = reshape(v, nx, ny)
write(base * ".f32", Float32.(colour_field((Z = m(r.Z), sigma = m(r.sigma)))))
write(base * ".i8", Int8.(clamp.(m(r.Z), -1, 127)))
open(base * ".json", "w") do io
    print(io, """{"system": "custom", "title": "delayed PD oscillator", "nx": $nx, "ny": $ny,
 "xr": [$(first(xs)), $(last(xs))], "yr": [$(first(ys)), $(last(ys))],
 "xl": "P", "yl": "D", "device": "$(device_name())", "kernel_ms": $(1e3t)}""")
end
println("saved ", base, ".{f32,i8,json}")
