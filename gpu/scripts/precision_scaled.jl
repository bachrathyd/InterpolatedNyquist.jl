# Narrow number formats with a rescaled frequency: Float64 ... Float8, Float4.
#
# Mixed precision: the march (frequency, step control, phase sum) runs in
# Float32, the characteristic function is evaluated in the format under test
# (`Teval`). Float16 is native; BFloat16, Float8 (E5M2, E4M3) and Float4 (E2M1)
# are emulated bit-accurately (every operation rounded to the format) -- GPUs
# have no scalar Float8/Float4 units, so for those only the ACCURACY is
# meaningful here, not the time.
#
# Scaling (4th-order benchmark, D ~ c1 λ⁴): μ = λ/Ω, D̂(μ) = D(Ωμ)/(c1 Ω⁴).
# Over the march the values of D̂ range from the constant term k0 = 1/(c1 Ω⁴)
# (ω -> 0) up to (ω_max/Ω)⁴ (ω = ω_max). Strategies for Ω:
#   raw      -- no scaling (Ω = 1, D itself)
#   ωmax     -- Ω = ω_max: every power μ^k ≤ 1, largest value 1
#   fitmax   -- Ω such that the largest value (ω_max/Ω)⁴ = floatmax/4
#   centered -- the value range [k0, (ω_max/Ω)⁴] centred (geometrically) in
#               [floatmin, floatmax] of the format
#
# Run:  julia --project=gpu/scripts -t auto gpu/scripts/precision_scaled.jl [--res 400] [--wmax 15]

include(joinpath(@__DIR__, "common.jl"))
sys = SYSTEMS["fourth"]
nx, ny = parse_res(arg("res", ON_GPU ? "1920x1080" : "400"))
const WMAX = parse(Float64, arg("wmax", "15"))
pts = grid_points(range(sys.xr...; length = nx), range(sys.yr...; length = ny))
const OUT = arg("out", joinpath(@__DIR__, "..", "results", "maps"))
mkpath(OUT)
print_device()

const FORMATS = [
    ("Float64", Float64, Float64, true), ("Float32", Float32, Float32, true),
    ("Float16", Float32, Float16, true), ("BFloat16 (emu)", Float32, BFloat16_emu, false),
    ("Float8 E5M2 (emu)", Float32, Float8_E5M2, false), ("Float8 E4M3 (emu)", Float32, Float8_E4M3, false),
    ("Float4 E2M1 (emu)", Float32, Float4_E2M1, false)]

function omega_for(strategy, TE; n = 4, an = 0.03, a0 = 1.0)
    fmax, fmin = format_range(TE)
    fmax = min(fmax, 1e30); fmin = max(fmin, 1e-30)
    strategy === :ωmax && return WMAX
    strategy === :fitmax && return WMAX / (fmax / 4)^(1 / n)
    # centered: top·bottom = fmax·fmin  with top = (ωmax/Ω)^n, bottom = a0/(an Ω^n)
    return (WMAX^n * a0 / (an * fmax * fmin))^(1 / (2n))
end

function run_case(T, TE, strategy; native)
    if strategy === :raw
        D, c, Ω = sys.D, sys.c, 1.0
    else
        Ω = omega_for(strategy, TE)
        D, c = D_fourth_scaled, fourth_scaled_consts(Ω)
    end
    plan = plan_sweep(pts; backend = BACKEND, T = T, Teval = TE, n_power = 4, nroots = 4,
        ω_max = WMAX / Ω, ω0 = 1e-9 / Ω, h0 = 1e-2 / Ω, maxsteps = 4000, lanes = default_lanes())
    run!(plan, D, c)
    t = timed(() -> run!(plan, D, c))
    return t, fetch_result(plan), Ω
end

ref = sweep(sys.D, pts; c = sys.c, backend = BACKEND, T = Float64, n_power = 4, lanes = default_lanes())
@printf("\n4th-order chart %dx%d, ω_max = %g, reference: Float64, ω_max = 1e5\n", nx, ny, WMAX)
@printf("%-19s %-9s %10s | %10s %8s %9s %8s %8s %6s\n", "format", "scaling", "Ω",
    "wrong", "unflag.", "flagged", "failed", "time ms", "evals")
rows = String["format,scaling,Omega,wrong_pct,wrong_unflagged_pct,flagged_pct,failed_pct,time_ms,evals_med,native"]
errmaps = Dict{String, BitMatrix}()
for (name, T, TE, native) in FORMATS, strategy in (:raw, :ωmax, :fitmax, :centered)
    (TE in (Float64, Float32) && strategy !== :raw) && continue     # no need to scale
    t, r, Ω = run_case(T, TE, strategy; native)
    N = length(r.Z)
    wrong = r.Z .!= ref.Z
    unfl = wrong .& (r.flags .== 0)
    @printf("%-19s %-9s %10.3g | %9.3f%% %7.3f%% %8.2f%% %7.2f%% %8.1f %6d%s\n", name, strategy, Ω,
        100count(wrong) / N, 100count(unfl) / N, 100count(!=(0), r.flags) / N,
        100count(==(-1), r.Z) / N, 1e3t, round(Int, median(r.evals)), native ? "" : "  (emulated)")
    push!(rows, join([name, strategy, Ω, 100count(wrong) / N, 100count(unfl) / N,
        100count(!=(0), r.flags) / N, 100count(==(-1), r.Z) / N, 1e3t, median(r.evals), native], ','))
    strategy === :centered && (errmaps[name] = reshape(wrong, nx, ny))
end
open(joinpath(OUT, "precision_scaled_$(nx)x$(ny).csv"), "w") do io
    foreach(l -> println(io, l), rows)
end
# error maps (centered scaling) for plotting: wrong pixels per format
for (name, m) in errmaps
    tag = replace(name, r"[^A-Za-z0-9]+" => "_")
    write(joinpath(OUT, "wrong_$(tag)_$(nx)x$(ny).u8"), UInt8.(m))
end
save_field(OUT, sys, (Z = reshape(ref.Z, nx, ny), sigma = reshape(ref.sigma, nx, ny)), nx, ny, 0.0;
    tag = "_precref")
println("written: ", joinpath(OUT, "precision_scaled_$(nx)x$(ny).csv"))
