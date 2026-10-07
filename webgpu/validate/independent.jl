# Independent check of the newer examples of the WebGPU demo (shimmy, CTCR) -- not the port,
# the counts themselves:
#
#   julia --project=gpu/scripts -t auto webgpu/validate/independent.jl --cpu [--nx 160 --ny 90]
#
# reference A: the page's settings (:unwrap, Float32, tol 0.3, the page's form of D: the
#              integral(...) closed form for the shimmy), as in ref_counts.json;
# reference B: the validated ODE back-end (:bs3, Float64, tol 1e-8) on the paper's own formula
#              (shimmy: Eq. (31) as printed, 2/λ² form and 1/(L - 1 - Σ); started at ω0 = 1e-3,
#              where that form is still accurate in Float64), ω_max = 1e4 (CTCR: 1e3 -- with the
#              delays up to 6 the ODE needs too many steps for 1e4; the truncation residual of the
#              count is ~|6λ + 2λ|/|λ²|/π = 8/(π ω_max) < 0.003).
# Points where the ODE march failed (count -1) are reported separately.
# Prints the counts that differ, and those at points flagged by neither run (no root on or
# next to the line).

include(joinpath(@__DIR__, "..", "..", "gpu", "scripts", "common.jl"))
using LinearAlgebra, Random
include(joinpath(@__DIR__, "web_systems.jl"))

const NX = parse(Int, arg("nx", "160"))
const NY = parse(Int, arg("ny", "90"))

print_device()
const ONLY = arg("only", "")
for key in ("shimmy", "ctcr"), alt in (false, true)
    (isempty(ONLY) || key in split(ONLY, ',')) || continue
    s = WEB[key]
    c = alt ? s.alt : s.c
    np = s.npow(c, s.wmax)
    plan = plan_grid(s.xr, s.yr, NX, NY; backend = BACKEND, T = Float32, n_power = np, nroots = 4,
                     ω_max = s.wmax, refine = 0, certify = false, s.kw(c)...)
    run!(plan, s.D, c)
    ra = fetch_result(plan)
    Dind = key == "shimmy" ? D_shimmy_paper : s.D
    planb = plan_grid(s.xr, s.yr, NX, NY; backend = BACKEND, T = Float64, n_power = np, nroots = 4,
                      method = :bs3, tol = 1e-8, ω0 = key == "shimmy" ? 1e-3 : 1e-9, ω_max = key == "ctcr" ? 1e3 : s.wmax,
                      maxsteps = 2_000_000,
                      refine = 0, certify = false)
    tb = timed(() -> run!(planb, Dind, c))
    rb = fetch_result(planb)
    ok = rb.Z .>= 0
    nf = count(.!ok)
    nd = count((ra.Z .!= rb.Z) .& ok)
    nu = count((ra.Z .!= rb.Z) .& ok .& (((ra.flags .| rb.flags) .& 6) .== 0))
    @printf("%-8s %-4s :unwrap F32 (page form) vs :bs3 F64 tol 1e-8 (%s): %d of %d counts differ, %d unflagged, %d failed ODE marches; stable %.1f %% / %.1f %%; bs3 %.1f s\n",
            key, alt ? "alt" : "dflt", key == "shimmy" ? "Eq. (31) as printed" : "same D",
            nd, NX * NY, nu, nf, 100 * count(==(0), ra.Z) / (NX * NY), 100 * count(==(0), rb.Z) / (NX * NY), tb)
    if nu > 0
        xs = range(s.xr[1], s.xr[2]; length = NX)
        ys = range(s.yr[1], s.yr[2]; length = NY)
        for i in findall((ra.Z .!= rb.Z) .& ok .& (((ra.flags .| rb.flags) .& 6) .== 0))[1:min(end, 8)]
            @printf("    (%.4f, %.4f): unwrap %d (flags %d), bs3 %d (flags %d)\n", xs[(i - 1) % NX + 1],
                    ys[(i - 1) ÷ NX + 1], ra.Z[i], ra.flags[i], rb.Z[i], rb.flags[i])
        end
    end
end
