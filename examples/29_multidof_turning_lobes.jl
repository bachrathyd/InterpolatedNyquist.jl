# Example 29: Multi-DOF regenerative turning -- stability lobe diagram
#
# Classic regenerative machining model with a MULTI-MODE flexible structure.
# The oriented FRF of the structure is a modal sum
#   G(λ) = Σ_k g_k / (λ² + 2 ζ_k ω_k λ + ω_k²)
# and the regenerative chip-thickness variation yields the characteristic function
#   D(λ) = 1 + w (1 - e^{-λτ}) G(λ),      τ = 2π/Ω  (one revolution)
# with chip-width (depth of cut) w and spindle speed Ω.
#
# Two features matter for the paper:
#  * D(λ) is RATIONAL (structural poles in the open left half-plane), showing
#    that the method needs no polynomial form: at high frequency D -> 1, so the
#    estimated leading order n ≈ 0 and the integrand decays fast.
#  * Toward LOW spindle speed the lobes become infinitely dense: methods that
#    trace all root-crossing D-curves waste effort there, because almost all
#    curves lie deep inside the unstable domain. The direct boundary tracing
#    via the sign(Z==0)·|σ_est| objective only follows the true stability limit.

using InterpolatedNyquist
using GLMakie
using MDBM

GLMakie.closeall()
GLMakie.activate!(; title="Multi-DOF turning stability lobes")

# Modal parameters (normalized): two dominant modes
const OM = (1.0, 2.4)      # natural angular frequencies
const ZE = (0.02, 0.03)    # damping ratios
const GK = (1.0, 0.45)     # modal gains (oriented)

function D_turning(λ::T, p) where T
    Ω, w = p
    τ = 2π / Ω
    G = GK[1] / (λ^2 + 2 * ZE[1] * OM[1] * λ + OM[1]^2) +
        GK[2] / (λ^2 + 2 * ZE[2] * OM[2] * λ + OM[2]^2)
    return one(T) + w * (1 - exp(-τ * λ)) * G
end

# ---------------------------------------------------------------------------
# 1. Coarse grid sweep
# ---------------------------------------------------------------------------
Ωv = LinRange(0.08, 1.2, 120)
wv = LinRange(0.01, 1.2, 80)   # w = 0 makes D constant (no dynamics)
params_vec = vec([(Ωv[i], wv[j]) for i in 1:length(Ωv), j in 1:length(wv)])

println("Grid sweep (multi-mode turning, $(length(params_vec)) points)...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec =
    calculate_unstable_roots_p_vec(D_turning, params_vec; ω_max=1e4, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(Ωv), length(wv))
σ_mat_est = reshape(σ_ests_vec, length(Ωv), length(wv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# ---------------------------------------------------------------------------
# 2. MDBM boundary trace (only the true stability limit, no D-curve clutter)
# ---------------------------------------------------------------------------
println("\nTracing stability lobes with MDBM...")
function mdbm_wrapper(Ω, w)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_turning, (Ω, w); ω_max=1e4)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    g = sign_val * abs(es)
    return isfinite(g) ? g : sign_val * 1.0e3
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(0.08, 1.2, 40), LinRange(0.01, 1.2, 30)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------
f = Figure(size=(1100, 650))
ax = GLMakie.Axis(f[1, 1], title="Multi-DOF turning: stability lobe diagram",
    xlabel="spindle speed Ω", ylabel="chip width w")
hm = heatmap!(ax, Ωv, wv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Z (unstable) / σ_est (stable)")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=1.5, label="MDBM boundary")
end

mkpath("output_figures")
save("output_figures/example_29.png", f)
display(f)
