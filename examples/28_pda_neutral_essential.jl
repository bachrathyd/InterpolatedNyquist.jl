# Example 28: Delayed PDA (acceleration feedback) control -> NEUTRAL system
#              with both "normal" Hopf-type stability boundaries AND
#              ESSENTIAL loss of stability (Z -> inf) visible on one chart.
#
# Plant: damped oscillator with full delayed PDA feedback
#   x''(t) + 2ζ x'(t) + x(t) = -P x(t-τ) - D x'(t-τ) - A x''(t-τ)
# Characteristic function (NEUTRAL delay equation due to the delayed acceleration):
#   D(λ) = λ² + 2ζλ + 1 + (P + Dλ + Aλ²) e^{-λτ}
#
# The neutral coefficient is (1 + A e^{-λτ}): for |A| > 1 the essential spectrum
#   Re(λ) -> ln(|A|)/τ > 0
# contains infinitely many unstable roots (essential instability). With a finite
# integration limit ω_max the computed Z saturates at a large but finite value
# (~ ω_max·τ/(2π)), so the stable/unstable CLASSIFICATION remains correct even
# though the count itself is truncated. The analytic essential boundary |A| = 1
# must appear in the chart alongside the regular Hopf lobes.

using InterpolatedNyquist
using GLMakie
using MDBM

GLMakie.closeall()
GLMakie.activate!(; title="PDA neutral: Hopf + essential instability")

const ZETA = 0.05
const TAU = 1.0
const D_GAIN = 0.1

function D_pda(λ::T, p) where T
    P, A = p
    return λ^2 + 2 * ZETA * λ + one(T) + (P + D_GAIN * λ + A * λ^2) * exp(-TAU * λ)
end

# ---------------------------------------------------------------------------
# 1. Coarse grid sweep across the essential boundary |A| = 1
#    (moderate ω_max: enough to classify; reltol relaxed for the stiff
#     neutral phase integrand)
# ---------------------------------------------------------------------------
Pv = LinRange(-1.5, 2.5, 90)
Av = LinRange(-1.6, 1.6, 90)
params_vec = vec([(Pv[i], Av[j]) for i in 1:length(Pv), j in 1:length(Av)])

println("Grid sweep (neutral PDA, $(length(params_vec)) points)...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec =
    calculate_unstable_roots_p_vec(D_pda, params_vec;
        ω_max=500.0, reltol=1e-4, abstol=1e-4, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(Pv), length(Av))
σ_mat_est = reshape(σ_ests_vec, length(Pv), length(Av))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

println("Z range on the grid: ", extrema(Z_mat_int),
    "  (large values = essential instability region)")

# ---------------------------------------------------------------------------
# 2. MDBM boundary trace (finds both Hopf lobes and the essential boundary)
# ---------------------------------------------------------------------------
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(pp, aa)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_pda, (pp, aa);
        ω_max=500.0, reltol=1e-4, abstol=1e-4)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(-1.5, 2.5, 30), LinRange(-1.6, 1.6, 30)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# ---------------------------------------------------------------------------
# 3. Plot: cap the color scale so the Hopf structure stays visible next to
#    the huge Z of the essential region; overlay the analytic |A| = 1 lines.
# ---------------------------------------------------------------------------
Z_CAP = 10
C_capped = clamp.(C_to_plot, -2.0, Z_CAP)

f = Figure(size=(1000, 650))
ax = GLMakie.Axis(f[1, 1], title="Delayed PDA control (neutral): Hopf + essential instability",
    xlabel="P (proportional gain)", ylabel="A (acceleration gain)")
hm = heatmap!(ax, Pv, Av, C_capped, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Z (capped at $Z_CAP) / σ_est (stable)")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM boundary")
end
hlines!(ax, [1.0, -1.0], color=:red, linestyle=:dash, linewidth=2,
    label="analytic essential boundary |A| = 1")
axislegend(ax, position=:rt)

mkpath("output_figures")
save("output_figures/example_28.png", f)
display(f)
