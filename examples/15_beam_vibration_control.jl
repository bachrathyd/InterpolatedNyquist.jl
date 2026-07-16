using InterpolatedNyquist
using GLMakie
using MDBM

# 1. Infinite-Dimensional Beam Characteristic Equation
# Longitudinal bar: clamped at x=0, controlled at x=L, no spatial discretization.
#
# TWO modelling choices matter, and both are made deliberately here:
#
# (a) KELVIN-VOIGT (material) damping, EA(1 + c d/dt) u'' = ρA u_tt, so
#     γ = λ / sqrt(1 + c λ)  (NOT sqrt(λ² + c λ), which is external/viscous drag).
#     With viscous damping every mode is damped equally and the roots pile up
#     on a vertical line -> the count is 0 or ∞ (a neutral-type obstruction).
#     Kelvin-Voigt damps high modes ever harder (Re λ_k -> -∞), so only finitely
#     many roots sit near the axis and the count is well posed. Real material
#     damping IS rate-dependent, so this is also the physically correct model.
#
# (b) RETURN-DIFFERENCE form D = 1 + Kp e^{-λτ} sech(γL) rather than the raw
#     cosh(γL) + Kp e^{-λτ}. Same zeros; the added poles are the open-loop roots
#     (left half-plane), and D -> 1 at infinity so the leading order is exactly 0.
function D_chareq(λ::T, p) where T
    Kp, τ_val = p
    L = T(1.0)
    c_damping = T(0.05)                       # Kelvin-Voigt coefficient
    γ = λ / sqrt(one(T) + c_damping * λ)
    return one(T) + Kp * exp(-λ * τ_val) / cosh(γ * L)
end

# 2. Hybrid Strategy
Kpv = LinRange(0.0, 2.0, 60)
tauv = LinRange(0.1, 3.0, 50)
params_vec = vec([(Kpv[i], tauv[j]) for i in 1:length(Kpv), j in 1:length(tauv)])

println("Grid sweep (Infinite DOF Beam Model)...")
# A moderate ω_max suffices: the Kelvin-Voigt damping suppresses the high
# beam modes, so only the low-frequency ones can destabilize
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, ω_max=200.0, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(Kpv), length(tauv))
σ_mat_est = reshape(σ_ests_vec, length(Kpv), length(tauv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. MDBM Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(Kp, τ_val)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (Kp, τ_val), ω_max=200.0, verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(0.0, 2.0, 15), LinRange(0.1, 3.0, 15)])
@time MDBM.solve!(boundary_mdbm, 3, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="Beam Vibration Control: Transcendental Stability Chart", 
    xlabel="Kp (Gain)", ylabel="τ (Delay)")

hm = heatmap!(ax, Kpv, tauv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end

mkpath("output_figures")
save("output_figures/example_15.png", f)
display(f)
