using InterpolatedNyquist
using GLMakie
using MDBM

# 1. Define High-Gain Neutral System
# D(λ) = λ^2 + a*λ^2*exp(-τλ) + b*λ + c*exp(-τλ)
# Neutral systems are extremely sensitive near |a| = 1.
function D_chareq(λ::T, p) where T
    a, c = p
    b = T(5.0) # Higher damping
    τ = T(1.0)
    return λ^2 + a * λ^2 * exp(-τ * λ) + b * λ + c * exp(-τ * λ)
end

# 2. Hybrid Strategy
av = LinRange(-0.95, 0.95, 60)
cv = LinRange(-10.0, 10.0, 50)
params_vec = vec([(av[i], cv[j]) for i in 1:length(av), j in 1:length(cv)])

println("Grid sweep (High-Gain Neutral)...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, reltol=1e-4, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(av), length(cv))
σ_mat_est = reshape(σ_ests_vec, length(av), length(cv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. MDBM Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(a, c)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (a, c), reltol=1e-4, verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(-0.95, 0.95, 20), LinRange(-10.0, 10.0, 20)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="High-Gain Neutral System Stability", xlabel="a (neutral term)", ylabel="c (stiffness)")
hm = heatmap!(ax, av, cv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end

mkpath("output_figures")
save("output_figures/example_20.png", f)
display(f)
