using InterpolatedNyquist
using CairoMakie
using MDBM
using LinearAlgebra

# GLMakie.closeall()
# GLMakie.activate!(; title="Stability - Neutral Delay System")

# 1. Define characteristic equation D(λ, p)
# Neutral system: x''(t) + a*x''(t-τ) + b*x(t) + c*x(t-τ) = 0
# D(λ) = λ^2 + a*λ^2*exp(-τ*λ) + b + c*exp(-τ*λ)
function D_chareq(λ::T, p) where T
    a, c = p
    b = T(1.0) # fix b
    τ = T(1.0)
    
    return λ^2 + a * λ^2 * exp(-τ * λ) + b + c * exp(-τ * λ)
end

# 2. Hybrid Strategy Part 1: Coarse Grid Sweep
av = LinRange(-0.9, 0.9, 60) # Neutral systems are unstable for |a| >= 1
cv = LinRange(-2.0, 2.0, 50)
params_vec = vec([(av[i], cv[j]) for i in 1:length(av), j in 1:length(cv)])

println("Grid sweep (Neutral System)...")
# Neutral systems can be very stiff, we use a slightly larger tolerance for the grid
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, reltol=1e-4, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(av), length(cv))
σ_mat_est = reshape(σ_ests_vec, length(av), length(cv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Hybrid Strategy Part 2: Detailed MDBM Boundary Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(a, c)::Float64
    # For neutral systems, we must be careful with stiffness
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (a, c), verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

# MDBM focusing on the boundary
boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(-0.9, 0.9, 20), LinRange(-2.0, 2.0, 20)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = Axis(f[1, 1], title="Neutral Delay System Stability", xlabel="a (neutral term)", ylabel="c (delayed stiffness)")

hm = heatmap!(ax, av, cv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end

mkpath("output_figures")
save("output_figures/example_12.png", f)
# display(f)
