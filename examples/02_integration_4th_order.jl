using InterpolatedNyquist
using GLMakie
using MDBM
using LinearAlgebra

GLMakie.closeall()
GLMakie.activate!(; title="Hybrid Stability - 4th Order")

# 1. Define characteristic equation D(λ, p)
function D_chareq(λ::T, p) where T
    P, D = p
    c1 = T(0.03)
    τ  = T(0.5)
    ζ  = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 2. Hybrid Strategy Part 1: Grid sweep
Pv = LinRange(-2.01, 4.0, 60)
Dv = LinRange(-2.01, 5.0, 50)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

println("Grid sweep (4th Order)...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(Pv), length(Dv))
σ_mat_est = reshape(σ_ests_vec, length(Pv), length(Dv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Hybrid Strategy Part 2: Detailed MDBM Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(p, d)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (p, d), verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(-2.01, 4.0, 20), LinRange(-2.01, 5.0, 20)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="Stability Chart (Hybrid Approach)", xlabel="p", ylabel="d")
hm = heatmap!(ax, Pv, Dv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end

# Save figure to disk
mkpath("output_figures")
save("output_figures/example_02.png", f)
display(f)
