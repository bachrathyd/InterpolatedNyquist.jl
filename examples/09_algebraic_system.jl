using InterpolatedNyquist
using GLMakie
using MDBM

GLMakie.closeall()
GLMakie.activate!(; title="Stability - Algebraic System")

# 1. Define characteristic equation D(λ, p)
function D_chareq(λ::T, p) where T
    a, b = p
    return λ^2 + a * λ + b * exp(-T(0.5) * λ)
end

# 2. Hybrid Strategy Part 1: Coarse Grid Sweep
av = LinRange(-1.0, 1.0, 60)
bv = LinRange(-1.0, 1.0, 50)
params_vec = vec([(av[i], bv[j]) for i in 1:length(av), j in 1:length(bv)])

println("Grid sweep using ODE solver...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(av), length(bv))
σ_mat_est = reshape(σ_ests_vec, length(av), length(bv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Hybrid Strategy Part 2: Detailed MDBM Boundary Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(a, b)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (a, b), verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(-1.0, 1.0, 30), LinRange(-1.0, 1.0, 30)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="Stability Chart (Algebraic System)", xlabel="a", ylabel="b")

hm = heatmap!(ax, av, bv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Metric (σ_est if stable, Z if unstable)")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end

# Save figure to disk
mkpath("output_figures")
save("output_figures/example_09.png", f)
display(f)
