using InterpolatedNyquist
using GLMakie
using MDBM
using LinearAlgebra

GLMakie.closeall()
GLMakie.activate!(; title="Hybrid σ-Contour Comparison")

# 1. Define characteristic equation D(λ, p)
function D_chareq(λ::T, p) where T
    P, D = p
    c1 = T(0.03)
    τ  = T(0.5)
    ζ  = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 2. Hybrid Part 1: Coarse sweep for background
Pv = LinRange(-2.0, 4.0, 60)
Dv = LinRange(-2.0, 5.0, 50)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

println("Grid sweep for background...")
@time Z_ints_vec, _, _, σ_ests_vec, _ = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=1)
Z_mat_int = reshape(Z_ints_vec, length(Pv), length(Dv))
σ_mat_est = reshape(σ_ests_vec, length(Pv), length(Dv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Hybrid Part 2: Multiple σ-Contours with MDBM
println("\nTracing σ-contours with MDBM...")
f = Figure(size=(1000, 800))
ax = GLMakie.Axis(f[1, 1], title="Hybrid σ-Contour Comparison", xlabel="P", ylabel="D")

hm = heatmap!(ax, Pv, Dv, C_to_plot, colormap=:viridis, alpha=0.6)
Colorbar(f[1, 2], hm, label="Stability Metric")

σ_levels = LinRange(0.0, -1.4, 5)
colors = [:black, :blue, :green, :orange, :red]

for (idx, σ_loc) in enumerate(σ_levels)
    @info "Tracing σ = $σ_loc level..."
    
    function mdbm_wrapper_sigma(p, d)::Float64
        zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (p, d), σ_loc, verbosity=0)
        sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
        return sign_val * abs(es)
    end

    boundary_mdbm = MDBM_Problem(mdbm_wrapper_sigma, [LinRange(-2.0, 4.0, 20), LinRange(-2.0, 5.0, 20)])
    MDBM.solve!(boundary_mdbm, 3, verbosity=0)
    
    xyz_sol = getinterpolatedsolution(boundary_mdbm)
    DT1 = MDBM.connect(boundary_mdbm)
    edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]
    
    if !isempty(edge2plot_xyz)
        lines!(ax, edge2plot_xyz..., color=colors[idx], linewidth=2.5, label="MDBM σ = $σ_loc")
    end
end

axislegend(ax, position=:rb)

# Save figure to disk
mkpath("output_figures")
save("output_figures/example_05.png", f)
display(f)

