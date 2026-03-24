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
Pv = LinRange(-2.0, 4.0, 160)
Dv = LinRange(-2.0, 5.0, 150)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])


println("Grid sweep for background...")
# # sigma approximation based on the closest point of D curve
# @time Z_ints_vec, _, _, σ_ests_vec, _ = 
#     calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=1)


# sigma approximation based on the closest root to the imaginary axis (Z_int) and its estimated sigma (σ_est)
 @time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec_multi, ω_crits_vec = 
     calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=1, n_roots_to_track=2)
  σ_ests_vec = map(σ_ests_vec_multi) do v
     v[argmin(ifelse.(isnan.(v), Inf, abs.(v)))]
 end


    
Z_mat_int = reshape(Z_ints_vec, length(Pv), length(Dv))
σ_mat_est = reshape(σ_ests_vec, length(Pv), length(Dv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Hybrid Part 2: Multiple σ-Contours with MDBM
println("\nTracing σ-contours with MDBM...")
f = Figure(size=(1000, 800))
ax = GLMakie.Axis(f[1, 1], title="Hybrid σ-Contour Comparison", xlabel="P", ylabel="D")

hm = heatmap!(ax, Pv, Dv, C_to_plot, colormap=:viridis, alpha=0.6)
Colorbar(f[1, 2], hm, label="Stability Metric")

σ_levels = LinRange(0.0, -1.4, 10)

# Add coarse contour lines from the heatmap data to compare with MDBM
contour!(ax, Pv, Dv, C_to_plot, levels=σ_levels, labels=true, color=:white, linestyle=:dash, linewidth=1.5)

display(f)

colors = [cgrad([:red, :blue])[t] for t in LinRange(0, 1, length(σ_levels))]

for (idx, σ_loc) in enumerate(σ_levels)
    @info "Tracing σ = $σ_loc level..."
    
    function mdbm_wrapper_sigma(p, d)::Float64
        zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (p, d), σ_loc, verbosity=0)
        sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
        return sign_val * abs(es)
    end

    boundary_mdbm = MDBM_Problem(mdbm_wrapper_sigma, [LinRange(-2.0, 4.0, 20), LinRange(-2.0, 5.0, 20)])
    MDBM.solve!(boundary_mdbm, 4, verbosity=0)
    
    xyz_sol = getinterpolatedsolution(boundary_mdbm)
    DT1 = MDBM.connect(boundary_mdbm)
    edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]
    
    if !isempty(edge2plot_xyz)
        lines!(ax, edge2plot_xyz..., color=colors[idx], linewidth=2.5, label="MDBM σ = $σ_loc")
    end
display(f)
end

axislegend(ax, position=:rb)

# Save figure to disk
mkpath("output_figures")
save("output_figures/example_05.png", f)
display(f)


