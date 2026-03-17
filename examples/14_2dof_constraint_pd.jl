using InterpolatedNyquist
using LinearAlgebra
using StaticArrays
using GLMakie
using MDBM

# 1. Define 2DOF System with Delayed PD Control
function D_chareq(λ::T, p) where T
    P, D = p
    τ = T(0.2)
    
    M = @SMatrix [T(1.0) T(0.0); T(0.0) T(0.5)]
    C = @SMatrix [T(0.1) T(-0.05); T(-0.05) T(0.1)]
    K = @SMatrix [T(2.0) T(-1.0); T(-1.0) T(1.0)]
    
    Q = λ^2 * M + λ * C + K
    Q_delayed = @SMatrix [(P + D * λ) * exp(-τ * λ) T(0.0); T(0.0) T(0.0)]
    
    return det(Q + Q_delayed)
end

# 2. Hybrid Strategy Part 1: Coarse Grid Sweep
Pv = LinRange(0.0, 10.0, 60)
Dv = LinRange(0.0, 2.0, 50)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

println("Grid sweep (2DOF Matrix Determinant)...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(Pv), length(Dv))
σ_mat_est = reshape(σ_ests_vec, length(Pv), length(Dv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Hybrid Strategy Part 2: Detailed MDBM Boundary Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(p, d)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (p, d), verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(0.0, 10.0, 30), LinRange(0.0, 2.0, 30)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="2DOF System: Stability Chart", xlabel="P (Gain)", ylabel="D (Damping)")
hm = heatmap!(ax, Pv, Dv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end

mkpath("output_figures")
save("output_figures/example_14.png", f)
display(f)
