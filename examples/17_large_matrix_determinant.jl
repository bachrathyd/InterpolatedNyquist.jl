using InterpolatedNyquist
using LinearAlgebra
using GLMakie
using MDBM

# 1. Define Ultra-High DOF System (50x50 Matrix)
# Reduced from 100x100 to allow reasonable hybrid execution time
const N_size = 50
const M_base = rand(N_size, N_size)
const M_mat = M_base' * M_base + I
const K_base = rand(N_size, N_size)
const K_mat = K_base' * K_base + I

function D_chareq(λ::T, p) where T
    gain, τ = p
    Q = λ^2 .* T.(M_mat) .+ T.(K_mat) .+ gain * exp(-λ * τ) .* I(N_size)
    return det(Q)
end

# 2. Hybrid Strategy Part 1: Coarse Grid Sweep (Very coarse for this heavy model)
gv = LinRange(0.0, 2.0, 20)
tv = LinRange(0.1, 1.0, 15)
params_vec = vec([(gv[i], tv[j]) for i in 1:length(gv), j in 1:length(tv)])

println("Grid sweep (Ultra-High DOF Matrix)...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, ω_max=20.0, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(gv), length(tv))
σ_mat_est = reshape(σ_ests_vec, length(gv), length(tv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Hybrid Strategy Part 2: Detailed MDBM Boundary Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(gain, τ)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (gain, τ), ω_max=20.0, verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(0.0, 2.0, 10), LinRange(0.1, 1.0, 10)])
@time MDBM.solve!(boundary_mdbm, 3, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="Ultra-High DOF Matrix Stability", xlabel="Gain", ylabel="τ (Delay)")
hm = heatmap!(ax, gv, tv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end

mkpath("output_figures")
save("output_figures/example_17.png", f)
display(f)
