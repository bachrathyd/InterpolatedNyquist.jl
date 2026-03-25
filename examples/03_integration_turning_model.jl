using InterpolatedNyquist
using GLMakie
using MDBM
using LinearAlgebra
using GeometryBasics

GLMakie.closeall()
GLMakie.activate!(; title="Hybrid Stability - Turning Model")

# 1. Define characteristic equation D(λ, p)
function D_chareq(λ::T, p) where T
    invΩ, w = p
    τ = T(2π) * invΩ
    ζ = T(0.01)
    H = one(T) / (λ^2 + T(2) * ζ * λ + one(T))
    return (one(T) / H + one(T) + w * (one(T) - exp(-τ * λ)))
end

# 2. Hybrid Strategy Part 1: Grid sweep
invΩ_v = LinRange(0.0, 4.0, 120)
w_v = LinRange(0.0, 2.0, 100)
params_vec = vec([(invΩ_v[i], w_v[j]) for i in 1:length(invΩ_v), j in 1:length(w_v)])

println("Grid sweep (Turning Model)...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(invΩ_v), length(w_v))
σ_mat_est = reshape(σ_ests_vec, length(invΩ_v), length(w_v))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Hybrid Strategy Part 2: Detailed MDBM Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(invΩ, w)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (invΩ, w), verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(0.0, 4.0, 20), LinRange(0.0, 2.0, 20)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="Stability Chart (Turning Model)", xlabel="1/Ω", ylabel="w")

hm = heatmap!(ax, invΩ_v, w_v, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Metric (σ_est if stable, Z if unstable)")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end


println("Finding largest inscribed circle in the stable region...in a $(size(invΩ_v,1))x$(size(w_v,1)) matrix")
@time R,circ_x,circ_y,i,j = find_largest_circle(Z_mat_int .== 0,invΩ_v, w_v,N=0) # not the N as a 2^N coarse-to-fine (multi-resolution) search strategy. It might loose circle smaller than 2^N, but it is much faster than the brute-force search. You can set N=0 for the brute-force search.
@show  R
circle = GeometryBasics.Circle(Point2f(circ_x, circ_y), R)
scatter!(ax, [circ_x], [circ_y], color=:magenta, marker=:star5, markersize=15, strokecolor=:black, strokewidth=1, label="Largest Inscribed Circle Center")
lines!(ax, circle, color=:magenta, linewidth=2, label="Largest Inscribed Circle")

# Save figure to disk
mkpath("output_figures")
save("output_figures/example_03.png", f)
display(f)
