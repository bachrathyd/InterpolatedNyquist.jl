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
    τ = T(0.5)
    ζ = T(0.02)
    #c1 = T(0.03)
    #τ = T(0.848)
    #ζ = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end
# 2. Hybrid Strategy Part 1: Grid sweep
Pv = LinRange(-2.01, 4.0, 150)
Dv = LinRange(-2.01, 5.0, 100)

params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])


println("Grid sweep - Brute Force")
#@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
#    calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=1)

@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec_many, ω_crits_vec =
    calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=0, n_roots_to_track=5)
σ_ests_vec = map(σ_ests_vec_many) do v
    v[argmin(ifelse.(isnan.(v), Inf, abs.(v)))]
end



Z_mat_int = reshape(Z_ints_vec, length(Pv), length(Dv))
σ_mat_est = reshape(σ_ests_vec, length(Pv), length(Dv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est


# 3. Hybrid Strategy Part 2: Detailed MDBM Trace
println("\nTracing stability boundary with MDBM...")

# function mdbm_wrapper(p, d)::Float64
#     Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec_many, ω_crits_vec =
#         calculate_unstable_roots_p_vec(D_chareq, [(p,d)], verbosity=0, n_roots_to_track=5)
#       @show   valid_sigma=filter(!isnan,   σ_ests_vec_many[1])
#     return length(valid_sigma) > 0 ? maximum(valid_sigma) : -Inf
# end
 function mdbm_wrapper(p, d)::Float64
     zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (p, d), verbosity=0)
     sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
     return sign_val * abs(es)
 end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(-2.01, 4.0, 37), LinRange(-2.01, 5.0, 37)])
@time MDBM.solve!(boundary_mdbm, 3, verbosity=0, interpolationorder=0)#This way we can eleimiate the "V shape solution" at the higher order boundary lines
interpolate!(boundary_mdbm, interpolationorder=1)

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


xyz_sol = getinterpolatedsolution(boundary_mdbm)
# this is only for a fast initalization
xy_val = getevaluatedpoints(boundary_mdbm)
fval = getevaluatedfunctionvalues(boundary_mdbm)
filter = fval .> 0

S = [xy_val[1][filter], xy_val[2][filter]] #Stable
U = [xy_val[1][.!filter], xy_val[2][.!filter]] # Unstable
B = xyz_sol #Boundary points

scatter!(S..., color=:green, label="S")
scatter!(U..., color=:red, label="U")
scatter!(B..., color=:blue, label="B")
scatter!(B..., label="B")






display(f)


##
# -------------------- Finding largest inscribed circle in the stable region ---------------------
println("-------------------------------------------------------------")
println("Finding largest inscribed circle in the stable region.")
# 1. Brute force grid based solution
my_scale_x = 1.4 # use the scale to unify the very different grid size - leading to fitting an ellipse
my_scale_y = 1.0 # use the scale to unify the very different grid size - leading to fitting an ellipse


println("Circle_ based on the Brute Force poitns...in a $(size(Pv,1))x$(size(Dv,1)) matrix")
@time circ_R, circ_x, circ_y, i, j = find_largest_circle(Z_mat_int .== 0, Pv, Dv, N=0, scale_x=my_scale_x, scale_y=my_scale_y) # not the N as a 2^N coarse-to-fine (multi-resolution) search strategy. It might loose circle smaller than 2^N, but it is much faster than the brute-force search. You can set N=0 for the brute-force search.
# Get the points for plotting
pts = generate_ellipse_points(circ_x, circ_y, circ_R,
    scale_x=my_scale_x, scale_y=my_scale_y)
scatter!(ax, [circ_x], [circ_y], color=:magenta, marker=:star5, markersize=15)
lines!(ax, pts[1], pts[2], color=:magenta, linewidth=2, label="Largest Inscribed Circle")#

display(f)

## -------------- circle based on MDBM result ---------------------
# based on a gridded solution, with random point cloud (Stable, Unstable, and Boundary points)
# Substep1: the center of the circle is must be on a given Stalbe points
# Substep2: refinement of the center location interatively
#note that the center is moveed along only in num_angles directions. The final resolution and the CPU time could be sensitive to this parameter (16-40 is a good range)

println("Circle_ based on the MDBM points")
@time final_circle = find_largest_circle(boundary_mdbm, N=5, scale_x=my_scale_x, scale_y=my_scale_y, num_angles=32, tol=1e-2)


# 2. Generate the plot points
pts = generate_ellipse_points(final_circle.x, final_circle.y, final_circle.R_scaled,
    scale_x=my_scale_x, scale_y=my_scale_y)

scatter!(ax, [final_circle.x], [final_circle.y], color=:yellow, marker=:star5, markersize=15)
lines!(ax, pts[1], pts[2], color=:yellow, linewidth=2, label="Largest Inscribed Ellipse")

display(f)




# Save figure to disk
mkpath("output_figures")
save("output_figures/example_02.png", f)
display(f)