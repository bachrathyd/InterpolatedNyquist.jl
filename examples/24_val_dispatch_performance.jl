using InterpolatedNyquist
using GLMakie
using BenchmarkTools
using LinearAlgebra
using StaticArrays

# 1. Define characteristic equation D(λ, p)
function D_chareq(λ::T, p) where T
    P, D = p
    c1 = T(0.03)
    τ = T(0.5)
    ζ = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# Parameters for testing
p_test = (-0.2, 0.5)

println("Performance comparison at p=$p_test:")

# Warmup to ensure JIT compilation doesn't affect benchmark
calculate_unstable_roots_direct(D_chareq, p_test, n_roots_to_track=0)
calculate_unstable_roots_direct(D_chareq, p_test, n_roots_to_track=1)
calculate_unstable_roots_direct(D_chareq, p_test, n_roots_to_track=10)



# Benchmarking
t0 = @belapsed calculate_unstable_roots_direct($D_chareq, $p_test, n_roots_to_track=0) 
t1 = @belapsed calculate_unstable_roots_direct($D_chareq, $p_test, n_roots_to_track=1)
t10 = @belapsed calculate_unstable_roots_direct($D_chareq, $p_test, n_roots_to_track=10) 

println("Val{0} (Max Speed): $(round(t0*1e6, digits=2)) μs")
println("Val{1} (Single):    $(round(t1*1e6, digits=2)) μs")
println("Val{10} (Multi):    $(round(t10*1e6, digits=2)) μs")

# Get results for plotting
# res1: (Z_int, Z_raw, min_D, sigma, omega)
res1 = calculate_unstable_roots_direct(D_chareq, p_test, n_roots_to_track=1)
# res10: (Z_int, Z_raw, min_Ds_vec, sigmas_vec, omegas_vec)
res10 = calculate_unstable_roots_direct(D_chareq, p_test, n_roots_to_track=10)

Z_int, Z_raw, min_D1, σ1, ω1 = res1
_, _, min_Ds, σs, ωs = res10

# Filter out uninitialized values (NaN/Inf)
mask = .!isnan.(σs) .& (min_Ds .< 1e5)
σs_plot = σs[mask]
ωs_plot = ωs[mask]

# Plotting
f = Figure(size=(800, 600))
ax = GLMakie.Axis(f[1, 1],
    title="Root Approximation (Z_raw=$(round(Z_raw, digits=3)), σ_closest=$(round(σ1, digits=3)))",
    xlabel="Re(λ) [σ]", ylabel="Im(λ) [ω]")

# Define grid for contour plot based on found roots
σ_range = [min(-0.5, isempty(σs_plot) ? -0.5 : minimum(σs_plot) - 0.1),
    max(0.5, isempty(σs_plot) ? 0.5 : maximum(σs_plot) + 0.1)]
ω_range = [0.0, max(10.0, isempty(ωs_plot) ? 10.0 : maximum(ωs_plot) + 1.0)]

σ_grid = LinRange(σ_range[1], σ_range[2], 150)
ω_grid = LinRange(ω_range[1], ω_range[2], 150)

# Evaluate D over the complex plane grid
D_vals = [D_chareq(Complex(s, w), p_test) for s in σ_grid, w in ω_grid]

# Plot contours for Re(D)=0 and Im(D)=0
contour!(ax, σ_grid, ω_grid, real.(D_vals), levels=[0.0], color=:blue, linewidth=2)
contour!(ax, σ_grid, ω_grid, imag.(D_vals), levels=[0.0], color=:red, linewidth=2)

# Legend entries for contours (dummy lines)
lines!(ax, [0, 0], [0, 0], color=:blue, label="Re(D)=0")
lines!(ax, [0, 0], [0, 0], color=:red, label="Im(D)=0")

# Scatter approximated roots
scatter!(ax, σs_plot, ωs_plot, color=:black, marker=:circle, markersize=10, label="Val{10} Roots")
scatter!(ax, [σ1], [ω1], color=:yellow, marker=:star5, markersize=15, strokecolor=:black, strokewidth=1, label="Val{1} Closest")

axislegend(ax, position=:rt)

# Ensure output directory exists
mkpath("output_figures")
save("output_figures/example_24.png", f)

println("Figure saved to output_figures/example_24.png")
display(f)
