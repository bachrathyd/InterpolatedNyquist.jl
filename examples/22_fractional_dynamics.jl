using InterpolatedNyquist
using GLMakie

# Fractional-Order Delayed System
# Example inspired by common literature on fractional-order delay systems
# D(s) = s^α + a*s^β + k*e^{-sτ} = 0
# with 1 < α < 2 and 0 < β < 1

function D_fractional(λ::T, p) where T
    α, β, a, k, τ = p
    # Use complex power for fractional order
    return λ^α + a * λ^β + k * exp(-τ * λ)
end

# Parameters: α=1.8, β=0.8, a=0.5, τ=1.0
α_val = 1.8
β_val = 0.8
a_val = 0.5
τ_val = 1.0

# Sweep over gain 'k'
kv = LinRange(0.0, 5.0, 200)
params_vec = [(α_val, β_val, a_val, k, τ_val) for k in kv]

println("Calculating stability for Fractional-Order System...")
@time Z_ints, Z_raws, min_Ds, σ_ests, ω_crits = calculate_unstable_roots_p_vec(D_fractional, params_vec, ω_max=100.0)

# 2D Parameter Sweep: Gain 'k' vs Delay 'τ'
kv_2d = LinRange(0.0, 5.0, 50)
τv_2d = LinRange(0.1, 2.0, 50)
params_vec_2d = vec([(α_val, β_val, a_val, k, τ) for k in kv_2d, τ in τv_2d])

println("Grid sweep (k vs τ)...")
@time Z_ints_2d, Z_raws_2d, min_Ds_2d, σ_ests_2d, ω_crits_2d = 
    calculate_unstable_roots_p_vec(D_fractional, params_vec_2d, ω_max=100.0)

Z_mat = reshape(Z_ints_2d, length(kv_2d), length(τv_2d))
σ_mat = reshape(σ_ests_2d, length(kv_2d), length(τv_2d))
C_plot = Z_mat .+ (Z_mat .== 0) .* σ_mat

# Visualization
f = Figure(size=(800, 600))
ax = Axis(f[1, 1], title="Fractional Dynamics Stability (α=1.8, β=0.8)", xlabel="Gain k", ylabel="Delay τ")
hm = heatmap!(ax, kv_2d, τv_2d, C_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")

save("output_figures/example_22_fractional.png", f)
display(f)
