using InterpolatedNyquist
using GLMakie
using Statistics
using Printf

# 1. Define a standard test system (4th Order)
function D_chareq(λ::T, p) where T
    P, D = p
    τ = T(0.5)
    ζ = T(0.02)
    # Characteristic equation for a delayed system
    return (T(0.03) * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 2. Setup Parameter Grid
Pv = LinRange(-2.0, 4.0, 50)
Dv = LinRange(-2.0, 5.0, 50)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

println("Calculating stability and error field over $(length(params_vec)) points...")

# 3. Perform Sweep
# Using a moderate tolerance to see some "visible" numerical error if it exists
reltol = 1e-4
abstol = 1e-4
ω_max = 1e4

# Custom loop to collect raw Z values
Z_raws = zeros(length(params_vec))
Z_ints = zeros(Int, length(params_vec))

@time Threads.@threads for i in 1:length(params_vec)
    # calculate_unstable_roots_direct returns (Z_int, Z_raw, ...)
    res = calculate_unstable_roots_direct(D_chareq, params_vec[i]; 
        ω_max=ω_max, reltol=reltol, abstol=abstol)
    Z_ints[i] = res[1]
    Z_raws[i] = res[2]
end

# 4. Error Estimation
# The theory states that Z must be an integer. 
# Thus, the distance to the nearest integer is a measure of numerical error.
errors = abs.(Z_raws .- round.(Z_raws))

Z_mat = reshape(Z_ints, length(Pv), length(Dv))
err_mat = reshape(errors, length(Pv), length(Dv))

# 5. Plotting
f = Figure(size=(1200, 500))

# Axis 1: Stability Chart
ax1 = Axis(f[1, 1], title="Stability Chart (Z_int)", xlabel="P", ylabel="D")
hm1 = heatmap!(ax1, Pv, Dv, Z_mat, colormap=:viridis)
Colorbar(f[1, 2], hm1, label="Number of Unstable Roots (Z)")

# Axis 2: Numerical Error Field
# Use log scale for error to see small variations
ax2 = Axis(f[1, 3], title="Numerical Error Field |Z_raw - Z_int|", xlabel="P", ylabel="D")
hm2 = heatmap!(ax2, Pv, Dv, err_mat, colormap=:inferno, colorrange=(1e-10, 1e-1))
Colorbar(f[1, 4], hm2, label="Absolute Error")

println("Average Error: ", mean(errors))
println("Max Error: ", maximum(errors))

mkpath("output_figures")
save("output_figures/example_21_error_field.png", f)
display(f)
