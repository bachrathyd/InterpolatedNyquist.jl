using InterpolatedNyquist
using GLMakie
using ForwardDiff
using LinearAlgebra
using Printf

# 1. Define characteristic equation D(λ, p)
function D_chareq(λ::T, p) where T
    P, D = p
    c1 = T(0.03)
    τ = T(0.5)
    ζ = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 2. Setup parameters
p_fixed = (0.5, 1.0)
println("Analyzing system at P=$(p_fixed[1]), D=$(p_fixed[2])...")

# 3. Get initial estimates (tracking 10 roots)
# CRITICAL: We explicitly use :Linear here to get the RAW integrator results as a baseline
zi, zr, md_vec, σ_vec, ω_vec = calculate_unstable_roots_direct(D_chareq, p_fixed; 
    n_roots_to_track=10, ω_max=10000.0, refinement_method=:Linear)


# Filter out NaNs
valid_idx = .!isnan.(σ_vec)
initial_roots = [σ_vec[i] + 1im * ω_vec[i] for i in 1:length(σ_vec) if valid_idx[i]]

# 4. Refine roots with different methods
println("Refining roots...")

# Reference: Newton with 15 steps (treat as "exact")
roots_ref = refine_roots(D_chareq, p_fixed, initial_roots, method=:Newton, steps=15)

# Method 1: Linear (Degree 1 Polynomial / Newton 1 step)
# Note: initial_roots are already linear approximations from the integrator
roots_lin = initial_roots

# Method 2: Polynomial Degree 2
roots_poly2 = refine_roots(D_chareq, p_fixed, initial_roots, method=:Polynomial, degree=2)

# Method 3: Polynomial Degree 3
roots_poly3 = refine_roots(D_chareq, p_fixed, initial_roots, method=:Polynomial, degree=3)

# Method 4: Newton 4 steps
roots_newton4 = refine_roots(D_chareq, p_fixed, initial_roots, method=:Newton, steps=4)


# 5. Display Errors
println("\nRoot Refinement Error Comparison (vs Newton-15):")
for i in 1:length(roots_ref)
    println("Root $i (Initial estimate: $(initial_roots[i]))")
    err_lin = abs(roots_lin[i] - roots_ref[i])
    err_p2 = abs(roots_poly2[i] - roots_ref[i])
    err_p3 = abs(roots_poly3[i] - roots_ref[i])
    err_n4 = abs(roots_newton4[i] - roots_ref[i])
    @printf("  Linear error:     %.2e\n", err_lin)
    @printf("  Poly-2 error:     %.2e\n", err_p2)
    @printf("  Poly-3 error:     %.2e\n", err_p3)
    @printf("  Newton-4 error:   %.2e\n", err_n4)
end

# 6. Plotting
fig = Figure(size=(1000, 800))
ax = GLMakie.Axis(fig[1, 1],
    title="Root Refinement Comparison (Z_raw = $(round(zr, digits=4)), σ_closest = $(round(real(roots_ref[1]), digits=4)))",
    xlabel="Re(λ)", ylabel="Im(λ)")

# Grid for contours
σ_range = LinRange(-1.0, 1.0, 200)
ω_range = LinRange(0.0, 40.0, 200)
D_grid = [D_chareq(s + 1im * w, p_fixed) for s in σ_range, w in ω_range]
Re_D = real.(D_grid)
Im_D = imag.(D_grid)

# Contour lines Re(D)=0 and Im(D)=0
contour!(ax, σ_range, ω_range, Re_D, levels=[0.0], color=:blue, linewidth=2, label="Re(D)=0")
contour!(ax, σ_range, ω_range, Im_D, levels=[0.0], color=:red, linewidth=2, label="Im(D)=0")

# Scatter estimates
scatter!(ax, real.(roots_lin), imag.(roots_lin), marker=:circle, color=:gray, markersize=12, label="Linear (Integrator)")
scatter!(ax, real.(roots_poly2), imag.(roots_poly2), marker=:cross, color=:green, markersize=15, label="Poly-2")
scatter!(ax, real.(roots_poly3), imag.(roots_poly3), marker=:star5, color=:orange, markersize=15, label="Poly-3")
scatter!(ax, real.(roots_newton4), imag.(roots_newton4), marker=:utriangle, color=:purple, markersize=15, label="Newton-4")
scatter!(ax, real.(roots_ref), imag.(roots_ref), marker=:diamond, color=:black, markersize=10, label="Newton-10 (Ref)")

axislegend(ax, position=:rt)

# Zoom into the first root
ax_zoom = GLMakie.Axis(fig[1, 2], title="Zoom: Closest Root", xlabel="Re(λ)", ylabel="Im(λ)")
r1 = roots_ref[1]
zoom_size = 0.05
σ_z = LinRange(real(r1) - zoom_size, real(r1) + zoom_size, 100)
ω_z = LinRange(imag(r1) - zoom_size, imag(r1) + zoom_size, 100)
D_z = [D_chareq(s + 1im * w, p_fixed) for s in σ_z, w in ω_z]
contour!(ax_zoom, σ_z, ω_z, real.(D_z), levels=[0.0], color=:blue, linewidth=2)
contour!(ax_zoom, σ_z, ω_z, imag.(D_z), levels=[0.0], color=:red, linewidth=2)

scatter!(ax_zoom, [real(roots_lin[1])], [imag(roots_lin[1])], marker=:circle, color=:gray, markersize=12)
scatter!(ax_zoom, [real(roots_poly2[1])], [imag(roots_poly2[1])], marker=:cross, color=:green, markersize=15)
scatter!(ax_zoom, [real(roots_poly3[1])], [imag(roots_poly3[1])], marker=:star5, color=:orange, markersize=15)
scatter!(ax_zoom, [real(roots_newton4[1])], [imag(roots_newton4[1])], marker=:utriangle, color=:purple, markersize=15)
scatter!(ax_zoom, [real(roots_ref[1])], [imag(roots_ref[1])], marker=:diamond, color=:black, markersize=10)

display(fig)
save("output_figures/example_26_refinement.png", fig)
println("Figure saved to output_figures/example_26_refinement.png")

##
# ------------------------------------------------------------------
# NEW SECTION: 2D Sweep and Error Comparison
# ------------------------------------------------------------------
println("\nStarting 2D sweep for error comparison...")

Pv_sweep = LinRange(-1.0, 1.0, 60)
Dv_sweep = LinRange(0.0, 2.0, 40)
params_sweep = vec([(p, d) for p in Pv_sweep, d in Dv_sweep])

# Method settings
methods_cfg = [
    (name="Linear", method=:Linear, steps=0, degree=0),
    (name="Poly-2", method=:Polynomial, steps=0, degree=2),
    (name="Poly-3", method=:Polynomial, steps=0, degree=3),
    (name="Newton-4", method=:Newton, steps=4, degree=0)
]

results_σ = []
results_err = []

# Ground Truth Calculation (Newton-15)
println("Calculating Ground Truth (Newton-10)...")
@time zi_gt, zr_gt, md_gt, es_gt, wc_gt = calculate_unstable_roots_p_vec(D_chareq, params_sweep;
    refinement_method=:Newton, refinement_steps=10)

for m in methods_cfg
    println("Processing $(m.name)...")
    @time zi, zr, md, es, wc = calculate_unstable_roots_p_vec(D_chareq, params_sweep;
        refinement_method=m.method, refinement_steps=m.steps, refinement_degree=m.degree)

    # Stability Metric: Z + sigma (if stable)
    σ_metric = zi .+ (zi .== 0) .* es
    push!(results_σ, reshape(σ_metric, length(Pv_sweep), length(Dv_sweep)))

    # Error: absolute difference in sigma compared to ground truth
    err = abs.(es .- es_gt)
    push!(results_err, reshape(err, length(Pv_sweep), length(Dv_sweep)))
end

# Multi-panel Plotting
fig2 = Figure(size=(1600, 800))
for i in 1:4
    # Top Row: Stability Metric
    ax_top = GLMakie.Axis(fig2[1, i], title="$(methods_cfg[i].name) Metric", xlabel="p", ylabel="d")
    hm_top = heatmap!(ax_top, Pv_sweep, Dv_sweep, results_σ[i], colormap=:viridis, colorrange=(-0.5, 1.5))
    if i == 4
        Colorbar(fig2[1, 5], hm_top)
    end

    # Bottom Row: Error (Log scale)
    ax_bot = GLMakie.Axis(fig2[2, i], title="$(methods_cfg[i].name) Error", xlabel="p", ylabel="d")
    # Use log10 for error visualization, capped at 1e-15 for floating point precision limits
    log_err = log10.(max.(results_err[i], 1e-16))
    hm_bot = heatmap!(ax_bot, Pv_sweep, Dv_sweep, log_err, colormap=:inferno, colorrange=(-15, 0))
    if i == 4
        Colorbar(fig2[2, 5], hm_bot, label="log10(Error)")
    end
end

display(fig2)

save("output_figures/example_26_error_map.png", fig2)
println("2D Error Map saved to output_figures/example_26_error_map.png")
