using InterpolatedNyquist
using GLMakie
using DifferentialEquations
using Statistics
using QuadGK
using Printf

# 1. Define a standard test system (Type-Stable)
function D_chareq(λ::T, p) where T
    P, D = p
    τ = T(0.5)
    ζ = T(0.02)
    return (T(0.03) * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

p_test = (2.0, 1.0) # Unstable point
ω_max = 1e4

# 2. Reference Value (Ultra High Precision)
println("Calculating reference value...")
# Return is (Z_int, Z_raw, min_D, estimated_sigma, ω_crit)
res_ref = calculate_unstable_roots_direct(D_chareq, p_test; 
    ω_max=ω_max, reltol=1e-15, abstol=1e-15, solver=Rosenbrock23(), maxiters=Int(1e7))
phi_ref = res_ref[2]
println("Reference Z_raw: $phi_ref")

# 3. Parameter Sweep for Convergence
tols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

# We will test:
# - Rosenbrock23 (Stiff ODE)
# - Tsit5 (Non-stiff ODE)
# - QuadGK (Adaptive Quadrature)
methods = [
    (tol -> calculate_unstable_roots_direct(D_chareq, p_test; ω_max=ω_max, reltol=tol, abstol=tol, solver=Rosenbrock23(), maxiters=Int(1e7)), "Rosenbrock23 (Stiff)"),
    (tol -> calculate_unstable_roots_direct(D_chareq, p_test; ω_max=ω_max, reltol=tol, abstol=tol, solver=Tsit5(), maxiters=Int(1e7)), "Tsit5 (Non-stiff)"),
    (tol -> calculate_unstable_roots_quadgk(D_chareq, p_test; ω_max=ω_max, reltol=tol, abstol=tol), "QuadGK (Quadrature)")
]

results = []

for (foo, name) in methods
    errors = Float64[]
    times = Float64[]
    println("Benchmarking method: $name")
    for tol in tols
        # Warmup
        foo(tol)
        
        # Measure time
        t = @elapsed begin
            res = foo(tol)
        end
        phi_val = res[2]
        
        push!(errors, abs(phi_val - phi_ref))
        push!(times, t)
    end
    push!(results, (name, errors, times))
end

# 4. Plotting
f = Figure(size=(1000, 800))
ax1 = GLMakie.Axis(f[1, 1], title="Convergence: Error vs. Requested Tolerance", 
    xlabel="Tolerance (reltol=abstol)", ylabel="Absolute Error |phi - phi_ref|", 
    xscale=log10, yscale=log10)

ax2 = GLMakie.Axis(f[2, 1], title="Efficiency: Time vs. Accuracy", 
    xlabel="Absolute Error", ylabel="Execution Time (s)", 
    xscale=log10, yscale=log10)

# Add a diagonal line for O(tol) reference
lines!(ax1, tols, tols, color=:grey, linestyle=:dash, label="O(tol)")

for (name, errors, times) in results
    # Add small epsilon to errors to avoid log(0)
    err_plot = errors .+ 1e-17
    scatterlines!(ax1, tols, err_plot, label=name)
    scatterlines!(ax2, err_plot, times, label=name)
end

# --- PRINT TABLE FOR LATEX ---
println("\n\\begin{table}[h]")
println("\\centering")
println("\\caption{Performance comparison of different solvers across various tolerances}")
println("\\begin{tabular}{c|ccc|ccc}")
println("Tol & \\multicolumn{3}{c|}{Error (Integral)} & \\multicolumn{3}{c}{Time (s)} \\\\")
println(" & Rosen23 & Tsit5 & QuadGK & Rosen23 & Tsit5 & QuadGK \\\\ \\hline")
for i in 1:length(tols)
    @printf("%.0e & %.2e & %.2e & %.2e & %.4f & %.4f & %.4f \\\\\n", 
        tols[i], results[1][2][i], results[2][2][i], results[3][2][i],
        results[1][3][i], results[2][3][i], results[3][3][i])
end
println("\\end{tabular}")
println("\\end{table}")

f[1, 2] = Legend(f[1, 2], ax1, "Method")
mkpath("output_figures")
save("output_figures/example_13.png", f)
display(f)
