using InterpolatedNyquist
using BenchmarkTools
using ForwardDiff
using StaticArrays

# 1. VERSION A: Clean but no explicit types
function D_no_types(λ, p)
    P, D = p
    τ = 0.5
    ζ = 0.02
    return (0.03 * λ^4 + λ^2 + 2 * ζ * λ + 1 + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 2. VERSION B: Explicit types
function D_types(λ::T, p) where T
    P, D = p
    c1 = T(0.03)
    τ  = T(0.5)
    ζ  = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 3. BENCHMARK
Pv = LinRange(-2.01, 4.0, 40)
Dv = LinRange(-2.01, 5.0, 40)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

println("--- Compilation vs. Execution Investigation ---")

println("\n>> TEST 1: NO EXPLICIT TYPES")
println("First run (including compilation):")
@time calculate_unstable_roots_p_vec(D_no_types, params_vec, verbosity=0)
println("Second run (execution only):")
@time calculate_unstable_roots_p_vec(D_no_types, params_vec, verbosity=0)

println("\n>> TEST 2: EXPLICIT TYPES")
println("First run (including compilation):")
@time calculate_unstable_roots_p_vec(D_types, params_vec, verbosity=0)
println("Second run (execution only):")
@time calculate_unstable_roots_p_vec(D_types, params_vec, verbosity=0)
