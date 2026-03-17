using InterpolatedNyquist
using BenchmarkTools
using ForwardDiff
using StaticArrays

# NAIVE VERSION
global_coeff = 0.03 
function D_naive(λ, p)
    P, D = p
    τ = 0.5
    ζ = 0.02
    return (global_coeff * λ^4 + λ^2 + 2 * ζ * λ + 1 + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# TYPE-STABLE VERSION
function D_stable(λ::T, p) where T
    P, D = p
    c1 = T(0.03)
    τ  = T(0.5)
    ζ  = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# BENCHMARKING
p_test = (0.5, 0.1)
params_vec = [p_test for _ in 1:100]

println("--- Type Stability Impact Benchmark ---")
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

println("\n1. Single evaluation (Naive):")
@btime D_naive(1.5im, $p_test)

println("2. Single evaluation (Type-Stable):")
@btime D_stable(1.5im, $p_test)

println("\n--- Full Integration Sweep (100 points) ---")
println("3. Integration Sweep (Naive):")
@time calculate_unstable_roots_p_vec(D_naive, params_vec, ω_max=100.0, verbosity=0)

println("4. Integration Sweep (Type-Stable):")
@time calculate_unstable_roots_p_vec(D_stable, params_vec, ω_max=100.0, verbosity=0)
