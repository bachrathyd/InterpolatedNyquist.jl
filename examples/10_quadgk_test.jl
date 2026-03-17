using InterpolatedNyquist
using BenchmarkTools

# A simple DDE system to test compilation overhead
function D_chareq_algebraic(λ::T, p) where T
    a, b = p
    return λ^2 + a * λ + b * exp(-0.5 * λ)
end

Pv = LinRange(-1.0, 1.0, 10)
Dv = LinRange(-1.0, 1.0, 10)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

println("--- ODE vs QuadGK Benchmark ---")

println("\n1. ODE Solver (Compilation + Execution):")
@time calculate_unstable_roots_p_vec(D_chareq_algebraic, params_vec, verbosity=0)
println("1. ODE Solver (Execution only):")
@time calculate_unstable_roots_p_vec(D_chareq_algebraic, params_vec, verbosity=0)

println("\n2. QuadGK (Compilation + Execution):")
@time calculate_unstable_roots_quadgk_p_vec(D_chareq_algebraic, params_vec, verbosity=0)
println("2. QuadGK (Execution only):")
@time calculate_unstable_roots_quadgk_p_vec(D_chareq_algebraic, params_vec, verbosity=0)
