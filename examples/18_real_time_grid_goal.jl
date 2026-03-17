using InterpolatedNyquist
using BenchmarkTools

# 1. Simplest possible CharEq to test the floor of solver overhead
function D_chareq(λ::T, p) where T
    a, b = p
    return λ^2 + a * λ + b
end

# 2. 100x100 Grid (10,000 points)
av = LinRange(0.1, 2.0, 100)
bv = LinRange(0.1, 2.0, 100)
params_vec = vec([(av[i], bv[j]) for i in 1:length(av), j in 1:length(bv)])

println("--- Real-Time Goal Test (100x100 = 10,000 points) ---")

println("\n1. Standard ODE Sweep (reltol=1e-3):")
# Pre-warm
calculate_unstable_roots_p_vec(D_chareq, params_vec[1:10], reltol=1e-3, verbosity=0)
@time calculate_unstable_roots_p_vec(D_chareq, params_vec, reltol=1e-3, verbosity=0)

println("\n2. QuadGK Sweep:")
# Pre-warm
calculate_unstable_roots_quadgk_p_vec(D_chareq, params_vec[1:10], verbosity=0)
@time calculate_unstable_roots_quadgk_p_vec(D_chareq, params_vec, verbosity=0)

println("\n3. Fixed-Step Sweep (steps=500):")
# Pre-warm
calculate_unstable_roots_fixed_step_p_vec(D_chareq, params_vec[1:10], steps=500, verbosity=0)
@time calculate_unstable_roots_fixed_step_p_vec(D_chareq, params_vec, steps=500, verbosity=0)

println("\nNote: Goal is < 0.1 seconds.")
