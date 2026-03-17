using InterpolatedNyquist
using BenchmarkTools

# Standard 4th order system
function D_chareq(λ::T, p) where T
    P, D = p
    c1 = T(0.03)
    τ  = T(0.5)
    ζ  = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

Pv = LinRange(-2.0, 4.0, 20)
Dv = LinRange(-2.0, 5.0, 20)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

println("--- Solver Performance Floor (400 points) ---")

println("\n1. Adaptive ODE Solver (reltol=1e-3):")
@time calculate_unstable_roots_p_vec(D_chareq, params_vec, reltol=1e-3, verbosity=0)

println("\n2. Fixed-Step Trapezoidal (steps=200):")
@time calculate_unstable_roots_fixed_step_p_vec(D_chareq, params_vec, steps=200, verbosity=0)

println("\n3. Fixed-Step Trapezoidal (steps=1000):")
@time calculate_unstable_roots_fixed_step_p_vec(D_chareq, params_vec, steps=1000, verbosity=0)

println("\nObservations:")
println("- Fixed-step is extremely fast but requires careful selection of 'steps' for convergence.")
println("- Perfect for real-time visualization where small integer errors in Z might be acceptable.")
