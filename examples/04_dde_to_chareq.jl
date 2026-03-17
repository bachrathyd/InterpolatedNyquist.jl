using InterpolatedNyquist
using StaticArrays
using LinearAlgebra
using BenchmarkTools

# 1. Define time-domain DDE model
function turning_model(u, h, p, t)
    invΩ, w = p
    τ = 2π * invΩ
    ζ = 0.01

    u_delayed = h(p, t - τ)

    du1 = u[2]
    du2 = -(2.0 + w)*u[1] - 2.0*ζ*u[2] + w*u_delayed[1]

    return SA[du1, du2]
end

# 2. Analytical solution for comparison
# Type-stable version
function D_chareq_analytical(λ::T, p) where T
    invΩ, w = p
    τ = T(2π) * invΩ
    ζ = T(0.01)
    H = one(T) / (λ^2 + T(2) * ζ * λ + one(T))
    return (one(T) / H + one(T) + w * (one(T) - exp(-τ * λ)))
end

# 3. Test extraction
p_test = (0.5, -0.1)
ω_test = 2.5
σ_test = -0.05
λ_test = σ_test + 1im * ω_test

println("--- DDE to CharEq Extraction Test ---")
println("Test point: λ = $λ_test")

# Extract using black-box method
D_numerical = get_D_from_model(turning_model, λ_test, p_test, Val(2))
D_analytical = D_chareq_analytical(λ_test, p_test)

println("1. Analytical D(λ): ", D_analytical)
println("2. Numerical D(λ):  ", D_numerical)

difference = abs(D_analytical - D_numerical)
println("\nDifference: ", difference)

if difference < 1e-10
    println("-> SUCCESS: Numerical extraction matches analytical solution!")
else
    println("-> ERROR: Mismatch detected.")
end

# 4. Benchmark
println("\n--- Performance Benchmark ---")
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1
println("1. Analytical:")
@btime D_chareq_analytical($λ_test, $p_test)
println("2. Numerical (Black-box):")
@btime get_D_from_model($turning_model, $λ_test, $p_test, Val(2))
