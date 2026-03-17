using InterpolatedNyquist
using BenchmarkTools
using ForwardDiff
using StaticArrays
using MDBM

# 1. NAIVE VERSION
global_zeta = 0.02 
function D_naive(λ, p)
    P, D = p
    τ = 0.5
    return (0.03 * λ^4 + λ^2 + 2 * global_zeta * λ + 1 + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 2. TYPE-STABLE VERSION
function D_stable(λ::T, p) where T
    P, D = p
    c1 = T(0.03)
    τ  = T(0.5)
    ζ  = T(0.02)
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 3. BENCHMARK CONFIGURATION
Pv = LinRange(-2.01, 4.0, 40)
Dv = LinRange(-2.01, 5.0, 40)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

function mdbm_wrapper_stable(p, d)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_stable, (p, d), verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

function mdbm_wrapper_naive(p, d)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_naive, (p, d), verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

println("--- Professional Performance Benchmark (Grid: 1600 points) ---")
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5.0

println("\n1. Grid Sweep (Naive):")
@time calculate_unstable_roots_p_vec(D_naive, params_vec, verbosity=0)

println("\n2. Grid Sweep (Type-Stable):")
@time calculate_unstable_roots_p_vec(D_stable, params_vec, verbosity=0)

println("\n3. MDBM Trace (Naive):")
@time begin
    boundary = MDBM_Problem(mdbm_wrapper_naive, [LinRange(-2,4,15), LinRange(-2,5,15)])
    MDBM.solve!(boundary, 3, verbosity=0)
end

println("\n4. MDBM Trace (Type-Stable):")
@time begin
    boundary = MDBM_Problem(mdbm_wrapper_stable, [LinRange(-2,4,15), LinRange(-2,5,15)])
    MDBM.solve!(boundary, 3, verbosity=0)
end
