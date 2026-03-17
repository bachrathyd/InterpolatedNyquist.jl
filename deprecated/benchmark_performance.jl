using InterpolatedNyquist
using BenchmarkTools
using MDBM

# Setup characteristic equation
function D_chareq(ω, p, σ=0.0)
    P, D = p
    λ = 1im * ω + σ
    τ = 0.5
    ζ = 0.02
    return (0.03 * λ^4 + λ^2 + 2 * ζ * λ + 1 + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 20x20 grid for speed
Pv = LinRange(-2.0, 4.0, 20)
Dv = LinRange(-2.0, 5.0, 20)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

println("--- BENCHMARK BASELINE ---")
println("1. Grid Sweep (400 points):")
@btime calculate_unstable_roots_p_vec($D_chareq, $params_vec, verbosity=0) samples=1 evals=1

println("\n2. MDBM Trace (Stability Boundary):")
function mdbm_wrapper(p, d)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (p, d), verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end
boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(-2,4,15), LinRange(-2,5,15)])
@btime MDBM.solve!($boundary_mdbm, 3, verbosity=0) samples=1 evals=1
