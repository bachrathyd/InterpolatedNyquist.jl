using Test
using InterpolatedNyquist

@testset "InterpolatedNyquist.jl Core Tests" begin

    @testset "4th-Order Delayed System" begin
        # 1. Define characteristic equation D(λ, p)
        function D_chareq(λ::T, p) where T
            P, D = p
            c1 = T(0.03)
            τ  = T(0.5)
            ζ  = T(0.02)
            return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
        end

        # Test 1: Single point evaluation (Val{1} dispatch)
        p_test = (-0.2, 0.5)
        Z, Z_raw, min_D, σ_est, ω_crit = calculate_unstable_roots_direct(D_chareq, p_test, n_roots_to_track=1)
        
        @test Z >= 0
        @test typeof(Z) == Int
        @test Z == round(Int, Z_raw)
        @test !isnan(σ_est)

        # Test 2: Vectorized sweep with Multi-Root Tracking (Val{N} dispatch)
        Pv = LinRange(-1.0, 1.0, 3)
        Dv = LinRange(-1.0, 1.0, 3)
        params_vec = vec([(p, d) for p in Pv, d in Dv])

        Z_ints, Z_raws, min_Ds_vec, σ_ests_vec_many, ω_crits_vec = 
            calculate_unstable_roots_p_vec(D_chareq, params_vec, verbosity=0, n_roots_to_track=3)

        @test length(Z_ints) == 9
        @test length(σ_ests_vec_many) == 9
        
        # Test ignoring NaNs logic exactly as written in the examples
        σ_ests_vec = map(σ_ests_vec_many) do v
            v[argmin(ifelse.(isnan.(v), Inf, abs.(v)))]
        end
        @test length(σ_ests_vec) == 9
    end
end