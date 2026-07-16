using Test
using InterpolatedNyquist
using StaticArrays

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

    @testset "Shifted σ-line: absolute root coordinates" begin
        # D(λ) = (λ - r)(λ - conj(r)) with known root pair r = -0.3 ± 2im
        function D_pair(λ::T, p) where T
            r_re, r_im = p
            return (λ - (r_re + 1im * r_im)) * (λ - (r_re - 1im * r_im))
        end
        p = (-0.3, 2.0)

        # σ = 0 (imaginary axis): stable, closest root at -0.3 + 2im
        Z, Z_raw, min_D, es, wc = calculate_unstable_roots_direct(D_pair, p; ω_max=1e4)
        @test Z == 0
        @test isapprox(es, -0.3; atol=1e-6)
        @test isapprox(wc, 2.0; atol=1e-3)

        # σ = -1: both roots lie right of the shifted line -> Z = 2,
        # and σ_est must be the ABSOLUTE real part (-0.3), not relative to the line.
        Z, Z_raw, min_D, es, wc = calculate_unstable_roots_direct(D_pair, p, -1.0; ω_max=1e4)
        @test Z == 2
        @test isapprox(es, -0.3; atol=1e-6)
        @test isapprox(wc, 2.0; atol=1e-3)

        # QuadGK backend (unrefined estimate, looser tolerance)
        Zq, _, _, esq, wcq = calculate_unstable_roots_quadgk(D_pair, p, -1.0; ω_max=1e4)
        @test Zq == 2
        @test isapprox(esq, -0.3; atol=5e-2)

        # Fixed-step backend
        Zf, _, _, esf, _ = calculate_unstable_roots_fixed_step(D_pair, p, -1.0; ω_max=1e3, steps=20000)
        @test Zf == 2
        @test isapprox(esf, -0.3; atol=5e-2)
    end

    @testset "Leading-order estimation (real-axis probe)" begin
        # The arc at infinity lies in the RIGHT half-plane, where delayed terms
        # decay like exp(-sτ); probing there removes the ripple that an
        # imaginary-axis fit has to average out.

        # retarded with delayed velocity feedback (delay term one order below
        # the leading power -- worst case for an imaginary-axis point estimator)
        D_vel(λ::T, p) where T = λ^4 + λ^2 + (p[1] + p[2] * λ^3) * exp(-λ / 2)
        @test isapprox(get_n_power_max(D_vel, (1.0, 0.5)), 4.0; atol=1e-8)

        # neutral: effective order 2 (an imaginary-axis point estimator diverges)
        D_neut(λ::T, p) where T = λ^2 + p[1] * λ^2 * exp(-λ) + one(T) + p[2] * exp(-λ)
        @test isapprox(get_n_power_max(D_neut, (0.5, 1.0)), 2.0; atol=1e-8)
        @test isapprox(get_n_power_max(D_neut, (0.9, 1.0)), 2.0; atol=1e-8)
        Z_n, Z_raw_n = calculate_unstable_roots_direct(D_neut, (0.5, 1.0);
            n_roots_to_track=0, ω_max=200.0, reltol=1e-6, abstol=1e-6)
        @test Z_n == 2   # matches the explicit n_power_max=2 reference count

        # fractional: non-integer leading order 1.8 (corrections decay slower)
        D_frac(λ::T, p) where T = λ^T(1.8) + T(0.5) * λ^T(0.8) + p[1] * exp(-p[2] * λ)
        @test isapprox(get_n_power_max(D_frac, (2.0, 1.0)), 1.8; atol=1e-6)

        # rational (turning-type): D -> 1, so the effective order is 0
        function D_rat(λ::T, p) where T
            Ω, w = p
            return one(T) + w * (1 - exp(-2π / Ω * λ)) / (λ^2 + 2 * T(0.02) * λ + one(T))
        end
        @test isapprox(get_n_power_max(D_rat, (0.5, 0.6)), 0.0; atol=1e-8)

        # overflow guard: |D| of a high-order determinant overflows at s = 1e8,
        # so the probe must back off automatically instead of returning NaN.
        # Accuracy degrades for such systems because the back-off caps s well
        # below the asymptotic regime -- pass n_power_max explicitly if known.
        D_ovf(λ::T, p) where T = (λ^2 + p[1] * λ + p[2])^50   # order 100
        n_ovf = get_n_power_max(D_ovf, (0.5, 1.0))
        @test isfinite(n_ovf)
        @test isapprox(n_ovf, 100.0; rtol=1e-2)
    end

    @testset "Overflow-safe integrand (huge determinant scale)" begin
        # Large-matrix determinants reach |D| beyond floatmax; the phase
        # integrand Im(D'/D) is scale-invariant and must survive that.
        D_huge(λ::T, p) where T = T(1e280) * (λ^2 + p[1] * λ + p[2] * exp(-λ / 2))
        D_norm(λ::T, p) where T = λ^2 + p[1] * λ + p[2] * exp(-λ / 2)
        for p in ((0.5, 1.0), (-0.2, 0.8), (0.1, -0.5))
            Zh, Zrh = calculate_unstable_roots_direct(D_huge, p; ω_max = 1e4, n_roots_to_track = 0)
            Zn, Zrn = calculate_unstable_roots_direct(D_norm, p; ω_max = 1e4, n_roots_to_track = 0)
            @test Zh == Zn
            @test isapprox(Zrh, Zrn; atol = 1e-3)
        end
    end

    @testset "Mass-matrix / DAE extraction" begin
        # Descriptor system with singular mass matrix E = diag(1, m, 0):
        #   x' = v
        #   m v' = -k x - c v + F
        #   0   = F + P x(t-τ)      (algebraic equation defining the delayed force)
        # Analytic characteristic function: det(λE - J) = -(m λ² + c λ + k + P e^{-λτ})
        function dae_rhs(u, h, p, t)
            k, c, m, P, τ = p
            x, v, F = u
            x_d = h(p, t - τ; idxs=1)
            return SA[v, -k * x - c * v + F, F + P * x_d]
        end
        p_dae = (1.0, 0.1, 1.3, 0.4, 0.5)
        k, c, m, P, τ = p_dae
        E = @SMatrix [1.0 0.0 0.0; 0.0 1.3 0.0; 0.0 0.0 0.0]

        D_true(λ) = -(m * λ^2 + c * λ + k + P * exp(-τ * λ))

        for λtest in (0.3 + 1.7im, -0.2 + 0.9im, 1.1 - 2.4im)
            D_ext = get_D_from_model(dae_rhs, λtest, p_dae, Val(3); mass_matrix=E)
            @test abs(D_ext - D_true(λtest)) < 1e-10
        end

        # Default mass_matrix = I must reproduce the old behaviour (explicit ODE part)
        function ode_rhs(u, h, p, t)
            k2, c2, P2, τ2 = p
            x, v = u
            x_d = h(p, t - τ2; idxs=1)
            return SA[v, -k2 * x - c2 * v - P2 * x_d]
        end
        p_ode = (1.0, 0.1, 0.4, 0.5)
        D_ode_true(λ) = λ^2 + p_ode[2] * λ + p_ode[1] + p_ode[3] * exp(-p_ode[4] * λ)
        D_ext_I = get_D_from_model(ode_rhs, 0.3 + 1.7im, p_ode, Val(2))
        @test abs(D_ext_I - D_ode_true(0.3 + 1.7im)) < 1e-10

        # Full pipeline through the DAE-extracted characteristic function:
        # root count must match the equivalent reduced retarded oscillator.
        D_dae(λ, pp) = get_D_from_model(dae_rhs, λ, pp, Val(3); mass_matrix=E)
        D_red(λ, pp) = pp[3] * λ^2 + pp[2] * λ + pp[1] + pp[4] * exp(-pp[5] * λ)
        for P_test in (0.4, 3.0, -2.0)
            p_t = (k, c, m, P_test, τ)
            Z_dae, _, _, es_dae, _ = calculate_unstable_roots_direct(D_dae, p_t; ω_max=1e4)
            Z_red, _, _, es_red, _ = calculate_unstable_roots_direct(D_red, p_t; ω_max=1e4)
            @test Z_dae == Z_red
            @test isapprox(es_dae, es_red; atol=1e-6)
        end
    end
end