using Test
using InterpolatedNyquist
using StaticArrays
import MDBM

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

        # transcendental continuum: D = cosh(sqrt(λ²+cλ)) + Kp exp(-λτ) grows
        # like e^λ in the right half-plane, so it is NOT polynomially bounded
        # and the real-axis probe is meaningless there. The estimator must
        # detect that (the slope is not scale-invariant) and fall back to the
        # imaginary axis, where |D| is bounded and oscillatory -> effective n = 0.
        function D_beam(λ::T, p) where T
            Kp, τ = p
            return cosh(sqrt(λ^2 + T(0.4) * λ)) + Kp * exp(-λ * τ)
        end
        n_beam = get_n_power_max(D_beam, (0.5, 1.0))
        @test isfinite(n_beam)
        @test abs(n_beam) < 0.5          # effective order 0, not a huge number
        Z_b, Zr_b = calculate_unstable_roots_direct(D_beam, (0.5, 1.0);
            n_roots_to_track=0, ω_max=200.0)
        # Like a neutral system, |D(iω)| stays oscillatory for all ω, so the
        # effective order carries a bounded residual; the count still rounds
        # correctly (the residual stays well inside the 1/2 threshold).
        @test abs(Zr_b - round(Zr_b)) < 0.25

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

    @testset "Non-finite count degrades to the invalid marker -1" begin
        # A root exactly ON the integration line (or any other violation of the
        # counting assumptions) must not throw an InexactError inside a threaded
        # sweep; the count degrades to -1, which the boundary objectives already
        # treat as invalid via max(Z, 0).
        D_ok(λ::T, p) where T = λ^2 + p[1] * λ + p[2] * exp(-λ / 2)
        Z_nf, Zr_nf = calculate_unstable_roots_direct(D_ok, (0.5, 1.0);
            n_roots_to_track=0, ω_max=1e3, n_power_max=NaN)
        @test Z_nf == -1
        @test isnan(Zr_nf)
    end

    @testset "Refinement stays local (no runaway into overflow)" begin
        # A characteristic function that overflows if evaluated far from the
        # origin -- exactly what a runaway Newton step causes, and the overflow
        # then happens INSIDE the user's D where no guard of ours can intercept.
        # A refinement step must therefore stay local.
        function D_fragile(λ::T, p) where T
            abs(λ) > 1e50 && error("D evaluated absurdly far from the seed: $λ")
            return (λ^2 + p[1] * λ + p[2])^60      # overflows for |λ| ≳ 1e5
        end
        for m in (:Newton, :Polynomial)
            for seed in (0.5 + 1.0im, 1e3 + 0.0im, -2.0 + 5.0im)
                r = refine_roots(D_fragile, (0.5, 1.0), seed; method = m,
                                 steps = 6, degree = 3)
                @test isfinite(abs(r))
            end
        end
        # a seed where D' vanishes must not throw either
        D_flat(λ::T, p) where T = one(T) + 0 * λ + 0 * p[1]
        @test isfinite(abs(refine_roots(D_flat, (1.0, 1.0), 1.0 + 1.0im; method = :Newton, steps = 4)))
    end

    @testset "QuadGK panel coverage over a wide frequency window" begin
        # Regression test: with a huge ω_max and a fast-decaying integrand
        # ripple (position-only feedback, ω^-3), the FIRST Gauss-Kronrod panel
        # over [0, ω_max] has its lowest node above the entire resonance
        # region; without interior breakpoints the quadrature converges after
        # one panel to Z_raw = n/2 -- a wrong count with a PERFECT integer
        # residual. The log-spaced breakpoints must prevent this.
        D_pos(λ::T, p) where T =
            T(0.03) * λ^4 + λ^2 + T(0.04) * λ + one(T) + p[1] * exp(-λ / 2)
        for P in (0.5, 2.0, -1.5, 3.5)
            Z_ref, _ = calculate_unstable_roots_direct(D_pos, (P,); ω_max=1e6,
                n_roots_to_track=0, reltol=1e-8, abstol=1e-8)
            Z_gk, Zr_gk = calculate_unstable_roots_quadgk(D_pos, (P,); ω_max=1e6)
            @test Z_gk == Z_ref
            @test abs(Zr_gk - Z_gk) < 0.1
        end
    end

    @testset "MDBM enrichment accepts the estimated (non-integer) order" begin
        # get_n_power_max returns a Float64; the enrichment path passes it to
        # calculate_encirclement_number, whose keyword must therefore accept
        # any Real (a ::Integer restriction makes the documented workflow
        # throw a TypeError at run time).
        D_enr(λ, p) = λ^2 + 0.1 * λ + p + 0.5 * exp(-λ)
        n_est = get_n_power_max(D_enr, 1.0)
        @test n_est isa Float64   # the exact situation that used to throw (::Integer kwarg)
        ωs = collect(range(0.0, 50.0, length=400))
        ωs_full = vcat(ωs, -ωs)
        Dv = [D_enr(1im * w, 1.0) for w in ωs]
        Nc = calculate_encirclement_number(vcat(Dv, conj.(Dv)), ωs_full; n_power_max=n_est)
        @test isfinite(Nc)
        @test abs(Nc - round(Nc)) < 0.1

        # smoke test of the full exported workflow (would have caught the bug)
        foo_ri(p, ω) = (real(D_enr(1im * ω, p)), imag(D_enr(1im * ω, p)))
        mdbm = InterpolatedNyquist.MDBM.MDBM_Problem(foo_ri,
            [LinRange(0.5, 2.0, 4), LinRange(0.0, 10.0, 10)])
        InterpolatedNyquist.MDBM.solve!(mdbm, 1, verbosity=0)
        p_uniq, Ncirc = argument_principle_with_MDBM(D_enr, mdbm, ωs)
        @test !isempty(Ncirc)
        @test all(isfinite, Ncirc)
    end

    @testset "Multi-root tracking on a shifted σ-line (absolute coords)" begin
        # two known conjugate pairs: -0.3 ± 2im and -1.5 ± 8im (well separated
        # in ω so both produce distinct |D| minima along the march)
        function D_two(λ::T, p) where T
            r1 = p[1] + 1im * p[2]; r2 = p[3] + 1im * p[4]
            return (λ - r1) * (λ - conj(r1)) * (λ - r2) * (λ - conj(r2))
        end
        p2 = (-0.3, 2.0, -1.5, 8.0)
        Z, Zr, mds, ess, wcs = calculate_unstable_roots_direct(D_two, p2, -2.0;
            n_roots_to_track=5, ω_max=1e3, refinement_steps=8)
        @test Z == 4                        # all four roots right of Re λ = -2
        found = [complex(ess[i], wcs[i]) for i in eachindex(ess) if isfinite(ess[i])]
        @test any(r -> abs(r - (-0.3 + 2.0im)) < 1e-6, found)
        @test maximum(real, found) ≈ -0.3 atol=1e-6   # dominant root, absolute coords
        # every reported root is a true root, and no refined duplicates remain
        @test all(r -> abs(D_two(r, p2)) < 1e-6, found)
        for i in 1:length(found), j in i+1:length(found)
            @test abs(found[i] - found[j]) > 1e-3
        end
    end

    @testset "Multi-root tracking captures a real root at ω = 0" begin
        # dominant root is REAL (divergence-type instability): only the ω = 0
        # boundary branch can capture it. The ω = 0 Newton seed is crude for a
        # cubic (it lands at 2.0 for this D), so allow enough polish steps.
        D_real(λ::T, p) where T = (λ - p[1]) * (λ + 1) * (λ + 2)
        Z, Zr, mds, ess, wcs = calculate_unstable_roots_direct(D_real, (0.5,);
            n_roots_to_track=3, ω_max=1e3, refinement_steps=12)
        @test Z == 1
        good = findall(isfinite, ess)
        i = good[argmax(ess[good])]
        @test ess[i] ≈ 0.5 atol=1e-8
        @test abs(wcs[i]) < 1e-6
    end

    @testset "Overflow scale: quadgk and fixed-step backends" begin
        D_huge2(λ::T, p) where T = T(1e280) * (λ^2 + p[1] * λ + p[2] * exp(-λ / 2))
        D_norm2(λ::T, p) where T = λ^2 + p[1] * λ + p[2] * exp(-λ / 2)
        for p in ((0.5, 1.0), (0.1, -0.5))
            Zg_h, _ = calculate_unstable_roots_quadgk(D_huge2, p; ω_max=1e4)
            Zg_n, _ = calculate_unstable_roots_quadgk(D_norm2, p; ω_max=1e4)
            @test Zg_h == Zg_n
            Zf_h, _ = calculate_unstable_roots_fixed_step(D_huge2, p; ω_max=1e3, steps=20000)
            Zf_n, _ = calculate_unstable_roots_fixed_step(D_norm2, p; ω_max=1e3, steps=20000)
            @test Zf_h == Zf_n
        end
    end

    @testset "Total overflow yields an invalid root estimate, not a fabricated one" begin
        # |D| overflows at EVERY sample -> no minimum can be tracked; the root
        # estimate must come back NaN instead of a "root" polished from the
        # 0 + 0im placeholder.
        D_allovf(λ::T, p) where T = T(1e200) * (T(1e200) * (λ^2 + p[1] * λ + p[2]))
        Z_o, Zr_o, md_o, es_o, wc_o = calculate_unstable_roots_direct(D_allovf, (0.5, 1.0);
            ω_max=1e3, n_power_max=2.0)
        @test isnan(es_o) && isnan(wc_o)
        @test !isfinite(md_o)
    end

    @testset "User-supplied leading order (n_power_max) in the sweeps" begin
        # A neutral system: |D| = |λ²(1 + a e^{-λ}) + ...| never settles onto a
        # power law on the imaginary axis, but n = 2 by inspection.
        D_neu(λ, p) = λ^2 + p[1] * λ^2 * exp(-λ) + 1 + p[2] * exp(-λ)
        params = [(0.5, c) for c in LinRange(-2.0, 2.0, 12)]

        Z_est, Zr_est = calculate_unstable_roots_p_vec(D_neu, params; ω_max = 200.0,
            n_roots_to_track = 0)
        Z_usr, Zr_usr = calculate_unstable_roots_p_vec(D_neu, params; ω_max = 200.0,
            n_roots_to_track = 0, n_power_max = 2.0)
        # the estimator finds n = 2 here, so supplying it must not change a thing
        @test Z_usr == Z_est
        @test Zr_usr ≈ Zr_est rtol = 1e-8

        # ... and a deliberately WRONG order must shift Z_raw by exactly the
        # difference of the n/2 terms, proving the value is really being used
        Z_bad, Zr_bad = calculate_unstable_roots_p_vec(D_neu, params; ω_max = 200.0,
            n_roots_to_track = 0, n_power_max = 4.0)
        @test Zr_bad ≈ Zr_est .+ 1.0 rtol = 1e-8

        # the other two back-ends honour it as well
        Zq, Zrq = calculate_unstable_roots_quadgk_p_vec(D_neu, params; ω_max = 200.0,
            n_power_max = 4.0)
        @test Zrq ≈ Zr_bad rtol = 1e-4
        Zf, Zrf = calculate_unstable_roots_fixed_step_p_vec(D_neu, params; ω_max = 200.0,
            steps = 2000, n_power_max = 4.0)
        @test all(isfinite, Zrf)
    end

    @testset "n_power_max is accepted by the MDBM-level entry points" begin
        # These paths estimated n internally with no way to override it, and
        # argument_principle_with_MDBM additionally fed a Float64 estimate into
        # a kwarg once typed ::Integer -- i.e. the documented workflow threw a
        # TypeError before it could produce a number.
        D_osc(λ, p) = λ^2 + 0.2 * λ + 1.0 + p[1] * exp(-λ)
        ax = [LinRange(0.0, 2.0, 5), LinRange(0.0, 6.0, 5)]   # (p, ω)
        prob = MDBM.MDBM_Problem((p, ω) -> begin
                D = D_osc(1im * ω, (p,))
                (real(D), imag(D))
            end, ax)
        MDBM.solve!(prob, 2, verbosity = 0)

        # estimated (the default) must not throw ...
        p_uniq, Nc = argument_principle_with_MDBM((λ, p) -> D_osc(λ, p), prob, [0.0, 1.0])
        @test length(p_uniq) == length(Nc)
        @test all(isfinite, Nc)

        # ... and a supplied order must be honoured: n = 2 here, and feeding
        # n = 4 must shift every count by exactly (4-2)/2 = 1
        _, Nc2 = argument_principle_with_MDBM((λ, p) -> D_osc(λ, p), prob, [0.0, 1.0];
            n_power_max = 2.0)
        _, Nc4 = argument_principle_with_MDBM((λ, p) -> D_osc(λ, p), prob, [0.0, 1.0];
            n_power_max = 4.0)
        @test Nc4 ≈ Nc2 .+ 1.0
    end

    @testset "Peak-skip cross-check" begin
        # sound direction: a tracked root right of the line contradicts Z == 0
        @test peak_skip_suspect(0, 1e-4)              # stable count, unstable root
        @test peak_skip_suspect(1, -1e-4)             # unstable count, stable root
        @test !peak_skip_suspect(0, -1e-4)            # consistent
        @test !peak_skip_suspect(2, 1e-4)             # consistent
        # distance guard: a disagreement far from the line is legitimate (the
        # tracked minimum need not be the dominant root) and must NOT fire
        @test !peak_skip_suspect(2, -0.5)
        @test peak_skip_suspect(2, -0.5; h = 1.0)     # ...unless h says otherwise
        # shifted line: the comparison is against sigma, not zero
        @test peak_skip_suspect(0, -0.4 + 1e-5, -0.4)
        @test !peak_skip_suspect(0, -0.5, -0.4)
        # non-finite estimate (|D| overflowed everywhere) must not fire
        @test !peak_skip_suspect(0, NaN)
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
            # a plain (non-static) Matrix must work identically
            D_ext_mat = get_D_from_model(dae_rhs, λtest, p_dae, Val(3); mass_matrix=Matrix(E))
            @test abs(D_ext_mat - D_true(λtest)) < 1e-10
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