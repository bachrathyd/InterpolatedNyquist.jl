# NyquistGPU unit tests: self-contained (analytic references only), CPU backend.
# Run:  julia --project=gpu -t auto -e "using Pkg; Pkg.test()"
# (the same kernels run on CUDABackend(); see gpu/scripts/gpu_check.jl)

using NyquistGPU, Test

# --- analytic references ----------------------------------------------------
# Hayes: x'(t) = a x(t) + b x(t-1), D(λ) = λ - a - b e^{-λ}, n = 1.
# Asymptotically stable iff a < 1, a + b < 0 and b > -sqrt(a² + ν²), where
# ν ∈ (0, π) solves ν = a tan(ν) (ν = π/2 for a = 0).
function hayes_stable(a, b)
    (a < 1 && a + b < 0) || return false
    abs(a) < 1e-14 && return b > -π / 2
    f(v) = v - a * tan(v)
    lo, hi = a > 0 ? (1e-12, π / 2 - 1e-12) : (π / 2 + 1e-12, π - 1e-12)
    for _ in 1:200
        m = (lo + hi) / 2
        (f(lo) * f(m) <= 0) ? (hi = m) : (lo = m)
    end
    ν = (lo + hi) / 2
    return b > -sqrt(a^2 + ν^2)
end
# true if the analytic answer does not change within ±d (skip boundary-grazing points)
hayes_safe(a, b; d = 0.03) =
    all(hayes_stable(a + da, b + db) == hayes_stable(a, b) for da in (-d, 0, d), db in (-d, 0, d))

D_hayes(λ, p, c) = λ - p[1] - p[2] * exp(-λ)
D_hayes_tau(λ, p, c) = λ - p[1] - p[2] * exp(-p[3] * λ)           # 3-D parameter point
# cubic with known roots p1 ± i p2 and -1:  Z = 2 for p1 > 0, 0 for p1 < 0
D_cubic(λ, p, c) = (λ^2 - 2 * p[1] * λ + p[1]^2 + p[2]^2) * (λ + 1)
D_promoting(λ, p, c) = λ^2 + 0.5 * λ + p[1]                        # Float64 literal!

@testset "NyquistGPU" begin

    @testset "Hayes equation, analytic region ($T, $meth)" for T in (Float64, Float32),
                                                                 meth in (:unwrap, :bs3)
        as = range(-3.0, 0.95; length = 41)
        bs = range(-3.0, 3.0; length = 41)
        pts = grid_points(as, bs)
        r = sweep(D_hayes, pts; n_power = 1, T = T, method = meth,
            (meth === :bs3 ? (ω_max = 1e4, tol = 1e-6) : (;))...)
        ok = [hayes_safe(p...) for p in pts]
        expect = [hayes_stable(p...) for p in pts]
        @test all((r.Z .== 0)[ok] .== expect[ok])
        @test count(ok) > 0.9 * length(pts)
        @test all(>=(0), r.Z)                       # no failed marches
    end

    @testset "3-D point list (Hayes with delay τ as 3rd parameter)" begin
        # λ' = λτ maps the τ-delay problem onto Hayes with (aτ, bτ)
        as = range(-2.0, 0.9; length = 9)
        bs = range(-2.5, 2.5; length = 9)
        τs = (0.5, 1.0, 2.0)
        pts = grid_points(as, bs, τs)
        @test length(pts) == 9 * 9 * 3 && length(first(pts)) == 3
        r = sweep(D_hayes_tau, pts; n_power = 1)
        Z3 = reshape(r.Z, length(as), length(bs), length(τs))
        for (k, τ) in enumerate(τs), (j, b) in enumerate(bs), (i, a) in enumerate(as)
            hayes_safe(a * τ, b * τ) || continue
            @test (Z3[i, j, k] == 0) == hayes_stable(a * τ, b * τ)
        end
    end

    @testset "scattered points + root tracking (known cubic)" begin
        # an arbitrary point cloud -- no grid at all
        pts = [(0.8 * sin(1.3k), 0.5 + abs(cos(0.7k))) for k in 1:300]
        for T in (Float64, Float32)
            r = sweep(D_cubic, pts; n_power = 3, T = T, nroots = 4)
            @test r.Z == [p[1] > 0 ? 2 : 0 for p in pts]
            # dominant tracked root: σ = max(p1, -1); one Newton step from the
            # |D| minimum -> first-order accurate near the line
            near = [abs(p[1]) < 0.05 for p in pts]
            σtrue = [max(p[1], -1.0) for p in pts]
            @test maximum(abs.(r.sigma[near] .- σtrue[near])) < 5e-3
        end
    end

    @testset "schedules are bit-identical" begin
        pts = grid_points(range(-3.0, 0.9; length = 23), range(-3.0, 3.0; length = 19))
        rs = [sweep(D_hayes, pts; n_power = 1, schedule = s) for s in (:pixel, :strided, :queue)]
        for r in rs[2:end]
            @test r.Zraw == rs[1].Zraw
            @test isequal(r.sigma, rs[1].sigma)
            @test r.steps == rs[1].steps
        end
    end

    @testset "chart() = 2-D special case" begin
        r = chart(D_hayes, (-3.0, 0.9), (-3.0, 3.0), 17, 13; n_power = 1)
        @test size(r.Z) == (17, 13)
        @test r.Z[17, 1] >= 1        # a = 0.9, b = -3: unstable
        @test r.Z[1, 7] == 0         # a = -3, b = 0: stable
    end

    @testset "boundary-grazing points are flagged, Float64 recheck fixes them" begin
        # a = 0: Hopf boundary at b = -π/2 (root crosses at ω = π/2)
        pts = [(0.0, -π / 2 - 1e-9), (0.0, -π / 2 + 1e-9), (0.0, -1.0), (0.0, -2.0)]
        r32 = sweep(D_hayes, pts; n_power = 1, T = Float32)
        @test r32.flags[3] == 0 && r32.flags[4] == 0
        @test r32.Z[3] == 0 && r32.Z[4] == 2
        @test all(r32.flags[1:2] .!= 0)                       # sub-resolution in Float32
        r64 = sweep(D_hayes, pts; n_power = 1, T = Float64)
        @test r64.Z == [2, 0, 0, 2]                           # resolved in Float64
        n = recheck!(r32, D_hayes, pts; n_power = 1)
        @test n == 2 && r32.Z == [2, 0, 0, 2]
    end

    @testset "root exactly on the line -> residual flag" begin
        # Hayes with a + b = 0: D(0) = -a - b = 0, Z_raw is a half-integer
        r = sweep(D_hayes, [(-1.0, 1.0), (-1.0, 0.5)]; n_power = 1, T = Float64)
        @test r.flags[1] & 4 != 0
        @test abs(r.Zraw[1] - round(r.Zraw[1])) > 0.25
        @test r.flags[2] == 0 && r.Z[2] == 0
    end

    @testset "unrolled literal powers of the dual λ" begin
        D4p(λ, p, c) = λ^4 + p[1] * λ^3 + λ^-2
        D4m(λ, p, c) = λ * λ * λ * λ + p[1] * (λ * λ * λ) + inv(λ * λ)
        for ω in (0.3, 1.7, 12.0)
            a = NyquistGPU.eval_line(NyquistGPU.CharFn(D4p), (0.7,), (), 0.1, ω)
            b = NyquistGPU.eval_line(NyquistGPU.CharFn(D4m), (0.7,), (), 0.1, ω)
            @test all(isapprox.(a, b; rtol = 1e-12))
        end
    end

    @testset "emulated narrow formats" begin
        xs = [0.1, 1.0625, -3.3, 65504.0, 7e4, 1e-6, 2.5, 449.0, 7.0]
        @test all(Float32(Float16_emu(x)) == Float32(Float16(x)) for x in xs[1:6])
        @test Float32(Float8_E4M3(1.0625)) == 1.0          # ties to even, 3 mantissa bits
        @test Float32(Float8_E4M3(449.0)) == 448.0         # E4M3FN saturates at 448
        @test Float32(Float8_E5M2(7e4)) == Inf32           # E5M2 overflows to Inf (max 57344)
        @test sort(unique(Float32.(Float4_E2M1.(0:0.01:7)))) == Float32[0, 0.5, 1, 1.5, 2, 3, 4, 6]
        @test Float32(Float8_E4M3(0.1)) == 0.1015625f0                      # 3 mantissa bits
        @test Float32(Float8_E4M3(0.1) * Float8_E4M3(3.0)) == 0.3125f0      # 0.3046875 rounded
        # mixed precision: march in Float32, D evaluated in (emulated) Float16
        pts = grid_points(range(-3.0, 0.9; length = 15), range(-3.0, 3.0; length = 15))
        ok = [hayes_safe(p...) for p in pts]
        # ω_max stays below the Float16 range (65504): λ itself would overflow at 1e5
        r = sweep(D_hayes, pts; n_power = 1, T = Float32, Teval = Float16_emu, ω_max = 1e3)
        @test all((r.Z .== 0)[ok] .== [hayes_stable(p...) for p in pts][ok])
    end

    @testset "device-generated grid, regrid!, recheck_flagged!" begin
        xs, ys = range(-3.0, 0.9; length = 23), range(-3.0, 3.0; length = 17)
        ref = sweep(D_hayes, grid_points(xs, ys); n_power = 1, T = Float64)
        g = plan_grid((-3.0, 0.9), (-3.0, 3.0), 23, 17; n_power = 1, T = Float64)
        @test all(isapprox.(collect.(Array(g.points)), collect.(grid_points(xs, ys)); atol = 1e-12))
        @test fetch_result(run!(g, D_hayes)).Z == ref.Z
        # zoom without reallocating
        regrid!(g, (-1.0, 0.5), (0.0, 2.0), 23, 17)
        @test fetch_result(run!(g, D_hayes)).Z ==
              sweep(D_hayes, grid_points(range(-1.0, 0.5; length = 23), range(0.0, 2.0; length = 17));
                    n_power = 1, T = Float64).Z
        @test_throws ErrorException regrid!(g, (0, 1), (0, 1), 10, 10)
        # second pass: Float16-evaluated sweep, flagged points redone in Float64 on the device
        regrid!(g, (-3.0, 0.9), (-3.0, 3.0), 23, 17)
        h = plan_grid((-3.0, 0.9), (-3.0, 3.0), 23, 17; n_power = 1, T = Float32,
                      Teval = Float16_emu, ω_max = 1e3, schedule = :pixel)
        run!(h, D_hayes)
        flagged = count(!=(0), Array(h.flags))
        r64 = plan_grid((0.0, 1.0), (0.0, 1.0), 23, 17; n_power = 1, T = Float32, schedule = :pixel)
        @test recheck_flagged!(h, r64, D_hayes) == flagged
        @test r64.npts == 23 * 17                       # capacity restored
        ok = [hayes_safe(p...) for p in grid_points(xs, ys)]
        @test (fetch_result(h).Z .== ref.Z)[ok] == trues(count(ok))
    end

    @testset "root refinement: Newton polish and certification by counting" begin
        # roots p1 ± i p2 and -3 ± 2i: the rightmost real part is max(p1, -3)
        Dq(λ, p, c) = ((λ - p[1])^2 + p[2]^2) * ((λ + 3)^2 + 4)
        pts = vec([(a, b) for a in range(-2.5, -0.1; length = 9), b in range(0.3, 3.0; length = 7)])
        exact = [max(a, -3.0) for (a, b) in pts]
        r0 = sweep(Dq, pts; n_power = 4, T = Float64)
        rn = sweep(Dq, pts; n_power = 4, T = Float64, refine = 25)
        rc = sweep(Dq, pts; n_power = 4, T = Float64, refine = 5, certify = true, σtol = 1e-6)
        rc32 = sweep(Dq, pts; n_power = 4, T = Float32, certify = true, σtol = 1e-4)
        @test all(r0.Z .== 0)
        # Newton polishes the tracked roots; where no |D| minimum was tracked at all
        # (the blind spot of the line-based estimate) it has nothing to polish
        ok = isfinite.(rn.sigma)
        @test isnan.(rn.sigma) == isnan.(r0.sigma)
        # ... and every polished value is a true root, though not always the rightmost
        # one (its minimum may be the untracked one): only counting guarantees that
        @test all(min(abs(s - a), abs(s + 3)) < 1e-6 for (s, (a, b)) in zip(rn.sigma[ok], pts[ok]))
        @test count(abs.(rn.sigma[ok] .- exact[ok]) .> 1e-6) < count(abs.(r0.sigma[ok] .- exact[ok]) .> 1e-6)
        @test maximum(abs.(rc.sigma .- exact)) <= 1e-6         # counting brackets all of them
        @test maximum(abs.(rc32.sigma .- exact)) <= 2e-4
        # set_march! changes the settings in place
        g = plan_grid((-2.5, -0.1), (0.3, 3.0), 9, 7; n_power = 4, T = Float64)
        run!(g, Dq)
        set_march!(g; refine = 5, certify = true, σtol = 1e-6, ω_max = 1e4)
        @test g.mp.newton == 5 && g.mp.bisect == 1 && g.mp.ωmax == 1e4
        @test maximum(abs.(fetch_result(run!(g, Dq)).sigma .- exact)) <= 1e-6
    end

    @testset "precision check" begin
        @test check_eltype(D_hayes, (0.0, 0.0), (), Float32)
        @test !check_eltype(D_promoting, (0.0,), (), Float32)
    end

    @testset "schedule simulator" begin
        @test simulate_schedule(fill(10, 4096); schedule = :pixel_rowmajor) ≈ 1
        s = [isodd(i) ? 10 : 100 for i in 1:4096]
        @test simulate_schedule(s; schedule = :pixel_rowmajor) < 0.6
        @test simulate_schedule(s; schedule = :queue, lanes = 256) > 0.9
    end
end
