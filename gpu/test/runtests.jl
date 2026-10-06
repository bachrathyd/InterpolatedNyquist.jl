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
        # a lane-schedule re-check plan: its stride belongs to its full capacity (with a common
        # factor of stride and the flagged count, points were skipped)
        h2 = plan_grid((-3.0, 0.9), (-3.0, 3.0), 23, 17; n_power = 1, T = Float32,
                       Teval = Float16_emu, ω_max = 1e3, schedule = :pixel)
        run!(h2, D_hayes)
        rs = plan_grid((0.0, 1.0), (0.0, 1.0), 23, 17; n_power = 1, T = Float32, schedule = :strided,
                       lanes = 7)
        st0 = rs.stride
        @test recheck_flagged!(h2, rs, D_hayes) == flagged
        @test fetch_result(h2).Z == fetch_result(h).Z
        @test fetch_result(h2).steps == fetch_result(h).steps
        @test (rs.npts, rs.stride, rs.lanes) == (23 * 17, st0, 7)
        # full charts run one warp per 4 x 8 pixel tile (k_tile!): same results as a point list,
        # also for sizes that are not multiples of the tile
        for (nx, ny) in ((23, 17), (8, 8), (5, 3))
            gt = plan_grid((-3.0, 0.9), (-3.0, 3.0), nx, ny; n_power = 1, T = Float64, schedule = :pixel)
            rt = fetch_result(run!(gt, D_hayes))
            # the same (device-generated) points as a list: bit-identical results
            rl = sweep(D_hayes, Array(gt.points); n_power = 1, T = Float64, schedule = :pixel)
            @test rt.Z == rl.Z && rt.steps == rl.steps && isequal(rt.sigma, rl.sigma)
        end
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

    # robustness checks (code review round 2): impossible counts, parity, end rule, trust radius
    # a circle march of a polynomial in w = 1/z = e^{-2πμ}: a real zero p[1] and a pair r e^{±iθ},
    # D = (1 - a w)(1 - 2 r cosθ w + r² w²) -> D(∞) = 1, count = [|a| > 1] + 2 [r > 1]
    D_circ(μ, p, c) = (w = exp(-c[1] * μ); (1 - p[1] * w) * (1 - 2 * p[2] * c[2] * w + p[2]^2 * w * w))
    # the same with a pole at w = 1/2 (z = 2): not entire in |z| > 1, the winding counts it -1
    D_pole(μ, p, c) = (w = exp(-c[1] * μ); (1 - p[1] * w) / (1 - 2w))
    circ_kw = (n_power = 0, ω0 = 1e-9, ω_max = 0.5, h0 = 0.05, hrel = 0.25, nroots = 1, T = Float64)
    c_circ = (2π, cos(0.9))

    @testset "robustness: impossible counts, parity, end rule, trust radius" begin
        pts = [(a, r) for a in (-1.3, -0.9, 0.5, 1.6) for r in (0.7, 1.2)]
        expect = [Int(abs(a) > 1) + 2 * Int(r > 1) for (a, r) in pts]
        r = sweep(D_circ, pts; c = c_circ, circle = true, parity = true, zmax = 3, circ_kw...)
        @test r.Z == expect
        @test all(==(0), r.flags)                       # parity holds, counts possible
        # zmax: the same counts against a too small bound are flagged (bit 8), nothing else
        r2 = sweep(D_circ, pts; c = c_circ, zmax = 2, circ_kw...)
        @test r2.Z == expect
        @test [f & 8 != 0 for f in r2.flags] == (expect .> 2)
        # a negative count (a pole outside the circle) is flagged as impossible
        rp = sweep(D_pole, [(0.5, 0.0)]; c = c_circ, circ_kw...)
        @test rp.Z == [-1] && rp.flags[1] & 8 != 0
        # end rule: a real zero just inside z = -1 (a stable flip multiplier) is a |D| minimum AT
        # μ = 1/2; only the circle march reports it (σ_μ = log(0.9)/2π at ω = 1/2)
        pe = [(-0.9, 0.3)]
        re = sweep(D_circ, pe; c = c_circ, circle = true, circ_kw...)
        ro = sweep(D_circ, pe; c = c_circ, circle = false, circ_kw...)
        @test re.Z == [0] && re.omega[1] == 0.5
        @test abs(re.sigma[1] - log(0.9) / 2π) < 2e-3
        @test !(ro.omega[1] == 0.5)
        # parity rule (white box): count 1 with D(1) < 0 and D(-1) < 0 is inconsistent
        mp = NyquistGPU.MarchParams{Float64, Float64}(0.0, 1e-9, 0.5, 0.05, 0.3, 0.3, 0.25, Inf, 0.0, 0.0,
            Int32(1000), Int32(0), Int32(0), 1e-3, Inf, 4.0, Int32(1), Int32(1))
        st = NyquistGPU.blank_state((0.0,), Float64, Val(1))
        mk(Φ, Dend, neg0) = NyquistGPU.MarchState{Float64, 1, Tuple{Float64}}(st.p, 0.5, 0.1, Φ, Dend, st.Dwa,
            st.ua, 1.0, 0.0, 0.0, Int32(5), Int8(1), neg0 ? NyquistGPU.FLAG_NEG0 : Int8(0), st.dd, st.ds, st.dw, st.ρ2, st.rej)
        @test NyquistGPU.finish(mk(-π, complex(-1.0), true), mp)[5] == 16         # Z = 1, parity 0
        @test NyquistGPU.finish(mk(-π, complex(+1.0), true), mp)[5] == 0          # Z = 1, parity 1
        @test NyquistGPU.finish(mk(-2π, complex(-1.0), true), mp)[5] == 0         # Z = 2, parity 0
        # the internal sign bit never leaves finish
        @test NyquistGPU.finish(mk(0.0, complex(1.0), true), mp)[5] & NyquistGPU.FLAG_NEG0 == 0
        # trust radius: a root estimate far beyond the resolving step is discarded (a shallow
        # minimum of a nearly flat D: Newton jumps far), the old radius max(|λ|, 1) accepted it
        # a shallow minimum (|D| ≈ 1, D' ≈ 2): Newton jumps |q| ≈ 0.5, fifty times the step h = 0.01
        args = (complex(1.0, 0.0067), complex(-0.5, 0.0), 1.0, complex(1.0, -0.0067), complex(0.5, 0.0), 1.0, 0.2, 0.01, 0.0)
        d_wide, s_wide, w_wide = NyquistGPU.dip_root(args..., 1e6)    # (practically) no trust radius
        qlen = hypot(s_wide - 0.0, w_wide - (0.2 + 0.01 * 0.5))        # ≈ |Newton step| (t ≈ 1/2)
        @test isfinite(d_wide) && qlen > 4 * 0.01                      # longer than 4 steps ...
        @test isnan(NyquistGPU.dip_root(args..., 4.0)[1])              # ... so discarded at qtrust = 4
        @test qlen^2 <= max(0.2^2, 1.0)                                # (the old radius kept it)
    end

    @testset "adaptive engine: run_list!, run_adaptive!, rho certificate, stepctl" begin
        nx, ny = 61, 47
        N = nx * ny
        full = plan_grid((-3.0, 0.9), (-3.0, 3.0), nx, ny; n_power = 1, T = Float64, schedule = :pixel)
        rf = fetch_result(run!(full, D_hayes))
        @test all(isfinite, rf.rho) && all(>(0), rf.rho)
        # run_list!: the listed points get the results of the full run, the others are untouched
        idx = collect(Int32, 5:7:N)
        rest = setdiff(1:N, idx)
        for (sched, kw) in ((:pixel, (;)), (:strided, (lanes = 13,)), (:queue, (lanes = 3,)))
            g = plan_grid((-3.0, 0.9), (-3.0, 3.0), nx, ny; n_power = 1, T = Float64, schedule = sched, kw...)
            run_list!(g, D_hayes, (), idx, length(idx))
            r = fetch_result(g)
            @test r.Z[idx] == rf.Z[idx] && r.steps[idx] == rf.steps[idx] && isequal(r.rho[idx], rf.rho[idx])
            @test all(==(0), r.steps[rest])
        end
        # the certificate rho = min |D/D'| is small only near the boundary (a root near the line)
        pts = Array(full.points)
        near = [!hayes_safe(p...; d = 0.1) for p in pts]
        @test maximum(rf.rho[.!near]) > 10 * minimum(rf.rho)
        # run_adaptive!: the counts of the full chart from a fraction of the points
        a = plan_grid((-3.0, 0.9), (-3.0, 3.0), nx, ny; n_power = 1, T = Float64, schedule = :pixel)
        info = run_adaptive!(a, D_hayes)                  # strides (16, 8, 4, 2, 1)
        ra = fetch_result(a)
        @test ra.Z == rf.Z
        @test info.points == sum(info.passes) && info.points < 0.8 * N   # 0.72 N (0.34 N at 481 x 361)
        @test count(>(0), Array(a.aux.lev)) == info.points
        # warm restart (slider move without a change): the same chart in fewer passes
        info2 = run_adaptive!(a, D_hayes; warm = true)
        @test fetch_result(a).Z == rf.Z && length(info2.passes) <= length(info.passes)
        # the damped step controller: the same counts away from the boundary
        d = plan_grid((-3.0, 0.9), (-3.0, 3.0), nx, ny; n_power = 1, T = Float64, schedule = :pixel,
                      stepctl = :damped)
        rd = fetch_result(run!(d, D_hayes))
        ok = [hayes_safe(p...) for p in pts]
        @test (rd.Z .== rf.Z)[ok] == trues(count(ok))
        @test d.mp.growmax == 2 && d.mp.hold == 1
        @test_throws ErrorException plan_grid((0, 1), (0, 1), 4, 4; n_power = 1, stepctl = :fast)
    end
end
