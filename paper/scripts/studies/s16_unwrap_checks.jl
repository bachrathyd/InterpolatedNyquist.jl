# Study s16: three numerical checks of the discrete phase-unwrapping back-end
# (`calculate_unstable_roots_unwrap`), against the ODE (Vern9) march at tight
# tolerances and against BigFloat Newton-polished roots.
#
#   1. neutral systems (appendix gallery A.4, A.5, A.6): Z agreement, evaluation
#      counts, integer residuals, omega_max scaling of the evaluation count
#   2. accuracy of the rightmost-root estimate at the three tab:semidisc points
#   3. boundary stress on both sides of the showcase Hopf boundary, with the
#      true crossing root and a parity cross-check
#
# Produces: paper/data/unwrap_checks/*.csv (results.md is written by hand from them)
# Writes nothing under paper/tables, paper/generated or paper/sections.
#
# Run (light environment, as s13):
#   julia --project=gpu/validate -t auto paper/scripts/studies/s16_unwrap_checks.jl

using InterpolatedNyquist, Statistics, Printf
const IN = InterpolatedNyquist
const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUT = joinpath(REPO, "paper", "data", "unwrap_checks")
mkpath(OUT)
include(joinpath(REPO, "gpu", "scripts", "systems.jl"))
println("threads: ", Threads.nthreads(), "   cpu: ", strip(Sys.cpu_info()[1].model))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
resid(z) = isfinite(z) ? abs(z - round(z)) : NaN
fmed(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))
fmax(v) = (w = filter(isfinite, v); isempty(w) ? NaN : maximum(w))
uw_evals(D, p; kw...) = IN._unwrap_march(IN.NyquistWrapper{typeof(p)}(D), p, 0.0, 1; kw...)[3]
function uw_evals_vec(D, pts; kw...)
    ev = zeros(Int, length(pts))
    Threads.@threads for i in eachindex(pts)
        ev[i] = uw_evals(D, pts[i]; kw...)
    end
    return ev
end
# parity rule for a real-coefficient D: Z is odd iff D(0) and the leading
# coefficient (sign of D at a large real s) have opposite signs
function parity_expected_odd(D, p)
    s0 = sign(real(D(complex(0.0), p)))
    sinf = sign(real(D(complex(1e6), p)))
    (s0 == 0 || sinf == 0) && return missing
    return s0 != sinf
end
parity_ok(Z, odd) = (odd === missing || Z < 0) ? missing : (isodd(Z) == odd)
function write_rows(name, header, rows)
    path = joinpath(OUT, name * ".csv")
    open(path, "w") do io
        println(io, join(header, ","))
        q(x) = (s = string(x); occursin(',', s) ? "\"" * s * "\"" : s)
        foreach(r -> println(io, join(q.(r), ",")), rows)
    end
    println("written ", path)
end

# ===========================================================================
# CHECK 1 -- neutral systems of the appendix gallery (D's copied from s08)
# ===========================================================================
function D_neutral(λ::T, p) where T
    a, c = p
    return λ^2 + a * λ^2 * exp(-λ) + one(T) + c * exp(-λ)
end
function D_neutral_hg(λ::T, p) where T
    a, c = p
    return λ^2 + a * λ^2 * exp(-λ) + T(5.0) * λ + c * exp(-λ)
end
function D_pda(λ::T, p) where T
    P, A = p
    return λ^2 + T(0.1) * λ + one(T) + (P + T(0.1) * λ + A * λ^2) * exp(-λ)
end

# ranges / ω_max / n_power_max exactly as the gallery SPECS; `nc` = index of the
# neutral coefficient (|coef| < 1 is the region outside essential instability)
CASES = [
    (id = "A4_neutral",    D = D_neutral,    xr = (-1.2, 1.2), yr = (-1.2, 1.2),  ω = 200.0, npow = 2.0, nc = 1),
    (id = "A5_neutral_hg", D = D_neutral_hg, xr = (-1.2, 1.2), yr = (-1.0, 10.0), ω = 200.0, npow = 2.0, nc = 1),
    (id = "A6_pda",        D = D_pda,        xr = (-1.1, 1.4), yr = (-1.15, 1.15), ω = 500.0, npow = 2.0, nc = 2),
]
const NG = 40
const ODE_TOL = 1e-9

grid_rows = Any[]
sum_rows = Any[]
for cs in CASES
    xv = range(cs.xr...; length = NG); yv = range(cs.yr...; length = NG)
    pts = vec([(x, y) for x in xv, y in yv])
    # warm-up (compilation) outside the timers
    calculate_unstable_roots_p_vec(cs.D, pts[1:2]; n_roots_to_track = 0, ω_max = cs.ω,
        reltol = ODE_TOL, abstol = ODE_TOL, n_power_max = cs.npow)
    calculate_unstable_roots_unwrap_p_vec(cs.D, pts[1:2]; n_roots_to_track = 0, n_power_max = cs.npow)
    t_ode = @elapsed Zo, Zro = calculate_unstable_roots_p_vec(cs.D, pts; n_roots_to_track = 0,
        ω_max = cs.ω, reltol = ODE_TOL, abstol = ODE_TOL, n_power_max = cs.npow)
    t_ud = @elapsed Zd, Zrd = calculate_unstable_roots_unwrap_p_vec(cs.D, pts; n_roots_to_track = 0,
        n_power_max = cs.npow)                                   # defaults: ω_max = 1e5, tol = 0.3
    t_ug = @elapsed Zg, Zrg = calculate_unstable_roots_unwrap_p_vec(cs.D, pts; n_roots_to_track = 0,
        ω_max = cs.ω, n_power_max = cs.npow)                     # gallery ω_max
    evd = uw_evals_vec(cs.D, pts)
    evg = uw_evals_vec(cs.D, pts; ω_max = cs.ω)
    npow_est = get_n_power_max(cs.D, pts[1])
    inside = [abs(p[cs.nc]) < 1 for p in pts]
    odd = [parity_expected_odd(cs.D, p) for p in pts]
    for i in eachindex(pts)
        push!(grid_rows, (cs.id, pts[i][1], pts[i][2], inside[i], Zo[i], Zro[i], Zd[i], Zrd[i], evd[i],
            Zg[i], Zrg[i], evg[i], odd[i] === missing ? "" : Int(odd[i])))
    end
    ro, rd, rg = resid.(Zro), resid.(Zrd), resid.(Zrg)
    for (lbl, Z, Zr, r, ev, t, wm) in (("ODE Vern9 tol=1e-9", Zo, Zro, ro, fill(0, length(pts)), t_ode, cs.ω),
                                       ("unwrap default (wmax=1e5)", Zd, Zrd, rd, evd, t_ud, 1e5),
                                       ("unwrap gallery wmax", Zg, Zrg, rg, evg, t_ug, cs.ω))
        dif_all = count(Z .!= Zo); dif_in = count((Z .!= Zo) .& inside)
        dif_out = count((Z .!= Zo) .& .!inside)
        nfail = count(<(0), Z)
        par_in = count(i -> inside[i] && parity_ok(Z[i], odd[i]) === false, eachindex(pts))
        push!(sum_rows, (cs.id, lbl, wm, length(pts), count(inside), dif_all, dif_in, dif_out,
            nfail, lbl[1:3] == "ODE" ? "" : median(ev), lbl[1:3] == "ODE" ? "" : maximum(ev),
            fmed(r), fmax(r), count(>(0.25), filter(isfinite, r)),
            count(>(0.25), filter(isfinite, r[inside])), par_in, t, npow_est))
        @printf("%-14s %-27s wmax=%-7.0f diffZ all/in/out = %4d/%4d/%4d fail=%d evals med/max=%s/%s  eps med=%.2e max=%.3f n(eps>1/4)=%d (in %d) parity-viol(in)=%d  t=%.2fs\n",
            cs.id, lbl, wm, dif_all, dif_in, dif_out, nfail, string(sum_rows[end][10]), string(sum_rows[end][11]),
            fmed(r), fmax(r), sum_rows[end][14], sum_rows[end][15], par_in, t)
    end
    println("   n_power estimate at first point: ", npow_est)
end
write_rows("check1_neutral_grid",
    ["case", "x", "y", "inside_abs_coef_lt_1", "Z_ode", "Zraw_ode", "Z_uw_default", "Zraw_uw_default",
     "evals_uw_default", "Z_uw_gallerywmax", "Zraw_uw_gallerywmax", "evals_uw_gallerywmax",
     "parity_expected_odd"], grid_rows)
write_rows("check1_neutral_summary",
    ["case", "method", "omega_max", "n_points", "n_inside", "nZdiff_vs_ode_all", "nZdiff_inside",
     "nZdiff_outside", "n_failed", "evals_median", "evals_max", "eps_median", "eps_max",
     "n_eps_gt_quarter", "n_eps_gt_quarter_inside", "n_parity_viol_inside", "wall_time_s",
     "npow_estimate_first_point"], sum_rows)

# ω_max scaling of the unwrap evaluation count, A.4 neutral
wrows = Any[]
for p in ((0.0, 0.5), (0.5, -0.5), (0.95, 0.5), (-1.1, 0.5))
    zref = calculate_unstable_roots_direct(D_neutral, p; n_roots_to_track = 0, ω_max = 200.0,
        reltol = ODE_TOL, abstol = ODE_TOL, n_power_max = 2.0)
    for wm in (1e2, 1e3, 1e4, 1e5)
        Z, Zr = calculate_unstable_roots_unwrap(D_neutral, p; n_roots_to_track = 0, ω_max = wm,
            n_power_max = 2.0)
        ev = uw_evals(D_neutral, p; ω_max = wm)
        t = (calculate_unstable_roots_unwrap(D_neutral, p; n_roots_to_track = 0, ω_max = wm, n_power_max = 2.0);
             minimum(@elapsed(calculate_unstable_roots_unwrap(D_neutral, p; n_roots_to_track = 0,
                 ω_max = wm, n_power_max = 2.0)) for _ in 1:3))
        push!(wrows, (p[1], p[2], wm, Z, Zr, resid(Zr), ev, ev / wm, 1e6t, zref[1], zref[2]))
        @printf("A4 p=(%5.2f,%5.2f) wmax=%.0e  Z=%d Zraw=%.4f evals=%d evals/wmax=%.3f t=%.0f us  (ODE ref Z=%d Zraw=%.4f)\n",
            p..., wm, Z, Zr, ev, ev / wm, 1e6t, zref[1], zref[2])
    end
end
write_rows("check1_wmax_scaling",
    ["a", "c", "omega_max", "Z_uw", "Zraw_uw", "eps_uw", "evals_uw", "evals_per_unit_omega",
     "time_us", "Z_ode_ref_wmax200", "Zraw_ode_ref_wmax200"], wrows)

# ===========================================================================
# BigFloat Newton (independent reference roots)
# ===========================================================================
function newton_big(D, p, λ0; prec = 256, iters = 40)
    setprecision(BigFloat, prec) do
        pb = BigFloat.(p)
        λ = Complex{BigFloat}(λ0)
        h = BigFloat(10)^(-30)
        for _ in 1:iters
            f = D(λ, pb)
            df = (D(λ + h, pb) - D(λ - h, pb)) / (2h)
            δ = f / df
            λ -= δ
            abs(δ) < BigFloat(10)^(-60) && break
        end
        return λ, abs(D(λ, pb))
    end
end

# ===========================================================================
# CHECK 2 -- rightmost-root accuracy at the tab:semidisc points
# ===========================================================================
const SC = SYSTEMS["showcase"]
Dsc = SC.ref                                  # = D_showcase_reduced of paper/scripts/studies/systems.jl
POINTS = [("stable", (1.8, 1.0)), ("near boundary", (2.4, 1.4)), ("unstable", (3.2, 0.6))]
METHODS_REF = (:Linear, :Newton, :Combined)
argmaxre(es, wc) = (g = findall(isfinite, es); i = g[argmax(view(es, g))]; (es[i], wc[i]))

r2 = Any[]
for (lbl, p) in POINTS
    # reference as in s07 (ODE tol 1e-9, 10 roots, Newton 15) then BigFloat-polished
    _, _, _, es, wc = calculate_unstable_roots_direct(Dsc, p; ω_max = 1e4, reltol = 1e-9,
        abstol = 1e-9, n_roots_to_track = 10, n_power_max = 4, refinement_method = :Newton,
        refinement_steps = 15)
    s0, w0 = argmaxre(es, wc)
    λb, resb = newton_big(Dsc, p, complex(s0, w0))
    σref = Float64(real(λb)); ωref = Float64(imag(λb))
    # independent cross-check: every unwrap-tracked minimum (10), BigFloat-polished, max Re
    _, _, _, eu, wu = calculate_unstable_roots_unwrap(Dsc, p; n_roots_to_track = 10, n_power_max = 4)
    pol = [newton_big(Dsc, p, complex(eu[j], wu[j]); prec = 128)[1] for j in eachindex(eu) if isfinite(eu[j])]
    σalt = maximum(Float64(real(z)) for z in pol)
    @printf("%-14s ref root %.15f %+.12fi  |D|=%.1e  (max-Re over polished unwrap minima: %.15f)\n",
        lbl, σref, ωref, Float64(resb), σalt)
    for m in METHODS_REF
        a = calculate_unstable_roots_direct(Dsc, p; ω_max = 1e4, n_power_max = 4,
            n_roots_to_track = 1, refinement_method = m)
        a10 = calculate_unstable_roots_direct(Dsc, p; ω_max = 1e4, n_power_max = 4,
            n_roots_to_track = 10, refinement_method = m)
        b = calculate_unstable_roots_unwrap(Dsc, p; n_power_max = 4, n_roots_to_track = 1,
            refinement_method = m)
        c = calculate_unstable_roots_unwrap(Dsc, p; n_power_max = 4, n_roots_to_track = 10,
            refinement_method = m)
        for (meth, s, w, Z) in (("ODE Vern9 1e-5; 1 root", a[4], a[5], a[1]),
                                ("ODE Vern9 1e-5; 10 roots max-Re", argmaxre(a10[4], a10[5])..., a10[1]),
                                ("unwrap; 1 root", b[4], b[5], b[1]),
                                ("unwrap; 10 roots max-Re", argmaxre(c[4], c[5])..., c[1]))
            push!(r2, (lbl, p[1], p[2], meth, m, Z, s, w, abs(s - σref), abs(complex(s, w) - complex(σref, ωref)),
                σref, ωref, Float64(resb)))
            @printf("   %-32s %-9s Z=%d  σ̂=%+.12f ω̂=%.6f  |σ̂-σ|=%.2e  |λ̂-λ|=%.2e\n", meth, m, Z, s, w,
                abs(s - σref), abs(complex(s, w) - complex(σref, ωref)))
        end
    end
end
write_rows("check2_root_accuracy",
    ["point", "P", "D", "method", "refinement", "Z", "sigma_hat", "omega_hat", "abs_err_sigma",
     "abs_err_lambda", "sigma_ref", "omega_ref", "ref_abs_D_bigfloat"], r2)

# ===========================================================================
# CHECK 3 -- boundary stress, both sides of the showcase Hopf boundary (D = 1.5)
# ===========================================================================
const DG = 1.5
σ_of(P) = calculate_unstable_roots_direct(Dsc, (P, DG); ω_max = 1e4, reltol = 1e-10, abstol = 1e-10,
    n_power_max = 4, refinement_method = :Newton, refinement_steps = 15)[4]
PB = let lo = 2.0, hi = 3.0           # exactly as s13 / s03
    for _ in 1:60
        m = (lo + hi) / 2
        σ_of(m) < 0 ? (lo = m) : (hi = m)
    end
    (lo + hi) / 2
end
# crossing root at P_b (seed) and the BigFloat crossing parameter for reference
λc0 = let r = calculate_unstable_roots_direct(Dsc, (PB, DG); ω_max = 1e4, reltol = 1e-10, abstol = 1e-10,
        n_power_max = 4, refinement_method = :Newton, refinement_steps = 15)
    complex(r[4], r[5])
end
crossing(P) = newton_big(Dsc, (P, DG), λc0)[1]       # P may be Float64 (exact) or BigFloat
PB_big = setprecision(BigFloat, 256) do
    a, b = big(PB) - big(1e-3), big(PB) + big(1e-3)
    fa, fb = real(crossing(a)), real(crossing(b))
    for _ in 1:12                                    # secant on Re λ_c(P)
        c = b - fb * (b - a) / (fb - fa)
        a, fa, b, fb = b, fb, c, real(crossing(c))
        abs(fb) < big(10)^(-70) && break
    end
    b
end
@printf("P_b (Float64 bisection, as s13) = %.17f ; BigFloat crossing P_b = %.20f ; diff = %.2e\n",
    PB, Float64(PB_big), Float64(big(PB) - PB_big))
Zminus = calculate_unstable_roots_direct(Dsc, (PB - 1e-2, DG); n_roots_to_track = 0, ω_max = 1e4,
    reltol = 1e-10, abstol = 1e-10, n_power_max = 4)[1]
Zplus = calculate_unstable_roots_direct(Dsc, (PB + 1e-2, DG); n_roots_to_track = 0, ω_max = 1e4,
    reltol = 1e-10, abstol = 1e-10, n_power_max = 4)[1]
println("Z at P_b -/+ 1e-2 (ODE 1e-10): ", Zminus, " / ", Zplus)

r3 = Any[]
for s in (-1, 1), d in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
    P = PB + s * d
    p = (P, DG)
    λt = crossing(P)
    σt = Float64(real(λt))
    Ztrue = real(λt) > 0 ? Zplus : Zminus
    odd = parity_expected_odd(Dsc, p)
    od = calculate_unstable_roots_direct(Dsc, p; ω_max = 1e4, n_power_max = 4)              # default tol 1e-5
    o8 = calculate_unstable_roots_direct(Dsc, p; ω_max = 1e4, reltol = 1e-8, abstol = 1e-8, n_power_max = 4)
    uw = calculate_unstable_roots_unwrap(Dsc, p; n_power_max = 4)
    ev = uw_evals(Dsc, p)
    sgn_ok(σh) = sign(σh) == sign(σt)
    push!(r3, (s * d, Float64(big(P) - PB_big), Ztrue, σt, Float64(imag(λt)),
        od[1], od[2], od[4], od[5], sgn_ok(od[4]), parity_ok(od[1], odd),
        o8[1], o8[2], o8[4], sgn_ok(o8[4]), parity_ok(o8[1], odd),
        uw[1], uw[2], uw[4], uw[5], sgn_ok(uw[4]), parity_ok(uw[1], odd), ev,
        odd === missing ? "" : Int(odd)))
    @printf("dP=%+.0e  Ztrue=%d σ_true=%+.3e | ODE1e-5 Z=%d σ̂=%+.3e sgn=%s par=%s | ODE1e-8 Z=%d σ̂=%+.3e sgn=%s par=%s | unwrap Z=%d σ̂=%+.3e sgn=%s par=%s ev=%d\n",
        s * d, Ztrue, σt, od[1], od[4], sgn_ok(od[4]), parity_ok(od[1], odd), o8[1], o8[4],
        sgn_ok(o8[4]), parity_ok(o8[1], odd), uw[1], uw[4], sgn_ok(uw[4]), parity_ok(uw[1], odd), ev)
end
write_rows("check3_boundary_stress",
    ["dP", "P_minus_Pb_bigfloat", "Z_true", "sigma_true", "omega_true",
     "Z_ode_1e-5", "Zraw_ode_1e-5", "sigma_hat_ode_1e-5", "omega_hat_ode_1e-5", "sign_ok_ode_1e-5", "parity_ok_ode_1e-5",
     "Z_ode_1e-8", "Zraw_ode_1e-8", "sigma_hat_ode_1e-8", "sign_ok_ode_1e-8", "parity_ok_ode_1e-8",
     "Z_uw", "Zraw_uw", "sigma_hat_uw", "omega_hat_uw", "sign_ok_uw", "parity_ok_uw", "evals_uw",
     "parity_expected_odd"], r3)
open(joinpath(OUT, "check3_meta.txt"), "w") do io
    @printf(io, "P_b_float64_bisection=%.17f\nP_b_bigfloat=%s\nD=%.2f\nZ_minus=%d\nZ_plus=%d\nD(0)=%.6g\nD(1e6)=%.6g\n",
        PB, string(PB_big), DG, Zminus, Zplus, real(Dsc(complex(0.0), (PB, DG))), real(Dsc(complex(1e6), (PB, DG))))
end
println("s16 done")
