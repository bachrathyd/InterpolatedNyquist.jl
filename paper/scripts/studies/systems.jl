# Shared system definitions and chart helpers used by several studies.
# (include-guarded so multiple studies can include it in one session)

if !@isdefined(SYSTEMS_INCLUDED)

using InterpolatedNyquist
using OrdinaryDiffEq
using ForwardDiff
using StaticArrays
using LinearAlgebra
using MDBM

# ===========================================================================
# THE SHOWCASE SYSTEM: constrained 2-DOF structure with delayed PD control
# (statically unstable main mass -- inverted-pendulum-like, k1 < 0 -- carrying
#  a rigidly locked absorber unit: three-mass chain with masses 2 and 3
#  rigidly linked -> DAE, singular mass matrix. Delayed collocated PD control
#  force acts on the main mass.)
# ===========================================================================
const SM = (m1 = 1.0, m2 = 0.3, m3 = 0.2, k1 = -1.0, k2 = 1.0,
            c1 = 0.05, c2 = 0.05, tau = 0.5)

function showcase_rhs(z, h, p, t)
    P, D = p
    x1, x2, x3, v1, v2, v3, Fc = z
    x1d = h(p, t - SM.tau; idxs = 1)
    v1d = h(p, t - SM.tau; idxs = 4)
    u = -P * x1d - D * v1d
    return SA[
        v1,
        v2,
        v3,
        -SM.k1 * x1 - SM.c1 * v1 + SM.k2 * (x2 - x1) + SM.c2 * (v2 - v1) + u,
        -SM.k2 * (x2 - x1) - SM.c2 * (v2 - v1) + Fc,
        -Fc,
        x3 - x2,
    ]
end

const E_SHOWCASE = SMatrix{7,7,Float64}(Diagonal(SA[1.0, 1.0, 1.0, SM.m1, SM.m2, SM.m3, 0.0]))

"Characteristic function of the showcase DAE, extracted automatically from the RHS."
D_showcase(λ, p) = get_D_from_model(showcase_rhs, λ, p, Val(7); mass_matrix = E_SHOWCASE)

"Hand-derived reduced 2-DOF quasi-polynomial (verification reference)."
function D_showcase_reduced(λ::T, p) where T
    P, D = p
    m23 = SM.m2 + SM.m3
    a11 = SM.m1 * λ^2 + (SM.c1 + SM.c2) * λ + (SM.k1 + SM.k2) + (P + D * λ) * exp(-SM.tau * λ)
    a12 = -(SM.c2 * λ + SM.k2)
    a21 = -(SM.c2 * λ + SM.k2)
    a22 = m23 * λ^2 + SM.c2 * λ + SM.k2
    return a11 * a22 - a12 * a21
end

# Parameter window of the showcase chart (tuned for a bounded stable island)
const SHOWCASE_PRANGE = (0.5, 3.0)
const SHOWCASE_DRANGE = (-0.5, 3.5)

"First-order (A, B) matrices of the reduced showcase system, x'(t) = A x(t) + B x(t-tau)."
function showcase_AB(p)
    P, Dg = p
    m23 = SM.m2 + SM.m3
    K = [SM.k1 + SM.k2 -SM.k2; -SM.k2 SM.k2]
    C = [SM.c1 + SM.c2 -SM.c2; -SM.c2 SM.c2]
    Minv = [1 / SM.m1 0.0; 0.0 1 / m23]
    A = [zeros(2, 2) I; -Minv*K -Minv*C]
    B = zeros(4, 4)
    B[3, 1] = -P / SM.m1
    B[3, 3] = -Dg / SM.m1
    return A, B
end

# ===========================================================================
# 4th-order delayed oscillator (benchmark workhorse, same as tests/examples)
# ===========================================================================
function D_fourth(λ::T, p) where T
    P, D = p
    c1 = T(0.03); τ = T(0.5); ζ = T(0.02)
    return c1 * λ^4 + λ^2 + 2ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ)
end

# ===========================================================================
# Chart helpers
# ===========================================================================
"""
Threaded brute-force sweep over a 2-parameter grid; returns matrices + wall time.

Extra keywords (`n_power_max`, tolerances, ...) are forwarded to the solver, so
a panel whose leading order is known by inspection can pass `n_power_max = 2`
and skip the estimation entirely.
"""
function sweep_grid(D_func, xv, yv; kwargs...)
    params = vec([(x, y) for x in xv, y in yv])
    calculate_unstable_roots_p_vec(D_func, params[1:1]; kwargs...)  # JIT warm-up, outside the timer
    t0 = time()
    Z_ints, Z_raws, min_Ds, sigmas, crits = calculate_unstable_roots_p_vec(D_func, params; kwargs...)
    t = time() - t0
    nx, ny = length(xv), length(yv)
    return (Z = reshape(Z_ints, nx, ny), Z_raw = reshape(Z_raws, nx, ny),
            min_D = reshape(min_Ds, nx, ny), sigma = reshape(sigmas, nx, ny),
            omega = reshape(crits, nx, ny), t = t)
end

"""
Threaded sweep tracking several roots per point; the `sigma` field holds the
maximal real part of the tracked roots (the dominant root), which is the
correct spectral-gap measure even where root branches cross. Uses the package
default refinement (`:Linear`, i.e. the raw first-order estimate): it is the
smoothest field over a chart, which is exactly what the interpolable colouring
wants. Pass `refinement_method` through `kwargs` for an accurate map.
"""
# nroots = 15 by default: the tracker keeps the N DEEPEST |D| dips (robust --
# a spurious minimum with a garbage sigma estimate is shallow and gets evicted),
# and the dominant (rightmost) root is picked by max-Re over them. With only 5
# slots, deeper NON-dominant roots can crowd the dominant one out at some
# pixels, so its sigma jumps -- the speckle in the stable colouring. Measured on
# the showcase: 5 -> 15 roots drops the rough pixels 78 -> 34 (the real
# branch-crossing floor), tolerance-independent, and the march cost is flat in
# nroots. Ranking the buffer by real part instead would be cheaper but is
# fragile: the raw sigma estimate blows up at degenerate minima (|D'| ~ 0,
# common in transcendental D), and a real-part ranked buffer would surface that
# garbage. Depth-ranking avoids it.
function sweep_grid_dominant(D_func, xv, yv; nroots = 15, kwargs...)
    params = vec([(x, y) for x in xv, y in yv])
    calculate_unstable_roots_p_vec(D_func, params[1:1]; n_roots_to_track = nroots, kwargs...)  # JIT warm-up
    t0 = time()
    Z_ints, Z_raws, min_Ds, es_list, wc_list =
        calculate_unstable_roots_p_vec(D_func, params; n_roots_to_track = nroots, kwargs...)
    t = time() - t0
    sig = map(es_list) do v
        m = maximum(filter(isfinite, v); init = -Inf)
        isfinite(m) ? m : NaN
    end
    nx, ny = length(xv), length(yv)
    return (Z = reshape(Z_ints, nx, ny), Z_raw = reshape(Z_raws, nx, ny),
            sigma = reshape(sig, nx, ny), t = t)
end

# ===========================================================================
# Unwrap chart back-end (CHART_BACKEND in common.jl)
# ===========================================================================
"""
`sweep_grid_dominant` with the discrete phase-unwrapping march
(`calculate_unstable_roots_unwrap_p_vec`). `D_func` must be ENTIRE -- rational
characteristic functions with their (stable) denominators cleared -- and
`n_power_max` is its leading order (or the effective order of a cleared
denominator, see `denominator_order`). Same outputs as `sweep_grid_dominant`.
The march finishes most charts in milliseconds, where one run is dominated by
scheduling noise, so the time is the best of `reps` runs (a run over 5 s is
timed once).
"""
function sweep_grid_dominant_unwrap end

"""
Root estimate of a |D| minimum AT the origin, exactly as the phase-ODE back-end
records it: one Newton step from λ = σ + 10⁻⁹i, taken when |D|² grows from
ω = 0. The march records this estimate only inside the trust region
|D/D'| <= max(|σ|, 1), so a real dominant root further left is lost from its
buffer -- 15 % of the stable pixels of the delayed-oscillator panel were left
without any σ (painted as boundary colour). NaN when |D| does not grow from 0.
"""
function origin_estimate(D_func, p; σ = 0.0)
    d = ForwardDiff.Dual(1e-9, 1.0)
    res = D_func(σ + 1im * d, p)
    vr, vi = ForwardDiff.value(real(res)), ForwardDiff.value(imag(res))
    dr, di = ForwardDiff.partials(real(res), 1), ForwardDiff.partials(imag(res), 1)
    vr * dr + vi * di > 0 || return NaN
    s = σ - (vr * di - vi * dr) / (di^2 + dr^2)
    return isfinite(s) ? s : NaN
end

function sweep_grid_dominant_unwrap(D_func, xv, yv; nroots = 15, n_power_max, reps = 3,
                                    kwargs...)
    params = vec([(x, y) for x in xv, y in yv])
    calculate_unstable_roots_unwrap_p_vec(D_func, params[1:2]; n_roots_to_track = nroots,
        n_power_max = n_power_max, kwargs...)                     # JIT warm-up
    res = nothing
    s0 = fill(NaN, length(params))
    t = Inf
    for _ in 1:reps
        t0 = time_ns()
        res = calculate_unstable_roots_unwrap_p_vec(D_func, params; n_roots_to_track = nroots,
            n_power_max = n_power_max, kwargs...)
        Threads.@threads for i in eachindex(params)
            s0[i] = origin_estimate(D_func, params[i])
        end
        t = min(t, (time_ns() - t0) / 1e9)
        t > 5 && break
    end
    Z_ints, Z_raws, _, es_list, _ = res
    # Consistency filter for the colouring. Where the count proves the point
    # stable (Z = 0) no root lies right of the line, so a tracked estimate with
    # σ_est > 0 there is spurious: a shallow minimum of the high-frequency delay
    # ripple, located on a long step's Hermite model, can return σ_est in the
    # hundreds (seen on the fractional Gao charts), which max-Re would pick and
    # the |σ| colour scale would paint as deeply stable. Such estimates are
    # dropped (a margin of 0.02 keeps genuine near-boundary roots).
    sig = map(Z_ints, es_list, s0) do z, v, o
        m = maximum(filter(s -> isfinite(s) && (z != 0 || s <= 0.02), [v; o]); init = -Inf)
        isfinite(m) ? m : NaN
    end
    nx, ny = length(xv), length(yv)
    return (Z = reshape(Z_ints, nx, ny), Z_raw = reshape(Z_raws, nx, ny),
            sigma = reshape(sig, nx, ny), t = t)
end

"""
    denominator_order(Dden, ω_max; p = (1.0, 1.0))

Effective order n_eff = 2Φ_den/π of a PARAMETER-INDEPENDENT stable denominator,
Φ_den its phase increment over [0, ω_max] measured by the same march. A rational
D = N/Dden written as a return difference (D -> 1, n = 0) has
Φ_D = Φ_N - Φ_den, so counting the entire numerator N with n_power_max = n_eff
gives exactly the return-difference count, with the same truncation at ω_max,
while the march never sees the poles. Dden must have no zeros in Re λ >= 0.
"""
denominator_order(Dden, ω_max; p = (1.0, 1.0)) =
    -2 * calculate_unstable_roots_unwrap(Dden, p; n_roots_to_track = 0, ω_max = ω_max,
        n_power_max = 0.0)[2]

"""
    chart_grid(D, xv, yv; nroots, ω_max, ode_kw, uw)

The sweep behind every example chart. `uw === nothing` (or CHART_BACKEND[] ==
:ode) -> the phase-ODE back-end `sweep_grid_dominant(D, ...; ode_kw...)`.
Otherwise `uw = (D = entire form, n_power_max = n, kw = march options)` and the
chart is computed with the unwrap march; with CHART_CHECK[] the ODE sweep is
repeated on the same grid, and every point where the two counts differ is
re-counted by the ODE back-end at reltol = abstol = 1e-10 (on the entire form)
to tell which one is off. Returns the grid fields plus `backend`, `t_ode` (the
ODE grid on the same machine, one run; the unwrap time is the best of three)
and `check`.
"""
function chart_grid(D, xv, yv; nroots = 15, ω_max, ode_kw = (;), uw = nothing)
    if uw === nothing || CHART_BACKEND[] == :ode
        g = sweep_grid_dominant(D, xv, yv; nroots = nroots, ω_max = ω_max, ode_kw...)
        return merge(g, (backend = "ode", t_ode = g.t, check = nothing, uw_npow = NaN))
    end
    g = sweep_grid_dominant_unwrap(uw.D, xv, yv; nroots = nroots, ω_max = ω_max,
        n_power_max = uw.n_power_max, uw.kw...)
    check = nothing
    if CHART_CHECK[]
        go = sweep_grid_dominant(D, xv, yv; nroots = nroots, ω_max = ω_max, ode_kw...)
        params = [(x, y) for x in xv, y in yv]
        # Referee at the differing points: the phase ODE at reltol = abstol =
        # 1e-10 on the ENTIRE form with the march's order -- an independent
        # algorithm, and free of the pole-zero pairs of a rational form. A point
        # whose count changes when the line is shifted by ±1e-6 has a root ON
        # the integration line (e.g. lambda = 0 at K = 1 on the bar panels, the
        # pair ±i on the a = c diagonal of the neutral panel): its count is
        # undefined ("degenerate"), and neither back-end can be called wrong
        # there. (The residual is no test for this on the neutral panels, whose
        # tail oscillation alone reaches 0.5.)
        idx = findall(g.Z .!= go.Z)
        diffs = Vector{Tuple}(undef, length(idx))
        isdeg = falses(length(idx))
        kref = (n_roots_to_track = 0, ω_max = ω_max, n_power_max = uw.n_power_max,
                reltol = 1e-10, abstol = 1e-10)
        Threads.@threads for k in eachindex(idx)
            i = idx[k]
            zref, zrref = calculate_unstable_roots_direct(uw.D, params[i], 0.0; kref...)
            zp = calculate_unstable_roots_direct(uw.D, params[i], 1e-6; kref...)[1]
            zm = calculate_unstable_roots_direct(uw.D, params[i], -1e-6; kref...)[1]
            isdeg[k] = zp != zm
            diffs[k] = (params[i]..., go.Z[i], go.Z_raw[i], g.Z[i], g.Z_raw[i], zref, zrref, isdeg[k])
        end
        degen(d) = d[9]
        check = (t_ode = go.t, n_diff = length(diffs),
                 n_degenerate = count(degen, diffs),
                 n_uw_wrong = count(d -> !degen(d) && d[5] != d[7], diffs),
                 n_ode_wrong = count(d -> !degen(d) && d[3] != d[7], diffs),
                 n_class_diff = count((g.Z .== 0) .!= (go.Z .== 0)),
                 diffs = diffs, Z_ode = go.Z, Z_raw_ode = go.Z_raw)
        @info "unwrap vs ODE counts" n_points = length(g.Z) n_diff = check.n_diff check.n_degenerate check.n_uw_wrong check.n_ode_wrong check.n_class_diff t_unwrap = g.t t_ode = go.t
    end
    return merge(g, (backend = "unwrap", t_ode = check === nothing ? NaN : check.t_ode,
                     check = check, uw_npow = Float64(uw.n_power_max)))
end

"""
Cache-name suffix of a chart grid: unwrap grids never reuse an ODE cache and
vice versa, and the march settings (order, step caps) are part of the name.
"""
chart_suffix(uw) = (CHART_BACKEND[] == :unwrap && uw !== nothing) ?
    "_uw3" * string(hash((Float64(uw.n_power_max), uw.kw)); base = 16) : ""

# The showcase chart (s01, reused by s09): the reduced form is an entire
# quasi-polynomial of order 4. The step cap over the resonance band ω < 6 is
# for the COLOURING, not the count: the counts are identical without it, but
# the march then locates some |D| minima on too coarse a Hermite model, and at
# ~1% of the stable pixels the dominant root is lost from the σ field (errors
# up to 0.28 against Newton-refined roots, vs 0.06 for the ODE grid; with the
# cap: max 0.06, median 6e-4 vs 5e-3 for the ODE grid).
const SHOWCASE_UW = (D = D_showcase_reduced, n_power_max = 4.0, kw = (hmax = 0.1, ωband = 6.0))

"Write data/unwrap_check_<study>.csv (one row per chart) and ..._points.csv (every differing point)."
function write_chart_checks(study, charts)
    med_res(Zr) = (r = filter(isfinite, abs.(vec(Zr) .- round.(vec(Zr)))); isempty(r) ? NaN : median(r))
    rows = Tuple[]
    pts = Tuple[]
    for (id, g, settings) in charts
        c = g.check
        nc(f) = c === nothing ? -1 : getfield(c, f)
        push!(rows, (id, g.backend, length(g.Z), g.t, g.t_ode, nc(:n_diff), nc(:n_degenerate),
            nc(:n_uw_wrong), nc(:n_ode_wrong), nc(:n_class_diff), g.uw_npow, med_res(g.Z_raw),
            c === nothing ? NaN : med_res(c.Z_raw_ode), settings))
        c === nothing || foreach(d -> push!(pts, (id, d...)), c.diffs)
    end
    write_csv("unwrap_check_$(study)",
        ["chart", "backend", "n_points", "chart_time_s", "ode_time_s_same_machine", "n_diff_vs_ode",
         "n_diff_degenerate", "n_diff_unwrap_wrong", "n_diff_ode_wrong", "n_stability_class_diff",
         "unwrap_n_power", "median_int_residual", "median_int_residual_ode", "unwrap_settings"], rows)
    write_csv("unwrap_check_$(study)_points",
        ["chart", "x", "y", "Z_ode", "Zraw_ode", "Z_unwrap", "Zraw_unwrap", "Z_ref_1e-10",
         "Zraw_ref_1e-10", "root_on_line"], pts)
end

"""
MDBM boundary trace with the scalar sign(Z==0)*|σ_dom - σ| objective.

The objective uses the DOMINANT root -- several tracked |D| minima, all
refined, maximal real part -- not the single closest minimum. This matters and
is not a refinement of taste: the single tracked minimum switches root branch
wherever another mode becomes the closest one, so its σ_est jumps
discontinuously (on the showcase system, from -0.194 to -0.053 across one cell
as ω_crit jumps 0.96 -> 2.34). MDBM fits a linear model inside each cell, and
across such a cliff it manufactures a spurious zero -- a phantom "boundary"
point in the middle of the stable domain, which then blocks the largest
inscribed circle. The dominant root varies smoothly through the branch
crossing (it is the max over branches, not the nearest one), so the objective
stays continuous and the trace stays clean.
"""
function mdbm_boundary(D_func, xrange, yrange; ngrid = 30, Niter = 4, σ = 0.0,
                       ω_max = 1e6, reltol = 1e-5, abstol = 1e-5, nroots = 5,
                       n_power_max = nothing)
    function wrapper(x, y)::Float64
        zi, zr, md, es, wc = calculate_unstable_roots_direct(D_func, (x, y), σ;
            ω_max = ω_max, reltol = reltol, abstol = abstol,
            n_roots_to_track = nroots, n_power_max = n_power_max)
        sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
        σ_dom = maximum(filter(isfinite, es); init = -Inf)
        g = sign_val * abs(σ_dom - σ)
        # undefined root estimate (D' ≡ 0, e.g. zero-gain edge): far from boundary
        return isfinite(g) ? g : sign_val * 1.0e3
    end
    prob = MDBM_Problem(wrapper, [LinRange(xrange..., ngrid), LinRange(yrange..., ngrid)])
    t = @elapsed MDBM.solve!(prob, Niter, verbosity = 0,interpolationorder=0)
    println("--------------------- interp 0-1 -----------------------------")
    MDBM.interpolate!(prob,interpolationorder=1)
    xyz = getinterpolatedsolution(prob)
    DT1 = MDBM.connect(prob)
    edges = if isempty(DT1)
        nothing
    else
        [reduce(hcat, [s[getindex.(DT1, 1)], s[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for s in xyz]
    end
    return (prob = prob, edges = edges, t = t)
end

"Standard stability-chart panel: combined-metric heatmap + boundary curve."
function stability_panel!(ax, xv, yv, C; edges = nothing, crange = nothing, rasterize = 8)
    hm = if crange === nothing
        heatmap!(ax, xv, yv, C; colormap = CHART_CMAP, rasterize = rasterize)
    else
        heatmap!(ax, xv, yv, C; colormap = CHART_CMAP, colorrange = crange, rasterize = rasterize)
    end
    if edges !== nothing
        lines!(ax, edges[1], edges[2]; color = BOUNDARY_COLOR, linewidth = BOUNDARY_LW)
    end
    return hm
end

"Phase-ODE solution with all accepted adaptive steps saved (for the walkthrough figure)."
function phase_ode_solution(D_func, p; σ = 0.0, ω_max = 1e6, reltol = 1e-5, abstol = 1e-5)
    function f(y, _, ω)
        w = max(ω, 1e-9)
        Dval = D_func(σ + 1im * w, p)
        Dp = ForwardDiff.derivative(x -> D_func(σ + 1im * x, p), w)
        return SA[imag(Dp / Dval)]
    end
    prob = ODEProblem{false}(f, SA[0.0], (0.0, Float64(ω_max)))
    return solve(prob, Vern9();
        reltol = reltol, abstol = abstol, save_everystep = true, maxiters = 10^6)
end

"Pointwise phase integrand (dense sampling for reference curves)."
function phase_integrand(D_func, p, ω; σ = 0.0)
    w = max(ω, 1e-9)
    Dval = D_func(σ + 1im * w, p)
    Dp = ForwardDiff.derivative(x -> D_func(σ + 1im * x, p), w)
    return imag(Dp / Dval)
end

const SYSTEMS_INCLUDED = true
end # include guard
