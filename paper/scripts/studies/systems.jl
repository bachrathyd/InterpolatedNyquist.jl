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
const SHOWCASE_PRANGE = (0.0, 4.0)
const SHOWCASE_DRANGE = (-0.5, 3.0)

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
"Threaded brute-force sweep over a 2-parameter grid; returns matrices + wall time."
function sweep_grid(D_func, xv, yv; kwargs...)
    params = vec([(x, y) for x in xv, y in yv])
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
maximal real part of the refined tracked roots (the dominant root), which is
the correct spectral-gap measure even where root branches cross.
"""
function sweep_grid_dominant(D_func, xv, yv; nroots = 5, kwargs...)
    params = vec([(x, y) for x in xv, y in yv])
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

"MDBM boundary trace with the scalar sign(Z==0)*|σ_est - σ| objective."
function mdbm_boundary(D_func, xrange, yrange; ngrid = 30, Niter = 4, σ = 0.0,
                       ω_max = 1e6, reltol = 1e-5, abstol = 1e-5)
    function wrapper(x, y)::Float64
        zi, zr, md, es, wc = calculate_unstable_roots_direct(D_func, (x, y), σ;
            ω_max = ω_max, reltol = reltol, abstol = abstol)
        sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
        g = sign_val * abs(es - σ)
        # undefined root estimate (D' ≡ 0, e.g. zero-gain edge): far from boundary
        return isfinite(g) ? g : sign_val * 1.0e3
    end
    prob = MDBM_Problem(wrapper, [LinRange(xrange..., ngrid), LinRange(yrange..., ngrid)])
    t = @elapsed MDBM.solve!(prob, Niter, verbosity = 0)
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
    return solve(prob, AutoTsit5(Rosenbrock23());
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
