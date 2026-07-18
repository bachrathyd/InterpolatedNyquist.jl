# Study s08: appendix case-study gallery. One standardized stability-chart
# panel per system class (retarded, distributed, neutral, essential/PDA,
# transcendental, FEM, large matrix, fractional, multi-DOF turning).
# Produces: figures/fig_gallery_a.pdf, figures/fig_gallery_b.pdf,
#           data/gallery_timings.csv

include(joinpath(@__DIR__, "systems.jl"))
using Random
using BenchmarkTools: @belapsed

# ---------------------------------------------------------------------------
# System definitions (mirroring examples/ but headless and seeded)
# ---------------------------------------------------------------------------
D_algebraic(λ::T, p) where T = λ^2 + p[1] * λ + p[2] * exp(-T(0.5) * λ)

function D_distributed(λ::T, p) where T
    a, b = p
    τ = T(1.0)
    dist = abs(λ) < 1e-8 ? τ - (τ^2 * λ) / T(2.0) : (one(T) - exp(-τ * λ)) / λ
    return λ^2 + a * λ + b * dist
end

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

# ===========================================================================
# ELASTIC BAR WITH DELAYED BOUNDARY FEEDBACK -- Zhang & Stepan, JSV 367 (2016)
# 219-232, doi:10.1016/j.jsv.2016.01.002. This panel reproduces their Fig. 8.
#
# Physical model (their Sec. 2 and 6): a bar of length l, fixed at x = l, where
# the normal force F(l,t) = A E u'(l,t) is sensed and fed back with gain K and
# delay tau to the free end x = 0, so F(0,t) = K F(l, t-tau). With internal
# viscous ("Kelvin-Voigt") damping eta, proportional to the elastic forces:
#
#   PDE  (their 49):  u_tt - eta c^2 u_txx - c^2 u_xx = 0,   c = sqrt(E/rho)
#   BCs  (their 5-6): u(l,t) = 0,   u'(0,t) - K u'(l,t-tau) = 0
#   char (their 54):  D(lam) = lam * ( cosh(T lam / sqrt(1+eta lam)) - K e^{-tau lam} )
#
# with T = l/c the wave travel time. The lam = 0 factor is the trivial rigid
# translation (their remark after Eq. 13) and is dropped. Non-dimensionalizing
# with T = 1 leaves exactly two parameters -- the delay ratio tau/T and the
# gain K -- which are the axes of their Fig. 8, at eta_tilde = eta/T = 0.01.
#
# NOTE ON THE DAMPING: the paper's internal damping enters as
# 1/sqrt(1 + eta*lam) inside the cosh, i.e. gamma ~ sqrt(lam/eta) for large
# lam, so the high modes are damped ever harder and Re lam_k -> -inf. That is
# what makes the count finite at all: the UNDAMPED bar (their Fig. 6) has its
# roots on finitely many vertical lines and is at best marginally stable, so no
# argument-principle count exists for it. This is the same conclusion this
# appendix reaches on physical grounds, and it is the paper's too.
const BEAM_ETA = 0.01          # eta_tilde = eta/T, the value of their Fig. 8
function D_beam(λ::T, p) where T
    r, K = p                   # r = tau/T (delay ratio), K = feedback gain
    γ = λ / sqrt(one(T) + T(BEAM_ETA) * λ)
    # Return-difference form of (54) after dropping the trivial lam factor:
    # same zeros, but D -> 1 at infinity, hence n = 0 exactly.
    return one(T) - K * exp(-r * λ) / cosh(γ)
end

# FINITE ELEMENT COUNTERPART of the very same bar (A.9), over the SAME axes.
#
# Zhang & Stepan's warning that the exact chart "draws attention to the
# numerical difficulties of finite degree of freedom approximations" is made
# for the UNDAMPED bar (their Figs. 5-6), whose stable set has measure zero in
# the delay ratio: no finite-DoF model can reproduce a zero-measure set. It does
# NOT transfer to Fig. 8, the damped case, where the stable regions have finite
# width -- and the damping suppresses precisely the high modes a coarse mesh
# gets wrong. Measured here rather than asserted (see fem_vs_exact below): the
# two stable maps agree almost everywhere.
#
# Linear bar elements, EA = rho A = l = 1 (hence c = 1 and T = l/c = 1, matching
# the non-dimensionalization of D_beam). Node 1 sits at x = 0 (the driven, free
# end), node n_el+1 at x = l (clamped: u = 0, removed from the DOFs).
# Stiffness-proportional damping C = eta*K is the discrete form of the paper's
# internal damping u_txx term, with the same eta.
function build_fem(n_el)
    h = 1.0 / n_el
    n_nodes = n_el + 1
    M = zeros(n_nodes, n_nodes); K = zeros(n_nodes, n_nodes)
    for i in 1:n_el
        ke = (1 / h) * [1 -1; -1 1]; me = (h / 6) * [2 1; 1 2]
        K[i:i+1, i:i+1] += ke; M[i:i+1, i:i+1] += me
    end
    # drop the clamped last node -> DOFs 1..n_el, DOF 1 = the free end x = 0
    return M[1:end-1, 1:end-1], BEAM_ETA .* K[1:end-1, 1:end-1], K[1:end-1, 1:end-1], h
end
# 12 elements: enough to resolve the low modes that the damping leaves alive,
# and ~6x cheaper per evaluation than 29 (each is a dense solve per frequency).
const M_f, C_f, K_f, H_FEM = build_fem(12)   # 12 elements -> 12 DOF, length 1
const N_FEM = size(K_f, 1)
# Written as a RETURN DIFFERENCE det(I + Q0^-1 E) rather than the raw
# determinant det(Q0 + E). Same zeros; the poles it introduces are the
# open-loop roots, all in the left half-plane, so they do not affect the
# right-half-plane count. The pay-off is that D -> 1 at infinity (n = 0)
# instead of D ~ λ^58, which (i) removes the 1e298-scale magnitudes that
# overflow, and (ii) removes the need to estimate a large leading order at all:
# the raw form leaves an integer residual of ~0.4 (a coin flip), this one ~1e-7.
# The feedback is a single collocated sensor-actuator pair, so the feedback
# matrix F = c * e_n * e_1' is RANK ONE, and the matrix determinant lemma
#     det(I + Q0^-1 F) = det(I + c (Q0^-1 e_n) e_1') = 1 + c * (Q0 \ e_n)[1]
# replaces a full 29x29 solve plus a 29x29 determinant by ONE solve with a
# single right-hand side. Same number, a few times cheaper -- this is the
# large-system optimization Section 8 points to, exercised here.
const E_1_FEM = begin
    v = zeros(N_FEM); v[1] = 1.0; v         # the free, driven end x = 0
end
function D_fem(λ::T, p) where T
    r, K = p                                # same axes as D_beam: (tau/T, K)
    Q0 = λ^2 .* T.(M_f) .+ λ .* T.(C_f) .+ T.(K_f)
    # Boundary condition u'(0,t) = K u'(l, t-tau). The sensed quantity is the
    # STRAIN at the clamped end: u'(l) ~ (u_{N+1} - u_N)/h = -u_N/h, since the
    # last node is fixed -- and it is that MINUS sign that the load at x = 0
    # carries. (Getting it wrong mirrors the whole chart in K: the up-peak at
    # tau/T ~ 1 becomes a down-spike. Verified against the paper's own critical
    # curves, Eqs. 57-58, where |D| must vanish: with this sign the FEM gives
    # 1e-4..1e-2 there, i.e. discretization error only, and its stable map
    # differs from the continuum at 0.4% of points instead of 18%.)
    # The weak form turns the prescribed strain at x = 0 into a nodal load
    # carrying the damped modulus EA(1 + eta*lam) of the natural boundary term.
    # The feedback is RANK ONE, so the determinant lemma leaves a single solve.
    c = -K * (one(T) + T(BEAM_ETA) * λ) / T(H_FEM) * exp(-r * λ)
    return one(T) + c * (Q0 \ T.(E_1_FEM))[N_FEM]
end

# ===========================================================================
# CONNECTED CRUISE CONTROL, N VEHICLES ON A RING -- Ge, Orosz, Hajdu, Insperger
# & Moehlis, "To Delay or Not to Delay -- Stability of Connected Cruise
# Control", Advances in Delays and Dynamics, Springer 2017, pp. 263-282.
# This panel reproduces the linear (plant) stability of their Fig. 4(d).
#
# N vehicles on a ring road, each following the one ahead with gains alpha
# (headway) and beta (velocity) and a driver reaction delay sigma. The circulant
# structure block-diagonalizes into N independent wave numbers k, giving their
# Eq. 27 -- here multiplied through by e^{-s sigma} to put it in RETARDED form:
#
#   D_k(s) = s^2 + e^{-s sigma}[ (alpha+beta)s + alpha f* - (beta s + alpha f*) z_k ]
#   z_k = exp(2 pi i k / N),   f* = V'(h*) = pi/2 [1/s] at v* = 15 m/s, h* = 20 m
#
# and the ring's characteristic function is the product over k. Three points
# decide the implementation, and none of them is optional:
#
#  * k = 0 is DROPPED. It has a root at s = 0 for every (alpha, beta) -- the
#    ring's translational invariance, which the paper states explicitly. Keeping
#    it would put a root exactly on the integration line and no count would
#    exist anywhere on the chart.
#  * The product must be NORMALIZED. Raw, it grows like s^(2(N-1)) = s^198;
#    at omega = 1e4 that is 1e792, and Float64 dies at 1e308. Dividing each
#    factor by (s+a)^2 with a > 0 places the extra poles in the LEFT half-plane,
#    where they cannot change the right-half-plane count, and makes D -> 1, i.e.
#    n = 0 exactly -- no leading-order estimate needed at all.
#  * The product over k = 1..N-1 is used, not individual factors: each factor
#    alone has COMPLEX coefficients (z_k is complex), so its roots are not
#    conjugate-symmetric and the half-line formula would not apply. The full
#    product is conjugate-symmetric (verified to 5e-15) because {z_k} is closed
#    under conjugation.
#
# With A = G/(s+a)^2 and B = (beta s + alpha f*)e^{-s sigma}/(s+a)^2, the
# product over k=1..N-1 telescopes to (A^N - B^N)/(A - B) = sum_j A^{N-1-j}B^j,
# evaluated below by a division-free recurrence that never forms either power.
const CCC_N = 100              # vehicles on the ring, as in their Fig. 4
const CCC_F = pi / 2           # f* [1/s], their maximum range-policy slope
const CCC_SIG = 0.2            # driver reaction delay [s], their panel (d)
const CCC_A = 1.0              # normalization pole: LHP, cancels from the count
function D_ccc(s::T, p) where T
    β, α = p
    e = exp(-T(CCC_SIG) * s)
    d = (s + T(CCC_A))^2
    A = (s^2 + ((α + β) * s + α * T(CCC_F)) * e) / d
    B = ((β * s + α * T(CCC_F)) * e) / d
    S = one(T); pA = one(T)
    for _ in 2:CCC_N
        pA *= A
        S = pA + B * S
    end
    return S
end

# 50x50 dense stress test. The previous parameters put the whole window deep in
# the unstable region (a featureless block); adding light damping and sweeping
# a delayed STIFFNESS gain against the delay exposes the lobe structure.
Random.seed!(42)
const N_big = 50
const _B1 = rand(N_big, N_big); const M_big = _B1' * _B1 + I
const _B2 = rand(N_big, N_big); const K_big = _B2' * _B2 + I
const C_big = 0.25 .* K_big
function D_bigmat(λ::T, p) where T
    gain, τ = p
    Q = λ^2 .* T.(M_big) .+ λ .* T.(C_big) .+ T.(K_big) .+
        (gain * exp(-λ * τ)) .* T.(K_big)
    return det(Q)
end

D_frac(λ, p) = λ^1.8 + 0.5 * λ^0.8 + p[1] * exp(-p[2] * λ)

function D_turning(λ::T, p) where T
    Ω, w = p
    τ = 2π / Ω
    G = T(1.0) / (λ^2 + 2 * T(0.02) * λ + one(T)) +
        T(0.45) / (λ^2 + 2 * T(0.03) * T(2.4) * λ + T(2.4)^2)
    return one(T) + w * (1 - exp(-τ * λ)) * G
end

# ---------------------------------------------------------------------------
# Panel specifications
# ---------------------------------------------------------------------------
# This gallery is the paper's SPEED argument. Every panel uses the same
# generous 100x100 background grid and tol = 1e-4 (one decade looser than the
# package default): at those settings a count can be off by one on a few
# boundary-adjacent pixels (Section 6.3), which is exactly the trade a first
# exploration should make and is invisible at chart resolution.
#
# omega_max = 1e4 is the paper's standard window and is used wherever it is
# affordable. Three panels keep a smaller window, for reasons of substance
# rather than convenience:
#   * neutral, high-gain neutral, PDA (200-500): the phase ripple of a NEUTRAL
#     system does not decay at all -- the integrand oscillates up to infinite
#     frequency -- so a larger window buys no accuracy whatsoever. What bounds
#     the truncation error is the analytic asin(|a|)/pi < 1/2 estimate of
#     Section 2.2, not omega_max. Raising it would only multiply the cost.
#   * beam, FEM bar (200): the Kelvin-Voigt damping makes |sech(gamma)| decay
#     like e^{-sqrt(omega/2c)}, so the integrand is dead long before omega=200;
#     and each evaluation costs a 29x29 solve, so a 50x larger window would
#     turn a 7-minute panel into a 6-hour one for no change in the chart.
#   * 50x50 determinant (200): same argument, with a dense 50x50 determinant
#     per evaluation.
#
# MDBM: a 7x7 initial mesh refined until the traced boundary has ~4x the
# per-axis resolution of the panel's background grid, from only 49 blanket
# evaluations. The refinement depth is derived from the grid rather than fixed,
# because a fixed depth is either too coarse for a fine panel or -- on the two
# panels whose D costs milliseconds (FEM bar, 50x50 determinant) -- spends
# minutes resolving a boundary far beyond the resolution anyone will look at.
half(n) = FAST[] ? max(12, n ÷ 2) : n
const NBF = 75        # background grid, all panels
const MDBM_N0_G = 7    # MDBM initial mesh per axis
mdbm_levels(nx) = clamp(ceil(Int, log2(4 * nx / MDBM_N0_G)), 3, 5)
mdbm_equiv(nx) = MDBM_N0_G * 2^mdbm_levels(nx)
SPECS = [
    # closed stable island (measured bbox x[-0.8,1.4] y[0.16,1.9]); range set to
    # fill the frame with an 8% margin
    (id = "fourth",  D = D_fourth,     xl = "P",  yl = "D",  xr = (-1.5, 2.0), yr = (-0.5, 2.5),
     nx = half(NBF), ny = half(NBF), ω = 1e4, tol = 1e-4, title = "4th-order + delayed PD"),
    (id = "algebraic", D = D_algebraic, xl = "a", yl = "b", xr = (-1.0, 10.0), yr = (-1.0, 10.0),
     nx = half(NBF), ny = half(NBF), ω = 1e4, tol = 1e-4, title = "delayed oscillator"),
    (id = "distributed", D = D_distributed, xl = "a", yl = "b", xr = (-0.5, 2.0), yr = (-1.0, 5.0),
     nx = half(NBF), ny = half(NBF), ω = 1e4, tol = 1e-4, title = "distributed delay"),
    # NOTE: neutral integrands oscillate persistently up to infinite frequency,
    # so the truncation must stay moderate; the tail error is bounded by
    # asin(|a|)/pi < 1/2, hence the rounded count remains correct.
    # stable set is bounded by the |a| < 1 essential-instability limit on the
    # sides; y tightened to the measured band (+-0.88) so the island fills
    (id = "neutral", D = D_neutral, xl = "a", yl = "c", xr = (-1.2, 1.2), yr = (-1.2, 1.2),
     nx = half(NBF), ny = half(NBF), ω = 200.0, tol = 1e-4, npow = 2.0,
     title = "neutral DDE"),
    # same |a| < 1 side limit; y tightened to the measured band [0.34, 8.43]
    (id = "neutral_hg", D = D_neutral_hg, xl = "a", yl = "c", xr = (-1.2, 1.2), yr = (-1.0, 10.0),
     nx = half(NBF), ny = half(NBF), ω = 200.0, tol = 1e-4, npow = 2.0,
     title = "high-gain neutral DDE"),
    # closed island (measured bbox x[-0.97,1.24] y[+-0.96]); range fills it and
    # still shows the A = +-1 essential-instability lines
    (id = "pda", D = D_pda, xl = "P", yl = "A", xr = (-1.1, 1.4), yr = (-1.15, 1.15),
     nx = half(NBF), ny = half(NBF), ω = 500.0, tol = 1e-4, npow = 2.0, cap = 10.0,
     hlines = [-1.0, 1.0], title = "PDA control (neutral, essential)"),
    # w starts slightly above 0: at w = 0 the rational D is constant (no roots)
    (id = "turning", D = D_turning, xl = "Ω", yl = "w", xr = (0.10, 1.2), yr = (0.01, 1.1),
     nx = half(NBF), ny = half(NBF), ω = 1e4, tol = 1e-4, title = "multi-DOF turning lobes"),
    # Reproduction of Zhang & Stepan (2016) Fig. 8: the exact axes and parameters
    # of the paper -- delay ratio tau/T against gain K, at eta_tilde = 0.01.
    (id = "beam", D = D_beam, xl = "τ/T", yl = "K", xr = (0.02, 10.5), yr = (-0.75, 1.0),
     nx = half(NBF), ny = half(NBF), ω = 400.0, tol = 1e-4, npow = 0.0,
     title = "elastic bar, exact (Zhang & Stépán Fig. 8)"),
    # SAME window as the beam panel (same physical feedback law: tip force
    # from clamped-end strain, KV damping) so the two charts are directly
    # comparable. ω above the highest structural mode (FE bar modes reach ~60).
    # The SAME bar, discretized: same axes, same eta -> the two panels are a
    # direct test of the convergence the paper warns about.
    (id = "fem", D = D_fem, xl = "τ/T", yl = "K", xr = (0.02, 10.5), yr = (-0.75, 1.0),
     nx = half(NBF), ny = half(NBF), ω = 400.0, tol = 1e-4, npow = 0.0,
     title = "same bar, 12-DOF FEM"),
    # Reproduction of Ge, Orosz, Hajdu, Insperger & Moehlis (2017) Fig. 4(d):
    # the exact axes and parameters of the paper.
    # alpha starts just above 0: at alpha = 0 EVERY wave number has D_k(0) = 0,
    # so a root sits exactly on the integration line and no count exists. That
    # line is the paper's own Eq. 12 stability boundary; our residual diagnostic
    # flags it unprompted (it was the only row of uncertain points on the chart).
    # This panel is the one place the gallery's tol = 1e-4 is not enough, and the
    # reason is worth recording. The (s+a)^2 normalization leaves a phase tail
    # ~2aN/omega which N = 100 amplifies, so the two knobs pull against each
    # other: raising omega_max shortens the tail but lengthens the march, and at
    # a loose tolerance the march's own error wins. Measured on a 40x40 grid: at
    # tol = 1e-6 the median residual falls 0.016 -> 0.0022 going from omega_max
    # 1e4 to 1e5, but at tol = 1e-4 it RISES 0.022 -> 0.032. Hence omega_max =
    # 1e4 with a tightened tol = 1e-5. What remains is not alarming in context:
    # this chart counts up to Z ~ 240 roots, so a residual of ~0.02 is a
    # RELATIVE error of order 1e-4.
    (id = "ccc", D = D_ccc, xl = "β [1/s]", yl = "α [1/s]", xr = (-5.0, 5.0), yr = (0.05, 10.0),
     nx = half(NBF), ny = half(NBF), ω = 1e4, tol = 1e-5, npow = 0.0,
     title = "$(CCC_N)-vehicle ring, CCC (Ge & Orosz Fig. 4d)"),
    (id = "bigmat", D = D_bigmat, xl = "gain", yl = "τ", xr = (-0.95, 1.0), yr = (0.05, 1.5),
     nx = half(40), ny = half(40), ω = 200.0, tol = 1e-4, title = "50x50 matrix determinant"),
    (id = "frac", D = D_frac, xl = "k", yl = "τ", xr = (0.0, 5.0), yr = (0.1, 2.0),
     nx = half(NBF), ny = half(NBF), ω = 1e4, tol = 1e-4, title = "fractional oscillator"),
]

getcap(s) = hasproperty(s, :cap) ? s.cap : nothing
gethl(s) = hasproperty(s, :hlines) ? s.hlines : nothing
# A panel may state its leading order instead of having it estimated. This is
# the recommended practice wherever n is known by inspection, and it matters
# most for the NEUTRAL panels: their |D| = |λ²(1 + a e^{-λ}) + ...| oscillates
# by a factor (1+|a|)/(1-|a|) -- up to 39x at a = 0.95 -- forever, instead of
# settling onto a power law, so the estimator is working uphill for a number
# that is simply 2.
getnpow(s) = hasproperty(s, :npow) ? s.npow : nothing

function gallery_panel!(fig, r, c, spec)
    xv = LinRange(spec.xr..., spec.nx)
    yv = LinRange(spec.yr..., spec.ny)
    # the cache key hashes every numeric knob of the spec, so editing a
    # panel's ranges / ω_max / tolerance can never silently reuse a stale
    # grid computed for different axes (the classic stale-figure trap)
    mit = mdbm_levels(spec.nx)
    # 15 tracked roots removes the sigma-colour speckle (see systems.jl); the
    # three panels whose D is expensive per evaluation (a dense solve or a
    # 100-factor product) keep 5, where the extra refinement would cost minutes
    # for a colour improvement invisible at chart resolution.
    nr = spec.id in ("fem", "bigmat", "ccc") ? 5 : 15
    # the cache key hashes every numeric knob of the spec, so editing a
    # panel's ranges / ω_max / tolerance / nroots can never silently reuse a
    # stale grid computed for different settings (the classic stale-figure trap)
    skey = string(hash((spec.xr, spec.yr, spec.ω, spec.tol, MDBM_N0_G, mit,
                        getnpow(spec), nr)); base = 16)
    # the DOMINANT root (max Re over several tracked minima) -- a single
    # tracked minimum can belong to a non-dominant branch away from the
    # boundary, which shows up as discontinuous shading
    grid = with_cache("s08_$(spec.id)_$(skey)_$(spec.nx)x$(spec.ny)") do
        np = getnpow(spec)
        kw = np === nothing ? (;) : (; n_power_max = np)
        sweep_grid_dominant(spec.D, xv, yv; nroots = nr, ω_max = spec.ω,
            reltol = spec.tol, abstol = spec.tol, kw...)
    end
    bnd = with_cache("s08_$(spec.id)_$(skey)_mdbm") do
        mdbm_boundary(spec.D, spec.xr, spec.yr; ngrid = MDBM_N0_G,
            Niter = FAST[] ? max(3, mit - 2) : mit,
            ω_max = spec.ω, reltol = spec.tol, abstol = spec.tol,
            n_power_max = getnpow(spec))
    end
    # Signed bilinear field: the stable half is scaled by this panel's own
    # sigma_min and the unstable half by its own Z_max, so a chart with Z up to
    # 6 and sigma down to only -0.1 still spends half the colour range on each.
    Zc = grid.Z
    cap = getcap(spec)
    cap !== nothing && (Zc = clamp.(Zc, -1, round(Int, cap)))
    C, σ_min, Z_max = bilinear_metric(Zc, grid.sigma)
    ax = MAxis(fig[r, c], xlabel = spec.xl, ylabel = spec.yl, title = spec.title,
        titlesize = 8)
    heatmap!(ax, xv, yv, C; colormap = BILINEAR_CMAP, colorrange = (-1, 1),
        rasterize = 8)
    bnd === nothing || lines!(ax, bnd.edges[1], bnd.edges[2];
        color = BOUNDARY_COLOR, linewidth = BOUNDARY_LW)
    hl = gethl(spec)
    hl !== nothing && hlines!(ax, hl; color = :red, linestyle = :dash, linewidth = 1.0)
    # CPU cost of BOTH stages, plus this panel's own colour limits: each panel
    # is normalized to its own extremes, so those numbers must travel with it.
    lab = "$(spec.nx)×$(spec.ny): $(tex_time_plain(grid.t))"
    bnd !== nothing && (lab *= "\nMDBM: $(tex_time_plain(bnd.t))")
    lab *= "\nσ≤$(round(σ_min, sigdigits = 2)) Z≤$(Z_max)"
    # Put the plate where it hides the least: over a solidly unstable corner
    # (a flat colour plateau) rather than on the stable island, whose shading
    # is the only part of the chart that actually varies.
    annotate_panel!(ax, spec.xr, spec.yr, lab; corner = pick_annotation_corner(grid.Z))
    # The integer residual PROVES this panel's settings are adequate: a
    # truncated tail or an under-resolved march shows up as Z_raw sitting away
    # from an integer. It costs nothing (Z_raw is already computed) and turns
    # the choice of omega_max from a judgement call into a checkable claim.
    #
    # Report the MEDIAN and the number of uncertain points, not just the max:
    # the max over 10^4 points is set by the single worst pixel and says almost
    # nothing about the chart. (On the beam panel, for instance, the max is 0.47
    # while the median is 1e-4 and 5 points in 10^4 exceed 0.25 -- all of them
    # deep in the unstable domain, where a dozen roots have each contributed a
    # little error and the stable/unstable classification is unaffected.)
    r = filter(isfinite, abs.(vec(grid.Z_raw) .- round.(vec(grid.Z_raw))))
    n_nonfinite = count(!isfinite, grid.Z_raw)
    resid_max = isempty(r) ? NaN : maximum(r)
    resid_med = isempty(r) ? NaN : median(r)
    n_uncert = count(>(0.25), r)
    @info "panel residual" spec.id ω = spec.ω resid_med resid_max n_uncert n_nonfinite
    return (spec.id, spec.nx * spec.ny, grid.t, bnd === nothing ? NaN : bnd.t,
            spec.ω, spec.tol, MDBM_N0_G, mit, mdbm_equiv(spec.nx),
            resid_med, resid_max, n_uncert, n_nonfinite)
end

# ---------------------------------------------------------------------------
# FEM diagnostics quoted in the appendix: the raw determinant vs the
# return-difference form, and the determinant-lemma speed-up. Every number the
# appendix states about this example is measured here.
# ---------------------------------------------------------------------------
# Both forms below must implement the SAME boundary feedback as the panel's
# D_fem (minus sign from the clamped-end strain, damped modulus (1 + eta*lam));
# D_fem returns 1 + c * (Q0^-1)[N,1], i.e. F = c * e_1 * e_N'.
_fem_c(λ::T, r, K) where T =
    -K * (one(T) + T(BEAM_ETA) * λ) / T(H_FEM) * exp(-r * λ)
function D_fem_raw(λ::T, p) where T          # the naive formulation
    r, K = p
    Q = λ^2 .* T.(M_f) .+ λ .* T.(C_f) .+ T.(K_f)
    Q[1, N_FEM] += _fem_c(λ, r, K)
    return det(Q)
end
function D_fem_full(λ::T, p) where T         # return difference, full NxN solve
    r, K = p
    Q0 = λ^2 .* T.(M_f) .+ λ .* T.(C_f) .+ T.(K_f)
    F = zeros(T, size(Q0)); F[1, N_FEM] = _fem_c(λ, r, K)
    return det(one(T) * I + (Q0 \ F))
end
fem_diag = with_cache("s08_fem_diag_v2") do
    # interior point of the fem panel (tau/T = 1, K = 0.5): off every boundary,
    # so the residuals measure the formulation, not a root on the contour
    p_fd = (1.0, 0.5)
    λs = (0.3 + 2.0im, -0.1 + 15.0im, 0.05 - 40.0im)
    agree = maximum(abs(D_fem_full(λ, p_fd) - D_fem(λ, p_fd)) /
                    max(abs(D_fem_full(λ, p_fd)), 1e-300) for λ in λs)
    # Cost per evaluation of each form. BenchmarkTools with interpolated
    # arguments, NOT a hand-rolled @elapsed loop through a function-valued
    # closure: the closure adds a dynamically dispatched call to every
    # iteration, which is a fixed overhead added to both forms and therefore
    # compresses the ratio toward 1 (it reported 1.5x where the true ratio is
    # several times that).
    λb = 0.3 + 12.0im
    t_full = @belapsed D_fem_full($λb, $p_fd)
    t_lem  = @belapsed D_fem($λb, $p_fd)
    # Magnitude of the RAW determinant at the TOP of the integration range --
    # the largest value the march actually has to represent, which is what
    # decides whether it overflows. (Probing at some interior omega understates
    # it by many orders: |D| ~ omega^58 here.)
    mag_raw = abs(D_fem_raw(200.0im, p_fd))
    n_raw = get_n_power_max(D_fem_raw, p_fd)
    n_rd  = get_n_power_max(D_fem, p_fd)
    _, zr_raw = calculate_unstable_roots_direct(D_fem_raw, p_fd; ω_max = 200.0,
        n_roots_to_track = 0)
    _, zr_rd = calculate_unstable_roots_direct(D_fem, p_fd; ω_max = 200.0,
        n_roots_to_track = 0)
    (agree = agree, t_full = t_full, t_lem = t_lem, speedup = t_full / t_lem,
     mag_raw = mag_raw, n_raw = n_raw, n_rd = n_rd,
     res_raw = abs(zr_raw - round(zr_raw)), res_rd = abs(zr_rd - round(zr_rd)))
end
@info "FEM diagnostics" fem_diag
write_csv("fem_diagnostics", ["key", "value"],
    [(string(k), getfield(fem_diag, k)) for k in propertynames(fem_diag)])
write_macros("fem_numbers", [
    "FemLemmaSpeedup" => @sprintf("%.1f", fem_diag.speedup),
    "FemLemmaAgree"   => tex_sci_bare(fem_diag.agree),
    "FemRawMag"       => @sprintf("10^{%d}", round(Int, log10(max(fem_diag.mag_raw, 1.0)))),
    "FemRawOrder"     => @sprintf("%.1f", fem_diag.n_raw),
    "FemRawResid"     => @sprintf("%.2f", fem_diag.res_raw),
    "FemRdResid"      => tex_sci_bare(fem_diag.res_rd),
])

# ---------------------------------------------------------------------------
# DOES the finite-DoF bar reproduce the continuum chart? Measure it; do not
# assert it either way.
#
# Zhang & Stepan's warning about finite-DoF approximations is made for the
# UNDAMPED bar (their Figs. 5-6), where the stable set has measure zero in the
# delay ratio -- no finite-DoF model can reproduce a zero-measure set. Fig. 8 is
# the DAMPED case, where the stable regions have finite width, and there is no
# reason a modest FE model should fail on those: the damping kills exactly the
# high modes the discretization gets wrong. So the honest question is not
# "does it fail" but "by how much, and where".
# ---------------------------------------------------------------------------
fem_vs_exact = with_cache("s08_femcmp_v1") do
    xv = LinRange(0.02, 10.5, 100); yv = LinRange(-0.75, 1.0, 100)
    params = vec([(x, y) for x in xv, y in yv])
    Ze, = calculate_unstable_roots_p_vec(D_beam, params; ω_max = 400.0,
        reltol = 1e-4, abstol = 1e-4, n_roots_to_track = 0, n_power_max = 0.0)
    Zf, = calculate_unstable_roots_p_vec(D_fem, params; ω_max = 400.0,
        reltol = 1e-4, abstol = 1e-4, n_roots_to_track = 0, n_power_max = 0.0)
    # what a reader compares is the stable/unstable MAP, not the exact count
    stable_e = Ze .== 0; stable_f = Zf .== 0
    disagree = stable_e .!= stable_f
    # where do they disagree? report the mean delay ratio of those points
    r_dis = isempty(findall(disagree)) ? NaN : mean(first.(params[findall(disagree)]))
    (n = length(params), n_dis = count(disagree),
     frac = count(disagree) / length(params),
     n_stable_e = count(stable_e), n_stable_f = count(stable_f), r_dis = r_dis)
end
@info "12-DoF FEM vs exact continuum (Zhang & Stepan Fig. 8 plane)" fem_vs_exact
write_csv("fem_vs_exact", ["key", "value"],
    [(string(k), getfield(fem_vs_exact, k)) for k in propertynames(fem_vs_exact)])
write_macros("femcmp_numbers", [
    "FemCmpN"      => string(fem_vs_exact.n),
    "FemCmpDis"    => string(fem_vs_exact.n_dis),
    "FemCmpPct"    => @sprintf("%.1f", 100 * fem_vs_exact.frac),
    "FemCmpRdis"   => isnan(fem_vs_exact.r_dis) ? "--" : @sprintf("%.1f", fem_vs_exact.r_dis),
])

timings = Tuple[]
const N_A = 6           # first figure holds the first six panels
figa = Figure(size = (W_FULL, W_FULL * 0.60))
for (k, spec) in enumerate(SPECS[1:N_A])
    r, c = fldmod1(k, 3)
    push!(timings, gallery_panel!(figa, r, c, spec))
end
save_fig(figa, "fig_gallery_a")

figb = Figure(size = (W_FULL, W_FULL * 0.60))
for (k, spec) in enumerate(SPECS[N_A+1:end])
#for (k, spec) in enumerate(SPECS[[7,8,10]])
    r, c = fldmod1(k, 3)
    push!(timings, gallery_panel!(figb, r, c, spec))
end
save_fig(figb, "fig_gallery_b")

write_csv("gallery_timings",
    ["system", "n_points", "grid_time_s", "mdbm_time_s", "wmax", "tol",
     "mdbm_n0", "mdbm_levels", "mdbm_equiv_res",
     "median_int_residual", "max_int_residual", "n_uncertain", "n_nonfinite"],
    timings)
# A panel is suspect when a non-trivial FRACTION of its points are uncertain --
# that is what a truncated tail or an under-resolved march looks like. A single
# bad pixel is not: it is the tail of a distribution whose median is ~1e-4.
let bad = [(t[1], t[12], t[2]) for t in timings if t[12] > 0.01 * t[2]]
    isempty(bad) || @warn "gallery: panels where >1% of points are uncertain" bad
end
@info "gallery residual summary" worst_median = maximum(t[10] for t in timings) total_uncertain =
    sum(t[12] for t in timings) total_points = sum(t[2] for t in timings)
