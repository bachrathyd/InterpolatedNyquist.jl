# Study s08: appendix case-study gallery. One standardized stability-chart
# panel per system class (retarded, distributed, neutral, essential/PDA,
# transcendental, FEM, large matrix, fractional, multi-DOF turning).
# Produces: figures/fig_gallery_a.pdf, figures/fig_gallery_b.pdf,
#           data/gallery_timings.csv

include(joinpath(@__DIR__, "systems.jl"))
using Random

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

# Transcendental beam with KELVIN-VOIGT (material) damping and delayed boundary
# feedback, in return-difference form D = 1 + Kp e^{-λτ} sech(γ L), γ = λ/√(1+cλ).
#
# The choice of damping model is what makes this a valid counting problem.
# External/viscous damping (γ = √(λ²+cλ)) damps every mode equally, so the
# roots accumulate on a vertical line and Z is 0 or ∞ -- the essential-spectrum
# obstruction of a neutral system. Kelvin-Voigt damping is both physically
# correct (real material damping is rate-dependent) and mathematically
# benign: γ ~ √(λ/c) for large λ, so the high modes are damped ever harder,
# Re λ_k → −∞, only finitely many roots sit near the axis, and |sech(γ)| decays
# like e^{−√ω} on the imaginary axis. The count is then well defined.
# The return-difference form additionally gives D → 1 (n = 0), no discretization.
# Parameter window (Kp, tau) = (0..2, 0.1..3) follows the delayed-boundary-
# control analysis of the elastic bar in the literature (Zhang & Stepan 2016);
# BEAM_C is a SMALL Kelvin-Voigt damping added on top: the undamped bar has
# infinitely many undamped modes on the imaginary axis, so no finite count
# exists, while any physical material damping restores one (see the appendix).
const BEAM_C = 0.02      # Kelvin-Voigt (rate-dependent) damping coefficient
function D_beam(λ::T, p) where T
    Kp, τ = p
    γ = λ / sqrt(one(T) + T(BEAM_C) * λ)
    return one(T) + Kp * exp(-λ * τ) / cosh(γ)
end

# n_el elements of length h = 1/n_el on n_el+1 nodes -> the bar has UNIT length,
# so the FE model discretizes exactly the continuum bar of the beam panel and
# the two charts are directly comparable. (An earlier version assembled N-1
# elements of length 1/N: a bar of length (N-1)/N, whose ~3% frequency shift
# visibly displaces the stability bands.)
function build_fem(n_el)
    h = 1.0 / n_el
    n_nodes = n_el + 1
    M = zeros(n_nodes, n_nodes); K = zeros(n_nodes, n_nodes)
    for i in 1:n_el
        ke = (1 / h) * [1 -1; -1 1]; me = (h / 6) * [2 1; 1 2]
        K[i:i+1, i:i+1] += ke; M[i:i+1, i:i+1] += me
    end
    # clamp node 1 -> n_el free DOFs. C = BEAM_C * K is the discrete form of the
    # same Kelvin-Voigt law the continuum panel uses, so the two charts are
    # directly comparable.
    return M[2:end, 2:end], BEAM_C .* K[2:end, 2:end], K[2:end, 2:end], h
end
const M_f, C_f, K_f, H_FEM = build_fem(29)   # 29 elements -> 29 DOF, length 1
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
const E_N_FEM = begin
    v = zeros(size(K_f, 1)); v[end] = 1.0; v
end
function D_fem(λ::T, p) where T
    Kp, τ = p
    Q0 = λ^2 .* T.(M_f) .+ λ .* T.(C_f) .+ T.(K_f)
    c = Kp / T(H_FEM) * exp(-λ * τ)          # strain reading at the clamped end
    return one(T) + c * (Q0 \ T.(E_N_FEM))[1]
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
# This gallery is the paper's SPEED argument (Section 7 makes the accuracy
# case at ω_max = 1e6 and tight tolerances). Every panel therefore uses the
# same generous 100x100 background grid and settings tuned for a chart in a
# fraction of a second: ω_max a few decades above the highest resonance and
# tol = 1e-4, one decade looser than the default. At those settings a count
# can be off by one on a few boundary-adjacent pixels (Section 6.3) -- which
# is exactly the trade a first exploration should make, and invisible at
# chart resolution.
#
# MDBM: a 7x7 initial mesh with 6 refinement levels -> 7*2^6 = 448 equivalent
# per-axis resolution, i.e. ~4.5x the background grid, from only 49 blanket
# evaluations. The refinement concentrates on the boundary, so the trace stays
# cheaper than the sweep it overlays.
half(n) = FAST[] ? max(12, n ÷ 2) : n
const NBF = 100        # background grid, all panels
const MDBM_N0_G = 7    # MDBM initial mesh per axis
const MDBM_IT_G = 6    # MDBM refinement levels -> 448-equivalent
SPECS = [
    (id = "fourth",  D = D_fourth,     xl = "P",  yl = "D",  xr = (-2.0, 4.0), yr = (-2.0, 5.0),
     nx = half(NBF), ny = half(NBF), ω = 1e4, tol = 1e-4, title = "4th-order + delayed PD"),
    (id = "algebraic", D = D_algebraic, xl = "a", yl = "b", xr = (-1.0, 1.0), yr = (-1.0, 1.0),
     nx = half(NBF), ny = half(NBF), ω = 1e3, tol = 1e-4, title = "delayed oscillator"),
    (id = "distributed", D = D_distributed, xl = "a", yl = "b", xr = (0.0, 2.0), yr = (-1.0, 5.0),
     nx = half(NBF), ny = half(NBF), ω = 1e3, tol = 1e-4, title = "distributed delay"),
    # NOTE: neutral integrands oscillate persistently up to infinite frequency,
    # so the truncation must stay moderate; the tail error is bounded by
    # asin(|a|)/pi < 1/2, hence the rounded count remains correct.
    (id = "neutral", D = D_neutral, xl = "a", yl = "c", xr = (-0.9, 0.9), yr = (-2.0, 2.0),
     nx = half(NBF), ny = half(NBF), ω = 200.0, tol = 1e-4, title = "neutral DDE"),
    (id = "neutral_hg", D = D_neutral_hg, xl = "a", yl = "c", xr = (-0.95, 0.95), yr = (-10.0, 10.0),
     nx = half(NBF), ny = half(NBF), ω = 200.0, tol = 1e-4, title = "high-gain neutral DDE"),
    (id = "pda", D = D_pda, xl = "P", yl = "A", xr = (-1.5, 2.5), yr = (-1.6, 1.6),
     nx = half(NBF), ny = half(NBF), ω = 500.0, tol = 1e-4, cap = 10.0,
     hlines = [-1.0, 1.0], title = "PDA control (neutral, essential)"),
    # w starts slightly above 0: at w = 0 the rational D is constant (no roots)
    (id = "turning", D = D_turning, xl = "Ω", yl = "w", xr = (0.08, 1.2), yr = (0.01, 1.2),
     nx = half(NBF), ny = half(NBF), ω = 1e3, tol = 1e-4, title = "multi-DOF turning lobes"),
    (id = "beam", D = D_beam, xl = "Kp", yl = "τ", xr = (0.0, 2.0), yr = (0.1, 3.0),
     nx = half(NBF), ny = half(NBF), ω = 200.0, tol = 1e-4, title = "transcendental beam"),
    # SAME window as the beam panel (same physical feedback law: tip force
    # from clamped-end strain, KV damping) so the two charts are directly
    # comparable. ω above the highest structural mode (FE bar modes reach ~60).
    (id = "fem", D = D_fem, xl = "Kp", yl = "τ", xr = (0.0, 2.0), yr = (0.1, 3.0),
     nx = half(NBF), ny = half(NBF), ω = 200.0, tol = 1e-4, title = "29-DOF FEM bar"),
    # gain > -1: at gain = -1 exactly, the delayed stiffness cancels the static
    # one and a characteristic root sits ON the integration line (Z undefined)
    (id = "bigmat", D = D_bigmat, xl = "gain", yl = "τ", xr = (-0.95, 1.0), yr = (0.05, 1.5),
     nx = half(40), ny = half(40), ω = 200.0, tol = 1e-4, title = "50x50 matrix determinant"),
    (id = "frac", D = D_frac, xl = "k", yl = "τ", xr = (0.0, 5.0), yr = (0.1, 2.0),
     nx = half(NBF), ny = half(NBF), ω = 100.0, tol = 1e-4, title = "fractional oscillator"),
]

getcap(s) = hasproperty(s, :cap) ? s.cap : nothing
gethl(s) = hasproperty(s, :hlines) ? s.hlines : nothing

function gallery_panel!(fig, r, c, spec)
    xv = LinRange(spec.xr..., spec.nx)
    yv = LinRange(spec.yr..., spec.ny)
    # the cache key hashes every numeric knob of the spec, so editing a
    # panel's ranges / ω_max / tolerance can never silently reuse a stale
    # grid computed for different axes (the classic stale-figure trap)
    skey = string(hash((spec.xr, spec.yr, spec.ω, spec.tol, MDBM_N0_G, MDBM_IT_G)); base = 16)
    # the DOMINANT root (max Re over several tracked minima) -- a single
    # tracked minimum can belong to a non-dominant branch away from the
    # boundary, which shows up as discontinuous shading
    grid = with_cache("s08_$(spec.id)_$(skey)_$(spec.nx)x$(spec.ny)") do
        sweep_grid_dominant(spec.D, xv, yv; nroots = 5, ω_max = spec.ω,
            reltol = spec.tol, abstol = spec.tol)
    end
    bnd = with_cache("s08_$(spec.id)_$(skey)_mdbm") do
        mdbm_boundary(spec.D, spec.xr, spec.yr; ngrid = MDBM_N0_G,
            Niter = FAST[] ? MDBM_IT_G - 2 : MDBM_IT_G,
            ω_max = spec.ω, reltol = spec.tol, abstol = spec.tol)
    end
    C = combined_metric(grid.Z, grid.sigma)
    cap = getcap(spec)
    cap !== nothing && (C = clamp.(C, -2.0, cap))
    # Colour range: the spectral gap varies over orders of magnitude, so a raw
    # extrema() lets a few deeply-stable pixels flatten the whole map. Clip the
    # stable end at a robust quantile of sigma instead.
    fin = filter(isfinite, vec(C))
    lo = isempty(fin) ? -1.0 : quantile(fin, 0.02)
    hi = isempty(fin) ? 1.0 : maximum(fin)
    crange = lo < hi ? (lo, hi) : nothing
    ax = MAxis(fig[r, c], xlabel = spec.xl, ylabel = spec.yl, title = spec.title,
        titlesize = 8)
    stability_panel!(ax, xv, yv, C; edges = bnd === nothing ? nothing : bnd.edges,
        crange = crange)
    hl = gethl(spec)
    hl !== nothing && hlines!(ax, hl; color = :red, linestyle = :dash, linewidth = 1.0)
    # CPU cost of BOTH stages, printed on the chart itself
    lab = "grid $(spec.nx)×$(spec.ny): $(tex_time_plain(grid.t))"
    bnd !== nothing && (lab *= "\nMDBM: $(tex_time_plain(bnd.t))")
    text!(ax, 0.03, 0.03; text = lab, space = :relative, align = (:left, :bottom),
        fontsize = 6, color = :white,
        strokecolor = :black, strokewidth = 0.6)
    return (spec.id, spec.nx * spec.ny, grid.t, bnd === nothing ? NaN : bnd.t)
end

# ---------------------------------------------------------------------------
# FEM diagnostics quoted in the appendix: the raw determinant vs the
# return-difference form, and the determinant-lemma speed-up. Every number the
# appendix states about this example is measured here.
# ---------------------------------------------------------------------------
function D_fem_raw(λ::T, p) where T          # the naive formulation
    Kp, τ = p
    Q = λ^2 .* T.(M_f) .+ λ .* T.(C_f) .+ T.(K_f)
    Q[end, 1] += Kp / T(H_FEM) * exp(-λ * τ)
    return det(Q)
end
function D_fem_full(λ::T, p) where T         # return difference, full 29x29 solve
    Kp, τ = p
    Q0 = λ^2 .* T.(M_f) .+ λ .* T.(C_f) .+ T.(K_f)
    F = zeros(T, size(Q0)); F[end, 1] = Kp / T(H_FEM) * exp(-λ * τ)
    return det(one(T) * I + (Q0 \ F))
end
fem_diag = with_cache("s08_fem_diag_v1") do
    p_fd = (1.0, 1.0)
    λs = (0.3 + 2.0im, -0.1 + 15.0im, 0.05 - 40.0im)
    agree = maximum(abs(D_fem_full(λ, p_fd) - D_fem(λ, p_fd)) /
                    max(abs(D_fem_full(λ, p_fd)), 1e-300) for λ in λs)
    # cost per evaluation of each form (median of repeats)
    bench(f) = (f(λs[1], p_fd); median([(@elapsed for _ in 1:200; f(λs[1], p_fd); end) / 200
                                        for _ in 1:5]))
    t_full = bench(D_fem_full); t_lem = bench(D_fem)
    # magnitude and integer residual of the RAW determinant vs the return difference
    mag_raw = abs(D_fem_raw(1.0 + 50.0im, p_fd))
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
    "FemLemmaAgree"   => tex_sci(fem_diag.agree),
    "FemRawMag"       => @sprintf("10^{%d}", round(Int, log10(max(fem_diag.mag_raw, 1.0)))),
    "FemRawOrder"     => @sprintf("%.1f", fem_diag.n_raw),
    "FemRawResid"     => @sprintf("%.2f", fem_diag.res_raw),
    "FemRdResid"      => tex_sci(fem_diag.res_rd),
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
    r, c = fldmod1(k, 3)
    push!(timings, gallery_panel!(figb, r, c, spec))
end
save_fig(figb, "fig_gallery_b")

write_csv("gallery_timings", ["system", "n_points", "grid_time_s", "mdbm_time_s"], timings)
