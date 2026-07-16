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
const BEAM_C = 0.05      # Kelvin-Voigt damping coefficient
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
    # clamp node 1 -> n_el free DOFs
    return M[2:end, 2:end], 0.05 .* K[2:end, 2:end], K[2:end, 2:end], h
end
const M_f, C_f, K_f, H_FEM = build_fem(29)   # 29 elements -> 29 DOF, length 1
# Written as a RETURN DIFFERENCE det(I + Q0^-1 E) rather than the raw
# determinant det(Q0 + E). Same zeros; the poles it introduces are the
# open-loop roots, all in the left half-plane, so they do not affect the
# right-half-plane count. The pay-off is that D -> 1 at infinity (n = 0)
# instead of D ~ λ^58, which (i) removes the 1e298-scale magnitudes that
# overflow, and (ii) removes the need to estimate a large leading order at all:
# the raw form leaves an integer residual of ~0.4 (a coin flip), this one ~1e-7.
function D_fem(λ::T, p) where T
    Kp, τ = p
    Q0 = λ^2 .* T.(M_f) .+ λ .* T.(C_f) .+ T.(K_f)
    F = zeros(T, size(Q0))
    F[end, 1] = Kp / T(H_FEM) * exp(-λ * τ)   # strain reading at the clamped end
    return det(one(T) * I + (Q0 \ F))
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
half(n) = FAST[] ? max(12, n ÷ 2) : n
SPECS = [
    (id = "fourth",  D = D_fourth,     xl = "P",  yl = "D",  xr = (-2.0, 4.0), yr = (-2.0, 5.0),
     nx = half(80), ny = half(60), ω = 1e6, tol = 1e-5, mdbm = 25, title = "4th-order + delayed PD"),
    (id = "algebraic", D = D_algebraic, xl = "a", yl = "b", xr = (-1.0, 1.0), yr = (-1.0, 1.0),
     nx = half(60), ny = half(50), ω = 1e6, tol = 1e-5, mdbm = 25, title = "delayed oscillator"),
    (id = "distributed", D = D_distributed, xl = "a", yl = "b", xr = (0.0, 2.0), yr = (-1.0, 5.0),
     nx = half(60), ny = half(50), ω = 1e6, tol = 1e-5, mdbm = 25, title = "distributed delay"),
    # NOTE: neutral integrands oscillate persistently up to infinite frequency,
    # so the truncation must stay moderate; the tail error is bounded by
    # asin(|a|)/pi < 1/2, hence the rounded count remains correct.
    (id = "neutral", D = D_neutral, xl = "a", yl = "c", xr = (-0.9, 0.9), yr = (-2.0, 2.0),
     nx = half(60), ny = half(50), ω = 200.0, tol = 1e-4, mdbm = 20, title = "neutral DDE"),
    (id = "neutral_hg", D = D_neutral_hg, xl = "a", yl = "c", xr = (-0.95, 0.95), yr = (-10.0, 10.0),
     nx = half(60), ny = half(50), ω = 200.0, tol = 1e-4, mdbm = 20, title = "high-gain neutral DDE"),
    (id = "pda", D = D_pda, xl = "P", yl = "A", xr = (-1.5, 2.5), yr = (-1.6, 1.6),
     nx = half(70), ny = half(70), ω = 500.0, tol = 1e-4, mdbm = 25, cap = 10.0,
     hlines = [-1.0, 1.0], title = "PDA control (neutral, essential)"),
    # w starts slightly above 0: at w = 0 the rational D is constant (no roots)
    (id = "turning", D = D_turning, xl = "Ω", yl = "w", xr = (0.08, 1.2), yr = (0.01, 1.2),
     nx = half(100), ny = half(70), ω = 1e4, tol = 1e-5, mdbm = 35, title = "multi-DOF turning lobes"),
    (id = "beam", D = D_beam, xl = "Kp", yl = "τ", xr = (0.0, 5.0), yr = (0.2, 3.0),
     nx = half(60), ny = half(50), ω = 1e4, tol = 1e-5, mdbm = 25, title = "transcendental beam"),
    # SAME window as the beam panel (same physical feedback law: tip force
    # from clamped-end strain, KV damping) so the two charts are directly
    # comparable. ω above the highest structural mode (FE bar modes reach ~60).
    (id = "fem", D = D_fem, xl = "Kp", yl = "τ", xr = (0.0, 5.0), yr = (0.2, 3.0),
     nx = half(60), ny = half(50), ω = 500.0, tol = 1e-5, mdbm = 20, title = "29-DOF FEM bar"),
    # gain > -1: at gain = -1 exactly, the delayed stiffness cancels the static
    # one and a characteristic root sits ON the integration line (Z undefined)
    (id = "bigmat", D = D_bigmat, xl = "gain", yl = "τ", xr = (-0.95, 1.0), yr = (0.05, 1.5),
     nx = half(28), ny = half(22), ω = 200.0, tol = 1e-5, mdbm = 10, title = "50x50 matrix determinant"),
    (id = "frac", D = D_frac, xl = "k", yl = "τ", xr = (0.0, 5.0), yr = (0.1, 2.0),
     nx = half(50), ny = half(50), ω = 100.0, tol = 1e-5, mdbm = 20, title = "fractional oscillator"),
]

getcap(s) = hasproperty(s, :cap) ? s.cap : nothing
gethl(s) = hasproperty(s, :hlines) ? s.hlines : nothing

function gallery_panel!(fig, r, c, spec)
    xv = LinRange(spec.xr..., spec.nx)
    yv = LinRange(spec.yr..., spec.ny)
    # the cache key hashes every numeric knob of the spec, so editing a
    # panel's ranges / ω_max / tolerance can never silently reuse a stale
    # grid computed for different axes (the classic stale-figure trap)
    skey = string(hash((spec.xr, spec.yr, spec.ω, spec.tol, spec.mdbm)); base = 16)
    grid = with_cache("s08_$(spec.id)_$(skey)_$(spec.nx)x$(spec.ny)") do
        sweep_grid(spec.D, xv, yv; ω_max = spec.ω, reltol = spec.tol, abstol = spec.tol)
    end
    bnd = spec.mdbm > 0 ? with_cache("s08_$(spec.id)_$(skey)_mdbm") do
            mdbm_boundary(spec.D, spec.xr, spec.yr; ngrid = spec.mdbm,
                Niter = FAST[] ? 3 : 4, ω_max = spec.ω, reltol = spec.tol, abstol = spec.tol)
        end : nothing
    C = combined_metric(grid.Z, grid.sigma)
    cap = getcap(spec)
    cap !== nothing && (C = clamp.(C, -2.0, cap))
    ax = MAxis(fig[r, c], xlabel = spec.xl, ylabel = spec.yl, title = spec.title,
        titlesize = 8)
    stability_panel!(ax, xv, yv, C; edges = bnd === nothing ? nothing : bnd.edges)
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
