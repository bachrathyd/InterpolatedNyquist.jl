# Hill determinant + argument principle for time-periodic delayed systems (CPU core).
#
# Delayed Mathieu equation   x'' + κ x' + (δ + ε cos(ω_p t)) x = b x(t - τ),  T = 2π/ω_p
#
# Floquet ansatz x = e^{λt} Σ_k c_k e^{i k ω_p t}:  harmonic k obeys
#     q(λ + i k ω_p) c_k + (ε/2)(c_{k-1} + c_{k+1}) = 0,    q(s) = s² + κ s + δ - b e^{-sτ}
# Hill matrix: tridiagonal, diagonal d_k = q(λ + i k ω_p), off-diagonals ε/2.
# Regularized (Hill's normalization, rows divided by the diagonal):
#     Δ_N(λ) = det(I + D⁻¹E)    (continuant recursion, O(N), dual-number friendly)
# Δ_N → 1 as Re λ → ∞, and the infinite Δ is periodic in the imaginary direction
# (period i ω_p). Argument principle on the half-strip {Re λ > 0, a < Im λ/ω_p < a+1}:
# the top/bottom edges cancel by periodicity, the right edge contributes nothing, so
#     Z_strip = P_N - (1/2π) Δarg_{ω ∈ [a, a+1] ω_p} Δ_N(iω)
# where Z_strip = number of unstable Floquet exponents (= multipliers with |μ| > 1) and
# P_N = poles of Δ_N in the strip = zeros of the d_k there = the RHP roots of the LTI
# quasi-polynomial q (each RHP root s of q lies in the strip shifted by exactly one k),
# counted with the LTI argument principle (n = 2).
# a is generic (0.237): exponents of real multipliers have Im λ/ω_p ∈ {0, 1/2} (mod 1)
# -- Hopf-type crossings at μ = +1 and flip (period doubling) at μ = -1 -- and must not
# sit on the strip edges.

using InterpolatedNyquist, ForwardDiff
const IN = InterpolatedNyquist

Base.@kwdef struct Mathieu
    κ::Float64 = 0.1
    ε::Float64 = 1.0
    τ::Float64 = 2π
    ωp::Float64 = 1.0          # principal frequency 2π/T
end

@inline q_lti(s, δ, b, m::Mathieu) = s^2 + m.κ * s + δ - b * exp(-m.τ * s)

# regularized Hill determinant on harmonics k = -N..N (continuant of I + D⁻¹E)
function hill_delta(λ, δ, b, N::Int, m::Mathieu)
    e2 = (m.ε / 2)^2
    dprev = q_lti(λ - 1im * N * m.ωp, δ, b, m)
    fprev = one(λ)           # f_{-1}
    f = one(λ)               # f_0  (1x1 block: diagonal 1)
    for k in (-N + 1):N
        d = q_lti(λ + 1im * k * m.ωp, δ, b, m)
        f, fprev = f - e2 / (d * dprev) * fprev, f
        dprev = d
    end
    return f
end

# Pole-free variant. Hill's row scaling by d_k puts a POLE of Δ at every zero of
# every d_k, i.e. at every LTI root of q (shifted by -ik). A root of q just left of
# the axis next to an unstable Floquet exponent just right of it (e.g. a flip
# exponent at Im λ = ω_p/2) is a zero-pole pair straddling the contour: the phase
# drops by 2π within one step, invisible to the end-point check of the march.
# Scaling the rows by r_k = (λ + ikω_p + c)² instead keeps Δ̃ → 1 (Re λ → ∞) and the
# periodicity, and moves all poles to λ = -c - ikω_p, far left of the contour:
#     Δ̃ = det H_N / Π r_k,  zeros = Floquet exponents only,  Z = -(1/2π) Δarg Δ̃.
# The diagonal factor Π d_k/r_k converges like O(1/k²) per ring; its rings beyond the
# coupling window are cheap scalars, included up to K = ktail·N.
function hill_delta_pf(λ, δ, b, N::Int, m::Mathieu; c = m.ωp, ktail = 8)
    e2 = (m.ε / 2)^2
    rk(k) = (λ + 1im * k * m.ωp + c)^2
    rprev = rk(-N)
    fprev = one(λ)
    dr = q_lti(λ - 1im * N * m.ωp, δ, b, m) / rprev
    f = dr
    P = dr                                               # Π_{|k|≤N} d_k / r_k
    for k in (-N + 1):N
        r = rk(k)
        dr = q_lti(λ + 1im * k * m.ωp, δ, b, m) / r
        f, fprev = dr * f - e2 / (r * rprev) * fprev, f
        P *= dr
        rprev = r
    end
    if is_commensurate(m)
        # τ ω_p = 2π j: e^{-(λ+ikω_p)τ} = e^{-λτ} for every k, so d_k = (s_k - z1)(s_k - z2)
        # with s_k = λ + ikω_p and the infinite diagonal product is closed-form:
        #   Π_k d_k/r_k = sinh(π(λ-z1)/ω_p) sinh(π(λ-z2)/ω_p) / sinh²(π(λ+c)/ω_p)
        B = δ - b * exp(-m.τ * λ)
        sq = sqrt(complex(m.κ^2 - 4B))
        z1, z2 = (-m.κ + sq) / 2, (-m.κ - sq) / 2
        w = π / m.ωp
        G = sinh(w * (λ - z1)) * sinh(w * (λ - z2)) / sinh(w * (λ + c))^2
        return f * (G / P)                               # × Π_{|k|>N} d_k/r_k (exact)
    end
    for k in (N + 1):(ktail * N)                          # otherwise: diagonal tail to ktail·N
        f *= q_lti(λ + 1im * k * m.ωp, δ, b, m) / rk(k) * q_lti(λ - 1im * k * m.ωp, δ, b, m) / rk(-k)
    end
    return f
end
is_commensurate(m::Mathieu) = abs(m.τ * m.ωp / 2π - round(m.τ * m.ωp / 2π)) < 1e-12

# a priori truncation from the tolerance: adding ring k changes Δ by about
# (ε/2)²/|d_k d_{k-1}| ≈ (ε/2)²/(k² ω_p² - δ)²  ->  invert for k
n_apriori(δ, tol, m::Mathieu) = ceil(Int, sqrt(max(δ, 0.0) + m.ε / (2 * sqrt(tol))) / m.ωp) + 1

# a posteriori: relative change of Δ when one more ring is added, at probe frequencies
function ring_change(δ, b, N, m::Mathieu; a = 0.237, probes = (0.0, 0.25, 0.5, 0.75), det = hill_delta_pf)
    e = 0.0
    for t in probes
        λ = 1im * (a + t) * m.ωp
        d0 = det(λ, δ, b, N, m)
        d1 = det(λ, δ, b, N + 1, m)
        e = max(e, abs(d1 / d0 - 1))
    end
    return e
end

# adaptive N per point: start at the a priori value, grow until the ring change < tol
function choose_N(δ, b, tol, m::Mathieu; Nmax = 60)
    N = n_apriori(δ, tol, m)
    e = ring_change(δ, b, N, m)
    while e > tol && N < Nmax
        N += 1
        e = ring_change(δ, b, N, m)
    end
    return N, e
end

# phase march of Δ_N along ω ∈ [a, a+1] ω_p (discrete unwrapping of the package)
function strip_winding(δ, b, N, m::Mathieu; a = 0.237, tol = 0.3)
    f = IN.NyquistWrapper{NTuple{3, Float64}}((λ, p) -> hill_delta(λ, p[1], p[2], Int(p[3]), m))
    Φ, ok, ev = IN._unwrap_march(f, (δ, b, Float64(N)), 0.0, 0; ω0 = a * m.ωp,
        ω_max = (a + 1) * m.ωp, tol = tol, h0 = 1e-3 * m.ωp, hrel = 0.05)
    return -Φ / (2π), ok, ev
end

function strip_winding_pf(δ, b, N, m::Mathieu; a = 0.237, tol = 0.3)
    f = IN.NyquistWrapper{NTuple{3, Float64}}((λ, p) -> hill_delta_pf(λ, p[1], p[2], Int(p[3]), m))
    Φ, ok, ev, dd, ds, dw = IN._unwrap_march(f, (δ, b, Float64(N)), 0.0, 1; ω0 = a * m.ωp,
        ω_max = (a + 1) * m.ωp, tol = tol, h0 = 1e-3 * m.ωp, hrel = 0.05)
    return -Φ / (2π), ok, ev, ds[1], dw[1]
end

"Pole-free count: unstable Floquet exponents, no LTI pole count needed."
function floquet_count_pf(δ, b, m::Mathieu; tol = 1e-4, N = nothing)
    Ne, err = N === nothing ? choose_N(δ, b, tol, m) : (N, ring_change(δ, b, N, m))
    W, ok, ev, σd, ωd = strip_winding_pf(δ, b, Ne, m)
    return (Z = ok ? round(Int, W) : -1, Zraw = W, N = Ne, err = err, evals = ev, σ = σd, ω = ωd)
end

# P: RHP roots of the LTI quasi-polynomial q (n = 2)
lti_count(δ, b, m::Mathieu) =
    calculate_unstable_roots_unwrap((λ, p) -> q_lti(λ, p[1], p[2], m), (δ, b), 0.0;
        n_roots_to_track = 0, n_power_max = 2)

"Unstable Floquet exponents of the delayed Mathieu equation at (δ, b)."
function floquet_count(δ, b, m::Mathieu; tol = 1e-4, N = nothing)
    Ne, err = N === nothing ? choose_N(δ, b, tol, m) : (N, ring_change(δ, b, N, m))
    W, ok, ev = strip_winding(δ, b, Ne, m)
    P, Praw = lti_count(δ, b, m)
    Zraw = P + W
    return (Z = ok ? round(Int, Zraw) : -1, Zraw = Zraw, N = Ne, err = err, P = P, evals = ev)
end

# ---------------------------------------------------------------------------
# Reference: Floquet multipliers of the RK4-discretized monodromy map (τ = T only).
# History = (x, v) at t_j = j h, j = 0..m over the last period; the delayed term
# b x(t - T) at the RK4 stages comes from the previous period (midpoints by cubic
# Hermite with x and v). 4th-order accurate; dimension 2(m+1).
# ---------------------------------------------------------------------------
function monodromy(δ, b, m::Mathieu; msteps = 80)
    @assert isapprox(m.τ, 2π / m.ωp) "reference implemented for τ = T"
    T = 2π / m.ωp
    h = T / msteps
    n = msteps + 1
    f(t, x, v, xd) = (v, -m.κ * v - (δ + m.ε * cos(m.ωp * t)) * x + b * xd)
    M = zeros(2n, 2n)
    Y = zeros(2n)
    for col in 1:2n
        fill!(Y, 0.0); Y[col] = 1.0
        xo, vo = view(Y, 1:n), view(Y, n+1:2n)
        xn = zeros(n); vn = zeros(n)
        xn[1], vn[1] = xo[n], vo[n]
        for j in 1:msteps
            t = (j - 1) * h
            x, v = xn[j], vn[j]
            xa, xb = xo[j], xo[j+1]
            xm = (xa + xb) / 2 + h * (vo[j] - vo[j+1]) / 8
            k1 = f(t, x, v, xa)
            k2 = f(t + h/2, x + h/2 * k1[1], v + h/2 * k1[2], xm)
            k3 = f(t + h/2, x + h/2 * k2[1], v + h/2 * k2[2], xm)
            k4 = f(t + h, x + h * k3[1], v + h * k3[2], xb)
            xn[j+1] = x + h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
            vn[j+1] = v + h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
        end
        M[1:n, col] = xn; M[n+1:2n, col] = vn
    end
    return M
end

using LinearAlgebra
function reference_count(δ, b, m::Mathieu; msteps = 80)
    μ = eigvals(monodromy(δ, b, m; msteps))
    return count(>(1), abs.(μ)), maximum(abs.(μ))
end
