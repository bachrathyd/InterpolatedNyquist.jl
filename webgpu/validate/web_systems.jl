# The appendix-gallery examples of the WebGPU demo (webgpu/examples.js) in NyquistGPU form
# D(λ, p, c), written operation for operation like the WGSL `charD` of each example (the
# Float32 rounding then matches closely). Only constants from `c` and integer literals, so a
# Float32 run stays in Float32.
#
# Entries: D, c (defaults), alt (a second constant set: the sliders moved), xr, yr, wmax,
# npow(c, wmax) (leading order, or the effective order of a cleared stable denominator),
# kw(c) (march options), cert (true: the rightmost root of the reference points may be
# certified by counting on shifted lines -- not for the fractional powers (branch cut) or the
# bar models (the effective order depends on the line)).
#
# Paper: INq-paper/paper/scripts/studies/s08_gallery.jl (A.2-A.9, A.11) and
# s10_fractional_controller.jl (A.12).

if !@isdefined(WEB_SYSTEMS_INCLUDED)
const WEB_SYSTEMS_INCLUDED = true

# --- A.2 delayed oscillator (generalized): λ² + aλ + k + (b + gλ) e^{-τλ},  p = (a, b) --------
function D_algebraic(λ, p, c)
    τ, k, g = c
    return λ * λ + p[1] * λ + k + (p[2] + g * λ) * exp(-τ * λ)
end

# --- A.3 distributed delay: λ² + aλ + k + b e^{-τ0 λ} (1 - e^{-τλ})/λ ------------------------
# (1 - e^{-x})/x with x = τλ: Taylor series (Horner) for |x| < 1/2 -- the removable
# singularity at λ = 0 and the cancellation of 1 - e^{-x} near it
# (coefficients built in the evaluation type: a Float64 literal would promote Float32)
function E1T(x::Complex{S}) where {S}
    o = one(S)
    if abs(x) < o / 2
        return o + x * (-o / 2 + x * (o / 6 + x * (-o / 24 + x * (o / 120 + x * (-o / 720 +
               x * (o / 5040 + x * (-o / 40320)))))))
    end
    return (o - exp(-x)) / x
end
function D_distributed(λ, p, c)
    τ, k, τ0 = c
    return λ * λ + p[1] * λ + k + p[2] * τ * exp(-τ0 * λ) * E1T(τ * λ)
end

# --- A.4 / A.5 neutral: λ² + a λ² e^{-τλ} + dλ + k + c e^{-τλ},  p = (a, c) -------------------
function D_neutral(λ, p, c)
    τ, d, k = c
    e = exp(-τ * λ)
    l2 = λ * λ
    return l2 + p[1] * l2 * e + d * λ + k + p[2] * e
end

# --- A.6 PDA control: λ² + 2ζλ + 1 + (P + Dλ + Aλ²) e^{-τλ},  p = (P, A) ----------------------
function D_pda(λ, p, c)
    ζ, Dg, τ = c
    l2 = λ * λ
    return l2 + 2 * ζ * λ + 1 + (p[1] + Dg * λ + p[2] * l2) * exp(-τ * λ)
end

# --- A.8 continuum rod (Zhang & Stépán), entire numerator in scale-free form ------------------
#   D_rd = 1 - K e^{-rλ}/cosh γ,  γ = λ sqrt(1 + c_e/λ) / sqrt(1 + ηλ)   (c_e = 0: the paper)
# numerator cosh γ - K e^{-rλ} times e^{-γ} (analytic, no zeros):
#   N~ = (1 + e^{-2γ})/2 - K e^{-rλ-γ},     denominator~ = (1 + e^{-2γ})/2
# Z = (Φ_den~ - Φ_N~)/π: the e^{-γ} factor cancels; nothing overflows (cosh γ ~ e^{120} at
# ω = 400 does in Float32).
function rod_gamma(λ, c)
    η, ce = c
    return λ * sqrt(1 + ce / λ) / sqrt(1 + η * λ)
end
function D_rod(λ, p, c)
    γ = rod_gamma(λ, c)
    num = (1 + exp(-2 * γ)) / 2
    return num - p[2] * exp(-p[1] * λ - γ)
end
D_rod_den(λ, p, c) = (1 + exp(-2 * rod_gamma(λ, c))) / 2
# the paper's form (s08: D_beam_entire, D_beam_den), Float64 cross-check only
D_rod_paper(λ, p, c) = cosh(λ / sqrt(1 + c[1] * λ)) - p[2] * exp(-p[1] * λ)
D_rod_paper_den(λ, p, c) = cosh(λ / sqrt(1 + c[1] * λ))

# --- A.9 FEM bar: N linear elements, C = ηK, rank-one delayed boundary feedback ----------------
# Q0 = λ²M + λC + K is tridiagonal (uniform elements, h = 1/N, free end = DOF 1):
#   a_1 = λ² 2h/6 + (1+ηλ)/h,  a_k = λ² 4h/6 + 2(1+ηλ)/h,  o = λ² h/6 - (1+ηλ)/h
# det Q0 by the continuant recurrence θ_k = a_k θ_{k-1} - o² θ_{k-2}, and
# (Q0^{-1})_{N,1} = (-1)^{N+1} o^{N-1} / det Q0, so the entire numerator of the return
# difference 1 + c(λ)(Q0^{-1})_{N,1} (s08: D_fem) is
#   det Q0 + c(λ) (-1)^{N+1} o^{N-1},   c(λ) = -K (1+ηλ) e^{-rλ} / h.
# Everything is divided by q^N, q = (4h/6)(λ + √3/h)² (= a_k at λ = 0; poles at λ = -√3 N only,
# far in the left half-plane; the same divisor for numerator and denominator cancels from the
# count): |det Q0| ~ 1e47 at ω = 400 overflows Float32, and so does a divisor that does not
# match a_k at low frequency (432^24 for N = 24).
function fem_parts(λ, c)
    η, Nf = c
    N = round(Int, Nf)
    h = 1 / Nf
    one_eta = 1 + η * λ
    l2 = λ * λ
    lp1 = λ + sqrt(3 * one(h)) / h
    q = (4 * h / 6) * (lp1 * lp1)
    iq = 1 / q
    a1 = (l2 * (2 * h / 6) + one_eta * (1 / h)) * iq
    ak = (l2 * (4 * h / 6) + one_eta * (2 / h)) * iq
    o = (l2 * (h / 6) - one_eta * (1 / h)) * iq
    o2 = o * o
    thp = one(a1)
    th = a1
    on = one(a1)
    for _ in 2:N
        t = ak * th - o2 * thp
        thp = th
        th = t
        on = on * o
    end
    return th, on, iq, one_eta, N, h
end
function D_fem(λ, p, c)
    th, on, iq, one_eta, N, h = fem_parts(λ, c)
    sgn = iseven(N) ? -1 : 1                       # (-1)^{N+1}
    return th + (-p[2] / h * sgn) * (one_eta * exp(-p[1] * λ)) * (on * iq)
end
D_fem_den(λ, p, c) = fem_parts(λ, c)[1]
# the paper's form (s08: build_fem / D_fem_entire / D_fem_den), Float64 cross-check only
function build_fem_paper(n_el, η)
    h = 1.0 / n_el
    n = n_el + 1
    M = zeros(n, n); K = zeros(n, n)
    for i in 1:n_el
        K[i:i+1, i:i+1] .+= (1 / h) .* [1 -1; -1 1]
        M[i:i+1, i:i+1] .+= (h / 6) .* [2 1; 1 2]
    end
    return M[1:end-1, 1:end-1], η .* K[1:end-1, 1:end-1], K[1:end-1, 1:end-1], h
end
function make_fem_paper(η, n_el)
    M, C, K, h = build_fem_paper(n_el, η)
    N = size(K, 1)
    e1 = zeros(N); e1[1] = 1.0
    den(λ, p, c) = LinearAlgebra.det(λ^2 .* M .+ λ .* C .+ K)
    function num(λ, p, c)
        F = LinearAlgebra.lu(λ^2 .* M .+ λ .* C .+ K)
        cc = -p[2] * (1 + η * λ) / h * exp(-p[1] * λ)
        return LinearAlgebra.det(F) * (1 + cc * (F \ e1)[N])
    end
    return num, den
end

# --- A.11 fractional oscillator: λ^α + c λ^β + k e^{-τλ},  p = (k, τ), principal branch -------
function D_frac(λ, p, c)
    α, β, cc = c
    L = log(λ)
    return exp(α * L) + cc * exp(β * L) + p[1] * exp(-p[2] * λ)
end

# --- A.12 Gao, Zhai & Liu (2017) Ex. 1: s^μ (T s^ν + 1) + Kp e^{-Ls} (kp s^μ + ki) ------------
function D_gao(λ, p, c)
    μ, Ld, Kp, Tc, ν = c
    L = log(λ)
    sm = exp(μ * L)
    return sm * (Tc * exp(ν * L) + 1) + Kp * exp(-Ld * λ) * (p[1] * sm + p[2])
end

# --- integral(...) closed forms of the page: φ_k(w) = Σ_j w^j/(j+k)! (k ≥ 2; φ_1 = exprel) -------
function phik(w::Complex{S}, k) where {S}
    if abs(w) < 2 + k
        c = one(S)
        for i in 1:(23 + k)
            c /= S(i)
        end
        r = Complex{S}(c)
        for j in 22:-1:0
            c *= S(j + 1 + k)
            r = w * r + c
        end
        return r
    end
    r = exp(w)
    f = one(S)
    for i in 1:k
        r = (r - f) / w
        f /= S(i)
    end
    return r
end
exprelT(x) = E1T(-x)          # (e^x - 1)/x, the page's dexprel (E1T(y) = (1 - e^{-y})/y)

# --- shimmy, stretched-string tyre (Takács, Orosz & Stépán 2009, Eq. 31),  p = (V, L), c = (Σ, ζ)
# contact-patch memory  2/λ²[(L-1)λ + 2 - ((L+1)λ + 2)e^{-λ}] = ∫₀¹ (2(L-1) + 4θ) e^{-λθ} dθ
#   = (2(L-1) + 4) φ_1(-λ) - 4 φ_2(-λ)   (the page's closed form of integral(...))
function D_shimmy(λ, p, c)
    V, L = p
    Σ, ζ = c
    o = one(V)
    A = (2 * (L - 1) + 4) * exprelT(-λ) - 4 * phik(-λ, 2)
    P = Σ * (V * V) * (λ * λ * λ) + 2 * V * (V + Σ * ζ) * (λ * λ) + (Σ + 4 * ζ * V) * λ + 2
    Q = (L - 1 - Σ) * (A + (L - 1 - Σ) * (2 * Σ * ζ * V * λ + Σ + 4 * ζ * V) +
         (L + 1 + Σ) * (2 * Σ * ζ * V * λ + Σ - 4 * ζ * V) * exp(-λ)) + 4 * ζ * V * L * (1 + Σ) * (2 + Σ * λ)
    return P - Q / (L * L + o / 3 + Σ * (L * L + 1 + Σ))
end
# Eq. (31) as printed (Float64 cross-check only; cancels near λ = 0, divides by L - 1 - Σ)
function D_shimmy_paper(λ, p, c)
    V, L = p
    Σ, ζ = c
    denom = L^2 + 1 / 3 + Σ * (L^2 + 1 + Σ)
    nf = L - 1 - Σ
    tA = (2 / λ^2) * ((L - 1) * λ + 2 - ((L + 1) * λ + 2) * exp(-λ))
    tB = 4ζ * V * L * (1 + Σ) * (2 + Σ * λ) / nf
    tC = nf * (2Σ * ζ * V * λ + Σ + 4ζ * V)
    tD = (L + 1 + Σ) * (2Σ * ζ * V * λ + Σ - 4ζ * V) * exp(-λ)
    poly = Σ * V^2 * λ^3 + 2V * (V + Σ * ζ) * λ^2 + (Σ + 4ζ * V) * λ + 2
    return poly - nf / denom * (tA + tB + tC + tD)
end

# --- two delays with cross-talk, CTCR (Sipahi & Olgac 2004, Eq. 18),  p = (τ1, τ2),
#     c = (c12, a0, b1, b2)
function D_ctcr(λ, p, c)
    τ1, τ2 = p
    c12, a0, b1, b2 = c
    S = typeof(τ1)
    return λ * λ + S(7.1) * λ + a0 + (6 * λ + b1) * exp(-τ1 * λ) + (2 * λ + b2) * exp(-τ2 * λ) +
           c12 * exp(-(τ1 + τ2) * λ)
end

neutral_kw(c) = (hmax = π / (2 * c[1]), ωband = Inf)
const CTCR_KW = (hmax = π / 12, ωband = 50.0)    # h ≤ π/(2(τ1 + τ2)_max) for ω < 50
pda_kw(c) = (hmax = π / (2 * c[3]), ωband = Inf)
const BAR_KW = (hmax = π / 21, ωband = 40.0)        # r = τ/T up to 10.5: π/(2 r_max) over ω < 40

const WEB = Dict(
    "algebraic" => (D = D_algebraic, c = (0.5, 0.0, 0.0), alt = (0.8, 0.5, 0.3),
        xr = (-1.0, 10.0), yr = (-1.0, 10.0), wmax = 1e4, npow = (c, w) -> 2.0, kw = c -> (;), cert = true),
    "distributed" => (D = D_distributed, c = (1.0, 0.0, 0.0), alt = (1.5, 0.3, 0.2),
        xr = (-0.5, 2.0), yr = (-1.0, 5.0), wmax = 1e4, npow = (c, w) -> 2.0, kw = c -> (;), cert = true),
    "neutral" => (D = D_neutral, c = (1.0, 0.0, 1.0), alt = (1.3, 0.3, 1.5),
        xr = (-1.2, 1.2), yr = (-1.2, 1.2), wmax = 200.0, npow = (c, w) -> 2.0, kw = neutral_kw, cert = true),
    "neutral_hg" => (D = D_neutral, c = (1.0, 5.0, 0.0), alt = (0.8, 3.0, 0.5),
        xr = (-1.2, 1.2), yr = (-1.0, 10.0), wmax = 200.0, npow = (c, w) -> 2.0, kw = neutral_kw, cert = true),
    "pda" => (D = D_pda, c = (0.05, 0.1, 1.0), alt = (0.1, 0.3, 0.7),
        xr = (-1.1, 1.4), yr = (-1.15, 1.15), wmax = 500.0, npow = (c, w) -> 2.0, kw = pda_kw, cert = true),
    "rod" => (D = D_rod, c = (0.01, 0.0), alt = (0.03, 0.3),
        xr = (0.02, 10.5), yr = (-0.75, 1.0), wmax = 400.0, npow = nothing, den = D_rod_den,
        kw = c -> BAR_KW, cert = false),
    "fem" => (D = D_fem, c = (0.01, 12.0), alt = (0.02, 8.0),
        xr = (0.02, 10.5), yr = (-0.75, 1.0), wmax = 400.0, npow = nothing, den = D_fem_den,
        kw = c -> BAR_KW, cert = false),
    "frac" => (D = D_frac, c = (1.8, 0.8, 0.5), alt = (1.6, 0.5, 1.0),
        xr = (0.0, 5.0), yr = (0.1, 2.0), wmax = 1e4, npow = (c, w) -> c[1], kw = c -> (;), cert = false),
    "shimmy" => (D = D_shimmy, c = (1.8, 0.02), alt = (1.2, 0.06),
        xr = (0.001, 0.6), yr = (-0.2, 7.0), wmax = 1e4, npow = (c, w) -> 3.0, kw = c -> (;), cert = true),
    "ctcr" => (D = D_ctcr, c = (8.0, 21.1425, 14.8, 7.3), alt = (12.0, 18.0, 14.8, 7.3),
        xr = (0.0, 3.0), yr = (0.0, 3.0), wmax = 1e4, npow = (c, w) -> 2.0, kw = c -> CTCR_KW, cert = true),
    "gao" => (D = D_gao, c = (1.5, 0.4, 5.0, 10.0, 0.5), alt = (1.2, 0.3, 4.0, 10.0, 0.5),
        xr = (-1.0, 6.0), yr = (-1.0, 25.0), wmax = 1e4, npow = (c, w) -> c[1] + c[5], kw = c -> (;), cert = false),
)

end # include guard
