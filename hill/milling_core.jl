# Hill determinant + argument principle for milling (Tests 2 and 3), CPU core.
#
# 1-DOF milling, time scaled by ω_n, w = a_p K_t/(m ω_n²) (dimensionless depth of cut):
#   x'' + 2ζx' + x = -(w/L) Σ_j ∫_0^L g(φ_j) f(φ_j) [x(t) - x(t - τ_j(ζ))] dζ
#   f(φ) = sin φ (cos φ + k_r sin φ),  g = 1 on the cutting window [φ_en, φ_ex]
#   φ_j(t, ζ) = Ω t + θ_j - ζ tanβ_j / R      (tooth j, height ζ ∈ [0, L], L = a_p)
#   τ_j(ζ) = (p_j + ζ (tanβ_j - tanβ_{j-1})/R) / Ω,  p_j = pitch angle to the previous tooth
# Ω = spindle speed / ω_n. Test 2: straight teeth (β = 0), uniform pitch -> tooth period,
# one point delay τ = T. Test 3: different helix angles -> spindle period, every tooth a
# delay distributed linearly over the axial depth.
#
# Floquet ansatz x = e^{λt} Σ_k c_k e^{ikω_p t}, s_k = λ + ikω_p. Hill matrix
#   A_kl = δ_kl p(s_k) + w C_{k-l} K_{k-l}(s_l),  p(s) = s² + 2ζs + 1
# C_m: Fourier coefficients of the cutting function, K_m(s): geometry/delay kernel.
# Rows are scaled by r_k = (s_k + c)² (pole-free regularization, see hill_core.jl), the
# determinant is an LU of the (2N+1)² block times the diagonal tail Π_{|k|>N} d_k/r_k.

using InterpolatedNyquist, ForwardDiff, LinearAlgebra
const IN = InterpolatedNyquist

Base.@kwdef struct Mill
    ζ::Float64 = 0.011
    kr::Float64 = 1 / 3              # K_n / K_t
    z::Int = 2
    aD::Float64 = 0.05               # radial immersion a/D
    down::Bool = true
    w1::Float64 = 0.4478             # w per mm of axial depth (Insperger-Stépán benchmark)
    R::Float64 = 8.0                 # tool radius [mm] (helix: angular lag ζ tanβ/R)
    tanβ::Vector{Float64} = zeros(2) # per tooth (all equal and uniform pitch: tooth period)
    pitch::Vector{Float64} = fill(π, 2)   # p_j [rad], sums to 2π
    c::Float64 = 1.0                 # row-scale shift (poles of the scaled determinant at -c-ikω_p)
end
Mill2(; kw...) = Mill(; kw...)       # Test 2 defaults
window(m::Mill) = m.down ? (acos(2m.aD - 1), Float64(π)) : (0.0, acos(1 - 2m.aD))
fcut(φ, m::Mill) = sin(φ) * (cos(φ) + m.kr * sin(φ))
straight_uniform(m::Mill) = all(iszero, m.tanβ) && all(≈(2π / m.z), m.pitch)
θs(m::Mill) = [-sum(m.pitch[2:j]; init = 0.0) for j in 1:m.z]   # θ_1 = 0, tooth j trails j-1

# Gauss-Legendre nodes on [-1, 1] (Golub-Welsch)
function gauss(n)
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end
const GL = gauss(400)

# C_m = (1/2π) ∫_{φen}^{φex} f(φ) e^{-i m q φ} dφ   (q = 1 spindle harmonics, q = z tooth harmonics: ×z)
function fourier(m::Mill, M::Int; q = 1)
    a, b = window(m)
    x, wgl = GL
    φ = (b - a) / 2 .* x .+ (a + b) / 2
    jac = (b - a) / 2
    return [sum(wgl .* fcut.(φ, Ref(m)) .* cis.(-mm * q .* φ)) * jac / 2π for mm in -M:M]
end

# robust log(sinh x) (no overflow), only exp of sums of it is used
lsinh(x) = real(x) >= 0 ? x + log((1 - exp(-2x)) / 2) : -x + log((exp(2x) - 1) / 2)

# expm1(x)/x, stable at x -> 0
@inline phi1(x) = abs(x) < 1e-4 ? 1 + x / 2 + x^2 / 6 : expm1c(x) / x
@inline expm1c(x) = exp(x) - 1

# --- problem setup per parameter point ---------------------------------------------
struct MillPoint{T}
    ωp::Float64          # principal frequency of the Hill expansion
    w::Float64
    Ω::Float64
    L::Float64           # axial depth [mm]
    C::Vector{ComplexF64}   # coefficients C_m, m = -2N..2N (index m + M + 1)
    M::Int
    tooth::Bool          # tooth-period formulation (Test 2)
end

function setup(m::Mill, Ω, ap, Nmax)
    w = m.w1 * ap
    if straight_uniform(m)
        ωp = m.z * Ω
        H = m.z .* fourier(m, 2Nmax; q = m.z)          # tooth harmonics: H_m = z C_{mz}
        return MillPoint{Float64}(ωp, w, Ω, ap, H, 2Nmax, true)
    end
    return MillPoint{Float64}(Ω, w, Ω, ap, fourier(m, 2Nmax; q = 1), 2Nmax, false)
end

# kernel K_m(s) (spindle formulation): (1/L) Σ_j e^{imθ_j} ∫_0^L e^{-imb_jζ}(1 - e^{-sτ_j(ζ)}) dζ
# exact (closed form, linear delays) or with `ns` axial slices (finite-point kernel)
function Kker(mm, s, m::Mill, P::MillPoint, θ; ns = 0)
    L = P.L; Ω = P.Ω
    acc = zero(s)
    for j in 1:m.z
        jp = j == 1 ? m.z : j - 1
        b = m.tanβ[j] / m.R
        Δ = (m.tanβ[j] - m.tanβ[jp]) / m.R
        pj = m.pitch[j]
        if ns == 0
            a1 = -1im * mm * b * L
            a2 = a1 - s * Δ * L / Ω
            acc += cis(mm * θ[j]) * (phi1(a1) - exp(-s * pj / Ω) * phi1(a2))
        else
            for l in 1:ns
                ζ = (l - 0.5) * L / ns
                acc += cis(mm * (θ[j] - b * ζ)) * (1 - exp(-s * (pj + Δ * ζ) / Ω)) / ns
            end
        end
    end
    return acc
end

# scaled Hill determinant at λ with harmonics -N..N
function mill_det(λ, N::Int, m::Mill, P::MillPoint; ns = 0, ktail = 4)
    ωp = P.ωp; w = P.w; c = m.c
    n = 2N + 1
    T = typeof(λ)
    A = Matrix{T}(undef, n, n)
    sk(k) = λ + 1im * k * ωp
    Cm(mm) = P.C[mm + P.M + 1]
    θ = θs(m)
    if P.tooth
        E = 1 - exp(-2π / ωp * λ)                     # common delay factor (τ = T)
        B = 1 + w * Cm(0) * E
        for i in 1:n, j in 1:n
            k = i - N - 1; l = j - N - 1
            s = sk(k)
            A[i, j] = (i == j ? s^2 + 2m.ζ * s + B : w * Cm(k - l) * E) / (s + c)^2
        end
        Bt = B                                         # tail exact (d_k = s_k² + 2ζ s_k + B)
    else
        for i in 1:n, j in 1:n
            k = i - N - 1; l = j - N - 1
            s = sk(k)
            v = w * Cm(k - l) * Kker(k - l, sk(l), m, P, θ; ns)
            A[i, j] = (i == j ? s^2 + 2m.ζ * s + 1 + v : v) / (s + c)^2
        end
        Bt = 1 + w * Cm(0) * m.z                       # mean (non-delayed) part for the far tail
    end
    det_A = lu_det!(A, n)
    # diagonal tail: closed form with constant B, Π_k (s_k - z1)(s_k - z2)/(s_k + c)²
    sq = sqrt(complex(4m.ζ^2 - 4Bt))
    z1, z2 = (-2m.ζ + sq) / 2, (-2m.ζ - sq) / 2
    wpi = π / ωp
    lG = lsinh(wpi * (λ - z1)) + lsinh(wpi * (λ - z2)) - 2 * lsinh(wpi * (λ + c))
    lP = zero(λ)
    for k in -N:N
        s = sk(k)
        lP += log((s - z1) * (s - z2) / (s + c)^2)
    end
    tail = exp(lG - lP)
    if !P.tooth                                        # exact d_k vs the mean for N < |k| ≤ ktail N
        for k in vcat(-ktail*N:-N-1, N+1:ktail*N)
            s = sk(k)
            v = w * Cm0(P) * Kker(0, s, m, P, θ; ns)
            tail *= (s^2 + 2m.ζ * s + 1 + v) / ((s - z1) * (s - z2))
        end
    end
    return det_A * tail
end
Cm0(P::MillPoint) = P.C[P.M + 1]

# LU with partial pivoting (pivot on the primal magnitude), returns det
function lu_det!(A, n)
    d = one(eltype(A))
    for k in 1:n
        p = k
        best = abs2(ForwardDiff.value(real(A[k, k]))) + abs2(ForwardDiff.value(imag(A[k, k])))
        for i in k+1:n
            v = abs2(ForwardDiff.value(real(A[i, k]))) + abs2(ForwardDiff.value(imag(A[i, k])))
            if v > best
                best = v; p = i
            end
        end
        if p != k
            for j in k:n
                A[k, j], A[p, j] = A[p, j], A[k, j]
            end
            d = -d
        end
        piv = A[k, k]
        d *= piv
        for i in k+1:n
            f = A[i, k] / piv
            for j in k+1:n
                A[i, j] -= f * A[k, j]
            end
        end
    end
    return d
end

ForwardDiff.value(x::Real) = x

# --- truncation, winding, count ---------------------------------------------------------
function ring_change(N, m::Mill, P::MillPoint; a = 0.237, probes = (0.0, 0.25, 0.5, 0.75), ns = 0)
    e = 0.0
    for t in probes
        λ = 1im * (a + t) * P.ωp
        e = max(e, abs(mill_det(λ, N + 1, m, P; ns) / mill_det(λ, N, m, P; ns) - 1))
    end
    return e
end

n_apriori(m::Mill, P::MillPoint, tol) =
    ceil(Int, sqrt(1 + P.w * maximum(abs, P.C) / sqrt(tol)) / P.ωp) + 1

function choose_N(m::Mill, P::MillPoint, tol; Nmax = 40, ns = 0)
    N = min(n_apriori(m, P, tol), Nmax)
    e = ring_change(N, m, P; ns)
    while e > tol && N < Nmax
        N += 1
        e = ring_change(N, m, P; ns)
    end
    return N, e
end

function strip_count(N, m::Mill, P::MillPoint; a = 0.237, ns = 0, tol = 0.3)
    f = IN.NyquistWrapper{NTuple{1, Float64}}((μ, p) -> mill_det(μ * P.ωp, N, m, P; ns))
    Φ, ok, ev, dd, ds, dw = IN._unwrap_march(f, (0.0,), 0.0, 1; ω0 = a, ω_max = a + 1, tol = tol,
        h0 = 1e-3, hrel = 0.05)
    return -Φ / (2π), ok, ev, ds[1] * P.ωp, dw[1]
end

"Unstable Floquet exponents of the milling model at spindle speed Ω (/ω_n) and depth ap [mm]."
function mill_count(m::Mill, Ω, ap; tol = 1e-3, N = nothing, Nmax = 40, ns = 0)
    P = setup(m, Ω, ap, Nmax + 1)
    Ne, err = N === nothing ? choose_N(m, P, tol; Nmax, ns) : (N, ring_change(N, m, P; ns))
    W, ok, ev, σ, μ = strip_count(Ne, m, P; ns)
    return (Z = ok ? round(Int, W) : -1, Zraw = W, N = Ne, err = err, evals = ev, σ = σ, θ = mod(μ, 1.0))
end

# --- time-domain reference: RK4 monodromy, history (x, v) on the grid of one period ------
# Test 2 (tooth period, τ = T): grid aligned with the cutting window (exact RK4 order).
function ref_tooth(m::Mill, Ω, ap; m1 = 40, m2 = 40)
    @assert straight_uniform(m)
    w = m.w1 * ap
    T = 2π / (m.z * Ω)
    φen, φex = window(m)
    tc = (φex - φen) / Ω
    ts = vcat(range(0, tc; length = m1 + 1), range(tc, T; length = m2 + 1)[2:end])
    n = length(ts)
    hfun(t) = (t <= tc + 1e-14) ? fcut(φen + Ω * t, m) : 0.0
    f(t, x, v, xd, inc) = (v, -2m.ζ * v - x - w * (inc ? hfun(t) : 0.0) * (x - xd))
    M = zeros(2n, 2n)
    for col in 1:2n
        Y = zeros(2n); Y[col] = 1
        xo, vo = Y[1:n], Y[n+1:2n]
        xn = zeros(n); vn = zeros(n)
        xn[1], vn[1] = xo[n], vo[n]
        for j in 1:n-1
            t = ts[j]; h = ts[j+1] - ts[j]
            inc = j <= m1                                # inside the cutting window
            x, v = xn[j], vn[j]
            xa, xb = xo[j], xo[j+1]
            xm = (xa + xb) / 2 + h * (vo[j] - vo[j+1]) / 8
            hf(tt) = inc ? fcut(φen + Ω * tt, m) : 0.0
            g(tt, x, v, xd) = (v, -2m.ζ * v - x - w * hf(tt) * (x - xd))
            k1 = g(t, x, v, xa)
            k2 = g(t + h/2, x + h/2 * k1[1], v + h/2 * k1[2], xm)
            k3 = g(t + h/2, x + h/2 * k2[1], v + h/2 * k2[2], xm)
            k4 = g(t + h, x + h * k3[1], v + h * k3[2], xb)
            xn[j+1] = x + h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
            vn[j+1] = v + h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
        end
        M[1:n, col] = xn; M[n+1:2n, col] = vn
    end
    μ = eigvals(M)
    return count(>(1), abs.(μ)), maximum(abs.(μ)), μ
end

# --- Test 3: general reference with distributed delays (spindle period) ------------------
# RK4 on a uniform grid of the spindle period T_s; every tooth j and axial slice l is a
# point delay τ_jl (< T_s); delayed values by cubic Hermite interpolation of (x, v) on the
# grid (previous and current period). The axial integral uses `ns` midpoint slices.
function ref_general(m::Mill, Ω, ap; msteps = 400, ns = 32)
    w = m.w1 * ap
    Ts = 2π / Ω
    h = Ts / msteps
    n = msteps + 1
    θ = θs(m)
    φen, φex = window(m)
    L = ap
    # slices: angle offset ψ_jl and delay τ_jl
    ψ = Float64[]; τs = Float64[]
    for j in 1:m.z, l in 1:ns
        jp = j == 1 ? m.z : j - 1
        ζ = (l - 0.5) * L / ns
        push!(ψ, θ[j] - ζ * m.tanβ[j] / m.R)
        push!(τs, (m.pitch[j] + ζ * (m.tanβ[j] - m.tanβ[jp]) / m.R) / Ω)
    end
    @assert maximum(τs) < Ts && minimum(τs) > 2h
    incut(φ) = (φm = mod(φ, 2π); φen <= φm <= φex)
    hjl(t, k) = (φ = Ω * t + ψ[k]; incut(φ) ? fcut(mod(φ, 2π), m) : 0.0)
    M = zeros(2n, 2n)
    tg = collect(0:msteps) .* h
    for col in 1:2n
        Y = zeros(2n); Y[col] = 1
        xo, vo = Y[1:n], Y[n+1:2n]
        xn = zeros(n); vn = zeros(n)
        xn[1], vn[1] = xo[n], vo[n]
        function xdel(td, jmax)                  # x at time td ∈ [-Ts, t]
            if td < 0
                u = (td + Ts) / h; i = clamp(floor(Int, u), 0, msteps - 1)
                xa, xb, va, vb = xo[i+1], xo[i+2], vo[i+1], vo[i+2]
            else
                u = td / h; i = clamp(floor(Int, u), 0, jmax - 1)
                xa, xb, va, vb = xn[i+1], xn[i+2], vn[i+1], vn[i+2]
            end
            s = u - i; s2 = s * s; s3 = s2 * s
            return (2s3 - 3s2 + 1) * xa + (s3 - 2s2 + s) * h * va + (-2s3 + 3s2) * xb + (s3 - s2) * h * vb
        end
        function rhs(t, x, v, jmax)
            F = 0.0
            for k in eachindex(τs)
                hk = hjl(t, k)
                hk == 0 && continue
                F += hk * (x - xdel(t - τs[k], jmax))
            end
            return (v, -2m.ζ * v - x - w / ns * F)
        end
        for j in 1:msteps
            t = tg[j]; x, v = xn[j], vn[j]
            k1 = rhs(t, x, v, j - 1)
            k2 = rhs(t + h/2, x + h/2 * k1[1], v + h/2 * k1[2], j - 1)
            k3 = rhs(t + h/2, x + h/2 * k2[1], v + h/2 * k2[2], j - 1)
            k4 = rhs(t + h, x + h * k3[1], v + h * k3[2], j - 1)
            xn[j+1] = x + h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
            vn[j+1] = v + h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
        end
        M[1:n, col] = xn; M[n+1:2n, col] = vn
    end
    μ = eigvals(M)
    return count(>(1), abs.(μ)), maximum(abs.(μ))
end

# Test 3 tool: two flutes, uniform pitch, helix 30° and 45°, R = 8 mm
Mill3(; kw...) = Mill(; tanβ = [tand(30.0), tand(45.0)], kw...)
