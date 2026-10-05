# GPU (KernelAbstractions) forms of the time-periodic characteristic functions, for the
# unchanged NyquistGPU kernel. Every D here is a function of μ = λ/ω_p(point): the period
# strip is then μ ∈ [a, a+1] for every point (march ω0 = a, ω_max = a + 1, n_power = 0),
# and the kernel's Z_raw = -Φ/π is twice the count: Z = Z_raw / 2.
# Constants come in one flat tuple c (Float32/Float64 on the device); Fourier coefficients
# are packed as real and imaginary parts.
using StaticArrays, ForwardDiff

const A_STRIP = 0.237
const NM2 = 24          # max harmonics (tooth period, Test 2): matrix 49 x 49 per thread
const NM3 = 26          # max harmonics (spindle period, Test 3): matrix 53 x 53 per thread
# a priori N for milling, calibrated against the adaptive (ring-change) N of the CPU study:
#   N = ⌈2 sqrt(1 + w H_max/√tol) / ω_p⌉ + 3   (the 1/m tail of the cutting harmonics couples
#   about twice as far as the Mathieu rule)

@inline _val(x::ForwardDiff.Dual) = ForwardDiff.value(x)
@inline _val(x::Real) = x
@inline _mag2(z) = _val(real(z))^2 + _val(imag(z))^2
# complex division without Base's scaled algorithm: for dual-number components it calls
# exponent(), which THROWS on an exact zero (a kernel exception on the GPU); here a zero
# denominator just gives Inf/NaN, which the march treats as a failed sample
# (scaled by the primal magnitude first, so neither |b|² nor its derivative underflows in Float32)
@inline function cinv(b)
    sc = max(abs(_val(real(b))), abs(_val(imag(b))))
    sc = ifelse(sc > 0, sc, one(sc))
    bs = b / sc
    return conj(bs) * inv(real(bs) * real(bs) + imag(bs) * imag(bs)) / sc
end
@inline cdiv(a, b) = a * cinv(b)

# LU with partial pivoting on the leading n x n block of an MMatrix, returns det
@inline function lu_det!(A, n)
    d = one(eltype(A))
    for k in 1:n
        p = k
        best = _mag2(@inbounds A[k, k])
        for i in (k + 1):n
            v = _mag2(@inbounds A[i, k])
            if v > best
                best = v
                p = i
            end
        end
        if p != k
            for j in k:n
                @inbounds a = A[k, j]
                @inbounds A[k, j] = A[p, j]
                @inbounds A[p, j] = a
            end
            d = -d
        end
        @inbounds piv = A[k, k]
        d *= piv
        ip = cinv(piv)
        for i in (k + 1):n
            @inbounds f = A[i, k] * ip
            for j in (k + 1):n
                @inbounds A[i, j] -= f * A[k, j]
            end
        end
    end
    return d
end

# log sinh without overflow (only exp of sums of it is used)
# complex log without Base's scaled algorithm (it throws on an exact zero for dual numbers)
@inline function clog(z)
    sc = max(abs(_val(real(z))), abs(_val(imag(z))))
    sc = ifelse(sc > 0, sc, one(sc))
    zs = z / sc
    return Complex(log(sc) + log(real(zs) * real(zs) + imag(zs) * imag(zs)) / 2, atan(imag(z), real(z)))
end
@inline lsinh(x) = real(x) >= 0 ? x + clog((1 - exp(-2x)) * oftype(real(x), 0.5)) : -x + clog((exp(2x) - 1) * oftype(real(x), 0.5))

# closed-form diagonal factor: Π_k (s_k - z1)(s_k - z2)/(s_k + cs)², s_k = λ + ikω_p (all k)
@inline function diag_closed(λ, z1, z2, cs, ωp)
    w = π / ωp
    return exp(lsinh(w * (λ - z1)) + lsinh(w * (λ - z2)) - 2 * lsinh(w * (λ + cs)))
end

# ---------------------------------------------------------------------------------------
# delayed Mathieu  x'' + κx' + (δ + ε cos t)x = b x(t - 2π);  p = (δ, b), c = (κ, ε, tol)
# ---------------------------------------------------------------------------------------
function D_mathieu(μ, p, c)
    δ, b = p
    κ, ε, tol = c
    λ = μ                                         # ω_p = 1
    N = unsafe_trunc(Int32, sqrt(max(δ, zero(δ)) + ε / (2 * sqrt(tol)))) + Int32(2)
    e2 = (ε / 2)^2
    B = δ - b * exp(-2 * oftype(δ, π) * λ)
    cc = one(δ)
    s = λ - Complex(zero(δ), oftype(δ, N))
    rprev = (s + cc)^2
    dr = cdiv(s * s + κ * s + B, rprev)
    f = dr; fprev = one(f); P = dr
    k = -N + Int32(1)
    while k <= N
        s = λ + Complex(zero(δ), oftype(δ, k))
        r = (s + cc)^2
        dr = cdiv(s * s + κ * s + B, r)
        fn = dr * f - e2 * cinv(r * rprev) * fprev
        fprev = f; f = fn; P *= dr; rprev = r
        k += Int32(1)
    end
    sq = sqrt(κ * κ - 4 * B)
    return f * cdiv(diag_closed(λ, (-κ + sq) / 2, (-κ - sq) / 2, cc, one(δ)), P)
end
mathieu_consts(κ, ε; tol = 1e-4) = (κ, ε, tol)
mathieu_ωp(p, c) = one(p[1])

# ---------------------------------------------------------------------------------------
# Test 2: straight-fluted 1-DOF milling, tooth period.  p = (rpm/1000, a_p [mm])
# c = (ζ, w1, z, cs, tol, Hmax, fn [Hz], H_re(0..2NM2)..., H_im(0..2NM2)...)
# ---------------------------------------------------------------------------------------
@inline _H(c, m, off, NM) = m >= 0 ? Complex(c[off + m], c[off + 2NM + 1 + m]) :
                                    Complex(c[off - m], -c[off + 2NM + 1 - m])

mill2_ωp(p, c) = c[3] * p[1] * 1000 / (60 * c[7])

function D_mill2(μ, p, c)
    rk, ap = p
    ζ, w1, z, cs, tol, Hmax = c[1], c[2], c[3], c[4], c[5], c[6]
    ωp = mill2_ωp(p, c)
    λ = μ * ωp
    w = w1 * ap
    N = min(unsafe_trunc(Int32, 2 * sqrt(1 + w * Hmax / sqrt(tol)) / ωp) + Int32(4), Int32(NM2))
    n = 2N + 1
    E = 1 - exp(-(2 * oftype(ζ, π) / ωp) * λ)
    B = 1 + w * _H(c, 0, 8, NM2) * E
    A = MMatrix{2NM2 + 1, 2NM2 + 1, typeof(λ)}(undef)
    for j in 1:n, i in 1:n
        k = i - N - 1
        l = j - N - 1
        s = λ + Complex(zero(ζ), k * ωp)
        r = (s + cs)^2
        @inbounds A[i, j] = cdiv(i == j ? s * s + 2ζ * s + B : w * _H(c, k - l, 8, NM2) * E, r)
    end
    dA = lu_det!(A, n)
    sq = sqrt(4ζ * ζ - 4 * B)
    z1 = (-2ζ + sq) / 2
    z2 = (-2ζ - sq) / 2
    P = one(λ)
    for k in (-N):N
        s = λ + Complex(zero(ζ), k * ωp)
        P *= cdiv((s - z1) * (s - z2), (s + cs)^2)
    end
    return dA * cdiv(diag_closed(λ, z1, z2, cs, ωp), P)
end

# ---------------------------------------------------------------------------------------
# Test 3: two flutes with helix angles β1, β2 (uniform pitch π), spindle period.
# p = (rpm/1000, a_p [mm]); c = (ζ, w1, cs, tol, Cmax, fn, tanβ1, tanβ2, R, C_re(0..2NM3)..., C_im...)
# ---------------------------------------------------------------------------------------
mill3_ωp(p, c) = p[1] * 1000 / (60 * c[6])

@inline phi1(x) = _mag2(x) < 1e-6 ? 1 + x / 2 + x * x / 6 + x * x * x / 24 : cdiv(exp(x) - 1, x)

# K_m(s) = (1/L) Σ_j e^{imθ_j} ∫_0^L e^{-im b_j ζ} (1 - e^{-s τ_j(ζ)}) dζ, closed form
@inline function kernel3(m, s, Ω, L, tb1, tb2, R)
    T = typeof(Ω)
    acc = zero(s)
    for j in 1:2
        θ = j == 1 ? zero(T) : -oftype(Ω, π)
        b = (j == 1 ? tb1 : tb2) / R
        Δ = (j == 1 ? tb1 - tb2 : tb2 - tb1) / R
        a1 = Complex(zero(T), -m * b * L)
        a2 = a1 - s * (Δ * L / Ω)
        acc += cis(m * θ) * (phi1(a1) - exp(-s * (oftype(Ω, π) / Ω)) * phi1(a2))
    end
    return acc
end

function D_mill3(μ, p, c)
    rk, ap = p
    ζ, w1, cs, tol, Cmax, fn, tb1, tb2, R = c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]
    Ω = mill3_ωp(p, c)
    ωp = Ω
    λ = μ * ωp
    w = w1 * ap
    L = ap
    N = min(unsafe_trunc(Int32, 2 * sqrt(1 + w * Cmax / sqrt(tol)) / ωp) + Int32(4), Int32(NM3))
    n = 2N + 1
    A = MMatrix{2NM3 + 1, 2NM3 + 1, typeof(λ)}(undef)
    for j in 1:n, i in 1:n
        k = i - N - 1
        l = j - N - 1
        s = λ + Complex(zero(ζ), k * ωp)
        sl = λ + Complex(zero(ζ), l * ωp)
        v = w * _H(c, k - l, 10, NM3) * kernel3(k - l, sl, Ω, L, tb1, tb2, R)
        @inbounds A[i, j] = cdiv(i == j ? s * s + 2ζ * s + 1 + v : v, (s + cs)^2)
    end
    dA = lu_det!(A, n)
    Bt = 1 + w * real(_H(c, 0, 10, NM3)) * 2          # mean (non-delayed) part for the far tail
    sq = sqrt(Complex(4ζ * ζ - 4 * Bt, zero(ζ)))
    z1 = (-2ζ + sq) / 2
    z2 = (-2ζ - sq) / 2
    P = one(λ)
    for k in (-N):N
        s = λ + Complex(zero(ζ), k * ωp)
        P *= cdiv((s - z1) * (s - z2), (s + cs)^2)
    end
    tail = cdiv(diag_closed(λ, z1, z2, cs, ωp), P)
    for k in (N + 1):(4N)                             # exact d_k / mean d_k, N < |k| ≤ 4N
        for sg in (-1, 1)
            s = λ + Complex(zero(ζ), sg * k * ωp)
            v = w * _H(c, 0, 10, NM3) * kernel3(0, s, Ω, L, tb1, tb2, R)
            tail *= cdiv(s * s + 2ζ * s + 1 + v, (s - z1) * (s - z2))
        end
    end
    return dA * tail
end

# --- host-side constant builders (Fourier coefficients of the cutting function) --------
function cut_fourier(aD, kr, down, M; q = 1)
    a, b = down ? (acos(2aD - 1), Float64(π)) : (0.0, acos(1 - 2aD))
    n = 400
    x = [cos(π * (k - 0.5) / n) for k in 1:n]              # Gauss-Chebyshev-like nodes on [-1,1]
    # composite midpoint on the window is accurate enough for M ≤ 100 with 4000 points
    np = 4000
    φ = [a + (b - a) * (k - 0.5) / np for k in 1:np]
    f = sin.(φ) .* (cos.(φ) .+ kr .* sin.(φ))
    return [sum(f .* cis.(-m * q .* φ)) * (b - a) / np / 2π for m in 0:M]
end

function mill2_consts(; ζ = 0.011, aD = 0.05, kr = 1 / 3, down = true, z = 2, w1 = 0.4478, fn = 922.0,
                      tol = 1e-3, cs = 1.0)
    C = cut_fourier(aD, kr, down, 2NM2; q = z)
    H = z .* C                                          # tooth harmonics
    Hmax = maximum(abs, H)
    return (ζ, w1, Float64(z), cs, tol, Hmax, fn, real.(H)..., imag.(H)...)
end

function mill3_consts(; ζ = 0.011, aD = 0.05, kr = 1 / 3, down = true, β1 = 30.0, β2 = 45.0, R = 8.0,
                      w1 = 0.4478, fn = 922.0, tol = 1e-3, cs = 1.0)
    C = cut_fourier(aD, kr, down, 2NM3; q = 1)
    return (ζ, w1, cs, tol, maximum(abs, C), fn, tand(β1), tand(β2), R, real.(C)..., imag.(C)...)
end
