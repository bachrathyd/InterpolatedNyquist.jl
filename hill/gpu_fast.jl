# Fast GPU form of the milling characteristic function (Test 2): the infinite Hill determinant
# compressed to a Q x Q determinant (see fredholm_core.jl), counted along the upper half of the
# unit circle of the Floquet multiplier z = e^{λT}.
#
#   F(z) = det(I_Q + α(λ) M(λ)),   α = w (1 - e^{-λT}),   M_pq = c_q S(ψ_p - ψ_q; λ),
#   S(Δ) = Σ_{k∈Z} e^{ikΔ}/p(λ + ikω) = T/(r1 - r2) [e^{-a1Δ/ω}/(1 - e^{-a1 T}) - (1 ↔ 2)],
#   a_i = λ - r_i,  r_{1,2} = -ζ ± i sqrt(1 - ζ²) (roots of p),  Δ ∈ [0, 2π).
# F is periodic in Im λ (a similarity transform), so a function of z; its only poles are the
# free-oscillator multipliers e^{r_i T} (inside the unit circle) and F → 1 as |z| → ∞: the winding
# along |z| = 1 counts the unstable multipliers (the generalized Nyquist criterion of the
# regenerative loop, compressed to the Q quadrature nodes of the cutting window). F(z̄) = conj F(z):
#   Z = -(1/π) Δarg F over Im λ ∈ [0, ω/2]  ->  march μ = λ/ω from 0 to 1/2, n_power = 0,
#   and the kernel's Z_raw = -Φ/π is the count itself.
# Per evaluation: 2Q complex exponentials (dual numbers) + Q² entries + an unrolled Q x Q LU.
using ForwardDiff

if !@isdefined(cinv)        # shared with gpu_models.jl
    @inline _val(x::ForwardDiff.Dual) = ForwardDiff.value(x)
    @inline _val(x::Real) = x
    @inline _mag2(z) = _val(real(z))^2 + _val(imag(z))^2
    @inline function cinv(b)
        sc = max(abs(_val(real(b))), abs(_val(imag(b))))
        isc = inv(ifelse(sc > 0, sc, one(sc)))          # one division instead of nine
        bs = b * isc
        return conj(bs) * (inv(real(bs) * real(bs) + imag(bs) * imag(bs)) * isc)
    end
end

# Code generation: a Q x Q LU with branch-free partial pivoting on scalar variables a_i_j
# (registers; no arrays, no run-time indexing), leaving the determinant in `d`.
_a(i, j) = Symbol(:a_, i, :_, j)
function lu_exprs(Q)
    ex = Any[:(d = one(a_1_1))]
    for k in 1:Q
        if k < Q
            push!(ex, :(pk = $k))
            push!(ex, :(best = _mag2($(_a(k, k)))))
            for i in (k + 1):Q
                push!(ex, :(v = _mag2($(_a(i, k)))))
                push!(ex, :(pk = ifelse(v > best, $i, pk)))
                push!(ex, :(best = ifelse(v > best, v, best)))
            end
            for i in (k + 1):Q, j in k:Q            # swap row k with row pk (at most one i matches)
                push!(ex, :(s = pk == $i))
                push!(ex, :(t = $(_a(k, j))))
                push!(ex, :($(_a(k, j)) = ifelse(s, $(_a(i, j)), t)))
                push!(ex, :($(_a(i, j)) = ifelse(s, t, $(_a(i, j)))))
            end
            push!(ex, :(d = ifelse(pk == $k, d, -d)))
        end
        push!(ex, :(d *= $(_a(k, k))))
        if k < Q
            push!(ex, :(ip = cinv($(_a(k, k)))))
            for i in (k + 1):Q
                push!(ex, :(f = $(_a(i, k)) * ip))
                for j in (k + 1):Q
                    push!(ex, :($(_a(i, j)) -= f * $(_a(k, j))))
                end
            end
        end
    end
    return ex
end

# p = (rpm/1000, a_p [mm]);  c = (ζ, w1, z, f_n, ψ_1..ψ_Q (ascending), c_1..c_Q)
mill2q_ωp(p, c) = c[3] * p[1] * 1000 / (60 * c[4])

# the whole characteristic function generated for the number of nodes Q = (length(c) - 4) / 2
@generated function D_mill2q(μ, p, c::NTuple{L, TT}) where {L, TT}
    Q = (L - 4) ÷ 2
    ex = Any[quote
        rk, ap = p
        ζ, w1 = c[1], c[2]
        twoπ = TT(6.283185307179586)
        ω = mill2q_ωp(p, c)
        T = twoπ / ω
        λ = μ * ω
        w = w1 * ap
        sζ = sqrt(1 - ζ * ζ)
        r1 = Complex(-ζ, sζ)                     # r2 = conj(r1)
        E = exp(-λ * T)                          # e^{-λT} = 1/z
        α = w * (1 - E)
        er1 = exp(r1 * T)
        W1 = er1 * E                             # e^{-a1 T}
        W2 = conj(er1) * E                       # e^{-a2 T}
        inv2is = Complex(zero(TT), -1 / (2 * sζ))
        k1 = T * inv2is * cinv(1 - W1)
        k2 = T * inv2is * cinv(1 - W2)
        iω = 1 / ω
    end]
    for i in 1:Q                                 # e^{∓a_i ψ/ω} = e^{∓λψ/ω} e^{±r_i ψ/ω} per node
        push!(ex, quote
            $(Symbol(:x_, i)) = c[$(4 + i)] * iω
            $(Symbol(:el_, i)) = exp(-λ * $(Symbol(:x_, i)))
            $(Symbol(:er_, i)) = exp(r1 * $(Symbol(:x_, i)))
            $(Symbol(:u1_, i)) = $(Symbol(:el_, i)) * $(Symbol(:er_, i))
            $(Symbol(:u2_, i)) = $(Symbol(:el_, i)) * conj($(Symbol(:er_, i)))
            $(Symbol(:v1_, i)) = cinv($(Symbol(:u1_, i)))
            $(Symbol(:v2_, i)) = cinv($(Symbol(:u2_, i)))
            $(Symbol(:g_, i)) = α * c[$(4 + Q + i)]
        end)
    end
    for i in 1:Q, j in 1:Q                       # M_ij = δ_ij + α c_j S(ψ_i - ψ_j)
        e1 = i < j ? :($(Symbol(:u1_, i)) * $(Symbol(:v1_, j)) * W1) : :($(Symbol(:u1_, i)) * $(Symbol(:v1_, j)))
        e2 = i < j ? :($(Symbol(:u2_, i)) * $(Symbol(:v2_, j)) * W2) : :($(Symbol(:u2_, i)) * $(Symbol(:v2_, j)))
        diag = i == j ? :(one(λ)) : :(zero(λ))
        push!(ex, :($(_a(i, j)) = $diag + $(Symbol(:g_, j)) * (k1 * $e1 - k2 * $e2)))
    end
    append!(ex, lu_exprs(Q))
    push!(ex, :(return d))
    return Expr(:block, ex...)
end

# --- host side: quadrature nodes of the cutting window(s) in the tooth phase ψ ∈ [0, 2π) ----
function gl(Q)
    β = [k / sqrt(4k^2 - 1) for k in 1:Q-1]
    J = zeros(Q, Q)
    for k in 1:Q-1
        J[k, k+1] = β[k]; J[k+1, k] = β[k]
    end
    E = eigen_sym(J)
    return E
end
# Golub-Welsch without LinearAlgebra's eigen on the GPU env: tiny Jacobi eigensolver
function eigen_sym(J)
    n = size(J, 1)
    A = copy(J); V = Matrix{Float64}(I_n(n))
    for sweep in 1:100
        off = sum(abs2, A) - sum(abs2, [A[i, i] for i in 1:n])
        off < 1e-30 && break
        for p in 1:n-1, q in p+1:n
            abs(A[p, q]) < 1e-300 && continue
            θ = (A[q, q] - A[p, p]) / (2A[p, q])
            t = sign(θ) / (abs(θ) + sqrt(θ^2 + 1)); θ == 0 && (t = 1.0)
            cs = 1 / sqrt(t^2 + 1); sn = t * cs
            for k in 1:n
                akp, akq = A[k, p], A[k, q]
                A[k, p] = cs * akp - sn * akq; A[k, q] = sn * akp + cs * akq
            end
            for k in 1:n
                apk, aqk = A[p, k], A[q, k]
                A[p, k] = cs * apk - sn * aqk; A[q, k] = sn * apk + cs * aqk
            end
            for k in 1:n
                vkp, vkq = V[k, p], V[k, q]
                V[k, p] = cs * vkp - sn * vkq; V[k, q] = sn * vkp + cs * vkq
            end
        end
    end
    x = [A[i, i] for i in 1:n]
    w = 2 .* V[1, :] .^ 2
    o = sortperm(x)
    return x[o], w[o]
end
I_n(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

function mill2q_consts(; ζ = 0.011, aD = 0.05, kr = 1 / 3, down = true, z = 2, w1 = 0.4478,
                       fn = 922.0, Q = 8)
    φen, φex = down ? (acos(2aD - 1), Float64(π)) : (0.0, acos(1 - 2aD))
    L = z * (φex - φen)                        # window length in the tooth phase
    x, wq = gl(Q)
    ψ = Float64[]; cc = Float64[]
    s = 0.0
    while s < L - 1e-12                        # one piece per tooth period covered by the window
        e = min(L, s + 2π)
        for (xi, wi) in zip(x, wq)
            ψi = (e - s) / 2 * xi + (e + s) / 2
            push!(ψ, mod(ψi, 2π))
            φ = φen + ψi / z
            push!(cc, (e - s) / 2 * wi * sin(φ) * (cos(φ) + kr * sin(φ)) / 2π)
        end
        s = e
    end
    o = sortperm(ψ)
    return (ζ, w1, Float64(z), fn, ψ[o]..., cc[o]...)
end

# ---------------------------------------------------------------------------------------------
# The same Q x Q determinant in O(Q): M is semiseparable.
#   M_ij = δ_ij + Σ_{s=1,2} a_s(i) b_s(j) · (ψ_i ≥ ψ_j ? 1 : W_s),
#   a_1 = k1 u1, a_2 = -k2 u2, b_s = α c_j v_s,  u_s = e^{-a_s ψ/ω}, v_s = 1/u_s,  W_s = e^{-a_s T}.
# Split M = T + A W Bᵀ with T = I + lower∘(A (I - W) Bᵀ). Since k_s (1 - W_s) = T/(r1 - r2) for
# both roots, the diagonal of the lower part cancels: T is UNIT lower triangular (det T = 1,
# the causal free-oscillator response), and by the determinant lemma
#   det M = det(I_2 + W Bᵀ T⁻¹ A),
# where Bᵀ T⁻¹ A (2 x 2) is accumulated by one division-free forward sweep over the nodes with
# four running sums P_st = Σ_{j<i} b_s(j) Y_j[t],  Y_i[t] = a_t(i) - Σ_s a_s(i)(1 - W_s) P_st.
# Identical value to D_mill2q, O(Q) work, O(1) state: Q = 16 ... 32 is cheap.
# ---------------------------------------------------------------------------------------------
@generated function D_mill2s(μ, p, c::NTuple{L, TT}) where {L, TT}
    Q = (L - 4) ÷ 2
    ex = Any[quote
        rk, ap = p
        ζ, w1 = c[1], c[2]
        twoπ = TT(6.283185307179586)
        ω = mill2q_ωp(p, c)
        T = twoπ / ω
        λ = μ * ω
        w = w1 * ap
        sζ = sqrt(1 - ζ * ζ)
        r1 = Complex(-ζ, sζ)
        E = exp(-λ * T)
        α = w * (1 - E)
        er1 = exp(r1 * T)
        W1 = er1 * E
        W2 = conj(er1) * E
        Tr = T * Complex(zero(TT), -1 / (2 * sζ))         # T/(r1 - r2)
        k1 = Tr * cinv(1 - W1)
        k2 = Tr * cinv(1 - W2)
        iω = 1 / ω
        u1 = one(λ); u2 = one(λ); v1 = one(λ); v2 = one(λ)
        P11 = zero(λ); P12 = zero(λ); P21 = zero(λ); P22 = zero(λ)
        ψp = zero(TT)
    end]
    for i in 1:Q
        push!(ex, quote
            δ = (c[$(4 + i)] - ψp) * iω
            ψp = c[$(4 + i)]
            el = exp(-λ * δ)
            er = exp(r1 * δ)
            ρ1 = el * er
            ρ2 = el * conj(er)
            u1 *= ρ1; u2 *= ρ2
            v1 *= cinv(ρ1); v2 *= cinv(ρ2)
            g = α * c[$(4 + Q + i)]
            a1 = k1 * u1; a2 = -k2 * u2
            b1 = g * v1;  b2 = g * v2
            s1 = Tr * u1; s2 = -Tr * u2                    # a_s (1 - W_s)
            Y1 = a1 - (s1 * P11 + s2 * P21)
            Y2 = a2 - (s1 * P12 + s2 * P22)
            P11 += b1 * Y1; P12 += b1 * Y2
            P21 += b2 * Y1; P22 += b2 * Y2
        end)
    end
    push!(ex, :(return (1 + W1 * P11) * (1 + W2 * P22) - W1 * P12 * W2 * P21))
    return Expr(:block, ex...)
end

# ---------------------------------------------------------------------------------------------
# Fastest form: equispaced (midpoint-rule) nodes + rescaled recursion. With constant node
# spacing δ the propagators ρ_s = e^{-a_s δ/ω} are the same for every node (2 exponentials per
# evaluation instead of 2 per node), and the scaled sums P̃_st = u_s(i) P_st obey
#   Y_i[t] = κ_t u_t(i) - T/(r1 - r2) (P̃_1t - P̃_2t),   P̃_st ← ρ_s (P̃_st + α c_i Y_i[t]),
# (κ_1 = k1, κ_2 = -k2) -- no reciprocals inside the loop; ~14 complex multiply-adds per node.
# c = (ζ, w1, z, f_n, δψ, ψ_1, c_1..c_Q); single cutting window per tooth period (z(φex-φen) ≤ 2π).
# ---------------------------------------------------------------------------------------------
@generated function D_mill2m(μ, p, c::NTuple{L, TT}) where {L, TT}
    Q = L - 6
    ex = Any[quote
        rk, ap = p
        ζ, w1 = c[1], c[2]
        twoπ = TT(6.283185307179586)
        ω = mill2q_ωp(p, c)
        T = twoπ / ω
        λ = μ * ω
        w = w1 * ap
        sζ = sqrt(1 - ζ * ζ)
        r1 = Complex(-ζ, sζ)
        E = exp(-λ * T)
        α = w * (1 - E)
        er1 = exp(r1 * T)
        W1 = er1 * E
        W2 = conj(er1) * E
        Tr = T * Complex(zero(TT), -1 / (2 * sζ))
        k1 = Tr * cinv(1 - W1)
        k2 = Tr * cinv(1 - W2)
        δ = c[5] / ω
        x1 = c[6] / ω
        el = exp(-λ * δ); er = exp(r1 * δ)
        ρ1 = el * er; ρ2 = el * conj(er)
        e1 = exp(-λ * x1); f1 = exp(r1 * x1)
        u1 = e1 * f1; u2 = e1 * conj(f1)
        P11 = zero(λ); P12 = zero(λ); P21 = zero(λ); P22 = zero(λ)
    end]
    for i in 1:Q
        push!(ex, quote
            g = α * c[$(6 + i)]
            Y1 = k1 * u1 - Tr * (P11 - P21)
            Y2 = -k2 * u2 - Tr * (P12 - P22)
            P11 += g * Y1; P12 += g * Y2; P21 += g * Y1; P22 += g * Y2
        end)
        if i < Q
            push!(ex, quote
                P11 *= ρ1; P12 *= ρ1; P21 *= ρ2; P22 *= ρ2
                u1 *= ρ1; u2 *= ρ2
            end)
        end
    end
    push!(ex, quote
        w1v = W1 * cinv(u1)                        # W_s / u_s(ψ_Q)
        w2v = W2 * cinv(u2)
        return (1 + w1v * P11) * (1 + w2v * P22) - w1v * P12 * w2v * P21
    end)
    return Expr(:block, ex...)
end

function mill2m_consts(; ζ = 0.011, aD = 0.05, kr = 1 / 3, down = true, z = 2, w1 = 0.4478,
                       fn = 922.0, Q = 16)
    φen, φex = down ? (acos(2aD - 1), Float64(π)) : (0.0, acos(1 - 2aD))
    Lw = z * (φex - φen)
    Lw <= 2π + 1e-12 || error("mill2m: the cutting window exceeds one tooth period (use D_mill2q/D_mill2s)")
    δ = Lw / Q
    cc = [begin
        ψi = (i - 0.5) * δ
        φ = φen + ψi / z
        δ * sin(φ) * (cos(φ) + kr * sin(φ)) / 2π
    end for i in 1:Q]
    return (ζ, w1, Float64(z), fn, δ, δ / 2, cc...)
end
# the same nodes for the general forms (consistency check)
function mill2m_as_q(cm)
    Q = length(cm) - 6
    ψ = [cm[6] + (i - 1) * cm[5] for i in 1:Q]
    return (cm[1:4]..., ψ..., cm[7:end]...)
end

# rolled-loop variant of D_mill2m (one copy of the node body; the constants are read with a
# run-time index) -- smaller code and fewer live values than the fully unrolled form
function D_mill2r(μ, p, c::NTuple{L, TT}) where {L, TT}
    Q = L - 6
    rk, ap = p
    ζ, w1 = c[1], c[2]
    twoπ = TT(6.283185307179586)
    ω = mill2q_ωp(p, c)
    T = twoπ / ω
    λ = μ * ω
    w = w1 * ap
    sζ = sqrt(1 - ζ * ζ)
    r1 = Complex(-ζ, sζ)
    E = exp(-λ * T)
    α = w * (1 - E)
    er1 = exp(r1 * T)
    W1 = er1 * E
    W2 = conj(er1) * E
    Tr = T * Complex(zero(TT), -1 / (2 * sζ))
    k1 = Tr * cinv(1 - W1)
    k2 = Tr * cinv(1 - W2)
    δ = c[5] / ω
    x1 = c[6] / ω
    el = exp(-λ * δ); er = exp(r1 * δ)
    ρ1 = el * er; ρ2 = el * conj(er)
    e1 = exp(-λ * x1); f1 = exp(r1 * x1)
    u1 = e1 * f1; u2 = e1 * conj(f1)
    P11 = zero(λ); P12 = zero(λ); P21 = zero(λ); P22 = zero(λ)
    i = 1
    while true
        g = α * @inbounds(c[6 + i])
        Y1 = k1 * u1 - Tr * (P11 - P21)
        Y2 = -k2 * u2 - Tr * (P12 - P22)
        P11 += g * Y1; P12 += g * Y2; P21 += g * Y1; P22 += g * Y2
        i == Q && break
        P11 *= ρ1; P12 *= ρ1; P21 *= ρ2; P22 *= ρ2
        u1 *= ρ1; u2 *= ρ2
        i += 1
    end
    w1v = W1 * cinv(u1)
    w2v = W2 * cinv(u2)
    return (1 + w1v * P11) * (1 + w2v * P22) - w1v * P12 * w2v * P21
end

# Normalized form of D_mill2r: the same determinant, cheaper per node. Row s of P only ever
# appears divided by u_s (w_s = W_s/u_s at the end), and P_s· and u_s are scaled by the same ρ_s
# -- so with A_st = P_st/u_s the scalings drop out, and with them every λ-dependent exponential
# except E = e^{-λT} = 1/z (F is a function of the Floquet multiplier alone). What is left of the
# node position is the ratio κ = u2/u1 = e^{-2i sζ ψ/ω}: a unit rotation, independent of λ (no
# dual numbers), advanced by ϱ = e^{-2i sζ δ/ω} per node. B = T/(r1 - r2) · A absorbs the
# constant factor of the update:
#   y1 = h (k1 - B11 + κ B21),  y2 = h (B22 - k2 - κ̄ B12),  h = T/(r1 - r2) α c_i,
#   B11 += y1,  B21 += κ̄ y1,  B22 += y2,  B12 += κ y2,  κ ← κ ϱ,
#   F = (1 + v1 B11)(1 + v2 B22) - v1 B12 v2 B21,  v_s = W_s (r1 - r2)/T.
# ~64 real multiplications per node instead of ~140 (dual numbers), 1 dual exponential per
# evaluation instead of 3.
function D_mill2n(μ, p, c::NTuple{L, TT}) where {L, TT}
    Q = L - 6
    rk, ap = p
    ζ, w1 = c[1], c[2]
    twoπ = TT(6.283185307179586)
    ω = mill2q_ωp(p, c)
    T = twoπ / ω
    λ = μ * ω
    w = w1 * ap
    sζ = sqrt(1 - ζ * ζ)
    r1 = Complex(-ζ, sζ)
    E = exp(-λ * T)                                # 1/z
    er1 = exp(r1 * T)
    W1 = er1 * E
    W2 = conj(er1) * E
    Tr = T * Complex(zero(TT), -1 / (2 * sζ))      # T/(r1 - r2)
    k1 = Tr * cinv(1 - W1)
    k2 = Tr * cinv(1 - W2)
    G = Tr * (w * (1 - E))
    s2 = -2 * sζ / ω
    κ = cis(s2 * c[6])                             # u2/u1 at the first node (x1 = c[6]/ω)
    ϱ = cis(s2 * c[5])                             # node to node (δ = c[5]/ω)
    B11 = zero(λ); B12 = zero(λ); B21 = zero(λ); B22 = zero(λ)
    i = 1
    while true
        h = G * @inbounds(c[6 + i])
        y1 = h * (k1 - B11 + κ * B21)
        y2 = h * (B22 - k2 - conj(κ) * B12)
        B11 += y1; B21 += conj(κ) * y1
        B22 += y2; B12 += κ * y2
        i == Q && break
        κ *= ϱ
        i += 1
    end
    iTr = cinv(Tr)
    v1 = W1 * iTr; v2 = W2 * iTr
    return (1 + v1 * B11) * (1 + v2 * B22) - v1 * B12 * v2 * B21
end

# ---------------------------------------------------------------------------------------------
# Fastest form: D_mill2p -- pole-free, with the per-point constants prepared once (found in the
# code review; NyquistGPU.prepare). From D_mill2n:
#  * C21 = κB21, C12 = κ̄B12: |κ| = 1 and only the product B12 B21 enters F, so κ drops out and
#    only the node-to-node rotation ϱ = e^{-2i sζ δψ/ω} is left;
#  * B11, C21 ∝ k1 and B22, C12 ∝ k2, with v_s k_s = W_s/(1 - W_s): multiplying F by
#    (1 - W1)(1 - W2) removes k_s, every division and the poles at the free-oscillator multipliers
#    e^{r_s T}. The factor has no zeros or poles outside |z| < 1 (zero winding on |z| = 1) and is
#    real positive at z = ±1, so the count and the parity rule are unchanged; the values have
#    ~100x less dynamic range (no Float16 overflow at light damping, fewer march steps).
#   y1 = h (β11 + B21),  y2 = h (β22 - B12),  β11 -= y1,  β22 += y2,
#   B21 = ϱ (B21 + y1),  B12 = ϱ̄ (B12 + y2),       h = T w/(r1 - r2) (1 - E) c_i,  E = e^{-2πμ}
#   F̃ = (1 - W1 β11)(1 + W2 β22) - W1 W2 B12 B21,  W1 = e^{r1 T} E,  W2 = conj(e^{r1 T}) E
# Per point (march precision, once): ϱ, e^{r1 T}, T w/(r1 - r2) -- in Float32 also when D runs in
# Float16 (the point and ω are never rounded to Float16: no overflow, no merged chart columns).
# Per evaluation: one exponential, no division; per node ~44 real multiplications (dual numbers).
# ---------------------------------------------------------------------------------------------
function mill2p_prepare(p, c)
    rk, ap = p
    ζ, w1, z, fn, δψ = c[1], c[2], c[3], c[4], c[5]
    TT = typeof(ζ)
    ω = z * rk * (TT(1000) / (60 * fn))
    T = TT(6.283185307179586) / ω
    sζ = sqrt(1 - ζ * ζ)
    er1 = exp(Complex(-ζ, sζ) * T)
    ϱ = cis(-2 * sζ * δψ / ω)
    return (real(ϱ), imag(ϱ), real(er1), imag(er1), zero(TT), -T * w1 * ap / (2 * sζ))
end

function D_mill2p(μ, q, c::NTuple{L, TT}) where {L, TT}
    Q = L - 6
    ϱ = Complex(q[1], q[2])
    er1 = Complex(q[3], q[4])
    Tw = Complex(q[5], q[6])                       # T w/(r1 - r2)
    E = exp(-TT(6.283185307179586) * μ)            # 1/z
    W1 = er1 * E
    W2 = conj(er1) * E
    G = Tw * (1 - E)
    β11 = one(E); β22 = -one(E); B21 = zero(E); B12 = zero(E)
    i = 1
    while true
        h = G * @inbounds(c[6 + i])
        y1 = h * (β11 + B21)
        y2 = h * (β22 - B12)
        β11 -= y1; β22 += y2
        B21 = ϱ * (B21 + y1); B12 = conj(ϱ) * (B12 + y2)
        i == Q && break
        i += 1
    end
    return (1 - W1 * β11) * (1 + W2 * β22) - W1 * W2 * (B12 * B21)
end
@isdefined(NyquistGPU) && (NyquistGPU.prepare(::typeof(D_mill2p), p, c) = mill2p_prepare(p, c))

# ---------------------------------------------------------------------------------------------
# D_mill2g -- Test 2 in the pole-free prepared form of D_mill2p, with GAUSS nodes on the cutting
# window and the DIAGONAL KINK CORRECTION, 4th order in Q instead of 2nd (code review round 2,
# mathematics). Q = 6-8 is as accurate as D_mill2p at Q = 64.
#
# Why: every row of the compressed operator has a slope kink at its own node (the free Green's
# function: S'(0+) - S'(0-) = 2π/ω²). A rule integrates the ramp (ψ - ψ_i)_+ with the error
#   E_i = (Lw - ψ_i)²/2 - Σ_{q>i} W_q (ψ_q - ψ_i)        (λ-free, host constant)
# and adding  ε_i = f(φ_i) E_i / ω²  to the diagonal of the compressed matrix M (I + αM, α = w(1 - E))
# removes that error. For Gauss nodes (no endpoint error) the determinant then converges like Q⁻⁴.
# In the recursion of D_mill2p the diagonal is the pivot of the causal (unit lower triangular) part:
# it becomes d_i = 1 + α ε_i = 1 + u g_i  (u = 1 - E, g_i = w ε_i), i.e. the node weight
#   h_i = G C_i / d_i      (G = T w/(r1 - r2) (1 - E), C_i = W_i f(φ_i)/2π)
# and the determinant is taken WITHOUT the factor Π d_i (that factor only adds an O(1/Q) smooth
# error to |F|; it has no zeros outside |z| < 1). Extra poles: d_i = 0 at z = g_i/(1 + g_i), inside
# the unit circle iff g_i > -1/2 (host check: mill2g_polecheck).
# Gauss nodes are not equidistant: the node-to-node rotation ϱ_i = e^{-2i sζ (ψ_{i+1} - ψ_i)/ω} is
# prepared per point (spacings are symmetric, so only Q÷2 distinct values).
# Cost per node vs D_mill2p: + ~30 real multiplications (dual numbers) and one REAL reciprocal
# (1/|d_i|², no complex division); per evaluation + 3 multiplications.
#
# c = (ζ, w1, z, f_n, ψ_1..ψ_Q, C_1..C_Q, ê_1..ê_Q)   (L = 4 + 3Q),  ê_i = f(φ_i) E_i
# q = prepare = (Re e^{r1 T}, Im e^{r1 T}, -T w/(2 sζ), w/ω², ϱ_1, ..., ϱ_{Q÷2})  (4 + 2(Q÷2) reals)

function mill2g_consts(; ζ = 0.011, aD = 0.05, kr = 1 / 3, down = true, z = 2, w1 = 0.4478,
                       fn = 922.0, Q = 8)
    φen, φex = down ? (acos(2aD - 1), Float64(π)) : (0.0, acos(1 - 2aD))
    Lw = z * (φex - φen)
    Lw <= 2π + 1e-12 || error("mill2g: the cutting window exceeds one tooth period")
    x, wq = gl(Q)                                   # Gauss-Legendre on [-1, 1], ascending
    ψ = Lw / 2 .* (x .+ 1)
    W = Lw / 2 .* wq
    f = [sin(φ) * (cos(φ) + kr * sin(φ)) for φ in φen .+ ψ ./ z]
    C = W .* f ./ 2π
    E = [(Lw - ψ[i])^2 / 2 - sum((W[k] * (ψ[k] - ψ[i]) for k in (i + 1):Q); init = 0.0) for i in 1:Q]
    return (ζ, w1, Float64(z), fn, ψ..., C..., (f .* E)...)
end

function mill2g_prepare(p, c::NTuple{L, TT}) where {L, TT}
    Q = (L - 4) ÷ 3
    H = Q ÷ 2
    rk, ap = p
    ζ, w1, z, fn = c[1], c[2], c[3], c[4]
    ω = z * rk * (TT(1000) / (60 * fn))
    T = TT(6.283185307179586) / ω
    sζ = sqrt(1 - ζ * ζ)
    er1 = exp(Complex(-ζ, sζ) * T)
    s2 = -2 * sζ / ω
    return ntuple(Val(4 + 2H)) do k
        if k == 1
            real(er1)
        elseif k == 2
            imag(er1)
        elseif k == 3
            -T * w1 * ap / (2 * sζ)                 # T w/(r1 - r2) = i·(this)
        elseif k == 4
            w1 * ap / (ω * ω)                       # g_i = (w/ω²) ê_i
        else
            j = (k - 3) ÷ 2                         # spacing j = ψ_{j+1} - ψ_j (= spacing Q - j)
            sn, cs = sincos(s2 * (c[4 + j + 1] - c[4 + j]))
            isodd(k) ? cs : sn
        end
    end
end

function D_mill2g(μ, q, c::NTuple{L, TT}) where {L, TT}
    Q = (L - 4) ÷ 3
    er1 = Complex(q[1], q[2])
    Tw = Complex(zero(TT), q[3])
    κω = q[4]
    E = exp(-TT(6.283185307179586) * μ)            # 1/z
    W1 = er1 * E
    W2 = conj(er1) * E
    u = 1 - E
    G = Tw * u
    ur2 = 2 * real(u)
    uu = real(u) * real(u) + imag(u) * imag(u)
    β11 = one(E); β22 = -one(E); B21 = zero(E); B12 = zero(E)
    i = 1
    while true
        g = κω * @inbounds(c[4 + 2Q + i])
        d = 1 + u * g                               # pivot 1 + α ε_i
        s = @inbounds(c[4 + Q + i]) * inv(1 + g * (ur2 + g * uu))   # C_i/|d|²
        h = (G * conj(d)) * s                       # G C_i / d
        y1 = h * (β11 + B21)
        y2 = h * (β22 - B12)
        β11 -= y1; β22 += y2
        B21 += y1; B12 += y2
        i == Q && break
        j = min(i, Q - i)                           # Gauss spacings are symmetric
        ϱ = Complex(@inbounds(q[3 + 2j]), @inbounds(q[4 + 2j]))
        B21 = ϱ * B21; B12 = conj(ϱ) * B12
        i += 1
    end
    return (1 - W1 * β11) * (1 + W2 * β22) - W1 * W2 * (B12 * B21)
end
@isdefined(NyquistGPU) && (NyquistGPU.prepare(::typeof(D_mill2g), p, c) = mill2g_prepare(p, c))

"host check: the extra poles z = g_i/(1 + g_i) lie inside |z| < 1 iff min_i g_i > -1/2 (worst point of a chart)"
function mill2g_polecheck(c, rks, aps)
    Q = (length(c) - 4) ÷ 3
    ê = c[4 + 2Q + 1:4 + 3Q]
    gmin = Inf; gmax = -Inf
    for rk in rks, ap in aps
        ω = c[3] * rk * 1000 / (60 * c[4])
        κ = c[2] * ap / ω^2
        gmin = min(gmin, κ * minimum(ê)); gmax = max(gmax, κ * maximum(ê))
    end
    return (gmin = gmin, gmax = gmax, ok = gmin > -0.5, zmax = maximum(abs(g / (1 + g)) for g in (gmin, gmax)))
end
