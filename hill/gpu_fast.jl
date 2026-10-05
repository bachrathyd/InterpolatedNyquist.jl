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
        sc = ifelse(sc > 0, sc, one(sc))
        bs = b / sc
        return conj(bs) * inv(real(bs) * real(bs) + imag(bs) * imag(bs)) / sc
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
