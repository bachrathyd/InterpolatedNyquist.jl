# Test 3 -- two flutes with different helix angles, delays distributed over the axial depth -- in
# the compressed (Nystrom) form of the infinite Hill determinant, counted along the unit circle of
# the Floquet multiplier z = e^{λT_s} (T_s: spindle period). Needs hill/gpu_fast.jl (helpers) and
# NyquistGPU (prepare hook, workspace). Derived in the code review (machining and linear-algebra
# reviewers; cross-checked by the mathematics reviewer).
#
# Nodes n = (tooth j, axial Gauss node ζ_l on [0, a_p], window Gauss node φ_i) in the WORKPIECE
# frame: angle ψ_n = φ_i - θ_j + ζ_l b_j (b_j = tanβ_j/R), weight c_n = w1 Wζ_l Wφ_i f(φ_i)/Ω.
# The delayed time of node n, t_n - τ_j(ζ_l), is exactly the time of the SAME material node on the
# previous tooth, π(n) = (j-1, l, i); one revolution earlier when the arc from π(n) to n crosses
# the time origin ("wrapped"): x(t_n - τ_n) = z^{-1} x(t_π(n)). With the periodic Green's function
# S_z of the structure (all harmonics in closed form) and C = diag(c):
#   D(z) = det(I + (I - P(z)) S_z C),   P(z) = P0 + z^{-1} P1   (rows: previous-tooth partners)
# D depends on λ only through z, D(z̄) = conj D(z), D(∞) = 1: the count is -Φ/π over μ ∈ [0, 1/2].
#
# Reduction, once per point (λ-free): S_z = G + a Γ(z) bᵀ with G the causal free response and
# Γ = diag(ρ_k q_k/(1 - q_k)), q_k = e^{r_k T_s}/z; A0 = I + (I - P0) G C is unit lower triangular in
# time order, and
#   D(z) = det(I_{nw+2} + R(z) A0⁻¹ L(z)),
# nw = number of wrapped nodes (the time origin is chosen to minimize it: about one tooth's nodes).
# A0⁻¹ is applied by forward sweeps with two running sums (G is semiseparable): O(N (nw + 4)) per
# point, no N x N matrix. Per evaluation: one (nw + 2) x (nw + 2) LU with dual numbers. The two rows
# with Γ are scaled by (1 - q_k): the factor has no zeros or poles outside |z| < 1 and is positive at
# z = ±1 (same count), and removes the poles at the free-oscillator multipliers (light damping).
#
# Workspace (per lane, complex dual entries; real data in the values, zero partials):
#   node fields (NF per node), running sums (2 per sorted position), Xa (LD x LD), Xb (LD x 2),
#   the LU matrix (LD x LD). LD = maximal nw + 2; points with more wrapped nodes return NaN (flagged).

# c = (ζ, w1, f_n, R, tanβ1, tanβ2, Q, n_s, LD, φ_1..φ_Q, Wf_1..Wf_Q, x_1..x_ns, wz_1..wz_ns)
#   Wf_i = Wφ_i f(φ_i) (Gauss on the cutting window), x_l, wz_l: Gauss on [0, 1] (ζ = a_p x_l)
function mill3c_consts(; ζ = 0.011, aD = 0.05, kr = 1 / 3, down = true, β1 = 30.0, β2 = 45.0, R = 8.0,
                       w1 = 0.4478, fn = 922.0, Q = 8, ns = 4, LD = ns * Q + 2 + 8)
    φen, φex = down ? (acos(2aD - 1), Float64(π)) : (0.0, acos(1 - 2aD))
    xg, wg = gl_nodes_weights(Q)
    φ = (φex - φen) / 2 .* xg .+ (φex + φen) / 2
    Wf = (φex - φen) / 2 .* wg .* sin.(φ) .* (cos.(φ) .+ kr .* sin.(φ))
    xz, wz = gl_nodes_weights(ns)
    return (ζ, w1, fn, R, tand(β1), tand(β2), Float64(Q), Float64(ns), Float64(LD),
            φ..., Wf..., ((xz .+ 1) ./ 2)..., (wz ./ 2)...)
end
gl_nodes_weights(n) = gl(n)                     # Gauss-Legendre nodes, weights on [-1, 1]

const NF3 = 8                                   # node fields
mill3c_wslen(Q, ns, LD) = 2Q * ns * (NF3 + 2) + 2LD * LD + 2LD
mill3c_wslen(c::Tuple) = mill3c_wslen(Int(c[7]), Int(c[8]), Int(c[9]))

@inline _cv(z) = Complex(_val(real(z)), _val(imag(z)))      # value of a stored entry

# node fields: 1 (t_n, c_n)   2 a1 = e^{r1 t}   3 b1c = e^{-r1 t} c   4 (partner, wrapped)
#              5 (ψ_n, Ωτ_n)  6 (perm[s], rank[n])   7 y (current column)   8 (W_i, wi[n])
@inline nf(n, f) = (n - 1) * NF3 + f

# prepare: nodes, time origin, order, the reduced matrices -- all λ-free, march precision
function mill3c_prepare(p, cw)
    c, A = cw
    rk, ap = p
    TT = typeof(rk)
    ζ, w1, fn, R, tb1, tb2 = c[1], c[2], c[3], c[4], c[5], c[6]
    Q = unsafe_trunc(Int, c[7]); ns = unsafe_trunc(Int, c[8]); LD = unsafe_trunc(Int, c[9])
    N = 2 * ns * Q
    SOFF = N * NF3
    XOFF = SOFF + 2N
    XBOFF = XOFF + LD * LD
    twoπ = TT(6.283185307179586)
    πT = TT(3.141592653589793)
    Ω = rk * (TT(1000) / (60 * fn))
    Ts = twoπ / Ω
    sζ = sqrt(1 - ζ * ζ)
    r1 = Complex(-ζ, sζ)
    ρ1 = Complex(zero(TT), -1 / (2 * sζ))
    b1 = tb1 / R; b2 = tb2 / R
    L = max(ap, TT(1e-6))
    # 1. nodes: angle, arc length to the partner, weight
    for j in 1:2, l in 1:ns, i in 1:Q
        n = ((j - 1) * ns + (l - 1)) * Q + i
        φ = c[9 + i]; Wf = c[9 + Q + i]; x = c[9 + 2Q + l]; wz = c[9 + 2Q + ns + l]
        ζl = L * x
        bj = j == 1 ? b1 : b2
        bp = j == 1 ? b2 : b1
        θj = j == 1 ? zero(TT) : -πT
        ψ = mod(φ - θj + ζl * bj, twoπ)
        Ωτ = πT + ζl * (bj - bp)                   # uniform pitch π
        A[nf(n, 1)] = Complex(zero(TT), w1 * (L * wz) * Wf / Ω)       # (t later, c_n)
        A[nf(n, 5)] = Complex(ψ, Ωτ)
        jp = 3 - j
        A[nf(n, 4)] = Complex(TT(((jp - 1) * ns + (l - 1)) * Q + i), zero(TT))
    end
    # 2. time origin at the node that minimizes the number of wrapped nodes (arcs through it)
    best = N + 1; kb = 1
    for k in 1:N
        ψk = real(_cv(A[nf(k, 5)]))
        cnt = 0
        for n in 1:N
            a = _cv(A[nf(n, 5)])
            cnt += Int(mod(real(a) - ψk, twoπ) < imag(a))
        end
        if cnt < best
            best = cnt; kb = k
        end
    end
    nw = best
    ψ0 = real(_cv(A[nf(kb, 5)]))
    # 3. times, wrapped flags, exponentials
    wcount = 0
    for n in 1:N
        a = _cv(A[nf(n, 5)])
        d = mod(real(a) - ψ0, twoπ)
        t = d / Ω
        wr = d < imag(a)
        cn = imag(_cv(A[nf(n, 1)]))
        A[nf(n, 1)] = Complex(t, cn)
        e = exp(r1 * t)
        A[nf(n, 2)] = e
        A[nf(n, 3)] = cn * exp(-r1 * t)              # no complex division (Base widens it to Float64)
        pr = real(_cv(A[nf(n, 4)]))
        A[nf(n, 4)] = Complex(pr, wr ? one(TT) : zero(TT))
        if wr
            wcount += 1
            A[nf(wcount, 8)] = Complex(TT(n), imag(_cv(A[nf(wcount, 8)])))   # W_i = n
            A[nf(n, 8)] = Complex(real(_cv(A[nf(n, 8)])), TT(wcount))       # wi[n]
        end
        A[nf(n, 6)] = Complex(TT(n), zero(TT))                              # perm = identity
    end
    # 4. sort by time (insertion sort on perm), then rank
    for s in 2:N
        m = unsafe_trunc(Int, real(_cv(A[nf(s, 6)])))
        tm = real(_cv(A[nf(m, 1)]))
        r = s - 1
        while r >= 1
            o = unsafe_trunc(Int, real(_cv(A[nf(r, 6)])))
            real(_cv(A[nf(o, 1)])) <= tm && break
            A[nf(r + 1, 6)] = Complex(TT(o), imag(_cv(A[nf(r + 1, 6)])))
            r -= 1
        end
        A[nf(r + 1, 6)] = Complex(TT(m), imag(_cv(A[nf(r + 1, 6)])))
    end
    for s in 1:N
        m = unsafe_trunc(Int, real(_cv(A[nf(s, 6)])))
        A[nf(m, 6)] = Complex(real(_cv(A[nf(m, 6)])), TT(s))
    end
    # 5. columns of A0⁻¹ [E_W, u0, u1] by forward sweeps; the reduced matrices
    n2 = nw + 2
    if n2 <= LD
        for col in 1:(nw + 4)
            S1 = zero(r1); S2 = zero(r1)
            for s in 1:N
                n = unsafe_trunc(Int, real(_cv(A[nf(s, 6)])))
                A[SOFF + 2s - 1] = S1
                A[SOFF + 2s] = S2
                a1 = _cv(A[nf(n, 2)])
                pw = _cv(A[nf(n, 4)])
                pn = unsafe_trunc(Int, real(pw))
                wr = imag(pw) > 0
                ap1 = _cv(A[nf(pn, 2)])
                # right-hand side of this column
                rhs = zero(r1)
                if col <= nw
                    rhs = unsafe_trunc(Int, imag(_cv(A[nf(n, 8)]))) == col && wr ? one(r1) : zero(r1)
                elseif col <= nw + 2                    # u0 = (I - P0) a
                    ak = col == nw + 1 ? a1 : conj(a1)
                    pk = col == nw + 1 ? ap1 : conj(ap1)
                    rhs = wr ? ak : ak - pk
                else                                    # u1 = P1 a
                    pk = col == nw + 3 ? ap1 : conj(ap1)
                    rhs = wr ? pk : zero(r1)
                end
                y = rhs - (ρ1 * a1 * S1 + conj(ρ1) * conj(a1) * S2)
                if !wr                                  # the partner is earlier in time
                    sp = unsafe_trunc(Int, imag(_cv(A[nf(pn, 6)])))
                    y += ρ1 * ap1 * _cv(A[SOFF + 2sp - 1]) + conj(ρ1) * conj(ap1) * _cv(A[SOFF + 2sp])
                end
                bc = _cv(A[nf(n, 3)])
                S1 += bc * y
                S2 += conj(bc) * y
            end
            # K rows (wrapped nodes: their partner, later in time) and the totals
            jc = col <= nw ? col : (col <= nw + 2 ? col : col - 2)   # column of Xa (E_W, u0) / Xb (u1)
            for i in 1:nw
                wn = unsafe_trunc(Int, real(_cv(A[nf(i, 8)])))
                pn = unsafe_trunc(Int, real(_cv(A[nf(wn, 4)])))
                sp = unsafe_trunc(Int, imag(_cv(A[nf(pn, 6)])))
                ap1 = _cv(A[nf(pn, 2)])
                v = ρ1 * ap1 * _cv(A[SOFF + 2sp - 1]) + conj(ρ1) * conj(ap1) * _cv(A[SOFF + 2sp])
                if col <= nw + 2
                    A[XOFF + i + (jc - 1) * LD] = v
                else
                    A[XBOFF + i + (col - nw - 3) * LD] = v
                end
            end
            if col <= nw + 2
                A[XOFF + nw + 1 + (jc - 1) * LD] = S1
                A[XOFF + nw + 2 + (jc - 1) * LD] = S2
            else
                A[XBOFF + nw + 1 + (col - nw - 3) * LD] = S1
                A[XBOFF + nw + 2 + (col - nw - 3) * LD] = S2
            end
        end
    end
    er = exp(r1 * Ts)
    return (Ts, TT(nw), real(er), imag(er))
end

# LU determinant with leading dimension LD at offset off (dual numbers, partial pivoting), in the
# arithmetic of E (the evaluation type -- the workspace stores the march precision)
@inline function lu_det_ld!(A, off, n, LD, ::Type{E}) where {E}
    d = one(E)
    for k in 1:n
        pk = k
        best = _mag2(E(A[off + k + (k - 1) * LD]))
        for i in (k + 1):n
            v = _mag2(E(A[off + i + (k - 1) * LD]))
            if v > best
                best = v
                pk = i
            end
        end
        if pk != k
            for j in k:n
                a = A[off + k + (j - 1) * LD]
                A[off + k + (j - 1) * LD] = A[off + pk + (j - 1) * LD]
                A[off + pk + (j - 1) * LD] = a
            end
            d = -d
        end
        piv = E(A[off + k + (k - 1) * LD])
        d *= piv
        ip = cinv(piv)
        for i in (k + 1):n
            f = E(A[off + i + (k - 1) * LD]) * ip
            for j in (k + 1):n
                A[off + i + (j - 1) * LD] = E(A[off + i + (j - 1) * LD]) - f * E(A[off + k + (j - 1) * LD])
            end
        end
    end
    return d
end

# the characteristic function: μ = λ/Ω on the half circle (z = e^{2πμ}), q from mill3c_prepare
function D_mill3c(μ, q, cw)
    c, A = cw
    TT = typeof(q[1])
    ζ = c[1]
    Q = unsafe_trunc(Int, c[7]); ns = unsafe_trunc(Int, c[8]); LD = unsafe_trunc(Int, c[9])
    N = 2 * ns * Q
    XOFF = N * NF3 + 2N
    XBOFF = XOFF + LD * LD
    AOFF = XBOFF + 2LD
    nw = unsafe_trunc(Int, q[2])
    n2 = nw + 2
    w = exp(-TT(6.283185307179586) * μ)        # 1/z
    n2 > LD && return w * TT(NaN)              # more wrapped nodes than the workspace holds
    sζ = sqrt(1 - ζ * ζ)
    ρ1 = Complex(zero(TT), -1 / (2 * sζ))
    er = Complex(q[3], q[4])
    q1 = er * w
    q2 = conj(er) * w
    for j in 1:n2, i in 1:n2
        x = Complex{TT}(_cv(A[XOFF + i + (j - 1) * LD])) + zero(w)
        if j > nw
            x -= w * Complex{TT}(_cv(A[XBOFF + i + (j - nw - 1) * LD]))
        end
        v = if i <= nw
            -w * x
        elseif i == nw + 1                     # row scaled by (1 - q1): pole-free
            ρ1 * q1 * x
        else
            conj(ρ1) * q2 * x
        end
        if i == j
            v += i <= nw ? one(w) : (i == nw + 1 ? 1 - q1 : 1 - q2)
        end
        A[AOFF + i + (j - 1) * LD] = v
    end
    return lu_det_ld!(A, AOFF, n2, LD, typeof(w))
end

if @isdefined(NyquistGPU)
    NyquistGPU.prepare(::typeof(D_mill3c), p, c) = mill3c_prepare(p, c)
end
mill3c_ωp(p, c) = p[1] * 1000 / (60 * c[3])         # spindle frequency / f_n (σ scaling)
