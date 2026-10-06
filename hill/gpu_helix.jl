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
#   The layout is that of the MAXIMAL axial count NS = c[8] (fixed offsets for every point), so that
#   D_mill3c / D_mill3h read the same offsets whatever n_s the point uses.
#
# v2 (math review, round 2b) -- two changes, both in prepare only (no cost per evaluation):
# * Diagonal kink correction. Every row has a slope kink of the causal Green's function G at its own
#   node (G(0) = 0, G'(0+) = 1); the window rule integrates the ramp (φ - φ_i)_+ with the error
#   E^φ_i = (φ_ex - φ_i)²/2 - Σ_{k>i} Wφ_k (φ_k - φ_i). Adding ε_n = w1 a_p wz_l f(φ_i) E^φ_i / Ω² to the
#   diagonal of S_z C, i.e. using (I - P(z))(S_z C + diag ε), removes that error. The correction is
#   λ-free and real: A0 = I + (I - P0)(G C + diag ε) is lower triangular with pivots 1 + ε_n (forward
#   sweep: y_n = (… + ε_π(n) y_π(n)) / (1 + ε_n) for a non-wrapped row), and a wrapped row of R picks up
#   ε_π(n) y_π(n). The off-node (oblique) kinks on the other slices are not corrected.
# * Adaptive axial count per point: n_s = clamp(ceil(2 + κ_ns a_p b_max/(2πΩ)), 2, NS)
#   (b_max = max tanβ/R; a_p b_max/(2πΩ) = the helix lag across the depth in natural periods);
#   Gauss tables for n_s = 2..NS in the constants. ns_fixed > 0 gives a fixed n_s (as v1).

# c = (ζ, w1, f_n, R, tanβ1, tanβ2, Q, NS, LD, ns_fixed, kink, κ_ns,
#      φ_1..φ_Q, Wf_1..Wf_Q, Ef_1..Ef_Q, [x_1..x_n, wz_1..wz_n for n = 2..NS])
#   Wf_i = Wφ_i f(φ_i) (Gauss on the cutting window), Ef_i = f(φ_i) E^φ_i (kink), x_l, wz_l: Gauss on
#   [0, 1] (ζ = a_p x_l); the table of n starts after c[mill3c_tab(Q, n)].
function mill3c_consts(; ζ = 0.011, aD = 0.05, kr = 1 / 3, down = true, β1 = 30.0, β2 = 45.0, R = 8.0,
                       w1 = 0.4478, fn = 922.0, Q = 8, ns = 0, nsmax = ns > 0 ? ns : 6, kink = true,
                       κns = 1.6, LD = nsmax * Q + 2 + 8)
    ns <= nsmax || error("mill3c_consts: ns > nsmax")
    φen, φex = down ? (acos(2aD - 1), Float64(π)) : (0.0, acos(1 - 2aD))
    xg, wg = gl_nodes_weights(Q)
    φ = (φex - φen) / 2 .* xg .+ (φex + φen) / 2
    Wφ = (φex - φen) / 2 .* wg
    f = sin.(φ) .* (cos.(φ) .+ kr .* sin.(φ))
    Wf = Wφ .* f
    Eφ = [(φex - φ[i])^2 / 2 - sum((Wφ[k] * (φ[k] - φ[i]) for k in (i + 1):Q); init = 0.0) for i in 1:Q]
    tab = Float64[]
    for n in 2:nsmax
        xz, wz = gl_nodes_weights(n)
        append!(tab, (xz .+ 1) ./ 2); append!(tab, wz ./ 2)
    end
    return (ζ, w1, fn, R, tand(β1), tand(β2), Float64(Q), Float64(nsmax), Float64(LD), Float64(ns),
            kink ? 1.0 : 0.0, κns, φ..., Wf..., (f .* Eφ)..., tab...)
end
gl_nodes_weights(n) = gl(n)                     # Gauss-Legendre nodes, weights on [-1, 1]
@inline mill3c_tab(Q, n) = 12 + 3Q + n * n - n - 2   # c[mill3c_tab(Q, n) + l] = x_l, + n + l: wz_l

"axial count of a point (rule of the machining review; NS = c[8] the maximum)"
@inline function mill3c_ns(rk, ap, c)
    TT = typeof(rk)
    NS = unsafe_trunc(Int, c[8]); nsf = unsafe_trunc(Int, c[10])
    nsf > 0 && return nsf
    Ω = rk * (TT(1000) / (60 * c[3]))
    bmax = max(abs(c[5]), abs(c[6])) / c[4]
    x = 2 + c[12] * max(ap, zero(TT)) * bmax / (TT(6.283185307179586) * Ω)
    return clamp(unsafe_trunc(Int, ceil(x)), 2, NS)
end

const NF3 = 8                                   # node fields
mill3c_wslen(Q, ns, LD) = 2Q * ns * (NF3 + 2) + 2LD * LD + 2LD
mill3c_wslen(c::Tuple) = mill3c_wslen(Int(c[7]), Int(c[8]), Int(c[9]))

@inline _cv(z) = Complex(_val(real(z)), _val(imag(z)))      # value of a stored entry

# node fields: 1 (t_n, c_n)   2 a1 = e^{r1 t}   3 b1c = e^{-r1 t} c   4 (partner, wrapped)
#              5 (ψ_n, Ωτ_n), after step 3: (ε_n, 1/(1 + ε_n))   6 (perm[s], rank[n])
#              7 y (current column)   8 (W_i, wi[n])
@inline nf(n, f) = (n - 1) * NF3 + f

# prepare: nodes, time origin, order, the reduced matrices -- all λ-free, march precision
function mill3c_prepare(p, cw)
    c, A = cw
    rk, ap = p
    TT = typeof(rk)
    ζ, w1, fn, R, tb1, tb2 = c[1], c[2], c[3], c[4], c[5], c[6]
    Q = unsafe_trunc(Int, c[7]); NS = unsafe_trunc(Int, c[8]); LD = unsafe_trunc(Int, c[9])
    kink = c[11] > 0
    ns = mill3c_ns(rk, ap, c)                   # axial nodes of this point (≤ NS)
    T0 = mill3c_tab(Q, ns)
    N = 2 * ns * Q
    NM = 2 * NS * Q                             # the layout is that of NS (fixed offsets)
    SOFF = NM * NF3
    XOFF = SOFF + 2NM
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
        φ = c[12 + i]; Wf = c[12 + Q + i]; x = c[T0 + l]; wz = c[T0 + ns + l]
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
            d = real(a) - ψk                       # both in [0, 2π): no mod (review: 4096 fmod per point)
            d += d < 0 ? twoπ : zero(TT)
            cnt += Int(d < imag(a))
        end
        if cnt < best
            best = cnt; kb = k
        end
    end
    ψ0 = real(_cv(A[nf(kb, 5)]))
    # 3. times and exponentials (the wrapped flags follow from the time order, after step 4)
    for n in 1:N
        a = _cv(A[nf(n, 5)])
        d = real(a) - ψ0
        d += d < 0 ? twoπ : zero(TT)
        t = d / Ω
        cn = imag(_cv(A[nf(n, 1)]))
        A[nf(n, 1)] = Complex(t, cn)
        e = exp(r1 * t)
        A[nf(n, 2)] = e
        A[nf(n, 3)] = cn * exp(-r1 * t)              # no complex division (Base widens it to Float64)
        A[nf(n, 6)] = Complex(TT(n), zero(TT))                              # perm = identity
        # kink correction of the node (field 5 is free from here on)
        ε = zero(TT)
        if kink
            i = (n - 1) % Q + 1
            l = ((n - 1) ÷ Q) % ns + 1
            ε = w1 * (L * c[T0 + ns + l]) * c[12 + 2Q + i] / (Ω * Ω)
        end
        A[nf(n, 5)] = Complex(ε, 1 / (1 + ε))
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
    # wrapped flags from the time ORDER: the sweep needs "not wrapped <=> the partner is earlier".
    # At a tie (the partner at the time origin) the angle test above and the sort can disagree by
    # rounding -- then a sweep read a stale running sum (found in the GPU review). Both readings
    # are the same delayed time, so the order decides; nw and the wrapped numbering are recounted.
    nw = 0
    for n in 1:N
        pw = _cv(A[nf(n, 4)])
        pn = unsafe_trunc(Int, real(pw))
        wr = imag(_cv(A[nf(pn, 6)])) > imag(_cv(A[nf(n, 6)]))
        A[nf(n, 4)] = Complex(real(pw), wr ? one(TT) : zero(TT))
        if wr
            nw += 1
            A[nf(nw, 8)] = Complex(TT(n), imag(_cv(A[nf(nw, 8)])))                # W_i = n
            A[nf(n, 8)] = Complex(real(_cv(A[nf(n, 8)])), TT(nw))                  # wi[n]
        end
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
                    kink && (y += real(_cv(A[nf(pn, 5)])) * _cv(A[nf(pn, 7)]))   # + ε_π y_π
                end
                if kink                                 # pivot 1 + ε_n
                    y *= imag(_cv(A[nf(n, 5)]))
                    A[nf(n, 7)] = y
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
                kink && (v += real(_cv(A[nf(pn, 5)])) * _cv(A[nf(pn, 7)]))      # + ε_π y_π
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
    return (Ts, TT(nw), real(er), imag(er), TT(ns))   # q[5] = n_s (diagnostics; the D's read q[1:4])
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

# =============================================================================================
# Hessenberg variant D_mill3h (code review, round 2): the same determinant as D_mill3c, with the
# λ-free block reduced once per point -- per evaluation O(nw²) instead of O(nw³).
#
# D_mill3c evaluates, per μ, the (nw+2) x (nw+2) matrix (w = 1/z, q_k = e^{r_k T_s} w, g_k = ρ_k q_k)
#   M(w) = [ I - w X11              -w (X12a - w X12b)                      ]
#          [ diag(g) X21            diag(1 - q) + diag(g) (X22a - w X22b)   ]
# by a dense dual LU. X11 (nw x nw) is λ-free, so once per point (prepare, march precision):
#   X11 = U H Uᴴ (Householder, H upper Hessenberg),  Y12a = Uᴴ X12a, Y12b = Uᴴ X12b, Y21 = X21 U,
# and det M = det of the same matrix with (H, Y12a, Y12b, Y21) in place of (X11, X12a, X12b, X21)
# (a unitary similarity of the leading block). Per evaluation the bordered Hessenberg matrix
#   [ I - w H      -w (Y12a - w Y12b) ]
#   [ diag(g) Y21   ...               ]
# is factored by Gaussian elimination with PARTIAL PIVOTING over the only rows that can be nonzero in
# column k: three active dense rows (initially row 1 and the two border rows) and the new Hessenberg
# row k+1 -- i.e. exactly the pivot choices of the dense LU, without its operations on zeros.
# The pivot row is consumed; the new row is stored into the pivot's buffer. 3 row buffers, O(n²).
#
# Workspace: as D_mill3c, but the (LD x LD) LU area is replaced by 3 row buffers of LD entries
# (the Householder vectors in prepare use the same area).

mill3h_wslen(Q, ns, LD) = 2Q * ns * (NF3 + 2) + LD * LD + 2LD + 3LD
mill3h_wslen(c::Tuple) = mill3h_wslen(Int(c[7]), Int(c[8]), Int(c[9]))

# prepare: mill3c's reduction, then the Hessenberg reduction of X11 in place (march precision)
function mill3h_prepare(p, cw)
    q = mill3c_prepare(p, cw)
    c, A = cw
    TT = typeof(q[1])
    Q = unsafe_trunc(Int, c[7]); ns = unsafe_trunc(Int, c[8]); LD = unsafe_trunc(Int, c[9])
    N = 2 * ns * Q
    XOFF = N * NF3 + 2N
    XBOFF = XOFF + LD * LD
    VOFF = XBOFF + 2LD                         # Householder vector (the row-buffer area)
    nw = unsafe_trunc(Int, q[2])
    nw + 2 <= LD || return q
    for k in 1:(nw - 2)
        m = nw - k
        xm = zero(TT)                          # scale first (entries ~ a_p: squares underflow at a_p -> 0)
        for i in 1:m
            x = _cv(A[XOFF + (k + i) + (k - 1) * LD])
            xm = max(xm, abs(real(x)), abs(imag(x)))
        end
        xm > 0 || continue
        sc = 1 / xm
        nx2 = zero(TT)
        for i in 1:m                           # v = x / xm,  x = H[k+1:nw, k]
            x = _cv(A[XOFF + (k + i) + (k - 1) * LD]) * sc
            nx2 += real(x) * real(x) + imag(x) * imag(x)
            A[VOFF + i] = x
        end
        nx = sqrt(nx2)
        x1 = _cv(A[VOFF + 1])
        a1 = sqrt(real(x1) * real(x1) + imag(x1) * imag(x1))
        ph = a1 > 0 ? x1 * (1 / a1) : one(x1)  # x1/|x1| without a complex division
        A[VOFF + 1] = x1 + ph * nx             # v1 = x1 - α, α = -ph |x|  (no cancellation)
        τ = 1 / (nx * (nx + a1))               # 2 / (vᴴv), scale-free in v
        # from the left: column k -> α e1; columns k+1..nw of H, Y12a (Xa cols nw+1:nw+2), Y12b (Xb)
        A[XOFF + (k + 1) + (k - 1) * LD] = -ph * (nx * xm)
        for i in 2:m
            A[XOFF + (k + i) + (k - 1) * LD] = zero(x1)
        end
        for j in (k + 1):(nw + 4)
            base = j <= nw + 2 ? XOFF + (j - 1) * LD : XBOFF + (j - nw - 3) * LD
            s = zero(x1)
            for i in 1:m
                s += conj(_cv(A[VOFF + i])) * _cv(A[base + k + i])
            end
            s *= τ
            for i in 1:m
                A[base + k + i] = _cv(A[base + k + i]) - s * _cv(A[VOFF + i])
            end
        end
        # from the right: columns k+1..nw of rows 1..nw (H) and nw+1..nw+2 (Y21)
        for r in 1:(nw + 2)
            s = zero(x1)
            for i in 1:m
                s += _cv(A[XOFF + r + (k + i - 1) * LD]) * _cv(A[VOFF + i])
            end
            s *= τ
            for i in 1:m
                A[XOFF + r + (k + i - 1) * LD] = _cv(A[XOFF + r + (k + i - 1) * LD]) - s * conj(_cv(A[VOFF + i]))
            end
        end
    end
    return q
end

# pivot magnitude (value part; Float32 at least, so that Float16 entries > 256 do not overflow)
@inline _m2p(z, ::Type{P}) where {P} = (x = P(_val(real(z))); y = P(_val(imag(z))); x * x + y * y)

function D_mill3h(μ, q, cw)
    c, A = cw
    TT = typeof(q[1])
    PT = TT === Float16 ? Float32 : TT
    ζ = c[1]
    Q = unsafe_trunc(Int, c[7]); ns = unsafe_trunc(Int, c[8]); LD = unsafe_trunc(Int, c[9])
    N = 2 * ns * Q
    XOFF = N * NF3 + 2N
    XBOFF = XOFF + LD * LD
    BOFF = XBOFF + 2LD                         # 3 row buffers of LD entries
    nw = unsafe_trunc(Int, q[2])
    n2 = nw + 2
    w = exp(-TT(6.283185307179586) * μ)        # 1/z
    n2 > LD && return w * TT(NaN)
    E = typeof(w)
    sζ = sqrt(1 - ζ * ζ)
    ρ1 = Complex(zero(TT), -1 / (2 * sζ))
    er = Complex(q[3], q[4])
    q1 = er * w
    q2 = conj(er) * w
    g1 = ρ1 * q1
    g2 = conj(ρ1) * q2
    # row i (1..nw) of [I - wH | -w(Y12a - w Y12b)] at column j
    @inline hrow(i, j) = j <= nw ? (i == j ? one(E) : zero(E)) - w * Complex{TT}(_cv(A[XOFF + i + (j - 1) * LD])) :
        -w * (Complex{TT}(_cv(A[XOFF + i + (j - 1) * LD])) - w * Complex{TT}(_cv(A[XBOFF + i + (j - nw - 1) * LD])))
    # border row b (1, 2) at column j
    @inline brow(b, j) = begin
        g = b == 1 ? g1 : g2
        if j <= nw
            g * Complex{TT}(_cv(A[XOFF + nw + b + (j - 1) * LD]))
        else
            v = g * (Complex{TT}(_cv(A[XOFF + nw + b + (j - 1) * LD])) - w * Complex{TT}(_cv(A[XBOFF + nw + b + (j - nw - 1) * LD])))
            j - nw == b ? v + (b == 1 ? 1 - q1 : 1 - q2) : v
        end
    end
    # active rows: slots 1..3 (buffers), original row indices
    for j in 1:n2
        nw >= 1 && (A[BOFF + j] = hrow(1, j))
        A[BOFF + LD + j] = brow(1, j)
        A[BOFF + 2LD + j] = brow(2, j)
    end
    sA = 1; sB = 2; sC = 3
    oA = 1; oB = nw + 1; oC = nw + 2
    if nw == 0                                 # only the two border rows (not reached for two teeth)
        sA = 2; oA = nw + 1; sB = 3; oB = nw + 2
    end
    d = one(E)
    for k in 1:nw
        hasnew = k < nw
        eA = E(A[BOFF + (sA - 1) * LD + k])
        eB = E(A[BOFF + (sB - 1) * LD + k])
        eC = E(A[BOFF + (sC - 1) * LD + k])
        eN = hasnew ? -w * Complex{TT}(_cv(A[XOFF + (k + 1) + (k - 1) * LD])) : zero(E)
        pv = 1; best = _m2p(eA, PT)
        mB = _m2p(eB, PT); mC = _m2p(eC, PT); mN = _m2p(eN, PT)
        if mB > best; best = mB; pv = 2; end
        if mC > best; best = mC; pv = 3; end
        if hasnew && mN > best; best = mN; pv = 4; end
        piv = pv == 1 ? eA : (pv == 2 ? eB : (pv == 3 ? eC : eN))
        r = pv == 1 ? oA : (pv == 2 ? oB : (pv == 3 ? oC : k + 1))
        sp = pv == 1 ? sA : (pv == 2 ? sB : sC)            # pivot's buffer (pv ≤ 3)
        # sign: rows not yet eliminated with a smaller original index than the pivot row
        cnt = Int(pv != 1 && oA < r) + Int(pv != 2 && oB < r) + Int(pv != 3 && oC < r) +
              Int(hasnew && pv != 4 && k + 1 < r) + (r > nw ? max(nw - k - 1, 0) : 0)
        d *= piv
        isodd(cnt) && (d = -d)
        ip = cinv(piv)
        # update targets t1, t2 (in place), t3 (in place if pv == 4, else the new row into sp)
        t1 = pv == 1 ? sB : sA
        t2 = pv == 3 ? sB : sC
        e1 = pv == 1 ? eB : eA
        e2 = pv == 3 ? eB : eC
        f1 = e1 * ip; f2 = e2 * ip
        f3 = (pv == 4 ? eC : eN) * ip
        t3 = pv == 4 ? sC : sp
        if pv == 4                                         # targets A, B, C
            t1 = sA; t2 = sB
            f1 = eA * ip; f2 = eB * ip
        end
        for j in (k + 1):n2
            pj = pv == 4 ? hrow(k + 1, j) : E(A[BOFF + (sp - 1) * LD + j])
            A[BOFF + (t1 - 1) * LD + j] = E(A[BOFF + (t1 - 1) * LD + j]) - f1 * pj
            A[BOFF + (t2 - 1) * LD + j] = E(A[BOFF + (t2 - 1) * LD + j]) - f2 * pj
            if pv == 4
                A[BOFF + (t3 - 1) * LD + j] = E(A[BOFF + (t3 - 1) * LD + j]) - f3 * pj
            elseif hasnew
                A[BOFF + (t3 - 1) * LD + j] = hrow(k + 1, j) - f3 * pj
            end
        end
        # new active set
        if pv != 4
            if hasnew                                      # the new row took the pivot's buffer
                pv == 1 && (oA = k + 1)
                pv == 2 && (oB = k + 1)
                pv == 3 && (oC = k + 1)
            else                                           # last column: drop the pivot row
                if pv == 1
                    sA = sB; oA = oB; sB = sC; oB = oC
                elseif pv == 2
                    sB = sC; oB = oC
                end
            end
        end
    end
    # the 2 x 2 Schur complement in the rows (A, B), columns nw+1, nw+2
    a1 = E(A[BOFF + (sA - 1) * LD + nw + 1]); a2 = E(A[BOFF + (sA - 1) * LD + nw + 2])
    b1 = E(A[BOFF + (sB - 1) * LD + nw + 1]); b2 = E(A[BOFF + (sB - 1) * LD + nw + 2])
    d2 = a1 * b2 - a2 * b1
    return oB < oA ? -(d * d2) : d * d2
end

if @isdefined(NyquistGPU)
    NyquistGPU.prepare(::typeof(D_mill3h), p, c) = mill3h_prepare(p, c)
end
