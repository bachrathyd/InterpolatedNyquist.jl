# Warp-cooperative D_mill3h (hill/gpu_helix.jl, HEAD e90381e, v2 setup: adaptive n_s, kink
# correction, wrapped flags from the time order): ONE WARP PER POINT.
#   prepare (once per point): mill3c_prepare + mill3h_prepare, lane-parallel
#     nodes / time origin / times / ranks: lanes = nodes (or candidate origins)
#     forward sweeps: lanes = right-hand-side columns (one column per lane, own running sums)
#     Householder reduction of X11: lanes = columns (left reflector), lanes = rows (right reflector)
#   evaluation: the bordered Hessenberg elimination of D_mill3h, lanes = columns of the 3 active
#     row buffers (+ the new Hessenberg row); the pivot candidates (column k) are read from shared
#     memory by every lane, so every lane takes the same pivot decision (no divergence).
# All per-element operations are those of the serial code, in the same order: the results are
# bit-identical to mill3h_prepare / D_mill3h in the same precision (checked by test_warp3h_cpu.jl).
# The X matrices (value-only, march precision, two planes) and the 3 row buffers (dual, four planes)
# live in SHARED memory; node fields and the per-lane running sums in the warp's contiguous slot.
# One code path for the GPU (GpuWarp: own lane, sync_warp) and a CPU emulation (EmuWarp: every phase
# loops over the 32 lanes). Requires NyquistGPU, hill/gpu_fast.jl, hill/gpu_helix.jl.
using ForwardDiff

abstract type WarpCtx end
struct EmuWarp <: WarpCtx end
struct GpuWarp <: WarpCtx
    lane::Int32
end
@inline lanes(::EmuWarp) = 0:31
@inline lanes(w::GpuWarp) = Int(w.lane):Int(w.lane)
@inline wsync(::EmuWarp) = nothing

# per-warp shared buffers (F: real planes of the march precision, I: Int32) -- not a Number, so
# NyquistGPU.conv_consts passes it through. Layout of F (offsets from fb):
#   X re, X im: ldx x nx each (cols 1..nw+2: Xa, cols nw+3..nw+4: Xb)
#   rows: 4 planes (Re val, Re partial, Im val, Im partial) x 3 buffers x ldr
#   Householder vector: re, im (ldx each)
# Layout of I (offsets from ib): 64 candidate counts/nodes, then W list, perm, rank (nn each)
struct WB3h{AF, AI, C <: WarpCtx}
    F::AF
    fb::Int32
    I::AI
    ib::Int32
    ldx::Int32
    nx::Int32
    ldr::Int32
    nn::Int32
    ctx::C
end
wb3h_nf(ldx, nx, ldr) = 2ldx * nx + 12ldr + 2ldx
wb3h_ni(nn) = 64 + 3nn

@inline function getX(wb::WB3h, i::Int, j::Int)
    k = Int(wb.fb) + i + (j - 1) * Int(wb.ldx)
    return Complex(@inbounds(wb.F[k]), @inbounds(wb.F[k + Int(wb.ldx) * Int(wb.nx)]))
end
@inline function setX!(wb::WB3h, i::Int, j::Int, v)
    k = Int(wb.fb) + i + (j - 1) * Int(wb.ldx)
    R = eltype(wb.F)
    @inbounds wb.F[k] = R(real(v))
    @inbounds wb.F[k + Int(wb.ldx) * Int(wb.nx)] = R(imag(v))
    return nothing
end
@inline _vbase(wb::WB3h) = Int(wb.fb) + 2 * Int(wb.ldx) * Int(wb.nx) + 12 * Int(wb.ldr)
@inline getV(wb::WB3h, i::Int) = Complex(@inbounds(wb.F[_vbase(wb) + i]), @inbounds(wb.F[_vbase(wb) + Int(wb.ldx) + i]))
@inline function setV!(wb::WB3h, i::Int, v)
    R = eltype(wb.F)
    @inbounds wb.F[_vbase(wb) + i] = R(real(v))
    @inbounds wb.F[_vbase(wb) + Int(wb.ldx) + i] = R(imag(v))
    return nothing
end
@inline function getR(wb::WB3h, s::Int, j::Int)
    ps = 3 * Int(wb.ldr)
    k = Int(wb.fb) + 2 * Int(wb.ldx) * Int(wb.nx) + (s - 1) * Int(wb.ldr) + j
    F = wb.F
    return Complex(ForwardDiff.Dual{NyquistGPU.PhaseTag}(@inbounds(F[k]), @inbounds(F[k + ps])),
                   ForwardDiff.Dual{NyquistGPU.PhaseTag}(@inbounds(F[k + 2ps]), @inbounds(F[k + 3ps])))
end
@inline function setR!(wb::WB3h, s::Int, j::Int, v)
    ps = 3 * Int(wb.ldr)
    k = Int(wb.fb) + 2 * Int(wb.ldx) * Int(wb.nx) + (s - 1) * Int(wb.ldr) + j
    F = wb.F
    R = eltype(F)
    @inbounds F[k] = R(ForwardDiff.value(real(v)))
    @inbounds F[k + ps] = R(ForwardDiff.partials(real(v), 1))
    @inbounds F[k + 2ps] = R(ForwardDiff.value(imag(v)))
    @inbounds F[k + 3ps] = R(ForwardDiff.partials(imag(v), 1))
    return nothing
end
@inline geti(wb::WB3h, k::Int) = Int(@inbounds(wb.I[Int(wb.ib) + k]))
@inline seti!(wb::WB3h, k::Int, v) = (@inbounds(wb.I[Int(wb.ib) + k] = Int32(v)); nothing)
const IW3 = 64                                   # W list at IW3 + i, perm at IW3 + nn + s, rank at IW3 + 2nn + n
# warp slot (Complex{T} entries): node fields of NM = 2 NS Q nodes, then per lane (interleaved by
# lane: coalesced) 2 running sums per sorted position and the kink column y per node
mill3w_wslen(c::Tuple) = (NM = 2 * Int(c[7]) * Int(c[8]); NM * NF3 + 96NM)

# running sums of lane ln at sorted position s (two per position), interleaved by lane: coalesced
@inline _scr(NM, ln, idx) = NM * NF3 + (idx - 1) * 32 + ln + 1             # idx = 1..2NM
@inline _ycol(NM, ln, n) = NM * NF3 + 64NM + (n - 1) * 32 + ln + 1          # y_n of lane ln

# ---- prepare: mill3c_prepare + mill3h_prepare, lane-parallel, same arithmetic ----
function prep3h_warp!(p, c, A, wb::WB3h, ::Val{HH} = Val(true)) where {HH}
    ctx = wb.ctx
    rk, ap = p
    TT = typeof(rk)
    ζ, w1, fn, R, tb1, tb2 = c[1], c[2], c[3], c[4], c[5], c[6]
    Q = unsafe_trunc(Int, c[7]); NS = unsafe_trunc(Int, c[8]); LD = unsafe_trunc(Int, c[9])
    kink = c[11] > 0
    ns = mill3c_ns(rk, ap, c)                   # axial nodes of this point (<= NS)
    T0 = mill3c_tab(Q, ns)
    N = 2 * ns * Q
    NM = 2 * NS * Q                             # slot layout of NS (fixed offsets)
    nn = Int(wb.nn)
    twoπ = TT(6.283185307179586)
    πT = TT(3.141592653589793)
    Ω = rk * (TT(1000) / (60 * fn))
    Ts = twoπ / Ω
    sζ = sqrt(1 - ζ * ζ)
    r1 = Complex(-ζ, sζ)
    ρ1 = Complex(zero(TT), -1 / (2 * sζ))
    b1 = tb1 / R; b2 = tb2 / R
    L = max(ap, TT(1e-6))
    er = exp(r1 * Ts)
    N > nn && return (Ts, TT(LD + 1), real(er), imag(er), TT(ns))   # does not fit this kernel: overflow
    # 1. nodes (lanes = nodes)
    for ln in lanes(ctx)
        n = ln + 1
        while n <= N
            i = (n - 1) % Q + 1
            l = ((n - 1) ÷ Q) % ns + 1
            j = (n - 1) ÷ (Q * ns) + 1
            φ = c[12 + i]; Wf = c[12 + Q + i]; x = c[T0 + l]; wz = c[T0 + ns + l]
            ζl = L * x
            bj = j == 1 ? b1 : b2
            bp = j == 1 ? b2 : b1
            θj = j == 1 ? zero(TT) : -πT
            ψ = mod(φ - θj + ζl * bj, twoπ)
            Ωτ = πT + ζl * (bj - bp)
            A[nf(n, 1)] = Complex(zero(TT), w1 * (L * wz) * Wf / Ω)
            A[nf(n, 5)] = Complex(ψ, Ωτ)
            jp = 3 - j
            A[nf(n, 4)] = Complex(TT(((jp - 1) * ns + (l - 1)) * Q + i), zero(TT))
            n += 32
        end
    end
    wsync(ctx)
    # 2. time origin: lanes = candidate nodes; the FIRST node with the minimal number of wrapped arcs
    for ln in lanes(ctx)
        bc = N + 1; bk = N + 1
        k = ln + 1
        while k <= N
            ψk = real(_cv(A[nf(k, 5)]))
            cnt = 0
            for n in 1:N
                a = _cv(A[nf(n, 5)])
                d = real(a) - ψk
                d += d < 0 ? twoπ : zero(TT)
                cnt += Int(d < imag(a))
            end
            if cnt < bc
                bc = cnt; bk = k
            end
            k += 32
        end
        seti!(wb, ln + 1, bc); seti!(wb, ln + 33, bk)
    end
    wsync(ctx)
    h = 16
    while h >= 1
        for ln in lanes(ctx)
            if ln < h
                c1 = geti(wb, ln + 1); k1 = geti(wb, ln + 33)
                c2 = geti(wb, ln + h + 1); k2 = geti(wb, ln + h + 33)
                if (c2 < c1) | ((c2 == c1) & (k2 < k1))
                    seti!(wb, ln + 1, c2); seti!(wb, ln + 33, k2)
                end
            end
        end
        wsync(ctx)
        h >>= 1
    end
    kb = geti(wb, 33)
    ψ0 = real(_cv(A[nf(kb, 5)]))
    wsync(ctx)                                    # the candidate slots are reused below
    # 3. times, exponentials, kink correction (lanes = nodes)
    for ln in lanes(ctx)
        n = ln + 1
        while n <= N
            a = _cv(A[nf(n, 5)])
            d = real(a) - ψ0
            d += d < 0 ? twoπ : zero(TT)
            t = d / Ω
            cn = imag(_cv(A[nf(n, 1)]))
            A[nf(n, 1)] = Complex(t, cn)
            e = exp(r1 * t)
            A[nf(n, 2)] = e
            A[nf(n, 3)] = cn * exp(-r1 * t)
            ε = zero(TT)
            if kink
                i = (n - 1) % Q + 1
                l = ((n - 1) ÷ Q) % ns + 1
                ε = w1 * (L * c[T0 + ns + l]) * c[12 + 2Q + i] / (Ω * Ω)
            end
            A[nf(n, 5)] = Complex(ε, 1 / (1 + ε))
            n += 32
        end
    end
    wsync(ctx)
    # 4. ranks (stable sort by time = the insertion sort of mill3c_prepare), lanes = nodes
    for ln in lanes(ctx)
        n = ln + 1
        while n <= N
            tn = real(_cv(A[nf(n, 1)]))
            r = 1
            for m in 1:N
                tm = real(_cv(A[nf(m, 1)]))
                r += Int((tm < tn) | ((tm == tn) & (m < n)))
            end
            seti!(wb, IW3 + 2nn + n, r)                     # rank[n]
            seti!(wb, IW3 + nn + r, n)                      # perm[r]
            n += 32
        end
    end
    wsync(ctx)
    # wrapped flags from the time order (wr = rank[partner] > rank[n]), lanes = nodes
    for ln in lanes(ctx)
        n = ln + 1
        while n <= N
            pw = _cv(A[nf(n, 4)])
            pn = unsafe_trunc(Int, real(pw))
            wr = geti(wb, IW3 + 2nn + pn) > geti(wb, IW3 + 2nn + n)
            A[nf(n, 4)] = Complex(real(pw), wr ? one(TT) : zero(TT))
            n += 32
        end
    end
    wsync(ctx)
    # wrapped numbering in node order (prefix count): W_i (I array), wi[n] (field 8), nw
    nw = 0
    for m in 1:N                                       # every lane: the same count
        nw += Int(imag(_cv(A[nf(m, 4)])) > 0)
    end
    for ln in lanes(ctx)
        n = ln + 1
        while n <= N
            if imag(_cv(A[nf(n, 4)])) > 0
                wc = 0
                for m in 1:n
                    wc += Int(imag(_cv(A[nf(m, 4)])) > 0)
                end
                seti!(wb, IW3 + wc, n)                      # W_wc = n
                A[nf(n, 8)] = Complex(zero(TT), TT(wc))     # wi[n] = wc
            end
            n += 32
        end
    end
    wsync(ctx)
    n2 = nw + 2
    if (n2 <= LD) & (n2 <= Int(wb.ldx)) & (nw + 4 <= Int(wb.nx)) & (n2 <= Int(wb.ldr))
        # 5. forward sweeps: lanes = columns (E_W, u0, u1), each with its own running sums and y
        for ln in lanes(ctx)
            col = ln + 1
            while col <= nw + 4
                S1 = zero(r1); S2 = zero(r1)
                for s in 1:N
                    n = geti(wb, IW3 + nn + s)
                    A[_scr(NM, ln, 2s - 1)] = S1
                    A[_scr(NM, ln, 2s)] = S2
                    a1 = _cv(A[nf(n, 2)])
                    pw = _cv(A[nf(n, 4)])
                    pn = unsafe_trunc(Int, real(pw))
                    wr = imag(pw) > 0
                    ap1 = _cv(A[nf(pn, 2)])
                    rhs = zero(r1)
                    if col <= nw
                        rhs = wr && unsafe_trunc(Int, imag(_cv(A[nf(n, 8)]))) == col ? one(r1) : zero(r1)
                    elseif col <= nw + 2
                        ak = col == nw + 1 ? a1 : conj(a1)
                        pk = col == nw + 1 ? ap1 : conj(ap1)
                        rhs = wr ? ak : ak - pk
                    else
                        pk = col == nw + 3 ? ap1 : conj(ap1)
                        rhs = wr ? pk : zero(r1)
                    end
                    y = rhs - (ρ1 * a1 * S1 + conj(ρ1) * conj(a1) * S2)
                    if !wr
                        sp = geti(wb, IW3 + 2nn + pn)
                        y += ρ1 * ap1 * _cv(A[_scr(NM, ln, 2sp - 1)]) + conj(ρ1) * conj(ap1) * _cv(A[_scr(NM, ln, 2sp)])
                        kink && (y += real(_cv(A[nf(pn, 5)])) * _cv(A[_ycol(NM, ln, pn)]))   # + eps_pi y_pi
                    end
                    if kink                                 # pivot 1 + eps_n
                        y *= imag(_cv(A[nf(n, 5)]))
                        A[_ycol(NM, ln, n)] = y
                    end
                    bc = _cv(A[nf(n, 3)])
                    S1 += bc * y
                    S2 += conj(bc) * y
                end
                for i in 1:nw                         # K rows: the partners of the wrapped nodes
                    wn = geti(wb, IW3 + i)
                    pn = unsafe_trunc(Int, real(_cv(A[nf(wn, 4)])))
                    sp = geti(wb, IW3 + 2nn + pn)
                    ap1 = _cv(A[nf(pn, 2)])
                    v = ρ1 * ap1 * _cv(A[_scr(NM, ln, 2sp - 1)]) + conj(ρ1) * conj(ap1) * _cv(A[_scr(NM, ln, 2sp)])
                    kink && (v += real(_cv(A[nf(pn, 5)])) * _cv(A[_ycol(NM, ln, pn)]))       # + eps_pi y_pi
                    setX!(wb, i, col, v)
                end
                setX!(wb, nw + 1, col, S1)
                setX!(wb, nw + 2, col, S2)
                col += 32
            end
        end
        wsync(ctx)
        # 6. Householder reduction of X11 to upper Hessenberg form (mill3h_prepare)
        for k in 1:(HH ? nw - 2 : 0)
            m = nw - k
            xm = zero(TT)
            for i in 1:m
                x = getX(wb, k + i, k)
                xm = max(xm, abs(real(x)), abs(imag(x)))
            end
            xm > 0 || continue
            sc = 1 / xm
            nx2 = zero(TT)
            for i in 1:m
                x = getX(wb, k + i, k) * sc
                nx2 += real(x) * real(x) + imag(x) * imag(x)
            end
            for ln in lanes(ctx)
                i = ln + 1
                while i <= m
                    setV!(wb, i, getX(wb, k + i, k) * sc)
                    i += 32
                end
            end
            wsync(ctx)
            nx = sqrt(nx2)
            x1 = getV(wb, 1)
            a1 = sqrt(real(x1) * real(x1) + imag(x1) * imag(x1))
            ph = a1 > 0 ? x1 * (1 / a1) : one(x1)
            τ = 1 / (nx * (nx + a1))
            wsync(ctx)                             # every lane has read v1 and column k
            for ln in lanes(ctx)
                i = ln + 1
                while i <= m
                    if i == 1
                        setV!(wb, 1, x1 + ph * nx)
                        setX!(wb, k + 1, k, -ph * (nx * xm))
                    else
                        setX!(wb, k + i, k, zero(x1))
                    end
                    i += 32
                end
            end
            wsync(ctx)
            for ln in lanes(ctx)                   # from the left: lanes = columns k+1..nw+4
                j = k + 1 + ln
                while j <= nw + 4
                    s = zero(x1)
                    for i in 1:m
                        s += conj(getV(wb, i)) * getX(wb, k + i, j)
                    end
                    s *= τ
                    for i in 1:m
                        setX!(wb, k + i, j, getX(wb, k + i, j) - s * getV(wb, i))
                    end
                    j += 32
                end
            end
            wsync(ctx)
            for ln in lanes(ctx)                   # from the right: lanes = rows 1..nw+2
                r = ln + 1
                while r <= nw + 2
                    s = zero(x1)
                    for i in 1:m
                        s += getX(wb, r, k + i) * getV(wb, i)
                    end
                    s *= τ
                    for i in 1:m
                        setX!(wb, r, k + i, getX(wb, r, k + i) - s * conj(getV(wb, i)))
                    end
                    r += 32
                end
            end
            wsync(ctx)
        end
    end
    return (Ts, TT(nw), real(er), imag(er), TT(ns))
end

# pivot magnitude as in D_mill3h
@inline _m2pw(z, ::Type{P}) where {P} = (x = P(_val(real(z))); y = P(_val(imag(z))); x * x + y * y)

# ---- evaluation: the bordered Hessenberg elimination of D_mill3h, lanes = columns ----
struct W3h end
@inline function (::W3h)(μ, q, cw)
    c, A, wb = cw
    TT = typeof(q[1])
    PT = TT === Float16 ? Float32 : TT
    ζ = c[1]
    LD = unsafe_trunc(Int, c[9])
    nw = unsafe_trunc(Int, q[2])
    n2 = nw + 2
    w = exp(-TT(6.283185307179586) * μ)        # 1/z
    n2 > LD && return w * TT(NaN)
    ((n2 > Int(wb.ldx)) | (nw + 4 > Int(wb.nx)) | (n2 > Int(wb.ldr))) && return w * TT(NaN)
    E = typeof(w)
    ctx = wb.ctx
    sζ = sqrt(1 - ζ * ζ)
    ρ1 = Complex(zero(TT), -1 / (2 * sζ))
    er = Complex(q[3], q[4])
    q1 = er * w
    q2 = conj(er) * w
    g1 = ρ1 * q1
    g2 = conj(ρ1) * q2
    # row i (1..nw) of [I - wH | -w(Y12a - w Y12b)] at column j (Xb columns are nw+3, nw+4)
    @inline hrow(i, j) = j <= nw ? (i == j ? one(E) : zero(E)) - w * Complex{TT}(getX(wb, i, j)) :
        -w * (Complex{TT}(getX(wb, i, j)) - w * Complex{TT}(getX(wb, i, j + 2)))
    @inline brow(b, j) = begin
        g = b == 1 ? g1 : g2
        if j <= nw
            g * Complex{TT}(getX(wb, nw + b, j))
        else
            v = g * (Complex{TT}(getX(wb, nw + b, j)) - w * Complex{TT}(getX(wb, nw + b, j + 2)))
            j - nw == b ? v + (b == 1 ? 1 - q1 : 1 - q2) : v
        end
    end
    for ln in lanes(ctx)
        j = ln + 1
        while j <= n2
            nw >= 1 && setR!(wb, 1, j, hrow(1, j))
            setR!(wb, 2, j, brow(1, j))
            setR!(wb, 3, j, brow(2, j))
            j += 32
        end
    end
    wsync(ctx)
    sA = 1; sB = 2; sC = 3
    oA = 1; oB = nw + 1; oC = nw + 2
    if nw == 0
        sA = 2; oA = nw + 1; sB = 3; oB = nw + 2
    end
    d = one(E)
    for k in 1:nw
        hasnew = k < nw
        eA = E(getR(wb, sA, k))
        eB = E(getR(wb, sB, k))
        eC = E(getR(wb, sC, k))
        eN = hasnew ? -w * Complex{TT}(getX(wb, k + 1, k)) : zero(E)
        pv = 1; best = _m2pw(eA, PT)
        mB = _m2pw(eB, PT); mC = _m2pw(eC, PT); mN = _m2pw(eN, PT)
        if mB > best; best = mB; pv = 2; end
        if mC > best; best = mC; pv = 3; end
        if hasnew && mN > best; best = mN; pv = 4; end
        piv = pv == 1 ? eA : (pv == 2 ? eB : (pv == 3 ? eC : eN))
        r = pv == 1 ? oA : (pv == 2 ? oB : (pv == 3 ? oC : k + 1))
        sp = pv == 1 ? sA : (pv == 2 ? sB : sC)
        cnt = Int(pv != 1 && oA < r) + Int(pv != 2 && oB < r) + Int(pv != 3 && oC < r) +
              Int(hasnew && pv != 4 && k + 1 < r) + (r > nw ? max(nw - k - 1, 0) : 0)
        d *= piv
        isodd(cnt) && (d = -d)
        ip = cinv(piv)
        t1 = pv == 1 ? sB : sA
        t2 = pv == 3 ? sB : sC
        e1 = pv == 1 ? eB : eA
        e2 = pv == 3 ? eB : eC
        f1 = e1 * ip; f2 = e2 * ip
        f3 = (pv == 4 ? eC : eN) * ip
        t3 = pv == 4 ? sC : sp
        if pv == 4
            t1 = sA; t2 = sB
            f1 = eA * ip; f2 = eB * ip
        end
        for ln in lanes(ctx)                       # lanes = columns k+1..n2
            j = k + 1 + ln
            while j <= n2
                pj = pv == 4 ? hrow(k + 1, j) : E(getR(wb, sp, j))
                setR!(wb, t1, j, E(getR(wb, t1, j)) - f1 * pj)
                setR!(wb, t2, j, E(getR(wb, t2, j)) - f2 * pj)
                if pv == 4
                    setR!(wb, t3, j, E(getR(wb, t3, j)) - f3 * pj)
                elseif hasnew
                    setR!(wb, t3, j, hrow(k + 1, j) - f3 * pj)
                end
                j += 32
            end
        end
        wsync(ctx)
        if pv != 4
            if hasnew
                pv == 1 && (oA = k + 1)
                pv == 2 && (oB = k + 1)
                pv == 3 && (oC = k + 1)
            else
                if pv == 1
                    sA = sB; oA = oB; sB = sC; oB = oC
                elseif pv == 2
                    sB = sC; oB = oC
                end
            end
        end
    end
    a1 = E(getR(wb, sA, nw + 1)); a2 = E(getR(wb, sA, nw + 2))
    b1 = E(getR(wb, sB, nw + 1)); b2 = E(getR(wb, sB, nw + 2))
    d2 = a1 * b2 - a2 * b1
    wsync(ctx)                                     # the buffers are rewritten by the next evaluation
    return oB < oA ? -(d * d2) : d * d2
end
