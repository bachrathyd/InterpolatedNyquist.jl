"""
    NyquistGPU

Brute-force stability sweeps of a *scalar* characteristic equation on the GPU.

The model is one scalar expression

    D(λ, p, c)     # λ complex, p = one K-dimensional parameter point (a tuple), c = tuple of constants

evaluated over an arbitrary list of parameter points -- any number of
dimensions K, any layout (rectangular grids, scattered samples, refinement
sets, lines through the space ...). A 2-D stability chart is just the special
case `points = grid_points(xs, ys)`.

Every point runs the same frequency march of the argument principle along
`λ = σ + iω`, `ω ∈ [ω0, ωmax]`:

    Z = n/2 - Δarg D(σ+iω) / π        (roots with Re λ > σ, real-coefficient D)

and, on the fly, tracks the deepest minima of `|D(σ+iω)|` (each one gives a
one-Newton-step estimate of a root near the line), as the CPU package
`InterpolatedNyquist.jl` does. Two marches are available:

* `:bs3`    -- the paper's phase ODE `dΦ/dω = Im(D'/D)` integrated with an
               adaptive Bogacki-Shampine 3(2) pair (3 evaluations / step).
               This is the validated port of the CPU method.
* `:unwrap` -- the phase is an exact differential, so it does not have to be
               *integrated* at all: the march only has to *unwrap*
               `arg D(σ+iω)` without losing a branch. Each step costs ONE
               evaluation; the observed increment `angle(D_b / D_a)` is
               checked against the trapezoid prediction `h (θ'_a + θ'_b)/2`
               from the exact derivatives, and the step is rejected when they
               disagree (a skipped near-root peak shows up as a ≈ π mismatch).
               The slowly decaying delay ripple, which forces thousands of
               steps on an integrator, is invisible once its phase amplitude
               drops below the tolerance -- the end-point phase is exact.

Three schedules map points to GPU threads:

* `:pixel`   -- one thread = one point (baseline).
* `:strided` -- a fixed pool of `lanes` threads; lane `l` processes points
                `perm(l), perm(l + lanes), ...`, where `perm` is a coprime-stride
                scramble of the point index (statistical load balancing, no atomics).
* `:queue`   -- a fixed pool of `lanes` persistent threads that pull the next
                point from a global atomic counter whenever they finish one.

In `:strided` and `:queue` the kernel loop body is *one march step*, and a
lane that finishes a point is refilled with the next one inside the same loop
-- so all lanes of a warp keep executing the same step instructions, and
adaptivity (points needing 30 or 3000 steps) no longer leaves lanes idle.

Everything is a KernelAbstractions kernel: the identical code runs on `CPU()`
(validation, any machine) and on `CUDABackend()` (after `using CUDA`).
"""
module NyquistGPU

using KernelAbstractions
using ForwardDiff
import Atomix

export LowFloat, Float8_E4M3, Float8_E5M2, Float4_E2M1, Float16_emu, BFloat16_emu, format_range
export SweepPlan, plan_sweep, run!, fetch_result, sweep, grid_points, chart, recheck!,
       check_eltype, simulate_schedule, coprime_stride, colour_field,
       plan_grid, regrid!, recheck_flagged!, set_march!

include("lowfloat.jl")

struct PhaseTag end

# Julia does not specialize a method on a `::Function` argument that the
# method only passes on without calling it -- which is exactly what the
# kernels do with D -- so on the CPU backend every evaluation of D would be a
# dynamic dispatch (measured: ~5x slower, heap-allocating). A callable wrapper
# that is not a `Function` is always specialized. (GPU compilation specializes
# on every argument type anyway; there the wrapper is a zero-size ghost.)
struct CharFn{F}
    f::F
end
@inline (m::CharFn)(λ, p, c) = m.f(λ, p, c)

# Literal integer powers of the dual-number λ (λ^4 in a user's D): unrolled
# products instead of Base's generic power_by_squaring loop, which is not
# inlined (2.5x slower on the CPU, a loop + call on the GPU). Only our own
# PhaseTag duals are affected.
const PhaseDualC{T} = Complex{ForwardDiff.Dual{PhaseTag, T, 1}}
@inline Base.literal_pow(::typeof(^), z::PhaseDualC, ::Val{p}) where {p} = _upow(z, Val(p))
@inline _upow(z, ::Val{0}) = one(z)
@inline _upow(z, ::Val{1}) = z
@inline _upow(z, ::Val{2}) = z * z
@inline function _upow(z, ::Val{p}) where {p}
    p < 0 && return inv(_upow(z, Val(-p)))
    h = _upow(z, Val(p ÷ 2))
    return isodd(p) ? h * h * z : h * h
end

# ===========================================================================
# March parameters (isbits, passed by value to the kernels)
# ===========================================================================
# T: precision of the march (frequency, step control, phase sum);
# TE: precision in which D is evaluated (TE = T, or a narrower format such as
# Float16 / an emulated Float8 -- mixed precision)
struct MarchParams{T, TE}
    σ::T         # integration line Re λ = σ
    ω0::T        # start of the march (tiny > 0, like the CPU package)
    ωmax::T      # end of the march
    h0::T        # first trial step
    rtol::T      # :bs3 relative tolerance
    atol::T      # :bs3 absolute tolerance  |  :unwrap phase tolerance [rad]
    hrel::T      # step cap  h <= hrel * max(ω, 1)
    hmax::T      # absolute step cap inside the resonance band ω < ωband
    ωband::T
    npow::T      # leading order n of D (D ~ c λ^n)
    maxsteps::Int32
    newton::Int32  # Newton steps polishing each tracked root after the march (0: first-order estimate)
    bisect::Int32  # 1: certify the rightmost root of stable points by counting on shifted lines
    σtol::T        # its tolerance in Re λ
end
evaltype(::MarchParams{T, TE}) where {T, TE} = TE

# ===========================================================================
# One evaluation: D and dD/dω on the line λ = σ + iω (one dual-number call)
# ===========================================================================
@inline eval_line(D::F, p, c, σ::T, ω::T) where {F, T} = eval_line(D, p, c, σ, ω, T)
@inline eval_line(D::F, p, c, mp::MarchParams{T, TE}, ω::T) where {F, T, TE} =
    eval_line(D, p, c, mp.σ, ω, TE)

# D is evaluated in TE (σ, ω and the point p are rounded to TE first; c already
# is TE), the results are returned in the march precision T
@inline function eval_line(D::F, p, c, σ::T, ω::T, ::Type{TE}) where {F, T, TE}
    w = ForwardDiff.Dual{PhaseTag}(TE(ω), one(TE))
    s = ForwardDiff.Dual{PhaseTag}(TE(σ), zero(TE))
    v = D(Complex(s, w), map(TE, p), c)
    re, im = real(v), imag(v)
    Dv = Complex{T}(T(ForwardDiff.value(re)), T(ForwardDiff.value(im)))
    Dw = Complex{T}(T(ForwardDiff.partials(re, 1)), T(ForwardDiff.partials(im, 1)))
    return Dv, Dw
end

# Scale-free view of a sample. With s = max(|Re D|, |Im D|) and u = D/s
# (|u|² ∈ [1, 2]) nothing overflows even when |D|² would (|D| ~ 1e20 in
# Float32), and no sqrt / hypot / complex division is needed -- the latter
# matters on the GPU, where Base widens Complex{Float32} division to Float64.
#   θ' = dΦ/dω = Im(D' conj D) / |D|²     (the phase integrand)
#   g  = Re(D' conj u)  -- same sign as d|D|/dω (minimum tracking)
@inline function sample_data(Dv::Complex{T}, Dw::Complex{T}) where {T}
    s = max(abs(real(Dv)), abs(imag(Dv)))
    u = Complex(real(Dv) / s, imag(Dv) / s)
    u2 = real(u)^2 + imag(u)^2
    th = (real(u) * imag(Dw) - imag(u) * real(Dw)) / (s * u2)
    g = real(u) * real(Dw) + imag(u) * imag(Dw)
    return u, th, g, s
end

# principal increment of arg between two (scaled) samples
@inline function dphase(ua::Complex{T}, ub::Complex{T}) where {T}
    pr = real(ub) * real(ua) + imag(ub) * imag(ua)      # ub * conj(ua)
    pi_ = imag(ub) * real(ua) - real(ub) * imag(ua)
    return atan(pi_, pr)
end

# q = D / D' (robust, no complex division). One Newton step from ω:
#   Δλ = -D / (dD/dλ) = -i D/Dω  ->  σ_est = σ + Im q,  ω_est = ω - Re q
@inline function newton_q(Dv::Complex{T}, Dw::Complex{T}) where {T}
    sw = max(abs(real(Dw)), abs(imag(Dw)))
    vr = real(Dw) / sw
    vi = imag(Dw) / sw
    v2 = vr * vr + vi * vi
    ar = real(Dv) / sw
    ai = imag(Dv) / sw
    return Complex((ar * vr + ai * vi) / v2, (ai * vr - ar * vi) / v2)
end

# Cubic Hermite model of the complex D(ω) on a step [a, a+h], t ∈ [0, 1]:
# value and dD/dω. D is smooth even where its phase is near-singular, so this
# locates a |D| minimum between two accepted samples at no evaluation cost.
@inline function hermite(A::Complex{T}, Aw::Complex{T}, B::Complex{T}, Bw::Complex{T},
                         h::T, t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    p = h00 * A + (h10 * h) * Aw + h01 * B + (h11 * h) * Bw
    d00 = 6 * t2 - 6 * t
    d10 = 3 * t2 - 4 * t + 1
    d11 = 3 * t2 - 2 * t
    dp = (d00 / h) * (A - B) + d10 * Aw + d11 * Bw
    return p, dp
end

# Minimum of |D| inside an accepted step whose end-point slopes bracket it
# (g_a < 0 < g_b): regula falsi (Illinois) on Re(conj(p) p') of the Hermite
# model, then one Newton step from the located minimum -> root estimate.
# Returns (depth |D_min|, σ_est, ω_est).
@inline function dip_root(Da::Complex{T}, Dwa::Complex{T}, sa::T,
                          Db::Complex{T}, Dwb::Complex{T}, sb::T,
                          a::T, h::T, σ::T) where {T}
    sc = inv(max(sa, sb))
    A = Da * sc
    Aw = Dwa * sc
    B = Db * sc
    Bw = Dwb * sc
    tl = zero(T)
    tr = one(T)
    fl = real(A) * real(Aw) + imag(A) * imag(Aw)        # < 0
    fr = real(B) * real(Bw) + imag(B) * imag(Bw)        # > 0
    t = clamp(fl / (fl - fr), T(0.01), T(0.99))
    side = 0
    for _ in 1:4
        p, dp = hermite(A, Aw, B, Bw, h, t)
        f = real(p) * real(dp) + imag(p) * imag(dp)
        if f < 0
            tl = t; fl = f
            side == -1 && (fr /= 2)
            side = -1
        else
            tr = t; fr = f
            side == 1 && (fl /= 2)
            side = 1
        end
        den = fr - fl
        t = den != 0 ? (tl * fr - tr * fl) / den : (tl + tr) / 2
        t = clamp(t, tl, tr)
    end
    p, dp = hermite(A, Aw, B, Bw, h, t)
    q = newton_q(p, dp)                                  # scale cancels in D/D'
    sp = max(abs(real(p)), abs(imag(p)))
    depth = sp * sqrt((real(p) / sp)^2 + (imag(p) / sp)^2) / sc
    ωm = a + t * h
    # trust region (as the package's refine_roots): a Newton step longer than
    # max(|λ_seed|, 1) has left the basin -- typically a degenerate minimum
    # with D' ≈ 0 in a flat tail -- and would report a garbage root
    trusted = abs2(q) <= max(σ * σ + ωm * ωm, one(T))
    return (trusted ? depth : T(NaN)), σ + imag(q), ωm - real(q)
end

# Insert (depth, σ, ω) into the depth-sorted N-slot buffer; the deepest N
# minima survive (depth ranking, as in the package: a spurious minimum with a
# garbage σ is shallow and gets evicted).
@inline function insert_root(dd::NTuple{N, T}, ds::NTuple{N, T}, dw::NTuple{N, T},
                             d::T, s::T, w::T) where {N, T}
    (isfinite(d) & isfinite(s) & isfinite(w)) || return dd, ds, dw
    k = 1 + sum(ntuple(j -> Int(dd[j] <= d), Val(N)))
    k > N && return dd, ds, dw
    ndd = ntuple(j -> j < k ? dd[j] : (j == k ? d : dd[max(j - 1, 1)]), Val(N))
    nds = ntuple(j -> j < k ? ds[j] : (j == k ? s : ds[max(j - 1, 1)]), Val(N))
    ndw = ntuple(j -> j < k ? dw[j] : (j == k ? w : dw[max(j - 1, 1)]), Val(N))
    return ndd, nds, ndw
end

# ===========================================================================
# Per-point march state: an immutable isbits struct, updated one step at a
# time -- the unit of work that the refill schedules interleave.
# ===========================================================================
struct MarchState{T, N, P}
    p::P                 # the parameter point (NTuple{K,T})
    ω::T
    h::T
    Φ::T                 # unwrapped phase increment / integral of θ' since ω0
    Da::Complex{T}       # D(σ + iω)
    Dwa::Complex{T}      # dD/dω
    ua::Complex{T}       # Da / sa
    sa::T
    tha::T               # θ'(ω)
    ga::T                # sign of d|D|/dω
    steps::Int32         # attempted steps (accepted + rejected)
    status::Int8         # 0 running, 1 finished, 2 failed
    flags::Int8          # bit 2: a sub-resolution transition was decided by the root side
    dd::NTuple{N, T}     # depths of the tracked minima (ascending)
    ds::NTuple{N, T}     # their σ estimates
    dw::NTuple{N, T}     # their ω estimates
end

@inline function blank_state(p::P, ::Type{T}, ::Val{N}) where {P, T, N}
    z = zero(T)
    cz = Complex(z, z)
    nan = ntuple(_ -> T(NaN), Val(N))
    return MarchState{T, N, P}(p, z, z, z, cz, cz, cz, z, z, z, Int32(0), Int8(1), Int8(0),
        ntuple(_ -> T(Inf), Val(N)), nan, nan)
end

@inline function seed(D::F, p::P, c, mp::MarchParams{T}, ::Val{N}) where {F, P, T, N}
    ω = mp.ω0
    Dv, Dw = eval_line(D, p, c, mp, ω)
    u, th, g, s = sample_data(Dv, Dw)
    dd = ntuple(_ -> T(Inf), Val(N))
    ds = ntuple(_ -> T(NaN), Val(N))
    dw = ds
    # |D(σ+iω)| is even in ω for real-coefficient D: if it grows away from
    # ω = 0, the minimum sits AT ω = 0 (a real root near the line).
    if g > 0
        q = newton_q(Dv, Dw)
        depth = s * sqrt(real(u)^2 + imag(u)^2)
        trusted = abs2(q) <= max(mp.σ * mp.σ, one(T))
        trusted && ((dd, ds, dw) = insert_root(dd, ds, dw, depth, mp.σ + imag(q), zero(T)))
    end
    ok = isfinite(th) & isfinite(s) & (s > 0)
    return MarchState{T, N, P}(p, ω, mp.h0, zero(T), Dv, Dw, u, s, th, g,
        Int32(0), ok ? Int8(0) : Int8(2), Int8(0), dd, ds, dw)
end

# common accept/reject bookkeeping
@inline function advance(st::MarchState{T, N, P}, mp::MarchParams{T}, accept::Bool,
                         b::T, h::T, hn::T, Φn::T,
                         Db, Dwb, ub, sb, thb, gb, fl::Int8 = Int8(0)) where {T, N, P}
    steps = st.steps + Int32(1)
    if accept
        dd, ds, dw = st.dd, st.ds, st.dw
        if (st.ga < 0) & (gb > 0)
            depth, se, we = dip_root(st.Da, st.Dwa, st.sa, Db, Dwb, sb, st.ω, h, mp.σ)
            dd, ds, dw = insert_root(dd, ds, dw, depth, se, we)
        end
        status = b >= mp.ωmax ? Int8(1) : (steps >= mp.maxsteps ? Int8(2) : Int8(0))
        return MarchState{T, N, P}(st.p, b, hn, Φn, Db, Dwb, ub, sb, thb, gb,
            steps, status, st.flags | fl, dd, ds, dw)
    else
        stuck = hn <= hfloor(st.ω, mp)
        status = (stuck | (steps >= mp.maxsteps)) ? Int8(2) : Int8(0)
        return MarchState{T, N, P}(st.p, st.ω, hn, st.Φ, st.Da, st.Dwa, st.ua, st.sa,
            st.tha, st.ga, steps, status, st.flags, st.dd, st.ds, st.dw)
    end
end

@inline function trial_step(st::MarchState{T}, mp::MarchParams{T}) where {T}
    rest = mp.ωmax - st.ω
    h = min(st.h, mp.hrel * max(st.ω, one(T)))
    h = st.ω < mp.ωband ? min(h, mp.hmax) : h
    last = h >= rest
    return (last ? rest : h), (last ? mp.ωmax : st.ω + h)
end

@inline finite_or_inf(e::T) where {T} = ifelse(isfinite(e), e, T(Inf))

# smallest step that still moves ω in precision T
@inline hfloor(ω::T) where {T} = 8 * eps(T) * max(ω, one(T))
# with a narrower evaluation format, frequencies closer than its relative
# spacing evaluate to the same point: steps below that are wasted
@inline hfloor(ω::T, ::MarchParams{T, T}) where {T} = hfloor(ω)
@inline hfloor(ω::T, mp::MarchParams{T, TE}) where {T, TE} =
    8 * max(eps(T), T(eps(TE))) * max(ω, mp.ω0)

# --- :unwrap  (1 evaluation per step) --------------------------------------
@inline function march_step(D::F, st::MarchState{T, N}, c, mp::MarchParams{T},
                            ::Val{:unwrap}) where {F, T, N}
    h, b = trial_step(st, mp)
    Db, Dwb = eval_line(D, st.p, c, mp, b)
    ub, thb, gb, sb = sample_data(Db, Dwb)
    Δ = dphase(st.ua, ub)
    err = finite_or_inf(abs(Δ - h * (st.tha + thb) / 2))
    tol = mp.atol
    valid = isfinite(thb) & (sb > 0) & isfinite(Δ)
    accept = (err <= tol) & valid
    fac = clamp(T(0.9) * cbrt(tol / max(err, floatmin(T))), T(0.2), T(4))
    hn = h * fac
    if !accept & valid & (hn <= hfloor(st.ω, mp))
        # A root so close to the line that its ±π phase transition is narrower
        # than the smallest representable step (|Re λ - σ| ≲ eps·ω): the step
        # cannot be refined further, and the principal increment is ambiguous
        # near ±π. Decide the branch by the SIDE of the root -- one Newton step
        # from the sample (the paper's peak-repair idea, at no extra
        # evaluation): a root right of the line lowers the phase by π, one
        # left of it raises the phase by π.
        q = newton_q(Db, Dwb)
        right = imag(q) > 0
        Δr = Δ
        if abs(Δ) > T(π) / 2
            Δr = (right & (Δ > 0)) ? Δ - 2 * T(π) : (((!right) & (Δ < 0)) ? Δ + 2 * T(π) : Δ)
        end
        return advance(st, mp, true, b, h, h, st.Φ + Δr, Db, Dwb, ub, sb, thb, gb, Int8(2))
    end
    return advance(st, mp, accept, b, h, hn, st.Φ + Δ, Db, Dwb, ub, sb, thb, gb)
end

# --- :bs3  (Bogacki-Shampine 3(2), FSAL: 3 evaluations per step) ------------
@inline function theta_at(D::F, st::MarchState{T}, c, mp::MarchParams{T}, ω::T) where {F, T}
    Dv, Dw = eval_line(D, st.p, c, mp, ω)
    _, th, _, _ = sample_data(Dv, Dw)
    return th
end

@inline function march_step(D::F, st::MarchState{T, N}, c, mp::MarchParams{T},
                            ::Val{:bs3}) where {F, T, N}
    h, b = trial_step(st, mp)
    th1 = st.tha
    th2 = theta_at(D, st, c, mp, st.ω + h / 2)
    th3 = theta_at(D, st, c, mp, st.ω + 3 * h / 4)
    ynew = st.Φ + h * (T(2 / 9) * th1 + T(1 / 3) * th2 + T(4 / 9) * th3)
    Db, Dwb = eval_line(D, st.p, c, mp, b)
    ub, th4, gb, sb = sample_data(Db, Dwb)
    zlow = st.Φ + h * (T(7 / 24) * th1 + T(1 / 4) * th2 + T(1 / 3) * th3 + T(1 / 8) * th4)
    err = finite_or_inf(abs(ynew - zlow))
    tol = mp.atol + mp.rtol * abs(ynew)
    accept = (err <= tol) & isfinite(th4) & (sb > 0)
    fac = clamp(T(0.9) * cbrt(tol / max(err, floatmin(T))), T(0.2), T(5))
    return advance(st, mp, accept, b, h, h * fac, ynew, Db, Dwb, ub, sb, th4, gb)
end

evals_per_step(::Val{:unwrap}) = 1
evals_per_step(::Val{:bs3}) = 3

# --- results ----------------------------------------------------------------
# Z_raw (NaN if the march failed), the dominant tracked root (largest σ_est
# among the deepest minima -- the spectral-gap colouring of the paper), and
# the attempted step count.
@inline function finish(st::MarchState{T, N}, mp::MarchParams{T}) where {T, N}
    Zraw = st.status == Int8(1) ? mp.npow / 2 - st.Φ / T(π) : T(NaN)
    σd = T(-Inf)
    ωd = T(NaN)
    for j in 1:N
        s = st.ds[j]
        if isfinite(s) & (s > σd)
            σd = s
            ωd = st.dw[j]
        end
    end
    σd = isfinite(σd) ? σd : T(NaN)
    fl = st.flags | (st.status == Int8(2) ? Int8(1) : Int8(0))
    # the paper's integer residual: Z_raw far from an integer means a root ON
    # the line (e.g. D(σ) = 0 exactly gives Z_raw = k + 1/2) or a truncation
    # problem -- the count is not trustworthy either way
    fl |= (abs(Zraw - round(Zraw)) > T(0.25)) ? Int8(4) : Int8(0)
    return Zraw, σd, ωd, st.steps, fl
end

# Newton polish of one tracked root in the complex plane, in the march
# precision T (also when D is evaluated in a narrower format during the march):
# λ ← λ - D/D', with D' = -i dD/dω from the same dual-number evaluation.
# Steps longer than the trust radius max(|λ|, 1) are shortened to it (damped
# Newton). Returns (NaN, NaN) if the last full step is not small (not converged).
@inline function newton_root(D::F, p, c, s::T, w::T, k::Int32) where {F, T}
    # a seed on the real axis (the ω = 0 minimum): for real-coefficient D, Newton
    # started there never leaves the axis and cannot reach a complex pair -- start
    # slightly above it (a real root still attracts the iteration)
    if abs(w) < T(1e-3) * max(abs(s), one(T))
        w = T(0.25) * max(abs(s), T(0.1))
    end
    qa = T(Inf)
    for _ in 1:k
        Dv, Dw = eval_line(D, p, c, s, w, T)
        q = newton_q(Dv, Dw)
        qa = abs2(q)
        isfinite(qa) || return T(NaN), T(NaN)
        r2 = max(s * s + w * w, one(T))           # trust radius² max(|λ|, 1)²
        if qa > r2                                # damped: shorten, do not reject
            q *= sqrt(r2 / qa)
        end
        s += imag(q)
        w -= real(q)
    end
    conv = qa <= T(1e-6) * max(s * s + w * w, one(T))
    return (conv ? s : T(NaN)), (conv ? w : T(NaN))
end

# the count on the shifted line Re λ = σ (a full march; -1 if it failed). Always in the
# march precision T: with a narrow evaluation format (Float16) a root near the shifted
# line is below its resolution, and the counts -- unlike the main sweep, whose doubtful
# points are flagged -- would scatter the certified σ.
@inline function count_at(D::F, p, c, mp::MarchParams{T, TE}, meth, σ::T) where {F, T, TE}
    m = MarchParams{T, T}(σ, mp.ω0, mp.ωmax, mp.h0, mp.rtol, mp.atol, mp.hrel, mp.hmax,
        mp.ωband, mp.npow, mp.maxsteps, Int32(0), Int32(0), mp.σtol)
    cT = map(T, c)
    st = seed(D, p, cT, m, Val(1))
    while st.status == Int8(0)
        st = march_step(D, st, cT, m, meth)
    end
    st.status == Int8(1) || return Int32(-1)
    return round(Int32, m.npow / 2 - st.Φ / T(π))
end

# Rightmost root of a stable point (count 0 on Re λ = σ) by counting: confirm the
# estimate σ̂ with the counts at σ̂ ± σtol, otherwise bracket and bisect.
# a count that does not fail on a line through a root (D = 0 on the line):
# retry on a slightly shifted line
@inline function count_near(D::F, p, c, mp::MarchParams{T}, meth, σ::T) where {F, T}
    k = count_at(D, p, c, mp, meth, σ)
    k >= Int32(0) && return k
    return count_at(D, p, c, mp, meth, σ - mp.σtol / 7)
end

@inline function certify_sigma(D::F, p, c, mp::MarchParams{T}, meth, σh::T) where {F, T}
    σ0 = mp.σ
    tol = mp.σtol
    g = isfinite(σh) ? min(σh, σ0) : σ0 - one(T)
    hi = min(g + tol, σ0)
    lo = g - tol
    if (hi >= σ0 || count_near(D, p, c, mp, meth, hi) == Int32(0)) &&
       count_near(D, p, c, mp, meth, lo) >= Int32(1)
        return (lo + hi) / 2                              # the estimate was right
    end
    hi = σ0                                               # count 0 here (stable point)
    span = max(abs(g - σ0), T(0.1))
    lo = σ0 - span
    found = false
    for _ in 1:5                                          # bracket: some root right of lo
        k = count_near(D, p, c, mp, meth, lo)
        k < 0 && return T(NaN)
        if k >= 1
            found = true
            break
        end
        hi = lo
        span *= 2
        lo = σ0 - span
    end
    found || return T(NaN)
    for _ in 1:40
        hi - lo <= tol && break
        m = (lo + hi) / 2
        k = count_near(D, p, c, mp, meth, m)
        k < 0 && return T(NaN)
        if k >= 1
            lo = m
        else
            hi = m
        end
    end
    return (lo + hi) / 2
end

# rightmost of the polished roots (NaN if none converged)
@inline function refined_dominant(D::F, st::MarchState{T, N}, c, mp::MarchParams{T}) where {F, T, N}
    cT = map(T, c)
    σd = T(-Inf)
    ωd = T(NaN)
    for j in 1:N
        s0 = st.ds[j]
        if isfinite(s0)
            s, w = newton_root(D, st.p, cT, s0, st.dw[j], mp.newton)
            if isfinite(s) & (s > σd)
                σd = s
                ωd = abs(w)
            end
        end
    end
    return (isfinite(σd) ? σd : T(NaN)), ωd
end

@inline function store!(Zr, Sg, Om, St, Fl, i, st, mp, D, c, meth)
    z, s, w, n, f = finish(st, mp)
    if mp.newton > 0
        s2, w2 = refined_dominant(D, st, c, mp)
        if isfinite(s2)
            s, w = s2, w2
        end
    end
    if (mp.bisect > 0) & isfinite(z)
        if round(z) == 0
            s = certify_sigma(D, st.p, c, mp, meth, s)
        end
    end
    @inbounds Zr[i] = z
    @inbounds Sg[i] = s
    @inbounds Om[i] = w
    @inbounds St[i] = n
    @inbounds Fl[i] = f
    return nothing
end

# ===========================================================================
# Kernels -- one per schedule. Pts is a device vector of NTuple{K,T}.
# ===========================================================================
@kernel function k_pixel!(Zr, Sg, Om, St, Fl, D, c, @Const(Pts), npts, mp, meth, nr)
    i = @index(Global, Linear)
    if i <= npts
        st = seed(D, @inbounds(Pts[i]), c, mp, nr)
        while st.status == Int8(0)
            st = march_step(D, st, c, mp, meth)
        end
        store!(Zr, Sg, Om, St, Fl, i, st, mp, D, c, meth)
    end
end

# scrambled index: j ↦ (j·P mod n) + 1 is a bijection for gcd(P, n) = 1
@inline scramble(j, P, n) = Int32(mod(Int64(j) * Int64(P), Int64(n)) + 1)

@kernel function k_strided!(Zr, Sg, Om, St, Fl, D, c, @Const(Pts), npts, mp, meth, nr, lanes, P)
    l = @index(Global, Linear)
    if l <= lanes
        j = Int32(l - 1)                       # this lane's k-th point: j = l-1 + k*lanes
        i = j < npts ? scramble(j, P, npts) : Int32(0)
        st = blank_state(@inbounds(Pts[1]), typeof(mp.σ), nr)
        if i > 0
            st = seed(D, @inbounds(Pts[i]), c, mp, nr)
        end
        while i > 0
            if st.status == Int8(0)
                st = march_step(D, st, c, mp, meth)
            else                                # finished: store and refill the lane
                store!(Zr, Sg, Om, St, Fl, i, st, mp, D, c, meth)
                j += lanes
                i = j < npts ? scramble(j, P, npts) : Int32(0)
                if i > 0
                    st = seed(D, @inbounds(Pts[i]), c, mp, nr)
                end
            end
        end
    end
end

# atomic fetch-and-add; `+=` returns the NEW value = the next 1-based point
@inline next_point!(counter) = Atomix.@atomic :monotonic counter[1] += Int32(1)

@kernel function k_queue!(Zr, Sg, Om, St, Fl, D, c, @Const(Pts), npts, mp, meth, nr, counter)
    i = next_point!(counter)
    st = blank_state(@inbounds(Pts[1]), typeof(mp.σ), nr)
    if i <= npts
        st = seed(D, @inbounds(Pts[i]), c, mp, nr)
    end
    while i <= npts
        if st.status == Int8(0)
            st = march_step(D, st, c, mp, meth)
        else
            store!(Zr, Sg, Om, St, Fl, i, st, mp, D, c, meth)
            i = next_point!(counter)
            if i <= npts
                st = seed(D, @inbounds(Pts[i]), c, mp, nr)
            end
        end
    end
end

# ===========================================================================
# Host side
# ===========================================================================
"""
    coprime_stride(n) -> P

A multiplier with `gcd(P, n) == 1` near `n / golden ratio`, so that
`j ↦ j·P mod n` scatters consecutive lane indices across the whole point list.
"""
function coprime_stride(n::Integer)
    n <= 2 && return 1
    P = max(1, round(Int, n / Base.MathConstants.golden))
    while gcd(P, n) != 1
        P += 1
    end
    return P
end

"""
    grid_points(axes...) -> Vector{NTuple{K,Float64}}

All points of the rectangular grid spanned by `K` axis vectors/ranges, first
axis fastest -- the layout of `vec([(x, y) for x in xs, y in ys])`. Results of
a sweep over it reshape with `reshape(res.Z, length.(axes)...)`.
"""
grid_points(axes...) = vec([Tuple(I) for I in Iterators.product(axes...)])
grid_points(axes::Tuple) = grid_points(axes...)

# any iterable of fixed-length vectors/tuples -> Vector{NTuple{K,T}}
function to_points(points, ::Type{T}) where {T}
    p1 = first(points)
    K = length(p1)
    return [ntuple(k -> T(q[k]), K) for q in points]
end

"""
    check_eltype(D, p, c, T) -> Bool

`true` if `D(λ, p, c)` keeps the working precision `T` (p: one parameter
point). A Float64 literal (`0.5 * λ`) silently promotes a Float32 kernel to
Float64 -- correct, but ~32-64x slower on consumer and RTX-PRO GPUs. Put
constants in `c`, or write `0.5f0`, `T(0.5)`, or integers.
"""
function check_eltype(D, p, c, ::Type{T}) where {T}
    d = ForwardDiff.Dual{PhaseTag}(T(1), one(T))
    v = D(Complex(d, d), map(T, Tuple(p)), map(T, Tuple(c)))
    return ForwardDiff.valtype(real(v)) === T
end

mutable struct SweepPlan{T, N, B, A, AI, AF, AP}
    backend::B
    npts::Int
    K::Int
    method::Symbol
    schedule::Symbol
    lanes::Int
    workgroup::Int
    stride::Int
    mp::MarchParams{T}
    points::AP           # device vector of NTuple{K,T}
    Zraw::A
    sigma::A
    omega::A
    steps::AI
    flags::AF
    counter::AI
end

"""
    plan_sweep(points; kw...) -> SweepPlan

Upload `points` (any iterable of equal-length vectors/tuples: an N-dimensional
parameter list, `grid_points(xs, ys)` for a chart) and allocate the result
buffers -- once; reuse the plan for every frame / constant set.

Keywords (defaults in brackets):
- `backend` [`CPU()`] -- `CUDABackend()` after `using CUDA`
- `T` [`Float32`] -- working precision of the march (consumer GPUs: Float32)
- `Teval` [`T`] -- precision in which D itself is evaluated (mixed precision):
  `Float16`, or an emulated `Float8_E4M3`, `Float8_E5M2`, `Float4_E2M1`, `BFloat16_emu`
  (accuracy studies only -- GPUs have no scalar Float8/Float4 units). Narrow formats
  overflow easily: write D in a rescaled frequency, see `scripts/precision_scaled.jl`
- `method` [`:unwrap`] -- `:unwrap` (1 eval/step) or `:bs3` (validated port)
- `schedule` [GPU: `:pixel`, CPU: `:queue`] -- `:pixel`, `:strided` or `:queue` (on a T4 the
  plain one-thread-per-point `:pixel` was fastest; `:queue` balances CPU threads best)
- `nroots` [`4`] -- tracked |D| minima per point (dominant root = max σ among them)
- `n_power` (required) -- leading order n of D
- `σ` [`0`], `ω0` [`1e-9`], `ω_max` [`1e5` for :unwrap, `1e4` for :bs3] -- the :unwrap end-point
  phase is exact, so ω_max only sets the truncation residual (~|ε(ω_max)|/π);
  much larger values just push Float32 trig into slow large-argument reduction
- `tol` -- `:unwrap`: phase tolerance in rad [`0.3`]; `:bs3`: abs = rel tolerance [`1e-5`]
- `hrel` [`1.0`] -- step cap h ≤ hrel·max(ω, 1)
- `hmax` [`Inf`], `ωband` [`0`] -- absolute step cap h ≤ hmax while ω < ωband
  (forces a minimum sampling density over the resonance band; see the notes on
  rational D in PLAN.md)
- `maxsteps` [`200_000`]
- `refine` [`0`] -- Newton steps polishing each tracked root after the march (complex
  plane, march precision; one evaluation each). 0 keeps the first-order estimate of the
  paper; 3-5 remove its bias deep in the stable domain. A root whose Newton iteration does
  not converge is discarded (if none converges, the first-order estimate is kept).
- `certify` [`false`], `σtol` [`1e-3`] -- for every stable point (Z = 0), determine the
  rightmost root by COUNTING: σ* is where the count on the shifted line Re λ = σ jumps from
  0 to ≥ 1. The (polished) estimate σ̂ is confirmed with two extra marches at σ̂ ± σtol;
  where it is wrong -- the tracked |D| minima can miss the rightmost root deep in the
  stable domain -- a bracket-and-bisect search finds it. Exact up to σtol, independent of
  the root tracking; `sigma` is NaN if no root lies right of σ = -16(1 + |σ̂|).
- `lanes` -- persistent threads for `:strided`/`:queue`. GPU: pass
  `#SMs × resident threads per SM` (the benchmark scripts do); default 2^18.
  CPU: one lane per Julia thread (`:queue`) or 8 per thread (`:strided`).
- `workgroup` -- GPU: 256. CPU: 1 for the lane schedules (each lane is its own
  task -- a CPU workgroup runs its items one after another, so a bigger group
  would let one item drain the whole queue), 64 for `:pixel`.
"""
function plan_sweep(points; backend = CPU(), T::Type = Float32, Teval::Union{Nothing, Type} = nothing,
                    method::Symbol = :unwrap,
                    schedule::Union{Nothing, Symbol} = nothing, nroots::Integer = 4, n_power,
                    σ = 0.0, ω0 = 1e-9, ω_max = nothing, tol = nothing, h0 = 1e-2,
                    hrel = 1.0, hmax = Inf, ωband = 0.0, maxsteps::Integer = 200_000,
                    refine::Integer = 0, certify::Bool = false, σtol = 1e-3,
                    lanes::Union{Nothing, Integer} = nothing,
                    workgroup::Union{Nothing, Integer} = nothing)
    method in (:unwrap, :bs3) || error("method must be :unwrap or :bs3")
    schedule = something(schedule, backend isa CPU ? :queue : :pixel)
    schedule in (:pixel, :strided, :queue) || error("schedule must be :pixel, :strided or :queue")
    ω_max = something(ω_max, method === :unwrap ? 1e5 : 1e4)
    tol = something(tol, method === :unwrap ? 0.3 : 1e-5)
    hpts = to_points(points, T)
    npts = length(hpts)
    npts < typemax(Int32) || error("too many points for Int32 indexing")
    cpu = backend isa CPU
    nt = Threads.nthreads()
    lanes = something(lanes, cpu ? (schedule === :queue ? nt : 8 * nt) : 1 << 18)
    lanes = min(lanes, npts)
    workgroup = something(workgroup, cpu ? (schedule === :pixel ? 64 : 1) : 256)
    TE = something(Teval, T)
    mp = MarchParams{T, TE}(T(σ), T(ω0), T(ω_max), T(h0), T(tol), T(tol), T(hrel), T(hmax),
        T(ωband), T(n_power), Int32(maxsteps), Int32(refine), Int32(certify), T(σtol))
    dpts = KernelAbstractions.allocate(backend, eltype(hpts), npts)
    copyto!(dpts, hpts)
    Zraw = KernelAbstractions.zeros(backend, T, npts)
    sig = KernelAbstractions.zeros(backend, T, npts)
    om = KernelAbstractions.zeros(backend, T, npts)
    st = KernelAbstractions.zeros(backend, Int32, npts)
    fl = KernelAbstractions.zeros(backend, Int8, npts)
    ctr = KernelAbstractions.zeros(backend, Int32, 1)
    return SweepPlan{T, Int(nroots), typeof(backend), typeof(Zraw), typeof(st), typeof(fl),
                     typeof(dpts)}(
        backend, npts, length(hpts[1]), method, schedule, lanes, workgroup,
        coprime_stride(npts), mp, dpts, Zraw, sig, om, st, fl, ctr)
end

"""
    set_march!(plan; ω_max, refine, certify, σtol, σ, tol) -> plan

Change march settings of an existing plan without reallocating (slider-driven
use): the end of the march `ω_max`, the Newton polish `refine`, the line `σ`,
the tolerance `tol`. Unspecified settings are kept.
"""
function set_march!(p::SweepPlan{T}; ω_max = nothing, refine = nothing, σ = nothing,
                    tol = nothing, certify = nothing, σtol = nothing) where {T}
    m = p.mp
    TE = evaltype(m)
    t = tol === nothing ? m.atol : T(tol)
    p.mp = MarchParams{T, TE}(σ === nothing ? m.σ : T(σ), m.ω0,
        ω_max === nothing ? m.ωmax : T(ω_max), m.h0, p.method === :unwrap ? t : m.rtol, t,
        m.hrel, m.hmax, m.ωband, m.npow, m.maxsteps,
        refine === nothing ? m.newton : Int32(refine),
        certify === nothing ? m.bisect : Int32(certify), σtol === nothing ? m.σtol : T(σtol))
    return p
end

"""
    run!(plan, D, c = ())

Run the sweep into the plan's device buffers (synchronizes). `c` is the tuple
of constants passed to `D(λ, p, c)` (converted to the plan's `T`); changing
`c` between calls needs no re-upload (slider-driven real-time use).
"""
function run!(p::SweepPlan{T, N}, D0::F, c = ()) where {T, N, F}
    be = p.backend
    D = D0 isa CharFn ? D0 : CharFn(D0)
    cT = map(evaltype(p.mp), Tuple(c))
    meth = Val(p.method)
    nr = Val(N)
    n = Int32(p.npts)
    if p.schedule === :pixel
        k = k_pixel!(be, p.workgroup)
        k(p.Zraw, p.sigma, p.omega, p.steps, p.flags, D, cT, p.points, n, p.mp, meth, nr;
            ndrange = p.npts)
    elseif p.schedule === :strided
        k = k_strided!(be, p.workgroup)
        k(p.Zraw, p.sigma, p.omega, p.steps, p.flags, D, cT, p.points, n, p.mp, meth, nr,
            Int32(p.lanes), Int32(p.stride); ndrange = p.lanes)
    else
        fill!(p.counter, Int32(0))
        k = k_queue!(be, p.workgroup)
        k(p.Zraw, p.sigma, p.omega, p.steps, p.flags, D, cT, p.points, n, p.mp, meth, nr, p.counter;
            ndrange = p.lanes)
    end
    KernelAbstractions.synchronize(be)
    return p
end

"""
    fetch_result(plan) -> NamedTuple of host vectors (one entry per point)

`Z` (Int, -1 where the march failed), `Zraw`, `sigma` (dominant tracked root,
NaN if none), `omega`, `steps`, `evals` (D evaluations per point), and
`flags`: bit 1 = the march failed (step underflow / `maxsteps`), bit 2 = a
root closer to the line than the precision can resolve was counted by its
side (a boundary-grazing point: its count is a decision, not a measurement),
bit 3 = integer residual |Zraw - round(Zraw)| > 0.25 (a root on the line).
"""
function fetch_result(p::SweepPlan)
    Zraw = Array(p.Zraw)
    steps = Array(p.steps)
    Z = map(z -> isfinite(z) ? round(Int, z) : -1, Zraw)
    return (Z = Z, Zraw = Zraw, sigma = Array(p.sigma), omega = Array(p.omega),
        steps = steps, evals = 1 .+ evals_per_step(Val(p.method)) .* Int.(steps),
        flags = Array(p.flags))
end

"""
    sweep(D, points; c = (), kw...) -> NamedTuple

One-shot convenience: `plan_sweep` + `run!` + `fetch_result`, plus the wall
time `t` of the compute (kernel + sync, excluding upload and download).
"""
function sweep(D, points; c = (), kw...)
    p = plan_sweep(points; kw...)
    T = evaltype(p.mp)
    check_eltype(D, first(points), c, T) || @warn "D(λ, p, c) promotes $(T) to a wider " *
        "type -- use constants from `c` or $(T) literals (e.g. 0.5f0) for full GPU speed" maxlog = 1
    t = @elapsed run!(p, D, c)
    return merge(fetch_result(p), (t = t, plan = p))
end

"""
    chart(D, xr, yr, nx, ny; c = (), kw...) -> NamedTuple

The 2-D special case: sweep the `nx × ny` grid over `xr × yr` and return
`nx × ny` matrices (`Z`, `Zraw`, `sigma`, `omega`, `steps`, `evals`) with the
axes `xs`, `ys`.
"""
function chart(D, xr, yr, nx::Integer, ny::Integer; c = (), kw...)
    xs = range(xr[1], xr[2]; length = nx)
    ys = range(yr[1], yr[2]; length = ny)
    r = sweep(D, grid_points(xs, ys); c = c, kw...)
    m(v) = reshape(v, nx, ny)
    return (Z = m(r.Z), Zraw = m(r.Zraw), sigma = m(r.sigma), omega = m(r.omega),
        steps = m(r.steps), evals = m(r.evals), flags = m(r.flags), xs = xs, ys = ys,
        t = r.t, plan = r.plan)
end

@kernel function k_grid!(Pts, x0, dx, y0, dy, nx)
    k = @index(Global, Linear)
    i = (k - 1) % nx
    j = (k - 1) ÷ nx
    @inbounds Pts[k] = (x0 + i * dx, y0 + j * dy)
end

"""
    plan_grid(xr, yr, nx, ny; kw...) -> SweepPlan

A plan for the `nx × ny` chart over `xr × yr` (first axis fastest, as
`grid_points`) whose points are generated ON THE DEVICE: no host array and no
upload, which matters at 4K/8K (33 M points). Same keywords as `plan_sweep`.
Change the axis ranges later with [`regrid!`](@ref), without reallocating.
"""
function plan_grid(xr, yr, nx::Integer, ny::Integer; backend = CPU(), T::Type = Float32,
                   lanes::Union{Nothing, Integer} = nothing, kw...)
    p = plan_sweep(((xr[1], yr[1]), (xr[2], yr[2])); backend, T, lanes, kw...)
    n = nx * ny
    n < typemax(Int32) || error("too many points for Int32 indexing")
    alloc(S) = KernelAbstractions.zeros(backend, S, n)
    p.npts = n
    p.points = KernelAbstractions.allocate(backend, NTuple{2, T}, n)
    p.Zraw, p.sigma, p.omega = alloc(T), alloc(T), alloc(T)
    p.steps, p.flags = alloc(Int32), alloc(Int8)
    nt = Threads.nthreads()
    cpu = backend isa CPU
    p.lanes = min(something(lanes, cpu ? (p.schedule === :queue ? nt : 8 * nt) : 1 << 18), n)
    p.stride = coprime_stride(n)
    return regrid!(p, xr, yr, nx, ny)
end

"""
    regrid!(plan, xr, yr, nx, ny) -> plan

Refill the points of a [`plan_grid`](@ref) plan with the `nx × ny` grid over
`xr × yr` on the device (`nx * ny` must equal the plan's point count).
"""
function regrid!(p::SweepPlan{T}, xr, yr, nx::Integer, ny::Integer) where {T}
    nx * ny == length(p.points) || error("regrid!: the plan holds $(length(p.points)) points")
    p.npts = nx * ny
    dx = nx > 1 ? T((xr[2] - xr[1]) / (nx - 1)) : zero(T)
    dy = ny > 1 ? T((yr[2] - yr[1]) / (ny - 1)) : zero(T)
    k_grid!(p.backend, 256)(p.points, T(xr[1]), dx, T(yr[1]), dy, Int32(nx); ndrange = p.npts)
    KernelAbstractions.synchronize(p.backend)
    return p
end

"""
    recheck_flagged!(plan, rplan, D, c = ()) -> n

Device-side second pass: gather every flagged point of `plan` (`flags != 0`)
into `rplan` -- a plan of at least the same capacity, typically Float32 with
production settings -- run it there, and scatter `Zraw`, `sigma`, `omega`,
`flags` back into `plan`. Nothing crosses the bus except the count `n`.
`rplan` must use the `:pixel` schedule (the GPU default).
"""
function recheck_flagged!(p::SweepPlan, rp::SweepPlan, D, c = ())
    idx = findall(!=(Int8(0)), p.flags)
    n = length(idx)
    n == 0 && return 0
    n <= length(rp.points) || error("recheck_flagged!: rplan is too small")
    cap = rp.npts
    rp.npts = n
    view(rp.points, 1:n) .= view(p.points, idx)
    run!(rp, D, c)
    view(p.Zraw, idx) .= view(rp.Zraw, 1:n)
    view(p.sigma, idx) .= view(rp.sigma, 1:n)
    view(p.omega, idx) .= view(rp.omega, 1:n)
    view(p.flags, idx) .= view(rp.flags, 1:n)
    KernelAbstractions.synchronize(p.backend)
    rp.npts = cap
    return n
end

"""
    recheck!(res, D, points; c = (), mask = nothing, T = Float64, kw...) -> n

Re-run a subset of the points -- by default every flagged one (`res.flags .!= 0`:
failed marches and boundary-grazing points) -- in precision `T` and patch the
results into `res` in place (`Z`, `Zraw`, `sigma`, `omega`, `flags`). The
natural second pass after a fast Float32 GPU sweep: the flagged points are few,
so they can run in Float64 on the CPU (`backend = CPU()`) or the GPU. Pass
`mask` (same shape as `res.Z`) to choose the points yourself, e.g. the
pixels next to a count change. Returns the number of points rechecked.
"""
function recheck!(res, D, points; c = (), mask = nothing, T::Type = Float64, kw...)
    m = vec(something(mask, res.flags .!= 0))
    idx = findall(m)
    isempty(idx) && return 0
    pts = collect(points)[idx]
    r = sweep(D, pts; c = c, T = T, kw...)
    for (k, i) in enumerate(idx)
        res.Z[i] = r.Z[k]
        res.Zraw[i] = r.Zraw[k]
        res.sigma[i] = r.sigma[k]
        res.omega[i] = r.omega[k]
        res.flags[i] = r.flags[k]
    end
    return length(idx)
end

"""
    colour_field(res; σcap = -1.5, Zcap = 6)

The paper's interpolable colouring: the dominant σ estimate inside the stable
domain (`Z == 0`), the capped integer count outside.
"""
function colour_field(res; σcap = -1.5, Zcap = 6)
    return map(res.Z, res.sigma) do z, s
        z == 0 ? (isfinite(s) ? max(s, σcap) : σcap) : Float64(min(z, Zcap))
    end
end

"""
    simulate_schedule(steps; warp = 32, lanes = 4096, schedule = :queue) -> efficiency

Idealized SIMT utilisation of a schedule given the per-point step counts
(every lane of a warp executes one step per iteration; a warp lives until its
slowest lane finishes). `:pixel_rowmajor` / `:pixel_random`: one point per
thread in the given / shuffled order. `:strided` and `:queue`: `lanes`
persistent lanes refilled in-loop (`:queue` assumes greedy lanes of equal
speed). Efficiency = useful lane-steps / issued lane-steps.
"""
function simulate_schedule(steps::AbstractArray{<:Integer}; warp::Int = 32,
                           lanes::Int = 4096, schedule::Symbol = :queue, P::Int = 0)
    s = Int.(vec(steps))
    n = length(s)
    total = sum(s)
    if schedule in (:pixel_rowmajor, :pixel_random)
        v = schedule === :pixel_random ? s[randperm_local(n)] : s
        issued = 0
        for i in 1:warp:n
            w = view(v, i:min(i + warp - 1, n))
            issued += warp * maximum(w)
        end
        return total / issued
    elseif schedule === :strided
        Pm = P == 0 ? coprime_stride(n) : P
        L = min(lanes, n)
        work = zeros(Int, L)
        for j in 0:(n - 1)
            work[(j % L) + 1] += s[mod(j * Pm, n) + 1] + 1      # +1: the seed evaluation
        end
        issued = 0
        for i in 1:warp:L
            issued += warp * maximum(view(work, i:min(i + warp - 1, L)))
        end
        return total / issued
    elseif schedule === :queue
        L = min(lanes, n)
        # greedy list scheduling: the lane that frees up first takes the next
        # point (min-heap of (finish time, lane), sorted array = valid heap)
        ht = [s[l] + 1 for l in 1:L]
        hl = collect(1:L)
        o = sortperm(ht)
        ht = ht[o]; hl = hl[o]
        for nxt in (L + 1):n
            ht[1] += s[nxt] + 1                    # root lane takes the point
            i = 1                                  # sift down
            while true
                c = 2i
                c > L && break
                (c < L && ht[c + 1] < ht[c]) && (c += 1)
                ht[i] <= ht[c] && break
                ht[i], ht[c] = ht[c], ht[i]
                hl[i], hl[c] = hl[c], hl[i]
                i = c
            end
        end
        finish_t = zeros(Int, L)
        finish_t[hl] = ht
        issued = 0
        for i in 1:warp:L
            issued += warp * maximum(view(finish_t, i:min(i + warp - 1, L)))
        end
        return total / issued
    else
        error("unknown schedule $schedule")
    end
end

# tiny local shuffle (avoid a Random dependency in the package)
function randperm_local(n)
    p = collect(1:n)
    x = UInt64(0x9E3779B97F4A7C15)
    for i in n:-1:2
        x ⊻= x << 13; x ⊻= x >> 7; x ⊻= x << 17
        j = Int(x % UInt64(i)) + 1
        p[i], p[j] = p[j], p[i]
    end
    return p
end

end # module
