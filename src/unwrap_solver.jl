# Discrete phase unwrapping ("unwrap" march) -- the counting back-end that does
# not integrate the phase derivative at all.
# This file is part of InterpolatedNyquist.jl
#
# The winding number only needs the END-POINT value of the continuous phase
# Φ(ω) = arg D(σ + iω). Between two samples a, b the increment is known exactly
# up to a multiple of 2π: Δ = angle(D_b / D_a). The march therefore only has to
# keep the samples dense enough that no branch is lost. With the exact phase
# derivative θ' = Im(D'/D) from the same dual-number evaluation, the trapezoid
# value h(θ'_a + θ'_b)/2 predicts the increment; the step is accepted when the
# observed and predicted increments agree within `tol` (radians), and the next
# step is h·0.9·(tol/err)^(1/3) (trapezoid error O(h³)), clamped to [0.2, 4]·h
# -- the step-size control of an embedded Runge-Kutta pair, with the exact
# increment in place of the higher-order solution. A skipped near-root peak
# shows up as a ≈ π mismatch and is refined. One evaluation of D per step.
#
# Minima of |D| between two accepted samples (sign change of d|D|/dω) are
# located on the cubic Hermite model of D built from the end values and
# derivatives (no extra evaluation), followed by one Newton step -> root
# estimate (σ_est, ω_est), as in the ODE back-end. A transition narrower than
# the smallest representable step is decided by the side of that root.

# scale-free sample: u = D/s with s = max(|Re D|, |Im D|)
@inline function _uw_sample(Dv::ComplexF64, Dw::ComplexF64)
    s = max(abs(real(Dv)), abs(imag(Dv)))
    u = Dv / s
    u2 = abs2(u)
    θ = (real(u) * imag(Dw) - imag(u) * real(Dw)) / (s * u2)
    g = real(u) * real(Dw) + imag(u) * imag(Dw)          # sign of d|D|/dω
    return u, θ, g, s
end

@inline function _uw_dphase(ua::ComplexF64, ub::ComplexF64)
    p = ub * conj(ua)
    return atan(imag(p), real(p))
end

# q = D / D_ω (robust); one Newton step from ω: σ_est = σ + Im q, ω_est = ω - Re q
@inline function _uw_newton(Dv::ComplexF64, Dw::ComplexF64)
    sw = max(abs(real(Dw)), abs(imag(Dw)))
    v = Dw / sw
    return (Dv / sw) * conj(v) / abs2(v)
end

@inline function _uw_hermite(A, Aw, B, Bw, h, t)
    t2 = t * t; t3 = t2 * t
    p = (2t3 - 3t2 + 1) * A + ((t3 - 2t2 + t) * h) * Aw + (-2t3 + 3t2) * B + ((t3 - t2) * h) * Bw
    dp = ((6t2 - 6t) / h) * (A - B) + (3t2 - 4t + 1) * Aw + (3t2 - 2t) * Bw
    return p, dp
end

# minimum of |D| inside [a, a+h] (Illinois regula falsi on the Hermite model),
# then one Newton step; returns (depth, σ_est, ω_est), depth = NaN if untrusted
function _uw_dip(Da, Dwa, sa, Db, Dwb, sb, a, h, σ)
    sc = 1 / max(sa, sb)
    A, Aw, B, Bw = Da * sc, Dwa * sc, Db * sc, Dwb * sc
    tl, tr = 0.0, 1.0
    fl = real(A) * real(Aw) + imag(A) * imag(Aw)
    fr = real(B) * real(Bw) + imag(B) * imag(Bw)
    t = clamp(fl / (fl - fr), 0.01, 0.99)
    side = 0
    for _ in 1:4
        p, dp = _uw_hermite(A, Aw, B, Bw, h, t)
        f = real(p) * real(dp) + imag(p) * imag(dp)
        if f < 0
            tl, fl = t, f
            side == -1 && (fr /= 2)
            side = -1
        else
            tr, fr = t, f
            side == 1 && (fl /= 2)
            side = 1
        end
        den = fr - fl
        t = clamp(den != 0 ? (tl * fr - tr * fl) / den : (tl + tr) / 2, tl, tr)
    end
    p, dp = _uw_hermite(A, Aw, B, Bw, h, t)
    q = _uw_newton(p, dp)
    ωm = a + t * h
    trusted = abs2(q) <= max(σ^2 + ωm^2, 1.0)       # trust region, as in refine_roots
    return (trusted ? abs(p) / sc : NaN), σ + imag(q), ωm - real(q)
end

# depth-sorted N-slot buffer of tracked minima
@inline function _uw_insert!(dd, ds, dw, d, s, w)
    (isfinite(d) && isfinite(s) && isfinite(w)) || return nothing
    N = length(dd)
    k = 1
    while k <= N && dd[k] <= d
        k += 1
    end
    k > N && return nothing
    for j in N:-1:k+1
        dd[j] = dd[j-1]; ds[j] = ds[j-1]; dw[j] = dw[j-1]
    end
    dd[k] = d; ds[k] = s; dw[k] = w
    return nothing
end

@inline function _uw_eval(D_func::NyquistWrapper{P}, p::P, σ, ω) where {P}
    dual_ω = ForwardDiff.Dual{StandardTag}(ω, 1.0)
    res = D_func(σ + 1im * dual_ω, p)
    rv, iv = real(res), imag(res)
    return ComplexF64(ForwardDiff.value(rv), ForwardDiff.value(iv)),
           ComplexF64(ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1))
end

"""
    _unwrap_march(D_func, p, σ, N; ω_max, tol, h0, hrel, hmax, ωband, maxsteps, ω0)

The march itself. Returns `(Φ, ok, evals, depths, σs, ωs)` with Φ the
unwrapped phase increment over [ω0, ω_max] and the N deepest tracked minima.
"""
function _unwrap_march(D_func::NyquistWrapper{P}, p::P, σ, N::Int; ω_max = 1e5, tol = 0.3,
                       h0 = 1e-2, hrel = 1.0, hmax = Inf, ωband = 0.0, maxsteps = 200_000,
                       ω0 = 1e-9) where {P}
    dd = fill(Inf, N); ds = fill(NaN, N); dw = fill(NaN, N)
    ω = Float64(ω0)
    Da, Dwa = _uw_eval(D_func, p, σ, ω)
    ua, θa, ga, sa = _uw_sample(Da, Dwa)
    evals = 1
    (isfinite(θa) && sa > 0) || return (NaN, false, evals, dd, ds, dw)
    if ga > 0 && N > 0                            # |D| grows from ω = 0: minimum at 0
        q = _uw_newton(Da, Dwa)
        abs2(q) <= max(σ^2, 1.0) && _uw_insert!(dd, ds, dw, sa * abs(ua), σ + imag(q), 0.0)
    end
    Φ = 0.0
    h = Float64(h0)
    floor_h(w) = 8 * eps(Float64) * max(w, 1.0)
    for _ in 1:maxsteps
        hh = min(h, hrel * max(ω, 1.0))
        ω < ωband && (hh = min(hh, hmax))
        last = hh >= ω_max - ω
        hh = last ? ω_max - ω : hh
        b = last ? Float64(ω_max) : ω + hh
        Db, Dwb = _uw_eval(D_func, p, σ, b)
        evals += 1
        ub, θb, gb, sb = _uw_sample(Db, Dwb)
        Δ = _uw_dphase(ua, ub)
        valid = isfinite(θb) && sb > 0 && isfinite(Δ)
        err = abs(Δ - hh * (θa + θb) / 2)
        isfinite(err) || (err = Inf)
        accept = valid && err <= tol
        fac = clamp(0.9 * cbrt(tol / max(err, floatmin(Float64))), 0.2, 4.0)
        hn = hh * fac
        if !accept && valid && hn <= floor_h(ω)
            # sub-resolution transition: decide the branch by the root side
            q = _uw_newton(Db, Dwb)
            right = imag(q) > 0
            if abs(Δ) > π / 2
                Δ = (right && Δ > 0) ? Δ - 2π : ((!right && Δ < 0) ? Δ + 2π : Δ)
            end
            accept = true
            hn = hh
        end
        if accept
            if N > 0 && ga < 0 && gb > 0
                d, se, we = _uw_dip(Da, Dwa, sa, Db, Dwb, sb, ω, hh, σ)
                _uw_insert!(dd, ds, dw, d, se, we)
            end
            Φ += Δ
            ω = b
            Da, Dwa, ua, θa, ga, sa = Db, Dwb, ub, θb, gb, sb
            last && return (Φ, true, evals, dd, ds, dw)
        elseif hn <= floor_h(ω)
            return (NaN, false, evals, dd, ds, dw)
        end
        h = hn
    end
    return (NaN, false, evals, dd, ds, dw)
end

"""
    calculate_unstable_roots_unwrap(D_func, p, σ=0.0; n_roots_to_track=1, ω_max=1e5,
        tol=0.3, n_power_max=nothing, hmax=Inf, ωband=0.0, refinement_method=:Linear, ...)

Counting by **discrete phase unwrapping**: `arg D(σ+iω)` is followed on an
adaptive frequency grid and its exact increments `angle(D_b/D_a)` are summed;
nothing is integrated. The step control compares each observed increment with
the trapezoid value of the exact phase derivative and rejects a step whose
mismatch exceeds `tol` (radians) -- a skipped near-root peak costs ≈ π and is
refined. One evaluation of `D` per step (typically 30–130 per point versus
thousands for the phase ODE), and the same tracked-minimum root estimates as
[`calculate_unstable_roots_direct`](@ref). Returns the same tuples:
`(Z, Z_raw)` for `n_roots_to_track = 0`, `(Z, Z_raw, min_D, σ_est, ω_crit)` for 1
(the deepest tracked |D| minimum), vectors of length N otherwise.

Write `D` as an entire function when possible: a rational `D` places a pole next
to every lightly damped mode, and a pole–zero pair can wind the phase by -2π
inside one step (multiply out the denominators; stable poles do not change the
count). Where chains of roots run close to the axis (e.g. regenerative delays),
cap the step with `hmax ≈ π/(2τ_max)` for `ω < ωband`.
"""
function calculate_unstable_roots_unwrap(@nospecialize(D_func), p::P, σ::S = 0.0;
    n_roots_to_track = 1, ω_max = 1e5, tol = 0.3, n_power_max = nothing, h0 = 1e-2,
    hrel = 1.0, hmax = Inf, ωband = 0.0, maxsteps = 200_000,
    refinement_method = :Linear, refinement_steps = 4, refinement_degree = 3) where {P, S}
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    return _unwrap_impl(wrapped_D, p, Float64(σ), n_roots_to_track; ω_max, tol, n_power_max,
        h0, hrel, hmax, ωband, maxsteps, refinement_method, refinement_steps, refinement_degree)
end

function _unwrap_impl(D_func::NyquistWrapper{P}, p::P, σ::Float64, N::Int; ω_max = 1e5,
    tol = 0.3, n_power_max = nothing, h0 = 1e-2, hrel = 1.0, hmax = Inf, ωband = 0.0,
    maxsteps = 200_000, refinement_method = :Linear, refinement_steps = 4,
    refinement_degree = 3) where {P}
    Φ, ok, _, dd, ds, dw = _unwrap_march(D_func, p, σ, max(N, 0); ω_max, tol, h0, hrel,
        hmax, ωband, maxsteps)
    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = ok ? n_pow / 2 - Φ / π : NaN
    zi = _safe_round_count(Z_raw)
    N == 0 && return zi, Z_raw
    refine(r) = refinement_method == :Linear ? r :
        refine_roots(D_func, p, r; method = refinement_method, steps = refinement_steps,
            degree = refinement_degree)
    if N == 1
        r = refine(ComplexF64(ds[1], dw[1]))
        return zi, Z_raw, dd[1], real(r), imag(r)
    end
    rs = [refine(ComplexF64(ds[j], dw[j])) for j in 1:N]
    return zi, Z_raw, copy(dd), real.(rs), imag.(rs)
end

"""
    calculate_unstable_roots_unwrap_p_vec(D_func, params_vec; n_roots_to_track=1, σ=0.0,
        ω_max=1e5, tol=0.3, n_power_max=nothing, parameter_independent_nmax=true, kwargs...)

Threaded sweep of [`calculate_unstable_roots_unwrap`](@ref) over a parameter
vector (same outputs as [`calculate_unstable_roots_p_vec`](@ref)).
"""
function calculate_unstable_roots_unwrap_p_vec(@nospecialize(D_func), params_vec::AbstractVector{P};
    n_roots_to_track = 1, σ = 0.0, ω_max = 1e5, tol = 0.3, n_power_max = nothing,
    parameter_independent_nmax = true, kwargs...) where {P}
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    n = length(params_vec)
    n_pow = n_power_max !== nothing ? Float64(n_power_max) :
        (parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ) : nothing)
    N = n_roots_to_track
    Zs = zeros(Int, n); Zr = zeros(Float64, n)
    if N == 0
        Threads.@threads for i in 1:n
            Zs[i], Zr[i] = _unwrap_impl(wrapped_D, params_vec[i], Float64(σ), 0;
                ω_max, tol, n_power_max = n_pow, kwargs...)
        end
        return Zs, Zr
    elseif N == 1
        md = zeros(n); se = zeros(n); wc = zeros(n)
        Threads.@threads for i in 1:n
            Zs[i], Zr[i], md[i], se[i], wc[i] = _unwrap_impl(wrapped_D, params_vec[i],
                Float64(σ), 1; ω_max, tol, n_power_max = n_pow, kwargs...)
        end
        return Zs, Zr, md, se, wc
    else
        md = [zeros(N) for _ in 1:n]; se = [zeros(N) for _ in 1:n]; wc = [zeros(N) for _ in 1:n]
        Threads.@threads for i in 1:n
            Zs[i], Zr[i], md[i], se[i], wc[i] = _unwrap_impl(wrapped_D, params_vec[i],
                Float64(σ), N; ω_max, tol, n_power_max = n_pow, kwargs...)
        end
        return Zs, Zr, md, se, wc
    end
end
