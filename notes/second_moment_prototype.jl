# Prototype: second-moment integrals on top of the unwrap march
# (see notes/stochastic_second_moment.md, Secs. 3, 8 and 10).
#
# The march is the one of src/unwrap_solver.jl (same step control, same phase
# bookkeeping, same tracked |D| minima, one dual-number evaluation per step). The
# model returns the characteristic function D together with numerators n_j that
# share D as their denominator; along the accepted nodes the march also integrates
#
#     I_j = 1/(2π) ∫_{-∞}^{∞} |n_j(s + iω) / D(s + iω)|² dω
#
# with the Hermite-corrected trapezoid rule (f' from the same dual number) and an
# analytic power-law tail beyond ω_max. Real systems: |G(s - iω)| = |G(s + iω)|,
# so only ω ≥ 0 is marched.
#
# Peak subtraction (pass 2): near a root λ_r close to the line, n/D ≈ res/(iω - λ_r)
# with res = n(λ_r)/D_λ(λ_r). The pole pair L(ω) = |res|²(1/|iω-λ_r|² + 1/|iω-λ̄_r|²)
# is subtracted from f and its integral over [ω0, ω_max] added back in closed form
# (arctan). The subtraction is exact algebra whatever λ_r and res are; their accuracy
# only decides how smooth the remainder is.

using InterpolatedNyquist
const IN = InterpolatedNyquist
const FD = IN.ForwardDiff

struct SMTag end

@inline function _sm_eval(model, p, s, ω)
    d = FD.Dual{SMTag}(ω, 1.0)
    r = model(s + 1im * d, p)
    v = map(z -> ComplexF64(FD.value(real(z)), FD.value(imag(z))), r)
    w = map(z -> ComplexF64(FD.partials(real(z), 1), FD.partials(imag(z), 1)), r)
    return v, w
end

# a subtracted pole pair: root λ (Re λ < s) and the |res_j|² of every numerator
struct SMPole{J}
    λ::ComplexF64
    a::NTuple{J, Float64}
end

# f_j = |n_j|²/|D|² - Σ_poles L_j and the ω-derivative
@inline function _sm_f(v, w, s, ω, poles)
    D, Dw = v[1], w[1]
    a2 = abs2(D)
    dlog = 2 * real(conj(D) * Dw) / a2
    f = map((n, nw) -> abs2(n) / a2, Base.tail(v), Base.tail(w))
    fw = map((n, nw, fj) -> 2 * real(conj(n) * nw) / a2 - fj * dlog, Base.tail(v), Base.tail(w), f)
    for P in poles
        σr = real(P.λ) - s; ωr = imag(P.λ)
        u1 = ω - ωr; u2 = ω + ωr; q1 = u1^2 + σr^2; q2 = u2^2 + σr^2
        L = 1 / q1 + 1 / q2
        Lw = -2u1 / q1^2 - 2u2 / q2^2
        f = map((x, a) -> x - a * L, f, P.a)
        fw = map((x, a) -> x - a * Lw, fw, P.a)
    end
    return f, fw
end

# ∫_{a}^{b} of the subtracted pole pairs (closed form)
function _sm_pole_integral(poles, s, a, b, ::Val{J}) where {J}
    acc = ntuple(_ -> 0.0, J)
    for P in poles
        σr = abs(real(P.λ) - s); ωr = imag(P.λ)
        F(ω) = (atan((ω - ωr) / σr) + atan((ω + ωr) / σr)) / σr
        acc = map((x, aj) -> x + aj * (F(b) - F(a)), acc, P.a)
    end
    return acc
end

# step helpers as top-level functions (closures over the reassigned hh / ω would be boxed)
@inline _sm_relerr(Ij, x, y, xw, yw, hh, sc) =
    abs(hh^2 / 12 * (xw - yw)) / max(sc, abs(Ij + hh / 2 * (x + y)), floatmin(Float64))
@inline _sm_add(Ij, x, y, xw, yw, hh) = Ij + hh / 2 * (x + y) + hh^2 / 12 * (xw - yw)
# tail f ≈ c ω^(-k):  ∫_ω^∞ f = f ω / (k - 1)
@inline _sm_tail(x, xw, ω) = (k = -ω * xw / x; k > 1.05 ? x * ω / (k - 1) : NaN)

"""
    second_moment_unwrap(model, p, s = 0.0; n_power_max, ω_max = 1e5, tol = 0.3,
                         rtol_I = 1e-3, Iscale = nothing, poles = (), n_track = 0, ...)

`model(λ, p)` returns `(D, n_1, …, n_J)`. Marches λ = s + iω, ω ∈ [ω0, ω_max] and
returns `(Z, I, evals, tracked)`: the unstable-root count of D right of Re λ = s,
the tuple `I_j = (1/2π)∫|n_j/D|² dω` (valid when Z == 0), the number of evaluations
and the `n_track` deepest |D| minima `(depth, σ_est, ω_est)` (as `_unwrap_march`).
A step is accepted when the phase test (`tol`, as in the unwrap march) and the
integral test (`rtol_I`: plain-trapezoid error estimate relative to the running
integral, or to `Iscale` -- a prior estimate of `I` -- when given) both pass;
`rtol_I = Inf` reproduces the deterministic march's nodes. `poles` are subtracted.
"""
function second_moment_unwrap(model, p, s = 0.0; n_power_max, ω_max = 1e5, tol = 0.3,
                              rtol_I = 1e-3, Iscale = nothing, poles = (), n_track = 0,
                              h0 = 1e-2, hrel = 1.0, hmax = Inf, ωband = 0.0,
                              maxsteps = 200_000, ω0 = 1e-9)
    dd = fill(Inf, n_track); ds = fill(NaN, n_track); dw = fill(NaN, n_track)
    ω = Float64(ω0)
    v, w = _sm_eval(model, p, s, ω)
    ua, θa, ga, sa = IN._uw_sample(v[1], w[1])
    Da, Dwa = v[1], w[1]
    fa, fwa = _sm_f(v, w, s, ω, poles)
    I = map(zero, fa)
    J = length(I)
    scI = Iscale === nothing ? map(zero, fa) : map(x -> π * x, Iscale)   # same units as the running sum
    evals = 1
    Φ = 0.0
    h = Float64(h0)
    floor_h(x) = 8 * eps(Float64) * max(x, 1.0)
    fail = (-1, map(_ -> NaN, I), evals, (dd, ds, dw))
    for _ in 1:maxsteps
        hh = min(h, hrel * max(ω, 1.0))
        ω < ωband && (hh = min(hh, hmax))
        last = hh >= ω_max - ω
        hh = last ? ω_max - ω : hh
        b = last ? Float64(ω_max) : ω + hh
        v, w = _sm_eval(model, p, s, b)
        evals += 1
        ub, θb, gb, sb = IN._uw_sample(v[1], w[1])
        Δ = IN._uw_dphase(ua, ub)
        valid = isfinite(θb) && sb > 0 && isfinite(Δ)
        err = abs(Δ - hh * (θa + θb) / 2)
        isfinite(err) || (err = Inf)
        # second criterion: the Hermite correction h²/12 (f'_a - f'_b) is the error
        # of the plain trapezoid, relative to the accumulated + local integral
        fb, fwb = _sm_f(v, w, s, b, poles)
        errI = let h1 = hh; maximum(map((a, b, c, d, e, sc) -> _sm_relerr(a, b, c, d, e, h1, sc), I, fa, fb, fwa, fwb, scI)) end
        isfinite(errI) || (errI = Inf)
        accept = valid && err <= tol && errI <= rtol_I
        fac = clamp(0.9 * cbrt(min(tol / max(err, floatmin(Float64)),
                                   rtol_I / max(errI, floatmin(Float64)))), 0.2, 4.0)
        hn = hh * fac
        if !accept && valid && hn <= floor_h(ω)
            q = IN._uw_newton(v[1], w[1])
            right = imag(q) > 0
            if abs(Δ) > π / 2
                Δ = (right && Δ > 0) ? Δ - 2π : ((!right && Δ < 0) ? Δ + 2π : Δ)
            end
            accept = true
            hn = hh
        end
        if accept
            if n_track > 0 && ga < 0 && gb > 0
                d_, se, we = IN._uw_dip(Da, Dwa, sa, v[1], w[1], sb, ω, hh, s)
                IN._uw_insert!(dd, ds, dw, d_, se, we)
            end
            # Hermite-corrected trapezoid, local error O(h^5)
            I = let h1 = hh; map((a, b, c, d, e) -> _sm_add(a, b, c, d, e, h1), I, fa, fb, fwa, fwb) end
            Φ += Δ
            ω = b
            ua, θa, ga, sa, fa, fwa, Da, Dwa = ub, θb, gb, sb, fb, fwb, v[1], w[1]
            if last
                # tail of the full integrand (the pole pairs are integrated exactly up to ω_max)
                ffull, fwfull = _sm_f(v, w, s, ω, ())
                tail = let w1 = ω; map((a, b) -> _sm_tail(a, b, w1), ffull, fwfull) end
                Ipol = _sm_pole_integral(poles, s, ω0, ω_max, Val(J))
                I = map((Ij, t, pl) -> (Ij + t + pl) / π, I, tail, Ipol)   # (1/2π)·2·∫_0^∞
                Z = round(Int, n_power_max / 2 - Φ / π)
                return Z, I, evals, (dd, ds, dw)
            end
        elseif hn <= floor_h(ω)
            return fail
        end
        h = hn
    end
    return fail
end

# Newton polish of a root estimate on D (complex λ), D_λ = -i D_ω from the dual number
function _sm_polish(model, p, λ; steps = 8)
    for _ in 1:steps
        v, w = _sm_eval(model, p, real(λ), imag(λ))
        Dλ = -1im * w[1]
        δ = v[1] / Dλ
        isfinite(δ) || return λ, false
        λ -= δ
        abs(δ) < 1e-12 * max(1.0, abs(λ)) && return λ, true
    end
    return λ, true
end

function _sm_residues(model, p, λ)
    v, w = _sm_eval(model, p, real(λ), imag(λ))
    Dλ = -1im * w[1]
    return map(n -> abs2(n / Dλ), Base.tail(v))
end

"""
    second_moment(model, p, s = 0.0; n_power_max, rtol_I = 1e-3, subtract = false,
                  n_track = 4, kw...)

Two passes. Pass 1 is the deterministic unwrap march (`rtol_I = Inf`): it gives the
count `Z`, the `n_track` deepest |D| minima and, on the same nodes, a coarse `I`.
If `Z == 0`, the minima are polished by Newton to roots λ_r (kept when they are
left of the line, Re λ_r < s), their pole pairs are subtracted (`subtract = true`),
and pass 2 re-marches the remainder with the integral test scaled by the coarse `I`.
Returns `(Z, I, evals_total, roots)`.
"""
function second_moment(model, p, s = 0.0; n_power_max, rtol_I = 1e-3, subtract = false,
                       n_track = 4, kw...)
    Z, I1, ev1, (dd, ds, dw) = second_moment_unwrap(model, p, s; n_power_max, rtol_I = Inf,
                                                    n_track, kw...)
    Z == 0 || return Z, map(_ -> NaN, I1), ev1, ComplexF64[]
    J = length(I1)
    poles = SMPole{J}[]
    roots = ComplexF64[]
    evp = 0
    if subtract
        for k in eachindex(dd)
            (isfinite(dd[k]) && isfinite(ds[k]) && isfinite(dw[k])) || continue
            λ, ok = _sm_polish(model, p, ComplexF64(ds[k], dw[k])); evp += 9
            (ok && real(λ) < s && imag(λ) >= 0) || continue
            any(r -> abs(r - λ) < 1e-6 * max(1, abs(λ)), roots) && continue
            push!(roots, λ)
            push!(poles, SMPole{J}(λ, _sm_residues(model, p, λ)))
        end
    end
    Z2, I2, ev2, _ = second_moment_unwrap(model, p, s; n_power_max, rtol_I, Iscale = I1,
                                         poles = Tuple(poles), kw...)
    return Z2, I2, ev1 + evp + ev2, roots
end

"""
    ms_exponent(model, p; channel = 2, σ_lo = -2.0, σ_hi = 1.0, n_power_max, iters = 40, kw...)

Mean-square Lyapunov exponent λ_MS = max(2 max Re λ_det, λ*), r(λ*) = 1, by
bisection on the monotone predicate "Z(σ/2) == 0 and r(σ) < 1", where
r(σ) = I[channel] evaluated on the line Re s = σ/2 (one march per step).
Scalar noise channel (p = 1, one Wiener process).
"""
function ms_exponent(model, p; channel = 2, σ_lo = -2.0, σ_hi = 1.0, n_power_max, iters = 40, kw...)
    ok(σ) = (r = second_moment_unwrap(model, p, σ / 2; n_power_max, kw...);
             r[1] == 0 && r[2][channel] < 1)
    ok(σ_hi) || error("σ_hi too small")
    ok(σ_lo) && error("σ_lo too large")
    for _ in 1:iters
        mid = (σ_lo + σ_hi) / 2
        ok(mid) ? (σ_hi = mid) : (σ_lo = mid)
    end
    return (σ_lo + σ_hi) / 2
end
