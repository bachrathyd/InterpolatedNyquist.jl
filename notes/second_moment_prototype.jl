# Prototype: second-moment integrals on top of the unwrap march
# (see notes/stochastic_second_moment.md, Secs. 3 and 8).
#
# The march is the one of src/unwrap_solver.jl (same step control, same phase
# bookkeeping, one dual-number evaluation per step). The model returns the
# characteristic function D together with numerators n_j that share D as their
# denominator; along the accepted nodes the march also integrates
#
#     I_j = 1/(2π) ∫_{-∞}^{∞} |n_j(s + iω) / D(s + iω)|² dω
#
# with the Hermite-corrected trapezoid rule (f' from the same dual number) and an
# analytic power-law tail beyond ω_max. Real systems: |G(s - iω)| = |G(s + iω)|,
# so only ω ≥ 0 is marched.

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

# f_j = |n_j|²/|D|² and its ω-derivative
@inline function _sm_f(v, w)
    D, Dw = v[1], w[1]
    a2 = abs2(D)
    dlog = 2 * real(conj(D) * Dw) / a2
    f = map((n, nw) -> abs2(n) / a2, Base.tail(v), Base.tail(w))
    fw = map((n, nw, fj) -> 2 * real(conj(n) * nw) / a2 - fj * dlog, Base.tail(v), Base.tail(w), f)
    return f, fw
end


# step helpers as top-level functions (closures over the reassigned hh / ω would be boxed)
@inline _sm_relerr(Ij, x, y, xw, yw, hh, sc) =
    abs(hh^2 / 12 * (xw - yw)) / max(sc, Ij + hh / 2 * (x + y), floatmin(Float64))
@inline _sm_add(Ij, x, y, xw, yw, hh) = Ij + hh / 2 * (x + y) + hh^2 / 12 * (xw - yw)
# tail f ≈ c ω^(-k):  ∫_ω^∞ f = f ω / (k - 1)
@inline _sm_tail(x, xw, ω) = (k = -ω * xw / x; k > 1.05 ? x * ω / (k - 1) : NaN)

"""
    second_moment_unwrap(model, p, s = 0.0; n_power_max, ω_max = 1e5, tol = 0.3, ...)

`model(λ, p)` returns `(D, n_1, …, n_J)`. Marches λ = s + iω, ω ∈ [ω0, ω_max]
and returns `(Z, I, evals)`: the unstable-root count of D right of Re λ = s, the
tuple `I_j = (1/2π)∫|n_j/D|² dω` (valid when Z == 0), and the number of evaluations.
A step is accepted when the phase test (`tol`, as in the unwrap march) and the
integral test (`rtol_I`: plain-trapezoid error estimate relative to the running
integral, or to `Iscale` -- a prior estimate of `I` -- when given) both pass;
`rtol_I = Inf` reproduces the deterministic march's nodes.
"""
function second_moment_unwrap(model, p, s = 0.0; n_power_max, ω_max = 1e5, tol = 0.3, rtol_I = 1e-3,
                              Iscale = nothing,
                              h0 = 1e-2, hrel = 1.0, hmax = Inf, ωband = 0.0,
                              maxsteps = 200_000, ω0 = 1e-9)
    ω = Float64(ω0)
    v, w = _sm_eval(model, p, s, ω)
    ua, θa, _, sa = IN._uw_sample(v[1], w[1])
    fa, fwa = _sm_f(v, w)
    I = map(zero, fa)
    scI = Iscale === nothing ? map(zero, fa) : map(x -> π * x, Iscale)   # same units as the running sum
    evals = 1
    Φ = 0.0
    h = Float64(h0)
    floor_h(x) = 8 * eps(Float64) * max(x, 1.0)
    for _ in 1:maxsteps
        hh = min(h, hrel * max(ω, 1.0))
        ω < ωband && (hh = min(hh, hmax))
        last = hh >= ω_max - ω
        hh = last ? ω_max - ω : hh
        b = last ? Float64(ω_max) : ω + hh
        v, w = _sm_eval(model, p, s, b)
        evals += 1
        ub, θb, _, sb = IN._uw_sample(v[1], w[1])
        Δ = IN._uw_dphase(ua, ub)
        valid = isfinite(θb) && sb > 0 && isfinite(Δ)
        err = abs(Δ - hh * (θa + θb) / 2)
        isfinite(err) || (err = Inf)
        # second criterion: the Hermite correction h²/12 (f'_a - f'_b) is the error
        # of the plain trapezoid, relative to the accumulated + local integral
        fb, fwb = _sm_f(v, w)
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
            # Hermite-corrected trapezoid, local error O(h^5)
            I = let h1 = hh; map((a, b, c, d, e) -> _sm_add(a, b, c, d, e, h1), I, fa, fb, fwa, fwb) end
            Φ += Δ
            ω = b
            ua, θa, sa, fa, fwa = ub, θb, sb, fb, fwb
            if last
                tail = let w1 = ω; map((a, b) -> _sm_tail(a, b, w1), fa, fwa) end
                I = map((Ij, t) -> (Ij + t) / π, I, tail)        # (1/2π)·2·∫_0^∞
                Z = round(Int, n_power_max / 2 - Φ / π)
                return Z, I, evals
            end
        elseif hn <= floor_h(ω)
            return -1, map(_ -> NaN, I), evals
        end
        h = hn
    end
    return -1, map(_ -> NaN, I), evals
end

"""
    second_moment(model, p, s = 0.0; n_power_max, rtol_I = 1e-3, kw...)

Two passes. Pass 1 is the deterministic unwrap march (`rtol_I = Inf`): it gives
the count `Z` and, on the same nodes, a coarse `I` (~10 % accurate). If `Z == 0`,
pass 2 re-marches with the integral test scaled by that estimate, so flat regions
where the integrand is negligible get no extra nodes. Returns `(Z, I, evals_total)`.
"""
function second_moment(model, p, s = 0.0; n_power_max, rtol_I = 1e-3, kw...)
    Z, I1, ev1 = second_moment_unwrap(model, p, s; n_power_max, rtol_I = Inf, kw...)
    Z == 0 || return Z, map(_ -> NaN, I1), ev1
    Z2, I2, ev2 = second_moment_unwrap(model, p, s; n_power_max, rtol_I, Iscale = I1, kw...)
    return Z2, I2, ev1 + ev2
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
