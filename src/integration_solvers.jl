# Core logic for Method 2: Integration-based solvers
# This file is part of InterpolatedNyquist.jl

using OrdinaryDiffEq
using SciMLBase # Just in case you need base types explicitly

using ForwardDiff
using StaticArrays
# QuadGK and FunctionWrappers are imported in the main module

# The phase integrand Im(D'/D) is scale-invariant, but its naive evaluation
# overflows once |D|^2 exceeds floatmax (e.g. determinants of large matrices
# reach |D| ~ 1e250). This helper computes it overflow-safely; such points are
# astronomically far from any root, so skipping the min-|D| tracking there is
# exact.
@inline function _safe_darg(re_val, im_val, dre, dim)
    s = max(abs(re_val), abs(im_val), abs(dre), abs(dim))
    (isfinite(s) && s > 0) || return 0.0
    rs = re_val / s; is_ = im_val / s
    drs = dre / s; dis = dim / s
    d_sq_s = rs^2 + is_^2
    d_sq_s < 1e-300 && return 0.0
    return (dis * rs - drs * is_) / d_sq_s
end

# The counting formula presumes no characteristic root sits exactly ON the
# integration line. If that assumption is violated (e.g. a parameter point on a
# structural singularity, such as a delayed stiffness exactly cancelling the
# static one), Z_raw can come out non-finite. Rather than throwing an
# InexactError deep inside a threaded sweep, return the count -1 -- the
# "invalid" marker the boundary objectives already treat as such via
# `max(Z, 0)` -- and let the caller see the non-finite Z_raw.
@inline _safe_round_count(Z_raw) = isfinite(Z_raw) ? round(Int, Z_raw) : -1

# Standardized Tag for ForwardDiff specialization
struct NyquistTag end
const StandardTag = ForwardDiff.Tag{NyquistTag, Float64}

# Exact types for FunctionWrapper to prevent recompilation
const StandardDual = ForwardDiff.Dual{StandardTag, Float64, 1}
const ComplexDual = Complex{StandardDual}

"""
    NyquistWrapper{P}

Type-stable wrapper for D(λ, p) where λ is a Complex Dual and p is the parameter collection.
Using this wrapper prevents Julia from recompiling the solver logic when D_func is redefined.
"""
const NyquistWrapper{P} = FunctionWrapper{ComplexDual, Tuple{ComplexDual, P}}

"""
    get_n_power_max(D_func, p, σ=0.0; probe=1e8, decades=1.0, samples=8)

Estimates the leading (highest-power) exponent `n` of `D(λ)` for `|λ| → ∞`,
i.e. `D(λ) ~ c λ^n`, as the least-squares slope of `log|D(s)|` versus `log s`
over `samples` log-spaced points on the **positive real axis**
`s ∈ [probe/10^decades, probe]`.

Probing along the real axis is both the natural and the numerically robust
choice. `n` is needed for the phase contribution of the arc at infinity, and
that arc lies in the **right half-plane**, where every delayed term
`e^{-λτ}` (τ > 0) decays exponentially. On the real axis those terms are
therefore already negligible: `|D(s)| → |c| s^n` with a smooth algebraic
correction that decays like `1/s`, so the fitted slope converges as `O(1/probe)`
(and as `O(probe^(-1/2))` for fractional-order corrections).

The fitted slope is validated against a second window one decade lower (the
two must agree to 5%, which accepts the genuine `O(1/s)` drift but rejects
exponential growth), and the two windows are Richardson-extrapolated, removing
the leading `1/s` correction — the retarded-case convergence is therefore
effectively `O(1/probe^2)`. The validation window doubles the evaluation cost
and extends the `s`-range down to `probe/10^(2 decades)`, which `D` must
tolerate numerically.

Evaluating instead along the imaginary axis -- where `|e^{-iωτ}| = 1` -- makes
the delay terms superimpose a persistent ripple on `log|D|` that a log-spaced
fit aliases rather than averages out; that variant loses several orders of
accuracy for delayed velocity feedback and for neutral equations. For a
neutral system `D ~ λ^n (1 + Σ b_j e^{-λτ_j})` the real-axis probe returns the
integer `n`, which is exactly the value the counting formula needs; the
oscillating factor only contributes the bounded tail discussed in the
documentation. Genuinely non-integer orders (fractional systems) are returned
as floats.

`probe` is reduced automatically (by factors of ten) if `|D|` overflows there,
which happens for determinants of large matrices (`|D| ~ s^n` exceeds
`floatmax` at `s ≈ 10^(308/n)`). For very high orders the back-off caps `s`
below the asymptotic regime and the estimate degrades accordingly; pass
`n_power_max` explicitly to the solvers if the leading order is known.
"""
function get_n_power_max(@nospecialize(D_func), p::P, σ=0.0; probe=1e8, decades=1.0, samples=8) where P
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    return _get_n_power_max_impl(wrapped_D, p, σ; probe=probe, decades=decades, samples=samples)
end

# Least-squares slope of log|D| vs log(coordinate) over one decade-wide window.
# `axis = :real` walks the positive real axis, `:imag` the line σ + iω.
# Returns (slope, n_used_samples).
function _ls_slope(D_func::NyquistWrapper{P}, p::P, σ, hi, decades, samples, axis::Symbol) where P
    sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0; K = 0
    zero_dual = ForwardDiff.Dual{StandardTag}(0.0, 0.0)
    for t in range(log(hi) - decades * log(10.0), log(hi); length=samples)
        v = exp(t)
        λ = if axis === :real
            Complex(ForwardDiff.Dual{StandardTag}(v, 0.0), zero_dual)
        else
            Complex(ForwardDiff.Dual{StandardTag}(σ, 0.0), ForwardDiff.Dual{StandardTag}(v, 0.0))
        end
        res = D_func(λ, p)
        absD = hypot(ForwardDiff.value(real(res)), ForwardDiff.value(imag(res)))
        (isfinite(absD) && absD > 0) || continue
        y = log(absD)
        sx += t; sy += y; sxx += t * t; sxy += t * y; K += 1
    end
    K < 2 && return (NaN, K)
    denom = K * sxx - sx * sx
    denom > 0 || return (NaN, K)
    return ((K * sxy - sx * sy) / denom, K)
end

function _get_n_power_max_impl(D_func::NyquistWrapper{P}, p::P, σ=0.0; probe=1e8, decades=1.0, samples=8) where P
    # --- preferred: real-axis probe, backing off until |D| is representable ---
    s_hi = Float64(probe)
    for _ in 1:16
        n_hi, K = _ls_slope(D_func, p, σ, s_hi, decades, samples, :real)
        if isfinite(n_hi) && K >= samples ÷ 2
            # Validate the power-law assumption: for D ~ c s^n (1 + O(1/s)) the
            # slope must be nearly the same one window lower. It is NOT for
            # functions that are not polynomially bounded in the right half-plane
            # -- e.g. the cosh-type characteristic function of an elastic
            # continuum grows like e^s, whose "slope" scales with s itself --
            # and there the real-axis probe is meaningless.
            #
            # The tolerance must be loose enough to accept the genuine O(1/s)
            # drift (which is large in absolute terms for high-order
            # determinants: ~0.2 out of 58 for a 29-DOF FEM model) and tight
            # enough to reject exponential growth (where the two slopes differ
            # by an order of magnitude). 5% separates the two cases cleanly.
            n_lo, _ = _ls_slope(D_func, p, σ, s_hi / 10.0^decades, decades, samples, :real)
            if isfinite(n_lo) && abs(n_hi - n_lo) <= 0.05 * max(1.0, abs(n_hi))
                if s_hi < Float64(probe)
                    # Overflow back-off can push the fit window below the
                    # asymptotic regime (|D| ~ s^n hits floatmax at
                    # s ≈ 10^(308/n), i.e. s ≈ 10 for n ≈ 300, where the
                    # estimate is unusable). The validation above catches most
                    # such cases, but the user should know the probe moved.
                    @warn "get_n_power_max: |D| overflowed at the requested probe; " *
                        "fitted the leading order on s ∈ [$(s_hi / 10.0^decades), $(s_hi)] " *
                        "instead. For very high orders (n ≳ 150) this window is " *
                        "pre-asymptotic and the estimate degrades; pass n_power_max " *
                        "explicitly if the leading order is known." maxlog = 1
                end
                # Richardson: the windows differ by a factor 10^decades in s and
                # the leading correction decays like 1/s, so extrapolate it away.
                r = 10.0^decades
                return n_hi + (n_hi - n_lo) / (r - 1.0)
            end
            break   # not a power law on the real axis -> fall back
        end
        s_hi /= 10.0
    end

    # --- fallback: imaginary-axis fit ---
    # Used when |D| is not polynomially bounded in the right half-plane. There
    # |D(σ+iω)| is typically bounded and oscillatory, and the least-squares
    # slope recovers the effective order the counting formula needs (0 for a
    # cosh-type continuum). The delay ripple makes this less accurate than the
    # real-axis probe, which is why it is only the fallback -- and it is
    # subject to the same two-window drift validation (with a looser tolerance
    # matching the ripple amplitude), so that a pre-asymptotic fit cannot be
    # returned silently as a confident answer.
    n_first = NaN
    for ω_hi in (1e4, 1e3, 1e2, 1e1)
        n_est, K = _ls_slope(D_func, p, σ, ω_hi, 2.0, 4 * samples, :imag)
        (isfinite(n_est) && K >= samples) || continue
        isnan(n_first) && (n_first = n_est)
        n_lo, K2 = _ls_slope(D_func, p, σ, ω_hi / 10.0, 2.0, 4 * samples, :imag)
        if isfinite(n_lo) && K2 >= samples && abs(n_est - n_lo) <= 0.25 * max(1.0, abs(n_est))
            return n_est
        end
    end
    if isfinite(n_first)
        @warn "get_n_power_max: the leading-order fit did not pass slope-invariance " *
            "validation on either axis; returning the unvalidated imaginary-axis " *
            "estimate $(n_first). Pass n_power_max explicitly if the leading order " *
            "is known, and treat non-integer Z_raw values with suspicion." maxlog = 1
        return n_first
    end
    @warn "get_n_power_max: could not fit a leading order at all (|D| non-finite " *
        "or zero on every probe window); returning 0.0. Pass n_power_max " *
        "explicitly." maxlog = 1
    return 0.0
end

"""
    peak_skip_suspect(Z, σ_est, σ=0.0; h=1e-3)

Cross-check for the one failure mode the integer residual cannot see.

A frequency march that steps *over* a near-singular peak of the phase
integrand loses exactly `±π`, so `Z_raw` shifts by exactly one and stays just
as close to an integer as before: `|Z_raw - round(Z_raw)|` certifies the
quadrature, not the count. The tracked root, however, comes from the minima of
`|D|²` -- a mechanism independent of the accumulated phase -- so the two must
agree: a root right of the line (`σ_est > σ`) contradicts `Z == 0`.

Returns `true` when `sign(σ_est - σ)` disagrees with the predicate `Z == 0`
**and** the root is within `h` of the line.

The distance guard is essential, not cosmetic. Far from the line the tracked
minimum need not belong to the *dominant* root (see `n_roots_to_track`): deep
inside an unstable domain the roots responsible for `Z > 0` may sit far to the
right and leave no dip near the line, so a disagreement there is legitimate
and the unguarded test fires on a large fraction of a perfectly converged
chart. A peak can only be skipped where a root is close to the line, which is
exactly where this test is sharp. Choose `h` a few orders of magnitude below
the spread of `σ_est` over the chart.
"""
@inline function peak_skip_suspect(Z::Integer, σ_est::Real, σ::Real=0.0; h::Real=1e-3)
    isfinite(σ_est) || return false
    d = σ_est - σ
    return ((d < 0) != (Z == 0)) && abs(d) < h
end

# ---------------------------------------------------------------------------
# Peak repair (opt-in, `peak_repair = true`): the cure for what
# `peak_skip_suspect` detects. A rootfinding callback stops the march at each
# minimum of |D(σ+iω)|² -- the exact center of the phase integrand's
# near-singular peak when a root lies close to the σ-line -- then the phase
# increment of the step that crossed the minimum is recomputed with quadrature
# forced into the located peak, and the march restarts at the peak, where step
# control resolves its right half on its own.
# ---------------------------------------------------------------------------

# Quadrature over [a, b] with an integrand peak KNOWN to sit at the right
# endpoint b. Geometric breakpoints stacked toward b force subdivision into
# the peak: a plain adaptive call silently accepts whenever the first rule's
# nodes all miss the peak (node spacing ~5e-3 of the interval), which happens
# for any peak narrower than that -- the same estimator blindness that lets
# fixed-tolerance quadrature miscount near a boundary in the first place.
function _endpoint_ladder_quad(g, a::Float64, b::Float64, rtol, atol)
    L = b - a
    pts = Float64[a]
    k = 1
    while L * 2.0^-k > max(1e-13 * abs(b), 1e-300) && k < 46
        push!(pts, b - L * 2.0^-k)
        k += 1
    end
    push!(pts, b)
    I, _ = quadgk(g, pts...; rtol = rtol, atol = atol)
    return I
end

# Tracking-free evaluation of (phase integrand, d|D|²/dω) at ω. The repair
# quadrature and the callback's condition checks sample ω out of march order,
# so they must not touch the minimum-tracking state of the Val{1}/Val{N}
# integrands.
function _integrand_and_dsq_deriv(D_func::NyquistWrapper{P}, p::P, σ, ω::Float64) where {P}
    pure_ω = max(ω, 1e-9)
    dual_ω = ForwardDiff.Dual{StandardTag}(pure_ω, 1.0)
    res = D_func(σ + 1im * dual_ω, p)
    rv, iv = real(res), imag(res)
    re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
    dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)
    d_sq = re_val^2 + im_val^2
    g = if d_sq < 1e-20 || !isfinite(d_sq) || !(isfinite(dre) && isfinite(dim))
        _safe_darg(re_val, im_val, dre, dim)
    else
        (dim * re_val - dre * im_val) / d_sq
    end
    return g, 2 * (re_val * dre + im_val * dim)
end

function _peak_repair_callback(D_func::NyquistWrapper{P}, p::P, σ, rtol, atol) where {P}
    enabled = Ref(true)
    hits = Ref(0)
    last_t = Ref(-Inf)
    stuck = Ref(0)
    cond = function (y, ω, integ)
        # once disabled, a constant sign means no further crossings to find
        enabled[] || return 1.0
        return _integrand_and_dsq_deriv(D_func, p, σ, ω)[2]
    end
    aff = function (integ)
        hits[] += 1
        a, b = integ.tprev, integ.t
        # A root within ~1e3 ulps of the σ-line makes the callback re-fire in
        # place (the crossing cannot be stepped past in Float64); such a point
        # is numerically ON the boundary, so stop repairing and let the march
        # finish -- a graceful wrong count instead of a maxiters hang.
        stuck[] = (abs(b - last_t[]) <= 100 * eps(b)) ? stuck[] + 1 : 0
        last_t[] = b
        if stuck[] >= 3 || hits[] > 1000
            enabled[] = false
            return nothing
        end
        if b > a
            g = w -> _integrand_and_dsq_deriv(D_func, p, σ, w)[1]
            I = _endpoint_ladder_quad(g, a, b, rtol, atol)
            integ.u = SA[integ.uprev[1] + I]
            u_modified!(integ, true)
        end
        return nothing
    end
    return ContinuousCallback(cond, aff; affect_neg! = nothing,
        rootfind = OrdinaryDiffEq.SciMLBase.LeftRootFind,
        save_positions = (false, false))
end

"""
    calculate_unstable_roots_direct(D_func, p, σ=0.0; ...)

Calculates the number of unstable roots using direct integration of the phase.
Returns `(Z_int, Z_raw, min_D, σ_est, ω_crit)` where `Z` counts the roots with
`Re(λ) > σ` and `σ_est + im*ω_crit` is the estimated location of the root
closest to the integration line `λ = σ + im*ω`. `σ_est` is the **absolute**
real part of the root (i.e. already includes the shift `σ`).
Use `n_roots_to_track` to optimize:
- 0: Max speed, only Z calculation.
- 1: Track the closest root (default).
- N: Track up to N local minima.

Refinement options:
- `refinement_method`: `:Linear` (default), `:Polynomial`, `:Newton`, or
  `:Combined`.
- `refinement_steps`: Number of steps for `:Newton`/`:Combined` (default 4).
- `refinement_degree`: Degree for `:Polynomial` (default 3).

`:Linear` (the default) returns the unrefined tracked estimate `σ_est`, one
Newton step off the minimum of `|D|`: it is first-order accurate but is the
*smoothest* field over a parameter chart, and its sign — all that stability
classification needs — is already correct. It is also the cheapest. When an
accurate rightmost root is wanted, select a polish:
- `:Polynomial` (degree 2 or 3) — a local Taylor root; robust across problem
  classes and a good default polish. Degree 2 is the simplest reliable choice.
- `:Newton` — reaches machine precision on smooth quasi-polynomials, but near
  a pole of a rational characteristic function it can settle on a neighbouring
  root and speckle a chart; prefer `:Polynomial` there.
- `:Combined` — runs the polynomial and Newton polishes and keeps the smallest
  `|D|` per root: the closest to an actual root of `D` (best residual on every
  system tested), at the sum of their costs (still only ~30% over the raw
  solve). Use it when you want the most accurate root and can accept that,
  minimising `|D|`, it may occasionally prefer a neighbouring root to the
  raw seed's, so it is marginally less smooth than a single polynomial polish.

Each polish is monotone: it is accepted only if it lowers `|D|` below the seed,
otherwise the raw estimate is returned, so refinement never degrades the
estimate. For a precise decay-rate *map*, the σ-level contour route is usually
what you want rather than a per-pixel polish.

The default integrator is `Vern9()`. The phase ODE `dy/dω = Im(D'/D)` has a
right-hand side that does not depend on `y`, i.e. it is a quadrature problem
with a zero Jacobian and no stiffness whatsoever; a high-order explicit
Runge-Kutta pair is therefore both the most accurate and (at tight
tolerances) the fastest choice. Low-order pairs can lose the count entirely
at loose tolerances.

`peak_repair = true` (default `false`) arms a detect-and-repair safeguard
against peak skipping, the march's one silent failure mode (a step over a
near-singular integrand peak loses exactly ±π, shifting the count by one
while the integer residual stays clean; see [`peak_skip_suspect`](@ref)).
A rootfinding callback stops the march at every minimum of `|D(σ+iω)|²` —
the peak's exact center, detectable from far away because `|D|²` dips over
an O(1) frequency range even when the peak itself is arbitrarily narrow.
The phase increment of the step that crossed the minimum is then recomputed
by quadrature forced into the located peak with a geometric breakpoint
ladder, and the march restarts at the peak, where step control resolves the
right half on its own. Measured on the showcase system at default
tolerances: the plain march miscounts within `~1e-6` of a Hopf boundary,
the adaptive-quadrature backend silently miscounts at `~1e-10`, while the
repaired march stays correct down to `~1e-10` — at ≈ 2–3× the march cost,
comparable to the quadrature backend (per-step condition checks dominate;
the repair quadrature itself is ~0.2% of evaluations). Off by default
because its benefit is deliberately narrow: the repair matters only in the
immediate vicinity of a stability boundary, and arming it for a whole
chart doubles the sweep to certify at most a few boundary-grazing pixels —
rarely a good trade. It is one keyword to switch on where it pays: to
re-check pixels flagged by [`peak_skip_suspect`](@ref), or when evaluating
deliberately close to a boundary. Roots within ~1e3 ulps of the σ-line
would make the callback re-fire in place; a stuck guard then disables
further repairs for the rest of that march (a point that close is
numerically on the boundary).
"""
function calculate_unstable_roots_direct(@nospecialize(D_func), p::P, σ::S=0.0;
    n_roots_to_track=1,
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(),
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6), peak_repair=false,
    refinement_method=:Linear, refinement_steps=4, refinement_degree=3) where {P, S}

    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    return _calculate_unstable_roots_direct_impl(wrapped_D, p, σ, Val(n_roots_to_track);
        ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver,
        n_power_max=n_power_max, verbosity=verbosity, maxiters=maxiters, peak_repair=peak_repair,
        refinement_method=refinement_method, refinement_steps=refinement_steps, refinement_degree=refinement_degree)
end

# Default for backward compatibility
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S; 
    kwargs...) where {P, S}
    return _calculate_unstable_roots_direct_impl(D_func, p, σ, Val(1); kwargs...)
end

# Val{0}: Maximum Speed (No tracking)
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S, ::Val{0};
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(),
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6), peak_repair=false,
    refinement_method=:Linear, refinement_steps=4, refinement_degree=3) where {P, S}

    function phase_ode(y, params, ω)
        pure_ω = max(ForwardDiff.value(ω), 1e-9)
        dual_ω = ForwardDiff.Dual{StandardTag}(pure_ω, 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)

        rv, iv = real(res), imag(res)
        re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
        dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)

        d_sq = re_val^2 + im_val^2
        if d_sq < 1e-20 || !isfinite(d_sq) || !(isfinite(dre) && isfinite(dim))
            return SA[_safe_darg(re_val, im_val, dre, dim)]
        end

        return SA[(dim * re_val - dre * im_val) / d_sq]
    end

    prob = ODEProblem{false}(phase_ode, SA[0.0], (0.0, Float64(ω_max)))
    cb = peak_repair ? _peak_repair_callback(D_func, p, σ, reltol, abstol) : nothing
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=maxiters, callback=cb)

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    return _safe_round_count(Z_raw), Z_raw
end

# Val{1}: Single Root Tracking
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S, ::Val{1};
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(),
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6), peak_repair=false,
    refinement_method=:Linear, refinement_steps=4, refinement_degree=3) where {P, S}

    min_D_sq = Ref(Inf)
    # NaN until the first representable |D|^2 is seen (see quadgk impl note);
    # refine_roots passes NaN seeds through unchanged, so a point where |D|
    # overflows everywhere reports (Inf, NaN, NaN) instead of a fabricated
    # root polished from the 0+0im placeholder.
    root_ref = Ref(NaN + NaN*im)

    function phase_ode(y, params, ω)
        pure_ω = max(ForwardDiff.value(ω), 1e-9)
        dual_ω = ForwardDiff.Dual{StandardTag}(pure_ω, 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)

        rv, iv = real(res), imag(res)
        re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
        dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)

        d_sq = re_val^2 + im_val^2
        if d_sq < 1e-20 || !isfinite(d_sq) || !(isfinite(dre) && isfinite(dim))
            return SA[_safe_darg(re_val, im_val, dre, dim)]
        end

        if d_sq < min_D_sq[]
            min_D_sq[] = d_sq
            # Real part of one Newton step -D/D' relative to the line, shifted by σ
            # so that the stored root position is absolute in the complex plane.
            est_sigma = σ - (re_val * dim - im_val * dre) / (dim^2 + dre^2)
            root_ref[] = est_sigma + 1im * ForwardDiff.value(ω)
        end

        return SA[(dim * re_val - dre * im_val) / d_sq]
    end

    prob = ODEProblem{false}(phase_ode, SA[0.0], (0.0, Float64(ω_max)))
    cb = peak_repair ? _peak_repair_callback(D_func, p, σ, reltol, abstol) : nothing
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=maxiters, callback=cb)

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    zi, zr, md, es, wc = _safe_round_count(Z_raw), Z_raw, sqrt(min_D_sq[]), real(root_ref[]), imag(root_ref[])
    
    if refinement_method != :Linear
        refined_root = refine_roots(D_func, p, es + 1im*wc; method=refinement_method, steps=refinement_steps, degree=refinement_degree)
        return zi, zr, md, real(refined_root), imag(refined_root)
    end
    return zi, zr, md, es, wc
end

# Val{N}: Multi-Root Tracking
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S, ::Val{N};
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(),
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6), peak_repair=false,
    refinement_method=:Linear, refinement_steps=4, refinement_degree=3) where {P, S, N}

    d_sq_vec = MVector{N, Float64}(fill(Inf, N))
    roots_vec = MVector{N, ComplexF64}(fill(NaN + NaN*im, N))
    
    # Boundary check at ω = 0
    # Because |D(ω)|^2 is even, a positive derivative at ω ≈ 0 implies a local
    # minimum at 0. NOTE: evenness holds for real-coefficient D (conjugate
    # symmetry D(conj(λ)) = conj(D(λ))) -- the same assumption the half-line
    # counting formula itself rests on; for complex-coefficient systems the
    # full-line integral must be used anyway.
    pure_ω_0 = 1e-9
    dual_ω_0 = ForwardDiff.Dual{StandardTag}(pure_ω_0, 1.0)
    dual_λ_0 = σ + 1im * dual_ω_0
    res_0 = D_func(dual_λ_0, p)
    rv_0, iv_0 = real(res_0), imag(res_0)
    re_val_0, im_val_0 = ForwardDiff.value(rv_0), ForwardDiff.value(iv_0)
    dre_0, dim_0 = ForwardDiff.partials(rv_0, 1), ForwardDiff.partials(iv_0, 1)
    d_sq_0 = re_val_0^2 + im_val_0^2
    d_sq_deriv_0 = 2*(re_val_0 * dre_0 + im_val_0 * dim_0)

    if d_sq_deriv_0 > 0
        est_sigma_0 = σ - (re_val_0 * dim_0 - im_val_0 * dre_0) / (dim_0^2 + dre_0^2)
        d_sq_vec[1] = d_sq_0
        roots_vec[1] = est_sigma_0 + 0.0im
    end

    prev_d_sq_deriv = Ref(d_sq_deriv_0)
    prev_ω = Ref(0.0)

    function phase_ode(y, params, ω)
        pure_ω = max(ForwardDiff.value(ω), 1e-9)
        dual_ω = ForwardDiff.Dual{StandardTag}(pure_ω, 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)

        rv, iv = real(res), imag(res)
        re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
        dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)

        d_sq = re_val^2 + im_val^2
        if d_sq < 1e-20 || !isfinite(d_sq) || !(isfinite(dre) && isfinite(dim))
            # same near-root/overflow handling as the other backends
            return SA[_safe_darg(re_val, im_val, dre, dim)]
        end
        d_sq_deriv = 2*(re_val * dre + im_val * dim)
        curr_ω = ForwardDiff.value(ω)

        # Detect local minimum: derivative crosses 0 from below
        if prev_d_sq_deriv[] < 0 && d_sq_deriv > 0 && curr_ω > prev_ω[]
            est_sigma = σ - (re_val * dim - im_val * dre) / (dim^2 + dre^2)
            new_root = est_sigma + 1im * curr_ω

            # The solver's RK stages and rejected/retried steps do not sample
            # ω monotonically, so the same minimum can be detected more than
            # once at nearly identical ω. Deduplicate by ω-proximity: keep the
            # deeper of the two detections instead of storing both.
            insert_val = d_sq
            dup_idx = 0
            for j in 1:N
                if isfinite(imag(roots_vec[j])) &&
                   abs(curr_ω - imag(roots_vec[j])) <= 1e-6 * max(curr_ω, 1.0)
                    dup_idx = j
                    break
                end
            end
            if dup_idx > 0
                if insert_val < d_sq_vec[dup_idx]
                    # drop the shallower duplicate, then re-insert sorted below
                    for j in dup_idx:N-1
                        d_sq_vec[j] = d_sq_vec[j+1]
                        roots_vec[j] = roots_vec[j+1]
                    end
                    d_sq_vec[N] = Inf
                    roots_vec[N] = NaN + NaN*im
                else
                    insert_val = Inf   # shallower re-detection: skip the insert
                end
            end

            if insert_val < d_sq_vec[N]
                idx = N
                while idx > 1 && insert_val < d_sq_vec[idx-1]
                    idx -= 1
                end
                for j in N:-1:idx+1
                    d_sq_vec[j] = d_sq_vec[j-1]
                    roots_vec[j] = roots_vec[j-1]
                end
                d_sq_vec[idx] = insert_val
                roots_vec[idx] = new_root
            end
        end
        prev_d_sq_deriv[] = d_sq_deriv
        prev_ω[] = curr_ω

        return SA[(dim * re_val - dre * im_val) / d_sq]
    end

    prob = ODEProblem{false}(phase_ode, SA[0.0], (0.0, Float64(ω_max)))
    cb = peak_repair ? _peak_repair_callback(D_func, p, σ, reltol, abstol) : nothing
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=maxiters, callback=cb)

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    zi, zr, mds, ess, wcs = _safe_round_count(Z_raw), Z_raw, sqrt.(d_sq_vec), real.(roots_vec), imag.(roots_vec)

    if refinement_method != :Linear
        roots = ess .+ 1im .* wcs
        refined_roots = [refine_roots(D_func, p, r; method=refinement_method, steps=refinement_steps, degree=refinement_degree) for r in roots]
        # The ω-proximity dedupe above cannot catch re-detections of the same
        # minimum at stage abscissae further apart than its window; those
        # collapse onto the SAME root only after refinement. Blank the
        # duplicates (entries are depth-sorted, so the first is the deepest).
        for j in 2:N
            for k in 1:j-1
                if isfinite(abs(refined_roots[j])) && isfinite(abs(refined_roots[k])) &&
                   abs(refined_roots[j] - refined_roots[k]) <= 1e-6 * max(1.0, abs(refined_roots[k]))
                    refined_roots[j] = NaN + NaN*im
                    break
                end
            end
        end
        return zi, zr, mds, real.(refined_roots), imag.(refined_roots)
    end
    return zi, zr, mds, ess, wcs
end

"""
    calculate_unstable_roots_p_vec(D_func, params_vec::AbstractVector; n_power_max=nothing, ...)

Vectorized stability sweep using multi-threading.

`n_power_max` is the leading order `n` of `D` in the counting
formula `Z = n/2 - (1/π)∫₀^ωmax Im(D'/D) dω`:

* `nothing` (default): estimate it automatically, once, at `params_vec[1]`
  (see `parameter_independent_nmax`) via [`get_n_power_max`](@ref).
* a number: use it as given for every parameter point, and skip the estimation
  entirely.

**Supply it whenever you know it.** It is exact where the estimator can only be
accurate, it costs nothing, and for some classes the estimator is working
uphill: for a neutral system `|D|` does not settle onto a power law at all but
oscillates by a factor `(1+|a|)/(1-|a|)` forever, and for a transcendental
`D` (e.g. `cosh`-type) it is not polynomially bounded in the right half-plane,
so the fit falls back to a weaker estimate. The order is usually obvious by
inspection — `D = λ²(1 + a e^{-λ}) + …` has `n = 2`, since `e^{-s} → 0` along
the real axis — and one number removes a whole class of doubt. Non-integer
orders are legitimate (fractional systems).

Note that a correct `n` does not by itself make a neutral count *certain*: the
truncated tail of a neutral system contributes an oscillation bounded by
`asin(|a|)/π < 1/2` regardless of `ω_max`, which is what keeps the rounded
count right and what stops the integer residual from being a useful check
there.
"""
function calculate_unstable_roots_p_vec(@nospecialize(D_func), params_vec::AbstractVector{P};
    n_roots_to_track=1,
    σ::S=0.0, ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(),
    n_power_max=nothing,
    parameter_independent_nmax=true, verbosity=0, maxiters=Int(1e6), peak_repair=false,
    refinement_method=:Linear, refinement_steps=4, refinement_degree=3) where {P, S}

    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)

    n_params = length(params_vec)
    # A user-supplied order wins outright: it is exact, it costs nothing, and
    # it is the documented escape hatch for the classes where the estimator is
    # least comfortable.
    n_pow_fixed = n_power_max !== nothing ? Float64(n_power_max) :
        (parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ) : nothing)

    if n_roots_to_track == 1
        Z_ints = zeros(Int, n_params)
        Z_raws = zeros(Float64, n_params)
        min_Ds = zeros(Float64, n_params)
        sigmas = zeros(Float64, n_params)
        crits = zeros(Float64, n_params)

        if verbosity > 0
            println("Calculating stability over \$n_params points (tracking 1 root)...")
        end

        @inbounds Threads.@threads for i in 1:n_params
            zi, zr, md, es, wc = _calculate_unstable_roots_direct_impl(wrapped_D, params_vec[i], σ, Val(1);
                ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver,
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters, peak_repair=peak_repair,
                refinement_method=refinement_method, refinement_steps=refinement_steps, refinement_degree=refinement_degree)
            Z_ints[i] = zi
            Z_raws[i] = zr
            min_Ds[i] = md
            sigmas[i] = es
            crits[i] = wc
        end
        return Z_ints, Z_raws, min_Ds, sigmas, crits
    elseif n_roots_to_track == 0
        Z_ints = zeros(Int, n_params)
        Z_raws = zeros(Float64, n_params)

        if verbosity > 0
            println("Calculating stability over \$n_params points (max speed)...")
        end

        @inbounds Threads.@threads for i in 1:n_params
            zi, zr = _calculate_unstable_roots_direct_impl(wrapped_D, params_vec[i], σ, Val(0);
                ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver,
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters, peak_repair=peak_repair,
                refinement_method=refinement_method, refinement_steps=refinement_steps, refinement_degree=refinement_degree)
            Z_ints[i] = zi
            Z_raws[i] = zr
        end
        return Z_ints, Z_raws
    else
        Z_ints = zeros(Int, n_params)
        Z_raws = zeros(Float64, n_params)
        min_Ds_list = [zeros(Float64, n_roots_to_track) for _ in 1:n_params]
        sigmas_list = [zeros(Float64, n_roots_to_track) for _ in 1:n_params]
        crits_list = [zeros(Float64, n_roots_to_track) for _ in 1:n_params]

        if verbosity > 0
            println("Calculating stability over \$n_params points (tracking \$n_roots_to_track roots)...")
        end

        # hoisted: one runtime Val construction instead of one per point (the
        # per-point call still dispatches dynamically, amortized by the solve)
        V = Val(n_roots_to_track)
        @inbounds Threads.@threads for i in 1:n_params
            zi, zr, md, es, wc = _calculate_unstable_roots_direct_impl(wrapped_D, params_vec[i], σ, V;
                ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver,
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters, peak_repair=peak_repair,
                refinement_method=refinement_method, refinement_steps=refinement_steps, refinement_degree=refinement_degree)
            Z_ints[i] = zi
            Z_raws[i] = zr
            min_Ds_list[i] .= md
            sigmas_list[i] .= es
            crits_list[i] .= wc
        end
        return Z_ints, Z_raws, min_Ds_list, sigmas_list, crits_list
    end
end

"""
    calculate_unstable_roots_quadgk(D_func, p, σ=0.0; ω_max=1e6, reltol=1e-5, abstol=1e-5, n_power_max=nothing)

Calculates the number of unstable roots using QuadGK.jl (adaptive 1D quadrature).
"""
function calculate_unstable_roots_quadgk(@nospecialize(D_func), p::P, σ::S=0.0; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, n_power_max=nothing) where {P, S}
    
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    return _calculate_unstable_roots_quadgk_impl(wrapped_D, p, σ; 
        ω_max=ω_max, reltol=reltol, abstol=abstol, n_power_max=n_power_max)
end

function _calculate_unstable_roots_quadgk_impl(D_func::NyquistWrapper{P}, p::P, σ::S=0.0; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, n_power_max=nothing) where {P, S}
    
    min_D_sq = Ref(Inf)
    # NaN until the first representable |D|^2 is seen: if |D| overflows at
    # EVERY sample (astronomically scaled determinants), no tracking is
    # possible and the root estimate must come back invalid, not fabricated.
    estimated_sigma = Ref(NaN)
    ω_crit = Ref(NaN)

    function phase_integrand(ω)
        # TRICK: Add a tiny offset to avoid singularities in fractional derivatives at ω=0
        pure_ω = max(ω, 1e-9)
        dual_ω = ForwardDiff.Dual{StandardTag}(pure_ω, 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)
        rv, iv = real(res), imag(res)
        re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
        dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)
        d_sq = re_val^2 + im_val^2
        if d_sq < 1e-20 || !isfinite(d_sq) || !(isfinite(dre) && isfinite(dim))
            return _safe_darg(re_val, im_val, dre, dim)
        end
        if d_sq < min_D_sq[]
            min_D_sq[] = d_sq
            ω_crit[] = ω
            estimated_sigma[] = σ - (re_val * dim - im_val * dre) / (dim^2 + dre^2)
        end
        return (dim * re_val - dre * im_val) / d_sq
    end

    # Log-spaced interior breakpoints (one per two decades) force at least one
    # G7-K15 panel into every frequency band. Without them the FIRST panel
    # spans [0, ω_max] and its lowest node sits at ω ≈ 0.0043·ω_max; for
    # ω_max = 1e6 and a system whose resonances live at ω ~ 1-10 the quadrature
    # then never samples the resonance region at all, the fast-decaying tail
    # looks converged after 15 evaluations, and Z_raw = n/2 is returned with a
    # PERFECT integer residual -- a silent wrong count the self-diagnostic
    # cannot flag. The extra panels cost a few dozen evaluations.
    ω_hi = Float64(ω_max)
    interior = ω_hi > 10.0 ? exp10.(0.0:2.0:(log10(ω_hi) - 1.0)) : Float64[]
    integral, err = quadgk(phase_integrand, 0.0, interior..., ω_hi, rtol=reltol, atol=abstol)
    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * integral + n_pow / 2.0
    return _safe_round_count(Z_raw), Z_raw, sqrt(min_D_sq[]), estimated_sigma[], ω_crit[]
end

"""
    calculate_unstable_roots_quadgk_p_vec(D_func, params_vec::AbstractVector; n_power_max=nothing, ...)

Vectorized stability sweep using QuadGK and multi-threading.

`n_power_max`: pass the leading order explicitly when it is known; `nothing`
estimates it once at `params_vec[1]`. See
[`calculate_unstable_roots_p_vec`](@ref).
"""
function calculate_unstable_roots_quadgk_p_vec(@nospecialize(D_func), params_vec::AbstractVector{P};
    σ::S=0.0, ω_max=1e6, reltol=1e-5, abstol=1e-5, n_power_max=nothing,
    parameter_independent_nmax=true, verbosity=0) where {P, S}

    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)

    n_params = length(params_vec)
    n_pow_fixed = n_power_max !== nothing ? Float64(n_power_max) :
        (parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ) : nothing)

    Z_ints = zeros(Int, n_params)
    Z_raws = zeros(Float64, n_params)
    min_Ds = zeros(Float64, n_params)
    sigmas = zeros(Float64, n_params)
    crits = zeros(Float64, n_params)

    if verbosity > 0
        println("Calculating stability (QuadGK) over $n_params points...")
    end

    @inbounds Threads.@threads for i in 1:n_params
        zi, zr, md, es, wc = _calculate_unstable_roots_quadgk_impl(wrapped_D, params_vec[i], σ; 
            ω_max=ω_max, reltol=reltol, abstol=abstol, n_power_max=n_pow_fixed)
        Z_ints[i] = zi
        Z_raws[i] = zr
        min_Ds[i] = md
        sigmas[i] = es
        crits[i] = wc
    end

    return Z_ints, Z_raws, min_Ds, sigmas, crits
end

"""
    calculate_unstable_roots_fixed_step(D_func, p, σ=0.0; ω_max=1e6, steps=1000, n_power_max=nothing)

ULTRA-FAST: Fixed-step trapezoidal integration. Zero overhead, perfect for real-time sweeps.
"""
function calculate_unstable_roots_fixed_step(@nospecialize(D_func), p::P, σ::S=0.0; 
    ω_max=1e6, steps=1000, n_power_max=nothing) where {P, S}
    
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    return _calculate_unstable_roots_fixed_step_impl(wrapped_D, p, σ; 
        ω_max=ω_max, steps=steps, n_power_max=n_power_max)
end

function _calculate_unstable_roots_fixed_step_impl(D_func::NyquistWrapper{P}, p::P, σ::S=0.0; 
    ω_max=1e6, steps=1000, n_power_max=nothing) where {P, S}
    
    # Refs, not plain locals: reassigning a captured local inside the closure
    # would box all three (Core.Box) and make the innermost loop dynamically
    # typed -- in the one backend whose point is raw speed.
    min_D_sq = Ref(Inf)
    estimated_sigma = Ref(NaN)
    ω_crit = Ref(NaN)

    h = Float64(ω_max) / steps
    integral = 0.0

    function get_darg(ω_val)
        # TRICK: Add a tiny offset to avoid singularities in fractional derivatives at ω=0
        pure_ω = max(ω_val, 1e-9)
        dual_ω = ForwardDiff.Dual{StandardTag}(pure_ω, 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)
        rv, iv = real(res), imag(res)
        re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
        dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)
        d_sq = re_val^2 + im_val^2

        if d_sq < 1e-20 || !isfinite(d_sq) || !(isfinite(dre) && isfinite(dim))
            return _safe_darg(re_val, im_val, dre, dim)
        end

        if d_sq < min_D_sq[]
            min_D_sq[] = d_sq
            ω_crit[] = ω_val
            estimated_sigma[] = σ - (re_val * dim - im_val * dre) / (dim^2 + dre^2)
        end

        return (dim * re_val - dre * im_val) / d_sq
    end

    f_prev = get_darg(0.0)
    for i in 1:steps
        ω = i * h
        f_curr = get_darg(ω)
        integral += (f_prev + f_curr) * 0.5 * h
        f_prev = f_curr
    end

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * integral + n_pow / 2.0
    return _safe_round_count(Z_raw), Z_raw, sqrt(min_D_sq[]), estimated_sigma[], ω_crit[]
end

"""
    refine_roots(D_func, p, roots; method=:Polynomial, steps=4, degree=3, fix_omega=false)

Refines the estimated roots. (The chart-level default is `:Linear`, i.e. no
refinement; select one of these when an accurate rightmost root is wanted.)
- `method=:Polynomial`: local Taylor root of `degree` (2 or 3); robust polish.
- `method=:Newton`: `steps` Newton-Raphson iterations; machine-precision on
  smooth `D`, less reliable near a rational pole.
- `method=:Combined`: the polynomial and Newton polishes, keeping the smallest
  `|D|` per root.
- `fix_omega`: If true, only the real part (sigma) is updated.

Refinement is a **local** polish and is confined to a trust region: iterates
that leave the disc of radius `max(|λ_seed|, 1)` around the seed are rejected
and the last sane iterate is returned. Without it the iteration can converge
to a different -- perfectly valid, but non-dominant -- root of `D`: for a
distributed-delay system, where `|D|` grows without bound to the left, a
single Newton step can land on a root at `Re λ ~ -1e4` and have it reported as
the nearest one. The guard also keeps the next evaluation of the user's
`D_func` away from `λ ~ 1e300`, where `λ^2` overflows inside `D_func` before
anything here could intercept it.

The trust region bounds the damage of a poor seed but does not make the
iteration globally convergent; refinement quality is assured only near the
tracked minimum it is designed to polish. Pass `method=:Linear` to switch
refinement off and keep the raw tracked estimate.

Returns the refined complex roots as standard `ComplexF64` values.
"""
function refine_roots(@nospecialize(D_func), p::P, roots::AbstractArray; kwargs...) where P
    return ComplexF64[refine_roots(D_func, p, r; kwargs...) for r in roots]
end

function refine_roots(@nospecialize(D_func), p::P, λ::Complex{T};
    method=:Newton, steps=4, degree=3, fix_omega=false) where {P, T}
    
    if isnan(λ) || isinf(λ)
        return ComplexF64(λ)
    end

    # COMBINED refinement: run each single method and keep the candidate with
    # the smallest |D| -- the honest, method-independent root-quality measure.
    # No single classical polish is best on every system: Newton reaches
    # machine precision on the smooth quasi-polynomials but overshoots near the
    # poles of a rational FRF (turning), where the Taylor polishes stay closer;
    # taking the best per root is robustly at least as good as any of them
    # (measured: on turning it beats every single method's median residual by
    # ~9x, and it ties Newton's 1e-15 on the showcase). Costs the sum of the
    # three -- still only ~10% over the phase integration -- so it is the
    # recommended choice when accuracy matters more than the last few percent
    # of speed.
    if method == :Combined
        seed = ComplexF64(λ)
        cands = (refine_roots(D_func, p, seed; method = :Polynomial, degree = 2, fix_omega = fix_omega),
                 refine_roots(D_func, p, seed; method = :Polynomial, degree = 3, fix_omega = fix_omega),
                 refine_roots(D_func, p, seed; method = :Newton,     steps = steps, fix_omega = fix_omega))
        _absD(z) = isfinite(z) ? abs(D_func(z, p)) : Inf
        best = seed
        best_d = _absD(seed)
        for c in cands
            d = _absD(c)
            if d < best_d
                best_d = d
                best = c
            end
        end
        return best
    end

    curr_λ = ComplexF64(λ)

    # TRUST REGION. Refinement is a local polish of an estimate that is already
    # one Newton step from the tracked minimum of |D|, so the true root sits a
    # short distance away and any large step means the iteration has left the
    # basin it was meant to refine. It will then happily converge to some OTHER
    # root -- a perfectly valid root of D, and useless here: for a distributed
    # delay, where D ~ b*e^{-λτ}/λ grows without bound to the left, a step can
    # land on a root at Re λ ~ -1e4 and report it as the dominant one, which
    # destroys the chart's colour scale. Steps leaving the disc of radius
    # `trust` around the SEED are therefore rejected and the last sane iterate
    # is returned. This also keeps the next evaluation of a user-supplied D away
    # from λ ~ 1e300, where λ^2 overflows inside D itself, beyond our reach.
    seed_λ = curr_λ
    trust = max(abs(seed_λ), 1.0)
    # cumulative displacement from the seed, so a sequence of small steps
    # cannot walk out of the region either
    step_ok(Δ) = isfinite(abs(Δ)) && abs((curr_λ + Δ) - seed_λ) <= trust

    # Helper to evaluate D and its 1st derivative efficiently without leaking Duals
    function eval_D_and_deriv(s, w)
        dual_s = ForwardDiff.Dual{StandardTag}(s, 1.0)
        res = D_func(dual_s + 1im * w, p)
        val = ComplexF64(ForwardDiff.value(real(res)), ForwardDiff.value(imag(res)))
        deriv = ComplexF64(ForwardDiff.partials(real(res), 1), ForwardDiff.partials(imag(res), 1))
        return val, deriv
    end

    # MONOTONE REFINEMENT. Refinement is a polish of an estimate (`est_sigma`,
    # one Newton step from the tracked minimum) that is already a good root
    # locator -- for the rational FRF of a turning model the unrefined estimate
    # reproduces a semi-discretization reference across the whole stable domain.
    # Iterated refinement must therefore only ever IMPROVE it: near a pole of a
    # rational D the local Newton/Taylor step points away from the shallow
    # near-line root toward a far root or the overflow region, and returning
    # that wandered iterate as the root fabricates a dominant root at
    # Re λ = -100 (or +2117) with |D| far larger than at the seed -- which then
    # destroys the σ colouring. So compare |D| at the refined iterate with |D|
    # at the seed and keep whichever is smaller; refinement can help but never
    # hurt. Only when BOTH are non-finite (the seed itself sits in overflow) is
    # there nothing to report -> NaN, the established invalid-root marker that
    # every dominance selection filters. `fix_omega` keeps the old contract --
    # its iteration is confined to a horizontal line and is used where the
    # caller wants exactly that.
    seed_absD = let (v, _) = eval_D_and_deriv(real(seed_λ), imag(seed_λ)); abs(v) end
    function keep_better(λc::ComplexF64)
        fix_omega && return λc
        cand_absD = isfinite(abs(λc)) ? abs(eval_D_and_deriv(real(λc), imag(λc))[1]) : Inf
        s_ok = isfinite(seed_absD)
        c_ok = isfinite(cand_absD)
        if c_ok && (!s_ok || cand_absD <= seed_absD)
            return λc
        elseif s_ok
            return seed_λ
        else
            return ComplexF64(NaN, NaN)
        end
    end

    if method == :Newton
        for _ in 1:steps
            val, deriv = eval_D_and_deriv(real(curr_λ), imag(curr_λ))
            if abs(deriv) < 1e-15 || !isfinite(abs(val)) || !isfinite(abs(deriv))
                break
            end
            Δλ = -val / deriv
            step_ok(Δλ) || break
            next_λ = fix_omega ? ComplexF64(real(curr_λ) + real(Δλ), imag(curr_λ)) :
                                 curr_λ + Δλ
            isfinite(abs(next_λ)) || break
            curr_λ = next_λ
        end
        return keep_better(curr_λ)

    elseif method == :Polynomial
        if degree == 1
            return refine_roots(D_func, p, curr_λ, method=:Newton, steps=1, fix_omega=fix_omega)
        end

        # For polynomial > 1, we need higher derivatives. To avoid nested ForwardDiff
        # which is extremely slow to compile and type-infer, we use finite differences
        # on the analytically computed first derivative.
        val, D1 = eval_D_and_deriv(real(curr_λ), imag(curr_λ))
        if !(isfinite(abs(val)) && isfinite(abs(D1)))
            # overflowing D at the seed itself: nothing to polish
            return fix_omega ? curr_λ : keep_better(curr_λ)
        end
        D0 = val
        
        h = 1e-5
        _, D1_plus = eval_D_and_deriv(real(curr_λ) + h, imag(curr_λ))
        _, D1_minus = eval_D_and_deriv(real(curr_λ) - h, imag(curr_λ))
        
        D2 = (D1_plus - D1_minus) / (2h)
        
        if degree == 2
            a = 0.5 * D2
            b = D1
            c = D0

            if abs(a) < 1e-15 || !isfinite(abs(a))
                Δλ = -c / b
            else
                disc = sqrt(b^2 - 4*a*c)
                Δλ1 = (-b + disc) / (2*a)
                Δλ2 = (-b - disc) / (2*a)
                Δλ = (abs(Δλ1) < abs(Δλ2)) ? Δλ1 : Δλ2
            end
            step_ok(Δλ) || return keep_better(curr_λ)

            if fix_omega
                return ComplexF64(real(curr_λ) + real(Δλ), imag(curr_λ))
            else
                return keep_better(curr_λ + Δλ)
            end

        elseif degree == 3
            # 3rd derivative via central finite difference on the AD-exact D1
            D3 = (D1_plus - 2*D1 + D1_minus) / (h^2)
            
            d = (1/6) * D3
            a = 0.5 * D2
            b = D1
            c = D0
            
            use_cubic = abs(d) >= 1e-15 &&
                isfinite(abs(a/d)) && isfinite(abs(b/d)) && isfinite(abs(c/d))
            if use_cubic
                M = ComplexF64[0.0 0.0 -c/d; 1.0 0.0 -b/d; 0.0 1.0 -a/d]
                evs = eigvals(M)
                Δλ = evs[argmin(abs.(evs))]
            elseif abs(a) >= 1e-15 && isfinite(abs(a))
                disc = sqrt(b^2 - 4*a*c)
                Δλ1 = (-b + disc) / (2*a)
                Δλ2 = (-b - disc) / (2*a)
                Δλ = (abs(Δλ1) < abs(Δλ2)) ? Δλ1 : Δλ2
            else
                Δλ = -c/b
            end
            step_ok(Δλ) || return keep_better(curr_λ)

            if fix_omega
                return ComplexF64(real(curr_λ) + real(Δλ), imag(curr_λ))
            else
                return keep_better(curr_λ + Δλ)
            end
        else
            error("Polynomial refinement only supported up to degree 3.")
        end
    else
        error("Unknown refinement method: $method")
    end
end

"""
    calculate_unstable_roots_fixed_step_p_vec(D_func, params_vec::AbstractVector; ...)

Vectorized stability sweep using ultra-fast fixed-step integration.
"""
function calculate_unstable_roots_fixed_step_p_vec(@nospecialize(D_func), params_vec::AbstractVector{P};
    σ::S=0.0, ω_max=1e6, steps=500, n_power_max=nothing,
    parameter_independent_nmax=true, verbosity=0) where {P, S}

    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)

    n_params = length(params_vec)
    n_pow_fixed = n_power_max !== nothing ? Float64(n_power_max) :
        (parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ) : nothing)

    Z_ints = zeros(Int, n_params)
    Z_raws = zeros(Float64, n_params)
    min_Ds = zeros(Float64, n_params)
    sigmas = zeros(Float64, n_params)
    crits = zeros(Float64, n_params)

    if verbosity > 0
        println("Calculating stability (Fixed-Step) over $n_params points...")
    end

    @inbounds Threads.@threads for i in 1:n_params
        zi, zr, md, es, wc = _calculate_unstable_roots_fixed_step_impl(wrapped_D, params_vec[i], σ; 
            ω_max=ω_max, steps=steps, n_power_max=n_pow_fixed)
        Z_ints[i] = zi
        Z_raws[i] = zr
        min_Ds[i] = md
        sigmas[i] = es
        crits[i] = wc
    end

    return Z_ints, Z_raws, min_Ds, sigmas, crits
end
