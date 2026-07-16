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

function _get_n_power_max_impl(D_func::NyquistWrapper{P}, p::P, σ=0.0; probe=1e8, decades=1.0, samples=8) where P
    s_hi = Float64(probe)
    zero_dual = ForwardDiff.Dual{StandardTag}(0.0, 0.0)
    for _ in 1:16   # back off until |D| is representable at the probe location
        sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0; K = 0
        for t in range(log(s_hi) - decades * log(10.0), log(s_hi); length=samples)
            dual_s = ForwardDiff.Dual{StandardTag}(exp(t), 0.0)
            res = D_func(Complex(dual_s, zero_dual), p)
            absD = hypot(ForwardDiff.value(real(res)), ForwardDiff.value(imag(res)))
            (isfinite(absD) && absD > 0) || continue
            y = log(absD)
            sx += t; sy += y; sxx += t * t; sxy += t * y; K += 1
        end
        if K >= 2
            denom = K * sxx - sx * sx
            if denom > 0
                n_est = (K * sxy - sx * sy) / denom
                isfinite(n_est) && return n_est
            end
        end
        s_hi /= 10.0
    end
    return 0.0
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
- `refinement_method`: `:Newton` (default), `:Polynomial`, or `:Linear`.
- `refinement_steps`: Number of steps for `:Newton` (default 4).
- `refinement_degree`: Degree for `:Polynomial` (default 3).

Refinement costs only a few percent of the integration, so it is on by
default; `:Linear` returns the unrefined tracked estimate.

The default integrator is `Vern9()`. The phase ODE `dy/dω = Im(D'/D)` has a
right-hand side that does not depend on `y`, i.e. it is a quadrature problem
with a zero Jacobian and no stiffness whatsoever; a high-order explicit
Runge-Kutta pair is therefore both the most accurate and (at tight
tolerances) the fastest choice. Low-order pairs can lose the count entirely
at loose tolerances.
"""
function calculate_unstable_roots_direct(@nospecialize(D_func), p::P, σ::S=0.0;
    n_roots_to_track=1,
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(), 
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6),
    refinement_method=:Newton, refinement_steps=4, refinement_degree=3) where {P, S}

    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    return _calculate_unstable_roots_direct_impl(wrapped_D, p, σ, Val(n_roots_to_track); 
        ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver, 
        n_power_max=n_power_max, verbosity=verbosity, maxiters=maxiters,
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
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6),
    refinement_method=:Newton, refinement_steps=4, refinement_degree=3) where {P, S}

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
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=maxiters)

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    return round(Int, Z_raw), Z_raw
end

# Val{1}: Single Root Tracking
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S, ::Val{1}; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(), 
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6),
    refinement_method=:Newton, refinement_steps=4, refinement_degree=3) where {P, S}

    min_D_sq = Ref(Inf)
    root_ref = Ref(0.0 + 0.0im)
    
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
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=maxiters)

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    zi, zr, md, es, wc = round(Int, Z_raw), Z_raw, sqrt(min_D_sq[]), real(root_ref[]), imag(root_ref[])
    
    if refinement_method != :Linear
        refined_root = refine_roots(D_func, p, es + 1im*wc; method=refinement_method, steps=refinement_steps, degree=refinement_degree)
        return zi, zr, md, real(refined_root), imag(refined_root)
    end
    return zi, zr, md, es, wc
end

# Val{N}: Multi-Root Tracking
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S, ::Val{N}; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(), 
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6),
    refinement_method=:Newton, refinement_steps=4, refinement_degree=3) where {P, S, N}

    d_sq_vec = MVector{N, Float64}(fill(Inf, N))
    roots_vec = MVector{N, ComplexF64}(fill(NaN + NaN*im, N))
    
    # Boundary check at ω = 0
    # Because |D(ω)|^2 is even, a positive derivative at ω ≈ 0 implies a local minimum at 0.
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
        if !isfinite(d_sq) || !(isfinite(dre) && isfinite(dim))
            return SA[_safe_darg(re_val, im_val, dre, dim)]
        end
        d_sq_deriv = 2*(re_val * dre + im_val * dim)
        curr_ω = ForwardDiff.value(ω)

        # Detect local minimum: derivative crosses 0 from below
        if prev_d_sq_deriv[] < 0 && d_sq_deriv > 0 && curr_ω > prev_ω[]
            est_sigma = σ - (re_val * dim - im_val * dre) / (dim^2 + dre^2)
            new_root = est_sigma + 1im * curr_ω
            
            if d_sq < d_sq_vec[N]
                idx = N
                while idx > 1 && d_sq < d_sq_vec[idx-1]
                    idx -= 1
                end
                for j in N:-1:idx+1
                    d_sq_vec[j] = d_sq_vec[j-1]
                    roots_vec[j] = roots_vec[j-1]
                end
                d_sq_vec[idx] = d_sq
                roots_vec[idx] = new_root
            end
        end
        prev_d_sq_deriv[] = d_sq_deriv
        prev_ω[] = curr_ω

        d_sq_safe = max(d_sq, 1e-20)
        return SA[(dim * re_val - dre * im_val) / d_sq_safe]
    end

    prob = ODEProblem{false}(phase_ode, SA[0.0], (0.0, Float64(ω_max)))
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=maxiters)

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    zi, zr, mds, ess, wcs = round(Int, Z_raw), Z_raw, sqrt.(d_sq_vec), real.(roots_vec), imag.(roots_vec)
    
    if refinement_method != :Linear
        roots = ess .+ 1im .* wcs
        refined_roots = [refine_roots(D_func, p, r; method=refinement_method, steps=refinement_steps, degree=refinement_degree) for r in roots]
        return zi, zr, mds, real.(refined_roots), imag.(refined_roots)
    end
    return zi, zr, mds, ess, wcs
end

"""
    calculate_unstable_roots_p_vec(D_func, params_vec::AbstractVector; ...)

Vectorized stability sweep using multi-threading.
"""
function calculate_unstable_roots_p_vec(@nospecialize(D_func), params_vec::AbstractVector{P}; 
    n_roots_to_track=1,
    σ::S=0.0, ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=Vern9(), 
    parameter_independent_nmax=true, verbosity=0, maxiters=Int(1e6),
    refinement_method=:Newton, refinement_steps=4, refinement_degree=3) where {P, S}
    
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    
    n_params = length(params_vec)
    n_pow_fixed = parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ) : nothing

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
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters,
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
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters,
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

        @inbounds Threads.@threads for i in 1:n_params
            zi, zr, md, es, wc = _calculate_unstable_roots_direct_impl(wrapped_D, params_vec[i], σ, Val(n_roots_to_track); 
                ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver, 
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters,
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
    estimated_sigma = Ref(Inf)
    ω_crit = Ref(0.0)
    
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

    integral, err = quadgk(phase_integrand, 0.0, Float64(ω_max), rtol=reltol, atol=abstol)
    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ) : n_power_max
    Z_raw = -(1.0 / π) * integral + n_pow / 2.0
    return round(Int, Z_raw), Z_raw, sqrt(min_D_sq[]), estimated_sigma[], ω_crit[]
end

"""
    calculate_unstable_roots_quadgk_p_vec(D_func, params_vec::AbstractVector; ...)

Vectorized stability sweep using QuadGK and multi-threading.
"""
function calculate_unstable_roots_quadgk_p_vec(@nospecialize(D_func), params_vec::AbstractVector{P}; 
    σ::S=0.0, ω_max=1e6, reltol=1e-5, abstol=1e-5, parameter_independent_nmax=true, verbosity=0) where {P, S}
    
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    
    n_params = length(params_vec)
    n_pow_fixed = parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ) : nothing

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
    
    min_D_sq = Inf
    estimated_sigma = Inf
    ω_crit = 0.0
    
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

        if d_sq < min_D_sq
            min_D_sq = d_sq
            ω_crit = ω_val
            estimated_sigma = σ - (re_val * dim - im_val * dre) / (dim^2 + dre^2)
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
    return round(Int, Z_raw), Z_raw, sqrt(min_D_sq), estimated_sigma, ω_crit
end

"""
    refine_roots(D_func, p, roots; method=:Newton, steps=3, degree=2, fix_omega=false)

Refines the estimated roots using Newton-Raphson or Polynomial (Taylor) approximation.
- `method=:Newton`: Performs `steps` iterations of Newton-Raphson.
- `method=:Polynomial`: Approximates D by a Taylor polynomial of `degree` and finds its root.
- `fix_omega`: If true, only the real part (sigma) is updated.

Returns the refined complex roots as standard `ComplexF64` values.
"""
function refine_roots(@nospecialize(D_func), p::P, roots::AbstractArray; kwargs...) where P
    return ComplexF64[refine_roots(D_func, p, r; kwargs...) for r in roots]
end

function refine_roots(@nospecialize(D_func), p::P, λ::Complex{T}; 
    method=:Newton, steps=3, degree=2, fix_omega=false) where {P, T}
    
    if isnan(λ) || isinf(λ)
        return ComplexF64(λ)
    end

    curr_λ = ComplexF64(λ)

    # Helper to evaluate D and its 1st derivative efficiently without leaking Duals
    function eval_D_and_deriv(s, w)
        dual_s = ForwardDiff.Dual{StandardTag}(s, 1.0)
        res = D_func(dual_s + 1im * w, p)
        val = ComplexF64(ForwardDiff.value(real(res)), ForwardDiff.value(imag(res)))
        deriv = ComplexF64(ForwardDiff.partials(real(res), 1), ForwardDiff.partials(imag(res), 1))
        return val, deriv
    end

    if method == :Newton
        for _ in 1:steps
            val, deriv = eval_D_and_deriv(real(curr_λ), imag(curr_λ))
            if abs(deriv) < 1e-15
                break
            end
            Δλ = -val / deriv
            if fix_omega
                curr_λ = ComplexF64(real(curr_λ) + real(Δλ), imag(curr_λ))
            else
                curr_λ += Δλ
            end
        end
        return curr_λ

    elseif method == :Polynomial
        if degree == 1
            return refine_roots(D_func, p, curr_λ, method=:Newton, steps=1, fix_omega=fix_omega)
        end

        # For polynomial > 1, we need higher derivatives. To avoid nested ForwardDiff
        # which is extremely slow to compile and type-infer, we use finite differences
        # on the analytically computed first derivative.
        val, D1 = eval_D_and_deriv(real(curr_λ), imag(curr_λ))
        if !(isfinite(abs(val)) && isfinite(abs(D1)))
            return curr_λ   # overflowing D (e.g. far-off seed): leave the estimate as is
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
            if !isfinite(abs(Δλ))
                return curr_λ
            end

            if fix_omega
                return ComplexF64(real(curr_λ) + real(Δλ), imag(curr_λ))
            else
                return curr_λ + Δλ
            end
            
        elseif degree == 3
            # 3rd derivative via finite diff on D1
            _, D1_plus2 = eval_D_and_deriv(real(curr_λ) + 2h, imag(curr_λ))
            _, D1_minus2 = eval_D_and_deriv(real(curr_λ) - 2h, imag(curr_λ))
            
            D3 = (D1_plus2 - 2*D1_plus + 2*D1_minus - D1_minus2) / (2 * h^3) # Approx, but (D1_plus - 2D1 + D1_minus)/h^2 is better
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
            if !isfinite(abs(Δλ))
                return curr_λ
            end
            
            if fix_omega
                return ComplexF64(real(curr_λ) + real(Δλ), imag(curr_λ))
            else
                return curr_λ + Δλ
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
    σ::S=0.0, ω_max=1e6, steps=500, parameter_independent_nmax=true, verbosity=0) where {P, S}
    
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    
    n_params = length(params_vec)
    n_pow_fixed = parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ) : nothing

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
