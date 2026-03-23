# Core logic for Method 2: Integration-based solvers
# This file is part of InterpolatedNyquist.jl

using DifferentialEquations
using ForwardDiff
using StaticArrays
# QuadGK and FunctionWrappers are imported in the main module

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
    get_n_power_max(D_func, p, σ=0.0; ω_large=1e6)

Estimates the highest power of the polynomial (n_power_max).
"""
function get_n_power_max(@nospecialize(D_func), p::P, σ=0.0; ω_large=1e6) where P
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    return _get_n_power_max_impl(wrapped_D, p, σ; ω_large=ω_large)
end

function _get_n_power_max_impl(D_func::NyquistWrapper{P}, p::P, σ=0.0; ω_large=1e6) where P
    dual_ω = ForwardDiff.Dual{StandardTag}(ω_large, 1.0)
    dual_λ = σ + 1im * dual_ω
    res = D_func(dual_λ, p)
    rv, iv = real(res), imag(res)
    D_val_re, D_val_im = ForwardDiff.value(rv), ForwardDiff.value(iv)
    dD_dω_re, dD_dω_im = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)
    n_est = ω_large * (dD_dω_re * D_val_re + dD_dω_im * D_val_im) / (D_val_re^2 + D_val_im^2)
    return n_est
end

"""
    calculate_unstable_roots_direct(D_func, p, σ=0.0; ...)

Calculates the number of unstable roots using direct integration of the phase.
Use `n_roots_to_track` to optimize:
- 0: Max speed, only Z calculation.
- 1: Track the closest root (default).
- N: Track up to N local minima.
"""
function calculate_unstable_roots_direct(@nospecialize(D_func), p::P, σ::S=0.0; 
    n_roots_to_track=1,
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=AutoTsit5(Rosenbrock23()), 
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6)) where {P, S}

    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    return _calculate_unstable_roots_direct_impl(wrapped_D, p, σ, Val(n_roots_to_track); 
        ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver, 
        n_power_max=n_power_max, verbosity=verbosity, maxiters=maxiters)
end

# Default for backward compatibility
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S; 
    kwargs...) where {P, S}
    return _calculate_unstable_roots_direct_impl(D_func, p, σ, Val(1); kwargs...)
end

# Val{0}: Maximum Speed (No tracking)
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S, ::Val{0}; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=AutoTsit5(Rosenbrock23()), 
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6)) where {P, S}

    function phase_ode(y, params, ω)
        pure_ω = max(ForwardDiff.value(ω), 1e-9)
        dual_ω = ForwardDiff.Dual{StandardTag}(pure_ω, 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)

        rv, iv = real(res), imag(res)
        re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
        dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)

        d_sq = re_val^2 + im_val^2
        if d_sq < 1e-20
            return SA[0.0]
        end

        return SA[(dim * re_val - dre * im_val) / d_sq]
    end

    prob = ODEProblem{false}(phase_ode, SA[0.0], (0.0, Float64(ω_max)))
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=maxiters)

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ, ω_large=ω_max) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    return round(Int, Z_raw), Z_raw
end

# Val{1}: Single Root Tracking
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S, ::Val{1}; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=AutoTsit5(Rosenbrock23()), 
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6)) where {P, S}

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
        if d_sq < 1e-20
            return SA[0.0]
        end

        if d_sq < min_D_sq[]
            min_D_sq[] = d_sq
            est_sigma = -(re_val * dim - im_val * dre) / (dim^2 + dre^2)
            root_ref[] = est_sigma + 1im * ForwardDiff.value(ω)
        end

        return SA[(dim * re_val - dre * im_val) / d_sq]
    end

    prob = ODEProblem{false}(phase_ode, SA[0.0], (0.0, Float64(ω_max)))
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=maxiters)

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ, ω_large=ω_max) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    return round(Int, Z_raw), Z_raw, sqrt(min_D_sq[]), real(root_ref[]), imag(root_ref[])
end

# Val{N}: Multi-Root Tracking
function _calculate_unstable_roots_direct_impl(D_func::NyquistWrapper{P}, p::P, σ::S, ::Val{N}; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=AutoTsit5(Rosenbrock23()), 
    n_power_max=nothing, verbosity=0, maxiters=Int(1e6)) where {P, S, N}

    d_sq_vec = MVector{N, Float64}(fill(Inf, N))
    roots_vec = MVector{N, ComplexF64}(fill(NaN + NaN*im, N))
    prev_d_sq_deriv = Ref(0.0)
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
        d_sq_deriv = 2*(re_val * dre + im_val * dim)
        curr_ω = ForwardDiff.value(ω)

        # Detect local minimum: derivative crosses 0 from below
        if prev_d_sq_deriv[] < 0 && d_sq_deriv > 0 && curr_ω > prev_ω[]
            est_sigma = -(re_val * dim - im_val * dre) / (dim^2 + dre^2)
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

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ, ω_large=ω_max) : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    return round(Int, Z_raw), Z_raw, sqrt.(d_sq_vec), real.(roots_vec), imag.(roots_vec)
end

"""
    calculate_unstable_roots_p_vec(D_func, params_vec::AbstractVector; ...)

Vectorized stability sweep using multi-threading.
"""
function calculate_unstable_roots_p_vec(@nospecialize(D_func), params_vec::AbstractVector{P}; 
    n_roots_to_track=1,
    σ::S=0.0, ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=AutoTsit5(Rosenbrock23()), 
    parameter_independent_nmax=true, verbosity=0, maxiters=Int(1e6)) where {P, S}
    
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    
    n_params = length(params_vec)
    n_pow_fixed = parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ, ω_large=ω_max) : nothing

    if n_roots_to_track == 1
        Z_ints = zeros(Int, n_params)
        Z_raws = zeros(Float64, n_params)
        min_Ds = zeros(Float64, n_params)
        sigmas = zeros(Float64, n_params)
        crits = zeros(Float64, n_params)

        if verbosity > 0
            println("Calculating stability over $n_params points (tracking 1 root)...")
        end

        @inbounds Threads.@threads for i in 1:n_params
            zi, zr, md, es, wc = _calculate_unstable_roots_direct_impl(wrapped_D, params_vec[i], σ, Val(1); 
                ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver, 
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters)
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
            println("Calculating stability over $n_params points (max speed)...")
        end

        @inbounds Threads.@threads for i in 1:n_params
            zi, zr = _calculate_unstable_roots_direct_impl(wrapped_D, params_vec[i], σ, Val(0); 
                ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver, 
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters)
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
            println("Calculating stability over $n_params points (tracking $n_roots_to_track roots)...")
        end

        @inbounds Threads.@threads for i in 1:n_params
            zi, zr, md, es, wc = _calculate_unstable_roots_direct_impl(wrapped_D, params_vec[i], σ, Val(n_roots_to_track); 
                ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver, 
                n_power_max=n_pow_fixed, verbosity=verbosity, maxiters=maxiters)
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
        if d_sq < 1e-20
            return 0.0
        end
        if d_sq < min_D_sq[]
            min_D_sq[] = d_sq
            ω_crit[] = ω
            estimated_sigma[] = -(re_val * dim - im_val * dre) / (dim^2 + dre^2)
        end
        return (dim * re_val - dre * im_val) / d_sq
    end

    integral, err = quadgk(phase_integrand, 0.0, Float64(ω_max), rtol=reltol, atol=abstol)
    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ, ω_large=ω_max) : n_power_max
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
    n_pow_fixed = parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ, ω_large=ω_max) : nothing

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
        
        if d_sq < 1e-20
            return 0.0
        end

        if d_sq < min_D_sq
            min_D_sq = d_sq
            ω_crit = ω_val
            estimated_sigma = -(re_val * dim - im_val * dre) / (dim^2 + dre^2)
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

    n_pow = n_power_max === nothing ? _get_n_power_max_impl(D_func, p, σ, ω_large=ω_max) : n_power_max
    Z_raw = -(1.0 / π) * integral + n_pow / 2.0
    return round(Int, Z_raw), Z_raw, sqrt(min_D_sq), estimated_sigma, ω_crit
end

"""
    calculate_unstable_roots_fixed_step_p_vec(D_func, params_vec::AbstractVector; ...)

Vectorized stability sweep using ultra-fast fixed-step integration.
"""
function calculate_unstable_roots_fixed_step_p_vec(@nospecialize(D_func), params_vec::AbstractVector{P}; 
    σ::S=0.0, ω_max=1e6, steps=500, parameter_independent_nmax=true, verbosity=0) where {P, S}
    
    wrapped_D = (D_func isa NyquistWrapper{P}) ? D_func : NyquistWrapper{P}(D_func)
    
    n_params = length(params_vec)
    n_pow_fixed = parameter_independent_nmax ? _get_n_power_max_impl(wrapped_D, params_vec[1], σ, ω_large=ω_max) : nothing

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
