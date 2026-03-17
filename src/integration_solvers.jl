# Core logic for Method 2: Integration-based solvers
# This file is part of InterpolatedNyquist.jl

using DifferentialEquations
using ForwardDiff
using StaticArrays
using QuadGK

# Standardized Tag for ForwardDiff specialization
struct NyquistTag end
const StandardTag = ForwardDiff.Tag{NyquistTag, Float64}

"""
    get_n_power_max(D_func, p, σ=0.0; ω_large=1e6)

Estimates the highest power of the polynomial (n_power_max).
"""
function get_n_power_max(D_func, p, σ=0.0; ω_large=1e6)
    dual_ω = ForwardDiff.Dual{StandardTag}(ω_large, 1.0)
    # λ = σ + iω
    dual_λ = σ + 1im * dual_ω
    res = D_func(dual_λ, p)
    rv = real(res)
    iv = imag(res)
    D_val_re = ForwardDiff.value(rv)
    D_val_im = ForwardDiff.value(iv)
    dD_dω_re = ForwardDiff.partials(rv, 1)
    dD_dω_im = ForwardDiff.partials(iv, 1)
    n_est = ω_large * (dD_dω_re * D_val_re + dD_dω_im * D_val_im) / (D_val_re^2 + D_val_im^2)
    return round(Int, n_est), n_est
end

"""
    calculate_unstable_roots_direct(D_func, p, σ=0.0; ...)

Calculates the number of unstable roots using direct integration of the phase.
"""
function calculate_unstable_roots_direct(D_func::F, p::P, σ::S=0.0; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=AutoTsit5(Rosenbrock23()), 
    n_power_max=nothing, verbosity=0) where {F, P, S}

    min_D_sq = Ref(Inf)
    estimated_sigma = Ref(Inf)
    ω_crit = Ref(0.0)
    
    function phase_ode(y, params, ω)
        # Ensure we use the correct Tag for autodiff compatibility
        T_tag = (ω isa ForwardDiff.Dual) ? ForwardDiff.tagtype(ω) : StandardTag
        
        dual_ω = ForwardDiff.Dual{T_tag}(ForwardDiff.value(ω), 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)

        rv = real(res)
        iv = imag(res)
        re_val = ForwardDiff.value(rv)
        im_val = ForwardDiff.value(iv)
        dre = ForwardDiff.partials(rv, 1)
        dim = ForwardDiff.partials(iv, 1)

        d_sq = re_val^2 + im_val^2
        darg = (dim * re_val - dre * im_val) / d_sq

        if d_sq < min_D_sq[]
            min_D_sq[] = d_sq
            ω_crit[] = ForwardDiff.value(ω)
            estimated_sigma[] = -(re_val * dim - im_val * dre) / (dim^2 + dre^2)
        end

        return SA[darg]
    end

    prob = ODEProblem{false}(phase_ode, SA[0.0], (0.0, Float64(ω_max)))
    sol = solve(prob, solver, reltol=reltol, abstol=abstol, save_everystep=false, saveat=[ω_max], maxiters=Int(1e5))

    n_pow = n_power_max === nothing ? get_n_power_max(D_func, p, σ, ω_large=ω_max)[1] : n_power_max
    Z_raw = -(1.0 / π) * sol.u[end][1] + n_pow / 2.0

    return round(Int, Z_raw), Z_raw, sqrt(min_D_sq[]), estimated_sigma[], ω_crit[]
end

"""
    calculate_unstable_roots_p_vec(D_func, params_vec::AbstractVector; ...)

Vectorized stability sweep using multi-threading.
"""
function calculate_unstable_roots_p_vec(D_func::F, params_vec::AbstractVector{P}; 
    σ::S=0.0, ω_max=1e6, reltol=1e-5, abstol=1e-5, solver=AutoTsit5(Rosenbrock23()), 
    parameter_independent_nmax=true, verbosity=0) where {F, P, S}
    
    n_params = length(params_vec)
    n_pow_fixed = parameter_independent_nmax ? get_n_power_max(D_func, params_vec[1], σ, ω_large=ω_max)[1] : nothing

    Z_ints = zeros(Int, n_params)
    Z_raws = zeros(Float64, n_params)
    min_Ds = zeros(Float64, n_params)
    sigmas = zeros(Float64, n_params)
    crits = zeros(Float64, n_params)

    if verbosity > 0
        println("Calculating stability over $n_params points...")
    end

    @inbounds Threads.@threads for i in 1:n_params
        zi, zr, md, es, wc = calculate_unstable_roots_direct(D_func, params_vec[i], σ; 
            ω_max=ω_max, reltol=reltol, abstol=abstol, solver=solver, 
            n_power_max=n_pow_fixed, verbosity=verbosity)
        Z_ints[i] = zi
        Z_raws[i] = zr
        min_Ds[i] = md
        sigmas[i] = es
        crits[i] = wc
    end

    return Z_ints, Z_raws, min_Ds, sigmas, crits
end

"""
    calculate_unstable_roots_quadgk(D_func, p, σ=0.0; ω_max=1e6, reltol=1e-5, abstol=1e-5, n_power_max=nothing)

Calculates the number of unstable roots using QuadGK.jl (adaptive 1D quadrature).
"""
function calculate_unstable_roots_quadgk(D_func::F, p::P, σ::S=0.0; 
    ω_max=1e6, reltol=1e-5, abstol=1e-5, n_power_max=nothing) where {F, P, S}
    
    min_D_sq = Ref(Inf)
    estimated_sigma = Ref(Inf)
    ω_crit = Ref(0.0)
    
    function phase_integrand(ω)
        dual_ω = ForwardDiff.Dual{StandardTag}(ω, 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)
        rv = real(res)
        iv = imag(res)
        re_val = ForwardDiff.value(rv)
        im_val = ForwardDiff.value(iv)
        dre = ForwardDiff.partials(rv, 1)
        dim = ForwardDiff.partials(iv, 1)
        d_sq = re_val^2 + im_val^2
        if d_sq < min_D_sq[]
            min_D_sq[] = d_sq
            ω_crit[] = ω
            estimated_sigma[] = -(re_val * dim - im_val * dre) / (dim^2 + dre^2)
        end
        return (dim * re_val - dre * im_val) / d_sq
    end

    integral, err = quadgk(phase_integrand, 0.0, Float64(ω_max), rtol=reltol, atol=abstol)
    n_pow = n_power_max === nothing ? get_n_power_max(D_func, p, σ, ω_large=ω_max)[1] : n_power_max
    Z_raw = -(1.0 / π) * integral + n_pow / 2.0
    return round(Int, Z_raw), Z_raw, sqrt(min_D_sq[]), estimated_sigma[], ω_crit[]
end

"""
    calculate_unstable_roots_quadgk_p_vec(D_func, params_vec::AbstractVector; ...)

Vectorized stability sweep using QuadGK and multi-threading.
"""
function calculate_unstable_roots_quadgk_p_vec(D_func::F, params_vec::AbstractVector{P}; 
    σ::S=0.0, ω_max=1e6, reltol=1e-5, abstol=1e-5, parameter_independent_nmax=true, verbosity=0) where {F, P, S}
    
    n_params = length(params_vec)
    n_pow_fixed = parameter_independent_nmax ? get_n_power_max(D_func, params_vec[1], σ, ω_large=ω_max)[1] : nothing

    Z_ints = zeros(Int, n_params)
    Z_raws = zeros(Float64, n_params)
    min_Ds = zeros(Float64, n_params)
    sigmas = zeros(Float64, n_params)
    crits = zeros(Float64, n_params)

    if verbosity > 0
        println("Calculating stability (QuadGK) over $n_params points...")
    end

    @inbounds Threads.@threads for i in 1:n_params
        zi, zr, md, es, wc = calculate_unstable_roots_quadgk(D_func, params_vec[i], σ; 
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
function calculate_unstable_roots_fixed_step(D_func::F, p::P, σ::S=0.0; 
    ω_max=1e6, steps=1000, n_power_max=nothing) where {F, P, S}
    
    min_D_sq = Inf
    estimated_sigma = Inf
    ω_crit = 0.0
    
    h = Float64(ω_max) / steps
    integral = 0.0
    
    # Internal point calculation
    function get_darg(ω_val)
        dual_ω = ForwardDiff.Dual{StandardTag}(ω_val, 1.0)
        dual_λ = σ + 1im * dual_ω
        res = D_func(dual_λ, p)
        rv, iv = real(res), imag(res)
        re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
        dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)
        d_sq = re_val^2 + im_val^2
        
        # Track diagnostics
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

    n_pow = n_power_max === nothing ? get_n_power_max(D_func, p, σ, ω_large=ω_max)[1] : n_power_max
    Z_raw = -(1.0 / π) * integral + n_pow / 2.0
    return round(Int, Z_raw), Z_raw, sqrt(min_D_sq), estimated_sigma, ω_crit
end

"""
    calculate_unstable_roots_fixed_step_p_vec(D_func, params_vec::AbstractVector; ...)

Vectorized stability sweep using ultra-fast fixed-step integration.
"""
function calculate_unstable_roots_fixed_step_p_vec(D_func::F, params_vec::AbstractVector{P}; 
    σ::S=0.0, ω_max=1e6, steps=500, parameter_independent_nmax=true, verbosity=0) where {F, P, S}
    
    n_params = length(params_vec)
    n_pow_fixed = parameter_independent_nmax ? get_n_power_max(D_func, params_vec[1], σ, ω_large=ω_max)[1] : nothing

    Z_ints = zeros(Int, n_params)
    Z_raws = zeros(Float64, n_params)
    min_Ds = zeros(Float64, n_params)
    sigmas = zeros(Float64, n_params)
    crits = zeros(Float64, n_params)

    if verbosity > 0
        println("Calculating stability (Fixed-Step) over $n_params points...")
    end

    @inbounds Threads.@threads for i in 1:n_params
        zi, zr, md, es, wc = calculate_unstable_roots_fixed_step(D_func, params_vec[i], σ; 
            ω_max=ω_max, steps=steps, n_power_max=n_pow_fixed)
        Z_ints[i] = zi
        Z_raws[i] = zr
        min_Ds[i] = md
        sigmas[i] = es
        crits[i] = wc
    end

    return Z_ints, Z_raws, min_Ds, sigmas, crits
end
