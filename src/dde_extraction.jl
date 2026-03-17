# Core logic for DDE to CharEq extraction
# This file is part of InterpolatedNyquist.jl

using LinearAlgebra
using ForwardDiff
using StaticArrays

"""
    get_D_from_model(bc_model, λ::Complex, p, ::Val{N}; u_eq = zeros(SVector{N, Float64}), verbosity::Int = 0) where {N}

Extracts the characteristic equation value D(λ) from a black-box DDE model.
Uses ForwardDiff to extract the Jacobian matrix and calculates its determinant.
"""
function get_D_from_model(bc_model, λ::Complex, p, ::Val{N}; u_eq = zeros(SVector{N, Float64}), verbosity::Int = 0) where {N}
    
    # T_val adapts to external ForwardDiff layers (if λ is dual, T_val will be too)
    T_val = typeof(real(λ))
    
    # --- 1. PERTURBATIONS (Using Standard Tags) ---
    T_inner_tag = ForwardDiff.Tag{typeof(bc_model), T_val}
    
    A_duals = SVector{N, ForwardDiff.Dual{T_inner_tag, T_val, N}}(
        ntuple(i -> ForwardDiff.Dual{T_inner_tag, T_val, N}(
            zero(T_val), 
            ForwardDiff.Partials{N, T_val}(ntuple(j -> i == j ? one(T_val) : zero(T_val), Val(N)))
        ), Val(N))
    )
    
    # At t=0, exp(0) = 1
    u_current = u_eq .+ A_duals 
    
    # Mock history function
    h_mock(p_arg, t_arg; idxs=nothing) = begin
        exp_term = exp(λ * t_arg)
        if idxs === nothing
            return u_eq .+ A_duals .* exp_term
        else
            return u_eq[idxs] .+ A_duals[idxs] .* exp_term
        end
    end
    
    # Evaluate model
    du = bc_model(u_current, h_mock, p, zero(T_val))
    
    # Characteristic matrix equation: λ*A - (du - du_eq) = 0
    res = λ .* A_duals .- du
    
    # --- 2. JACOBIAN EXTRACTION (0-allocation SMatrix) ---
    get_p(x, j) = ForwardDiff.partials(x, j)
    
    M_lambda = SMatrix{N, N, Complex{T_val}, N*N}(
        ntuple(k -> begin
            i = (k - 1) % N + 1
            j = (k - 1) ÷ N + 1
            r = res[i]
            complex(get_p(real(r), j), get_p(imag(r), j))
        end, Val(N*N))
    )
    
    # --- 3. EQUILIBRIUM CHECK ---
    if verbosity > 0
        get_v(x) = ForwardDiff.value(x)
        val_eq = map(r -> complex(get_v(real(r)), get_v(imag(r))), res)
        
        _raw_float(x::Float64) = x
        _raw_float(x::ForwardDiff.Dual) = _raw_float(ForwardDiff.value(x))
        
        sum_err = sum(abs2, val_eq)
        if _raw_float(sum_err) > 1e-6
            @warn "At the given u_eq point, derivatives are not zero! Residual (du): $(val_eq)"
        end
    end
    
    return det(M_lambda)
end
