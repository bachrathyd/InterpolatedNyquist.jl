module InterpolatedNyquist

using LinearAlgebra
using StaticArrays
using Statistics
using DSP: unwrap
using MDBM
using DelaunayTriangulation
using GeometryBasics
using Interpolations
using ForwardDiff
using DifferentialEquations
using PrecompileTools

# Include sub-files
include("mdbm_enrichment.jl")
include("integration_solvers.jl")
include("dde_extraction.jl")

# Exports
export calculate_encirclement_number,
       argument_principle_with_MDBM,
       trinangulation_of_MDBM_results,
       argument_principle_solver_with_MDBM,
       get_n_power_max,
       calculate_unstable_roots_direct,
       calculate_unstable_roots_p_vec,
       calculate_unstable_roots_quadgk,
       calculate_unstable_roots_quadgk_p_vec,
       calculate_unstable_roots_fixed_step,
       calculate_unstable_roots_fixed_step_p_vec,
       get_D_from_model

# Precompilation Workload
@setup_workload begin
    # Representative characteristic equation
    function D_precompile(λ::T, p) where T
        P, D = p
        τ = T(0.5); ζ = T(0.02)
        return (T(0.03) * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
    end
    
    # Common parameter types
    p_tuple = (2.0, 1.0)
    p_vec = [2.0, 1.0]
    
    @compile_workload begin
        # Precompile single point solvers
        calculate_unstable_roots_direct(D_precompile, p_tuple; ω_max=100.0)
        calculate_unstable_roots_quadgk(D_precompile, p_tuple; ω_max=100.0)
        calculate_unstable_roots_fixed_step(D_precompile, p_tuple; ω_max=100.0, steps=10)
        
        # Precompile vectorized sweeps
        params_vec = [p_tuple, (1.0, 0.5)]
        calculate_unstable_roots_p_vec(D_precompile, params_vec; ω_max=100.0)
        
        # Precompile DDE extraction
        function turning_model_pre(u, h, p, t)
            return SA[u[2], -u[1] - 0.1*u[2] + p[1]*h(p, t-0.5)[1]]
        end
        get_D_from_model(turning_model_pre, 1.0 + 1.0im, p_vec, Val(2))
    end
end

end
