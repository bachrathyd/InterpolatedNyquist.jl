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

# Include sub-files (they use the 'using' from above)
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

end
