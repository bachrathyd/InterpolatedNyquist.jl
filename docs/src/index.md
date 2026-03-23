# InterpolatedNyquist.jl

`InterpolatedNyquist.jl` is a high-performance Julia package for determining the stability of delayed dynamical systems using Nyquist-based methods. It provides a suite of tools for calculating the number of unstable roots by treating the argument principle as an integration problem.

## Core Methodology

The package transforms the traditional Nyquist argument principle into an Ordinary Differential Equation (ODE) or an adaptive quadrature problem. By integrating the rate of change of the phase of the characteristic equation $D(\lambda, p)$ along the imaginary axis, we can robustly determine the number of encirclements of the origin.

### Key Advantages:
- **Autodiff-Enhanced:** Uses `ForwardDiff.jl` for exact calculation of phase derivatives, eliminating discretization errors.
- **Stiff-ODE Robustness:** Leverages `DifferentialEquations.jl` to handle "stiff" regions where the characteristic contour passes close to the origin.
- **MDBM Enrichment:** Combines global sweeps with the Multi-Dimensional Bisection Method for high-precision boundary refinement.
- **Generalization:** Works with any transcendental equation, high-dimensional FEM models, and neutral/distributed delays.

## Installation

```julia
using Pkg
Pkg.add("InterpolatedNyquist")
```

## Quick Start

### 1. Define your Characteristic Equation
The equation MUST accept two arguments: a complex frequency `λ` and a unified parameter collection `p`.

```julia
function D_chareq(λ::T, p) where T
    P, D = p
    τ = T(0.5); ζ = T(0.02)
    return (T(0.03) * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end
```

### 2. Calculate Stability at a Single Point
```julia
using InterpolatedNyquist
p = (2.0, 1.0)
Z, Z_raw, min_D, σ_est, ω_crit = calculate_unstable_roots_direct(D_chareq, p; ω_max=100.0)
println("Number of unstable roots: ", Z)
```

### 3. Perform a Vectorized Sweep
```julia
params_vec = [(p, 1.0) for p in LinRange(0.0, 5.0, 100)]
Z_vec, _, _, _, _ = calculate_unstable_roots_p_vec(D_chareq, params_vec)
```

## Solvers

The package provides three main integration-based solvers:
1. `calculate_unstable_roots_direct`: Uses `DifferentialEquations.jl` (Stiff-ODE). Most robust.
2. `calculate_unstable_roots_quadgk`: Uses `QuadGK.jl` (Adaptive Quadrature). Fast for smooth paths.
3. `calculate_unstable_roots_fixed_step`: Trapezoidal integration. Ultra-fast for real-time visualization.

For boundary refinement and interpolation, see the **MDBM Enrichment** section in the API reference.
