# InterpolatedNyquist.jl

`InterpolatedNyquist.jl` is a high-performance Julia package for determining the stability of delayed dynamical systems using Nyquist-based methods. It combines coarse brute-force sweeps with precise Multi-Dimensional Bisection Method (MDBM) boundary refinement.

## Features
- **Standardized Parameter Handling:** Consistent with `DifferentialEquations.jl` where all parameters are passed as a unified collection `p`.
- **Hybrid Strategy:** Fast global sweeps for background metrics, high-precision MDBM for boundaries.
- **Multiple Solvers:** Stiff ODE solvers (`Rosenbrock23`), adaptive 1D quadrature (`QuadGK`), and ultra-fast fixed-step integration.
- **Automatic Polynomial Order:** Asymptotic estimation of $n_{max}$.
- **Physics Models:** Supports matrix-determinant systems, transcendental equations, FEM models, and neutral/distributed delays.

## Installation
```julia
using Pkg
Pkg.add("InterpolatedNyquist")
```

## Basic Usage

1. Define a characteristic equation accepting a complex frequency `λ` and a parameter collection `p`.
2. Use solvers like `calculate_unstable_roots_direct` or perform batch sweeps using `calculate_unstable_roots_p_vec`.
