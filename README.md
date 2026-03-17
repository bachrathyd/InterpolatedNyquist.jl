# InterpolatedNyquist.jl

`InterpolatedNyquist.jl` is a high-performance Julia package for determining the stability of delayed dynamical systems using Nyquist-based methods. It combines coarse brute-force sweeps with precise Multi-Dimensional Bisection Method (MDBM) boundary refinement.

## Features
- **Standardized Parameter Handling:** Consistent with `DifferentialEquations.jl`.
- **Hybrid Strategy:** Fast global sweeps for background metrics, high-precision MDBM for boundaries.
- **Multiple Solvers:** Stiff ODE solvers (`Rosenbrock23`), adaptive 1D quadrature (`QuadGK`), and ultra-fast fixed-step integration.
- **Automatic Polynomial Order:** Asymptotic estimation of $n_{max}$.
- **Physics Models:** Supports matrix-determinant systems, transcendental equations, FEM models, and neutral/distributed delays.

## Installation
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Citing
If you use `InterpolatedNyquist.jl` in your research, please cite the following paper (in preparation):

> Your Name, et al. "High-Performance Nyquist Stability Analysis of Delayed Dynamical Systems using MDBM and Autodiff-Enhanced Stiff Integration." Journal of Sound and Vibration, 2026.

See `CITATION.bib` for the BibTeX entry.

## Examples Directory
- `01` - `03`: Hybrid stability charts for standard systems.
- `04`: DDE to CharEq black-box extraction.
- `05`: Tracing high-resolution σ-contours.
- `06` - `08`: Performance and compilation benchmarks.
- `09`: Algebraic system stability.
- `10`: ODE vs QuadGK comparison.
- `11`: Distributed delay stability.
- `12`: Neutral delay system stability.
- `13`: Convergence study (Error vs Tolerance).
- `14`: 2DOF matrix system with delayed PD control.
- `15`: Infinite-dimensional transcendental beam model.
- `16`: Discretized FEM beam (30+ DOF).
- `17`: Ultra-heavy 50x50 matrix determinant system.
- `18`: Real-time goal benchmark (10,000 points).
- `19`: Fixed-step vs Adaptive solver benchmarking.
- `20`: High-gain neutral system stability.
