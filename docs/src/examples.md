# Examples

`InterpolatedNyquist.jl` handles a wide variety of dynamical systems.

## 1. Standard 4th-Order Delayed System
A common benchmark for delay-induced instability in high-order systems.

```julia
using InterpolatedNyquist
using GLMakie

# Define characteristic equation: λ^4 + λ^2 + ... + P*exp(-τλ)
function D_4th(λ::T, p) where T
    P, D = p
    τ = T(0.5); ζ = T(0.02)
    return (T(0.03) * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# Multi-threaded parameter sweep
Pv = LinRange(-2.0, 4.0, 60)
Dv = LinRange(-2.0, 5.0, 50)
params_vec = vec([(p, d) for p in Pv, d in Dv])

Z_ints, Z_raws, min_Ds, σ_ests, ω_crits = calculate_unstable_roots_p_vec(D_4th, params_vec)
```

## 2. Neutral Delay Differential Equations (NDDE)
Systems where the derivative depends on delayed derivatives. These systems often have infinitely many unstable roots and require robust solvers.

```julia
function D_neutral(λ::T, p) where T
    k, τ = p
    return λ + T(1.0) + k * λ * exp(-τ * λ)
end

p = (0.5, 1.0)
Z, _ = calculate_unstable_roots_direct(D_neutral, p)
```

## 3. Extracting Characteristic Equations from State-Space Models
If you have a DDE defined in the `DifferentialEquations.jl` format, you can extract its characteristic equation $D(\lambda, p)$ at any point.

```julia
using InterpolatedNyquist
using StaticArrays

# Turning model: d²x/dt² + 0.1 dx/dt + x = p * x(t - 0.5)
function turning_model(u, h, p, t)
    return SA[u[2], -u[1] - 0.1*u[2] + p[1]*h(p, t-0.5)[1]]
end

# Evaluate D(1.0 + 1.0im) for the parameter p = [2.0]
λ_test = 1.0 + 1.0im
p_test = [2.0]
D_val = get_D_from_model(turning_model, λ_test, p_test, Val(2))
```

## 4. Stability Boundary Refinement (MDBM)
For high-precision boundary tracing without high-resolution global grids.

```julia
using InterpolatedNyquist
using MDBM

# Axis list for parameters (P, D, ω)
axlist = [Axis(-2.0, 4.0, name="P"), Axis(-2.0, 5.0, name="D"), Axis(0.0, 100.0, name="ω")]
ω_coars = LinRange(0.0, 100.0, 50)

# Automatic enrichment and triangulation
boundary_mdbm, p_uniq, Ncirc, mesh_points, mesh_faces, mesh_colors, edge2plot_xyz = 
    argument_principle_solver_with_MDBM(D_4th, axlist, ω_coars)
```

## 6. Higher-Order Root Refinement
For maximum precision near boundaries or inside stable regions, roots can be refined using polynomial or Newton-Raphson approximations.

```julia
# Use the robust 3rd-order polynomial refinement (default)
zi, zr, md, es, wc = calculate_unstable_roots_direct(D_4th, (1.0, 0.5), 
    refinement_method=:Polynomial, refinement_degree=3)

# Or use Newton-Raphson for machine precision
zi, zr, md, es, wc = calculate_unstable_roots_direct(D_4th, (1.0, 0.5), 
    refinement_method=:Newton, refinement_steps=5)

# Post-processing individual root estimates
initial_root = es + 1im * wc
refined_root = refine_roots(D_4th, (1.0, 0.5), initial_root; method=:Newton, steps=3)
```
