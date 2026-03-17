# Performance Optimization Report

This document tracks the step-by-step performance improvements made to the `InterpolatedNyquist.jl` integration-based stability solvers.

## Benchmark Configuration
- **System:** 4th Order Dynamical System (Example 02).
- **Grid Sweep:** 20x20 (400 points).
- **MDBM Trace:** 15x15 initial grid, 3 iterations.
- **Hardware:** Multi-threaded execution (`Threads.@threads`).

---

## Baseline Performance
Initial state of the refactored code (v0.1.0).
- **Grid Sweep:** 55.12 ms (46,834 allocations: 2.15 MiB)
- **MDBM Trace:** 1.096 s (1,750,586 allocations: 86.75 MiB)

---

## Optimization Step 1: Pre-calculate `n_power_max`
The polynomial order `n` was previously re-estimated for every single point in a parameter sweep using high-frequency asymptotics. 
- **Change:** `calculate_unstable_roots_p_vec` now computes `n_power_max` once at the start of the sweep and passes it down.
- **Result:** Significant reduction in redundant function evaluations.

---

## Optimization Step 2: Fix Type Instability & Closure Boxing
The `ForwardDiff.Tag` was being re-created inside the ODE kernel, and captured variables were causing "closure boxing" (storing values on the heap).
- **Change:** 
    1. Pre-calculated `TagType` outside the `phase_ode` kernel.
    2. Used explicit type parameters `{F, P, S}` in function signatures to ensure specialization.
    3. Simplified the ODE state to a single scalar (`phase`) while keeping diagnostics in `Ref` to avoid numerical stiffness issues.
- **Result:** Faster kernel execution and fewer allocations.

---

## Final Results (v0.1.1)
- **Grid Sweep:** 44.03 ms (~20% Speedup)
- **MDBM Trace:** 0.883 s (~18% Speedup)
- **Memory Allocations:** Reduced by ~70% in grid sweeps and ~30% in MDBM tracing.

## Summary of Changes
| Version | Grid Sweep (ms) | MDBM Trace (s) | Allocations (Grid) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 55.12 | 1.096 | 46,834 |
| **Optimized** | 44.03 | 0.883 | 14,022 |

These changes ensure that the package is not only robust and standardized but also highly efficient for large-scale stability analysis.
