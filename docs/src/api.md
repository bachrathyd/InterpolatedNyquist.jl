# API Reference

This page lists the public functions provided by `InterpolatedNyquist.jl`.

## MDBM Enrichment (Method 1)

```@docs
calculate_encirclement_number
argument_principle_with_MDBM
triangulation_of_MDBM_results
argument_principle_solver_with_MDBM
sensitivity_mapping_with_MDBM
```

## Integration-based Solvers (Method 2)

```@docs
calculate_unstable_roots_direct
calculate_unstable_roots_p_vec
calculate_unstable_roots_quadgk
calculate_unstable_roots_quadgk_p_vec
calculate_unstable_roots_fixed_step
calculate_unstable_roots_fixed_step_p_vec
get_n_power_max
```

## DDE Extraction

```@docs
get_D_from_model
```
