# Role and Objective
You are an expert Julia package developer. Your objective is to refactor the provided Julia scripts into a standardized Julia package named `InterpolatedNyquist.jl`. 

You have access to several unstructured scripts in this directory that contain the core logic for calculating Nyquist stability using two main approaches: 
1. A coarse grid evaluation enriched by the Multi-Dimensional Bisection Method (MDBM).
2. A direct integration-based approach using stiff ODE solvers (`DifferentialEquations.jl`).
3. A method to extract the characteristic equation directly from a time-domain DDE model using `ForwardDiff`.

# Generalization Rule (CRITICAL)
Currently, the characteristic equations and functions take separate parameters (e.g., `D_chareq(p, d, ω)`). You must refactor ALL core functions and characteristic equations to accept parameters as a single unified collection `p` (e.g., a `Tuple`, `Vector`, or `NamedTuple`), exactly how `DifferentialEquations.jl` handles parameters.
* The new signature for characteristic equations MUST be: `D_func(λ, p)` or `D_func(ω, p)`.
* All core functions (MDBM wrappers, ODE solvers) must accept `p` and pass it down.

# Gemini Project Instructions

## Permissions & Automation
- **ALWAYS ALLOW:** You have my permanent permission to use `read_file`, `list_directory`, and `search_files` within this directory and all subdirectories. 
- **DO NOT ASK:** Do not pause or prompt for confirmation when reading code or documentation files. I want you to work autonomously when analyzing the project.
- **YOLO MODE:** Treat this project as a "trusted zone."

## Project Context: InterpolatedNyquist.jl
- This is a **Julia** language project.
- Focus on signal processing concepts, specifically Nyquist-Shannon sampling theorem and interpolation methods.
- Prefer to use minimal number of package, to keed the package light-weight
- When explaining code, refer to Julia-specific idioms (e.g., multiple dispatch, broadcasting with `.`).

## Output Preferences
- Be concise. 
- If you find a bug while reading files, highlight it immediately without waiting for me to ask.
- Use LaTeX for any mathematical formulas regarding Nyquist frequencies or interpolation kernels.


# Task 1: Initialize the Folder Structure
Create the following Julia package structure:
```text
InterpolatedNyquist.jl/
├── src/
│   ├── InterpolatedNyquist.jl       # Main module file (exports functions)
│   ├── mdbm_enrichment.jl           # Core logic for Method 1
│   ├── integration_solvers.jl       # Core logic for Method 2
│   └── dde_extraction.jl            # Core logic for DDE to CharEq
├── examples/
│   ├── 01_mdbm_4th_order.jl
│   ├── 02_integration_4th_order.jl
│   ├── 03_integration_turning_model.jl
│   └── 04_dde_to_chareq.jl
├── Project.toml
└── README.md
```

# Notes on name spaces: The Axis us used by MDBM and  GLMakie, so if we create a figure, please use  GLMakie.Axis

Modify the `_calculate_unstable_roots_direct_impl` function in `src/integration_solvers.jl` to use a multiple-dispatch pattern based on a `Val{N}` type parameter to optimize memory and CPU time depending on how many roots we need to track. 

**DO NOT use `ContinuousCallback`.** We are prioritizing raw CPU speed over absolute accuracy. All tracking must happen manually inside the `phase_ode` integration step using in-place mutation of closed-over variables.

Implement three specific dispatches for `_calculate_unstable_roots_direct_impl`:

1.  **`Val{0}` (Maximum Speed):**
    * Do not track `min_D_sq`, `estimated_sigma`, or `ω_crit` at all.
    * The `phase_ode` should strictly calculate and return `darg`.
    * Return only `(round(Int, Z_raw), Z_raw)`.

2.  **`Val{1}` (Single Root Tracking):**
    * Track only the single closest root.
    * Use a closed-over `Ref{Float64}` for `min_D_sq` and a `Ref{Complex{Float64}}` to store the combined `σ_est + im * ω_crit`.
    * Update these references inside `phase_ode` if `d_sq < min_D_sq[]`.
    * Return `(round(Int, Z_raw), Z_raw, sqrt(min_D_sq[]), real(root_ref[]), imag(root_ref[]))`.

3.  **`Val{N}` (Multi-Root Tracking):**
    * Track up to `N` local minima of `|D|^2`.
    * Use an `MVector{N, Float64}` (from `StaticArrays`) for storing the local minimum `d_sq` values, initialized to `Inf`.
    * Use an `MVector{N, ComplexF64}` to store the corresponding `σ_est + im * ω_crit` values, initialized to `NaN + NaN*im`.
    * Inside `phase_ode`, detect a local minimum. (You can do this by tracking the previous step's `d_sq` and `d_sq_derivative` to find where the derivative crosses from negative to positive). 
    * When a local minimum is found, calculate its `σ_est` and `ω_crit` as a `Complex` number.
    * If this new minimum is smaller than the largest value currently in our `MVector`, insert it into the array (keeping the arrays sorted by `d_sq` from smallest to largest). 
    * Return `(round(Int, Z_raw), Z_raw, sqrt.(d_sq_vec), real.(roots_vec), imag.(roots_vec))`.

Ensure all code remains type-stable, allocation-free inside the ODE loop, and compatible with the existing `NyquistWrapper` and `ForwardDiff` logic.

4.  **The defailt value of should be N=1 (`Val{1}`) this way all the previouse code sould be complatible with the new implementaion**
    * Add a new example, which compares the CPU time of a single points evaluation of similar problem as in : 02_integration_4th_order.jl, use the fix paramtere p=-0.2 d=0.5. Compare Val{0}, Val{1} and Val{10}.
    for the selected points plot the contour lines fo Re(D)=0 and Im(D)=0 (so we can see the exact root location at the intersection) and then scatterplot the approximated roots by Val{10} (use different marke to the closest one provided by Val{1}). The title should contain the Z_raw and sigma of the closest root.






Further task:Furthremore, alwasy save the figures, and meke a clear documnetation, with all the feautes, and examples, and description how to use it. Make all the necessary documentation which is needed for Julia package registration.
   Furthermore, I would like to write a journal paper about it, so make a latex folred, in which you write the journal article, basic idea of nyquist stabiolity. Solition one. Enriching the coars grib with the MDB solution in which    
   the number of roots getting much betterin and only a coars resolution is satisfactor. + If we combine it withe a triangulation of tha arease with the same points and using the constraind edges in triangulation provided by the       
   MDBM then we can also reduce the grid resolution in the paramteres space too, becaue of the interpolation capability of the MDBM. Next chaper, use stiff  ode solver to the the best resolution at any point, it is most roust in a     
   single points calculation. (anyway, if low reolustion is enough quarature it ok, but the stiff equation solver is much more robust and still very fast, due p the excellent Julia Diff.Eq.jl implementation. Futhermot, if we store      
   the closest points, or basde on that we can approximate the colsest root, thatn we can alos use the interpolation capability of the MDBM late (or if we just want to find the stilbity limit). This is super fast. And usfull,
   because not only the boundary is given, but the sensitivity , the robustness inside the stable area. This is super efficinet. And can be applied to almost any kind of diffet which is not time dependent (in the concluasion there     
   should be a note, that for time dependent system infinity Hill dterminant based calculation could be used, however it is out of socpe, and for thet the robust calcuation would be hard, it would depend on the truncation sime of      
   the hill matirx, but for this system ther are more robust and also fast solution in e..g: time domins ) .

   After this (but maybe in the appendix) these should be many example. You can serarch similar problem in the literure, which is similart tou our examples and provide them as a tescase, or just use these examples as is.

   We should also discuss the CPU tha for different methods, MDBM-endriched - here we should state that the triangulation is the bottleneck, so we just analise the integration base Bfure for colored maps and MDBM solution for high resolution stability limit only , based on stiff integrator and/or Quadrature, and so on....