# Core logic for Method 1: MDBM Enrichment
# This file is part of InterpolatedNyquist.jl

using MDBM
using Statistics
using DSP: unwrap
using DelaunayTriangulation
using GeometryBasics
using Interpolations

"""
    calculate_encirclement_number(D_values, omega_values; n_power_max=0)

Calculates the number of encirclements of the origin for a given list of complex values `D_values`
corresponding to a list of frequencies `omega_values`.
"""
function calculate_encirclement_number(D_values::AbstractVector{<:Complex}, omega_values::AbstractVector{<:Real}; n_power_max::Integer=0)
    if length(D_values) != length(omega_values)
        error("D_values and omega_values must have the same length.")
    end
    if length(D_values) < 2
        return Float64(n_power_max / 2)
    end

    # Sort by omega to trace the path correctly
    perm = sortperm(omega_values)
    omega_sorted = omega_values[perm]
    D_sorted = D_values[perm]

    # Filter unique omega values
    mask = unique(i -> omega_sorted[i], 1:length(omega_sorted))
    omega_unique = omega_sorted[mask]
    D_unique = D_sorted[mask]

    if length(D_unique) < 2
        return Float64(n_power_max / 2)
    end

    # Calculate phase and unwrap to get continuous phase change
    angles = angle.(D_unique)
    unwrapped_angles = unwrap(angles)
    total_phase_change = unwrapped_angles[end] - unwrapped_angles[1]

    # Number of encirclements
    N = -total_phase_change / (2π)

    # Add contribution from poles at infinity
    return N + n_power_max / 2
end

"""
    calculate_encirclement_number(D_func, omega_values, p; n_power_max=0, σ=0.0)

Calculates the number of encirclements by evaluating a characteristic equation `D_func(λ, p)` where `λ = σ + 1im * ω`.
"""
function calculate_encirclement_number(D_func::Function, omega_values::AbstractVector{<:Real}, p; n_power_max::Integer=0, σ=0.0)
    D_complex = [D_func(σ + 1im * ω, p) for ω in omega_values]
    return calculate_encirclement_number(D_complex, omega_values; n_power_max=n_power_max)
end

"""
    argument_principle_with_MDBM(D_func, mdbm, omega_coars; σ=0.0)

Enriches a coarse grid evaluation of the argument principle with MDBM results.
Assumes the last axis of the MDBM problem is ω.
`n_power_max` is calculated internally.
"""
function argument_principle_with_MDBM(D_func, mdbm, omega_coars; σ=0.0)
    # 1. Get data from MDBM
    D_re_im = getevaluatedfunctionvalues(mdbm)
    D_comp = [D[1] + 1im * D[2] for D in D_re_im]

    xyz_val = getevaluatedpoints(mdbm)
    
    # Assume the last axis is omega
    n_axes = length(xyz_val)
    p_axes = xyz_val[1:n_axes-1]
    omega_vals = xyz_val[n_axes]

    # 2. Group results by parameter collection for efficiency
    grouped_results = Dict{Any, Vector{Tuple{Float64, ComplexF64}}}()
    for i in 1:length(omega_vals)
        # Handle different parameter collection types
        p_val = if n_axes > 2
             Tuple(a[i] for a in p_axes)
        else
             p_axes[1][i]
        end
        
        datapoint = (omega_vals[i], D_comp[i])
        if !haskey(grouped_results, p_val)
            grouped_results[p_val] = []
        end
        push!(grouped_results[p_val], datapoint)
    end

    p_uniq_keys = collect(keys(grouped_results))
    Ncirc = zeros(length(p_uniq_keys))

    # 3. Loop over unique parameter collections and calculate Ncirc
    for (i, p_val) in enumerate(p_uniq_keys)
        # a. Get MDBM results for this point
        mdbm_results = grouped_results[p_val]
        om_mdbm = getindex.(mdbm_results, 1)
        D_mdbm = getindex.(mdbm_results, 2)

        # b. Calculate D for the extra low-res points
        D_extra = [D_func(σ + 1im * ω, p_val) for ω in omega_coars]

        # c. Calculate n_power_max for this parameter set
        # Use a large frequency to estimate polynomial order
        n_power_max, _ = get_n_power_max(D_func, p_val, σ; ω_large=1e6)

        # d. Combine MDBM, extra, and symmetric points
        om_combined = vcat(om_mdbm, omega_coars)
        D_combined = vcat(D_mdbm, D_extra)
        om_full = vcat(om_combined, -om_combined)
        D_full = vcat(D_combined, conj.(D_combined))

        # e. Use the dedicated function to calculate encirclement number
        Ncirc[i] = calculate_encirclement_number(D_full, om_full; n_power_max=n_power_max)
    end

    return p_uniq_keys, Ncirc
end

"""
    trinangulation_of_MDBM_results(mdbm, p_uniq, Ncirc)

Triangulates the MDBM results for visualization.
"""
function trinangulation_of_MDBM_results(mdbm, p_uniq, Ncirc)
    xyz_sol = getinterpolatedsolution(mdbm)
    
    # Assuming 2 parameter dimensions for plotting
    pts_solution = [Point2f(xyz_sol[1][i], xyz_sol[2][i]) for i in 1:length(xyz_sol[1])]
    
    # Correctly handle p_uniq which might be Tuples
    pts_sampled = map(p_uniq) do p
        if p isa Tuple
            Point2f(p[1], p[2])
        else
            Point2f(p, 0.0)
        end
    end
    
    all_pts = vcat(pts_solution, pts_sampled)
    n_sol = length(pts_solution)

    DT1 = MDBM.connect(mdbm)
    edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

    # 2. Triangulate with constraints
    const_edges = Set{Tuple{Int,Int}}(DT1)
    tri = triangulate(all_pts, segments=const_edges)

    # 3. Build a "Decomposed" Mesh
    mesh_points = Point2f[]
    mesh_faces = GeometryBasics.TriangleFace{Int}[]
    mesh_colors = Float32[]
    face_count = 0

    for T in each_triangle(tri)
        u, v, w = T
        u > 0 && v > 0 && w > 0 || continue

        # Calculate average Ncirc for this triangle (only for sampled points)
        vals = [Ncirc[idx-n_sol] for idx in (u, v, w) if idx > n_sol]

        if !isempty(vals)
            avg_c = Float32(mean(vals))
            push!(mesh_points, all_pts[u], all_pts[v], all_pts[w])
            push!(mesh_faces, GeometryBasics.TriangleFace(face_count * 3 + 1, face_count * 3 + 2, face_count * 3 + 3))
            append!(mesh_colors, [avg_c, avg_c, avg_c])
            face_count += 1
        end
    end
    return mesh_points, mesh_faces, mesh_colors, edge2plot_xyz
end

"""
    argument_principle_solver_with_MDBM(D_func, axlist, ω_coars; σ=0.0)

Complete workflow for MDBM enrichment.
"""
function argument_principle_solver_with_MDBM(D_func, axlist, ω_coars; σ=0.0)
    
    function Dchar_foo_ReIm(p_all...)
        # Assume last parameter is ω, rest are p
        ω = p_all[end]
        p = if length(p_all) > 2 
            Tuple(p_all[1:end-1])
        else
            p_all[1]
        end
        D = D_func(σ + 1im * ω, p)
        return real(D), imag(D)
    end

    @info "Boundary Tracing (MDBM)"
    boundary_mdbm = MDBM_Problem(Dchar_foo_ReIm, axlist)
    Niter = 4
    MDBM.solve!(boundary_mdbm, Niter, verbosity=1, checkneighbourNum=2, doThreadprecomp=true, normp=10.0, ncubetolerance=0.6)

    @info "Stability Verification (Enrichment)"
    p_uniq, Ncirc = argument_principle_with_MDBM(D_func, boundary_mdbm, ω_coars; σ=σ)

    @info "Triangulation"
    mesh_points, mesh_faces, mesh_colors, edge2plot_xyz = trinangulation_of_MDBM_results(boundary_mdbm, p_uniq, Ncirc)

    return boundary_mdbm, p_uniq, Ncirc, mesh_points, mesh_faces, mesh_colors, edge2plot_xyz
end
