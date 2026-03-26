"""
    find_largest_circle(S::BitMatrix, Xvector::AbstractVector, Yvector::AbstractVector; 
                        N::Int=0, sub_pixel_depth::Int=5, scale_x::Float64=1.0, scale_y::Float64=1.0)
    find_largest_circle(S::BitMatrix; N::Int=0, sub_pixel_depth::Int=5, scale_x::Float64=1.0, scale_y::Float64=1.0)

Finds the largest physical circle containing only `true` values within a boolean matrix `S`.
Incorporates anisotropic scaling to correctly map ellipses in physical space.

# Keyword Arguments
- `N::Int`: Number of coarse subsampling iterations for speed. Default is `0`.
- `sub_pixel_depth::Int`: Number of fractional refinement steps. Default is `5`.
- `scale_x`, `scale_y`: Scaling factors to normalize the physical axes to a 1:1 aspect ratio.

# Returns
A `NamedTuple`: `(R_scaled, x, y, i, j)` where `R_scaled` is the radius in the scaled 
mathematical space, and `x, y` are the unscaled physical coordinates of the center.
"""
function find_largest_circle(S::BitMatrix, Xvector::AbstractVector, Yvector::AbstractVector; 
                             N::Int=0, sub_pixel_depth::Int=5, scale_x::Float64=1.0, scale_y::Float64=1.0)
    rows, cols = size(S)
    
    @assert length(Xvector) == rows "Length of Xvector must match number of rows in S"
    @assert length(Yvector) == cols "Length of Yvector must match number of columns in S"
    @assert issorted(Xvector) "Xvector must be sorted in ascending order"
    @assert issorted(Yvector) "Yvector must be sorted in ascending order"
    @assert N >= 0 && sub_pixel_depth >= 0 "Iteration steps must be non-negative"
    
    # 1. Pre-scale the coordinate vectors for blazing fast inner loops
    X_scaled = [Float64(x) * scale_x for x in Xvector]
    Y_scaled = [Float64(y) * scale_y for y in Yvector]
    
    x_min, x_max = X_scaled[1], X_scaled[end]
    y_min, y_max = Y_scaled[1], Y_scaled[end]
    
    # Helper function for linear interpolation of fractional indices
    function lerp_1d(V, idx::Float64)
        idx = clamp(idx, 1.0, Float64(length(V)))
        i = floor(Int, idx)
        if i == length(V)
            return Float64(V[end])
        end
        t = idx - i
        return (1.0 - t) * V[i] + t * V[i+1]
    end

    best_i, best_j = 0, 0
    best_R_sq = -1.0
    
    # --- PHASE 1: Integer Coarse-to-Fine Search ---
    for lvl in N:-1:0
        step = 2^lvl
        
        if lvl == N
            row_range = 1:step:rows
            col_range = 1:step:cols
        else
            prev_step = 2^(lvl + 1)
            search_radius = 2 * prev_step
            row_range = max(1, best_i - search_radius):step:min(rows, best_i + search_radius)
            col_range = max(1, best_j - search_radius):step:min(cols, best_j + search_radius)
        end
        
        obstacles = Tuple{Int, Int}[]
        for j in 1:step:cols
            for i in 1:step:rows
                if !S[i, j]
                    push!(obstacles, (i, j))
                end
            end
        end
        
        centers = Tuple{Int, Int, Float64}[]
        for j in col_range
            for i in row_range
                if S[i, j]
                    x, y = X_scaled[i], Y_scaled[j]
                    bound = min(x - x_min, x_max - x, y - y_min, y_max - y)^2
                    push!(centers, (i, j, bound))
                end
            end
        end
        
        if isempty(centers) continue end
        sort!(centers, by = item -> item[3], rev = true)
        
        local_best_R_sq = -1.0
        local_best_center = (best_i, best_j)
        
        for (i, j, bound_sq) in centers
            if bound_sq <= local_best_R_sq break end
            
            min_dist_sq = bound_sq
            x_c, y_c = X_scaled[i], Y_scaled[j]
            
            for (obs_i, obs_j) in obstacles
                dist_sq = (x_c - X_scaled[obs_i])^2 + (y_c - Y_scaled[obs_j])^2
                if dist_sq < min_dist_sq
                    min_dist_sq = dist_sq
                end
                if min_dist_sq <= local_best_R_sq break end
            end
            
            if min_dist_sq > local_best_R_sq
                local_best_R_sq = min_dist_sq
                local_best_center = (i, j)
            end
        end
        
        best_i, best_j = local_best_center
        best_R_sq = local_best_R_sq
    end
    
    # If no valid area is found, return early
    if best_R_sq < 0
        return (R_scaled = 0.0, x = 0.0, y = 0.0, i = 0.0, j = 0.0)
    end

    # --- PHASE 2: Fractional Sub-Pixel Refinement ---
    best_i_float = Float64(best_i)
    best_j_float = Float64(best_j)
    
    if sub_pixel_depth > 0
        full_obstacles = Tuple{Int, Int}[]
        for j in 1:cols, i in 1:rows
            if !S[i, j] push!(full_obstacles, (i, j)) end
        end

        for depth in 1:sub_pixel_depth
            step_size = 1.0 / (2^depth)
            search_radius = 2.0 * step_size 
            
            i_candidates = max(1.0, best_i_float - search_radius):step_size:min(Float64(rows), best_i_float + search_radius)
            j_candidates = max(1.0, best_j_float - search_radius):step_size:min(Float64(cols), best_j_float + search_radius)
            
            local_best_R_sq = best_R_sq
            local_best_i = best_i_float
            local_best_j = best_j_float
            
            for c_i in i_candidates
                for c_j in j_candidates
                    # Interpolate directly in the SCALED space
                    x_c = lerp_1d(X_scaled, c_i)
                    y_c = lerp_1d(Y_scaled, c_j)
                    
                    bound_sq = min(x_c - x_min, x_max - x_c, y_c - y_min, y_max - y_c)^2
                    if bound_sq <= local_best_R_sq continue end
                    
                    min_dist_sq = bound_sq
                    for (obs_i, obs_j) in full_obstacles
                        dist_sq = (x_c - X_scaled[obs_i])^2 + (y_c - Y_scaled[obs_j])^2
                        if dist_sq < min_dist_sq
                            min_dist_sq = dist_sq
                        end
                        if min_dist_sq <= local_best_R_sq break end
                    end
                    
                    if min_dist_sq > local_best_R_sq
                        local_best_R_sq = min_dist_sq
                        local_best_i = c_i
                        local_best_j = c_j
                    end
                end
            end
            
            best_R_sq = local_best_R_sq
            best_i_float = local_best_i
            best_j_float = local_best_j
        end
    end
    
    # We return the unscaled physical X/Y for the user by interpolating the original vectors
    return (
        R_scaled = sqrt(best_R_sq), 
        x = Float64(sub_pixel_depth > 0 ? lerp_1d(Xvector, best_i_float) : Xvector[best_i]),
        y = Float64(sub_pixel_depth > 0 ? lerp_1d(Yvector, best_j_float) : Yvector[best_j]),
        i = best_i_float, 
        j = best_j_float
    )
end

# Wrapper for default index space usage
function find_largest_circle(S::BitMatrix; N::Int=0, sub_pixel_depth::Int=5, scale_x::Float64=1.0, scale_y::Float64=1.0)
    return find_largest_circle(S, 1:size(S, 1), 1:size(S, 2); N=N, sub_pixel_depth=sub_pixel_depth, scale_x=scale_x, scale_y=scale_y)
end

"""
    find_largest_circle(true_pts, false_pts, boundary_pts; N=0, scale_x=1.0, scale_y=1.0)

Finds the largest inscribed ellipse using unordered 2D point clouds, incorporating 
anisotropic scaling to normalize the search space.

# Arguments
- `true_pts`: Array of valid 2D coordinates (e.g., `[(x1, y1), (x2, y2), ...]`). 
- `false_pts`: Array of invalid 2D coordinates (obstacles).
- `boundary_pts`: Array of precise 2D boundary points separating true and false regions.
- `N::Int`: Number of coarse subsampling iterations for speed. Default is 0.
- `scale_x`, `scale_y`: Scaling factors to normalize physical axes to a 1:1 aspect ratio.

# Returns
A `NamedTuple`: `(R_scaled, x, y)` representing the exact scaled radius and the 
unscaled physical coordinates of the center.
"""
function find_largest_circle(true_pts::AbstractVector, false_pts::AbstractVector, boundary_pts::AbstractVector; 
                             N::Int=0, scale_x::Float64=1.0, scale_y::Float64=1.0)
    
    # 1. Pre-scale all point clouds for blazing fast distance calculations
    t_scaled = [(Float64(pt[1]) * scale_x, Float64(pt[2]) * scale_y) for pt in true_pts]
    f_scaled = [(Float64(pt[1]) * scale_x, Float64(pt[2]) * scale_y) for pt in false_pts]
    b_scaled = [(Float64(pt[1]) * scale_x, Float64(pt[2]) * scale_y) for pt in boundary_pts]
    
    # OPTIMIZATION: Combine obstacles, putting boundary points FIRST.
    obstacles_scaled = vcat(b_scaled, f_scaled)
    
    best_R_sq = -1.0
    best_idx = 0 # Track the index so we can return the exact unscaled physical coordinates
    
    # Coarse-to-fine subsampling loop
    for lvl in N:-1:0
        step = 2^lvl
        
        # Subsample the candidate centers
        candidate_indices = 1:step:length(t_scaled)
        
        for idx in candidate_indices
            x_c, y_c = t_scaled[idx]
            
            min_dist_sq = Inf
            
            for obs in obstacles_scaled
                dist_sq = (x_c - obs[1])^2 + (y_c - obs[2])^2
                
                if dist_sq < min_dist_sq
                    min_dist_sq = dist_sq
                end
                
                # EARLY EXIT
                if min_dist_sq <= best_R_sq
                    break
                end
            end
            
            if min_dist_sq > best_R_sq && min_dist_sq != Inf
                best_R_sq = min_dist_sq
                best_idx = idx
            end
        end
    end
    
    if best_idx == 0
        return (R_scaled = 0.0, x = 0.0, y = 0.0)
    end
    
    return (
        R_scaled = sqrt(best_R_sq), 
        x = Float64(true_pts[best_idx][1]), # Return the original unscaled coordinates
        y = Float64(true_pts[best_idx][2])
    )
end

# Helper function to convert 2xN structures into an Array of (x, y) tuples
"""
    _to_tuples(pts)

Internal helper function to convert various 2D point cloud formats (2xN Matrix, 
Vector of Vectors [X, Y]) into a standard Vector of (x, y) tuples.
"""
function _to_tuples(pts)
    # Check if it is a 2xN Matrix
    if pts isa AbstractMatrix && size(pts, 1) == 2
        return [(Float64(pts[1, i]), Float64(pts[2, i])) for i in 1:size(pts, 2)]
        
    # Check if it is a Vector of Vectors: [X_coords, Y_coords]
    elseif pts isa AbstractVector && length(pts) == 2 && pts[1] isa AbstractVector
        return [(Float64(pts[1][i]), Float64(pts[2][i])) for i in 1:length(pts[1])]
        
    # If it happens to already be in the correct format, just pass it through
    else
        return pts 
    end
end

"""
    find_largest_circle_transposed(true_pts, false_pts, boundary_pts; N::Int=0, scale_x=1.0, scale_y=1.0)

A wrapper for `find_largest_circle` that accepts transposed 2D point cloud data.
Automatically parses `2xN` Matrices or `[X_vector, Y_vector]` Vectors into the correct format.
"""
function find_largest_circle_transposed(true_pts, false_pts, boundary_pts; 
                                        N::Int=0, scale_x::Float64=1.0, scale_y::Float64=1.0)
    # 1. Quickly translate the transposed data into paired (x, y) tuples
    t_pts = _to_tuples(true_pts)
    f_pts = _to_tuples(false_pts)
    b_pts = _to_tuples(boundary_pts)
    
    # 2. Call the core multi-dimensional function with the scaling parameters
    return find_largest_circle(t_pts, f_pts, b_pts; N=N, scale_x=scale_x, scale_y=scale_y)
end


"""
    refine_circle_robust(x_init, y_init, boundary_pts; U=[], scale_x=1.0, scale_y=1.0, 
                         max_iters=2000, tol=1e-6, decay_rate=0.9, num_angles=16)

An advanced radial pattern search with adaptive step sizing. It navigates narrow, 
angled corridors by searching in a full circle and smoothly shrinking/growing its 
step size to prevent getting trapped in local bottlenecks.
"""
function refine_circle_robust(x_init::Real, y_init::Real, boundary_pts::AbstractVector; 
                              U::AbstractVector=[], scale_x::Float64=1.0, scale_y::Float64=1.0, 
                              max_iters::Int=2000, tol::Float64=1e-6, 
                              decay_rate::Float64=0.9, num_angles::Int=36)
    
    if isempty(boundary_pts)
        return (R_scaled = 0.0, x = Float64(x_init), y = Float64(y_init), valid = false)
    end

    # Transform into scaled space
    x_s = Float64(x_init) * scale_x
    y_s = Float64(y_init) * scale_y
    bound_s = [(Float64(b[1]) * scale_x, Float64(b[2]) * scale_y) for b in boundary_pts]
    
    function get_R(cx, cy)
        minimum([sqrt((cx - b[1])^2 + (cy - b[2])^2) for b in bound_s])
    end

    current_R = get_R(x_s, y_s)
    
    # Start with a decent step size, and pre-calculate our search angles
    step_size = current_R * 0.2 
    angles = range(0, 2π, length=num_angles+1)[1:end-1]

    for iter in 1:max_iters
        best_x, best_y = x_s, y_s
        best_R = current_R
        moved = false
        
        # Cast a circular net to find the best possible direction
        for θ in angles
            test_x = x_s + cos(θ) * step_size
            test_y = y_s + sin(θ) * step_size
            test_R = get_R(test_x, test_y)
            
            if test_R > best_R
                best_R = test_R
                best_x = test_x
                best_y = test_y
                moved = true
            end
        end
        
        if moved
            # We found a better spot! Move there.
            x_s, y_s, current_R = best_x, best_y, best_R
            
            # MOMENTUM: If we are moving successfully, try expanding the step size slightly 
            # to sprint through wide open spaces (capped at 50% of current radius)
            step_size = min(step_size * 1.05, current_R * 0.25)
        else
            # We are stuck at this resolution. Use the gentle decay rate to squeeze tighter.
            step_size *= decay_rate
            if step_size < tol 
                break 
            end
        end
    end
    
    # Validation against false points (U)
    is_valid = true
    if !isempty(U)
        U_s = [(Float64(u[1]) * scale_x, Float64(u[2]) * scale_y) for u in U]
        min_U_dist = minimum([sqrt((x_s - u[1])^2 + (y_s - u[2])^2) for u in U_s])
        if min_U_dist < current_R - 1e-8
            is_valid = false
        end
    end
    return (
        R_scaled = current_R, 
        x = x_s / scale_x, 
        y = y_s / scale_y, 
        valid = is_valid
    )
end
"""
    generate_ellipse_points(x_center, y_center, R_scaled; scale_x=1.0, scale_y=1.0, Nellips=360)

Generates the `[X_points, Y_points]` arrays required to plot the resulting scaled 
circle as an ellipse in the original physical space.
"""
function generate_ellipse_points(x_center::Real, y_center::Real, R_scaled::Real; 
                                 scale_x::Float64=1.0, scale_y::Float64=1.0, Nellips::Int=360)
    
    theta = range(0, 2π, length=Nellips)
    
    # Un-scale the mathematical radius back into physical X/Y distances
    R_physical_x = R_scaled / scale_x
    R_physical_y = R_scaled / scale_y
    
    x_pts = x_center .+ R_physical_x .* cos.(theta)
    y_pts = y_center .+ R_physical_y .* sin.(theta)
    
    return [x_pts, y_pts]
end

"""
    find_largest_circle(mdbm; N=0, scale_x=1.0, scale_y=1.0, kwargs...)

Automatically extracts Stable (S), Unstable (U), and Boundary (B) point clouds from 
an `MDBM_Problem` object. It then performs a two-step optimization:
1. A fast, scaled grid search for an initial guess.
2. A robust, omnidirectional pattern search for mathematically exact sub-pixel refinement.

# Keyword Arguments
- `N::Int`: Number of subsampling steps for the initial guess. Default is `0`.
- `scale_x`, `scale_y`: Normalization scales to prevent anisotropic "pipe" traps.
- `max_iters`, `tol`, `decay_rate`, `num_angles`: Advanced tuning parameters passed 
  directly to the `refine_circle_robust` step.

# Returns
A `NamedTuple`: `(R_scaled, x, y, valid)` containing the exact scaled radius, 
the unscaled physical center coordinates, and a validation flag.
"""
function find_largest_circle(mdbm; N::Int=0, scale_x::Float64=1.0, scale_y::Float64=1.0,
                             max_iters::Int=2000, tol::Float64=1e-6, 
                             decay_rate::Float64=0.9, num_angles::Int=31)
    
    # 1. Extract data directly from the MDBM object
    B = getinterpolatedsolution(mdbm)
    xy_val = getevaluatedpoints(mdbm)
    fval = getevaluatedfunctionvalues(mdbm)
    
    # Create the boolean mask for Stable vs Unstable
    is_stable = fval .> 0
    
    # Generate the transposed 2xN Vector-of-Vector structures
    S = [xy_val[1][is_stable], xy_val[2][is_stable]]
    U = [xy_val[1][.!is_stable], xy_val[2][.!is_stable]]
    
    # 2. Phase 1: Fast Gridded Initial Guess
    guess = find_largest_circle_transposed(S, U, B; N=N, scale_x=scale_x, scale_y=scale_y)
    
    # Safety check: if the grid search completely failed, return zeroes
    if guess.R_scaled <= 0.0
        return (R_scaled = 0.0, x = guess.x, y = guess.y, valid = false)
    end
    
    # 3. Phase 2: Robust Omni-directional Refinement
    # Convert arrays to tuple format for the refinement engine
    B_tuples = _to_tuples(B)
    U_tuples = _to_tuples(U)
    
    final_circle = refine_circle_robust(guess.x, guess.y, B_tuples; 
                                        U = U_tuples, 
                                        scale_x = scale_x, 
                                        scale_y = scale_y,
                                        max_iters = max_iters, 
                                        tol = tol, 
                                        decay_rate = decay_rate, 
                                        num_angles = num_angles)
                                        
    return final_circle
end