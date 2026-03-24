"""
    find_largest_circle(S::BitMatrix, Xvector::AbstractVector, Yvector::AbstractVector; N::Int=0)
    find_largest_circle(S::BitMatrix; N::Int=0)

Finds the largest physical circle containing only `true` values within a boolean matrix `S`.

# Iterative Refinement (`N`)
If `N > 0`, the algorithm uses a coarse-to-fine multi-resolution search to drastically speed up 
computation on large matrices. 
1. It subsamples the matrix by taking every `2^N`th element.
2. It finds the best coarse center.
3. It iteratively halves the step size, searching only within a localized window around the 
   previous best center, until it reaches the native resolution (step size 1).
*Warning:* While blazing fast, `N > 0` may settle in a local maximum if the true largest 
circle is hidden behind a narrow bottleneck missed by the coarse subsampling. Default is `0` 
(exact global search).

# Arguments
- `S::BitMatrix`: A boolean matrix (`true` = valid space, `false` = obstacles).
- `Xvector::AbstractVector`: Physical row coordinates (must be strictly increasing).
- `Yvector::AbstractVector`: Physical column coordinates (must be strictly increasing).
- `N::Int`: Number of subsampling iterations. Default is 0 (exact search).

# Returns
A `NamedTuple`: `(R, i, j, x, y)` representing the radius, indices, and physical coordinates.
"""
function find_largest_circle(S::BitMatrix, Xvector::AbstractVector, Yvector::AbstractVector; N::Int=0)
    rows, cols = size(S)
    
    # Safety checks
    @assert length(Xvector) == rows "Length of Xvector must match number of rows in S"
    @assert length(Yvector) == cols "Length of Yvector must match number of columns in S"
    @assert issorted(Xvector) "Xvector must be sorted in ascending order"
    @assert issorted(Yvector) "Yvector must be sorted in ascending order"
    @assert N >= 0 "Iteration steps N must be non-negative"
    
    x_min, x_max = Float64(Xvector[1]), Float64(Xvector[end])
    y_min, y_max = Float64(Yvector[1]), Float64(Yvector[end])
    
    best_i, best_j = 0, 0
    best_R_sq = -1.0
    
    # Coarse-to-fine loop
    for lvl in N:-1:0
        step = 2^lvl
        
        # 1. Define the search space for centers
        if lvl == N
            # First pass: search the entire subsampled grid
            row_range = 1:step:rows
            col_range = 1:step:cols
        else
            # Refinement pass: search only locally around the previous best center
            prev_step = 2^(lvl + 1)
            search_radius = 2 * prev_step # Window size to catch the peak
            row_range = max(1, best_i - search_radius):step:min(rows, best_i + search_radius)
            col_range = max(1, best_j - search_radius):step:min(cols, best_j + search_radius)
        end
        
        # 2. Subsample obstacles to match current resolution
        obstacles = Tuple{Int, Int}[]
        for j in 1:step:cols
            for i in 1:step:rows
                if !S[i, j]
                    push!(obstacles, (i, j))
                end
            end
        end
        
        # 3. Collect and rank valid candidate centers in the current range
        centers = Tuple{Int, Int, Float64}[]
        for j in col_range
            for i in row_range
                if S[i, j]
                    x, y = Float64(Xvector[i]), Float64(Yvector[j])
                    bound = min(x - x_min, x_max - x, y - y_min, y_max - y)
                    push!(centers, (i, j, bound^2))
                end
            end
        end
        
        # If the coarse grid completely missed all 'true' values, abort this level
        if isempty(centers)
            continue 
        end
        
        # Sort for aggressive bounding-box pruning
        sort!(centers, by = item -> item[3], rev = true)
        
        local_best_R_sq = -1.0
        local_best_center = (best_i, best_j) # Default to previous best if nothing better is found
        
        # 4. Find the best center at this resolution
        for (i, j, bound_sq) in centers
            if bound_sq <= local_best_R_sq
                break # Pruning
            end
            
            min_dist_sq = bound_sq
            x_c, y_c = Float64(Xvector[i]), Float64(Yvector[j])
            
            for (obs_i, obs_j) in obstacles
                dist_sq = (x_c - Xvector[obs_i])^2 + (y_c - Yvector[obs_j])^2
                if dist_sq < min_dist_sq
                    min_dist_sq = dist_sq
                end
                
                if min_dist_sq <= local_best_R_sq
                    break # Early exit
                end
            end
            
            if min_dist_sq > local_best_R_sq
                local_best_R_sq = min_dist_sq
                local_best_center = (i, j)
            end
        end
        
        # Update globals for the next refinement pass
        best_i, best_j = local_best_center
        best_R_sq = local_best_R_sq
    end
    
    # If no circle was found at all
    if best_R_sq < 0
        return (R = 0.0, i = 0, j = 0, x = 0.0, y = 0.0)
    end
    
    return (
        R = sqrt(best_R_sq),
        x = Float64(Xvector[best_i]),
        y = Float64(Yvector[best_j]),        
        i = best_i, 
        j = best_j
    )
end

# Multiple Dispatch: Default to index space if vectors are omitted, while still accepting N
function find_largest_circle(S::BitMatrix; N::Int=0)
    return find_largest_circle(S, 1:size(S, 1), 1:size(S, 2); N=N)
end