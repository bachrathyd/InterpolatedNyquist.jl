using InterpolatedNyquist
using GLMakie
using MDBM

# Ensure GLMakie is active
GLMakie.closeall()
GLMakie.activate!()

# Fractional-Order Delayed System: D(s) = s^α + a*s^β + k*e^{-sτ}
function D_fractional(λ::T, p) where T
    α, β, a, k, τ = p
    return λ^α + a * λ^β + k * exp(-τ * λ)
end

# 1. Setup Reactive Parameters and Grid
res = 75 # Resolution for the background grid
kv = LinRange(0.0, 5.0, res)
τv = LinRange(0.1, 2.0, res)

# 2. Create Figure and SliderGrid
fig = Figure(size = (1200, 800))
sg = SliderGrid(
    fig[1, 2],
    (label = "α (Order 1)", range = LinRange(1.01, 1.99, 100), startvalue = 1.8),
    (label = "β (Order 2)", range = LinRange(0.01, 0.99, 100), startvalue = 0.8),
    (label = "a (Damping)", range = LinRange(0.0, 2.0, 100), startvalue = 0.5),
    width = 350,
    tellheight = false
)

# 3. Reactive Observables
C_plot = Observable(zeros(res, res))
edge_x = Observable(Float32[])
edge_y = Observable(Float32[])

function update_plot()
    # Access slider values
    α = sg.sliders[1].value[]
    β = sg.sliders[2].value[]
    a = sg.sliders[3].value[]
    
    # --- Part A: Fast Grid Sweep for Heatmap ---
    params_vec = vec([(α, β, a, k, τ) for k in kv, τ in τv])
    Z_ints, Z_raws, min_Ds, σ_ests, ω_crits = 
        calculate_unstable_roots_fixed_step_p_vec(D_fractional, params_vec; ω_max=50.0, steps=200)
    
    Z_mat = reshape(Z_ints, res, res)
    σ_mat = reshape(σ_ests, res, res)
    C_plot[] = Z_mat .+ (Z_mat .== 0) .* σ_mat
    
    # --- Part B: MDBM Boundary Tracing ---
    # Objective function for MDBM based on the current slider values
    # We add explicit types to the arguments and return value to help MDBM inference
    function mdbm_objective(k::Float64, τ::Float64)::Float64
        # Capture current slider values as Float64
        p_curr = (Float64(α), Float64(β), Float64(a), k, τ)
        zi, zr, md, es, wc = calculate_unstable_roots_fixed_step(D_fractional, p_curr; ω_max=50.0, steps=200)
        
        sign_val = (zi == 0) ? 1.0 : -1.0
        val = sign_val * abs(es)
        return isnan(val) ? 0.0 : Float64(val)
    end
    
    # Run a quick MDBM (low iterations for interactivity)
    # Explicitly providing the axes as a typed vector
    ax_mdbm = LinRange{Float64, Int64}[LinRange(0.0, 5.0, 11), LinRange(0.1, 2.0, 11)]
    prob = MDBM_Problem(mdbm_objective, ax_mdbm)
    MDBM.solve!(prob, 3) 
    
    # Extract edges for plotting with defensive checks
    try
        xyz_sol = getinterpolatedsolution(prob)
        DT = MDBM.connect(prob)
        
        # Check if any solutions and connections exist
        if !isempty(DT) && !isempty(xyz_sol) && length(xyz_sol) >= 2 && !isempty(xyz_sol[1])
            # Efficiently build the line segments with NaN separators
            # Convert to Float32 for Makie consistency
            ex = Float32.(reduce(hcat, [xyz_sol[1][getindex.(DT, 1)], xyz_sol[1][getindex.(DT, 2)], fill(NaN, length(DT))])'[:])
            ey = Float32.(reduce(hcat, [xyz_sol[2][getindex.(DT, 1)], xyz_sol[2][getindex.(DT, 2)], fill(NaN, length(DT))])'[:])
            edge_x[] = ex
            edge_y[] = ey
        else
            # Clear lines if no solution is found
            edge_x[] = Float32[]
            edge_y[] = Float32[]
        end
    catch e
        # Silently handle MDBM extraction errors (e.g. if everything is identical)
        edge_x[] = Float32[]
        edge_y[] = Float32[]
    end
end

# Trigger update when sliders move
for slider in sg.sliders
    on(slider.value) do _
        update_plot()
    end
end

# Initial calculation
update_plot()

# 4. Visualization Layout
ax = GLMakie.Axis(fig[1, 1], title = "Interactive Fractional Stability (Heatmap + MDBM Boundary)", 
          xlabel = "Gain k", ylabel = "Delay τ")

# Plot Heatmap
hm = heatmap!(ax, kv, τv, C_plot, colormap = :viridis, colorrange = (-0.5, 2.5))

# Plot MDBM Boundary Lines
lines!(ax, edge_x, edge_y, color = :black, linewidth = 2.5, label = "MDBM Boundary")

Colorbar(fig[2, 1], hm, vertical = false, label = "Stability Index (0 = Stable, >0 = Unstable Roots)")

# Legend for clarity
axislegend(ax, position = :rt)

println("Interactive plot with MDBM boundary ready.")
display(fig)
