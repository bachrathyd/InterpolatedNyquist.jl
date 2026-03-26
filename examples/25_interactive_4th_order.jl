using InterpolatedNyquist
using GLMakie
using MDBM
using LinearAlgebra

# Ensure GLMakie is active
GLMakie.closeall()
GLMakie.activate!()

# 1. Define characteristic equation D(λ, p)
function D_chareq(λ::T, p) where T
    P, D, c1, τ, ζ = p
    return (c1 * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 2. Setup Parameter Space
P_lims = (-2.01, 4.0)
D_lims = (-2.01, 5.0)

# 3. Create Type-Stable Wrapper Once (CRITICAL for Thread-Safety)
const P_TYPE = Tuple{Float64, Float64, Float64, Float64, Float64}
const wrapped_D = InterpolatedNyquist.NyquistWrapper{P_TYPE}(D_chareq)

# 4. Create Figure and Layout
fig = Figure(size = (1250, 900))
ax = GLMakie.Axis(fig[1, 1], xlabel = "P", ylabel = "D")

sg = SliderGrid(
    fig[1, 2][1, 1],
    (label = "c1", range = LinRange(0.01, 0.1, 100), startvalue = 0.03),
    (label = "τ (Delay)", range = LinRange(0.1, 2.0, 100), startvalue = 0.5),
    (label = "ζ (Damping)", range = LinRange(0.0, 0.1, 100), startvalue = 0.02),
    (label = "scale_x", range = LinRange(0.1, 5.0, 100), startvalue = 1.4),
    (label = "scale_y", range = LinRange(0.1, 5.0, 100), startvalue = 1.0),
    (label = "BF Res", range = 20:5:150, startvalue = 60),
    (label = "MDBM Iter", range = 1:6, startvalue = 3),
    width = 350,
    tellheight = false
)

# 4b. Toggles for Control (Vertically Arranged)
toggle_show_pts = Toggle(fig, active = false)
toggle_bf_active = Toggle(fig, active = true)
toggle_mdbm_active = Toggle(fig, active = true)
toggle_threaded = Toggle(fig, active = false)

t_grid = fig[1, 2][2, 1] = GridLayout(tellheight = true, halign = :left, padding = (10, 10, 10, 10))
t_grid[1, 1] = Label(fig, "Show MDBM Pts:")
t_grid[1, 2] = toggle_show_pts
t_grid[2, 1] = Label(fig, "Compute BF:")
t_grid[2, 2] = toggle_bf_active
t_grid[3, 1] = Label(fig, "Compute MDBM:")
t_grid[3, 2] = toggle_mdbm_active
t_grid[4, 1] = Label(fig, "MDBM Threaded:")
t_grid[4, 2] = toggle_threaded
rowgap!(t_grid, 5)

# 5. Reactive Observables
Pv_obs = Observable(LinRange(P_lims..., 60))
Dv_obs = Observable(LinRange(D_lims..., 60))
C_plot = Observable(zeros(60, 60))
ax_title = Observable("Interactive 4th Order Stability")

edge_x = Observable(Float32[])
edge_y = Observable(Float32[])

# MDBM points (S, U, B)
S_pts = Observable(Point2f[])
U_pts = Observable(Point2f[])
B_pts = Observable(Point2f[])

# MDBM Circle
ellipse_mdbm_x = Observable(Float32[])
ellipse_mdbm_y = Observable(Float32[])
center_mdbm_x = Observable(Float32[])
center_mdbm_y = Observable(Float32[])

# BF Circle
ellipse_bf_x = Observable(Float32[])
ellipse_bf_y = Observable(Float32[])
center_bf_x = Observable(Float32[])
center_bf_y = Observable(Float32[])

function update_plot()
    # Access values
    c1_val = sg.sliders[1].value[]
    τ_val = sg.sliders[2].value[]
    ζ_val = sg.sliders[3].value[]
    scale_x_val = Float64(sg.sliders[4].value[])
    scale_y_val = Float64(sg.sliders[5].value[])
    bf_res = sg.sliders[6].value[]
    mdbm_iters = sg.sliders[7].value[]
    
    # --- Part A: Brute Force Grid Sweep ---
    if toggle_bf_active.active[]
        Pv = LinRange(P_lims..., bf_res)
        Dv = LinRange(D_lims..., bf_res)
        Pv_obs[] = Pv
        Dv_obs[] = Dv
        
        params_vec = vec([(p, d, c1_val, τ_val, ζ_val) for p in Pv, d in Dv])
        Z_ints, Z_raws, min_Ds, σ_ests, ω_crits = 
            calculate_unstable_roots_p_vec(wrapped_D, params_vec; verbosity=0, ω_max=1000.0)
        
        Z_mat = reshape(Z_ints, bf_res, bf_res)
        σ_mat = reshape(σ_ests, bf_res, bf_res)
        C_plot[] = Z_mat .+ (Z_mat .== 0) .* σ_mat
        
        # BF Circle
        b_circle = find_largest_circle(Z_mat .== 0, Pv, Dv, scale_x=scale_x_val, scale_y=scale_y_val)
        if b_circle.R_scaled > 0
            pts = generate_ellipse_points(b_circle.x, b_circle.y, b_circle.R_scaled, 
                                          scale_x=scale_x_val, scale_y=scale_y_val)
            ellipse_bf_x[] = Float32.(pts[1]); ellipse_bf_y[] = Float32.(pts[2])
            center_bf_x[] = Float32[b_circle.x]; center_bf_y[] = Float32[b_circle.y]
        else
            ellipse_bf_x[] = Float32[]; ellipse_bf_y[] = Float32[]
            center_bf_x[] = Float32[]; center_bf_y[] = Float32[]
        end
    else
        # Clear heatmap and circles when BF is inactive
        C_plot[] = zeros(bf_res, bf_res)
        ellipse_bf_x[] = Float32[]; ellipse_bf_y[] = Float32[]
        center_bf_x[] = Float32[]; center_bf_y[] = Float32[]
    end
    
    # --- Part B: MDBM Boundary and Circle ---
    if toggle_mdbm_active.active[]
        function mdbm_objective(P::Float64, D::Float64)::Float64
            p_curr = (P, D, Float64(c1_val), Float64(τ_val), Float64(ζ_val))
            zi, zr, md, es, wc = calculate_unstable_roots_direct(wrapped_D, p_curr, verbosity=0, ω_max=1000.0)
            return (zi == 0) ? abs(es) : -abs(es)
        end
        
        ax_mdbm = [LinRange(P_lims..., 30), LinRange(D_lims..., 30)]
        prob = MDBM_Problem(mdbm_objective, ax_mdbm)
        MDBM.solve!(prob, mdbm_iters, verbosity=0, interpolationorder=0, doThreadprecomp=toggle_threaded.active[])
        interpolate!(prob, interpolationorder=1)
        
        # MDBM solution points
        xyz_sol = getinterpolatedsolution(prob)
        xy_val = getevaluatedpoints(prob)
        fval = getevaluatedfunctionvalues(prob)
        
        if !isempty(xyz_sol) && length(xyz_sol) >= 2
            B_pts[] = [Point2f(xyz_sol[1][i], xyz_sol[2][i]) for i in 1:length(xyz_sol[1])]
        else
            B_pts[] = Point2f[]
        end
        
        is_stable = fval .> 0
        S_pts[] = [Point2f(xy_val[1][i], xy_val[2][i]) for i in findall(is_stable)]
        U_pts[] = [Point2f(xy_val[1][i], xy_val[2][i]) for i in findall(.!is_stable)]
        
        # MDBM Boundary edges
        try
            DT = MDBM.connect(prob)
            if !isempty(DT) && !isempty(xyz_sol) && length(xyz_sol) >= 2
                ex = Float32.(reduce(hcat, [xyz_sol[1][getindex.(DT, 1)], xyz_sol[1][getindex.(DT, 2)], fill(NaN, length(DT))])'[:])
                ey = Float32.(reduce(hcat, [xyz_sol[2][getindex.(DT, 1)], xyz_sol[2][getindex.(DT, 2)], fill(NaN, length(DT))])'[:])
                edge_x[] = ex; edge_y[] = ey
            else
                edge_x[] = Float32[]; edge_y[] = Float32[]
            end
        catch
            edge_x[] = Float32[]; edge_y[] = Float32[]
        end
        
        # MDBM Circle
        m_circle = find_largest_circle(prob, scale_x=scale_x_val, scale_y=scale_y_val)
        if m_circle.valid
            pts = generate_ellipse_points(m_circle.x, m_circle.y, m_circle.R_scaled, 
                                          scale_x=scale_x_val, scale_y=scale_y_val)
            ellipse_mdbm_x[] = Float32.(pts[1]); ellipse_mdbm_y[] = Float32.(pts[2])
            center_mdbm_x[] = Float32[m_circle.x]; center_mdbm_y[] = Float32[m_circle.y]
            
            p_center = (m_circle.x, m_circle.y, Float64(c1_val), Float64(τ_val), Float64(ζ_val))
            zi, zr, md, es, wc = calculate_unstable_roots_direct(wrapped_D, p_center; ω_max=1000.0)
            ax_title[] = "Interactive 4th Order | Z_raw=$(round(zr, digits=3)), σ=$(round(es, digits=4)) at MDBM center"
        else
            ellipse_mdbm_x[] = Float32[]; ellipse_mdbm_y[] = Float32[]
            center_mdbm_x[] = Float32[]; center_mdbm_y[] = Float32[]
            ax_title[] = "Interactive 4th Order | No stable region found"
        end
    else
        edge_x[] = Float32[]; edge_y[] = Float32[]
        ellipse_mdbm_x[] = Float32[]; ellipse_mdbm_y[] = Float32[]
        center_mdbm_x[] = Float32[]; center_mdbm_y[] = Float32[]
        B_pts[] = Point2f[]; S_pts[] = Point2f[]; U_pts[] = Point2f[]
    end
end

# Trigger updates
for s in sg.sliders
    on(s.value) do _ update_plot() end
end
for t in [toggle_bf_active, toggle_mdbm_active, toggle_threaded]
    on(t.active) do _ update_plot() end
end

# Initial calculation
update_plot()

# 6. Visualization
#ax.title = ax_title

# Heatmap
hm = heatmap!(ax, Pv_obs, Dv_obs, C_plot, colormap = :viridis)

# MDBM Boundary Line
lines!(ax, edge_x, edge_y, color = :black, linewidth = 2, label = "MDBM Boundary")

# MDBM Points (toggled)
scatter!(ax, S_pts, color = :green, markersize = 6, label = "S", visible = toggle_show_pts.active)
scatter!(ax, U_pts, color = :red, markersize = 6, label = "U", visible = toggle_show_pts.active)
scatter!(ax, B_pts, color = :blue, markersize = 8, label = "B", visible = toggle_show_pts.active)

# Circles
lines!(ax, ellipse_mdbm_x, ellipse_mdbm_y, color = :yellow, linewidth = 3, label = "MDBM Circle")
scatter!(ax, center_mdbm_x, center_mdbm_y, color = :yellow, marker = :star5, markersize = 15)

lines!(ax, ellipse_bf_x, ellipse_bf_y, color = :cyan, linewidth = 2, label = "BF Circle", linestyle = :dash)
scatter!(ax, center_bf_x, center_bf_y, color = :cyan, marker = :circle, markersize = 10)

Colorbar(fig[2, 1], hm, vertical = false, label = "Stability Metric")
axislegend(ax, position = :rt)

println("Interactive plot with BF circle, MDBM iteration control, and toggles ready.")
display(fig)
