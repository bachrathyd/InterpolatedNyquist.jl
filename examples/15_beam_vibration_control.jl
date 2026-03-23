using InterpolatedNyquist
using GLMakie
using MDBM

# 1. Infinite-Dimensional Beam Characteristic Equation
# Longitudinal beam: clamped at x=0, controlled at x=L.
# Control law: Force(L, t) = -Kp * Force(0, t-τ)
# D(λ) = cosh(γ(λ)*L) + Kp * exp(-λ*τ)
function D_chareq(λ::T, p) where T
    Kp, τ_val = p
    
    # Beam parameters
    L = T(1.0)
    ρA = T(1.0)
    EA = T(1.0)
    c_damping = T(0.1) # Internal damping
    
    # Wave propagation constant
    γ = sqrt((ρA * λ^2 + c_damping * λ) / EA)
    
    # Transcendental characteristic equation
    # Force(L) = EA*U'(L) = EA*B*γ*cosh(γL)
    # Force(0) = EA*U'(0) = EA*B*γ
    # Control: Force(L) + Kp*Force(0)*exp(-λτ) = 0
    # => cosh(γL) + Kp*exp(-λτ) = 0
    return cosh(γ * L) + Kp * exp(-λ * τ_val)
end

# 2. Hybrid Strategy
Kpv = LinRange(0.0, 2.0, 60)
tauv = LinRange(0.1, 3.0, 50)
params_vec = vec([(Kpv[i], tauv[j]) for i in 1:length(Kpv), j in 1:length(tauv)])

println("Grid sweep (Infinite DOF Beam Model)...")
# We use a larger ω_max because beam modes extend to infinity
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq, params_vec, ω_max=200.0, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(Kpv), length(tauv))
σ_mat_est = reshape(σ_ests_vec, length(Kpv), length(tauv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. MDBM Trace
println("\nTracing stability boundary with MDBM...")
function mdbm_wrapper(Kp, τ_val)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_chareq, (Kp, τ_val), ω_max=200.0, verbosity=0)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    return sign_val * abs(es)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(0.0, 2.0, 15), LinRange(0.1, 3.0, 15)])
@time MDBM.solve!(boundary_mdbm, 3, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# 4. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="Beam Vibration Control: Transcendental Stability Chart", 
    xlabel="Kp (Gain)", ylabel="τ (Delay)")

hm = heatmap!(ax, Kpv, tauv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM Boundary")
end

mkpath("output_figures")
save("output_figures/example_15.png", f)
display(f)
