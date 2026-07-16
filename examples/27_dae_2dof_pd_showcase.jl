# Example 27: Constrained 2-DOF system with delayed PD control (DAE showcase)
#
# Physical model (the main showcase system of the paper):
#   - m1: statically UNSTABLE main structure (inverted-pendulum-like, k1 < 0)
#         on its suspension (k1, c1 to ground)
#   - m2: absorber housing mounted on the structure (k2, c2 between m1 and m2)
#   - m3: instrumentation block RIGIDLY bolted to the absorber -> algebraic
#         constraint x3 - x2 = 0 with internal (Lagrange-multiplier) force Fc
#   - digital controller: delayed collocated PD feedback acting on m1
#         u(t) = -P*x1(t-τ) - D*v1(t-τ)
#
# First-order DESCRIPTOR form with SINGULAR mass matrix (7 states):
#   z = (x1, x2, x3, v1, v2, v3, Fc)
#   E * dz/dt = f(z, z(t-τ))  with  E = diag(1,1,1, m1,m2,m3, 0)
#
# The characteristic function D(λ) = det(λE - J) is extracted AUTOMATICALLY
# from the right-hand side via get_D_from_model (mass_matrix keyword) --
# no by-hand derivation of the characteristic equation is needed.
# For verification, the reduced (constraint-eliminated) 2-DOF quasi-polynomial
# is also derived by hand: the two must agree up to a constant factor.

using InterpolatedNyquist
using StaticArrays
using LinearAlgebra
using GLMakie
using MDBM

GLMakie.closeall()
GLMakie.activate!(; title="DAE 2-DOF + delayed PD showcase")

# ---------------------------------------------------------------------------
# 1. Model definition (DifferentialEquations.jl DDE signature)
# ---------------------------------------------------------------------------
const M1, M2, M3 = 1.0, 0.3, 0.2      # masses (m2+m3 = 0.5)
const K1, K2 = -1.0, 1.0              # stiffnesses (K1 < 0: statically unstable)
const C1, C2 = 0.05, 0.05             # dampings
const TAU = 0.5                       # feedback delay

function dae_rhs(z, h, p, t)
    P, D = p
    x1, x2, x3, v1, v2, v3, Fc = z
    x1d = h(p, t - TAU; idxs=1)        # x1(t-τ)
    v1d = h(p, t - TAU; idxs=4)        # v1(t-τ)
    u = -P * x1d - D * v1d             # delayed PD control force on m1
    return SA[
        v1,
        v2,
        v3,
        -K1 * x1 - C1 * v1 + K2 * (x2 - x1) + C2 * (v2 - v1) + u,
        -K2 * (x2 - x1) - C2 * (v2 - v1) + Fc,
        -Fc,
        x3 - x2,                       # algebraic constraint (zero mass-matrix row)
    ]
end

const E_MASS = SMatrix{7,7,Float64}(Diagonal(SA[1.0, 1.0, 1.0, M1, M2, M3, 0.0]))

# Characteristic function extracted automatically from the model:
D_dae(λ, p) = get_D_from_model(dae_rhs, λ, p, Val(7); mass_matrix=E_MASS)

# ---------------------------------------------------------------------------
# 2. Verification against the hand-derived reduced 2-DOF quasi-polynomial
#    (constraint eliminated: merged mass m23 = m2 + m3 at position x2 = x3)
# ---------------------------------------------------------------------------
function D_reduced(λ::T, p) where T
    P, D = p
    m23 = M2 + M3
    a11 = M1 * λ^2 + (C1 + C2) * λ + (K1 + K2) + (P + D * λ) * exp(-TAU * λ)
    a12 = -(C2 * λ + K2)
    a21 = -(C2 * λ + K2)
    a22 = m23 * λ^2 + C2 * λ + K2
    return a11 * a22 - a12 * a21
end

p_check = (1.3, 0.4)
λ_checks = [0.2 + 1.5im, -0.4 + 3.1im, 0.05 - 0.7im]
ratios = [D_dae(λ, p_check) / D_reduced(λ, p_check) for λ in λ_checks]
println("D_dae / D_reduced ratios (must be a λ-independent constant):")
println.(ratios)
@assert maximum(abs, ratios ./ ratios[1] .- 1) < 1e-8 "DAE extraction mismatch!"
println("DAE extraction VERIFIED against reduced quasi-polynomial.\n")

# ---------------------------------------------------------------------------
# 3. Hybrid Strategy Part 1: coarse grid sweep (interpolable coloring)
# ---------------------------------------------------------------------------
Pv = LinRange(0.0, 4.0, 90)
Dv = LinRange(-0.5, 3.0, 70)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

# DEMO SETTINGS: deliberately fast rather than maximally safe.
#   ω_max = 1e4  -- three decades above the highest resonance. ω_max = 1e6 is
#                   the safe default and costs ~50x more evaluations here
#                   (this system's velocity feedback makes the phase ripple
#                   decay only like 1/ω, so the march must resolve the whole
#                   range); 1e4 leaves a truncated tail of ~0.2/ω_max ~ 2e-5
#                   in Z̃, far below the 1/2 rounding threshold -> same chart.
#   tol   = 1e-4 -- one decade looser than the package default. The chart and
#                   the traced boundary are indistinguishable from the
#                   reference; only the count of a few boundary-adjacent
#                   pixels can differ by one (see the paper, Sec. 6.3).
# For a final, publication-quality chart use the defaults (tol 1e-5, ω_max 1e6).
println("Grid sweep (7x7 DAE determinant, $(length(params_vec)) points)...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec =
    calculate_unstable_roots_p_vec(D_dae, params_vec; ω_max=1e4,
        reltol=1e-4, abstol=1e-4, verbosity=1)

Z_mat_int = reshape(Z_ints_vec, length(Pv), length(Dv))
σ_mat_est = reshape(σ_ests_vec, length(Pv), length(Dv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# ---------------------------------------------------------------------------
# 4. Hybrid Strategy Part 2: high-resolution MDBM boundary trace
# ---------------------------------------------------------------------------
println("\nTracing stability boundary with MDBM...")
# The objective uses the DOMINANT root (several tracked minima, max real part),
# not the single closest one: the closest minimum switches root branch where
# another mode overtakes it, and the resulting jump in σ_est makes MDBM
# interpolate a spurious "boundary" point in the middle of the stable domain.
function mdbm_wrapper(pp, dd)::Float64
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_dae, (pp, dd);
        ω_max=1e4, reltol=1e-4, abstol=1e-4, n_roots_to_track=5)
    sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
    σ_dom = maximum(filter(isfinite, es); init=-Inf)
    return sign_val * abs(σ_dom)
end

boundary_mdbm = MDBM_Problem(mdbm_wrapper, [LinRange(0.0, 4.0, 30), LinRange(-0.5, 3.0, 30)])
@time MDBM.solve!(boundary_mdbm, 4, verbosity=1)
xyz_sol = getinterpolatedsolution(boundary_mdbm)
DT1 = MDBM.connect(boundary_mdbm)
edge2plot_xyz = [reduce(hcat, [i_sol[getindex.(DT1, 1)], i_sol[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for i_sol in xyz_sol]

# ---------------------------------------------------------------------------
# 5. Most robust stable point (largest inscribed ellipse)
# ---------------------------------------------------------------------------
circle_res = find_largest_circle(boundary_mdbm; N=200, scale_x=1.0, scale_y=2.6)
println("Most robust point: P=$(circle_res.x), D=$(circle_res.y), R=$(circle_res.R_scaled)")

# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
f = Figure(size=(1000, 650))
ax = GLMakie.Axis(f[1, 1], title="Constrained 2-DOF + delayed PD (DAE showcase)",
    xlabel="P (proportional gain)", ylabel="D (derivative gain)")
hm = heatmap!(ax, Pv, Dv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Z (unstable) / σ_est (stable)")

if !isempty(edge2plot_xyz)
    lines!(ax, edge2plot_xyz..., color=:black, linewidth=2, label="MDBM boundary")
end
ellipse_pts = generate_ellipse_points(circle_res.x, circle_res.y, circle_res.R_scaled;
    scale_x=1.0, scale_y=2.6)
lines!(ax, ellipse_pts[1], ellipse_pts[2], color=:red, linewidth=2, label="largest stable ellipse")
scatter!(ax, [circle_res.x], [circle_res.y], color=:red, markersize=10)

mkpath("output_figures")
save("output_figures/example_27.png", f)
display(f)
