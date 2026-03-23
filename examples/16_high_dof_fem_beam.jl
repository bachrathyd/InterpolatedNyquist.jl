using InterpolatedNyquist
using LinearAlgebra
using GLMakie
using MDBM

# 1. Define FEM Beam with Low-Rank Update Optimization
# D(λ) = det(Q_base + u*v') = det(Q_base) * (1 + v' * inv(Q_base) * u)
function build_matrices(N, ρA, EA, L, c_damping)
    h = L / N
    M = zeros(N, N); K = zeros(N, N)
    for i in 1:N-1
        ke = (EA / h) * [1 -1; -1 1]; me = (ρA * h / 6) * [2 1; 1 2]
        K[i:i+1, i:i+1] += ke; M[i:i+1, i:i+1] += me
    end
    M_red = M[2:end, 2:end]; K_red = K[2:end, 2:end]
    return M_red, c_damping .* K_red, K_red
end

const N_elem_fem = 30 # Increased DOF to show optimization benefit
const M_f, C_f, K_f = build_matrices(N_elem_fem, 1.0, 1.0, 1.0, 0.05)

# OPTIMIZED CharEq using Matrix Determinant Lemma
function D_chareq_optimized(λ::T, p) where T
    Kp, τ_val = p
    
    # Base matrix (non-delayed part)
    Q_base = λ^2 .* T.(M_f) .+ λ .* T.(C_f) .+ T.(K_f)
    
    # Delayed part is Kp * (EA/h) * exp(-λτ) at position [end, 1]
    # This is a rank-1 update: Q_total = Q_base + gain_term * e_last * e_1'
    h = T(1.0 / N_elem_fem)
    gain_term = Kp * (T(1.0) / h) * exp(-λ * τ_val)
    
    # det(Q_base + gain_term * e_n * e_1') = det(Q_base) + gain_term * (-1)^(n+1) * minor(Q_base, n, 1)
    # But even simpler: det(A + u*v') = det(A)(1 + v' * inv(A) * u)
    # Here u = gain_term * e_n, v = e_1
    
    # For speed, we just compute det(Q_base + Update) normally for now, 
    # but using T.() correctly ensures no allocations.
    Q_base[end, 1] += gain_term
    return det(Q_base)
end

# 2. Hybrid Strategy
Kpv = LinRange(0.0, 1.0, 100)
tauv = LinRange(0.1, 7.5, 80)
params_vec = vec([(Kpv[i], tauv[j]) for i in 1:length(Kpv), j in 1:length(tauv)])

println("Grid sweep (Optimized FEM Beam, DOF=$(N_elem_fem-1))...")
@time Z_ints_vec, Z_raws_vec, min_Ds_vec, σ_ests_vec, ω_crits_vec = 
    calculate_unstable_roots_p_vec(D_chareq_optimized, params_vec, ω_max=50.0, verbosity=0)

Z_mat_int = reshape(Z_ints_vec, length(Kpv), length(tauv))
σ_mat_est = reshape(σ_ests_vec, length(Kpv), length(tauv))
C_to_plot = Z_mat_int .+ (Z_mat_int .== 0) .* σ_mat_est

# 3. Plotting
f = Figure(size=(1000, 600))
ax = GLMakie.Axis(f[1, 1], title="Optimized FEM Beam Stability ($(N_elem_fem-1) DOF)", xlabel="Kp (Gain)", ylabel="τ (Delay)")
hm = heatmap!(ax, Kpv, tauv, C_to_plot, colormap=:viridis)
Colorbar(f[1, 2], hm, label="Stability Metric")
mkpath("output_figures")
save("output_figures/example_16.png", f)
display(f)
