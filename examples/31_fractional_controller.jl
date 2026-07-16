# Example 31: Fractional-order PI^lambda controller of a fractional delayed
# plant -- reproduction of Gao, Zhai & Liu (2017), Example 1:
#   plant      G(s) = 5 e^{-0.4 s} / (10 s^0.5 + 1)
#   controller C(s) = kp + ki / s^lam
# Closed-loop characteristic function:
#   D(s) = s^lam (10 s^0.5 + 1) + 5 e^{-0.4 s} (kp s^lam + ki)
# Their charts show the stabilizing (kp, ki) regions, including regions with
# guaranteed stability DEGREE sigma_deg (all roots left of Re = -sigma_deg),
# which maps directly onto the shifted-line argument principle (σ = -sigma_deg).

using InterpolatedNyquist
using GLMakie
using MDBM

const LAM = 0.4               # try 1.5 for the second chart of the paper
function D_gao(s::T, p) where T
    kp, ki = p
    return s^T(LAM) * (10 * s^T(0.5) + one(T)) + 5 * exp(-T(0.4) * s) * (kp * s^T(LAM) + ki)
end

# Leading order is non-integer (10 s^{0.9}): estimated automatically
println("estimated leading order n = ", get_n_power_max(D_gao, (2.0, 3.0)))

kpv = LinRange(-1.0, 6.0, 80)
kiv = LinRange(-0.5, 8.0, 60)
params_vec = vec([(kp, ki) for kp in kpv, ki in kiv])

println("Grid sweep (fractional controller)...")
@time Z_ints, Z_raws, min_Ds, σ_ests, ω_crits =
    calculate_unstable_roots_p_vec(D_gao, params_vec; ω_max = 1e4, verbosity = 1)

Z_mat = reshape(Z_ints, length(kpv), length(kiv))
σ_mat = reshape(σ_ests, length(kpv), length(kiv))
C_plot = Z_mat .+ (Z_mat .== 0) .* σ_mat

f = Figure(size = (900, 600))
ax = GLMakie.Axis(f[1, 1], xlabel = "k_p", ylabel = "k_i",
    title = "Fractional PI^$(LAM) controller: stability degree regions (Gao et al. 2017, Ex. 1)")
hm = heatmap!(ax, kpv, kiv, C_plot, colormap = :viridis)
Colorbar(f[1, 2], hm, label = "Z (unstable) / σ_est (stable)")

# Boundaries for prescribed stability degrees via the shifted line σ = -σ_deg
for (i, σ_deg) in enumerate([0.0, 0.5, 1.0])
    function wrapper(kp, ki)::Float64
        zi, zr, md, es, wc = calculate_unstable_roots_direct(D_gao, (kp, ki), -σ_deg; ω_max = 1e4)
        sign_val = (max(zi, 0) == 0) ? 1.0 : -1.0
        return sign_val * abs(es + σ_deg)
    end
    bnd = MDBM_Problem(wrapper, [LinRange(-1.0, 6.0, 25), LinRange(-0.5, 8.0, 25)])
    MDBM.solve!(bnd, 4, verbosity = 0)
    xyz = getinterpolatedsolution(bnd)
    DT1 = MDBM.connect(bnd)
    if !isempty(DT1)
        edges = [reduce(hcat, [s[getindex.(DT1, 1)], s[getindex.(DT1, 2)], fill(NaN, length(DT1))])'[:] for s in xyz]
        lines!(ax, edges[1], edges[2], color = cgrad([:black, :red])[(i-1)/2],
            linewidth = 2, label = "σ_deg = $σ_deg")
    end
end

# Test points tabulated in the reference (λ = 0.4)
scatter!(ax, [2.0, 2.0, 2.0], [5.0, 3.0, 2.0], color = :white,
    strokecolor = :black, strokewidth = 1, markersize = 12)
axislegend(ax, position = :rb)

mkpath("output_figures")
save("output_figures/example_31.png", f)
display(f)
