# Study s01: the showcase system (constrained 2-DOF DAE + delayed PD).
# Produces:
#   figures/fig_method_walkthrough.pdf   (locus, integrand + adaptive steps, phase)
#   figures/fig_showcase_hybrid.pdf      (hybrid chart + robust ellipse)
#   data/showcase_summary.csv            (verification, timings, selected points)

include(joinpath(@__DIR__, "systems.jl"))

# ---------------------------------------------------------------------------
# 0. Verification: automatic DAE extraction vs hand-derived quasi-polynomial
# ---------------------------------------------------------------------------
p_check = (1.3, 0.4)
λ_checks = [0.2 + 1.5im, -0.4 + 3.1im, 0.05 - 0.7im]
ratios = [D_showcase(λ, p_check) / D_showcase_reduced(λ, p_check) for λ in λ_checks]
ratio_dev = maximum(abs, ratios ./ ratios[1] .- 1)
@assert ratio_dev < 1e-12 "DAE extraction mismatch (dev=$(ratio_dev))"
@info "showcase DAE extraction verified" ratio_dev

# ---------------------------------------------------------------------------
# 1. Hybrid chart: coarse sweep + MDBM boundary + robust ellipse
# ---------------------------------------------------------------------------
nP, nD = FAST[] ? (45, 35) : (90, 70)
Pv = LinRange(SHOWCASE_PRANGE..., nP)
Dv = LinRange(SHOWCASE_DRANGE..., nD)

grid = with_cache("s01_domgrid_$(nP)x$(nD)") do
    sweep_grid_dominant(D_showcase, Pv, Dv; ω_max = 1e4)
end
C = combined_metric(grid.Z, grid.sigma)

bnd = with_cache("s01_mdbm") do
    mdbm_boundary(D_showcase, SHOWCASE_PRANGE, SHOWCASE_DRANGE;
        ngrid = 30, Niter = FAST[] ? 3 : 4, ω_max = 1e4)
end

const ELL_SX = 1.0
const ELL_SY = (SHOWCASE_PRANGE[2] - SHOWCASE_PRANGE[1]) / (SHOWCASE_DRANGE[2] - SHOWCASE_DRANGE[1])
circ = find_largest_circle(bnd.prob; N = 5, scale_x = ELL_SX, scale_y = ELL_SY,
    num_angles = 32, tol = 1e-3)
ell = generate_ellipse_points(circ.x, circ.y, circ.R_scaled; scale_x = ELL_SX, scale_y = ELL_SY)

# Pick a representative stable point (deep stable) and unstable point (Z = 2).
# NaN (no finite tracked root) must map to Inf too, or argmin returns it.
stable_idx = argmin(replace(x -> (isnan(x) || x >= 0) ? Inf : x, C))
iu = findall(grid.Z .== 2)
unstable_idx = isempty(iu) ? argmax(grid.Z) : iu[cld(length(iu), 2)]
p_stable = (Pv[stable_idx[1]], Dv[stable_idx[2]])
p_unstable = (Pv[unstable_idx[1]], Dv[unstable_idx[2]])
@info "walkthrough points" p_stable p_unstable

fig = Figure(size = (W_ONEHALF, W_ONEHALF * 0.62))
ax = MAxis(fig[1, 1], xlabel = "proportional gain  P", ylabel = "derivative gain  D")
hm = stability_panel!(ax, Pv, Dv, C; edges = bnd.edges)
Colorbar(fig[1, 2], hm, label = "Z (unstable)  /  σ̂ (stable)")
lines!(ax, ell[1], ell[2]; color = :red, linewidth = 1.2)
scatter!(ax, [circ.x], [circ.y]; color = :red, marker = :star5, markersize = 9)
scatter!(ax, [p_stable[1]], [p_stable[2]]; color = :white, strokecolor = :black,
    strokewidth = 0.8, marker = :circle, markersize = 8)
text!(ax, p_stable[1], p_stable[2]; text = " S", color = :white, fontsize = 9)
scatter!(ax, [p_unstable[1]], [p_unstable[2]]; color = :white, strokecolor = :black,
    strokewidth = 0.8, marker = :rect, markersize = 8)
text!(ax, p_unstable[1], p_unstable[2]; text = " U", color = :white, fontsize = 9)
save_fig(fig, "fig_showcase_hybrid")

# ---------------------------------------------------------------------------
# 2. Method walkthrough figure at the two marked points
# ---------------------------------------------------------------------------
sol_s = phase_ode_solution(D_showcase, p_stable; ω_max = 1e4)
sol_u = phase_ode_solution(D_showcase, p_unstable; ω_max = 1e4)
n_s = get_n_power_max(D_showcase, p_stable)
Zraw_s = -sol_s.u[end][1] / π + n_s / 2
Zraw_u = -sol_u.u[end][1] / π + n_s / 2
@info "walkthrough winding numbers" Zraw_s Zraw_u nsteps_s = length(sol_s.t) nsteps_u = length(sol_u.t)

ω_show = 8.0
ω_dense = range(1e-6, ω_show; length = 3000)

fig2 = Figure(size = (W_FULL, W_FULL * 0.30))

# (a) normalized Mikhailov locus
axa = MAxis(fig2[1, 1], xlabel = "Re D / (1+|D|)", ylabel = "Im D / (1+|D|)",
    title = "(a) normalized locus of D(iω)", aspect = DataAspect())
for (pp, col, lab) in ((p_stable, Makie.wong_colors()[1], "stable (S)"),
                       (p_unstable, Makie.wong_colors()[6], "unstable (U)"))
    Dv_loc = [D_showcase_reduced(1im * w, pp) for w in ω_dense]
    Dn = Dv_loc ./ (1 .+ abs.(Dv_loc))
    lines!(axa, real.(Dn), imag.(Dn); color = col, label = lab)
end
scatter!(axa, [0.0], [0.0]; color = :black, marker = :cross, markersize = 8)
axislegend(axa; position = :lt, labelsize = 7)

# (b) phase integrand with the adaptive solver steps
axb = MAxis(fig2[1, 2], xlabel = "ω", ylabel = "dθ/dω",
    title = "(b) integrand and adaptive steps")
integ_u = [phase_integrand(D_showcase_reduced, p_unstable, w) for w in ω_dense]
lines!(axb, collect(ω_dense), integ_u; color = Makie.wong_colors()[6])
ts = filter(t -> t <= ω_show, sol_u.t)
scatter!(axb, ts, [phase_integrand(D_showcase_reduced, p_unstable, t) for t in ts];
    color = :black, markersize = 4)

# (c) accumulated phase over the whole (huge) frequency range, log-x
axc = MAxis(fig2[1, 3], xlabel = "ω", ylabel = "θ(ω)", xscale = log10,
    title = "(c) accumulated phase up to ω_max = 1e4")
tpos_s = max.(sol_s.t, 1e-3)
tpos_u = max.(sol_u.t, 1e-3)
lines!(axc, tpos_s, [u[1] for u in sol_s.u]; color = Makie.wong_colors()[1])
scatter!(axc, tpos_s, [u[1] for u in sol_s.u]; color = Makie.wong_colors()[1], markersize = 4)
lines!(axc, tpos_u, [u[1] for u in sol_u.u]; color = Makie.wong_colors()[6])
scatter!(axc, tpos_u, [u[1] for u in sol_u.u]; color = Makie.wong_colors()[6], markersize = 4)
save_fig(fig2, "fig_method_walkthrough")

# ---------------------------------------------------------------------------
# 3. Summary CSV (quoted in the manuscript text)
# ---------------------------------------------------------------------------
write_csv("showcase_summary", ["key", "value"], [
    ("extraction_ratio_dev", ratio_dev),
    ("grid_nx", nP), ("grid_ny", nD),
    ("grid_time_s", grid.t),
    ("mdbm_time_s", bnd.t),
    ("robust_P", circ.x), ("robust_D", circ.y), ("robust_R_scaled", circ.R_scaled),
    ("p_stable_P", p_stable[1]), ("p_stable_D", p_stable[2]),
    ("p_unstable_P", p_unstable[1]), ("p_unstable_D", p_unstable[2]),
    ("Zraw_stable", Zraw_s), ("Zraw_unstable", Zraw_u),
    ("odesteps_stable", length(sol_s.t)), ("odesteps_unstable", length(sol_u.t)),
])
