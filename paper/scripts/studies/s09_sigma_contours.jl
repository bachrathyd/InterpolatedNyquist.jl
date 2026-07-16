# Study s09: sigma-contours (guaranteed decay-rate boundaries) of the showcase
# system: the shifted argument principle traces gamma-stability boundaries.
# Produces: figures/fig_sigma_contours.pdf, data/sigma_contours.csv

include(joinpath(@__DIR__, "systems.jl"))

nP, nD = FAST[] ? (45, 35) : (90, 70)
Pv = LinRange(SHOWCASE_PRANGE..., nP)
Dv = LinRange(SHOWCASE_DRANGE..., nD)

grid = with_cache("s01_domgrid_$(nP)x$(nD)") do   # reuse the s01 cache if present
    sweep_grid_dominant(D_showcase, Pv, Dv; ω_max = 1e4)
end

# Choose contour levels from the actual depth of the stable region
σ_stable = grid.sigma[grid.Z .== 0]
σ_deep = isempty(σ_stable) ? -0.5 : quantile(σ_stable, 0.02)
σ_levels = collect(LinRange(0.0, 0.9 * σ_deep, 5))
@info "sigma levels" σ_levels

contours = with_cache("s09_sigma_mdbm") do
    map(σ_levels) do σ
        bnd = mdbm_boundary(D_showcase, SHOWCASE_PRANGE, SHOWCASE_DRANGE;
            ngrid = 24, Niter = FAST[] ? 3 : 4, σ = σ, ω_max = 1e4)
        @info "sigma contour traced" σ t = bnd.t
        (σ = σ, edges = bnd.edges, t = bnd.t)
    end
end

# ---------------------------------------------------------------------------
# Figure: sigma_est field in the stable region + MDBM gamma-stability contours
# ---------------------------------------------------------------------------
S = map((z, s) -> z == 0 ? s : NaN, grid.Z, grid.sigma)
fig = Figure(size = (W_ONEHALF, W_ONEHALF * 0.62))
ax = MAxis(fig[1, 1], xlabel = "proportional gain  P", ylabel = "derivative gain  D")
hm = heatmap!(ax, Pv, Dv, S; colormap = :viridis, rasterize = 8)
Colorbar(fig[1, 2], hm, label = "estimated decay rate σ̂ (stable region)")
cmap = cgrad([:red, :blue])
for (k, c) in enumerate(contours)
    c.edges === nothing && continue
    col = cmap[(k - 1) / max(1, length(contours) - 1)]
    lines!(ax, c.edges[1], c.edges[2]; color = col, linewidth = 1.2,
        label = "σ = $(round(c.σ; digits = 2))")
end
axislegend(ax; position = :rt, labelsize = 7)
save_fig(fig, "fig_sigma_contours")

write_csv("sigma_contours", ["sigma", "mdbm_time_s"],
    [(c.σ, c.t) for c in contours])
