# Study s09: sigma-contours (guaranteed decay-rate boundaries) of the showcase
# system: the shifted argument principle traces gamma-stability boundaries.
# Produces: figures/fig_sigma_contours.pdf, data/sigma_contours.csv

include(joinpath(@__DIR__, "systems.jl"))

nP, nD = FAST[] ? (45, 35) : (90, 70)
Pv = LinRange(SHOWCASE_PRANGE..., nP)
Dv = LinRange(SHOWCASE_DRANGE..., nD)

grid = with_cache("s01_domgrid_h_$(nP)x$(nD)") do   # reuse the s01 cache if present
    sweep_grid_dominant(D_showcase_reduced, Pv, Dv; ω_max = 1e4)
end

# Choose contour levels from the actual depth of the stable region
σ_stable = grid.sigma[grid.Z .== 0]
σ_deep = isempty(σ_stable) ? -0.5 : quantile(σ_stable, 0.02)
σ_levels = collect(LinRange(0.0, 0.9 * σ_deep, 5))
@info "sigma levels" σ_levels

contours = with_cache("s09_sigma_mdbm") do
    map(σ_levels) do σ
        bnd = mdbm_boundary(D_showcase_reduced, SHOWCASE_PRANGE, SHOWCASE_DRANGE;
            ngrid = 24, Niter = FAST[] ? 3 : 4, σ = σ, ω_max = 1e4)
        @info "sigma contour traced" σ t = bnd.t
        (σ = σ, edges = bnd.edges, t = bnd.t)
    end
end

# ---------------------------------------------------------------------------
# Figure: sigma_est field in the stable region + MDBM gamma-stability contours
# ---------------------------------------------------------------------------
# The unstable region keeps its root count (red -> black) rather than being
# blanked: the bilinear scale gives the stable half its own full blue -> green
# ramp anyway, so showing Z costs the decay-rate field nothing and the reader
# sees the whole chart in one picture.
Cb, σ_min, Z_max = bilinear_metric(grid.Z, grid.sigma)
fig = Figure(size = (W_ONEHALF, W_ONEHALF * 0.62))
ax = MAxis(fig[1, 1], xlabel = "proportional gain  P", ylabel = "derivative gain  D")
hm = heatmap!(ax, Pv, Dv, Cb; colormap = BILINEAR_CMAP, colorrange = (-1, 1),
    rasterize = 8)
tpos, tlab = bilinear_ticks(σ_min, Z_max)
Colorbar(fig[1, 2], hm, label = "σ̂ (stable)   |   Z (unstable)", ticks = (tpos, tlab))
# Contours are drawn in hues the map does not use (white -> magenta), so they
# stay legible over both the blue-green interior and the red-black exterior.
cmap = cgrad([:white, :magenta])
for (k, c) in enumerate(contours)
    c.edges === nothing && continue
    col = cmap[(k - 1) / max(1, length(contours) - 1)]
    lines!(ax, c.edges[1], c.edges[2]; color = col, linewidth = 1.4,
        label = "σ = $(round(c.σ; digits = 2))")
end
axislegend(ax; position = :rt, labelsize = 7, backgroundcolor = (:white, 0.8))
save_fig(fig, "fig_sigma_contours")

write_csv("sigma_contours", ["sigma", "mdbm_time_s"],
    [(c.σ, c.t) for c in contours])
