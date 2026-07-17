# Study s10: reproduction of a fractional-order delayed-controller stability
# chart from the literature: Gao, Zhai & Liu (2017), Example 1.
#   plant      G(s) = 5 e^{-0.4 s} / (10 s^{0.5} + 1)
#   controller C(s) = kp + ki / s^lam
#   D(s) = s^lam (10 s^{0.5} + 1) + 5 e^{-0.4 s} (kp s^lam + ki)
# Their charts show stabilizing regions in the (kp, ki) plane including
# regions with guaranteed stability DEGREE (all roots left of Re = -sigma_deg),
# which maps directly onto our shifted-line argument principle.
# Produces: figures/fig_fractional_controller.pdf, data/fractional_controller.csv

include(joinpath(@__DIR__, "systems.jl"))

function make_D_gao(lam)
    function D(s::T, p) where T
        kp, ki = p
        return s^lam * (10 * s^T(0.5) + one(T)) + 5 * exp(-T(0.4) * s) * (kp * s^lam + ki)
    end
    return D
end

CASES_G = [
    (lam = 0.4, kpr = (-1.0, 6.0), kir = (-0.5, 8.0), sig_degs = [0.0, 0.5, 1.0],
     test_points = [(2.0, 5.0), (2.0, 3.0), (2.0, 2.0)]),
    (lam = 1.5, kpr = (-1.0, 6.0), kir = (-1.0, 25.0), sig_degs = [0.0, 0.15, 0.3],
     test_points = [(4.0, 19.0), (4.0, 17.0), (4.0, 8.0)]),
]

fig = Figure(size = (W_FULL, W_FULL * 0.42))
rows_csv = Tuple[]

# Each case gets its OWN colour scale and its OWN colour bar. The two systems
# differ by an order of magnitude in spectral gap (mu = 0.4 reaches sigma ~ -1.7,
# mu = 1.5 only ~ -0.3), so a shared scale would leave the second panel
# uniformly blue -- while a shared BAR over per-panel scales would mislabel one
# of them. Two bars is the only arrangement that is both readable and honest.
NXY = FAST[] ? (40, 30) : (80, 60)
for (ic, case) in enumerate(CASES_G)
    D = make_D_gao(case.lam)
    kpv = LinRange(case.kpr..., NXY[1]); kiv = LinRange(case.kir..., NXY[2])
    grid = with_cache("s10_gao_lam$(case.lam)_$(NXY[1])x$(NXY[2])") do
        sweep_grid_dominant(D, kpv, kiv; nroots = 5, ω_max = 1e4)
    end
    col = 2ic - 1
    ax = MAxis(fig[1, col], xlabel = "k_p", ylabel = ic == 1 ? "k_i" : "",
        title = "μ = $(case.lam)")
    Cb, σ_min, Z_max = bilinear_metric(grid.Z, grid.sigma)
    hm = heatmap!(ax, kpv, kiv, Cb; colormap = BILINEAR_CMAP, colorrange = (-1, 1),
        rasterize = 8)
    tpos, tlab = bilinear_ticks(σ_min, Z_max; n = 2)
    Colorbar(fig[1, col + 1], hm, ticks = (tpos, tlab),
        label = ic == length(CASES_G) ? "σ̂ (stable)  |  Z (unstable)" : "")
    # sigma-degree contours in hues the map does not use
    cmap = cgrad([:white, :magenta])
    for (k, sd) in enumerate(case.sig_degs)
        bnd = with_cache("s10_gao_lam$(case.lam)_mdbm_sd$(sd)") do
            mdbm_boundary(D, case.kpr, case.kir; ngrid = 25,
                Niter = FAST[] ? 3 : 4, σ = -sd, ω_max = 1e4)
        end
        if bnd.edges !== nothing
            col = cmap[(k - 1) / max(1, length(case.sig_degs) - 1)]
            lines!(ax, bnd.edges[1], bnd.edges[2]; color = col, linewidth = 1.2,
                label = "σ_deg = $sd")
        end
        # verify the published test points against every sigma level
        for tp in case.test_points
            Z, Z_raw = calculate_unstable_roots_direct(D, tp, -sd;
                n_roots_to_track = 0, ω_max = 1e4)
            push!(rows_csv, (case.lam, sd, tp[1], tp[2], Z, Z_raw))
        end
    end
    scatter!(ax, first.(case.test_points), last.(case.test_points);
        color = :white, strokecolor = :black, strokewidth = 0.8, markersize = 7)
    axislegend(ax; position = :rb, labelsize = 7)
end
save_fig(fig, "fig_fractional_controller")

write_csv("fractional_controller",
    ["lambda", "sigma_deg", "kp", "ki", "Z", "Z_raw"], rows_csv)
