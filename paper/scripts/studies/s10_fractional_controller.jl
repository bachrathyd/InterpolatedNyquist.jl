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
for (ic, case) in enumerate(CASES_G)
    D = make_D_gao(case.lam)
    nx, ny = FAST[] ? (40, 30) : (80, 60)
    kpv = LinRange(case.kpr..., nx)
    kiv = LinRange(case.kir..., ny)
    grid = with_cache("s10_gao_lam$(case.lam)_$(nx)x$(ny)") do
        sweep_grid(D, kpv, kiv; ω_max = 1e4)
    end
    ax = MAxis(fig[1, ic], xlabel = "k_p", ylabel = ic == 1 ? "k_i" : "",
        title = "λ = $(case.lam)")
    stability_panel!(ax, kpv, kiv, combined_metric(grid.Z, grid.sigma))
    cmap = cgrad([:black, :red])
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
