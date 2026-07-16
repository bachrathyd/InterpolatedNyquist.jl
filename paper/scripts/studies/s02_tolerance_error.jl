# Study s02: solver tolerance vs. chart quality, CPU time and the integer
# residual field, on the showcase system.
# Produces:
#   figures/fig_tolerance_charts.pdf  (four charts, CPU time in the titles)
#   figures/fig_error_field.pdf       (log10 integer-residual fields + histograms)
#   tables/tab_tolerance.tex, data/tolerance_study.csv

include(joinpath(@__DIR__, "systems.jl"))

nP, nD = FAST[] ? (50, 40) : (100, 80)
Pv = LinRange(SHOWCASE_PRANGE..., nP)
Dv = LinRange(SHOWCASE_DRANGE..., nD)
TOLS = [1e-3, 1e-5, 1e-7, 1e-9]
# ω_max must be generous enough that the truncated tail does not become the
# error floor: the velocity-feedback ripple decays like 1/ω, so the tail
# contributes ~0.2/ω_max to Z̃.  ω_max = 1e4 would floor the study at ~2e-5.
const WMAX_TOL = 1e6

tol_grids = with_cache("s02_grids_v2_$(nP)x$(nD)") do
    map(TOLS) do tol
        g = sweep_grid(D_showcase, Pv, Dv; ω_max = WMAX_TOL, reltol = tol, abstol = tol)
        @info "tolerance grid done" tol t = g.t
        g
    end
end

errs = [abs.(g.Z_raw .- round.(g.Z_raw)) for g in tol_grids]

# Boundary (traced once, high accuracy) overlaid on every panel
bnd = with_cache("s02_mdbm_ref_v2") do
    mdbm_boundary(D_showcase, SHOWCASE_PRANGE, SHOWCASE_DRANGE;
        ngrid = 30, Niter = 4, ω_max = WMAX_TOL, reltol = 1e-8, abstol = 1e-8)
end

# ---------------------------------------------------------------------------
# Figure 1: the four charts with CPU times
# ---------------------------------------------------------------------------
crange = extrema(combined_metric(tol_grids[end].Z, tol_grids[end].sigma))
fig = Figure(size = (W_FULL, W_FULL * 0.26))
hm = nothing
for (i, tol) in enumerate(TOLS)
    g = tol_grids[i]
    ax = MAxis(fig[1, i], xlabel = "P", ylabel = i == 1 ? "D" : "",
        title = "tol = 1e$(round(Int, log10(tol)))   ($(tex_time_plain(g.t)))")
    global hm = stability_panel!(ax, Pv, Dv, combined_metric(g.Z, g.sigma);
        edges = bnd.edges, crange = crange)
    i > 1 && hideydecorations!(ax; ticks = false)
end
Colorbar(fig[1, length(TOLS) + 1], hm, label = "Z (unstable)  /  σ̂ (stable)")
save_fig(fig, "fig_tolerance_charts")

# ---------------------------------------------------------------------------
# Figure 2: integer-residual fields + histograms, with the 100*tol rule marked
# ---------------------------------------------------------------------------
fig2 = Figure(size = (W_FULL, W_FULL * 0.46))
hm2 = nothing
for (i, tol) in enumerate(TOLS)
    logerr = log10.(max.(errs[i], 1e-16))
    ax = MAxis(fig2[1, i], xlabel = "P", ylabel = i == 1 ? "D" : "",
        title = "tol = 1e$(round(Int, log10(tol)))")
    global hm2 = heatmap!(ax, Pv, Dv, logerr; colormap = :inferno,
        colorrange = (-14, 0), rasterize = 8)
    i > 1 && hideydecorations!(ax; ticks = false)
    axh = MAxis(fig2[2, i], xlabel = "log10 ε", ylabel = i == 1 ? "count" : "")
    hist!(axh, vec(logerr); bins = 40, color = Makie.wong_colors()[1])
    vlines!(axh, [log10(tol)]; color = :red, linestyle = :dash)
    vlines!(axh, [log10(100 * tol)]; color = :black, linestyle = :dot)
    xlims!(axh, -14, 0)
end
Colorbar(fig2[1, length(TOLS) + 1], hm2, label = "log10 ε")
save_fig(fig2, "fig_error_field")

# ---------------------------------------------------------------------------
# Table + CSV.  Tests the practical rule of thumb  ε ≈ 100 · tol.
# ---------------------------------------------------------------------------
rows_csv = Tuple[]
rows_tex = Vector{String}[]
for (i, tol) in enumerate(TOLS)
    g = tol_grids[i]
    e = errs[i]
    n_missed = count(>=(0.25), e)   # points where the rounded count would be uncertain
    push!(rows_csv, (tol, g.t, mean(e), median(e), maximum(e),
        mean(e) / tol, maximum(e) / tol, n_missed, nP * nD))
    push!(rows_tex, ["\$10^{$(round(Int, log10(tol)))}\$", tex_time(g.t),
        tex_sci(mean(e)), tex_sci(maximum(e)),
        @sprintf("%.0f", mean(e) / tol), @sprintf("%.0f", maximum(e) / tol),
        string(n_missed)])
end
write_csv("tolerance_study",
    ["tol", "grid_time_s", "mean_err", "median_err", "max_err",
     "mean_over_tol", "max_over_tol", "n_misclassified", "n_points"],
    rows_csv)
write_booktabs("tab_tolerance", "lcccccc",
    ["tolerance", "CPU time", "mean \$\\varepsilon\$", "max \$\\varepsilon\$",
     "mean/tol", "max/tol", "uncertain"],
    rows_tex)

@info "100*tol rule check" mean_ratios = [r[6] for r in rows_csv] max_ratios = [r[7] for r in rows_csv]
