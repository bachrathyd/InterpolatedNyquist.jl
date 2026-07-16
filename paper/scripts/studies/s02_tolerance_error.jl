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

# This study isolates the INTEGRATOR, so two deliberate choices:
#
# 1. ω_max = 1e6, not the 1e4 that the chart itself needs.  The velocity
#    feedback of this system makes the phase ripple decay only like 1/ω, so the
#    truncated tail contributes ~0.2/ω_max to Z̃; at ω_max = 1e4 that floors
#    the residual at ~2e-5 and the two tightest tolerances would be
#    indistinguishable.  (§6.3 states this trade-off explicitly.)
# 2. The reduced (hand-derived) form of the SAME characteristic function.  It
#    is identical to the automatically extracted one to machine precision
#    (§6.1) and gives the identical chart, but costs ~15x less per evaluation
#    because it skips the 7x7 dual-number determinant.  Since ω_max = 1e6
#    already multiplies the work by ~100, the extracted form would make this
#    study take hours to say exactly the same thing about the integrator.
const WMAX_TOL = 1e6
const D_TOL = D_showcase_reduced

# cache v3: v2 predates the D_TOL switch (its cached timings measured the
# 15x-costlier extracted form and are not comparable)
tol_grids = with_cache("s02_grids_v3_$(nP)x$(nD)") do
    map(TOLS) do tol
        g = sweep_grid(D_TOL, Pv, Dv; ω_max = WMAX_TOL, reltol = tol, abstol = tol)
        @info "tolerance grid done" tol t = g.t
        g
    end
end

errs = [abs.(g.Z_raw .- round.(g.Z_raw)) for g in tol_grids]

# The tightest-tolerance grid (1e-9) serves as the count reference: the
# "wrong Z" column then reports ACTUAL misclassifications of each looser
# chart, not merely points whose rounding is uncertain.
Z_ref_grid = tol_grids[end].Z
n_wrong = [count(g.Z .!= Z_ref_grid) for g in tol_grids]

# Boundary (traced once, high accuracy) overlaid on every panel
bnd = with_cache("s02_mdbm_ref_v3") do
    mdbm_boundary(D_TOL, SHOWCASE_PRANGE, SHOWCASE_DRANGE;
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
    n_uncertain = count(>=(0.25), e)   # points where the rounded count would be uncertain
    push!(rows_csv, (tol, g.t, mean(e), median(e), maximum(e),
        mean(e) / tol, maximum(e) / tol, n_uncertain, n_wrong[i], nP * nD))
    push!(rows_tex, ["\$10^{$(round(Int, log10(tol)))}\$", tex_time(g.t),
        tex_sci(mean(e)), tex_sci(maximum(e)),
        @sprintf("%.0f", mean(e) / tol), @sprintf("%.0f", maximum(e) / tol),
        string(n_uncertain), string(n_wrong[i])])
end
write_csv("tolerance_study",
    ["tol", "grid_time_s", "mean_err", "median_err", "max_err",
     "mean_over_tol", "max_over_tol", "n_uncertain", "n_wrong_vs_ref", "n_points"],
    rows_csv)
write_booktabs("tab_tolerance", "lccccccc",
    ["tolerance", "CPU time", "mean \$\\varepsilon\$", "max \$\\varepsilon\$",
     "mean/tol", "max/tol", "uncertain", "wrong \$Z\$"],
    rows_tex)

@info "100*tol rule check" mean_ratios = [r[6] for r in rows_csv] max_ratios = [r[7] for r in rows_csv]
