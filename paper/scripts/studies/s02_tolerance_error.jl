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
# The ladder deliberately starts at 1e-2, where eps ~ 100*tol predicts
# residuals of order ONE, i.e. genuine miscounts: the first panel shows what
# breaking the rule of thumb looks like, the rest how quickly certainty is
# bought back. (Note: the COUNT is never refined -- the Newton polish of
# Section 3.3 only touches the root estimate sigma_est, so the residual and
# wrong-Z columns measure the integrator alone.)
TOLS = [1e-2, 1e-4, 1e-6, 1e-8]

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

# cache v4: v3 used the 1e-3..1e-9 ladder
tol_grids = with_cache("s02_grids_v4_$(nP)x$(nD)") do
    map(TOLS) do tol
        g = sweep_grid(D_TOL, Pv, Dv; ω_max = WMAX_TOL, reltol = tol, abstol = tol)
        @info "tolerance grid done" tol t = g.t
        g
    end
end

errs = [abs.(g.Z_raw .- round.(g.Z_raw)) for g in tol_grids]

# The tightest-tolerance grid (1e-8) serves as the count reference: the
# "wrong Z" column then reports ACTUAL misclassifications of each looser
# chart, not merely points whose rounding is uncertain.
Z_ref_grid = tol_grids[end].Z
n_wrong = [count(g.Z .!= Z_ref_grid) for g in tol_grids]
# ... and of those, the ones that flip the stable/unstable map (the errors a
# stability chart actually shows):
n_flip = [count((g.Z .== 0) .!= (Z_ref_grid .== 0)) for g in tol_grids]

# ---------------------------------------------------------------------------
# The sigma-vs-Z cross-check (Section 3.4), MEASURED rather than asserted.
#
# The naive test sign(sigma_est) != (Z == 0) is useless on its own: far from
# the boundary the tracked |D| minimum need not belong to the DOMINANT root
# (the caveat of Sec. 3.2), so a disagreement there is legitimate and the
# test fires on ~30% of a fully converged chart. A peak can only be skipped
# where a root lies CLOSE to the line, so the test is restricted to a
# neighbourhood |sigma_est| < CHECK_H -- orders of magnitude below the
# typical |sigma_est| of the deep domain, quantified by ref_med below.
const CHECK_H = 1e-3
naive_flag(g) = [(g.sigma[i] < 0) != (g.Z[i] == 0) for i in eachindex(g.Z)]
near_flag(g) = [((g.sigma[i] < 0) != (g.Z[i] == 0)) && abs(g.sigma[i]) < CHECK_H
                for i in eachindex(g.Z)]
ref_g = tol_grids[end]
ref_med = median(abs.(ref_g.sigma[naive_flag(ref_g)]))
check_rows = Tuple[]
for (i, tol) in enumerate(TOLS)
    g = tol_grids[i]
    flips = findall((g.Z .== 0) .!= (Z_ref_grid .== 0))
    nf = near_flag(g)
    push!(check_rows, (tol, count(naive_flag(g)), count(nf), length(flips),
        count(nf[flips]), nP * nD))
end
write_csv("crosscheck",
    ["tol", "naive_flags", "near_flags", "n_flips", "flips_caught", "n_points"],
    check_rows)
@info "cross-check" naive_on_reference = count(naive_flag(ref_g)) near_on_reference =
    count(near_flag(ref_g)) median_abs_sigma_of_naive_flags = ref_med

# Numbers quoted in Sections 3.4 and 6.3 -> TeX macros, so the prose can never
# drift from the measurement.
loose_i = 1                       # the loosest tolerance of the ladder
loose_flips = check_rows[loose_i][4]
loose_caught = check_rows[loose_i][5]
write_macros("crosscheck_numbers", [
    "CheckNpoints"     => string(nP * nD),
    "CheckNaiveRef"    => string(count(naive_flag(ref_g))),
    "CheckNearRef"     => string(count(near_flag(ref_g))),
    "CheckMedianSigma" => @sprintf("%.2g", ref_med),
    "CheckH"           => @sprintf("%g", CHECK_H),
    "CheckLooseTol"    => @sprintf("10^{%d}", round(Int, log10(TOLS[loose_i]))),
    "CheckLooseFlips"  => string(loose_flips),
    "CheckLooseCaught" => string(loose_caught),
    "CheckLooseWrong"  => string(n_wrong[loose_i]),
    "CheckRefTol"      => @sprintf("10^{%d}", round(Int, log10(TOLS[end]))),
])

# Boundary (traced once, high accuracy) overlaid on every panel
bnd = with_cache("s02_mdbm_ref_v3") do
    mdbm_boundary(D_TOL, SHOWCASE_PRANGE, SHOWCASE_DRANGE;
        ngrid = 30, Niter = 4, ω_max = WMAX_TOL, reltol = 1e-8, abstol = 1e-8)
end

# ---------------------------------------------------------------------------
# Does a LOOSE tolerance move the traced BOUNDARY, or only the counts?
# The two outputs fail differently: Z is rounded, so it flips discontinuously
# once the phase error reaches ±π, whereas sigma_est is a local quantity that
# inherits the tolerance smoothly -- and the boundary is built from sigma_est
# alone. Measure the displacement (one-sided Hausdorff, in parameter units)
# of loose traces against the reference trace above.
# ---------------------------------------------------------------------------
bnd_loose = [with_cache("s02_mdbm_tol$(tol)_v1") do
        mdbm_boundary(D_TOL, SHOWCASE_PRANGE, SHOWCASE_DRANGE;
            ngrid = 30, Niter = 4, ω_max = WMAX_TOL, reltol = tol, abstol = tol)
    end for tol in TOLS]
ref_pts = let s = getinterpolatedsolution(bnd.prob)
    [(s[1][i], s[2][i]) for i in eachindex(s[1])]
end
function boundary_shift(prob)
    s = getinterpolatedsolution(prob)
    pts = [(s[1][i], s[2][i]) for i in eachindex(s[1])]
    isempty(pts) && return (NaN, NaN)
    d = [minimum(hypot(p[1] - q[1], p[2] - q[2]) for q in ref_pts) for p in pts]
    return (maximum(d), mean(d))
end
pix = max(step(Pv), step(Dv))     # one chart pixel, for scale
shift_rows = Tuple[]
for (i, tol) in enumerate(TOLS)
    mx, mn = boundary_shift(bnd_loose[i].prob)
    push!(shift_rows, (tol, mx, mn, mx / pix, pix))
    @info "boundary shift vs 1e-8 trace" tol max_shift = mx max_in_pixels = mx / pix
end
write_csv("boundary_shift",
    ["tol", "max_shift", "mean_shift", "max_shift_in_pixels", "pixel_size"], shift_rows)

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
        mean(e) / tol, maximum(e) / tol, n_uncertain, n_wrong[i], n_flip[i], nP * nD))
    push!(rows_tex, ["\$10^{$(round(Int, log10(tol)))}\$", tex_time(g.t),
        tex_sci(mean(e)), tex_sci(maximum(e)),
        @sprintf("%.0f", mean(e) / tol), @sprintf("%.0f", maximum(e) / tol),
        string(n_uncertain), string(n_wrong[i]), string(n_flip[i])])
end
write_csv("tolerance_study",
    ["tol", "grid_time_s", "mean_err", "median_err", "max_err",
     "mean_over_tol", "max_over_tol", "n_uncertain", "n_wrong_vs_ref",
     "n_flip_vs_ref", "n_points"],
    rows_csv)
write_booktabs("tab_tolerance", "lcccccccc",
    ["tolerance", "CPU time", "mean \$\\varepsilon\$", "max \$\\varepsilon\$",
     "mean/tol", "max/tol", "uncertain", "wrong \$\\Zint\$", "flips"],
    rows_tex)

@info "100*tol rule check" mean_ratios = [r[6] for r in rows_csv] max_ratios = [r[7] for r in rows_csv]
