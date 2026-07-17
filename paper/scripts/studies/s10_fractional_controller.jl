# Study s10: reproduction of a fractional-order delayed-controller stability
# chart from the literature: Gao, Zhai & Liu (2017), Example 1.
#   plant      G(s) = 5 e^{-0.4 s} / (10 s^{0.5} + 1)
#   controller C(s) = kp + ki / s^lam
#   D(s) = s^lam (10 s^{0.5} + 1) + 5 e^{-0.4 s} (kp s^lam + ki)
# Their charts show stabilizing regions in the (kp, ki) plane.
#
# ONLY the sigma = 0 boundary is computed here, and that restriction is the
# point of this study rather than a shortcut. The SHIFTED line does NOT carry
# over to this class: s^mu has a branch cut along the negative real axis, so a
# contour enclosing Re(lam) > -sigma_deg encloses the segment [-sigma_deg, 0]
# of that cut, D is not analytic inside it, and the argument principle does not
# apply. The consequence is visible, not subtle -- earlier versions of this
# figure drew sigma_deg = 0.5 contours that CROSSED the sigma_deg = 0 boundary,
# which is impossible for genuine gamma-stability regions (moving the line left
# can only ever add roots). The block below measures both facts and writes them
# out, so the appendix can state the limitation with evidence.
#
# sigma = 0 itself is fine: the cut lies on the negative reals, outside the
# closed right half-plane; only the branch POINT at the origin touches the
# contour, and D(0) = ki there is finite and non-zero, so the indentation
# around it contributes nothing in the limit.
# Produces: figures/fig_fractional_controller.pdf, data/fractional_controller.csv,
#           data/fractional_branchcut.csv

include(joinpath(@__DIR__, "systems.jl"))

function make_D_gao(lam)
    function D(s::T, p) where T
        kp, ki = p
        return s^lam * (10 * s^T(0.5) + one(T)) + 5 * exp(-T(0.4) * s) * (kp * s^lam + ki)
    end
    return D
end

CASES_G = [
    (lam = 0.4, kpr = (-1.0, 6.0), kir = (-0.5, 8.0), sig_degs = [0.0],
     test_points = [(2.0, 5.0), (2.0, 3.0), (2.0, 2.0)]),
    (lam = 1.5, kpr = (-1.0, 6.0), kir = (-1.0, 25.0), sig_degs = [0.0],
     test_points = [(4.0, 19.0), (4.0, 17.0), (4.0, 8.0)]),
]

# ---------------------------------------------------------------------------
# EVIDENCE that the shifted line is unusable for this class (see the header).
#   (a) D jumps across the negative real axis -> the cut is real, not folklore.
#   (b) Z(sigma) loses monotonicity: Z(-0.5) < Z(0) happens, although moving
#       the integration line LEFT can only ever add roots to its right.
# ---------------------------------------------------------------------------
branch_rows = Tuple[]
let D = make_D_gao(0.4), p = (2.0, 3.0)
    # (a) The cut IS there, on the negative reals.
    for ε in (1e-6, 1e-9, 1e-12)
        a = D(-0.25 + ε * im, p); b = D(-0.25 - ε * im, p)
        push!(branch_rows, ("jump_at_-0.25", ε, abs(a - b), NaN, NaN))
        @info "branch cut: D across the negative real axis" ε jump = abs(a - b)
    end
    # (b) ... and it is NOT on the imaginary axis, which is why sigma = 0 works.
    for ω in (1e-3, 1.0, 10.0)
        a = D(1e-30 + ω * im, p); b = D(-1e-30 + ω * im, p)
        push!(branch_rows, ("jump_across_imaginary_axis", ω, abs(a - b), NaN, NaN))
        @info "no cut on the imaginary axis" ω jump = abs(a - b)
    end
    # (c) The branch POINT at s = 0 does lie ON the contour: the integrand
    #     diverges like omega^(mu-1). That is integrable for mu > 0, and the
    #     indentation around it contributes nothing PROVIDED D(0) != 0 --
    #     D(0) = ki here, so the ki = 0 line of this very chart is degenerate.
    for p2 in ((2.0, 3.0), (2.0, 0.0))
        push!(branch_rows, ("D_at_origin", p2[1], p2[2], abs(D(1e-300 + 0im, p2)), NaN))
        @info "D at the branch point" kp = p2[1] ki = p2[2] absD0 = abs(D(1e-300 + 0im, p2))
    end
end
# (wrapped in a function: a bare top-level loop puts the counters in soft scope)
function count_violations!(rows)
    n_viol = 0; n_tot = 0
    for case in CASES_G
        D = make_D_gao(case.lam)
        for kp in LinRange(case.kpr..., 15), ki in LinRange(case.kir..., 15)
            z0, = calculate_unstable_roots_direct(D, (kp, ki), 0.0;
                ω_max = 1e4, n_roots_to_track = 0)
            z5, = calculate_unstable_roots_direct(D, (kp, ki), -0.5;
                ω_max = 1e4, n_roots_to_track = 0)
            n_tot += 1
            if z5 < z0
                n_viol += 1
                push!(rows, ("monotonicity_violation", case.lam, kp, ki, Float64(z0 - z5)))
            end
        end
    end
    return n_viol, n_tot
end
n_viol, n_tot = count_violations!(branch_rows)
@warn "gamma-stability monotonicity Z(-0.5) >= Z(0) violated (branch cut)" n_viol n_tot
write_csv("fractional_branchcut", ["kind", "a", "b", "c", "d"], branch_rows)
# Does the sigma = 0 count actually land on integers, despite the divergent
# integrand at the origin? That is the empirical test of admissibility.
resid_stats = map(CASES_G) do case
    D = make_D_gao(case.lam)
    params = vec([(kp, ki) for kp in LinRange(case.kpr..., 30),
                            ki in LinRange(max(case.kir[1], 0.5), case.kir[2], 30)])
    _, Zraw = calculate_unstable_roots_p_vec(D, params; ω_max = 1e4, n_roots_to_track = 0)
    r = abs.(Zraw .- round.(Zraw))
    @info "sigma=0 residual on the fractional chart" case.lam median_r = median(r) max_r = maximum(r)
    (lam = case.lam, med = median(r), mx = maximum(r), n_unc = count(>(0.25), r), n = length(r))
end
for s in resid_stats
    push!(branch_rows, ("sigma0_residual", s.lam, s.med, s.mx, Float64(s.n_unc)))
end
write_macros("fractional_numbers", [
    "FracJump"     => @sprintf("%.1f", branch_rows[1][3]),
    "FracViol"     => string(n_viol),
    "FracViolTot"  => string(n_tot),
    "FracAxisJump" => @sprintf("%.0e", maximum(r[3] for r in branch_rows
                                               if r[1] == "jump_across_imaginary_axis")),
    "FracResidMed" => tex_sci_bare(maximum(s.med for s in resid_stats)),
    "FracResidMax" => @sprintf("%.3f", maximum(s.mx for s in resid_stats)),
    "FracResidN"   => string(sum(s.n for s in resid_stats)),
])

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
    # Legends at the top, and on the side each panel's stable region leaves
    # free: the mu = 0.4 island hugs the lower left, the mu = 1.5 one the lower
    # right, so the legend goes right on the first panel and left on the second.
    axislegend(ax; position = ic == 1 ? :rt : :lt, labelsize = 7,
        backgroundcolor = (:white, 0.85))
end
save_fig(fig, "fig_fractional_controller")

write_csv("fractional_controller",
    ["lambda", "sigma_deg", "kp", "ki", "Z", "Z_raw"], rows_csv)
