# Study s04: accuracy and cost of the root refinement
# (Linear seed vs Taylor Poly-2/Poly-3 vs Newton iterations).
# All timings are PER PARAMETER POINT and are reported RELATIVE to the cost of
# the phase integration that produced the seed, so the reader can see that
# refinement is essentially free.
# Produces: figures/fig_refinement.pdf, tables/tab_refinement.tex,
#           data/refinement.csv

include(joinpath(@__DIR__, "studies_common.jl"))
include(joinpath(@__DIR__, "systems.jl"))

nP, nD = FAST[] ? (30, 20) : (60, 40)
Pv = LinRange(SHOWCASE_PRANGE..., nP)
Dv = LinRange(SHOWCASE_DRANGE..., nD)
const WMAX_REF = 1e4

REFINER_NAMES = ["Linear", "Poly-2", "Poly-3", "Newton-4"]
METHOD_SPEC = [(:Linear, 0), (:Polynomial, 2), (:Polynomial, 3), (:Newton, 4)]

refine_at(p, λ, method, deg_or_steps) = method === :Linear ? λ :
    method === :Newton ? refine_roots(D_showcase_reduced, p, λ; method = :Newton, steps = deg_or_steps) :
    refine_roots(D_showcase_reduced, p, λ; method = :Polynomial, degree = deg_or_steps)

data = with_cache("s04_refine_v2_$(nP)x$(nD)") do
    # Raw (unrefined) seeds from the integrator
    seeds = Matrix{ComplexF64}(undef, nP, nD)
    for (j, d) in enumerate(Dv), (i, p) in enumerate(Pv)
        zi, zr, md, es, wc = calculate_unstable_roots_direct(D_showcase_reduced, (p, d);
            ω_max = WMAX_REF, refinement_method = :Linear)
        seeds[i, j] = es + 1im * wc
    end

    # Baseline: what one parameter point costs WITHOUT any refinement.
    t_integration = time_point(() -> calculate_unstable_roots_direct(D_showcase_reduced,
        (Pv[nP ÷ 2], Dv[nD ÷ 2]); ω_max = WMAX_REF, refinement_method = :Linear))

    errs = Dict{String, Matrix{Float64}}()
    t_point = Dict{String, Float64}()
    gt = [refine_at((p, d), seeds[i, j], :Newton, 15)
          for (i, p) in enumerate(Pv), (j, d) in enumerate(Dv)]
    for (name, (m, k)) in zip(REFINER_NAMES, METHOD_SPEC)
        ref = [refine_at((p, d), seeds[i, j], m, k)
               for (i, p) in enumerate(Pv), (j, d) in enumerate(Dv)]
        errs[name] = abs.(ref .- gt)
        # per-point cost of the FULL solve (integration + this refinement)
        t_point[name] = m === :Linear ? t_integration :
            time_point(() -> calculate_unstable_roots_direct(D_showcase_reduced,
                (Pv[nP ÷ 2], Dv[nD ÷ 2]); ω_max = WMAX_REF,
                refinement_method = m,
                refinement_steps = (m === :Newton ? k : 4),
                refinement_degree = (m === :Polynomial ? k : 3)))
        @info "refinement" name t_point[name] med = median(filter(isfinite, vec(errs[name])))
    end
    (seeds = seeds, errs = errs, t_point = t_point, t_integration = t_integration)
end

# ---------------------------------------------------------------------------
# Figure: (a) zoomed root-plane view + the four log10 error maps
# ---------------------------------------------------------------------------
p_zoom = (Pv[round(Int, 0.7nP)], Dv[round(Int, 0.5nD)])
zi, zr, md, es, wc = calculate_unstable_roots_direct(D_showcase_reduced, p_zoom;
    ω_max = WMAX_REF, n_roots_to_track = 6, refinement_method = :Linear)
seeds_zoom = [e + 1im * w for (e, w) in zip(es, wc) if isfinite(e)]
ref_zoom = [refine_at(p_zoom, s, :Newton, 4) for s in seeds_zoom]
truth_zoom = [refine_at(p_zoom, s, :Newton, 15) for s in seeds_zoom]

# zoom tightly around the dominant root so seed and refined point separate
k_dom = argmax(real.(truth_zoom))
λ0 = truth_zoom[k_dom]
pad = max(3 * abs(seeds_zoom[k_dom] - λ0), 2e-3)
σs = range(real(λ0) - pad, real(λ0) + pad; length = 220)
ωs = range(imag(λ0) - pad, imag(λ0) + pad; length = 220)
ReD = [real(D_showcase_reduced(s + 1im * w, p_zoom)) for s in σs, w in ωs]
ImD = [imag(D_showcase_reduced(s + 1im * w, p_zoom)) for s in σs, w in ωs]

fig = Figure(size = (W_FULL, W_FULL * 0.55))
axr = MAxis(fig[1:2, 1], xlabel = "Re λ", ylabel = "Im λ",
    title = "(a) dominant root, zoom ±$(round(pad; sigdigits=2))")
contour!(axr, σs, ωs, ReD; levels = [0.0], color = Makie.wong_colors()[1])
contour!(axr, σs, ωs, ImD; levels = [0.0], color = Makie.wong_colors()[2])
scatter!(axr, [real(seeds_zoom[k_dom])], [imag(seeds_zoom[k_dom])];
    color = :black, marker = :cross, markersize = 13, label = "integrator seed")
scatter!(axr, [real(ref_zoom[k_dom])], [imag(ref_zoom[k_dom])];
    color = :red, marker = :circle, markersize = 7, label = "Newton-4 refined")
scatter!(axr, [real(λ0)], [imag(λ0)]; color = :white, strokecolor = :black,
    strokewidth = 0.7, marker = :diamond, markersize = 6, label = "reference")
axislegend(axr; position = :rb, labelsize = 6)

panel_pos = [(1, 2), (1, 3), (2, 2), (2, 3)]
hm = nothing
for (k, name) in enumerate(REFINER_NAMES)
    r, c = panel_pos[k]
    over = 100 * (data.t_point[name] - data.t_integration) / data.t_integration
    ttl = name == "Linear" ? "Linear (seed, $(tex_time_plain(data.t_integration))/point)" :
        "$(name)  (+$(round(over; digits=1))% per point)"
    ax = MAxis(fig[r, c], xlabel = r == 2 ? "P" : "", ylabel = c == 2 ? "D" : "", title = ttl)
    global hm = heatmap!(ax, Pv, Dv, log10.(max.(data.errs[name], 1e-16));
        colormap = :inferno, colorrange = (-14, 0), rasterize = 8)
end
Colorbar(fig[1:2, 4], hm, label = "log10 |λ - λ_ref|")
save_fig(fig, "fig_refinement")

# ---------------------------------------------------------------------------
# Table + CSV
# ---------------------------------------------------------------------------
rows_csv = Tuple[]
rows_tex = Vector{String}[]
for name in REFINER_NAMES
    e = filter(isfinite, vec(data.errs[name]))
    over = 100 * (data.t_point[name] - data.t_integration) / data.t_integration
    push!(rows_csv, (name, median(e), mean(e), maximum(e), data.t_point[name], over))
    push!(rows_tex, [name, tex_sci(median(e)), tex_sci(maximum(e)),
        tex_time(data.t_point[name]),
        name == "Linear" ? "--" : @sprintf("+%.1f\\%%", over)])
end
write_csv("refinement",
    ["method", "median_err", "mean_err", "max_err", "time_per_point_s", "overhead_pct"],
    rows_csv)
write_booktabs("tab_refinement", "lcccc",
    ["refinement", "median error", "max error", "time / point", "overhead"],
    rows_tex)
