# Study s04: accuracy and cost of the optional root refinement.
#
# The chart-level default is now :Linear (the raw tracked estimate, no polish);
# refinement is an opt-in for an accurate rightmost root. This study judges the
# five choices -- raw, Poly-2, Poly-3, Newton-4, Combined -- by TWO honest,
# method-INDEPENDENT metrics (the previous version measured error against
# Newton-15 itself, which trivially favoured Newton):
#   (1) residual |D(lambda_dom)|          -- distance to an actual root of D
#   (2) |Re lambda_dom - sigma_SD|         -- error vs a semi-discretization ref
# on TWO systems: a SMOOTH quasi-polynomial (the showcase) and a NON-smooth
# rational FRF with poles (turning lobes), so the "no single polish wins
# everywhere" point is visible. Both an error FIELD (per-pixel maps) and error
# BARS (distributions) are shown, as the two complementary views.
# Produces: figures/fig_refinement.pdf, tables/tab_refinement.tex,
#           data/refinement.csv, generated/refinement_numbers.tex

include(joinpath(@__DIR__, "studies_common.jl"))
include(joinpath(@__DIR__, "systems.jl"))
using SemiDiscretizationMethod

# ---------------------------------------------------------------------------
# The two systems and their first-order (A, B) forms for the SD reference
# ---------------------------------------------------------------------------
# showcase: reduced quasi-polynomial (smooth); A,B from systems.jl
showcase_ABt(p) = (showcase_AB(p)..., SM.tau)

# turning: two modal oscillators + regenerative delayed force -> rational FRF
const TW1=1.0; const TZ1=0.02; const TW2=2.4; const TZ2=0.03; const TKAP=0.45
function D_turning(λ::T, p) where T
    Ω, w = p; τ = 2π/Ω
    G = T(1.0)/(λ^2+2*T(TZ1)*T(TW1)*λ+T(TW1)^2) + T(TKAP)/(λ^2+2*T(TZ2)*T(TW2)*λ+T(TW2)^2)
    return one(T) + w*(1-exp(-τ*λ))*G
end
function turning_ABt(p)
    Ω, w = p
    A = [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0;
        -(TW1^2+w) -w -2TZ1*TW1 0.0; -TKAP*w -(TW2^2+TKAP*w) 0.0 -2TZ2*TW2]
    B = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; w w 0.0 0.0; TKAP*w TKAP*w 0.0 0.0]
    return A, B, 2π/Ω
end

# SD dominant real part (multiplication-free LR mapping). Uses the DENSE
# generalized eigensolve (LAPACK, thread-safe) rather than the sparse ARPACK
# spectralRadiusOfMapping: this assess() runs the grid under Threads.@threads,
# and ARPACK holds Fortran global state -- concurrent calls segfault the
# process. The pencil is only (n+1)*dim (~164x164 here), so dense is cheap.
function sd_sigma(ABt, p; n=40, order=2)
    A, B, τ = ABt(p); Δt = τ/n
    ld = LDDEProblem(ProportionalMX(A), [DelayMX(t->τ, B)], Additive(zeros(size(A,1))))
    m = DiscreteMapping_LR(ld, SemiDiscretization(order, Δt), τ; n_steps=1, calculate_additive=false)
    ρ = maximum(abs, eigen(collect(m.RmappingMX), collect(m.LmappingMX)).values)
    return log(ρ)/Δt
end

# ---------------------------------------------------------------------------
# Methods and the dominant-root evaluation
# ---------------------------------------------------------------------------
const REFINERS = ["raw", "Poly-2", "Poly-3", "Newton-4", "Combined"]
function refine_one(D, p, seed, name)
    (name == "raw" || !isfinite(seed)) && return seed
    name == "Poly-2"   && return refine_roots(D, p, seed; method=:Polynomial, degree=2)
    name == "Poly-3"   && return refine_roots(D, p, seed; method=:Polynomial, degree=3)
    name == "Newton-4" && return refine_roots(D, p, seed; method=:Newton, steps=4)
    return refine_roots(D, p, seed; method=:Combined, steps=4)   # Combined
end
function dominant(D, p, seeds, name)
    best = NaN+NaN*im
    for s in seeds
        λ = refine_one(D, p, s, name)
        (isfinite(λ) && (!isfinite(best) || real(λ) > real(best))) && (best = λ)
    end
    return best
end

# per-system per-method error FIELDS (residual and sigma-vs-SD), over the WHOLE
# parameter plane -- stable AND unstable -- so the maps carry information
# everywhere (the dominant-root real part is compared to the SD reference
# regardless of its sign). Near the boundary, where a root sits close to the
# integration line, the estimate is sharp and the refinements separate; deep in
# either domain the tracked line-minimum is no longer the true dominant root and
# every method's error grows, which is exactly the accuracy-near-the-boundary
# behaviour Section 3.2 predicts.
function assess(D, ABt, xv, yv; ω, tol, npow=nothing, sdn=40)
    npkw = npow === nothing ? (;) : (; n_power_max=npow)
    nx, ny = length(xv), length(yv)
    resid = Dict(m => fill(NaN, nx, ny) for m in REFINERS)
    σerr  = Dict(m => fill(NaN, nx, ny) for m in REFINERS)
    stable = fill(false, nx, ny)
    Threads.@threads for j in 1:ny
        for i in 1:nx
            p = (xv[i], yv[j])
            zi, zr, mds, es, wc = calculate_unstable_roots_direct(D, p; n_roots_to_track=15,
                ω_max=ω, reltol=tol, abstol=tol, refinement_method=:Linear, npkw...)
            stable[i, j] = (zi == 0)
            seeds = [e+1im*w for (e,w) in zip(es,wc) if isfinite(e)]
            isempty(seeds) && continue
            σsd = sd_sigma(ABt, p; n=sdn)
            for m in REFINERS
                λ = dominant(D, p, seeds, m)
                isfinite(λ) || continue
                resid[m][i, j] = abs(D(λ, p))
                isfinite(σsd) && (σerr[m][i, j] = abs(real(λ) - σsd))
            end
        end
    end
    return resid, σerr, stable
end

# ---------------------------------------------------------------------------
# Compute (cached)
# ---------------------------------------------------------------------------
SHOW_x = LinRange(SHOWCASE_PRANGE..., FAST[] ? 30 : 55)
SHOW_y = LinRange(SHOWCASE_DRANGE..., FAST[] ? 20 : 38)
TURN_x = LinRange(0.3, 1.3, FAST[] ? 30 : 50)
TURN_y = LinRange(0.01, 1.2, FAST[] ? 30 : 50)

# v2: fields computed over the WHOLE plane (was stable-only), + stability mask
sfields = with_cache("s04_showcase_fields_v2_$(length(SHOW_x))x$(length(SHOW_y))") do
    r, s, st = assess(D_showcase_reduced, showcase_ABt, SHOW_x, SHOW_y; ω=1e4, tol=1e-4)
    (resid=r, σerr=s, stable=st)
end
tfields = with_cache("s04_turning_fields_v2_$(length(TURN_x))x$(length(TURN_y))") do
    r, s, st = assess(D_turning, turning_ABt, TURN_x, TURN_y; ω=1e4, tol=1e-4)
    (resid=r, σerr=s, stable=st)
end

# per-point cost of the full solve under each method (showcase interior point)
methsym = Dict("raw"=>:Linear, "Poly-2"=>:Polynomial, "Poly-3"=>:Polynomial,
    "Newton-4"=>:Newton, "Combined"=>:Combined)
methdeg = Dict("raw"=>3, "Poly-2"=>2, "Poly-3"=>3, "Newton-4"=>3, "Combined"=>3)
costs = with_cache("s04_costs_v1") do
    Dict(m => time_point(() -> calculate_unstable_roots_direct(D_showcase_reduced, (1.8,1.0);
        n_roots_to_track=15, ω_max=1e4, reltol=1e-4, abstol=1e-4,
        refinement_method=methsym[m], refinement_degree=methdeg[m], refinement_steps=4);
        samples=7) for m in REFINERS)
end
t_raw = costs["raw"]

# ---------------------------------------------------------------------------
# Figure: error FIELDS over the WHOLE plane (2 systems x 5 methods,
# log10 |Δσ vs SD|) + error BARS summarising the STABLE domain (where the
# interpolable colouring lives and the refinement quality is the clean story).
# ---------------------------------------------------------------------------
finite(v) = filter(isfinite, v)
med(v) = isempty(finite(v)) ? NaN : median(finite(v))
# stable-domain values of a per-pixel field
smask(F, dict, m) = [dict[m][idx] for idx in CartesianIndices(F.stable)
                     if F.stable[idx] && isfinite(dict[m][idx])]
CR = (-12.0, 0.0)                                  # log10 error colour range

fig = Figure(size = (W_FULL, W_FULL * 0.58))
hm = nothing
for (blk, (name, xv, yv, F)) in enumerate((
        ("showcase (smooth)", SHOW_x, SHOW_y, sfields),
        ("turning (rational, poles)", TURN_x, TURN_y, tfields)))
    for (k, m) in enumerate(REFINERS)
        ax = MAxis(fig[blk, k]; title = k == 1 ? "$name  —  $m" : m,
            titlesize = 7, xticksvisible=false, yticksvisible=false,
            xticklabelsvisible=false, yticklabelsvisible=false)
        global hm = heatmap!(ax, xv, yv, log10.(max.(F.σerr[m], 1e-13));
            colormap = :inferno, colorrange = CR, rasterize = 8)
    end
end
Colorbar(fig[1:2, 6], hm, label = "log10 |Re λ_dom − σ_SD| (whole plane)",
    labelsize = 7, ticklabelsize = 6)

# bars row: residual + sigma-error distributions over the STABLE domain
function boxes!(ax, F, dict)
    for (k, m) in enumerate(REFINERS)
        v = log10.(max.(smask(F, dict, m), 1e-16))
        isempty(v) && continue
        boxplot!(ax, fill(k, length(v)), v; width=0.6, color=(:steelblue,0.55),
            strokecolor=:black, strokewidth=0.4, markersize=2)
    end
end
bx = [("showcase |Δσ| (stable)", sfields, sfields.σerr),
      ("turning |Δσ| (stable)", tfields, tfields.σerr),
      ("showcase resid |D|", sfields, sfields.resid),
      ("turning resid |D|", tfields, tfields.resid)]
for (k, (ttl, F, dat)) in enumerate(bx)
    ax = MAxis(fig[3, k]; title=ttl, titlesize=7, ylabel = k==1 ? "log10" : "",
        ylabelsize=7, xticks=(1:5, REFINERS), xticklabelrotation=0.5, xticklabelsize=6,
        yticklabelsize=6)
    boxes!(ax, F, dat)
end
axc = MAxis(fig[3, 5]; title="cost / point", titlesize=7, ylabel="ms", ylabelsize=7,
    xticks=(1:5, REFINERS), xticklabelrotation=0.5, xticklabelsize=6, yticklabelsize=6)
barplot!(axc, 1:5, [1e3*costs[m] for m in REFINERS]; color=(:darkorange,0.75))
rowsize!(fig.layout, 3, Relative(0.24))            # bars row kept compact
save_fig(fig, "fig_refinement")

# ---------------------------------------------------------------------------
# Table + CSV + macros  (statistics over the STABLE domain)
# ---------------------------------------------------------------------------
rows_csv = Tuple[]; rows_tex = Vector{String}[]
for m in REFINERS
    over = 100*(costs[m]-t_raw)/t_raw
    push!(rows_csv, (m, med(smask(sfields, sfields.resid, m)), med(smask(sfields, sfields.σerr, m)),
        med(smask(tfields, tfields.resid, m)), med(smask(tfields, tfields.σerr, m)), costs[m], over))
    push!(rows_tex, [m, tex_sci(med(smask(sfields, sfields.σerr, m))),
        tex_sci(med(smask(tfields, tfields.σerr, m))),
        tex_time(costs[m]), m == "raw" ? "--" : @sprintf("%+.0f\\%%", over)])
end
write_csv("refinement",
    ["method", "show_resid_med", "show_dsig_med", "turn_resid_med", "turn_dsig_med",
     "time_per_point_s", "overhead_pct"], rows_csv)
write_booktabs("tab_refinement", "lcccc",
    ["refinement", "\$|\\Delta\\sigma|\$ showcase", "\$|\\Delta\\sigma|\$ turning",
     "time / point", "overhead"], rows_tex)

# numbers quoted in the text -> macros (stable-domain medians)
ci(m, sys) = sys == :show ? med(smask(sfields, sfields.σerr, m)) :
                            med(smask(tfields, tfields.σerr, m))
# bare values (no $...$): the prose supplies the surrounding math mode, so
# these must NOT be double-wrapped -- hence tex_sci_bare, not tex_sci
write_macros("refinement_numbers", [
    "RefShowRaw"   => tex_sci_bare(ci("raw", :show)),
    "RefShowPolyTwo" => tex_sci_bare(ci("Poly-2", :show)),
    "RefShowNewton"=> tex_sci_bare(ci("Newton-4", :show)),
    "RefTurnRaw"   => tex_sci_bare(ci("raw", :turn)),
    "RefTurnPolyThree" => tex_sci_bare(ci("Poly-3", :turn)),
    "RefTurnNewton"=> tex_sci_bare(ci("Newton-4", :turn)),
    "RefTurnComb"  => tex_sci_bare(ci("Combined", :turn)),
    "RefCombOver"  => @sprintf("%.0f", 100*(costs["Combined"]-t_raw)/t_raw),
])
@info "s04 done" showcase_raw=ci("raw",:show) turning_newton=ci("Newton-4",:turn) turning_poly3=ci("Poly-3",:turn)
