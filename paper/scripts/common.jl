# Shared infrastructure for all paper studies.
# Include this file first from every study script / the driver.

using CairoMakie
const MAxis = CairoMakie.Axis   # avoid the MDBM.Axis / Makie.Axis export clash
using DelimitedFiles
using Printf
using Statistics
using LinearAlgebra
using Random
using Serialization
using Dates

# ---------------------------------------------------------------------------
# Paths (everything anchored to this file's location; ASCII-only artifact names)
# ---------------------------------------------------------------------------
const SCRIPTS_DIR = @__DIR__
const PAPER_DIR   = dirname(SCRIPTS_DIR)
const REPO_DIR    = dirname(PAPER_DIR)
const FIG_DIR     = joinpath(PAPER_DIR, "figures")
const TAB_DIR     = joinpath(PAPER_DIR, "tables")
const DATA_DIR    = joinpath(PAPER_DIR, "data")
const CACHE_DIR   = joinpath(DATA_DIR, "cache")
const GEN_DIR     = joinpath(PAPER_DIR, "generated")
foreach(mkpath, (FIG_DIR, TAB_DIR, DATA_DIR, CACHE_DIR, GEN_DIR))

# ---------------------------------------------------------------------------
# Global flags (set by generate_all.jl from ARGS)
# ---------------------------------------------------------------------------
const FORCE = Ref(false)   # ignore caches
const FAST  = Ref(false)   # draft mode: reduced resolutions

# ---------------------------------------------------------------------------
# Reproducibility / stable timings
# ---------------------------------------------------------------------------
Random.seed!(0)
LinearAlgebra.BLAS.set_num_threads(1)  # avoid oversubscription under Threads.@threads

# ---------------------------------------------------------------------------
# Figure theme: elsarticle 3p column widths (90 mm single, 190 mm full)
# ---------------------------------------------------------------------------
mm2pt(w_mm) = w_mm * 72 / 25.4
const W_SINGLE = mm2pt(90)
const W_ONEHALF = mm2pt(140)
const W_FULL   = mm2pt(190)

function paper_theme()
    Theme(
        fontsize = 9,
        figure_padding = 4,
        Axis = (
            xgridvisible = false, ygridvisible = false,
            xtickalign = 1, ytickalign = 1,
            spinewidth = 0.8, xtickwidth = 0.8, ytickwidth = 0.8,
            titlefont = :regular, titlesize = 9,
        ),
        Colorbar = (spinewidth = 0.8, tickwidth = 0.8, tickalign = 1),
        Lines = (linewidth = 1.5,),
        Legend = (framevisible = true, framewidth = 0.5, patchsize = (12, 6), rowgap = 0),
    )
end
set_theme!(paper_theme())

# Fixed color conventions used by EVERY stability chart in the paper
const CHART_CMAP = :viridis
const BOUNDARY_COLOR = :black
const BOUNDARY_LW = 1.2

"Combined interpolable coloring: Z in the unstable domain, sigma_est in the stable one."
combined_metric(Z_mat, sigma_mat) = Z_mat .+ (Z_mat .== 0) .* sigma_mat

# ---------------------------------------------------------------------------
# SIGNED BILINEAR colour scale (the one the charts actually use)
# ---------------------------------------------------------------------------
# A single linear axis over C = Z + (Z==0)*sigma wastes almost all of its range:
# the unstable side spans integers up to 5 or 6 while the stable side spans
# sigma in (-0.1, 0), so the entire spectral-gap information is squeezed into
# ~2% of the colour bar and reads as one flat shade.
#
# Instead, map each side to its OWN normalized coordinate with a common zero at
# the stability boundary:
#
#     t = -|sigma| / |sigma_min|   in [-1, 0)   where Z == 0  (stable)
#     t = +Z / Z_max              in (0, +1]   where Z >= 1  (unstable)
#
# Both halves are simply "value divided by that half's extreme". The zero of
# each axis is a real, labelled tick at the centre of the bar -- the boundary
# where sigma = 0 meets Z = 0 -- which is what makes the scale describable in
# one sentence. The colour AT zero on the unstable side is full red; the first
# actual count, Z = 1, therefore already carries some darkening, which is fine:
# what the eye needs is the ORDERING of the counts, and the boundary itself is
# marked by the blue-to-red break, not by a shade of red.
#
# so both halves use the full width of their own colour ramp regardless of how
# different their magnitudes are. The scale is linear on each side (no arctan
# compression is needed here: neither branch is singular at 0), and the break at
# t = 0 is deliberate -- that is the stability boundary, where the colour SHOULD
# jump.
#
#     stable:    sigma = 0 (boundary) -> BLUE ....... sigma_min -> GREEN
#     unstable:  Z small              -> RED ........ Z_max     -> BLACK
const COL_STABLE_ZERO = RGBf(0.15, 0.35, 1.00)   # sigma -> 0-   (marginally stable)
const COL_STABLE_FAR  = RGBf(0.00, 0.75, 0.25)   # sigma -> min  (deeply stable)
const COL_UNSTAB_ZERO = RGBf(1.00, 0.15, 0.10)   # Z -> 0+       (barely unstable)
# Deep maroon rather than pure black at the far end: on charts with only two or
# three distinct counts the top of the ramp covers large areas, and pure black
# turns them into heavy voids that dominate the figure and swallow any
# annotation drawn over them.
const COL_UNSTAB_FAR  = RGBf(0.28, 0.00, 0.05)   # Z = Z_max     (badly unstable)
# Built by sampling each branch SEPARATELY, rather than as a 4-stop gradient
# with the stops crowded around 0.5: any sample landing inside that tiny window
# is a blue-red average, i.e. purple, and it appears as a violet fringe just
# inside the stable domain -- exactly where sigma is closest to zero and the
# chart is read most carefully. Sampling per branch makes the break exact: the
# last stable level is blue, the first unstable level is red, and no
# intermediate colour exists.
const BILINEAR_CMAP = let n = 256
    lerp(a, b, t) = RGBf(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t,
                         a.b + (b.b - a.b) * t)
    cols = map(range(0, 1; length = n)) do t
        if t < 0.5
            lerp(COL_STABLE_FAR, COL_STABLE_ZERO, t / 0.5)   # green -> blue
        else
            lerp(COL_UNSTAB_ZERO, COL_UNSTAB_FAR, (t - 0.5) / 0.5)  # red -> maroon
        end
    end
    cgrad(cols)
end

"""
    bilinear_metric(Z_mat, sigma_mat; σ_ref=nothing, Z_ref=nothing) -> (T, sigma_min, Z_max)

Pass `σ_ref` / `Z_ref` to force the two limits instead of taking them from this
grid. Needed whenever several panels must be COMPARABLE (e.g. the same chart at
four tolerances): per-panel normalization would give each its own scale and
hide the very differences the figure is about.

Signed bilinear chart field `T ∈ [-1, 1]` with the stability boundary at 0
(see the comment above). Plot with `colormap = BILINEAR_CMAP` and
`colorrange = (-1, 1)`. Returns the two extremes as well, since they are what
the colour bar must be labelled with -- each panel is normalized to its OWN
`sigma_min` and `Z_max`, which is the entire point.

Points with the invalid marker (`Z < 0`, i.e. a root exactly on the line, so
the count is undefined) become `NaN` and are left unpainted.
"""
function bilinear_metric(Z_mat, sigma_mat; σ_quantile = 0.02,
                         σ_ref = nothing, Z_ref = nothing)
    stable_σ = [sigma_mat[i] for i in eachindex(Z_mat)
                if Z_mat[i] == 0 && isfinite(sigma_mat[i])]
    # ROBUST lower limit, not the outright minimum. A handful of deeply stable
    # pixels (sigma ~ -1.8) against a domain whose typical gap is ~ -0.1 would
    # otherwise squeeze the entire visible variation into a few percent of the
    # ramp -- the same defect, moved from the whole chart into the stable half.
    # The reference implementation solves this with an arctan; on a linear
    # scale the equivalent is to clip the tail, so the deepest few percent
    # saturate at green and the rest of the domain gets the full gradient.
    σ_min = σ_ref !== nothing ? σ_ref :
            (isempty(stable_σ) ? -1.0 : min(quantile(stable_σ, σ_quantile), -eps()))
    unstable_Z = [Z_mat[i] for i in eachindex(Z_mat) if Z_mat[i] > 0]
    Z_max = Z_ref !== nothing ? Z_ref : (isempty(unstable_Z) ? 1 : maximum(unstable_Z))
    T = map(eachindex(Z_mat)) do i
        z = Z_mat[i]
        if z < 0                      # invalid marker: count undefined here
            NaN
        elseif z == 0
            s = isfinite(sigma_mat[i]) ? sigma_mat[i] : 0.0
            -clamp(abs(s) / abs(σ_min), 0.0, 1.0)
        else
            _z_to_t(z, Z_max)
        end
    end
    return reshape(T, size(Z_mat)), σ_min, Z_max
end

# Z / Z_max: 0 sits exactly at the centre of the bar (and is a visible tick),
# Z_max at the top. Deliberately the simplest possible rule.
_z_to_t(z, Z_max) = clamp(z / max(Z_max, 1), 0.0, 1.0)

"""
    bilinear_ticks(σ_min, Z_max; n=3) -> (positions, labels)

Tick positions on the `t ∈ [-1,1]` axis together with the REAL values they
stand for: decay rates on the stable half, root counts on the unstable half.
"""
function bilinear_ticks(σ_min, Z_max; n = 3)
    pos = Float64[]; lab = String[]
    for k in n:-1:1                                  # stable half: t < 0
        t = -k / n
        # the end of the ramp is a CLIP, not the minimum -- say so
        s = @sprintf("%.2g", t * abs(σ_min))
        push!(pos, t); push!(lab, k == n ? "≤" * s : s)
    end
    push!(pos, 0.0); push!(lab, "0")             # the boundary, at the centre
    # unstable half: tick the ACTUAL integer counts, at the t they map to --
    # they are what the colours mean, and there are rarely more than a handful
    zs = Z_max <= 6 ? (1:Z_max) : round.(Int, range(1, Z_max; length = 5))
    for z in unique(zs)
        push!(pos, _z_to_t(z, Z_max)); push!(lab, string(z))
    end
    return pos, lab
end

"""
    annotate_panel!(ax, xr, yr, text_str)

Put an annotation in the lower-left corner of a chart on a semi-transparent
white plate, so that it stays readable over any colour -- in particular over
the black of a badly unstable region, where white-on-dark text disappears.
"""
function annotate_panel!(ax, xr, yr, text_str; fontsize = 6,
                         panel_pt = (W_FULL / 3, W_FULL * 0.60 / 2))
    w, h = xr[2] - xr[1], yr[2] - yr[1]
    lines_ = split(text_str, '\n')
    nlines = length(lines_)
    ncols = maximum(length, lines_)
    # The plate must be given in DATA coordinates while the text is sized in
    # POINTS, so its extent cannot be measured from the text plot (Makie only
    # knows it at render time; boundingbox returns the anchor with zero width).
    # Convert instead: estimate the text box in points from the font metrics and
    # divide by the panel's size in points. `panel_pt` is the drawable area of
    # ONE panel -- the caller must pass it if the layout is not the gallery's
    # 3x2 grid. Erring large is deliberate: text that runs past the plate lands
    # on the black of a badly unstable region and vanishes, which is the exact
    # failure the plate exists to prevent.
    char_pt = 0.62 * fontsize                     # mean advance width
    line_pt = 1.35 * fontsize                     # line height
    pw = min(0.95, (ncols * char_pt) / (0.74 * panel_pt[1]) + 0.05) * w
    ph = min(0.60, (nlines * line_pt) / (0.72 * panel_pt[2]) + 0.04) * h
    x0, y0 = xr[1] + 0.025w, yr[1] + 0.025h
    poly!(ax, Rect2f(x0, y0, pw, ph); color = (:white, 0.82),
        strokecolor = (:black, 0.35), strokewidth = 0.3)
    text!(ax, x0 + 0.02w, y0 + 0.02h; text = text_str, align = (:left, :bottom),
        fontsize = fontsize, color = :black)
    return nothing
end

# ---------------------------------------------------------------------------
# Saving figures (PDF with rasterized heatmaps by default; PNG on request)
# ---------------------------------------------------------------------------
function save_fig(fig, name::AbstractString; kind::Symbol = :vector, px_per_unit = 4)
    @assert isascii(name) "figure names must be ASCII: $name"
    FAST[] && @warn "DRAFT-mode figure (reduced resolution) -- rerun WITHOUT --fast before submission" name
    if kind == :vector
        path = joinpath(FIG_DIR, name * ".pdf")
        save(path, fig)
    else
        path = joinpath(FIG_DIR, name * ".png")
        save(path, fig; px_per_unit = px_per_unit)
    end
    # low-resolution PNG preview for quick visual inspection (not committed)
    preview_dir = joinpath(FIG_DIR, "preview")
    mkpath(preview_dir)
    save(joinpath(preview_dir, name * ".png"), fig; px_per_unit = 2)
    @info "figure saved" path
    return path
end

# ---------------------------------------------------------------------------
# Compute/render split: cache heavy results under data/cache
# ---------------------------------------------------------------------------
function with_cache(compute::Function, cache_name::AbstractString)
    path = joinpath(CACHE_DIR, cache_name * ".jls")
    if !FORCE[] && isfile(path)
        @info "cache hit" path
        return Serialization.deserialize(path)
    end
    result = compute()
    Serialization.serialize(path, result)
    @info "cache written" path
    return result
end

# ---------------------------------------------------------------------------
# Timing helper: median of warm repeats, GC between runs, allocations recorded
# ---------------------------------------------------------------------------
function benchmark_sweep(f::Function; repeats::Int = 5)
    f()  # warm-up (excluded)
    times = Float64[]
    mem = 0
    for r in 1:repeats
        GC.gc()
        stats = @timed f()
        push!(times, stats.time)
        r == 1 && (mem = stats.bytes)
    end
    return (t_med = median(times), t_min = minimum(times),
            t_mean = mean(times), t_std = std(times), mem_bytes = mem)
end

# ---------------------------------------------------------------------------
# CSV helpers (plain writedlm-based; header in first row)
# ---------------------------------------------------------------------------
function write_csv(name::AbstractString, header::Vector{String}, rows::Vector{<:Tuple})
    path = joinpath(DATA_DIR, name * ".csv")
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(string.(row), ","))
        end
    end
    @info "data written" path
    return path
end

function read_csv(name::AbstractString)
    path = joinpath(DATA_DIR, name * ".csv")
    raw, header = readdlm(path, ',', Any; header = true)
    return raw, vec(string.(header))
end

csv_exists(name::AbstractString) = isfile(joinpath(DATA_DIR, name * ".csv"))

# ---------------------------------------------------------------------------
# Booktabs table fragments (caption/label live in the hand-written TeX)
# ---------------------------------------------------------------------------
function write_booktabs(name::AbstractString, colspec::String,
                        header::Vector{String}, rows::Vector{Vector{String}})
    FAST[] && @warn "DRAFT-mode table (reduced resolution) -- rerun WITHOUT --fast before submission" name
    path = joinpath(TAB_DIR, name * ".tex")
    open(path, "w") do io
        println(io, "% auto-generated by paper/scripts -- do not edit by hand")
        FAST[] && println(io, "% !! DRAFT MODE (--fast): reduced resolutions, timings not final !!")
        println(io, "\\begin{tabular}{$colspec}")
        println(io, "\\toprule")
        println(io, join(header, " & "), " \\\\")
        println(io, "\\midrule")
        for row in rows
            println(io, join(row, " & "), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    @info "table written" path
    return path
end

"Format a number as LaTeX scientific notation, e.g. 3.2\\cdot10^{-6}."
function tex_sci(x::Real; digits::Int = 1)
    (x == 0 || !isfinite(x)) && return string(x)
    e = floor(Int, log10(abs(x)))
    m = x / 10.0^e
    return @sprintf("\$%.*f \\cdot 10^{%d}\$", digits, m, e)
end

"""
Scientific notation WITHOUT the surrounding `\$ \$`, for use inside a macro that
the manuscript will place in its own math context (`tex_sci` brings its own
delimiters, which then nest and break the build).
"""
function tex_sci_bare(x::Real)
    x == 0 && return "0"
    e = floor(Int, log10(abs(x)))
    m = x / 10.0^e
    return @sprintf("%.1f \\cdot 10^{%d}", m, e)
end

"""
Write a set of `\\newcommand` macros to `generated/<name>.tex`.

Every number quoted in the manuscript prose should come from here rather than
being copied by hand, so that a rerun cannot leave a stale figure in the text.
`pairs` maps macro name => value (already formatted).
"""
function write_macros(name::AbstractString, pairs::Vector{<:Pair})
    path = joinpath(GEN_DIR, name * ".tex")
    open(path, "w") do io
        println(io, "% auto-generated by paper/scripts -- do not edit by hand")
        for (k, v) in pairs
            println(io, "\\newcommand{\\", k, "}{", v, "}")
        end
    end
    @info "macros written" path
    return path
end

"Format seconds compactly for tables (ms below 1 s)."
function tex_time(t::Real)
    t < 1.0 ? @sprintf("%.1f\\,ms", 1000t) : @sprintf("%.2f\\,s", t)
end

"Plain-text variant for figure titles."
function tex_time_plain(t::Real)
    t < 1.0 ? @sprintf("%.0f ms", 1000t) : @sprintf("%.2f s", t)
end
