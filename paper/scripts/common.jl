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
# Saving figures (PDF with rasterized heatmaps by default; PNG on request)
# ---------------------------------------------------------------------------
function save_fig(fig, name::AbstractString; kind::Symbol = :vector, px_per_unit = 4)
    @assert isascii(name) "figure names must be ASCII: $name"
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
    path = joinpath(TAB_DIR, name * ".tex")
    open(path, "w") do io
        println(io, "% auto-generated by paper/scripts -- do not edit by hand")
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

"Format seconds compactly for tables (ms below 1 s)."
function tex_time(t::Real)
    t < 1.0 ? @sprintf("%.1f\\,ms", 1000t) : @sprintf("%.2f\\,s", t)
end

"Plain-text variant for figure titles."
function tex_time_plain(t::Real)
    t < 1.0 ? @sprintf("%.0f ms", 1000t) : @sprintf("%.2f s", t)
end
