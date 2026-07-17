# Driver: regenerates ALL paper data, figures and tables.
#
#   julia --project=paper/scripts -t auto paper/scripts/generate_all.jl [options]
#
# Options:
#   --force        ignore caches, recompute everything
#   --fast         draft mode (reduced grid resolutions, skips the solver zoo)
#   --only=s01,s03 run only the listed studies

include(joinpath(@__DIR__, "common.jl"))

FORCE[] = "--force" in ARGS
FAST[]  = "--fast" in ARGS
only_arg = findfirst(a -> startswith(a, "--only="), ARGS)
const ONLY = only_arg === nothing ? String[] :
    split(replace(ARGS[only_arg], "--only=" => ""), ",")

include(joinpath(@__DIR__, "machine_info.jl"))

const STUDIES = [
    "s01_showcase",
    "s02_tolerance_error",
    "s03_convergence",
    "s04_refinement",
    "s05_solver_zoo",
    "s06_grid_timings",
    "s07_semidisc",
    "s08_gallery",
    "s09_sigma_contours",
    "s10_fractional_controller",
    "s11_diagnostics",
    "s12_peak_repair",
]

__results = Dict{String, Any}()
for study in STUDIES
    if !isempty(ONLY) && !(study in ONLY)
        __results[study] = :skipped
        continue
    end
    if FAST[] && study == "s05_solver_zoo"
        __results[study] = :skipped_fast
        continue
    end
    path = joinpath(@__DIR__, "studies", study * ".jl")
    if !isfile(path)
        __results[study] = :missing
        continue
    end
    @info "===== running study =====" study
    t0 = time()
    try
        include(path)
        __results[study] = round(time() - t0; digits = 1)
    catch err
        @error "study failed" study err
        __results[study] = err
    end
end

println("\n===== SUMMARY =====")
for study in STUDIES
    r = get(__results, study, :unknown)
    status = r isa Number ? "OK  ($(r) s)" : string(r)
    println(rpad(study, 28), status)
end
