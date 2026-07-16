# Study s06: what a COMPLETE stability chart costs.
# Every brute-force row is a full 100x100 = 10 000-point chart, so the numbers
# are directly readable as "time for one chart"; the per-point cost is listed
# alongside. Every MDBM row reports the initial mesh, the refinement levels,
# the resulting equivalent resolution and the number of function evaluations
# actually performed.
# Produces: tables/tab_grid_timings.tex, tables/tab_mdbm.tex,
#           data/grid_timings.csv, data/mdbm_stats.csv, data/ttfx.csv

include(joinpath(@__DIR__, "studies_common.jl"))
include(joinpath(@__DIR__, "systems.jl"))

D_simple(λ, p) = λ^2 + p[1] * λ + p[2]

const NGRID = FAST[] ? 40 : 100           # 100 x 100 chart everywhere
const MDBM_N0 = 20                        # initial mesh per axis
const MDBM_IT = 4                         # refinement levels

CASES = [
    ("2nd-order, no delay", D_simple,   (-1.0, 2.0), (-1.0, 2.0), 1e3),
    ("4th-order + delay",   D_fourth,   (-2.0, 4.0), (-2.0, 5.0), 1e6),
    ("2-DOF DAE showcase",  D_showcase, SHOWCASE_PRANGE, SHOWCASE_DRANGE, 1e4),
]

BACKENDS = [
    ("ODE Vern9 + root (default)", (D, prm, wm) -> calculate_unstable_roots_p_vec(D, prm; ω_max = wm)),
    ("ODE Vern9, Z only",          (D, prm, wm) -> calculate_unstable_roots_p_vec(D, prm; ω_max = wm, n_roots_to_track = 0)),
    ("QuadGK, Z only",             (D, prm, wm) -> calculate_unstable_roots_quadgk_p_vec(D, prm; ω_max = wm)),
    ("fixed-step (500)",           (D, prm, wm) -> calculate_unstable_roots_fixed_step_p_vec(D, prm; ω_max = min(wm, 1e3), steps = 500)),
]

# ---------------------------------------------------------------------------
# Brute-force chart timings (median of repeats)
# ---------------------------------------------------------------------------
timing_rows = with_cache("s06_grid_timings_v2_$(NGRID)") do
    out = Tuple[]
    for (cname, D, xr, yr, wm) in CASES
        xv = LinRange(xr..., NGRID); yv = LinRange(yr..., NGRID)
        params = vec([(x, y) for x in xv, y in yv])
        for (bname, runner) in BACKENDS
            res = benchmark_sweep(() -> runner(D, params, wm); repeats = FAST[] ? 2 : 5)
            push!(out, (cname, bname, length(params), res.t_med,
                res.t_med / length(params) * 1e6, res.t_std, res.mem_bytes / 2^20))
            @info "chart timing" cname bname chart_s = res.t_med
        end
    end
    out
end
write_csv("grid_timings",
    ["system", "backend", "n_points", "chart_time_s", "per_point_us", "t_std_s", "mem_mb"],
    timing_rows)

# ---------------------------------------------------------------------------
# MDBM boundary tracing: cost AND what resolution it buys
# ---------------------------------------------------------------------------
mdbm_rows = with_cache("s06_mdbm_stats_v2") do
    out = Tuple[]
    for (cname, D, xr, yr, wm) in CASES
        f = () -> mdbm_boundary(D, xr, yr; ngrid = MDBM_N0, Niter = MDBM_IT, ω_max = wm)
        b = f()                                  # warm-up + result
        t = median([(@elapsed f()) for _ in 1:3])
        n_eval = length(getevaluatedpoints(b.prob)[1])
        n_sol = length(getinterpolatedsolution(b.prob)[1])
        equiv = MDBM_N0 * 2^MDBM_IT
        push!(out, (cname, MDBM_N0, MDBM_IT, equiv, n_eval, n_sol, t,
            equiv^2 / n_eval))
        @info "mdbm stats" cname n_eval equiv t
    end
    out
end
write_csv("mdbm_stats",
    ["system", "n0", "iters", "equiv_res", "n_evaluated", "n_boundary_pts",
     "time_s", "saving_factor"], mdbm_rows)

# ---------------------------------------------------------------------------
# Cold start probes (fresh processes) -> data/ttfx.csv
# ---------------------------------------------------------------------------
ttfx_csv = joinpath(DATA_DIR, "ttfx.csv")
if !isfile(ttfx_csv) || FORCE[]
    isfile(ttfx_csv) && rm(ttfx_csv)
    probe = joinpath(SCRIPTS_DIR, "ttfx_probe.jl")
    for backend in ["direct", "quadgk", "fixed"]
        @info "ttfx probe" backend
        run(`julia --project=$SCRIPTS_DIR --startup-file=no -t auto $probe $backend`)
    end
end

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
rows_tex = Vector{String}[]
for (cname, bname, n, t, per_pt, tstd, mem) in timing_rows
    push!(rows_tex, [cname, bname, tex_time(t), @sprintf("%.0f", per_pt),
        @sprintf("%.0f", mem)])
end
write_booktabs("tab_grid_timings", "llccc",
    ["system", "back-end", "chart ($(NGRID)\$\\times\$$(NGRID))", "per point [\$\\mu\$s]",
     "alloc [MB]"], rows_tex)

rows_mdbm = Vector{String}[]
for (cname, n0, it, equiv, nev, nsol, t, save) in mdbm_rows
    push!(rows_mdbm, [cname, "$(n0)\$\\times\$$(n0)", string(it),
        "$(equiv)\$\\times\$$(equiv)", string(nev), string(nsol), tex_time(t),
        @sprintf("%.0f\$\\times\$", save)])
end
write_booktabs("tab_mdbm", "lcccccc",
    ["system", "initial mesh", "levels", "equiv.\\ resolution", "evaluations",
     "boundary pts", "time", "saving"], rows_mdbm)
