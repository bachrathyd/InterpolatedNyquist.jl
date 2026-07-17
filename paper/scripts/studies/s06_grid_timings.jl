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

# omega_max = 1e4 for every chart in the paper: three decades above the highest
# resonance of these systems, leaving a truncated tail five orders below the
# rounding threshold. (For the 4th-order system, whose ripple decays like
# omega^-3, 1e6 would be nearly free -- 1426 vs 1330 evaluations -- but for the
# showcase, whose ripple decays like 1/omega, it costs ~60x. One window,
# chosen for the harder case.)
const WMAX_CHART = 1e4
CASES = [
    ("2nd-order, no delay", D_simple,   (-1.0, 2.0), (-1.0, 2.0), WMAX_CHART),
    ("4th-order + delay",   D_fourth,   (-2.0, 4.0), (-2.0, 5.0), WMAX_CHART),
    ("2-DOF DAE showcase",  D_showcase_reduced, SHOWCASE_PRANGE, SHOWCASE_DRANGE, WMAX_CHART),
]

BACKENDS = [
    ("ODE Vern9 + root (default)", (D, prm, wm) -> calculate_unstable_roots_p_vec(D, prm; ω_max = wm)),
    ("ODE Vern9, Z only",          (D, prm, wm) -> calculate_unstable_roots_p_vec(D, prm; ω_max = wm, n_roots_to_track = 0)),
    ("QuadGK, Z only",             (D, prm, wm) -> calculate_unstable_roots_quadgk_p_vec(D, prm; ω_max = wm)),
    ("fixed-step (500)",           (D, prm, wm) -> calculate_unstable_roots_fixed_step_p_vec(D, prm; ω_max = min(wm, 1e3), steps = 500)),
]

# ---------------------------------------------------------------------------
# Brute-force chart timings (median of repeats)
# Every backend's counts are verified against a tight-tolerance reference of
# the SAME chart before its time is recorded: a fast backend that returns
# wrong counts would otherwise be benchmarked as a "speedup". The adaptive
# backends must match everywhere; the fixed-step backend (hand-set resolution,
# real-time floor) reports its mismatches honestly in the table.
# ---------------------------------------------------------------------------
timing_rows = with_cache("s06_grid_timings_v4_$(NGRID)") do
    out = Tuple[]
    for (cname, D, xr, yr, wm) in CASES
        xv = LinRange(xr..., NGRID); yv = LinRange(yr..., NGRID)
        params = vec([(x, y) for x in xv, y in yv])
        Z_ref, _ = calculate_unstable_roots_p_vec(D, params; ω_max = wm,
            n_roots_to_track = 0, reltol = 1e-8, abstol = 1e-8)
        for (bname, runner) in BACKENDS
            Z_b = runner(D, params, wm)[1]
            n_wrong = count(Z_b .!= Z_ref)
            if n_wrong > 0 && !occursin("fixed-step", bname)
                error("s06: backend '$bname' returned $n_wrong wrong counts on '$cname'")
            end
            # Repeat only cheap sweeps. The DAE showcase chart takes ~1 min per
            # sweep; five repeats of it per back-end would spend twenty minutes
            # refining a number whose leading digits are already stable.
            f = () -> runner(D, params, wm)
            t1 = @elapsed f()
            res = t1 < 5.0 ? benchmark_sweep(f; repeats = FAST[] ? 2 : 5) :
                             (t_med = t1, t_std = 0.0, mem_bytes = 0)
            push!(out, (cname, bname, length(params), res.t_med,
                res.t_med / length(params) * 1e6, res.t_std, res.mem_bytes / 2^20,
                n_wrong))
            @info "chart timing" cname bname chart_s = res.t_med n_wrong
        end
    end
    out
end
write_csv("grid_timings",
    ["system", "backend", "n_points", "chart_time_s", "per_point_us", "t_std_s", "mem_mb", "n_wrong_Z"],
    timing_rows)

# ---------------------------------------------------------------------------
# MDBM boundary tracing: cost AND what resolution it buys
# ---------------------------------------------------------------------------
mdbm_rows = with_cache("s06_mdbm_stats_v3") do
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
# The paper's two operating points, measured on the SAME 100x100 chart and the
# SAME window (omega_max = 1e4). They differ only in the tolerance, which is
# the knob a user actually turns:
#   ACCURATE -- tol = 1e-5, the package default: certified counts.
#   FAST     -- tol = 1e-3, for a first exploration: a few boundary-adjacent
#               counts may be off by one, and the traced boundary is unmoved.
# Both are reported against the same reference count so the price of the fast
# preset is stated, not hidden.
# ---------------------------------------------------------------------------
preset_rows = with_cache("s06_presets_v2_$(NGRID)") do
    out = Tuple[]
    for (cname, D, xr, yr, wm) in CASES
        xv = LinRange(xr..., NGRID); yv = LinRange(yr..., NGRID)
        params = vec([(x, y) for x in xv, y in yv])
        # Reference counts at the same window, tight tolerance.
        Z_ref, _ = calculate_unstable_roots_p_vec(D, params; ω_max = WMAX_CHART,
            n_roots_to_track = 0, reltol = 1e-8, abstol = 1e-8)
        # Both presets use the standard window; they differ in the tolerance,
        # which is the knob a user actually turns. "accurate" is the package
        # default; "fast" is the exploratory setting of the gallery.
        for (pname, pwm, ptol) in (("accurate", WMAX_CHART, 1e-5),
                                   ("fast", WMAX_CHART, 1e-3))
            f = () -> calculate_unstable_roots_p_vec(D, params; ω_max = pwm,
                reltol = ptol, abstol = ptol)
            # Warm up on a handful of points (JIT only), then time ONE sweep.
            # Repeating is only worth it when a sweep is short: the showcase's
            # accurate preset is a ~50 min chart, and five repeats of that cost
            # four hours to sharpen a number whose leading digit is already
            # certain.
            calculate_unstable_roots_p_vec(D, params[1:min(20, end)]; ω_max = pwm,
                reltol = ptol, abstol = ptol)
            t1 = @elapsed Zt = f()
            t = t1
            if t1 < 10.0
                res = benchmark_sweep(f; repeats = FAST[] ? 2 : 5)
                t = res.t_med
            end
            Z = Zt[1]
            push!(out, (cname, pname, pwm, ptol, length(params), t,
                t / length(params) * 1e6, count(Z .!= Z_ref)))
            @info "preset" cname pname chart_s = t wrong = count(Z .!= Z_ref)
        end
    end
    out
end
write_csv("presets",
    ["system", "preset", "wmax", "tol", "n_points", "chart_time_s",
     "per_point_us", "n_wrong_vs_ref"], preset_rows)

# ---------------------------------------------------------------------------
# What the convenience of automatic extraction costs.
#
# The two characteristic functions are the SAME function (verified to machine
# precision in s01) and give bit-identical charts; they differ only in how they
# are obtained:
#   hand-derived : a flat scalar expression, no determinant, no AD -- what a
#                  user gets after doing the algebra once, by hand.
#   auto-extracted: one evaluation of the user's ODE right-hand side seeded
#                  with ForwardDiff duals, then a 7x7 determinant of the
#                  descriptor matrix -- no algebra at all.
# The counts are asserted identical before either time is recorded, so the
# ratio is a pure measure of the convenience, not of a different answer.
# ---------------------------------------------------------------------------
FORMS = [("hand-derived (no determinant, no AD)", D_showcase_reduced),
         ("automatic extraction from the RHS",    D_showcase)]
extraction_rows = with_cache("s06_extraction_v1_$(NGRID)") do
    xv = LinRange(SHOWCASE_PRANGE..., NGRID); yv = LinRange(SHOWCASE_DRANGE..., NGRID)
    params = vec([(x, y) for x in xv, y in yv])
    out = Tuple[]
    Z_first = nothing
    for (fname, D) in FORMS
        f = () -> calculate_unstable_roots_p_vec(D, params; ω_max = WMAX_CHART)
        f()                                            # warm-up / JIT
        t1 = @elapsed r = f()
        res = t1 < 5.0 ? benchmark_sweep(f; repeats = 3) : (t_med = t1, mem_bytes = 0)
        Z = r[1]
        if Z_first === nothing
            Z_first = Z
        else
            n_diff = count(Z .!= Z_first)
            n_diff == 0 || error("s06: the two forms of D disagree on $n_diff points")
        end
        push!(out, (fname, length(params), res.t_med,
            res.t_med / length(params) * 1e6, res.mem_bytes / 2^20))
        @info "extraction cost" fname chart_s = res.t_med
    end
    out
end
write_csv("extraction_cost",
    ["form", "n_points", "chart_time_s", "per_point_us", "mem_mb"], extraction_rows)

let t_hand = extraction_rows[1][3], t_auto = extraction_rows[2][3]
    # No memory column: the expensive form skips the repeat-benchmark that
    # measures allocations, so it would report a misleading 0.
    rows_ex = [[r[1], tex_time(r[3]), @sprintf("%.0f", r[4])] for r in extraction_rows]
    push!(rows_ex, ["\\emph{ratio}", @sprintf("%.0f\$\\times\$", t_auto / t_hand),
                    @sprintf("%.0f\$\\times\$", extraction_rows[2][4] / extraction_rows[1][4])])
    write_booktabs("tab_extraction", "lcc",
        ["form of \$\\Dfun\$", "chart ($(NGRID)\$\\times\$$(NGRID))",
         "per point [\$\\mu\$s]"], rows_ex)
    write_macros("extraction_numbers", [
        "ExtractRatio"   => @sprintf("%.0f", t_auto / t_hand),
        "ExtractHandT"   => tex_time(t_hand),
        "ExtractAutoT"   => tex_time(t_auto),
        "ExtractHandUs"  => @sprintf("%.0f", extraction_rows[1][4]),
        "ExtractAutoUs"  => @sprintf("%.0f", extraction_rows[2][4]),
    ])
end

rows_pre = Vector{String}[]
for (cname, pname, pwm, ptol, n, t, per_pt, nw) in preset_rows
    push!(rows_pre, [pname == "accurate" ? cname : "", pname,
        @sprintf("\$10^{%d}\$", round(Int, log10(pwm))),
        @sprintf("\$10^{%d}\$", round(Int, log10(ptol))),
        tex_time(t), @sprintf("%.0f", per_pt), string(nw)])
end
write_booktabs("tab_presets", "llccccc",
    ["system", "preset", "\$\\wmax\$", "tol", "chart ($(NGRID)\$\\times\$$(NGRID))",
     "per point [\$\\mu\$s]", "wrong \$\\Zint\$"], rows_pre)

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
for (cname, bname, n, t, per_pt, tstd, mem, n_wrong) in timing_rows
    push!(rows_tex, [cname, bname, tex_time(t), @sprintf("%.0f", per_pt),
        @sprintf("%.0f", mem), string(n_wrong)])
end
write_booktabs("tab_grid_timings", "llcccc",
    ["system", "back-end", "chart ($(NGRID)\$\\times\$$(NGRID))", "per point [\$\\mu\$s]",
     "alloc [MB]", "wrong \$Z\$"], rows_tex)

rows_mdbm = Vector{String}[]
for (cname, n0, it, equiv, nev, nsol, t, save) in mdbm_rows
    push!(rows_mdbm, [cname, "$(n0)\$\\times\$$(n0)", string(it),
        "$(equiv)\$\\times\$$(equiv)", string(nev), string(nsol), tex_time(t),
        @sprintf("%.0f\$\\times\$", save)])
end
write_booktabs("tab_mdbm", "lccccccc",
    ["system", "initial mesh", "levels", "equiv.\\ resolution", "evaluations",
     "boundary pts", "time", "saving"], rows_mdbm)
