# Study s05: "solver zoo" -- error vs CPU-time Pareto front for the fixed-step
# trapezoid, adaptive Gauss-Kronrod quadrature and several ODE solvers, on a
# 40x40 grid of the 4th-order benchmark oscillator.
# Checkpointed: one CSV row per (method, control) appended immediately, existing
# rows are skipped on rerun.
# Produces: figures/fig_solver_zoo.pdf, data/solver_zoo.csv

include(joinpath(@__DIR__, "systems.jl"))

const ZOO_WMAX = 1e3
const ZOO_TIME_CAP = 5.0     # stop escalating a method when a sweep exceeds this

nZ = FAST[] ? 20 : 40
Pz = LinRange(-2.0, 4.0, nZ)
Dz = LinRange(-2.0, 5.0, nZ)
params_zoo = vec([(p, d) for p in Pz, d in Dz])

# Reference field (tight stiff solve), cached
Zref = with_cache("s05_reference_$(nZ)") do
    _, Z_raws = calculate_unstable_roots_p_vec(D_fourth, params_zoo;
        n_roots_to_track = 0, ω_max = ZOO_WMAX, reltol = 1e-13, abstol = 1e-13,
        solver = Rosenbrock23(), maxiters = Int(1e8))
    Z_raws
end

ZOO_CSV = joinpath(DATA_DIR, "solver_zoo.csv")
if !isfile(ZOO_CSV) || FORCE[]
    open(ZOO_CSV, "w") do io
        println(io, "method,control,mean_err,median_err,max_err,time_s,mem_mb")
    end
end
# Existing rows are skipped on rerun, but their recorded time is still
# returned, so the escalation loops below re-hit the time cap in the same
# place instead of proceeding past the point a previous session stopped at.
existing_t = Dict{Tuple{String, Float64}, Float64}()
for line in readlines(ZOO_CSV)[2:end]
    f = split(line, ',')
    existing_t[(String(f[1]), parse(Float64, f[2]))] = parse(Float64, f[6])
end

function zoo_run!(method::String, control::Float64, sweep::Function)
    haskey(existing_t, (method, control)) && return existing_t[(method, control)]
    res = benchmark_sweep(sweep; repeats = 3)
    Z_raws = sweep()
    err = abs.(Z_raws .- Zref)
    open(ZOO_CSV, "a") do io
        println(io, join([method, control, mean(err), median(err), maximum(err),
            res.t_med, res.mem_bytes / 2^20], ","))
    end
    @info "zoo" method control mean_err = mean(err) t = res.t_med
    return res.t_med
end

# fixed-step trapezoid: control = number of steps
for steps in round.(Int, 10 .^ (1.5:0.25:4.5))
    t = zoo_run!("fixed-step", Float64(steps), () ->
        calculate_unstable_roots_fixed_step_p_vec(D_fourth, params_zoo;
            ω_max = ZOO_WMAX, steps = steps)[2])
    t !== nothing && t > ZOO_TIME_CAP && break
end

# adaptive methods: control = tolerance
TOLS_ZOO = 10.0 .^ (-2:-1.0:-11)
for tol in TOLS_ZOO
    t = zoo_run!("QuadGK", tol, () ->
        calculate_unstable_roots_quadgk_p_vec(D_fourth, params_zoo;
            ω_max = ZOO_WMAX, reltol = tol, abstol = tol)[2])
    t !== nothing && t > ZOO_TIME_CAP && break
end
for (name, solver) in [("BS3", BS3()), ("Tsit5", Tsit5()), ("Vern9", Vern9()),
                       ("Rosenbrock23", Rosenbrock23()), ("Rodas5P", Rodas5P()),
                       ("AutoTsit5(Rosenbrock23)", AutoTsit5(Rosenbrock23()))]
    for tol in TOLS_ZOO
        t = zoo_run!(name, tol, () ->
            calculate_unstable_roots_p_vec(D_fourth, params_zoo;
                n_roots_to_track = 0, ω_max = ZOO_WMAX, reltol = tol, abstol = tol,
                solver = solver)[2])
        t !== nothing && t > ZOO_TIME_CAP && break
    end
end

# ---------------------------------------------------------------------------
# Render the Pareto figure from the CSV
# ---------------------------------------------------------------------------
raw, hdr = read_csv("solver_zoo")
methods = unique(string.(raw[:, 1]))
fig = Figure(size = (W_ONEHALF, W_ONEHALF * 0.62))
ax = MAxis(fig[1, 1], xlabel = "sweep CPU time [s]  ($(nZ)x$(nZ) grid)",
    ylabel = "mean |Z̃ - Z̃_ref|", xscale = log10, yscale = log10)
cols = Makie.wong_colors()
for (i, m) in enumerate(methods)
    sel = findall(==(m), string.(raw[:, 1]))
    ts = Float64.(raw[sel, 6])
    es = max.(Float64.(raw[sel, 3]), 1e-16)
    ord = sortperm(ts)
    scatterlines!(ax, ts[ord], es[ord]; label = m, color = cols[mod1(i, 7)],
        linestyle = i > 7 ? :dash : :solid, markersize = 5)
end
axislegend(ax; position = :lb, labelsize = 7)
save_fig(fig, "fig_solver_zoo")
