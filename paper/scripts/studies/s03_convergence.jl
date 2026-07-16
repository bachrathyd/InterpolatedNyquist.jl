# Study s03: accuracy and cost of the integration back-ends.
#  (a) integer residual vs requested tolerance   (reference-free!)
#  (b) CPU time vs tolerance
#  (c) efficiency front: error (vertical) vs CPU time (horizontal)
#  (d) near-boundary robustness: does a back-end step over the collapsing peak?
# Produces: figures/fig_convergence.pdf, tables/tab_convergence.tex,
#           data/convergence.csv, data/boundary_stress.csv

include(joinpath(@__DIR__, "studies_common.jl"))
include(joinpath(@__DIR__, "systems.jl"))
using QuadGK
import SemiDiscretizationMethod   # only for the independent Z_EXACT check

# Evaluation point: the unstable point selected by s01 (fallback: fixed)
p_conv = if csv_exists("showcase_summary")
    raw, hdr = read_csv("showcase_summary")
    d = Dict(string(raw[i, 1]) => raw[i, 2] for i in 1:size(raw, 1))
    (Float64(d["p_unstable_P"]), Float64(d["p_unstable_D"]))
else
    (2.067415730337079, 2.3405797101449277)
end
@info "convergence study at" p_conv

# ω_max is chosen so that the truncated tail (≈0.2/ω_max for this
# velocity-feedback system) stays far below the tightest tolerance studied.
const WMAX_CONV = 1e6
n_pow = get_n_power_max(D_showcase, p_conv)
Z_EXACT = 2
@info "leading order" n_pow err = abs(n_pow - 4)

# Verify Z_EXACT with a method that shares NOTHING with the one under study:
# dense generalized eigenvalues of the semi-discretized transition mapping
# (no phase integral, no leading-order estimate, no ω_max truncation). Each
# characteristic exponent with Re λ > 0 maps to a multiplier |μ| > 1; the
# spurious discretization modes cluster far inside the unit circle.
z_check = with_cache("s03_zexact_check_v1") do
    A, B = showcase_AB(p_conv)
    Δt = SM.tau / 120
    lddep = SemiDiscretizationMethod.LDDEProblem(
        SemiDiscretizationMethod.ProportionalMX(A),
        [SemiDiscretizationMethod.DelayMX(t -> SM.tau, B)],
        SemiDiscretizationMethod.Additive(zeros(4)))
    m = SemiDiscretizationMethod.DiscreteMapping_LR(lddep,
        SemiDiscretizationMethod.SemiDiscretization(2, Δt), SM.tau;
        n_steps = 1, calculate_additive = true)
    μ = eigen(collect(m.RmappingMX), collect(m.LmappingMX)).values
    count(x -> abs(x) > 1, μ)
end
@assert z_check == Z_EXACT "independent semi-discretization count is $(z_check), expected $(Z_EXACT)"
@info "Z_EXACT verified by semi-discretization (dense eig, n=120, order 2)" z_check

TOLS = 10.0 .^ (-3:-1.0:-10)
METHODS = [
    ("Vern9 (default)", tol -> calculate_unstable_roots_direct(D_showcase, p_conv;
        n_roots_to_track = 0, ω_max = WMAX_CONV, reltol = tol, abstol = tol,
        solver = Vern9(), n_power_max = n_pow, maxiters = 10^9)[2]),
    ("Tsit5", tol -> calculate_unstable_roots_direct(D_showcase, p_conv;
        n_roots_to_track = 0, ω_max = WMAX_CONV, reltol = tol, abstol = tol,
        solver = Tsit5(), n_power_max = n_pow, maxiters = 10^9)[2]),
    ("BS3", tol -> calculate_unstable_roots_direct(D_showcase, p_conv;
        n_roots_to_track = 0, ω_max = WMAX_CONV, reltol = tol, abstol = tol,
        solver = BS3(), n_power_max = n_pow, maxiters = 10^9)[2]),
    ("Rosenbrock23", tol -> calculate_unstable_roots_direct(D_showcase, p_conv;
        n_roots_to_track = 0, ω_max = WMAX_CONV, reltol = tol, abstol = tol,
        solver = Rosenbrock23(), n_power_max = n_pow, maxiters = 10^9)[2]),
    ("QuadGK", tol -> calculate_unstable_roots_quadgk(D_showcase, p_conv;
        ω_max = WMAX_CONV, reltol = tol, abstol = tol, n_power_max = n_pow)[2]),
]

# A single evaluation of a low-order pair at a tight tolerance over this
# frequency range can take a minute, so each method stops escalating once one
# point exceeds the cap. The dropped points are logged rather than silently
# omitted, and the curves simply end where the method becomes impractical --
# which is itself the relevant information.
const T_CAP = 20.0

data = with_cache("s03_convergence_v3") do
    out = Dict{String, Vector{Tuple{Float64, Float64, Float64}}}()
    for (name, runner) in METHODS
        rows = Tuple{Float64, Float64, Float64}[]
        for tol in TOLS
            t = time_point(() -> runner(tol))
            Zraw = runner(tol)
            err = abs(Zraw - Z_EXACT)          # reference-free: exact integer
            push!(rows, (tol, err, t))
            @info "convergence" name tol err t
            if t > T_CAP
                @warn "stopping tolerance sweep: cost cap exceeded" name tol t T_CAP
                break
            end
        end
        out[name] = rows
    end
    out
end

# ---------------------------------------------------------------------------
# Figure: (a) error vs tol, (b) CPU vs tol, (c) efficiency front
#         -- error is on the VERTICAL axis in both (a) and (c)
# ---------------------------------------------------------------------------
fig = Figure(size = (W_FULL, W_FULL * 0.32))
axa = MAxis(fig[1, 1], xlabel = "requested tolerance", ylabel = "count error |Z̃ - Z|",
    xscale = log10, yscale = log10, title = "(a) accuracy")
axb = MAxis(fig[1, 2], xlabel = "requested tolerance", ylabel = "CPU time [s]",
    xscale = log10, yscale = log10, title = "(b) cost")
axc = MAxis(fig[1, 3], xlabel = "CPU time [s]", ylabel = "count error |Z̃ - Z|",
    xscale = log10, yscale = log10, title = "(c) efficiency front")
cols = Makie.wong_colors()
for (i, (name, _)) in enumerate(METHODS)
    rows = data[name]
    tols = getindex.(rows, 1)
    errs = max.(getindex.(rows, 2), 1e-16)
    ts = getindex.(rows, 3)
    scatterlines!(axa, tols, errs; color = cols[i], label = name, markersize = 5)
    scatterlines!(axb, tols, ts; color = cols[i], label = name, markersize = 5)
    scatterlines!(axc, ts, errs; color = cols[i], label = name, markersize = 5)
end
lines!(axa, TOLS, 100 .* TOLS; color = :black, linestyle = :dot)
text!(axa, 1e-6, 3e-4; text = "100·tol", color = :black, fontsize = 7)
axislegend(axa; position = :lt, labelsize = 6)
save_fig(fig, "fig_convergence")

# ---------------------------------------------------------------------------
# Near-boundary stress: approach a Hopf boundary geometrically.
# The peak half-width equals |σ_est|, so this is where a marching solver can
# step over the peak and silently lose exactly one root.
# ---------------------------------------------------------------------------
stress = with_cache("s03_boundary_stress") do
    sig(P) = calculate_unstable_roots_direct(D_showcase, (P, 1.5); ω_max = WMAX_CONV,
        reltol = 1e-10, abstol = 1e-10, n_power_max = n_pow,
        refinement_method = :Newton, refinement_steps = 15)[4]
    lo, hi = 2.0, 3.0
    for _ in 1:60
        mid = (lo + hi) / 2
        (sig(mid) < 0) ? (lo = mid) : (hi = mid)
    end
    P_b = (lo + hi) / 2
    @info "Hopf boundary located" P_b
    rows = Tuple[]
    for dP in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
        zo = calculate_unstable_roots_direct(D_showcase, (P_b + dP, 1.5); ω_max = WMAX_CONV,
            reltol = 1e-8, abstol = 1e-8, n_power_max = n_pow)
        zg = calculate_unstable_roots_quadgk(D_showcase, (P_b + dP, 1.5); ω_max = WMAX_CONV,
            reltol = 1e-8, abstol = 1e-8, n_power_max = n_pow)
        push!(rows, (dP, zo[4], zo[1], zg[1], abs(zo[2] - round(zo[2])),
            abs(zg[2] - round(zg[2]))))
        @info "boundary stress" dP Z_ode = zo[1] Z_gk = zg[1] sigma = zo[4]
    end
    (P_b = P_b, rows = rows)
end
write_csv("boundary_stress", ["dP", "sigma_est", "Z_ode", "Z_gk", "res_ode", "res_gk"],
    stress.rows)
write_booktabs("tab_boundary_stress", "lccccc",
    ["\$\\Delta P\$", "\$\\sest\$ (peak half-width)", "\$\\Zint\$ march",
     "\$\\Zint\$ quadrature", "\$\\varepsilon\$ march", "\$\\varepsilon\$ quadrature"],
    [[tex_sci(r[1]; digits = 0), tex_sci(r[2]), string(r[3]), string(r[4]),
      tex_sci(r[5]), tex_sci(r[6])] for r in stress.rows])

# ---------------------------------------------------------------------------
# Tables + CSV
# ---------------------------------------------------------------------------
rows_csv = Tuple[]
for (name, _) in METHODS, (tol, err, t) in data[name]
    push!(rows_csv, (name, tol, err, t))
end
write_csv("convergence", ["method", "tol", "err", "time_s"], rows_csv)

sel_tols = [1e-3, 1e-5, 1e-7, 1e-9]
rows_tex = Vector{String}[]
for tol in sel_tols
    row = ["\$10^{$(round(Int, log10(tol)))}\$"]
    for (name, _) in METHODS          # error block
        idx = findfirst(r -> r[1] == tol, data[name])
        push!(row, idx === nothing ? "--" : tex_sci(data[name][idx][2]))
    end
    for (name, _) in METHODS          # time block
        idx = findfirst(r -> r[1] == tol, data[name])
        push!(row, idx === nothing ? "--" : tex_time(data[name][idx][3]))
    end
    push!(rows_tex, row)
end
write_booktabs("tab_convergence", "l" * "c"^(2 * length(METHODS)),
    vcat(["tol"], ["\\multicolumn{1}{c}{$(m[1])}" for m in METHODS],
         ["\\multicolumn{1}{c}{$(m[1])}" for m in METHODS]),
    rows_tex)
