# Study s07: comparison with the (multiplication-free) semi-discretization
# method (SemiDiscretizationMethod.jl) on the reduced showcase system.
# Output is a TABLE only -- the two methods' error controls are not
# commensurable (steps-per-delay vs. requested tolerance), so plotting them on
# shared axes invites a false comparison.
# Produces: tables/tab_semidisc.tex, data/semidisc.csv

include(joinpath(@__DIR__, "studies_common.jl"))
include(joinpath(@__DIR__, "systems.jl"))
using SemiDiscretizationMethod

function showcase_AB(p)
    P, Dg = p
    m23 = SM.m2 + SM.m3
    K = [SM.k1 + SM.k2 -SM.k2; -SM.k2 SM.k2]
    C = [SM.c1 + SM.c2 -SM.c2; -SM.c2 SM.c2]
    Minv = [1 / SM.m1 0.0; 0.0 1 / m23]
    A = [zeros(2, 2) I; -Minv*K -Minv*C]
    B = zeros(4, 4)
    B[3, 1] = -P / SM.m1
    B[3, 3] = -Dg / SM.m1
    return A, B
end

function sd_dominant_exponent(p; n = 40, order = 1)
    for n_try in (n, n + 1, n + 2)      # ARPACK occasionally fails to converge
        try
            A, B = showcase_AB(p)
            Δt = SM.tau / n_try
            lddep = LDDEProblem(ProportionalMX(A), [DelayMX(t -> SM.tau, B)], Additive(zeros(4)))
            mapping = DiscreteMapping_LR(lddep, SemiDiscretization(order, Δt), SM.tau;
                n_steps = 1, calculate_additive = true)
            return log(spectralRadiusOfMapping(mapping)) / Δt
        catch err
            @warn "semi-discretization eigensolve failed, retrying" n_try err
        end
    end
    return NaN
end

# The dominant root of the present method: track several |D| minima, refine all,
# take the maximal real part (a single tracked minimum may belong to a
# non-dominant branch away from the stability boundary).
function our_dominant(p; tol = 1e-5, refinement = :Newton, steps = 4)
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_showcase_reduced, p;
        ω_max = 1e4, reltol = tol, abstol = tol, n_roots_to_track = 10,
        refinement_method = refinement, refinement_steps = steps)
    return maximum(filter(isfinite, es))
end
true_sigma(p) = our_dominant(p; tol = 1e-9, refinement = :Newton, steps = 15)

POINTS = [("stable", (1.8, 1.0)), ("near boundary", (2.4, 1.4)), ("unstable", (3.2, 0.6))]
NS = [10, 20, 40, 80, 160]

data = with_cache("s07_semidisc_v3") do
    rows = Tuple[]
    for (label, p) in POINTS
        σ_true = true_sigma(p)
        @info "dominant reference" label p σ_true
        for order in (1, 2), n in NS
            t = time_point(() -> sd_dominant_exponent(p; n = n, order = order); samples = 3)
            σ_sd = sd_dominant_exponent(p; n = n, order = order)
            push!(rows, ("SD order $order", label, p[1], p[2], Float64(n), σ_sd,
                abs(σ_sd - σ_true), t))
        end
        for tol in (1e-5, 1e-8)
            t = time_point(() -> our_dominant(p; tol = tol); samples = 3)
            σ_ours = our_dominant(p; tol = tol)
            push!(rows, ("interp. Nyquist tol=$tol", label, p[1], p[2], 0.0, σ_ours,
                abs(σ_ours - σ_true), t))
        end
    end
    rows
end
write_csv("semidisc",
    ["method", "point", "P", "D", "n_steps", "sigma_est", "err", "time_s"], data)

# ---------------------------------------------------------------------------
# Table: a few representative settings per method, at the near-boundary point
# (the one that matters for a stability chart)
# ---------------------------------------------------------------------------
sel(m, n, lbl) = findfirst(r -> r[1] == m && r[5] == n && r[2] == lbl, data)
rows_tex = Vector{String}[]
for lbl in ("stable", "near boundary", "unstable")
    for (m, n, show) in (("SD order 1", 20.0, "SD order 1, \$n=20\$"),
                         ("SD order 1", 160.0, "SD order 1, \$n=160\$"),
                         ("SD order 2", 20.0, "SD order 2, \$n=20\$"),
                         ("SD order 2", 160.0, "SD order 2, \$n=160\$"),
                         ("interp. Nyquist tol=1.0e-5", 0.0, "this work, tol \$10^{-5}\$"),
                         ("interp. Nyquist tol=1.0e-8", 0.0, "this work, tol \$10^{-8}\$"))
        i = sel(m, n, lbl)
        i === nothing && continue
        r = data[i]
        push!(rows_tex, [lbl == "stable" && show == "SD order 1, \$n=20\$" ? lbl :
                         (show == "SD order 1, \$n=20\$" ? lbl : ""),
                         show, isfinite(r[7]) ? tex_sci(r[7]) : "--", tex_time(r[8])])
    end
end
write_booktabs("tab_semidisc", "llcc",
    ["point", "method / setting", "error of \$\\Real\\lam_{\\mathrm{dom}}\$", "time / point"],
    rows_tex)
