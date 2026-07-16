# Study s07: comparison with the (multiplication-free) semi-discretization
# method (SemiDiscretizationMethod.jl) on the reduced showcase system.
# Output is a TABLE only -- the two methods' error controls are not
# commensurable (steps-per-delay vs. requested tolerance), so plotting them on
# shared axes invites a false comparison.
# Produces: tables/tab_semidisc.tex, data/semidisc.csv

include(joinpath(@__DIR__, "studies_common.jl"))
include(joinpath(@__DIR__, "systems.jl"))
using SemiDiscretizationMethod

function sd_mapping(p; n = 40, order = 1)
    A, B = showcase_AB(p)
    Δt = SM.tau / n
    lddep = LDDEProblem(ProportionalMX(A), [DelayMX(t -> SM.tau, B)], Additive(zeros(4)))
    return DiscreteMapping_LR(lddep, SemiDiscretization(order, Δt), SM.tau;
        n_steps = 1, calculate_additive = true), Δt
end

function sd_dominant_exponent(p; n = 40, order = 1)
    mapping, Δt = sd_mapping(p; n = n, order = order)
    ρ = try
        spectralRadiusOfMapping(mapping)   # sparse ARPACK path
    catch err
        # ARPACK non-convergence is routine at the largest n. The mapping is
        # only (n+1)*4 wide, so fall back to the dense generalized eigenvalue
        # problem R x = mu L x -- exact and still sub-second at n = 160.
        maximum(abs, eigen(collect(mapping.RmappingMX),
                           collect(mapping.LmappingMX)).values)
    end
    return log(ρ) / Δt
end

# The dominant root of the present method: track several |D| minima, refine all,
# take the maximal real part (a single tracked minimum may belong to a
# non-dominant branch away from the stability boundary).
function our_dominant_root(p; tol = 1e-5, refinement = :Newton, steps = 4)
    zi, zr, md, es, wc = calculate_unstable_roots_direct(D_showcase_reduced, p;
        ω_max = 1e4, reltol = tol, abstol = tol, n_roots_to_track = 10,
        refinement_method = refinement, refinement_steps = steps)
    good = findall(isfinite, es)
    i = good[argmax(view(es, good))]
    return complex(es[i], wc[i])
end
our_dominant(p; kw...) = real(our_dominant_root(p; kw...))

# Reference: the Newton-polished root of the quasi-polynomial itself, with its
# defect |D(lambda_ref)| recorded so the reference is verifiable independently
# of either method (de-circularization: the error columns measure distance to
# a root of D certified by this residual, not agreement with "our" sweep).
function true_sigma_with_residual(p)
    λ_ref = our_dominant_root(p; tol = 1e-9, refinement = :Newton, steps = 15)
    return real(λ_ref), abs(D_showcase_reduced(λ_ref, p))
end

POINTS = [("stable", (1.8, 1.0)), ("near boundary", (2.4, 1.4)), ("unstable", (3.2, 0.6))]
NS = [10, 20, 40, 80, 160]

data = with_cache("s07_semidisc_v4") do
    rows = Tuple[]
    for (label, p) in POINTS
        σ_true, res_ref = true_sigma_with_residual(p)
        @info "dominant reference" label p σ_true res_ref
        for order in (1, 2), n in NS
            t = time_point(() -> sd_dominant_exponent(p; n = n, order = order); samples = 3)
            σ_sd = sd_dominant_exponent(p; n = n, order = order)
            push!(rows, ("SD order $order", label, p[1], p[2], Float64(n), σ_sd,
                abs(σ_sd - σ_true), t, res_ref))
        end
        for tol in (1e-5, 1e-8)
            t = time_point(() -> our_dominant(p; tol = tol); samples = 3)
            σ_ours = our_dominant(p; tol = tol)
            push!(rows, ("interp. Nyquist tol=$tol", label, p[1], p[2], 0.0, σ_ours,
                abs(σ_ours - σ_true), t, res_ref))
            # the same sweep WITHOUT the Newton polish: the honest accuracy of
            # the tracked estimate itself (the polished rows share machinery
            # with the reference, so their error mostly reflects the polish)
            t_u = time_point(() -> our_dominant(p; tol = tol, refinement = :Linear); samples = 3)
            σ_u = our_dominant(p; tol = tol, refinement = :Linear)
            push!(rows, ("interp. Nyquist tol=$tol unrefined", label, p[1], p[2], 0.0, σ_u,
                abs(σ_u - σ_true), t_u, res_ref))
        end
    end
    rows
end
write_csv("semidisc",
    ["method", "point", "P", "D", "n_steps", "sigma_est", "err", "time_s", "ref_residual"], data)

# ---------------------------------------------------------------------------
# Table: a few representative settings per method, at the near-boundary point
# (the one that matters for a stability chart)
# ---------------------------------------------------------------------------
sel(m, n, lbl) = findfirst(r -> r[1] == m && r[5] == n && r[2] == lbl, data)
ROW_SPECS = (("SD order 1", 20.0, "SD order 1, \$n=20\$"),
             ("SD order 1", 160.0, "SD order 1, \$n=160\$"),
             ("SD order 2", 20.0, "SD order 2, \$n=20\$"),
             ("SD order 2", 160.0, "SD order 2, \$n=160\$"),
             ("interp. Nyquist tol=1.0e-5 unrefined", 0.0, "this work, tol \$10^{-5}\$, unrefined"),
             ("interp. Nyquist tol=1.0e-5", 0.0, "this work, tol \$10^{-5}\$"),
             ("interp. Nyquist tol=1.0e-8", 0.0, "this work, tol \$10^{-8}\$"))
rows_tex = Vector{String}[]
for lbl in ("stable", "near boundary", "unstable")
    first_of_block = true
    for (m, n, show) in ROW_SPECS
        i = sel(m, n, lbl)
        i === nothing && continue
        r = data[i]
        push!(rows_tex, [first_of_block ? lbl : "",
                         show, isfinite(r[7]) ? tex_sci(r[7]) : "--", tex_time(r[8])])
        first_of_block = false
    end
end
write_booktabs("tab_semidisc", "llcc",
    ["point", "method / setting", "error of \$\\Real\\lam_{\\mathrm{dom}}\$", "time / point"],
    rows_tex)
