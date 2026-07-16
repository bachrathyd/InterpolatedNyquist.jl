# Cold-start (time-to-first-X) probe. Run in a FRESH Julia process:
#   julia --project=paper/scripts --startup-file=no -t auto paper/scripts/ttfx_probe.jl <backend>
# where <backend> is one of: direct | quadgk | fixed
# Appends one CSV row to paper/data/ttfx.csv:
#   backend, t_load_s, t_first_point_s, t_first_sweep_s, n_sweep

backend = isempty(ARGS) ? "direct" : ARGS[1]

t_load = @elapsed using InterpolatedNyquist

function D_test(λ::T, p) where T
    P, D = p
    c1 = T(0.03); τ = T(0.5); ζ = T(0.02)
    return c1 * λ^4 + λ^2 + 2ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ)
end

Pv = LinRange(-2.0, 4.0, 40)
Dv = LinRange(-2.0, 5.0, 25)
params = vec([(p, d) for p in Pv, d in Dv])

if backend == "direct"
    t_first = @elapsed calculate_unstable_roots_direct(D_test, (-0.2, 0.5))
    t_sweep = @elapsed calculate_unstable_roots_p_vec(D_test, params)
elseif backend == "quadgk"
    t_first = @elapsed calculate_unstable_roots_quadgk(D_test, (-0.2, 0.5))
    t_sweep = @elapsed calculate_unstable_roots_quadgk_p_vec(D_test, params)
elseif backend == "fixed"
    t_first = @elapsed calculate_unstable_roots_fixed_step(D_test, (-0.2, 0.5))
    t_sweep = @elapsed calculate_unstable_roots_fixed_step_p_vec(D_test, params)
else
    error("unknown backend $backend")
end

csv = joinpath(dirname(@__DIR__), "data", "ttfx.csv")
if !isfile(csv)
    open(csv, "w") do io
        println(io, "backend,t_load_s,t_first_point_s,t_first_sweep_s,n_sweep")
    end
end
open(csv, "a") do io
    println(io, "$backend,$t_load,$t_first,$t_sweep,$(length(params))")
end
println("ttfx[$backend]: load=$(round(t_load; digits=2))s first=$(round(t_first; digits=2))s sweep($(length(params)))=$(round(t_sweep; digits=2))s")
