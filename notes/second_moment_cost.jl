# Cost of the second-moment march vs the deterministic unwrap march (turning GSTM):
# 1-pass, 2-pass, 2-pass with pole subtraction, and the band step cap, including
# points approaching the stability boundary (sharp stochastic-resonance peaks).
#   julia --project=. notes/second_moment_cost.jl
include("second_moment_turning_model.jl")
using Statistics

Dfun = (λ, q) -> turning(λ, q)[1]
med(f) = (f(); median([@elapsed(f()) for _ in 1:100]))
count0(K, Ω) = calculate_unstable_roots_unwrap(Dfun, (K, Ω, 0.1); n_roots_to_track = 0, n_power_max = 2)[1]

# critical K at Ω = 0.6 (bisection on the deterministic count)
function Kcrit(Ω; lo = 0.0, hi = 1.0)
    for _ in 1:50
        mid = (lo + hi) / 2
        count0(mid, Ω) == 0 ? (lo = mid) : (hi = mid)
    end
    return lo
end
Kc = Kcrit(0.6)
@printf("critical K at Ω = 0.6: %.6f\n\n", Kc)

schemes = [
    ("1-pass",              (m, p; kw...) -> second_moment_unwrap(m, p; kw...)),
    ("2-pass",              (m, p; kw...) -> second_moment(m, p; subtract = false, kw...)),
    ("2-pass + poles",      (m, p; kw...) -> second_moment(m, p; subtract = true, kw...)),
    ("2-pass + poles + cap",(m, p; kw...) -> second_moment(m, p; subtract = true, hmax = p[2] / 4, ωband = 5.0, kw...)),
]

println("== Cost and accuracy per parameter point (median wall time of 100 runs, 1 thread)")
cases = [(0.2, 1.0, 1.0), (0.2, 0.6, 0.7), (0.30, 0.5, 0.5),
         (0.9Kc, 0.6, 0.3), (0.99Kc, 0.6, 0.1), (0.999Kc, 0.6, 0.03)]
summary = Dict{String, Vector{NTuple{3, Float64}}}()
for p in cases
    ev_d = IN._unwrap_march(IN.NyquistWrapper{typeof(p)}(Dfun), p, 0.0, 0; ω_max = 1e5)[3]
    td = med(() -> calculate_unstable_roots_unwrap(Dfun, p; n_roots_to_track = 0, n_power_max = 2))
    ref = stats(p, reference(p))
    @printf("K=%.5f Ω=%.1f σ=%.2f: deterministic %d evals, %.1f µs | ref Var %.6g r0 %.4g\n",
            p..., ev_d, 1e6td, ref[3], ref[2])
    for (name, f) in schemes, rt in (1e-3, 1e-4)
        Z, I, ev = f(turning, p; n_power_max = 2, rtol_I = rt)
        s = stats(p, I)
        t = med(() -> f(turning, p; n_power_max = 2, rtol_I = rt))
        eV = abs(s[3] / ref[3] - 1); er = abs(s[2] / ref[2] - 1)
        push!(get!(summary, "$name $rt", NTuple{3, Float64}[]), (t / td, eV, er))
        @printf("   %-22s rtol_I=%-6s %4d evals %7.1f µs  %5.1f× time | rel.err Var %.1e  r0 %.1e\n",
                name, string(rt), ev, 1e6t, t / td, eV, er)
    end
end

println("\n== Summary over the cases: time ratio (median, max), max rel. error Var / r0")
for (name, _) in schemes, rt in (1e-3, 1e-4)
    r = summary["$name $rt"]
    @printf("   %-22s rtol_I=%-6s time %4.1f× (max %4.1f×)  err Var ≤ %.1e  r0 ≤ %.1e\n", name, string(rt),
            median(first.(r)), maximum(first.(r)), maximum(getindex.(r, 2)), maximum(getindex.(r, 3)))
end
