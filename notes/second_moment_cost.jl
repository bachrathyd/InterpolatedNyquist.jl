# Cost of the second-moment march vs the deterministic unwrap march (turning GSTM).
#   julia --project=. notes/second_moment_cost.jl
include("second_moment_turning_model.jl")
using Statistics

Dfun = (λ, q) -> turning(λ, q)[1]
med(f) = (f(); median([@elapsed(f()) for _ in 1:300]))

println("== Cost and accuracy per parameter point (median wall time of 300 runs, 1 thread)")
for p in [(0.2, 1.0, 1.0), (0.2, 0.6, 0.7), (0.30, 0.5, 0.5)]
    ev_d = IN._unwrap_march(IN.NyquistWrapper{typeof(p)}(Dfun), p, 0.0, 0; ω_max = 1e5)[3]
    td = med(() -> calculate_unstable_roots_unwrap(Dfun, p; n_roots_to_track = 0, n_power_max = 2))
    ref = stats(p, reference(p))
    @printf("K=%.1f Ω=%.1f σ=%.1f: deterministic %d evals, %.1f µs (%.0f ns/eval)\n",
            p..., ev_d, 1e6td, 1e9td / ev_d)
    for (name, f) in (("1-pass", second_moment_unwrap), ("2-pass", second_moment))
        for rt in (Inf, 1e-2, 1e-3, 1e-4)
            name == "2-pass" && rt == Inf && continue
            Z, I, ev = f(turning, p; n_power_max = 2, rtol_I = rt)
            s = stats(p, I)
            t = med(() -> f(turning, p; n_power_max = 2, rtol_I = rt))
            @printf("   %s rtol_I=%-6s %4d evals %6.1f µs  %5.1f× time %4.1f× evals | rel.err Var %.1e  r0 %.1e\n",
                    name, string(rt), ev, 1e6t, t / td, ev / ev_d, abs(s[3] / ref[3] - 1), abs(s[2] / ref[2] - 1))
        end
    end
end

# where do the nodes go?  (all evaluations, both passes)
function nodes(p, rt, f)
    ws = Float64[]
    m = (λ, q) -> (push!(ws, FD.value(imag(λ))); turning(λ, q))
    f(m, p; n_power_max = 2, rtol_I = rt)
    return ws
end
edges = [0, 0.5, 0.8, 1.2, 2, 5, 20, 100, 1e5]
println("\n== Node distribution for K=0.2 Ω=0.6, ω bands ", edges)
for (name, f) in (("1-pass", second_moment_unwrap), ("2-pass", second_moment)), rt in (Inf, 1e-2, 1e-4)
    name == "2-pass" && rt == Inf && continue
    ws = nodes((0.2, 0.6, 0.7), rt, f)
    println("   $name rtol_I=$(rpad(string(rt), 6)) ", [count(x -> edges[i] <= x < edges[i+1], ws) for i in 1:length(edges)-1])
end
