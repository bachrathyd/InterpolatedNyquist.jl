# Turning GSTM model, QuadGK reference and helpers (shared by the notes/second_moment_*.jl scripts)
include("second_moment_prototype.jl")
using QuadGK, Printf

const ζ = 0.1
# p = (K, Ω, σn):  D = λ² + 2ζλ + 1 + K(1 - e^{-λτ});  n1 = 1 (force -> displacement);
# n2 = Kσ(e^{-λτ} - 1) (noise-modulating output y_τ - y, scaled by the noise gain)
function turning(λ, p)
    K, Ω, σn = p
    e = exp(-λ * 2π / Ω)
    return (λ^2 + 2ζ * λ + 1 + K * (1 - e), one(λ), K * σn * (e - 1))
end

function reference(p, s = 0.0)
    f(ω, j) = (r = turning(s + 1im * ω, p); abs2(r[j+1] / r[1]))
    br = [0.0; collect(0.05:0.05:60.0); Inf]
    return ntuple(j -> quadgk(ω -> f(ω, j), br...; rtol = 1e-12, maxevals = 10^7)[1] / π, 2)
end

stats(p, I) = (va = (p[1] * p[3])^2 * I[1]; (va, I[2], va / (1 - I[2])))
