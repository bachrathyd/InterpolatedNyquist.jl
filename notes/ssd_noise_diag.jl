# Diagnose the factor 4 between SSD and the frequency method in the PD variance:
# additive-only (σm = 0), one shared noise ID, and the README's Hayes example.
#   julia notes/ssd_noise_diag.jl
using Pkg
Pkg.activate(; temp = true)
Pkg.develop([PackageSpec(path = joinpath(@__DIR__, "..")),
             PackageSpec(path = joinpath(@__DIR__, "..", "..", "StochasticSemiDiscretizationMethod.jl"))])
Pkg.add(["StaticArrays", "QuadGK"])
using StochasticSemiDiscretizationMethod, StaticArrays, Printf, QuadGK

function var_ssd(P, D, τ, ζ, σm, σa; shared = false, r = 20)
    AMx = ProportionalMX(@SMatrix [0.0 1.0; -1.0 -2ζ])
    BMx = DelayMX(τ, @SMatrix [0.0 0.0; -P -D])
    α1 = stCoeffMX(1, ProportionalMX(@SMatrix [0.0 0.0; 0.0 0.0]))
    β1 = stCoeffMX(1, DelayMX(τ, @SMatrix [0.0 0.0; -σm*P -σm*D]))
    σv = stAdditive(shared ? 1 : 2, Additive(@SVector [0.0, σa]))
    prob = LDDEProblem(AMx, [BMx], [α1], [β1], Additive(2), [σv])
    mp = DiscreteMapping_M2(prob, SemiDiscretization(1, τ / r), τ, n_steps = 1, calculate_additive = true)
    M2 = VecToCovMx(fixPointOfMapping(mp), length(mp.M1_Vs[1]))
    return M2[1, 1]
end
# exact additive-only variance: σa² (1/2π)∫|1/D(iω)|² dω, D = -ω² + 2iζω + 1 + (P + iωD)e^{-iωτ}
function var_exact(P, D, τ, ζ, σm, σa)
    Df(w) = -w^2 + 2im * ζ * w + 1 + (P + im * w * D) * exp(-im * w * τ)
    Gf(w) = σm * (P + im * w * D) * exp(-im * w * τ) / Df(w)
    iy = quadgk(w -> abs2(1 / Df(w)), 0, 5, 20, Inf; rtol = 1e-10)[1] / π
    ig = quadgk(w -> abs2(Gf(w)), 0, 5, 20, Inf; rtol = 1e-10)[1] / π
    return σa^2 * iy / (1 - ig)
end
for (lab, p) in [("σm = 0 (additive only)", (0.5, 0.4, 0.5, 0.05, 0.0, 0.1)),
                 ("σm = 0.6", (0.5, 0.4, 0.5, 0.05, 0.6, 0.1))]
    v2 = var_ssd(p...; shared = false); v1 = var_ssd(p...; shared = true); ve = var_exact(p...)
    @printf("%-24s exact %.6f | SSD two noise IDs %.6f (×%.3f) | SSD one noise ID %.6f (×%.3f)\n",
            lab, ve, v2, v2 / ve, v1, v1 / ve)
end
# Hayes (README): dx = a x dt + (β x(t-1) + 1) dW, a = -6, β = 2: exact 1/8
AMx = ProportionalMX(-6.0 * ones(1, 1)); BMx = DelayMX(1.0, zeros(1, 1))
prob = LDDEProblem(AMx, [BMx], [stCoeffMX(1, ProportionalMX(zeros(1, 1)))],
                   [stCoeffMX(1, DelayMX(1.0, 2.0 * ones(1, 1)))], Additive(1), [stAdditive(1, Additive(ones(1)))])
mp = DiscreteMapping_M2(prob, SemiDiscretization(0, 0.1), 1.0, n_steps = 10, calculate_additive = true)
@printf("Hayes a=-6 β=2: SSD %.6f, frequency formula 1/8 = 0.125\n", VecToCovMx(fixPointOfMapping(mp), length(mp.M1_Vs[1]))[1, 1])
