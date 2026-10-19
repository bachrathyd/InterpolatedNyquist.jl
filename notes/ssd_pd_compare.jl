# Delayed PD control with noisy feedback gain and force noise (the web example `spd`):
# stochastic semi-discretization (StochasticSemiDiscretizationMethod.jl, Sykora, Bachrathy & Stépán
# 2019) vs the frequency-domain second moment on the unwrap march (notes/stochastic_sdde.jl).
#
#   x'' + 2ζx' + x = -(P x(t-τ) + D x'(t-τ))(1 + σm ξ₁) + σa ξ₂        (Itô)
#
# Run (a throw-away environment with both packages):
#   julia notes/ssd_pd_compare.jl
using Pkg
Pkg.activate(; temp = true)
Pkg.develop([PackageSpec(path = joinpath(@__DIR__, "..")),
             PackageSpec(path = joinpath(@__DIR__, "..", "..", "StochasticSemiDiscretizationMethod.jl"))])
Pkg.add(["StaticArrays", "QuadGK"])
include(joinpath(@__DIR__, "stochastic_sdde.jl"))
using StochasticSemiDiscretizationMethod
using Printf, Statistics

# ---- SSD model ------------------------------------------------------------------------------
function ssd_problem(P, D, τ, ζ, σm, σa)
    AMx = ProportionalMX(@SMatrix [0.0 1.0; -1.0 -2ζ])
    BMx = DelayMX(τ, @SMatrix [0.0 0.0; -P -D])
    cVec = Additive(2)
    α1 = stCoeffMX(1, ProportionalMX(@SMatrix [0.0 0.0; 0.0 0.0]))
    β1 = stCoeffMX(1, DelayMX(τ, @SMatrix [0.0 0.0; -σm*P -σm*D]))
    # one noise ID for both terms: equivalent here (the mean is 0, so E[(γ + βx_τ)²] = γ² + E[(βx_τ)²]);
    # with the additive term on noise ID 2 the package returns 4× the variance (notes/ssd_noise_diag.jl)
    σ2 = stAdditive(1, Additive(@SVector [0.0, σa]))
    return LDDEProblem(AMx, [BMx], [α1], [β1], cVec, [σ2])
end
function ssd(p; r = 20, order = 1)
    P, D, τ, ζ, σm, σa = p
    prob = ssd_problem(P, D, τ, ζ, σm, σa)
    Δt = τ / r
    method = SemiDiscretization(order, Δt)
    mp = DiscreteMapping_M2(prob, method, τ, n_steps = 1, calculate_additive = true)
    ρ = spectralRadiusOfMapping(mp)
    M2 = VecToCovMx(fixPointOfMapping(mp), length(mp.M1_Vs[1]))
    return (Var = M2[1, 1], ρ = ρ, λ = log(ρ) / Δt)
end

# ---- frequency-domain model -----------------------------------------------------------------
function f_pd(u, h, p, t)
    P, D, τ, ζ, σm, σa = p
    uτ = h(p, t - τ)
    return SVector(u[2], -2ζ * u[2] - u[1] - (P * uτ[1] + D * uτ[2]))
end
function g_pd(u, h, p, t)
    P, D, τ, ζ, σm, σa = p
    uτ = h(p, t - τ)
    z = zero(u[1])
    return SMatrix{2, 2}(z, -σm * (P * uτ[1] + D * uτ[2]), z, σa + z)
end

points = [
    (0.5, 0.4, 0.5, 0.05, 0.6, 0.1),     # the web example's default
    (0.8, 0.5, 0.6, 0.05, 0.9, 0.1),     # MS unstable, mean stable
    (1.2, 0.2, 0.5, 0.05, 0.3, 0.1),
    (0.2, 0.8, 1.0, 0.05, 0.4, 0.1),
    (1.5, 0.9, 0.3, 0.1, 0.2, 0.2),
]
med(f; n = 5) = (f(); median([@elapsed(f()) for _ in 1:n]))

println("== frequency domain (unwrap march, rtol_I = 1e-4) ==")
freq = Dict()
for p in points
    sys = StochasticDDE(f_pd, g_pd, p, Val(2))
    r = stochastic_moments(sys; rtol_I = 1e-4)
    λ = ms_exponent_sdde(sys; σ_lo = -3.0, σ_hi = 2.0, iters = 30)
    t_m = med(() -> stochastic_moments(sys; rtol_I = 1e-4); n = 20)
    t_m3 = med(() -> stochastic_moments(sys; rtol_I = 1e-3); n = 20)
    t_λ = med(() -> ms_exponent_sdde(sys; σ_lo = -3.0, σ_hi = 2.0, iters = 30); n = 3)
    freq[p] = (r, λ)
    @printf("P=%.1f D=%.1f τ=%.1f σm=%.1f: Z=%d r0=%.5f Var=%s λ_MS=%.5f | %d evals, %.2f ms (rtol 1e-4), %.2f ms (1e-3), λ_MS %.0f ms\n",
            p[1], p[2], p[3], p[5], r.Z, r.r0, r.r0 < 1 ? @sprintf("%.6f", r.P[1, 1]) : "MS-unstable", λ, r.evals,
            1e3t_m, 1e3t_m3, 1e3t_λ)
end

println("\n== stochastic semi-discretization (1st order), convergence in r = τ/Δt ==")
for p in points
    r, λf = freq[p]
    @printf("P=%.1f D=%.1f τ=%.1f σm=%.1f   [freq: Var %s, λ_MS %.5f]\n", p[1], p[2], p[3], p[5],
            r.r0 < 1 ? @sprintf("%.6f", r.P[1, 1]) : "MS-unstable", λf)
    for rr in (10, 20, 40, 80)
        s = nothing
        t = @elapsed (s = try ssd(p; r = rr) catch e; nothing end)
        s === nothing && (println("   r=$rr: SSD failed (Arpack: spectral radius did not converge)"); continue)
        if rr == 10
            t = med(() -> ssd(p; r = rr); n = 3)
        end
        eV = r.r0 < 1 ? abs(s.Var / r.P[1, 1] - 1) : NaN
        @printf("   r=%4d: Var %s (rel.diff %.1e)  ρ=%.6f  λ=log ρ/Δt=%.5f (diff %.1e)  %.3f s\n", rr,
                s.ρ < 1 ? @sprintf("%.6f", s.Var) : "unstable", eV, s.ρ, s.λ, abs(s.λ - λf), t)
    end
end
