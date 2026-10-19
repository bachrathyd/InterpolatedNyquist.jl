# Checks of the SDDE / matrix model input (notes/stochastic_sdde.jl)
#   julia --project=. notes/stochastic_sdde_checks.jl
include("stochastic_sdde.jl")
using Printf, Random, Statistics

println("== 1. Turning GSTM as an SDDE (f, g functions) vs the scalar transfer form")
# x'' + 2ζx' + x = K(1 + x(t-τ) - x) + Kσ(1 + x(t-τ) - x) ξ     (equilibrium x = K)
function f_turn(u, h, p, t)
    K, Ω, σ, ζ = p
    xτ = h(p, t - 2π / Ω)[1]
    return SVector(u[2], -2ζ * u[2] - u[1] + K * (1 + xτ - u[1]))
end
function g_turn(u, h, p, t)            # one Wiener process -> N×1 matrix
    K, Ω, σ, ζ = p
    xτ = h(p, t - 2π / Ω)[1]
    return SMatrix{2, 1}(0.0 * u[1], K * σ * (1 + xτ - u[1]))
end
for (p, Varref, r0ref) in [((0.2, 0.6, 0.7, 0.1), 0.371578, 0.367976), ((0.2, 1.0, 1.0, 0.1), 0.103849, 0.077075)]
    sys = StochasticDDE(f_turn, g_turn, p, Val(2))
    r = stochastic_moments(sys; rtol_I = 1e-4)
    @printf("K=%.1f Ω=%.1f σ=%.1f: u_eq=%s p=%d  Z=%d  r0=%.6f (ref %.6f)  Var x=%.6f (ref %.6f)  evals=%d\n",
            p[1:3]..., round.(sys.u_eq, digits = 6), size(sys.E, 2), r.Z, r.r0, r0ref, r.P[1, 1], Varref, r.evals)
end

println("\n== 2. 2-DOF, 2 noise channels, one W, additive + multiplicative (no delay): exact Lyapunov")
A = [0 0 1 0; 0 0 0 1; -2 1 -0.1 0; 1 -1.5 0 -0.05]
Ein = [0 0; 0 0; 1.0 0; 0 1.0]; C = 0.5 * [0.6 -0.3 0.2 0; 0.1 0.5 0 0.1]; gv = [0.3, -0.2]
f2, g2 = LinearSDDE(A, zeros(4), [Ein * gv]; α = [Ein * C])
sys2 = StochasticDDE((u, h, p, t) -> SVector{4}(f2(u, h, p, t)), (u, h, p, t) -> SMatrix{4, 1}(g2(u, h, p, t)), nothing, Val(4))
r2 = stochastic_moments(sys2; rtol_I = 1e-5)
Lop = kron(I(4), A) + kron(A, I(4)) + kron(Ein * C, Ein * C)
Pex = reshape(Lop \ (-vec(Ein * gv * gv' * Ein')), 4, 4)
@printf("p=%d Z=%d r0=%.4f (exact 0.5323)  max|P - P_exact|/max|P_exact| = %.1e  evals=%d\n",
        size(sys2.E, 2), r2.Z, r2.r0, maximum(abs, r2.P - Pex) / maximum(abs, Pex), r2.evals)

println("\n== 3. Same turning model given as matrices (LinearSDDE) -> identical")
K, Ω, σ, ζ = 0.2, 0.6, 0.7, 0.1; τ = 2π / Ω
fm, gm = LinearSDDE([0 1; -(1 + K) -2ζ], [0.0, K], [[0.0, K * σ]];
                    B = [([0 0; K 0], τ)], α = [[0 0; -K*σ 0]], β = [[([0 0; K*σ 0], τ)]])
sys3 = StochasticDDE((u, h, p, t) -> SVector{2}(fm(u, h, p, t)), (u, h, p, t) -> SMatrix{2, 1}(gm(u, h, p, t)), nothing, Val(2))
r3 = stochastic_moments(sys3; rtol_I = 1e-4)
@printf("u_eq=%s  r0=%.6f  Var x=%.6f\n", round.(sys3.u_eq, digits = 6), r3.r0, r3.P[1, 1])

println("\n== 4. Delayed PD control, additive + multiplicative noise (two independent W), vs Monte Carlo")
# x'' + 2ζx' + x = -(P x(t-τ) + D x'(t-τ))(1 + σ_m ξ₁) + σ_a ξ₂
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
function mc_pd(p; ntraj = 4000, dt = 0.002, T = 1500.0, seed = 1)
    P, D, τ, ζ, σm, σa = p
    rng = MersenneTwister(seed); nd = round(Int, τ / dt); n = round(Int, T / dt)
    hx = zeros(nd, ntraj); hv = zeros(nd, ntraj); x = zeros(ntraj); v = zeros(ntraj)
    acc = 0.0; cnt = 0; sq = sqrt(dt)
    for k in 1:n
        j = mod1(k, nd)
        @inbounds for i in 1:ntraj
            fb = P * hx[j, i] + D * hv[j, i]
            v[i] += (-2ζ * v[i] - x[i] - fb) * dt - σm * fb * sq * randn(rng) + σa * sq * randn(rng)
            x[i] += v[i] * dt
            hx[j, i] = x[i]; hv[j, i] = v[i]
        end
        if k > n ÷ 3 && k % 10 == 0
            acc += mean(abs2, x); cnt += 1
        end
    end
    return acc / cnt
end
for p in [(0.5, 0.4, 0.5, 0.05, 0.6, 0.1), (0.8, 0.5, 0.6, 0.05, 0.9, 0.1)]
    sys = StochasticDDE(f_pd, g_pd, p, Val(2))
    r = stochastic_moments(sys; rtol_I = 1e-4)
    λms = ms_exponent_sdde(sys; σ_lo = -2.0, σ_hi = 1.0)
    @printf("P=%.1f D=%.1f τ=%.1f σm=%.1f σa=%.1f: p=%d Z=%d r0=%.4f Var x=%.6g  λ_MS=%.5f  evals=%d\n",
            p[1], p[2], p[3], p[5], p[6], size(sys.E, 2), r.Z, r.r0, r.P[1, 1], λms, r.evals)
    t = @elapsed m = mean(mc_pd(p; seed = s) for s in 1:2)
    @printf("    Monte Carlo E[x²] = %.6g  (%.0f s)\n", m, t)
end
