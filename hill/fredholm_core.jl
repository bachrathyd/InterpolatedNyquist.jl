# Compressed Hill determinant for milling (Test 2): the infinite Hill determinant evaluated
# exactly in the harmonics and reduced to a Q x Q determinant (CPU prototype).
#
# Straight teeth, uniform pitch, tooth period T = 2π/ω (ω = zΩ), delay τ = T:
#   A(λ) = D(λ) + α(λ) H,  D = diag p(λ + ikω),  p(s) = s² + 2ζs + 1,  α = w(1 - e^{-λT}),
#   H_kl = h_{k-l}: Toeplitz matrix of the cutting function h(ψ), ψ = ωt ∈ [0, 2π).
# h vanishes outside the cutting window(s); there it is smooth. Gauss quadrature of the
# Fourier integral h_m = (1/2π)∫ h(ψ) e^{-imψ} dψ on the window(s) factors H ≈ U Vᴴ (rank Q):
#   U_kq = c_q e^{-ikψ_q},  Vᴴ_ql = e^{ilψ_q},  c_q = W_q h(ψ_q)/2π.
# Matrix determinant lemma:  det(I + α D⁻¹ U Vᴴ) = det(I_Q + α Vᴴ D⁻¹ U), and
#   (Vᴴ D⁻¹ U)_pq = c_q S(ψ_p - ψ_q),   S(Δ) = Σ_{k∈Z} e^{ikΔ} / p(λ + ikω)
# summed over ALL harmonics in closed form (partial fractions of 1/p):
#   Σ_k e^{ikΔ}/(a + ikω) = T e^{-aΔ/ω} / (1 - e^{-aT}),  Δ ∈ [0, 2π)
# With the row scaling r_k = (λ + ikω + c)² (pole-free, see hill_core.jl):
#   F(λ) = G(λ) · det(I_Q + α M(λ)),  G = Π_k p(s_k)/r_k (closed form, sinh ratio).
# F depends on λ only through the Floquet multiplier z = e^{λT}; F(z̄) = conj F(z), so the
# winding along the unit circle |z| = 1 is twice that along its upper half:
#   Z = -(1/π) Δarg F(iy),  y ∈ [0, ω/2].
# The only approximation is the quadrature (Q nodes); there is no truncation in harmonics.

include(joinpath(@__DIR__, "milling_core.jl"))

lsinh_(x) = real(x) >= 0 ? x + log((1 - exp(-2x)) / 2) : -x + log((exp(2x) - 1) / 2)

# Gauss-Legendre nodes/weights on [a, b]
function gl_nodes(Q, a, b)
    x, wgl = gauss(Q)
    return (b - a) / 2 .* x .+ (a + b) / 2, (b - a) / 2 .* wgl
end

# nodes ψ_q and coefficients c_q on the cutting window(s) of one tooth period
function window_nodes(m::Mill, Q::Int)
    φen, φex = window(m)
    z = m.z
    # tooth phase ψ ∈ [0, 2π): the spindle angle of the cutting tooth is φ = φen + ψ/z (ψ from entry)
    # (pieces: one per tooth whose window intersects the tooth period; for φex - φen ≤ 2π/z one piece)
    L = z * (φex - φen)                         # window length in ψ
    pieces = Tuple{Float64, Float64, Float64}[]  # (ψ start, ψ end, φ offset)
    s = 0.0
    while s < L - 1e-12
        e = min(L, s + 2π)
        push!(pieces, (s, e, 0.0))
        s = e
    end
    ψ = Float64[]; c = Float64[]
    for (a, b, _) in pieces
        x, wq = gl_nodes(Q, a, b)
        for (xi, wi) in zip(x, wq)
            push!(ψ, mod(xi, 2π)); push!(c, wi * fcut(φen + xi / z, m) / 2π)
        end
    end
    return ψ, c
end

"F(λ) for the point (Ω, a_p); `nodes = window_nodes(m, Q)`; cs: row-scale shift."
function fredholm_det(λ, m::Mill, Ω, ap, nodes; cs = 1.0)
    ψ, cq = nodes
    Q = length(ψ)
    ω = m.z * Ω
    T = 2π / ω
    w = m.w1 * ap
    ζ = m.ζ
    r1 = complex(-ζ, sqrt(1 - ζ^2)); r2 = conj(r1)
    a1, a2 = λ - r1, λ - r2
    e1, e2 = exp(-a1 * T), exp(-a2 * T)
    k1 = T / (r1 - r2) / (1 - e1)
    k2 = T / (r1 - r2) / (1 - e2)
    α = w * (1 - exp(-λ * T))
    M = Matrix{typeof(λ)}(undef, Q, Q)
    for q in 1:Q, p in 1:Q
        Δ = ψ[p] - ψ[q]
        Δ < 0 && (Δ += 2π)
        S = k1 * exp(-a1 * Δ / ω) - k2 * exp(-a2 * Δ / ω)
        M[p, q] = (p == q ? 1 : 0) + α * cq[q] * S
    end
    dM = lu_det!(M, Q)
    lG = lsinh_(π * (λ - r1) / ω) + lsinh_(π * (λ - r2) / ω) - 2 * lsinh_(π * (λ + cs) / ω)
    return exp(lG) * dM
end

"Unstable Floquet exponents (multipliers |μ| > 1) by the compressed determinant."
function fredholm_count(m::Mill, Ω, ap; Q = 8, tol = 0.3, nodes = window_nodes(m, Q))
    ω = m.z * Ω
    f = IN.NyquistWrapper{NTuple{1, Float64}}((μ, p) -> fredholm_det(μ * ω, m, Ω, ap, nodes))
    Φ, ok, ev, dd, ds, dw = IN._unwrap_march(f, (0.0,), 0.0, 1; ω0 = 1e-9, ω_max = 0.5, tol = tol,
        h0 = 1e-3, hrel = 0.05)
    return (Z = ok ? round(Int, -Φ / π) : -1, Zraw = -Φ / π, evals = ev, σ = ds[1] * ω, θ = mod(dw[1], 1.0))
end
