# Stochastic delay models in DifferentialEquations.jl form -> first moment, mean-square
# stability margin r0, stationary second moment and MS exponent on the unwrap march.
# (Theory: notes/stochastic_second_moment.md, Part II.)
#
#   du = f(u, h, p, t) dt + g(u, h, p, t) dW          (Itô; StochasticDelayDiffEq signature)
#
# f returns an N-vector; g returns an N×m matrix (m Wiener processes, non-diagonal noise)
# or an N-vector (diagonal noise: component i has its own W_i). h(p, t - τ) is the
# history. Nothing about the delays has to be declared: as in `get_D_from_model`, the
# model is evaluated with ForwardDiff duals around the equilibrium and the mock history
# u_eq + a·e^{λt}, so at one λ
#
#   Δ(λ)  = λ·M - ∂f/∂a        (characteristic matrix, every delay as e^{-λτ})
#   Γ_k(λ) = ∂g_k/∂a           (noise modulation α_k + Σ β_ki e^{-λτ_i}, column k of g)
#   γ̃_k   = g_k(u_eq)          (effective additive noise at the equilibrium)
#
# Linear models can be given as matrices instead (`LinearSDDE`); they are wrapped into
# the same f/g form. Nonlinear f, g are linearised at the equilibrium (small noise).

include("second_moment_prototype.jl")
using LinearAlgebra, StaticArrays
const ForwardDiff = FD

struct StochasticDDE{F, G, P, N, M}
    f::F
    g::G
    p::P
    u_eq::SVector{N, Float64}
    mass::M
    m::Int                      # number of Wiener processes
    E::Matrix{Float64}          # N×p orthonormal basis of the noise directions
    diagonal::Bool              # g returns a vector (diagonal noise)
end

_gcols(gu::AbstractVector, k, ::Val{N}) where {N} = SVector{N}(ntuple(i -> i == k ? gu[i] : zero(gu[i]), Val(N)))
_gcols(gu::AbstractMatrix, k, ::Val{N}) where {N} = SVector{N}(ntuple(i -> gu[i, k], Val(N)))

# equilibrium of the drift with constant history: f(u, h ≡ u) = 0 (Newton, ForwardDiff)
function sdde_equilibrium(f, p, u0::SVector{N}; iters = 50) where {N}
    F(u) = f(u, (q, t; idxs = nothing) -> idxs === nothing ? u : u[idxs], p, 0.0)
    u = u0
    for _ in 1:iters
        J = ForwardDiff.jacobian(F, u)
        δ = J \ F(u)
        u = u - δ
        norm(δ) < 1e-13 * max(1, norm(u)) && return u
    end
    return u
end

"""
    StochasticDDE(f, g, p, Val(N); u_eq = nothing, mass_matrix = I)

Wrap an SDDE `du = f(u,h,p,t) dt + g(u,h,p,t) dW`. `u_eq` defaults to the equilibrium
of the drift (Newton from 0). The noise basis `E` spans every additive and
modulation direction of g (sampled at λ = 0, 0.7i, 1.9i).
"""
function StochasticDDE(f, g, p, ::Val{N}; u_eq = nothing, mass_matrix = LinearAlgebra.I) where {N}
    ue = u_eq === nothing ? sdde_equilibrium(f, p, zeros(SVector{N, Float64})) : SVector{N, Float64}(u_eq)
    h0(q, t; idxs = nothing) = idxs === nothing ? ue : ue[idxs]
    g0 = g(ue, h0, p, 0.0)
    diagonal = g0 isa AbstractVector
    m = diagonal ? N : size(g0, 2)
    tmp = StochasticDDE{typeof(f), typeof(g), typeof(p), N, typeof(mass_matrix)}(f, g, p, ue, mass_matrix, m,
                                                                              Matrix{Float64}(I, N, N), diagonal)
    cols = Vector{Float64}[]
    for k in 1:m
        push!(cols, Vector(_gcols(g0, k, Val(N))))
    end
    for λ in (0.0 + 0im, 0.7im, 1.9im)
        _, Γ = sdde_mats(tmp, λ)
        for k in 1:m, j in 1:N
            push!(cols, real.(Γ[k][:, j])); push!(cols, imag.(Γ[k][:, j]))
        end
    end
    S = svd(reduce(hcat, cols))
    tolr = 1e-12 * max(1.0, maximum(S.S; init = 0.0))
    p_ = count(>(tolr), S.S)
    E = S.U[:, 1:max(p_, 1)]
    return StochasticDDE{typeof(f), typeof(g), typeof(p), N, typeof(mass_matrix)}(f, g, p, ue, mass_matrix, m, E, diagonal)
end

"""
    sdde_mats(sys, λ) -> (Δ(λ), (Γ_1(λ), …, Γ_m(λ)))

One evaluation of f and g with inner ForwardDiff duals (one per state) and the mock
history `u_eq + a·e^{λ t}`; λ may itself carry an outer dual (the march's d/dω).
"""
function sdde_mats(sys::StochasticDDE{F, G, P, N}, λ::Complex) where {F, G, P, N}
    T = typeof(real(λ))
    Tag = ForwardDiff.Tag{typeof(sys.f), T}
    a = SVector{N}(ntuple(i -> ForwardDiff.Dual{Tag, T, N}(zero(T),
        ForwardDiff.Partials{N, T}(ntuple(j -> i == j ? one(T) : zero(T), Val(N)))), Val(N)))
    ue = sys.u_eq
    u = ue .+ a
    h(q, t; idxs = nothing) = (e = exp(λ * t);
                               idxs === nothing ? ue .+ a .* e : ue[idxs] .+ a[idxs] .* e)
    du = sys.f(u, h, sys.p, zero(T))
    res = λ .* (sys.mass * a) .- du
    jac(r, j) = complex(ForwardDiff.partials(real(r), j), ForwardDiff.partials(imag(r), j))
    Δ = SMatrix{N, N}(ntuple(k -> jac(res[(k - 1) % N + 1], (k - 1) ÷ N + 1), Val(N * N)))
    gu = sys.g(u, h, sys.p, zero(T))
    Γ = ntuple(kk -> begin
            col = _gcols(gu, kk, Val(N))
            SMatrix{N, N}(ntuple(k -> jac(complex(col[(k - 1) % N + 1]), (k - 1) ÷ N + 1), Val(N * N)))
        end, sys.m)
    return Δ, Γ
end

"""
    LinearSDDE(A, c, γ; B = [], α = [], β = []) -> (f, g)

Direct matrix form: `du = (A u + Σ_i B_i u(t-τ_i) + c) dt + Σ_k (γ_k + α_k u + Σ_i β_ki u(t-τ_i)) dW_k`.
`B = [(B_1, τ_1), …]`; `γ = [γ_1, …]` (one N-vector per Wiener process),
`α = [α_1, …]` (N×N each), `β = [[(β_11, τ_1), …], …]`. Returns f, g for `StochasticDDE`.
"""
function LinearSDDE(A, c, γ; B = (), α = (), β = ())
    m = length(γ)
    f(u, h, p, t) = A * u + c + sum((Bi * h(p, t - τi) for (Bi, τi) in B); init = zero(u))
    g(u, h, p, t) = hcat(ntuple(k -> γ[k] .+ (isempty(α) ? zero(u) : α[k] * u) .+
                         (isempty(β) ? zero(u) : sum((βi * h(p, t - τi) for (βi, τi) in β[k]); init = zero(u))), m)...)
    return f, g
end

# ---------------------------------------------------------------------------
# the march over a generic real integrand vector (phase test as _unwrap_march,
# integral test on the "diagonal" entries, Hermite-corrected trapezoid, power tail)

function _march_general(evalf, n_power_max, diagidx; ω0 = 1e-9, ω_max = 1e5, tol = 0.3,
                        rtol_I = 1e-3, Iscale = nothing, h0 = 1e-2, hrel = 1.0, hmax = Inf,
                        ωband = 0.0, maxsteps = 200_000)
    ω = Float64(ω0)
    D, Dw, fa, fwa = evalf(ω)
    ua, θa, _, sa = IN._uw_sample(D, Dw)
    I = zero(fa)
    sc = Iscale === nothing ? zero(fa) : π .* Iscale
    evals = 1; Φ = 0.0; h = Float64(h0)
    floor_h(x) = 8 * eps(Float64) * max(x, 1.0)
    for _ in 1:maxsteps
        hh = min(h, hrel * max(ω, 1.0))
        ω < ωband && (hh = min(hh, hmax))
        last = hh >= ω_max - ω
        hh = last ? ω_max - ω : hh
        b = last ? Float64(ω_max) : ω + hh
        Db, Dwb, fb, fwb = evalf(b)
        evals += 1
        ub, θb, _, sb = IN._uw_sample(Db, Dwb)
        Δ = IN._uw_dphase(ua, ub)
        valid = isfinite(θb) && sb > 0 && isfinite(Δ)
        err = abs(Δ - hh * (θa + θb) / 2)
        isfinite(err) || (err = Inf)
        errI = 0.0
        for j in diagidx
            e = abs(hh^2 / 12 * (fwa[j] - fwb[j])) /
                max(sc[j], abs(I[j] + hh / 2 * (fa[j] + fb[j])), floatmin(Float64))
            errI = max(errI, isfinite(e) ? e : Inf)
        end
        accept = valid && err <= tol && errI <= rtol_I
        fac = clamp(0.9 * cbrt(min(tol / max(err, floatmin(Float64)), rtol_I / max(errI, floatmin(Float64)))), 0.2, 4.0)
        hn = hh * fac
        if !accept && valid && hn <= floor_h(ω)
            q = IN._uw_newton(Db, Dwb)
            right = imag(q) > 0
            if abs(Δ) > π / 2
                Δ = (right && Δ > 0) ? Δ - 2π : ((!right && Δ < 0) ? Δ + 2π : Δ)
            end
            accept = true; hn = hh
        end
        if accept
            I = I .+ hh / 2 .* (fa .+ fb) .+ hh^2 / 12 .* (fwa .- fwb)
            Φ += Δ; ω = b
            ua, θa, sa, fa, fwa = ub, θb, sb, fb, fwb
            if last
                tail = map((x, xw) -> (x == 0 ? 0.0 : (k = -ω * xw / x; k > 1.05 ? x * ω / (k - 1) : NaN)), fa, fwa)
                return round(Int, n_power_max / 2 - Φ / π), (I .+ tail) ./ π, evals
            end
        elseif hn <= floor_h(ω)
            return -1, fill(NaN, length(I)), evals
        end
        h = hn
    end
    return -1, fill(NaN, length(I)), evals
end

struct SDDETag end

# integrand at λ = s + iω: D, dD/dω and the real parts of the outer products
#   vec(H) vec(H)^*  (H = Δ⁻¹E, N×p)   and   Σ_k vec(G_k) vec(G_k)^*  (G_k = Eᵀ Γ_k H, p×p)
function _sdde_integrand(sys::StochasticDDE{F, G, P, N}, s, ω) where {F, G, P, N}
    d = ForwardDiff.Dual{ForwardDiff.Tag{SDDETag, Float64}}(ω, 1.0)
    Δ, Γ = sdde_mats(sys, s + 1im * d)
    E = sys.E
    Δm = Matrix(Δ)
    D = det(Δm)
    H = Δm \ E
    Gs = [E' * (Matrix(Γk) * H) for Γk in Γ]
    val(z) = ComplexF64(ForwardDiff.value(real(z)), ForwardDiff.value(imag(z)))
    der(z) = ComplexF64(ForwardDiff.partials(real(z), 1), ForwardDiff.partials(imag(z), 1))
    xh = vec(H); xhv = val.(xh); xhd = der.(xh)
    nH = length(xh)
    fH = vec(real.(xhv * xhv')); fHw = vec(real.(xhd * xhv' + xhv * xhd'))
    np2 = length(Gs[1])
    fG = zeros(np2 * np2); fGw = zeros(np2 * np2)
    for Gk in Gs
        x = vec(Gk); xv = val.(x); xd = der.(x)
        fG .+= vec(real.(xv * xv')); fGw .+= vec(real.(xd * xv' + xv * xd'))
    end
    return val(D), der(D), vcat(fH, fG), vcat(fHw, fGw)
end

"""
    stochastic_moments(sys; s = 0.0, rtol_I = 1e-3, n_power_max = rank of the mass matrix, kw...)

Two passes of the march on λ = s + iω (pass 1 = the deterministic march: Z; pass 2
only if Z == 0, integral test scaled by pass 1). Returns a named tuple
`(Z, r0, stable_MS, mean, P, q, L, evals)`: `r0` = spectral radius of the
second-moment loop operator 𝕃 (MS stable ⇔ Z == 0 and r0 < 1), `mean` = u_eq,
`P` = stationary covariance of u (N×N), `q` = noise intensity in the basis E.
"""
function stochastic_moments(sys::StochasticDDE{F, G, P_, N}; s = 0.0, rtol_I = 1e-3,
                            n_power_max = rank(Matrix(sys.mass * Matrix{Float64}(I, N, N))), kw...) where {F, G, P_, N}
    p = size(sys.E, 2)
    nH = N * p; nG = p * p
    diagidx = vcat([(i - 1) * nH + i for i in 1:nH], [nH^2 + (i - 1) * nG + i for i in 1:nG])
    evalf = ω -> _sdde_integrand(sys, s, ω)
    Z, I1, ev1 = _march_general(evalf, n_power_max, diagidx; rtol_I = Inf, kw...)
    if Z != 0
        return (Z = Z, r0 = NaN, stable_MS = false, mean = sys.u_eq, P = fill(NaN, N, N),
                q = fill(NaN, p, p), L = fill(NaN, nG, nG), evals = ev1)
    end
    Z2, Iv, ev2 = _march_general(evalf, n_power_max, diagidx; rtol_I, Iscale = I1, kw...)
    IH = reshape(Iv[1:nH^2], nH, nH)               # ∫ Re(vec(H) vec(H)^*) / 2π
    IG = reshape(Iv[nH^2+1:end], nG, nG)
    idxH(i, c) = (c - 1) * N + i                    # vec(H)[idx] = H[i, c]
    idxG(a, c) = (c - 1) * p + a
    # (𝕃 q)_ab = Σ_cd q_cd ∫ G_ac conj(G_bd)
    L = zeros(nG, nG)
    for a in 1:p, b in 1:p, c in 1:p, d in 1:p
        L[idxG(a, b), idxG(c, d)] = IG[idxG(a, c), idxG(b, d)]
    end
    r0 = maximum(abs, eigvals(L))
    Γ0 = zeros(p, p)
    h0(q, t; idxs = nothing) = idxs === nothing ? sys.u_eq : sys.u_eq[idxs]
    g0 = sys.g(sys.u_eq, h0, sys.p, 0.0)
    for k in 1:sys.m
        σk = sys.E' * Vector(_gcols(g0, k, Val(N)))
        Γ0 .+= σk * σk'
    end
    if r0 < 1
        q = reshape((I - L) \ vec(Γ0), p, p)
        P = [sum(IH[idxH(i, c), idxH(j, d)] * q[c, d] for c in 1:p, d in 1:p) for i in 1:N, j in 1:N]
    else
        q = fill(NaN, p, p); P = fill(NaN, N, N)
    end
    return (Z = Z, r0 = r0, stable_MS = r0 < 1, mean = sys.u_eq, P = P, q = q, L = L, evals = ev1 + ev2)
end

"""
    ms_exponent_sdde(sys; σ_lo = -2.0, σ_hi = 1.0, iters = 40, kw...)

Mean-square Lyapunov exponent λ_MS (bisection on "Z(σ/2) == 0 and r(σ) < 1").
"""
function ms_exponent_sdde(sys; σ_lo = -2.0, σ_hi = 1.0, iters = 40, rtol_I = 1e-4, kw...)
    ok(σ) = (r = stochastic_moments(sys; s = σ / 2, rtol_I, kw...); r.Z == 0 && r.r0 < 1)
    ok(σ_hi) || error("σ_hi too small")
    ok(σ_lo) && error("σ_lo too large")
    for _ in 1:iters
        mid = (σ_lo + σ_hi) / 2
        ok(mid) ? (σ_hi = mid) : (σ_lo = mid)
    end
    return (σ_lo + σ_hi) / 2
end
