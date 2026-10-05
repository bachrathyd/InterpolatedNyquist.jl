# Test 1 on the GPU: delayed Mathieu chart by the pole-free Hill determinant + discrete
# phase unwrapping, one GPU thread per parameter point (NyquistGPU kernel, unchanged).
#   julia --project=gpu/scripts hill/gpu_hill.jl [--cpu] [--res 1920x1080] [--check 160x80]
#
# The kernel's counting formula is Z_raw = n/2 - Φ/π over [ω0, ω_max]; with n = 0 and
# the strip segment ω ∈ [a, a+1]ω_p it returns -Φ/π = 2 Z_strip, so Z = Z_raw / 2.
# N (harmonics -N..N) is derived per point inside D from the single tolerance.
include(joinpath(@__DIR__, "..", "gpu", "scripts", "common.jl"))

# c = (κ, ε, τ, ω_p, tol): pole-free determinant, τ ω_p = 2π (closed-form diagonal tail)
function D_hill(λ, p, c)
    δ, b = p
    κ, ε, τ, ωp, tol = c
    one_ = one(real(λ))
    N = unsafe_trunc(Int32, sqrt(max(δ, zero(δ)) + ε / (2 * sqrt(tol))) / ωp) + Int32(2)
    e2 = (ε / 2)^2
    B = δ - b * exp(-τ * λ)                       # d_k = s_k² + κ s_k + B, s_k = λ + ikω_p
    cc = ωp                                       # row scale r_k = (s_k + c)²
    s = λ - Complex(zero(ωp), N * ωp)
    rprev = (s + cc)^2
    dr = (s * s + κ * s + B) / rprev
    f = dr
    fprev = one(f)
    P = dr
    k = -N + Int32(1)
    while k <= N
        s = λ + Complex(zero(ωp), k * ωp)
        r = (s + cc)^2
        dr = (s * s + κ * s + B) / r
        fn = dr * f - e2 / (r * rprev) * fprev
        fprev = f
        f = fn
        P *= dr
        rprev = r
        k += Int32(1)
    end
    sq = sqrt(κ * κ - 4 * B)
    z1 = (-κ + sq) / 2
    z2 = (-κ - sq) / 2
    w = π * one_ / ωp
    G = sinh(w * (λ - z1)) * sinh(w * (λ - z2)) / sinh(w * (λ + cc))^2
    return f * (G / P)
end

const A_STRIP = 0.237
const CM = (0.1, 1.0, 2π, 1.0, 1e-4)             # κ, ε, τ = T = 2π, ω_p = 1, tol
kw_hill(T) = (n_power = 0, ω0 = A_STRIP, ω_max = A_STRIP + 1, h0 = 1e-3, hrel = 0.05, nroots = 1, T = T)

nx, ny = parse_res(arg("res", "1920x1080"))
cx, cy = parse_res(arg("check", "160x80"))
print_device()
δr, br = (-1.0, 5.0), (-1.5, 1.5)

# 1. correctness: GPU (Float32/Float64) vs the same kernel on the CPU backend in Float64
cpts = grid_points(range(δr...; length = cx), range(br...; length = cy))
ref = sweep(D_hill, cpts; c = CM, backend = CPU(), kw_hill(Float64)...)
Zref = round.(Int, ref.Zraw ./ 2)
for T in (Float64, Float32)
    r = sweep(D_hill, cpts; c = CM, backend = BACKEND, lanes = default_lanes(), kw_hill(T)...)
    Z = round.(Int, r.Zraw ./ 2)
    @printf("check %dx%d %-8s: differs from CPU Float64 at %d points; max |Zraw/2 - round| = %.2e; flagged %d\n",
        cx, cy, T, count(Z .!= Zref), maximum(abs.(r.Zraw ./ 2 .- Z)), count(!=(0), r.flags))
end
write(joinpath(arg("out", "."), "hill_check_Z.csv"), join(string.(Zref), ","))

# 2. timing at full resolution
for T in (Float64, Float32)
    g = plan_grid(δr, br, nx, ny; backend = BACKEND, lanes = default_lanes(), kw_hill(T)...)
    run!(g, D_hill, CM)
    ts = [timed(() -> run!(g, D_hill, CM)) for _ in 1:5]
    r = fetch_result(g)
    Z = round.(Int, r.Zraw ./ 2)
    @printf("%dx%d %-8s: kernel %.2f ms (%.1f Mpts/s), evals median %d, max |Zraw/2 - round| %.2e, unstable %.1f %%\n",
        nx, ny, T, 1e3 * median(ts), nx * ny / median(ts) / 1e6, round(Int, median(r.evals)),
        maximum(abs.(r.Zraw ./ 2 .- Z)), 100 * count(>(0), Z) / length(Z))
end
