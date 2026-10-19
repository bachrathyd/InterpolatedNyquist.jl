# Turning GSTM (dissertation Sec. 4): r0, additive-only and full stationary variance
# from the unwrap march, against a QuadGK reference and the Python numbers.
#   julia --project=. notes/second_moment_turning.jl
include("second_moment_turning_model.jl")

println("== 1. Python numbers (Sec. 4 of the note) vs march (rtol_I = 1e-4) vs QuadGK")
python = Dict((0.2, 1.0, 1.0) => (0.095845, 0.0771, 0.10385),
              (0.2, 0.6, 0.7) => (0.23485, 0.3680, 0.37158),
              (0.2, 0.6, 1.5) => (1.0784, 1.6897, NaN))
for p in [(0.2, 1.0, 1.0), (0.2, 0.6, 0.7), (0.2, 0.6, 1.5)]
    Z, I, ev = second_moment_unwrap(turning, p; n_power_max = 2, rtol_I = 1e-4)
    va, r0, v = stats(p, I); va_r, r0_r, v_r = stats(p, reference(p)); py = python[p]
    fmt(x, r) = r < 1 ? @sprintf("%.6f", x) : "MS-unstable"
    @printf("K=%.2f Ω=%.1f σ=%.1f  Z=%d evals=%d\n", p..., Z, ev)
    @printf("   Var_add  march %.6f  ref %.6f  py %.6f\n", va, va_r, py[1])
    @printf("   r0       march %.6f  ref %.6f  py %.4f\n", r0, r0_r, py[2])
    @printf("   Var      march %s  ref %s  py %s\n", fmt(v, r0), fmt(v_r, r0_r), isnan(py[3]) ? "MS-unstable" : @sprintf("%.5f", py[3]))
end

println("\n== 2. Accuracy vs extra nodes (rtol_I = Inf: exactly the deterministic march nodes)")
cases = [(0.2, 1.0, 1.0), (0.2, 0.6, 0.7), (0.30, 0.5, 0.5), (0.243, 0.6, 0.3)]
for p in cases
    ref = stats(p, reference(p))
    Zd = calculate_unstable_roots_unwrap((λ, q) -> turning(λ, q)[1], p; n_roots_to_track = 0, n_power_max = 2)[1]
    @printf("K=%.3f Ω=%.1f σ=%.1f  Z=%d  ref: Var=%.6g r0=%.4g\n", p..., Zd, ref[3], ref[2])
    Zd == 0 || continue
    for rt in (Inf, 1e-2, 1e-3, 1e-4, 1e-5)
        Z, I, ev = second_moment_unwrap(turning, p; n_power_max = 2, rtol_I = rt)
        s = stats(p, I)
        @printf("   rtol_I=%-6s evals=%5d  rel.err Var=%.1e  r0=%.1e\n", string(rt), ev,
                abs(s[3] / ref[3] - 1), abs(s[2] / ref[2] - 1))
    end
end

# wall time and node distribution: second_moment_cost.jl

println("\n== 4. MS exponent (bisection on Re s = σ/2 lines)")
# exact delay-free reference (check_exponent.py): y'' + 0.2y' + 1.5y = 0.6·s·(-y)ξ
nodelay(λ, p) = (λ^2 + 0.2λ + 1.5, one(λ), -0.5 * p[1] * one(λ))
for s in (1.2, 1.8)
    λ = ms_exponent(nodelay, (s,); n_power_max = 2, σ_lo = -1.0, σ_hi = 1.0, rtol_I = 1e-6)
    @printf("delay-free s=%.1f: λ_MS march %.5f   exact %s\n", s, λ, s == 1.2 ? "-0.07949" : "+0.06856")
end
for p in [(0.2, 0.6, 0.7), (0.2, 0.6, 1.5)]
    λ = ms_exponent(turning, p; n_power_max = 2, σ_lo = -1.0, σ_hi = 1.0, rtol_I = 1e-5)
    rq = reference(p, λ / 2)[2]
    @printf("GSTM K=%.1f Ω=%.1f σ=%.1f: λ_MS = %.5f   (QuadGK r(λ_MS) = %.4f)\n", p..., λ, rq)
end
