# Benchmark systems in NyquistGPU form:  D(λ, p, c).
#
# p is one parameter point (here 2-D: the two chart axes), c holds EVERY other
# constant. Keeping the constants in c (converted to the kernel precision by
# `run!`) and using only integer literals keeps a Float32 kernel in Float32 --
# a Float64 literal such as `0.5 * λ` would silently promote it (see
# `check_eltype`).
#
# Each entry also carries
#   ref  -- the CPU package's D(λ, p) for cross-validation (Float64 constants)
#   kw   -- recommended NyquistGPU options for this system
#
# The GPU march wants an ENTIRE D (a quasi-polynomial, no poles): a rational
# D(λ) puts a pole right next to every lightly damped mode, and when a root
# leaves that mode into the right half-plane the pole-zero pair winds the
# phase by a full -2π inside one step -- invisible to any end-point check.
# Clearing the (stable) denominators does not change the count in Re λ > σ.

if !@isdefined(GPU_SYSTEMS_INCLUDED)
const GPU_SYSTEMS_INCLUDED = true

"4th-order delayed oscillator (paper App. A.1 / solver zoo), axes (P, D)."
D_fourth(λ, p, c) = c[1] * λ^4 + λ^2 + 2 * c[2] * λ + 1 + (p[1] + p[2] * λ) * exp(-c[3] * λ)

"Showcase constrained 2-DOF structure, delayed PD control (reduced quasi-polynomial), axes (P, D)."
function D_showcase(λ, p, c)
    P, Dg = p
    m1, m23, k1, k2, c1, c2, τ = c
    a11 = m1 * λ^2 + (c1 + c2) * λ + (k1 + k2) + (P + Dg * λ) * exp(-τ * λ)
    a12 = -(c2 * λ + k2)
    a22 = m23 * λ^2 + c2 * λ + k2
    return a11 * a22 - a12 * a12
end

"""
Two-mode regenerative turning model (paper gallery), axes (spindle speed Ω,
chip width w), with the modal denominators cleared:
    D = M1 M2 + w (1 - e^{-τλ}) (M2 + A2 M1),   τ = 2π/Ω,   n = 4
(the paper's rational form 1 + w(1 - e^{-τλ})(1/M1 + A2/M2), times M1 M2).
"""
function D_turning(λ, p, c)
    Ω, w = p
    ζ1, A2, ζ2, ω2, twoπ = c
    M1 = λ^2 + 2 * ζ1 * λ + 1
    M2 = λ^2 + 2 * ζ2 * ω2 * λ + ω2^2
    return M1 * M2 + w * (1 - exp(-(twoπ / Ω) * λ)) * (M2 + A2 * M1)
end

"The paper's rational turning form (CPU-package reference)."
function D_turning_rational(λ, p)
    Ω, w = p
    τ = 2π / Ω
    G = 1 / (λ^2 + 2 * 0.02 * λ + 1) + 0.45 / (λ^2 + 2 * 0.03 * 2.4 * λ + 2.4^2)
    return 1 + w * (1 - exp(-τ * λ)) * G
end

const SYSTEMS = Dict(
    "fourth" => (name = "fourth", title = "4th-order delayed oscillator",
        D = D_fourth, c = (0.03, 0.02, 0.5), npow = 4,
        xr = (-2.0, 4.0), yr = (-2.0, 5.0), xl = "P", yl = "D",
        ref = (λ, p) -> D_fourth(λ, p, (0.03, 0.02, 0.5)), ref_npow = 4,
        kw = (;)),
    "showcase" => (name = "showcase", title = "showcase 2-DOF DAE (delayed PD)",
        D = D_showcase, c = (1.0, 0.5, -1.0, 1.0, 0.05, 0.05, 0.5), npow = 4,
        xr = (0.5, 3.0), yr = (-0.5, 3.5), xl = "P", yl = "D",
        ref = (λ, p) -> D_showcase(λ, p, (1.0, 0.5, -1.0, 1.0, 0.05, 0.05, 0.5)), ref_npow = 4,
        kw = (;)),
    # Band cap: the regenerative delay τ = 2π/Ω puts a chain of roots near the
    # axis spaced ≈ 2π/τ = Ω apart in frequency (≥ 0.1 on this chart); two
    # unstable chain roots inside one step would wind the phase by -2π
    # unseen, so the step is capped at ~Ω_min/2 over the resonance band.
    "turning" => (name = "turning", title = "two-mode turning lobes",
        D = D_turning, c = (0.02, 0.45, 0.03, 2.4, 2π), npow = 4,
        xr = (0.10, 1.2), yr = (0.01, 1.1), xl = "Ω", yl = "w",
        ref = D_turning_rational, ref_npow = 0,
        kw = (hmax = 0.05, ωband = 5.0)),
)

"CPU-package signature D(λ, p) for a system (Float64 constants)."
ref_D(sys) = sys.ref

end # include guard
