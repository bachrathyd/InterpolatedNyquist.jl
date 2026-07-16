# Example 30: Comparison with the (multiplication-free) semi-discretization
# method (SemiDiscretizationMethod.jl) on a 2-DOF delayed PD control system.
#
# Semi-discretization approximates the dominant characteristic exponent via
# the spectral radius of a finite transition mapping (converging polynomially
# in the steps-per-delay n), while InterpolatedNyquist integrates the
# argument-principle phase and refines the tracked rightmost root.
#
# NOTE: add SemiDiscretizationMethod to your environment first:
#   using Pkg; Pkg.add("SemiDiscretizationMethod")

using InterpolatedNyquist
using SemiDiscretizationMethod
using LinearAlgebra
using GLMakie

# Reduced showcase system (see example 27): statically unstable main mass with
# a rigidly locked absorber unit and delayed collocated PD control.
const m1, m23, k1, k2, c1, c2, τ = 1.0, 0.5, -1.0, 1.0, 0.05, 0.05, 0.5

function D_red(λ::T, p) where T
    P, D = p
    a11 = m1 * λ^2 + (c1 + c2) * λ + (k1 + k2) + (P + D * λ) * exp(-τ * λ)
    a12 = -(c2 * λ + k2)
    a22 = m23 * λ^2 + c2 * λ + k2
    return a11 * a22 - a12 * a12
end

function sd_dominant_exponent(p; n = 40, order = 1)
    P, Dg = p
    K = [k1 + k2 -k2; -k2 k2]
    C = [c1 + c2 -c2; -c2 c2]
    Minv = [1 / m1 0.0; 0.0 1 / m23]
    A = [zeros(2, 2) I; -Minv*K -Minv*C]
    B = zeros(4, 4); B[3, 1] = -P / m1; B[3, 3] = -Dg / m1
    Δt = τ / n
    lddep = LDDEProblem(ProportionalMX(A), [DelayMX(t -> τ, B)], Additive(zeros(4)))
    mapping = DiscreteMapping_LR(lddep, SemiDiscretization(order, Δt), τ;
        n_steps = 1, calculate_additive = true)
    return log(spectralRadiusOfMapping(mapping)) / Δt
end

p = (1.8, 1.0)

# Ground truth: refined rightmost root from the argument-principle sweep
zi, zr, md, σ_true, ω_true = calculate_unstable_roots_direct(D_red, p;
    ω_max = 1e4, reltol = 1e-9, abstol = 1e-9,
    refinement_method = :Newton, refinement_steps = 15)
println("rightmost root (refined): $σ_true + $ω_true im   (Z = $zi)")

ns = [10, 20, 40, 80, 160, 320]
f = Figure(size = (900, 500))
ax = GLMakie.Axis(f[1, 1], xscale = log10, yscale = log10,
    xlabel = "steps per delay n", ylabel = "|Re λ_dom error|",
    title = "Semi-discretization convergence vs interpolated Nyquist")
for order in (1, 2)
    errs = Float64[]
    for n in ns
        t = @elapsed σ_sd = sd_dominant_exponent(p; n = n, order = order)
        push!(errs, abs(σ_sd - σ_true))
        println("SD order $order, n = $n: Re λ = $σ_sd  (err $(errs[end]), $t s)")
    end
    scatterlines!(ax, ns, max.(errs, 1e-16), label = "SD order $order")
end
for tol in (1e-5, 1e-8)
    t = @elapsed _, _, _, σ_in, _ = calculate_unstable_roots_direct(D_red, p;
        ω_max = 1e4, reltol = tol, abstol = tol)
    println("interp. Nyquist tol=$tol: Re λ = $σ_in  ($t s)")
    hlines!(ax, [max(abs(σ_in - σ_true), 1e-16)], linestyle = :dash,
        label = "interp. Nyquist tol=$tol")
end
axislegend(ax, position = :lb)
mkpath("output_figures")
save("output_figures/example_30.png", f)
display(f)
