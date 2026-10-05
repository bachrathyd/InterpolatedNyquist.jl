# Near-boundary stress test (the paper's s03 study, extended to both sides
# and to the GPU kernels): approach the showcase Hopf boundary geometrically.
# The phase transition of the critical root has half-width |Re λ|, so this is
# where any frequency march can step over it and silently lose a root.
#
# Run (from the repo root):
#   julia --project=gpu/validate -t auto gpu/validate/boundary_stress.jl

using InterpolatedNyquist, NyquistGPU, Printf
include(joinpath(@__DIR__, "..", "scripts", "systems.jl"))

sys = SYSTEMS["showcase"]
D = ref_D(sys)
const DG = 1.5

σ_of(P) = calculate_unstable_roots_direct(D, (P, DG); ω_max = 1e4, reltol = 1e-10,
    abstol = 1e-10, n_power_max = sys.ref_npow, refinement_method = :Newton,
    refinement_steps = 15)[4]
const PB = let lo = 2.0, hi = 3.0
    for _ in 1:60
        mid = (lo + hi) / 2
        (σ_of(mid) < 0) ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end
@printf("Hopf boundary of the showcase at D = %.2f: P_b = %.16f\n\n", DG, PB)

offsets = [s * d for s in (-1, 1) for d in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)]
pts = [(PB + d, DG) for d in offsets]

pkg = [calculate_unstable_roots_direct(D, p; ω_max = 1e4, reltol = 1e-8, abstol = 1e-8,
    n_power_max = sys.ref_npow) for p in pts]
variants = [
    ("unwrap F64", (T = Float64, method = :unwrap)),
    ("unwrap F32", (T = Float32, method = :unwrap)),
    ("bs3 F64 1e-8", (T = Float64, method = :bs3, tol = 1e-8, ω_max = 1e4)),
    ("bs3 F32 1e-5", (T = Float32, method = :bs3, tol = 1e-5, ω_max = 1e4)),
]
res = [sweep(sys.D, pts; c = sys.c, n_power = sys.npow, nroots = 8, schedule = :pixel, kw...)
       for (_, kw) in variants]

@printf("%10s %12s | %8s", "dP", "σ (pkg)", "pkg Vern9")
foreach(v -> @printf(" | %-16s", v[1]), variants)
println()
for (k, d) in enumerate(offsets)
    expect = d < 0 ? "stable" : "unstable"
    @printf("%10.0e %12.3e | %8d", d, pkg[k][4], pkg[k][1])
    for r in res
        f = r.flags[k]
        tag = f & 1 != 0 ? " FAIL" : (f & 2 != 0 ? " side" : "")
        @printf(" | Z=%2d %5d ev%-5s", r.Z[k], r.evals[k], tag)
    end
    println("   ($expect side)")
end
println("""
ev = D evaluations; 'side' = the critical transition was narrower than the
precision's smallest step and its branch was decided by the root side
(flags bit 2); 'FAIL' = march aborted (Z reported as -1).""")
