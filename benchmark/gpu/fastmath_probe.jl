# Does @fastmath buy anything for the phase march? (CPU only; see GPU_NOTES.md)
#
# Two experiments on the SHOWCASE system, 100x100 chart, wmax=1e4, tol=1e-5:
#
#   A. REALISTIC: @fastmath on the user's characteristic function D(lambda),
#      swept through the package solver. This is what a user would actually
#      write. Expectation: no effect, because @fastmath is SYNTACTIC and does
#      not propagate into callees -- with lambda::Complex{Dual} every operator
#      immediately dispatches into Base/ForwardDiff methods whose instructions
#      are emitted without fast flags (inlining does not add them afterwards).
#
#   B. UPPER BOUND: the whole march written as ONE function body in real
#      scalar arithmetic (analytic dD/dlambda, no duals, no Complex), so every
#      float op really is inside the @fastmath scope. This is the most
#      fast-math could ever give this algorithm -- attainable only by
#      rewriting the solver, not by a keyword on the package.
#
# Run: julia --project=benchmark/gpu -t auto benchmark/gpu/fastmath_probe.jl

using InterpolatedNyquist, Printf, Statistics

const SM = (m1 = 1.0, m2 = 0.3, m3 = 0.2, k1 = -1.0, k2 = 1.0,
            c1 = 0.05, c2 = 0.05, tau = 0.5)
const PRANGE = (0.5, 3.0)
const DRANGE = (-0.5, 3.5)
const NX = 100
const WMAX = 1e4
const TOL = 1e-5
const NPOW = 4.0

# ---------------------------------------------------------------------------
# A. the user's D, plain vs @fastmath
# ---------------------------------------------------------------------------
function D_plain(λ, p)
    P, D = p
    m23 = SM.m2 + SM.m3
    a11 = SM.m1 * λ^2 + (SM.c1 + SM.c2) * λ + (SM.k1 + SM.k2) +
          (P + D * λ) * exp(-SM.tau * λ)
    a12 = -(SM.c2 * λ + SM.k2)
    a22 = m23 * λ^2 + SM.c2 * λ + SM.k2
    return a11 * a22 - a12 * a12
end

function D_fast(λ, p)
    P, D = p
    m23 = SM.m2 + SM.m3
    @fastmath begin
        a11 = SM.m1 * λ^2 + (SM.c1 + SM.c2) * λ + (SM.k1 + SM.k2) +
              (P + D * λ) * exp(-SM.tau * λ)
        a12 = -(SM.c2 * λ + SM.k2)
        a22 = m23 * λ^2 + SM.c2 * λ + SM.k2
        return a11 * a22 - a12 * a12
    end
end

# ---------------------------------------------------------------------------
# B. the whole march in one body, real scalars only: plain vs @fastmath.
# Analytic derivative of the same D on lambda = i*omega.
# The two functions below are byte-identical except for the @fastmath.
# ---------------------------------------------------------------------------
const MARCH_BODY = quote
        m23 = SM.m2 + SM.m3
        w = 1e-9; y = 0.0; h = 1e-2; steps = 0
        mind2 = Inf; sigbest = 0.0
        th1 = 0.0
        @inline function integ(w)
            c = cos(SM.tau * w); s = -sin(SM.tau * w)      # e^{-i tau w}
            w2 = w * w
            a11r = -SM.m1 * w2 + (SM.k1 + SM.k2) + P * c - Dg * w * s
            a11i = (SM.c1 + SM.c2) * w + P * s + Dg * w * c
            a12r = -SM.k2;              a12i = -SM.c2 * w
            a22r = SM.k2 - m23 * w2;    a22i = SM.c2 * w
            Dr = a11r * a22r - a11i * a22i - (a12r * a12r - a12i * a12i)
            Di = a11r * a22i + a11i * a22r - 2 * a12r * a12i
            Fr = Dg - SM.tau * P;       Fi = -SM.tau * Dg * w
            b11r = (SM.c1 + SM.c2) + Fr * c - Fi * s
            b11i = 2 * SM.m1 * w + Fr * s + Fi * c
            b22r = SM.c2;               b22i = 2 * m23 * w
            Lr = b11r * a22r - b11i * a22i + a11r * b22r - a11i * b22i +
                 2 * SM.c2 * a12i
            Li = b11r * a22i + b11i * a22r + a11r * b22i + a11i * b22r -
                 2 * SM.c2 * a12r
            Dwr = -Li; Dwi = Lr                            # dD/dw = i dD/dlam
            absD2 = Dr * Dr + Di * Di
            cross = Dr * Dwi - Di * Dwr
            return cross / absD2, absD2, -cross / max(Dwr * Dwr + Dwi * Dwi, 1e-300)
        end
        th1, d2, sg = integ(w)
        if d2 < mind2; mind2 = d2; sigbest = sg; end
        while w < WMAX && steps < 200_000
            h = min(h, WMAX - w)
            th2, d2a, sga = integ(w + h / 2)
            th3, d2b, sgb = integ(w + 3h / 4)
            ynew = y + h * (2th1 / 9 + th2 / 3 + 4th3 / 9)
            th4, d2c, sgc = integ(w + h)
            zlow = y + h * (7th1 / 24 + th2 / 4 + th3 / 3 + th4 / 8)
            err = abs(ynew - zlow)
            tol = TOL + TOL * abs(ynew)
            steps += 1
            if err <= tol
                w += h; y = ynew; th1 = th4
                if d2a < mind2; mind2 = d2a; sigbest = sga; end
                if d2b < mind2; mind2 = d2b; sigbest = sgb; end
                if d2c < mind2; mind2 = d2c; sigbest = sgc; end
            end
            fac = 0.9 * cbrt(tol / max(err, 1e-30))
            h *= min(max(fac, 0.2), 5.0)
        end
    (NPOW / 2 - y / pi, sigbest, steps)
end

# built by @eval (not a macro) so the body sees the arguments P, Dg directly;
# the two functions are identical except for the @fastmath
@eval march_plain(P, Dg) = $(MARCH_BODY)
@eval march_fast(P, Dg) = @fastmath $(MARCH_BODY)

# ---------------------------------------------------------------------------
function grid_params()
    Pv = range(PRANGE...; length = NX); Dv = range(DRANGE...; length = NX)
    return vec([(p, d) for p in Pv, d in Dv])
end

function sweep_kernel(f, params)
    out = Vector{Float64}(undef, length(params))
    sig = Vector{Float64}(undef, length(params))
    Threads.@threads for i in eachindex(params)
        z, s, _ = f(params[i][1], params[i][2])
        out[i] = z; sig[i] = s
    end
    return out, sig
end

timeit(f; n = 5) = (GC.gc(); median([(GC.gc(); @elapsed f()) for _ in 1:n]))

params = grid_params()
println("=== @fastmath probe: showcase chart $(NX)x$(NX), wmax=$WMAX, tol=$TOL,",
        " $(Threads.nthreads()) threads ===\n")

# --- A -------------------------------------------------------------------
pkg(D) = calculate_unstable_roots_p_vec(D, params; n_roots_to_track = 0,
    ω_max = WMAX, reltol = TOL, abstol = TOL, n_power_max = NPOW)[2]
pkg(D_plain); pkg(D_fast)                                   # JIT
tp = timeit(() -> pkg(D_plain)); zp = pkg(D_plain)
tf = timeit(() -> pkg(D_fast));  zf = pkg(D_fast)
@printf("A. package solver, user D:\n")
@printf("   plain      %6.3f s\n", tp)
@printf("   @fastmath  %6.3f s   (%+.1f %%)\n", tf, 100 * (tf - tp) / tp)
@printf("   counts differing: %d / %d;  max |dZraw| = %.2e\n\n",
    count(round.(zp) .!= round.(zf)), length(zp), maximum(abs, zp .- zf))

# --- B -------------------------------------------------------------------
sweep_kernel(march_plain, params[1:8]); sweep_kernel(march_fast, params[1:8])
tkp = timeit(() -> sweep_kernel(march_plain, params))
tkf = timeit(() -> sweep_kernel(march_fast, params))
zkp, skp = sweep_kernel(march_plain, params)
zkf, skf = sweep_kernel(march_fast, params)
@printf("B. whole march in real scalars (upper bound):\n")
@printf("   plain      %6.3f s\n", tkp)
@printf("   @fastmath  %6.3f s   (%+.1f %%)\n", tkf, 100 * (tkf - tkp) / tkp)
@printf("   counts differing: %d / %d;  max |dZraw| = %.2e;  max |dsigma| = %.2e\n",
    count(round.(zkp) .!= round.(zkf)), length(zkp),
    maximum(abs, zkp .- zkf), maximum(abs, skp .- skf))
resid(z) = abs.(z .- round.(z))
@printf("   integer residual  plain: med %.2e max %.2e\n",
    median(resid(zkp)), maximum(resid(zkp)))
@printf("   integer residual  fast : med %.2e max %.2e\n",
    median(resid(zkf)), maximum(resid(zkf)))
@printf("\n   (kernel counts vs package reference: %d / %d differ)\n",
    count(round.(zkp) .!= round.(zp)), length(zp))
