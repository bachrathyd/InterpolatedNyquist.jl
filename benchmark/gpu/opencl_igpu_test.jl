# Can the validated march kernel run on THIS machine's GPU?
# The AMD Radeon iGPU (gfx90c, Vega) has no CUDA, but exposes OpenCL (with
# FP64); OpenCL.jl compiles Julia kernels to SPIR-V and -- because AMD's
# Windows driver cannot ingest SPIR-V -- translates onward to OpenCL C.
#
# This file is the most PORTABLE variant found (see GPU_NOTES.md round 4):
#   - analytic dD/dlambda instead of ForwardDiff duals (the dual path trips
#     an invalid-select bug in the SPIR-V LLVM backend -- upstream issue)
#   - cexp() instead of Base complex exp (avoids Base.sincos)
#   - sqrt(sqrt()) step control instead of ^(1/3), floor(x+.5) instead of round
# With these, SPIR-V translation SUCCEEDS. On this machine it then dies in
# the vendor driver: AMD's 2022 OpenCL compiler (driver 30.0.13044) segfaults
# on ANY loop-bearing kernel (bisected: saxpy and loop-free transcendental
# kernels run fine; a trivial counted for-loop crashes aclWriteToMem).
# ==> update the AMD Adrenalin driver, then rerun this script.
#
# Run:  julia --project=benchmark/gpu -t auto benchmark/gpu/opencl_igpu_test.jl

using OpenCL, KernelAbstractions, Printf, Statistics

const NPOW = 4.0

@inline function cexp(z::Complex{T}) where {T}
    er = exp(real(z))
    return Complex(er * cos(imag(z)), er * sin(imag(z)))
end

@inline function theta_sigma(::Type{T}, w, P, Dg, c1, zeta, tau) where {T}
    lam = Complex(zero(T), T(w))
    E = cexp(-tau * lam)
    D  = c1 * lam^4 + lam^2 + 2 * zeta * lam + one(T) + (P + Dg * lam) * E
    Dl = 4 * c1 * lam^3 + 2 * lam + 2 * zeta + (Dg - tau * (P + Dg * lam)) * E
    Dw = Complex(-imag(Dl), real(Dl))          # dD/domega = i * dD/dlambda
    absD2 = abs2(D)
    th = (real(D) * imag(Dw) - imag(D) * real(Dw)) / absD2
    sig = -(real(D) * imag(Dw) - imag(D) * real(Dw)) / max(abs2(Dw), T(1e-30))
    return th, absD2, sig
end

@kernel function march_m!(C, @Const(Pv), @Const(Dv),
        c1::T, zeta::T, tau::T, wmax::T, rtol::T, atol::T) where {T}
    idx = @index(Global)
    P = Pv[idx]; Dg = Dv[idx]
    w = T(1e-9); y = zero(T); h = T(1e-2)
    steps = 0
    mind2 = T(Inf); sigbest = zero(T)
    th1, d2, sg = theta_sigma(T, w, P, Dg, c1, zeta, tau)
    if d2 < mind2; mind2 = d2; sigbest = sg; end
    while w < wmax && steps < 200_000
        h = min(h, wmax - w)
        th2, d2a, sga = theta_sigma(T, w + h / 2, P, Dg, c1, zeta, tau)
        th3, d2b, sgb = theta_sigma(T, w + 3 * h / 4, P, Dg, c1, zeta, tau)
        ynew = y + h * (T(2 / 9) * th1 + T(1 / 3) * th2 + T(4 / 9) * th3)
        th4, d2c, sgc = theta_sigma(T, w + h, P, Dg, c1, zeta, tau)
        zlow = y + h * (T(7 / 24) * th1 + T(1 / 4) * th2 +
                        T(1 / 3) * th3 + T(1 / 8) * th4)
        err = abs(ynew - zlow)
        tol = atol + rtol * abs(ynew)
        steps += 1
        if err <= tol
            w += h; y = ynew; th1 = th4
            if d2a < mind2; mind2 = d2a; sigbest = sga; end
            if d2b < mind2; mind2 = d2b; sigbest = sgb; end
            if d2c < mind2; mind2 = d2c; sigbest = sgc; end
        end
        fac = T(0.9) * sqrt(sqrt(tol / max(err, T(1e-30))))
        h *= min(max(fac, T(0.2)), T(5.0))
    end
    Zraw = T(NPOW) / 2 - y / T(pi)
    Z = floor(Zraw + T(0.5))
    C[idx] = Z == 0 ? max(sigbest, T(-1.5)) : min(Z, T(6.0))
end

const T32 = Float32

function sweep(backend, res; tol = 1f-4)
    Pz = range(-2.0, 4.0; length = res); Dz = range(-2.0, 5.0; length = res)
    Ps = T32.(vec([p for p in Pz, d in Dz]))
    Dgs = T32.(vec([d for p in Pz, d in Dz]))
    Pd = KernelAbstractions.allocate(backend, T32, res * res); copyto!(Pd, Ps)
    Dd = KernelAbstractions.allocate(backend, T32, res * res); copyto!(Dd, Dgs)
    C = KernelAbstractions.allocate(backend, T32, res * res)
    k = march_m!(backend)
    run() = (k(C, Pd, Dd, 0.03f0, 0.02f0, 0.5f0, 1f4, tol, tol;
        ndrange = res * res); KernelAbstractions.synchronize(backend))
    run()                                   # compile + warm-up
    t = @elapsed run()
    return Array(C), t
end

println("device: ", cl.device().name, "  (", cl.platform().name, ")")
for res in (100, 200, 400)
    Cg, tg = sweep(OpenCLBackend(), res)
    Cc, tc = sweep(CPU(), res)
    nd = count(abs.(Cg .- Cc) .> 0.05)
    @printf("res %3dx%-3d : GPU %7.1f ms/frame (%5.1f fps)   CPU-16t %7.1f ms   pixels differing: %d/%d\n",
        res, res, 1000tg, 1 / tg, 1000tc, nd, length(Cg))
end
