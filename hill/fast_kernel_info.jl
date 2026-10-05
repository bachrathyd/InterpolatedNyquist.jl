# Kernel diagnostics for the fast milling model: registers/thread, local memory (spills),
# Float64 instructions, timing.  julia --project=gpu/scripts hill/fast_kernel_info.jl
include(joinpath(@__DIR__, "..", "gpu", "scripts", "common.jl"))
include(joinpath(@__DIR__, "gpu_fast.jl"))
ON_GPU || error("needs a CUDA GPU")
const NG = NyquistGPU
print_device()

function cuda_pixel!(Zr, Sg, Om, St, Fl, D, c, Pts, npts, mp, meth, nr)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= npts
        st = NG.seed(D, @inbounds(Pts[i]), c, mp, nr)
        while st.status == Int8(0)
            st = NG.march_step(D, st, c, mp, meth)
        end
        NG.store!(Zr, Sg, Om, St, Fl, i, st, mp, D, c, meth)
    end
    return nothing
end

nx, ny = 1920, 1080
pts = grid_points(range(5.0, 25.0; length = nx), range(0.0, 5.0; length = ny))
@printf("\n%-4s %-16s | %5s %9s %6s | %9s %8s\n", "Q", "T/Teval", "regs", "local[B]", "f64", "1080p ms", "Mpts/s")
for Q in (8, 16), (T, TE) in ((Float32, Float32), (Float32, Float16)), maxregs in (nothing, 128)
    plan = plan_sweep(pts; backend = BACKEND, T = T, Teval = TE, nroots = 1, n_power = 0, ω0 = 1e-9,
        ω_max = 0.5, h0 = 1e-3, hrel = 0.05, schedule = :pixel)
    D = NG.CharFn(D_mill2m)
    c = map(TE, mill2m_consts(Q = Q))
    n = Int32(length(pts))
    args = (plan.Zraw, plan.sigma, plan.omega, plan.steps, plan.flags, D, c, plan.points, n,
        plan.mp, Val(:unwrap), Val(1))
    k = maxregs === nothing ? (@cuda launch = false cuda_pixel!(args...)) :
                              (@cuda launch = false maxregs = maxregs cuda_pixel!(args...))
    regs = CUDA.registers(k)
    mem = CUDA.memory(k)
    ptx = sprint(io -> CUDA.code_ptx(io, cuda_pixel!, Tuple{map(a -> typeof(CUDA.cudaconvert(a)), args)...}; kernel = true))
    nf64 = count(m -> true, eachmatch(r"\.f64\b", ptx))
    threads = 128
    blocks = cld(Int(n), threads)
    k(args...; threads = threads, blocks = blocks); CUDA.synchronize()
    t = minimum(1:3) do _
        CUDA.@elapsed k(args...; threads = threads, blocks = blocks)
    end
    @printf("%-4d %-16s | %5d %9d %6d | %9.2f %8.1f   %s\n", Q, "$T/$TE", regs, mem.local, nf64, 1e3t, n / t / 1e6,
        maxregs === nothing ? "" : "maxregs=$maxregs")
end
