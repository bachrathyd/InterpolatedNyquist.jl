# Kernel diagnostics for the fast milling model: registers/thread, local memory (spills),
# Float64 instructions, timing (plain CUDA launch and the KernelAbstractions kernel with two
# work-group sizes).  julia --project=gpu/scripts hill/fast_kernel_info.jl
include(joinpath(@__DIR__, "..", "gpu", "scripts", "common.jl"))
include(joinpath(@__DIR__, "gpu_fast.jl"))
ON_GPU || error("needs a CUDA GPU")
const NG = NyquistGPU
print_device()

function cuda_pixel!(Zr, Sg, Om, St, Fl, D, c, Pts, npts, mp, meth, nr)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= npts
        cE = NG.evalconsts(mp, c)
        st = NG.seed(D, @inbounds(Pts[i]), c, cE, mp, nr)
        while st.status == Int8(0)
            st = NG.march_step(D, st, cE, mp, meth)
        end
        NG.store!(Zr, Sg, Om, St, Fl, i, st, mp, D, c, meth, Val(:none))
    end
    return nothing
end

nx, ny = 1920, 1080
pts = grid_points(range(5.0, 25.0; length = nx), range(0.0, 5.0; length = ny))
@printf("\n%-6s %-4s %-16s | %5s %9s %6s | %9s %8s | %9s %9s %9s\n", "model", "Q", "T/Teval", "regs",
    "local[B]", "f64", "cuda128", "Mpts/s", "tile wg64", "wg128", "wg256")
for (mname, Dm, cf) in (("normal", D_mill2n, mill2m_consts), ("polefr", D_mill2p, mill2m_consts),
                       ("gausk", D_mill2g, mill2g_consts)), Q in (8, 16),
    (T, TE) in ((Float32, Float32), (Float32, Float16))
    plan = plan_sweep(pts; backend = BACKEND, T = T, Teval = TE, nroots = 1, n_power = 0, ω0 = 1e-9,
        ω_max = 0.5, h0 = 0.05, hrel = 0.25, schedule = :pixel)
    D = NG.CharFn(Dm)
    c = map(T, cf(Q = Q))                          # constants in the march precision
    n = Int32(length(pts))
    args = (plan.Zraw, plan.sigma, plan.omega, plan.steps, plan.flags, D, c, plan.points, n,
        plan.mp, Val(:unwrap), Val(1))
    k = @cuda launch = false always_inline = true cuda_pixel!(args...)
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
    tka = map((64, 128, 256)) do wg                # plan_grid: 4 x 8 pixel tiles per warp
        g = plan_grid((5.0, 25.0), (0.0, 5.0), nx, ny; backend = BACKEND, T = T, Teval = TE, nroots = 1,
            n_power = 0, ω0 = 1e-9, ω_max = 0.5, h0 = 0.05, hrel = 0.25, schedule = :pixel, workgroup = wg)
        run!(g, Dm, c)
        minimum(timed(() -> run!(g, Dm, c)) for _ in 1:3)
    end
    @printf("%-6s %-4d %-16s | %5d %9d %6d | %9.2f %8.1f | %9.2f %9.2f %9.2f\n", mname, Q, "$T/$TE", regs,
        mem.local, nf64, 1e3t, n / t / 1e6, 1e3tka[1], 1e3tka[2], 1e3tka[3])
end
