# ptxas resource usage of the warp-cooperative D_mill3h class instances ON THE GPU (registers, local
# memory = stack + spills, static shared memory, resident warps per SM) -- the numbers the offline PTX
# flow cannot give (no ptxas on the review machine). Same repo auto-detection as tour_warp3h.jl.
#   julia --project=<repo>/gpu/scripts kernel_info_warp3h.jl
const REPO = get(ENV, "NGPU_REPO", normpath(joinpath(@__DIR__, "..", "..")))   # the repository root
include(joinpath(REPO, "gpu", "scripts", "common.jl"))
include(joinpath(REPO, "hill", "gpu_fast.jl"))
include(joinpath(REPO, "hill", "gpu_helix.jl"))
include(joinpath(@__DIR__, "warp3h.jl"))
include(joinpath(@__DIR__, "warp3h_kernel.jl"))
ON_GPU || error("needs a CUDA GPU")
print_device()
c = mill3c_consts(Q = 8)
@printf("%-22s %5s %8s %8s %10s %12s\n", "instance", "regs", "local B", "shared B", "warps/SM", "time 64x32")
for TE in (Float32, Float16), ns in (2, 3, 4, 5, 6, 0)
    p = plan_grid((3.0, 30.0), (0.0, 10.0), 64, 32; backend = BACKEND, T = Float32, Teval = TE, n_power = 0,
                  ω0 = 1e-9, ω_max = 0.5, h0 = 0.05, hrel = 0.25, nroots = 1, circle = true, parity = true, zmax = 84)
    d = warp3h_dims(c, ns)
    T = Float32
    D = NyquistGPU.CharFn(W3h())
    cT = NyquistGPU.conv_consts(T, Tuple(c))
    wl = mill3w_wslen(Tuple(c))
    ws = CUDA.zeros(Complex{T}, wl)
    idx = CuArray(Int32[1])
    args = (p.Zraw, p.sigma, p.omega, p.steps, p.flags, p.rho, D, cT, p.points, idx, Int32(1), p.mp,
            Val(p.method), Val(1), ws, Int32(wl), Val(:none), Int32(d.ldx), Int32(d.nx), Int32(d.ldr), Int32(d.nn), Val(1))
    k = @cuda launch = false always_inline = true k_warp3h!(args...)
    regs = CUDA.registers(k)
    mem = CUDA.memory(k)
    shb = warp3h_shared_bytes(d, T)                            # dynamic shared memory of one warp
    blk = try CUDA.active_blocks(k.fun, 32; shmem = shb) catch; -1 end
    t = try
        CUDA.@elapsed run_warp3h!(p, c)
    catch err
        NaN
    end
    @printf("%-22s %5d %8d %8d %10d %12.2f\n", "n_s=$(ns == 0 ? "LD" : ns) $(TE)", regs, mem.local, shb, blk, 1e3t)
end
