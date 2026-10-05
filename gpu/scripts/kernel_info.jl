# GPU kernel diagnostics: registers per thread (-> occupancy), local memory
# (register spills), Float64 instructions hiding in a Float32 kernel, and the
# full-HD timing of a plain CUDA.jl version of the one-thread-per-point march
# for several root-slot counts.
#
# Run (CUDA machine / Colab):  julia --project=gpu/scripts gpu/scripts/kernel_info.jl

include(joinpath(@__DIR__, "common.jl"))
ON_GPU || error("kernel_info.jl needs a functional CUDA GPU")
const NG = NyquistGPU
print_device()

# the :pixel march as a plain CUDA.jl kernel (same NyquistGPU internals)
function cuda_pixel!(Zr, Sg, Om, St, Fl, D, c, Pts, npts, mp, meth, nr)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= npts
        st = NG.seed(D, @inbounds(Pts[i]), c, mp, nr)
        while st.status == Int8(0)
            st = NG.march_step(D, st, c, mp, meth)
        end
        NG.store!(Zr, Sg, Om, St, Fl, i, st, mp)
    end
    return nothing
end

dev = CUDA.device()
regs_per_sm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
max_thr_sm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

@printf("\n%-9s %-8s %3s | %5s %10s %9s %6s %7s | %10s %8s\n", "system", "T", "N",
    "regs", "local [B]", "occupancy", "f64", "maxthr", "1080p ms", "Mpts/s")
for name in ("fourth", "showcase", "turning"), T in (Float32, Float64), N in (2, 4, 8)
    sys = SYSTEMS[name]
    nx, ny = 1920, 1080
    pts = grid_points(range(sys.xr...; length = nx), range(sys.yr...; length = ny))
    plan = plan_sweep(pts; backend = BACKEND, T = T, nroots = N, n_power = sys.npow,
        schedule = :pixel, sys.kw...)
    D = NG.CharFn(sys.D)
    c = map(T, sys.c)
    n = Int32(length(pts))
    args = (plan.Zraw, plan.sigma, plan.omega, plan.steps, plan.flags, D, c, plan.points, n,
        plan.mp, Val(:unwrap), Val(N))
    k = @cuda launch = false cuda_pixel!(args...)
    regs = CUDA.registers(k)
    mem = CUDA.memory(k)
    occ = min(max_thr_sm, 32 * fld(regs_per_sm, 32 * max(regs, 1))) / max_thr_sm
    ptx = sprint(io -> CUDA.code_ptx(io, cuda_pixel!, Tuple{map(a -> typeof(CUDA.cudaconvert(a)), args)...}; kernel = true))
    nf64 = count(m -> true, eachmatch(r"\.f64\b", ptx))
    threads = 256
    blocks = cld(Int(n), threads)
    k(args...; threads = threads, blocks = blocks); CUDA.synchronize()          # warm-up
    t = minimum(1:3) do _
        CUDA.@elapsed k(args...; threads = threads, blocks = blocks)
    end
    @printf("%-9s %-8s %3d | %5d %10d %8.0f%% %6d %7d | %10.2f %8.2f\n", name, T, N,
        regs, mem.local, 100occ, nf64, CUDA.maxthreads(k), 1e3t, n / t / 1e6)
end
println("""

regs: registers/thread (T4/L4/A100: 64 K registers per SM, so >64 regs/thread lowers occupancy);
local: bytes of per-thread local memory (spills -- slow DRAM traffic);
f64: Float64 PTX instructions (a Float32 kernel should have ~0; FP64 runs at 1/32-1/64 rate on
T4/L4/RTX cards); occupancy: resident threads per SM allowed by the register count.""")
