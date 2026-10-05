# Thread schedules for the fast milling model on the GPU: one thread per point (:pixel) against
# persistent lanes that take the next point when theirs is done (:strided, :queue). The march needs
# 11..60 evaluations per point, so a warp of :pixel threads waits for its slowest point.
#   julia --project=gpu/scripts hill/fast_schedule_bench.jl [--q 16]
include(joinpath(@__DIR__, "..", "gpu", "scripts", "common.jl"))
include(joinpath(@__DIR__, "gpu_fast.jl"))
ON_GPU || error("needs a CUDA GPU")
const NG = NyquistGPU
print_device()
Q = parse(Int, arg("q", "16"))
xr, yr = (5.0, 25.0), (0.0, 5.0)
c = mill2m_consts(Q = Q)
kw = (n_power = 0, ω0 = 1e-9, ω_max = 0.5, h0 = 0.05, hrel = 0.25, nroots = 1)
sms = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
for (nx, ny) in ((1920, 1080), (3840, 2160)), (T, TE) in ((Float32, Float16), (Float32, Float32))
    println("\n$(nx)x$(ny), $(T)/$(TE), Q = $Q")
    ref = nothing
    for (sch, wg, lpm) in ((:pixel, 128, 0), (:pixel, 256, 0), (:pixel, 64, 0), (:queue, 128, 256), (:queue, 128, 384),
                           (:queue, 128, 512), (:queue, 256, 768), (:strided, 128, 384), (:strided, 128, 512))
        lanes = lpm == 0 ? nothing : sms * lpm
        g = plan_grid(xr, yr, nx, ny; backend = BACKEND, T = T, Teval = TE, schedule = sch, workgroup = wg,
                      lanes = lanes, kw...)
        run!(g, D_mill2r, c)
        t = median([timed(() -> run!(g, D_mill2r, c)) for _ in 1:5])
        r = fetch_result(g)
        ref === nothing && (ref = r.Z)
        eff = sch === :pixel ? NG.simulate_schedule(r.steps; schedule = :pixel_rowmajor) :
              NG.simulate_schedule(r.steps; schedule = sch, lanes = something(lanes, 1))
        @printf("  %-8s wg %3d lanes/SM %4s: %8.2f ms (%6.1f Mpts/s)  simulated SIMT efficiency %.2f  counts = :pixel: %s\n",
            sch, wg, lpm == 0 ? "-" : string(lpm), 1e3t, nx * ny / t / 1e6, eff, r.Z == ref)
    end
end
