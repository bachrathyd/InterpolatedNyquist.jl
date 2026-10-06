# Shared helpers for the GPU scripts: backend detection, device info, CLI args.
# Pass --cpu to force the CPU backend (validation on machines without CUDA).

using NyquistGPU, KernelAbstractions, Printf, Statistics, Dates
include(joinpath(@__DIR__, "systems.jl"))

if !("--cpu" in ARGS)
    try
        @eval using CUDA
    catch e
        @warn "CUDA.jl could not be loaded -- CPU backend" exception = e
    end
end
const ON_GPU = isdefined(Main, :CUDA) && CUDA.functional()
# always_inline: D (and the complex/dual arithmetic it calls) is compiled into the kernel; as a
# separate function its arguments and results went through local memory
const BACKEND = ON_GPU ? CUDA.CUDABackend(always_inline = true) : CPU()

"Short description of the compute device."
function device_name()
    ON_GPU || return "CPU $(Sys.CPU_NAME) x$(Threads.nthreads()) threads"
    return CUDA.name(CUDA.device())
end

"Persistent lanes that fill the device: #SMs × max resident threads per SM."
function default_lanes()
    ON_GPU || return Threads.nthreads()
    dev = CUDA.device()
    sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    tps = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    return sms * tps
end

function print_device()
    println("backend : ", ON_GPU ? "GPU (CUDA)" : "CPU (KernelAbstractions CPU backend)")
    println("device  : ", device_name())
    if ON_GPU
        dev = CUDA.device()
        @printf("SMs     : %d, max threads/SM %d -> %d resident lanes\n",
            CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
            default_lanes())
        println("CUDA    : driver ", CUDA.driver_version(), ", runtime ", CUDA.runtime_version())
    end
    println("Julia   : ", VERSION, ", ", Threads.nthreads(), " threads")
end

"Value of `--name value` on the command line, or `default`."
function arg(name, default)
    i = findfirst(==("--" * name), ARGS)
    return (i === nothing || i == length(ARGS)) ? default : ARGS[i + 1]
end

"Parse '512' -> (512, 512) and '1920x1080' -> (1920, 1080)."
function parse_res(s)
    p = split(s, 'x')
    return length(p) == 1 ? (parse(Int, p[1]), parse(Int, p[1])) :
           (parse(Int, p[1]), parse(Int, p[2]))
end

"Device synchronization-free wall timer (run! synchronizes)."
timed(f) = (t0 = time_ns(); f(); (time_ns() - t0) / 1e9)

"""
    save_field(out, sys, r, nx, ny, ms; tag = "")

Write a chart for plotting (gpu/colab/plot_fields.py): `<base>.f32` (colour
field: dominant σ where Z == 0, capped Z elsewhere), `<base>.i8` (counts) and
`<base>.json` (axes, device, time); `r` needs `nx × ny` fields `Z`, `sigma`.
"""
function save_field(out, sys, r, nx, ny, ms; tag = "")
    base = joinpath(out, "field_$(sys.name)$(tag)_$(nx)x$(ny)")
    write(base * ".f32", Float32.(colour_field(r)))
    write(base * ".i8", Int8.(clamp.(r.Z, -1, 127)))
    open(base * ".json", "w") do io
        print(io, """{"system": "$(sys.name)$(tag)", "title": "$(sys.title)", "nx": $nx, "ny": $ny,
 "xr": [$(sys.xr[1]), $(sys.xr[2])], "yr": [$(sys.yr[1]), $(sys.yr[2])],
 "xl": "$(sys.xl)", "yl": "$(sys.yl)", "device": "$(device_name())", "kernel_ms": $ms,
 "layout": "column-major nx*ny (x fastest): numpy reshape (ny, nx)"}""")
    end
    println("    saved ", base, ".{f32,i8,json}")
    return base
end
