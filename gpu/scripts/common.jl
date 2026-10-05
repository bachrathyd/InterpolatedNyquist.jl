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
const BACKEND = ON_GPU ? CUDA.CUDABackend() : CPU()

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
