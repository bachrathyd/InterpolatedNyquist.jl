# Speed of the number formats relative to Float32 (accuracy not checked): Test 3 with the
# warp-cooperative kernel (mill3w) and Test 2 (mill2g), on a small chart, formats F64 / F32 / F16.
#   julia --project=gpu/scripts hill/warp/speed_ratio.jl [--res 480x270] [--csv out.csv]
const REPO = get(ENV, "NGPU_REPO", normpath(joinpath(@__DIR__, "..", "..")))
include(joinpath(REPO, "hill", "gpu_tour_milling.jl"))
include(joinpath(@__DIR__, "warp3h.jl"))
include(joinpath(@__DIR__, "warp3h_kernel.jl"))

const SFMT = [("F64", Float64, Float64), ("F32", Float32, Float32), ("F16", Float32, Float16)]

function speed_ratio(; res = (480, 270), csv = nothing)
    gpu = ON_GPU ? CUDA.name(CUDA.device()) : "CPU"
    nx, ny = res
    m3 = MODELS["mill3h"]
    m2 = MODELS["mill2g"]
    cases = [("mill3w", (m3..., ws = false), W3h()), ("mill2g", m2, m2.D)]
    rows = String[]
    for (name, m, D) in cases
        t = Dict{String, Float64}()
        for (f, T, TE) in SFMT
            p = plan_for(m, m.c, m.xr, m.yr, nx, ny, T, TE)
            t1 = timed(() -> run!(p, D, m.c))                  # compiles
            reps = t1 > 10 ? 1 : (t1 > 1 ? 3 : 10)
            t[f] = median([timed(() -> run!(p, D, m.c)) for _ in 1:reps])
        end
        for (f, _, _) in SFMT
            r = t[f] / t["F32"]
            @printf("%-8s %-4s %10.2f ms   x%.2f of F32\n", name, f, 1e3t[f], r)
            push!(rows, @sprintf("%s,%s,%s,%d,%d,%.3f,%.4f", gpu, name, f, nx, ny, 1e3t[f], r))
        end
    end
    hdr = "gpu,model,format,nx,ny,ms,ratio_to_F32"
    println("\nCSV\n", hdr); foreach(println, rows)
    csv === nothing || write(csv, hdr * "\n" * join(rows, "\n") * "\n")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    print_device()
    ON_GPU || error("needs a CUDA GPU")
    speed_ratio(; res = parse_res(arg("res", "480x270")), csv = arg("csv", nothing))
end
