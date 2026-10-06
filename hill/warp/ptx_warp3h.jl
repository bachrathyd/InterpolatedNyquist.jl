# Offline PTX (no GPU, no ptxas here) of k_warp3h! per axial class and per architecture:
# sm_75 (T4), sm_80 (A100), sm_89 (L4), sm_120 (RTX PRO 6000); Float32 and Float16 evaluation.
# Reports the static shared memory (.shared declarations), the local depot, f64 and calls.
#   julia --project=gpu/scripts -t 1 ptx_warp3h.jl
const REPO = get(ENV, "NGPU_REPO", normpath(joinpath(@__DIR__, "..", "..")))   # the repository root
include(joinpath(REPO, "gpu", "scripts", "ptx_offline.jl"))
include(joinpath(REPO, "gpu", "scripts", "common.jl"))
include(joinpath(REPO, "hill", "gpu_fast.jl"))
include(joinpath(REPO, "hill", "gpu_helix.jl"))
include(joinpath(@__DIR__, "warp3h.jl"))
include(joinpath(@__DIR__, "warp3h_kernel.jl"))
using ForwardDiff
const NG = NyquistGPU
V(T) = DevVec{T}
c = mill3c_consts(Q = 8)
T = Float32
C = typeof(map(T, c))
shared_bytes(asm) = sum((parse(Int, m[1]) for m in eachmatch(r"\.shared[^\n\[]*\[(\d+)\]", asm)); init = 0)
cases = Any[(TE, cap, 3) for TE in (Float32,) for cap in (v"7.5", v"8.0", v"8.9", v"12.0")]
append!(cases, [(Float16, cap, 3) for cap in (v"8.9", v"12.0")])     # one kernel per format (dynamic shared)
for (TE, cap, ns) in cases
    d = warp3h_dims(c, ns)
    mpT = NG.MarchParams{T, TE}
    tt = Tuple{V(T), V(T), V(T), V(Int32), V(Int8), V(T), NG.CharFn{W3h}, C, V(NTuple{2, T}), V(Int32), Int32, mpT,
               Val{:unwrap}, Val{1}, V(Complex{T}), Int32, Val{:none}, Int32, Int32, Int32, Int32, Val{1}}
    lab = "eval $TE sm_$(cap.major)$(cap.minor)"
    try
        t = @elapsed asm = ptx_of(k_warp3h!, tt; cap = cap, ptx = v"8.7")
        s = ptxstats(asm)
        println(rpad(lab, 40), ": compiles ($(round(t; digits = 1)) s); ", s.n, " PTX instr, static shared ", shared_bytes(asm),
                " B (dynamic: ", warp3h_shared_bytes(d, T), " B per n_s = 3 warp), local depot ", s.local_depot, ", f64 ", s.f64,
                ", calls ", s.calls, " ", s.nvcalls)
    catch err
        println(rpad(lab, 40), ": FAILS -- ", first(sprint(showerror, err), 1200))
    end
end
