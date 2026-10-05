# Do the milling kernels compile for the GPU? PTX generated on the CPU (no GPU needed):
#   julia --project=gpu/scripts hill/gpu_compile_check.jl
const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "gpu", "scripts", "ptx_offline.jl"))
include(joinpath(REPO, "gpu", "scripts", "common.jl"))
include(joinpath(REPO, "hill", "gpu_models.jl"))
include(joinpath(REPO, "hill", "gpu_fast.jl"))
include(joinpath(REPO, "hill", "gpu_helix.jl"))
using ForwardDiff
const NG = NyquistGPU
function kws!(Zr, Sg, Om, St, Fl, D, c0, Pts, npts, mp, meth, nr, lanes, ws, rm)
    c = NG.with_slot(c0, ws, 1, lanes)
    cE = NG.evalconsts(mp, c)
    st = NG.seed(D, @inbounds(Pts[1]), c, cE, mp, nr)
    while st.status == Int8(0)
        st = NG.march_step(D, st, cE, mp, meth)
    end
    NG.store!(Zr, Sg, Om, St, Fl, 1, st, mp, D, c, meth, rm)
    return nothing
end
function kpx!(Zr, Sg, Om, St, Fl, D, c, Pts, npts, mp, meth, nr, rm)
    cE = NG.evalconsts(mp, c)
    st = NG.seed(D, @inbounds(Pts[1]), c, cE, mp, nr)
    while st.status == Int8(0)
        st = NG.march_step(D, st, cE, mp, meth)
    end
    NG.store!(Zr, Sg, Om, St, Fl, 1, st, mp, D, c, meth, rm)
    return nothing
end
V(T) = DevVec{T}
for (name, D, c, ws) in (("D_mill3h", D_mill3h, mill3c_consts(), true), ("D_mill3c", D_mill3c, mill3c_consts(), true), ("D_mill2g", D_mill2g, mill2g_consts(Q = 8), false), ("D_mill2p", D_mill2p, mill2m_consts(Q = 16), false),
                         ("D_mill3", D_mill3, mill3_consts(), true))
    for TE in (Float32, Float16)
        T = Float32
        name == "D_mill3" && TE === Float16 && continue
        C = typeof(map(T, c))
        mpT = NG.MarchParams{T, TE}
        rm = Val{:none}
        try
            asm = if ws
                E = Complex{ForwardDiff.Dual{NG.PhaseTag, T, 1}}
                ptx_of(kws!, Tuple{V(T), V(T), V(T), V(Int32), V(Int8), NG.CharFn{typeof(D)}, C, V(NTuple{2, T}), Int32, mpT,
                                   Val{:unwrap}, Val{1}, Int32, V(E), rm})
            else
                ptx_of(kpx!, Tuple{V(T), V(T), V(T), V(Int32), V(Int8), NG.CharFn{typeof(D)}, C, V(NTuple{2, T}), Int32, mpT,
                                   Val{:unwrap}, Val{1}, rm})
            end
            s = ptxstats(asm)
            println("$name $T/$TE: compiles; ", s.n, " PTX instructions, local depot ", s.local_depot, ", f64 ", s.f64, ", calls ", s.calls)
        catch err
            msg = sprint(showerror, err)
            println("$name $T/$TE: FAILS -- ", join(filter(l -> occursin("Reason", l), split(msg, '\n'))[1:min(end, 4)], " | "))
        end
    end
end
