# CPU validation of the warp-cooperative D_mill3h at HEAD e90381e (v2 setup: adaptive n_s, kink
# correction, wrapped flags from the time order). EmuWarp = the same code as the GPU kernel, every
# phase looped over the 32 lanes, with the shared buffers sized by the point's n_s CLASS (as the
# class kernel instances on the GPU).
#  (1) prepare: q (5 entries) and the reduced X matrices vs mill3h_prepare, bitwise
#  (2) D values vs D_mill3h, bitwise, Float32 and Float16 evaluation
#  (3) full marches with the tour's march settings (circle_kw(84)): Z_raw, steps, flags, ρ identical
# Points: stratified over 3-30 krpm x 0-10 mm so that every n_s = 2..6 occurs.
#   julia --project=gpu/scripts -t 2 test_warp3h_cpu.jl --cpu
const REPO = get(ENV, "NGPU_REPO", normpath(joinpath(@__DIR__, "..", "..")))   # the repository root
include(joinpath(REPO, "gpu", "scripts", "common.jl"))
include(joinpath(REPO, "hill", "gpu_fast.jl"))
include(joinpath(REPO, "hill", "gpu_helix.jl"))
include(joinpath(@__DIR__, "warp3h.jl"))
include(joinpath(@__DIR__, "warp3h_dims.jl"))
using ForwardDiff, Random
const NG = NyquistGPU

c64 = mill3c_consts(Q = 8)                     # nsmax = 6, kink, LD = 58 (as the tour's mill3h)
T = Float32
SE = Complex{ForwardDiff.Dual{NG.PhaseTag, T, 1}}
cT = NG.conv_consts(T, Tuple(c64))
Q, NS, LD = Int(c64[7]), Int(c64[8]), Int(c64[9])
NM = 2Q * NS
XOFF = NM * NF3 + 2NM
XBOFF = XOFF + LD * LD

newref() = NG.WorkSlot(zeros(SE, mill3h_wslen(c64)), 1, 1)
newwarp(ns) = begin
    d = warp3h_dims(c64, ns)
    (NG.WorkSlot(zeros(Complex{T}, mill3w_wslen(Tuple(c64))), 1, 1),
     WB3h(zeros(T, wb3h_nf(d.ldx, d.nx, d.ldr)), Int32(0), zeros(Int32, wb3h_ni(d.nn)), Int32(0),
          Int32(d.ldx), Int32(d.nx), Int32(d.ldr), Int32(d.nn), EmuWarp()))
end

# stratified points: K per class n_s = 2..6 (n_s = 2 needs a_p = 0)
rng = MersenneTwister(11)
K = 24
pts = Tuple{T, T}[]
for s in 2:NS
    got = 0
    while got < K
        p = (T(3 + 27rand(rng)), s == 2 ? zero(T) : T(10rand(rng)))
        if mill3c_ns(p[1], p[2], cT) == s
            push!(pts, p); got += 1
        end
    end
end
println("points: ", length(pts), " (", K, " per n_s = 2..", NS, "), class shared memory per warp (F32): ",
        join(["n_s=$s: $(warp3h_shared_bytes(c64, T, s)) B" for s in 2:NS], ", "), ", full LD: ", warp3h_shared_bytes(c64, T, 0), " B")

# (1) + (2)
nq = 0; nx = 0; nxt = 0; nD = Dict(Float32 => 0, Float16 => 0); nDt = 0
byns = Dict{Int, Vector{Int}}()
for p in pts
    s = mill3c_ns(p[1], p[2], cT)
    sR = newref(); sW, wb = newwarp(s)
    qR = mill3h_prepare(p, (cT, sR))
    qW = prep3h_warp!(p, cT, sW, wb)
    global nq += qR === qW
    nw = Int(qR[2])
    push!(get!(byns, s, Int[]), nw)
    for i in 1:(nw + 2), j in 1:(nw + 4)
        xr = j <= nw + 2 ? _cv(sR[XOFF + i + (j - 1) * LD]) : _cv(sR[XBOFF + i + (j - nw - 3) * LD])
        global nxt += 1
        global nx += xr === getX(wb, i, j)
    end
    for TE in (Float32, Float16)
        cE = NG.conv_consts(TE, cT); qE = map(TE, qR)
        for y in (0.0, 0.07, 0.21, 0.33, 0.5)
            μ = Complex(ForwardDiff.Dual{NG.PhaseTag}(TE(0), zero(TE)), ForwardDiff.Dual{NG.PhaseTag}(TE(y), one(TE)))
            d1 = D_mill3h(μ, qE, (cE, sR))
            d2 = W3h()(μ, qE, (cE, sW, wb))
            nD[TE] += (d1 === d2) | (isnan(real(d1).value) & isnan(real(d2).value))
            TE === Float32 && (global nDt += 1)
        end
    end
end
println("prepare: q (5 entries) identical $nq of $(length(pts)); X entries identical $nx of $nxt")
println("nw per n_s: ", join(["n_s=$s: nw $(extrema(byns[s]))" for s in sort(collect(keys(byns)))], ", "))
println("D values identical: Float32 $(nD[Float32]) of $nDt, Float16 $(nD[Float16]) of $nDt")

# (3) marches with the tour's settings (plan_grid on the CPU backend -> the same MarchParams)
kwt = (n_power = 0, ω0 = 1e-9, ω_max = 0.5, h0 = 0.05, hrel = 0.25, nroots = 1, circle = true, parity = true, zmax = 84)
for TE in (Float32, Float16)
    mp = plan_grid((3.0, 30.0), (0.0, 10.0), 4, 4; backend = CPU(), T = T, Teval = TE, kwt...).mp
    R = Dict{Symbol, Any}()
    for mode in (:ref, :warp)
        Z = zeros(T, length(pts)); S = zeros(Int32, length(pts)); F = zeros(Int8, length(pts)); P = zeros(T, length(pts))
        tm = @elapsed Threads.@threads for k in eachindex(pts)
            p = pts[k]
            if mode === :ref
                sR = newref()
                D = NG.CharFn(D_mill3h); cM = (cT, sR); cE = NG.evalconsts(mp, cM)
                st = NG.seed(D, p, cM, cE, mp, Val(1))
            else
                sW, wb = newwarp(mill3c_ns(p[1], p[2], cT))
                D = NG.CharFn(W3h()); cM = (cT, sW, wb); cE = NG.evalconsts(mp, cM)
                st = NG.seed_q(D, prep3h_warp!(p, cT, sW, wb), cE, mp, Val(1))
            end
            while st.status == Int8(0)
                st = NG.march_step(D, st, cE, mp, Val(:unwrap))
            end
            z, s, w, n, f = NG.finish(st, mp)
            Z[k] = z; S[k] = n; F[k] = f; P[k] = st.ρ2
        end
        R[mode] = (Z, S, F, P, tm)
    end
    (Z1, S1, F1, P1, t1), (Z2, S2, F2, P2, t2) = R[:ref], R[:warp]
    same(a, b) = count(i -> (a[i] === b[i]) | (isnan(a[i]) & isnan(b[i])), eachindex(a))
    println("march eval $TE, $(length(pts)) points: Z_raw identical $(same(Z1, Z2)), steps ", count(S1 .== S2),
        ", flags ", count(F1 .== F2), ", ρ² ", same(P1, P2), "; evals/point ", round(sum(S1 .+ 1) / length(S1); digits = 2),
        ", flagged ", count(!=(0), F1), "; CPU ref ", round(t1; digits = 1), " s, emulated warp ", round(t2; digits = 1), " s")
end
