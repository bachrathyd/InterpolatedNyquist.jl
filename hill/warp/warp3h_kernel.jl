# CUDA.jl kernel + launchers for the warp-cooperative D_mill3h (include after warp3h.jl; needs CUDA).
# One warp per point, persistent warps over an index list (stride = number of warps). The 32 lanes
# prepare the point together (prep3h_warp!), then run NyquistGPU's unchanged march (seed_q /
# march_step) on the warp-collective D = CharFn(W3h()) -- every lane gets the same D, so no
# divergence; lane 0 stores (with the ρ certificate). No refinement / certification in a warp.
#
# Shared memory is sized PER AXIAL CLASS n_s (warp3h_dims.jl): the points of a chart are split by
# n_s = mill3c_ns(rk, ap, c) on the device and every class runs its own launch (on the 3-30 krpm x
# 0-10 mm chart: n_s = 3 for 76.5 % of the points, 8.3 KB per warp instead of 33 KB at the
# constants' LD = 58). Points whose nw exceeds their class are flagged FLAG_W3OVER and re-run with
# the full-LD geometry. The geometry is a runtime argument and the shared memory DYNAMIC: one
# compiled kernel per number format serves every class (the class sizes only steer the occupancy).
#
#   run_warp3h!(plan, c)                 the whole chart (plan from plan_grid / plan_for, ws = false)
#   run_warp3h_list!(plan, c, idx, n)    only idx[1:n] (run_list! semantics)
#   NyquistGPU.run!(plan, W3h(), c), NyquistGPU.run_list!(plan, W3h(), c, idx, n): the same, so that
#   recheck_flagged!(plan, rplan, W3h(), c) (F16+) and run_adaptive!(plan, W3h(), c; rplan) drive it.
using CUDA, ForwardDiff
include(joinpath(@__DIR__, "warp3h_dims.jl"))
@inline wsync(::GpuWarp) = CUDA.sync_warp()

function k_warp3h!(Zr, Sg, Om, St, Fl, Rh, D, c, Pts, idx, n, mp, meth, nr, ws, wslen, rm,
                   LDX::Int32, NX::Int32, LDR::Int32, NN::Int32, ::Val{WPB}) where {WPB}
    T = typeof(mp.σ)
    tid = Int32(threadIdx().x) - Int32(1)
    lane = tid % Int32(32)
    wib = tid ÷ Int32(32)
    gw = (Int32(blockIdx().x) - Int32(1)) * Int32(WPB) + wib        # global warp (0-based)
    W = Int32(gridDim().x) * Int32(WPB)
    NFW = Int32(wb3h_nf(LDX, NX, LDR))
    NIW = Int32(wb3h_ni(NN))
    Fsh = CuDynamicSharedArray(T, Int(NFW) * WPB)
    Ish = CuDynamicSharedArray(Int32, Int(NIW) * WPB, Int(NFW) * WPB * sizeof(T))
    slot = NyquistGPU.WorkSlot(ws, Int(gw) * Int(wslen) + 1, 1)      # contiguous per warp
    wb = WB3h(Fsh, wib * NFW, Ish, wib * NIW, LDX, NX, LDR, NN, GpuWarp(lane))
    cM = (c, slot, wb)
    cE = NyquistGPU.evalconsts(mp, cM)
    t = gw + Int32(1)
    while t <= n
        i = @inbounds idx[t]
        q = prep3h_warp!(@inbounds(Pts[i]), c, slot, wb)
        if unsafe_trunc(Int, q[2]) + 2 > LDR                      # nw over this class
            if lane == Int32(0)
                @inbounds Zr[i] = T(NaN); @inbounds Sg[i] = T(NaN); @inbounds Om[i] = T(NaN)
                @inbounds St[i] = Int32(0); @inbounds Fl[i] = FLAG_W3OVER; @inbounds Rh[i] = T(NaN)
            end
        else
            st = NyquistGPU.seed_q(D, q, cE, mp, nr)
            while st.status == Int8(0)
                st = NyquistGPU.march_step(D, st, cE, mp, meth)
            end
            if lane == Int32(0)
                NyquistGPU.store!(Zr, Sg, Om, St, Fl, Rh, i, st, mp, D, cM, meth, rm)
            end
        end
        CUDA.sync_warp()
        t += W
    end
    return nothing
end

# one class instance over idx[1:n] (idx: device vector of Int32 point indices)
function warp3h_launch!(p::NyquistGPU.SweepPlan{T, N}, c, idx, n::Integer, ns::Integer;
                        warps_per_sm = nothing, warps_per_block = 1) where {T, N}
    n == 0 && return p
    (p.mp.newton == 0 && p.mp.bisect == 0) || error("run_warp3h!: no refinement / certification")
    d = warp3h_dims(c, ns)
    dev = CUDA.device()
    sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    shm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    wpsm = something(warps_per_sm, clamp(floor(Int, 0.95shm / warp3h_shared_bytes(d, T)), 1, 32))
    W = min(sms * wpsm, n)
    wl = mill3w_wslen(Tuple(c))
    ws = CUDA.zeros(Complex{T}, W * wl)
    cT = NyquistGPU.conv_consts(T, Tuple(c))
    D = NyquistGPU.CharFn(W3h())
    shmem = warp3h_shared_bytes(d, T) * warps_per_block
    args = (p.Zraw, p.sigma, p.omega, p.steps, p.flags, p.rho, D, cT, p.points, idx, Int32(n), p.mp,
        Val(p.method), Val(N), ws, Int32(wl), Val(:none), Int32(d.ldx), Int32(d.nx), Int32(d.ldr),
        Int32(d.nn), Val(warps_per_block))
    k = @cuda launch = false always_inline = true k_warp3h!(args...)
    shmem > 48 * 1024 && (CUDA.attributes(k.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem)
    k(args...; threads = 32 * warps_per_block, blocks = cld(W, warps_per_block), shmem = shmem)
    CUDA.unsafe_free!(ws)
    return p
end

"""
    run_warp3h_list!(plan, c, idx, n; warps_per_sm = nothing) -> plan

March the points `idx[1:n]` of `plan` (run_list! semantics) with the warp-cooperative D_mill3h,
split by the axial class n_s of each point (one kernel instance per class, shared memory sized by the
class), then the overflow points (nw over the class) with the full-LD instance.
"""
function run_warp3h_list!(p::NyquistGPU.SweepPlan{T}, c, idx, n::Integer; kw...) where {T}
    n == 0 && return p
    cT = NyquistGPU.conv_consts(T, Tuple(c))
    NS = Int(c[8])
    iv = CuArray{Int32}(view(idx, 1:n))
    pts = p.points[iv]
    nsv = map(q -> Int8(mill3c_ns(q[1], q[2], cT)), pts)
    for s in 2:NS
        sel = findall(==(Int8(s)), nsv)
        m = length(sel)
        m == 0 && continue
        warp3h_launch!(p, c, iv[sel], m, s; kw...)
    end
    over = findall(==(FLAG_W3OVER), p.flags[iv])
    isempty(over) || warp3h_launch!(p, c, iv[over], length(over), 0; kw...)
    CUDA.synchronize()
    return p
end

run_warp3h!(p::NyquistGPU.SweepPlan, c; kw...) =
    run_warp3h_list!(p, c, CuArray(Int32(1):Int32(p.npts)), p.npts; kw...)

# the engine's entry points for D = W3h(): run!, run_list! -- and with them recheck_flagged!,
# recheck_points!, run_adaptive!
NyquistGPU.run!(p::NyquistGPU.SweepPlan, ::W3h, c = ()) = run_warp3h!(p, c)
NyquistGPU.run_list!(p::NyquistGPU.SweepPlan, ::W3h, c, idx, n::Integer) = run_warp3h_list!(p, c, idx, n)
