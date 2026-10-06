# Kernel geometry of the warp-cooperative D_mill3h (no CUDA needed: shared with the CPU emulation).
# One kernel instance per axial class n_s: on the Test 3 chart nw = n_s Q exactly (20 736 checked
# points, nw_stats.jl), so the shared X of a class is sized (n_s Q + 3) x (n_s Q + 4) -- not by the
# maximal LD of the constants. Points whose nw exceeds their class (none seen) are flagged
# FLAG_W3OVER by the class kernel and re-run by the full-LD instance (class 0).
const FLAG_W3OVER = Int8(32)

# dims of class ns (2..NS), or of the full-LD instance (ns = 0)
function warp3h_dims(c, ns::Integer = 0)
    Q, NS, LD = Int(c[7]), Int(c[8]), Int(c[9])
    n2 = ns == 0 ? LD : min(ns * Q + 2, LD)
    ldx = isodd(n2) ? n2 : n2 + 1               # odd leading dimension: lanes = columns hit 32 banks
    nn = 2Q * (ns == 0 ? NS : ns)
    return (ldx = ldx, nx = n2 + 2, ldr = n2, nn = nn)
end
warp3h_shared_bytes(d::NamedTuple, ::Type{T}) where {T} = wb3h_nf(d.ldx, d.nx, d.ldr) * sizeof(T) + wb3h_ni(d.nn) * 4
warp3h_shared_bytes(c, ::Type{T}, ns::Integer = 0) where {T} = warp3h_shared_bytes(warp3h_dims(c, ns), T)
