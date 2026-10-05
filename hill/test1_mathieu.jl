# Test 1: delayed Mathieu equation -- Hill determinant + argument principle vs the
# RK4 monodromy reference. Writes hill/results/test1_*.{csv,png}.
#   julia --project=gpu/validate -t auto hill/test1_mathieu.jl [nx ny]
include(joinpath(@__DIR__, "hill_core.jl"))
include(joinpath(@__DIR__, "png.jl"))
using Printf, Statistics, Base.Threads

const OUT = mkpath(joinpath(@__DIR__, "results"))
m = Mathieu()                     # κ = 0.1, ε = 1, τ = T = 2π
nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 240
ny = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 120
δs = range(-1.0, 5.0; length = nx)
bs = range(-1.5, 1.5; length = ny)
TOL = 1e-4                        # the single user-facing tolerance (relative, on Δ)

# --- warm-up + chart -----------------------------------------------------------
floquet_count_pf(1.0, 0.1, m; tol = TOL)
floquet_count(1.0, 0.1, m; tol = TOL)
reference_count(1.0, 0.1, m)
ANG = zeros(nx, ny)
Z = zeros(Int, nx, ny); Zr = zeros(nx, ny); NN = zeros(Int, nx, ny); ER = zeros(nx, ny); EV = zeros(Int, nx, ny)
t_hill = @elapsed @threads for j in 1:ny
    for i in 1:nx
        r = floquet_count_pf(δs[i], bs[j], m; tol = TOL)
        Z[i, j], Zr[i, j], NN[i, j], ER[i, j], EV[i, j] = r.Z, r.Zraw, r.N, r.err, r.evals
        ANG[i, j] = mod(r.ω / m.ωp, 1.0)
    end
end
ZH = zeros(Int, nx, ny)            # Hill's normalization (rows / d_k) + LTI pole count
t_hillnorm = @elapsed @threads for j in 1:ny
    for i in 1:nx
        ZH[i, j] = floquet_count(δs[i], bs[j], m; tol = TOL).Z
    end
end
ZR = zeros(Int, nx, ny); RHO = zeros(nx, ny)
t_ref = @elapsed @threads for j in 1:ny
    for i in 1:nx
        ZR[i, j], RHO[i, j] = reference_count(δs[i], bs[j], m)
    end
end
wrong = count(Z .!= ZR)
wrong_stab = count((Z .== 0) .!= (ZR .== 0))
resid = maximum(abs.(Zr .- round.(Zr)))
@printf("chart %dx%d: Hill+AP %.2f s (%.1f µs/pt, %d threads), reference %.2f s\n", nx, ny, t_hill,
    1e6t_hill / (nx * ny) * nthreads(), nthreads(), t_ref)
@printf("count differs from reference: %d of %d (stability differs: %d); max |Zraw - round| = %.2e\n",
    wrong, nx * ny, wrong_stab, resid)
@printf("chosen N: %d..%d (median %d); a-posteriori error estimate max %.1e; evals median %d max %d\n",
    minimum(NN), maximum(NN), median(NN), maximum(ER), median(EV), maximum(EV))

# --- boundary type: the crossing multiplier's angle from the tracked |Δ| minimum ----
# (classification on boundary pixels only: Z changes to a neighbour)
BT = zeros(Int8, nx, ny)            # 0 none, 1 μ=+1, 2 flip, 3 Neimark-Sacker
@threads for j in 1:ny
    for i in 1:nx
        nb = (i < nx && (Z[i+1, j] == 0) != (Z[i, j] == 0)) || (j < ny && (Z[i, j+1] == 0) != (Z[i, j] == 0))
        nb || continue
        θ = ANG[i, j]
        d0 = min(θ, 1 - θ); dh = abs(θ - 0.5)
        BT[i, j] = d0 < 0.03 ? 1 : (dh < 0.03 ? 2 : 3)
    end
end
@printf("boundary pixels: μ=+1 %d, flip (μ=-1) %d, Neimark-Sacker %d\n", count(==(1), BT), count(==(2), BT), count(==(3), BT))

# --- convergence study in N on a coarse grid ------------------------------------------
cg = [(δ, b) for δ in range(-1, 5; length = 25), b in range(-1.5, 1.5; length = 13)]
Nref = 60
rows = String["N,median_est_err,max_est_err,median_true_err,max_true_err,wrong_counts,points"]
zref = [reference_count(p..., m)[1] for p in cg]
for N in 1:12
    est = Float64[]; tru = Float64[]; wr = 0
    for (k, (δ, b)) in enumerate(cg)
        push!(est, ring_change(δ, b, N, m))
        e = 0.0
        for t in (0.0, 0.25, 0.5, 0.75)
            λ = 1im * (0.237 + t)
            e = max(e, abs(hill_delta_pf(λ, δ, b, N, m) / hill_delta_pf(λ, δ, b, Nref, m) - 1))
        end
        push!(tru, e)
        wr += floquet_count_pf(δ, b, m; N = N).Z != zref[k]
    end
    push!(rows, @sprintf("%d,%.3e,%.3e,%.3e,%.3e,%d,%d", N, median(est), maximum(est), median(tru), maximum(tru), wr, length(cg)))
    @printf("N=%2d  est err median %.2e max %.2e | true err median %.2e max %.2e | wrong counts %d/%d\n",
        N, median(est), maximum(est), median(tru), maximum(tru), wr, length(cg))
end
write(joinpath(OUT, "test1_convergence.csv"), join(rows, "\n") * "\n")

# --- images ----------------------------------------------------------------------------
reds = [(255, 255, 255), (252, 187, 161), (251, 106, 74), (203, 24, 29), (103, 0, 13)]
img(f) = (A = zeros(UInt8, 3, nx, ny); for j in 1:ny, i in 1:nx; A[:, i, j] .= UInt8.(f(i, ny + 1 - j)); end; A)
chart = img((i, j) -> begin
    c = reds[clamp(Z[i, j], 0, 4) + 1]
    bref = (i < nx && (ZR[i+1, j] == 0) != (ZR[i, j] == 0)) || (j < ny && (ZR[i, j+1] == 0) != (ZR[i, j] == 0))
    Z[i, j] != ZR[i, j] ? (0, 160, 255) : (bref ? (0, 0, 0) : c)
end)
writepng(joinpath(OUT, "test1_chart.png"), chart)      # colour: count (Hill); black: reference boundary; blue: mismatch
nmin, nmax = minimum(NN), maximum(NN)
writepng(joinpath(OUT, "test1_N.png"), img((i, j) -> (t = nmax > nmin ? (NN[i, j] - nmin) / (nmax - nmin) : 0.5;
    (round(Int, 40 + 200t), round(Int, 80 + 100t), round(Int, 200 - 150t)))))
lo, hi = log10(minimum(ER)), log10(maximum(ER))
writepng(joinpath(OUT, "test1_errest.png"), img((i, j) -> (t = (log10(ER[i, j]) - lo) / max(hi - lo, 1e-9);
    (round(Int, 255t), round(Int, 255t), round(Int, 255(1 - t))))))
bcol = [(255, 255, 255), (0, 150, 0), (220, 0, 0), (0, 0, 220)]
writepng(joinpath(OUT, "test1_boundary_type.png"), img((i, j) -> BT[i, j] > 0 ? bcol[BT[i, j] + 1] :
    (Z[i, j] == 0 ? (225, 225, 225) : (255, 255, 255))))
open(joinpath(OUT, "test1_summary.txt"), "w") do io
    @printf(io, "grid %dx%d, δ∈[-1,5], b∈[-1.5,1.5], κ=0.1, ε=1, τ=T=2π, tol=%.0e\n", nx, ny, TOL)
    @printf(io, "Hill+AP: %.3f s on %d threads; reference (RK4 monodromy, 80 steps): %.3f s\n", t_hill, nthreads(), t_ref)
    @printf(io, "counts differing from reference: %d / %d; stability differing: %d\n", wrong, nx * ny, wrong_stab)
    @printf(io, "max |Zraw - round(Zraw)| = %.2e\n", resid)
    @printf(io, "N chosen: %d..%d; error estimate max %.2e; evals median %d max %d\n", nmin, nmax, maximum(ER), median(EV), maximum(EV))
    @printf(io, "boundary pixels: μ=+1 %d, flip %d, Neimark-Sacker %d\n", count(==(1), BT), count(==(2), BT), count(==(3), BT))
end
println("written to ", OUT)
