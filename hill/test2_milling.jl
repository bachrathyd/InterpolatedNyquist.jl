# Test 2: straight-fluted 1-DOF milling (Insperger-Stépán benchmark: z = 2, a/D = 0.05
# down milling, K_n/K_t = 1/3, ζ = 0.011, f_n = 922 Hz) -- Hill determinant + argument
# principle vs the RK4 monodromy reference. Writes hill/results/test2_*.
#   julia --project=gpu/validate -t auto hill/test2_milling.jl [nx ny]
include(joinpath(@__DIR__, "milling_core.jl"))
include(joinpath(@__DIR__, "png.jl"))
using Printf, Statistics, Base.Threads

const OUT = mkpath(joinpath(@__DIR__, "results"))
m = Mill2()
nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
ny = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
rpms = range(5000.0, 25000.0; length = nx)
aps = range(0.02, 5.0; length = ny)
Ωof(rpm) = rpm / (60 * 922)
TOL = 1e-3

mill_count(m, Ωof(10000), 2.0; tol = TOL); ref_tooth(m, Ωof(10000), 2.0)
Z = zeros(Int, nx, ny); Zr = zeros(nx, ny); NN = zeros(Int, nx, ny); ER = zeros(nx, ny)
EV = zeros(Int, nx, ny); TH = zeros(nx, ny)
t_hill = @elapsed @threads for j in 1:ny
    for i in 1:nx
        r = mill_count(m, Ωof(rpms[i]), aps[j]; tol = TOL)
        Z[i, j], Zr[i, j], NN[i, j], ER[i, j], EV[i, j], TH[i, j] = r.Z, r.Zraw, r.N, r.err, r.evals, r.θ
    end
end
ZR = zeros(Int, nx, ny)
t_ref = @elapsed @threads for j in 1:ny
    for i in 1:nx
        ZR[i, j] = ref_tooth(m, Ωof(rpms[i]), aps[j]; m1 = 80, m2 = 80)[1]
    end
end
wrong = count(Z .!= ZR); wrong_stab = count((Z .== 0) .!= (ZR .== 0))
@printf("chart %dx%d: Hill+AP %.2f s (%d threads), reference %.2f s\n", nx, ny, t_hill, nthreads(), t_ref)
@printf("count differs: %d of %d (stability: %d); max |Zraw - round| %.2e; failed %d\n", wrong, nx * ny,
    wrong_stab, maximum(abs.(Zr .- round.(Zr))), count(<(0), Z))
@printf("N: %d..%d, median %d; error estimate max %.1e; evals median %d max %d\n", minimum(NN), maximum(NN),
    median(NN), maximum(ER), median(EV), maximum(EV))

# the differing points, re-checked with a finer reference (320 steps) and N + 10 harmonics
for I in findall(Z .!= ZR)[1:min(end, 16)]
    i, j = Tuple(I)
    Ω = Ωof(rpms[i])
    zf, ρf = ref_tooth(m, Ω, aps[j]; m1 = 160, m2 = 160)
    zN = mill_count(m, Ω, aps[j]; N = NN[i, j] + 10, Nmax = NN[i, j] + 10).Z
    @printf("   rpm %6.0f ap %.3f: Hill %d (N %d; N+10: %d), reference %d, finer reference %d (rho %.5f)
",
        rpms[i], aps[j], Z[i, j], NN[i, j], zN, ZR[i, j], zf, ρf)
end

# boundary type from the crossing exponent of the tracked |Δ| minimum (μ = Im λ/ω_p mod 1)
BT = zeros(Int8, nx, ny)
for j in 1:ny, i in 1:nx
    nb = (i < nx && (Z[i+1, j] == 0) != (Z[i, j] == 0)) || (j < ny && (Z[i, j+1] == 0) != (Z[i, j] == 0))
    nb || continue
    θ = TH[i, j]
    BT[i, j] = min(θ, 1 - θ) < 0.03 ? 1 : (abs(θ - 0.5) < 0.03 ? 2 : 3)
end
@printf("boundary pixels: μ=+1 %d, flip %d, Neimark-Sacker %d\n", count(==(1), BT), count(==(2), BT), count(==(3), BT))

# convergence in N at a few speeds (low speed needs many harmonics)
rows = String["rpm,ap,N,est_err,true_err,Z,Zref"]
for (rpm, ap) in ((6000.0, 1.0), (6000.0, 3.0), (11000.0, 2.0), (15000.0, 1.5), (22000.0, 3.0))
    Ω = Ωof(rpm)
    P = setup(m, Ω, ap, 61)
    zr = ref_tooth(m, Ω, ap; m1 = 80, m2 = 80)[1]
    for N in 1:24
        e = ring_change(N, m, P)
        tr = 0.0
        for t in (0.0, 0.25, 0.5, 0.75)
            λ = 1im * (0.237 + t) * P.ωp
            tr = max(tr, abs(mill_det(λ, N, m, P) / mill_det(λ, 60, m, P) - 1))
        end
        zN = mill_count(m, Ω, ap; N = N).Z
        push!(rows, @sprintf("%.0f,%.2f,%d,%.3e,%.3e,%d,%d", rpm, ap, N, e, tr, zN, zr))
    end
end
write(joinpath(OUT, "test2_convergence.csv"), join(rows, "\n") * "\n")
for r in rows[2:end]
    v = split(r, ','); parse(Int, v[3]) in (2, 4, 6, 8, 12, 16, 20, 24) && println(r)
end

reds = [(255, 255, 255), (252, 187, 161), (251, 106, 74), (203, 24, 29), (103, 0, 13)]
img(f) = (A = zeros(UInt8, 3, nx, ny); for j in 1:ny, i in 1:nx; A[:, i, j] .= UInt8.(f(i, ny + 1 - j)); end; A)
writepng(joinpath(OUT, "test2_chart.png"), img((i, j) -> begin
    bref = (i < nx && (ZR[i+1, j] == 0) != (ZR[i, j] == 0)) || (j < ny && (ZR[i, j+1] == 0) != (ZR[i, j] == 0))
    Z[i, j] != ZR[i, j] ? (0, 160, 255) : (bref ? (0, 0, 0) : reds[clamp(Z[i, j], 0, 4) + 1])
end))
nmin, nmax = minimum(NN), maximum(NN)
writepng(joinpath(OUT, "test2_N.png"), img((i, j) -> (t = (NN[i, j] - nmin) / max(nmax - nmin, 1);
    (round(Int, 40 + 200t), round(Int, 80 + 100t), round(Int, 200 - 150t)))))
lo, hi = log10(minimum(ER)), log10(maximum(ER))
writepng(joinpath(OUT, "test2_errest.png"), img((i, j) -> (t = (log10(ER[i, j]) - lo) / max(hi - lo, 1e-9);
    (round(Int, 255t), round(Int, 255t), round(Int, 255(1 - t))))))
bcol = [(255, 255, 255), (0, 150, 0), (220, 0, 0), (0, 0, 220)]
writepng(joinpath(OUT, "test2_boundary_type.png"), img((i, j) -> BT[i, j] > 0 ? bcol[BT[i, j] + 1] :
    (Z[i, j] == 0 ? (225, 225, 225) : (255, 255, 255))))
open(joinpath(OUT, "test2_summary.txt"), "w") do io
    @printf(io, "grid %dx%d, rpm 5000..25000, a_p 0.02..5 mm, tol %.0e\n", nx, ny, TOL)
    @printf(io, "Hill+AP %.2f s on %d threads; reference (RK4 monodromy, 160 steps/period) %.2f s\n", t_hill, nthreads(), t_ref)
    @printf(io, "count differs %d / %d (stability %d)\n", wrong, nx * ny, wrong_stab)
    @printf(io, "N %d..%d (median %d); error estimate max %.1e; evals median %d max %d\n", nmin, nmax, median(NN), maximum(ER), median(EV), maximum(EV))
    @printf(io, "boundary pixels: μ=+1 %d, flip %d, Neimark-Sacker %d\n", count(==(1), BT), count(==(2), BT), count(==(3), BT))
end
println("written to ", OUT)
