# Test 3: two-flute tool with different helix angles (30°, 45°), uniform pitch, R = 8 mm,
# a/D = 0.05 down milling -> spindle period, every tooth a delay distributed linearly over
# the axial depth. Hill determinant with (a) the exact (closed-form) kernel and (b) the
# kernel sampled at n_s axial points, vs the time-domain reference (RK4 monodromy, 32
# slices). Writes hill/results/test3_*.
#   julia --project=gpu/validate -t auto hill/test3_helix.jl [nx ny]
include(joinpath(@__DIR__, "milling_core.jl"))
include(joinpath(@__DIR__, "png.jl"))
using Printf, Statistics, Base.Threads

const OUT = mkpath(joinpath(@__DIR__, "results"))
m = Mill3()
nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 120
ny = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 60
rpms = range(8000.0, 30000.0; length = nx)
aps = range(0.05, 10.0; length = ny)
Ωof(rpm) = rpm / (60 * 922)
TOL = 1e-3

mill_count(m, Ωof(15000), 2.0; tol = TOL); mill_count(m, Ωof(15000), 2.0; tol = TOL, ns = 4)
function chart(; ns = 0)
    Z = zeros(Int, nx, ny); NN = zeros(Int, nx, ny); ER = zeros(nx, ny); TH = zeros(nx, ny); Zr = zeros(nx, ny)
    t = @elapsed @threads for j in 1:ny
        for i in 1:nx
            r = mill_count(m, Ωof(rpms[i]), aps[j]; tol = TOL, ns = ns)
            Z[i, j], NN[i, j], ER[i, j], TH[i, j], Zr[i, j] = r.Z, r.N, r.err, r.θ, r.Zraw
        end
    end
    return Z, NN, ER, TH, Zr, t
end
Z, NN, ER, TH, Zr, t_exact = chart()
@printf("chart %dx%d, exact kernel: %.2f s (%d threads); N %d..%d (median %d); err est max %.1e; max |Zraw-round| %.1e; unstable %.1f %%\n",
    nx, ny, t_exact, nthreads(), minimum(NN), maximum(NN), median(NN), maximum(ER), maximum(abs.(Zr .- round.(Zr))),
    100count(>(0), Z) / length(Z))

# kernel approximation from n_s axial sample points
kern_rows = String["ns,time_s,differs_from_exact,stability_differs"]
for ns in (1, 2, 4, 8, 16)
    Zs, _, _, _, _, t = chart(; ns)
    d = count(Zs .!= Z); ds = count((Zs .== 0) .!= (Z .== 0))
    push!(kern_rows, @sprintf("%d,%.2f,%d,%d", ns, t, d, ds))
    @printf("kernel from %2d axial points: %.2f s, differs from the exact kernel at %d points (stability %d)\n", ns, t, d, ds)
end
write(joinpath(OUT, "test3_kernel_approx.csv"), join(kern_rows, "\n") * "\n")

# time-domain reference on a coarse sub-grid (every 4th point)
ii = 1:4:nx; jj = 1:4:ny
ZR = fill(-9, nx, ny)
cells = [(i, j) for i in ii, j in jj]
t_ref = @elapsed @threads for k in eachindex(cells)
    i, j = cells[k]
    ZR[i, j] = ref_general(m, Ωof(rpms[i]), aps[j]; msteps = 400, ns = 32)[1]
end
sub = [(i, j) for i in ii, j in jj]
wr = count(Z[i, j] != ZR[i, j] for (i, j) in sub); ws = count((Z[i, j] == 0) != (ZR[i, j] == 0) for (i, j) in sub)
@printf("reference (RK4 monodromy, 400 steps/rev, 32 slices) on %d points: %.1f s; count differs %d, stability differs %d\n",
    length(sub), t_ref, wr, ws)
bad = [(rpms[i], aps[j], Z[i, j], ZR[i, j]) for (i, j) in sub if Z[i, j] != ZR[i, j]]
for b in bad[1:min(end, 8)]
    @printf("   rpm %.0f ap %.2f: Hill %d, reference %d\n", b...)
end

# boundary types
BT = zeros(Int8, nx, ny)
for j in 1:ny, i in 1:nx
    nb = (i < nx && (Z[i+1, j] == 0) != (Z[i, j] == 0)) || (j < ny && (Z[i, j+1] == 0) != (Z[i, j] == 0))
    nb || continue
    θ = TH[i, j]
    BT[i, j] = min(θ, 1 - θ) < 0.03 ? 1 : (abs(θ - 0.5) < 0.03 ? 2 : 3)
end
@printf("boundary pixels: μ=+1 %d, flip %d, Neimark-Sacker %d\n", count(==(1), BT), count(==(2), BT), count(==(3), BT))

reds = [(255, 255, 255), (252, 187, 161), (251, 106, 74), (203, 24, 29), (103, 0, 13)]
img(f) = (A = zeros(UInt8, 3, nx, ny); for j in 1:ny, i in 1:nx; A[:, i, j] .= UInt8.(f(i, ny + 1 - j)); end; A)
writepng(joinpath(OUT, "test3_chart.png"), img((i, j) -> begin
    ZR[i, j] != -9 && ZR[i, j] != Z[i, j] ? (0, 160, 255) :
    ZR[i, j] != -9 ? (ZR[i, j] == 0 ? (0, 170, 0) : (0, 0, 0)) : reds[clamp(Z[i, j], 0, 4) + 1]
end))   # colour: Hill count; reference sub-grid dots: green stable / black unstable / blue mismatch
nmin, nmax = minimum(NN), maximum(NN)
writepng(joinpath(OUT, "test3_N.png"), img((i, j) -> (t = (NN[i, j] - nmin) / max(nmax - nmin, 1);
    (round(Int, 40 + 200t), round(Int, 80 + 100t), round(Int, 200 - 150t)))))
bcol = [(255, 255, 255), (0, 150, 0), (220, 0, 0), (0, 0, 220)]
writepng(joinpath(OUT, "test3_boundary_type.png"), img((i, j) -> BT[i, j] > 0 ? bcol[BT[i, j] + 1] :
    (Z[i, j] == 0 ? (225, 225, 225) : (255, 255, 255))))
open(joinpath(OUT, "test3_summary.txt"), "w") do io
    @printf(io, "grid %dx%d, rpm 8000..30000, a_p 0.05..10 mm, helix 30°/45°, R 8 mm, a/D 0.05 down, tol %.0e\n", nx, ny, TOL)
    @printf(io, "exact kernel: %.2f s on %d threads; N %d..%d (median %d); error estimate max %.1e\n", t_exact, nthreads(), nmin, nmax, median(NN), maximum(ER))
    for r in kern_rows[2:end]; println(io, "kernel approx (ns,time,differs,stab differs): ", r); end
    @printf(io, "reference on %d points (%.1f s): count differs %d, stability differs %d\n", length(sub), t_ref, wr, ws)
    @printf(io, "boundary pixels: μ=+1 %d, flip %d, Neimark-Sacker %d\n", count(==(1), BT), count(==(2), BT), count(==(3), BT))
end
println("written to ", OUT)
