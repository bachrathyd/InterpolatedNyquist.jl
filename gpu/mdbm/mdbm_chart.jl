# Stability chart by the Multi-Dimensional Bisection Method with GPU batches.
#
# MDBM (bachrathyd/MDBM.jl) traces the stability boundary: it starts from a
# coarse grid and bisects only the n-cubes that bracket a sign change of the
# stability objective
#     g = (Z == 0 ? +1 : -1) * |σ_dom|      (continuous across the boundary)
# With MDBM_Problem(...; vectorized = fv) every MDBM stage (initial grid, each
# refinement, each neighbour check) evaluates ALL its new points with ONE
# NyquistGPU sweep -- one kernel launch. The coarse grid doubles as the
# background colouring (counts Z and rightmost-root estimates σ).
#
# The result is compared with the brute-force chart on the grid that MDBM's
# finest level corresponds to: (nx0-1)·2^L + 1 points per axis.
#
# Run:  julia --project=gpu/mdbm gpu/mdbm/mdbm_chart.jl [--system fourth] [--coarse 241x136] [--levels 3]

include(joinpath(@__DIR__, "..", "scripts", "common.jl"))
using MDBM

const SYS = SYSTEMS[arg("system", "fourth")]
const NX0, NY0 = parse_res(arg("coarse", "241x136"))
const LEVELS = parse(Int, arg("levels", "3"))
const OUT = arg("out", joinpath(@__DIR__, "..", "results", "maps"))
mkpath(OUT)
print_device()

const KW = (c = SYS.c, backend = BACKEND, T = Float32, n_power = SYS.npow, nroots = 4,
            schedule = ON_GPU ? :pixel : :queue, lanes = default_lanes(), SYS.kw...)

"stability objective: > 0 stable (distance of the rightmost root), < 0 unstable"
objective(Z, σ) = Z == 0 ? (isfinite(σ) ? max(-σ, 1e-9) : 1.0) :
                  -(isfinite(σ) ? max(abs(σ), 1e-9) : 1.0)

# the vectorized evaluator MDBM calls once per stage; it also keeps (Z, σ)
const SIDE = Dict{NTuple{2, Float64}, Tuple{Int, Float64}}()
const STATS = (calls = Ref(0), points = Ref(0), time = Ref(0.0))
function fv(pts::AbstractVector)
    t = @elapsed begin
        r = sweep(SYS.D, pts; KW...)
        any(!=(0), r.flags) && recheck!(r, SYS.D, pts; merge(KW, (T = Float64,))...)
    end
    STATS.calls[] += 1
    STATS.points[] += length(pts)
    STATS.time[] += t
    out = Vector{Float64}(undef, length(pts))
    for (i, p) in enumerate(pts)
        SIDE[(Float64(p[1]), Float64(p[2]))] = (r.Z[i], Float64(r.sigma[i]))
        out[i] = objective(r.Z[i], r.sigma[i])
    end
    return out
end
f_scalar(x, y) = fv([(x, y)])[1]

xs0 = collect(range(SYS.xr...; length = NX0))
ys0 = collect(range(SYS.yr...; length = NY0))
nxf, nyf = (NX0 - 1) * 2^LEVELS + 1, (NY0 - 1) * 2^LEVELS + 1
@printf("\n%s: MDBM from a %dx%d grid, %d bisection levels -> %dx%d equivalent resolution\n",
    SYS.title, NX0, NY0, LEVELS, nxf, nyf)

# warm-up: compile the sweeps (Float32 + Float64 recheck) and MDBM's own
# methods for these types on a tiny problem, outside the timer
sweep(SYS.D, [(xs0[1], ys0[1])]; merge(KW, (T = Float64,))...)
let m = MDBM_Problem(f_scalar, [Axis(xs0[1:8:end], :x), Axis(ys0[1:8:end], :y)]; vectorized = fv)
    solve!(m, 1; interpolationorder = 0, verbosity = 0)
    interpolate!(m; interpolationorder = 1)
end
empty!(SIDE)
STATS.calls[] = 0; STATS.points[] = 0; STATS.time[] = 0.0

t_mdbm = @elapsed begin
    mdbm = MDBM_Problem(f_scalar, [Axis(xs0, :x), Axis(ys0, :y)]; vectorized = fv)
    # bracketing by a TRUE sign change (order 0, as in the paper's pipeline):
    # with order 1 MDBM also keeps cubes where the linear fit merely predicts a
    # zero nearby, and g = -|σ_dom| touches zero inside unstable regions
    # wherever the tracked root estimate passes 0 -> spurious curves
    solve!(mdbm, LEVELS; interpolationorder = 0, verbosity = 0,
        checkneighbourNum = parse(Int, arg("neighbour", "1")))
    interpolate!(mdbm; interpolationorder = 1)        # positions by the linear fit
end
sol = getinterpolatedsolution(mdbm)
nbnd = length(sol[1])
@printf("  MDBM:        %8.1f ms total, of which %8.1f ms in %d batched sweeps (%d points = %.1f %% of %dx%d)\n",
    1e3t_mdbm, 1e3STATS.time[], STATS.calls[], STATS.points[],
    100STATS.points[] / (nxf * nyf), nxf, nyf)
@printf("               %d boundary points\n", nbnd)

# brute force on the equivalent grid (same tick positions as MDBM's finest level)
xsf = range(SYS.xr...; length = nxf)
ysf = range(SYS.yr...; length = nyf)
ptsf = grid_points(xsf, ysf)
sweep(SYS.D, ptsf[1:4]; KW...)
t_bf = @elapsed (rbf = sweep(SYS.D, ptsf; KW...))
@printf("  brute force: %8.1f ms for %d points\n", 1e3t_bf, nxf * nyf)
@printf("  -> MDBM evaluates %.1fx fewer points; wall-clock ratio %.2fx\n",
    nxf * nyf / STATS.points[], t_bf / t_mdbm)

# accuracy: is every MDBM boundary point next to a brute-force boundary pixel,
# and is every brute-force boundary pixel next to an MDBM boundary point?
S = reshape(rbf.Z .== 0, nxf, nyf)
bnd = falses(nxf, nyf)
for j in 1:nyf, i in 1:nxf
    i < nxf && S[i, j] != S[i + 1, j] && (bnd[i, j] = bnd[i + 1, j] = true)
    j < nyf && S[i, j] != S[i, j + 1] && (bnd[i, j] = bnd[i, j + 1] = true)
end
dx, dy = step(xsf), step(ysf)
mark = falses(nxf, nyf)
# pixel distance of each MDBM point to the nearest brute-force boundary pixel (capped at 16)
bidx = Tuple.(findall(bnd))
dist = Float64[]
for k in 1:nbnd
    i = clamp(round(Int, (sol[1][k] - xsf[1]) / dx) + 1, 1, nxf)
    j = clamp(round(Int, (sol[2][k] - ysf[1]) / dy) + 1, 1, nyf)
    mark[i, j] = true
    d = 16.0
    for jj in max(1, j - 16):min(nyf, j + 16), ii in max(1, i - 16):min(nxf, i + 16)
        bnd[ii, jj] && (d = min(d, hypot(ii - i, jj - j)))
    end
    push!(dist, d)
end
near_ok = count(<=(1.5), dist)
@printf("  distance of MDBM points to the brute-force boundary [px]: median %.1f, p90 %.1f, max %.1f; within 1.5 px: %.1f %%, within 3 px: %.1f %%
",
    median(dist), quantile(dist, 0.9), maximum(dist), 100near_ok / nbnd, 100count(<=(3), dist) / nbnd)
sizes = [Int(maximum(nc.size)) for nc in mdbm.ncubes]
@printf("  final MDBM n-cube sizes (in finest-grid cells): %s
",
    join(["$(s)x: $(count(==(s), sizes))" for s in sort(unique(sizes))], ", "))
covered = count(I -> bnd[I] && any(mark[ii, jj] for ii in max(1, I[1] - 2):min(nxf, I[1] + 2),
                                    jj in max(1, I[2] - 2):min(nyf, I[2] + 2)), CartesianIndices(bnd))
@printf("  accuracy:    %.2f %% of MDBM boundary points within 1 pixel of the brute-force boundary\n",
    100near_ok / max(nbnd, 1))
@printf("               %.2f %% of brute-force boundary pixels within 2 pixels of an MDBM point\n",
    100covered / max(count(bnd), 1))

# outputs: coarse background field + boundary points
Zc = [SIDE[(x, y)][1] for x in xs0, y in ys0]
σc = [SIDE[(x, y)][2] for x in xs0, y in ys0]
save_field(OUT, SYS, (Z = Zc, sigma = σc), NX0, NY0, 1e3t_mdbm; tag = "_mdbm")
open(joinpath(OUT, "boundary_$(SYS.name)_mdbm.csv"), "w") do io
    println(io, "x,y")
    foreach(k -> println(io, sol[1][k], ",", sol[2][k]), 1:nbnd)
end
save_field(OUT, SYS, (Z = reshape(rbf.Z, nxf, nyf), sigma = reshape(rbf.sigma, nxf, nyf)),
    nxf, nyf, 1e3t_bf; tag = "_bf")
println("  wrote boundary_$(SYS.name)_mdbm.csv")
