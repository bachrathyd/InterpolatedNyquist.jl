# Benchmark of the compressed (Q x Q) milling characteristic function on the GPU (Test 2).
#   julia --project=gpu/scripts hill/gpu_fast_bench.jl [--cpu] [--res 1920x1080] [--check 160x80]
include(joinpath(@__DIR__, "..", "gpu", "scripts", "common.jl"))
include(joinpath(@__DIR__, "gpu_fast.jl"))

nx, ny = parse_res(arg("res", "1920x1080"))
cx, cy = parse_res(arg("check", "160x80"))
print_device()
xr, yr = (5.0, 25.0), (0.0, 5.0)          # spindle speed [1000 rpm], depth of cut [mm]
kwq(T, TE = T) = (n_power = 0, ω0 = 1e-9, ω_max = 0.5, h0 = 1e-3, hrel = 0.05, nroots = 1, T = T,
                  Teval = TE)

cpts = grid_points(range(xr...; length = cx), range(yr...; length = cy))
ref = sweep(D_mill2s, cpts; c = mill2q_consts(Q = 64), backend = CPU(), kwq(Float64)...)
Zref = round.(Int, ref.Zraw)
println("reference: CPU, Float64, semiseparable Q = 64, $(cx)x$(cy) points, unstable $(round(100count(>(0), Zref) / length(Zref); digits = 1)) %")
cases = [(:mid, 8), (:mid, 12), (:mid, 16), (:mid, 24), (:mid, 32), (:semi, 8)]
for (kind, Q) in cases, (T, TE) in ((Float32, Float32), (Float32, Float16))
    D = kind === :mid ? D_mill2m : (kind === :semi ? D_mill2s : D_mill2q)
    c = kind === :mid ? mill2m_consts(Q = Q) : mill2q_consts(Q = Q)
    lab = (kind === :mid ? "mid  " : kind === :semi ? "O(Q) " : "LU   ") * (TE === T ? string(T) : "$(T)/$(TE)")
    try
        r = sweep(D, cpts; c = c, backend = BACKEND, lanes = default_lanes(), kwq(T, TE)...)
        Z = round.(Int, r.Zraw)
        g = plan_grid(xr, yr, nx, ny; backend = BACKEND, lanes = default_lanes(), kwq(T, TE)...)
        run!(g, D, c)
        ts = [timed(() -> run!(g, D, c)) for _ in 1:5]
        rr = fetch_result(g)
        @printf("Q=%2d %-22s check: differs %4d/%d (failed %d) | %dx%d: kernel %8.2f ms (%7.1f Mpts/s, %6.1f fps), evals median %d\n",
            Q, lab, count(Z .!= Zref), length(Zref), count(isnan, r.Zraw), nx, ny, 1e3 * median(ts),
            nx * ny / median(ts) / 1e6, 1 / median(ts), round(Int, median(rr.evals)))
    catch err
        println("Q=$Q $lab: ", first(sprint(showerror, err), 300))
    end
end
