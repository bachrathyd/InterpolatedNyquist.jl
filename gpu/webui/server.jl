# The browser page of webgpu/ (the same page) with the march on the Colab GPU: this server serves
# the page and answers its chart requests with NyquistGPU (CUDA). In a Colab cell:
#     julia -t auto --project=gpu/webui gpu/webui/server.jl 8800      (background, see the notebook)
#     from google.colab import output; output.serve_kernel_port_as_iframe(8800, height=1100)
# The page detects the server (GET api/info) and sends its 2-D charts here; the colouring stays
# on the viewer's WebGPU (display.wgsl), 3-D views and integral(...) by quadrature stay local.
#
# POST api/march (JSON) {src (Julia `D(λ, p, c)`, webgpu/expr.js julia()), c, x0, x1, y0, y1, nx, ny,
#                        npow, w0, wmax, tol, hmax, wband, maxsteps, exact, prec ("F32" | "F64")}
# -> nx·ny records of 16 bytes as the page's march.wgsl writes them: Float32 z (Z_raw, 0 if the
#    march failed), Float32 s (σ, -3e38 if none), UInt32 steps, UInt32 flags (1 failed, 2 sub-
#    resolution, 4 residual, 8 impossible count, 128 no count, 256 σ valid)

using HTTP, JSON3, NyquistGPU, KernelAbstractions
const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
const BACKEND = HAS_CUDA ? CUDA.CUDABackend(always_inline = true) : KernelAbstractions.CPU()
const ROOT = normpath(joinpath(@__DIR__, "..", "..", "webgpu"))
const LOCK = ReentrantLock()

# helpers of the page's closed forms (as webgpu/validate/web_systems.jl)
const HELPERS = raw"""
function E1T(x::Complex{S}) where {S}
    o = one(S)
    if abs(x) < o / 2
        return o + x * (-o / 2 + x * (o / 6 + x * (-o / 24 + x * (o / 120 + x * (-o / 720 +
               x * (o / 5040 + x * (-o / 40320)))))))
    end
    return (o - exp(-x)) / x
end
exprelT(x) = E1T(-x)
function phik(w::Complex{S}, k) where {S}
    if abs(w) < 2 + k
        c = one(S)
        for i in 1:(23 + k)
            c /= S(i)
        end
        r = Complex{S}(c)
        for j in 22:-1:0
            c *= S(j + 1 + k)
            r = w * r + c
        end
        return r
    end
    r = exp(w)
    f = one(S)
    for i in 1:k
        r = (r - f) / w
        f /= S(i)
    end
    return r
end
"""

const MODELS = Dict{String, Any}()
function model_of(src::String)
    get!(MODELS, src) do
        m = Module(:NyqModel)
        Base.include_string(m, HELPERS * src)
        Base.invokelatest(getproperty, m, :D)
    end
end

# plans by size, precision and the settings a plan fixes (the rest: regrid! / set_march!)
const PLANS = Dict{Any, Any}()
const PLAN_ORDER = Any[]
function plan_for(nx, ny, T, q)
    key = (nx, ny, T, q.npow, q.w0, q.maxsteps, q.hmax, q.wband, q.exact)
    p = get(PLANS, key, nothing)
    if p === nothing
        while length(PLAN_ORDER) >= 6                   # keep the device memory bounded
            delete!(PLANS, popfirst!(PLAN_ORDER))
        end
        p = plan_grid((q.x0, q.x1), (q.y0, q.y1), nx, ny; backend = BACKEND, T = T, n_power = q.npow,
                      ω0 = q.w0, ω_max = q.wmax, tol = q.tol, hmax = q.hmax, ωband = q.wband,
                      maxsteps = q.maxsteps, refine = q.exact ? 4 : 0)
        PLANS[key] = p; push!(PLAN_ORDER, key)
    end
    return p
end

fin(v, big) = (v === nothing || !isfinite(Float64(v))) ? big : Float64(v)

function march(req)
    D = model_of(String(req.src))
    T = String(get(req, :prec, "F32")) == "F64" ? Float64 : Float32
    nx = Int(req.nx); ny = Int(req.ny)
    q = (x0 = Float64(req.x0), x1 = Float64(req.x1), y0 = Float64(req.y0), y1 = Float64(req.y1),
         npow = Float64(req.npow), w0 = Float64(req.w0), wmax = Float64(req.wmax), tol = Float64(req.tol),
         hmax = fin(req.hmax, Inf), wband = fin(req.wband, 0.0), maxsteps = Int(req.maxsteps), exact = Bool(req.exact))
    p = plan_for(nx, ny, T, q)
    regrid!(p, (q.x0, q.x1), (q.y0, q.y1), nx, ny)
    set_march!(p; ω_max = q.wmax, tol = q.tol)
    c = Tuple(T.(Float64.(req.c)))
    t0 = time()
    Base.invokelatest(run!, p, D, c)
    ms = 1000 * (time() - t0)
    r = fetch_result(p)
    n = nx * ny
    out = Vector{UInt32}(undef, 4n)
    @inbounds for k in 1:n
        z = r.Zraw[k]; s = r.sigma[k]
        fl = UInt32(reinterpret(UInt8, r.flags[k])) & 0x0f
        failed = !isfinite(z)
        failed && (fl |= 0x81)
        isfinite(s) && (fl |= 0x100)
        out[4k-3] = reinterpret(UInt32, failed ? 0f0 : Float32(z))
        out[4k-2] = reinterpret(UInt32, isfinite(s) ? Float32(s) : -3f38)
        out[4k-1] = UInt32(max(r.steps[k], 0))
        out[4k] = fl
    end
    return out, ms
end

bodybytes(req) = (b = req.body; b isa AbstractVector{UInt8} ? b : b.data)    # HTTP.jl 1.x / 2.x

const MIME = Dict(".html" => "text/html; charset=utf-8", ".js" => "text/javascript; charset=utf-8",
                  ".mjs" => "text/javascript; charset=utf-8", ".wgsl" => "text/plain; charset=utf-8",
                  ".json" => "application/json", ".md" => "text/plain; charset=utf-8", ".png" => "image/png")

function handle(req::HTTP.Request)
    path = HTTP.unescapeuri(HTTP.URI(req.target).path)
    if endswith(path, "/api/info")
        name = HAS_CUDA ? CUDA.name(CUDA.device()) : "CPU (no CUDA)"
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                             JSON3.write((device = "Colab: " * name, threads = Threads.nthreads(), cuda = HAS_CUDA)))
    elseif endswith(path, "/api/march") && req.method == "POST"
        try
            out, ms = lock(LOCK) do
                march(JSON3.read(bodybytes(req)))
            end
            return HTTP.Response(200, ["Content-Type" => "application/octet-stream", "X-Compute-Ms" => string(round(ms, digits = 1)),
                                       "Access-Control-Expose-Headers" => "X-Compute-Ms"], collect(reinterpret(UInt8, out)))
        catch err
            @error "march failed" exception = (err, catch_backtrace())
            return HTTP.Response(500, ["Content-Type" => "text/plain"], first(sprint(showerror, err), 2000))
        end
    end
    rel = lstrip(path, '/')
    full = normpath(joinpath(ROOT, isempty(rel) ? "index.html" : rel))
    (startswith(full, ROOT) && isfile(full)) || return HTTP.Response(404, "not found")
    ext = lowercase(splitext(full)[2])
    return HTTP.Response(200, ["Content-Type" => get(MIME, ext, "application/octet-stream"), "Cache-Control" => "no-cache"],
                         read(full))
end

port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8800
println("NyquistGPU web UI: device ", HAS_CUDA ? CUDA.name(CUDA.device()) : "CPU", ", serving $ROOT on port $port")
# warm-up: the fourth-order example (compiles the march kernels for Float32 and Float64)
let src = "function D(λ, p, c)\n    return c[1] * λ^4 + λ * λ + c[6] * c[2] * λ + c[7] + (p[1] + p[2] * λ) * exp(-c[5] * λ)\nend\n"
    for prec in ("F32", "F64")
        try
            req = JSON3.read(JSON3.write((src = src, c = [0.03, 0.02, 0.0, 0.0, 0.5, 2.0, 1.0], x0 = -2, x1 = 4, y0 = -2, y1 = 5,
                                          nx = 16, ny = 9, npow = 4, w0 = 1e-9, wmax = 1e5, tol = 0.3, hmax = nothing,
                                          wband = 0, maxsteps = 50000, exact = true, prec = prec)))
            march(req); println("warm-up ($prec) done")
        catch err
            @warn "warm-up failed" exception = err
        end
    end
end
println("ready")
flush(stdout)
HTTP.serve(handle, "0.0.0.0", port)
