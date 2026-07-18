# Real-time ("shader-style") stability chart -- GPU version.
#
# Target: a 400x400 chart that recomputes EVERYTHING live while parameter
# sliders move, one thread per pixel, like a fragment shader.
#
#   sliders --> Observable params --> KA kernel sweep (GPU or CPU) --> heatmap
#
# Backend is auto-detected: if CUDA.jl is installed and functional, the sweep
# runs on the GPU at full resolution; otherwise it falls back to the
# multithreaded CPU backend with progressive refinement (a coarse grid while
# dragging, the full grid ~0.5 s after the sliders rest). The kernel itself
# is IDENTICAL on both backends (KernelAbstractions), and was validated
# against the package solver in two_system_validation.jl (showcase: 10000/
# 10000 counts, FP32 included).
#
# Measured on this 16-thread CPU (FP32, tol 1e-4, 4th-order oscillator):
#   100x100: ~47 ms/frame -- 400x400: ~750 ms/frame (not real-time on CPU).
# The 400x400 target is exactly the regime that needs the GPU: with the
# measured 83-86 % warp efficiency and ~650 evaluations/pixel, a mid-range
# CUDA card is expected at 8-40 ms/frame, i.e. genuinely live.
#
# Julia's JIT makes this work for ARBITRARY user-supplied parametric D(lam):
# a new characteristic function specializes the kernel once (seconds), after
# which every slider move is a pure kernel launch -- the shader experience
# without transpiling the model.
#
# Run (GUI):    julia --project=benchmark/gpu -t auto benchmark/gpu/realtime_chart.jl
# Run (bench):  julia --project=benchmark/gpu -t auto benchmark/gpu/realtime_chart.jl --bench

using KernelAbstractions, ForwardDiff, Printf

# ---------------------------------------------------------------------------
# Backend auto-detection: GPU if present, CPU otherwise
# ---------------------------------------------------------------------------
const BACKEND, ON_GPU = let
    be, ok = CPU(), false
    try
        @eval using CUDA
        if CUDA.functional()
            be, ok = CUDABackend(), true
        else
            @info "CUDA.jl present but no functional GPU -- CPU backend"
        end
    catch
        @info "CUDA.jl not installed -- CPU backend (add CUDA to this env on a GPU machine)"
    end
    be, ok
end

const RES_IDLE = 400           # the full-resolution chart
const REFINE_DELAY = 0.5       # s of slider silence before CPU refinement
const T32 = Float32
const TOL = 1f-4
const WMAX = 1f4

struct PhaseTag end

# ---------------------------------------------------------------------------
# The parametric model: 4th-order delayed oscillator.
#   D(lam) = c1 lam^4 + lam^2 + 2 zeta lam + 1 + (P + Dg lam) e^(-lam tau)
# Chart axes: (P, Dg). Sliders: c1, zeta, tau. Any other model drops in the
# same way (see D_of in two_system_validation.jl).
# ---------------------------------------------------------------------------
const NPOW = 4.0
const XR = (-2.0, 4.0)
const YR = (-2.0, 5.0)

@inline function eval_D(::Type{T}, w, P, Dg, c1, zeta, tau) where {T}
    d = ForwardDiff.Dual{PhaseTag}(T(w), one(T))
    lam = Complex(zero(d), d)
    Dv = c1 * lam^4 + lam^2 + 2 * zeta * lam + one(T) +
         (P + Dg * lam) * exp(-tau * lam)
    Dre, Dim = real(Dv), imag(Dv)
    D_re, D_im = ForwardDiff.value(Dre), ForwardDiff.value(Dim)
    Dp_re, Dp_im = ForwardDiff.partials(Dre, 1), ForwardDiff.partials(Dim, 1)
    absD2 = D_re * D_re + D_im * D_im
    th = (D_re * Dp_im - D_im * Dp_re) / absD2
    absDp2 = Dp_re * Dp_re + Dp_im * Dp_im
    sig = -(D_re * Dp_im - D_im * Dp_re) / max(absDp2, T(1e-30))
    return th, absD2, sig
end

@kernel function march!(C, @Const(Pv), @Const(Dv),
        c1::T, zeta::T, tau::T, wmax::T, rtol::T, atol::T) where {T}
    idx = @index(Global)
    P = Pv[idx]; Dg = Dv[idx]
    w = T(1e-9); y = zero(T); h = T(1e-2)
    steps = Int32(0)
    mind2 = T(Inf); sigbest = zero(T)
    th1, d2, sg = eval_D(T, w, P, Dg, c1, zeta, tau)
    if d2 < mind2; mind2 = d2; sigbest = sg; end
    while w < wmax && steps < Int32(200_000)
        h = min(h, wmax - w)
        th2, d2a, sga = eval_D(T, w + h / 2, P, Dg, c1, zeta, tau)
        th3, d2b, sgb = eval_D(T, w + 3 * h / 4, P, Dg, c1, zeta, tau)
        ynew = y + h * (T(2 / 9) * th1 + T(1 / 3) * th2 + T(4 / 9) * th3)
        th4, d2c, sgc = eval_D(T, w + h, P, Dg, c1, zeta, tau)
        zlow = y + h * (T(7 / 24) * th1 + T(1 / 4) * th2 +
                        T(1 / 3) * th3 + T(1 / 8) * th4)
        err = abs(ynew - zlow)
        tol = atol + rtol * abs(ynew)
        steps += Int32(1)
        if err <= tol
            w += h; y = ynew; th1 = th4
            if d2a < mind2; mind2 = d2a; sigbest = sga; end
            if d2b < mind2; mind2 = d2b; sigbest = sgb; end
            if d2c < mind2; mind2 = d2c; sigbest = sgc; end
        end
        fac = T(0.9) * (tol / max(err, T(1e-30)))^T(1 / 3)
        h *= min(max(fac, T(0.2)), T(5.0))
    end
    Zraw = T(NPOW) / 2 - y / T(pi)
    Z = round(Zraw)
    # the paper's interpolable coloring, computed in-kernel:
    # sigma inside the stable domain, the (capped) integer count outside
    C[idx] = Z == 0 ? max(sigbest, T(-1.5)) : min(Z, T(6.0))
end

# ---------------------------------------------------------------------------
# Persistent per-resolution device buffers (allocate once, reuse every frame)
# ---------------------------------------------------------------------------
struct Buffers
    Pz::Vector{Float64}
    Dz::Vector{Float64}
    Ps                    # device vectors (CPU Array or CuArray)
    Dgs
    C
    host::Matrix{Float32} # staging buffer for the heatmap
end
const BUFFERS = Dict{Int, Buffers}()

function to_device(x::Vector{T32})
    d = KernelAbstractions.allocate(BACKEND, T32, length(x))
    copyto!(d, x)
    return d
end

function buffers(res::Int)
    get!(BUFFERS, res) do
        Pz = collect(range(XR...; length = res))
        Dz = collect(range(YR...; length = res))
        Ps = T32.(vec([p for p in Pz, d in Dz]))
        Dgs = T32.(vec([d for p in Pz, d in Dz]))
        Buffers(Pz, Dz, to_device(Ps), to_device(Dgs),
            KernelAbstractions.allocate(BACKEND, T32, res * res),
            Matrix{Float32}(undef, res, res))
    end
end

const KERNEL = march!(BACKEND)

"One frame: sweep the whole chart at resolution `res`. Returns the staging matrix."
function frame!(res::Int, c1, zeta, tau)
    b = buffers(res)
    KERNEL(b.C, b.Ps, b.Dgs, T32(c1), T32(zeta), T32(tau),
        WMAX, TOL, TOL; ndrange = res * res)
    KernelAbstractions.synchronize(BACKEND)
    copyto!(vec(b.host), Array(b.C))          # device -> host (640 kB at 400^2)
    return b.host
end

# ---------------------------------------------------------------------------
# Headless benchmark: the number that decides "real time"
# ---------------------------------------------------------------------------
function bench()
    println("backend: ", ON_GPU ? "GPU (CUDA)" : "CPU ($(Threads.nthreads()) threads)")
    for res in (100, 200, 400)
        frame!(res, 0.03, 0.02, 0.5)                       # warm-up / JIT
        n = res == 400 ? 5 : 20
        t = @elapsed for i in 1:n                          # slider sweep: tau moves
            frame!(res, 0.03, 0.02, 0.3 + 0.02 * i)
        end
        @printf("res %3dx%-3d : %7.1f ms/frame  (%.1f fps)\n",
            res, res, 1000t / n, n / t)
    end
end

# ---------------------------------------------------------------------------
# GUI: full-resolution live chart on GPU; progressive refinement on CPU
# ---------------------------------------------------------------------------
function pick_drag_res()
    ON_GPU && return RES_IDLE                 # GPU redraws full res live
    frame!(RES_IDLE, 0.03, 0.02, 0.5)         # warm-up
    t = @elapsed frame!(RES_IDLE, 0.03, 0.02, 0.5)
    t < 0.08 && return RES_IDLE
    # scale the pixel count so a drag frame costs ~80 ms
    r = clamp(round(Int, RES_IDLE * sqrt(0.08 / t)), 48, RES_IDLE)
    @info "CPU backend: full res $(RES_IDLE) takes $(round(1000t; digits=0)) ms; dragging at $(r)x$(r), refining on idle"
    return r
end

function gui()
    @eval using GLMakie
    Base.invokelatest() do
        res_drag = pick_drag_res()

        fig = Figure(size = (980, 760))
        ax = GLMakie.Axis(fig[1, 1], xlabel = "P", ylabel = "D",
            title = "live stability chart")
        sg = GLMakie.SliderGrid(fig[2, 1],
            (label = "c1", range = 0.001:0.001:0.10, startvalue = 0.03),
            (label = "zeta", range = 0.0:0.002:0.20, startvalue = 0.02),
            (label = "tau", range = 0.05:0.01:1.50, startvalue = 0.5),
            (label = "resolution", range = 100:50:800, startvalue = RES_IDLE))
        data = GLMakie.Observable(zeros(Float32, RES_IDLE, RES_IDLE))
        xs = GLMakie.Observable(buffers(RES_IDLE).Pz)
        ys = GLMakie.Observable(buffers(RES_IDLE).Dz)
        GLMakie.heatmap!(ax, xs, ys, data; colormap = GLMakie.Reverse(:RdYlGn_9),
            colorrange = (-1.5, 6.0))    # green = stable (deep gap), red = many unstable roots
        status = GLMakie.Observable("...")
        GLMakie.Label(fig[3, 1], status; tellwidth = false)

        last_move = Ref(time())
        refined = Ref(false)
        busy = Ref(false)
        res_target = Ref(RES_IDLE)

        backend_tag = ON_GPU ? "GPU" : "CPU, $(Threads.nthreads()) threads"
        function draw!(res)
            busy[] && return
            busy[] = true
            c1 = sg.sliders[1].value[]; z = sg.sliders[2].value[]
            tau = sg.sliders[3].value[]
            t = @elapsed (m = frame!(res, c1, z, tau))
            b = buffers(res)
            xs[] = b.Pz; ys[] = b.Dz; data[] = copy(m)
            ax.title[] = @sprintf("%d x %d   |   %.0f ms/frame   (%.1f fps)   [%s]",
                res, res, 1000t, 1 / t, backend_tag)
            status[] = @sprintf("c1 = %.3f    zeta = %.3f    tau = %.2f", c1, z, tau)
            busy[] = false
        end

        for s in sg.sliders[1:3]              # model parameters: drag redraw
            GLMakie.on(s.value) do _
                last_move[] = time(); refined[] = res_drag >= res_target[]
                draw!(min(res_drag, res_target[]))
            end
        end
        GLMakie.on(sg.sliders[4].value) do r  # resolution: preview, then refine
            res_target[] = r
            last_move[] = time(); refined[] = res_drag >= r
            draw!(min(res_drag, r))
        end
        @async while true                     # refine when the sliders rest
            sleep(0.1)
            if !refined[] && time() - last_move[] > REFINE_DELAY
                refined[] = true
                draw!(res_target[])
            end
        end

        draw!(RES_IDLE)
        scr = display(fig)
        return scr
    end
end

if "--bench" in ARGS
    bench()
else
    scr = gui()
    println("Window open -- move the sliders. Closing the window exits.")
    try
        while isopen(scr)
            sleep(0.2)
        end
    catch
        wait(Condition())              # fallback: run until the process is killed
    end
end
