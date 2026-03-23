using InterpolatedNyquist
using GLMakie
using Statistics
using Printf
using QuadGK
using DifferentialEquations
using ForwardDiff
using StaticArrays

# 1. Define the system (4th Order)
function D_chareq(λ::T, p) where T
    P, D = p
    τ = T(0.5)
    ζ = T(0.02)
    return (T(0.03) * λ^4 + λ^2 + T(2) * ζ * λ + one(T) + P * exp(-τ * λ) + D * λ * exp(-τ * λ))
end

# 2. Setup Grid
# Using 80x80 to keep the 10x repetition benchmarks within reasonable time
Pv = LinRange(-2.0, 4.0, 80)
Dv = LinRange(-2.0, 5.0, 80)
params_vec = vec([(Pv[i], Dv[j]) for i in 1:length(Pv), j in 1:length(Dv)])

# 3. Naive Fixed-Step Integration Implementation
struct NaiveTag end

function naive_z_raw(D_func, p, ω_max, steps)
    h = ω_max / steps
    integral = 0.0
    n_pow = get_n_power_max(D_func, p; ω_large=ω_max)
    
    function get_darg(ω_val)
        pure_ω = max(ω_val, 1e-9)
        dual_ω = ForwardDiff.Dual{NaiveTag}(pure_ω, 1.0)
        dual_λ = 1im * dual_ω
        res = D_func(dual_λ, p)
        rv, iv = real(res), imag(res)
        re_val, im_val = ForwardDiff.value(rv), ForwardDiff.value(iv)
        dre, dim = ForwardDiff.partials(rv, 1), ForwardDiff.partials(iv, 1)
        d_sq = re_val^2 + im_val^2
        if d_sq < 1e-20 return 0.0 end
        return (dim * re_val - dre * im_val) / d_sq
    end

    f_prev = get_darg(0.0)
    for i in 1:steps
        ω = i * h
        f_curr = get_darg(ω)
        integral += (f_prev + f_curr) * 0.5 * h
        f_prev = f_curr
    end
    return -(1.0 / π) * integral + n_pow / 2.0
end

# 4. Benchmarking Logic
mkpath("performance_tests/Naive")
mkpath("performance_tests/QuadGK")
mkpath("performance_tests/ODE")

method_results = Dict{String, Any}()

# Summary Helper
function update_summaries()
    all_methods = sort(collect(keys(method_results)))
    create_analysis_plot("All Methods Summary", all_methods, "summary_comparison.png")
    
    naive_keys = filter(k -> startswith(k, "Naive"), all_methods)
    create_analysis_plot("Naive Fixed-Step Approach", naive_keys, "approach_naive.png")
    
    quad_keys = filter(k -> startswith(k, "QuadGK"), all_methods)
    create_analysis_plot("Quadrature Approach", quad_keys, "approach_quadgk.png")
    
    ode_keys = filter(k -> startswith(k, "ODE_"), all_methods)
    if !isempty(ode_keys)
        create_analysis_plot("ODE Solver Approaches", ode_keys, "approach_ode.png")
    end
end

function create_analysis_plot(title_prefix, filter_keys, filename)
    f = Figure(size=(1800, 1000))
    ax1 = Axis(f[1, 1], title="Median Error vs. Complexity", xlabel="Complexity (Steps/1/Tol)", ylabel="Median Abs Error", xscale=log10, yscale=log10)
    ax2 = Axis(f[1, 2], title="CPU Time vs. Complexity", xlabel="Complexity (Steps/1/Tol)", ylabel="CPU Time [s] (Mean ± Std)", xscale=log10, yscale=log10)
    ax3 = Axis(f[1, 3], title="Error vs. CPU Time", xlabel="CPU Time [s]", ylabel="Error", xscale=log10, yscale=log10)
    ax4 = Axis(f[2, 2], title="Memory Consumption vs. Complexity", xlabel="Complexity (Steps/1/Tol)", ylabel="Allocations [MB]", xscale=log10, yscale=log10)
    
    # Apply CPU time limit
    ylims!(ax2, low=1e-3)

    any_plotted = false
    colors = Makie.wong_colors()
    for (i, name) in enumerate(filter_keys)
        if !haskey(method_results, name) continue end
        any_plotted = true
        c, e_med, e05, e95, t_mean, t_std, mem_vals = method_results[name]
        color = colors[mod1(i, length(colors))]
        
        # Error Plot (Median + 5/95 Quantile Band)
        band!(ax1, c, e05, e95, color=(color, 0.2))
        scatterlines!(ax1, c, e_med, label=name, color=color)
        
        # CPU Time Plot (Mean + Std Band)
        t_low = max.(1e-6, t_mean .- t_std) # Keep log scale happy
        t_high = t_mean .+ t_std
        band!(ax2, c, t_low, t_high, color=(color, 0.2))
        scatterlines!(ax2, c, t_mean, color=color)
        
        # Error vs Time
        band!(ax3, t_mean, e05, e95, color=(color, 0.2))
        scatterlines!(ax3, t_mean, e_med, color=color)

        # Memory Consumption (Allocated MB)
        scatterlines!(ax4, c, mem_vals ./ (1024^2), color=color)
    end
    
    if any_plotted
        f[1:2, 4] = Legend(f, ax1, "Method")
    end
    Label(f[0, :], title_prefix, fontsize=20, font=:bold)
    save("performance_tests/$filename", f)
end

function run_benchmark(method_base_name, complexity_seq, solver_func, ω_val; is_tol=true, n_repeats=10)
    method_name = "$(method_base_name)_ω$(Int(ω_val))"
    
    c_vals, e_med_vals, e05_vals, e95_vals, t_mean_vals, t_std_vals, mem_vals = [Float64[] for _ in 1:7]
    
    subfolder = if startswith(method_base_name, "Naive")
        "Naive/ω$(Int(ω_val))"
    elseif startswith(method_base_name, "QuadGK")
        "QuadGK/ω$(Int(ω_val))"
    else
        "ODE/ω$(Int(ω_val))"
    end
    mkpath("performance_tests/$(subfolder)")

    println("\nBenchmarking $method_name (ω_max=$ω_val)...")
    
    # Warm-up
    println("Warming up...")
    solver_func(params_vec[1], complexity_seq[1], ω_val)

    for c in complexity_seq
        all_z_raws = zeros(length(params_vec))
        times = Float64[]
        allocations = Float64[]
        
        println("  Running complexity c=$c (repeating $n_repeats times)...")
        
        # Perform repetitions
        for r in 1:n_repeats
            start_t = time()
            allocs = @allocated begin
                Threads.@threads for i in 1:length(params_vec)
                    res = solver_func(params_vec[i], c, ω_val)
                    if r == n_repeats
                        all_z_raws[i] = res
                    end
                end
            end
            push!(times, time() - start_t)
            push!(allocations, Float64(allocs))
        end
        
        elapsed_mean = mean(times)
        elapsed_std = std(times)
        mem_mean = mean(allocations)
        
        errors = abs.(all_z_raws .- round.(all_z_raws))
        e_med = median(errors)
        p05 = quantile(errors, 0.05)
        p95 = quantile(errors, 0.95)
        
        push!(c_vals, is_tol ? 1.0/c : c)
        push!(e_med_vals, e_med)
        push!(e05_vals, p05)
        push!(e95_vals, p95)
        push!(t_mean_vals, elapsed_mean)
        push!(t_std_vals, elapsed_std)
        push!(mem_vals, mem_mean)
        
        method_results[method_name] = (c_vals, e_med_vals, e05_vals, e95_vals, t_mean_vals, t_std_vals, mem_vals)
        @printf("    C: %.2e | Med Err: %.2e | Time: %.3f ± %.3f s | Mem: %.2f MB\n", c, e_med, elapsed_mean, elapsed_std, mem_mean/(1024^2))
        
        # Save Triple-Panel Figure
        f_step = Figure(size=(1800, 500))
        z_mat = reshape(all_z_raws, length(Pv), length(Dv))
        err_mat = reshape(errors, length(Pv), length(Dv))
        
        ax1 = Axis(f_step[1, 1], title="Z_raw Result (c=$c, ω=$ω_val)", xlabel="P", ylabel="D")
        heatmap!(ax1, Pv, Dv, z_mat, colormap=:viridis, colorrange=(-2, 10))
        
        ax2 = Axis(f_step[1, 2], title="Error Field (log10)", xlabel="P", ylabel="D")
        heatmap!(ax2, Pv, Dv, log10.(err_mat .+ 1e-18))
        
        ax3 = Axis(f_step[1, 3], title="Error Histogram", xlabel="Log10(Error)")
        hist!(ax3, log10.(errors .+ 1e-18), bins=50)
        
        save("performance_tests/$(subfolder)/$(method_name)_c$(c).png", f_step)
        
        update_summaries()

        if elapsed_mean > 10.0 # Stop if average time is too high
            println("    Average time limit reached. Skipping higher complexities.")
            break
        end
    end
end

# Define Sequences
naive_seq = [Int(ceil(10.0^(i/3))) for i in 5:40]
tol_seq = [10.0^(-i/3) for i in 3:21]

omegas = [100.0, 1000.0, 10000.0]

ode_solvers = [
     ("ODE_BS3", BS3()),
     ("ODE_RK4", RK4()),
     ("ODE_Tsit5", Tsit5()),
     ("ODE_Ros23", Rosenbrock23()),
     ("ODE_Rodas4", Rodas4()),
     ("ODE_TRBDF2", TRBDF2()),
     ("ODE_AutoStiff", AutoTsit5(Rosenbrock23()))
]

# Run Benchmarks
for ω in omegas
    # Naive
    run_benchmark("Naive", naive_seq, (p, c, ω_m) -> naive_z_raw(D_chareq, p, ω_m, Int(c)), ω; is_tol=false)
    # QuadGK
    run_benchmark("QuadGK", tol_seq, (p, c, ω_m) -> calculate_unstable_roots_quadgk(D_chareq, p; ω_max=ω_m, reltol=c, abstol=c)[2], ω)
    # ODE Solvers
    for (name, s) in ode_solvers
        run_benchmark(name, tol_seq, (p, c, ω_m) -> calculate_unstable_roots_direct(D_chareq, p; ω_max=ω_m, reltol=c, abstol=c, solver=s)[2], ω)
    end
end

println("All simulations and continuous summary updates complete.")
