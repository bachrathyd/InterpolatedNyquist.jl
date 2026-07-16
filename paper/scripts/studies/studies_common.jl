# Small helpers shared by the studies (include-guarded).

if !@isdefined(STUDIES_COMMON_INCLUDED)

using Statistics

"""
Median wall time of `f()`, auto-scaling the repetition count so that each
sample takes at least 20 ms (removes timer granularity), after a warm-up call.
"""
function time_point(f; samples = 7, min_batch_s = 0.02)
    f()
    t1 = @elapsed f()
    k = max(1, ceil(Int, min_batch_s / max(t1, 1e-9)))
    ts = Float64[]
    for _ in 1:samples
        GC.gc()
        push!(ts, (@elapsed for _ in 1:k; f(); end) / k)
    end
    return median(ts)
end

const STUDIES_COMMON_INCLUDED = true
end
