# Small helpers shared by the studies (include-guarded).

if !@isdefined(STUDIES_COMMON_INCLUDED)

using Statistics

"""
Median wall time of `f()`, after a warm-up call.

Cheap calls are batched so that each sample lasts at least `min_batch_s`
(removing timer granularity); expensive calls are sampled fewer times instead,
because a single evaluation can already take tens of seconds (a low-order pair
at a tight tolerance over a wide frequency range) and seven repeats of that
would dominate the whole study.
"""
function time_point(f; samples = 7, min_batch_s = 0.02)
    f()
    t1 = @elapsed f()
    if t1 > 1.0
        return t1                      # already precise; repeating is pure waste
    end
    n = t1 > 0.05 ? 3 : samples
    k = max(1, ceil(Int, min_batch_s / max(t1, 1e-9)))
    ts = Float64[]
    for _ in 1:n
        GC.gc()
        t = @elapsed begin
            for _ in 1:k
                f()
            end
        end
        push!(ts, t / k)
    end
    return median(ts)
end

const STUDIES_COMMON_INCLUDED = true
end
