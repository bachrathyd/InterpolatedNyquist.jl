# Study s11: evaluation-count diagnostics quoted in Sections 3.1 and 7.4.
#  (a) D-evaluations per parameter point for the ordered march vs adaptive
#      quadrature, on a slowly (ω^-1) and a fast (ω^-3) decaying ripple, at
#      two frequency windows -- with the COUNT verified against a tight
#      reference before any number is recorded (a cheap backend that returns
#      a wrong count must never be reported as a speedup).
#  (b) cost of enlarging ω_max to 1e8 / 1e10 (the "choose ω_max generously"
#      guidance of Section 3.1 must rest on measured numbers).
# Produces: tables/tab_evals.tex, data/eval_counts.csv, data/wmax_scaling.csv

include(joinpath(@__DIR__, "studies_common.jl"))
include(joinpath(@__DIR__, "systems.jl"))

# count D evaluations with a wrapping closure (thread-safe: single-point calls)
const EVAL_COUNTER = Ref(0)
function counted(D)
    return (λ, p) -> begin
        EVAL_COUNTER[] += 1
        D(λ, p)
    end
end

# unstable representative points (fixed, resolution-independent)
const P_SHOW = (3.0, 1.0)     # showcase reduced: velocity feedback, ω^-1 ripple
const P_FOUR = (2.0, 0.0)     # 4th-order with D = 0: position feedback, ω^-3 ripple

SYSTEMS_EV = [
    ("showcase (\$\\omega^{-1}\$ ripple)", D_showcase_reduced, P_SHOW),
    ("4th-order (\$\\omega^{-3}\$ ripple)", D_fourth, P_FOUR),
]

ev_rows = with_cache("s11_eval_counts_v1") do
    out = Tuple[]
    for (name, D, p) in SYSTEMS_EV
        n_pow = get_n_power_max(D, p)
        for wm in (1e4, 1e6)
            Z_ref, _ = calculate_unstable_roots_direct(D, p; ω_max = wm,
                n_roots_to_track = 0, reltol = 1e-9, abstol = 1e-9, n_power_max = n_pow)
            Dc = counted(D)
            EVAL_COUNTER[] = 0
            Z_m, _ = calculate_unstable_roots_direct(Dc, p; ω_max = wm,
                n_roots_to_track = 0, n_power_max = n_pow)
            n_march = EVAL_COUNTER[]
            EVAL_COUNTER[] = 0
            Z_q, _ = calculate_unstable_roots_quadgk(Dc, p; ω_max = wm, n_power_max = n_pow)
            n_quad = EVAL_COUNTER[]
            Z_m == Z_ref || error("s11: march count $Z_m != reference $Z_ref at ω_max=$wm ($name)")
            Z_q == Z_ref || error("s11: quadrature count $Z_q != reference $Z_ref at ω_max=$wm ($name)")
            push!(out, (name, wm, n_march, n_quad, Z_ref))
            @info "eval counts" name wm n_march n_quad Z_ref
        end
    end
    out
end
write_csv("eval_counts", ["system", "wmax", "n_march", "n_quad", "Z"], ev_rows)

fmtn(n) = replace(@sprintf("%d", n), r"(?<=\d)(?=(\d{3})+$)" => "\\,")
rows_tex = Vector{String}[]
for (name, _, _) in SYSTEMS_EV
    sel = [r for r in ev_rows if r[1] == name]
    r4 = sel[findfirst(r -> r[2] == 1e4, sel)]
    r6 = sel[findfirst(r -> r[2] == 1e6, sel)]
    push!(rows_tex, [name, "ordered march", fmtn(r4[3]), fmtn(r6[3])])
    push!(rows_tex, ["", "quadrature", fmtn(r4[4]), fmtn(r6[4])])
end
write_booktabs("tab_evals", "llcc",
    ["system", "back-end", "\$\\wmax=10^{4}\$", "\$\\wmax=10^{6}\$"], rows_tex)

# ---------------------------------------------------------------------------
# ω_max scaling of the march (the Section 3.1 guidance)
# ---------------------------------------------------------------------------
wmax_rows = with_cache("s11_wmax_scaling_v1") do
    out = Tuple[]
    D, p = D_fourth, P_FOUR
    n_pow = get_n_power_max(D, p)
    for wm in (1e6, 1e8, 1e10)
        Dc = counted(D)
        EVAL_COUNTER[] = 0
        Z, Zr = calculate_unstable_roots_direct(Dc, p; ω_max = wm,
            n_roots_to_track = 0, n_power_max = n_pow)
        n_ev = EVAL_COUNTER[]
        t = time_point(() -> calculate_unstable_roots_direct(D, p; ω_max = wm,
            n_roots_to_track = 0, n_power_max = n_pow))
        push!(out, (wm, n_ev, t, Z, Zr))
        @info "wmax scaling" wm n_ev t Z
    end
    out
end
write_csv("wmax_scaling", ["wmax", "n_evals", "time_s", "Z", "Z_raw"], wmax_rows)
