# Study s12: the peak-repair callback (Section 6.3), measured.
#
# The march's one silent failure mode is stepping over the near-singular
# integrand peak of a root close to the sigma-line (Sec. 3.4). The
# cross-check flags such pixels; `peak_repair = true` repairs them within
# the march itself: a rootfinding callback stops the integrator at every
# minimum of |D|^2 (the peak's exact center, detectable from far away
# because |D|^2 dips over an O(1) frequency range even when the peak is
# arbitrarily narrow), the phase increment of the step that crossed the
# minimum is recomputed by quadrature forced into the located peak by a
# geometric breakpoint ladder, and the march restarts at the minimum.
#
# This study measures where the plain march, the repaired march and the
# adaptive-quadrature backend lose the count as the showcase Hopf point is
# approached, and what the repair costs. Hard sanity gates: every claim the
# prose quotes is asserted here, so a regeneration on changed code fails
# loudly instead of shipping stale sentences.
# Produces: data/peak_repair.csv, generated/repair_numbers.tex

include(joinpath(@__DIR__, "studies_common.jl"))
include(joinpath(@__DIR__, "systems.jl"))

const D_REP = D_showcase_reduced
const D_GAIN_REP = 1.5          # vertical cut of the showcase chart
const NPOW_REP = 4              # leading order of the reduced quasi-polynomial

# count D evaluations with a wrapping closure (single-point calls, no threads)
const REP_COUNTER = Ref(0)
rep_counted(D) = (λ, p) -> (REP_COUNTER[] += 1; D(λ, p))

# Hopf point on the cut, bisected on the sign of the refined dominant sigma
# at tight tolerances. Deterministic for this D; cached like every grid.
Pb_rep = with_cache("s12_hopf_point_v1") do
    sig(P) = calculate_unstable_roots_direct(D_REP, (P, D_GAIN_REP); ω_max = 1e6,
        reltol = 1e-10, abstol = 1e-10, refinement_method = :Newton, refinement_steps = 15)[4]
    lo, hi = 2.0, 3.0
    for _ in 1:60
        m = (lo + hi) / 2
        (sig(m) < 0) ? (lo = m) : (hi = m)
    end
    (lo + hi) / 2
end
@info "s12 Hopf point" Pb_rep

# The three counts at distance dP from the boundary. Truth by continuity: Z
# is constant on each side of an isolated Hopf crossing, so the far point
# (dP = 1e-2, verified march == quadrature below) anchors the unstable side
# and truth minus two (the crossing pair) anchors the stable side.
DP_LADDER = (1e-2, 1e-6, 1e-8, 1e-10, -1e-6, -1e-8)

count_at(dP; kw...) = begin
    p = (Pb_rep + dP, D_GAIN_REP)
    Zp = calculate_unstable_roots_direct(D_REP, p; n_power_max = NPOW_REP, kw...)[1]
    Zr = calculate_unstable_roots_direct(D_REP, p; n_power_max = NPOW_REP,
        peak_repair = true, kw...)[1]
    Zq = calculate_unstable_roots_quadgk(D_REP, p; n_power_max = NPOW_REP)[1]
    return Zp, Zr, Zq
end

rep_rows = with_cache("s12_repair_ladder_v1") do
    out = Tuple[]
    for dP in DP_LADDER
        p = (Pb_rep + dP, D_GAIN_REP)
        Zp, Zr, Zq = count_at(dP)
        tp = time_point(() -> calculate_unstable_roots_direct(D_REP, p;
            n_power_max = NPOW_REP))
        tr = time_point(() -> calculate_unstable_roots_direct(D_REP, p;
            n_power_max = NPOW_REP, peak_repair = true))
        tq = time_point(() -> calculate_unstable_roots_quadgk(D_REP, p;
            n_power_max = NPOW_REP))
        Dc = rep_counted(D_REP)
        REP_COUNTER[] = 0
        calculate_unstable_roots_direct(Dc, p; n_power_max = NPOW_REP)
        np_ev = REP_COUNTER[]
        REP_COUNTER[] = 0
        calculate_unstable_roots_direct(Dc, p; n_power_max = NPOW_REP, peak_repair = true)
        nr_ev = REP_COUNTER[]
        REP_COUNTER[] = 0
        calculate_unstable_roots_quadgk(Dc, p; n_power_max = NPOW_REP)
        nq_ev = REP_COUNTER[]
        push!(out, (dP, Zp, Zr, Zq, tp, tr, tq, np_ev, nr_ev, nq_ev))
        @info "repair ladder" dP Zp Zr Zq tp tr tq
    end
    out
end

Z_far = rep_rows[1][2]
truth(dP) = dP > 0 ? Z_far : Z_far - 2

# --- sanity gates: exactly the claims Section 6.3 makes -------------------
let r = Dict(row[1] => row for row in rep_rows)
    r[1e-2][2] == r[1e-2][4] == Z_far ||
        error("s12: march and quadrature disagree at the far point")
    for dP in DP_LADDER
        r[dP][3] == truth(dP) ||
            error("s12: repaired march wrong at dP=$dP (got $(r[dP][3]), truth $(truth(dP)))")
    end
    r[1e-6][2] == truth(1e-6) &&
        error("s12: plain march is suddenly CORRECT at dP=1e-6; Section 6.3 prose is stale")
    r[1e-8][2] == truth(1e-8) &&
        error("s12: plain march is suddenly CORRECT at dP=1e-8; Section 6.3 prose is stale")
    r[1e-8][4] == truth(1e-8) ||
        error("s12: quadrature backend wrong already at dP=1e-8")
    r[1e-10][4] == truth(1e-10) &&
        error("s12: quadrature backend is suddenly CORRECT at dP=1e-10; Section 6.3 prose is stale")
    r[-1e-8][2] == truth(-1e-8) &&
        error("s12: plain march is suddenly CORRECT on the stable side; prose is stale")
end

write_csv("peak_repair",
    ["dP", "Z_truth", "Z_plain", "Z_repair", "Z_gk",
     "t_plain_s", "t_repair_s", "t_gk_s",
     "evals_plain", "evals_repair", "evals_gk"],
    [(row[1], truth(row[1]), row[2], row[3], row[4],
      row[5], row[6], row[7], row[8], row[9], row[10]) for row in rep_rows])

# cost factor at the representative near-boundary distance 1e-8
row8 = rep_rows[findfirst(row -> row[1] == 1e-8, rep_rows)]
cost_factor = row8[6] / row8[5]
gk_factor   = row8[7] / row8[5]
write_macros("repair_numbers", [
    "RepairPlainFailDP" => "10^{-6}",
    "RepairDeepDP"      => "10^{-10}",
    "RepairCostFactor"  => @sprintf("%.1f", cost_factor),
    "RepairGKFactor"    => @sprintf("%.1f", gk_factor),
])
@info "s12 cost" cost_factor gk_factor
