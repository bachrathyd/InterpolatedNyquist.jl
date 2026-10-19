// Stochastic mode: the steady-state second moment and the mean-square stability margin on the
// unwrap march (theory: notes/stochastic_second_moment.md; Julia prototype: notes/stochastic_sdde.jl).
// Spliced into march.wgsl at //#SMARCH# by engine.js when the model is in the state form (sde.js),
// whose generated code provides
//     fn smodel(l: CD, p) -> SMO     D(λ) = det M(λ), |Y|² and Σ_k |G_k|² with their d/dω
//     fn smstatic(p) -> vec2<f32>    Γ = Σ_k γ̃_k² (effective additive intensity), validity
//
// Per chart point, two passes of the march on λ = iω:
//   pass 1: the deterministic march (phase test only) -> the count Z (first-moment stability) and,
//           on the same nodes, coarse integrals I_Y, I_G;
//   pass 2 (only if Z == 0): the same march with a second acceptance test, the plain-trapezoid
//           error h²/12 |f'_a - f'_b| relative to the coarse integral (tolerance P.qtrust, which the
//           stochastic march uses as rtol_I: it tracks no |D| minima).
// The integrals use the Hermite-corrected trapezoid rule (f' from the dual numbers) and a power-law
// tail beyond ω_max:
//   r0 = (1/2π)∫ Σ_k |G_k|² dω   (mean-square stable iff Z == 0 and r0 < 1)
//   Var(x_0) = Γ (1/2π)∫ |Y|² dω / (1 - r0)
// Result (struct Res, 16 bytes as the deterministic march):
//   z = Z_raw,  s = log10 Var (flags & 4096) or r0 (flags & 2048: mean-square unstable),
//   steps = evaluations (bits 0..16) | round(r0 · 32767) << 17 (when mean-square stable),
//   flags = the march flags | 2048 (MS unstable) | 4096 (variance valid) | 8192 (pass 2 failed).

struct SMO {
    D: CD,          // det M(λ) and dD/dω
    fy: vec2<f32>,  // |Y|², d|Y|²/dω
    fg: vec2<f32>,  // Σ_k |G_k|², d/dω
}

var<private> SMm: array<CD, 36>;    // characteristic matrix (row-major, stride 6), overwritten by the LU
var<private> SMu: array<CD, 6>;     // M^{-1} e_r
var<private> SRm: array<f32, 36>;   // A + Σ B_d (real), overwritten
var<private> SRb: array<f32, 6>;    // -c, then the mean

// det M and SMu = M^{-1} e_r by Gaussian elimination with partial pivoting (complex dual numbers)
fn sm_solve(n: i32, r: i32) -> CD {
    for (var i = 0; i < n; i++) { SMu[i] = CD(C(select(0.0, 1.0, i == r), 0.0), C(0.0, 0.0)); }
    var det = CD(C(1.0, 0.0), C(0.0, 0.0));
    for (var k = 0; k < n; k++) {
        var piv = k;
        var best = cabs(SMm[k * 6 + k].v);
        for (var i = k + 1; i < n; i++) {
            let a = cabs(SMm[i * 6 + k].v);
            if (a > best) { best = a; piv = i; }
        }
        if (piv != k) {
            for (var j = k; j < n; j++) {
                let t = SMm[k * 6 + j];
                SMm[k * 6 + j] = SMm[piv * 6 + j];
                SMm[piv * 6 + j] = t;
            }
            let t = SMu[k];
            SMu[k] = SMu[piv];
            SMu[piv] = t;
            det = dneg(det);
        }
        let pk = SMm[k * 6 + k];
        det = dmul(det, pk);
        if (!(best > 0.0)) { return CD(C(0.0, 0.0), C(0.0, 0.0)); }
        for (var i = k + 1; i < n; i++) {
            let f = ddiv(SMm[i * 6 + k], pk);
            for (var j = k + 1; j < n; j++) {
                SMm[i * 6 + j] = dsub(SMm[i * 6 + j], dmul(f, SMm[k * 6 + j]));
            }
            SMu[i] = dsub(SMu[i], dmul(f, SMu[k]));
        }
    }
    for (var i = n - 1; i >= 0; i--) {
        var s = SMu[i];
        for (var j = i + 1; j < n; j++) { s = dsub(s, dmul(SMm[i * 6 + j], SMu[j])); }
        SMu[i] = ddiv(s, SMm[i * 6 + i]);
    }
    return det;
}

// (A + Σ B_d) m = -c (real, partial pivoting); the mean in SRb; false if singular
fn sm_rsolve(n: i32) -> bool {
    for (var k = 0; k < n; k++) {
        var piv = k;
        var best = abs(SRm[k * 6 + k]);
        for (var i = k + 1; i < n; i++) {
            let a = abs(SRm[i * 6 + k]);
            if (a > best) { best = a; piv = i; }
        }
        if (!(best > 0.0)) { return false; }
        if (piv != k) {
            for (var j = k; j < n; j++) {
                let t = SRm[k * 6 + j];
                SRm[k * 6 + j] = SRm[piv * 6 + j];
                SRm[piv * 6 + j] = t;
            }
            let t = SRb[k];
            SRb[k] = SRb[piv];
            SRb[piv] = t;
        }
        for (var i = k + 1; i < n; i++) {
            let f = SRm[i * 6 + k] / SRm[k * 6 + k];
            for (var j = k + 1; j < n; j++) { SRm[i * 6 + j] = SRm[i * 6 + j] - f * SRm[k * 6 + j]; }
            SRb[i] = SRb[i] - f * SRb[k];
        }
    }
    for (var i = n - 1; i >= 0; i--) {
        var s = SRb[i];
        for (var j = i + 1; j < n; j++) { s = s - SRm[i * 6 + j] * SRb[j]; }
        SRb[i] = s / SRm[i * 6 + i];
    }
    return true;
}

struct SMPass {
    status: u32,    // 1 finished, 2 failed
    phi: f32,
    iy: f32,        // (1/2π)∫|Y|²
    ig: f32,        // (1/2π)∫Σ|G_k|²
    steps: u32,
    flags: u32,
}

// relative error of the plain trapezoid on [a, b] (scale: the coarse integral, or the running one)
fn sm_err(acc: f32, fa: vec2<f32>, fb: vec2<f32>, h: f32, sc: f32) -> f32 {
    let e = abs(h * h / 12.0 * (fa.y - fb.y));
    let den = max(max(sc, abs(acc + 0.5 * h * (fa.x + fb.x))), FLOATMIN);
    let r = e / den;
    return select(BIG, r, r < BIG);
}

// one march; rtol = BIG: the deterministic nodes (phase test only)
fn sm_pass(p: vec2<f32>, rtol: f32, sy: f32, sg: f32) -> SMPass {
    let sig = P.sigma;
    var out: SMPass;
    out.flags = 0u;
    var w = P.w0;
    let o0 = smodel(CD(C(sig, w), C(0.0, 1.0)), p);
    let S = sample_data(o0.D.v, o0.D.d);
    var status = select(2u, 0u, S.ok);
    var h = P.h0;
    var phi = 0.0;
    var ua = S.u;
    var tha = S.th;
    var fya = o0.fy;
    var fga = o0.fg;
    var iy = 0.0;
    var ig = 0.0;
    var steps = 0u;
    // integrals in the march's units (∫ over ω ≥ 0); the scales likewise
    let scy = sy * PI;
    let scg = sg * PI;
    loop {
        if (status != 0u) { break; }
        let rest = P.wmax - w;
        var ht = min(h, P.hrel * max(w, 1.0));
        if (w < P.wband) { ht = min(ht, P.hmax); }
        let last = ht >= rest;
        let hs = select(ht, rest, last);
        let b = select(w + ht, P.wmax, last);
        let ob = smodel(CD(C(sig, b), C(0.0, 1.0)), p);
        let Sb = sample_data(ob.D.v, ob.D.d);
        var dl = 0.0;
        if (Sb.ok) { dl = dphase(ua, Sb.u); }
        let fin = all(abs(ob.fy) < vec2<f32>(BIG)) && all(abs(ob.fg) < vec2<f32>(BIG));
        let valid = Sb.ok && fin;
        var err = BIG;
        var errI = 0.0;
        if (valid) {
            err = abs(dl - hs * (tha + Sb.th) / 2.0);
            if (!(err < BIG)) { err = BIG; }
            if (rtol < BIG) {
                errI = max(sm_err(iy, fya, ob.fy, hs, scy), sm_err(ig, fga, ob.fg, hs, scg));
            }
        }
        let accept = valid && (err <= P.tol) && (errI <= rtol);
        let q = min(P.tol / max(err, FLOATMIN), select(BIG, rtol / max(errI, FLOATMIN), rtol < BIG));
        let fac = clamp(0.9 * cbrtp(q), 0.2, P.growmax);
        var hn = hs * fac;
        steps = steps + 1u;
        var take = accept;
        var phin = phi + dl;
        if (!accept && valid && (hn <= hfloor(w))) {
            let qn = newton_q(ob.D.v, ob.D.d);
            let right = (qn.z != 0.0) && (qn.y > 0.0);
            var dr = dl;
            if (abs(dl) > PI / 2.0) {
                if (right && (dl > 0.0)) { dr = dl - 2.0 * PI; }
                else if ((!right) && (dl < 0.0)) { dr = dl + 2.0 * PI; }
            }
            take = true;
            hn = hs;
            phin = phi + dr;
            out.flags = out.flags | 2u;
        }
        if (take) {
            iy = iy + 0.5 * hs * (fya.x + ob.fy.x) + hs * hs / 12.0 * (fya.y - ob.fy.y);
            ig = ig + 0.5 * hs * (fga.x + ob.fg.x) + hs * hs / 12.0 * (fga.y - ob.fg.y);
            if (b >= P.wmax) { status = 1u; } else if (steps >= P.maxsteps) { status = 2u; }
            w = b;
            h = hn;
            phi = phin;
            ua = Sb.u;
            tha = Sb.th;
            fya = ob.fy;
            fga = ob.fg;
        } else {
            if ((hn <= hfloor(w)) || (steps >= P.maxsteps)) { status = 2u; }
            h = hn;
        }
    }
    // power-law tails f ≈ c ω^(-k):  ∫_ω^∞ f = f ω / (k - 1)
    if (fya.x > 0.0) {
        let k = -w * fya.y / fya.x;
        if (k > 1.05) { iy = iy + fya.x * w / (k - 1.0); }
    }
    if (fga.x > 0.0) {
        let k = -w * fga.y / fga.x;
        if (k > 1.05) { ig = ig + fga.x * w / (k - 1.0); }
    }
    out.status = status;
    out.phi = phi;
    out.iy = iy / PI;
    out.ig = ig / PI;
    out.steps = steps;
    return out;
}

fn march_sm(p: vec2<f32>) -> Res {
    var res: Res;
    let a = sm_pass(p, BIG, 0.0, 0.0);
    var f = a.flags;
    var steps = a.steps;
    res.s = 0.0;
    if (a.status != 1u) {
        res.z = 0.0;
        res.steps = steps;
        res.flags = f | 1u | 128u;
        return res;
    }
    let zr = P.npow / 2.0 - a.phi / PI;
    res.z = zr;
    let Z = round(zr);
    if (abs(zr - Z) > 0.25) { f = f | 4u; }
    if (Z < 0.0) { f = f | 8u; }
    if (Z == 0.0) {
        let b = sm_pass(p, P.qtrust, a.iy, a.ig);
        steps = steps + b.steps;
        let st = smstatic(p);
        if ((b.status == 1u) && (st.y > 0.0)) {
            let r0 = b.ig;
            if (r0 < 1.0) {
                let v = st.x * b.iy / (1.0 - r0);
                res.s = log(max(v, 1e-37)) * 0.43429448190325176;
                f = f | 4096u;
                steps = (min(steps, 131071u)) | (u32(round(clamp(r0, 0.0, 1.0) * 32767.0)) << 17u);
            } else {
                res.s = r0;
                f = f | 2048u;
                steps = min(steps, 131071u);
            }
        } else {
            f = f | 8192u;
            steps = min(steps, 131071u);
        }
    } else {
        steps = min(steps, 131071u);
    }
    res.steps = steps;
    res.flags = f;
    return res;
}

@compute @workgroup_size(8, 8)
fn main_sm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let S = (P.lev & 0xffffu) + 1u;
    let ix = gid.x * S;
    let row = (P.row0 + gid.y) * S;
    if ((ix >= P.nx) || (row >= P.ny) || (gid.y >= P.rows)) { return; }
    if (((P.lev >> 16u) != 0u) && (ix % (2u * S) == 0u) && (row % (2u * S) == 0u)) { return; }
    let iz = row / P.nys;
    let iy = row - iz * P.nys;
    PZ = P.z0 + f32(iz) * P.dz;
    K[0] = P.c[0].x; K[1] = P.c[0].y; K[2] = P.c[0].z; K[3] = P.c[0].w;
    K[4] = P.c[1].x; K[5] = P.c[1].y; K[6] = P.c[1].z; K[7] = P.c[1].w;
    K[8] = P.c[2].x; K[9] = P.c[2].y; K[10] = P.c[2].z; K[11] = P.c[2].w;
    K[12] = P.c[3].x; K[13] = P.c[3].y; K[14] = P.c[3].z; K[15] = P.c[3].w;
    let p = vec2<f32>(P.x0 + f32(ix) * P.dx, P.y0 + f32(iy) * P.dy);
    R[row * P.nx + ix] = march_sm(p);
}
