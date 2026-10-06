// NyquistGPU :unwrap march in WGSL (Float32) -- a port of gpu/src/NyquistGPU.jl
// (seed_q / march_step(::Val{:unwrap}) / advance / finish, first-order root estimate:
// refine = 0, certify = false; schedule :pixel, one invocation per chart point).
//
// For every chart point p the phase of D(σ + iω, p) is unwrapped along the imaginary axis
// σ = 0, ω ∈ [ω0, ωmax], one evaluation of D and dD/dω (complex dual numbers) per step:
//     Z = n/2 - Δarg D / π       (roots with Re λ > σ, real-coefficient entire D)
// and the deepest |D| minima give first-order estimates of the roots near the line; the
// rightmost of them is σ, the colour of a stable point.
//
// The example's characteristic function is spliced in at the marker below (app.js):
//     fn charD(l: CD, p: vec2<f32>) -> CD      constants: K[0..7]
//
// Portability: WGSL does not guarantee IEEE infinities / NaN (and has no isnan), so the
// NaN / Inf sentinels of the Julia code are replaced by explicit validity flags and BIG.
// sin/cos/atan2 are written out (Cephes single-precision polynomials) instead of the
// builtins, whose accuracy WGSL only bounds on [-π, π] and which differ between GPUs.

alias C = vec2<f32>;

// complex dual number: value v and derivative d = dv/dω (λ = σ + iω  =>  dλ/dω = i)
struct CD {
    v: C,
    d: C,
}

struct Params {
    x0: f32, dx: f32, y0: f32, dy: f32,
    nx: u32, ny: u32, row0: u32, rows: u32,
    w0: f32, wmax: f32, h0: f32, tol: f32,
    hrel: f32, hmax: f32, wband: f32, npow: f32,
    qtrust: f32, growmax: f32, maxsteps: u32, sigma: f32,
    c0: vec4<f32>,
    c1: vec4<f32>,
}

struct Res {
    z: f32,         // Z_raw = n/2 - Φ/π (0 if the march failed)
    s: f32,         // σ of the rightmost tracked root (valid if flags & 256)
    steps: u32,     // attempted steps (accepted + rejected)
    flags: u32,     // 1 failed, 2 sub-resolution decision, 4 integer residual, 8 impossible (Z < 0)
                    // 128: no count (failed), 256: σ valid
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> R: array<Res>;

const BIG: f32 = 3.0e38;
const PI: f32 = 3.14159265358979;
const EPS32: f32 = 1.1920929e-7;
const FLOATMIN: f32 = 1.17549435e-38;

var<private> K: array<f32, 8>;

// ---------------------------------------------------------------------------
// elementary functions
// ---------------------------------------------------------------------------
fn cmul(a: C, b: C) -> C { return C(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// (sin x, cos x): reduction by π/2 (3-part Cody-Waite), Cephes sinf/cosf on [-π/4, π/4]
fn sincos(x: f32) -> vec2<f32> {
    let j = round(x * 0.63661977236758134);
    var r = x - j * 1.5703125;
    r = r - j * 4.837512969970703125e-4;
    r = r - j * 7.54978995489188216e-8;
    let z = r * r;
    let s = r + r * z * ((-1.9515295891e-4 * z + 8.3321608736e-3) * z - 1.6666654611e-1);
    let c = 1.0 - 0.5 * z + z * z * ((2.443315711809948e-5 * z - 1.388731625493765e-3) * z + 4.166664568298827e-2);
    let q = i32(j) & 3;
    if (q == 0) { return vec2<f32>(s, c); }
    if (q == 1) { return vec2<f32>(c, -s); }
    if (q == 2) { return vec2<f32>(-s, -c); }
    return vec2<f32>(-c, s);
}

// atan2 from Cephes atanf on t = min/max ∈ [0, 1]
fn atan2m(y: f32, x: f32) -> f32 {
    let ax = abs(x);
    let ay = abs(y);
    let mx = max(ax, ay);
    if (mx == 0.0) { return 0.0; }
    var t = min(ax, ay) / mx;
    var y0 = 0.0;
    if (t > 0.4142135623730950) {
        y0 = 0.25 * PI;
        t = (t - 1.0) / (t + 1.0);
    }
    let z = t * t;
    var a = y0 + ((((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z
                   - 3.33329491539e-1) * z * t + t);
    if (ay > ax) { a = 0.5 * PI - a; }
    if (x < 0.0) { a = PI - a; }
    if (y < 0.0) { a = -a; }
    return a;
}

fn cbrtp(x: f32) -> f32 { return exp2(log2(x) * (1.0 / 3.0)); }    // x > 0

// ---------------------------------------------------------------------------
// complex dual arithmetic (forward mode, one derivative: d/dω)
// ---------------------------------------------------------------------------
fn dadd(a: CD, b: CD) -> CD { return CD(a.v + b.v, a.d + b.d); }
fn dsub(a: CD, b: CD) -> CD { return CD(a.v - b.v, a.d - b.d); }
fn dneg(a: CD) -> CD { return CD(-a.v, -a.d); }
fn dmul(a: CD, b: CD) -> CD { return CD(cmul(a.v, b.v), cmul(a.d, b.v) + cmul(a.v, b.d)); }
fn dscale(a: CD, s: f32) -> CD { return CD(a.v * s, a.d * s); }          // real scalar
fn daddr(a: CD, r: f32) -> CD { return CD(a.v + C(r, 0.0), a.d); }       // + real constant
fn dexp(a: CD) -> CD {
    let e = exp(a.v.x);
    let sc = sincos(a.v.y);
    let v = C(e * sc.y, e * sc.x);
    return CD(v, cmul(v, a.d));
}

//#CHARD#

// ---------------------------------------------------------------------------
// the march (NyquistGPU.jl: eval_line, sample_data, dphase, newton_q, hermite, dip_root,
// insert_root, seed_q, trial_step, march_step(:unwrap), advance, finish)
// ---------------------------------------------------------------------------
struct Smp {
    u: C,       // D / s
    th: f32,    // θ' = dΦ/dω
    g: f32,     // sign of d|D|/dω
    s: f32,     // max(|Re D|, |Im D|)
    ok: bool,   // θ' finite and s > 0
}

fn sample_data(Dv: C, Dw: C) -> Smp {
    let s = max(abs(Dv.x), abs(Dv.y));
    if (!(s > 0.0) || !(s < BIG)) { return Smp(C(1.0, 0.0), 0.0, 0.0, 0.0, false); }
    let u = Dv / s;
    let u2 = u.x * u.x + u.y * u.y;
    let th = (u.x * Dw.y - u.y * Dw.x) / (s * u2);
    let g = u.x * Dw.x + u.y * Dw.y;
    let ok = (abs(th) < BIG) && (abs(g) < BIG);
    return Smp(u, th, g, s, ok);
}

// principal increment of arg between two scaled samples
fn dphase(ua: C, ub: C) -> f32 {
    let pr = ub.x * ua.x + ub.y * ua.y;
    let pi_ = ub.y * ua.x - ub.x * ua.y;
    return atan2m(pi_, pr);
}

// q = D / D' (D' = dD/dω); .z = 1 if valid. Newton step from ω: σ_est = σ + Im q.
fn newton_q(Dv: C, Dw: C) -> vec3<f32> {
    let sw = max(abs(Dw.x), abs(Dw.y));
    if (!(sw > 0.0)) { return vec3<f32>(0.0, 0.0, 0.0); }
    let vr = Dw.x / sw;
    let vi = Dw.y / sw;
    let v2 = vr * vr + vi * vi;
    let ar = Dv.x / sw;
    let ai = Dv.y / sw;
    return vec3<f32>((ar * vr + ai * vi) / v2, (ai * vr - ar * vi) / v2, 1.0);
}

// |q| <= r without overflowing |q|²
fn within(q: vec3<f32>, r: f32) -> bool {
    if (q.z == 0.0) { return false; }
    if (max(abs(q.x), abs(q.y)) > r) { return false; }
    return q.x * q.x + q.y * q.y <= r * r;
}

struct Herm {
    p: C,
    dp: C,
}

fn hermite(A: C, Aw: C, B: C, Bw: C, h: f32, t: f32) -> Herm {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    let p = h00 * A + (h10 * h) * Aw + h01 * B + (h11 * h) * Bw;
    let d00 = 6.0 * t2 - 6.0 * t;
    let d10 = 3.0 * t2 - 4.0 * t + 1.0;
    let d11 = 3.0 * t2 - 2.0 * t;
    let dp = (d00 / h) * (A - B) + d10 * Aw + d11 * Bw;
    return Herm(p, dp);
}

// minimum of |D| inside an accepted step whose end slopes bracket it (g_a < 0 < g_b):
// Illinois on Re(conj(p) p') of the Hermite model, one Newton step from the minimum.
// returns (depth, σ_est, trusted)
fn dip_root(Da: C, Dwa: C, sa: f32, Db: C, Dwb: C, sb: f32, h: f32, sig: f32, ct: f32) -> vec3<f32> {
    let sc = 1.0 / max(sa, sb);
    let A = Da * sc;
    let Aw = Dwa * sc;
    let B = Db * sc;
    let Bw = Dwb * sc;
    var tl = 0.0;
    var tr = 1.0;
    var fl = A.x * Aw.x + A.y * Aw.y;
    var fr = B.x * Bw.x + B.y * Bw.y;
    var t = clamp(fl / (fl - fr), 0.01, 0.99);
    var side = 0;
    for (var it = 0; it < 4; it++) {
        let hm = hermite(A, Aw, B, Bw, h, t);
        let f = hm.p.x * hm.dp.x + hm.p.y * hm.dp.y;
        if (f < 0.0) {
            tl = t; fl = f;
            if (side == -1) { fr = fr / 2.0; }
            side = -1;
        } else {
            tr = t; fr = f;
            if (side == 1) { fl = fl / 2.0; }
            side = 1;
        }
        let den = fr - fl;
        if (den != 0.0) { t = (tl * fr - tr * fl) / den; } else { t = (tl + tr) / 2.0; }
        t = clamp(t, tl, tr);
    }
    let hm = hermite(A, Aw, B, Bw, h, t);
    let q = newton_q(hm.p, hm.dp);
    let sp = max(abs(hm.p.x), abs(hm.p.y));
    if (!(sp > 0.0)) { return vec3<f32>(0.0, 0.0, 0.0); }
    let depth = sp * sqrt((hm.p.x / sp) * (hm.p.x / sp) + (hm.p.y / sp) * (hm.p.y / sp)) / sc;
    let trusted = within(q, ct * h) && (depth < BIG);
    return vec3<f32>(depth, sig + q.y, select(0.0, 1.0, trusted));
}

// tracked minima: depths dd (ascending, BIG = empty) and their σ estimates ds (-BIG = empty)
var<private> dd: vec4<f32>;
var<private> ds: vec4<f32>;

fn insert_root(d: f32, s: f32) {
    if (!(abs(s) < BIG)) { return; }
    let le = select(vec4<f32>(0.0), vec4<f32>(1.0), dd <= vec4<f32>(d));
    let k = i32(dot(le, vec4<f32>(1.0)));          // 0-based slot
    if (k > 3) { return; }
    let idx = vec4<i32>(0, 1, 2, 3);
    let kk = vec4<i32>(k);
    let shd = vec4<f32>(dd.x, dd.x, dd.y, dd.z);
    let shs = vec4<f32>(ds.x, ds.x, ds.y, ds.z);
    dd = select(select(shd, vec4<f32>(d), idx == kk), dd, idx < kk);
    ds = select(select(shs, vec4<f32>(s), idx == kk), ds, idx < kk);
}

fn hfloor(w: f32) -> f32 { return 8.0 * EPS32 * max(w, 1.0); }

fn march(p: vec2<f32>) -> Res {
    let sig = P.sigma;
    dd = vec4<f32>(BIG);
    ds = vec4<f32>(-BIG);

    // --- seed_q ---
    var w = P.w0;
    var L = charD(CD(C(sig, w), C(0.0, 1.0)), p);
    var S = sample_data(L.v, L.d);
    if (S.ok && S.g > 0.0) {
        // |D(σ+iω)| is even in ω: growing away from ω = 0, the minimum sits AT ω = 0
        let q = newton_q(L.v, L.d);
        let depth = S.s * sqrt(S.u.x * S.u.x + S.u.y * S.u.y);
        if (within(q, P.qtrust * P.hrel * max(w, 1.0))) { insert_root(depth, sig + q.y); }
    }
    var status = select(2u, 0u, S.ok);      // 0 running, 1 finished, 2 failed
    var h = P.h0;
    var phi = 0.0;
    var Da = L.v;
    var Dwa = L.d;
    var ua = S.u;
    var sa = S.s;
    var tha = S.th;
    var ga = S.g;
    var steps = 0u;
    var flags = 0u;

    loop {
        if (status != 0u) { break; }
        // --- trial_step ---
        let rest = P.wmax - w;
        var ht = min(h, P.hrel * max(w, 1.0));
        if (w < P.wband) { ht = min(ht, P.hmax); }
        let last = ht >= rest;
        let hs = select(ht, rest, last);
        let b = select(w + ht, P.wmax, last);
        // --- one evaluation ---
        let Lb = charD(CD(C(sig, b), C(0.0, 1.0)), p);
        let Sb = sample_data(Lb.v, Lb.d);
        var dl = 0.0;
        if (Sb.ok) { dl = dphase(ua, Sb.u); }
        let valid = Sb.ok;
        var err = BIG;
        if (valid) {
            err = abs(dl - hs * (tha + Sb.th) / 2.0);
            if (!(err < BIG)) { err = BIG; }
        }
        let accept = valid && (err <= P.tol);
        let fac = clamp(0.9 * cbrtp(P.tol / max(err, FLOATMIN)), 0.2, P.growmax);
        var hn = hs * fac;
        steps = steps + 1u;
        var take = accept;
        var phin = phi + dl;
        var fl = 0u;
        if (!accept && valid && (hn <= hfloor(w))) {
            // a root closer to the line than the smallest step: decide the branch by its side
            let q = newton_q(Lb.v, Lb.d);
            let right = (q.z != 0.0) && (q.y > 0.0);
            var dr = dl;
            if (abs(dl) > PI / 2.0) {
                if (right && (dl > 0.0)) { dr = dl - 2.0 * PI; }
                else if ((!right) && (dl < 0.0)) { dr = dl + 2.0 * PI; }
            }
            take = true;
            hn = hs;
            phin = phi + dr;
            fl = 2u;
        }
        if (take) {
            // --- advance (accepted) ---
            let fin = b >= P.wmax;
            if ((ga < 0.0) && (Sb.g > 0.0)) {
                let r = dip_root(Da, Dwa, sa, Lb.v, Lb.d, Sb.s, hs, sig, P.qtrust);
                if (r.z != 0.0) { insert_root(r.x, r.y); }
            }
            if (fin) { status = 1u; } else if (steps >= P.maxsteps) { status = 2u; }
            w = b;
            h = hn;
            phi = phin;
            Da = Lb.v;
            Dwa = Lb.d;
            ua = Sb.u;
            sa = Sb.s;
            tha = Sb.th;
            ga = Sb.g;
            flags = flags | fl;
        } else {
            // --- advance (rejected) ---
            if ((hn <= hfloor(w)) || (steps >= P.maxsteps)) { status = 2u; }
            h = hn;
        }
    }

    // --- finish ---
    var res: Res;
    res.steps = steps;
    var f = flags;
    if (status == 1u) {
        let zr = P.npow / 2.0 - phi / PI;
        res.z = zr;
        let Z = round(zr);
        if (abs(zr - Z) > 0.25) { f = f | 4u; }
        if (Z < 0.0) { f = f | 8u; }
    } else {
        res.z = 0.0;
        f = f | 1u | 128u;
    }
    let sd = max(max(ds.x, ds.y), max(ds.z, ds.w));
    res.s = sd;
    if (sd > -0.5 * BIG) { f = f | 256u; }
    res.flags = f;
    return res;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = P.row0 + gid.y;
    if ((ix >= P.nx) || (iy >= P.ny) || (gid.y >= P.rows)) { return; }
    K[0] = P.c0.x; K[1] = P.c0.y; K[2] = P.c0.z; K[3] = P.c0.w;
    K[4] = P.c1.x; K[5] = P.c1.y; K[6] = P.c1.z; K[7] = P.c1.w;
    let p = vec2<f32>(P.x0 + f32(ix) * P.dx, P.y0 + f32(iy) * P.dy);
    R[iy * P.nx + ix] = march(p);
}
