// NyquistGPU :unwrap march in WGSL (Float32) -- a port of gpu/src/NyquistGPU.jl
// (seed_q / march_step(::Val{:unwrap}) / advance / finish, first-order root estimate:
// refine = 0, certify = false; schedule :pixel, one invocation per chart point).
//
// EXACT (set by the host per pipeline): the 'exact root' mode -- the dominant-root recipe of
// the paper: the 8 deepest |D| minima are tracked (with their ω), each estimate and the
// one-step estimate from ω0 are polished by up to NREF damped Newton steps in the complex
// plane, and σ = the largest real part of the converged roots.
//
// For every chart point p the phase of D(σ + iω, p) is unwrapped along the imaginary axis
// σ = 0, ω ∈ [ω0, ωmax], one evaluation of D and dD/dω (complex dual numbers) per step:
//     Z = n/2 - Δarg D / π       (roots with Re λ > σ, real-coefficient entire D)
// and the deepest |D| minima give first-order estimates of the roots near the line; the
// rightmost of them is σ, the colour of a stable point.
//
// The example's characteristic function is spliced in at the marker below (app.js):
//     fn charD(l: CD, p: vec2<f32>) -> CD      constants: K[0..15]
// (generated from the typed equation by expr.js, or hand-written for a built-in example)
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
    c: array<vec4<f32>, 4>,     // the model constants K[0..15]
    z0: f32, dz: f32,           // 3D grids: the third axis parameter PZ = z0 + iz dz
    nys: u32, lev: u32,         // rows per z slice (ny); the grid has ny * nz rows
                                // lev: progressive level, (stride − 1) | skip << 16 (0: every point)
}

struct Res {
    z: f32,         // Z_raw = n/2 - Φ/π (0 if the march failed)
    s: f32,         // σ of the rightmost tracked root (valid if flags & 256)
    steps: u32,     // attempted steps (accepted + rejected)
    flags: u32,     // 1 failed, 2 sub-resolution decision, 4 integer residual, 8 impossible (Z < 0)
                    // 128: no count (failed), 256: σ valid, 512: σ from polished roots (exact
                    // mode), 1024: σ from the one-step estimate at ω0 outside its trust region
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> R: array<Res>;

const EXACT: bool = false;      // replaced by the host for the 'exact root' pipeline
const BRANCH0: bool = false;    // replaced by the host: D has a branch point at λ = 0 (fractional
                                // powers) -- its |D| minimum at ω0 is no root estimate
const NREF: i32 = 8;            // Newton evaluations per root at most (exact mode; converged ones stop early)
const BIG: f32 = 3.0e38;
const PI: f32 = 3.14159265358979;
const EPS32: f32 = 1.1920929e-7;
const FLOATMIN: f32 = 1.17549435e-38;

var<private> K: array<f32, 16>;
var<private> PZ: f32;            // the third axis parameter (3D grids)

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

// |z| without overflowing |z|²
fn cabs(z: C) -> f32 {
    let m = max(abs(z.x), abs(z.y));
    if (!(m > 0.0)) { return 0.0; }
    let a = z.x / m;
    let b = z.y / m;
    return m * sqrt(a * a + b * b);
}
// a / b, scaled by max(|Re b|, |Im b|) (no overflow of |b|²)
fn cdiv(a: C, b: C) -> C {
    let s = max(abs(b.x), abs(b.y));
    let br = b / s;
    let ar = a / s;
    let d = br.x * br.x + br.y * br.y;
    return C(ar.x * br.x + ar.y * br.y, ar.y * br.x - ar.x * br.y) / d;
}
// principal log and square root (branch cut on the negative real axis)
fn clog(z: C) -> C { return C(log(cabs(z)), atan2m(z.y, z.x)); }
fn csqrt(z: C) -> C {
    let r = cabs(z);
    if (!(r > 0.0)) { return C(0.0, 0.0); }
    if (z.x >= 0.0) {
        let t = sqrt(0.5 * (r + z.x));
        return C(t, 0.5 * z.y / t);
    }
    let t = sqrt(0.5 * (r - z.x));
    return C(0.5 * abs(z.y) / t, select(t, -t, z.y < 0.0));
}
fn dinv(a: CD) -> CD {
    let v = cdiv(C(1.0, 0.0), a.v);
    return CD(v, -cmul(a.d, cmul(v, v)));
}
fn ddiv(a: CD, b: CD) -> CD {
    let q = cdiv(a.v, b.v);
    return CD(q, cdiv(a.d - cmul(q, b.d), b.v));
}
fn dlog(a: CD) -> CD { return CD(clog(a.v), cdiv(a.d, a.v)); }
fn dsqrt(a: CD) -> CD {
    let v = csqrt(a.v);
    return CD(v, cdiv(a.d, 2.0 * v));
}
fn dpowr(a: CD, mu: f32) -> CD { return dexp(dscale(dlog(a), mu)); }    // principal a^mu

// --- helpers of the generated code (expr.js): mixed real / complex-constant / dual operands,
// trigonometric and hyperbolic functions, exprel(x) = (e^x - 1)/x
fn daddc(a: CD, c: C) -> CD { return CD(a.v + c, a.d); }                 // + complex constant
fn dmulc(a: CD, c: C) -> CD { return CD(cmul(a.v, c), cmul(a.d, c)); }   // * complex constant
fn ddivr(a: CD, r: f32) -> CD { return CD(a.v / r, a.d / r); }           // / real
fn ddivc(a: CD, c: C) -> CD { return CD(cdiv(a.v, c), cdiv(a.d, c)); }   // / complex constant
fn cexp(z: C) -> C {
    let e = exp(z.x);
    let sc = sincos(z.y);
    return C(e * sc.y, e * sc.x);
}
fn rtan(x: f32) -> f32 { let sc = sincos(x); return sc.x / sc.y; }
// sin(x+iy) = sin x cosh y + i cos x sinh y,  cos(x+iy) = cos x cosh y - i sin x sinh y
fn dsin(a: CD) -> CD {
    let sc = sincos(a.v.x);
    let ch = cosh(a.v.y);
    let sh = sinh(a.v.y);
    return CD(C(sc.x * ch, sc.y * sh), cmul(C(sc.y * ch, -sc.x * sh), a.d));
}
fn dcos(a: CD) -> CD {
    let sc = sincos(a.v.x);
    let ch = cosh(a.v.y);
    let sh = sinh(a.v.y);
    return CD(C(sc.y * ch, -sc.x * sh), -cmul(C(sc.x * ch, sc.y * sh), a.d));
}
// sinh(x+iy) = sinh x cos y + i cosh x sin y,  cosh(x+iy) = cosh x cos y + i sinh x sin y
fn dsinh(a: CD) -> CD {
    let sc = sincos(a.v.y);
    let ch = cosh(a.v.x);
    let sh = sinh(a.v.x);
    return CD(C(sh * sc.y, ch * sc.x), cmul(C(ch * sc.y, sh * sc.x), a.d));
}
fn dcosh(a: CD) -> CD {
    let sc = sincos(a.v.y);
    let ch = cosh(a.v.x);
    let sh = sinh(a.v.x);
    return CD(C(ch * sc.y, sh * sc.x), cmul(C(sh * sc.y, ch * sc.x), a.d));
}
// tanh z = (1 - e^{-2z})/(1 + e^{-2z}) for Re z >= 0 (odd): no overflow
fn dtanh(a: CD) -> CD {
    let s = select(1.0, -1.0, a.v.x < 0.0);
    let e = cexp(-2.0 * s * a.v);
    let t = s * cdiv(C(1.0, 0.0) - e, C(1.0, 0.0) + e);
    return CD(t, cmul(C(1.0, 0.0) - cmul(t, t), a.d));
}
// tan z = -i tanh(iz)
fn dtan(a: CD) -> CD {
    let t = dtanh(CD(C(-a.v.y, a.v.x), C(-a.d.y, a.d.x)));
    return CD(C(t.v.y, -t.v.x), C(t.d.y, -t.d.x));
}
// exprel(x) = (e^x - 1)/x, entire: Horner Taylor series for |x| < 1/2 (the removable
// singularity at 0 and the cancellation of e^x - 1 near it)
fn dexprel(x: CD) -> CD {
    if (cabs(x.v) < 0.5) {
        var r = daddr(dscale(x, 1.0 / 40320.0), 1.0 / 5040.0);
        r = daddr(dmul(x, r), 1.0 / 720.0);
        r = daddr(dmul(x, r), 1.0 / 120.0);
        r = daddr(dmul(x, r), 1.0 / 24.0);
        r = daddr(dmul(x, r), 1.0 / 6.0);
        r = daddr(dmul(x, r), 0.5);
        return daddr(dmul(x, r), 1.0);
    }
    let e = dexp(x);
    return ddiv(daddr(e, -1.0), x);
}
// φ_k(w) = Σ_{j≥0} w^j/(j+k)!, k = 2..4 (integral(...) of polynomial × exp kernels; φ_1 = exprel):
// 24-term Horner Taylor series for |w| < 2 + k, else φ_i = (φ_{i-1} - 1/(i-1)!)/w from φ_0 = e^w
fn dphik(w: CD, k: i32) -> CD {
    if (cabs(w.v) < 2.0 + f32(k)) {
        var c = 1.0;
        for (var i = 1; i <= 23 + k; i++) { c = c / f32(i); }      // 1/(23+k)!
        var r = CD(C(c, 0.0), C(0.0, 0.0));
        for (var j = 22; j >= 0; j--) {
            c = c * f32(j + 1 + k);
            r = daddr(dmul(w, r), c);
        }
        return r;
    }
    var r = dexp(w);
    var f = 1.0;
    for (var i = 1; i <= k; i++) {
        r = ddiv(daddr(r, -f), w);
        f = f / f32(i);
    }
    return r;
}
fn rphik(x: f32, k: i32) -> f32 {
    if (abs(x) < 2.0 + f32(k)) {
        var c = 1.0;
        for (var i = 1; i <= 23 + k; i++) { c = c / f32(i); }
        var r = c;
        for (var j = 22; j >= 0; j--) {
            c = c * f32(j + 1 + k);
            r = x * r + c;
        }
        return r;
    }
    var r = exp(x);
    var f = 1.0;
    for (var i = 1; i <= k; i++) {
        r = (r - f) / x;
        f = f / f32(i);
    }
    return r;
}
fn rexprel(x: f32) -> f32 {
    if (abs(x) < 0.5) {
        var r = x / 40320.0 + 1.0 / 5040.0;
        r = x * r + 1.0 / 720.0;
        r = x * r + 1.0 / 120.0;
        r = x * r + 1.0 / 24.0;
        r = x * r + 1.0 / 6.0;
        r = x * r + 0.5;
        return x * r + 1.0;
    }
    return (exp(x) - 1.0) / x;
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
// [t0, t1] ⊂ [0, 1]: the bracket of the minimum (default mode: the whole step)
// returns (depth, σ_est, ω_est, trusted)  (exact mode: an untrusted estimate -> (depth, σ, ω_min, 1))
fn hslope(A: C, Aw: C, B: C, Bw: C, h: f32, t: f32) -> f32 {
    let hm = hermite(A, Aw, B, Bw, h, t);
    return hm.p.x * hm.dp.x + hm.p.y * hm.dp.y;
}
fn dip_root(Da: C, Dwa: C, sa: f32, Db: C, Dwb: C, sb: f32, a: f32, h: f32, sig: f32, ct: f32,
            t0: f32, t1: f32) -> vec4<f32> {
    let sc = 1.0 / max(sa, sb);
    let A = Da * sc;
    let Aw = Dwa * sc;
    let B = Db * sc;
    let Bw = Dwb * sc;
    var tl = t0;
    var tr = t1;
    var fl = A.x * Aw.x + A.y * Aw.y;
    var fr = B.x * Bw.x + B.y * Bw.y;
    if (t0 > 0.0) { fl = hslope(A, Aw, B, Bw, h, t0); }
    if (t1 < 1.0) { fr = hslope(A, Aw, B, Bw, h, t1); }
    let wd = t1 - t0;
    var t = clamp(t0 + wd * (fl / (fl - fr)), t0 + 0.01 * wd, t1 - 0.01 * wd);
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
    if (!(sp > 0.0)) { return vec4<f32>(0.0); }
    let depth = sp * sqrt((hm.p.x / sp) * (hm.p.x / sp) + (hm.p.y / sp) * (hm.p.y / sp)) / sc;
    let trusted = within(q, ct * h) && (depth < BIG);
    if (EXACT && !trusted && (depth < BIG)) {
        // exact mode: an estimate outside the trust region (a root further from the line than
        // the local step resolves) is not dropped -- the minimum ON the line seeds the Newton
        // polish, which validates it (a root it does not converge to is discarded)
        // (stored with a negative ω: no first-order fallback for such a seed; ranked behind
        // every trusted minimum: they only fill free slots)
        return vec4<f32>(select(depth * 1e20, 0.5 * BIG, depth > 1e18), sig, -(a + t * h), 1.0);
    }
    return vec4<f32>(depth, sig + q.y, a + t * h - q.x, select(0.0, 1.0, trusted));
}

// tracked minima: depths dd (ascending, BIG = empty) and their σ estimates ds (-BIG = empty);
// exact mode: 8 slots (dd, dd1 ...) with the ω estimates dw
var<private> dd: vec4<f32>;
var<private> ds: vec4<f32>;
var<private> dd1: vec4<f32>;
var<private> ds1: vec4<f32>;
var<private> dw: vec4<f32>;
var<private> dw1: vec4<f32>;

fn insert8(d: f32, s: f32, w: f32) {
    if (!(abs(s) < BIG) || !(abs(w) < BIG)) { return; }
    let one = vec4<f32>(1.0);
    let k = i32(dot(select(vec4<f32>(0.0), one, dd <= vec4<f32>(d)), one) +
                dot(select(vec4<f32>(0.0), one, dd1 <= vec4<f32>(d)), one));
    if (k > 7) { return; }
    let idx = vec4<i32>(0, 1, 2, 3);
    let id1 = vec4<i32>(4, 5, 6, 7);
    let kk = vec4<i32>(k);
    let shd = vec4<f32>(dd.x, dd.x, dd.y, dd.z);
    let shs = vec4<f32>(ds.x, ds.x, ds.y, ds.z);
    let shw = vec4<f32>(dw.x, dw.x, dw.y, dw.z);
    let shd1 = vec4<f32>(dd.w, dd1.x, dd1.y, dd1.z);
    let shs1 = vec4<f32>(ds.w, ds1.x, ds1.y, ds1.z);
    let shw1 = vec4<f32>(dw.w, dw1.x, dw1.y, dw1.z);
    dd1 = select(select(shd1, vec4<f32>(d), id1 == kk), dd1, id1 < kk);
    ds1 = select(select(shs1, vec4<f32>(s), id1 == kk), ds1, id1 < kk);
    dw1 = select(select(shw1, vec4<f32>(w), id1 == kk), dw1, id1 < kk);
    dd = select(select(shd, vec4<f32>(d), idx == kk), dd, idx < kk);
    ds = select(select(shs, vec4<f32>(s), idx == kk), ds, idx < kk);
    dw = select(select(shw, vec4<f32>(w), idx == kk), dw, idx < kk);
}

fn track(d: f32, s: f32, w: f32) {
    if (EXACT) { insert8(d, s, w); } else { insert_root(d, s); }
}

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

// Newton polish of one root estimate λ = s + iw in the complex plane (exact mode; cf.
// newton_root of NyquistGPU.jl): λ ← λ − D/D_λ with D_λ = −i D_ω from the same dual-number
// evaluation; a step longer than max(|λ|, 1)/2 is shortened to it, and a step is kept only if it
// decreases |D| (otherwise it is halved, within the budget of NREF evaluations).
// Returns (σ, ω, converged, evaluations); converged: the last Newton step from the best point
// was below 1e-3 max(1, |λ|/100).
fn polish(p: vec2<f32>, s0: f32, w0: f32) -> vec4<f32> {
    var bs = s0;
    var bw = w0;
    // a seed on the real axis: Newton for a real-coefficient D would never leave it
    if (abs(bw) < 1e-3 * max(abs(bs), 1.0)) { bw = max(abs(bs), 0.1); }
    var s = bs;
    var w = bw;
    var bm = BIG;               // |D| at the best point
    var ns = 0.0;               // Newton step from the best point
    var nw = 0.0;
    var ts = 0.0;               // current trial step from the best point
    var tw = 0.0;
    var conv = false;
    var ne = 0.0;
    var nrej = 0;
    var qprev = 0.0;            // the previous accepted Newton step (0: none yet)
    for (var it = 0; it < NREF; it++) {
        let L = charD(CD(C(s, w), C(0.0, 1.0)), p);
        ne = ne + 1.0;
        let m = cabs(L.v);
        if (m < bm) {
            bs = s; bw = w; bm = m;
            let q = newton_q(L.v, L.d);
            if (q.z == 0.0) { conv = false; break; }
            var es = q.y;
            var ew = -q.x;
            let r = max(cabs(C(s, w)), 1.0);
            let rs = 0.5 * r;                          // longest step
            let ql = cabs(C(es, ew));
            if (!(ql < BIG)) { conv = false; break; }
            if (ql > rs) { es = es * (rs / ql); ew = ew * (rs / ql); }
            ns = es; nw = ew;
            ts = es; tw = ew;
            // converged: an absolute 1e-3 (the colour scale), relaxed above |λ| = 100 to the
            // Float32 resolution of λ (a relative test alone accepts garbage at high frequency)
            // and contracting like Newton's quadratic convergence (a drift into a branch point,
            // where D' blows up and D/D' -> 0 without a root, contracts only linearly)
            conv = (ql <= 4e-6 * r) || ((ql <= 1e-3 * max(1.0, 0.01 * r)) && (ql <= 0.1 * qprev));
            qprev = ql;
            if (ql <= 4e-6 * r) { break; }            // at the Float32 resolution
            nrej = 0;
        } else {
            if (it == 0) { break; }                    // D not finite at the seed
            nrej = nrej + 1;
            if (nrej >= 3) { break; }                  // stuck (a saddle of |D|, a branch point)
            ts = 0.5 * ts;
            tw = 0.5 * tw;
        }
        s = bs + ts;
        w = bw + tw;
    }
    // a root far from its seed is not the root of that minimum: Newton thrown off from a saddle
    // of |D| (two close roots sharing one dip), where a scaled D can be smaller far away
    let near = cabs(C(bs + ns - s0, bw + nw - w0)) <= max(1.0, 0.5 * cabs(C(s0, w0)));
    return vec4<f32>(bs + ns, abs(bw + nw), select(0.0, 1.0, conv && near && (bm < BIG)), ne);
}

fn march(p: vec2<f32>) -> Res {
    let sig = P.sigma;
    dd = vec4<f32>(BIG);
    ds = vec4<f32>(-BIG);
    if (EXACT) {
        dd1 = vec4<f32>(BIG);
        ds1 = vec4<f32>(-BIG);
        dw = vec4<f32>(0.0);
        dw1 = vec4<f32>(0.0);
    }

    // --- seed_q ---
    var w = P.w0;
    var L = charD(CD(C(sig, w), C(0.0, 1.0)), p);
    var S = sample_data(L.v, L.d);
    var sorg = -BIG;            // one-step estimate from ω0, also outside the trust region
    var otr = false;            // ... and it was tracked (inside the trust region)
    if (S.ok && (S.g > 0.0) && !BRANCH0) {
        // |D(σ+iω)| is even in ω: growing away from ω = 0, the minimum sits AT ω = 0
        let q = newton_q(L.v, L.d);
        let depth = S.s * sqrt(S.u.x * S.u.x + S.u.y * S.u.y);
        if (within(q, P.qtrust * P.hrel * max(w, 1.0))) { track(depth, sig + q.y, 0.0); otr = true; }
        if ((q.z != 0.0) && (abs(q.y) < 1e6)) { sorg = sig + q.y; }
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
            if (EXACT) {
                // exact mode: a minimum of the step's Hermite model also where the end slopes
                // do not bracket it (a long step can hold a maximum and a minimum): the first
                // sign change - -> + of d|p|²/dt over 4 sub-intervals
                let sc = 1.0 / max(sa, Sb.s);
                let A = Da * sc;
                let Aw = Dwa * sc;
                let B = Lb.v * sc;
                let Bw = Lb.d * sc;
                var tp = 0.0;
                var fp = A.x * Aw.x + A.y * Aw.y;
                var t0 = -1.0;
                for (var k = 1; k <= 4; k++) {
                    let tk = f32(k) * 0.25;
                    var fk = B.x * Bw.x + B.y * Bw.y;
                    if (k < 4) { fk = hslope(A, Aw, B, Bw, hs, tk); }
                    if ((t0 < 0.0) && (fp < 0.0) && (fk > 0.0)) { t0 = tp; }
                    tp = tk;
                    fp = fk;
                }
                if (t0 >= 0.0) {
                    let r = dip_root(Da, Dwa, sa, Lb.v, Lb.d, Sb.s, w, hs, sig, P.qtrust, t0, t0 + 0.25);
                    if (r.w != 0.0) { track(r.x, r.y, r.z); }
                }
            } else if ((ga < 0.0) && (Sb.g > 0.0)) {
                let r = dip_root(Da, Dwa, sa, Lb.v, Lb.d, Sb.s, w, hs, sig, P.qtrust, 0.0, 1.0);
                if (r.w != 0.0) { track(r.x, r.y, r.z); }
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
    var f = flags;
    var Z = 0.0;
    if (status == 1u) {
        let zr = P.npow / 2.0 - phi / PI;
        res.z = zr;
        Z = round(zr);
        if (abs(zr - Z) > 0.25) { f = f | 4u; }
        if (Z < 0.0) { f = f | 8u; }
    } else {
        res.z = 0.0;
        f = f | 1u | 128u;
    }
    // first-order estimate: the rightmost of the 4 deepest minima (NyquistGPU, nroots = 4).
    // Where the count proves the point stable, an estimate right of σ + 0.02 is spurious (a
    // shallow ripple minimum located on a long step -- the paper's consistency filter); the
    // engine keeps it, the colouring here does not.
    let lim = vec4<f32>(select(BIG, sig + 0.02, (status == 1u) && (Z == 0.0)));
    let d0 = select(vec4<f32>(-BIG), ds, ds <= lim);
    var sd = max(max(d0.x, d0.y), max(d0.z, d0.w));
    if (EXACT) {
        // (the line seeds of untrusted minima, ω < 0, carry no estimate)
        let a = select(vec4<f32>(-BIG), ds, (dw >= vec4<f32>(0.0)) & (ds <= lim));
        let b = select(vec4<f32>(-BIG), ds1, (dw1 >= vec4<f32>(0.0)) & (ds1 <= lim));
        let m = max(a, b);
        sd = max(max(m.x, m.y), max(m.z, m.w));
    }
    if (!(sd > -0.5 * BIG) && (sorg > -0.5 * BIG) && (sorg <= lim.x)) {
        // no trusted estimate at all: the one-step estimate from ω0 (as the paper's charts)
        sd = sorg;
        f = f | 1024u;
    }
    var nev = 0u;
    if (EXACT) {
        // polish the 8 tracked minima and the estimate from ω0; σ = the rightmost converged
        // root (where the count proves the point stable, a root right of the line is spurious)
        var sx = -BIG;
        for (var j = 0; j < 9; j++) {
            var s0 = select(sorg, -BIG, otr);     // (a tracked one is polished from its slot)
            var w0 = 0.0;
            if (j < 4) { s0 = ds[j]; w0 = dw[j]; }
            else if (j < 8) { s0 = ds1[j - 4]; w0 = dw1[j - 4]; }
            if (s0 > -0.5 * BIG) {
                let r = polish(p, s0, abs(w0));
                nev = nev + u32(r.w);
                // a converged root; else the first-order estimate of a trusted seed (Newton can
                // fail from a saddle of |D|, e.g. between two close roots that share one dip)
                var sc = r.x;
                if (r.z == 0.0) { sc = select(s0, -BIG, (w0 < 0.0) || (j == 8)); }
                if ((sc > sx) && (sc <= lim.x)) { sx = sc; }
            }
        }
        if (sx > -0.5 * BIG) {
            sd = sx;
            f = (f & ~1024u) | 512u;
        }
    }
    res.s = sd;
    if (sd > -0.5 * BIG) { f = f | 256u; }
    res.steps = steps + nev;
    res.flags = f;
    return res;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // progressive levels: the threads cover the stride-S lattice (row0, rows in lattice rows);
    // skip: the nodes of the stride-2S lattice are done already (the previous level)
    let S = (P.lev & 0xffffu) + 1u;
    let ix = gid.x * S;
    let row = (P.row0 + gid.y) * S;           // row of the (ny * nz)-row grid
    if ((ix >= P.nx) || (row >= P.ny) || (gid.y >= P.rows)) { return; }
    if (((P.lev >> 16u) != 0u) && (ix % (2u * S) == 0u) && (row % (2u * S) == 0u)) { return; }
    let iz = row / P.nys;
    let iy = row - iz * P.nys;
    PZ = P.z0 + f32(iz) * P.dz;
    // (unrolled: constant indices keep K in registers)
    K[0] = P.c[0].x; K[1] = P.c[0].y; K[2] = P.c[0].z; K[3] = P.c[0].w;
    K[4] = P.c[1].x; K[5] = P.c[1].y; K[6] = P.c[1].z; K[7] = P.c[1].w;
    K[8] = P.c[2].x; K[9] = P.c[2].y; K[10] = P.c[2].z; K[11] = P.c[2].w;
    K[12] = P.c[3].x; K[13] = P.c[3].y; K[14] = P.c[3].z; K[15] = P.c[3].w;
    let p = vec2<f32>(P.x0 + f32(ix) * P.dx, P.y0 + f32(iy) * P.dy);
    R[row * P.nx + ix] = march(p);
}
