// The examples of the demo: the three time-independent systems of gpu/scripts/systems.jl (with
// the slider knobs of gpu/interactive/server.jl) and the appendix-gallery case studies of the
// paper (INq-paper: paper/scripts/studies/s08_gallery.jl, A.2-A.9 and A.11;
// s10_fractional_controller.jl, A.12) with their chart axes and validated :unwrap settings.
//
// Each `wgsl` is the characteristic function
//     D(λ, p, c)   ->   fn charD(l: CD, p: vec2<f32>) -> CD,   c[i] = K[i - 1]
// written with the complex dual helpers of march.wgsl (dadd, dmul, dscale, daddr, dexp, dlog,
// dsqrt, ddiv, dinv ...), operation for operation as the Julia reference
// (gpu/scripts/systems.jl, webgpu/validate/web_systems.jl), so the Float32 rounding matches.
//
// Fields: key, group, title, formula (shown under the selector), c (default constants),
// cnames, npow (leading order: a number or c => number; or `prep(c, wmax)` -> the effective
// order of a cleared stable denominator, measured on the host), xr, yr, xl, yl (paper axes),
// wmax (ω_max of the paper / validated), wmaxFixed (ω_max must not be changed: neutral
// systems, where a larger window buys nothing but cost, and the bar models, whose effective
// order is measured at this ω_max), kw (march options; an object or (c, xr) => object), smin
// (σ colour floor), res (default resolution), knobs (sliders on constants), hlines (dashed
// reference lines), note (extra remark under the formula), branch0 (D has a branch point at
// λ = 0: the |D| minimum at ω0 gives no root estimate).

// ---------------------------------------------------------------------------------------------
// host-side complex arithmetic (Float64) for the effective orders of the bar models
// ---------------------------------------------------------------------------------------------
const cx = (re, im = 0) => ({ re, im });
const cadd = (a, b) => cx(a.re + b.re, a.im + b.im);
const csub = (a, b) => cx(a.re - b.re, a.im - b.im);
const cmul = (a, b) => cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const csc = (a, s) => cx(a.re * s, a.im * s);
const cdiv = (a, b) => {
  const s = Math.max(Math.abs(b.re), Math.abs(b.im));
  const br = b.re / s, bi = b.im / s, d = br * br + bi * bi;
  return cx((a.re / s * br + a.im / s * bi) / d, (a.im / s * br - a.re / s * bi) / d);
};
const cexp = (a) => { const e = Math.exp(a.re); return cx(e * Math.cos(a.im), e * Math.sin(a.im)); };
const csqrt = (z) => {
  const r = Math.hypot(z.re, z.im);
  if (!(r > 0)) return cx(0, 0);
  if (z.re >= 0) { const t = Math.sqrt(0.5 * (r + z.re)); return cx(t, 0.5 * z.im / t); }
  const t = Math.sqrt(0.5 * (r - z.re));
  return cx(0.5 * Math.abs(z.im) / t, z.im < 0 ? -t : t);
};

/**
 * Unwrapped phase increment of f(iω) over [w0, w1] (Float64): coarse steps h(ω), each split
 * recursively until every principal increment is below 0.1 rad. f must have no zeros on the
 * line; the coarse step must be short enough that no single step hides a full 2π turn (here:
 * zeros of a stable denominator next to the axis contribute at most π each, spaced far wider
 * than the coarse step).
 */
export function unwrapPhase(f, w0, w1, h) {
  let evals = 0;
  const F = (w) => { evals++; return f(w); };
  const seg = (a, za, b, zb, depth) => {
    const d = Math.atan2(zb.im * za.re - zb.re * za.im, zb.re * za.re + zb.im * za.im);
    if (Math.abs(d) <= 0.1 || depth >= 40) return d;
    const m = 0.5 * (a + b);
    const zm = F(m);
    return seg(a, za, m, zm, depth + 1) + seg(m, zm, b, zb, depth + 1);
  };
  let w = w0;
  let za = F(w);
  let phi = 0;
  while (w < w1) {
    const b = Math.min(w1, w + h(w));
    const zb = F(b);
    phi += seg(w, za, b, zb, 0);
    w = b;
    za = zb;
  }
  return { phi, evals };
}

// rod: the scale-free denominator (1 + e^{-2γ})/2,  γ = λ sqrt(1 + c_e/λ) / sqrt(1 + ηλ)
function rodGamma(l, eta, ce) {
  const s1 = csqrt(cadd(cx(1), csc(cdiv(cx(1), l), ce)));
  const s2 = csqrt(cadd(cx(1), csc(l, eta)));
  return cdiv(cmul(l, s1), s2);
}
function rodDen(w, c) {
  const g = rodGamma(cx(0, w), c[0], c[1]);
  return csc(cadd(cx(1), cexp(csc(g, -2))), 0.5);
}
// FEM bar: det Q0 / q^N by the continuant recurrence, q = (4h/6)(λ + √3/h)²  (q = a_k at λ = 0)
function femDen(w, c) {
  const eta = c[0], N = Math.max(1, Math.round(c[1])), h = 1 / N;
  const l = cx(0, w), l2 = cmul(l, l), oe = cadd(cx(1), csc(l, eta));
  const lp1 = cadd(l, cx(Math.sqrt(3) / h));
  const iq = cdiv(cx(1), csc(cmul(lp1, lp1), 4 * h / 6));
  const a1 = cmul(cadd(csc(l2, 2 * h / 6), csc(oe, 1 / h)), iq);
  const ak = cmul(cadd(csc(l2, 4 * h / 6), csc(oe, 2 / h)), iq);
  const o = cmul(csub(csc(l2, h / 6), csc(oe, 1 / h)), iq);
  const o2 = cmul(o, o);
  let thp = cx(1), th = a1;
  for (let k = 2; k <= N; k++) { const t = csub(cmul(ak, th), cmul(o2, thp)); thp = th; th = t; }
  return th;
}
const neffCache = new Map();
/** effective order 2Φ_den(ω_max)/π of a parameter-independent stable denominator */
function neff(key, den, c, wmax, h) {
  const k = `${key}|${c.join(',')}|${wmax}`;
  if (neffCache.has(k)) return neffCache.get(k);
  const t0 = performance.now();
  const r = unwrapPhase((w) => den(w, c), 1e-9, wmax, h);
  const v = { npow: 2 * r.phi / Math.PI, evals: r.evals, ms: performance.now() - t0 };
  if (neffCache.size > 200) neffCache.clear();
  neffCache.set(k, v);
  return v;
}

// ---------------------------------------------------------------------------------------------
export const GROUPS = [
  { id: 'ret', label: 'Retarded (quasi-polynomials)' },
  { id: 'neu', label: 'Neutral' },
  { id: 'bar', label: 'Elastic bar with delayed boundary feedback' },
  { id: 'frac', label: 'Fractional order' },
];

const barKw = (c, xr) => ({ hmax: Math.PI / (2 * Math.max(Math.abs(xr[0]), Math.abs(xr[1]), 1)), wband: 40.0 });

export const EXAMPLES = [
  {
    key: 'fourth',
    group: 'ret',
    title: 'A.1 Fourth-order delayed oscillator',
    formula: 'D(λ) = c₁λ⁴ + λ² + 2ζλ + 1 + (P + Dλ)·e^{−τλ}',
    c: [0.03, 0.02, 0.5],
    cnames: ['c₁', 'ζ', 'τ'],
    npow: 4,
    xr: [-2.0, 4.0], yr: [-2.0, 5.0], xl: 'P', yl: 'D',
    wmax: 1e5,
    kw: {},
    smin: -0.6,
    res: '960x540',
    knobs: [
      { i: 3, name: 'delay τ', lo: 0.1, hi: 1.5 },
      { i: 2, name: 'damping ζ', lo: 0.0, hi: 0.2 },
      { i: 1, name: 'c₁ (λ⁴ coeff.)', lo: 0.005, hi: 0.1 },
    ],
    // D_fourth(λ, p, c) = c[1] * λ^4 + λ^2 + 2 * c[2] * λ + 1 + (p[1] + p[2] * λ) * exp(-c[3] * λ)
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let l2 = dmul(l, l);
    let l4 = dmul(l2, l2);
    var r = dadd(dscale(l4, K[0]), l2);
    r = dadd(r, dscale(l, 2.0 * K[1]));
    r = daddr(r, 1.0);
    let e = dexp(dscale(l, -K[2]));
    let pd = daddr(dscale(l, p.y), p.x);
    return dadd(r, dmul(pd, e));
}`,
  },
  {
    key: 'algebraic',
    group: 'ret',
    title: 'A.2 Delayed oscillator',
    formula: 'D(λ) = λ² + aλ + k + (b + gλ)·e^{−τλ}     (paper: k = g = 0, τ = 1/2)',
    c: [0.5, 0.0, 0.0],
    cnames: ['τ', 'k', 'g'],
    npow: 2,
    xr: [-1.0, 10.0], yr: [-1.0, 10.0], xl: 'a', yl: 'b',
    wmax: 1e4,
    kw: {},
    smin: -1.2,
    res: '960x540',
    knobs: [
      { i: 1, name: 'delay τ', lo: 0.1, hi: 2.0 },
      { i: 2, name: 'stiffness k', lo: -1.0, hi: 2.0 },
      { i: 3, name: 'delayed velocity gain g', lo: -0.5, hi: 1.0 },
    ],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let tau = K[0]; let k = K[1]; let g = K[2];
    let r = daddr(dadd(dmul(l, l), dscale(l, p.x)), k);
    let e = dexp(dscale(l, -tau));
    return dadd(r, dmul(daddr(dscale(l, g), p.y), e));
}`,
  },
  {
    key: 'distributed',
    group: 'ret',
    title: 'A.3 Distributed delay',
    formula: 'D(λ) = λ² + aλ + k + b·e^{−τ₀λ}(1 − e^{−τλ})/λ     (paper: k = τ₀ = 0, τ = 1)',
    note: 'Uniform kernel ∫ x(t − τ₀ − ϑ) dϑ over [0, τ] in closed form; the removable singularity at λ = 0 is evaluated by its Taylor series for |τλ| < 1/2.',
    c: [1.0, 0.0, 0.0],
    cnames: ['τ', 'k', 'τ₀'],
    npow: 2,
    xr: [-0.5, 2.0], yr: [-1.0, 5.0], xl: 'a', yl: 'b',
    wmax: 1e4,
    kw: {},
    smin: -0.6,
    res: '960x540',
    knobs: [
      { i: 1, name: 'kernel length τ', lo: 0.2, hi: 3.0 },
      { i: 2, name: 'stiffness k', lo: -1.0, hi: 2.0 },
      { i: 3, name: 'kernel start τ₀', lo: 0.0, hi: 1.0 },
    ],
    // (1 - e^{-x})/x: Horner Taylor series for |x| < 1/2
    wgsl: /* wgsl */ `
fn e1(x: CD) -> CD {
    if (cabs(x.v) < 0.5) {
        var r = daddr(dscale(x, -1.0 / 40320.0), 1.0 / 5040.0);
        r = daddr(dmul(x, r), -1.0 / 720.0);
        r = daddr(dmul(x, r), 1.0 / 120.0);
        r = daddr(dmul(x, r), -1.0 / 24.0);
        r = daddr(dmul(x, r), 1.0 / 6.0);
        r = daddr(dmul(x, r), -0.5);
        return daddr(dmul(x, r), 1.0);
    }
    let e = dexp(dneg(x));
    return ddiv(CD(C(1.0, 0.0) - e.v, -e.d), x);
}
fn charD(l: CD, p: vec2<f32>) -> CD {
    let tau = K[0]; let k = K[1]; let t0 = K[2];
    let base = daddr(dadd(dmul(l, l), dscale(l, p.x)), k);
    let kern = dmul(dexp(dscale(l, -t0)), e1(dscale(l, tau)));
    return dadd(base, dscale(kern, p.y * tau));
}`,
  },
  {
    key: 'showcase',
    group: 'ret',
    title: 'Showcase: constrained 2-DOF DAE (delayed PD)',
    formula: 'D(λ) = a₁₁a₂₂ − a₁₂²,  a₁₁ = m₁λ² + (c₁+c₂)λ + k₁+k₂ + (P + Dλ)e^{−τλ}',
    c: [1.0, 0.5, -1.0, 1.0, 0.05, 0.05, 0.5],
    cnames: ['m₁', 'm₂₃', 'k₁', 'k₂', 'c₁', 'c₂', 'τ'],
    npow: 4,
    xr: [0.5, 3.0], yr: [-0.5, 3.5], xl: 'P', yl: 'D',
    wmax: 1e5,
    kw: {},
    smin: -0.4,
    res: '960x540',
    knobs: [
      { i: 7, name: 'delay τ', lo: 0.1, hi: 1.5 },
      { i: 5, name: 'damper c₁', lo: 0.0, hi: 0.5 },
      { i: 6, name: 'damper c₂', lo: 0.0, hi: 0.5 },
    ],
    // a11 = m1 λ² + (c1 + c2) λ + (k1 + k2) + (P + Dg λ) e^{-τλ},  a12 = -(c2 λ + k2),
    // a22 = m23 λ² + c2 λ + k2,  D = a11 a22 - a12 a12
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let m1 = K[0]; let m23 = K[1]; let k1 = K[2]; let k2 = K[3];
    let c1 = K[4]; let c2 = K[5]; let tau = K[6];
    let l2 = dmul(l, l);
    var a11 = dadd(dscale(l2, m1), dscale(l, c1 + c2));
    a11 = daddr(a11, k1 + k2);
    a11 = dadd(a11, dmul(daddr(dscale(l, p.y), p.x), dexp(dscale(l, -tau))));
    let a12 = dneg(daddr(dscale(l, c2), k2));
    let a22 = daddr(dadd(dscale(l2, m23), dscale(l, c2)), k2);
    return dsub(dmul(a11, a22), dmul(a12, a12));
}`,
  },
  {
    key: 'turning',
    group: 'ret',
    title: 'A.7 Multi-mode turning lobes',
    formula: 'D(λ) = M₁M₂ + w(1 − e^{−2πλ/Ω})(M₂ + A₂M₁),  M₁ = λ² + 2ζ₁λ + 1,  M₂ = λ² + 2ζ₂ω₂λ + ω₂²',
    note: 'The rational form 1 + w(1 − e^{−τλ})(1/M₁ + A₂/M₂) with the stable modal denominators cleared (entire, n = 4); step cap h ≤ 0.05 for ω < 5 (regenerative root chain).',
    c: [0.02, 0.45, 0.03, 2.4, 2 * Math.PI],
    cnames: ['ζ₁', 'A₂', 'ζ₂', 'ω₂', '2π'],
    npow: 4,
    xr: [0.10, 1.2], yr: [0.01, 1.1], xl: 'Ω', yl: 'w',
    wmax: 1e5,
    kw: { hmax: 0.05, wband: 5.0 },
    smin: -0.05,
    res: '960x540',
    knobs: [
      { i: 1, name: 'damping ζ₁', lo: 0.005, hi: 0.1 },
      { i: 2, name: 'mode-2 weight A₂', lo: 0.0, hi: 1.5 },
      { i: 4, name: 'mode-2 frequency ω₂', lo: 1.2, hi: 4.0 },
    ],
    // M1 = λ² + 2ζ1 λ + 1, M2 = λ² + 2ζ2 ω2 λ + ω2², D = M1 M2 + w (1 - e^{-(2π/Ω)λ}) (M2 + A2 M1)
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let z1 = K[0]; let A2 = K[1]; let z2 = K[2]; let w2 = K[3]; let twopi = K[4];
    let l2 = dmul(l, l);
    let M1 = daddr(dadd(l2, dscale(l, 2.0 * z1)), 1.0);
    let M2 = daddr(dadd(l2, dscale(l, 2.0 * z2 * w2)), w2 * w2);
    let e = dexp(dscale(l, -(twopi / p.x)));
    let one_e = CD(C(1.0, 0.0) - e.v, -e.d);
    let g = dadd(M2, dscale(M1, A2));
    return dadd(dmul(M1, M2), dmul(dscale(one_e, p.y), g));
}`,
  },
  {
    key: 'neutral',
    group: 'neu',
    title: 'A.4 Neutral DDE',
    formula: 'D(λ) = λ² + aλ²e^{−τλ} + dλ + k + c·e^{−τλ}     (paper: τ = 1, d = 0, k = 1)',
    note: 'Neutral: the phase ripple never decays, so ω_max = 200 (tail error < arcsin|a|/π < 1/2), n = 2 exactly, step cap h ≤ π/(2τ) along the whole line. |a| > 1: essential instability.',
    c: [1.0, 0.0, 1.0],
    cnames: ['τ', 'd', 'k'],
    npow: 2,
    xr: [-1.2, 1.2], yr: [-1.2, 1.2], xl: 'a', yl: 'c',
    wmax: 200, wmaxFixed: true,
    kw: (c) => ({ hmax: Math.PI / (2 * c[0]), wband: Infinity }),
    smin: -0.5,
    res: '480x270',
    knobs: [
      { i: 1, name: 'delay τ', lo: 0.3, hi: 2.0 },
      { i: 2, name: 'damping d', lo: 0.0, hi: 2.0 },
      { i: 3, name: 'stiffness k', lo: 0.0, hi: 3.0 },
    ],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let tau = K[0]; let d = K[1]; let k = K[2];
    let e = dexp(dscale(l, -tau));
    let l2 = dmul(l, l);
    var r = dadd(l2, dmul(dscale(l2, p.x), e));
    r = daddr(dadd(r, dscale(l, d)), k);
    return dadd(r, dscale(e, p.y));
}`,
  },
  {
    key: 'neutral_hg',
    group: 'neu',
    title: 'A.5 High-gain neutral DDE',
    formula: 'D(λ) = λ² + aλ²e^{−τλ} + dλ + k + c·e^{−τλ}     (paper: τ = 1, d = 5, k = 0)',
    note: 'The damped high-gain variant of A.4: ω_max = 200, n = 2, step cap h ≤ π/(2τ).',
    c: [1.0, 5.0, 0.0],
    cnames: ['τ', 'd', 'k'],
    npow: 2,
    xr: [-1.2, 1.2], yr: [-1.0, 10.0], xl: 'a', yl: 'c',
    wmax: 200, wmaxFixed: true,
    kw: (c) => ({ hmax: Math.PI / (2 * c[0]), wband: Infinity }),
    smin: -0.6,
    res: '480x270',
    knobs: [
      { i: 1, name: 'delay τ', lo: 0.3, hi: 2.0 },
      { i: 2, name: 'damping d', lo: 1.0, hi: 10.0 },
      { i: 3, name: 'stiffness k', lo: 0.0, hi: 3.0 },
    ],
    wgsl: null,            // = neutral (filled below)
  },
  {
    key: 'pda',
    group: 'neu',
    title: 'A.6 PDA control (neutral, essential instability)',
    formula: 'D(λ) = λ² + 2ζλ + 1 + (P + Dλ + Aλ²)·e^{−τλ}     (paper: ζ = 0.05, D = 0.1, τ = 1)',
    note: 'Delayed acceleration feedback makes the loop neutral: for |A| > 1 the essential spectrum sits at Re λ = ln|A|/τ > 0 (dashed lines) and the count saturates (≈ ω_max τ/2π). ω_max = 500, n = 2, h ≤ π/(2τ).',
    c: [0.05, 0.1, 1.0],
    cnames: ['ζ', 'D', 'τ'],
    npow: 2,
    xr: [-1.1, 1.4], yr: [-1.15, 1.15], xl: 'P', yl: 'A',
    wmax: 500, wmaxFixed: true,
    kw: (c) => ({ hmax: Math.PI / (2 * c[2]), wband: Infinity }),
    smin: -0.5,
    res: '480x270',
    hlines: [-1, 1],
    knobs: [
      { i: 1, name: 'damping ζ', lo: 0.0, hi: 0.3 },
      { i: 2, name: 'derivative gain D', lo: -0.5, hi: 0.5 },
      { i: 3, name: 'delay τ', lo: 0.3, hi: 2.0 },
    ],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let z = K[0]; let Dg = K[1]; let tau = K[2];
    let l2 = dmul(l, l);
    let base = daddr(dadd(l2, dscale(l, 2.0 * z)), 1.0);
    let pd = dadd(daddr(dscale(l, Dg), p.x), dscale(l2, p.y));
    return dadd(base, dmul(pd, dexp(dscale(l, -tau))));
}`,
  },
  {
    key: 'rod',
    group: 'bar',
    title: 'A.8 Exact transcendental rod (Zhang & Stépán)',
    formula: 'D(λ) = 1 − K·e^{−rλ}/cosh γ,   γ = λ/√(1 + ηλ)     (r = τ/T; with external damping c: γ = λ√(1 + c/λ)/√(1 + ηλ))',
    note: 'Bar with delayed boundary feedback, exact (no discretization), Kelvin–Voigt damping η. Counted on the entire numerator cosh γ − K e^{−rλ}, with the phase of the stable denominator cosh γ as effective order n_eff = 2Φ_den(ω_max)/π (the paper’s recipe; measured on the host whenever η or c change). In Float32 cosh γ ~ e^{120} overflows at ω = 400, so numerator and denominator are both multiplied by e^{−γ}: the march sees (1 + e^{−2γ})/2 − K e^{−rλ−γ}, and n_eff of (1 + e^{−2γ})/2 is ≈ 0 (the e^{−γ} phase cancels from the count). ω_max = 400, h ≤ π/(2 r_max) for ω < 40. K = 1: a root at λ = 0 on the line.',
    c: [0.01, 0.0],
    cnames: ['η', 'c'],
    prep: (c, wmax) => neff('rod', rodDen, c, wmax, (w) => (w < 40 ? 0.01 : 0.1)),
    xr: [0.02, 10.5], yr: [-0.75, 1.0], xl: 'τ/T', yl: 'K',
    wmax: 400, wmaxFixed: true,
    kw: barKw,
    smin: -0.3,
    res: '480x270',
    knobs: [
      { i: 1, name: 'Kelvin–Voigt damping η', lo: 0.002, hi: 0.05 },
      { i: 2, name: 'external viscous damping c', lo: 0.0, hi: 1.0 },
    ],
    // N~ = cosh(γ) e^{-γ} - K e^{-rλ-γ} = (1 + e^{-2γ})/2 - K e^{-rλ-γ}
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let eta = K[0]; let ce = K[1];
    let s1 = dsqrt(daddr(dscale(dinv(l), ce), 1.0));
    let s2 = dsqrt(daddr(dscale(l, eta), 1.0));
    let g = ddiv(dmul(l, s1), s2);
    let num = dscale(daddr(dexp(dscale(g, -2.0)), 1.0), 0.5);
    let fb = dexp(dsub(dscale(l, -p.x), g));
    return dsub(num, dscale(fb, p.y));
}`,
  },
  {
    key: 'fem',
    group: 'bar',
    title: 'A.9 Same bar, finite elements (N DOF; paper: 12)',
    formula: 'D(λ) = det(λ²M + λC + K + F(λ)),  F = c(λ)e₁e_Nᵀ,  c(λ) = −K(1 + ηλ)e^{−rλ}/h,  C = ηK',
    note: 'N linear bar elements (h = 1/N). Q₀ = λ²M + λC + K is tridiagonal: det Q₀ by its continuant and (Q₀⁻¹)_{N1} = (−1)^{N+1}o^{N−1}/det Q₀, so D = det Q₀ + c(λ)(−1)^{N+1}o^{N−1} costs O(N); divided by ((4h/6)(λ + √3/h)²)^N against overflow (poles far left). Effective order from the phase of det Q₀ (host).',
    c: [0.01, 12],
    cnames: ['η', 'N'],
    prep: (c, wmax) => neff('fem', femDen, c, wmax, (w) => (w < 60 ? 0.005 : 0.05)),
    xr: [0.02, 10.5], yr: [-0.75, 1.0], xl: 'τ/T', yl: 'K',
    wmax: 400, wmaxFixed: true,
    kw: barKw,
    smin: -0.3,
    res: '320x180',
    knobs: [
      { i: 1, name: 'Kelvin–Voigt damping η', lo: 0.002, hi: 0.05 },
      { i: 2, name: 'elements N', lo: 2, hi: 24, step: 1 },
    ],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let eta = K[0];
    let N = max(1, i32(round(K[1])));
    let h = 1.0 / f32(N);
    let one_eta = daddr(dscale(l, eta), 1.0);
    let l2 = dmul(l, l);
    let lp1 = daddr(l, 1.7320508075688772 / h);
    let iq = dinv(dscale(dmul(lp1, lp1), 4.0 * h / 6.0));
    let a1 = dmul(dadd(dscale(l2, 2.0 * h / 6.0), dscale(one_eta, 1.0 / h)), iq);
    let ak = dmul(dadd(dscale(l2, 4.0 * h / 6.0), dscale(one_eta, 2.0 / h)), iq);
    let o = dmul(dsub(dscale(l2, h / 6.0), dscale(one_eta, 1.0 / h)), iq);
    let o2 = dmul(o, o);
    var thp = CD(C(1.0, 0.0), C(0.0, 0.0));
    var th = a1;
    var on = CD(C(1.0, 0.0), C(0.0, 0.0));
    for (var k = 2; k <= N; k++) {
        let t = dsub(dmul(ak, th), dmul(o2, thp));
        thp = th;
        th = t;
        on = dmul(on, o);
    }
    let sgn = select(1.0, -1.0, (N & 1) == 0);
    let fb = dmul(dmul(one_eta, dexp(dscale(l, -p.x))), dmul(on, iq));
    return dadd(th, dscale(fb, -p.y / h * sgn));
}`,
  },
  {
    key: 'frac',
    branch0: true,               // λ^μ: the minimum of |D| at ω0 is the branch point, not a root
    group: 'frac',
    title: 'A.11 Fractional-order oscillator',
    formula: 'D(λ) = λ^α + c·λ^β + k·e^{−τλ}     (paper: α = 1.8, β = 0.8, c = 0.5)',
    note: 'Principal branch, λ^μ = exp(μ log λ); n = α (non-integer). Only σ = 0 is admissible (a shifted line would cross the branch cut); the march starts at ω₀ = 1e-9, next to the branch point.',
    c: [1.8, 0.8, 0.5],
    cnames: ['α', 'β', 'c'],
    npow: (c) => Math.max(c[0], c[1]),
    xr: [0.0, 5.0], yr: [0.1, 2.0], xl: 'k', yl: 'τ',
    wmax: 1e4,
    kw: {},
    smin: -0.2,
    res: '960x540',
    knobs: [
      { i: 1, name: 'order α', lo: 1.1, hi: 1.99 },
      { i: 2, name: 'order β', lo: 0.0, hi: 1.0 },
      { i: 3, name: 'coefficient c', lo: 0.0, hi: 2.0 },
    ],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let a = K[0]; let b = K[1]; let cc = K[2];
    let L = dlog(l);
    let r = dadd(dexp(dscale(L, a)), dscale(dexp(dscale(L, b)), cc));
    return dadd(r, dscale(dexp(dscale(l, -p.y)), p.x));
}`,
  },
  {
    key: 'gao',
    branch0: true,               // λ^μ: the minimum of |D| at ω0 is the branch point, not a root
    group: 'frac',
    title: 'A.12 Fractional PI controller (Gao, Zhai & Liu)',
    formula: 'D(s) = s^μ(T s^ν + 1) + K e^{−Ls}(k_p s^μ + k_i),   plant K e^{−Ls}/(T s^ν + 1),  C(s) = k_p + k_i/s^μ',
    note: 'Gao, Zhai & Liu (2017), Example 1, μ = 1.5 (K = 5, L = 0.4, T = 10, ν = 0.5); n = μ + ν; σ = 0 only (branch cut).',
    c: [1.5, 0.4, 5.0, 10.0, 0.5],
    cnames: ['μ', 'L', 'K', 'T', 'ν'],
    npow: (c) => c[0] + c[4],
    xr: [-1.0, 6.0], yr: [-1.0, 25.0], xl: 'k_p', yl: 'k_i',
    wmax: 1e4,
    kw: {},
    smin: -0.4,
    res: '480x270',
    knobs: [
      { i: 1, name: 'controller order μ', lo: 0.3, hi: 1.9 },
      { i: 2, name: 'plant delay L', lo: 0.05, hi: 1.0 },
      { i: 3, name: 'plant gain K', lo: 1.0, hi: 10.0 },
    ],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let mu = K[0]; let Ld = K[1]; let Kp = K[2]; let Tc = K[3]; let nu = K[4];
    let L = dlog(l);
    let sm = dexp(dscale(L, mu));
    let a = dmul(sm, daddr(dscale(dexp(dscale(L, nu)), Tc), 1.0));
    let b = dscale(dmul(dexp(dscale(l, -Ld)), daddr(dscale(sm, p.x), p.y)), Kp);
    return dadd(a, b);
}`,
  },
];

EXAMPLES.find((e) => e.key === 'neutral_hg').wgsl = EXAMPLES.find((e) => e.key === 'neutral').wgsl;

/** leading order n passed to the march for constants c (and ω_max) -> { npow, info } */
export function orderOf(ex, c, wmax) {
  if (ex.prep) {
    const r = ex.prep(c, wmax);
    return { npow: r.npow, info: `n_eff = ${r.npow.toFixed(4)} (host unwrap, ${r.evals} evaluations, ${r.ms.toFixed(1)} ms)` };
  }
  const n = typeof ex.npow === 'function' ? ex.npow(c) : ex.npow;
  return { npow: n, info: `n = ${+n.toFixed(4)}` };
}

/** march options for constants c and the x range */
export function kwOf(ex, c, xr) {
  return typeof ex.kw === 'function' ? ex.kw(c, xr) : ex.kw;
}
