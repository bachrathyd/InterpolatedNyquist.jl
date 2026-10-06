// The three time-independent examples of gpu/scripts/systems.jl with the slider knobs of
// gpu/interactive/server.jl (EXAMPLES). Each `wgsl` is the characteristic function
//     D(λ, p, c)   ->   fn charD(l: CD, p: vec2<f32>) -> CD,   c[i] = K[i - 1]
// written with the complex dual helpers of march.wgsl (dadd, dmul, dscale, daddr, dexp ...),
// operation for operation as the Julia source, so the Float32 rounding matches closely.

export const EXAMPLES = [
  {
    key: 'fourth',
    title: '4th-order delayed oscillator',
    formula: 'D(λ) = c₁λ⁴ + λ² + 2ζλ + 1 + (P + Dλ)·e^{−τλ}',
    c: [0.03, 0.02, 0.5],
    cnames: ['c₁', 'ζ', 'τ'],
    npow: 4,
    xr: [-2.0, 4.0], yr: [-2.0, 5.0], xl: 'P', yl: 'D',
    kw: {},
    smin: -0.6,
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
    key: 'showcase',
    title: 'showcase 2-DOF DAE (delayed PD)',
    formula: 'D(λ) = a₁₁a₂₂ − a₁₂²,  a₁₁ = m₁λ² + (c₁+c₂)λ + k₁+k₂ + (P + Dλ)e^{−τλ}',
    c: [1.0, 0.5, -1.0, 1.0, 0.05, 0.05, 0.5],
    cnames: ['m₁', 'm₂₃', 'k₁', 'k₂', 'c₁', 'c₂', 'τ'],
    npow: 4,
    xr: [0.5, 3.0], yr: [-0.5, 3.5], xl: 'P', yl: 'D',
    kw: {},
    smin: -0.4,
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
    title: 'two-mode turning lobes',
    formula: 'D(λ) = M₁M₂ + w(1 − e^{−2πλ/Ω})(M₂ + A₂M₁),  M₁ = λ² + 2ζ₁λ + 1,  M₂ = λ² + 2ζ₂ω₂λ + ω₂²',
    c: [0.02, 0.45, 0.03, 2.4, 2 * Math.PI],
    cnames: ['ζ₁', 'A₂', 'ζ₂', 'ω₂', '2π'],
    npow: 4,
    xr: [0.10, 1.2], yr: [0.01, 1.1], xl: 'Ω', yl: 'w',
    kw: { hmax: 0.05, wband: 5.0 },
    smin: -0.05,
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
];
