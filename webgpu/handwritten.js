// The hand-written charD of each text-defined example (the demo's first version, commit 1239398), kept for
// the speed comparison generated vs hand-written (index.html?bench=compare). K[i] = c[i] in the
// order below, p = the example's chart axes (the same as in examples.js). Not used otherwise
// (the built-in FEM bar keeps its hand-written charD in examples.js).

export const HANDWRITTEN = {
  fourth: {
    c: [0.03, 0.02, 0.5],
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
  algebraic: {
    c: [0.5, 0.0, 0.0],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let tau = K[0]; let k = K[1]; let g = K[2];
    let r = daddr(dadd(dmul(l, l), dscale(l, p.x)), k);
    let e = dexp(dscale(l, -tau));
    return dadd(r, dmul(daddr(dscale(l, g), p.y), e));
}`,
  },
  distributed: {
    c: [1.0, 0.0, 0.0],
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
  showcase: {
    c: [1.0, 0.5, -1.0, 1.0, 0.05, 0.05, 0.5],
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
  turning: {
    c: [0.02, 0.45, 0.03, 2.4, 2 * Math.PI],
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
  neutral: {
    c: [1.0, 0.0, 1.0],
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
  pda: {
    c: [0.05, 0.1, 1.0],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let z = K[0]; let Dg = K[1]; let tau = K[2];
    let l2 = dmul(l, l);
    let base = daddr(dadd(l2, dscale(l, 2.0 * z)), 1.0);
    let pd = dadd(daddr(dscale(l, Dg), p.x), dscale(l2, p.y));
    return dadd(base, dmul(pd, dexp(dscale(l, -tau))));
}`,
  },
  rod: {
    c: [0.01, 0.0],
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
  frac: {
    c: [1.8, 0.8, 0.5],
    wgsl: /* wgsl */ `
fn charD(l: CD, p: vec2<f32>) -> CD {
    let a = K[0]; let b = K[1]; let cc = K[2];
    let L = dlog(l);
    let r = dadd(dexp(dscale(L, a)), dscale(dexp(dscale(L, b)), cc));
    return dadd(r, dscale(dexp(dscale(l, -p.y)), p.x));
}`,
  },
  gao: {
    c: [1.5, 0.4, 5.0, 10.0, 0.5],
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
};
HANDWRITTEN.neutral_hg = { c: [1.0, 5.0, 0.0], wgsl: HANDWRITTEN.neutral.wgsl };
