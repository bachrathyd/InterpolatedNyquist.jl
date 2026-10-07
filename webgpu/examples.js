// The examples of the demo: the three time-independent systems of gpu/scripts/systems.jl (with
// the slider knobs of gpu/interactive/server.jl) and the appendix-gallery case studies of the
// paper (INq-paper: paper/scripts/studies/s08_gallery.jl, A.2-A.9 and A.11;
// s10_fractional_controller.jl, A.12) with their chart axes and validated :unwrap settings.
//
// Every example except the FEM bar is plain text in the equation language of expr.js and runs
// through the same path as a typed equation (parser -> generated WGSL); the texts are written
// operation for operation like the Julia reference (gpu/scripts/systems.jl,
// webgpu/validate/web_systems.jl), so the Float32 rounding matches closely.
//
// Fields: key, group, title, formula (shown under the selector), note, text (the equation; its
// range lines  name = min:max @ value  give each parameter's paper range and default value; an
// axis parameter's range is its chart range), axes: [x, y] (parameter names), set (march settings as expressions of
// the parameters: n (order; empty = estimated), wmax, tol, hmax, wband (h <= hmax for
// ω < wband; empty = the whole line), w0, branch ('auto' / 'on' / 'off': D has a branch
// point at λ = 0, its |D| minimum at ω0 is no root estimate)); an axis parameter in a
// setting stands for max(|min|, |max|) of its range. smin (σ colour floor), res (default
// resolution), hlines ({ param, at }: dashed lines where `param` is the y axis), cmap (the
// parameter of each constant c[i] of validate/ref_counts.json; null: built into the text),
// builtin (not expressible as text: hand-written charD, fixed parameters / axes, n_eff from
// the host).

// ---------------------------------------------------------------------------------------------
// host-side complex arithmetic (Float64) for the effective order of the FEM bar
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

/**
 * Unwrapped phase increment of f(iω) over [w0, w1] (Float64): coarse steps h(ω), each split
 * recursively until every principal increment is below 0.1 rad. f must have no zeros on the
 * line; the coarse step must be short enough that no single step hides a full 2π turn.
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
  { id: 'own', label: 'Your equation' },
  { id: 'ret', label: 'Retarded (quasi-polynomials)' },
  { id: 'neu', label: 'Neutral' },
  { id: 'bar', label: 'Elastic bar with delayed boundary feedback' },
  { id: 'frac', label: 'Fractional order' },
];

const NEUTRAL_NOTE = 'Neutral: the phase ripple never decays, so a small ω_max (tail error < arcsin|a|/π < 1/2) and the step cap h ≤ π/(2τ) along the whole line; the default ω_max = 1e5 fails here.';
const BAR_SET = { wmax: '400', hmax: 'pi/(2*max(r, 1))', wband: '40' };

export const EXAMPLES = [
  {
    key: 'fourth',
    group: 'ret',
    title: 'A.1 Fourth-order delayed oscillator',
    formula: 'D(λ) = c₁λ⁴ + λ² + 2ζλ + 1 + (P + Dλ)·e^{−τλ}',
    text: 'c₁ = 0.005:0.1 @ 0.03\n' +
      'ζ = 0:0.2 @ 0.02\n' +
      'P = -2:4                # x axis\n' +
      'D = -2:5                # y axis\n' +
      'τ = 0.1:1.5 @ 0.5\n' +
      'c₁*λ^4 + λ^2 + 2*ζ*λ + 1 + (P + D*λ)*exp(-τ*λ)',
    axes: ['P', 'D'],
    set: { wmax: '1e5' },
    cmap: ['c₁', 'ζ', 'τ'],
    smin: -0.6,
    res: '960x540',
  },
  {
    key: 'shimmy',
    group: 'ret',
    title: 'Shimmy, stretched-string tyre (Takács, Orosz & Stépán 2009)',
    formula: 'D(λ) = ΣV²λ³ + 2V(V + Σζ)λ² + (Σ + 4ζV)λ + 2 − (L − 1 − Σ)/(L² + 1/3 + Σ(L² + 1 + Σ)) {2/λ²[(L − 1)λ + 2 − ((L + 1)λ + 2)e^{−λ}] + 4ζVL(1 + Σ)(2 + Σλ)/(L − 1 − Σ) + (L − 1 − Σ)(2ΣζVλ + Σ + 4ζV) + (L + 1 + Σ)(2ΣζVλ + Σ − 4ζV)e^{−λ}}',
    note: 'Towed wheel with a stretched-string tyre, Eur. J. Mech. A/Solids 28 (2009), Eq. (31), dimensionless: towing speed V, caster length L, relaxation length Σ = σ/a, tyre damping ζ; the delay 1 is the time a contact point needs to cross the contact patch. The term 2/λ²[...] is the memory of the contact patch, ∫₀¹ (2(L − 1) + 4θ) e^{−λθ} dθ (integral form: no removable singularity at λ = 0); the factor L − 1 − Σ of the ζ term cancels, so nothing divides by zero at L = 1 + Σ. Retarded, n = 3. The 3D button adds ζ as the third axis (the Hopf surface of the paper’s MDBM figure).',
    text: '# Takács, Orosz & Stépán (2009), Eq. (31): shimmy of a towed wheel, stretched-string tyre\n' +
      'V = 0.001:0.6           # x axis: towing speed\n' +
      'L = -0.2:7              # y axis: caster length\n' +
      'ζ = 0:0.1 @ 0.02        # tyre damping (z axis in 3D)\n' +
      'Σ = 0.5:3 @ 1.8         # relaxation length σ/a\n' +
      '# contact-patch memory 2/λ²[(L-1)λ + 2 - ((L+1)λ + 2)e^{-λ}] as an integral over the patch\n' +
      'A = integral((2*(L - 1) + 4*θ)*exp(-λ*θ), θ, 0, 1)\n' +
      'P = Σ*V^2*λ^3 + 2*V*(V + Σ*ζ)*λ^2 + (Σ + 4*ζ*V)*λ + 2\n' +
      '# braces of Eq. (31) times (L - 1 - Σ): the 1/(L - 1 - Σ) of the ζ term cancels\n' +
      'Q = (L - 1 - Σ)*(A + (L - 1 - Σ)*(2*Σ*ζ*V*λ + Σ + 4*ζ*V) + (L + 1 + Σ)*(2*Σ*ζ*V*λ + Σ - 4*ζ*V)*exp(-λ)) + 4*ζ*V*L*(1 + Σ)*(2 + Σ*λ)\n' +
      'P - Q/(L^2 + 1/3 + Σ*(L^2 + 1 + Σ))',
    axes: ['V', 'L'],
    axes3d: ['V', 'L', 'ζ'],
    look3d: { alpha: 0.06, surf: 0.5 },     // 3D button: mostly the stability boundary
    set: { wmax: '1e4' },
    cmap: ['Σ', 'ζ'],
    smin: -0.3,
    res: '960x540',
  },
  {
    key: 'ctcr',
    group: 'ret',
    title: 'Two delays, CTCR example (Sipahi & Olgac 2004)',
    formula: 'CE(s) = s² + 7.1s + 21.1425 + (6s + 14.80)e^{−τ₁s} + (2s + 7.3)e^{−τ₂s} + 8e^{−(τ₁+τ₂)s}',
    note: 'Sipahi & Olgac, "A novel stability study on multiple time-delay systems (MTDS) using the root clustering paradigm", ACC 2004, Eq. (18) (cluster treatment of characteristic roots). The last term is the cross-talk of the two delays (gain c₁₂ = 8). Stable at τ₁ = τ₂ = 0 (roots −9.95, −5.15). Step cap h ≤ π/(2(τ₁ + τ₂)_max) for ω < 50. The 3D button adds the cross-talk gain as the third axis.',
    text: '# Sipahi & Olgac (2004), Eq. (18): two delays with cross-talk\n' +
      'τ₁ = 0:3                # x axis\n' +
      'τ₂ = 0:3                # y axis\n' +
      'c₁₂ = 0:16 @ 8          # cross-talk gain (z axis in 3D)\n' +
      'a₀ = 10:30 @ 21.1425\n' +
      'b₁ = 0:30 @ 14.8\n' +
      'b₂ = 0:15 @ 7.3\n' +
      'λ^2 + 7.1*λ + a₀ + (6*λ + b₁)*exp(-τ₁*λ) + (2*λ + b₂)*exp(-τ₂*λ) + c₁₂*exp(-(τ₁ + τ₂)*λ)',
    axes: ['τ₁', 'τ₂'],
    axes3d: ['τ₁', 'τ₂', 'c₁₂'],
    // the delays add up to 6: the phase turns fast at low ω; without the cap the default step
    // control misses a root pair at ~0.1 % of the points (checked against the ODE back-end)
    set: { wmax: '1e4', hmax: 'pi/(2*(τ₁ + τ₂))', wband: '50' },
    cmap: ['c₁₂', 'a₀', 'b₁', 'b₂'],
    smin: -1.0,
    res: '480x270',
  },
  {
    key: 'algebraic',
    group: 'ret',
    title: 'A.2 Delayed oscillator',
    formula: 'D(λ) = λ² + aλ + k + (b + gλ)·e^{−τλ}     (paper: k = g = 0, τ = 1/2)',
    text: 'a = -1:10         # x axis\n' +
      'k = -1:2 @ 0\n' +
      'b = -1:10         # y axis\n' +
      'g = -0.5:1 @ 0\n' +
      'τ = 0.1:2 @ 0.5\n' +
      'λ^2 + a*λ + k + (b + g*λ)*exp(-τ*λ)',
    axes: ['a', 'b'],
    set: { wmax: '1e4' },
    cmap: ['τ', 'k', 'g'],
    smin: -1.2,
    res: '960x540',
  },
  {
    key: 'distributed',
    group: 'ret',
    title: 'A.3 Distributed delay',
    formula: 'D(λ) = λ² + aλ + k + b·e^{−τ₀λ}(1 − e^{−τλ})/λ     (paper: k = τ₀ = 0, τ = 1)',
    note: 'Uniform kernel ∫ x(t − θ) dθ over [τ₀, τ₀ + τ], typed as integral(...): the page integrates polynomial × exponential kernels in closed form, τ e^{−τ₀λ} exprel(−τλ) with exprel(x) = (eˣ − 1)/x, whose removable singularity at λ = 0 is evaluated by its Taylor series.',
    text: '# distributed delay: the uniform kernel over [τ₀, τ₀ + τ]\n' +
      'a = -0.5:2      # x axis\n' +
      'k = -1:2 @ 0\n' +
      'b = -1:5        # y axis\n' +
      'τ = 0.2:3 @ 1\n' +
      'τ₀ = 0:1 @ 0\n' +
      'λ^2 + a*λ + k + b*integral(exp(-λ*θ), θ, τ₀, τ₀ + τ)',
    axes: ['a', 'b'],
    set: { wmax: '1e4' },
    cmap: ['τ', 'k', 'τ₀'],
    smin: -0.6,
    res: '960x540',
  },
  {
    key: 'showcase',
    group: 'ret',
    title: 'Showcase: constrained 2-DOF DAE (delayed PD)',
    formula: 'D(λ) = a₁₁a₂₂ − a₁₂²,  a₁₁ = m₁λ² + (c₁+c₂)λ + k₁+k₂ + (P + Dλ)e^{−τλ}',
    text: 'm₁ = 0.2:2 @ 1\n' +
      'c₁ = 0:0.5 @ 0.05\n' +
      'c₂ = 0:0.5 @ 0.05\n' +
      'k₁ = -2:1 @ -1\n' +
      'k₂ = 0:2 @ 1\n' +
      'P = 0.5:3             # x axis\n' +
      'D = -0.5:3.5          # y axis\n' +
      'τ = 0.1:1.5 @ 0.5\n' +
      'm₂₃ = 0.1:1.5 @ 0.5\n' +
      'a₁₁ = m₁*λ^2 + (c₁ + c₂)*λ + (k₁ + k₂) + (P + D*λ)*exp(-τ*λ)\n' +
      'a₁₂ = -(c₂*λ + k₂)\n' +
      'a₂₂ = m₂₃*λ^2 + c₂*λ + k₂\n' +
      'a₁₁*a₂₂ - a₁₂^2',
    axes: ['P', 'D'],
    set: { wmax: '1e5' },
    cmap: ['m₁', 'm₂₃', 'k₁', 'k₂', 'c₁', 'c₂', 'τ'],
    smin: -0.4,
    res: '960x540',
  },
  {
    key: 'turning',
    group: 'ret',
    title: 'A.7 Multi-mode turning lobes',
    formula: 'D(λ) = M₁M₂ + w(1 − e^{−2πλ/Ω})(M₂ + A₂M₁),  M₁ = λ² + 2ζ₁λ + 1,  M₂ = λ² + 2ζ₂ω₂λ + ω₂²',
    note: 'The rational form 1 + w(1 − e^{−τλ})(1/M₁ + A₂/M₂) with the stable modal denominators cleared (entire, n = 4); step cap h ≤ 0.05 for ω < 5 (regenerative root chain).',
    text: 'ζ₁ = 0.005:0.1 @ 0.02\n' +
      'ζ₂ = 0.005:0.1 @ 0.03\n' +
      'ω₂ = 1.2:4 @ 2.4\n' +
      'w = 0.01:1.1            # y axis\n' +
      'Ω = 0.1:1.2             # x axis\n' +
      'A₂ = 0:1.5 @ 0.45\n' +
      'M₁ = λ^2 + 2*ζ₁*λ + 1\n' +
      'M₂ = λ^2 + 2*ζ₂*ω₂*λ + ω₂^2\n' +
      'M₁*M₂ + w*(1 - exp(-2*pi/Ω*λ))*(M₂ + A₂*M₁)',
    axes: ['Ω', 'w'],
    set: { wmax: '1e5', hmax: '0.05', wband: '5' },
    cmap: ['ζ₁', 'A₂', 'ζ₂', 'ω₂', null],
    smin: -0.05,
    res: '960x540',
  },
  {
    key: 'neutral',
    group: 'neu',
    title: 'A.4 Neutral DDE',
    formula: 'D(λ) = λ² + aλ²e^{−τλ} + dλ + k + c·e^{−τλ}     (paper: τ = 1, d = 0, k = 1)',
    note: NEUTRAL_NOTE + ' |a| > 1: essential instability.',
    text: 'a = -1.2:1.2    # x axis\n' +
      'τ = 0.3:2 @ 1\n' +
      'd = 0:2 @ 0\n' +
      'k = 0:3 @ 1\n' +
      'c = -1.2:1.2    # y axis\n' +
      'λ^2 + a*λ^2*exp(-τ*λ) + d*λ + k + c*exp(-τ*λ)',
    axes: ['a', 'c'],
    set: { wmax: '200', hmax: 'pi/(2*τ)' },
    cmap: ['τ', 'd', 'k'],
    smin: -0.5,
    res: '480x270',
  },
  {
    key: 'neutral_hg',
    group: 'neu',
    title: 'A.5 High-gain neutral DDE',
    formula: 'D(λ) = λ² + aλ²e^{−τλ} + dλ + k + c·e^{−τλ}     (paper: τ = 1, d = 5, k = 0)',
    note: 'The damped high-gain variant of A.4. ' + NEUTRAL_NOTE,
    text: 'a = -1.2:1.2    # x axis\n' +
      'τ = 0.3:2 @ 1\n' +
      'd = 1:10 @ 5\n' +
      'k = 0:3 @ 0\n' +
      'c = -1:10       # y axis\n' +
      'λ^2 + a*λ^2*exp(-τ*λ) + d*λ + k + c*exp(-τ*λ)',
    axes: ['a', 'c'],
    set: { wmax: '200', hmax: 'pi/(2*τ)' },
    cmap: ['τ', 'd', 'k'],
    smin: -0.6,
    res: '480x270',
  },
  {
    key: 'pda',
    group: 'neu',
    title: 'A.6 PDA control (neutral, essential instability)',
    formula: 'D(λ) = λ² + 2ζλ + 1 + (P + Dλ + Aλ²)·e^{−τλ}     (paper: ζ = 0.05, D = 0.1, τ = 1)',
    note: 'Delayed acceleration feedback makes the loop neutral: for |A| > 1 the essential spectrum sits at Re λ = ln|A|/τ > 0 (dashed lines) and the count saturates (≈ ω_max τ/2π). ω_max = 500, h ≤ π/(2τ).',
    text: 'ζ = 0:0.3 @ 0.05\n' +
      'P = -1.1:1.4         # x axis\n' +
      'D = -0.5:0.5 @ 0.1\n' +
      'A = -1.15:1.15       # y axis\n' +
      'τ = 0.3:2 @ 1\n' +
      'λ^2 + 2*ζ*λ + 1 + (P + D*λ + A*λ^2)*exp(-τ*λ)',
    axes: ['P', 'A'],
    set: { wmax: '500', hmax: 'pi/(2*τ)' },
    cmap: ['ζ', 'D', 'τ'],
    smin: -0.5,
    res: '480x270',
    hlines: { param: 'A', at: [-1, 1] },
  },
  {
    key: 'rod',
    group: 'bar',
    title: 'A.8 Exact transcendental rod (Zhang & Stépán)',
    formula: 'D(λ) = 1 − K·e^{−rλ}/cosh γ,   γ = λ√(1 + c/λ)/√(1 + ηλ)     (r = τ/T; paper: c = 0)',
    note: 'Bar with delayed boundary feedback, exact (no discretization), Kelvin–Voigt damping η, external damping c. Counted on the entire numerator cosh γ − K e^{−rλ}; in Float32 cosh γ ~ e^{120} overflows at ω = 400, so it is multiplied by the analytic, zero-free e^{−γ}: the march sees (1 + e^{−2γ})/2 − K e^{−rλ−γ}, whose leading order is n ≈ 0 (the e^{−γ} phase cancels against that of the stable denominator). ω_max = 400, h ≤ π/(2 r_max) for ω < 40. K = 1: a root at λ = 0 on the line.',
    text: '# numerator cosh γ - K e^{-rλ}, times e^{-γ} (no overflow in Float32)\n' +
      'c = 0:1 @ 0\n' +
      'η = 0.002:0.05 @ 0.01\n' +
      'K = -0.75:1             # y axis\n' +
      'r = 0.02:10.5           # x axis\n' +
      'γ = λ*sqrt(1 + c/λ)/sqrt(1 + η*λ)\n' +
      '(1 + exp(-2*γ))/2 - K*exp(-r*λ - γ)',
    axes: ['r', 'K'],
    set: { ...BAR_SET },
    cmap: ['η', 'c'],
    smin: -0.3,
    res: '480x270',
  },
  {
    key: 'fem',
    group: 'bar',
    title: 'A.9 Same bar, finite elements (N DOF; paper: 12)',
    formula: 'D(λ) = det(λ²M + λC + K + F(λ)),  F = c(λ)e₁e_Nᵀ,  c(λ) = −K(1 + ηλ)e^{−rλ}/h,  C = ηK',
    note: 'Built-in (a loop over the N elements is not a closed expression): N linear bar elements (h = 1/N). Q₀ = λ²M + λC + K is tridiagonal: det Q₀ by its continuant and (Q₀⁻¹)_{N1} = (−1)^{N+1}o^{N−1}/det Q₀, so D = det Q₀ + c(λ)(−1)^{N+1}o^{N−1} costs O(N); divided by ((4h/6)(λ + √3/h)²)^N against overflow (poles far left). Effective order n_eff = 2Φ_den(ω_max)/π from the phase of det Q₀ (host).',
    text: '# built-in example (hand-written WGSL, not editable): the N-element FEM bar\n' +
      '#   D(λ) = det(λ²M + λC + K + F(λ)),  F = c(λ) e₁ e_Nᵀ,  c(λ) = -K (1 + ηλ) e^{-rλ} / h\n' +
      '# evaluated by the continuant recurrence of the tridiagonal λ²M + λC + K (O(N) per point).\n' +
      '# Parameters: η (Kelvin-Voigt damping), N (elements); axes r = τ/T and K.\n' +
      '# Copy the rod example (A.8) for an editable exact-bar equation.\n' +
      'η = 0.002:0.05 @ 0.01\n' +
      'N = 2:1:24 @ 12          # step 1: integer slider\n' +
      'r = 0.02:10.5           # x axis\n' +
      'K = -0.75:1             # y axis',
    axes: ['r', 'K'],
    set: { ...BAR_SET },
    cmap: ['η', 'N'],
    smin: -0.3,
    res: '320x180',
    builtin: {
      // K[0] = η, K[1] = N (parameter order above), p = (r, K)
      prep: (c, wmax) => neff('fem', femDen, c, wmax, (w) => (w < 60 ? 0.005 : 0.05)),
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
  },
  {
    key: 'frac',
    group: 'frac',
    title: 'A.11 Fractional-order oscillator',
    formula: 'D(λ) = λ^α + c·λ^β + k·e^{−τλ}     (paper: α = 1.8, β = 0.8, c = 0.5)',
    note: 'Principal branch, λ^μ = exp(μ log λ); n = α (non-integer, estimated). Only σ = 0 is admissible (a shifted line would cross the branch cut); the march starts at ω₀ = 1e-9, next to the branch point (branch point at λ = 0 detected: its |D| minimum is no root estimate).',
    text: 'α = 1.1:1.99 @ 1.8\n' +
      'c = 0:2 @ 0.5\n' +
      'β = 0:1 @ 0.8\n' +
      'k = 0:5              # x axis\n' +
      'τ = 0.1:2            # y axis\n' +
      'λ^α + c*λ^β + k*exp(-τ*λ)',
    axes: ['k', 'τ'],
    set: { wmax: '1e4' },
    cmap: ['α', 'β', 'c'],
    smin: -0.2,
    res: '960x540',
  },
  {
    key: 'gao',
    group: 'frac',
    title: 'A.12 Fractional PI controller (Gao, Zhai & Liu)',
    formula: 'D(s) = s^μ(T s^ν + 1) + K e^{−Ls}(k_p s^μ + k_i),   plant K e^{−Ls}/(T s^ν + 1),  C(s) = k_p + k_i/s^μ',
    note: 'Gao, Zhai & Liu (2017), Example 1, μ = 1.5 (K = 5, L = 0.4, T = 10, ν = 0.5); n = μ + ν (estimated); σ = 0 only (branch cut).',
    text: 'μ = 0.3:1.9 @ 1.5\n' +
      'T = 1:20 @ 10\n' +
      'ν = 0.1:1 @ 0.5\n' +
      'K = 1:10 @ 5\n' +
      'L = 0.05:1 @ 0.4\n' +
      'k_p = -1:6          # x axis\n' +
      'k_i = -1:25         # y axis\n' +
      'λ^μ*(T*λ^ν + 1) + K*exp(-L*λ)*(k_p*λ^μ + k_i)',
    axes: ['k_p', 'k_i'],
    set: { wmax: '1e4' },
    cmap: ['μ', 'L', 'K', 'T', 'ν'],
    smin: -0.4,
    res: '480x270',
  },
];

// 'Own example': starts as the fourth-order equation (A.1), editable, kept in localStorage
{
  const f = EXAMPLES.find((e) => e.key === 'fourth');
  EXAMPLES.unshift({
    ...f,
    key: 'own',
    group: 'own',
    title: 'Own example (type your equation)',
    formula: '',
    note: 'Type any characteristic function D(λ) below; every unknown name becomes a parameter. Saved in this browser.',
    text: '# your characteristic function D(λ); starts as A.1 (fourth-order delayed oscillator)\n' + f.text,
    cmap: null,
  });
}

/** default march settings (strings: expressions of the parameters) */
export const SET_DEFAULTS = { n: '', wmax: '1e5', tol: '0.3', hmax: '', wband: '', w0: '1e-9', branch: 'auto' };
