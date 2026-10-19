// Host checks of the stochastic state form (sde.js):  node webgpu/validate/sde_check.mjs
// 1. compile both examples, print the generated WGSL of the turning model;
// 2. D(λ) from the state form vs the closed-form characteristic function;
// 3. a Float64 port of the GPU computation (same formulas: dual-free, fine trapezoid) of r0 and
//    Var against the Julia reference values (notes/second_moment_turning.jl).
import { compileSDE, isStateForm } from '../sde.js';
import { EXAMPLES } from '../examples.js';

const ex = (k) => EXAMPLES.find((e) => e.key === k);
for (const k of ['sturning', 'spd']) {
  const t = ex(k).text;
  const m = compileSDE(t);
  console.log(`${k}: state form ${isStateForm(t)}, params [${m.params.join(', ')}], N = ${m.stochastic.N} (${m.stochastic.states.join(', ')}), ` +
    `noises ${m.stochastic.noises.join(', ')}, multiplicative ${m.stochastic.multiplicative}, warnings ${JSON.stringify(m.warnings)}`);
}
const m = compileSDE(ex('sturning').text);
console.log('\n--- generated WGSL (turning, axes Ω, K) ---\n' + m.code(0, 1));

// parameter vector in m.params order
const pvec = (mm, o) => mm.params.map((n) => o[n]);
const cx = (re, im = 0) => ({ re, im });
const cmul = (a, b) => cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const cadd = (a, b) => cx(a.re + b.re, a.im + b.im);
const cexp = (a) => cx(Math.exp(a.re) * Math.cos(a.im), Math.exp(a.re) * Math.sin(a.im));

console.log('\n--- D(λ): state form vs closed form  λ² + 2ζλ + 1 + K(1 - e^{-λτ}) ---');
for (const [K, Om, z] of [[0.2, 0.6, 0.1], [0.45, 1.3, 0.05]]) {
  const pv = pvec(m, { Ω: Om, K, σ: 0.7, ζ: z });
  for (const lam of [cx(0.1, 0.7), cx(-0.3, 4.2), cx(0, 30)]) {
    const a = m.evalD(lam, pv);
    const e = cexp(cx(-lam.re * 2 * Math.PI / Om, -lam.im * 2 * Math.PI / Om));
    let b = cadd(cmul(lam, lam), cx(2 * z * lam.re + 1 + K * (1 - e.re), 2 * z * lam.im - K * e.im));
    console.log(`  K=${K} Ω=${Om} λ=${lam.re}+${lam.im}i: |D_state - D_closed| / |D| = ${(Math.hypot(a.re - b.re, a.im - b.im) / Math.hypot(b.re, b.im)).toExponential(1)}`);
  }
  console.log(`  mean / Γ: ${JSON.stringify(m.stochastic.statics(pv))}  (expect mean [K, 0], Γ = (Kσ)² = ${(K * 0.7) ** 2})`);
}

console.log('\n--- r0, Var by a Float64 trapezoid on the same transfer functions (vs Julia QuadGK) ---');
// u = M^{-1} e_r for N = 2: M = [[λ, -1], [1 + K - K e, λ + 2ζ]] (row r = 1 is the noise row)
function rv(K, Om, s, z) {
  const tau = 2 * Math.PI / Om;
  let iy = 0, ig = 0;
  const dw = 1e-4, W = 400;
  for (let w = dw / 2; w < W; w += dw) {
    const e = cexp(cx(0, -w * tau));
    const D = cadd(cmul(cx(0, w), cx(2 * z, w)), cx(1 + K - K * e.re, -K * e.im));   // λ(λ + 2ζ) + 1 + K - K e
    const d2 = D.re * D.re + D.im * D.im;
    const Y2 = 1 / d2;                                                     // |u_0|² = |1/D|²
    const g = cx(K * s * (e.re - 1), K * s * e.im);                          // (α + β e) u_0 = Kσ(e - 1)/D
    iy += Y2 * dw;
    ig += (g.re * g.re + g.im * g.im) / d2 * dw;
  }
  return { r0: ig / Math.PI, Var: (K * s) ** 2 * (iy / Math.PI) / (1 - ig / Math.PI) };
}
for (const [K, Om, s, refV, refR] of [[0.2, 0.6, 0.7, 0.371578, 0.367976], [0.2, 1.0, 1.0, 0.103849, 0.077075]]) {
  const r = rv(K, Om, s, 0.1);
  console.log(`  K=${K} Ω=${Om} σ=${s}: r0 ${r.r0.toFixed(6)} (Julia ${refR}), Var ${r.Var.toFixed(6)} (Julia ${refV})`);
}
