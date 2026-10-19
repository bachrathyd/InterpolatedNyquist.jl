// The stochastic state form: a linear delay system with additive and multiplicative white noise,
// typed as state equations (cf. the Julia version notes/stochastic_sdde.jl and the theory in
// notes/stochastic_second_moment.md).
//
//   ζ = 0.1                                   helpers / parameter ranges as in expr.js
//   τ = 2*pi/Ω
//   x'' = -2*ζ*x' - x + K*(x(t-τ) - x) + K*σ*(1 + x(t-τ) - x)*ξ
//
// * a line  name' = rhs  (or  name'' = ..., up to 4 primes) defines states: x'' = f adds the
//   states x and x' (x' is the velocity, usable on right-hand sides); several such lines build
//   an N-dimensional system (N <= 6);
// * x(t-τ) is a delayed state (τ any real expression of the parameters; several delays are
//   allowed, also delayed velocities x'(t-τ));
// * ξ (or ξ₁ ... ξ₄, xi, xi1 ...) are independent Gaussian white noises (Itô): every term
//   must be linear in the states, and a noise may multiply a state (multiplicative noise) or a
//   constant (additive noise), never another noise;
// * the noise may enter one equation only (a force channel: p = 1); the output is the first
//   state.
//
// Compiled exactly (no differentiation needed: the equations are affine): every right-hand side
// is lowered to an affine form over {1, x_j, x_j(t-τ_d), ξ_k, ξ_k x_j, ξ_k x_j(t-τ_d)} with
// real coefficient expressions of the parameters. From these:
//   M(λ) = λI - A - Σ_d B_d e^{-λτ_d}           (characteristic matrix, D = det M)
//   u(λ) = M(λ)^{-1} e_r                         (r: the noise equation)
//   Y = u_0 (output transfer),  G_k = Σ_j (α_kj + Σ_d β_kjd e^{-λτ_d}) u_j
//   mean m = -(A + Σ B_d)^{-1} c,  γ̃_k = γ_k + Σ_j (α_kj + Σ_d β_kjd) m_j,  Γ = Σ_k γ̃_k²
// and on the GPU (smarch.wgsl):  r0 = (1/2π)∫ Σ|G_k|² dω,  Var(x_0) = Γ (1/2π)∫|Y|² dω / (1 - r0).

import {
  tokenize, splitStatements, parseStatement, IR, emitNode, f32lit, buildDecls, CONSTS, LAMBDA, ARITY,
  NON_ANALYTIC, cx, CF, ExprError, MAX_PARAMS,
} from './expr.js';

export const MAX_STATES = 6;
export const MAX_NOISES = 4;
const NOISE_RE = /^(?:ξ|xi)([₀-₉0-9]*)$/u;
const SUB = { '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9' };

/** a text in the stochastic state form: some line defines  name' = ... */
export function isStateForm(src) {
  try {
    return splitStatements(tokenize(src)).some((t) => t.length >= 2 && t[0].t === 'id' && t[0].v.endsWith("'") &&
      t[1].t === 'op' && t[1].v === '=');
  } catch (e) {
    return false;
  }
}

const baseOf = (nm) => nm.replace(/'+$/, '');
const primes = (nm) => nm.length - baseOf(nm).length;

export function compileSDE(src) {
  if (typeof src !== 'string') throw new ExprError('no equations');
  if (src.length > 20000) throw new ExprError('the text is too long (20000 characters at most)');
  const all = splitStatements(tokenize(src)).map(parseStatement);
  const declStmts = all.filter((s) => s.kind === 'decl');
  const stmts = all.filter((s) => s.kind !== 'decl');
  const odes = stmts.filter((s) => s.name && s.name.endsWith("'"));
  const helperStmts = stmts.filter((s) => !(s.name && s.name.endsWith("'")));
  for (const s of helperStmts) {
    if (s.name === null) throw new ExprError("a plain expression: in the state form every line is  name = expression  or a state equation  x' = ...", s.line, s.col);
    if (s.name === 'D') throw new ExprError("D = ... is the characteristic-function form; the state form defines x' = ... (do not mix them)", s.line, s.col);
  }
  // ---- states: x'' = f defines x, x' (the equation of x' is f, the one of x is x') -------------
  const states = [];
  const eqOf = new Map();       // state name -> { ast } | { chain: next state }
  for (const s of odes) {
    const b = baseOf(s.name), k = primes(s.name);
    if (k > 4) throw new ExprError('at most 4 primes (x\'\'\'\')', s.line, s.col);
    if (NOISE_RE.test(b) || LAMBDA.has(b) || CONSTS[b] || ARITY[b] || b === 't') throw new ExprError(`'${b}' cannot be a state`, s.line, s.col);
    for (let m = 0; m < k; m++) {
      const nm = b + "'".repeat(m);
      if (eqOf.has(nm)) throw new ExprError(`the state '${nm}' has two equations`, s.line, s.col);
      states.push(nm);
      eqOf.set(nm, m < k - 1 ? { chain: b + "'".repeat(m + 1) } : { ast: s.ast, line: s.line, col: s.col });
    }
  }
  const N = states.length;
  if (!N) throw new ExprError("no state equation  x' = ...");
  if (N > MAX_STATES) throw new ExprError(`${N} states: at most ${MAX_STATES} are supported in the browser (use notes/stochastic_sdde.jl)`);
  const sidx = new Map(states.map((n, i) => [n, i]));

  // ---- helpers, parameters ----------------------------------------------------------------------
  const ir = new IR();
  const defAt = new Map();
  const reserved = (nm, s) => {
    if (LAMBDA.has(nm)) throw new ExprError(`'${nm}' is the Laplace variable and cannot be defined`, s.line, s.col);
    if (CONSTS[nm]) throw new ExprError(`'${nm}' is a constant and cannot be redefined`, s.line, s.col);
    if (ARITY[nm] || NON_ANALYTIC.has(nm)) throw new ExprError(`'${nm}' is a function name`, s.line, s.col);
    if (nm === 't' || NOISE_RE.test(nm) || sidx.has(nm)) throw new ExprError(`'${nm}' is reserved (time, noise or a state)`, s.line, s.col);
  };
  helperStmts.forEach((s, k) => {
    reserved(s.name, s);
    if (defAt.has(s.name)) throw new ExprError(`'${s.name}' is defined twice`, s.line, s.col);
    defAt.set(s.name, k);
  });
  const decls = buildDecls(declStmts, defAt, helperStmts, reserved);
  const pnames = [...decls.keys()];
  const helperForm = new Map();
  let curHelper = Infinity;

  // ---- affine forms: Map key -> IR node (real coefficient) --------------------------------------
  // keys: '1' | 'x|j' | 'd|j|di' | 'n|k' | 'nx|k|j' | 'nd|k|j|di'
  const delays = [];            // IR nodes of the distinct delays
  const noises = [];            // noise names in order of appearance
  const F1 = (node) => new Map([['1', node]]);
  const isConst = (F) => [...F.keys()].every((k) => k === '1');
  const hasNoise = (F) => [...F.keys()].some((k) => k[0] === 'n');
  const stateFree = (F) => [...F.keys()].every((k) => k === '1' || k.startsWith('n|'));
  const stateAffine = (F) => !hasNoise(F);
  const cval = (F) => F.get('1') ?? ir.lit(0);
  const fadd = (A, B, sgn = 1) => {
    const out = new Map(A);
    for (const [k, v] of B) {
      const w = sgn < 0 ? ir.neg(v) : v;
      out.set(k, out.has(k) ? ir.add(out.get(k), w) : w);
    }
    return out;
  };
  const fscale = (A, c) => new Map([...A].map(([k, v]) => [k, ir.mul(v, c)]));
  const where = (a) => [a.line, a.col];
  function fmul(A, B, ast) {
    if (isConst(A)) return fscale(B, cval(A));
    if (isConst(B)) return fscale(A, cval(B));
    // (constant + noises) × (constant + states): the only admissible product of two non-constant forms
    let nz = null, st = null;
    if (stateFree(A) && stateAffine(B)) { nz = A; st = B; } else if (stateFree(B) && stateAffine(A)) { nz = B; st = A; }
    if (!nz) {
      throw new ExprError(hasNoise(A) && hasNoise(B) ? 'a product of two noises is not allowed' :
        'a product of states is nonlinear: the state form must be linear in the states (linearize, or use notes/stochastic_sdde.jl)', ...where(ast));
    }
    let out = new Map();
    for (const [kn, vn] of nz) {
      for (const [ks, vs] of st) {
        const c = ir.mul(vn, vs);
        let key;
        if (kn === '1') key = ks;
        else {
          const k = kn.slice(2);
          key = ks === '1' ? `n|${k}` : ks[0] === 'x' ? `nx|${k}|${ks.slice(2)}` : `nd|${k}|${ks.slice(2)}`;
        }
        out = fadd(out, new Map([[key, c]]));
      }
    }
    return out;
  }
  function lowerF(ast) {
    switch (ast.t) {
      case 'num': return F1(ir.lit(ast.v));
      case 'id': {
        const n = ast.name;
        if (LAMBDA.has(n)) throw new ExprError("λ does not appear in the state form (delays are written x(t-τ))", ...where(ast));
        if (n === 't') throw new ExprError("t only appears inside a delayed state x(t-τ)", ...where(ast));
        if (CONSTS[n]) {
          if (n === 'i') throw new ExprError('i: the coefficients must be real', ...where(ast));
          return F1(ir.lit(CONSTS[n][0]));
        }
        if (ARITY[n]) throw new ExprError(`'${n}' is a function: write ${n}(...)`, ...where(ast));
        if (sidx.has(n)) return new Map([[`x|${sidx.get(n)}`, ir.lit(1)]]);
        const nm = NOISE_RE.exec(n);
        if (nm) {
          if (!noises.includes(n)) noises.push(n);
          if (noises.length > MAX_NOISES) throw new ExprError(`at most ${MAX_NOISES} independent noises`, ...where(ast));
          return new Map([[`n|${noises.indexOf(n)}`, ir.lit(1)]]);
        }
        if (n.endsWith("'")) throw new ExprError(`'${n}' is not a state (define ${baseOf(n)}${"'".repeat(primes(n) + 1)} = ... to make it one)`, ...where(ast));
        if (defAt.has(n)) {
          const k = defAt.get(n);
          if (k >= curHelper) throw new ExprError(`'${n}' is used before its definition`, ...where(ast));
          return helperForm.get(n);
        }
        if (!pnames.includes(n)) pnames.push(n);
        return F1(ir.par(n));
      }
      case 'neg': return fscale(lowerF(ast.a), ir.lit(-1));
      case 'bin': {
        const A = lowerF(ast.a);
        const B = lowerF(ast.b);
        switch (ast.op) {
          case '+': return fadd(A, B);
          case '-': return fadd(A, B, -1);
          case '*': return fmul(A, B, ast);
          case '/':
            if (!isConst(B)) throw new ExprError('division by a state or a noise is not linear', ...where(ast));
            return fscale(A, ir.div(ir.lit(1), cval(B)));
          default:
            if (isConst(A) && isConst(B)) return F1(ir.pow(cval(A), cval(B)));
            if (isConst(B) && ir.isLit(cval(B), 1)) return A;
            throw new ExprError('a power of a state or a noise is not linear', ...where(ast));
        }
      }
      case 'call': {
        const f = ast.f;
        if (sidx.has(f) || (f.endsWith("'") && !sidx.has(f) && sidx.has(baseOf(f)))) {
          if (!sidx.has(f)) throw new ExprError(`'${f}' is not a state`, ...where(ast));
          if (ast.args.length !== 1) throw new ExprError(`a delayed state is written ${f}(t-τ)`, ...where(ast));
          const a = ast.args[0];
          if (a.t === 'id' && a.name === 't') return new Map([[`x|${sidx.get(f)}`, ir.lit(1)]]);
          if (!(a.t === 'bin' && a.op === '-' && a.a.t === 'id' && a.a.name === 't')) {
            throw new ExprError(`a delayed state is written ${f}(t-τ) with τ an expression of the parameters`, ...where(ast));
          }
          const T = lowerF(a.b);
          if (!isConst(T)) throw new ExprError('the delay must not contain states or noises', ...where(ast));
          const tn = cval(T);
          if (ir.nodes[tn].ty !== 'R') throw new ExprError('the delay must be real', ...where(ast));
          let di = delays.indexOf(tn);
          if (di < 0) { delays.push(tn); di = delays.length - 1; }
          if (delays.length > 4) throw new ExprError('at most 4 different delays', ...where(ast));
          return new Map([[`d|${sidx.get(f)}|${di}`, ir.lit(1)]]);
        }
        if (f === 'integral') throw new ExprError('integral(...) is not available in the state form', ...where(ast));
        const ar = ARITY[f];
        if (ar === undefined) {
          if (NON_ANALYTIC.has(f)) throw new ExprError(`${f}(...) is not supported`, ...where(ast));
          throw new ExprError(`unknown function '${f}'`, ...where(ast));
        }
        if (ast.args.length !== ar) throw new ExprError(`${f} takes ${ar} argument${ar > 1 ? 's' : ''}`, ...where(ast));
        const args = ast.args.map(lowerF);
        if (!args.every(isConst)) throw new ExprError(`${f}(...) of a state or a noise is not linear`, ...where(ast));
        const v = args.map(cval);
        if (f === 'pow') return F1(ir.pow(v[0], v[1]));
        return F1(ir.op2(f === 'ln' ? 'log' : f, ...v));
      }
      default: throw new ExprError('internal: unknown node');
    }
  }

  helperStmts.forEach((s, k) => { curHelper = k; helperForm.set(s.name, lowerF(s.ast)); });
  curHelper = Infinity;
  const rows = states.map((nm) => {
    const e = eqOf.get(nm);
    if (e.chain) return new Map([[`x|${sidx.get(e.chain)}`, ir.lit(1)]]);
    return lowerF(e.ast);
  });

  // ---- coefficients ----------------------------------------------------------------------------
  const Z0 = ir.lit(0);
  const get = (F, key) => F.get(key) ?? Z0;
  const nd = delays.length;
  const A = rows.map((F) => states.map((_, j) => get(F, `x|${j}`)));
  const B = delays.map((_, d) => rows.map((F) => states.map((_, j) => get(F, `d|${j}|${d}`))));
  const c = rows.map((F) => get(F, '1'));
  const noiseRows = rows.map((F, i) => (hasNoise(F) ? i : -1)).filter((i) => i >= 0);
  const warnings = [];
  if (noiseRows.length > 1) {
    throw new ExprError(`noise appears in the equations of ${noiseRows.map((i) => states[i]).join(' and ')}: the browser version ` +
      'supports one noise channel (all noise in one equation, e.g. the force); use notes/stochastic_sdde.jl for several');
  }
  if (!noiseRows.length) warnings.push('no noise term (ξ): the variance is zero, only the deterministic count is meaningful');
  const r = noiseRows.length ? noiseRows[0] : N - 1;
  const nk = Math.max(noises.length, 1);
  const Fr = rows[r];
  const gam = Array.from({ length: nk }, (_, k) => get(Fr, `n|${k}`));
  const alp = Array.from({ length: nk }, (_, k) => states.map((_, j) => get(Fr, `nx|${k}|${j}`)));
  const bet = Array.from({ length: nk }, (_, k) => states.map((_, j) => delays.map((_, d) => get(Fr, `nd|${k}|${j}|${d}`))));
  const multiplicative = alp.some((a) => a.some((v) => !ir.isLit(v, 0))) || bet.some((a) => a.some((b) => b.some((v) => !ir.isLit(v, 0))));
  const allNodes = [...A.flat(), ...B.flat(2), ...c, ...gam, ...alp.flat(), ...bet.flat(2), ...delays];
  for (const id of allNodes) {
    if (ir.nodes[id].ty !== 'R') throw new ExprError('a coefficient is complex (log / sqrt of a negative parameter?): the state form needs real coefficients');
  }
  // reachable coefficient nodes in creation (topological) order
  const reach = new Uint8Array(ir.nodes.length);
  const st = [...allNodes];
  while (st.length) { const id = st.pop(); if (reach[id]) continue; reach[id] = 1; for (const a of ir.nodes[id].args) st.push(a); }
  const order = [];
  for (let id = 0; id < ir.nodes.length; id++) if (reach[id]) order.push(id);
  const usesPar = (nm) => order.some((id) => ir.nodes[id].op === 'par' && ir.nodes[id].name === nm);
  const params = pnames.filter(usesPar);
  for (const nm of pnames) if (!params.includes(nm)) warnings.push(`parameter '${nm}' has no effect`);
  if (params.length > MAX_PARAMS) throw new ExprError(`${params.length} parameters: at most ${MAX_PARAMS} are supported`);
  const pidx = new Map(params.map((nm, i) => [nm, i]));
  for (const id of order) { const n = ir.nodes[id]; if (n.op === 'lit') f32lit(n.val.re); }

  // ---- WGSL ---------------------------------------------------------------------------------------
  const codeCache = new Map();
  function code(ix, iy, iz = -1) {
    const ck = ix + ',' + iy + ',' + iz;
    if (codeCache.has(ck)) return codeCache.get(ck);
    const name = (id) => {
      const n = ir.nodes[id];
      if (n.op === 'lit') return f32lit(n.val.re);
      if (n.op === 'par') {
        const j = pidx.get(n.name);
        return j === ix ? 'p.x' : j === iy ? 'p.y' : j === iz ? 'PZ' : `K[${j}]`;
      }
      return `t${id}`;
    };
    const coef = [];
    for (const id of order) {
      const n = ir.nodes[id];
      if (n.op === 'lit' || n.op === 'par') continue;
      const Bn = n.args.map((a) => ir.nodes[a]);
      coef.push(`    let t${id} = ${emitNode(n, n.args.map(name), Bn.map((x) => x.ty), Bn)};`);
    }
    const isZ = (id) => ir.isLit(id, 0);
    // characteristic matrix entries  M_ij = δ_ij λ - A_ij - Σ_d B_dij E_d
    const ml = [];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let e = i === j ? 'l' : 'CD(C(0.0, 0.0), C(0.0, 0.0))';
        if (!isZ(A[i][j])) e = `daddr(${e}, -(${name(A[i][j])}))`;
        for (let d = 0; d < nd; d++) if (!isZ(B[d][i][j])) e = `dsub(${e}, dscale(E${d}, ${name(B[d][i][j])}))`;
        ml.push(`    SMm[${i * 6 + j}] = ${e};`);
      }
    }
    // noise modulation  G_k = Σ_j (α_kj + Σ_d β_kjd E_d) u_j
    const gl = [];
    for (let k = 0; k < nk; k++) {
      let terms = [];
      for (let j = 0; j < N; j++) {
        let w = isZ(alp[k][j]) ? null : `dscale(SMu[${j}], ${name(alp[k][j])})`;
        for (let d = 0; d < nd; d++) {
          if (isZ(bet[k][j][d])) continue;
          const t = `dmul(dscale(E${d}, ${name(bet[k][j][d])}), SMu[${j}])`;
          w = w ? `dadd(${w}, ${t})` : t;
        }
        if (w) terms.push(w);
      }
      const g = terms.length ? terms.reduce((a, b) => `dadd(${a}, ${b})`) : null;
      if (g) gl.push(`    let G${k} = ${g};\n    fg = fg + vec2<f32>(dot(G${k}.v, G${k}.v), 2.0 * dot(G${k}.v, G${k}.d));`);
    }
    // the mean and the effective additive noise (real)
    const rl = [];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let e = isZ(A[i][j]) ? '0.0' : name(A[i][j]);
        for (let d = 0; d < nd; d++) if (!isZ(B[d][i][j])) e = `${e} + ${name(B[d][i][j])}`;
        rl.push(`    SRm[${i * 6 + j}] = ${e};`);
      }
      rl.push(`    SRb[${i}] = ${isZ(c[i]) ? '0.0' : `-(${name(c[i])})`};`);
    }
    const gt = [];
    for (let k = 0; k < nk; k++) {
      let e = isZ(gam[k]) ? '0.0' : name(gam[k]);
      for (let j = 0; j < N; j++) {
        const parts = [];
        if (!isZ(alp[k][j])) parts.push(name(alp[k][j]));
        for (let d = 0; d < nd; d++) if (!isZ(bet[k][j][d])) parts.push(name(bet[k][j][d]));
        if (parts.length) e = `${e} + (${parts.join(' + ')}) * SRb[${j}]`;
      }
      gt.push(`    let g${k} = ${e};\n    gs = gs + g${k} * g${k};`);
    }
    const ed = delays.map((tn, d) => `    let E${d} = dexp(dscale(l, -(${name(tn)})));`);
    const s = `// generated by sde.js: ${N} states (${states.join(', ')}), ${nd} delay(s), ${noises.length} noise(s), noise in the equation of ${states[r]}
fn smodel(l: CD, p: vec2<f32>) -> SMO {
${coef.join('\n')}
${ed.join('\n')}
${ml.join('\n')}
    let Dt = sm_solve(${N}, ${r});
    var fg = vec2<f32>(0.0, 0.0);
${gl.join('\n')}
    let Y = SMu[0];
    return SMO(Dt, vec2<f32>(dot(Y.v, Y.v), 2.0 * dot(Y.v, Y.d)), fg);
}
// Γ = Σ_k γ̃_k² (effective additive intensity at the mean); .y = 0 if the mean is not defined
fn smstatic(p: vec2<f32>) -> vec2<f32> {
${coef.join('\n')}
${rl.join('\n')}
    if (!sm_rsolve(${N})) { return vec2<f32>(0.0, 0.0); }
    var gs = 0.0;
${gt.join('\n')}
    return vec2<f32>(gs, 1.0);
}
fn charD(l: CD, p: vec2<f32>) -> CD { return smodel(l, p).D; }`;
    codeCache.set(ck, s);
    return s;
  }

  // ---- host evaluation (Float64): coefficients, D(λ) = det M(λ), mean ----------------------------
  function coefValues(pv) {
    const r = new Array(ir.nodes.length);
    for (const id of order) {
      const n = ir.nodes[id];
      if (n.op === 'lit') r[id] = n.val.re;
      else if (n.op === 'par') r[id] = pv[pidx.get(n.name)];
      else {
        const a = n.args.map((x) => cx(r[x], 0));
        const v = a.length === 2 ? CF[n.op](a[0], a[1]) : CF[n.op](a[0]);
        r[id] = v.re;
      }
    }
    return r;
  }
  const cmul = (a, b) => cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
  const csub = (a, b) => cx(a.re - b.re, a.im - b.im);
  const cdiv = (a, b) => CF.div(a, b);
  function evalD(lam, pv) {
    const v = coefValues(pv);
    const E = delays.map((tn) => CF.exp(cx(-lam.re * v[tn], -lam.im * v[tn])));
    const M = [];
    for (let i = 0; i < N; i++) {
      const row = [];
      for (let j = 0; j < N; j++) {
        let e = i === j ? cx(lam.re, lam.im) : cx(0, 0);
        e = cx(e.re - v[A[i][j]], e.im);
        for (let d = 0; d < nd; d++) e = csub(e, cx(E[d].re * v[B[d][i][j]], E[d].im * v[B[d][i][j]]));
        row.push(e);
      }
      M.push(row);
    }
    let det = cx(1, 0);
    for (let k = 0; k < N; k++) {
      let p = k;
      for (let i = k + 1; i < N; i++) if (Math.hypot(M[i][k].re, M[i][k].im) > Math.hypot(M[p][k].re, M[p][k].im)) p = i;
      if (p !== k) { [M[p], M[k]] = [M[k], M[p]]; det = cx(-det.re, -det.im); }
      const pk = M[k][k];
      det = cmul(det, pk);
      if (pk.re === 0 && pk.im === 0) return cx(0, 0);
      for (let i = k + 1; i < N; i++) {
        const f = cdiv(M[i][k], pk);
        for (let j = k + 1; j < N; j++) M[i][j] = csub(M[i][j], cmul(f, M[k][j]));
      }
    }
    return det;
  }
  /** host check of a parameter point: mean (equilibrium) and effective additive intensity */
  function statics(pv) {
    const v = coefValues(pv);
    const M = A.map((row, i) => row.map((a, j) => v[a] + B.reduce((s, Bd) => s + v[Bd[i][j]], 0)));
    const b = c.map((x) => -v[x]);
    for (let k = 0; k < N; k++) {
      let p = k;
      for (let i = k + 1; i < N; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
      [M[p], M[k]] = [M[k], M[p]]; [b[p], b[k]] = [b[k], b[p]];
      if (M[k][k] === 0) return null;
      for (let i = k + 1; i < N; i++) {
        const f = M[i][k] / M[k][k];
        for (let j = k; j < N; j++) M[i][j] -= f * M[k][j];
        b[i] -= f * b[k];
      }
    }
    const m = new Array(N).fill(0);
    for (let i = N - 1; i >= 0; i--) {
      let s = b[i];
      for (let j = i + 1; j < N; j++) s -= M[i][j] * m[j];
      m[i] = s / M[i][i];
    }
    let G = 0;
    for (let k = 0; k < nk; k++) {
      let g = v[gam[k]];
      for (let j = 0; j < N; j++) g += (v[alp[k][j]] + bet[k][j].reduce((s, x) => s + v[x], 0)) * m[j];
      G += g * g;
    }
    return { mean: m, Gamma: G };
  }

  return {
    params, decls, helpers: [...defAt.keys()], warnings, branchAuto: false, code, julia: null, evalD,
    nodes: order.length, quadrature: false,
    stochastic: { N, states, noises, delays: nd, noiseState: states[r], output: states[0], multiplicative, statics },
  };
}
