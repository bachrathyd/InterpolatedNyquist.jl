// The equation language of the demo: a characteristic function D(λ) typed as text.
//
//   # comment
//   γ = λ/sqrt(1 + η*λ)                  helper (usable in later lines)
//   (1 + exp(-2*γ))/2 - K*exp(-r*λ - γ)   the last line (or a line  D = ...) is D(λ)
//
// numbers (1.5e-3), + - * / ^ (** and superscripts ² ³ ⁻¹ as aliases), unary minus, parentheses;
// functions exp log (ln) sqrt sin cos tan sinh cosh tanh exprel pow(a, b); constants pi (π),
// i; the variable λ (or `lambda`). Every other identifier is a parameter. A line that starts or
// ends with an operator, or an open parenthesis, continues on the next line.
// Parameter ranges (not helpers):  P = 2.1:20,  P = 2.1:0.1:10 (start:step:stop; the step sets
// the grid size along an axis),  P = 2.1:20 @ 5 (initial value; default: the midpoint).
// A plain  name = number  is a fixed constant (a helper).
//
// Safety: the text is tokenized with a whitelist and parsed into an AST; the WGSL and the
// host (Float64) evaluator are generated from a typed expression DAG built from the AST. No
// user text reaches the shader: parameters become K[j] / p.x / p.y, helpers become t<id>,
// numbers are re-printed from their parsed value. No eval / new Function anywhere.
//
// The DAG (IR) is hash-consed: identical subexpressions (a helper used twice, the log λ of two
// fractional powers, the e^{-τλ} of two terms ...) are computed once. Types: R (real f32,
// independent of λ), C (complex constant, independent of λ), D (complex dual number in λ:
// value and d/dω). Integer powers become repeated squaring, other powers exp(b·log a) on the
// principal branch.

export class ExprError extends Error {
  constructor(msg, line = 0, col = 0) {
    super(msg);
    this.name = 'ExprError';
    this.line = line;
    this.col = col;
  }
  get where() { return this.line ? `line ${this.line}, column ${this.col}: ` : ''; }
}

export const MAX_PARAMS = 16;            // K[0..15] of march.wgsl
const ARITY = { exp: 1, log: 1, ln: 1, sqrt: 1, sin: 1, cos: 1, tan: 1, sinh: 1, cosh: 1, tanh: 1, exprel: 1, pow: 2 };
const SETTING_ARITY = { min: 2, max: 2, abs: 1 };          // real-valued settings only
const NON_ANALYTIC = new Set(['abs', 'real', 'imag', 'conj', 'min', 'max', 'arg', 'angle', 'sign', 'floor', 'ceil',
  'round', 're', 'im', 'Re', 'Im', 'hypot', 'mod', 'cabs', 'abs2']);
const CONSTS = { pi: [Math.PI, 0], π: [Math.PI, 0], i: [0, 1] };
const LAMBDA = new Set(['λ', 'lambda']);
const SUPER = { '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9' };
const OPCHARS = { '+': '+', '-': '-', '−': '-', '–': '-', '*': '*', '·': '*', '⋅': '*', '×': '*', '/': '/', '÷': '/',
  '^': '^', '(': '(', ')': ')', ',': ',', '=': '=', ':': ':', '@': '@' };
const SPACE = new Set([' ', '\t', '\r', ' ', ' ', ' ', '​']);
const NUM_RE = /(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/y;
const ID_RE = /[\p{L}_][\p{L}\p{Nd}\p{M}_₀-₉]*/uy;

// ---------------------------------------------------------------------------------------------
// tokenizer
// ---------------------------------------------------------------------------------------------
function tokenize(src) {
  const toks = [];
  let i = 0;
  let line = 1;
  let ls = 0;
  const n = src.length;
  while (i < n) {
    const ch = src[i];
    const col = i - ls + 1;
    if (ch === '\n') { toks.push({ t: 'nl', line, col }); i++; line++; ls = i; continue; }
    if (SPACE.has(ch)) { i++; continue; }
    if (ch === '#') { while (i < n && src[i] !== '\n') i++; continue; }
    if (ch === '*' && src[i + 1] === '*') { toks.push({ t: 'op', v: '^', line, col }); i += 2; continue; }
    if (OPCHARS[ch]) { toks.push({ t: 'op', v: OPCHARS[ch], line, col }); i++; continue; }
    if (SUPER[ch] !== undefined || ch === '⁻') {
      let s = '';
      const neg = ch === '⁻';
      if (neg) i++;
      while (i < n && SUPER[src[i]] !== undefined) s += SUPER[src[i++]];
      if (!s) throw new ExprError("a superscript minus must be followed by superscript digits", line, col);
      toks.push({ t: 'op', v: '^', line, col });
      if (neg) toks.push({ t: 'op', v: '-', line, col });
      toks.push({ t: 'num', v: Number(s), line, col, text: s });
      continue;
    }
    NUM_RE.lastIndex = i;
    let m = NUM_RE.exec(src);
    if (m) {
      const v = Number(m[0]);
      if (!Number.isFinite(v)) throw new ExprError(`the number ${m[0]} is too large`, line, col);
      toks.push({ t: 'num', v, line, col, text: m[0] });
      i += m[0].length;
      continue;
    }
    ID_RE.lastIndex = i;
    m = ID_RE.exec(src);
    if (m) {
      if (m[0].length > 40) throw new ExprError('identifier too long (40 characters at most)', line, col);
      toks.push({ t: 'id', v: m[0], line, col });
      i += m[0].length;
      continue;
    }
    const cp = String.fromCodePoint(src.codePointAt(i));
    throw new ExprError(`unexpected character '${cp}'`, line, col);
  }
  // line continuation: inside parentheses, after an operator, or before a line that starts
  // with + - * / ^
  const out = [];
  let depth = 0;
  for (let k = 0; k < toks.length; k++) {
    const t = toks[k];
    if (t.t === 'op') {
      if (t.v === '(') depth++;
      else if (t.v === ')') depth = Math.max(0, depth - 1);
    }
    if (t.t !== 'nl') { out.push(t); continue; }
    const prev = out[out.length - 1];
    let j = k + 1;
    while (j < toks.length && toks[j].t === 'nl') j++;
    const nx = toks[j];
    if (!prev || prev.t === 'nl') continue;
    if (depth > 0) continue;
    if (prev.t === 'op' && prev.v !== ')' && prev.v !== ':' && prev.v !== '@') continue;
    if (nx && nx.t === 'op' && '+-*/^'.includes(nx.v)) continue;
    out.push(t);
  }
  return out;
}

function splitStatements(toks) {
  const out = [[]];
  for (const t of toks) {
    if (t.t === 'nl') { if (out[out.length - 1].length) out.push([]); } else out[out.length - 1].push(t);
  }
  if (!out[out.length - 1].length) out.pop();
  return out;
}

// ---------------------------------------------------------------------------------------------
// parser (recursive descent):  expr := term (± term)*,  term := unary (*/ unary)*,
// unary := -unary | +unary | power,  power := primary (^ unary)?   (-x^2 = -(x^2), right-assoc.)
// ---------------------------------------------------------------------------------------------
class Parser {
  constructor(toks) {
    this.t = toks;
    this.k = 0;
    const last = toks[toks.length - 1];
    this.eof = { t: 'eof', line: last ? last.line : 1, col: last ? last.col + 1 : 1 };
  }
  peek() { return this.t[this.k] || this.eof; }
  next() { return this.t[this.k++] || this.eof; }
  isOp(v) { const t = this.peek(); return t.t === 'op' && t.v === v; }
  fail(msg, t = this.peek()) { throw new ExprError(msg, t.line, t.col); }
  expect(v) {
    if (!this.isOp(v)) {
      const t = this.peek();
      this.fail(t.t === 'eof' ? `missing '${v}' at the end` : `expected '${v}' before ${describe(t)}`);
    }
    return this.next();
  }
  expr() {
    let a = this.term();
    while (this.isOp('+') || this.isOp('-')) {
      const o = this.next();
      a = { t: 'bin', op: o.v, a, b: this.term(), line: o.line, col: o.col };
    }
    return a;
  }
  term() {
    let a = this.unary();
    while (this.isOp('*') || this.isOp('/')) {
      const o = this.next();
      a = { t: 'bin', op: o.v, a, b: this.unary(), line: o.line, col: o.col };
    }
    return a;
  }
  unary() {
    if (this.isOp('-')) { const o = this.next(); return { t: 'neg', a: this.unary(), line: o.line, col: o.col }; }
    if (this.isOp('+')) { this.next(); return this.unary(); }
    return this.power();
  }
  power() {
    const b = this.primary();
    if (this.isOp('^')) {
      const o = this.next();
      return { t: 'bin', op: '^', a: b, b: this.unary(), line: o.line, col: o.col };
    }
    return b;
  }
  primary() {
    const t = this.next();
    let node;
    if (t.t === 'num') node = { t: 'num', v: t.v, line: t.line, col: t.col };
    else if (t.t === 'id') {
      if (this.isOp('(')) {
        this.next();
        const args = [];
        if (!this.isOp(')')) {
          args.push(this.expr());
          while (this.isOp(',')) { this.next(); args.push(this.expr()); }
        }
        this.expect(')');
        node = { t: 'call', f: t.v, args, line: t.line, col: t.col };
      } else node = { t: 'id', name: t.v, line: t.line, col: t.col };
    } else if (t.t === 'op' && t.v === '(') {
      node = this.expr();
      this.expect(')');
    } else if (t.t === 'eof') this.fail('the expression ends too early', t);
    else this.fail(`unexpected ${describe(t)}`, t);
    const nx = this.peek();
    if (nx.t === 'num' || nx.t === 'id' || (nx.t === 'op' && nx.v === '(')) {
      this.fail(`missing operator before ${describe(nx)} (implicit multiplication is not supported: write e.g. 2*λ)`, nx);
    }
    return node;
  }
}

function describe(t) {
  if (t.t === 'num') return `the number ${t.text ?? t.v}`;
  if (t.t === 'id') return `'${t.v}'`;
  if (t.t === 'eof') return 'the end of the line';
  return `'${t.v}'`;
}

function parseStatement(toks) {
  let name = null;
  let body = toks;
  if (toks.length >= 2 && toks[0].t === 'id' && toks[1].t === 'op' && toks[1].v === '=') {
    name = toks[0].v;
    body = toks.slice(2);
    if (!body.length) throw new ExprError(`nothing after '${name} ='`, toks[1].line, toks[1].col);
    // a parameter range  name = start:stop  or  start:step:stop,  optionally  @ value
    let depth = 0;
    const cuts = [];
    let at = -1;
    body.forEach((t, k) => {
      if (t.t !== 'op') return;
      if (t.v === '(') depth++;
      else if (t.v === ')') depth--;
      else if (depth === 0 && t.v === ':' && at < 0) cuts.push(k);
      else if (depth === 0 && t.v === '@') {
        if (at >= 0) throw new ExprError("a second '@'", t.line, t.col);
        at = k;
      }
    });
    if (cuts.length || at >= 0) {
      if (!cuts.length) throw new ExprError(`'@' belongs to a range:  ${name} = min:max @ value`, body[at].line, body[at].col);
      if (cuts.length > 2) throw new ExprError('a range is  start:stop  or  start:step:stop', body[cuts[2]].line, body[cuts[2]].col);
      const end = at >= 0 ? at : body.length;
      const bounds = [-1, ...cuts, end];
      const part = (a, b, what, ref) => {
        const seg = body.slice(a + 1, b);
        if (!seg.length) throw new ExprError(`the ${what} of the range is missing`, ref.line, ref.col);
        const q = new Parser(seg);
        const e = q.expr();
        if (q.peek().t !== 'eof') q.fail(`unexpected ${describe(q.peek())}`);
        return e;
      };
      const names = cuts.length === 1 ? ['start', 'stop'] : ['start', 'step', 'stop'];
      const parts = names.map((w, k) => part(bounds[k], bounds[k + 1], w, k ? body[bounds[k]] : toks[1]));
      const value = at >= 0 ? part(at, body.length, 'value after @', body[at]) : null;
      return { kind: 'decl', name, parts, value, line: toks[0].line, col: toks[0].col };
    }
  }
  const p = new Parser(body);
  const ast = p.expr();
  const rest = p.peek();
  if (rest.t !== 'eof') {
    if (rest.t === 'op' && rest.v === '=') p.fail("unexpected '=': a definition is  name = expression,  one per line", rest);
    if (rest.t === 'op' && rest.v === ')') p.fail("unmatched ')'", rest);
    p.fail(`unexpected ${describe(rest)}`, rest);
  }
  return { kind: 'stmt', name, ast, line: toks[0].line, col: toks[0].col };
}

/** { stmts, decls }: the statements (the last one, or the line `D = ...`, is D) and the
 * parameter-range declarations (anywhere) */
export function parseProgram(src) {
  if (typeof src !== 'string') throw new ExprError('no equation');
  if (src.length > 20000) throw new ExprError('the equation is too long (20000 characters at most)');
  const all = splitStatements(tokenize(src)).map(parseStatement);
  const decls = all.filter((s) => s.kind === 'decl');
  const stmts = all.filter((s) => s.kind !== 'decl');
  if (!stmts.length) throw new ExprError(decls.length ? 'D(λ) is missing: after the parameter ranges, the last line is the characteristic function' : 'empty: type D(λ), e.g.   λ^2 + a*λ + b*exp(-τ*λ)');
  const dIdx = stmts.findIndex((s) => s.name === 'D');
  if (dIdx >= 0 && dIdx !== stmts.length - 1) {
    const s = stmts[dIdx + 1];
    throw new ExprError('nothing may follow the line  D = ...', s.line, s.col);
  }
  stmts.forEach((s, k) => {
    if (k < stmts.length - 1 && s.name === null) {
      throw new ExprError('only the last line can be a plain expression (it is D); write  name = expression  for a helper', s.line, s.col);
    }
  });
  return { stmts, decls };
}

// a constant (range bound): numbers, pi, arithmetic
function constValue(ast, what) {
  const ir = new IR();
  const ctx = {
    settings: false,
    isName: () => false,
    ident: (a) => {
      if (CONSTS[a.name] && a.name !== 'i') return ir.lit(CONSTS[a.name][0]);
      throw new ExprError(`the ${what} must be a number (pi allowed), not '${a.name}'`, a.line, a.col);
    },
  };
  const n = ir.nodes[lowerAst(ir, ast, ctx)];
  if (n.op !== 'lit' || n.val.im !== 0 || !Number.isFinite(n.val.re)) throw new ExprError(`the ${what} is not a finite real number`, ast.line, ast.col);
  return n.val.re;
}

// ---------------------------------------------------------------------------------------------
// host complex arithmetic (Float64): constant folding, the evaluator, settings
// ---------------------------------------------------------------------------------------------
const cx = (re, im = 0) => ({ re, im });
function cdivH(a, b) {
  if (b.im === 0) return cx(a.re / b.re, a.im / b.re);
  const s = Math.max(Math.abs(b.re), Math.abs(b.im));
  const br = b.re / s, bi = b.im / s, d = br * br + bi * bi;
  return cx((a.re / s * br + a.im / s * bi) / d, (a.im / s * br - a.re / s * bi) / d);
}
function cexpH(a) {
  const e = Math.exp(a.re);
  return a.im === 0 ? cx(e, 0) : cx(e * Math.cos(a.im), e * Math.sin(a.im));
}
function csqrtH(z) {
  const r = Math.hypot(z.re, z.im);
  if (!(r > 0)) return cx(0, 0);
  if (z.re >= 0) { const t = Math.sqrt(0.5 * (r + z.re)); return cx(t, 0.5 * z.im / t); }
  const t = Math.sqrt(0.5 * (r - z.re));
  return cx(0.5 * Math.abs(z.im) / t, z.im < 0 ? -t : t);
}
function ctanhH(a) {
  const s = a.re < 0 ? -1 : 1;
  const e = cexpH(cx(-2 * s * a.re, -2 * s * a.im));
  const t = cdivH(cx(1 - e.re, -e.im), cx(1 + e.re, e.im));
  return cx(s * t.re, s * t.im);
}
function cexprelH(a) {
  if (Math.hypot(a.re, a.im) < 0.5) {
    // Σ x^k/(k+1)!, k = 0..19 (Horner)
    let r = cx(1 / 20922789888000, 0);         // 1/20!   ... built downwards
    let f = 20922789888000;
    for (let k = 19; k >= 1; k--) {
      f /= (k + 1);
      r = cx(a.re * r.re - a.im * r.im + 1 / f, a.re * r.im + a.im * r.re);
    }
    return r;
  }
  const e = cexpH(a);
  return cdivH(cx(e.re - 1, e.im), a);
}
const CF = {
  add: (a, b) => cx(a.re + b.re, a.im + b.im),
  sub: (a, b) => cx(a.re - b.re, a.im - b.im),
  mul: (a, b) => (a.im === 0 && b.im === 0 ? cx(a.re * b.re, 0) : cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)),
  div: cdivH,
  neg: (a) => cx(-a.re, -a.im),
  inv: (a) => cdivH(cx(1, 0), a),
  exp: cexpH,
  log: (a) => cx(Math.log(Math.hypot(a.re, a.im)), a.im === 0 ? (a.re < 0 ? Math.PI : 0) : Math.atan2(a.im, a.re)),
  sqrt: csqrtH,
  sin: (a) => cx(Math.sin(a.re) * Math.cosh(a.im), Math.cos(a.re) * Math.sinh(a.im)),
  cos: (a) => cx(Math.cos(a.re) * Math.cosh(a.im), -Math.sin(a.re) * Math.sinh(a.im)),
  sinh: (a) => cx(Math.sinh(a.re) * Math.cos(a.im), Math.cosh(a.re) * Math.sin(a.im)),
  cosh: (a) => cx(Math.cosh(a.re) * Math.cos(a.im), Math.sinh(a.re) * Math.sin(a.im)),
  tanh: ctanhH,
  tan: (a) => { const t = ctanhH(cx(-a.im, a.re)); return cx(t.im, -t.re); },
  exprel: cexprelH,
  min: (a, b) => cx(Math.min(a.re, b.re)),
  max: (a, b) => cx(Math.max(a.re, b.re)),
  abs: (a) => cx(Math.hypot(a.re, a.im)),
};

// ---------------------------------------------------------------------------------------------
// typed expression DAG
// ---------------------------------------------------------------------------------------------
const maxTy = (a, b) => ((a === 'D' || b === 'D') ? 'D' : (a === 'C' || b === 'C') ? 'C' : 'R');
const SAME_TY = new Set(['neg', 'inv', 'exp', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exprel']);

class IR {
  constructor() {
    this.nodes = [];
    this.map = new Map();
  }
  _add(op, ty, args, extra = '', init = null) {
    const key = op + '|' + args.join(',') + '|' + extra;
    let id = this.map.get(key);
    if (id === undefined) {
      id = this.nodes.length;
      this.nodes.push({ id, op, ty, args, ...(init || {}) });
      this.map.set(key, id);
    }
    return id;
  }
  lit(re, im = 0) {
    if (Object.is(re, -0)) re = 0;
    if (Object.is(im, -0)) im = 0;
    return this._add('lit', im === 0 ? 'R' : 'C', [], `${re},${im}`, { val: cx(re, im) });
  }
  lam() { return this._add('lam', 'D', []); }
  par(name) { return this._add('par', 'R', [], name, { name }); }
  isLit(a, v) { const n = this.nodes[a]; return n.op === 'lit' && n.val.im === 0 && n.val.re === v; }
  add(a, b) { if (this.isLit(a, 0)) return b; if (this.isLit(b, 0)) return a; return this.op2('add', a, b); }
  sub(a, b) { if (this.isLit(b, 0)) return a; if (this.isLit(a, 0)) return this.neg(b); return this.op2('sub', a, b); }
  mul(a, b) {
    if (this.isLit(a, 1)) return b;
    if (this.isLit(b, 1)) return a;
    if (this.isLit(a, -1)) return this.neg(b);
    if (this.isLit(b, -1)) return this.neg(a);
    return this.op2('mul', a, b);
  }
  div(a, b) { if (this.isLit(b, 1)) return a; return this.op2('div', a, b); }
  neg(a) {
    const n = this.nodes[a];
    if (n.op === 'neg') return n.args[0];
    if (n.op === 'mul' && n.ty !== 'R') {
      // -(r·x) -> (-r)·x: the real factor carries the sign (dscale(x, -r))
      const [x, y] = n.args;
      if (this.nodes[x].ty === 'R') return this.mul(this.neg(x), y);
      if (this.nodes[y].ty === 'R') return this.mul(x, this.neg(y));
    }
    return this.op2('neg', a);
  }
  op2(o, ...args) {
    const N = args.map((a) => this.nodes[a]);
    if (N.every((n) => n.op === 'lit')) {
      const r = CF[o](...N.map((n) => n.val));
      return this.lit(r.re, r.im);
    }
    let ty;
    if (args.length === 2 && o !== 'min' && o !== 'max') ty = maxTy(N[0].ty, N[1].ty);
    else if (SAME_TY.has(o)) ty = N[0].ty;
    else if (o === 'log' || o === 'sqrt') ty = N[0].ty === 'R' ? 'C' : N[0].ty;
    else ty = 'R';
    // commutative: one node for a*b and b*a (exact in IEEE arithmetic, also for the dual products)
    const kargs =(o === 'add' || o === 'mul') ? args.slice().sort((x, y) => x - y) : args;
    const key = o + '|' + kargs.join(',') + '|';
    let id = this.map.get(key);
    if (id === undefined) {
      id = this.nodes.length;
      this.nodes.push({ id, op: o, ty, args });
      this.map.set(key, id);
    }
    return id;
  }
  ipow(a, n) {
    if (n === 0) return this.lit(1);
    if (n < 0) return this.op2('inv', this.ipow(a, -n));
    if (n === 1) return a;
    if (n % 2 === 0) { const h = this.ipow(a, n / 2); return this.mul(h, h); }
    return this.mul(this.ipow(a, n - 1), a);
  }
  pow(a, b) {
    const B = this.nodes[b];
    if (B.op === 'lit' && B.val.im === 0) {
      const e = B.val.re;
      if (Number.isInteger(e) && Math.abs(e) <= 1e6) return this.ipow(a, e);
      if (e === 0.5) return this.op2('sqrt', a);
      if (e === -0.5) return this.op2('inv', this.op2('sqrt', a));
    }
    // principal branch a^b = exp(b log a)
    return this.op2('exp', this.mul(b, this.op2('log', a)));
  }
}

// ---------------------------------------------------------------------------------------------
// lowering AST -> IR
// ---------------------------------------------------------------------------------------------
function lowerAst(ir, ast, ctx) {
  const L = (a) => lowerAst(ir, a, ctx);
  switch (ast.t) {
    case 'num': return ir.lit(ast.v);
    case 'id': return ctx.ident(ast);
    case 'neg': return ir.neg(L(ast.a));
    case 'bin': {
      const a = L(ast.a);
      const b = L(ast.b);
      switch (ast.op) {
        case '+': return ir.add(a, b);
        case '-': return ir.sub(a, b);
        case '*': return ir.mul(a, b);
        case '/': return ir.div(a, b);
        default: return ir.pow(a, b);
      }
    }
    case 'call': {
      const f = ast.f;
      const ar = ARITY[f] ?? (ctx.settings ? SETTING_ARITY[f] : undefined);
      if (ar === undefined) {
        if (NON_ANALYTIC.has(f)) {
          throw new ExprError(`${f}(...) is not analytic in λ: the argument principle needs an analytic D (use only + - * / ^ exp log sqrt sin cos tan sinh cosh tanh exprel)`, ast.line, ast.col);
        }
        if (LAMBDA.has(f) || CONSTS[f] || ctx.isName(f)) {
          throw new ExprError(`'${f}' is not a function (implicit multiplication is not supported: write ${f}*(...))`, ast.line, ast.col);
        }
        throw new ExprError(`unknown function '${f}' (known: exp, log, sqrt, sin, cos, tan, sinh, cosh, tanh, exprel, pow)`, ast.line, ast.col);
      }
      if (ast.args.length !== ar) {
        throw new ExprError(`${f} takes ${ar} argument${ar > 1 ? 's' : ''}, got ${ast.args.length}`, ast.line, ast.col);
      }
      const args = ast.args.map(L);
      if (f === 'pow') return ir.pow(args[0], args[1]);
      if (f === 'ln') return ir.op2('log', args[0]);
      return ir.op2(f, ...args);
    }
    default: throw new ExprError('internal: unknown node');
  }
}

// ---------------------------------------------------------------------------------------------
// WGSL emission
// ---------------------------------------------------------------------------------------------
function f32lit(v) {
  if (!Number.isFinite(v)) throw new ExprError(`a constant subexpression evaluates to ${v}`);
  if (Math.abs(v) > 3.4e38) throw new ExprError(`the constant ${v} is outside the Float32 range`);
  let s = String(v);
  if (!/[.eE]/.test(s)) s += '.0';
  return v < 0 ? `(${s})` : s;
}

function emitNode(nd, A, T, B) {
  const [a, b] = A;
  const [ta, tb] = T;
  const R = (x) => `C(${x}, 0.0)`;          // a real as a complex
  const Z = (x) => `CD(${x}, C(0.0, 0.0))`; // a complex constant as a dual number
  switch (nd.op) {
    case 'add':
      if (ta === 'R' && tb === 'R') return `(${a} + ${b})`;
      if (ta !== 'D' && tb !== 'D') return `(${ta === 'R' ? R(a) : a} + ${tb === 'R' ? R(b) : b})`;
      if (ta === 'D' && tb === 'D') return `dadd(${a}, ${b})`;
      {
        const [d, o, to] = ta === 'D' ? [a, b, tb] : [b, a, ta];
        return to === 'R' ? `daddr(${d}, ${o})` : `daddc(${d}, ${o})`;
      }
    case 'sub':
      if (ta === 'R' && tb === 'R') return `(${a} - ${b})`;
      if (ta !== 'D' && tb !== 'D') return `(${ta === 'R' ? R(a) : a} - ${tb === 'R' ? R(b) : b})`;
      if (ta === 'D' && tb === 'D') return `dsub(${a}, ${b})`;
      if (ta === 'D') return tb === 'R' ? `daddr(${a}, -(${b}))` : `daddc(${a}, -(${b}))`;
      return ta === 'R' ? `daddr(dneg(${b}), ${a})` : `daddc(dneg(${b}), ${a})`;
    case 'mul':
      if (ta === 'R' && tb === 'R') return `(${a} * ${b})`;
      if (ta === 'C' && tb === 'C') return `cmul(${a}, ${b})`;
      if (ta !== 'D' && tb !== 'D') return ta === 'C' ? `(${a} * ${b})` : `(${b} * ${a})`;
      if (ta === 'D' && tb === 'D') return `dmul(${a}, ${b})`;
      {
        const [d, o, to] = ta === 'D' ? [a, b, tb] : [b, a, ta];
        return to === 'R' ? `dscale(${d}, ${o})` : `dmulc(${d}, ${o})`;
      }
    case 'div':
      if (ta === 'R' && tb === 'R') return `(${a} / ${b})`;
      if (ta === 'C' && tb === 'R') return `(${a} / ${b})`;
      if (ta === 'R' && tb === 'C') return `cdiv(${R(a)}, ${b})`;
      if (ta === 'C' && tb === 'C') return `cdiv(${a}, ${b})`;
      if (ta === 'D' && tb === 'D') return `ddiv(${a}, ${b})`;
      if (ta === 'D' && tb === 'R') {
        // division by a power of two: an exact multiplication
        if (B[1].op === 'lit') {
          const v = B[1].val.re;
          const m = Math.log2(Math.abs(v));
          if (Number.isInteger(m) && Math.abs(m) < 100) return `dscale(${a}, ${f32lit(1 / v)})`;
        }
        return `ddivr(${a}, ${b})`;
      }
      if (ta === 'D') return `ddivc(${a}, ${b})`;
      return tb === 'D' && ta === 'R' ? `dscale(dinv(${b}), ${a})` : `dmulc(dinv(${b}), ${a})`;
    case 'neg': return ta === 'D' ? `dneg(${a})` : `(-${a})`;
    case 'inv': return ta === 'R' ? `(1.0 / ${a})` : ta === 'C' ? `cdiv(C(1.0, 0.0), ${a})` : `dinv(${a})`;
    case 'exp': return ta === 'R' ? `exp(${a})` : ta === 'C' ? `cexp(${a})` : `dexp(${a})`;
    case 'log': return ta === 'R' ? `clog(${R(a)})` : ta === 'C' ? `clog(${a})` : `dlog(${a})`;
    case 'sqrt': return ta === 'R' ? `csqrt(${R(a)})` : ta === 'C' ? `csqrt(${a})` : `dsqrt(${a})`;
    case 'sin': case 'cos': case 'tan': case 'sinh': case 'cosh': case 'tanh': case 'exprel': {
      const f = nd.op;
      if (ta === 'D') return `d${f}(${a})`;
      if (ta === 'C') return `d${f}(${Z(a)}).v`;
      if (f === 'sin') return `sincos(${a}).x`;
      if (f === 'cos') return `sincos(${a}).y`;
      if (f === 'tan') return `rtan(${a})`;
      if (f === 'exprel') return `rexprel(${a})`;
      return `${f}(${a})`;
    }
    default: throw new ExprError(`internal: no WGSL for ${nd.op}`);
  }
}

// parameter ranges: name = start:stop | start:step:stop [@ value]  ->  Map name -> { min, max,
// step, count (grid points along an axis), value, line }
function buildDecls(declStmts, defAt, stmts, reserved) {
  const decls = new Map();
  for (const s of declStmts) {
    const nm = s.name;
    reserved(nm, s);
    if (decls.has(nm)) throw new ExprError(`the range of '${nm}' is declared twice (lines ${decls.get(nm).line} and ${s.line})`, s.line, s.col);
    if (defAt.has(nm)) throw new ExprError(`'${nm}' is declared as a parameter range and defined as a helper (line ${stmts[defAt.get(nm)].line})`, s.line, s.col);
    const v = s.parts.map((a, k) => constValue(a, s.parts.length === 3 && k === 1 ? 'step' : 'range bound'));
    const min = v[0];
    const max = v[v.length - 1];
    const step = v.length === 3 ? v[1] : null;
    if (!(max > min)) throw new ExprError(`the range of '${nm}' must have start < stop`, s.line, s.col);
    let count = null;
    if (step !== null) {
      if (!(step > 0)) throw new ExprError(`the step of '${nm}' must be positive`, s.line, s.col);
      count = Math.floor((max - min) / step + 1e-9) + 1;
      if (count < 2) throw new ExprError(`the step of '${nm}' is larger than its range`, s.line, s.col);
    }
    const value = s.value ? constValue(s.value, 'value after @') : null;
    decls.set(nm, { min, max, step, count, value, line: s.line });
  }
  return decls;
}

/** the parameter ranges of a text without compiling D (built-in examples) */
export function rangeDecls(src) {
  const all = splitStatements(tokenize(src)).map(parseStatement);
  return buildDecls(all.filter((s) => s.kind === 'decl'), new Map(), [], () => {});
}

// ---------------------------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------------------------
/**
 * Compile the text of a characteristic function.
 * -> { params: [names] (first appearance, only those D depends on), helpers, warnings,
 *      branchAuto (λ^x with non-integer x, log λ or sqrt λ: a branch point at λ = 0),
 *      code(ix, iy, iz) -> WGSL `fn charD(l: CD, p: vec2<f32>) -> CD` (param ix -> p.x, iy -> p.y,
 *      iz -> PZ (3D grids), others -> K[j]), evalD(λ, values) -> D(λ) in Float64, nodes (DAG size) }
 */
export function compileModel(src) {
  const { stmts, decls: declStmts } = parseProgram(src);
  const ir = new IR();
  const defAt = new Map();
  const helpers = new Map();
  const used = new Map();
  const last = stmts.length - 1;
  const reserved = (nm, s) => {
    if (LAMBDA.has(nm)) throw new ExprError(`'${nm}' is the variable and cannot be defined`, s.line, s.col);
    if (CONSTS[nm]) throw new ExprError(`'${nm}' is a constant and cannot be redefined`, s.line, s.col);
    if (ARITY[nm] || NON_ANALYTIC.has(nm)) throw new ExprError(`'${nm}' is a function name`, s.line, s.col);
  };
  stmts.forEach((s, k) => {
    if (s.name === null || (k === last && s.name === 'D')) return;
    const nm = s.name;
    reserved(nm, s);
    if (defAt.has(nm)) throw new ExprError(`'${nm}' is defined twice (lines ${stmts[defAt.get(nm)].line} and ${s.line})`, s.line, s.col);
    defAt.set(nm, k);
  });
  const decls = buildDecls(declStmts, defAt, stmts, reserved);
  const pnames = [...decls.keys()];          // declared parameters first, in declaration order
  let cur = 0;
  const ctx = {
    settings: false,
    isName: (n) => defAt.has(n) || pnames.includes(n),
    ident: (ast) => {
      const n = ast.name;
      if (LAMBDA.has(n)) return ir.lam();
      if (CONSTS[n]) return ir.lit(CONSTS[n][0], CONSTS[n][1]);
      if (ARITY[n]) throw new ExprError(`'${n}' is a function: write ${n}(...)`, ast.line, ast.col);
      if (defAt.has(n)) {
        const k = defAt.get(n);
        if (k >= cur) {
          throw new ExprError(k === cur ? `'${n}' is used in its own definition` :
            `'${n}' is used before its definition in line ${stmts[k].line}`, ast.line, ast.col);
        }
        used.set(n, (used.get(n) || 0) + 1);
        return helpers.get(n);
      }
      if (!pnames.includes(n)) pnames.push(n);
      return ir.par(n);
    },
  };
  let root = null;
  stmts.forEach((s, k) => {
    cur = k;
    const id = lowerAst(ir, s.ast, ctx);
    if (k === last) root = id;
    else helpers.set(s.name, id);
  });
  const warnings = [];
  for (const [nm] of helpers) if (!used.get(nm)) warnings.push(`helper '${nm}' is not used`);
  if (ir.nodes[root].ty !== 'D') {
    throw new ExprError('D does not depend on λ (write the variable as λ or lambda)', stmts[last].line, stmts[last].col);
  }
  // reachable nodes (creation order is a topological order)
  const reach = new Uint8Array(ir.nodes.length);
  const stack = [root];
  while (stack.length) {
    const id = stack.pop();
    if (reach[id]) continue;
    reach[id] = 1;
    for (const a of ir.nodes[id].args) stack.push(a);
  }
  const order = [];
  for (let id = 0; id < ir.nodes.length; id++) if (reach[id]) order.push(id);
  const params = pnames.filter((n) => order.some((id) => ir.nodes[id].op === 'par' && ir.nodes[id].name === n));
  for (const n of pnames) if (!params.includes(n)) warnings.push(`parameter '${n}' has no effect on D`);
  if (params.length > MAX_PARAMS) throw new ExprError(`${params.length} parameters: at most ${MAX_PARAMS} are supported`);
  const pidx = new Map(params.map((n, i) => [n, i]));
  const lamId = ir.map.get('lam||');
  const branchAuto = order.some((id) => {
    const n = ir.nodes[id];
    return (n.op === 'log' || n.op === 'sqrt') && n.args[0] === lamId;
  });
  // Float32 range of the literals: checked once here (also when the code is never emitted)
  for (const id of order) {
    const n = ir.nodes[id];
    if (n.op === 'lit') { f32lit(n.val.re); f32lit(n.val.im); }
  }

  const codeCache = new Map();
  function code(ix, iy, iz = -1) {
    const ck = ix + ',' + iy + ',' + iz;
    if (codeCache.has(ck)) return codeCache.get(ck);
    const name = (id) => {
      const n = ir.nodes[id];
      if (n.op === 'lit') return n.ty === 'R' ? f32lit(n.val.re) : `C(${f32lit(n.val.re)}, ${f32lit(n.val.im)})`;
      if (n.op === 'lam') return 'l';
      if (n.op === 'par') {
        const j = pidx.get(n.name);
        return j === ix ? 'p.x' : j === iy ? 'p.y' : j === iz ? 'PZ' : `K[${j}]`;
      }
      return `t${id}`;
    };
    const lines = [];
    for (const id of order) {
      const n = ir.nodes[id];
      if (n.op === 'lit' || n.op === 'lam' || n.op === 'par') continue;
      const B = n.args.map((a) => ir.nodes[a]);
      lines.push(`    let t${id} = ${emitNode(n, n.args.map(name), B.map((x) => x.ty), B)};`);
    }
    const s = `fn charD(l: CD, p: vec2<f32>) -> CD {\n${lines.join('\n')}\n    return ${name(root)};\n}`;
    codeCache.set(ck, s);
    return s;
  }

  function evalD(lam, pv) {
    const r = new Array(ir.nodes.length);
    for (const id of order) {
      const n = ir.nodes[id];
      switch (n.op) {
        case 'lit': r[id] = n.val; break;
        case 'lam': r[id] = lam; break;
        case 'par': r[id] = cx(pv[pidx.get(n.name)], 0); break;
        default: r[id] = n.args.length === 2 ? CF[n.op](r[n.args[0]], r[n.args[1]]) : CF[n.op](r[n.args[0]]);
      }
    }
    return r[root];
  }

  return { params, decls, helpers: [...helpers.keys()], warnings, branchAuto, code, evalD, nodes: order.length };
}

// ---------------------------------------------------------------------------------------------
// settings: real expressions of the parameters (ω_max, h_max, ...), with min, max, abs and inf
// ---------------------------------------------------------------------------------------------
const settingAst = new Map();
/**
 * Value of a setting expression; env: { name: value }. Empty -> null.
 * Throws ExprError (unknown name, λ, complex or non-finite result unless `inf`).
 */
export function evalSetting(src, env) {
  const s = String(src ?? '').trim();
  if (!s) return null;
  let ast = settingAst.get(s);
  if (!ast) {
    if (s.length > 500) throw new ExprError('too long');
    const toks = tokenize(s).filter((t) => t.t !== 'nl');
    const p = new Parser(toks);
    ast = p.expr();
    if (p.peek().t !== 'eof') p.fail(`unexpected ${describe(p.peek())}`);
    if (settingAst.size > 500) settingAst.clear();
    settingAst.set(s, ast);
  }
  const ir = new IR();
  const ctx = {
    settings: true,
    isName: (n) => Object.prototype.hasOwnProperty.call(env, n),
    ident: (a) => {
      const n = a.name;
      if (n === 'inf' || n === 'Inf' || n === '∞') return ir.lit(Infinity);
      if (CONSTS[n]) return ir.lit(CONSTS[n][0], CONSTS[n][1]);
      if (LAMBDA.has(n)) throw new ExprError('a setting cannot depend on λ', a.line, a.col);
      if (Object.prototype.hasOwnProperty.call(env, n)) return ir.lit(env[n]);
      throw new ExprError(`unknown name '${n}' (use parameters, numbers, pi, inf, min, max, abs)`, a.line, a.col);
    },
  };
  const id = lowerAst(ir, ast, ctx);
  const n = ir.nodes[id];
  if (n.op !== 'lit') throw new ExprError('internal: setting not constant');
  const v = n.val;
  if (Math.abs(v.im) > 1e-12 * Math.max(1, Math.abs(v.re))) throw new ExprError('a setting must be real');
  if (Number.isNaN(v.re)) throw new ExprError('the setting evaluates to NaN');
  return v.re;
}

// ---------------------------------------------------------------------------------------------
// leading order (paper, eq. npow): least-squares slope of ln|D(s)| over log-spaced real s in
// [s*/10, s*], s* = 1e8 reduced until |D| is finite; one decade lower as a check
// ---------------------------------------------------------------------------------------------
export function leadingOrder(evalD, pv) {
  const f = (s) => { const z = evalD(cx(s, 0), pv); return Math.hypot(z.re, z.im); };
  const slope = (hi) => {
    const m = 9;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let k = 0; k < m; k++) {
      const s = hi * Math.pow(10, -k / (m - 1));
      const v = f(s);
      if (!(v > 0 && v < Infinity)) return NaN;
      const x = Math.log(s), y = Math.log(v);
      sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    return (m * sxy - sx * sy) / (m * sxx - sx * sx);
  };
  let hi = 1e8;
  let n1 = NaN;
  for (; hi >= 1; hi /= 10) {
    n1 = slope(hi);
    if (Number.isFinite(n1)) break;
  }
  if (!Number.isFinite(n1)) {
    return { n: NaN, hi: 0, warn: '|D(s)| is not finite (or zero) along the real axis up to 1e8: set the order n manually (Advanced)' };
  }
  const n2 = slope(hi / 10);
  let warn = null;
  if (!(Math.abs(n1 - n2) <= 0.02)) {
    warn = `ln|D(s)| is not linear in ln s on the real axis (slope ${n1.toFixed(3)} on [${hi / 10}, ${hi}], ` +
      `${Number.isFinite(n2) ? n2.toFixed(3) : 'undefined'} one decade lower): exponential growth (an advanced term?) ` +
      'or not yet asymptotic. Check n or set it manually (Advanced).';
  } else if (hi < 1e8) {
    warn = `|D(s)| overflows above s = ${hi}: n estimated on [${hi / 10}, ${hi}]`;
  }
  let n = n1;
  if (Math.abs(n - Math.round(n)) < 1e-6) n = Math.round(n);
  return { n, hi, warn };
}

/** D(λ̄) = conj D(λ) at two test points (real coefficients: the count over ω ≥ 0 relies on it) */
export function realCoefficients(evalD, pv) {
  for (const l of [cx(0.31, 1.73), cx(-0.47, 0.29)]) {
    const a = evalD(l, pv);
    const b = evalD(cx(l.re, -l.im), pv);
    const d = Math.hypot(a.re - b.re, a.im + b.im);
    const s = 1 + Math.hypot(a.re, a.im);
    if (Number.isFinite(d) && d > 1e-9 * s) return false;
  }
  return true;
}
