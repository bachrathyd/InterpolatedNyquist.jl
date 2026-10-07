// UI of the WebGPU stability-chart demo: the equation box (text -> expr.js -> generated WGSL)
// with parameter-range lines kept in sync with the parameter table (sliders, ranges, X / Y / Z
// axis choice), march settings, automatic resolution (about 20 frames per second), a 5 s
// watchdog, a render scheduler (the newest settings win, a running chart is finished first),
// 2D axes / box zoom / hover read-out, the experimental 3D view, shareable links, and the
// ?validate=1 / ?bench=1 / ?bench=compare modes.

import { EXAMPLES, GROUPS, SET_DEFAULTS } from './examples.js';
import { Engine, WebGPUUnavailable, countOf } from './engine.js';
import { compileModel, rangeDecls, evalSetting, leadingOrder, realCoefficients, ExprError } from './expr.js';
import { Volume3D, volumeBytes, defaultCamera, project, boundaryMesh, runMesh, meshSTL } from './view3d.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const ZCAP = 6;
const OWN_KEY = 'nyquistgpu.own.v1';
const RES_CHOICES = ['auto', '320x180', '480x270', '960x540', '1920x1080', '3840x2160'];
const TARGET_MS = 50;            // auto resolution: one frame in about 50 ms (20 fps)
const DEADLINE_MS = 5000;        // watchdog in auto mode: no further row band after 5 s
// (a fixed resolution uses the user's time limit, state.limit seconds, 1 .. 600)
const SET_INPUTS = ['n', 'w0', 'hmax', 'wband', 'branch'];      // text fields of Advanced
const BG = [0x11 / 255, 0x13 / 255, 0x18 / 255];
const exByKey = new Map(EXAMPLES.map((e) => [e.key, e]));
const dict = () => Object.create(null);          // name-keyed maps (parameter names are user text)

class UserError extends Error {
  constructor(msg, where = 'eq') { super(msg); this.where = where; }
}

const state = {
  key: exByKey.has(params.get('ex')) ? params.get('ex') : 'fourth',
  exact: params.get('exact') !== '0',             // default: exact roots; &exact=0: fast mode
  boundary: true,
  flags: false,
  live: true,
  limit: 5,                // time limit [s] of a chart at a fixed resolution
  alpha: 0.25,             // 3D: interior opacity (0: only the boundary, 1: solid)
  surf: 0.22,              // 3D: opacity of the boundary surface
  smooth: true,            // 3D: boundary as a marching-tetrahedra mesh (else ray-marched)
  cam: defaultCamera(),
};
const slots = new Map();
let engine = null;
let vol3d = null;
let last = null;           // last read-back chart { nx, ny, xr, yr, z, s, steps, flags, xl, yl }
let lastJob = null;
let lastFrame = null;      // camera frame of the last 3D draw

function logMsg(s) {
  console.log(s);
  $('log').textContent += s + '\n';
}
function showError(html) {
  const m = $('msg');
  m.innerHTML = html;
  m.style.display = 'block';
}

const fmt = (v) => {
  const a = Math.abs(v);
  if (!Number.isFinite(v)) return String(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(2);
  return (+v.toPrecision(4)).toString();
};
const fmtIn = (v) => String(+(+v).toPrecision(6));
const fmtN = (n) => (Number.isInteger(n) ? String(n) : n.toFixed(4));
const sig2 = (v) => +v.toPrecision(2);
const is3D = (job) => !!job && job.nz > 1;

// ---------------------------------------------------------------------------------------------
// slots: the state of each example (equation text, parameters, axes, settings)
// ---------------------------------------------------------------------------------------------
function freshSlot(ex) {
  return {
    key: ex.key, text: ex.text, params: dict(), axes: [...ex.axes.slice(0, 2), null], set: { ...SET_DEFAULTS, ...ex.set },
    smin: ex.smin, res: 'auto', model: null, modelText: null, err: null, consts: dict(), declSig: dict(), jobWarns: [],
  };
}

function getSlot(key) {
  if (!slots.has(key)) {
    const s = freshSlot(exByKey.get(key));
    compileSlot(s);
    slots.set(key, s);
  }
  return slots.get(key);
}
const cur = () => getSlot(state.key);
const curEx = () => exByKey.get(state.key);

function defaultParam(slot, name, d) {
  if (d) return { min: d.min, max: d.max, step: d.step, value: d.value ?? (d.min + d.max) / 2 };
  const v = slot.consts[name];                  // a deleted helper  name = number
  if (Number.isFinite(v)) return v === 0 ? { value: 0, min: -1, max: 1 } : { value: v, min: Math.min(0, 2 * v), max: Math.max(0, 2 * v) };
  return { value: 1, min: 0, max: 2 };
}

function builtinModel(ex, text) {
  const decls = rangeDecls(text);
  return { params: [...decls.keys()], decls, builtin: true, helpers: [], warnings: [], branchAuto: false, code: () => ex.builtin.wgsl, evalD: null, nodes: 0 };
}

/** (re)compile the slot's text; on an error the last valid model stays (the chart keeps it) */
function compileSlot(slot) {
  if (slot.model && slot.modelText === slot.text) return true;
  const ex = exByKey.get(slot.key);
  let m;
  try {
    m = ex.builtin ? builtinModel(ex, slot.text) : compileModel(slot.text);
  } catch (e) {
    if (!(e instanceof ExprError)) console.error(e);
    slot.err = e;
    slot.modelText = slot.text;
    return false;
  }
  slot.err = null;
  slot.model = m;
  slot.modelText = slot.text;
  // parameters: a declared range (name = min:max @ value) sets min / max (and the value) when
  // the declaration is new or changed; undeclared parameters keep their table values
  for (const n of m.params) {
    const d = m.decls.get(n);
    const sig = d ? `${d.min}|${d.max}|${d.step}|${d.value}` : null;
    let p = slot.params[n];
    if (!p) p = slot.params[n] = defaultParam(slot, n, d);
    else if (d && slot.declSig[n] !== sig) {
      p.min = d.min;
      p.max = d.max;
      p.step = d.step;
      if (d.value !== null) p.value = d.value;
      else if (!(p.value >= d.min && p.value <= d.max)) p.value = (d.min + d.max) / 2;
    }
    if (d) slot.declSig[n] = sig;
  }
  // axes: keep the valid ones in place, fill X / Y with the first free parameters (Z optional)
  const ax = slot.axes.map((a) => (a && m.params.includes(a) ? a : null));
  for (let k = 0; k < 3; k++) for (let j = 0; j < k; j++) if (ax[k] && ax[k] === ax[j]) ax[k] = null;
  for (let k = 0; k < 2; k++) {
    if (ax[k]) continue;
    ax[k] = m.params.find((n) => !ax.includes(n)) || null;
  }
  slot.axes = ax;
  return true;
}

// remember numeric helpers (τ = 0.5): when the line is deleted, τ becomes a parameter at 0.5
function harvestConsts(slot) {
  for (const line of slot.text.split('\n')) {
    const m = /^\s*([\p{L}_][\p{L}\p{Nd}\p{M}_₀-₉]*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:#.*)?$/u.exec(line);
    if (m && m[1] !== 'D') slot.consts[m[1]] = Number(m[2]);
  }
}

/**
 * Write the range of parameter `name` into the text: rewrite its declaration line (keeping a
 * trailing comment, the step and an @ value) or insert one before the first non-comment line.
 * step: undefined = keep the declared step, null = none, number = new step.
 */
function writeRange(slot, name, min, max, step) {
  const m = slot.model;
  if (!m) return;
  const d = m.decls.get(name);
  const f = (v) => String(+(+v).toPrecision(6));
  let st = step === undefined ? (d ? d.step : null) : step;
  if (st !== null && !(Math.floor((max - min) / st + 1e-9) >= 1)) st = null;
  let body = `${name} = ${f(min)}:${st !== null ? f(st) + ':' : ''}${f(max)}`;
  const p = slot.params[name];
  const isAxis = slot.axes.includes(name);
  if (d && d.value !== null && !isAxis) body += ` @ ${f(p.value)}`;
  const lines = slot.text.split('\n');
  if (d && lines[d.line - 1] !== undefined) {
    const L = lines[d.line - 1];
    const h = L.indexOf('#');
    const pad = h >= 0 ? Math.max(1, h - body.length) : 0;
    lines[d.line - 1] = h >= 0 ? body + ' '.repeat(pad) + L.slice(h) : body;
  } else {
    let k = 0;
    while (k < lines.length && /^\s*(#.*)?$/.test(lines[k])) k++;
    lines.splice(k, 0, body);
  }
  slot.text = lines.join('\n');
  compileSlot(slot);
  if (slot === cur()) {
    $('eq').value = slot.text;
    fitTextarea();
    updateModified();
  }
  saveOwn();
}

// ---------------------------------------------------------------------------------------------
// jobs
// ---------------------------------------------------------------------------------------------
/**
 * Chart job of a slot -> { code, branch0, exact, c, npow, nx, ny, nz, xr, yr, zr, xl, yl, zl,
 * march, info, warns, auto, counts }. opts: { exact, res ('WxH'), size: { nx, ny, nz } };
 * with the slot's resolution 'auto' and no res / size the sizes are left to autoSize().
 */
function makeJob(slot, opts = {}) {
  const ex = exByKey.get(slot.key);
  const m = slot.model;
  if (!m) throw new UserError(slot.err ? slot.err.message : 'no valid equation');
  const names = m.params;
  if (names.length < 2) throw new UserError(`The chart needs two parameters as its axes; D has ${names.length ? 'only ' + names[0] : 'none'}.`);
  const ix = names.indexOf(slot.axes[0]);
  const iy = names.indexOf(slot.axes[1]);
  const iz = slot.axes[2] ? names.indexOf(slot.axes[2]) : -1;
  if (ix < 0 || iy < 0 || ix === iy) throw new UserError('Select two different parameters as the chart axes (X and Y).');
  const P = names.map((n) => slot.params[n]);
  const vals = P.map((p) => p.value);
  if (!vals.every(Number.isFinite)) throw new UserError('a parameter value is not a number');
  const xr = [P[ix].min, P[ix].max];
  const yr = [P[iy].min, P[iy].max];
  const zr = iz >= 0 ? [P[iz].min, P[iz].max] : null;
  if (!(xr[1] > xr[0]) || !(yr[1] > yr[0]) || (zr && !(zr[1] > zr[0]))) throw new UserError('axis range: max must be larger than min');
  const axisIdx = [ix, iy, iz];
  const env = dict();
  names.forEach((n, i) => { env[n] = axisIdx.includes(i) ? Math.max(Math.abs(P[i].min), Math.abs(P[i].max)) : P[i].value; });
  const S = (k, dflt, check, what) => {
    let v;
    try { v = evalSetting(slot.set[k], env); } catch (e) { throw new UserError(`${what}: ${e.message}`, 'set'); }
    if (v === null) return dflt;
    if (check && !check(v)) throw new UserError(`${what}: ${v} is not admissible`, 'set');
    return v;
  };
  const pos = (v) => v > 0;
  const wmax = S('wmax', 1e5, (v) => v > 0 && Number.isFinite(v), 'ω_max');
  const tol = S('tol', 0.3, (v) => v > 0 && v < 3, 'tolerance');
  const hmax = S('hmax', Infinity, pos, 'h_max');
  const wband = S('wband', Infinity, (v) => v >= 0, 'ω band');
  const w0 = S('w0', 1e-9, (v) => v > 0 && v < wmax, 'ω₀');
  const warns = [...m.warnings];
  let npow, info;
  const nset = S('n', null, Number.isFinite, 'order n');
  // a generic interior point of the axis ranges (the order should not depend on it)
  const pv = vals.slice();
  pv[ix] = xr[0] + 0.382 * (xr[1] - xr[0]);
  pv[iy] = yr[0] + 0.618 * (yr[1] - yr[0]);
  if (iz >= 0) pv[iz] = zr[0] + 0.447 * (zr[1] - zr[0]);
  if (nset !== null) {
    npow = nset;
    info = `n = ${fmtN(npow)} (set in Advanced)`;
  } else if (m.builtin) {
    const r = ex.builtin.prep(vals, wmax);
    npow = r.npow;
    info = `n_eff = ${r.npow.toFixed(4)} (host unwrap, ${r.evals} evaluations, ${r.ms.toFixed(1)} ms)`;
  } else {
    const lo = leadingOrder(m.evalD, pv);
    if (!Number.isFinite(lo.n)) throw new UserError(lo.warn);
    npow = lo.n;
    info = `n = ${fmtN(npow)} (automatic: slope of ln|D(s)|, s ∈ [${fmt(lo.hi / 10)}, ${fmt(lo.hi)}])`;
    if (lo.warn) warns.push(lo.warn);
    for (const [a, b] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const q = pv.slice();
      q[ix] = xr[a];
      q[iy] = yr[b];
      const r = leadingOrder(m.evalD, q);
      if (Number.isFinite(r.n) && Math.abs(r.n - npow) > 0.02) {
        warns.push(`The order depends on the axis parameters (n = ${fmtN(r.n)} at ${names[ix]} = ${fmt(q[ix])}, ${names[iy]} = ${fmt(q[iy])}): the count is valid only where n is the order of D.`);
        break;
      }
    }
    if (!realCoefficients(m.evalD, pv)) {
      warns.push('D(λ̄) ≠ conj D(λ): complex coefficients (e.g. i, or log / sqrt / non-integer powers of a negative parameter). The count over ω ≥ 0 assumes a real-coefficient D.');
    }
  }
  const branch0 = slot.set.branch === 'on' || (slot.set.branch !== 'off' && m.branchAuto);
  const hl = ex.hlines && ex.hlines.param === names[iy] && iz < 0 ? ex.hlines.at : null;
  const cnt = (i) => (i >= 0 && m.decls.get(names[i]) ? m.decls.get(names[i]).count : null);
  let nx = 10, ny = 10, nz = iz >= 0 ? 10 : 1;
  const res = opts.res || (slot.res !== 'auto' ? slot.res : null);
  if (opts.size) ({ nx, ny, nz } = opts.size);
  else if (res) {
    [nx, ny] = res.split('x').map(Number);
    if (iz >= 0) nx = ny = nz = Math.min(256, Math.max(16, Math.round(Math.cbrt(nx * ny))));
  }
  return {
    key: slot.key, code: m.code(ix, iy, iz), branch0, exact: opts.exact ?? state.exact, c: vals, npow,
    nx, ny, nz, xr, yr, zr, xl: names[ix], yl: names[iy], zl: iz >= 0 ? names[iz] : null, hlines: hl, smin: slot.smin,
    march: { wmax, tol, hmax, wband: Number.isFinite(hmax) ? wband : 0, w0 }, info, warns,
    auto: !opts.size && !res, counts: [cnt(ix), cnt(iy), cnt(iz)],
  };
}

function marchInfo(job) {
  const m = job.march;
  const cap = Number.isFinite(m.hmax) ? `, h ≤ ${fmt(m.hmax)}${Number.isFinite(m.wband) ? ` for ω < ${fmt(m.wband)}` : ' on the whole line'}` : '';
  const g = is3D(job) ? `${job.nx}×${job.ny}×${job.nz}` : `${job.nx}×${job.ny}`;
  return `${g}${job.auto ? ' (auto)' : ''} · ${job.info}, ω_max = ${fmt(m.wmax)}, tol = ${fmt(m.tol)}${cap}${job.branch0 ? ', branch point at 0' : ''} · ${job.exact ? 'exact roots' : 'fast (first-order σ)'}`;
}

// ---------------------------------------------------------------------------------------------
// automatic resolution. Frame time model t(N) = lat + c N: a small grid is latency-bound (the
// longest single march), a large one throughput-bound. Probe grids 10², 20², 40², ... (3D: 10³,
// 20³, ...) when the equation, mode or dimension changes: lat = the 10² time, c from the largest
// probe; then c follows the measured frames. The grid gets N = max(TARGET - lat, lat) / c points
// (a frame of ~50 ms, or 2 lat where one march alone takes longer than that).
// ---------------------------------------------------------------------------------------------
const auto = { key: null, skey: null, lat: 0, cost: null, size: null, fromSteps: false, aspect: 16 / 9 };

async function probe(job) {
  const d3 = is3D(job);
  let s = 10;
  let lat = null;
  let cost = null;
  for (;;) {
    const r = await engine.compute({ ...job, nx: s, ny: s, nz: d3 ? s : 1 }, { deadlineMs: DEADLINE_MS });
    if (r.timedOut) return { lat: r.wallMs, cost: r.wallMs / Math.max(1, r.rowsDone * s) };
    if (lat === null) lat = r.wallMs;
    cost = Math.max(r.wallMs - lat, 0.25 * r.wallMs) / r.npts;
    if (r.wallMs > Math.max(12, 3 * lat) || s >= (d3 ? 40 : 160)) break;
    s *= 2;
  }
  return { lat, cost };
}

const budget = () => Math.max(TARGET_MS - auto.lat, auto.lat) / Math.max(auto.cost, 1e-9);

function sizeFor(N, d3, aspect) {
  if (d3) {
    const n = Math.min(128, Math.max(16, Math.round(Math.cbrt(N))));
    return { nx: n, ny: n, nz: n };
  }
  let ny = Math.round(Math.sqrt(N / aspect));
  ny = Math.min(2160, Math.max(20, ny));
  const nx = Math.min(3840, Math.max(20, Math.round(ny * aspect)));
  return { nx, ny, nz: 1 };
}

async function autoSize(job) {
  const d3 = is3D(job);
  const key = `${job.code}|${job.exact}|${job.branch0}|${d3}`;
  const skey = `${job.key}|${job.xl}|${job.yl}|${job.zl}|${job.counts.join()}`;
  if (auto.key !== key) {
    const p = await probe(job);
    auto.key = key;
    auto.lat = p.lat;
    auto.cost = p.cost;
    auto.size = null;
  }
  if (auto.skey !== skey) {
    auto.skey = skey;
    auto.aspect = 16 / 9;
    // a declared step (name = start:step:stop) gives the next frame's grid along that axis
    const [cx, cy, cz] = job.counts;
    if (cx || cy || cz) {
      if (d3) {
        // the stepped axes as declared (at most 256), the others share the frame budget
        const c3 = [cx, cy, cz].map((c) => (c ? Math.min(c, 256) : null));
        const prod = c3.reduce((a, c) => a * (c || 1), 1);
        const k = c3.filter((c) => !c).length;
        const rest = k ? Math.min(128, Math.max(16, Math.round(Math.pow(Math.max(1, budget() / prod), 1 / k)))) : 0;
        auto.size = { nx: c3[0] || rest, ny: c3[1] || rest, nz: c3[2] || rest };
      } else {
        const nx = cx || Math.round(cy * 16 / 9);
        const ny = cy || Math.round(cx * 9 / 16);
        auto.size = { nx: Math.min(3840, Math.max(2, nx)), ny: Math.min(2160, Math.max(2, ny)), nz: 1 };
        auto.aspect = auto.size.nx / auto.size.ny;
      }
      auto.fromSteps = true;
      return auto.size;
    }
  }
  if (auto.fromSteps && auto.size) return auto.size;
  const want = sizeFor(budget(), d3, auto.aspect);
  const n = (s) => s.nx * s.ny * s.nz;
  if (auto.size && (auto.size.nz > 1) === d3 && Math.abs(Math.log(n(want) / n(auto.size))) < Math.log(1.3)) return auto.size;
  return want;
}

function autoUpdate(job, r) {
  if (!job.auto) return;
  if (r.timedOut) {
    const meas = r.wallMs / Math.max(1, r.rowsDone * job.nx);
    auto.cost = Math.max(auto.cost || 0, meas) * 1.5;
    auto.size = null;
    auto.fromSteps = false;
    return;
  }
  const meas = Math.max(r.wallMs - auto.lat, 0.25 * r.wallMs) / r.npts;
  auto.cost = auto.cost ? Math.exp(0.5 * Math.log(auto.cost) + 0.5 * Math.log(meas)) : meas;
  auto.size = { nx: job.nx, ny: job.ny, nz: job.nz };
  auto.fromSteps = false;
}

// ---------------------------------------------------------------------------------------------
// model panel
// ---------------------------------------------------------------------------------------------
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function buildExampleSelect() {
  const sel = $('example');
  for (const g of GROUPS) {
    const og = document.createElement('optgroup');
    og.label = g.label;
    EXAMPLES.forEach((e) => { if (e.group === g.id) og.appendChild(new Option(e.title, e.key)); });
    sel.appendChild(og);
  }
  sel.value = state.key;
  sel.addEventListener('change', () => {
    state.key = sel.value;
    setUrl();
    loadModelUI();
    request();
  });
}

function setUrl() {
  try { history.replaceState(null, '', '?ex=' + state.key + (state.exact ? '' : '&exact=0')); } catch (e) { /* ignore */ }
}

function fitTextarea() {
  const ta = $('eq');
  ta.style.height = 'auto';
  ta.style.height = Math.min(220, ta.scrollHeight + 4) + 'px';
}

function loadModelUI() {
  const ex = curEx();
  const slot = cur();
  $('formula').textContent = ex.formula || '';
  $('formula').style.display = ex.formula ? '' : 'none';
  $('note').textContent = ex.note || '';
  $('note').style.display = ex.note ? '' : 'none';
  $('about').hidden = !ex.formula && !ex.note;
  $('aboutSum').textContent = ex.key === 'own' ? 'How it works' : 'About this example (formula, source, settings)';
  const ta = $('eq');
  ta.value = slot.text;
  ta.readOnly = !!ex.builtin;
  ta.classList.toggle('ro', !!ex.builtin);
  fitTextarea();
  $('insLam').disabled = !!ex.builtin;
  $('res').value = slot.res;
  $('smin').value = slot.smin;
  $('sminLab').textContent = fmt(slot.smin);
  for (const k of SET_INPUTS) $('set_' + k).value = slot.set[k];
  syncMarchSliders();
  updateModified();
  update3dButton();
  buildParamTable();
  renderStatus();
}

// ω_max (log scale 10 .. 1e6) and the phase tolerance 10^x rad (x in [-2, 0])
function settingNumber(slot, k, dflt) {
  try { const v = evalSetting(slot.set[k], dict()); return v === null ? dflt : v; } catch (e) { return dflt; }
}
function syncMarchSliders() {
  const slot = cur();
  const w = settingNumber(slot, 'wmax', 1e5);
  const t = settingNumber(slot, 'tol', 0.3);
  $('wmaxS').value = Math.log10(w);
  $('wmaxO').textContent = fmt(w);
  $('tolS').value = Math.log10(t);
  $('tolO').textContent = fmt(t) + ' rad';
}

function update3dButton() {
  $('stlRow').hidden = !cur().axes[2];
  const ex = curEx();
  const b = $('view3d');
  b.hidden = !ex.axes3d;
  if (ex.axes3d) b.textContent = cur().axes[2] ? '2D view' : `3D view (Z = ${ex.axes3d[2]})`;
}

function updateModified() {
  const ex = curEx();
  const slot = cur();
  const same = slot.text === ex.text;
  $('eqMod').textContent = ex.builtin ? '(built-in, read-only)' : (ex.key === 'own' ? '(saved in this browser)' : (same ? '' : '(modified)'));
}

function renderStatus() {
  const slot = cur();
  const st = $('eqStatus');
  st.innerHTML = '';
  if (slot.err) {
    const e = slot.err;
    const d = el('div', 'err');
    d.appendChild(el('b', '', 'Error: '));
    d.appendChild(document.createTextNode((e.where || '') + e.message + (slot.model ? ' (the chart shows the last valid equation)' : '')));
    st.appendChild(d);
    if (e.line) {
      const src = slot.text.split('\n')[e.line - 1];
      if (src !== undefined) {
        const pre = el('pre', 'caret');
        pre.textContent = src + '\n' + ' '.repeat(Math.max(0, e.col - 1)) + '^';
        st.appendChild(pre);
      }
    }
  } else if (slot.model) {
    const m = slot.model;
    const parts = [`${m.params.length} parameter${m.params.length === 1 ? '' : 's'}`];
    if (m.helpers.length) parts.push(`helper${m.helpers.length > 1 ? 's' : ''} ${m.helpers.join(', ')}`);
    if (m.nodes) parts.push(`${m.nodes} operations after sharing common subexpressions`);
    st.appendChild(el('div', 'ok', parts.join(' · ')));
  }
  if (slot.runErr) st.appendChild(el('div', 'err', slot.runErr));
  for (const w of new Set([...(slot.model ? slot.model.warnings : []), ...slot.jobWarns])) st.appendChild(el('div', 'warn', w));
  $('setStatus').textContent = slot.setErr || '';
}

function buildParamTable() {
  const slot = cur();
  const ex = curEx();
  const m = slot.model;
  const box = $('ptable');
  box.innerHTML = '';
  if (!m) return;
  for (const name of m.params) {
    const p = slot.params[name];
    const axis = slot.axes.indexOf(name);
    const d = m.decls.get(name);
    const row = el('div', 'prow' + (axis >= 0 ? ' axis' : ''));
    const top = el('div', 'ptop');
    const nm = el('b', 'pname', name);
    nm.title = name + (d ? ' (range declared in the text)' : ' (automatic range: declare it as  ' + name + ' = min:max)');
    top.appendChild(nm);
    let val = null;
    if (axis >= 0) top.appendChild(el('span', 'axlab', ['x', 'y', 'z'][axis] + ' axis' + (d && d.count ? ` · ${d.count} pts` : '')));
    else {
      val = el('input', 'pval');
      val.type = 'number';
      val.step = 'any';
      val.value = fmtIn(p.value);
      val.title = 'value';
      top.appendChild(val);
    }
    const axb = el('span', 'axbtns');
    for (const k of [0, 1, 2]) {
      const b = el('button', 'ax' + (axis === k ? ' on' : ''), ['X', 'Y', 'Z'][k]);
      b.title = k < 2 ? `use ${name} as the ${k ? 'y' : 'x'} axis of the chart` :
        (axis === 2 ? 'back to a 2D chart' : `3D (experimental): use ${name} as the third axis`);
      b.disabled = !!ex.builtin;
      b.addEventListener('click', () => setAxis(name, k));
      axb.appendChild(b);
    }
    top.appendChild(axb);
    const bot = el('div', 'pbot');
    const mn = el('input', 'pmin');
    const mx = el('input', 'pmax');
    for (const [inp, v, t] of [[mn, p.min, 'min'], [mx, p.max, 'max']]) {
      inp.type = 'number';
      inp.step = 'any';
      inp.value = fmtIn(v);
      inp.title = t + (d ? ' (writes the range line of the text)' : ' (adds a range line to the text)');
    }
    bot.appendChild(mn);
    let sl = null;
    if (axis >= 0) bot.appendChild(el('span', 'axrange', '◂ chart range ▸'));
    else {
      sl = el('input', 'pslider');
      sl.type = 'range';
      sl.min = p.min;
      sl.max = p.max;
      sl.step = p.step || (p.max - p.min) / 1000 || 'any';
      sl.value = p.value;
      bot.appendChild(sl);
    }
    bot.appendChild(mx);
    row.appendChild(top);
    row.appendChild(bot);
    box.appendChild(row);
    if (sl) {
      sl.addEventListener('input', () => {
        p.value = +sl.value;
        val.value = fmtIn(p.value);
        paramChanged(false);
      });
      sl.addEventListener('change', () => paramChanged(true));
      val.addEventListener('change', () => {
        const v = parseFloat(val.value);
        if (!Number.isFinite(v)) { val.value = fmtIn(p.value); return; }
        p.value = v;
        sl.value = v;
        paramChanged(true);
      });
    }
    const range = (inp, k) => inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      const o = { min: p.min, max: p.max };
      o[k] = v;
      if (!Number.isFinite(v) || !(o.max > o.min)) { inp.value = fmtIn(p[k]); return; }
      p.min = o.min;
      p.max = o.max;
      writeRange(slot, name, o.min, o.max);
      buildParamTable();
      request();
    });
    range(mn, 'min');
    range(mx, 'max');
  }
}

function paramChanged(final) {
  saveOwn();
  if (state.live || final) request();
}

function setAxis(name, k) {
  const slot = cur();
  if (k === 2 && slot.axes[2] === name) slot.axes[2] = null;        // Z off: back to 2D
  else {
    if (slot.axes[k] === name) return;
    const j = slot.axes.indexOf(name);
    if (j >= 0) slot.axes[j] = slot.axes[k];
    slot.axes[k] = name;
    if (!slot.axes[0] || !slot.axes[1]) {                          // keep X and Y filled
      const free = slot.model.params.filter((n) => !slot.axes.includes(n));
      for (const q of [0, 1]) if (!slot.axes[q]) slot.axes[q] = free.shift() || null;
    }
  }
  buildParamTable();
  update3dButton();
  saveOwn();
  request();
}

let eqTimer = null;
function applyText() {
  clearTimeout(eqTimer);
  const slot = cur();
  if (curEx().builtin || slot.text === $('eq').value) return;
  harvestConsts(slot);
  slot.text = $('eq').value;
  slot.jobWarns = [];
  slot.runErr = null;
  const ok = compileSlot(slot);
  updateModified();
  buildParamTable();
  renderStatus();
  saveOwn();
  if (ok) request();
}

function bindModelControls() {
  const ta = $('eq');
  ta.addEventListener('input', () => { fitTextarea(); clearTimeout(eqTimer); eqTimer = setTimeout(applyText, 400); });
  ta.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); applyText(); } });
  $('insLam').addEventListener('click', () => {
    if (ta.readOnly) return;
    ta.setRangeText('λ', ta.selectionStart, ta.selectionEnd, 'end');
    ta.focus();
    ta.dispatchEvent(new Event('input'));
  });
  $('resetEx').addEventListener('click', () => {
    const ex = curEx();
    slots.set(ex.key, freshSlot(ex));
    compileSlot(cur());
    saveOwn();
    loadModelUI();
    request();
  });
  for (const k of SET_INPUTS) {
    $('set_' + k).addEventListener('change', () => {
      const slot = cur();
      slot.set[k] = String($('set_' + k).value).slice(0, 200);
      saveOwn();
      request();
    });
  }
  $('wmaxS').addEventListener('input', () => {
    const v = sig2(Math.pow(10, +$('wmaxS').value));
    cur().set.wmax = String(v);
    $('wmaxO').textContent = fmt(v);
    paramChanged(false);
  });
  $('wmaxS').addEventListener('change', () => paramChanged(true));
  $('tolS').addEventListener('input', () => {
    const v = sig2(Math.pow(10, +$('tolS').value));
    cur().set.tol = String(v);
    $('tolO').textContent = fmt(v) + ' rad';
    paramChanged(false);
  });
  $('tolS').addEventListener('change', () => paramChanged(true));
}

// ---------------------------------------------------------------------------------------------
// own example in localStorage; shareable links (#m=<base64url JSON>)
// ---------------------------------------------------------------------------------------------
function serialize(slot) {
  const p = {};
  const names = slot.model ? slot.model.params : Object.keys(slot.params);
  for (const n of names) {
    const q = slot.params[n];
    if (q) p[n] = [q.min, q.value, q.max];
  }
  const s = {};
  for (const k of Object.keys(SET_DEFAULTS)) if (slot.set[k] !== SET_DEFAULTS[k]) s[k] = slot.set[k];
  return { v: 1, k: slot.key, t: slot.text, p, a: slot.axes, s, f: slot.smin, r: slot.res };
}

function applySerialized(slot, o) {
  if (!o || typeof o !== 'object') return;
  const ex = exByKey.get(slot.key);
  if (typeof o.t === 'string' && !ex.builtin) slot.text = o.t.slice(0, 20000);
  if (Array.isArray(o.a) && !ex.builtin) slot.axes = [0, 1, 2].map((k) => (typeof o.a[k] === 'string' ? o.a[k] : null));
  if (o.s && typeof o.s === 'object') {
    for (const k of Object.keys(SET_DEFAULTS)) slot.set[k] = typeof o.s[k] === 'string' ? o.s[k].slice(0, 200) : SET_DEFAULTS[k];
    if (ex.set) for (const k of Object.keys(ex.set)) if (typeof o.s[k] !== 'string') slot.set[k] = ex.set[k];
  }
  if (Number.isFinite(o.f) && o.f < 0) slot.smin = o.f;
  if (RES_CHOICES.includes(o.r)) slot.res = o.r;
  slot.modelText = null;
  slot.model = null;
  compileSlot(slot);                       // (declared ranges first, then the link's values)
  if (o.p && typeof o.p === 'object') {
    for (const [n, a] of Object.entries(o.p)) {
      if (n.length > 40 || !Array.isArray(a) || a.length !== 3 || !a.every(Number.isFinite) || !(a[2] > a[0])) continue;
      const q = slot.params[n];
      slot.params[n] = { min: a[0], value: a[1], max: a[2], step: q ? q.step : undefined };
    }
  }
}

let saveTimer = null;
function saveOwn() {
  if (state.key !== 'own') return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(OWN_KEY, JSON.stringify(serialize(getSlot('own')))); } catch (e) { /* no storage: fine */ }
  }, 300);
}
function loadOwn() {
  let o = null;
  try { o = JSON.parse(localStorage.getItem(OWN_KEY) || 'null'); } catch (e) { o = null; }
  if (o) applySerialized(getSlot('own'), o);
}

function b64urlEncode(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function b64urlDecode(s) {
  let t = s.replace(/-/g, '+').replace(/_/g, '/');
  while (t.length % 4) t += '=';
  const bin = atob(t);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

function linkFor(slot) {
  const o = serialize(slot);
  o.x = state.exact ? 1 : 0;
  if (state.limit !== 5) o.w = state.limit;
  return location.origin + location.pathname + '?ex=' + slot.key + (state.exact ? '' : '&exact=0') + '#m=' + b64urlEncode(JSON.stringify(o));
}

async function copyLink() {
  applyText();
  const url = linkFor(cur());
  try { history.replaceState(null, '', url); } catch (e) { /* ignore */ }
  let ok = false;
  try { await navigator.clipboard.writeText(url); ok = true; } catch (e) { ok = false; }
  const out = $('linkOut');
  out.value = url;
  out.hidden = false;
  out.select();
  $('linkMsg').textContent = ok ? `Link copied (${url.length} characters).` : 'Copy the link below.';
}

function restoreFromHash() {
  const h = location.hash;
  if (!h.startsWith('#m=')) return;
  try {
    const o = JSON.parse(b64urlDecode(h.slice(3)));
    if (!o || typeof o !== 'object') throw new Error('not an object');
    const key = exByKey.has(o.k) ? o.k : 'own';
    state.key = key;
    if (o.x === 1 || o.x === 0) state.exact = o.x === 1;
    if (Number.isFinite(o.w) && o.w >= 1 && o.w <= 600) state.limit = o.w;
    applySerialized(getSlot(key), o);
  } catch (e) {
    showError('The shared link could not be read (' + (e.message || e) + '); showing the default example.');
  }
}

// ---------------------------------------------------------------------------------------------
// chart controls
// ---------------------------------------------------------------------------------------------
// resizable split between the controls and the chart (desktop); width kept in localStorage
const SPLIT_KEY = 'nyquistgpu.split.v1';
function bindSplit() {
  const main = document.querySelector('main');
  const sp = $('split');
  const clamp = (w) => Math.round(Math.max(240, Math.min(0.7 * window.innerWidth, w)));
  const apply = (w) => { main.style.setProperty('--side', clamp(w) + 'px'); };
  let saved = null;
  try { saved = parseFloat(localStorage.getItem(SPLIT_KEY)); } catch (e) { saved = null; }
  if (Number.isFinite(saved)) apply(saved);
  let drag = null;
  let refit = null;
  const after = () => {
    redraw();
    clearTimeout(refit);
    refit = setTimeout(() => request(), 150);                 // the chart re-fits, then recomputes
  };
  sp.addEventListener('pointerdown', (ev) => {
    drag = { x: ev.clientX, w: document.querySelector('aside').getBoundingClientRect().width };
    sp.classList.add('drag');
    try { sp.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
    ev.preventDefault();
  });
  sp.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    apply(drag.w + ev.clientX - drag.x);
    after();
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    sp.classList.remove('drag');
    try { localStorage.setItem(SPLIT_KEY, String(document.querySelector('aside').getBoundingClientRect().width)); } catch (e) { /* ignore */ }
  };
  sp.addEventListener('pointerup', end);
  sp.addEventListener('pointercancel', end);
  sp.addEventListener('dblclick', () => {
    main.style.removeProperty('--side');
    try { localStorage.removeItem(SPLIT_KEY); } catch (e) { /* ignore */ }
    after();
  });
  window.addEventListener('resize', () => {
    let w = null;
    try { w = parseFloat(localStorage.getItem(SPLIT_KEY)); } catch (e) { w = null; }
    if (Number.isFinite(w)) apply(w);
  });
}

// help overlay: the ? button, #help or ?help=1; Esc or a click outside closes it
function bindHelp() {
  const bg = $('helpBg');
  const open = () => { bg.hidden = false; $('helpBox').focus(); };
  const close = () => {
    bg.hidden = true;
    if (location.hash === '#help') { try { history.replaceState(null, '', location.pathname + location.search); } catch (e) { /* ignore */ } }
  };
  $('helpBtn').addEventListener('click', open);
  $('helpClose').addEventListener('click', close);
  bg.addEventListener('click', (ev) => { if (ev.target === bg) close(); });
  document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && !bg.hidden) close(); });
  window.addEventListener('hashchange', () => { if (location.hash === '#help') open(); });
  if (location.hash === '#help' || params.get('help') === '1') open();
}

function bindChartControls() {
  $('res').addEventListener('change', () => { cur().res = $('res').value; saveOwn(); request(); });
  $('resetAxes').addEventListener('click', resetAxes);
  $('limit').value = state.limit;
  $('limit').addEventListener('change', () => {
    const v = parseFloat($('limit').value);
    if (Number.isFinite(v) && v >= 1 && v <= 600) state.limit = v; else $('limit').value = state.limit;
  });
  $('live').addEventListener('change', () => { state.live = $('live').checked; });
  $('bnd').addEventListener('change', () => { state.boundary = $('bnd').checked; redraw(); });
  $('flags').addEventListener('change', () => { state.flags = $('flags').checked; redraw(); });
  $('smin').addEventListener('change', () => {
    const v = parseFloat($('smin').value);
    const slot = cur();
    if (Number.isFinite(v) && v < 0) { slot.smin = v; $('sminLab').textContent = fmt(v); saveOwn(); if (is3D(lastJob)) request(); else redraw(); } else $('smin').value = slot.smin;
  });
  const a3 = () => { $('alphaO').textContent = state.alpha.toFixed(2); $('surfO').textContent = state.surf.toFixed(2); };
  $('alpha').addEventListener('input', () => { state.alpha = +$('alpha').value; a3(); redraw(); });
  $('surfA').addEventListener('input', () => { state.surf = +$('surfA').value; a3(); redraw(); });
  $('smooth3d').checked = state.smooth;
  $('smooth3d').addEventListener('change', () => {
    state.smooth = $('smooth3d').checked;
    if (state.smooth && is3D(lastJob) && last && !lastMesh) buildMesh(last, [lastJob.nx, lastJob.ny, lastJob.nz], cur().smin);
    if (!state.smooth) meshJob++;
    redraw();
  });
  a3();
  $('view3d').addEventListener('click', () => {
    const slot = cur();
    const ex = curEx();
    if (!ex.axes3d || !slot.model) return;
    if (slot.axes[2]) slot.axes = [slot.axes[0], slot.axes[1], null];
    else if (ex.axes3d.every((n) => slot.model.params.includes(n))) {
      slot.axes = ex.axes3d.slice();
      if (ex.look3d) {
        state.alpha = ex.look3d.alpha;
        state.surf = ex.look3d.surf;
        $('alpha').value = state.alpha;
        $('surfA').value = state.surf;
        a3();
      }
    }
    state.cam = defaultCamera();
    buildParamTable();
    update3dButton();
    saveOwn();
    request();
  });
  $('mode').value = state.exact ? 'exact' : 'fast';
  $('mode').addEventListener('change', () => {
    state.exact = $('mode').value === 'exact';
    setUrl();
    request();
  });
  $('render').addEventListener('click', () => { applyText(); request(); });
  $('bench').addEventListener('click', benchCurrent);
  $('savePng').addEventListener('click', savePng);
  $('copyLink').addEventListener('click', copyLink);
  $('saveStl').addEventListener('click', saveStl);
}

function resetAxes() {
  const slot = cur();
  const ex = curEx();
  let d0;
  try { d0 = rangeDecls(ex.text); } catch (e) { d0 = new Map(); }
  for (const n of slot.axes) {
    const d = n && d0.get(n);
    if (d && slot.params[n]) {
      slot.params[n].min = d.min;
      slot.params[n].max = d.max;
      writeRange(slot, n, d.min, d.max, d.step);
    }
  }
  state.cam = defaultCamera();
  buildParamTable();
  request();
}

// ---------------------------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------------------------
function displayFactor(nx) {
  const css = $('chartclip').getBoundingClientRect().width || 960;
  const px = Math.max(480, css * (window.devicePixelRatio || 1));
  return Math.max(1, Math.ceil(nx / px - 1e-9));
}

function viewOpts(nx) {
  return { f: displayFactor(nx), boundary: state.boundary, flags: state.flags, smin: cur().smin, zcap: ZCAP };
}

let ctx = null;
function draw3d(target = ctx) {
  if (!vol3d || !vol3d.tex) return;
  const cv = target.canvas;
  if (target === ctx) {
    const css = $('chartclip').getBoundingClientRect().width || 960;
    const W = Math.round(Math.min(1600, css * (window.devicePixelRatio || 1)));
    const H = Math.round(W * 9 / 16);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  }
  const F = vol3d.draw(target, state.cam, { fog: state.alpha, surf: state.surf, smooth: state.smooth, bg: BG });
  if (target === ctx) lastFrame = F;
}

function redraw() {
  if (!engine) return;
  if (is3D(lastJob)) draw3d();
  else if (engine.resBuf) engine.draw(ctx, viewOpts(engine.nx));
  drawAxes();
}

let busy = false;
let want = false;
function request() {
  want = true;
  if (!busy && engine) pump();
}
async function pump() {
  busy = true;
  try {
    while (want) {
      want = false;
      const slot = cur();
      let job;
      slot.setErr = null;
      slot.runErr = null;
      try {
        job = makeJob(slot);
      } catch (e) {
        if (!(e instanceof UserError) && !(e instanceof ExprError)) throw e;
        if (e.where === 'set') slot.setErr = e.message; else slot.runErr = e.message;
        slot.jobWarns = [];
        renderStatus();
        continue;
      }
      slot.jobWarns = job.warns;
      $('orderInfo').textContent = job.info;
      $('wgslOut').textContent = job.code.trim();
      try {
        if (job.auto) Object.assign(job, await autoSize(job));
        await renderOnce(job, slot);
      } catch (e) {
        if (!e.charD) throw e;
        slot.runErr = e.message;
        $('wgslOut').textContent = e.charD.trim();
      }
      renderStatus();
    }
  } catch (e) {
    showError('Rendering failed: ' + e.message);
    console.error(e);
  } finally {
    busy = false;
  }
}

function setProgress(frac) {
  $('progress').firstElementChild.style.width = (100 * frac).toFixed(1) + '%';
}

async function renderOnce(job, slot) {
  let rafPending = false;
  const d3 = is3D(job);
  const deadline = job.auto ? DEADLINE_MS : 1000 * state.limit;
  const t0 = performance.now();
  const long = !job.auto && state.limit > 5;
  const r = await engine.compute(job, {
    targetMs: 60,
    deadlineMs: deadline,
    onBand: (row0, frac) => {
      setProgress(frac);
      if (long && performance.now() - t0 > 1000) {
        $('hover').textContent = `computing ${job.nx}×${job.ny}${d3 ? '×' + job.nz : ''}: ${(100 * frac).toFixed(0)} % after ${((performance.now() - t0) / 1000).toFixed(0)} s (time limit ${state.limit} s)`;
      }
      if (!d3 && !rafPending) {
        rafPending = true;
        requestAnimationFrame(() => { rafPending = false; engine.draw(ctx, viewOpts(job.nx)); });
      }
    },
  });
  setProgress(0);
  autoUpdate(job, r);
  if (r.timedOut) {
    slot.runErr = `Stopped after ${(deadline / 1000).toFixed(0)} s (time limit; ${(100 * r.rowsDone / (job.ny * job.nz)).toFixed(0)} % computed): the equation is too expensive or the march does not finish — reduce ω_max or the resolution${job.auto ? ' (the automatic resolution is lowered for the next frame)' : ''}.`;
    if (!d3) { lastJob = job; engine.draw(ctx, viewOpts(job.nx)); drawAxes(); }
    return r;
  }
  if (long) $('hover').textContent = ' ';
  lastJob = job;
  showTiming(r, job);
  $('marchInfo').textContent = marchInfo(job);
  const rb = await engine.readback();
  last = { ...rb, xr: job.xr, yr: job.yr, xl: job.xl, yl: job.yl };
  if (d3) {
    if (!vol3d) vol3d = await Volume3D.create(engine.device, engine.format);
    vol3d.setVolume([job.nx, job.ny, job.nz], volumeBytes(rb, [job.nx, job.ny, job.nz], slot.smin, countOf));
    draw3d();                                   // (the ray-marched surface until the mesh is ready)
    buildMesh(rb, [job.nx, job.ny, job.nz], slot.smin);
  } else engine.draw(ctx, viewOpts(job.nx));
  updatePreview();
  drawAxes();
  showStats(last);
  return r;
}

function showTiming(r, job) {
  $('sGpu').innerHTML = `${r.gpuMs.toFixed(r.gpuMs < 100 ? 1 : 0)} <small>ms${r.tsUsed ? '' : ' (wall)'}${job.exact ? ' · exact' : ' · fast'}</small>`;
  $('sMpts').innerHTML = `${(r.npts / r.gpuMs / 1e3).toFixed(2)} <small>Mpts/s</small>`;
  $('sWall').innerHTML = `${r.wallMs.toFixed(0)} <small>ms</small>`;
  const g = is3D(job) ? `${job.nx}³` : `${job.nx}×${job.ny}`;
  $('sPts').innerHTML = `${g} <small>· ${(r.npts / 1e6).toFixed(2)}M${job.auto ? ' · auto' : ''}</small>`;
}

function showStats(L) {
  const n = L.nx * L.ny;
  let ev = 0, st = 0, fl = 0;
  for (let i = 0; i < n; i++) {
    ev += 1 + L.steps[i];
    if (countOf(L, i) === 0) st++;
    if (L.flags[i] & 15) fl++;
  }
  $('sSteps').innerHTML = `${(ev / n).toFixed(0)}`;
  $('sStable').innerHTML = `${(100 * st / n).toFixed(1)}% <small>· ${(100 * fl / n).toFixed(2)}%</small>`;
}

// ---------------------------------------------------------------------------------------------
// axes, zoom, hover (2D); box edges and orbit camera (3D)
// ---------------------------------------------------------------------------------------------
function niceTicks(a, b, n) {
  const span = b - a;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let t = Math.ceil(a / step - 1e-9) * step; t <= b + 1e-9 * span; t += step) out.push(Math.abs(t) < 1e-12 * span ? 0 : t);
  return out;
}

let zoomRect = null;
function overlay() {
  const wrap = $('plotwrap');
  const ax = $('axes');
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (ax.width !== Math.round(W * dpr) || ax.height !== Math.round(H * dpr)) {
    ax.width = Math.round(W * dpr); ax.height = Math.round(H * dpr);
    ax.style.width = W + 'px'; ax.style.height = H + 'px';
  }
  const g = ax.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const cr = $('chartclip').getBoundingClientRect();
  const wr = wrap.getBoundingClientRect();
  return { g, L: cr.left - wr.left, T: cr.top - wr.top, CW: cr.width, CH: cr.height };
}

function drawAxes() {
  const { g, L, T, CW, CH } = overlay();
  if (!lastJob) return;
  const v = is3D(lastJob) ? lastJob : { ...lastJob, ...curView() };
  const cs = getComputedStyle(document.documentElement);
  const ink = cs.getPropertyValue('--muted').trim() || '#666';
  g.strokeStyle = ink; g.fillStyle = ink;
  g.font = '12px system-ui, sans-serif';
  g.lineWidth = 1;
  if (is3D(v)) { drawBox3D(g, L, T, CW, CH); return; }
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (const t of niceTicks(v.xr[0], v.xr[1], Math.max(3, Math.floor(CW / 90)))) {
    const x = L + (t - v.xr[0]) / (v.xr[1] - v.xr[0]) * CW;
    g.beginPath(); g.moveTo(x, T + CH); g.lineTo(x, T + CH + 5); g.stroke();
    g.fillText(fmt(t), x, T + CH + 7);
  }
  g.fillText(v.xl, L + CW / 2, T + CH + 22);
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const t of niceTicks(v.yr[0], v.yr[1], Math.max(3, Math.floor(CH / 60)))) {
    const y = T + CH - (t - v.yr[0]) / (v.yr[1] - v.yr[0]) * CH;
    g.beginPath(); g.moveTo(L - 5, y); g.lineTo(L, y); g.stroke();
    g.fillText(fmt(t), L - 7, y);
  }
  g.save();
  g.translate(14, T + CH / 2); g.rotate(-Math.PI / 2);
  g.textAlign = 'center'; g.fillText(v.yl, 0, 0);
  g.restore();
  if (v.hlines) {
    g.save();
    g.beginPath(); g.rect(L, T, CW, CH); g.clip();
    g.strokeStyle = 'rgba(255,255,255,0.75)'; g.setLineDash([6, 4]);
    for (const yv of v.hlines) {
      const y = T + CH - (yv - v.yr[0]) / (v.yr[1] - v.yr[0]) * CH;
      g.beginPath(); g.moveTo(L, y); g.lineTo(L + CW, y); g.stroke();
    }
    g.restore();
  }
  if (zoomRect) {
    g.strokeStyle = '#ffffff'; g.setLineDash([4, 3]);
    g.strokeRect(L + zoomRect.x0, T + zoomRect.y0, zoomRect.x1 - zoomRect.x0, zoomRect.y1 - zoomRect.y0);
    g.setLineDash([]);
  }
}

function drawBox3D(g, L, T, CW, CH) {
  const F = lastFrame;
  if (!F) return;
  const v = lastJob;
  const P = (p) => { const q = project(F, p, CW, CH); return q ? [L + q[0], T + q[1]] : null; };
  const h = 0.5;
  g.save();
  g.beginPath(); g.rect(L, T, CW, CH); g.clip();
  g.strokeStyle = 'rgba(205, 210, 225, 0.45)';
  for (const a of [-h, h]) {
    for (const b of [-h, h]) {
      for (const [p, q] of [[[-h, a, b], [h, a, b]], [[a, -h, b], [a, h, b]], [[a, b, -h], [a, b, h]]]) {
        const s = P(p), e = P(q);
        if (s && e) { g.beginPath(); g.moveTo(...s); g.lineTo(...e); g.stroke(); }
      }
    }
  }
  g.fillStyle = 'rgba(230, 233, 240, 0.95)';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  const label = (p0, p1, name, r, dx, dy) => {
    const a = P(p0), b = P(p1), m = P(p0.map((x, i) => (x + p1[i]) / 2));
    if (!a || !b || !m) return;
    g.font = '12px system-ui, sans-serif';
    g.fillText(fmt(r[0]), a[0] + dx, a[1] + dy);
    g.fillText(fmt(r[1]), b[0] + dx, b[1] + dy);
    g.font = '600 13px system-ui, sans-serif';
    g.fillText(name, m[0] + 1.6 * dx, m[1] + 1.4 * dy);
  };
  label([-h, -h, -h], [h, -h, -h], v.xl, v.xr, 0, 14);
  label([h, -h, -h], [h, h, -h], v.yl, v.yr, 10, 12);
  label([-h, -h, -h], [-h, -h, h], v.zl, v.zr, -18, 0);
  g.restore();
}

function chartCoords(ev) {
  const r = $('chartclip').getBoundingClientRect();
  return { x: Math.min(Math.max(ev.clientX - r.left, 0), r.width), y: Math.min(Math.max(ev.clientY - r.top, 0), r.height), w: r.width, h: r.height };
}

// 2D view changes (wheel, middle-drag, box zoom, pinch): the axis ranges change at once, the
// last image is shown transformed (CSS) until the new chart arrives, the chart is recomputed
// continuously (at the automatic resolution), and the range lines of the text are rewritten
// when the interaction pauses (600 ms) -- not on every event, so typing is not disturbed.
function curView() {
  const slot = cur();
  if (lastJob && !is3D(lastJob) && lastJob.key === slot.key && slot.params[lastJob.xl] && slot.params[lastJob.yl] &&
      slot.axes[0] === lastJob.xl && slot.axes[1] === lastJob.yl) {
    const px = slot.params[lastJob.xl], py = slot.params[lastJob.yl];
    return { xr: [px.min, px.max], yr: [py.min, py.max] };
  }
  return lastJob ? { xr: lastJob.xr, yr: lastJob.yr } : null;
}

function updatePreview() {
  const cv = $('chart');
  const v = curView();
  if (!lastJob || is3D(lastJob) || !v) { cv.style.transform = ''; return; }
  const { xr, yr } = lastJob;
  const r = $('chartclip').getBoundingClientRect();
  const W = r.width, H = r.height;
  const sx = (xr[1] - xr[0]) / (v.xr[1] - v.xr[0]);
  const sy = (yr[1] - yr[0]) / (v.yr[1] - v.yr[0]);
  const tx = (xr[0] - v.xr[0]) / (v.xr[1] - v.xr[0]) * W;
  const ty = (v.yr[1] - yr[1]) / (v.yr[1] - v.yr[0]) * H;
  const id = Math.abs(sx - 1) < 1e-9 && Math.abs(sy - 1) < 1e-9 && Math.abs(tx) < 0.01 && Math.abs(ty) < 0.01;
  cv.style.transform = id ? '' : `translate(${tx}px, ${ty}px) scale(${sx}, ${sy})`;
}

function syncRangeInputs() {
  const slot = cur();
  for (const row of document.querySelectorAll('#ptable .prow')) {
    const p = slot.params[row.querySelector('.pname').textContent];
    if (!p) continue;
    row.querySelector('.pmin').value = fmtIn(p.min);
    row.querySelector('.pmax').value = fmtIn(p.max);
  }
}

let commitTimer = null;
function commitView() {
  clearTimeout(commitTimer);
  commitTimer = null;
  const slot = cur();
  if (!slot.model || !lastJob) return;
  for (const name of [lastJob.xl, lastJob.yl]) {
    const p = slot.params[name];
    if (!p) continue;
    const d = slot.model.decls.get(name);
    // round to 1e-4 of the span (the text stays readable; the view moves by < 0.01 %)
    const q = Math.pow(10, Math.floor(Math.log10(p.max - p.min)) - 4);
    const r = (x) => +(Math.round(x / q) * q).toPrecision(12);
    const lo = r(p.min), hi = r(p.max);
    const step = d && d.count ? +((hi - lo) / (d.count - 1)).toPrecision(4) : undefined;   // keep the grid size
    writeRange(slot, name, lo, hi, step);
  }
  buildParamTable();
}

function setView(xr, yr, commitNow = false) {
  const slot = cur();
  if (!lastJob || is3D(lastJob) || lastJob.key !== slot.key) return;
  const px = slot.params[lastJob.xl], py = slot.params[lastJob.yl];
  if (!px || !py || !(xr[1] > xr[0]) || !(yr[1] > yr[0]) || ![...xr, ...yr].every(Number.isFinite)) return;
  px.min = xr[0]; px.max = xr[1];
  py.min = yr[0]; py.max = yr[1];
  syncRangeInputs();
  updatePreview();
  drawAxes();
  request();
  clearTimeout(commitTimer);
  if (commitNow) commitView(); else commitTimer = setTimeout(commitView, 600);
}

function bindPointer() {
  const cv = $('chart');
  const box = $('chartclip');
  let start = null;              // left drag: box zoom (2D) / orbit (3D)
  let pan = null;                // middle drag (2D)
  const touches = new Map();     // active touch pointers
  let pinch = null;
  let raf = false;
  const orbit = () => { if (!raf) { raf = true; requestAnimationFrame(() => { raf = false; draw3d(); drawAxes(); }); } };
  const coords = chartCoords;
  const capture = (ev) => { try { box.setPointerCapture(ev.pointerId); } catch (e) { /* synthetic event */ } };
  box.addEventListener('mousedown', (ev) => { if (ev.button === 1) ev.preventDefault(); });   // no autoscroll
  box.addEventListener('auxclick', (ev) => { if (ev.button === 1) ev.preventDefault(); });
  box.addEventListener('pointerdown', (ev) => {
    const p = coords(ev);
    if (ev.pointerType === 'touch') {
      touches.set(ev.pointerId, p);
      capture(ev);
      if (touches.size === 2 && !is3D(lastJob)) {
        const [a, b] = [...touches.values()];
        const v = curView();
        if (v) pinch = { v, m: [(a.x + b.x) / 2, (a.y + b.y) / 2], d: Math.max(10, Math.hypot(a.x - b.x, a.y - b.y)) };
        start = null;
      } else if (touches.size === 1 && is3D(lastJob)) start = { ...p, cam: { ...state.cam } };
      return;
    }
    if (ev.button === 1) {
      ev.preventDefault();
      const v = curView();
      if (v && !is3D(lastJob)) { pan = { ...p, v }; capture(ev); }
      return;
    }
    if (ev.button !== 0) return;
    start = { ...p, cam: { ...state.cam } };
    capture(ev);
  });
  box.addEventListener('pointermove', (ev) => {
    const p = coords(ev);
    if (ev.pointerType === 'touch' && touches.has(ev.pointerId)) {
      touches.set(ev.pointerId, p);
      if (pinch && touches.size === 2) {
        const [a, b] = [...touches.values()];
        const m = [(a.x + b.x) / 2, (a.y + b.y) / 2];
        const s = pinch.d / Math.max(10, Math.hypot(a.x - b.x, a.y - b.y));
        const { v } = pinch;
        const W = p.w, H = p.h;
        const cx0 = v.xr[0] + pinch.m[0] / W * (v.xr[1] - v.xr[0]);
        const cy0 = v.yr[1] - pinch.m[1] / H * (v.yr[1] - v.yr[0]);
        const wx = (v.xr[1] - v.xr[0]) * s, wy = (v.yr[1] - v.yr[0]) * s;
        const x0 = cx0 - m[0] / W * wx;
        const y1 = cy0 + m[1] / H * wy;
        setView([x0, x0 + wx], [y1 - wy, y1]);
        return;
      }
    }
    if (is3D(lastJob)) {
      if (start) {
        state.cam.yaw = start.cam.yaw - (p.x - start.x) * 0.01;
        state.cam.pitch = Math.max(-1.45, Math.min(1.45, start.cam.pitch + (p.y - start.y) * 0.01));
        orbit();
      }
      return;
    }
    hover(p);
    if (pan) {
      const { v } = pan;
      const dx = (p.x - pan.x) / p.w * (v.xr[1] - v.xr[0]);
      const dy = (p.y - pan.y) / p.h * (v.yr[1] - v.yr[0]);
      setView([v.xr[0] - dx, v.xr[1] - dx], [v.yr[0] + dy, v.yr[1] + dy]);
      return;
    }
    if (start && ev.pointerType !== 'touch') {
      zoomRect = { x0: Math.min(start.x, p.x), y0: Math.min(start.y, p.y), x1: Math.max(start.x, p.x), y1: Math.max(start.y, p.y) };
      drawAxes();
    }
  });
  const end = (ev) => {
    if (ev.pointerType === 'touch') {
      touches.delete(ev.pointerId);
      if (touches.size < 2 && pinch) { pinch = null; commitView(); }
      if (touches.size === 0) start = null;
      return;
    }
    if (pan) { pan = null; commitView(); return; }
    const z = zoomRect;
    const st = start;
    start = null; zoomRect = null;
    if (is3D(lastJob) || !st) return;
    drawAxes();
    const v = curView();
    if (z && z.x1 - z.x0 > 6 && z.y1 - z.y0 > 6 && v) {
      const p = coords(ev);
      const X = (x) => v.xr[0] + x / p.w * (v.xr[1] - v.xr[0]);
      const Y = (y) => v.yr[1] - y / p.h * (v.yr[1] - v.yr[0]);
      setView([X(z.x0), X(z.x1)], [Y(z.y1), Y(z.y0)], true);
    }
  };
  box.addEventListener('pointerup', end);
  box.addEventListener('pointercancel', end);
  box.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    if (is3D(lastJob)) {
      state.cam.dist = Math.max(1.0, Math.min(8, state.cam.dist * Math.exp(ev.deltaY * 0.001)));
      orbit();
      return;
    }
    const v = curView();
    if (!v) return;
    const p = coords(ev);
    const dy = ev.deltaMode === 1 ? ev.deltaY * 33 : ev.deltaMode === 2 ? ev.deltaY * 400 : ev.deltaY;
    const f = Math.exp(Math.max(-1, Math.min(1, dy * 0.0015)));
    const cx = v.xr[0] + p.x / p.w * (v.xr[1] - v.xr[0]);
    const cy = v.yr[1] - p.y / p.h * (v.yr[1] - v.yr[0]);
    setView([cx + (v.xr[0] - cx) * f, cx + (v.xr[1] - cx) * f], [cy + (v.yr[0] - cy) * f, cy + (v.yr[1] - cy) * f]);
  }, { passive: false });
  box.addEventListener('dblclick', () => {
    if (is3D(lastJob)) { state.cam = defaultCamera(); orbit(); } else { clearTimeout(commitTimer); resetAxes(); }
  });
  box.addEventListener('pointerleave', () => {
    $('hover').textContent = is3D(lastJob) ? '3D view: drag to rotate, wheel to zoom, double-click to reset the view.' :
      '2D: wheel zooms at the cursor, middle-drag pans, left-drag zooms to a box, double-click resets; touch: pinch / two-finger drag.';
  });
}

function hover(p) {
  if (!last || is3D(lastJob)) return;
  const { nx, ny, xr, yr } = last;
  const v = curView() || last;
  const xv = v.xr[0] + p.x / p.w * (v.xr[1] - v.xr[0]);
  const yv = v.yr[1] - p.y / p.h * (v.yr[1] - v.yr[0]);
  const ix = Math.round((xv - xr[0]) / (xr[1] - xr[0]) * (nx - 1));
  const iy = Math.round((yv - yr[0]) / (yr[1] - yr[0]) * (ny - 1));
  if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) { $('hover').textContent = ' '; return; }
  const i = ix + iy * nx;
  const x = xr[0] + ix * (xr[1] - xr[0]) / (nx - 1);
  const y = yr[0] + iy * (yr[1] - yr[0]) / (ny - 1);
  const Z = countOf(last, i);
  const sig = (last.flags[i] & 256) ? fmt(last.s[i]) + ((last.flags[i] & 512) ? ' (refined)' : (last.flags[i] & 1024) ? ' (from ω₀)' : '') : '–';
  const f = last.flags[i] & 15;
  const fl = f ? ` · flags ${[1, 2, 4, 8].filter((b) => f & b).map((b) => ({ 1: 'failed', 2: 'sub-resolution', 4: 'residual', 8: 'Z<0' })[b]).join(', ')}` : '';
  $('hover').textContent = `${last.xl} = ${fmt(x)}, ${last.yl} = ${fmt(y)}:  Z = ${Z < 0 && (last.flags[i] & 128) ? 'failed' : Z}` +
    `${Z === 0 ? ` · σ ≈ ${sig}` : ''} · ${1 + last.steps[i]} evaluations${fl}`;
}

// ---------------------------------------------------------------------------------------------
// benchmark, PNG, validation
// ---------------------------------------------------------------------------------------------
const median = (a) => { const s = a.slice().sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

async function timeJob(job, reps) {
  await engine.compute(job);                    // warm-up (pipeline compile, clocks)
  const g = [], w = [];
  let r;
  for (let k = 0; k < reps; k++) {
    r = await engine.compute(job);
    g.push(r.gpuMs); w.push(r.wallMs);
  }
  return { gpu: median(g), wall: median(w), gpuMin: Math.min(...g), npts: r.npts, bands: r.bands, ts: r.tsUsed };
}

async function benchCurrent() {
  if (busy || !lastJob) return;
  busy = true;
  $('bench').disabled = true;
  try {
    const job = lastJob;
    const t = await timeJob(job, 5);
    logMsg(`benchmark ${job.key} ${job.nx}x${job.ny}${is3D(job) ? 'x' + job.nz : ''} (${job.exact ? 'exact' : 'fast'}): GPU median ${t.gpu.toFixed(1)} ms (min ${t.gpuMin.toFixed(1)})${t.ts ? '' : ' [wall]'}, ` +
      `wall ${t.wall.toFixed(1)} ms, ${(t.npts / t.gpu / 1e3).toFixed(2)} Mpts/s, ${t.bands} submissions`);
  } catch (e) {
    logMsg('benchmark failed: ' + e.message);
  } finally {
    busy = false;
    $('bench').disabled = false;
    want = true;
    pump();
  }
}

async function savePng() {
  if (!engine || !engine.resBuf || !lastJob) return;
  const cv = document.createElement('canvas');
  const c2 = cv.getContext('webgpu');
  if (is3D(lastJob)) {
    cv.width = 1600; cv.height = 900;
    draw3d(c2);
  } else engine.draw(c2, { ...viewOpts(engine.nx), f: 1 });
  cv.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${lastJob.key}_${lastJob.xl}_${lastJob.yl}${lastJob.zl ? '_' + lastJob.zl : ''}_${lastJob.nx}x${lastJob.ny}${is3D(lastJob) ? 'x' + lastJob.nz : ''}.png`.replace(/[^\w.-]+/g, '_');
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, 'image/png');
}

// the smooth boundary mesh of the current 3D chart: built in time slices (~12 ms) between frames,
// so the page stays responsive; a newer chart cancels an unfinished build
let meshJob = 0;
let lastMesh = null;
async function buildMesh(rb, dims, smin) {
  const id = ++meshJob;
  lastMesh = null;
  if (!state.smooth) return;
  const t0 = performance.now();
  const gen = boundaryMesh(rb, dims, smin, countOf);
  let r;
  let slice = performance.now();
  for (;;) {
    r = gen.next();
    if (r.done) break;
    if (performance.now() - slice > 12) {
      await new Promise((res) => setTimeout(res, 0));     // (not rAF: it stops in hidden tabs)
      if (id !== meshJob) return;
      slice = performance.now();
    }
  }
  if (id !== meshJob || !vol3d) return;
  vol3d.setMesh(r.value);
  lastMesh = { mesh: r.value, dims, ms: performance.now() - t0 };
  window.__meshMs = lastMesh.ms;
  draw3d();
  drawAxes();
}

function saveStl() {
  if (!last || !is3D(lastJob)) return;
  const j = lastJob;
  const t0 = performance.now();
  const dims = [j.nx, j.ny, j.nz];
  const mesh = lastMesh && lastMesh.dims.join() === dims.join() ? lastMesh.mesh : runMesh(boundaryMesh(last, dims, cur().smin, countOf));
  const { blob, triangles } = meshSTL(mesh, dims, [j.xr, j.yr, j.zr], $('stlUnit').checked);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${j.key}_${j.xl}_${j.yl}_${j.zl}_${j.nx}x${j.ny}x${j.nz}.stl`.replace(/[^\w.-]+/g, '_');
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  logMsg(`STL: ${triangles} triangles (closed at the box faces), ${(blob.size / 1e6).toFixed(1)} MB, ${(performance.now() - t0).toFixed(0)} ms`);
}

function diffCanvas(nx, ny, zw, zr) {
  const cv = document.createElement('canvas');
  cv.width = nx; cv.height = ny;
  const g = cv.getContext('2d');
  const img = g.createImageData(nx, ny);
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const i = ix + iy * nx;
      const o = 4 * (ix + (ny - 1 - iy) * nx);
      let col;
      if (zw[i] === zr[i]) col = zr[i] === 0 ? [205, 210, 220] : (zr[i] > 0 ? [120, 124, 134] : [60, 60, 60]);
      else col = zw[i] > zr[i] ? [230, 40, 40] : [30, 90, 230];
      img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return cv;
}

// Speckle measure of a σ field as it is coloured (σ clamped to [smin, 0]): stable points (with
// σ) whose 8 neighbours are all stable with σ, and whose σ differs from the neighbours' median
// by more than 5 % of the colour range.
function roughCount(r, nx, ny, smin) {
  const thr = 0.05 * Math.abs(smin);
  const cl = (v) => Math.min(0, Math.max(smin, v));
  let n = 0;
  const nb = new Array(8);
  for (let iy = 1; iy < ny - 1; iy++) {
    for (let ix = 1; ix < nx - 1; ix++) {
      const i = ix + iy * nx;
      if (countOf(r, i) !== 0 || !(r.flags[i] & 256)) continue;
      let k = 0;
      let ok = true;
      for (let dy = -1; dy <= 1 && ok; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const j = i + dx + dy * nx;
          if (countOf(r, j) !== 0 || !(r.flags[j] & 256)) { ok = false; break; }
          nb[k++] = cl(r.s[j]);
        }
      }
      if (!ok) continue;
      if (Math.abs(cl(r.s[i]) - median(nb)) > thr) n++;
    }
  }
  return n;
}

/** the job of a reference grid: the example's text and settings with the constants c of the
 * reference (mapped by ex.cmap) and its axis ranges; checks the settings against the reference */
function refJob(e, ex, nx, ny, exact) {
  const slot = freshSlot(ex);
  compileSlot(slot);
  if (!slot.model) throw new Error(`${ex.key}: ${slot.err && slot.err.message}`);
  (ex.cmap || []).forEach((n, i) => {
    if (n === null) return;
    if (!slot.params[n]) throw new Error(`${ex.key}: no parameter ${n}`);
    slot.params[n].value = e.c[i];
  });
  const [px, py] = slot.axes.map((n) => slot.params[n]);
  px.min = e.xr[0]; px.max = e.xr[1];
  py.min = e.yr[0]; py.max = e.yr[1];
  const job = makeJob(slot, { exact, res: `${nx}x${ny}` });
  // settings vs the reference (hmax / wband: null = none / whole line)
  const bad = [];
  const near = (a, b) => (a === b) || Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));
  if (!near(job.march.wmax, e.wmax ?? 1e5)) bad.push(`ω_max ${job.march.wmax} vs ${e.wmax}`);
  const hr = e.kw.hmax === undefined || e.kw.hmax === null ? Infinity : e.kw.hmax;
  if (!near(job.march.hmax, hr)) bad.push(`h_max ${job.march.hmax} vs ${hr}`);
  if (Number.isFinite(hr)) {
    const wb = e.kw.wband === undefined || e.kw.wband === null ? Infinity : e.kw.wband;
    if (!near(job.march.wband, wb)) bad.push(`ω_band ${job.march.wband} vs ${wb}`);
  }
  return { job, bad };
}

async function runValidation() {
  const rep = $('report');
  rep.innerHTML = '<div class="panel"><h2>Validation against the Julia engine</h2><div id="vstat">loading reference ...</div></div>';
  let ref;
  try {
    ref = await (await fetch('validate/ref_counts.json')).json();
  } catch (e) {
    $('vstat').textContent = 'cannot load validate/ref_counts.json: ' + e.message;
    return null;
  }
  const only = params.get('only') ? params.get('only').split(',') : null;
  const rows = [];
  const figs = [];
  for (const [name, e] of Object.entries(ref.examples)) {
    if (only && !only.includes(e.sys)) continue;
    $('vstat').textContent = `computing ${name} (${ref.nx} x ${ref.ny}) ...`;
    const ex = exByKey.get(e.sys);
    if (!ex) continue;
    const { job, bad } = refJob(e, ex, ref.nx, ref.ny, false);
    if (bad.length) logMsg(`${name}: settings differ from the reference: ${bad.join('; ')}`);
    const t = await engine.compute(job);
    const r = await engine.readback();
    const n = ref.nx * ref.ny;
    const z64 = Int32Array.from(e.Z32);
    if (e.Z64d) for (const [i, z] of e.Z64d) z64[i] = z;
    else if (e.Z64) z64.set(e.Z64);
    const zw = new Int32Array(n);
    let d32 = 0, d64 = 0, dflag = 0, nfl = 0;
    const ds = [];
    for (let i = 0; i < n; i++) {
      zw[i] = countOf(r, i);
      if (r.flags[i] & 15) nfl++;
      if (zw[i] !== e.Z32[i]) { d32++; if ((r.flags[i] & 6) || (e.F32[i] & 6)) dflag++; }
      if (zw[i] !== z64[i]) d64++;
      if (!job.branch0 && zw[i] === 0 && e.Z32[i] === 0 && (r.flags[i] & 256) && !(r.flags[i] & 1024) && e.S32[i] !== null && e.S32[i] <= 0.02) ds.push(Math.abs(r.s[i] - e.S32[i]));
    }
    ds.sort((a, b) => a - b);
    const q = (p) => (ds.length ? ds[Math.min(ds.length - 1, Math.floor(p * ds.length))] : NaN);
    const rough0 = roughCount(r, ref.nx, ref.ny, ex.smin);
    // the 'exact root' mode on the same grid: reference dominant roots, speckle, cost
    const tx = await engine.compute({ ...job, exact: true });
    const rx = await engine.readback();
    let nroot = 0, nhit = 0, maxd = 0;
    const miss = [];
    for (const [i, sref, wref] of e.roots || []) {
      // (fractional powers: the engine's estimate from the |D| minimum at the branch point,
      // ω = 0, is no root -- the page does not track it)
      if (job.branch0 && wref === 0) continue;
      nroot++;
      const sw = (rx.flags[i] & 256) ? rx.s[i] : NaN;
      const d = Math.abs(sw - sref);
      if (d <= 1e-3 * Math.max(1, Math.abs(sref))) nhit++;
      else miss.push(`#${i}: ${Number.isFinite(sw) ? sw.toPrecision(5) : '–'} vs ${sref.toPrecision(5)}`);
      if (Number.isFinite(d)) maxd = Math.max(maxd, d);
    }
    const roughX = roughCount(rx, ref.nx, ref.ny, ex.smin);
    let zx = 0, ev0 = 0, evx = 0;
    for (let i = 0; i < n; i++) {
      if (countOf(rx, i) !== zw[i]) zx++;
      ev0 += 1 + r.steps[i];
      evx += 1 + rx.steps[i];
    }
    rows.push({
      name, n, d32, d64, p32: 100 * d32 / n, p64: 100 * d64 / n, dflag, nfl, und: d32 - dflag,
      sigMed: q(0.5), sig99: q(0.99), sigN: ds.length, gpuMs: t.gpuMs, gpuX: tx.gpuMs,
      evals: ev0 / n, evalsX: evx / n, rough0, roughX, nroot, nhit, maxd, miss, zx,
      npowRef: e.npow, npowHost: job.npow, setOk: bad.length === 0, path: ex.builtin ? 'built-in' : 'text',
    });
    if (miss.length) logMsg(`${name}: exact-mode roots off the reference: ${miss.join('; ')}`);
    const fig = document.createElement('figure');
    fig.appendChild(diffCanvas(ref.nx, ref.ny, zw, Int32Array.from(e.Z32)));
    const cap = document.createElement('figcaption');
    cap.textContent = `${name}: ${d32} differing (${d32 - dflag} unflagged)`;
    fig.appendChild(cap);
    figs.push(fig);
  }
  const ok = rows.every((r) => r.und === 0 && r.zx === 0 && r.setOk);
  const fe = (v) => (Number.isFinite(v) ? v.toExponential(1) : '–');
  const tr = rows.map((r) => `<tr><td>${r.name}</td><td>${r.path}</td><td>${r.d32}</td><td class="${r.und === 0 ? 'pass' : 'fail'}">${r.und}</td>` +
    `<td>${r.p32.toFixed(3)}%</td><td>${r.p64.toFixed(3)}%</td><td>${r.nfl}</td><td>${fe(r.sigMed)}</td><td>${fe(r.sig99)}</td>` +
    `<td>${fe(Math.abs(r.npowHost - r.npowRef))}</td><td class="${r.setOk ? 'pass' : 'fail'}">${r.setOk ? 'same' : 'differ'}</td>` +
    `<td class="${r.nhit === r.nroot ? 'pass' : 'fail'}">${r.nhit}/${r.nroot}</td><td>${fe(r.maxd)}</td>` +
    `<td>${r.rough0} → ${r.roughX}</td><td>${r.gpuMs.toFixed(1)} / ${r.gpuX.toFixed(1)}</td><td>${r.evals.toFixed(0)} / ${r.evalsX.toFixed(0)}</td></tr>`).join('');
  $('vstat').innerHTML = `<p>${ref.nx} × ${ref.ny} grid per example; reference: NyquistGPU on ${ref.device}, each example with its own ω<sub>max</sub> and step caps. ` +
    'Every example runs through the page\'s own path: its equation text is parsed and compiled to WGSL (the FEM bar: its built-in charD), the order n is estimated on the host, the march settings are evaluated from their expressions. ' +
    `Target: no differing count Z at unflagged points (flagged = a root on the line or closer to it than the Float32 resolution, ` +
    `e.g. the K = 1 row of the bar charts, where λ = 0 is a root), the same counts in exact mode and the reference's march settings. ` +
    `<b class="${ok ? 'pass' : 'fail'}">${ok ? 'PASS' : 'FAIL'}</b></p>` +
    `<table><thead><tr><th>grid</th><th>D from</th><th>differing Z</th><th>unflagged</th><th>vs Julia F32</th><th>vs Julia F64</th><th>flagged (web)</th>` +
    `<th>|Δσ| median</th><th>|Δσ| 99%</th><th>|Δn|</th><th>settings</th><th>exact roots</th><th>max |Δσ| exact</th><th>speckle default → exact</th>` +
    `<th>GPU ms default / exact</th><th>evals / pt</th></tr></thead><tbody>${tr}</tbody></table>` +
    `<p>|Δσ|: first-order estimate vs the Julia Float32 run (default mode; stable points where the engine's estimate is not spurious, σ ≤ 0.02). |Δn|: the order used here (estimated from ln|D(s)| on the real axis; FEM: host unwrap of the denominator) vs the order of the reference. ` +
    `Exact roots: the 'exact root' mode at 16 stable points per grid against Float64 dominant roots (15 tracked minima, Newton-polished, ` +
    `certified by counting on shifted lines where that is admissible), tolerance 1e-3. Speckle: stable points whose σ (clamped to the colour ` +
    `range [σ floor, 0]) differs from the median of their 8 neighbours by more than 5 % of that range.</p>`;
  const dv = document.createElement('div');
  dv.className = 'diffs';
  figs.forEach((f) => dv.appendChild(f));
  $('vstat').appendChild(dv);
  window.__validation = { ok, rows: rows.map(({ miss, ...r }) => r), device: engine.deviceName };
  logMsg('validation: ' + JSON.stringify(window.__validation));
  return window.__validation;
}

function benchBox(title) {
  const box = document.createElement('div');
  box.className = 'panel';
  box.style.marginTop = '12px';
  box.innerHTML = `<h2>${title}</h2><div class="bstat">running ...</div>`;
  $('report').appendChild(box);
  return box.querySelector('.bstat');
}

async function runBench() {
  const hd = params.get('bench') === 'hd';
  const bs = benchBox(`Benchmark (${hd ? 'full HD' : 'default resolution of each example'}, default parameters)`);
  const rows = [];
  const only = params.get('only') ? params.get('only').split(',') : null;
  for (const ex of EXAMPLES) {
    if (ex.key === 'own' || (only && !only.includes(ex.key))) continue;
    const res = hd ? '1920x1080' : (ex.res || '960x540');
    const out = { key: ex.key, res };
    for (const exact of [false, true]) {
      bs.textContent = `running ${ex.key}${exact ? ' (exact)' : ''} ...`;
      const slot = freshSlot(ex);
      compileSlot(slot);
      const job = makeJob(slot, { exact, res });
      const t = await timeJob(job, 3);
      const rb = await engine.readback();
      let ev = 0;
      for (let i = 0; i < rb.steps.length; i++) ev += 1 + rb.steps[i];
      out[exact ? 'x' : 'n'] = { gpuMs: t.gpu, wallMs: t.wall, mpts: t.npts / t.gpu / 1e3, evals: ev / rb.steps.length, ts: t.ts,
        rough: roughCount(rb, job.nx, job.ny, ex.smin) };
    }
    rows.push(out);
  }
  bs.innerHTML = `<table><thead><tr><th>example</th><th>resolution</th><th>GPU ms</th><th>Mpts/s</th><th>evals / pt</th>` +
    `<th>GPU ms exact</th><th>evals / pt exact</th><th>exact / default</th><th>speckle default → exact</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${r.key}</td><td>${r.res}</td><td>${r.n.gpuMs.toFixed(1)}${r.n.ts ? '' : ' (wall)'}</td><td>${r.n.mpts.toFixed(2)}</td>` +
      `<td>${r.n.evals.toFixed(0)}</td><td>${r.x.gpuMs.toFixed(1)}</td><td>${r.x.evals.toFixed(0)}</td><td>${(r.x.gpuMs / r.n.gpuMs).toFixed(2)}</td><td>${r.n.rough} → ${r.x.rough}</td></tr>`).join('') +
    `</tbody></table><p>Median of 3 after a warm-up. Device: ${engine.deviceName}${engine.hasTS ? ' (timestamp queries)' : ' (no timestamp queries: wall-clock busy time)'}</p>`;
  window.__bench = { rows, device: engine.deviceName, timestamps: engine.hasTS };
  logMsg('bench: ' + JSON.stringify(window.__bench));
  return window.__bench;
}

// generated (text -> WGSL) vs the hand-written charD of the first version, same settings
async function runCompare() {
  const { HANDWRITTEN } = await import('./handwritten.js');
  const hd = params.get('bench') === 'compare-hd';
  const bs = benchBox(`Generated vs hand-written WGSL (${hd ? 'full HD' : 'default resolution'}, default parameters)`);
  const reps = 5;
  const rows = [];
  const only = params.get('only') ? params.get('only').split(',') : null;
  for (const [key, hw] of Object.entries(HANDWRITTEN)) {
    if (only && !only.includes(key)) continue;
    const ex = exByKey.get(key);
    const res = hd ? '1920x1080' : (ex.res || '960x540');
    const row = { key, res };
    for (const exact of [false, true]) {
      bs.textContent = `running ${key}${exact ? ' (exact)' : ''} ...`;
      const slot = freshSlot(ex);
      compileSlot(slot);
      const jg = makeJob(slot, { exact, res });
      const jh = { ...jg, code: hw.wgsl, c: hw.c };
      await engine.compute(jg);
      await engine.compute(jh);
      const tg = [], th = [];
      for (let k = 0; k < reps; k++) {           // interleaved: clock drift hits both alike
        tg.push((await engine.compute(jg)).gpuMs);
        th.push((await engine.compute(jh)).gpuMs);
      }
      const rh = await engine.readback();
      await engine.compute(jg);
      const rg = await engine.readback();
      let dz = 0, ds = 0, de = 0;
      for (let i = 0; i < rg.z.length; i++) {
        if (countOf(rg, i) !== countOf(rh, i)) dz++;
        if (countOf(rg, i) === 0 && countOf(rh, i) === 0 && (rg.flags[i] & 256) && (rh.flags[i] & 256)) ds =Math.max(ds, Math.abs(rg.s[i] - rh.s[i]));
        if (rg.steps[i] !== rh.steps[i]) de++;
      }
      row[exact ? 'x' : 'n'] = { gen: median(tg), hand: median(th), dz, ds, de, npts: rg.z.length };
    }
    rows.push(row);
  }
  const f1 = (v) => v.toFixed(1);
  bs.innerHTML = `<table><thead><tr><th>example</th><th>resolution</th><th>GPU ms generated</th><th>hand-written</th><th>ratio</th>` +
    `<th>exact: generated</th><th>hand-written</th><th>ratio</th><th>differing Z</th><th>points with other step counts</th><th>max |Δσ| (stable points)</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${r.key}</td><td>${r.res}</td><td>${f1(r.n.gen)}</td><td>${f1(r.n.hand)}</td><td>${(r.n.gen / r.n.hand).toFixed(2)}</td>` +
      `<td>${f1(r.x.gen)}</td><td>${f1(r.x.hand)}</td><td>${(r.x.gen / r.x.hand).toFixed(2)}</td><td>${r.n.dz} / ${r.x.dz}</td>` +
      `<td>${(100 * r.n.de / r.n.npts).toFixed(2)}%</td><td>${r.n.ds.toExponential(1)}</td></tr>`).join('') +
    `</tbody></table><p>Median of ${reps} interleaved runs after a warm-up. Device: ${engine.deviceName}</p>`;
  window.__compare = { rows, device: engine.deviceName };
  logMsg('compare: ' + JSON.stringify(window.__compare));
  return window.__compare;
}


// ---------------------------------------------------------------------------------------------
async function main() {
  getSlot('own');
  loadOwn();
  restoreFromHash();
  buildExampleSelect();
  $('example').value = state.key;
  bindModelControls();
  bindChartControls();
  bindPointer();
  bindSplit();
  bindHelp();
  loadModelUI();
  drawAxes();
  window.__app = { state, slots, auto, getSlot, makeJob, compileSlot, linkFor, applyText, writeRange };
  window.addEventListener('resize', () => redraw());
  try {
    engine = await Engine.create(logMsg);
  } catch (e) {
    const why = e instanceof WebGPUUnavailable ? e.message : 'WebGPU initialisation failed: ' + e.message;
    showError(`<b>WebGPU is not available.</b> ${why}<br>Use a current Chrome or Edge (version 113 or newer) on Windows, macOS or ChromeOS; ` +
      `on Linux / Android enable it under chrome://flags (#enable-unsafe-webgpu). Check chrome://gpu for "WebGPU: Hardware accelerated". ` +
      'The equation box still checks your input.');
    $('device').textContent = '';
    for (const id of ['render', 'bench', 'savePng']) $(id).disabled = true;
    return;
  }
  $('device').textContent = `GPU: ${engine.deviceName}${engine.hasTS ? '' : ' (no timestamp queries: wall-clock timing)'}`;
  ctx = $('chart').getContext('webgpu');
  window.__engine = engine;
  busy = true;
  try {
    if (params.has('validate')) await runValidation();
    const b = params.get('bench');
    if (b === 'compare' || b === 'compare-hd') await runCompare();
    else if (params.has('bench')) await runBench();
  } catch (e) {
    showError('Validation / benchmark failed: ' + e.message);
    console.error(e);
  } finally {
    busy = false;
  }
  request();
}

main();
