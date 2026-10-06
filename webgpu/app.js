// UI of the WebGPU stability-chart demo: example / constants / resolution controls, a render
// scheduler (the newest settings win, a running chart is finished first), axes, box zoom,
// hover read-out, timing, and the ?validate=1 / ?bench=1 modes.

import { EXAMPLES } from './examples.js';
import { Engine, WebGPUUnavailable, countOf } from './engine.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const ZCAP = 6;

const state = {
  ex: 0,
  c: EXAMPLES.map((e) => e.c.slice()),
  view: EXAMPLES.map((e) => ({ xr: e.xr.slice(), yr: e.yr.slice() })),
  smin: EXAMPLES.map((e) => e.smin),
  res: '960x540',
  wmax: 1e5,
  boundary: true,
  flags: false,
  live: true,
};
let engine = null;
let last = null;           // last read-back chart { nx, ny, xr, yr, z, s, steps, flags }
let lastJob = null;

function logMsg(s) {
  console.log(s);
  const el = $('log');
  el.textContent += s + '\n';
}
function showError(html) {
  const m = $('msg');
  m.innerHTML = html;
  m.style.display = 'block';
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------
const fmt = (v) => {
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(2);
  return (+v.toPrecision(4)).toString();
};

function buildExampleSelect() {
  const sel = $('example');
  EXAMPLES.forEach((e, i) => sel.add(new Option(e.title, i)));
  sel.value = state.ex;
  sel.addEventListener('change', () => { state.ex = +sel.value; buildModelControls(); request(); });
}

function buildModelControls() {
  const ex = EXAMPLES[state.ex];
  const c = state.c[state.ex];
  $('formula').textContent = ex.formula;
  const kn = $('knobs');
  kn.innerHTML = '';
  for (const k of ex.knobs) {
    const lab = document.createElement('label');
    lab.className = 'row';
    const id = 'knob' + k.i;
    lab.innerHTML = `<span><b>${k.name}</b><output id="${id}o"></output></span>` +
      `<input type="range" id="${id}" min="${k.lo}" max="${k.hi}" step="${(k.hi - k.lo) / 400}">`;
    kn.appendChild(lab);
    const inp = lab.querySelector('input');
    const out = lab.querySelector('output');
    inp.value = c[k.i - 1];
    out.textContent = fmt(c[k.i - 1]);
    inp.addEventListener('input', () => {
      c[k.i - 1] = +inp.value;
      out.textContent = fmt(+inp.value);
      syncConsts();
      if (state.live) request();
    });
    inp.addEventListener('change', () => request());
  }
  const cs = $('consts');
  cs.innerHTML = '';
  ex.cnames.forEach((nm, i) => {
    const lab = document.createElement('label');
    lab.className = 'row';
    lab.innerHTML = `<span><b>c${i + 1} = ${nm}</b></span><input type="number" step="any" id="cst${i}">`;
    cs.appendChild(lab);
    const inp = lab.querySelector('input');
    inp.value = c[i];
    inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      if (!Number.isFinite(v)) { inp.value = c[i]; return; }
      c[i] = v;
      syncKnobs();
      request();
    });
  });
  const rst = document.createElement('div');
  rst.className = 'buttons';
  rst.style.gridColumn = '1 / -1';
  rst.innerHTML = '<button>Default constants</button>';
  rst.querySelector('button').addEventListener('click', () => {
    state.c[state.ex] = ex.c.slice();
    buildModelControls();
    request();
  });
  cs.appendChild(rst);
  $('smin').value = state.smin[state.ex];
  $('sminLab').textContent = fmt(state.smin[state.ex]);
  syncAxes();
}

function syncConsts() {
  const c = state.c[state.ex];
  c.forEach((v, i) => { const el = $('cst' + i); if (el) el.value = +v.toPrecision(6); });
}
function syncKnobs() {
  const ex = EXAMPLES[state.ex];
  const c = state.c[state.ex];
  for (const k of ex.knobs) {
    $('knob' + k.i).value = c[k.i - 1];
    $('knob' + k.i + 'o').textContent = fmt(c[k.i - 1]);
  }
}
function syncAxes() {
  const ex = EXAMPLES[state.ex];
  const v = state.view[state.ex];
  $('x0').value = +v.xr[0].toPrecision(6); $('x1').value = +v.xr[1].toPrecision(6);
  $('y0').value = +v.yr[0].toPrecision(6); $('y1').value = +v.yr[1].toPrecision(6);
  $('xlab').textContent = $('xlab2').textContent = ex.xl;
  $('ylab').textContent = $('ylab2').textContent = ex.yl;
}

function bindChartControls() {
  $('res').addEventListener('change', () => { state.res = $('res').value; request(); });
  for (const id of ['x0', 'x1', 'y0', 'y1']) {
    $(id).addEventListener('change', () => {
      const v = state.view[state.ex];
      const a = [+$('x0').value, +$('x1').value, +$('y0').value, +$('y1').value];
      if (a.every(Number.isFinite) && a[1] > a[0] && a[3] > a[2]) {
        v.xr = [a[0], a[1]]; v.yr = [a[2], a[3]];
        request();
      } else syncAxes();
    });
  }
  $('resetAxes').addEventListener('click', resetAxes);
  $('live').addEventListener('change', () => { state.live = $('live').checked; });
  $('bnd').addEventListener('change', () => { state.boundary = $('bnd').checked; redraw(); });
  $('flags').addEventListener('change', () => { state.flags = $('flags').checked; redraw(); });
  $('smin').addEventListener('change', () => {
    const v = parseFloat($('smin').value);
    if (Number.isFinite(v) && v < 0) { state.smin[state.ex] = v; $('sminLab').textContent = fmt(v); redraw(); }
    else $('smin').value = state.smin[state.ex];
  });
  $('wmax').addEventListener('change', () => { state.wmax = +$('wmax').value; request(); });
  $('render').addEventListener('click', () => request());
  $('bench').addEventListener('click', benchCurrent);
  $('savePng').addEventListener('click', savePng);
}

function resetAxes() {
  const ex = EXAMPLES[state.ex];
  state.view[state.ex] = { xr: ex.xr.slice(), yr: ex.yr.slice() };
  syncAxes();
  request();
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------
function currentJob() {
  const ex = EXAMPLES[state.ex];
  const [nx, ny] = state.res.split('x').map(Number);
  const v = state.view[state.ex];
  return {
    ex, c: state.c[state.ex].slice(), npow: ex.npow, nx, ny, xr: v.xr.slice(), yr: v.yr.slice(),
    march: { wmax: state.wmax, ...ex.kw },
  };
}

function displayFactor(nx) {
  const css = $('chart').getBoundingClientRect().width || 960;
  const px = Math.max(480, css * (window.devicePixelRatio || 1));
  return Math.max(1, Math.ceil(nx / px - 1e-9));
}

function viewOpts(nx) {
  return { f: displayFactor(nx), boundary: state.boundary, flags: state.flags, smin: state.smin[state.ex], zcap: ZCAP };
}

let ctx = null;
function redraw() {
  if (!engine || !engine.resBuf) return;
  engine.draw(ctx, viewOpts(engine.nx));
  drawAxes();
}

let busy = false;
let want = false;
function request() {
  want = true;
  if (!busy) pump();
}
async function pump() {
  busy = true;
  try {
    while (want) {
      want = false;
      await renderOnce(currentJob());
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

async function renderOnce(job) {
  let rafPending = false;
  const r = await engine.compute(job, {
    targetMs: 60,
    onBand: (row0, frac) => {
      setProgress(frac);
      if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(() => { rafPending = false; engine.draw(ctx, viewOpts(job.nx)); });
      }
    },
  });
  setProgress(0);
  lastJob = job;
  engine.draw(ctx, viewOpts(job.nx));
  drawAxes();
  showTiming(r);
  const rb = await engine.readback();
  last = { ...rb, xr: job.xr, yr: job.yr, ex: job.ex };
  showStats(last);
  return r;
}

function showTiming(r) {
  $('sGpu').innerHTML = `${r.gpuMs.toFixed(r.gpuMs < 100 ? 1 : 0)} <small>ms${r.tsUsed ? '' : ' (wall)'}</small>`;
  $('sMpts').innerHTML = `${(r.npts / r.gpuMs / 1e3).toFixed(2)} <small>Mpts/s</small>`;
  $('sWall').innerHTML = `${r.wallMs.toFixed(0)} <small>ms</small>`;
  $('sPts').innerHTML = `${(r.npts / 1e6).toFixed(2)}M <small>· ${r.bands}</small>`;
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

// ---------------------------------------------------------------------------
// axes, zoom, hover
// ---------------------------------------------------------------------------
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
function drawAxes() {
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
  const cr = $('chart').getBoundingClientRect();
  const wr = wrap.getBoundingClientRect();
  const L = cr.left - wr.left, T = cr.top - wr.top, CW = cr.width, CH = cr.height;
  const ex = EXAMPLES[state.ex];
  const v = lastJob ? { xr: lastJob.xr, yr: lastJob.yr } : state.view[state.ex];
  const exl = lastJob ? lastJob.ex : ex;
  const cs = getComputedStyle(document.documentElement);
  const ink = cs.getPropertyValue('--muted').trim() || '#666';
  g.strokeStyle = ink; g.fillStyle = ink;
  g.font = '12px system-ui, sans-serif';
  g.lineWidth = 1;
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (const t of niceTicks(v.xr[0], v.xr[1], Math.max(3, Math.floor(CW / 90)))) {
    const x = L + (t - v.xr[0]) / (v.xr[1] - v.xr[0]) * CW;
    g.beginPath(); g.moveTo(x, T + CH); g.lineTo(x, T + CH + 5); g.stroke();
    g.fillText(fmt(t), x, T + CH + 7);
  }
  g.fillText(exl.xl, L + CW / 2, T + CH + 22);
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const t of niceTicks(v.yr[0], v.yr[1], Math.max(3, Math.floor(CH / 60)))) {
    const y = T + CH - (t - v.yr[0]) / (v.yr[1] - v.yr[0]) * CH;
    g.beginPath(); g.moveTo(L - 5, y); g.lineTo(L, y); g.stroke();
    g.fillText(fmt(t), L - 7, y);
  }
  g.save();
  g.translate(14, T + CH / 2); g.rotate(-Math.PI / 2);
  g.textAlign = 'center'; g.fillText(exl.yl, 0, 0);
  g.restore();
  if (zoomRect) {
    g.strokeStyle = '#ffffff'; g.setLineDash([4, 3]);
    g.strokeRect(L + zoomRect.x0, T + zoomRect.y0, zoomRect.x1 - zoomRect.x0, zoomRect.y1 - zoomRect.y0);
    g.setLineDash([]);
  }
}

function chartCoords(ev) {
  const r = $('chart').getBoundingClientRect();
  return { x: Math.min(Math.max(ev.clientX - r.left, 0), r.width), y: Math.min(Math.max(ev.clientY - r.top, 0), r.height), w: r.width, h: r.height };
}

function bindPointer() {
  const cv = $('chart');
  let start = null;
  cv.addEventListener('pointerdown', (ev) => { start = chartCoords(ev); cv.setPointerCapture(ev.pointerId); });
  cv.addEventListener('pointermove', (ev) => {
    const p = chartCoords(ev);
    hover(p);
    if (start) {
      zoomRect = { x0: Math.min(start.x, p.x), y0: Math.min(start.y, p.y), x1: Math.max(start.x, p.x), y1: Math.max(start.y, p.y) };
      drawAxes();
    }
  });
  cv.addEventListener('pointerup', (ev) => {
    const z = zoomRect;
    start = null; zoomRect = null;
    drawAxes();
    if (z && z.x1 - z.x0 > 6 && z.y1 - z.y0 > 6 && lastJob) {
      const p = chartCoords(ev);
      const { xr, yr } = lastJob;
      const X = (x) => xr[0] + x / p.w * (xr[1] - xr[0]);
      const Y = (y) => yr[1] - y / p.h * (yr[1] - yr[0]);
      state.view[state.ex] = { xr: [X(z.x0), X(z.x1)], yr: [Y(z.y1), Y(z.y0)] };
      syncAxes();
      request();
    }
  });
  cv.addEventListener('dblclick', resetAxes);
  cv.addEventListener('pointerleave', () => { $('hover').textContent = ' '; });
}

function hover(p) {
  if (!last) return;
  const { nx, ny, xr, yr } = last;
  const ix = Math.round(p.x / p.w * (nx - 1));
  const iy = Math.round((1 - p.y / p.h) * (ny - 1));
  const i = ix + iy * nx;
  const x = xr[0] + ix * (xr[1] - xr[0]) / (nx - 1);
  const y = yr[0] + iy * (yr[1] - yr[0]) / (ny - 1);
  const Z = countOf(last, i);
  const sig = (last.flags[i] & 256) ? fmt(last.s[i]) : '–';
  const f = last.flags[i] & 15;
  const fl = f ? ` · flags ${[1, 2, 4, 8].filter((b) => f & b).map((b) => ({ 1: 'failed', 2: 'sub-resolution', 4: 'residual', 8: 'Z<0' })[b]).join(', ')}` : '';
  $('hover').textContent = `${last.ex.xl} = ${fmt(x)}, ${last.ex.yl} = ${fmt(y)}:  Z = ${Z < 0 && (last.flags[i] & 128) ? 'failed' : Z}` +
    `${Z === 0 ? ` · σ ≈ ${sig}` : ''} · ${1 + last.steps[i]} evaluations${fl}`;
}

// ---------------------------------------------------------------------------
// benchmark, PNG, validation
// ---------------------------------------------------------------------------
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
  if (busy) return;
  busy = true;
  $('bench').disabled = true;
  try {
    const job = currentJob();
    const t = await timeJob(job, 5);
    lastJob = job;
    redraw();
    logMsg(`benchmark ${job.ex.key} ${job.nx}x${job.ny}: GPU median ${t.gpu.toFixed(1)} ms (min ${t.gpuMin.toFixed(1)})${t.ts ? '' : ' [wall]'}, ` +
      `wall ${t.wall.toFixed(1)} ms, ${(t.npts / t.gpu / 1e3).toFixed(2)} Mpts/s, ${t.bands} submissions`);
  } finally {
    busy = false;
    $('bench').disabled = false;
    if (want) pump();
  }
}

async function savePng() {
  if (!engine || !engine.resBuf || !lastJob) return;
  const cv = document.createElement('canvas');
  const c2 = cv.getContext('webgpu');
  engine.draw(c2, { ...viewOpts(engine.nx), f: 1 });
  cv.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${lastJob.ex.key}_${lastJob.nx}x${lastJob.ny}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, 'image/png');
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
  const rows = [];
  const figs = [];
  for (const [name, e] of Object.entries(ref.examples)) {
    $('vstat').textContent = `computing ${name} (${ref.nx} x ${ref.ny}) ...`;
    const ex = EXAMPLES.find((x) => x.key === e.sys);
    const kw = {};
    if (e.kw.hmax !== undefined) kw.hmax = e.kw.hmax;
    if (e.kw['ωband'] !== undefined) kw.wband = e.kw['ωband'];
    const job = { ex, c: e.c, npow: e.npow, nx: ref.nx, ny: ref.ny, xr: e.xr, yr: e.yr, march: { wmax: ref.wmax, ...kw } };
    const t = await engine.compute(job);
    const r = await engine.readback();
    const n = ref.nx * ref.ny;
    const zw = new Int32Array(n);
    let d32 = 0, d64 = 0, dflag = 0, nfl = 0;
    const ds = [];
    for (let i = 0; i < n; i++) {
      zw[i] = countOf(r, i);
      if (r.flags[i] & 15) nfl++;
      if (zw[i] !== e.Z32[i]) { d32++; if ((r.flags[i] & 15) || e.F32[i]) dflag++; }
      if (zw[i] !== e.Z64[i]) d64++;
      if (zw[i] === 0 && e.Z32[i] === 0 && (r.flags[i] & 256) && e.S32[i] !== null) ds.push(Math.abs(r.s[i] - e.S32[i]));
    }
    ds.sort((a, b) => a - b);
    const q = (p) => ds.length ? ds[Math.min(ds.length - 1, Math.floor(p * ds.length))] : NaN;
    const row = {
      name, n, d32, d64, p32: 100 * d32 / n, p64: 100 * d64 / n, dflag, nfl,
      sigMed: q(0.5), sig99: q(0.99), sigN: ds.length, gpuMs: t.gpuMs,
    };
    rows.push(row);
    const fig = document.createElement('figure');
    fig.appendChild(diffCanvas(ref.nx, ref.ny, zw, Int32Array.from(e.Z32)));
    const cap = document.createElement('figcaption');
    cap.textContent = `${name}: ${d32} differing (red: WebGPU higher, blue: lower)`;
    fig.appendChild(cap);
    figs.push(fig);
  }
  const ok = rows.every((r) => r.p32 < 0.5);
  const tr = rows.map((r) => `<tr><td>${r.name}</td><td>${r.d32}</td><td class="${r.p32 < 0.5 ? 'pass' : 'fail'}">${r.p32.toFixed(3)}%</td>` +
    `<td>${r.p64.toFixed(3)}%</td><td>${r.dflag}</td><td>${r.nfl}</td><td>${r.sigMed.toExponential(1)}</td><td>${r.sig99.toExponential(1)}</td><td>${r.gpuMs.toFixed(1)}</td></tr>`).join('');
  $('vstat').innerHTML = `<p>${ref.nx} × ${ref.ny} grid per example; reference: NyquistGPU on ${ref.device}, ω<sub>max</sub> = ${ref.wmax}. ` +
    `Target: &lt; 0.5% differing counts. <b class="${ok ? 'pass' : 'fail'}">${ok ? 'PASS' : 'FAIL'}</b></p>` +
    `<table><thead><tr><th>example</th><th>differing Z</th><th>vs Julia F32</th><th>vs Julia F64</th><th>of them flagged</th><th>flagged (web)</th>` +
    `<th>|Δσ| median</th><th>|Δσ| 99%</th><th>GPU ms</th></tr></thead><tbody>${tr}</tbody></table>`;
  const dv = document.createElement('div');
  dv.className = 'diffs';
  figs.forEach((f) => dv.appendChild(f));
  $('vstat').appendChild(dv);
  window.__validation = { ok, rows, device: engine.deviceName };
  logMsg('validation: ' + JSON.stringify(window.__validation));
  return window.__validation;
}

async function runBench() {
  const rep = $('report');
  const box = document.createElement('div');
  box.className = 'panel';
  box.style.marginTop = '12px';
  box.innerHTML = '<h2>Full-HD benchmark (default constants, ω<sub>max</sub> = 1e5)</h2><div id="bstat">running ...</div>';
  rep.appendChild(box);
  const rows = [];
  for (const ex of EXAMPLES) {
    $('bstat').textContent = `running ${ex.key} ...`;
    const job = { ex, c: ex.c.slice(), npow: ex.npow, nx: 1920, ny: 1080, xr: ex.xr, yr: ex.yr, march: { wmax: 1e5, ...ex.kw } };
    const t = await timeJob(job, 3);
    const rb = await engine.readback();
    let ev = 0;
    for (let i = 0; i < rb.steps.length; i++) ev += 1 + rb.steps[i];
    rows.push({ key: ex.key, gpuMs: t.gpu, gpuMin: t.gpuMin, wallMs: t.wall, mpts: t.npts / t.gpu / 1e3, bands: t.bands, evals: ev / rb.steps.length, ts: t.ts });
  }
  $('bstat').innerHTML = `<table><thead><tr><th>example</th><th>GPU ms (median of 3)</th><th>min</th><th>wall ms</th><th>Mpts/s</th><th>evaluations / pt</th><th>submissions</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${r.key}</td><td>${r.gpuMs.toFixed(1)}${r.ts ? '' : ' (wall)'}</td><td>${r.gpuMin.toFixed(1)}</td><td>${r.wallMs.toFixed(1)}</td><td>${r.mpts.toFixed(2)}</td><td>${r.evals.toFixed(0)}</td><td>${r.bands}</td></tr>`).join('') +
    `</tbody></table><p>Device: ${engine.deviceName}${engine.hasTS ? ' (timestamp queries)' : ' (no timestamp queries: wall-clock busy time)'}</p>`;
  window.__bench = { rows, device: engine.deviceName, timestamps: engine.hasTS };
  logMsg('bench: ' + JSON.stringify(window.__bench));
  return window.__bench;
}

// ---------------------------------------------------------------------------
async function main() {
  buildExampleSelect();
  buildModelControls();
  bindChartControls();
  bindPointer();
  drawAxes();
  window.addEventListener('resize', () => redraw());
  try {
    engine = await Engine.create(logMsg);
  } catch (e) {
    const why = e instanceof WebGPUUnavailable ? e.message : 'WebGPU initialisation failed: ' + e.message;
    showError(`<b>WebGPU is not available.</b> ${why}<br>Use a current Chrome or Edge (version 113 or newer) on Windows, macOS or ChromeOS; ` +
      `on Linux / Android enable it under chrome://flags (#enable-unsafe-webgpu). Check chrome://gpu for "WebGPU: Hardware accelerated".`);
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
    if (params.has('bench')) await runBench();
  } catch (e) {
    showError('Validation / benchmark failed: ' + e.message);
    console.error(e);
  } finally {
    busy = false;
  }
  request();
}

main();
