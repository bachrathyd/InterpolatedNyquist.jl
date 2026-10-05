# Cell 3 of NyquistGPU_Interactive.ipynb (inserted by make_interactive.py): an HTML/JS front end
# that calls Python through google.colab.kernel.invokeFunction. Each request returns its frame
# directly (no background threads -- Colab does not flush widget updates made from them), and
# the page keeps at most one request in flight, always sending the newest state next.
import io, os, json, time, base64, numpy as np
from PIL import Image as PImage
from IPython.display import HTML, JSON, display
from google.colab import output

DISP_FILE, FULL_FILE = '/dev/shm/nyq_disp.rgb', '/dev/shm/nyq_full.rgb'

def _cmd(name, p):
    return name + ' ' + ' '.join(f'{k}={v}' for k, v in p.items())

def _params(q):
    c = ','.join(repr(float(v)) for v in q['c'])
    return dict(ex=q['ex'], fmt=q['fmt'], nx=int(q['nx']), ny=int(q['ny']), x0=float(q['x0']),
                x1=float(q['x1']), y0=float(q['y0']), y1=float(q['y1']), c=c, maxw=int(q['maxw']),
                maxh=int(q['maxh']), smin=float(q['smin']), flags=int(q['flags']), bnd=int(q['bnd']),
                wmax=float(q['wmax']), refine=str(q['refine']), out=DISP_FILE)

def _render(q):
    try:
        r = srv.ask(_cmd('render', _params(q)))
        t0 = time.time()
        a = np.fromfile(DISP_FILE, np.uint8, count=3 * r['dw'] * r['dh']).reshape(r['dh'], r['dw'], 3)
        buf = io.BytesIO()
        # the frame travels base64-encoded through the kernel channel: its size, not the GPU,
        # sets the frame rate -- JPEG is ~5x smaller than PNG for these colour gradients
        if q.get('enc') == 'png':
            PImage.fromarray(a).save(buf, 'PNG', compress_level=1)
        else:
            PImage.fromarray(a).save(buf, 'JPEG', quality=90, subsampling=0)
        r['img'] = f"data:image/{'png' if q.get('enc') == 'png' else 'jpeg'};base64," + base64.b64encode(buf.getvalue()).decode()
        r['t_enc'], r['kb'] = (time.time() - t0) * 1e3, buf.tell() / 1024
        return JSON(r)
    except Exception as err:
        return JSON({'error': str(err)})

def _bench(q):
    try:
        p = _params(q)
        ts = [srv.ask(_cmd('render', p))['t_kernel'] for _ in range(20)]
        return JSON({'med': float(np.median(ts)), 'min': float(min(ts)), 'n': p['nx'] * p['ny']})
    except Exception as err:
        return JSON({'error': str(err)})

def _save(q):
    try:
        p = _params(q)
        srv.ask(_cmd('render', p))
        r = srv.ask(f'save out={FULL_FILE}')
        a = np.fromfile(FULL_FILE, np.uint8, count=3 * r['w'] * r['h']).reshape(r['h'], r['w'], 3)
        fn = f"/content/{p['ex']}_{p['fmt'].replace('+', 'r')}_{r['w']}x{r['h']}.png"
        PImage.fromarray(a).save(fn)
        return JSON({'file': fn, 'mb': os.path.getsize(fn) / 1e6})
    except Exception as err:
        return JSON({'error': str(err)})

output.register_callback('nyq.render', _render)
output.register_callback('nyq.bench', _bench)
output.register_callback('nyq.save', _save)

APP = r'''
<style>
 #nq {font: 13px system-ui, sans-serif; color: inherit}
 #nq .row {display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center; margin: 4px 0}
 #nq label {display: inline-flex; align-items: center; gap: 6px}
 #nq select, #nq input[type=number] {font: inherit; padding: 2px 4px}
 #nq input[type=number] {width: 90px}
 #nq input[type=range] {width: 220px}
 #nq button {font: inherit; padding: 3px 9px; cursor: pointer}
 #nq .val {display: inline-block; min-width: 64px; font-variant-numeric: tabular-nums}
 #nq #st {font: 12px ui-monospace, monospace; margin: 6px 0; min-height: 2.6em}
 #nq img {max-width: 100%; display: block; border: 1px solid #8884}
 #nq #cap {font-size: 12px; opacity: .8; margin-top: 4px}
</style>
<div id="nq">
 <div class="row">
  <label>example <select id="ex"></select></label>
  <label>format <select id="fmt"></select></label>
  <label>resolution <select id="res"></select></label>
  <label>image <select id="enc"><option value="jpeg">JPEG (fast transfer)</option><option value="png">PNG (exact pixels)</option></select></label>
 </div>
 <div class="row">
  <label>rightmost root <select id="ref"></select></label>
  <label>&omega;<sub>max</sub> <input type="range" id="wm" min="0.3" max="5" step="0.01"><span class="val" id="wmv"></span></label>
 </div>
 <div class="row" id="knobs"></div>
 <div class="row">
  <label>&sigma; colour floor <input type="range" id="smin" min="-2" max="-0.005" step="0.005"><span class="val" id="sminv"></span></label>
  <label><input type="checkbox" id="bnd" checked> boundary (white)</label>
  <label><input type="checkbox" id="flg"> show flagged points</label>
 </div>
 <div class="row">
  <label>x min <input type="number" id="x0" step="any"></label><label>x max <input type="number" id="x1" step="any"></label>
  <label>y min <input type="number" id="y0" step="any"></label><label>y max <input type="number" id="y1" step="any"></label>
 </div>
 <div class="row">
  <button id="zin">zoom in &times;2</button><button id="zout">zoom out &times;2</button><button id="rst">reset view</button>
  <button id="pl">&larr;</button><button id="pr">&rarr;</button><button id="pd">&darr;</button><button id="pu">&uarr;</button>
  <button id="bench">benchmark 20 frames</button><button id="save">save full-resolution PNG</button>
 </div>
 <div id="st">starting&hellip;</div>
 <div id="st2" style="font: 12px ui-monospace, monospace"></div>
 <img id="im">
 <div id="cap"></div>
 <div id="note" style="font-size: 12px; opacity: .8"></div>
</div>
<script>
(() => {
 const META = %META%;
 const RES = [['320x180',320,180],['480x270',480,270],['640x360 (nHD)',640,360],['960x540 (qHD)',960,540],['1280x720 (HD)',1280,720],['1920x1080 (full HD)',1920,1080],
              ['2560x1440 (QHD)',2560,1440],['3840x2160 (4K UHD)',3840,2160],['7680x4320 (8K UHD)',7680,4320]];
 const $ = id => document.getElementById(id);
 const EX = {}; META.examples.forEach(e => EX[e.key] = e);
 META.examples.forEach(e => $('ex').add(new Option(e.title, e.key)));
 META.formats.forEach(([k, l]) => $('fmt').add(new Option(l, k)));
 RES.forEach(([l, w, h]) => $('res').add(new Option(l, w + 'x' + h)));
 META.refines.forEach(([k, l]) => $('ref').add(new Option(l, k)));
 const isF16 = () => $('fmt').value.startsWith('F16');
 const wmax = () => Math.pow(10, +$('wm').value);
 function showW() { if (EX[$('ex').value].hill) { $('wmv').innerHTML = 'not used (period strip)'; return; }
   const w = wmax(), cap = isF16() && w > META.w16;
   $('wmv').innerHTML = (cap ? META.w16 : +w.toPrecision(3)).toLocaleString() + (cap ? ' (Float16 cap)' : ''); }
 function defaultW() { $('wm').value = isF16() ? Math.log10(META.w16) : 5; showW(); }
 $('res').value = '1920x1080';
 let knobs = [];
 const fmt = v => Number(v).toPrecision(4);
 function buildKnobs(e) {
  $('knobs').innerHTML = ''; knobs = [];
  e.knobs.forEach(k => {
   const lab = document.createElement('label');
   const r = document.createElement('input');
   Object.assign(r, {type: 'range', min: k.lo, max: k.hi, step: (k.hi - k.lo) / 1000});
   r.value = k.value;
   const v = document.createElement('span'); v.className = 'val'; v.textContent = fmt(k.value);
   r.oninput = () => { v.textContent = fmt(r.value); go(); };
   lab.append(k.name + ' ', r, v); $('knobs').append(lab);
   knobs.push([k.i - 1, r]);
  });
 }
 function setView(x0, x1, y0, y1) { $('x0').value = +x0.toPrecision(8); $('x1').value = +x1.toPrecision(8);
                                    $('y0').value = +y0.toPrecision(8); $('y1').value = +y1.toPrecision(8); }
 function reset() { const e = EX[$('ex').value]; buildKnobs(e); setView(...e.xr, ...e.yr);
                    $('smin').value = e.smin; $('sminv').textContent = e.smin;
                    // time-periodic (Hill) examples: Float32/Float64, Newton polish, no ω_max
                    if (e.hill) { if (isF16() && !e.f16) $('fmt').value = 'F32'; if ($('ref').value === 'count') $('ref').value = 'none'; }
                    if (e.res) $('res').value = e.res;   // heavy examples start at a low resolution
                    $('wm').disabled = !!e.hill; showW(); $('note').textContent = e.note || ''; go(); }
 function params() {
  const e = EX[$('ex').value], c = e.c.slice();
  knobs.forEach(([i, r]) => c[i] = +r.value);
  const [nx, ny] = $('res').value.split('x').map(Number);
  return {ex: e.key, fmt: $('fmt').value, nx, ny, x0: +$('x0').value, x1: +$('x1').value,
          y0: +$('y0').value, y1: +$('y1').value, c, maxw: 1600, maxh: 900, smin: +$('smin').value,
          flags: $('flg').checked ? 1 : 0, bnd: $('bnd').checked ? 1 : 0,
          wmax: isF16() ? Math.min(wmax(), META.w16) : wmax(), refine: $('ref').value, enc: $('enc').value};
 }
 // Two requests in flight: the kernel channel has a fixed latency of tens of ms per call, so
 // the next frame is computed while the previous one travels; only the newest state is
 // queued, and a response older than the frame on screen is dropped.
 const DEPTH = 2;
 let inflight = 0, dirty = false, seq = 0, shown = 0, tShown = 0, fps = 0;
 async function go() {
  if (inflight >= DEPTH) { dirty = true; return; }
  inflight++; dirty = false;
  const id = ++seq, p = params(), t0 = performance.now();
  const slow = setTimeout(() => $('st').innerHTML = '<i>computing&hellip; (the first use of an example / format ' +
                                  'compiles its kernel, ~10-30 s; a new resolution allocates the plan)</i>', 400);
  try {
   const res = await google.colab.kernel.invokeFunction('nyq.render', [p], {});
   const r = res.data['application/json'];
   clearTimeout(slow);
   if (id < shown) {}                                   // overtaken by a newer frame
   else if (r.error) { $('st').innerHTML = '<b style="color:#d33">error:</b> ' + r.error; }
   else {
    $('im').src = r.img;
    const now = performance.now(), rt = now - t0, e = EX[p.ex];
    if (id === shown + 1 && t0 < tShown) {   // sent before the previous frame arrived: continuous updates
     const f = 1e3 / (now - tShown); fps = fps > 0 ? 0.7 * fps + 0.3 * f : f;
    } else fps = 0;
    shown = id; tShown = now;
    const re = p.fmt === 'F16+' ? ` | re-check ${r.n_recheck.toLocaleString()} pts ${r.t_recheck.toFixed(2)} ms` : '';
    const al = r.t_plan > 2 ? ` | plan allocation ${r.t_plan.toFixed(0)} ms` : '';
    $('st').innerHTML = `${p.nx}&times;${p.ny} = ${(r.n / 1e6).toFixed(2)} Mpts | <b>GPU kernel ${r.t_kernel.toFixed(2)} ms</b> ` +
      `(${Math.round(r.mpts).toLocaleString()} Mpts/s)${re} | colour + downsample ${r.t_colour.toFixed(2)} ms | ` +
      `read-back ${r.t_read.toFixed(2)} ms${al} | flagged ${r.flagged_pct.toFixed(3)} %` +
      (r.wmax === null ? ' | one period strip (Hill)' : ` | &omega;<sub>max</sub> = ${(+r.wmax.toPrecision(3)).toLocaleString()}`) + `<br>` +
      `frame round trip ${rt.toFixed(0)} ms: GPU work above, ${p.enc.toUpperCase()} ${r.kb.toFixed(0)} KB encoded in ` +
      `${r.t_enc.toFixed(1)} ms, the rest is the transfer to the browser` +
      (fps > 0 ? ` | <b>display rate ~${fps.toFixed(1)} fps</b> (while moving a slider, ${DEPTH} frames in flight)` : '') +
      `; ${META.device}`;
    $('cap').innerHTML = `<b>${e.title}</b> &mdash; horizontal: ${e.xl} &isin; [${fmt(p.x0)}, ${fmt(p.x1)}], vertical: ` +
      `${e.yl} &isin; [${fmt(p.y0)}, ${fmt(p.y1)}] &mdash; red: number of unstable roots (darker = more), ` +
      `purple to yellow: rightmost root &sigma; in the stable domain (yellow = close to the boundary)`;
   }
  } catch (err) { clearTimeout(slow); $('st').textContent = 'error: ' + err; }
  inflight--;
  if (dirty) go();
 }
 function zoom(f) { const cx = (+$('x0').value + +$('x1').value) / 2, cy = (+$('y0').value + +$('y1').value) / 2;
  const hx = (+$('x1').value - +$('x0').value) / 2 * f, hy = (+$('y1').value - +$('y0').value) / 2 * f;
  setView(cx - hx, cx + hx, cy - hy, cy + hy); go(); }
 function pan(dx, dy) { const sx = (+$('x1').value - +$('x0').value) * dx, sy = (+$('y1').value - +$('y0').value) * dy;
  setView(+$('x0').value + sx, +$('x1').value + sx, +$('y0').value + sy, +$('y1').value + sy); go(); }
 $('ex').onchange = reset; $('res').onchange = go; $('ref').onchange = go; $('enc').onchange = go;
 $('fmt').onchange = () => { const e = EX[$('ex').value]; if (e.hill && isF16() && !e.f16) $('fmt').value = 'F32'; defaultW(); go(); };
 $('wm').oninput = () => { showW(); go(); };
 $('smin').oninput = () => { $('sminv').textContent = $('smin').value; go(); };
 $('bnd').onchange = go; $('flg').onchange = go;
 ['x0', 'x1', 'y0', 'y1'].forEach(id => $(id).onchange = go);
 $('zin').onclick = () => zoom(0.5); $('zout').onclick = () => zoom(2); $('rst').onclick = reset;
 $('pl').onclick = () => pan(-0.25, 0); $('pr').onclick = () => pan(0.25, 0);
 $('pd').onclick = () => pan(0, -0.25); $('pu').onclick = () => pan(0, 0.25);
 async function extra(name, busyText, show) {
  $('st2').innerHTML = '<i>' + busyText + '</i>';
  const r = (await google.colab.kernel.invokeFunction(name, [params()], {})).data['application/json'];
  $('st2').innerHTML = r.error ? 'error: ' + r.error : show(r);
 }
 // whole frames as the page receives them (GPU, encoding, transfer, decoding), `depth` requests in flight
 async function displayRate(n, depth) {
  const p = params(); let sent = 0, got = 0; const t0 = performance.now();
  await new Promise(done => {
   const one = async () => {
    sent++;
    const r = (await google.colab.kernel.invokeFunction('nyq.render', [p], {})).data['application/json'];
    if (!r.error) $('im').src = r.img;
    if (++got === n) done(); else if (sent < n) one();
   };
   for (let i = 0; i < Math.min(depth, n); i++) one();
  });
  return (performance.now() - t0) / n;
 }
 $('bench').onclick = async () => {
  await extra('nyq.bench', 'benchmarking the GPU kernel&hellip;', r =>
   `benchmark: kernel median ${r.med.toFixed(3)} ms, min ${r.min.toFixed(3)} ms over 20 frames ` +
   `(${Math.round(r.n / r.med / 1e3).toLocaleString()} Mpts/s)`);
  const k = $('st2').innerHTML;
  $('st2').innerHTML = k + '<br><i>measuring the display rate&hellip;</i>';
  const d1 = await displayRate(20, 1), d2 = await displayRate(20, DEPTH);
  $('st2').innerHTML = k + `<br>display (whole frames in this browser, 20 each): ${d1.toFixed(0)} ms per frame ` +
   `(${(1e3 / d1).toFixed(1)} fps) one at a time, <b>${d2.toFixed(0)} ms (${(1e3 / d2).toFixed(1)} fps) with ${DEPTH} in flight</b>`;
 };
 $('save').onclick = () => extra('nyq.save', 'saving the full-resolution image&hellip;', r =>
   `saved <tt>${r.file}</tt> (${r.mb.toFixed(1)} MB): Files panel on the left`);
 if (META.default && EX[META.default]) $('ex').value = META.default;   // DEFAULT_EX in Python
 defaultW();
 reset();
 if (META.autobench) setTimeout(() => $('bench').onclick(), 3000);     // AUTOBENCH = True in Python
})();
</script>
'''
display(HTML(APP.replace('%META%', json.dumps(dict(META, default=globals().get('DEFAULT_EX'),
                                                     autobench=bool(globals().get('AUTOBENCH')))))))
