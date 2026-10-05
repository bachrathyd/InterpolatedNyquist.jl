# Cell 3 of NyquistGPU_Interactive.ipynb (inserted by make_interactive.py): the HTML/JS front end of
# the GPU server. Two transports for the frames:
#  * 'stream' (default): a small HTTP server in this kernel, shown through Colab's port proxy. The page
#    posts every new state; one long-lived response streams the frames back, and the server always
#    renders the newest state as soon as the previous frame is out. The frame rate is then set by the
#    GPU + encoding, not by the ~50 ms round trip of a request.
#  * 'kernel': google.colab.kernel.invokeFunction, one request per frame, two in flight (fallback:
#    set TRANSPORT = 'kernel' before running this cell if the proxied frame stays empty).
import io, os, json, time, base64, struct, threading, numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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

def _frame(q):
    "Render the page state q on the GPU server and encode the display image -> (info, bytes)."
    t0 = time.time()
    r = srv.ask(_cmd('render', _params(q)))
    t1 = time.time()
    a = np.fromfile(DISP_FILE, np.uint8, count=3 * r['dw'] * r['dh']).reshape(r['dh'], r['dw'], 3)
    buf = io.BytesIO()
    png = q.get('enc') == 'png'            # JPEG is ~5x smaller than PNG for these colour gradients
    if png:
        PImage.fromarray(a).save(buf, 'PNG', compress_level=1)
    else:
        PImage.fromarray(a).save(buf, 'JPEG', quality=90, subsampling=0)
    r['t_enc'], r['kb'] = (time.time() - t1) * 1e3, buf.tell() / 1024
    r['mime'] = 'image/png' if png else 'image/jpeg'
    r['t_srv'] = (time.time() - t0) * 1e3
    return r, buf.getvalue()

def _bench_d(q):
    p = _params(q)
    ts = [srv.ask(_cmd('render', p))['t_kernel'] for _ in range(20)]
    return {'med': float(np.median(ts)), 'min': float(min(ts)), 'n': p['nx'] * p['ny']}

def _save_d(q):
    p = _params(q)
    srv.ask(_cmd('render', p))
    r = srv.ask(f'save out={FULL_FILE}')
    a = np.fromfile(FULL_FILE, np.uint8, count=3 * r['w'] * r['h']).reshape(r['h'], r['w'], 3)
    fn = f"/content/{p['ex']}_{p['fmt'].replace('+', 'r')}_{r['w']}x{r['h']}.png"
    PImage.fromarray(a).save(fn)
    return {'file': fn, 'mb': os.path.getsize(fn) / 1e6}

def _guard(f):
    def g(q):
        try:
            return JSON(f(q))
        except Exception as err:
            return JSON({'error': str(err)})
    return g

def _render_kernel(q):
    r, img = _frame(q)
    r['img'] = f"data:{r['mime']};base64," + base64.b64encode(img).decode()
    return r

output.register_callback('nyq.render', _guard(_render_kernel))
output.register_callback('nyq.bench', _guard(_bench_d))
output.register_callback('nyq.save', _guard(_save_d))

# --- stream transport ----------------------------------------------------------------------------
class _Stream:
    "The newest state posted by the page, and benchmark bursts (one stream at a time)."
    def __init__(self):
        self.cv = threading.Condition()
        self.q, self.seq, self.burst, self.gen = None, 0, [], 0

_ST = _Stream()

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ct='application/json'):
        self.send_response(200)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0].strip('/')
        if path == '':
            self._send(_PAGE.encode(), 'text/html; charset=utf-8')
        elif path == 'stream':
            self._stream()
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split('?')[0].strip('/')
        q = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        if path == 'state':                       # newest state wins (posts may overtake each other)
            with _ST.cv:
                if q.get('seq', 0) > _ST.seq:
                    _ST.q, _ST.seq = q, q['seq']
                    _ST.cv.notify_all()
            self._send(b'{}')
        elif path == 'burst':                     # benchmark: the same state n times, back to back
            with _ST.cv:
                _ST.burst = [q] * int(q.get('n', 30))
                _ST.cv.notify_all()
            self._send(b'{}')
        elif path in ('bench', 'save'):
            try:
                r = (_bench_d if path == 'bench' else _save_d)(q)
            except Exception as err:
                r = {'error': str(err)}
            self._send(json.dumps(r).encode())
        else:
            self.send_error(404)

    def _stream(self):
        # frames: [header length][image length] (big-endian uint32), header JSON, image bytes.
        # The stream keeps ITS state object: after the cell is re-run (a new _ST), a stream of the
        # old page must not take the new page's frames.
        st = _ST
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
        with st.cv:
            st.gen += 1
            gen = st.gen
            st.cv.notify_all()                    # an older stream (reloaded page) stops
        done = 0
        try:
            while True:
                with st.cv:
                    st.cv.wait_for(lambda: st.gen != gen or st.seq > done or st.burst, timeout=10)
                    if st.gen != gen or st is not _ST:
                        return
                    k = 0
                    if st.burst:
                        q = st.burst.pop()
                        k = len(st.burst) + 1     # countdown to 1
                    elif st.seq > done:
                        q, done = st.q, st.seq
                    else:
                        q = None
                if q is None:                     # keep-alive for the proxy
                    self.wfile.write(struct.pack('>II', 2, 0) + b'{}')
                    self.wfile.flush()
                    continue
                try:
                    info, img = _frame(q)
                except Exception as err:
                    info, img = {'error': str(err)}, b''
                info.update(seq=q.get('seq', 0), burst=k, q=q)
                h = json.dumps(info).encode()
                self.wfile.write(struct.pack('>II', len(h), len(img)) + h + img)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

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
 #nq canvas {max-width: 100%; height: auto; display: block; border: 1px solid #8884}
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
  <button id="bench">benchmark</button><button id="save">save full-resolution PNG</button>
 </div>
 <div id="st">starting&hellip;</div>
 <div id="st2" style="font: 12px ui-monospace, monospace"></div>
 <canvas id="cv" width="960" height="540"></canvas>
 <div id="cap"></div>
 <div id="note" style="font-size: 12px; opacity: .8"></div>
</div>
<script>
(() => {
 const META = %META%;
 const STREAM = !!META.stream;
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
 // --- shared: drawing and the status line ---
 const cv = $('cv'), ctx = cv.getContext('2d');
 async function draw(blob) {                     // decoded off the main thread, drawn at once
  const b = await createImageBitmap(blob);
  if (cv.width !== b.width || cv.height !== b.height) { cv.width = b.width; cv.height = b.height; }
  ctx.drawImage(b, 0, 0); b.close();
 }
 let fps = 0, tLast = 0, slow = null;
 function rate(now, continuous) {                // display rate while the picture keeps changing
  if (continuous && tLast > 0) { const f = 1e3 / (now - tLast); fps = fps > 0 ? 0.7 * fps + 0.3 * f : f; }
  else fps = 0;
  tLast = now;
 }
 function armSlow() { clearTimeout(slow); slow = setTimeout(() => $('st').innerHTML = '<i>computing&hellip; (the first use of an ' +
   'example / format compiles its kernel, ~10-30 s; a new resolution allocates the plan)</i>', 400); }
 function status(r, p, line) {
  const e = EX[p.ex];
  const re = p.fmt === 'F16+' ? ` | re-check ${r.n_recheck.toLocaleString()} pts ${r.t_recheck.toFixed(2)} ms` : '';
  const al = r.t_plan > 2 ? ` | plan allocation ${r.t_plan.toFixed(0)} ms` : '';
  $('st').innerHTML = `${p.nx}&times;${p.ny} = ${(r.n / 1e6).toFixed(2)} Mpts | <b>GPU kernel ${r.t_kernel.toFixed(2)} ms</b> ` +
    `(${Math.round(r.mpts).toLocaleString()} Mpts/s)${re} | colour + downsample ${r.t_colour.toFixed(2)} ms | ` +
    `read-back ${r.t_read.toFixed(2)} ms${al} | flagged ${r.flagged_pct.toFixed(3)} %` +
    (r.wmax === null ? ' | one period strip (Hill)' : ` | &omega;<sub>max</sub> = ${(+r.wmax.toPrecision(3)).toLocaleString()}`) +
    `<br>${p.enc.toUpperCase()} ${r.kb.toFixed(0)} KB encoded in ${r.t_enc.toFixed(1)} ms, server ${r.t_srv.toFixed(0)} ms per frame | ` +
    line + (fps > 0 ? ` | <b>display rate ~${fps.toFixed(1)} fps</b>` : '') + `; ${META.device}`;
  $('cap').innerHTML = `<b>${e.title}</b> &mdash; horizontal: ${e.xl} &isin; [${fmt(p.x0)}, ${fmt(p.x1)}], vertical: ` +
    `${e.yl} &isin; [${fmt(p.y0)}, ${fmt(p.y1)}] &mdash; red: number of unstable roots (darker = more), ` +
    `purple to yellow: rightmost root &sigma; in the stable domain (yellow = close to the boundary)`;
 }
 async function call(name, p) {                  // one-shot requests: benchmark, save
  if (STREAM) return await (await fetch(name, {method: 'POST', body: JSON.stringify(p)})).json();
  return (await google.colab.kernel.invokeFunction('nyq.' + name, [p], {})).data['application/json'];
 }
 // --- transport 'kernel': one invokeFunction per frame, two in flight (the next frame is computed
 //     while the previous one travels; a response older than the frame on screen is dropped) ---
 const DEPTH = 2;
 let inflight = 0, dirty = false, seq = 0, shown = 0;
 function b64blob(d) {                           // data URL -> Blob (no fetch of data: URLs in the output frame)
  const i = d.indexOf(','), bin = atob(d.slice(i + 1)), u = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) u[k] = bin.charCodeAt(k);
  return new Blob([u], {type: d.slice(5, d.indexOf(';'))});
 }
 async function goKernel() {
  if (inflight >= DEPTH) { dirty = true; return; }
  inflight++; dirty = false;
  const id = ++seq, p = params(), t0 = performance.now();
  armSlow();
  try {
   const r = (await google.colab.kernel.invokeFunction('nyq.render', [p], {})).data['application/json'];
   clearTimeout(slow);
   if (id < shown) {}
   else if (r.error) { $('st').innerHTML = '<b style="color:#d33">error:</b> ' + r.error; }
   else {
    await draw(b64blob(r.img));
    const now = performance.now();
    rate(now, id === shown + 1 && t0 < tLast);   // sent before the previous frame arrived: continuous
    shown = id;
    status(r, p, `frame round trip ${(now - t0).toFixed(0)} ms (kernel channel, ${DEPTH} requests in flight)`);
   }
  } catch (err) { clearTimeout(slow); $('st').textContent = 'error: ' + err; }
  inflight--;
  if (dirty) goKernel();
 }
 async function kernelRate(n, depth) {          // whole frames through the kernel channel
  const p = params(); let sent = 0, got = 0; const t0 = performance.now();
  await new Promise(done => {
   const one = async () => {
    sent++;
    const r = (await google.colab.kernel.invokeFunction('nyq.render', [p], {})).data['application/json'];
    if (!r.error) await draw(b64blob(r.img));
    if (++got === n) done(); else if (sent < n) one();
   };
   for (let i = 0; i < Math.min(depth, n); i++) one();
  });
  return (performance.now() - t0) / n;
 }
 // --- transport 'stream': post the state, frames come back on one long-lived response ---
 let sseq = 0, posting = 0, pdirty = false, burstT = [], burstDone = null;
 const sentAt = new Map();
 function goStream() {
  if (posting >= 4) { pdirty = true; return; }
  const p = params(); p.seq = ++sseq; sentAt.set(p.seq, performance.now());
  posting++; armSlow();
  fetch('state', {method: 'POST', body: JSON.stringify(p)}).catch(() => {})
   .finally(() => { posting--; if (pdirty) { pdirty = false; goStream(); } });
 }
 async function onStreamFrame(r, img) {
  clearTimeout(slow);
  if (r.error) { $('st').innerHTML = '<b style="color:#d33">error:</b> ' + r.error; return; }
  await draw(new Blob([img], {type: r.mime}));
  const now = performance.now();
  if (r.burst) {                                // benchmark frames, counting down to 1
   burstT.push(now);
   if (r.burst === 1 && burstDone) burstDone();
   return;
  }
  const t0 = sentAt.get(r.seq);
  let newer = false;
  for (const k of [...sentAt.keys()]) { if (k <= r.seq) sentAt.delete(k); else newer = true; }
  rate(now, newer);                            // more states pending: the server renders back to back
  status(r, r.q, `latency ${t0 ? (now - t0).toFixed(0) : '?'} ms from the state to the picture (stream)`);
 }
 function burst(n) {                            // n frames of the same state, back to back
  return new Promise(res => {
   burstT = [];
   const fin = () => { if (!burstDone) return; burstDone = null; const t = burstT;
                       res(t.length > 1 ? (t[t.length - 1] - t[0]) / (t.length - 1) : NaN); };
   burstDone = fin; setTimeout(fin, 15000);
   const p = params(); p.n = n; p.seq = sseq;
   fetch('burst', {method: 'POST', body: JSON.stringify(p)});
  });
 }
 async function readStream() {
  const concat = cs => { const out = new Uint8Array(cs.reduce((s, c) => s + c.length, 0)); let o = 0;
                         for (const c of cs) { out.set(c, o); o += c.length; } return out; };
  for (;;) {
   try {
    const rd = (await fetch('stream', {cache: 'no-store'})).body.getReader();
    let chunks = [], have = 0;
    const take = async n => {                  // the next n bytes of the stream
     while (have < n) { const {done, value} = await rd.read(); if (done) throw new Error('closed'); chunks.push(value); have += value.length; }
     const all = chunks.length === 1 ? chunks[0] : concat(chunks), rest = all.subarray(n);
     chunks = rest.length ? [rest] : []; have = rest.length;
     return all.subarray(0, n);
    };
    $('st2').textContent = '';
    for (;;) {
     const hd = await take(8), dv = new DataView(hd.buffer, hd.byteOffset, 8);
     const hl = dv.getUint32(0), il = dv.getUint32(4);
     const info = JSON.parse(new TextDecoder().decode(await take(hl)));
     if (il > 0) await onStreamFrame(info, (await take(il)).slice());
    }
   } catch (err) {
    $('st2').textContent = `frame stream: ${err} -- reconnecting`;
    await new Promise(r => setTimeout(r, 1000));
   }
  }
 }
 function go() { STREAM ? goStream() : goKernel(); }
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
 $('bench').onclick = async () => {
  $('st2').innerHTML = '<i>benchmarking the GPU kernel&hellip;</i>';
  const r = await call('bench', params());
  if (r.error) { $('st2').innerHTML = 'error: ' + r.error; return; }
  const k = `benchmark: kernel median ${r.med.toFixed(3)} ms, min ${r.min.toFixed(3)} ms over 20 frames ` +
            `(${Math.round(r.n / r.med / 1e3).toLocaleString()} Mpts/s)`;
  $('st2').innerHTML = k + '<br><i>measuring the display rate&hellip;</i>';
  if (STREAM) {
   const d = await burst(40);
   $('st2').innerHTML = k + `<br>display (whole frames in this browser, 40 back to back through the stream): ` +
     `<b>${d.toFixed(1)} ms per frame (${(1e3 / d).toFixed(1)} fps)</b>`;
  } else {
   const d1 = await kernelRate(20, 1), d2 = await kernelRate(20, DEPTH);
   $('st2').innerHTML = k + `<br>display (whole frames in this browser, 20 each): ${d1.toFixed(0)} ms per frame ` +
     `(${(1e3 / d1).toFixed(1)} fps) one at a time, <b>${d2.toFixed(0)} ms (${(1e3 / d2).toFixed(1)} fps) with ${DEPTH} in flight</b>`;
  }
 };
 $('save').onclick = async () => {
  $('st2').innerHTML = '<i>saving the full-resolution image&hellip;</i>';
  const r = await call('save', params());
  $('st2').innerHTML = r.error ? 'error: ' + r.error : `saved <tt>${r.file}</tt> (${r.mb.toFixed(1)} MB): Files panel on the left`;
 };
 if (META.default && EX[META.default]) $('ex').value = META.default;   // DEFAULT_EX in Python
 defaultW();
 if (STREAM) readStream();
 reset();
 if (META.autobench) setTimeout(() => $('bench').onclick(), 3000);     // AUTOBENCH = True in Python
})();
</script>
'''

_meta = dict(META, default=globals().get('DEFAULT_EX'), autobench=bool(globals().get('AUTOBENCH')))
if globals().get('TRANSPORT', 'stream') == 'stream':
    _PAGE = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
             '<style>:root {color-scheme: light dark} body {margin: 6px; background: Canvas; color: CanvasText}</style>'
             '</head><body>' + APP.replace('%META%', json.dumps(dict(_meta, stream=True))) + '</body></html>')
    PORT = globals().get('PORT', 8790)
    try:                                         # re-running the cell restarts the frame server
        _httpd.shutdown()
        _httpd.server_close()
    except NameError:
        pass
    _httpd = ThreadingHTTPServer(('', PORT), _Handler)
    _httpd.daemon_threads = True
    threading.Thread(target=_httpd.serve_forever, daemon=True).start()
    output.serve_kernel_port_as_iframe(PORT, height=1060)
else:
    display(HTML(APP.replace('%META%', json.dumps(dict(_meta, stream=False)))))
