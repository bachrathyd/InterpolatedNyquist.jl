"""Generate gpu/colab/NyquistGPU_Interactive.ipynb (interactive stability charts on a Colab GPU).

Edit the cells here, then run:  python gpu/colab/make_interactive.py
"""
import json
import os

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)})


def code(src, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(True)})


md(r"""
# Interactive GPU stability charts (NyquistGPU)

Brute-force stability charts of delayed systems, recomputed on the GPU while you move the sliders:
**up to 8K (7680×4320 = 33 million parameter points) per frame**, one GPU thread per point, with the
discrete phase-unwrapping march of [InterpolatedNyquist.jl](https://github.com/bachrathyd/InterpolatedNyquist.jl)
(branch `gpu-cuda`, folder `gpu/`).

**Runtime:** *Runtime → Change runtime type → Python 3 + **G4 (RTX PRO 6000)***, the fastest Colab GPU
(full HD in Float16: 0.7 ms per chart). Any other NVIDIA GPU works too (T4: ~6 ms).
Then *Runtime → Run all*. The first start takes ~5–10 min (Julia + CUDA packages); afterwards the
server starts in ~1 min. **When you are done: *Runtime → Disconnect and delete runtime*.**

How it works: a persistent Julia process keeps the sweep plan (points + result buffers) on the GPU,
evaluates the chart, colours it and box-filters it to the display size **on the GPU**, and only the
display image crosses the bus. Changing a constant re-runs the kernel only; changing the resolution,
example or number format re-allocates the plan; the first use of an example/format compiles its kernel
(~10–30 s).

Number formats:
* **Float16 (fastest)**: the characteristic function is evaluated in half precision (march in
  Float32, frequency window ω ≤ 15, since Float16 ends at 65504). Points whose count is a decision
  rather than a measurement are *flagged* (tick *show flagged points* to see them in magenta).
  Accurate for the 4th-order and showcase models (<0.05 % wrong, all flagged); the turning model
  has lightly damped modes and ~33 % flagged points in Float16 -- use the re-check or Float32 there.
* **Float16 + Float32 re-check**: the flagged points are redone in Float32 on the GPU.
* **Float32**: exact charts (production setting, ω_max = 10⁵). **Float64**: reference (slow on
  graphics-class GPUs, which run Float64 at 1/64 rate).
""")

md("## 1. Setup (Julia, code, packages)")
code(r"""
REPO, BRANCH, JULIA_CHANNEL = "https://github.com/bachrathyd/InterpolatedNyquist.jl", "gpu-cuda", "1.12"
REPO_DIR = '/content/InterpolatedNyquist.jl'
import os, subprocess

def sh(cmd):
    "Run a shell command, stream its output, and STOP on failure."
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', text=True, bufsize=1,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end='')
    if p.wait() != 0:
        raise RuntimeError(f'command failed (exit code {p.returncode}) -- see the output above')

sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')
os.environ['PATH'] = '/root/.juliaup/bin:' + os.environ['PATH']
os.environ['JULIA_NUM_THREADS'] = 'auto'
if not os.path.exists('/root/.juliaup/bin/julia'):
    sh(f'curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel {JULIA_CHANNEL} > /dev/null')
if os.path.isdir(REPO_DIR):
    sh(f'cd {REPO_DIR} && git fetch -q --depth 1 origin {BRANCH} && git reset -q --hard FETCH_HEAD')
else:
    sh(f'git clone -q -b {BRANCH} --depth 1 {REPO} {REPO_DIR}')
sh(f"cd {REPO_DIR} && git log -1 --format='%h  %ad  %s' --date=short")
sh(f"cd {REPO_DIR} && julia --project=gpu/scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 2>&1 | tail -3")
""")

md("## 2. Start the GPU server (~1 min: loads CUDA and compiles the default kernel)")
code(r"""
import json, threading, time

class NyquistServer:
    "The persistent Julia process (gpu/interactive/server.jl), one request per line."
    def __init__(self):
        self.log = open('/content/nyquist_server.log', 'w')
        self.p = subprocess.Popen(['julia', '--project=gpu/scripts', 'gpu/interactive/server.jl'],
                                  cwd=REPO_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, bufsize=1)
        self.lock = threading.Lock()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('the server stopped -- see /content/nyquist_server.log')
            if line.startswith('READY'):
                print(line.strip())
                break
            print(line, end='')

    def ask(self, cmd):
        with self.lock:
            self.p.stdin.write(cmd + '\n')
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        tag, _, rest = line.strip().partition(' ')
        if tag == 'ERR' or not line:
            raise RuntimeError(rest or 'server stopped (see /content/nyquist_server.log)')
        return json.loads(rest)

    def close(self):
        try:
            self.p.stdin.write('quit\n'); self.p.stdin.flush(); self.p.wait(5)
        except Exception:
            self.p.kill()

try:
    srv.close()           # re-running this cell restarts the server
except NameError:
    pass
t0 = time.time()
srv = NyquistServer()
META = srv.ask('meta')
print(f"server up in {time.time() - t0:.0f} s on {META['device']}")
""")

md("## 3. The interactive chart")
code(r"""
import io, numpy as np, ipywidgets as W
from PIL import Image as PImage
from IPython.display import display

EX = {e['key']: e for e in META['examples']}
RES = [('960×540 (qHD)', (960, 540)), ('1280×720 (HD)', (1280, 720)), ('1920×1080 (full HD)', (1920, 1080)),
       ('2560×1440 (QHD)', (2560, 1440)), ('3840×2160 (4K UHD)', (3840, 2160)),
       ('7680×4320 (8K UHD)', (7680, 4320))]
DISP = (1600, 900)                                    # display size (the GPU box-filters down to it)
DISP_FILE = '/dev/shm/nyq_disp.rgb'

w_ex = W.Dropdown(options=[(e['title'], e['key']) for e in META['examples']], description='example')
w_fmt = W.Dropdown(options=[(l, k) for k, l in META['formats']], value='F16', description='format',
                   layout=W.Layout(width='420px'))
w_res = W.Dropdown(options=RES, value=(1920, 1080), description='resolution')
w_bnd = W.Checkbox(value=True, description='boundary (white)')
w_flag = W.Checkbox(value=False, description='show flagged points')
w_smin = W.FloatSlider(description='σ colour floor', min=-2.0, max=-0.005, step=0.005, readout_format='.3f',
                       continuous_update=True)
w_x0, w_x1, w_y0, w_y1 = (W.FloatText(description=d, layout=W.Layout(width='190px'))
                          for d in ('x min', 'x max', 'y min', 'y max'))
b = lambda txt, tip: W.Button(description=txt, tooltip=tip, layout=W.Layout(width='auto'))
b_in, b_out, b_reset = b('zoom in ×2', 'halve both ranges'), b('zoom out ×2', ''), b('reset view', '')
b_l, b_r, b_d, b_u = b('←', 'pan left'), b('→', 'pan right'), b('↓', 'pan down'), b('↑', 'pan up')
b_bench, b_save = b('benchmark 20 frames', 'kernel timing at the current settings'), \
                  b('save full-resolution PNG', 'writes /content/<example>_<res>.png')
knob_box = W.VBox()
img = W.Image(format='png', layout=W.Layout(width='100%', max_width=f'{DISP[0]}px'))
caption = W.HTML()
status = W.HTML()
knobs = []

def build_knobs(e):
    global knobs
    knobs = []
    for k in e['knobs']:
        s = W.FloatSlider(value=k['value'], min=k['lo'], max=k['hi'], step=(k['hi'] - k['lo']) / 400,
                          description=k['name'], readout_format='.4g', continuous_update=True,
                          style={'description_width': '150px'}, layout=W.Layout(width='520px'))
        s.ci = k['i'] - 1
        s.observe(lambda ch: None if _quiet[0] else request(), 'value')
        knobs.append(s)
    knob_box.children = knobs

def constants():
    c = list(EX[w_ex.value]['c'])
    for s in knobs:
        c[s.ci] = s.value
    return c

def params():
    nx, ny = w_res.value
    return dict(ex=w_ex.value, fmt=w_fmt.value, nx=nx, ny=ny, x0=w_x0.value, x1=w_x1.value,
                y0=w_y0.value, y1=w_y1.value, c=','.join(repr(float(v)) for v in constants()),
                maxw=DISP[0], maxh=DISP[1], smin=w_smin.value, flags=int(w_flag.value),
                bnd=int(w_bnd.value), out=DISP_FILE)

def cmd(name, p):
    return name + ' ' + ' '.join(f'{k}={v}' for k, v in p.items())

def draw(p):
    t0 = time.time()
    status.value = '<i>computing… (the first use of an example / format compiles its kernel, ~10–30 s)</i>'
    r = srv.ask(cmd('render', p))
    a = np.fromfile(DISP_FILE, np.uint8, count=3 * r['dw'] * r['dh']).reshape(r['dh'], r['dw'], 3)
    buf = io.BytesIO()
    PImage.fromarray(a).save(buf, 'PNG', compress_level=1)
    img.value = buf.getvalue()
    rt = 1e3 * (time.time() - t0)
    e = EX[p['ex']]
    caption.value = (f"<b>{e['title']}</b> — horizontal: {e['xl']} ∈ [{p['x0']:.4g}, {p['x1']:.4g}], "
                     f"vertical: {e['yl']} ∈ [{p['y0']:.4g}, {p['y1']:.4g}] — "
                     f"red: number of unstable roots, blue→yellow: rightmost root σ in the stable domain")
    re = f" | re-check {r['n_recheck']:,} pts {r['t_recheck']:.2f} ms" if p['fmt'] == 'F16+' else ''
    alloc = f" | plan allocation {r['t_plan']:.0f} ms" if r['t_plan'] > 2 else ''
    status.value = (f"<tt>{p['nx']}×{p['ny']} = {r['n'] / 1e6:.2f} Mpts | <b>GPU kernel {r['t_kernel']:.2f} ms</b> "
                    f"({r['mpts']:,.0f} Mpts/s){re} | colour + downsample {r['t_colour']:.2f} ms | "
                    f"read-back {r['t_read']:.2f} ms{alloc} | flagged {r['flagged_pct']:.3f} % | "
                    f"frame round trip {rt:.0f} ms (incl. PNG + browser)</tt>")

# latest-request-wins worker: slider events never queue up behind a slow frame
_lock, _state = threading.Lock(), {'pending': None, 'busy': False}
def request(*_):
    with _lock:
        _state['pending'] = params()
        if _state['busy']:
            return
        _state['busy'] = True
    threading.Thread(target=_worker, daemon=True).start()

def _worker():
    while True:
        with _lock:
            p, _state['pending'] = _state['pending'], None
            if p is None:
                _state['busy'] = False
                return
        try:
            draw(p)
        except Exception as err:
            status.value = f'<b style="color:#c00">error:</b> <tt>{err}</tt>'

_quiet = [False]
def set_view(x0, x1, y0, y1):
    _quiet[0] = True
    w_x0.value, w_x1.value, w_y0.value, w_y1.value = x0, x1, y0, y1
    _quiet[0] = False
    request()

def on_example(*_):
    e = EX[w_ex.value]
    _quiet[0] = True
    w_smin.value = e['smin']
    w_x0.value, w_x1.value = e['xr']
    w_y0.value, w_y1.value = e['yr']
    build_knobs(e)
    _quiet[0] = False
    request()

def zoom(f):
    cx, cy = (w_x0.value + w_x1.value) / 2, (w_y0.value + w_y1.value) / 2
    hx, hy = (w_x1.value - w_x0.value) / 2 * f, (w_y1.value - w_y0.value) / 2 * f
    set_view(cx - hx, cx + hx, cy - hy, cy + hy)

def pan(dx, dy):
    sx, sy = (w_x1.value - w_x0.value) * dx, (w_y1.value - w_y0.value) * dy
    set_view(w_x0.value + sx, w_x1.value + sx, w_y0.value + sy, w_y1.value + sy)

def bench(*_):
    p = params()
    ts = [srv.ask(cmd('render', p))['t_kernel'] for _ in range(20)]
    status.value += f"<br><tt>benchmark: kernel median {np.median(ts):.3f} ms, min {min(ts):.3f} ms " \
                    f"over 20 frames ({p['nx'] * p['ny'] / np.median(ts) / 1e3:,.0f} Mpts/s)</tt>"

def save(*_):
    p = params()
    status.value = '<i>saving the full-resolution image…</i>'
    srv.ask(cmd('render', p))
    r = srv.ask('save out=/dev/shm/nyq_full.rgb')
    a = np.fromfile('/dev/shm/nyq_full.rgb', np.uint8, count=3 * r['w'] * r['h']).reshape(r['h'], r['w'], 3)
    fn = f"/content/{p['ex']}_{p['fmt'].replace('+', 'r')}_{r['w']}x{r['h']}.png"
    PImage.fromarray(a).save(fn, optimize=False)
    status.value = f"saved <tt>{fn}</tt> ({os.path.getsize(fn) / 1e6:.1f} MB) — Files panel on the left, or " \
                   f"<tt>from google.colab import files; files.download('{fn}')</tt>"

for w in (w_fmt, w_res, w_bnd, w_flag, w_smin, w_x0, w_x1, w_y0, w_y1):
    w.observe(lambda ch: None if _quiet[0] else request(), 'value')
w_ex.observe(on_example, 'value')
b_in.on_click(lambda _: zoom(0.5)); b_out.on_click(lambda _: zoom(2.0))
b_reset.on_click(lambda _: on_example())
b_l.on_click(lambda _: pan(-0.25, 0)); b_r.on_click(lambda _: pan(0.25, 0))
b_d.on_click(lambda _: pan(0, -0.25)); b_u.on_click(lambda _: pan(0, 0.25))
b_bench.on_click(bench); b_save.on_click(save)

ui = W.VBox([
    W.HBox([w_ex, w_fmt, w_res]),
    W.HBox([knob_box, W.VBox([w_smin, W.HBox([w_bnd, w_flag])])]),
    W.HBox([w_x0, w_x1, w_y0, w_y1]),
    W.HBox([b_in, b_out, b_reset, b_l, b_r, b_d, b_u, b_bench, b_save]),
    status, img, caption])
display(ui)
on_example()
""")

md("""
## 4. Done?
Stop the server and **disconnect the runtime** (*Runtime → Disconnect and delete runtime*) so the GPU
stops consuming compute units.
""")
code(r"""
srv.close()
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "G4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NyquistGPU_Interactive.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("written", out)
