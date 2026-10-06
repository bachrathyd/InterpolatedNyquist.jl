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
REPO, BRANCH, JULIA_CHANNEL = "https://github.com/bachrathyd/InterpolatedNyquist.jl", "%BRANCH%", "1.12"
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

md("""
## 3. The interactive chart
Move a slider, pick another example / format / resolution, zoom or pan: the chart is recomputed on
the GPU. The status line gives the GPU and server times, the latency from a slider move to the new
picture, and the display rate while the picture keeps changing. *benchmark* measures the kernel and
the frame rate in your browser. *save full-resolution PNG* writes the exact chart.

How frames reach the browser: the chart is shown through Colab's port proxy. A small HTTP server in
this notebook receives every new slider state and **streams** the frames back on one open
connection. The GPU always renders the newest state as soon as the previous frame is out, so the
frame rate is set by the GPU and the image encoding, not by the ~50 ms round trip of a request. The
live view is sent as JPEG, about 5× smaller than PNG; choose *PNG* for exact pixels. If the chart
below stays empty (port proxy blocked), add `TRANSPORT = 'kernel'` at the top of the cell and run it
again. That uses Colab's kernel channel instead, with one request per frame and two in flight.
""")
code(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interactive_app.py"),
          encoding="utf-8").read())

md("""
## 4. Done?
Stop the server and **disconnect the runtime** (*Runtime → Disconnect and delete runtime*) so the GPU
stops consuming compute units.
""")
code(r"""
# (commented out so that *Run all* does not stop the server right after starting it)
# srv.close()
""")

def write(out, branch, extra_md=None, default_ex=None):
    cs = json.loads(json.dumps(cells).replace("%BRANCH%", branch))
    if default_ex:                     # the example shown first (the app reads DEFAULT_EX)
        app = next(c for c in cs if c["cell_type"] == "code" and any("APP = r'''" in l for l in c["source"]))
        app["source"].insert(0, f"DEFAULT_EX = '{default_ex}'\n")
    if extra_md:
        cs.insert(1, {"cell_type": "markdown", "metadata": {}, "source": extra_md.strip(chr(10)).splitlines(True)})
    nb = {"cells": cs, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "G4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("written", path)




def write_tour(out, branch):
    "A short notebook: setup (cell 1 of the interactive notebook) + the GPU tour of the milling charts."
    setup = next(c for c in cells if c["cell_type"] == "code")

    def cc(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.strip("\n").splitlines(True)}
    cs = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# GPU tour: milling stability charts (branch `hill-argument-principle`)\n",
            "\n",
            "Times full-HD milling stability charts on the GPU of this runtime, in every number format\n",
            "(Float32; Float16 evaluation with a Float32 march; Float16 + a Float32 re-check of the flagged\n",
            "points), each checked against a Float64 reference on a coarse grid:\n",
            "* `hill/warp/tour_warp3h.jl`: the hardest case (Test 3, helix 30°/45°, 3–30 krpm × 0–10 mm) with\n",
            "  the warp-cooperative kernel (one warp per point), full chart and adaptive refinement;\n",
            "* `hill/gpu_tour_milling.jl`: Test 2 (`mill2g`), Test 3 with one thread per point (`mill3h`), and\n",
            "  the dense Hill reference (`mill3d`, 480×270).\n",
            "\n",
            "Pick the GPU under *Runtime → Change runtime type*, then *Run all* (~20–40 min with the Julia\n",
            "setup). **Afterwards: *Runtime → Disconnect and delete runtime*.**\n"]},
        json.loads(json.dumps(setup).replace("%BRANCH%", branch)),
        cc(r'''
# hardest case, warp-cooperative kernel: smoke test (GPU vs CPU-validated reference), then full HD
# (the logs also go to /content/*.log)
sh(f"set -o pipefail; cd {REPO_DIR} && julia --project=gpu/scripts hill/warp/tour_warp3h.jl --res 320x180 "
   f"--check 48x27 --adaptive no 2>&1 | tee /content/tour_warp_smoke.log")
sh(f"set -o pipefail; cd {REPO_DIR} && julia --project=gpu/scripts hill/warp/tour_warp3h.jl --res 1920x1080 "
   f"--check 192x108 --csv /content/tour_warp.csv 2>&1 | tee /content/tour_warp.log")
'''),
        cc(r'''
MODELS, ADAPTIVE = 'mill2g,mill3h,mill3d', 'yes'   # see MODELS in hill/gpu_tour_milling.jl
sh(f"set -o pipefail; cd {REPO_DIR} && julia --project=gpu/scripts hill/gpu_tour_milling.jl --models {MODELS} "
   f"--res 1920x1080 --check 192x108 --adaptive {ADAPTIVE} --csv /content/tour_main.csv 2>&1 | tee /content/tour_main.log")
'''),
        cc(r'''
# registers / local memory / occupancy of the warp kernel, and the CSV rows of both tours
sh(f"set -o pipefail; cd {REPO_DIR} && julia --project=gpu/scripts hill/warp/kernel_info_warp3h.jl 2>&1 | tee /content/kinfo.log")
sh("cat /content/tour_warp.csv /content/tour_main.csv")
'''),
    ]
    nb = {"cells": cs, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("written", path)

write("NyquistGPU_Interactive.ipynb", "gpu-cuda")
write("NyquistGPU_Hill_Interactive.ipynb", "hill-argument-principle", r"""
## Time-periodic systems (branch `hill-argument-principle`)
Besides the three time-independent examples, the dropdown offers four **time-periodic** delayed
systems, decided by a **Hill determinant + argument principle** (no time integration):
* *milling, straight flutes (Test 2, compressed Hill, fast)*, shown first. 1-DOF, z = 2, down milling,
  f_n = 922 Hz, a cutting-force coefficient with a jump at tooth entry; axes: spindle speed [1000 rpm]
  and depth of cut [mm].
  - The infinite Hill determinant is used with all harmonics in closed form. The matrix determinant
    lemma reduces it to the 16 quadrature nodes of the cutting window, and its semiseparable
    structure gives that 16×16 determinant in O(16) operations.
  - The count runs along the unit circle of the Floquet multiplier.
  - On a G4, a full HD chart takes **5 ms on the GPU in Float16** (about 100 frames per second in
    the browser), 7 ms with the Float32 re-check of the flagged points, and 16 ms in Float32.
* *delayed Mathieu* x'' + κx' + (δ + ε cos t)x = b x(t − 2π); axes δ and b.
* *milling, straight flutes (Test 2, dense Hill, slow)*: the same chart from the truncated dense Hill
  matrix, the reference implementation.
* *milling, different helix angles* (Test 3): helix 30° and β₂, with the delay of every tooth
  distributed linearly over the axial depth; spindle period.

The dense milling examples are heavy: a dense Hill matrix and its LU per point and frequency sample,
stored in GPU memory. They start at 480×270 / 320×180, about a second per frame; higher resolutions
work but take proportionally longer.

How the dense forms count: unstable Floquet exponents in one period strip of the imaginary axis,
from the phase of the row-scaled (pole-free) Hill determinant. The number of harmonics is derived per
point from one tolerance. The rightmost-root colouring uses the Newton-polished estimate. Float16 is
offered for the fast example only.
""", default_ex="mill2q")
write_tour("NyquistGPU_Hill_Tour.ipynb", "hill-argument-principle")

def write_speed(out, branch, script="hill/warp/speed_ratio.jl --res 480x270", title="Speed of Float64 / Float32 / Float16 relative to Float32 (milling Tests 3 and 2, 480x270)"):
    "Setup + the format speed ratios (F64 / F32 / F16) on a small chart."
    setup = next(c for c in cells if c["cell_type"] == "code")
    cs = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Speed of Float64 / Float32 / Float16 relative to Float32 (milling Tests 3 and 2, 480x270)\n",
            "Pick the GPU, *Run all*. **Afterwards: Runtime -> Disconnect and delete runtime.**\n"]},
        json.loads(json.dumps(setup).replace("%BRANCH%", branch)),
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "sh(f\"set -o pipefail; cd {REPO_DIR} && julia --project=gpu/scripts " + script + " \"\n",
            "   f\"--csv /content/speed.csv 2>&1 | tee /content/speed.log\")\n"]},
    ]
    nb = {"cells": cs, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), out), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("written", out)

write_speed("NyquistGPU_Hill_Speed.ipynb", "hill-argument-principle")
write_speed("NyquistGPU_Lane_Scaling.ipynb", "hill-argument-principle", "gpu/scripts/gpu_lane_scaling.jl",
            "GPU thread scaling: chart time vs number of GPU threads (Float64), three time-independent examples")
