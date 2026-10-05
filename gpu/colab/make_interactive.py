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
the GPU. The page always shows the newest request (intermediate slider positions are skipped while
a frame is in flight). The status line gives the GPU times; the frame round trip adds the PNG
encoding and the transfer to your browser.
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

def write(out, branch, extra_md=None):
    cs = json.loads(json.dumps(cells).replace("%BRANCH%", branch))
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


write("NyquistGPU_Interactive.ipynb", "gpu-cuda")
write("NyquistGPU_Hill_Interactive.ipynb", "hill-argument-principle", r"""
## Time-periodic systems (branch `hill-argument-principle`)
Besides the three time-independent examples, the dropdown offers three **time-periodic** delayed
systems, decided by a **Hill determinant + argument principle** (no time integration):
* *delayed Mathieu* x'' + κx' + (δ + ε cos t)x = b x(t − 2π) (axes δ, b);
* *milling, straight flutes* (Test 2): 1-DOF, z = 2, down milling, f_n = 922 Hz, cutting-force
  coefficient with a jump at tooth entry (axes: spindle speed [1000 rpm], depth of cut [mm]);
* *milling, different helix angles* (Test 3): helix 30° and β₂, the delay of every tooth distributed
  linearly over the axial depth, spindle period.

The milling examples are heavy (a dense Hill matrix and its LU per point and frequency sample,
stored in GPU memory): they start at 480×270 / 320×180 (about a second per frame); higher resolutions
work but take proportionally longer.

Counting: unstable Floquet exponents in one period strip of the imaginary axis, from the phase of the
row-scaled (pole-free) Hill determinant; the number of harmonics is derived per point from one
tolerance. Float32/Float64 only; the rightmost-root colouring uses the Newton-polished estimate.
""")
