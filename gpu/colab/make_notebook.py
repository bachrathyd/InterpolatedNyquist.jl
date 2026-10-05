"""Generate gpu/colab/NyquistGPU_Colab.ipynb (Python runtime driving Julia).

Edit the cells here, then run:  python gpu/colab/make_notebook.py
"""
import json
import os

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(True)})


md(r"""
# NyquistGPU on Colab: brute-force stability charts on an NVIDIA GPU

Runs the `gpu-cuda` branch of [InterpolatedNyquist.jl](https://github.com/bachrathyd/InterpolatedNyquist.jl)
(`gpu/` = the NyquistGPU engine) on a Colab GPU and writes every result to your Google Drive.

**Before you start:** *Runtime → Change runtime type → **Python 3** + a **GPU***
(this notebook drives Julia from the Python runtime, because only Python can mount Google Drive).

| GPU | FP32 | FP64 | ≈ compute units / h | use it for |
|---|---|---|---|---|
| T4 | 8 TFLOPS | 1/32 rate | ~1.2 | setup, debugging (also on the free tier) |
| L4 | 30 TFLOPS | 1/64 rate | ~1.7 | best value for Float32 charts |
| A100 | 19.5 TFLOPS | **9.7 TFLOPS** | ~5.4 | Float64 sweeps |
| RTX PRO 6000 (G4) | ~120 TFLOPS | 1/64 rate | ~8.7 | the fastest Float32 timings |

A full pass of this notebook takes roughly 15–25 min (most of it is the one-time Julia package setup),
i.e. well under 1 compute unit on a T4/L4. **When you are done: *Runtime → Disconnect and delete runtime*,**
otherwise an idle GPU keeps consuming units.

Steps: 1 settings · 2 GPU + Drive · 3 Julia · 4 code · 5 packages · 6 checks · 7 benchmark · 8 tables + charts · 9 your own model
""")

md("## 1. Settings")
code(r"""
REPO          = "https://github.com/bachrathyd/InterpolatedNyquist.jl"
BRANCH        = "gpu-cuda"
DRIVE_FOLDER  = "NyquistGPU"   # folder in *My Drive* that receives the results (created if missing);
                               # a nested path such as "Research/NyquistGPU" works too
JULIA_CHANNEL = "1.12"         # juliaup channel (the branch is tested with Julia 1.12)
""")

md("## 2. GPU and Google Drive")
code(r"""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
""")
code(r"""
import os, datetime, subprocess
from google.colab import drive
drive.mount('/content/drive')

root = '/content/drive/MyDrive'
OUT = os.path.join(root, DRIVE_FOLDER)
if not os.path.isdir(OUT):
    tops = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f'"{DRIVE_FOLDER}" not found in My Drive -- creating it. (Top-level folders: {tops[:40]})')
    os.makedirs(OUT, exist_ok=True)

gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip().split('\n')[0]
tag = (gpu or 'noGPU').replace(' ', '_')
RUN = os.path.join(OUT, 'runs', datetime.datetime.now().strftime('%Y%m%d_%H%M') + '_' + tag)
os.makedirs(RUN, exist_ok=True)
print('GPU     :', gpu or 'NONE -- select a GPU runtime!')
print('results ->', RUN)
""")

md("## 3. Julia (juliaup, ~1 min)")
code(r"""
import os
os.environ['PATH'] = '/root/.juliaup/bin:' + os.environ['PATH']
os.environ['JULIA_NUM_THREADS'] = 'auto'
if not os.path.exists('/root/.juliaup/bin/julia'):
    !curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel {JULIA_CHANNEL} > /dev/null
!julia --version
""")

md("## 4. Code (clone or update the branch)")
code(r"""
REPO_DIR = '/content/InterpolatedNyquist.jl'
if os.path.isdir(REPO_DIR):
    !cd {REPO_DIR} && git fetch -q --depth 1 origin {BRANCH} && git reset -q --hard FETCH_HEAD
else:
    !git clone -q -b {BRANCH} --depth 1 {REPO} {REPO_DIR}
!cd {REPO_DIR} && git log -1 --format='%h  %ad  %s' --date=short
""")

md("""
## 5. Julia packages (first time on a fresh runtime: ~5–10 min)
Installs CUDA.jl, KernelAbstractions and NyquistGPU, downloads the matching CUDA runtime,
and prints the CUDA configuration.
""")
code(r"""
!cd {REPO_DIR} && julia --project=gpu/scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA; CUDA.versioninfo()'
""")

md("""
## 6. Checks
* **GPU vs CPU**: the same kernels on the GPU and on the CPU backend must give the same
  counts on every unflagged point (the CPU backend is validated against the CPU package
  `InterpolatedNyquist.jl`, see `gpu/validate/`). Must end with `PASS`.
* Optional: the package unit tests (analytic Hayes region, 3-D point lists, flags), CPU only.
""")
code(r"""
!cd {REPO_DIR} && julia --project=gpu/scripts gpu/scripts/gpu_check.jl --n 128 2>&1 | tee "{RUN}/gpu_check.log"
""")
code(r"""
# optional (~2 min):
# !cd {REPO_DIR} && julia --project=gpu -t auto -e 'using Pkg; Pkg.test()'
""")

md("""
## 7. Benchmark ladder
Resolutions 100² → 1920×1080 for the three paper systems, Float32 and Float64, the three
schedules (`pixel` = one thread per point, `strided` = statistical load balancing,
`queue` = persistent threads with an atomic work queue). Also times a "slider loop" (a
constant changes every frame) at full HD. Each new configuration compiles once (~10–30 s).

Faster variant: `--systems showcase --res 512,1920x1080 --T Float32 --schedules queue`
""")
code(r"""
!cd {REPO_DIR} && julia --project=gpu/scripts gpu/scripts/bench_ladder.jl --out "{RUN}" 2>&1 | tee "{RUN}/bench.log"
""")

md("## 8. Results")
code(r"""
import glob, pandas as pd
csv = sorted(glob.glob(f'{RUN}/bench_*.csv'))[-1]
df = pd.read_csv(csv)
print('kernel time [ms], median of the repetitions:')
display(df.pivot_table(index=['system', 'nx', 'ny', 'method', 'T'], columns='schedule',
                       values='t_med_ms').round(3))
display(df[['system', 'nx', 'ny', 'T', 'schedule', 't_med_ms', 'frame_ms', 'mpts_per_s',
            'evals_med', 'evals_max', 'flagged', 'count_diff_vs_f64']])
""")
code(r"""
import sys
sys.path.insert(0, f'{REPO_DIR}/gpu/colab')
import plot_fields
fig = plot_fields.plot_all(RUN, f'{RUN}/charts.png')
""")

md("""
## 9. Your own model
`gpu/scripts/example_custom.jl` is a template: `D(λ, p, c)` (λ complex, `p` one parameter point
of any length, `c` constants), an arbitrary point list (a chart is `grid_points(xs, ys)`),
a Float32 GPU sweep, a Float64 re-check of the flagged points, and the saved field.

1. Run the next cell once. It copies the template to `gpu/scripts/my_model.jl` and to your Drive folder.
2. Edit the copy in your Drive folder: double-click `my_model.jl` in the Files pane under `drive/MyDrive/...`.
3. Run the cell after it.

Rules for `D`: write it as an **entire** function (multiply out rational denominators; stable poles
do not change the count), and take constants from `c` or use integer literals. A literal such as
`0.5*λ` silently turns a Float32 kernel into a much slower Float64 one; `check_eltype` warns about it.
""")
code(r"""
import shutil
MY = os.path.join(OUT, 'my_model.jl')
if not os.path.exists(MY):
    shutil.copy(f'{REPO_DIR}/gpu/scripts/example_custom.jl', MY)
print('edit this file:', MY)
""")
code(r"""
shutil.copy(MY, f'{REPO_DIR}/gpu/scripts/my_model.jl')
!cd {REPO_DIR} && julia --project=gpu/scripts gpu/scripts/my_model.jl --out "{RUN}" --res 1920x1080
fig = plot_fields.plot_all(RUN, f'{RUN}/charts.png')
""")

md("""
---
**Done? *Runtime → Disconnect and delete runtime*.** Everything worth keeping is in your Drive folder
(`runs/<date>_<GPU>/`: `bench_*.csv/.md`, `gpu_check.log`, `bench.log`, `field_*` and `charts.png`).
""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "L4", "name": "NyquistGPU_Colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NyquistGPU_Colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote", out, len(cells), "cells")
