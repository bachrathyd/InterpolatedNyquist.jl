"""CPU threads and GPU threads (lanes) vs time for 10^6 points (a 1000x1000 chart), log-log, for the three
time-independent examples. Reads thread_scaling_*.csv (CPU, 400x400) and lane_scaling_*.csv (GPU lanes)."""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
series = []   # (label, color, marker, system -> [(n, ms_per_Mpt)])
cpu_style = {"local": ("CPU: your PC (i5-10400, 6 cores)", "#2a78d6", "o"),
             "colab": ("CPU: Colab G4 host (48 threads)", "#0d366b", "s")}
for f in sorted(glob.glob(os.path.join(here, "thread_scaling_*.csv"))):
    tag = os.path.basename(f)[len("thread_scaling_"):-4]
    lab, col, mk = cpu_style.get(tag, (tag, "#555", "o"))
    d = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["device"] == "CPU" and r["T"] == "Float64":
            npts = int(r["nx"]) * int(r["ny"])
            d.setdefault(r["system"], []).append((int(r["threads"]), float(r["ms"]) / npts * 1e6))
    series.append((lab + ", Float64", col, mk, d, "-"))
gpu_style = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": ("GPU: RTX PRO 6000 (G4)", "#eb6834", "^"),
             "NVIDIA A100-SXM4-40GB": ("GPU: A100", "#1baf7a", "v"),
             "NVIDIA L4": ("GPU: L4", "#eda100", "D"), "Tesla T4": ("GPU: T4", "#e87ba4", "P")}
for f in sorted(glob.glob(os.path.join(here, "lane_scaling_*.csv"))):
    d, dev, T = {}, None, None
    for r in csv.DictReader(open(f, encoding="utf-8")):
        dev, T = r["device"], r["T"]
        d.setdefault(r["system"], []).append((int(r["threads"]), float(r["ms_per_Mpt"])))
    lab, col, mk = gpu_style.get(dev, (dev, "#999", "x"))
    series.append((lab + ", " + T, col, mk, d, "-" if T == "Float64" else "--"))

systems = [("fourth", "4th-order oscillator"), ("showcase", "2-DOF DAE (showcase)"), ("turning", "two-mode turning")]
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for ax, (s, title) in zip(axes, systems):
    for lab, col, mk, d, ls in series:
        pts = sorted(d.get(s, []))
        if not pts:
            continue
        n, t = zip(*pts)
        ax.plot(n, t, marker=mk, color=col, lw=1.8, ms=5, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title, fontsize=11); ax.set_xlabel("number of threads (CPU threads / GPU threads)")
    ax.grid(color="#e5e5e5", lw=0.8, which="both"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("time for 10⁶ points = 1000×1000 chart [ms]\n(1 s ≙ 1 µs per point)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9)
fig.suptitle("Parallel scaling (solid: Float64, dashed: Float32): time of a 1000 × 1000 chart vs number of threads", fontsize=12)
fig.tight_layout(rect=(0, 0.12, 1, 1))
out = os.path.join(here, "parallel_scaling.png")
fig.savefig(out, dpi=150)
print("written", out)
