"""Thread scaling of the three time-independent examples: chart time vs CPU threads (log-log), with the
ideal 1/n line from 1 thread and the GPU time as a horizontal reference. Reads thread_scaling_*.csv."""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(here, "thread_scaling_*.csv"))):
    tag = os.path.basename(f)[len("thread_scaling_"):-4]
    for r in csv.DictReader(open(f, encoding="utf-8")):
        r["tag"] = tag
        rows.append(r)
systems = [("fourth", "4th-order oscillator"), ("showcase", "2-DOF DAE (showcase)"), ("turning", "two-mode turning")]
cpus = [("local", "your PC (i5-10400, 6 cores)", "#2a78d6", "o"), ("colab", "Colab G4 host CPU", "#eb6834", "s")]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=False)
for ax, (s, title) in zip(axes, systems):
    for tag, lab, col, mk in cpus:
        pts = sorted((int(r["threads"]), float(r["ms"])) for r in rows
                     if r["tag"] == tag and r["system"] == s and r["device"] == "CPU" and r["nx"] == "400")
        if not pts:
            continue
        n, t = zip(*pts)
        ax.plot(n, t, marker=mk, color=col, lw=2, ms=6, label=lab)
        ax.plot([1, max(n)], [t[0], t[0] / max(n)], ls=":", color=col, lw=1)
    for T, ls in (("Float64", "--"), ("Float32", "-.")):
        g = [float(r["ms"]) for r in rows if r["system"] == s and r["device"] != "CPU" and r["nx"] == "400" and r["T"] == T]
        if g:
            ax.axhline(g[0], color="#1baf7a", lw=2, ls=ls, label=f"GPU G4, {T}: {g[0]:.2g} ms")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 48], ["1", "2", "4", "8", "16", "32", "48"])
    ax.set_title(title, fontsize=11); ax.set_xlabel("CPU threads")
    ax.grid(color="#e5e5e5", lw=0.8, which="both"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
axes[0].set_ylabel("time per 400 × 400 chart [ms] (dotted: ideal 1/n)")
fig.suptitle("Chart time vs. CPU threads (Float64) and on the GPU, 400 × 400 = 160 000 points", fontsize=12)
fig.tight_layout()
out = os.path.join(here, "thread_scaling.png")
fig.savefig(out, dpi=150)
print("written", out)
