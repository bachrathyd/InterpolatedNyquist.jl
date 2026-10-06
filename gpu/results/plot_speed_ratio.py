"""Bar chart of the format speed ratios (time / time of Float32) from gpu/results/hill_speed_*.csv."""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(here, "hill_speed_*.csv"))):
    rows += list(csv.DictReader(open(f, encoding="utf-8")))
short = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": "G4", "NVIDIA A100-SXM4-40GB": "A100",
         "NVIDIA L4": "L4", "Tesla T4": "T4"}
gpus = [g for g in ("G4", "A100", "L4", "T4") if any(short.get(r["gpu"], r["gpu"]) == g for r in rows)]
fmts = [("F64", "#2a78d6"), ("F32", "#eb6834"), ("F16", "#1baf7a")]
models = [("mill3w", "Test 3 (helix), warp kernel"), ("mill2g", "Test 2 (straight flutes)")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (m, title) in zip(axes, models):
    w = 0.26
    for k, (f, col) in enumerate(fmts):
        xs, ys = [], []
        for i, g in enumerate(gpus):
            r = [r for r in rows if short.get(r["gpu"], r["gpu"]) == g and r["model"] == m and r["format"] == f]
            if r:
                xs.append(i + (k - 1) * w); ys.append(float(r[0]["ratio_to_F32"]))
        bars = ax.bar(xs, ys, w * 0.92, color=col, label=f)
        for x, y in zip(xs, ys):
            ax.text(x, y * 1.08, f"{y:.2g}×", ha="center", va="bottom", fontsize=8, color="#333")
    ax.axhline(1, color="#888", lw=1, zorder=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(gpus)), gpus)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("time / time in Float32 (log)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Speed of the number formats relative to Float32 (480 × 270 chart)", fontsize=12)
fig.tight_layout()
out = os.path.join(here, "hill_speed_ratio.png")
fig.savefig(out, dpi=150)
print("written", out)
