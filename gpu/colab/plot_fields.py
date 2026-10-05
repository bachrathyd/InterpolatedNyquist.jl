"""Plot the stability-chart fields written by gpu/scripts/bench_ladder.jl.

Each chart is three files with a common base name:
    <base>.f32   colour field, float32, nx*ny, x fastest (Julia column-major)
                 = dominant root estimate sigma inside the stable domain (Z == 0),
                   the integer count Z (capped) outside
    <base>.i8    counts Z, int8 (-1 = march failed)
    <base>.json  nx, ny, axis ranges and labels, device, kernel time
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_field(base):
    with open(base + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    nx, ny = meta["nx"], meta["ny"]
    C = np.fromfile(base + ".f32", dtype=np.float32).reshape(ny, nx)
    Z = np.fromfile(base + ".i8", dtype=np.int8).reshape(ny, nx)
    return meta, C, Z


def plot_field(base, ax=None, sigma_min=None, z_max=6):
    """Paper-style interpolable chart: green-to-yellow spectral gap in the stable
    domain, discrete reds for the number of unstable roots, white boundary.
    `sigma_min` (colour floor of the spectral gap) defaults to the 2nd
    percentile of the stable-domain estimates."""
    meta, C, Z = load_field(base)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if sigma_min is None:
        s = C[(Z == 0) & np.isfinite(C)]
        sigma_min = min(float(np.percentile(s, 2)), -1e-3) if s.size else -1.0
    ext = [meta["xr"][0], meta["xr"][1], meta["yr"][0], meta["yr"][1]]
    common = dict(origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    ax.imshow(np.ma.masked_where(Z <= 0, C), cmap="Reds", vmin=0, vmax=z_max + 1, **common)
    ax.imshow(np.ma.masked_where(Z != 0, C), cmap="viridis", vmin=sigma_min, vmax=0, **common)
    ax.imshow(np.ma.masked_where(Z != -1, np.ones_like(C)), cmap="gray", vmin=0, vmax=2, **common)
    xs = np.linspace(ext[0], ext[1], meta["nx"])
    ys = np.linspace(ext[2], ext[3], meta["ny"])
    if (Z == 0).any() and (Z != 0).any():
        ax.contour(xs, ys, (Z == 0).astype(float), levels=[0.5], colors="white", linewidths=0.8)
    ax.set_xlabel(meta["xl"])
    ax.set_ylabel(meta["yl"])
    ax.set_title(f'{meta["title"]}\n{meta["nx"]}x{meta["ny"]}: {meta["kernel_ms"]:.1f} ms '
                 f'on {meta["device"]}', fontsize=9)
    return ax


def plot_all(folder, out_png=None):
    """One panel per saved field in `folder`; optionally save the figure."""
    bases = sorted(p[:-5] for p in glob.glob(os.path.join(folder, "field_*.json")))
    if not bases:
        print("no field_*.json in", folder)
        return None
    fig, axs = plt.subplots(1, len(bases), figsize=(6 * len(bases), 4.8), squeeze=False)
    for ax, b in zip(axs[0], bases):
        plot_field(b, ax)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
        print("saved", out_png)
    return fig
