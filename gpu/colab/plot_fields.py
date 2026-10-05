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


def colour_rgb(C, Z, sigma_min, z_max=6):
    """RGBA image of a chart (row 0 = lowest y): viridis spectral gap in the
    stable domain, Reds for the unstable counts, grey for failed points and a
    one-pixel white stability boundary."""
    import matplotlib as mpl
    rgb = np.zeros(C.shape + (4,))
    st, un, fail = Z == 0, Z > 0, Z < 0
    rgb[st] = mpl.colormaps["viridis"](np.clip((C[st] - sigma_min) / (0 - sigma_min), 0, 1))
    rgb[un] = mpl.colormaps["Reds"](np.clip(C[un] / (z_max + 1), 0, 1))
    rgb[fail] = (0.5, 0.5, 0.5, 1.0)
    edge = np.zeros_like(st)
    edge[:-1, :] |= st[:-1, :] != st[1:, :]
    edge[:, :-1] |= st[:, :-1] != st[:, 1:]
    rgb[edge] = (1.0, 1.0, 1.0, 1.0)
    return rgb


def _sigma_floor(C, Z):
    s = C[(Z == 0) & np.isfinite(C)]
    return min(float(np.percentile(s, 2)), -1e-3) if s.size else -1.0


def save_native(base, png):
    """The chart as a pixel-exact image (nx x ny pixels, no axes)."""
    meta, C, Z = load_field(base)
    plt.imsave(png, np.flipud(colour_rgb(C, Z, _sigma_floor(C, Z))))
    return png


def plot_series(bases, out_png, ncols=3):
    """Panel figure of a resolution series (same chart, increasing nx*ny)."""
    bases = sorted(bases, key=lambda b: (lambda m: m["nx"] * m["ny"])(load_field(b)[0]))
    nrows = -(-len(bases) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.4 * nrows), squeeze=False)
    for ax, b in zip(axs.flat, bases):
        plot_field(b, ax)
    for ax in list(axs.flat)[len(bases):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def render_all(folder):
    """Annotated + native PNG for every field in `folder`, and a panel of any
    resolution series (fields named field_<system>_series_<nx>x<ny>)."""
    out = []
    bases = sorted(p[:-5] for p in glob.glob(os.path.join(folder, "field_*.json")))
    for b in bases:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_field(b, ax)
        fig.tight_layout()
        fig.savefig(b + "_chart.png", dpi=150)
        plt.close(fig)
        out += [b + "_chart.png", save_native(b, b + "_native.png")]
    series = [b for b in bases if "_series_" in os.path.basename(b)]
    if series:
        out.append(plot_series(series, os.path.join(folder, "resolution_series.png")))
    for p in out:
        print("saved", p)
    return out


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    render_all(sys.argv[1] if len(sys.argv) > 1 else ".")
