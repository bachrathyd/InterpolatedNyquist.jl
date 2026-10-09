// The march on a server instead of WebGPU: the Colab notebook runs gpu/webui/server.jl
// (NyquistGPU on the Colab GPU) and serves this page. RemoteEngine is the WebGPU Engine with
// compute() replaced for 2-D charts: the server returns the same 16-byte records march.wgsl would
// write; they are scattered into the result buffer (progressive levels: the stride-s nodes) and
// coloured locally by display.wgsl as before. 3-D views and models without Julia code (integral
// by quadrature, the hand-written examples) are computed locally.
// Coarse levels (opt.final false) in Float32, the final level in Float64.

import { Engine } from './engine.js';

export class RemoteEngine extends Engine {
  /** the server, if this page was served by it (GET api/info); else null */
  static async detect() {
    try {
      const r = await fetch('api/info', { cache: 'no-store' });
      if (!r.ok || !(r.headers.get('content-type') || '').includes('json')) return null;
      return await r.json();
    } catch (e) { return null; }
  }

  /** the WebGPU engine (for the colouring) turned into a remote one */
  static async create(log = () => {}, info = null) {
    const eng = await Engine.create(log);
    Object.setPrototypeOf(eng, RemoteEngine.prototype);
    eng.server = info;
    eng.remote = true;
    eng.mirror = null;
    eng.rate = new Map();
    return eng;
  }

  get deviceName() { return this.server ? this.server.device : super.deviceName; }

  async compute(job, opts = {}) {
    if (!job.julia || (job.nz || 1) > 1) return super.compute(job, opts);
    const nx = job.nx, ny = job.ny;
    const S = opts.stride || 1;
    const nxs = Math.ceil(nx / S), nys = Math.ceil(ny / S);
    const dx = nx > 1 ? (job.xr[1] - job.xr[0]) / (nx - 1) : 0, dy = ny > 1 ? (job.yr[1] - job.yr[0]) / (ny - 1) : 0;
    const fresh = this.ensureBuffers(nx, ny);
    this.grid3d = false;
    if (fresh || !this.mirror || this.mirror.byteLength !== nx * ny * 16) this.mirror = new ArrayBuffer(nx * ny * 16);
    const M = new Uint32Array(this.mirror);
    const m = { w0: 1e-9, wmax: 1e5, tol: 0.3, hmax: Infinity, wband: 0, maxsteps: 50000, ...job.march };
    const prec = opts.final === false ? 'F32' : 'F64';
    const c = [...job.c, ...job.julia.lits];
    const key = job.julia.src + '|' + prec;
    const t0 = performance.now();
    let row = 0, gpuMs = 0, bands = 0, cancelled = false, timedOut = false;
    // requests of ~0.5 s (rows of the stride-S lattice), from the measured rate
    let rows = Math.max(1, Math.min(nys, Math.floor((this.rate.get(key) || 20) * 500 / nxs)));
    while (row < nys) {
      if (opts.isCancelled?.()) { cancelled = true; break; }
      if (opts.deadlineMs && performance.now() - t0 > opts.deadlineMs) { cancelled = true; timedOut = true; break; }
      const nr = Math.min(rows, nys - row);
      const y0 = job.yr[0] + row * S * dy;
      const body = JSON.stringify({ src: job.julia.src, c, x0: job.xr[0], x1: job.xr[0] + (nxs - 1) * S * dx,
        y0, y1: y0 + (nr - 1) * S * dy, nx: nxs, ny: nr, npow: job.npow, w0: m.w0, wmax: m.wmax, tol: m.tol,
        hmax: Number.isFinite(m.hmax) ? m.hmax : null, wband: Number.isFinite(m.wband) ? m.wband : null,
        maxsteps: m.maxsteps, exact: !!job.exact, prec });
      const tc = performance.now();
      const resp = await fetch('api/march', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
      if (!resp.ok) throw new Error('server: ' + (await resp.text()).slice(0, 400));
      const out = new Uint32Array(await resp.arrayBuffer());
      gpuMs += +(resp.headers.get('X-Compute-Ms') || 0);
      for (let j = 0; j < nr; j++) {
        const dst = (row + j) * S * nx;
        for (let i = 0; i < nxs; i++) {
          const o = 4 * (j * nxs + i), d = 4 * (dst + i * S);
          M[d] = out[o]; M[d + 1] = out[o + 1]; M[d + 2] = out[o + 2]; M[d + 3] = out[o + 3];
        }
      }
      this.rate.set(key, nr * nxs / Math.max(1, performance.now() - tc));
      row += nr; bands++;
      rows = Math.max(1, Math.min(nys, Math.floor(this.rate.get(key) * 500 / nxs)));
      opts.onBand?.(0, row / nys);
    }
    this.device.queue.writeBuffer(this.resBuf, 0, this.mirror);
    this.rowlo = 0;
    const wallMs = performance.now() - t0;
    const npts = row * nxs;
    if (cancelled) return { cancelled: true, timedOut, wallMs, bands, npts, rowsDone: row * S };
    return { gpuMs, wallMs, bands, npts, tsUsed: false, cancelled: false, timedOut: false };
  }
}
