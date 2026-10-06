// WebGPU host side of the NyquistGPU port: device setup, one compute pipeline per example
// (march.wgsl with the example's charD spliced in), banded dispatches (each submission stays
// far below the browser / OS GPU watchdog), GPU timing (timestamp queries if available) and
// the colouring render pass (display.wgsl).

const RES_BYTES = 16;            // struct Res { z: f32, s: f32, steps: u32, flags: u32 }
const PARAM_BYTES = 112;         // struct Params of march.wgsl
const VIEW_BYTES = 48;           // struct View of display.wgsl
const MAX_BANDS = 512;           // timestamp slots: 2 per band
const BIG = 3.0e38;

// march defaults of NyquistGPU.plan_sweep for :unwrap (the server's "F32" format)
export const MARCH_DEFAULTS = {
  sigma: 0.0, w0: 1e-9, wmax: 1e5, h0: 1e-2, tol: 0.3, hrel: 1.0, hmax: Infinity, wband: 0.0,
  maxsteps: 200000, qtrust: 4.0, growmax: 4.0,
};

export class WebGPUUnavailable extends Error {}

export class Engine {
  static async create(log = () => {}) {
    if (!('gpu' in navigator)) {
      throw new WebGPUUnavailable(window.isSecureContext
        ? 'This browser does not expose WebGPU (navigator.gpu is missing).'
        : 'WebGPU needs a secure context: open the page via https:// or http://localhost.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new WebGPUUnavailable('WebGPU is present, but no GPU adapter is available (blocklisted driver or GPU disabled).');
    const hasTS = adapter.features.has('timestamp-query');
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: hasTS ? ['timestamp-query'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 1 << 28),
        maxBufferSize: Math.min(lim.maxBufferSize, 1 << 28),
      },
    });
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    const fetchText = async (u) => {
      const r = await fetch(u);
      if (!r.ok) throw new Error(`cannot load ${u} (${r.status})`);
      return r.text();
    };
    let marchSrc, dispSrc;
    try {
      [marchSrc, dispSrc] = await Promise.all([fetchText('march.wgsl'), fetchText('display.wgsl')]);
    } catch (e) {
      throw new Error('Cannot load the WGSL shaders (' + e.message + '). Serve the folder over http, ' +
        'e.g. "python -m http.server" in webgpu/, instead of opening index.html as a file.');
    }
    const eng = new Engine(device, info, hasTS, marchSrc, dispSrc, log);
    return eng;
  }

  constructor(device, info, hasTS, marchSrc, dispSrc, log) {
    this.device = device;
    this.info = info;
    this.hasTS = hasTS;
    this.marchSrc = marchSrc;
    this.log = log;
    this.pipelines = new Map();
    this.lost = null;
    device.lost.then((e) => { this.lost = e; log('GPU device lost: ' + (e.message || e.reason)); });
    device.addEventListener?.('uncapturederror', (ev) => log('WebGPU error: ' + ev.error.message));
    this.paramBuf = device.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.viewBuf = device.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (hasTS) {
      this.qset = device.createQuerySet({ type: 'timestamp', count: 2 * MAX_BANDS });
      this.qres = device.createBuffer({ size: 16 * MAX_BANDS, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.qread = device.createBuffer({ size: 16 * MAX_BANDS, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    this.dispModule = device.createShaderModule({ code: dispSrc, label: 'display.wgsl' });
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.dispPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: this.dispModule, entryPoint: 'vs' },
      fragment: { module: this.dispModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.resBuf = null;
    this.npts = 0;
    this.nx = 0;
    this.ny = 0;
    this.rowlo = 0;
    this.maxInflight = 2;        // submissions in flight: the next band is queued while one runs
    this.bandLog = [];
  }

  get deviceName() {
    const i = this.info || {};
    return [i.vendor, i.architecture, i.device, i.description].filter((s) => s).join(' ') || 'unknown GPU';
  }

  async pipeline(ex) {
    if (this.pipelines.has(ex.key)) return this.pipelines.get(ex.key);
    const code = this.marchSrc.replace('//#CHARD#', ex.wgsl);
    const module = this.device.createShaderModule({ code, label: 'march:' + ex.key });
    const ci = await module.getCompilationInfo();
    const errs = ci.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error('WGSL compile error (' + ex.key + '): ' + errs.map((m) => `line ${m.lineNum}: ${m.message}`).join('; '));
    const p = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: ex.key });
    const entry = { pipeline: p, bgCache: new WeakMap() };
    this.pipelines.set(ex.key, entry);
    return entry;
  }

  ensureBuffers(nx, ny) {
    const n = nx * ny;
    if (this.resBuf && this.npts === n && this.nx === nx) return false;
    this.resBuf?.destroy();
    this.readBuf?.destroy();
    this.resBuf = this.device.createBuffer({ size: n * RES_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.readBuf = null;
    this.npts = n; this.nx = nx; this.ny = ny;
    this.dispBG = this.device.createBindGroup({
      layout: this.dispPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.viewBuf } }, { binding: 1, resource: { buffer: this.resBuf } }],
    });
    this.rowlo = ny;                      // nothing computed yet
    return true;
  }

  writeParams(job, row0, rows) {
    const m = { ...MARCH_DEFAULTS, ...job.march };
    const b = new ArrayBuffer(PARAM_BYTES);
    const f = new Float32Array(b);
    const u = new Uint32Array(b);
    const { nx, ny, xr, yr } = job;
    f[0] = xr[0]; f[1] = nx > 1 ? (xr[1] - xr[0]) / (nx - 1) : 0;
    f[2] = yr[0]; f[3] = ny > 1 ? (yr[1] - yr[0]) / (ny - 1) : 0;
    u[4] = nx; u[5] = ny; u[6] = row0; u[7] = rows;
    f[8] = m.w0; f[9] = m.wmax; f[10] = m.h0; f[11] = m.tol;
    f[12] = m.hrel; f[13] = Number.isFinite(m.hmax) ? m.hmax : BIG; f[14] = m.wband; f[15] = job.npow;
    f[16] = m.qtrust; f[17] = m.growmax; u[18] = m.maxsteps; f[19] = m.sigma;
    for (let i = 0; i < 8; i++) f[20 + i] = i < job.c.length ? job.c[i] : 0;
    this.device.queue.writeBuffer(this.paramBuf, 0, b);
  }

  /**
   * Compute one chart.  job = { ex, c, npow, nx, ny, xr, yr, march }
   * opts = { targetMs, onBand(rowlo, frac), isCancelled() }
   * -> { gpuMs, wallMs, bands, npts, tsUsed, cancelled }
   */
  async compute(job, opts = {}) {
    const dev = this.device;
    const { nx, ny } = job;
    const fresh = this.ensureBuffers(nx, ny);
    const { pipeline, bgCache } = await this.pipeline(job.ex);
    let bg = bgCache.get(this.resBuf);
    if (!bg) {
      bg = dev.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.paramBuf } }, { binding: 1, resource: { buffer: this.resBuf } }],
      });
      bgCache.set(this.resBuf, bg);
    }
    this.rowlo = fresh ? ny : 0;          // fresh buffer: rows below the progress are blank
    const target = opts.targetMs ?? 60;
    const minRows = Math.max(1, Math.ceil(ny / (MAX_BANDS - 8)));
    let rowsBand = Math.max(minRows, Math.min(ny, 8 * Math.round(Math.max(1, 4096 / nx))));
    let row = ny;
    let nb = 0;
    const inflight = [];
    let busyMs = 0;
    let lastEnd = 0;
    let cancelled = false;
    const t0 = performance.now();
    this.bandLog = [];
    const settle = async () => {
      const b = inflight.shift();
      const tEnd = await b.done;
      const dt = tEnd - Math.max(lastEnd, b.tSub);
      lastEnd = tEnd;
      busyMs += dt;
      this.bandLog.push({ rows: b.rows, dt, wait: tEnd - b.tSub });
      // band sizing: aim at `target` ms per submission (well below any GPU watchdog)
      const want = b.rows * target / Math.max(dt, 0.5);
      rowsBand = Math.max(minRows, Math.min(ny, Math.round(Math.min(want, 4 * b.rows, rowsBand * 4) / 8) * 8 || minRows));
      if (fresh) this.rowlo = b.row0;
      opts.onBand?.(b.row0, 1 - b.row0 / ny);
    };
    while (row > 0) {
      if (opts.isCancelled?.()) { cancelled = true; break; }
      const rows = Math.min(rowsBand, row);
      const row0 = row - rows;
      this.writeParams(job, row0, rows);
      const enc = dev.createCommandEncoder();
      const desc = {};
      if (this.hasTS && nb < MAX_BANDS) {
        desc.timestampWrites = { querySet: this.qset, beginningOfPassWriteIndex: 2 * nb, endOfPassWriteIndex: 2 * nb + 1 };
      }
      const pass = enc.beginComputePass(desc);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(rows / 8));
      pass.end();
      const tSub = performance.now();
      dev.queue.submit([enc.finish()]);
      inflight.push({ rows, row0, tSub, done: dev.queue.onSubmittedWorkDone().then(() => performance.now()) });
      nb++;
      row = row0;
      if (inflight.length >= this.maxInflight) await settle();
    }
    while (inflight.length) await settle();
    const wallMs = performance.now() - t0;
    if (cancelled) return { cancelled: true, wallMs, bands: nb, npts: nx * ny };
    this.rowlo = 0;
    let gpuMs = busyMs;
    let tsUsed = false;
    if (this.hasTS && nb <= MAX_BANDS) {
      const enc = dev.createCommandEncoder();
      enc.resolveQuerySet(this.qset, 0, 2 * nb, this.qres, 0);
      enc.copyBufferToBuffer(this.qres, 0, this.qread, 0, 16 * nb);
      dev.queue.submit([enc.finish()]);
      await this.qread.mapAsync(GPUMapMode.READ, 0, 16 * nb);
      const t = new BigUint64Array(this.qread.getMappedRange(0, 16 * nb).slice(0));
      this.qread.unmap();
      let ns = 0n;
      let ok = true;
      for (let i = 0; i < nb; i++) {
        const d = t[2 * i + 1] - t[2 * i];
        if (t[2 * i + 1] < t[2 * i] || d === 0n) ok = false;
        ns += d;
      }
      if (ok) { gpuMs = Number(ns) / 1e6; tsUsed = true; }
    }
    return { gpuMs, wallMs, bands: nb, npts: nx * ny, tsUsed, cancelled: false };
  }

  /** Colour the current result buffer into a canvas context (f x f box filter). */
  draw(ctx, view) {
    if (!this.resBuf) return;
    const f = view.f;
    const dw = Math.ceil(this.nx / f);
    const dh = Math.ceil(this.ny / f);
    const canvas = ctx.canvas;
    if (canvas.width !== dw || canvas.height !== dh) { canvas.width = dw; canvas.height = dh; }
    if (!ctx._configured || ctx._configured !== this.device) {
      ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
      ctx._configured = this.device;
    }
    const b = new ArrayBuffer(VIEW_BYTES);
    const u = new Uint32Array(b);
    const fl = new Float32Array(b);
    u[0] = this.nx; u[1] = this.ny; u[2] = f; u[3] = dw;
    u[4] = dh; u[5] = view.boundary ? 1 : 0; u[6] = view.flags ? 1 : 0; u[7] = this.rowlo;
    fl[8] = view.smin; fl[9] = view.zcap;
    this.device.queue.writeBuffer(this.viewBuf, 0, b);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.dispPipeline);
    pass.setBindGroup(0, this.dispBG);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Copy the results to the host -> { z: Float32Array, s: Float32Array, steps: Uint32Array, flags: Uint32Array } */
  async readback() {
    const size = this.npts * RES_BYTES;
    const rb = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.resBuf, 0, rb, 0, size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const raw = rb.getMappedRange().slice(0);
    rb.unmap();
    rb.destroy();
    const f = new Float32Array(raw);
    const u = new Uint32Array(raw);
    const n = this.npts;
    const out = { nx: this.nx, ny: this.ny, z: new Float32Array(n), s: new Float32Array(n), steps: new Uint32Array(n), flags: new Uint32Array(n) };
    for (let i = 0; i < n; i++) {
      out.z[i] = f[4 * i]; out.s[i] = f[4 * i + 1]; out.steps[i] = u[4 * i + 2]; out.flags[i] = u[4 * i + 3];
    }
    return out;
  }
}

/** Z of a read-back point as fetch_result reports it: -1 where the march failed. */
export function countOf(r, i) {
  return (r.flags[i] & 128) ? -1 : Math.round(r.z[i]);
}
