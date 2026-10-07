// Experimental 3D view: a stability chart over three parameters (X, Y, Z) computed as a
// brute-force grid by the same march, shown by ray marching (volume.wgsl). Orbit camera
// (drag: rotate, wheel: zoom), labelled box edges on the 2D overlay canvas.

const FOV = 35 * Math.PI / 180;

export function defaultCamera() {
  return { yaw: -0.85, pitch: 0.42, dist: 2.5 };
}

/** camera frame (world z up; the cube [-1/2, 1/2]^3: x = X, y = Y, z = Z) */
export function cameraFrame(cam, aspect) {
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const eye = [cam.dist * cp * Math.cos(cam.yaw), cam.dist * cp * Math.sin(cam.yaw), cam.dist * sp];
  const n = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
  const fwd = n(eye.map((x) => -x));
  const right = n([fwd[1] * 1 - fwd[2] * 0, fwd[2] * 0 - fwd[0] * 1, 0]);      // fwd × z
  const up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]];
  return { eye, fwd, right, up, tanh: Math.tan(FOV / 2), aspect };
}

/** world point -> [x, y] in a W x H box (null behind the camera) */
export function project(F, p, W, H) {
  const v = [p[0] - F.eye[0], p[1] - F.eye[1], p[2] - F.eye[2]];
  const zc = v[0] * F.fwd[0] + v[1] * F.fwd[1] + v[2] * F.fwd[2];
  if (zc <= 1e-6) return null;
  const xc = v[0] * F.right[0] + v[1] * F.right[1] + v[2] * F.right[2];
  const yc = v[0] * F.up[0] + v[1] * F.up[1] + v[2] * F.up[2];
  const nx = xc / (zc * F.tanh * F.aspect);
  const ny = yc / (zc * F.tanh);
  return [(nx + 1) / 2 * W, (1 - ny) / 2 * H];
}

/**
 * The combined field of a read-back 3D chart as bytes (C + 1)/2 * 255:
 * stable: C = max(σ, smin)/|smin| (at most -0.02), otherwise +0.6.
 */
export function volumeBytes(rb, n, smin, countOf) {
  const [nx, ny, nz] = n;
  const out = new Uint8Array(nx * ny * nz);
  const s = Math.abs(smin);
  for (let i = 0; i < out.length; i++) {
    let C = 0.6;
    if (countOf(rb, i) === 0) {
      C = (rb.flags[i] & 256) ? Math.min(-0.02, Math.max(rb.s[i], smin) / s) : -0.3;
    }
    out[i] = Math.round((C + 1) * 127.5);
  }
  return out;
}

export class Volume3D {
  static async create(device, format) {
    const r = await fetch('volume.wgsl', { cache: 'no-cache' });
    if (!r.ok) throw new Error('cannot load volume.wgsl');
    return new Volume3D(device, format, await r.text());
  }

  constructor(device, format, src) {
    this.device = device;
    this.format = format;
    const module = device.createShaderModule({ code: src, label: 'volume.wgsl' });
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.ubuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    this.tex = null;
    this.dims = null;
  }

  setVolume(dims, bytes) {
    const [nx, ny, nz] = dims;
    if (!this.tex || this.dims.join() !== dims.join()) {
      this.tex?.destroy();
      this.tex = this.device.createTexture({ size: [nx, ny, nz], dimension: '3d', format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dims = dims.slice();
      this.bg = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.ubuf } }, { binding: 1, resource: this.tex.createView() }, { binding: 2, resource: this.sampler }],
      });
    }
    this.device.queue.writeTexture({ texture: this.tex }, bytes, { bytesPerRow: nx, rowsPerImage: ny }, [nx, ny, nz]);
  }

  /** ray-march into a canvas context (W x H set by the caller); opts { fog (interior opacity 0..1), surf, bg: [r, g, b] } */
  draw(ctx, cam, opts) {
    if (!this.tex) return null;
    const canvas = ctx.canvas;
    if (ctx._configured !== this.device) {
      ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
      ctx._configured = this.device;
    }
    const F = cameraFrame(cam, canvas.width / canvas.height);
    const f = new Float32Array(24);
    f.set(F.eye, 0); f.set(F.right, 4); f.set(F.up, 8); f.set(F.fwd, 12);
    f[16] = F.tanh; f[17] = F.aspect; f[18] = opts.fog; f[19] = opts.surf;
    f.set(opts.bg, 20); f[23] = 1 / Math.max(...this.dims);
    this.device.queue.writeBuffer(this.ubuf, 0, f);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bg);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    return F;
  }
}

/**
 * Stability boundary of a read-back 3D chart as a closed binary STL.
 * Field (full Float32 precision): f = σ of the rightmost root (stable points: ≤ -1e-6,
 * unstable: ≥ +1e-6), clipped to [smin, -smin]; no σ: ±|smin|/2 by the count. The zero level of f
 * is the boundary (σ passes 0 continuously there), extracted by marching tetrahedra (6 per cell,
 * linear interpolation: a smooth surface, no voxel faces). The grid is padded with an unstable
 * layer and the vertices are clamped to the box, so the surface is closed at the box faces
 * (printable). Triangles are oriented outward (from the stable side to the unstable one).
 * ranges: [xr, yr, zr]; unit: true -> coordinates in the unit box [0, 1]^3.
 * -> { blob, triangles }
 */
export function boundarySTL(rb, dims, ranges, smin, countOf, unit) {
  const [nx, ny, nz] = dims;
  const m = Math.abs(smin);
  const X = nx + 2, Y = ny + 2, Z = nz + 2;
  // padding: 'very unstable' -- the zero crossing towards it lies at the boundary node itself
  // (t = v/(v - 1e30) ~ 0), which closes the surface exactly at the box faces
  const PAD = 1e30;
  const f = new Float32Array(X * Y * Z).fill(PAD);
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const q = i + nx * (j + ny * k);
        const Zc = countOf(rb, q);
        const has = (rb.flags[q] & 256) && Number.isFinite(rb.s[q]);
        let v;
        if (Zc === 0) v = has ? Math.min(-1e-6, Math.max(rb.s[q], -m)) : -m / 2;
        else v = has ? Math.max(1e-6, Math.min(rb.s[q], m)) : m / 2;
        f[(i + 1) + X * ((j + 1) + Y * (k + 1))] = v;
      }
    }
  }
  const [xr, yr, zr] = ranges;
  const map = (g, n, r) => {                  // padded grid coordinate -> axis units
    const t = (g - 1) / (n - 1);
    return unit ? t : r[0] + t * (r[1] - r[0]);
  };
  // the 8 cube corners and the 6 tetrahedra around the main diagonal 0-6
  const C = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
  const T = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]];
  const tri = [];
  const p = new Array(4), v = new Array(4);
  const P = new Array(8), V = new Array(8);
  const cut = (a0, b0) => {
    // always from the stable (negative) end: the same point for every tetrahedron sharing the edge
    const [a, b] = v[a0] < 0 ? [a0, b0] : [b0, a0];
    const t = v[a] / (v[a] - v[b]);
    return [p[a][0] + t * (p[b][0] - p[a][0]), p[a][1] + t * (p[b][1] - p[a][1]), p[a][2] + t * (p[b][2] - p[a][2])];
  };
  const same = (P1, P2) => P1[0] === P2[0] && P1[1] === P2[1] && P1[2] === P2[2];
  const emit = (A, B, Cc, pos, neg) => {
    if (same(A, B) || same(B, Cc) || same(A, Cc)) return;            // degenerate (at the box faces)
    // outward: the normal points from the stable (negative) to the unstable (positive) vertices
    const u = [B[0] - A[0], B[1] - A[1], B[2] - A[2]];
    const w = [Cc[0] - A[0], Cc[1] - A[1], Cc[2] - A[2]];
    const nrm = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const avg = (ids) => [0, 1, 2].map((d) => ids.reduce((s, i) => s + p[i][d], 0) / ids.length);
    const pp = avg(pos), pn = avg(neg);
    const dir = (pp[0] - pn[0]) * nrm[0] + (pp[1] - pn[1]) * nrm[1] + (pp[2] - pn[2]) * nrm[2];
    tri.push(dir >= 0 ? [A, B, Cc] : [A, Cc, B]);
  };
  for (let k = 0; k < Z - 1; k++) {
    for (let j = 0; j < Y - 1; j++) {
      for (let i = 0; i < X - 1; i++) {
        let neg = 0;
        for (let c = 0; c < 8; c++) {
          const [a, b, d] = C[c];
          V[c] = f[(i + a) + X * ((j + b) + Y * (k + d))];
          if (V[c] < 0) neg++;
        }
        if (neg === 0 || neg === 8) continue;
        for (let c = 0; c < 8; c++) P[c] = [map(i + C[c][0], nx, xr), map(j + C[c][1], ny, yr), map(k + C[c][2], nz, zr)];
        for (const t of T) {
          for (let q = 0; q < 4; q++) { p[q] = P[t[q]]; v[q] = V[t[q]]; }
          const ins = [0, 1, 2, 3].filter((q) => v[q] < 0);
          const out = [0, 1, 2, 3].filter((q) => v[q] >= 0);
          if (ins.length === 0 || ins.length === 4) continue;
          if (ins.length === 1 || ins.length === 3) {
            const [s] = ins.length === 1 ? ins : out;
            const o = (ins.length === 1 ? out : ins);
            emit(cut(s, o[0]), cut(s, o[1]), cut(s, o[2]), out, ins);
          } else {
            const [a, b] = ins;
            const [c1, d] = out;
            const q1 = cut(a, c1), q2 = cut(a, d), q3 = cut(b, d), q4 = cut(b, c1);
            emit(q1, q2, q3, out, ins);
            emit(q1, q3, q4, out, ins);
          }
        }
      }
    }
  }
  const buf = new ArrayBuffer(84 + 50 * tri.length);
  const dv = new DataView(buf);
  const head = 'NyquistGPU WebGPU: stability boundary (zero level of sigma), marching tetrahedra';
  for (let i = 0; i < 80; i++) dv.setUint8(i, i < head.length ? head.charCodeAt(i) & 127 : 32);
  dv.setUint32(80, tri.length, true);
  let o = 84;
  for (const [A, B, Cc] of tri) {
    const u = [B[0] - A[0], B[1] - A[1], B[2] - A[2]];
    const w = [Cc[0] - A[0], Cc[1] - A[1], Cc[2] - A[2]];
    let nrm = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const l = Math.hypot(...nrm) || 1;
    nrm = nrm.map((x) => x / l);
    for (const val of [...nrm, ...A, ...B, ...Cc]) { dv.setFloat32(o, val, true); o += 4; }
    dv.setUint16(o, 0, true);
    o += 2;
  }
  return { blob: new Blob([buf], { type: 'model/stl' }), triangles: tri.length };
}
