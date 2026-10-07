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

  /** ray-march into a canvas context (W x H set by the caller); opts { fog, surf, bg: [r, g, b] } */
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
