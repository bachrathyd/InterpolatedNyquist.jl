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

// the smooth boundary mesh (grid coordinates g = node index; world = (g + 1/2)/n - 1/2, the
// texel centres of the volume texture): translucent, two-sided Lambert, premultiplied alpha
const MESH_WGSL = /* wgsl */ `
struct M {
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    p: vec4<f32>,       // tan(fov/2), aspect, opacity, -
    gs: vec4<f32>,      // 1/nx, 1/ny, 1/nz, -
}
@group(0) @binding(0) var<uniform> m: M;
struct VO {
    @builtin(position) pos: vec4<f32>,
    @location(0) n: vec3<f32>,
}
@vertex
fn vs(@location(0) g: vec3<f32>, @location(1) n: vec3<f32>) -> VO {
    let w = (g + vec3<f32>(0.5)) * m.gs.xyz - vec3<f32>(0.5);
    let v = w - m.eye.xyz;
    let zc = dot(v, m.fwd.xyz);
    var o: VO;
    o.pos = vec4<f32>(dot(v, m.right.xyz) / (m.p.x * m.p.y), dot(v, m.up.xyz) / m.p.x, 0.5 * zc, zc);
    o.n = n / m.gs.xyz;            // grid -> world normal (inverse transpose of the scaling)
    return o;
}
@fragment
fn fs(i: VO) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.4, -0.5, 0.75));
    let nl = length(i.n);
    var shade = 0.7;
    if (nl > 0.0) { shade = 0.3 + 0.7 * abs(dot(i.n / nl, light)); }
    let a = m.p.z;
    return vec4<f32>(vec3<f32>(0.93, 0.95, 1.0) * shade * a, a);
}`;

const BLEND = {
  color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

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
      fragment: { module, entryPoint: 'fs', targets: [{ format, blend: BLEND }] },
      primitive: { topology: 'triangle-list' },
    });
    const mm = device.createShaderModule({ code: MESH_WGSL, label: 'boundary mesh' });
    this.bgl = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    const meshPipe = (cullMode) => device.createRenderPipeline({
      layout,
      vertex: {
        module: mm, entryPoint: 'vs',
        buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }],
      },
      fragment: { module: mm, entryPoint: 'fs', targets: [{ format, blend: BLEND }] },
      primitive: { topology: 'triangle-list', cullMode, frontFace: 'ccw' },
    });
    this.meshBack = meshPipe('front');     // the far side of the boundary (drawn before the fog)
    this.meshFront = meshPipe('back');     // the near side (drawn after the fog)
    this.ubuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.mbuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.mbg = device.createBindGroup({ layout: this.bgl, entries: [{ binding: 0, resource: { buffer: this.mbuf } }] });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    this.tex = null;
    this.dims = null;
    this.mesh = null;
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
    // (the mesh is kept: the last completed one stays on screen until its successor is ready)
  }

  clearMesh() {
    if (this.mesh) { this.mesh.vb.destroy(); this.mesh.ib.destroy(); }
    this.mesh = null;
  }

  /** upload a boundary mesh (boundaryMesh) of a volume of grid size dims */
  setMesh(mesh, dims) {
    this.clearMesh();
    if (!mesh.idx.length) { this.mesh = { vb: { destroy() {} }, ib: { destroy() {} }, count: 0, dims }; return; }
    const nv = mesh.pos.length / 3;
    const inter = new Float32Array(6 * nv);
    for (let i = 0; i < nv; i++) {
      inter[6 * i] = mesh.pos[3 * i]; inter[6 * i + 1] = mesh.pos[3 * i + 1]; inter[6 * i + 2] = mesh.pos[3 * i + 2];
      inter[6 * i + 3] = mesh.nrm[3 * i]; inter[6 * i + 4] = mesh.nrm[3 * i + 1]; inter[6 * i + 5] = mesh.nrm[3 * i + 2];
    }
    const vb = this.device.createBuffer({ size: inter.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(vb, 0, inter);
    // on screen without the caps that close the surface at the box faces (kept in the STL)
    const [nx, ny, nz] = dims;
    const P = mesh.pos;
    const onFace = (a, b, c) => {
      for (let d = 0; d < 3; d++) {
        const e = [nx, ny, nz][d] - 1;
        const x = P[3 * a + d];
        if ((x === 0 || x === e) && P[3 * b + d] === x && P[3 * c + d] === x) return true;
      }
      return false;
    };
    const keep = new Uint32Array(mesh.idx.length);
    let k = 0;
    for (let t = 0; t < mesh.idx.length; t += 3) {
      const a = mesh.idx[t], b = mesh.idx[t + 1], c = mesh.idx[t + 2];
      if (onFace(a, b, c)) continue;
      keep[k++] = a; keep[k++] = b; keep[k++] = c;
    }
    const sub = keep.subarray(0, Math.max(k, 3));
    const ib = this.device.createBuffer({ size: sub.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(ib, 0, sub);
    this.mesh = { vb, ib, count: k, dims: dims.slice() };
  }

  /**
   * Draw into a canvas context (W x H set by the caller). opts { fog (interior opacity 0..1),
   * surf (boundary opacity), smooth (use the uploaded mesh for the boundary), bg: [r, g, b] }.
   * Order (premultiplied alpha): far side of the mesh, ray-marched fog, near side of the mesh.
   */
  draw(ctx, cam, opts) {
    if (!this.tex) return null;
    const canvas = ctx.canvas;
    if (ctx._configured !== this.device) {
      ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
      ctx._configured = this.device;
    }
    // smooth: the boundary only as a mesh (the last completed one; none yet: the fog alone)
    const useMesh = !!opts.smooth;
    const F = cameraFrame(cam, canvas.width / canvas.height);
    const f = new Float32Array(24);
    f.set(F.eye, 0); f.set(F.right, 4); f.set(F.up, 8); f.set(F.fwd, 12);
    f[16] = F.tanh; f[17] = F.aspect; f[18] = opts.fog; f[19] = useMesh ? 0 : opts.surf;
    f.set(opts.bg, 20); f[23] = 1 / Math.max(...this.dims);
    this.device.queue.writeBuffer(this.ubuf, 0, f);
    if (useMesh && this.mesh) {
      const g = new Float32Array(24);
      g.set(F.eye, 0); g.set(F.right, 4); g.set(F.up, 8); g.set(F.fwd, 12);
      g[16] = F.tanh; g[17] = F.aspect; g[18] = opts.surf;
      g[20] = 1 / this.mesh.dims[0]; g[21] = 1 / this.mesh.dims[1]; g[22] = 1 / this.mesh.dims[2];
      this.device.queue.writeBuffer(this.mbuf, 0, g);
    }
    const enc = this.device.createCommandEncoder();
    const [r, gg, b] = opts.bg;
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r, g: gg, b, a: 1 } }],
    });
    const drawMesh = (pipe) => {
      if (!useMesh || !this.mesh || !this.mesh.count || !(opts.surf > 0)) return;
      pass.setPipeline(pipe);
      pass.setBindGroup(0, this.mbg);
      pass.setVertexBuffer(0, this.mesh.vb);
      pass.setIndexBuffer(this.mesh.ib, 'uint32');
      pass.drawIndexed(this.mesh.count);
    };
    drawMesh(this.meshBack);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bg);
    pass.draw(3);
    drawMesh(this.meshFront);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    return F;
  }
}

/**
 * Stability boundary of a read-back 3D chart as a smooth, closed triangle mesh (generator: yields
 * after each z layer of cells, returns { pos (grid coordinates), nrm, idx (Uint32Array) }).
 * Field (full Float32 precision): f = σ of the rightmost root (stable points: ≤ -1e-6,
 * unstable: ≥ +1e-6), clipped to [smin, -smin]; no σ: ±|smin|/2 by the count. Its zero level is
 * the boundary (σ passes 0 continuously there), extracted by marching tetrahedra (the 6 Kuhn
 * tetrahedra of each cell around its main diagonal, linear interpolation along the edges: no
 * voxel faces). The grid is padded with a 'very unstable' layer (1e30: the crossing lies at the
 * boundary node itself), so the surface is closed at the box faces (printable). Vertices are
 * shared through their grid edge (watertight), triangles are oriented outward (from the stable
 * to the unstable side), vertex normals are the area-weighted averages of the face normals.
 */
export function* boundaryMesh(rb, dims, smin, countOf) {
  const [nx, ny, nz] = dims;
  const m = Math.abs(smin);
  const X = nx + 2, Y = ny + 2, Z = nz + 2;
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
  // cube corner offsets (bit 0: x, bit 1: y, bit 2: z) and the 6 Kuhn tetrahedra 0-..-7
  const CO = [0, 1, 3, 2, 4, 5, 7, 6];        // corner number -> offset bits (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)
  const T = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]];
  const off = (b) => (b & 1) + X * (((b >> 1) & 1) + Y * ((b >> 2) & 1));
  const pos = [];
  const idx = [];
  const vmap = new Map();
  const cn = new Int32Array(4);     // node index of each tetra corner
  const cb = new Int32Array(4);     // offset bits
  const cv = new Float64Array(4);   // field values
  const base = [0, 0, 0];
  // the vertex on the edge between tetra corners a, b (one stable, one not): shared by edge key
  const vert = (a, b) => {
    const sA = cv[a] < 0;
    const s = sA ? a : b, u = sA ? b : a;                // from the stable end
    const t = cv[s] / (cv[s] - cv[u]);
    let key;
    if (t < 1e-7) key = -1 - cn[s];                      // at the node (the padding): one vertex per node
    else key = (cn[a] < cn[b] ? cn[a] : cn[b]) * 8 + (cb[a] ^ cb[b]);
    let id = vmap.get(key);
    if (id !== undefined) return id;
    id = pos.length / 3;
    for (let d = 0; d < 3; d++) {
      const ps = base[d] + ((cb[s] >> d) & 1), pu = base[d] + ((cb[u] >> d) & 1);
      pos.push((t < 1e-7 ? ps : ps + t * (pu - ps)) - 1);   // padded -> node coordinates
    }
    vmap.set(key, id);
    return id;
  };
  // orientation from the midpoints of the three tetrahedron edges (exact half-integers, the same
  // combinatorial orientation as the cut points, also where those nearly coincide); twice the
  // midpoints: integers
  const tri = (a0, b0, a1, b1, a2, b2, ins, nin) => {
    const i0 = vert(a0, b0), i1 = vert(a1, b1), i2 = vert(a2, b2);
    if (i0 === i1 || i1 === i2 || i0 === i2) return;
    const m = (a, b, d) => ((cb[a] >> d) & 1) + ((cb[b] >> d) & 1);
    const Ax = m(a0, b0, 0), Ay = m(a0, b0, 1), Az = m(a0, b0, 2);
    const ux = m(a1, b1, 0) - Ax, uy = m(a1, b1, 1) - Ay, uz = m(a1, b1, 2) - Az;
    const wx = m(a2, b2, 0) - Ax, wy = m(a2, b2, 1) - Ay, wz = m(a2, b2, 2) - Az;
    const nx_ = uy * wz - uz * wy, ny_ = uz * wx - ux * wz, nz_ = ux * wy - uy * wx;
    let sx = 0, sy = 0, sz = 0;
    for (let q = 0; q < nin; q++) { const c = ins[q]; sx += 2 * (cb[c] & 1); sy += 2 * ((cb[c] >> 1) & 1); sz += 2 * ((cb[c] >> 2) & 1); }
    const dot = (nin * Ax - sx) * nx_ + (nin * Ay - sy) * ny_ + (nin * Az - sz) * nz_;
    if (dot >= 0) idx.push(i0, i1, i2); else idx.push(i0, i2, i1);
  };
  const ins = new Int32Array(4), out = new Int32Array(4);
  for (let k = 0; k < Z - 1; k++) {
    for (let j = 0; j < Y - 1; j++) {
      for (let i = 0; i < X - 1; i++) {
        const n0 = i + X * (j + Y * k);
        let neg = 0;
        for (let c = 0; c < 8; c++) if (f[n0 + off(CO[c])] < 0) neg++;
        if (neg === 0 || neg === 8) continue;
        base[0] = i; base[1] = j; base[2] = k;
        for (const t of T) {
          let ni = 0, no = 0;
          for (let q = 0; q < 4; q++) {
            cb[q] = CO[t[q]];
            cn[q] = n0 + off(cb[q]);
            cv[q] = f[cn[q]];
            if (cv[q] < 0) ins[ni++] = q; else out[no++] = q;
          }
          if (ni === 0 || ni === 4) continue;
          if (ni === 1) tri(ins[0], out[0], ins[0], out[1], ins[0], out[2], ins, ni);
          else if (ni === 3) tri(out[0], ins[0], out[0], ins[1], out[0], ins[2], ins, ni);
          else {
            const a = ins[0], b = ins[1], c1 = out[0], d = out[1];
            tri(a, c1, a, d, b, d, ins, ni);
            tri(a, c1, b, d, b, c1, ins, ni);
          }
        }
      }
    }
    yield k / (Z - 1);
  }
  const P = new Float32Array(pos);
  const I = new Uint32Array(idx);
  const N = new Float32Array(P.length);
  for (let t = 0; t < I.length; t += 3) {
    const a = 3 * I[t], b = 3 * I[t + 1], c = 3 * I[t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const wx = P[c] - P[a], wy = P[c + 1] - P[a + 1], wz = P[c + 2] - P[a + 2];
    const nx_ = uy * wz - uz * wy, ny_ = uz * wx - ux * wz, nz_ = ux * wy - uy * wx;   // area-weighted
    for (const v of [a, b, c]) { N[v] += nx_; N[v + 1] += ny_; N[v + 2] += nz_; }
  }
  for (let v = 0; v < N.length; v += 3) {
    const l = Math.hypot(N[v], N[v + 1], N[v + 2]) || 1;
    N[v] /= l; N[v + 1] /= l; N[v + 2] /= l;
  }
  return { pos: P, nrm: N, idx: I };
}

/** run a generator to completion synchronously */
export function runMesh(gen) {
  let r;
  do { r = gen.next(); } while (!r.done);
  return r.value;
}

/**
 * Binary STL of a boundary mesh: coordinates in axis units (node i -> min + i/(n-1) (max - min))
 * or, with unit, in [0, 1]^3. -> { blob, triangles }
 */
export function meshSTL(mesh, dims, ranges, unit) {
  const tri = mesh.idx.length / 3;
  const buf = new ArrayBuffer(84 + 50 * tri);
  const dv = new DataView(buf);
  const head = 'NyquistGPU WebGPU: stability boundary (zero level of sigma), marching tetrahedra';
  for (let i = 0; i < 80; i++) dv.setUint8(i, i < head.length ? head.charCodeAt(i) & 127 : 32);
  dv.setUint32(80, tri, true);
  const map = (v, d) => {
    const t = mesh.pos[3 * v + d] / (dims[d] - 1);
    return unit ? t : ranges[d][0] + t * (ranges[d][1] - ranges[d][0]);
  };
  let o = 84;
  for (let t = 0; t < tri; t++) {
    const V = [0, 1, 2].map((k) => mesh.idx[3 * t + k]);
    const A = [0, 1, 2].map((d) => map(V[0], d));
    const B = [0, 1, 2].map((d) => map(V[1], d));
    const C = [0, 1, 2].map((d) => map(V[2], d));
    const u = [B[0] - A[0], B[1] - A[1], B[2] - A[2]];
    const w = [C[0] - A[0], C[1] - A[1], C[2] - A[2]];
    let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const l = Math.hypot(...n) || 1;
    n = n.map((x) => x / l);
    for (const val of [...n, ...A, ...B, ...C]) { dv.setFloat32(o, val, true); o += 4; }
    dv.setUint16(o, 0, true);
    o += 2;
  }
  return { blob: new Blob([buf], { type: 'model/stl' }), triangles: tri };
}
