// Colouring pass (gpu/interactive/server.jl, k_display! / pixel_colour): one display pixel =
// an f x f block of chart points (box filter).
//   Z == 0 (stable): viridis of σ, t = 1 - σ/smin   (bright: σ near 0, dark: σ <= smin)
//   Z >= 1         : reds by the count, capped at zcap
//   failed march / Z < 0: grey;   white: stability boundary (stable and unstable points in the
//   block or its right / lower neighbours);   optional magenta tint of flagged points.

struct Res {
    z: f32,
    s: f32,
    steps: u32,
    flags: u32,
}

struct View {
    nx: u32, ny: u32, f: u32, dw: u32,
    dh: u32, bnd: u32, showflags: u32, rowlo: u32,   // rows iy < rowlo: not computed yet
    smin: f32, zcap: f32, pad1: f32, pad2: f32,
}

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> R: array<Res>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = select(-1.0, 3.0, i == 1u);
    let y = select(-1.0, 3.0, i == 2u);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn viridis(t0: f32) -> vec3<f32> {
    var tab = array<vec3<f32>, 9>(
        vec3<f32>(68.0, 1.0, 84.0), vec3<f32>(71.0, 44.0, 122.0), vec3<f32>(59.0, 81.0, 139.0),
        vec3<f32>(44.0, 113.0, 142.0), vec3<f32>(33.0, 144.0, 141.0), vec3<f32>(39.0, 173.0, 129.0),
        vec3<f32>(92.0, 200.0, 99.0), vec3<f32>(170.0, 220.0, 50.0), vec3<f32>(253.0, 231.0, 37.0));
    let x = clamp(t0, 0.0, 1.0) * 8.0;
    let i = min(u32(x), 7u);
    let w = x - f32(i);
    return mix(tab[i], tab[i + 1u], w);
}

fn reds(t0: f32) -> vec3<f32> {
    var tab = array<vec3<f32>, 4>(
        vec3<f32>(252.0, 187.0, 161.0), vec3<f32>(251.0, 106.0, 74.0),
        vec3<f32>(203.0, 24.0, 29.0), vec3<f32>(103.0, 0.0, 13.0));
    let x = clamp(t0, 0.0, 1.0) * 3.0;
    let i = min(u32(x), 2u);
    let w = x - f32(i);
    return mix(tab[i], tab[i + 1u], w);
}

// -2: not computed yet, -1: no count (failed march or Z < 0), else Z
fn count_of(r: Res, iy: u32) -> i32 {
    if (iy < V.rowlo) { return -2; }
    if ((r.flags & 128u) != 0u) { return -1; }
    let Z = i32(round(r.z));
    return select(Z, -1, Z < 0);
}

fn colour(r: Res, Z: i32) -> vec3<f32> {
    if (Z == -2) { return vec3<f32>(28.0, 30.0, 36.0); }
    if (Z < 0) { return vec3<f32>(128.0, 128.0, 128.0); }
    if (Z == 0) {
        var t = 0.0;
        if ((r.flags & 256u) != 0u) { t = r.s / V.smin * -1.0 + 1.0; }
        return viridis(t);
    }
    return reds((min(f32(Z), V.zcap) - 1.0) / max(V.zcap - 1.0, 1.0));
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let u = u32(pos.x);
    let v = u32(pos.y);
    let f = V.f;
    var acc = vec3<f32>(0.0);
    var cnt = 0.0;
    var st = false;
    var un = false;
    var fl = false;
    for (var bj = 0u; bj <= f; bj++) {
        for (var bi = 0u; bi <= f; bi++) {
            let i = u * f + bi;
            let jr = v * f + bj;                       // row counted from the top
            if ((i < V.nx) && (jr < V.ny)) {
                let iy = V.ny - 1u - jr;
                let r = R[i + iy * V.nx];
                let Z = count_of(r, iy);
                st = st || (Z == 0);
                un = un || (Z > 0);
                if ((bi < f) && (bj < f)) {
                    acc += colour(r, Z);
                    cnt += 1.0;
                    fl = fl || (((r.flags & 15u) != 0u) && (Z != -2));
                }
            }
        }
    }
    var c = acc / max(cnt, 1.0);
    if ((V.bnd != 0u) && st && un) {
        c = vec3<f32>(255.0);
    } else if ((V.showflags != 0u) && fl) {
        c = vec3<f32>(0.5 * c.x + 127.5, 0.5 * c.y, 0.5 * c.z + 127.5);
    }
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(255.0)) / 255.0, 1.0);
}
