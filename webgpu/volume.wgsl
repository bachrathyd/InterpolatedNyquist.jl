// 3D view of a stability chart over three parameters (experimental): ray marching of a 3D
// texture holding the combined field C per grid point,
//     stable (Z = 0):  C = max(σ, σ_floor) / |σ_floor|  in [-1, 0)   (more negative: more stable)
//     otherwise:       C = +0.6                                       (not drawn)
// stored as r8unorm (C + 1)/2 and trilinearly filtered. The unstable region is not drawn; the
// stability boundary is the zero level of C (a translucent, shaded surface); the stable interior
// is a fog coloured by the σ colour map, its density growing with the stability margin -C.

struct U {
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    p: vec4<f32>,       // tan(fov/2), aspect, fog density, surface opacity
    bg: vec4<f32>,      // background colour; .w = 1 / grid size (gradient step)
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var smp: sampler;

struct VO {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VO {
    let x = select(-1.0, 3.0, i == 1u);
    let y = select(-1.0, 3.0, i == 2u);
    var o: VO;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

fn viridis(t0: f32) -> vec3<f32> {
    var tab = array<vec3<f32>, 9>(
        vec3<f32>(68.0, 1.0, 84.0), vec3<f32>(71.0, 44.0, 122.0), vec3<f32>(59.0, 81.0, 139.0),
        vec3<f32>(44.0, 113.0, 142.0), vec3<f32>(33.0, 144.0, 141.0), vec3<f32>(39.0, 173.0, 129.0),
        vec3<f32>(92.0, 200.0, 99.0), vec3<f32>(170.0, 220.0, 50.0), vec3<f32>(253.0, 231.0, 37.0));
    let x = clamp(t0, 0.0, 1.0) * 8.0;
    let i = min(u32(x), 7u);
    let w = x - f32(i);
    return mix(tab[i], tab[i + 1u], w) / 255.0;
}

fn field(q: vec3<f32>) -> f32 {
    return textureSampleLevel(vol, smp, q, 0.0).r * 2.0 - 1.0;
}

@fragment
fn fs(v: VO) -> @location(0) vec4<f32> {
    let eye = u.eye.xyz;
    let dir = normalize(u.fwd.xyz + v.uv.x * u.p.x * u.p.y * u.right.xyz + v.uv.y * u.p.x * u.up.xyz);
    // the unit cube [-1/2, 1/2]^3
    let inv = 1.0 / dir;
    let ta = (vec3<f32>(-0.5) - eye) * inv;
    let tb = (vec3<f32>(0.5) - eye) * inv;
    let tn = min(ta, tb);
    let tf = max(ta, tb);
    let t0 = max(max(max(tn.x, tn.y), tn.z), 0.0);
    let t1 = min(min(tf.x, tf.y), tf.z);
    if (t1 <= t0) { return vec4<f32>(u.bg.xyz, 1.0); }
    let dt = 1.0 / 320.0;
    let light = normalize(vec3<f32>(0.4, -0.5, 0.75));
    let h = u.bg.w;
    var col = vec3<f32>(0.0);
    var alpha = 0.0;
    var t = t0;
    var prev = field(eye + dir * t + 0.5);
    for (var k = 0; k < 600; k++) {
        if ((t > t1) || (alpha > 0.985)) { break; }
        let q = eye + dir * t + 0.5;
        let c = field(q);
        if ((k > 0) && ((c < 0.0) != (prev < 0.0))) {
            // the stability boundary C = 0: a translucent, shaded sheet
            let g = vec3<f32>(field(q + vec3<f32>(h, 0.0, 0.0)) - field(q - vec3<f32>(h, 0.0, 0.0)),
                              field(q + vec3<f32>(0.0, h, 0.0)) - field(q - vec3<f32>(0.0, h, 0.0)),
                              field(q + vec3<f32>(0.0, 0.0, h)) - field(q - vec3<f32>(0.0, 0.0, h)));
            let gl = length(g);
            var shade = 0.7;
            if (gl > 0.0) { shade = 0.35 + 0.65 * abs(dot(g / gl, light)); }
            let a = u.p.w;
            col += (1.0 - alpha) * a * vec3<f32>(0.93, 0.95, 1.0) * shade;
            alpha += (1.0 - alpha) * a;
        }
        if (c < 0.0) {
            let a = 1.0 - exp(-u.p.z * (-c) * dt);
            col += (1.0 - alpha) * a * viridis(1.0 + c);
            alpha += (1.0 - alpha) * a;
        }
        prev = c;
        t += dt;
    }
    return vec4<f32>(col + (1.0 - alpha) * u.bg.xyz, 1.0);
}
