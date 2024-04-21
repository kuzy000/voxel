struct View {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    // viewport(x_origin, y_origin, width, height)
//    viewport: vec4<f32>,
//    frustum: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<storage, read_write> voxel_tree: array<u32>;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 8)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = vec3<i32>(invocation_id);
    
    let id = invocation_id;
    let num = num_workgroups;

    let v = id.x * num.y * num.z + id.y * num.x + id.z;

    let randomNumber = randomFloat(v);

    let alive = randomNumber > 0.95;
    let color = vec4<f32>(f32(v % 2u == 0u));

    voxel_tree[v] = u32(v % 2u == 0u);
}

fn is_alive(location: vec2<i32>, offset_x: i32, offset_y: i32) -> i32 {
    let value: vec4<f32> = textureLoad(texture, location + vec2<i32>(offset_x, offset_y));
    return i32(value.x);
}

fn count_alive(location: vec2<i32>) -> i32 {
    return is_alive(location, -1, -1) +
           is_alive(location, -1,  0) +
           is_alive(location, -1,  1) +
           is_alive(location,  0, -1) +
           is_alive(location,  0,  1) +
           is_alive(location,  1, -1) +
           is_alive(location,  1,  0) +
           is_alive(location,  1,  1);
}

const SIZE = vec2<i32>(1280, 720);

// https://iquilezles.org/articles/distfunctions/
fn sdf_plane(p: vec3<f32>, n: vec3<f32>, h: f32) -> f32 {
    return dot(p, n) + h;
}

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0);
}

fn sdf_round_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

fn sdf_cube_tex(p: vec3<f32>) -> f32 {
    let size = 4;
    
    var res = 1e9;
    
//    for (var x = 0; x < size; x++) {
//        for (var y = 0; y < size; y++) {
//            for (var z = 0; z < size; z++) {
//                let loc = vec3<i32>(x, y, z);
//                let value: vec4<f32> = textureLoad(cube, loc);
//                if (value.x > 0f) {
//                    res = min(res, sdf_box(p + vec3<f32>(loc), vec3<f32>(0.5f)));
//                }
//            }
//        }
//    }

    return res;
}

fn sdf_world(p: vec3<f32>) -> f32 {
    let size = 1.f;

    var res: f32 = 1e9;

    res = min(res, sdf_plane(p, vec3(0f, 1f, 0f), 5.));
    // res = min(res, sdf_round_box(p, vec3<f32>(size), 0.1f));
    // res = min(res, sdf_sphere(p - vec3(2f, 0f, 2f), 1.));
    // res = min(res, sdf_cube_tex(p - vec3(10f, 5f, 2f)));
    res = min(res, sdf_cube_tex(p));
    

    return res;
}

fn normal_world(p: vec3<f32>) -> vec3<f32> {
    let step = vec2(0.001f, 0.f);
    
    let x = sdf_world(p + step.xyy) - sdf_world(p - step.xyy);
    let y = sdf_world(p + step.yxy) - sdf_world(p - step.yxy);
    let z = sdf_world(p + step.yyx) - sdf_world(p - step.yyx);
    
    return normalize(vec3(x, y, z));
}

fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    let step_size = .01f;
    let steps = 64;
    let min_dst = 0.01f;
    let max_dst = 1000.f;

    var ray_len = 0.f;

    for (var i = 0; i < steps; i++) {
        let p = ro + rd * ray_len;
        let dst = sdf_world(p);
        
        if (dst < min_dst) {
            return normal_world(p) * 0.5 + 0.5;
        }
        
        if (dst > max_dst) {
            break;
        }
        
        ray_len += dst;
    }

    return vec3<f32>(0.f);
}

const VOXEL_SIZE: f32 = 1.;
const MAX_STEPS: i32 = 128;

// fn get_voxel(ipos: vec3<i32>) -> bool {
//     let p = vec3<f32>(ipos) + vec3(0.5);
//     let d = min(max(-sdf_sphere(p, 7.5), sdf_box(p, vec3(6.0))), -sdf_sphere(p, 25.0));
//     return d < 0.;
// }

fn get_voxel(ipos: vec3<i32>) -> bool {
    if (ipos.x < 0 || ipos.x >= 4) { return false; }
    if (ipos.y < 0 || ipos.y >= 4) { return false; }
    if (ipos.z < 0 || ipos.z >= 4) { return false; }

    let i = ipos.x * 4 * 4 + ipos.y * 4 + ipos.z;
    
    let v = voxel_tree[i];
    return v > 0u;
}

// Branchless Voxel Raycasting
// https://www.shadertoy.com/view/4dX3zl
// http://www.cse.yorku.ca/~amana/research/grid.pdf
fn traverse(pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var ipos = vec3<i32>(floor(pos));
    let istep = vec3<i32>(sign(dir));
    
    //let delta = abs(vec3(length(dir)) / dir);
    let delta = abs(1. / dir);
    var tmax = (sign(dir) * (vec3<f32>(ipos) - pos) + (sign(dir) * 0.5) + 0.5) * delta;
    
    var mask = vec3<bool>(false);
    
    for (var i = 0; i < MAX_STEPS; i++) {
        if (get_voxel(ipos)) {
            let normal = -normalize(vec3<f32>(mask) * vec3<f32>(istep));
            let color = normal * .5 + .5;
            return color;
        }

        mask = tmax.xyz <= min(tmax.yzx, tmax.zxy);
        
        tmax += vec3<f32>(mask) * delta;
        ipos += vec3<i32>(mask) * istep;
    }
    
    return vec3<f32>(0.f);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    
    let uv = (vec2<f32>(location) / vec2<f32>(SIZE)) * vec2(2.f, -2.f) - vec2(1.f, -1.f);

    let a4 = (view.inverse_view_proj * vec4(uv, -1.0f, 1.f));
    let a = a4.xyz / a4.w;

    let b4 = (view.inverse_view_proj * vec4(uv, 1.0f, 1.f));
    let b = b4.xyz / b4.w;
    
    let rd = normalize(b - a);
    
    // let a4 = (view.inverse_view * vec4(0f, 0f, 0f, 1f));
    // let a = a4.xyz / a4.w;

    // let right = (view.inverse_view * vec4(1f, 0f, 0f, 0f)).xyz;
    // let up = (view.inverse_view * vec4(0f, 1f, 0f, 0f)).xyz;
    // let forward = (view.inverse_view * vec4(0f, 0f, -1f, 0f)).xyz;
    // 
    // let rd = right * uv.x + up * uv.y + forward;
    
    let pos = a;
    let dir = rd;

    let c = traverse(pos, dir);

    let color = vec4<f32>(c, 1.f);

    storageBarrier();
    textureStore(texture, location, color);
}