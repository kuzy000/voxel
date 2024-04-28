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

const VOXEL_SIZE: f32 = 1.0f;
const VOXEL_DIM: i32 = 4;
const VOXEL_COUNT: i32 = VOXEL_DIM * VOXEL_DIM * VOXEL_DIM;
const VOXEL_TREE_DEPTH: i32 = 6;

struct Voxel {
    value: u32,
}

struct VoxelLeaf {
    mask: array<u32, 2>,
    voxels: array<Voxel, VOXEL_COUNT>,
}

struct VoxelNode {
    mask: array<u32, 2>,
    // Indices to either `voxel_nodes` or voxel_leafs` depending on the current depth
    indices: array<u32, VOXEL_COUNT>,
}

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<storage, read> voxel_nodes: array<VoxelNode>;
@group(0) @binding(3) var<storage, read> voxel_leafs: array<VoxelLeaf>;

const MAX_STEPS: i32 = 128;

struct RayMarchResult {
    normal: vec3<f32>,
    color: vec3<f32>,
    distance: f32,
}



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

    // voxel_tree[v] = u32(v % 2u == 0u);
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
    return abs(dot(p, n) + h);
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

    // res = min(res, sdf_plane(p, vec3(0f, 1f, 0f), 0.));
    // res = min(res, sdf_round_box(p - vec3(5., 5., 0.), vec3<f32>(size), 0.1f));
    // res = min(res, sdf_sphere(p - vec3(2f, 0f, 2f), 1.));
    // res = min(res, sdf_cube_tex(p - vec3(10f, 5f, 2f)));
    // res = min(res, sdf_cube_tex(p));

    res = min(res, sdf_sphere(p, 1.));
    

    return res;
}

fn normal_world(p: vec3<f32>) -> vec3<f32> {
    let step = vec2(0.001f, 0.f);
    
    let x = sdf_world(p + step.xyy) - sdf_world(p - step.xyy);
    let y = sdf_world(p + step.yxy) - sdf_world(p - step.yxy);
    let z = sdf_world(p + step.yyx) - sdf_world(p - step.yyx);
    
    return normalize(vec3(x, y, z));
}

fn ray_march_sdf(pos: vec3<f32>, dir: vec3<f32>) -> RayMarchResult {
    let step_size = .01f;
    let steps = 64;
    let min_dst = 0.01f;
    let max_dst = 1000.f;

    var ray_len = 0.f;

    for (var i = 0; i < steps; i++) {
        let p = pos + dir * ray_len;
        let dst = sdf_world(p);
        
        if (dst < min_dst) {
            let normal = normal_world(p);
            let color = normal * 0.5 + 0.5;
            return RayMarchResult(normal, color, ray_len);
        }
        
        if (dst > max_dst) {
            break;
        }
        
        ray_len += dst;
    }

    return RayMarchResult(vec3<f32>(), vec3<f32>(), 1e9f);
}

// fn get_voxel(ipos: vec3<i32>) -> bool {
//     let p = vec3<f32>(ipos) + vec3(0.5);
//     let d = min(max(-sdf_sphere(p, 7.5), sdf_box(p, vec3(6.0))), -sdf_sphere(p, 25.0));
//     return d < 0.;
// }

fn pos_to_idx(ipos: vec3<i32>) -> u32 {
    return u32(ipos.x * 4 * 4 + ipos.y * 4 + ipos.z);
}

fn get_voxel(ipos: vec3<i32>) -> bool {
    if (ipos.x < 0 || ipos.x >= 4) { return false; }
    if (ipos.y < 0 || ipos.y >= 4) { return false; }
    if (ipos.z < 0 || ipos.z >= 4) { return false; }

    let i = pos_to_idx(ipos);
    let leaf = voxel_leafs[0];
    
    if (i < 32) {
        return (leaf.mask[0] & (1u << (i - 0))) > 0;
    }
    else {
        return (leaf.mask[1] & (1u << (i - 32))) > 0;
    }
}

// Branchless Voxel Raycasting
// https://www.shadertoy.com/view/4dX3zl
// http://www.cse.yorku.ca/~amana/research/grid.pdf
fn ray_march_voxel_simple(pos: vec3<f32>, dir: vec3<f32>, offset: vec3<i32>, voxel_size: f32) -> RayMarchResult {
    // Apply offset and scale as if `voxel_size = 1.f`
    let local_pos = (pos - vec3<f32>(offset * VOXEL_DIM)) / voxel_size;

    var ipos = vec3<i32>(floor(local_pos));
    let istep = vec3<i32>(sign(dir));
    
    // let delta = abs(vec3(length(dir)) / dir);
    let delta = abs(1. / dir);
    var tmax = (sign(dir) * (vec3<f32>(ipos) - local_pos) + (sign(dir) * 0.5) + 0.5) * delta;
    
    var mask = vec3<bool>(false);
    
    for (var i = 0; i < MAX_STEPS; i++) {
        if (get_voxel(ipos)) {
            let normal = -normalize(vec3<f32>(mask) * vec3<f32>(istep));
            let color = normal * .5 + .5;
            let distance = min(tmax.x, min(tmax.y, tmax.z)) * voxel_size;
            return RayMarchResult(normal, color, distance);
        }

        mask = tmax.xyz <= min(tmax.yzx, tmax.zxy);
        
        tmax += vec3<f32>(mask) * delta;
        ipos += vec3<i32>(mask) * istep;
    }
    
    return RayMarchResult(vec3<f32>(), vec3<f32>(), 1e9f);
}

fn get_voxel_leaf(index: u32, ipos: vec3<i32>) -> bool {
    // TODO remove it
    if (ipos.x < 0 || ipos.x >= 4) { return false; }
    if (ipos.y < 0 || ipos.y >= 4) { return false; }
    if (ipos.z < 0 || ipos.z >= 4) { return false; }
    
//    var s = true;
//    s = s && ipos.x >= 1 && ipos.x <= 3;
//    s = s && ipos.y >= 1 && ipos.y <= 3;
//    s = s && ipos.z >= 1 && ipos.z <= 3;
//
//    return s;
//    
    let i = pos_to_idx(ipos);
    let leaf = voxel_leafs[index];
    
    if (i < 32) {
        return (leaf.mask[0] & (1u << (i - 0))) > 0;
    }
    else {
        return (leaf.mask[1] & (1u << (i - 32))) > 0;
    }
}

fn get_voxel_nodes(index: u32, ipos: vec3<i32>) -> bool {
    // TODO remove it
    if (ipos.x < 0 || ipos.x >= 4) { return false; }
    if (ipos.y < 0 || ipos.y >= 4) { return false; }
    if (ipos.z < 0 || ipos.z >= 4) { return false; }
    
    let i = pos_to_idx(ipos);
    let node = voxel_nodes[index];
    
    if (i < 32) {
        return (node.mask[0] & (1u << (i - 0))) > 0;
    }
    else {
        return (node.mask[1] & (1u << (i - 32))) > 0;
    }
}


fn ray_march_voxel_leaf(pos: vec3<f32>, dir: vec3<f32>, offset: vec3<i32>, voxel_size: f32, index: u32) -> RayMarchResult {
    // Apply offset and scale as if `voxel_size = 1.f`
    let local_pos = (pos - vec3<f32>(offset * VOXEL_DIM)) / voxel_size;

    var ipos = vec3<i32>(floor(local_pos));
    let istep = vec3<i32>(sign(dir));
    
    // let delta = abs(vec3(length(dir)) / dir);
    let delta = abs(1. / dir);
    var tmax = (sign(dir) * (vec3<f32>(ipos) - local_pos) + (sign(dir) * 0.5) + 0.5) * delta;
    
    var mask = vec3<bool>(false);
    
    for (var i = 0; i < MAX_STEPS; i++) {
        if (get_voxel_leaf(index, ipos)) {
            let normal = -normalize(vec3<f32>(mask) * vec3<f32>(istep));
            let color = normal * .5 + .5;
            let distance = min(tmax.x, min(tmax.y, tmax.z)) * voxel_size;
            return RayMarchResult(normal, color, distance);
        }

        mask = tmax.xyz <= min(tmax.yzx, tmax.zxy);
        
        tmax += vec3<f32>(mask) * delta;
        ipos += vec3<i32>(mask) * istep;
    }
    
    return RayMarchResult(vec3<f32>(), vec3<f32>(), 1e9f);
}

fn ray_march_voxel_node(pos: vec3<f32>, dir: vec3<f32>, offset: vec3<i32>, voxel_size: f32, index: u32) -> vec3<i32> {
    // Apply offset and scale as if `voxel_size = 1.f`
    let local_pos = (pos - vec3<f32>(offset * VOXEL_DIM)) / voxel_size;

    var ipos = vec3<i32>(floor(local_pos));
    let istep = vec3<i32>(sign(dir));
    
    // let delta = abs(vec3(length(dir)) / dir);
    let delta = abs(1. / dir);
    var tmax = (sign(dir) * (vec3<f32>(ipos) - local_pos) + (sign(dir) * 0.5) + 0.5) * delta;
    
    var mask = vec3<bool>(false);
    
    for (var i = 0; i < MAX_STEPS; i++) {
        if (get_voxel_nodes(index, ipos)) {
            return ipos;
        }

        mask = tmax.xyz <= min(tmax.yzx, tmax.zxy);
        
        tmax += vec3<f32>(mask) * delta;
        ipos += vec3<i32>(mask) * istep;
    }
    
    return vec3<i32>(-1);
}

struct Intersection {
    has: bool,
    t: f32,
}

// https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
fn ray_bbox(org: vec3f, dir: vec3f, lb: vec3f, rt: vec3f) -> Intersection {
    // r.dir is unit direction vector of ray
    var dirfrac = vec3f(0.f);
    dirfrac.x = 1.0f / dir.x;
    dirfrac.y = 1.0f / dir.y;
    dirfrac.z = 1.0f / dir.z;
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    let t1 = (lb.x - org.x)*dirfrac.x;
    let t2 = (rt.x - org.x)*dirfrac.x;
    let t3 = (lb.y - org.y)*dirfrac.y;
    let t4 = (rt.y - org.y)*dirfrac.y;
    let t5 = (lb.z - org.z)*dirfrac.z;
    let t6 = (rt.z - org.z)*dirfrac.z;

    let tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    let tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0)
    {
        return Intersection(false, tmax);
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        return Intersection(false, tmax);
    }

    return Intersection(true, tmin);
}

struct RayMarchFrame {
    index: u32,
    local_pos: vec3<f32>,
    ipos: vec3<i32>,
    tmax: vec3<f32>,
    tmax_prev: vec3<f32>,
}

fn is_inside(pos: vec3f, a: vec3f, b: vec3f) -> bool {
    if (pos.x < a.x) { return false; }
    if (pos.y < a.y) { return false; }
    if (pos.z < a.z) { return false; }
    if (pos.x > b.x) { return false; }
    if (pos.y > b.y) { return false; }
    if (pos.z > b.z) { return false; }
    
    return true;
}

// TODO: try to remove me
const BIAS: f32 = 0.01f;

fn get_voxel_size(depth: u32) -> f32 {
    // TODO: rewrite with loop?
    return VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - depth)));
}

fn ray_march_voxel(pos: vec3<f32>, dir: vec3<f32>) -> RayMarchResult {
    var frames: array<RayMarchFrame, VOXEL_TREE_DEPTH>;
    
    var inter_t = 0.f;
    if (!is_inside(pos, vec3f(0.), vec3f(get_voxel_size(0u)))) {
        let intersection = ray_bbox(pos, dir, vec3f(0.), vec3f(get_voxel_size(0u)));
        if (!intersection.has) {
            return RayMarchResult(vec3<f32>(), vec3<f32>(), 1e9f);
        }
        inter_t = intersection.t;
    }
    
    // Pos at which the ray enters the voxel tree
    // Add small bias so the in_pos0 is always inside the box
    let in_pos0 = pos + dir * (inter_t + BIAS);
    var depth = 0;

    // TODO remove it it. It should already be inside the box
    let global_pos = clamp(in_pos0, vec3f(0.), vec3f(get_voxel_size(0u)));

    // Apply offset and scale as if `voxel_size = 1.f`
    frames[depth].local_pos = global_pos / get_voxel_size(1u);

    frames[depth].index = 0u;
    frames[depth].ipos = vec3<i32>(floor(frames[depth].local_pos));
    let istep = vec3<i32>(sign(dir));
    
    // let delta = abs(vec3(length(dir)) / dir);
    let delta = abs(1. / dir);
    frames[depth].tmax = (sign(dir) * (vec3<f32>(frames[depth].ipos) - frames[depth].local_pos) + (sign(dir) * 0.5) + 0.5) * delta;
    frames[depth].tmax_prev = vec3f();
    
    var mask = vec3<bool>(false);
   // mask = frames[depth].tmax.xyz <= min(frames[depth].tmax.yzx, frames[depth].tmax.zxy);
    
    var diff = (frames[depth].local_pos - vec3<f32>(frames[depth].ipos));
    diff = abs(diff - 0.5f);
    mask = diff.xyz >= max(diff.xyz, max(diff.yzx, diff.zxy));

    // if (true) {
    //     // return RayMarchResult(vec3<f32>(), vec3<f32>(diff), 0.f);
    //     return RayMarchResult(vec3<f32>(), vec3<f32>(mask), 0.f);
    // }

    for (var i = 0; i < MAX_STEPS; i++) {
        let ipos = frames[depth].ipos;
        if (ipos.x < 0 || ipos.x >= VOXEL_DIM || ipos.y < 0 || ipos.y >= VOXEL_DIM || ipos.z < 0 || ipos.z >= VOXEL_DIM) {
            if (depth == 0) {
                break;
            }
            depth -= 1;
            let tmax = frames[depth].tmax;
            mask = tmax.xyz <= min(tmax.yzx, tmax.zxy);
            
            frames[depth].tmax_prev = frames[depth].tmax;
            frames[depth].tmax += vec3<f32>(mask) * delta;
            frames[depth].ipos += vec3<i32>(mask) * istep;
        }

        if (depth == VOXEL_TREE_DEPTH - 1) {
            let index = frames[depth].index;
            let ipos = frames[depth].ipos;
            let local_pos = frames[depth].local_pos;
            let tmax = frames[depth].tmax;

            let color = vec3f(local_pos);
            if (get_voxel_leaf(index, ipos)) {
                let normal = -normalize(vec3<f32>(mask) * vec3<f32>(istep));
                let color = normal * .5 + .5;
                // let color = vec3f(ipos) / 4.f;
                
                var distance = 0.f;
                var voxel_size = VOXEL_SIZE;
                for (var j = VOXEL_TREE_DEPTH - 1; j >= 0; j--) {
                    let tmax = frames[j].tmax_prev;
                    distance += min(tmax.x, min(tmax.y, tmax.z)) * voxel_size + inter_t;
                    voxel_size *= f32(VOXEL_DIM);
                }
                
                return RayMarchResult(normal, color, distance);
            }
        }
        else {
            let index = frames[depth].index;
            let ipos = frames[depth].ipos;
            if (get_voxel_nodes(index, ipos)) {
                let tmax = frames[depth].tmax;
                let tmax_prev = frames[depth].tmax_prev;
                let local_pos = frames[depth].local_pos;
                let distance = min(tmax_prev.x, min(tmax_prev.y, tmax_prev.z));
                
                // TODO try remove bias
                let current_local_pos = local_pos + dir * (distance + BIAS);
                let child_local_pos = (current_local_pos - vec3f(ipos)) * vec3f(VOXEL_DIM);

                depth += 1;

                frames[depth].index = voxel_nodes[index].indices[pos_to_idx(ipos)];
                frames[depth].local_pos = child_local_pos;
                frames[depth].ipos = vec3<i32>(floor(frames[depth].local_pos));
                frames[depth].tmax = (sign(dir) * (vec3<f32>(frames[depth].ipos) - frames[depth].local_pos) + (sign(dir) * 0.5) + 0.5) * delta;
                frames[depth].tmax_prev = vec3f();

                let nipos = frames[depth].ipos;
                let nindex = frames[depth].index;
                let nlocal_pos = frames[depth].local_pos;
                let ntmax = frames[depth].tmax;

                // if (nlocal_pos.x >= 0 && depth == 1) {
                //     let m = voxel_nodes[nindex].mask[1];
                //     let color = vec3f(m);
                //     return RayMarchResult(vec3f(), color, 0f);
                // }

                continue;
            }
        }
        
        
        let tmax = frames[depth].tmax;
        mask = tmax.xyz <= min(tmax.yzx, tmax.zxy);
        
        frames[depth].tmax_prev = frames[depth].tmax;
        frames[depth].tmax += vec3<f32>(mask) * delta;
        frames[depth].ipos += vec3<i32>(mask) * istep;
    }
    
    return RayMarchResult(vec3<f32>(), vec3<f32>(), 1e9f);
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

    let res_vox = ray_march_voxel(pos, dir);
    let res_sdf = ray_march_sdf(pos, dir);
    
    var color: vec4<f32>;
    if (res_vox.distance < res_sdf.distance) {
        color = vec4(res_vox.color, 1.);
    }
    else {
        color = vec4(res_sdf.color, 1.);
    }


    storageBarrier();
    textureStore(texture, location, color);
}