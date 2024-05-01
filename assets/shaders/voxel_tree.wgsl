#import bevy_pbr::{
    pbr_functions::alpha_discard,
    pbr_fragment::pbr_input_from_standard_material,
    mesh_view_bindings::view,
    view_transformations::position_world_to_view,
    view_transformations::position_clip_to_view,
    view_transformations::position_ndc_to_view,
    view_transformations::view_z_to_depth_ndc,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
}
#endif

#import bevy_render::{
    view::View,
}

#ifdef MESHLET_MESH_MATERIAL_PASS
#import bevy_pbr::meshlet_visibility_buffer_resolve::resolve_vertex_output
#endif

const VOXEL_SIZE: f32 = 1.0f;
const VOXEL_DIM: i32 = 4;
const VOXEL_COUNT: i32 = VOXEL_DIM * VOXEL_DIM * VOXEL_DIM;
const VOXEL_TREE_DEPTH: i32 = 6;

const DST_MAX = 1e9f;

struct Voxel {
    color: vec3f,
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


@group(2) @binding(100) var<storage, read> voxel_nodes: array<VoxelNode>;
@group(2) @binding(101) var<storage, read> voxel_leafs: array<VoxelLeaf>;


const MAX_STEPS: i32 = 256;

struct RayMarchResult {
    normal: vec3<f32>,
    color: vec3<f32>,
    distance: f32,
}


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

fn sdf_world(p: vec3<f32>) -> f32 {
    let size = 1.f;

    var res: f32 = DST_MAX;

    res = min(res, sdf_sphere(p, 0.5));
    res = min(res, sdf_box(p - vec3(3., 0., 0.), vec3f(1.)));
    res = min(res, sdf_box(p - vec3(0., 3., 0.), vec3f(1.)));
    res = min(res, sdf_box(p - vec3(0., 0., 3.), vec3f(1.)));
    

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

    return RayMarchResult(vec3<f32>(), vec3<f32>(), DST_MAX);
}


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
    
    return RayMarchResult(vec3<f32>(), vec3<f32>(), DST_MAX);
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
    let t1 = (lb.x - org.x) * dirfrac.x;
    let t2 = (rt.x - org.x) * dirfrac.x;
    let t3 = (lb.y - org.y) * dirfrac.y;
    let t4 = (rt.y - org.y) * dirfrac.y;
    let t5 = (lb.z - org.z) * dirfrac.z;
    let t6 = (rt.z - org.z) * dirfrac.z;

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

fn is_inside(pos: vec3f, a: vec3f, b: vec3f) -> bool {
    if (pos.x < a.x) { return false; }
    if (pos.y < a.y) { return false; }
    if (pos.z < a.z) { return false; }
    if (pos.x > b.x) { return false; }
    if (pos.y > b.y) { return false; }
    if (pos.z > b.z) { return false; }
    
    return true;
}


fn get_voxel_size(depth: u32) -> f32 {
    // TODO: rewrite with loop?
    return VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - depth)));
}

struct RayMarchFrame {
    index: u32,
    local_pos: vec3<f32>,
    ipos: vec3<i32>,
    tmax: vec3<f32>,
    tmax_prev: vec3<f32>,
}

fn ray_march_voxel(pos: vec3<f32>, dir: vec3<f32>) -> RayMarchResult {
    var frames: array<RayMarchFrame, VOXEL_TREE_DEPTH>;
    
    var inter_t = 0.f;
    if (!is_inside(pos, vec3f(0.), vec3f(get_voxel_size(0u)))) {
        let intersection = ray_bbox(pos, dir, vec3f(0.), vec3f(get_voxel_size(0u)));
        if (!intersection.has) {
            return RayMarchResult(vec3<f32>(), vec3<f32>(), DST_MAX);
        }
        inter_t = intersection.t;
    }
    
    // Pos at which the ray enters the voxel tree
    // Add small bias so the in_pos0 is always inside the box
    let in_pos0 = pos + dir * inter_t;
    var depth = 0;

    // TODO remove it it. It should already be inside the box
    let global_pos = clamp(in_pos0, vec3f(0.), vec3f(get_voxel_size(0u)));

    // Apply offset and scale as if `voxel_size = 1.f`
    frames[depth].local_pos = global_pos / get_voxel_size(1u);

    frames[depth].index = 0u;
    frames[depth].ipos = clamp(vec3<i32>(floor(frames[depth].local_pos)), vec3i(0), vec3i(VOXEL_DIM - 1));
    let istep = vec3<i32>(sign(dir));
    
    // let delta = abs(vec3(length(dir)) / dir);
    let delta = abs(1. / dir);
    frames[depth].tmax = (sign(dir) * (vec3<f32>(frames[depth].ipos) - frames[depth].local_pos) + (sign(dir) * 0.5) + 0.5) * delta;
    //frames[depth].tmax += vec3f(10.);
    // frames[depth].tmax_prev = frames[depth].tmax;
    
    var mask = vec3<bool>(false);
    var side_point = vec3f(); // TODO vec2?
   // mask = frames[depth].tmax.xyz <= min(frames[depth].tmax.yzx, frames[depth].tmax.zxy);
    
    var diff = (frames[depth].local_pos - vec3<f32>(frames[depth].ipos));
    // diff = abs(diff - 0.5f);
    
    
    // inside the root box
    if (inter_t == 0.f) {
    }
    else {
        side_point = global_pos / get_voxel_size(0u);
        let v = abs(side_point - 0.5);
        mask = v.xyz >= max(v.xyz, max(v.yzx, v.zxy));

        frames[depth].tmax_prev = frames[depth].tmax - vec3f(mask) * delta;
    }
    
    if (false) {
        let color = vec3f(mask);
        return RayMarchResult(vec3<f32>(), side_point, 0.f);
        // return RayMarchResult(vec3<f32>(), vec3<f32>(mask), 0.f);
    }
    
    var dst = 0.;

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
                let voxel = voxel_leafs[index].voxels[pos_to_idx(ipos)];

                let normal = -normalize(vec3<f32>(mask) * vec3<f32>(istep));
                let color = voxel.color; //normal * .5 + .5;
                // let color = vec3f(ipos) / 4.f;
                
                var distance = 0.f;
                var voxel_size = VOXEL_SIZE;
                for (var j = VOXEL_TREE_DEPTH - 1; j >= 0; j--) {
                    let tmax = frames[j].tmax_prev;
                    distance += min(tmax.x, min(tmax.y, tmax.z)) * voxel_size;
                    voxel_size *= f32(VOXEL_DIM);
                }
                
                return RayMarchResult(normal, color, distance + inter_t);
            }
        }
        else {
            let index = frames[depth].index;
            let ipos = frames[depth].ipos;
            let lpos = frames[depth].local_pos;
            if (get_voxel_nodes(index, ipos)) {
                let tmax = frames[depth].tmax_prev;
                
                var ipos_new = vec3i(0);
                var lpos_new = vec3f(0);
                var tmax_new = vec3f(0.);
                var tmax_prev_new = vec3f(0.);
                if (any(mask)) {
                    if (tmax.x < tmax.y) {
                        if (tmax.z < tmax.x) {
                            // zxy
                            let len = tmax.z;
                            
                            let irx = 1. - (tmax.x - len) / delta.x;
                            let iry = 1. - (tmax.y - len) / delta.y;

                            if (sign(dir.z) > 0) {
                                lpos_new.z = 0.;
                                ipos_new.z = 0;
                                tmax_new.z = delta.z;
                            }
                            else {
                                lpos_new.z = f32(VOXEL_DIM);
                                ipos_new.z = VOXEL_DIM - 1;
                                tmax_new.z = delta.z;
                            }

                            if (sign(dir.x) > 0) {
                                lpos_new.x = f32(VOXEL_DIM) * irx;
                                ipos_new.x = i32(floor(lpos_new.x));
                                tmax_new.x = delta.x * (1. - fract(lpos_new.x));
                            }
                            else {
                                lpos_new.x = f32(VOXEL_DIM) - f32(VOXEL_DIM) * irx;
                                ipos_new.x = i32(floor(lpos_new.x));
                                tmax_new.x = delta.x * fract(lpos_new.x);
                            }

                            if (sign(dir.y) > 0) {
                                lpos_new.y = f32(VOXEL_DIM) * iry;
                                ipos_new.y = i32(floor(lpos_new.y));
                                tmax_new.y = delta.y * (1. - fract(lpos_new.y));
                            }
                            else {
                                lpos_new.y = f32(VOXEL_DIM) - f32(VOXEL_DIM) * iry;
                                ipos_new.y = i32(floor(lpos_new.y));
                                tmax_new.y = delta.y * fract(lpos_new.y);
                            }

                        }
                        else {
                            // xzy
                            let len = tmax.x;
                            
                            let irz = 1. - (tmax.z - len) / delta.z;
                            let iry = 1. - (tmax.y - len) / delta.y;

                            if (sign(dir.x) > 0) {
                                lpos_new.x = 0.;
                                ipos_new.x = 0;
                                tmax_new.x = delta.x;
                            }
                            else {
                                lpos_new.x = f32(VOXEL_DIM);
                                ipos_new.x = VOXEL_DIM - 1;
                                tmax_new.x = delta.x;
                            }

                            if (sign(dir.z) > 0) {
                                lpos_new.z = f32(VOXEL_DIM) * irz;
                                ipos_new.z = i32(floor(lpos_new.z));
                                tmax_new.z = delta.z * (1. - fract(lpos_new.z));
                            }
                            else {
                                lpos_new.z = f32(VOXEL_DIM) - f32(VOXEL_DIM) * irz;
                                ipos_new.z = i32(floor(lpos_new.z));
                                tmax_new.z = delta.z * fract(lpos_new.z);
                            }

                            if (sign(dir.y) > 0) {
                                lpos_new.y = f32(VOXEL_DIM) * iry;
                                ipos_new.y = i32(floor(lpos_new.y));
                                tmax_new.y = delta.y * (1. - fract(lpos_new.y));
                            }
                            else {
                                lpos_new.y = f32(VOXEL_DIM) - f32(VOXEL_DIM) * iry;
                                ipos_new.y = i32(floor(lpos_new.y));
                                tmax_new.y = delta.y * fract(lpos_new.y);
                            }
                        }
                    }
                    else {
                        if (tmax.z < tmax.y) {
                            // zyx
                            let len = tmax.z;
                            
                            let irx = 1. - (tmax.x - len) / delta.x;
                            let iry = 1. - (tmax.y - len) / delta.y;

                            if (sign(dir.z) > 0) {
                                lpos_new.z = 0.;
                                ipos_new.z = 0;
                                tmax_new.z = delta.z;
                            }
                            else {
                                lpos_new.z = f32(VOXEL_DIM);
                                ipos_new.z = VOXEL_DIM - 1;
                                tmax_new.z = delta.z;
                            }

                            if (sign(dir.x) > 0) {
                                lpos_new.x = f32(VOXEL_DIM) * irx;
                                ipos_new.x = i32(floor(lpos_new.x));
                                tmax_new.x = delta.x * (1. - fract(lpos_new.x));
                            }
                            else {
                                lpos_new.x = f32(VOXEL_DIM) - f32(VOXEL_DIM) * irx;
                                ipos_new.x = i32(floor(lpos_new.x));
                                tmax_new.x = delta.x * fract(lpos_new.x);
                            }

                            if (sign(dir.y) > 0) {
                                lpos_new.y = f32(VOXEL_DIM) * iry;
                                ipos_new.y = i32(floor(lpos_new.y));
                                tmax_new.y = delta.y * (1. - fract(lpos_new.y));
                            }
                            else {
                                lpos_new.y = f32(VOXEL_DIM) - f32(VOXEL_DIM) * iry;
                                ipos_new.y = i32(floor(lpos_new.y));
                                tmax_new.y = delta.y * fract(lpos_new.y);
                            }
                        }
                        else {
                            // yzx
                            let len = tmax.y;
                            
                            let irx = 1. - (tmax.x - len) / delta.x;
                            let irz = 1. - (tmax.z - len) / delta.z;

                            if (sign(dir.y) > 0) {
                                lpos_new.y = 0.;
                                ipos_new.y = 0;
                                tmax_new.y = delta.y;
                            }
                            else {
                                lpos_new.y = f32(VOXEL_DIM);
                                ipos_new.y = VOXEL_DIM - 1;
                                tmax_new.y = delta.y;
                            }

                            if (sign(dir.x) > 0) {
                                lpos_new.x = f32(VOXEL_DIM) * irx;
                                ipos_new.x = i32(floor(lpos_new.x));
                                tmax_new.x = delta.x * (1. - fract(lpos_new.x));
                            }
                            else {
                                lpos_new.x = f32(VOXEL_DIM) - f32(VOXEL_DIM) * irx;
                                ipos_new.x = i32(floor(lpos_new.x));
                                tmax_new.x = delta.x * fract(lpos_new.x);
                            }

                            if (sign(dir.z) > 0) {
                                lpos_new.z = f32(VOXEL_DIM) * irz;
                                ipos_new.z = i32(floor(lpos_new.z));
                                tmax_new.z = delta.z * (1. - fract(lpos_new.z));
                            }
                            else {
                                lpos_new.z = f32(VOXEL_DIM) - f32(VOXEL_DIM) * irz;
                                ipos_new.z = i32(floor(lpos_new.z));
                                tmax_new.z = delta.z * fract(lpos_new.z);
                            }
                        }
                    }
                    tmax_prev_new = tmax_new - vec3f(mask) * delta;
                }
                else {
                    // Apply offset and scale as if `voxel_size = 1.f`
                    lpos_new = (lpos - vec3f(ipos)) * f32(VOXEL_DIM);

                    ipos_new = vec3<i32>(floor(lpos_new));
                    tmax_new = (sign(dir) * (vec3<f32>(ipos_new) - lpos_new) + (sign(dir) * 0.5) + 0.5) * delta;
                }
                
                //let local_pos = frames[depth].local_pos;
                //let distance = min(tmax.x, min(tmax.y, tmax.z));
                //
                //// TODO try remove bias
                //let current_local_pos = local_pos + dir * (distance + BIAS);
                //let child_local_pos = (current_local_pos - vec3f(ipos)) * vec3f(VOXEL_DIM);

                //depth += 1;

                //frames[depth].index = voxel_nodes[index].indices[pos_to_idx(ipos)];
                //frames[depth].local_pos = child_local_pos;
                //frames[depth].ipos = vec3<i32>(floor(frames[depth].local_pos));
                //frames[depth].tmax = (sign(dir) * (vec3<f32>(frames[depth].ipos) - frames[depth].local_pos) + (sign(dir) * 0.5) + 0.5) * delta;
                //frames[depth].tmax_prev = vec3f();

                //let nipos = frames[depth].ipos;
                //let nindex = frames[depth].index;
                //let nlocal_pos = frames[depth].local_pos;
                //let ntmax = frames[depth].tmax;

                // if (depth == 1) {
                //      let color = vec3f(lpos_new / f32(VOXEL_DIM));
                //      return RayMarchResult(vec3f(), color, 0f);
                // }

                depth += 1;

                frames[depth].index = voxel_nodes[index].indices[pos_to_idx(ipos)];
                frames[depth].local_pos = lpos_new;
                frames[depth].ipos = ipos_new;
                frames[depth].tmax = tmax_new;
                frames[depth].tmax_prev = tmax_prev_new;
                
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
    
    return RayMarchResult(vec3<f32>(), vec3<f32>(), DST_MAX);
}


struct FragmentOutputWithDepth {
    @location(0) normal: vec4<f32>,
    @location(1) motion_vector: vec2<f32>,
    @location(2) deferred: vec4<u32>,
    @location(3) deferred_lighting_pass_id: u32,
    @builtin(frag_depth) frag_depth: f32,
}


@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutputWithDepth {

    // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    
    // alpha discard
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);


    let pos4 = view.view * vec4f(0.f, 0.f, 0.f, 1.f);
    let pos = pos4.xyz / pos4.w;
    let dir = normalize(-pbr_input.V);

    let res_vox = ray_march_voxel(pos, dir);
    let res_sdf = ray_march_sdf(pos, dir);
    
    var res: RayMarchResult;
    if (res_vox.distance < res_sdf.distance) {
        res = res_vox;
    }
    else {
        res = res_sdf;
    }

    pbr_input.material.base_color = vec4f(res.color, 1.);
    pbr_input.N = res.normal;

    // TODO take into account distance from the camera to the near clipping plane
    let depth = view_z_to_depth_ndc(-res.distance);

    var out = deferred_output(in, pbr_input);
    var newOut: FragmentOutputWithDepth;
    newOut.frag_depth = depth;
    newOut.deferred = out.deferred;
    newOut.deferred_lighting_pass_id = out.deferred_lighting_pass_id;
    newOut.normal = vec4f(res.normal, 1.);
    newOut.motion_vector = out.motion_vector;

    return newOut;
}

