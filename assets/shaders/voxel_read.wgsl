#define_import_path voxel_tracer::voxel_read

#import voxel_tracer::common::{
    DST_MAX,
    Intersection,
    RayMarchResult,
    ray_bbox,
    is_inside,
}

#import voxel_tracer::voxel_common::{
    VOXEL_SIZE,
    VOXEL_DIM,
    VOXEL_COUNT,
    VOXEL_TREE_DEPTH,
    VOXEL_MASK_LEN,
    VOXEL_SIZES,
    pos_to_idx,
}

struct VoxelInfo {
    nodes_len: u32,
    nodes_cap: u32,

    leafs_len: u32,
    leafs_cap: u32,
}

struct Voxel {
    color: u32,
}

struct VoxelLeaf {
    mask: array<u32, VOXEL_MASK_LEN>,
    voxels: array<Voxel, VOXEL_COUNT>,
}

struct VoxelNode {
    mask: array<u32, VOXEL_MASK_LEN>,
    // Indices to either `nodes` or `leafs` depending on the current depth
    indices: array<u32, VOXEL_COUNT>,
}

@group(0) @binding(0) var<storage, read_write> info : VoxelInfo;
@group(0) @binding(1) var<storage, read_write> nodes: array<VoxelNode>;
@group(0) @binding(2) var<storage, read_write> leafs: array<VoxelLeaf>;

const MAX_STEPS: i32 = 512;

fn get_voxel_leaf(index: u32, ipos: vec3<i32>) -> bool {
    let i = pos_to_idx(ipos);
    let leaf = &leafs[index];
    
    let ti = i >> 5;
    let oi = ti * 32;
    
    return ((*leaf).mask[ti] & (1u << (i - oi))) > 0;
}

fn get_voxel_nodes(index: u32, ipos: vec3<i32>) -> bool {
    let i = pos_to_idx(ipos);
    let node = &nodes[index];
    
    let ti = i >> 5;
    let oi = ti * 32;
    
    return ((*node).mask[ti] & (1u << (i - oi))) > 0;
}

struct RayMarchFrame {
    index: u32,
    ipos: vec3<i32>,
    tmax: vec3<f32>,
}

fn trace(pos: vec3<f32>, dir: vec3<f32>) -> RayMarchResult {
    var frames: array<RayMarchFrame, VOXEL_TREE_DEPTH>;
    
    var inter_t = 0.f;
    if (!is_inside(pos, vec3f(0.), vec3f(VOXEL_SIZES[0]))) {
        let intersection = ray_bbox(pos, dir, vec3f(0.), vec3f(VOXEL_SIZES[0u]));
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
    let global_pos = clamp(in_pos0, vec3f(0.), vec3f(VOXEL_SIZES[0u]));

    // Apply offset and scale as if `voxel_size = 1.f`
    let local_pos = global_pos / VOXEL_SIZES[1u];

    frames[depth].index = 0u;
    frames[depth].ipos = clamp(vec3<i32>(floor(local_pos)), vec3i(0), vec3i(VOXEL_DIM - 1));
    let istep = vec3<i32>(sign(dir));
    
    // let delta = abs(vec3(length(dir)) / dir);
    let delta = abs(1. / dir);
    frames[depth].tmax = (sign(dir) * (vec3<f32>(frames[depth].ipos) - local_pos) + (sign(dir) * 0.5) + 0.5) * delta;
    //frames[depth].tmax += vec3f(10.);
    // frames[depth].tmax_prev = frames[depth].tmax;
    
    var mask = vec3<bool>(false);
    var side_point = vec3f(); // TODO vec2?
   // mask = frames[depth].tmax.xyz <= min(frames[depth].tmax.yzx, frames[depth].tmax.zxy);
    
    var diff = (local_pos - vec3<f32>(frames[depth].ipos));
    // diff = abs(diff - 0.5f);
    
    var tmax_prev: vec3f;
    
    // inside the root box
    if (inter_t == 0.f) {
        // Calc the starting depth
        var lpos = local_pos;
        for (var i = 0; i < VOXEL_TREE_DEPTH - 1; i++) {
            let index = frames[depth].index;
            let ipos = frames[depth].ipos;

            if (get_voxel_nodes(index, ipos)) {
                let lpos_new = (lpos - vec3f(ipos)) * f32(VOXEL_DIM);

                let ipos_new = vec3<i32>(floor(lpos_new));
                let tmax_new = (sign(dir) * (vec3<f32>(ipos_new) - lpos_new) + (sign(dir) * 0.5) + 0.5) * delta;

                depth += 1;

                frames[depth].index = nodes[index].indices[pos_to_idx(ipos)];
                frames[depth].ipos = ipos_new;
                frames[depth].tmax = tmax_new;
                lpos = lpos_new;
            }
        }
    }
    else {
        side_point = global_pos / VOXEL_SIZES[0];
        let v = abs(side_point - 0.5);
        mask = v.xyz >= max(v.xyz, max(v.yzx, v.zxy));

        tmax_prev = frames[depth].tmax - vec3f(mask) * delta;
    }
    
    // if (false) {
    //     let color = vec3f(mask);
    //     return RayMarchResult(vec3<f32>(), side_point, 0.f);
    //     // return RayMarchResult(vec3<f32>(), vec3<f32>(mask), 0.f);
    // }
    
    var dst = 0.;
    
    let sign_dir01 = (sign(dir) * 0.5) + 0.5;
    
    let lpos_const = (1. - sign_dir01) * f32(VOXEL_DIM);
    let ipos_const = vec3i(1. - sign_dir01) * (VOXEL_DIM - 1);

    var distance = 0.;
    for (var i = 0; i < MAX_STEPS; i++) {
        var ipos = frames[depth].ipos;
        if (ipos.x < 0 || ipos.x >= VOXEL_DIM || ipos.y < 0 || ipos.y >= VOXEL_DIM || ipos.z < 0 || ipos.z >= VOXEL_DIM) {
            if (depth == 0) {
                break;
            }

            depth -= 1;
            let tmax = frames[depth].tmax;
            mask = tmax.xyz <= min(tmax.yzx, tmax.zxy);
            
            tmax_prev = frames[depth].tmax;
            frames[depth].tmax += vec3<f32>(mask) * delta;
            frames[depth].ipos += vec3<i32>(mask) * istep;
            continue;
        }

        let index = frames[depth].index;
        ipos = frames[depth].ipos;
        // let lpos = frames[depth].local_pos;

        if (depth == VOXEL_TREE_DEPTH - 1) {
            let index = frames[depth].index;
            let ipos = frames[depth].ipos;
            // let local_pos = frames[depth].local_pos;
            let tmax = frames[depth].tmax;

            // let color = vec3f(local_pos);
            if (get_voxel_leaf(index, ipos)) {
                let voxel = leafs[index].voxels[pos_to_idx(ipos)];

                let normal = -normalize(vec3<f32>(mask) * vec3<f32>(istep));
                let color = vec3f(unpack4x8unorm(voxel.color).xyz); //normal * .5 + .5;
                // let color = vec3f(ipos) / 4.f;
                
                var distance = 0.f;
                var voxel_size = VOXEL_SIZE;
                for (var j = VOXEL_TREE_DEPTH - 1; j >= 0; j--) {
                    //let tmax = frames[j].tmax_prev; accurate
                    let tmax = frames[j].tmax;
                    distance += min(tmax.x, min(tmax.y, tmax.z)) * voxel_size;
                    voxel_size *= f32(VOXEL_DIM);
                }
                
                return RayMarchResult(normal, color, distance + inter_t);
            }
        }
        else if (get_voxel_nodes(index, ipos)) {
            let tmax = tmax_prev;
            
            var ipos_new = vec3i(0);
            var lpos_new = vec3f(0);
            var tmax_new = vec3f(0.);
            var tmax_prev_new = vec3f(0.);
            
            //var lmask = tmax.xyz <= min(tmax.yzx, tmax.zxy);
            var lmask = mask;
            // if (!any(lmask)) {
            //     lmask[0] = true;
            // }
            
            {
                let len = dot(tmax, vec3f(lmask));
                
                let ir = 1. - (tmax - len) / delta;

                let lpos_new_t = f32(VOXEL_DIM) * ir;
                tmax_new = delta * (1. - fract(lpos_new_t));

                lpos_new = sign(dir) * lpos_new_t + lpos_const;
                ipos_new = vec3i(floor(lpos_new));
                
                lpos_new *= vec3f(!lmask);
                ipos_new *= vec3i(!lmask);

                lpos_new += vec3f(lmask) * lpos_const;
                ipos_new += vec3i(lmask) * ipos_const;
            }

            tmax_prev_new = tmax_new - vec3f(mask) * delta;
            

            depth += 1;

            frames[depth].index = nodes[index].indices[pos_to_idx(ipos)];
            // frames[depth].local_pos = lpos_new;
            frames[depth].ipos = ipos_new;
            frames[depth].tmax = tmax_new;
            tmax_prev = tmax_prev_new;
            
            // if (nlocal_pos.x >= 0 && depth == 1) {
            //     let m = voxel_nodes[nindex].mask[1];
            //     let color = vec3f(m);
            //     return RayMarchResult(vec3f(), color, 0f);
            // }

            continue;
        }
        
        let tmax = frames[depth].tmax;
        mask = tmax.xyz <= min(tmax.yzx, tmax.zxy);

        tmax_prev = frames[depth].tmax;
        frames[depth].tmax += vec3<f32>(mask) * delta;
        frames[depth].ipos += vec3<i32>(mask) * istep;
    }
    
    return RayMarchResult(vec3<f32>(), vec3<f32>(), DST_MAX);
}

fn place(pos: vec3i, voxel: Voxel) {
    var parent_idx = 0;
    for (var depth = 1; depth < VOXEL_TREE_DEPTH; depth++) {
        let lpos = (pos / VOXEL_SIZES[depth]) % VOXEL_DIM;

        let idx = pos_to_idx(lpos);
        let node = &nodes[parent_idx];
        
        let ti = idx >> 5;
        let oi = ti * 32;
        
        return ((*node).mask[ti] & (1u << (idx - oi))) > 0;
    }
}