#import voxel_tracer::common::RayMarchResult
#import voxel_tracer::common::DST_MAX
#import voxel_tracer::common::perlin_noise
#import voxel_tracer::sdf as sdf
#import voxel_tracer::voxel_common::{
    VOXEL_TREE_DEPTH,
    VOXEL_IDX_EMPTY,
    VOXEL_COUNT,
    VOXEL_SIZE,
    VOXEL_DIM,
    VOXEL_SIZES,
}
#import voxel_tracer::voxel_write as vox

@compute @workgroup_size(#{WG_X}, #{WG_Y}, #{WG_Z})
fn clear_world(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = invocation_id.x * #{WG_Y} * #{WG_Z} + invocation_id.y * #{WG_Y} + invocation_id.z;
    vox::clear(idx);
}

// @compute @workgroup_size(#{WG_X}, #{WG_Y}, #{WG_Z})
// fn draw_old(
//     @builtin(global_invocation_id) invocation_id: vec3<u32>,
//     @builtin(num_workgroups) num_workgroups: vec3<u32>
// ) {
//     let min = vox::push_constants.min.xyz;
//     let max = vox::push_constants.max.xyz;
//     let depth = vox::push_constants.depth;
// 
//     var ipos = vec3i(invocation_id) + min;
//     let ipos_real = ipos;
//     if (any(ipos >= max)) {
//         return;
//     }
//     
//     if (depth == 5) {
//         // Force previous LoD
//         // ipos = ipos / 2 * 2;
//     }
// 
//     let grad = vec3f(ipos - min) / vec3f(max - min - vec3i(1));
//     let color = pack4x8unorm(vec4(grad, 0.));
//     
//     let voxel_size = VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH - 1) - depth)));
//     
//     let vmin = vec3f(ipos) * voxel_size;
//     let vmax = vec3f(ipos + vec3i(1)) * voxel_size;
//     
//     let center = vmin + (vmax - vmin) * .5;
//     
//     let value = perlin_noise(center.xz, 0.01, 6, 0.5, 2.0, 13u);
//     
//     let world_value = (value * .5 + .5) * 100 + 50.;
//     
//     if ((center.y - world_value) < voxel_size * 2) {
//         vox::place(ipos_real, color);
//     }
// 
//     
// //    let vmin = vec3f(ipos) * voxel_size;
// //    let vmax = vec3f(ipos + vec3i(1)) * voxel_size;
// //    
// //    let center = vec3f(100, 100, 100);
// //    let radius = 30.f;
// //    
// //    var smin = center - vmin;
// //    var smax = center - vmax;
// //    smin *= smin;
// //    smax *= smax;
// //    
// //    let tmin = vec3f(center < vmin) * smin;
// //    let tmax = vec3f(center > vmax) * smax;
// //    
// //    var ds = radius * radius;
// //    ds -= tmin.x + tmin.y + tmin.z;
// //    ds -= tmax.x + tmax.y + tmax.z;
// //    
// //    if (ds > 0.) {
// //       vox::place(ipos, color);
// //    }
// 
// //    let p = (vec3f(ipos) + vec3f(.5)) * voxel_size;
// //    let d = sdf::sdf_world(p);
// //    let voxel_box = sdf::sdf_box(vec3f(0.), vec3f(voxel_size));
// //    
// //    let r = max(voxel_box, d);
// //    
// //    if (r <= sqrt(2.) * voxel_size * .6f) {
// //        vox::place(ipos, color);
// //    }
// }


var <workgroup> draw_buffer: array<u32, VOXEL_COUNT>;
var <workgroup> num_different: atomic<u32>;
var <workgroup> parent_ptr: u32;

fn draw_inner(ipos: vec3i) -> u32 {
    let min = vox::push_constants.world_min.xyz;
    let max = vox::push_constants.world_max.xyz;

    let grad = vec3f(ipos - min) / vec3f(max - min - vec3i(1));
    let color = pack4x8unorm(vec4f(grad, 0.));
    
    if (2 == 1) {
        return color;
    }

    let center = min + (max - min) / 2;
    let radius = (max - min) / 2 - 1;

    if (length(vec3f(ipos - center)) < f32(radius.x)) {
        return color;
    }

    return VOXEL_IDX_EMPTY;
}

@compute @workgroup_size(VOXEL_DIM, VOXEL_DIM, VOXEL_DIM)
fn draw_leafs(
    @builtin(local_invocation_id) lpos_u: vec3<u32>,
    @builtin(global_invocation_id) gpos_u: vec3<u32>,
    @builtin(workgroup_id) wpos_u: vec3<u32>,
    @builtin(num_workgroups) wsize_u: vec3<u32>
) {
    let min = vox::push_constants.min.xyz;
    let max = vox::push_constants.max.xyz;
    let world_min = vox::push_constants.world_min.xyz;
    let world_max = vox::push_constants.world_max.xyz;
    let depth = vox::push_constants.depth;

    let lpos = vec3i(lpos_u);
    let gpos = vec3i(gpos_u);
    let wpos = vec3i(wpos_u);
    let wsize = vec3i(wsize_u);

    let lidx = lpos.x * VOXEL_DIM * VOXEL_DIM + lpos.y * VOXEL_DIM + lpos.z;
    let widx = wpos.x * wsize.z * wsize.y + wpos.y * wsize.z + wpos.z;
    
    // Position in grid at current `depth`
    let ipos = (min / VOXEL_DIM) * VOXEL_DIM + gpos;
    
    let draw_area = &vox::draw_area_0;
    
    let q = vox::query(ipos, depth);
    parent_ptr = q.parent_idx;
    if (parent_ptr != VOXEL_IDX_EMPTY) {
        draw_buffer[lidx] = vox::leafs[q.parent_idx].voxels[q.idx].color;
    }
    else {
        draw_buffer[lidx] = VOXEL_IDX_EMPTY;
    }

    atomicStore(&num_different, 0u);

    if (all(ipos >= min) && all(ipos < max)) {
        draw_buffer[lidx] = draw_inner(ipos);
    }
    else {
        draw_buffer[lidx] = VOXEL_IDX_EMPTY;
    }

    workgroupBarrier();

    // TODO: LODs
    // let lidx_next = (lidx + 1) % (VOXEL_DIM * VOXEL_DIM * VOXEL_DIM);
    // if (draw_buffer[lidx] != draw_buffer[lidx_next]) {
    //     atomicAdd(&num_different, 1u);
    // }
    
    // Simply calculate occupied cells
    if (draw_buffer[lidx] != VOXEL_IDX_EMPTY) {
        atomicAdd(&num_different, 1u);
    }

    workgroupBarrier();
    
    if (atomicLoad(&num_different) == 0u) {
        let value = draw_buffer[lidx]; // the same for the whole chunk
        // if (value == VOXEL_IDX_EMPTY) { // it is always true
            (*draw_area)[widx] = VOXEL_IDX_EMPTY;
        // }

        // TODO: destroy leaf if allocated
        return;
    }
    
    // Allocate chunk in global memory
    if (lidx == 0 && parent_ptr == VOXEL_IDX_EMPTY) {
        parent_ptr = atomicAdd(&vox::info.leafs_len, 1u);
        (*draw_area)[widx] = parent_ptr;
    }
    
    let gptr = workgroupUniformLoad(&parent_ptr);
    vox::leafs[gptr].voxels[lidx].color = draw_buffer[lidx];
}

fn get_draw_area(draw_area_index: u32, index: u32) -> u32 {
    if (draw_area_index == 0) {
        return vox::draw_area_0[index];
    }
    else {
        return vox::draw_area_1[index];
    }
}

fn set_draw_area(draw_area_index: u32, index: u32, value: u32) {
    if (draw_area_index == 0) {
        vox::draw_area_0[index] = value;
    }
    else {
        vox::draw_area_1[index] = value;
    }
}


@compute @workgroup_size(VOXEL_DIM, VOXEL_DIM, VOXEL_DIM)
fn draw_nodes(
    @builtin(local_invocation_id) lpos_u: vec3<u32>,
    @builtin(global_invocation_id) gpos_u: vec3<u32>,
    @builtin(workgroup_id) wpos_u: vec3<u32>,
    @builtin(num_workgroups) wsize_u: vec3<u32>
) {
    let min = vox::push_constants.min.xyz;
    let max = vox::push_constants.max.xyz;
    let world_min = vox::push_constants.world_min.xyz;
    let world_max = vox::push_constants.world_max.xyz;
    let depth = vox::push_constants.depth;
    let wsize_prev = vox::push_constants.wsize_children;

    let lpos = vec3i(lpos_u);
    let gpos = vec3i(gpos_u);
    let wpos = vec3i(wpos_u);
    let wsize = vec3i(wsize_u);

    let lidx = lpos.x * VOXEL_DIM * VOXEL_DIM + lpos.y * VOXEL_DIM + lpos.z;
    let widx = wpos.x * wsize.z * wsize.y + wpos.y * wsize.z + wpos.z;
    
    // Position in grid at current `depth`
    let ipos = (min / VOXEL_DIM) * VOXEL_DIM + gpos;
    
    let draw_area_index = (u32(VOXEL_TREE_DEPTH) - 1u - depth) % 2u;
    let draw_area_index_children = (draw_area_index + 1u) % 2u;
    
    let q = vox::query(ipos, depth);
    parent_ptr = q.parent_idx;
    if (parent_ptr != VOXEL_IDX_EMPTY) {
        draw_buffer[lidx] = vox::nodes[q.parent_idx].indices[q.idx];
    }
    else {
        draw_buffer[lidx] = VOXEL_IDX_EMPTY;
    }

    atomicStore(&num_different, 0u);
    
    if (all(ipos >= min) && all(ipos < max)) {
        //if (depth == u32(VOXEL_TREE_DEPTH) - 2u) {
        if (1 == 1) {
            let cpos = ipos - min;

            let cidx = cpos.x * wsize_prev.y * wsize_prev.z + cpos.y * wsize_prev.z + cpos.z;
            draw_buffer[lidx] = get_draw_area(draw_area_index_children, u32(cidx));
        }
        else {
            let cpos = ipos - min;
            let cidx = cpos.x * i32(VOXEL_DIM) * i32(VOXEL_DIM) + cpos.y * i32(VOXEL_DIM) + cpos.z;
            draw_buffer[lidx] = get_draw_area(draw_area_index_children, u32(cidx));
        }
    }
    else {
        draw_buffer[lidx] = VOXEL_IDX_EMPTY;
    }

    workgroupBarrier();

    // TODO: LODs
    // let lidx_next = (lidx + 1) % (VOXEL_DIM * VOXEL_DIM * VOXEL_DIM);
    // if (draw_buffer[lidx] != draw_buffer[lidx_next]) {
    //     atomicAdd(&num_different, 1u);
    // }
    
    // Simply calculate occupied cells
    if (draw_buffer[lidx] != VOXEL_IDX_EMPTY) {
        atomicAdd(&num_different, 1u);
    }

    workgroupBarrier();
    
    if (atomicLoad(&num_different) == 0u) {
        let value = draw_buffer[lidx]; // the same for the whole chunk
        // if (value == VOXEL_IDX_EMPTY) { // it is always true
            set_draw_area(draw_area_index, u32(widx), u32(VOXEL_IDX_EMPTY));
        // }

        // TODO: destroy leaf if allocated
        return;
    }
    
    // Allocate chunk in global memory
    if (lidx == 0 && parent_ptr == VOXEL_IDX_EMPTY) {
        parent_ptr = atomicAdd(&vox::info.nodes_len, 1u);
        set_draw_area(draw_area_index, u32(widx), parent_ptr);
    }
    
    let gptr = workgroupUniformLoad(&parent_ptr);
    vox::nodes[gptr].indices[lidx] = draw_buffer[lidx];
}