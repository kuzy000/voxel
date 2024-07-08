#import voxel_tracer::common::RayMarchResult
#import voxel_tracer::common::DST_MAX
#import voxel_tracer::common::{
    perlin_noise,
    perlin_noise3,
}
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

var <workgroup> draw_buffer: array<u32, VOXEL_COUNT>;
var <workgroup> num_occupied: atomic<u32>;
var <workgroup> num_different: atomic<u32>;
var <workgroup> num_divided: atomic<u32>;
var <workgroup> sum_r: atomic<u32>;
var <workgroup> sum_g: atomic<u32>;
var <workgroup> sum_b: atomic<u32>;
var <workgroup> parent_ptr: u32;
var <workgroup> lod_ptr: u32;


fn draw_inner_sphere(ipos: vec3i, current: u32) -> u32 {
    if (current != VOXEL_IDX_EMPTY) {
        return current;
    }

    let min = vox::push_constants.world_min.xyz;
    let max = vox::push_constants.world_max.xyz;

    let grad = vec3f(ipos - min) / vec3f(max - min - vec3i(1));
    //let color = pack4x8unorm(vec4f(grad, 0.));
    let color = pack4x8unorm(vec4f(1., 1., 1., 0.));
    
    if (2 == 1) {
        return color;
    }

    let center = min + (max - min) / 2;
    let radius = (max - min) / 2 ;

    if (length(vec3f(ipos - center)) < f32(radius.x)) {
        return color;
    }

    return VOXEL_IDX_EMPTY;
}

fn draw_inner(ipos: vec3i, current: u32) -> u32 {
    var PALETTE = array<u32, 8>(
        0xd53e4f,
        0xf46d43,
        0xfdae61,
        0xfee08b,
        0xe6f598,
        0xabdda4,
        0x66c2a5,
        0x3288bd,
    );

    if (current != VOXEL_IDX_EMPTY) {
        return current;
    }

    let min = vox::push_constants.world_min.xyz;
    let max = vox::push_constants.world_max.xyz;

    let grad = vec3f(ipos - min) / vec3f(max - min - vec3i(1));
    //let color = pack4x8unorm(vec4f(grad, 0.));
    let color = pack4x8unorm(vec4f(1., 1., 1., 0.));
    
    if (2 == 1) {
        return color;
    }
    
    
    let p = vec3f(ipos);
    let lpos = p / vec3f(max - min);
    
    let lands = perlin_noise(p.xz, .0005, 6, .5, 2., 123u) * .5 + .5;
    let caves = perlin_noise3(p, .002, 6, .5, 2., 123u) * .5 + .5;

    if (lands > lpos.y && caves > 0.5) {
        let c = (caves - .5) * 2.;
        let cv = u32(c * 9.);
        let cu = PALETTE[cv];
        
        let b = (cu >>  0u) & 0xFFu;
        let g = (cu >>  8u) & 0xFFu;
        let r = (cu >> 16u) & 0xFFu;
        let col = vec3f(vec3u(r, g, b)) / 255.;

        return pack4x8unorm(vec4f(col, 0.));
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
    
    let q = vox::query(ipos, depth);
    parent_ptr = q.parent_idx;

    if (parent_ptr != VOXEL_IDX_EMPTY) {
        draw_buffer[lidx] = vox::leafs[q.parent_idx].voxels[q.idx].color;
    }
    else {
        draw_buffer[lidx] = q.value_if_empty;
    }

    atomicStore(&num_different, 0u);
    atomicStore(&num_occupied, 0u);
    atomicStore(&sum_r, 0u);
    atomicStore(&sum_g, 0u);
    atomicStore(&sum_b, 0u);

    if (all(ipos >= min) && all(ipos < max)) {
        draw_buffer[lidx] = draw_inner(ipos, draw_buffer[lidx]);
    }

    workgroupBarrier();

    // TODO: LODs
    let lidx_next = (lidx + 1) % (VOXEL_DIM * VOXEL_DIM * VOXEL_DIM);
    if (draw_buffer[lidx] != draw_buffer[lidx_next]) {
        atomicAdd(&num_different, 1u);
    }
    
    // Simply calculate occupied cells
    if (draw_buffer[lidx] != VOXEL_IDX_EMPTY) {
        atomicAdd(&num_occupied, 1u);

        let cur = vec3u(unpack4x8unorm(draw_buffer[lidx]).xyz * 255.f);
        atomicAdd(&sum_r, cur.x);
        atomicAdd(&sum_g, cur.y);
        atomicAdd(&sum_b, cur.z);
    }

    workgroupBarrier();
    
    let num_occupied_v = atomicLoad(&num_occupied);
    let num_different_v = atomicLoad(&num_different);

    if (num_occupied_v == 0u) {
        // let value = draw_buffer[lidx]; // the same for the whole chunk
        // if (value == VOXEL_IDX_EMPTY) { // it is always true
            // (*draw_area)[widx] = VOXEL_IDX_EMPTY;
        // }

        vox::set_draw_area(0u, u32(widx), vox::DrawResult(VOXEL_IDX_EMPTY, VOXEL_IDX_EMPTY));
        
        // TODO: destroy leaf if allocated
        return;
    }
    
    let sum_color_u = vec3u(atomicLoad(&sum_r), atomicLoad(&sum_g), atomicLoad(&sum_b));
    let mean_color = vec3f(sum_color_u) / f32(num_occupied_v) / 255.f;
    var mean_color_u = pack4x8unorm(vec4f(mean_color, 0.));
//    if (num_occupied_v < u32(VOXEL_COUNT) / 16u) {
//        mean_color_u = VOXEL_IDX_EMPTY;
//    }

    if (num_different_v == 0u) {
        // let value = draw_buffer[lidx]; // the same for the whole chunk
        // if (value == VOXEL_IDX_EMPTY) { // it is always true
            // (*draw_area)[widx] = VOXEL_IDX_EMPTY;
        // }

        vox::set_draw_area(0u, u32(widx), vox::DrawResult(VOXEL_IDX_EMPTY, mean_color_u));
        
        // TODO: destroy leaf if allocated
        return;
    }

    // Allocate chunk in global memory
    if (lidx == 0 && parent_ptr == VOXEL_IDX_EMPTY) {
        parent_ptr = atomicAdd(&vox::info.leafs_len, 1u);
    }
    
    let gptr = workgroupUniformLoad(&parent_ptr);
    vox::set_draw_area(0u, u32(widx), vox::DrawResult(gptr, mean_color_u));
    vox::leafs[gptr].voxels[lidx].color = draw_buffer[lidx];
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

    var child_ptr: u32 = VOXEL_IDX_EMPTY;
    if (parent_ptr != VOXEL_IDX_EMPTY) {
        child_ptr = vox::nodes[q.parent_idx].indices[q.idx];
        lod_ptr = vox::nodes[q.parent_idx].leaf;
        draw_buffer[lidx] = vox::leafs[lod_ptr].voxels[q.idx].color;
    }
    else {
        child_ptr = VOXEL_IDX_EMPTY;
        lod_ptr = VOXEL_IDX_EMPTY;
        draw_buffer[lidx] = q.value_if_empty;
    }

    atomicStore(&num_different, 0u);
    atomicStore(&num_occupied, 0u);
    atomicStore(&num_divided, 0u);
    atomicStore(&sum_r, 0u);
    atomicStore(&sum_g, 0u);
    atomicStore(&sum_b, 0u);
    
    if (all(ipos >= min) && all(ipos < max)) {
        let cpos = ipos - min;

        let cidx = cpos.x * wsize_prev.y * wsize_prev.z + cpos.y * wsize_prev.z + cpos.z;
        
        let child_res = vox::get_draw_area(draw_area_index_children, u32(cidx));
        child_ptr = child_res.idx;
        draw_buffer[lidx] = child_res.value;
    }

    workgroupBarrier();

    // TODO: LODs
    let lidx_next = (lidx + 1) % (VOXEL_DIM * VOXEL_DIM * VOXEL_DIM);
    if (draw_buffer[lidx] != draw_buffer[lidx_next]) {
        atomicAdd(&num_different, 1u);
    }
    
    // Simply calculate occupied cells
    if (draw_buffer[lidx] != VOXEL_IDX_EMPTY) {
        atomicAdd(&num_occupied, 1u);

        let cur = vec3u(unpack4x8unorm(draw_buffer[lidx]).xyz * 255.f);
        atomicAdd(&sum_r, cur.x);
        atomicAdd(&sum_g, cur.y);
        atomicAdd(&sum_b, cur.z);
    }
    
    if (child_ptr != VOXEL_IDX_EMPTY) {
        atomicAdd(&num_divided, 1u);
    }

    workgroupBarrier();
    
    let num_occupied_v = atomicLoad(&num_occupied);
    let num_different_v = atomicLoad(&num_different);
    let num_divided_v = atomicLoad(&num_divided);
    if (num_occupied_v == 0u) {
        let value = draw_buffer[lidx]; // the same for the whole chunk
        // if (value == VOXEL_IDX_EMPTY) { // it is always true
            // set_draw_area(draw_area_index, u32(widx), u32(VOXEL_IDX_EMPTY));
            vox::set_draw_area(draw_area_index, u32(widx), vox::DrawResult(VOXEL_IDX_EMPTY, VOXEL_IDX_EMPTY));
        // }

        // TODO: destroy leaf if allocated
        return;
    }

    let sum_color_u = vec3u(atomicLoad(&sum_r), atomicLoad(&sum_g), atomicLoad(&sum_b));
    let mean_color = vec3f(sum_color_u) / f32(num_occupied_v) / 255.f;
    var mean_color_u = pack4x8unorm(vec4f(mean_color, 0.));
//    if (num_occupied_v < u32(VOXEL_COUNT) / 16u) {
//        mean_color_u = VOXEL_IDX_EMPTY;
//    }

    if (num_different_v == 0u && num_divided_v == 0u) {
        // let value = draw_buffer[lidx]; // the same for the whole chunk
        // if (value == VOXEL_IDX_EMPTY) { // it is always true
            // (*draw_area)[widx] = VOXEL_IDX_EMPTY;
        // }

        vox::set_draw_area(draw_area_index, u32(widx), vox::DrawResult(VOXEL_IDX_EMPTY, mean_color_u));
        
        // TODO: destroy leaf if allocated
        return;
    }

    // Allocate chunk in global memory
    if (lidx == 0 && parent_ptr == VOXEL_IDX_EMPTY) {
        parent_ptr = atomicAdd(&vox::info.nodes_len, 1u);
        lod_ptr = atomicAdd(&vox::info.leafs_len, 1u);
    }
    
    let gptr = workgroupUniformLoad(&parent_ptr);
    let lptr = workgroupUniformLoad(&lod_ptr);

    vox::set_draw_area(draw_area_index, u32(widx), vox::DrawResult(gptr, mean_color_u));
    vox::nodes[gptr].leaf = lptr;
    vox::nodes[gptr].indices[lidx] = child_ptr;
    vox::leafs[lptr].voxels[lidx].color = draw_buffer[lidx];
}