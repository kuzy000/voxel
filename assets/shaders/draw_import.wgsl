#import voxel_tracer::common::{
    ComputeBuiltins,
    perlin_noise,
    perlin_noise3,
}
#import voxel_tracer::draw::{
    DrawParams,
    draw_begin,
    draw_end,
}
#import voxel_tracer::voxel_common::{
    VOXEL_IDX_EMPTY,
    VOXEL_TREE_DEPTH,
    VOXEL_COUNT,
    VOXEL_DIM,
    pos_to_idx,
}

@group(1) @binding(0) var<storage, read_write> import_nodes: array<array<u32, VOXEL_COUNT>>;
@group(1) @binding(1) var<storage, read_write> import_leafs: array<array<u32, VOXEL_COUNT>>;

fn query_import(world_pos: vec3i) -> u32 {
    let depth = u32(VOXEL_TREE_DEPTH - 1);
    var parent_idx = 0u;
    var voxel_size = 1u; // pow(f32(VOXEL_DIM), f32(depth));

    for (var i = 0u; i < depth; i++) {
        voxel_size *= u32(VOXEL_DIM);
    }
    
    for (var i = 0u; i < depth; i++) {
        let lpos = (world_pos / vec3i(voxel_size)) % vec3i(VOXEL_DIM);
        let idx = pos_to_idx(lpos);

        var child_idx = import_nodes[parent_idx][idx];
        if (child_idx == VOXEL_IDX_EMPTY) {
            return VOXEL_IDX_EMPTY;
        }

        parent_idx = child_idx;
        voxel_size = voxel_size / u32(VOXEL_DIM);
    }

    let lpos = world_pos % vec3i(VOXEL_DIM);
    let idx = pos_to_idx(lpos);
    return import_leafs[parent_idx][idx];
}

fn draw_import(params: DrawParams) -> u32 {
    return query_import(params.world_pos);
}


@compute @workgroup_size(VOXEL_DIM, VOXEL_DIM, VOXEL_DIM)
fn draw(
    @builtin(local_invocation_id) lpos_u: vec3<u32>,
    @builtin(global_invocation_id) gpos_u: vec3<u32>,
    @builtin(workgroup_id) wpos_u: vec3<u32>,
    @builtin(num_workgroups) wsize_u: vec3<u32>
) {
    let comp = ComputeBuiltins(lpos_u, gpos_u, wpos_u, wsize_u);
    
    let params = draw_begin(comp);
    let voxel = draw_import(params);
    draw_end(comp, voxel);
}