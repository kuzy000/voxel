#define_import_path voxel_tracer::voxel_write

#import voxel_tracer::common::{
    SPIN_LOCK_MAX
}

#import voxel_tracer::voxel_common::{
    VOXEL_SIZE,
    VOXEL_DIM,
    VOXEL_COUNT,
    VOXEL_TREE_DEPTH,
    VOXEL_MASK_LEN,
    VOXEL_SIZES,
    VOXEL_IDX_EMPTY,
    VOXEL_IDX_ALLOCATING,
    pos_to_idx,
    Voxel,
    VoxelLeaf,
    VoxelNode
}

struct Freelist {
    count: atomic<u32>,
    list: array<u32>,
}

struct VoxelInfo {
    nodes_cap: u32,
    nodes_len: atomic<u32>,
    nodes_free_count: atomic<u32>,

    leafs_cap: u32,
    leafs_len: atomic<u32>,
    leafs_free_count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> info : VoxelInfo;
@group(0) @binding(1) var<storage, read_write> nodes: array<VoxelNode>;
@group(0) @binding(2) var<storage, read_write> leafs: array<VoxelLeaf>;

@group(0) @binding(3) var<storage, read_write> free_nodes: array<u32>;
@group(0) @binding(4) var<storage, read_write> free_leafs: array<u32>;

struct DrawResult {
    idx: u32, // to `leafs` or `nodes`
    value: u32, // Current LOD
}

@group(0) @binding(5) var<storage, read_write> draw_area_0: array<DrawResult>;
@group(0) @binding(6) var<storage, read_write> draw_area_1: array<DrawResult>;

struct PushConstants {
    min: vec4i, // including
    max: vec4i, // excluding
    world_min: vec4i, // including
    world_max: vec4i, // excluding
    wsize_children: vec4i,
    depth: u32,
}

var <push_constant> push_constants: PushConstants;

fn get_draw_area(draw_area_index: u32, index: u32) -> DrawResult {
    if (draw_area_index == 0) {
        return draw_area_0[index];
    }
    else {
        return draw_area_1[index];
    }
}

fn set_draw_area(draw_area_index: u32, index: u32, value: DrawResult) {
    if (draw_area_index == 0) {
        draw_area_0[index] = value;
    }
    else {
        draw_area_1[index] = value;
    }
}


fn clear_nodes(idx: u32) {
    for (var i = 0; i < VOXEL_COUNT; i++) {
        nodes[idx].leaf = VOXEL_IDX_EMPTY;
        nodes[idx].indices[i] = VOXEL_IDX_EMPTY;
    }
    
    if (idx == 0) {
        nodes[idx].leaf = 0u;
    }
}

fn clear_leafs(idx: u32) {
    for (var i = 0; i < VOXEL_COUNT; i++) {
        leafs[idx].voxels[i].color = VOXEL_IDX_EMPTY;
    }
}

fn clear(idx: u32) {
    atomicStore(&info.nodes_len, 1u); // reserve root node
    atomicStore(&info.leafs_len, 1u); // reserve root node LOD

    atomicStore(&info.nodes_free_count, 0u);
    atomicStore(&info.leafs_free_count, 0u);

    if (idx < info.leafs_cap) {
        clear_leafs(idx);
    }
    else if (idx - info.leafs_cap < info.nodes_cap) {  
        clear_nodes(idx - info.leafs_cap);
    }
}

// Depending on query's depth is either:
// - `nodes[parent_idx].indices[idx]` 
// - `leafs[parent_idx].voxels[idx]
struct QueryResult {
    parent_idx: u32, 
    idx: u32, 

    // Filled only if `parent_idx` 
    value_if_empty: u32,
}

// pos - local coords of `depth`
// 0 <= depth < VOXEL_TREE_DEPTH
fn query(pos: vec3i, depth: u32) -> QueryResult {
    var parent_idx = 0u;
    var voxel_size = 1u; // pow(f32(VOXEL_DIM), f32(depth));

    for (var i = 0u; i < depth; i++) {
        voxel_size *= u32(VOXEL_DIM);
    }
    
    for (var i = 0u; i < depth; i++) {
        let lpos = (pos / vec3i(voxel_size)) % vec3i(VOXEL_DIM);
        let idx = pos_to_idx(lpos);

        var child_idx = nodes[parent_idx].indices[idx];
        if (child_idx == VOXEL_IDX_EMPTY) {
            let value = leafs[nodes[parent_idx].leaf].voxels[idx].color;
            return QueryResult(VOXEL_IDX_EMPTY, VOXEL_IDX_EMPTY, value);
        }

        parent_idx = child_idx;
        voxel_size = voxel_size / u32(VOXEL_DIM);
    }

    let lpos = pos % vec3i(VOXEL_DIM);
    let idx = pos_to_idx(lpos);
    return QueryResult(parent_idx, idx, VOXEL_IDX_EMPTY);
}