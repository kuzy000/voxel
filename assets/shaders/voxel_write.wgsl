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

struct Voxel {
    color: u32,
}

struct VoxelLeaf {
    voxels: array<Voxel, VOXEL_COUNT>,
}

struct VoxelNode {
    // Indices to either `nodes` or `leafs` depending on the current depth
    indices: array<atomic<u32>, VOXEL_COUNT>,
}

@group(0) @binding(0) var<storage, read_write> info : VoxelInfo;
@group(0) @binding(1) var<storage, read_write> nodes: array<VoxelNode>;
@group(0) @binding(2) var<storage, read_write> leafs: array<VoxelLeaf>;

@group(0) @binding(3) var<storage, read_write> free_nodes: array<u32>;
@group(0) @binding(4) var<storage, read_write> free_leafs: array<u32>;

@group(0) @binding(5) var<storage, read_write> draw_area_0: array<u32>;
@group(0) @binding(6) var<storage, read_write> draw_area_1: array<u32>;

struct PushConstants {
    min: vec4i, // including
    max: vec4i, // excluding
    world_min: vec4i, // including
    world_max: vec4i, // excluding
    wsize_children: vec4i,
    depth: u32,
}

var <push_constant> push_constants: PushConstants;

fn place_old(pos: vec3i, data: u32) {
    var parent_idx = 0u;
    var voxel_size = i32(VOXEL_SIZES[1]);

    for (var depth = 0; depth < VOXEL_TREE_DEPTH - 2; depth++) {
        let lpos = (pos / vec3i(voxel_size)) % vec3i(VOXEL_DIM);
        let idx = pos_to_idx(lpos);

        var child_idx = 0u;
        let node_ptr = &nodes[parent_idx];
        let res = atomicCompareExchangeWeak(&(*node_ptr).indices[idx], VOXEL_IDX_EMPTY, VOXEL_IDX_ALLOCATING);
        if (res.exchanged) {
            // Allocate new chunk
            let new_idx = atomicAdd(&info.nodes_len, 1u);
            if new_idx >= info.nodes_cap {
                return;
            }
            
            atomicStore(&(*node_ptr).indices[idx], new_idx);
            child_idx = new_idx;
        }
        else {
            child_idx = res.old_value;
            for (var i = 0; i < SPIN_LOCK_MAX; i++) {
                child_idx = atomicLoad(&(*node_ptr).indices[idx]);

                if (child_idx != VOXEL_IDX_ALLOCATING) {
                    break;
                }
            }

            if (child_idx == VOXEL_IDX_ALLOCATING) {
                return;
            }
        }
        
        parent_idx = child_idx;
        voxel_size = voxel_size / VOXEL_DIM;
    }

    {
        let lpos = (pos / vec3i(voxel_size)) % vec3i(VOXEL_DIM);
        let idx = pos_to_idx(lpos);

        var child_idx = 0u;
        let node_ptr = &nodes[parent_idx];
        let res = atomicCompareExchangeWeak(&(*node_ptr).indices[idx], VOXEL_IDX_EMPTY, VOXEL_IDX_ALLOCATING);
        if (res.exchanged) {
            // Allocate new chunk
            let new_idx = atomicAdd(&info.leafs_len, 1u);
            if new_idx >= info.leafs_cap {
                return;
            }

            atomicStore(&(*node_ptr).indices[idx], new_idx);
            child_idx = new_idx;
        }
        else {
            child_idx = res.old_value;
            for (var i = 0; i < SPIN_LOCK_MAX; i++) {
                child_idx = atomicLoad(&(*node_ptr).indices[idx]);
                if (child_idx != VOXEL_IDX_ALLOCATING) {
                    break;
                }
            }

            if (child_idx == VOXEL_IDX_ALLOCATING) {
                return;
            }
        }

        parent_idx = child_idx;
        voxel_size = voxel_size / VOXEL_DIM;
    }


    let lpos = pos % vec3i(VOXEL_DIM);
    let idx = pos_to_idx(lpos);
    leafs[parent_idx].voxels[idx].color = data;
}

fn place(pos: vec3i, data: u32) {
    var parent_idx = 0u;
    // var voxel_size = i32(VOXEL_SIZES[u32(VOXEL_TREE_DEPTH) - push_constants.depth]);
    var voxel_size = f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - (u32(VOXEL_TREE_DEPTH) - push_constants.depth))));

    for (var depth = 0u; depth < min(u32(VOXEL_TREE_DEPTH - 2), push_constants.depth + 1); depth++) {
        let lpos = (pos / vec3i(voxel_size)) % vec3i(VOXEL_DIM);

        let idx = pos_to_idx(lpos);
        let p = &nodes[parent_idx].indices[idx];

        var child_idx = 0u;
        if (*p == VOXEL_IDX_EMPTY) {
            if (depth != push_constants.depth) {
                return;
            }

            // Allocate new chunk
            let new_idx = atomicAdd(&info.nodes_len, 1u);
            if new_idx >= info.nodes_cap {
                return;
            }
            
            *p = new_idx;
        }
        child_idx = *p;

        parent_idx = child_idx;
        voxel_size = voxel_size / f32(VOXEL_DIM);
    }

    if (push_constants.depth >= u32(VOXEL_TREE_DEPTH - 2)) {
        let lpos = (pos / vec3i(voxel_size)) % vec3i(VOXEL_DIM);
        let idx = pos_to_idx(lpos);
        let p = &nodes[parent_idx].indices[idx];

        var child_idx = 0u;
        if (*p == VOXEL_IDX_EMPTY) {
            if (u32(VOXEL_TREE_DEPTH - 2) != push_constants.depth) {
                return;
            }

            // Allocate new chunk
            let new_idx = atomicAdd(&info.leafs_len, 1u);
            if new_idx >= info.leafs_cap {
                return;
            }

            *p = new_idx;
        }
        child_idx = *p;

        parent_idx = child_idx;
        voxel_size = voxel_size / f32(VOXEL_DIM);
    }

    if (push_constants.depth == u32(VOXEL_TREE_DEPTH - 1)) {
        let lpos = pos % vec3i(VOXEL_DIM);
        let idx = pos_to_idx(lpos);
        leafs[parent_idx].voxels[idx].color = data;
    }
}

fn clear_nodes(idx: u32) {
    for (var i = 0; i < VOXEL_COUNT; i++) {
        nodes[idx].indices[i] = VOXEL_IDX_EMPTY;
    }
}

fn clear_leafs(idx: u32) {
    for (var i = 0; i < VOXEL_COUNT; i++) {
        leafs[idx].voxels[i].color = VOXEL_IDX_EMPTY;
    }
}

fn clear(idx: u32) {
    atomicStore(&info.nodes_len, 1u); // reserve root node
    atomicStore(&info.leafs_len, 0u);

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
            return QueryResult(VOXEL_IDX_EMPTY, VOXEL_IDX_EMPTY);
        }

        parent_idx = child_idx;
        voxel_size = voxel_size / u32(VOXEL_DIM);
    }

    let lpos = pos % vec3i(VOXEL_DIM);
    let idx = pos_to_idx(lpos);
    return QueryResult(parent_idx, idx);
}