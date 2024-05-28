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

struct VoxelInfo {
    nodes_len: atomic<u32>,
    nodes_cap: u32,

    leafs_len: atomic<u32>,
    leafs_cap: u32,
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

fn place(pos: vec3i, data: u32) {
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