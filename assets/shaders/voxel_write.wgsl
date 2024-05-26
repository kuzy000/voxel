#define_import_path voxel_tracer::voxel_write

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
    nodes_len: atomic<u32>,
    nodes_cap: atomic<u32>,

    leafs_len: atomic<u32>,
    leafs_cap: atomic<u32>,
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