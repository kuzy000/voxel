#define_import_path voxel_tracer::voxel_common

const VOXEL_SIZE: f32 = 1.0f;
const VOXEL_DIM: i32 = #{VOXEL_DIM};
const VOXEL_COUNT: i32 = VOXEL_DIM * VOXEL_DIM * VOXEL_DIM;
const VOXEL_TREE_DEPTH: i32 = #{VOXEL_TREE_DEPTH};
// const VOXEL_MASK_LEN: i32 = #{VOXEL_MASK_LEN};

const VOXEL_IDX_EMPTY: u32 = #{VOXEL_IDX_EMPTY};
const VOXEL_IDX_ALLOCATING: u32 = VOXEL_IDX_EMPTY - 1;

const VOXEL_SIZES = array(
    VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - 0))),
    VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - 1))),
    VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - 2))),
    VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - 3))),
    VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - 4))),
    VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - 5))),
    VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH) - 6))),
);

fn pos_to_idx(ipos: vec3<i32>) -> u32 {
    return u32(ipos.x * VOXEL_DIM * VOXEL_DIM + ipos.y * VOXEL_DIM + ipos.z);
}
