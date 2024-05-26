#import voxel_tracer::common::RayMarchResult
#import voxel_tracer::common::DST_MAX
#import voxel_tracer::sdf as sdf
#import voxel_tracer::voxel_common::{
    VOXEL_TREE_DEPTH,
    VOXEL_COUNT,
}
#import voxel_tracer::voxel_write as vox

@compute @workgroup_size(1, 1, 1)
fn draw(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    for (var i = 0; i < VOXEL_TREE_DEPTH; i++) {
        vox::leafs[0].mask[i] = ~u32(0);
    }

    let color = pack4x8unorm(vec4(1., 0., 0., 1.));

    for (var i = 0; i < VOXEL_COUNT; i++) {
        vox::leafs[0].voxels[i].color = color;
    }
}