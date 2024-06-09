#import voxel_tracer::common::RayMarchResult
#import voxel_tracer::common::DST_MAX
#import voxel_tracer::sdf as sdf
#import voxel_tracer::voxel_common::{
    VOXEL_TREE_DEPTH,
    VOXEL_COUNT,
}
#import voxel_tracer::voxel_write as vox

@compute @workgroup_size(#{WG_X}, #{WG_Y}, #{WG_Z})
fn draw(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let x = i32(invocation_id.x);
    let y = i32(invocation_id.y);
    let z = i32(invocation_id.z);

    let wgsize = vec3i(#{WG_X}, #{WG_Y}, #{WG_Z});
    let size = wgsize * vec3i(num_workgroups);

    let v = vec3f(f32(x), f32(y), f32(z));
    let pos = vec3i(x, y, z) + vec3i(10);

    let vn = (v / vec3f(size - vec3i(1)));
    let color = pack4x8unorm(vec4(vn, 0.));

    let len = length(vn * 2. - 1.);
    if (len < 1. && len > .9) {
        vox::place(pos, color);
    }
}