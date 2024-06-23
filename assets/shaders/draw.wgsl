#import voxel_tracer::common::RayMarchResult
#import voxel_tracer::common::DST_MAX
#import voxel_tracer::sdf as sdf
#import voxel_tracer::voxel_common::{
    VOXEL_TREE_DEPTH,
    VOXEL_COUNT,
    VOXEL_SIZE,
    VOXEL_DIM,
    VOXEL_SIZES,
}
#import voxel_tracer::voxel_write as vox

@compute @workgroup_size(#{WG_X}, #{WG_Y}, #{WG_Z})
fn draw(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let min = vox::push_constants.min.xyz; 
    let max = vox::push_constants.max.xyz; 
    let depth = vox::push_constants.depth; 

    var ipos = vec3i(invocation_id) + min;
    if (any(ipos >= max)) {
        return;
    }

    let grad = vec3f(ipos - min) / vec3f(max - min - vec3i(1));
    let color = pack4x8unorm(vec4(grad, 0.));
    
    let voxel_size = VOXEL_SIZE * f32(pow(f32(VOXEL_DIM), f32(u32(VOXEL_TREE_DEPTH - 1) - depth)));
    
    let p = (vec3f(ipos) + vec3f(.5)) * voxel_size;
    let d = sdf::sdf_world(p);
    let voxel_box = sdf::sdf_box(vec3f(0.), vec3f(voxel_size));
    
    let r = max(voxel_box, d);
    
    if (r < sqrt(2.) * voxel_size) {
        vox::place(ipos, color);
    }
}