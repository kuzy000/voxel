#import bevy_render::view::View
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_pbr::pbr_types::PbrInput 
#import bevy_pbr::pbr_types::pbr_input_new
#import bevy_pbr::pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT
#import bevy_pbr::pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE
#import bevy_pbr::pbr_deferred_functions::deferred_gbuffer_from_pbr_input

#import voxel_tracer::common::RayMarchResult
#import voxel_tracer::common::DST_MAX
#import voxel_tracer::sdf as sdf
#import voxel_tracer::voxel_read as vox

@group(1) @binding(0) var<uniform> view : View;

struct FragmentOutputWithDepth {
    @location(0) normal: vec4<f32>,
    @location(1) motion_vector: vec2<f32>,
    @location(2) deferred: vec4<u32>,
    @location(3) deferred_lighting_pass_id: u32,
    @builtin(frag_depth) frag_depth: f32,
}

fn view_z_to_depth_ndc(view_z: f32) -> f32 {
    let ndc_pos = view.projection * vec4(0.0, 0.0, view_z, 1.0);
    return ndc_pos.z / ndc_pos.w;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutputWithDepth {
    var ndc = in.uv * 2. - 1.;
    ndc.y = -ndc.y;
    
    var dir_eye = view.inverse_projection * vec4(ndc, -1., 1.);
    dir_eye.w = 0.;

    // bevy's `view.view` is actually inversed view (and vice versa)
    let dir_w = (view.view * dir_eye).xyz;
 
    let pos4 = view.view * vec4f(0.f, 0.f, 0.f, 1.f);
    let pos = pos4.xyz / pos4.w;
    let dir = normalize(dir_w);
    // let res_vox = ray_march_voxel(pos, dir);
    let res_sdf = sdf::trace(pos, dir);
    let res_vox = vox::trace(pos, dir);

    var res: RayMarchResult;
    if (res_sdf.distance < res_vox.distance) {
        res = res_sdf;
    }
    else {
        res = res_vox;
    }
    
    //if (res.distance >= DST_MAX) {
    //    discard;
    //}
    
    //let color = res.normal + 1. * 0.5; 
    let color = res.color * (dot(res.normal, normalize(vec3f(1., 1., 1.,))) * .5 + .5) + (res.color_debug.xyz) * res.color_debug.w;

    var pbr_input = pbr_input_new();
    
    pbr_input.frag_coord = vec4(in.uv, 0.5, 1.);
    // pbr_input.material.base_color = vec4f(res.normal * 0.5 + 0.5, 1.);
    pbr_input.material.base_color = vec4f(color, 1.);
    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_UNLIT_BIT;
    //pbr_input.N = res.position;
 
    // TODO take into account distance from the camera to the near clipping plane
    let depth = view_z_to_depth_ndc(-res.distance);
 
    var newOut: FragmentOutputWithDepth;
    newOut.frag_depth = depth;
    // newOut.frag_depth = 1.;
    newOut.deferred = deferred_gbuffer_from_pbr_input(pbr_input);
    newOut.deferred_lighting_pass_id = pbr_input.material.deferred_lighting_pass_id;
    newOut.normal = vec4f(res.normal, 1.);
    newOut.motion_vector = vec2f(0., 0.);
 
    return newOut;
}