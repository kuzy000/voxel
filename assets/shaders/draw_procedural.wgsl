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
#import voxel_tracer::voxel_common::VOXEL_IDX_EMPTY

fn draw_sphere(params: DrawParams) -> u32 {
    if (params.voxel != VOXEL_IDX_EMPTY) {
        return params.voxel;
    }

    let ipos = params.world_pos;
    let min = params.world_min;
    let max = params.world_max;

    let grad = vec3f(ipos - min) / vec3f(max - min - vec3i(1));
    //let color = pack4x8unorm(vec4f(grad, 0.));
    let color = pack4x8unorm(vec4f(1., 1., 1., 0.));
    
    if (2 == 1) {
        return color;
    }

    let center = min + (max - min) / 2;
    let radius = (max - min) / 2 ;

    if (length(vec3f(ipos - center)) < f32(radius.x)) {
        return color;
    }

    return VOXEL_IDX_EMPTY;
}

fn draw_caves(params: DrawParams) -> u32 {
    var PALETTE = array<u32, 8>(
        0xd53e4f,
        0xf46d43,
        0xfdae61,
        0xfee08b,
        0xe6f598,
        0xabdda4,
        0x66c2a5,
        0x3288bd,
    );

    if (params.voxel != VOXEL_IDX_EMPTY) {
        return params.voxel;
    }

    let ipos = params.world_pos;
    let min = params.world_min;
    let max = params.world_max;

    let grad = vec3f(ipos - min) / vec3f(max - min - vec3i(1));
    //let color = pack4x8unorm(vec4f(grad, 0.));
    let color = pack4x8unorm(vec4f(1., 1., 1., 0.));
    
    if (2 == 1) {
        return color;
    }
    
    
    let p = vec3f(ipos);
    let lpos = p / vec3f(max - min);
    
    let lands = perlin_noise(p.xz, .0005, 6, .5, 2., 123u) * .5 + .5;
    let caves = perlin_noise3(p, .002, 6, .5, 2., 123u) * .5 + .5;

    if (lands > lpos.y && caves > 0.5) {
        let c = (caves - .5) * 2.;
        let cv = u32(c * 9.);
        let cu = PALETTE[cv];
        
        let b = (cu >>  0u) & 0xFFu;
        let g = (cu >>  8u) & 0xFFu;
        let r = (cu >> 16u) & 0xFFu;
        let col = vec3f(vec3u(r, g, b)) / 255.;

        return pack4x8unorm(vec4f(col, 0.));
    }

    return VOXEL_IDX_EMPTY;
}

@compute @workgroup_size(#{WG_X}, #{WG_Y}, #{WG_Z})
fn draw(
    @builtin(local_invocation_id) lpos_u: vec3<u32>,
    @builtin(global_invocation_id) gpos_u: vec3<u32>,
    @builtin(workgroup_id) wpos_u: vec3<u32>,
    @builtin(num_workgroups) wsize_u: vec3<u32>
) {
    let comp = ComputeBuiltins(lpos_u, gpos_u, wpos_u, wsize_u);
    
    let params = draw_begin(comp);
    let voxel = draw_caves(params);
    draw_end(comp, voxel);
}