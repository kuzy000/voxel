#define_import_path voxel_tracer::sdf

#import voxel_tracer::common::{RayMarchResult, DST_MAX}

// https://iquilezles.org/articles/distfunctions/
fn sdf_plane(p: vec3<f32>, n: vec3<f32>, h: f32) -> f32 {
    return abs(dot(p, n) + h);
}

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0);
}

fn sdf_round_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

fn sdf_world(p: vec3<f32>) -> f32 {
    let size = 1.f;

    var res: f32 = DST_MAX;

    res = min(res, sdf_sphere(p, 0.5));
    res = min(res, sdf_box(p - vec3(3., 0., 0.), vec3f(1.)));
    res = min(res, sdf_box(p - vec3(0., 3., 0.), vec3f(1.)));
    res = min(res, sdf_box(p - vec3(0., 0., 3.), vec3f(1.)));
    

    return res;
}

fn normal_world(p: vec3<f32>) -> vec3<f32> {
    let step = vec2(0.001f, 0.f);
    
    let x = sdf_world(p + step.xyy) - sdf_world(p - step.xyy);
    let y = sdf_world(p + step.yxy) - sdf_world(p - step.yxy);
    let z = sdf_world(p + step.yyx) - sdf_world(p - step.yyx);
    
    return normalize(vec3(x, y, z));
}

fn trace(pos: vec3<f32>, dir: vec3<f32>) -> RayMarchResult {
    let step_size = .01f;
    let steps = 64;
    let min_dst = 0.01f;
    let max_dst = 1000.f;

    var ray_len = 0.f;

    for (var i = 0; i < steps; i++) {
        let p = pos + dir * ray_len;
        let dst = sdf_world(p);
        
        if (dst < min_dst) {
            let normal = normal_world(p);
            let color = normal * 0.5 + 0.5;
            return RayMarchResult(normal, color, ray_len, vec4f(0.));
        }
        
        if (dst > max_dst) {
            break;
        }
        
        ray_len += dst;
    }

    return RayMarchResult(vec3<f32>(), vec3<f32>(), DST_MAX, vec4f(0.));
}

