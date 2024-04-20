struct View {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    // viewport(x_origin, y_origin, width, height)
//    viewport: vec4<f32>,
//    frustum: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> view: View;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let randomNumber = randomFloat(invocation_id.y * num_workgroups.x + invocation_id.x);
    let alive = randomNumber > 0.9;
    let color = vec4<f32>(f32(alive));

    textureStore(texture, location, color);
}

fn is_alive(location: vec2<i32>, offset_x: i32, offset_y: i32) -> i32 {
    let value: vec4<f32> = textureLoad(texture, location + vec2<i32>(offset_x, offset_y));
    return i32(value.x);
}

fn count_alive(location: vec2<i32>) -> i32 {
    return is_alive(location, -1, -1) +
           is_alive(location, -1,  0) +
           is_alive(location, -1,  1) +
           is_alive(location,  0, -1) +
           is_alive(location,  0,  1) +
           is_alive(location,  1, -1) +
           is_alive(location,  1,  0) +
           is_alive(location,  1,  1);
}

const SIZE = vec2<i32>(1280, 720);

// https://iquilezles.org/articles/distfunctions/
fn sdf_plane(p: vec3<f32>, n: vec3<f32>, h: f32) -> f32 {
    return dot(p, n) + h;
}

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_round_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

fn sdf_world(p: vec3<f32>) -> f32 {
    let size = 1.f;

    var res: f32 = 1e9;

    res = min(res, sdf_round_box(p, vec3<f32>(size), 0.1f));
    res = min(res, sdf_plane(p, vec3(0f, 1f, 0f), 1.));
    res = min(res, sdf_sphere(p - vec3(2f, 0f, 2f), 1.));
    

    return res;
}

fn normal_world(p: vec3<f32>) -> vec3<f32> {
    let step = vec2(0.001f, 0.f);
    
    let x = sdf_world(p + step.xyy) - sdf_world(p - step.xyy);
    let y = sdf_world(p + step.yxy) - sdf_world(p - step.yxy);
    let z = sdf_world(p + step.yyx) - sdf_world(p - step.yyx);
    
    return normalize(vec3(x, y, z));
}

fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    let step_size = .01f;
    let steps = 64;
    let min_dst = 0.01f;
    let max_dst = 1000.f;

    var ray_len = 0.f;

    for (var i = 0; i < steps; i++) {
        let p = ro + rd * ray_len;
        let dst = sdf_world(p);
        
        if (dst < min_dst) {
            return normal_world(p) * 0.5 + 0.5;
        }
        
        if (dst > max_dst) {
            break;
        }
        
        ray_len += dst;
    }

    return vec3<f32>(0.f);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    
    let uv = (vec2<f32>(location) / vec2<f32>(SIZE)) * vec2(2.f, -2.f) - vec2(1.f, -1.f);

    let a4 = (view.inverse_view_proj * vec4(uv, -1.0f, 1.f));
    let a = a4.xyz / a4.w;

    let b4 = (view.inverse_view_proj * vec4(uv, 1.0f, 1.f));
    let b = b4.xyz / b4.w;
    
    let rd = normalize(b - a);
    
    // let a4 = (view.inverse_view * vec4(0f, 0f, 0f, 1f));
    // let a = a4.xyz / a4.w;

    // let right = (view.inverse_view * vec4(1f, 0f, 0f, 0f)).xyz;
    // let up = (view.inverse_view * vec4(0f, 1f, 0f, 0f)).xyz;
    // let forward = (view.inverse_view * vec4(0f, 0f, -1f, 0f)).xyz;
    // 
    // let rd = right * uv.x + up * uv.y + forward;

    let c = ray_march(a, rd);

    let color = vec4<f32>(c, 1.f);

    storageBarrier();
    textureStore(texture, location, color);
}