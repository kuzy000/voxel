#define_import_path voxel_tracer::common

const DST_MAX = 1e9f;
const SPIN_LOCK_MAX = 2048;

struct ComputeBuiltins {
    lpos_u: vec3<u32>, // local_invocation_id
    gpos_u: vec3<u32>, // global_invocation_id
    wpos_u: vec3<u32>, // workgroup_id
    wsize_u: vec3<u32> // num_workgroups
}

struct RayMarchResult {
    normal: vec3f,
    color: vec3f,
    distance: f32,
    color_debug: vec4f,
}

struct Intersection {
    has: bool,
    t: f32,
}

// https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
fn ray_bbox(org: vec3f, dir: vec3f, lb: vec3f, rt: vec3f) -> Intersection {
    // r.dir is unit direction vector of ray
    var dirfrac = vec3f(0.f);
    dirfrac.x = 1.0f / dir.x;
    dirfrac.y = 1.0f / dir.y;
    dirfrac.z = 1.0f / dir.z;
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    let t1 = (lb.x - org.x) * dirfrac.x;
    let t2 = (rt.x - org.x) * dirfrac.x;
    let t3 = (lb.y - org.y) * dirfrac.y;
    let t4 = (rt.y - org.y) * dirfrac.y;
    let t5 = (lb.z - org.z) * dirfrac.z;
    let t6 = (rt.z - org.z) * dirfrac.z;

    let tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    let tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0)
    {
        return Intersection(false, tmax);
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        return Intersection(false, tmax);
    }

    return Intersection(true, tmin);
}

fn is_inside(pos: vec3f, a: vec3f, b: vec3f) -> bool {
    if (pos.x < a.x) { return false; }
    if (pos.y < a.y) { return false; }
    if (pos.z < a.z) { return false; }
    if (pos.x > b.x) { return false; }
    if (pos.y > b.y) { return false; }
    if (pos.z > b.z) { return false; }
    
    return true;
}

// https://www.shadertoy.com/view/NlSGDz
// MurmurHash
fn hash(x: u32, seed: u32) -> u32 {
    let m = 0x5bd1e995u;
    var hash = seed;
    // process input
    var k = x;
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
    // some final mixing
    hash ^= hash >> 13;
    hash *= m;
    hash ^= hash >> 15;
    return hash;
}

// https://www.shadertoy.com/view/NlSGDz
// MurmurHash
fn hash2(x: vec2u, seed: u32) -> u32 {
    let m = 0x5bd1e995u;
    var hash = seed;
    // process first vector element
    var k = x.x; 
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
    // process second vector element
    k = x.y; 
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
	// some final mixing
    hash ^= hash >> 13;
    hash *= m;
    hash ^= hash >> 15;
    return hash;
}

fn hash3(x: vec3u, seed: u32) -> u32 {
    let m = 0x5bd1e995u;
    var hash = seed;
    // process first vector element
    var k = x.x; 
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
    // process second vector element
    k = x.y; 
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
    // process third vector element
    k = x.z; 
    k *= m;
    k ^= k >> 24;
    k *= m;
    hash *= m;
    hash ^= k;
	// some final mixing
    hash ^= hash >> 13;
    hash *= m;
    hash ^= hash >> 15;
    return hash;
}

// https://www.shadertoy.com/view/NlSGDz
fn random_dir2(hash: u32) -> vec2f {
    switch (i32(hash) & 3) { // look at the last two bits to pick a gradient direction
        case 0: {
            return vec2f(1.0, 1.0);
        }
        case 1: {
            return vec2f(-1.0, 1.0);
        }
        case 2: {
            return vec2f(1.0, -1.0);
        }
        case 3, default: {
            return vec2f(-1.0, -1.0);
        }
    }
}

fn random_dir3(hash: u32) -> vec3f {
    switch (i32(hash) & 7) { 
        case 0: {
            return vec3f(1.0, 1.0, 1.0);
        }
        case 1: {
            return vec3f(-1.0, 1.0, 1.0);
        }
        case 2: {
            return vec3f(1.0, -1.0, 1.0);
        }
        case 3 {
            return vec3f(-1.0, -1.0, 1.0);
        }
        case 4: {
            return vec3f(1.0, 1.0, -1.0);
        }
        case 5: {
            return vec3f(-1.0, 1.0, -1.0);
        }
        case 6: {
            return vec3f(1.0, -1.0, -1.0);
        }
        case 7, default: {
            return vec3f(-1.0, -1.0, -1.0);
        }
    }
}

// https://www.shadertoy.com/view/NlSGDz
fn perlin_noise_octave(position: vec2f, seed: u32) -> f32 {
    let floorPosition = floor(position);
    let fractPosition = position - floorPosition;
    let cellCoordinates = vec2u(floorPosition);

    let value1 = dot(random_dir2(hash2(cellCoordinates, seed)), fractPosition);
    let value2 = dot(random_dir2(hash2((cellCoordinates + vec2u(1u, 0u)), seed)), fractPosition - vec2f(1.0, 0.0));
    let value3 = dot(random_dir2(hash2((cellCoordinates + vec2u(0u, 1u)), seed)), fractPosition - vec2f(0.0, 1.0));
    let value4 = dot(random_dir2(hash2((cellCoordinates + vec2u(1u, 1u)), seed)), fractPosition - vec2f(1.0, 1.0));

    // 6t^5 - 15t^4 + 10t^3
    var t = fractPosition;
    t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);

    return mix(mix(value1, value2, t.x), mix(value3, value4, t.x), t.y);
}

fn perlin_noise_octave3(position: vec3f, seed: u32) -> f32 {
    let floorPosition = floor(position);
    let fractPosition = position - floorPosition;
    let cellCoordinates = vec3u(floorPosition);

    let value1 = dot(random_dir3(hash3((cellCoordinates + vec3u(0u, 0u, 0u)), seed)), fractPosition - vec3f(0.0, 0.0, 0.0));
    let value2 = dot(random_dir3(hash3((cellCoordinates + vec3u(1u, 0u, 0u)), seed)), fractPosition - vec3f(1.0, 0.0, 0.0));
    let value3 = dot(random_dir3(hash3((cellCoordinates + vec3u(0u, 1u, 0u)), seed)), fractPosition - vec3f(0.0, 1.0, 0.0));
    let value4 = dot(random_dir3(hash3((cellCoordinates + vec3u(1u, 1u, 0u)), seed)), fractPosition - vec3f(1.0, 1.0, 0.0));
    let value5 = dot(random_dir3(hash3((cellCoordinates + vec3u(0u, 0u, 1u)), seed)), fractPosition - vec3f(0.0, 0.0, 1.0));
    let value6 = dot(random_dir3(hash3((cellCoordinates + vec3u(1u, 0u, 1u)), seed)), fractPosition - vec3f(1.0, 0.0, 1.0));
    let value7 = dot(random_dir3(hash3((cellCoordinates + vec3u(0u, 1u, 1u)), seed)), fractPosition - vec3f(0.0, 1.0, 1.0));
    let value8 = dot(random_dir3(hash3((cellCoordinates + vec3u(1u, 1u, 1u)), seed)), fractPosition - vec3f(1.0, 1.0, 1.0));

    // 6t^5 - 15t^4 + 10t^3
    var t = fractPosition;
    t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    
    let z1 = mix(mix(value1, value2, t.x), mix(value3, value4, t.x), t.y);
    let z2 = mix(mix(value5, value6, t.x), mix(value7, value8, t.x), t.y);

    return mix(z1, z2, t.z);
}

// https://www.shadertoy.com/view/NlSGDz
fn perlin_noise(position: vec2f, frequency: f32, octaveCount: i32, persistence: f32, lacunarity: f32, seed: u32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var currentFrequency = frequency;
    var currentSeed = seed;
    for (var i = 0; i < octaveCount; i++) {
        currentSeed = hash(currentSeed, 0u); // create a new seed for each octave
        value += perlin_noise_octave(position * currentFrequency, currentSeed) * amplitude;
        amplitude *= persistence;
        currentFrequency *= lacunarity;
    }
    return value;
}

fn perlin_noise3(position: vec3f, frequency: f32, octaveCount: i32, persistence: f32, lacunarity: f32, seed: u32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var currentFrequency = frequency;
    var currentSeed = seed;
    for (var i = 0; i < octaveCount; i++) {
        currentSeed = hash(currentSeed, 0u); // create a new seed for each octave
        value += perlin_noise_octave3(position * currentFrequency, currentSeed) * amplitude;
        amplitude *= persistence;
        currentFrequency *= lacunarity;
    }
    return value;
}
