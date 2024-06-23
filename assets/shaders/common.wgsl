#define_import_path voxel_tracer::common

const DST_MAX = 1e9f;
const SPIN_LOCK_MAX = 2048;

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