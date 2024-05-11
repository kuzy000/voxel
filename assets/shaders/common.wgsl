#define_import_path voxel_tracer::common

const DST_MAX = 1e9f;

struct RayMarchResult {
    normal: vec3f,
    color: vec3f,
    distance: f32,
}