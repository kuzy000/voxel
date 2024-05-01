use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::{AsBindGroup, Buffer, ShaderRef},
};

pub type VoxelTreeMaterial = ExtendedMaterial<StandardMaterial, VoxelTreeMaterialExtension>;

#[derive(Asset, TypePath, Debug, AsBindGroup, Clone)]
pub struct VoxelTreeMaterialExtension {
    #[storage(100, read_only, buffer, visibility(fragment))]
    pub voxel_nodes: Buffer,

    #[storage(101, read_only, buffer, visibility(fragment))]
    pub voxel_leafs: Buffer,
}

impl MaterialExtension for VoxelTreeMaterialExtension {
    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Path("shaders/voxel_tree.wgsl".into())
    }
}
